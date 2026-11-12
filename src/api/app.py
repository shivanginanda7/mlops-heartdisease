"""FastAPI service for the heart disease risk model.

Run locally from the repo root:
    uvicorn src.api.app:app --port 8000

Endpoints:
    POST /v1/predict   one patient as JSON -> prediction, probability, confidence
    GET  /v1/model     model card of the served model
    GET  /healthz      liveness: the process is up
    GET  /readyz       readiness: the model is loaded (503 until it is)
    GET  /metrics      Prometheus metrics
    GET  /docs         interactive Swagger UI
"""
import hashlib
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from src import settings as s
from src.inference import Predictor

# ---------------------------------------------------------------- logging
log = logging.getLogger("heart_api")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def log_event(**fields) -> None:
    """One JSON line per event, easy to grep and to ship to a log store."""
    fields["ts"] = datetime.now(UTC).isoformat(timespec="milliseconds")
    log.info(json.dumps(fields, default=str))


# ---------------------------------------------------------------- metrics
REQUESTS = Counter("heart_api_requests_total", "HTTP requests",
                   ["method", "path", "status"])
LATENCY = Histogram("heart_api_request_seconds", "Request latency in seconds", ["path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_predictions_total", "Predictions by outcome", ["label"])
PROBABILITY = Histogram("heart_prediction_probability", "Predicted disease probability",
                        buckets=[i / 10 for i in range(1, 11)])
ROLLING_MEAN = Gauge("heart_prediction_probability_rolling_mean",
                     "Mean predicted probability over the last 100 predictions "
                     "(a simple drift signal)")
MODEL_INFO = Gauge("heart_model_info", "Served model", ["version", "family", "threshold"])

_recent = deque(maxlen=100)


# ---------------------------------------------------------------- schemas
EXAMPLE = {
    "age": 58, "sex": 1, "chest_pain": 4, "resting_bp": 140, "cholesterol": 260,
    "fasting_sugar": 0, "rest_ecg": 2, "max_hr": 132, "exercise_angina": 1,
    "st_depression": 1.8, "st_slope": 2, "major_vessels": 1, "thal": 7,
}


class Patient(BaseModel):
    """One patient. Codes follow the UCI data dictionary."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    age: int = Field(ge=18, le=100, description="Years")
    sex: Literal[0, 1] = Field(description="0 = female, 1 = male")
    chest_pain: Literal[1, 2, 3, 4] = Field(
        description="1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    resting_bp: float = Field(ge=70, le=250, description="Resting blood pressure, mm Hg")
    cholesterol: float = Field(ge=100, le=650, description="Serum cholesterol, mg/dl")
    fasting_sugar: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    rest_ecg: Literal[0, 1, 2] = Field(description="Resting ECG result")
    max_hr: float = Field(ge=60, le=230, description="Maximum heart rate achieved")
    exercise_angina: Literal[0, 1] = Field(description="Exercise-induced angina")
    st_depression: float = Field(ge=0, le=7, description="ST depression (oldpeak), mm")
    st_slope: Literal[1, 2, 3] = Field(description="Slope of peak exercise ST segment")
    major_vessels: Literal[0, 1, 2, 3] | None = Field(
        default=None, description="Vessels coloured by fluoroscopy; null if unknown")
    thal: Literal[3, 6, 7] | None = Field(
        default=None, description="3 normal, 6 fixed defect, 7 reversible; null if unknown")


class PredictionOut(BaseModel):
    request_id: str
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_band: str
    threshold: float
    model_version: str
    model_family: str


# ---------------------------------------------------------------- app
def create_app(model_dir=None) -> FastAPI:
    model_dir = model_dir or os.getenv("MODEL_DIR", str(s.MODELS_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = Predictor(model_dir)
            p = app.state.predictor
            MODEL_INFO.labels(p.version, p.family, str(p.threshold)).set(1)
            log_event(event="model_loaded", model_dir=str(model_dir),
                      version=p.version, family=p.family, threshold=p.threshold)
        except FileNotFoundError as exc:
            app.state.predictor = None
            log_event(event="model_missing", model_dir=str(model_dir), error=str(exc))
        yield

    app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(response.status_code)).inc()
            LATENCY.labels(path).observe(elapsed)
            log_event(event="request", request_id=request_id, method=request.method,
                      path=path, status=response.status_code,
                      latency_ms=round(elapsed * 1000, 2))
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "alive"}

    @app.get("/readyz")
    def readyz(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ready", "model_version": request.app.state.predictor.version}

    @app.get("/v1/model")
    def model_card(request: Request):
        predictor = _require_model(request)
        return predictor.card

    @app.post("/v1/predict", response_model=PredictionOut)
    def predict(patient: Patient, request: Request):
        predictor = _require_model(request)
        payload = patient.model_dump()
        result = predictor.predict([payload])[0]

        PREDICTIONS.labels(result["label"]).inc()
        PROBABILITY.observe(result["probability_disease"])
        _recent.append(result["probability_disease"])
        ROLLING_MEAN.set(sum(_recent) / len(_recent))

        # Log a hash of the input instead of the patient's values
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        log_event(event="prediction", request_id=request.state.request_id,
                  input_hash=digest[:16], prediction=result["prediction"],
                  probability=result["probability_disease"],
                  model_version=result["model_version"])
        return {"request_id": request.state.request_id, **result}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _require_model(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


app = create_app()
