import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import settings as s
from src.api.app import EXAMPLE, create_app
from src.data.ingest import clean
from src.features import build_pipeline
from src.inference import Predictor

from .test_ingest import raw_frame  # noqa: F401


@pytest.fixture
def model_dir(raw_frame, tmp_path):  # noqa: F811
    df, _ = clean(raw_frame)
    model = LogisticRegression(max_iter=500)
    pipe = build_pipeline(model).fit(df[s.FEATURES], df[s.TARGET])
    joblib.dump(pipe, tmp_path / "model.joblib")
    card = {"decision_threshold": 0.3, "registry_version": 7, "model_family": "test"}
    (tmp_path / "model_card.json").write_text(json.dumps(card))
    return tmp_path


@pytest.fixture
def client(model_dir):
    with TestClient(create_app(model_dir)) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "alive"}


def test_readyz_when_model_loaded(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["model_version"] == "7"


def test_readyz_503_without_model(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/readyz").status_code == 503
        assert c.post("/v1/predict", json=EXAMPLE).status_code == 503


def test_predict_returns_all_fields(client):
    r = client.post("/v1/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_disease"] <= 1
    assert 0 <= body["confidence"] <= 1
    assert body["threshold"] == 0.3
    assert body["model_version"] == "7"
    assert r.headers["x-request-id"] == body["request_id"]


def test_prediction_follows_threshold(client):
    body = client.post("/v1/predict", json=EXAMPLE).json()
    assert body["prediction"] == int(body["probability_disease"] >= body["threshold"])


def test_unknown_values_allowed_as_null(client):
    payload = {**EXAMPLE, "major_vessels": None, "thal": None}
    assert client.post("/v1/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("age", 5), ("chest_pain", 9), ("thal", 4), ("cholesterol", -1), ("sex", "male"),
])
def test_invalid_input_rejected(client, field, value):
    r = client.post("/v1/predict", json={**EXAMPLE, field: value})
    assert r.status_code == 422


def test_missing_and_extra_fields_rejected(client):
    missing = {k: v for k, v in EXAMPLE.items() if k != "age"}
    assert client.post("/v1/predict", json=missing).status_code == 422
    assert client.post("/v1/predict", json={**EXAMPLE, "weight": 80}).status_code == 422


def test_model_card_endpoint(client):
    assert client.get("/v1/model").json()["model_family"] == "test"


def test_metrics_exposed(client):
    client.post("/v1/predict", json=EXAMPLE)
    text = client.get("/metrics").text
    for name in ("heart_api_requests_total", "heart_predictions_total",
                 "heart_prediction_probability_rolling_mean", "heart_model_info"):
        assert name in text


def test_predictor_confidence_is_zero_at_threshold(model_dir):
    p = Predictor(model_dir)
    assert p._result(p.threshold)["confidence"] == 0
    assert p._result(1.0)["confidence"] == 1
    assert p._result(0.0)["confidence"] == 1
