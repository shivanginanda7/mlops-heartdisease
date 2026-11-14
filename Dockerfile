# Heart disease risk API - serving image
#
# Two stages: the first installs Python packages, the second copies only the
# installed packages plus the app code and model. No compilers, no training
# libraries (MLflow, Optuna, Jupyter) and no data end up in the final image.

# ---------- stage 1: install dependencies ----------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt


# ---------- stage 2: runtime ----------
FROM python:3.12-slim

LABEL org.opencontainers.image.title="heart-disease-risk-api" \
      org.opencontainers.image.description="FastAPI service for the UCI heart disease classifier"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

# XGBoost needs the OpenMP runtime on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user, never as root
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY models/model.joblib models/model_card.json ./models/

USER appuser
EXPOSE 8000

# Healthy only when the model is loaded (readiness, not just "process is up").
# python:slim has no curl, so the check uses Python's standard library.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; \
r = urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2); \
sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
