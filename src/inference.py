"""Load the trained pipeline and its model card, and score patients.

Shared by the API (src/api/app.py) and the command line (src/predict.py), so
both apply the same threshold and return the same fields.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src import settings as s


class Predictor:
    def __init__(self, model_dir: Path = s.MODELS_DIR):
        model_dir = Path(model_dir)
        self.pipeline = joblib.load(model_dir / s.MODEL_FILE.name)
        self.card = json.loads((model_dir / s.MODEL_CARD_FILE.name).read_text())
        self.threshold = float(self.card["decision_threshold"])
        self.version = str(self.card.get("registry_version") or "local")
        self.family = self.card.get("model_family", "unknown")

    def predict(self, records: list[dict]) -> list[dict]:
        frame = pd.DataFrame(records, columns=s.FEATURES).astype(float)
        probs = self.pipeline.predict_proba(frame)[:, 1]
        return [self._result(float(p)) for p in probs]

    def _result(self, p: float) -> dict:
        t = self.threshold
        pred = int(p >= t)
        # Confidence = how far the probability sits from the decision threshold,
        # scaled to 0-1 on each side: 0 at the threshold, 1 at certainty.
        confidence = (p - t) / (1 - t) if pred else (t - p) / t
        return {
            "prediction": pred,
            "label": "disease" if pred else "no disease",
            "probability_disease": round(p, 4),
            "confidence": round(confidence, 4),
            "risk_band": _risk_band(p, t),
            "threshold": t,
            "model_version": self.version,
            "model_family": self.family,
        }


def _risk_band(p: float, t: float) -> str:
    if p < t:
        return "low"
    if p < 0.5:
        return "elevated"
    return "high"
