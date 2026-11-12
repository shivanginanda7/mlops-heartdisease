import json

import joblib
import numpy as np
import optuna
import pytest

from src import settings as s
from src.data.ingest import clean
from src.train import make_pipeline, metrics_at, train_all, tune_threshold

from .test_ingest import raw_frame  # noqa: F401


@pytest.fixture
def xy(raw_frame):  # noqa: F811
    df, _ = clean(raw_frame)
    return df[s.FEATURES], df[s.TARGET]


def test_metrics_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = metrics_at(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["accuracy"] == m["recall"] == m["roc_auc"] == 1.0


def test_threshold_in_search_range():
    y = np.array([0, 0, 1, 1, 1])
    t = tune_threshold(y, np.array([0.1, 0.4, 0.45, 0.6, 0.9]))
    assert 0.1 <= t <= 0.7


@pytest.mark.parametrize("name", ["logreg_elasticnet", "svm_rbf", "xgboost"])
def test_each_model_fits_and_predicts(name, xy):
    X, y = xy
    trial = optuna.trial.FixedTrial({
        "engineered": True, "C": 1.0, "l1_ratio": 0.5, "gamma": 0.1,
        "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9,
        "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0,
    })
    pipe = make_pipeline(name, trial).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_train_all_end_to_end(xy, tmp_path):
    X, y = xy
    X = X.sample(80, replace=True, random_state=0).reset_index(drop=True)
    y = y.sample(80, replace=True, random_state=0).reset_index(drop=True)
    out = train_all(
        X, y, trials=2,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=(tmp_path / "mlruns").as_uri(),
        models_dir=tmp_path / "models",
        report_dir=tmp_path / "report",
        models=["logreg_elasticnet"],
    )
    assert (tmp_path / "models" / "model.joblib").exists()
    card = json.loads((tmp_path / "models" / "model_card.json").read_text())
    assert str(card["registry_version"]) == "1"
    assert 0.1 <= card["decision_threshold"] <= 0.7
    pipe = joblib.load(tmp_path / "models" / "model.joblib")
    assert pipe.predict_proba(X.head(2)).shape == (2, 2)
    assert (tmp_path / "report" / "model_comparison.png").exists()
    assert out["champion"]["model"] == "logreg_elasticnet"
