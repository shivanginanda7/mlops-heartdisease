import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import settings as s
from src.data.ingest import clean
from src.features import ClinicalFeatures, build_pipeline, build_preprocessor

from .test_ingest import raw_frame  # noqa: F401


@pytest.fixture
def xy(raw_frame):  # noqa: F811
    df, _ = clean(raw_frame)
    return df[s.FEATURES], df[s.TARGET]


def test_engineered_values():
    X = pd.DataFrame([{**{c: 0 for c in s.FEATURES}, "age": 60, "max_hr": 120,
                       "exercise_angina": 1, "st_depression": 2.0}])
    out = ClinicalFeatures().transform(X)
    assert out.loc[0, "pct_max_hr"] == pytest.approx(120 / 160)
    assert out.loc[0, "exercise_ischaemia"] == 1


def test_engineering_can_be_switched_off(xy):
    X, _ = xy
    out = ClinicalFeatures(enabled=False).transform(X)
    assert "pct_max_hr" not in out.columns


def test_preprocessor_output_shape_and_no_nans(xy):
    X, _ = xy
    steps = build_pipeline(LogisticRegression())[:-1]
    Xt = steps.fit_transform(X)
    # 6 numeric + 17 one-hot (4+3+3+4+3) + 4 binary
    assert Xt.shape == (len(X), 28)
    assert not np.isnan(Xt).any()


def test_unseen_category_does_not_crash(xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.iloc[[0]].copy()
    row["thal"] = 5  # code never seen in training
    assert pipe.predict_proba(row).shape == (1, 2)


def test_missing_values_imputed_at_predict_time(xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    row = X.iloc[[0]].copy()
    row[["major_vessels", "thal"]] = np.nan
    proba = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_predict_proba_in_range(xy):
    X, y = xy
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_preprocessor_without_engineering_has_fewer_columns(xy):
    X, _ = xy
    with_eng = build_preprocessor(True).fit(ClinicalFeatures(True).transform(X))
    without = build_preprocessor(False).fit(ClinicalFeatures(False).transform(X))
    assert len(with_eng.get_feature_names_out()) == len(without.get_feature_names_out()) + 2
