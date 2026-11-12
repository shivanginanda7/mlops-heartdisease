import numpy as np
import pandas as pd
import pytest

from src import settings as s
from src.data.ingest import clean


@pytest.fixture
def raw_frame():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(object),
        "thal": rng.choice([3, 6, 7], n).astype(object),
        "num": rng.integers(0, 5, n),
    })
    df.loc[0, "ca"] = "?"
    df.loc[1, "thal"] = "?"
    df.loc[2, "cp"] = 9  # invalid code
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # one duplicate row
    return df


def test_columns_renamed_and_ordered(raw_frame):
    df, _ = clean(raw_frame)
    assert list(df.columns) == s.FEATURES + [s.TARGET]


def test_target_is_binary(raw_frame):
    df, _ = clean(raw_frame)
    assert set(df[s.TARGET].unique()) <= {0, 1}
    raw = raw_frame.drop_duplicates().reset_index(drop=True)
    expected = (pd.to_numeric(raw["num"]) > 0).astype(int)
    assert (df[s.TARGET].values == expected.values).all()


def test_question_marks_become_missing(raw_frame):
    df, report = clean(raw_frame)
    assert df["major_vessels"].isna().sum() >= 1
    assert df["thal"].isna().sum() >= 1
    assert "major_vessels" in report["missing_by_column"]


def test_invalid_codes_set_missing(raw_frame):
    df, report = clean(raw_frame)
    assert report["invalid_codes_set_missing"].get("chest_pain") == 1
    assert df["chest_pain"].dropna().isin(s.VALID_CODES["chest_pain"]).all()


def test_duplicates_removed(raw_frame):
    df, report = clean(raw_frame)
    assert report["duplicates_removed"] == 1
    assert len(df) == len(raw_frame) - 1


def test_numeric_columns_have_no_missing(raw_frame):
    df, _ = clean(raw_frame)
    assert df[s.NUMERIC].isna().sum().sum() == 0
