"""Feature engineering and the preprocessing pipeline.

The same Pipeline object is used for training and for the API, so a single
JSON patient goes through exactly the same steps as the training data.

Run from the repo root to inspect it:  python -m src.features
"""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import settings as s

log = logging.getLogger(__name__)

TEST_SIZE = 0.2
ENGINEERED_NUMERIC = ["pct_max_hr"]
ENGINEERED_BINARY = ["exercise_ischaemia"]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds two features with a clinical rationale.

    pct_max_hr: max_hr / (220 - age), the share of the age-predicted maximum
        heart rate the patient reached. A low value (chronotropic incompetence)
        is a known marker of coronary disease, and it removes the age effect
        that raw max_hr carries.
    exercise_ischaemia: 1 when exercise-induced angina AND ST depression
        above 1 mm occur together, i.e. symptom and ECG sign agree.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).astype(float)
        if not self.enabled:
            return X
        X = X.copy()
        X["pct_max_hr"] = X["max_hr"] / (220 - X["age"])
        X["exercise_ischaemia"] = (
            (X["exercise_angina"] == 1) & (X["st_depression"] > 1.0)
        ).astype(float)
        return X


def build_preprocessor(engineered: bool = True) -> ColumnTransformer:
    """Impute, scale and encode. Fitted on training folds only."""
    numeric = s.NUMERIC + (ENGINEERED_NUMERIC if engineered else [])
    binary = s.BINARY + (ENGINEERED_BINARY if engineered else [])

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric_steps, numeric),
            ("cat", categorical_steps, s.CATEGORICAL),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model, engineered: bool = True) -> Pipeline:
    """Full pipeline: engineered features -> preprocessing -> model."""
    return Pipeline([
        ("features", ClinicalFeatures(enabled=engineered)),
        ("prep", build_preprocessor(engineered)),
        ("model", model),
    ])


def load_xy(path=s.CLEAN_FILE) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df[s.FEATURES], df[s.TARGET]


def split(X, y):
    """Stratified hold-out split; the test set is only used for final reporting."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=s.SEED)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    X, y = load_xy()
    X_train, X_test, y_train, y_test = split(X, y)

    steps = Pipeline([("features", ClinicalFeatures()), ("prep", build_preprocessor())])
    Xt = steps.fit_transform(X_train)
    names = steps.named_steps["prep"].get_feature_names_out()

    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    log.info("Disease rate - train: %.3f | test: %.3f", y_train.mean(), y_test.mean())
    log.info("Features: %d raw -> %d after encoding", X.shape[1], Xt.shape[1])
    log.info("Encoded feature names: %s", ", ".join(names))


if __name__ == "__main__":
    main()
