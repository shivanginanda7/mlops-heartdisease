"""Project-wide paths, column definitions and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
EDA_DIR = ROOT / "screenshots" / "eda"

RAW_FILE = DATA_RAW_DIR / "cleveland_raw.csv"
CLEAN_FILE = DATA_PROCESSED_DIR / "cleveland_clean.csv"
DATA_REPORT_FILE = DATA_PROCESSED_DIR / "data_report.json"

SEED = 2024

# Source: UCI Machine Learning Repository, Heart Disease (Cleveland subset)
UCI_ID = 45
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Original UCI column names, in file order
SOURCE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Readable names used everywhere downstream (training, API, report)
RENAME = {
    "cp": "chest_pain",
    "trestbps": "resting_bp",
    "chol": "cholesterol",
    "fbs": "fasting_sugar",
    "restecg": "rest_ecg",
    "thalach": "max_hr",
    "exang": "exercise_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "ca": "major_vessels",
}

TARGET = "has_disease"

NUMERIC = ["age", "resting_bp", "cholesterol", "max_hr", "st_depression"]
CATEGORICAL = ["chest_pain", "rest_ecg", "st_slope", "major_vessels", "thal"]
BINARY = ["sex", "fasting_sugar", "exercise_angina"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

# Allowed codes per the UCI data dictionary; anything else is treated as missing
VALID_CODES = {
    "sex": {0, 1},
    "chest_pain": {1, 2, 3, 4},
    "fasting_sugar": {0, 1},
    "rest_ecg": {0, 1, 2},
    "exercise_angina": {0, 1},
    "st_slope": {1, 2, 3},
    "major_vessels": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}

# --- Training and tracking (Phase 3) ---
MODELS_DIR = ROOT / "models"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"
TRAINING_DIR = ROOT / "screenshots" / "training"

# Absolute paths, so the MLflow UI finds the same store from any folder
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "heart-disease-optuna"
REGISTERED_MODEL = "heart-risk-classifier"
MODEL_ALIAS = "champion"

CV_FOLDS = 5
OPTUNA_TRIALS = 25
F_BETA = 2  # recall counts twice as much as precision: a missed patient costs more
