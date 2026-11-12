"""Tune three models with Optuna, track everything in MLflow, pick a champion.

Run from the repo root:
    python -m src.train              # full run (about 3-5 minutes)
    python -m src.train --trials 3   # quick check (under a minute)

Design choices:
  * Optuna maximises mean 5-fold CV ROC-AUC on the training set. Whether to use
    the engineered features is itself a hyperparameter.
  * The decision threshold is tuned on out-of-fold predictions to maximise F2
    (missing a sick patient is worse than a false alarm), subject to a
    precision floor of 0.65 so it does not flag almost everyone.
  * The champion is chosen on CV F2 (ROC-AUC breaks ties). The hold-out test
    set is only used for reporting, never for selection.
"""
import argparse
import json
import logging
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.svm import SVC  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src import settings as s  # noqa: E402
from src.features import build_pipeline, load_xy, split  # noqa: E402

log = logging.getLogger(__name__)

MODEL_NAMES = ["logreg_elasticnet", "svm_rbf", "xgboost"]
THRESHOLDS = np.round(np.arange(0.10, 0.71, 0.01), 2)
MIN_PRECISION = 0.65


# ---------------------------------------------------------------- models
def make_model(name: str, trial):
    """Return an estimator with hyperparameters drawn from the Optuna trial."""
    if name == "logreg_elasticnet":
        return LogisticRegression(
            solver="saga",
            C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            max_iter=5000,
            random_state=s.SEED,
        )
    if name == "svm_rbf":
        svc = SVC(
            kernel="rbf",
            C=trial.suggest_float("C", 1e-2, 50.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            random_state=s.SEED,
        )
        # Platt scaling gives usable probabilities (needed for ROC-AUC and the threshold)
        return CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400, step=25),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            eval_metric="logloss",
            n_jobs=2,
            random_state=s.SEED,
        )
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name: str, trial):
    engineered = trial.suggest_categorical("engineered", [True, False])
    return build_pipeline(make_model(name, trial), engineered=engineered)


# ---------------------------------------------------------------- metrics
def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "f2": fbeta_score(y_true, pred, beta=s.F_BETA),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def tune_threshold(y_true, proba) -> float:
    """Best F2 among thresholds that keep precision >= MIN_PRECISION.

    F2 alone drifts towards flagging almost everyone; the precision floor keeps
    the false-alarm rate clinically usable. Falls back to plain F2 if no
    threshold meets the floor.
    """
    scores = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        f2 = fbeta_score(y_true, pred, beta=s.F_BETA)
        ok = precision_score(y_true, pred, zero_division=0) >= MIN_PRECISION
        scores.append((ok, f2))
    if any(ok for ok, _ in scores):
        best = max(range(len(THRESHOLDS)), key=lambda i: (scores[i][0], scores[i][1]))
    else:
        best = int(np.argmax([f2 for _, f2 in scores]))
    return float(THRESHOLDS[best])


def fold_metrics(y, oof, cv_splits, threshold) -> dict:
    """Mean and std of each metric across the CV folds, from out-of-fold predictions."""
    per_fold = [metrics_at(y.iloc[idx], oof[idx], threshold) for _, idx in cv_splits]
    out = {}
    for key in per_fold[0]:
        vals = [m[key] for m in per_fold]
        out[f"cv_{key}_mean"] = float(np.mean(vals))
        out[f"cv_{key}_std"] = float(np.std(vals))
    return out


# ---------------------------------------------------------------- plots
def save_plots(name, pipe, X_test, y_test, proba_test, y_train, oof, threshold, out: Path):
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax[0], name=name)
    ax[0].plot([0, 1], [0, 1], ls=":", c="grey")
    ax[0].set_title("ROC curve (hold-out test)")
    PrecisionRecallDisplay.from_predictions(y_test, proba_test, ax=ax[1], name=name)
    ax[1].set_title("Precision-recall (hold-out test)")
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba_test >= threshold).astype(int),
        display_labels=["No disease", "Disease"], cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # How precision, recall and F2 move with the threshold (out-of-fold, train set)
    rows = [metrics_at(y_train, oof, t) for t in THRESHOLDS]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key in ("precision", "recall", "f2"):
        ax.plot(THRESHOLDS, [r[key] for r in rows], label=key)
    ax.axvline(threshold, ls="--", c="black", lw=1, label=f"chosen = {threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_title("Threshold tuning on out-of-fold predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "threshold_tuning.png", dpi=120)
    plt.close(fig)

    # Model-agnostic importance on the raw input features
    imp = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=s.SEED, n_jobs=1)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(X_test.columns[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#55A868")
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Permutation feature importance (test)")
    fig.tight_layout()
    fig.savefig(out / "permutation_importance.png", dpi=120)
    plt.close(fig)

    report = classification_report(y_test, (proba_test >= threshold).astype(int),
                                   target_names=["No disease", "Disease"])
    (out / "classification_report.txt").write_text(report)


# ---------------------------------------------------------------- one model
def run_model(name, X_train, y_train, X_test, y_test, trials, cv) -> dict:
    cv_splits = list(cv.split(X_train, y_train))

    with mlflow.start_run(run_name=name) as parent:
        mlflow.set_tags({"model_family": name, "stage": "tuning"})

        def objective(trial):
            pipe = make_pipeline(name, trial)
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
            with mlflow.start_run(run_name=f"{name}-trial-{trial.number:02d}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metrics({"cv_roc_auc_mean": scores.mean(),
                                    "cv_roc_auc_std": scores.std()})
            return scores.mean()

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=s.SEED))
        study.optimize(objective, n_trials=trials)

        best = optuna.trial.FixedTrial(study.best_params)
        pipe = make_pipeline(name, best)

        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = tune_threshold(y_train, oof)
        cv_metrics = fold_metrics(y_train, oof, cv_splits, threshold)

        pipe.fit(X_train, y_train)
        proba_test = pipe.predict_proba(X_test)[:, 1]
        test_raw = metrics_at(y_test, proba_test, threshold)
        test_metrics = {f"test_{k}": v for k, v in test_raw.items()}

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_params({"n_trials": trials, "cv_folds": s.CV_FOLDS,
                           "f_beta": s.F_BETA, "min_precision": MIN_PRECISION})
        mlflow.log_metric("decision_threshold", threshold)
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(test_metrics)

        with tempfile.TemporaryDirectory() as tmp:
            save_plots(name, pipe, X_test, y_test, proba_test,
                       y_train, oof, threshold, Path(tmp))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")
        mlflow.log_dict(study.best_params, "best_params.json")
        with tempfile.TemporaryDirectory() as tmp:
            trials_csv = Path(tmp) / "optuna_trials.csv"
            study.trials_dataframe().to_csv(trials_csv, index=False)
            mlflow.log_artifact(str(trials_csv), "optuna")

        # cloudpickle + code_paths: the pipeline contains our own ClinicalFeatures
        # class, which the default skops format refuses to load
        X_sig = X_train.astype(float)
        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            code_paths=[str(s.ROOT / "src")],
            signature=infer_signature(X_sig, pipe.predict_proba(X_sig)[:, 1]),
            input_example=X_sig.head(3),
        )

        log.info("%-18s CV ROC-AUC %.3f | CV F2 %.3f | CV recall %.3f | threshold %.2f",
                 name, cv_metrics["cv_roc_auc_mean"], cv_metrics["cv_f2_mean"],
                 cv_metrics["cv_recall_mean"], threshold)

        return {
            "model": name,
            "run_id": parent.info.run_id,
            "model_uri": info.model_uri,
            "pipeline": pipe,
            "threshold": threshold,
            "best_params": study.best_params,
            **cv_metrics,
            **test_metrics,
        }


# ---------------------------------------------------------------- orchestration
def train_all(X, y, trials=s.OPTUNA_TRIALS, tracking_uri=s.MLFLOW_TRACKING_URI,
              artifact_root=s.MLFLOW_ARTIFACT_ROOT, models_dir=s.MODELS_DIR,
              report_dir=s.TRAINING_DIR, register=True, models=MODEL_NAMES) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(s.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(s.MLFLOW_EXPERIMENT, artifact_location=artifact_root)
    mlflow.set_experiment(s.MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test = split(X, y)
    cv = StratifiedKFold(n_splits=s.CV_FOLDS, shuffle=True, random_state=s.SEED)

    results = [run_model(m, X_train, y_train, X_test, y_test, trials, cv) for m in models]

    # Champion: best CV F2, ROC-AUC as tie-breaker. Test scores are not used here.
    champion = max(results, key=lambda r: (round(r["cv_f2_mean"], 3), r["cv_roc_auc_mean"]))

    version = None
    if register:
        mv = mlflow.register_model(champion["model_uri"], s.REGISTERED_MODEL)
        version = mv.version
        client = MlflowClient()
        client.set_registered_model_alias(s.REGISTERED_MODEL, s.MODEL_ALIAS, version)
        client.set_model_version_tag(s.REGISTERED_MODEL, version, "threshold",
                                     str(champion["threshold"]))

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion["pipeline"], models_dir / s.MODEL_FILE.name)
    card = {
        "model_name": s.REGISTERED_MODEL,
        "model_family": champion["model"],
        "registry_version": version,
        "mlflow_run_id": champion["run_id"],
        "decision_threshold": champion["threshold"],
        "engineered_features": champion["best_params"].get("engineered"),
        "best_params": champion["best_params"],
        "features": list(X.columns),
        "cv_metrics": _subset(champion, "cv_"),
        "test_metrics": _subset(champion, "test_"),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (models_dir / s.MODEL_CARD_FILE.name).write_text(json.dumps(card, indent=2))

    skip = ("pipeline", "best_params")
    rows = [{k: v for k, v in r.items() if k not in skip} for r in results]
    table = pd.DataFrame(rows).set_index("model")
    save_comparison(table, Path(report_dir))
    return {"champion": champion, "table": table, "card": card}


def _subset(result: dict, prefix: str) -> dict:
    return {k: round(v, 4) for k, v in result.items() if k.startswith(prefix)}


def save_comparison(table: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cols = ["cv_roc_auc_mean", "cv_roc_auc_std", "cv_f2_mean", "cv_recall_mean",
            "cv_precision_mean", "cv_accuracy_mean", "threshold",
            "test_roc_auc", "test_recall", "test_f2"]
    table[cols].round(3).to_csv(out / "model_comparison.csv")

    metrics = ["cv_roc_auc_mean", "cv_f2_mean", "cv_recall_mean", "cv_precision_mean"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    width = 0.8 / len(table)
    for i, (name, row) in enumerate(table.iterrows()):
        errors = [row[m.replace("_mean", "_std")] for m in metrics]
        ax.bar(x + i * width, row[metrics], width, yerr=errors, label=name, capsize=3)
    ax.set_xticks(x + width * (len(table) - 1) / 2)
    ax.set_xticklabels([m.replace("cv_", "").replace("_mean", "") for m in metrics])
    ax.set_ylim(0.5, 1.0)
    ax.set_title("5-fold CV comparison (tuned threshold, error bars = fold std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=s.OPTUNA_TRIALS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for noisy in ("matplotlib", "mlflow", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    X, y = load_xy()
    out = train_all(X, y, trials=args.trials)

    cols = ["cv_roc_auc_mean", "cv_f2_mean", "cv_recall_mean", "cv_precision_mean",
            "threshold", "test_roc_auc", "test_recall"]
    print("\n" + out["table"][cols].round(3).to_string())
    card = out["card"]
    print(f"\nChampion: {card['model_family']} | threshold {card['decision_threshold']} | "
          f"registry version {card['registry_version']}")
    print(f"Saved {s.MODEL_FILE.name} and {s.MODEL_CARD_FILE.name} in models/")


if __name__ == "__main__":
    main()
