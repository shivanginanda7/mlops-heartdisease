"""Exploratory analysis: saves figures and a summary to screenshots/eda/.

Run from the repo root after ingest:  python -m src.explore
"""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import settings as s  # noqa: E402

log = logging.getLogger(__name__)

LABELS = {0: "No disease", 1: "Disease"}
COLOURS = {0: "#4C72B0", 1: "#C44E52"}


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path.name)
    return path


def plot_class_balance(df, out_dir):
    counts = df[s.TARGET].value_counts().sort_index()
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.barh([LABELS[i] for i in counts.index], counts.values,
            color=[COLOURS[i] for i in counts.index])
    for i, (n, p) in enumerate(zip(counts.values, pct.values, strict=True)):
        ax.text(n + 2, i, f"{n} ({p:.1f}%)", va="center")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("Patients")
    ax.set_title("Target class balance")
    return _save(fig, out_dir, "01_class_balance.png")


def plot_numeric_by_target(df, out_dir):
    fig, axes = plt.subplots(1, len(s.NUMERIC), figsize=(18, 3.6))
    for ax, col in zip(axes, s.NUMERIC, strict=True):
        for t in (0, 1):
            sns.histplot(df.loc[df[s.TARGET] == t, col], ax=ax, kde=True,
                         stat="density", alpha=0.45, color=COLOURS[t], label=LABELS[t])
        ax.set_title(col)
        ax.set_ylabel("")
    axes[0].legend()
    fig.suptitle("Numeric feature distributions by outcome", y=1.04)
    return _save(fig, out_dir, "02_numeric_hist_by_target.png")


def plot_spearman_heatmap(df, out_dir):
    corr = df[s.FEATURES + [s.TARGET]].astype(float).corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.4, ax=ax)
    ax.set_title("Spearman rank correlation (lower triangle)")
    return _save(fig, out_dir, "03_spearman_correlation.png")


def plot_disease_rate_by_category(df, out_dir):
    cols = s.CATEGORICAL + s.BINARY
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols, strict=True):
        grp = df.groupby(col, dropna=True)[s.TARGET].agg(["mean", "count"])
        ax.bar(grp.index.astype(int).astype(str), grp["mean"] * 100, color="#8172B2")
        for x, (rate, n) in enumerate(zip(grp["mean"], grp["count"], strict=True)):
            ax.text(x, rate * 100 + 2, f"n={n}", ha="center", fontsize=8)
        ax.axhline(df[s.TARGET].mean() * 100, ls="--", c="grey", lw=1)
        ax.set_ylim(0, 110)
        ax.set_title(col)
        ax.set_ylabel("Disease rate (%)")
    fig.suptitle("Disease rate per category level (dashed line = overall rate)", y=1.01)
    fig.tight_layout()
    return _save(fig, out_dir, "04_disease_rate_by_category.png")


def plot_age_vs_max_hr(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in (0, 1):
        sub = df[df[s.TARGET] == t]
        ax.scatter(sub["age"], sub["max_hr"], s=22, alpha=0.7,
                   c=COLOURS[t], label=LABELS[t])
    ages = np.linspace(df["age"].min(), df["age"].max(), 50)
    ax.plot(ages, 220 - ages, ls=":", c="black", label="Age-predicted max (220 - age)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Max heart rate achieved")
    ax.set_title("Exercise heart-rate response vs age")
    ax.legend()
    return _save(fig, out_dir, "05_age_vs_max_hr.png")


def write_summary(df, out_dir):
    lines = ["# EDA summary (auto-generated)", ""]
    lines.append(f"Rows: {len(df)}, features: {len(s.FEATURES)}")
    lines.append("")
    lines.append("## Missing values")
    miss = df.isna().sum()
    for col, n in miss[miss > 0].items():
        lines.append(f"- {col}: {n}")
    lines.append("")
    lines.append("## Numeric features, mean by outcome")
    means = df.groupby(s.TARGET)[s.NUMERIC].mean().T.round(2)
    means.columns = [LABELS[c] for c in means.columns]
    lines.append(means.to_markdown() if _has_tabulate() else means.to_string())
    lines.append("")
    lines.append("## Spearman correlation with target")
    corr = df[s.FEATURES + [s.TARGET]].astype(float).corr(method="spearman")[s.TARGET]
    for col, v in corr.drop(s.TARGET).sort_values(key=abs, ascending=False).items():
        lines.append(f"- {col}: {v:+.2f}")
    path = out_dir / "eda_summary.md"
    path.write_text("\n".join(lines) + "\n")
    log.info("Saved %s", path.name)
    return path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


def run(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    return [
        plot_class_balance(df, out_dir),
        plot_numeric_by_target(df, out_dir),
        plot_spearman_heatmap(df, out_dir),
        plot_disease_rate_by_category(df, out_dir),
        plot_age_vs_max_hr(df, out_dir),
        write_summary(df, out_dir),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_csv(s.CLEAN_FILE)
    files = run(df, s.EDA_DIR)
    log.info("EDA complete: %d files in %s", len(files), s.EDA_DIR.relative_to(s.ROOT))


if __name__ == "__main__":
    main()
