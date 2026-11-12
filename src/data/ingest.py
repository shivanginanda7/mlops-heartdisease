"""Download the UCI Cleveland heart disease data, clean it and save it.

Run from the repo root:  python -m src.data.ingest
"""
import json
import logging

import pandas as pd

from src import settings as s

log = logging.getLogger(__name__)


def fetch_raw() -> tuple[pd.DataFrame, str]:
    """Fetch the raw data. Tries the ucimlrepo API first, then the direct file."""
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=s.UCI_ID)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        source = f"ucimlrepo (id={s.UCI_ID})"
    except Exception as exc:  # network or API change: fall back to the raw file
        log.warning("ucimlrepo failed (%s); using direct UCI file", exc)
        df = pd.read_csv(s.UCI_URL, header=None, names=s.SOURCE_COLUMNS)
        source = s.UCI_URL
    return df[s.SOURCE_COLUMNS].copy(), source


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned frame and a small report of what was changed."""
    df = raw.copy()

    # '?' marks missing values in the original file; coerce everything to numbers
    df = df.apply(pd.to_numeric, errors="coerce")

    rows_in = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates = rows_in - len(df)

    df = df.rename(columns=s.RENAME)

    # num is 0 (no disease) to 4 (severe); the task is presence vs absence
    df[s.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    # Codes outside the data dictionary are treated as missing, not guessed
    invalid = {}
    for col, allowed in s.VALID_CODES.items():
        bad = df[col].notna() & ~df[col].isin(allowed)
        invalid[col] = int(bad.sum())
        df.loc[bad, col] = pd.NA

    for col in s.CATEGORICAL + s.BINARY:
        df[col] = df[col].astype("Int64")

    # Missing values are kept as NaN on purpose: they are imputed inside the
    # training pipeline so the same logic runs at prediction time.
    report = {
        "rows_raw": rows_in,
        "duplicates_removed": duplicates,
        "rows_clean": len(df),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        "invalid_codes_set_missing": {k: v for k, v in invalid.items() if v},
        "class_counts": {
            "no_disease": int((df[s.TARGET] == 0).sum()),
            "disease": int((df[s.TARGET] == 1).sum()),
        },
    }
    return df[s.FEATURES + [s.TARGET]], report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    s.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw, source = fetch_raw()
    raw.to_csv(s.RAW_FILE, index=False)
    log.info("Fetched %d rows from %s", len(raw), source)

    df, report = clean(raw)
    report["source"] = source
    df.to_csv(s.CLEAN_FILE, index=False)
    s.DATA_REPORT_FILE.write_text(json.dumps(report, indent=2))

    log.info("Saved %s", s.CLEAN_FILE.relative_to(s.ROOT))
    log.info("Report: %s", json.dumps(report))


if __name__ == "__main__":
    main()
