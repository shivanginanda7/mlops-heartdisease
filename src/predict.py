"""Score patients from the command line with the saved model.

Examples (from the repo root):
    python -m src.predict sample_patient.json
    python -m src.predict --row 0          # first patient in the cleaned dataset
"""
import argparse
import json

import pandas as pd

from src import settings as s
from src.inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart disease risk prediction")
    parser.add_argument("json_file", nargs="?", help="JSON file: one patient or a list")
    parser.add_argument("--row", type=int, help="Row index in the cleaned dataset")
    args = parser.parse_args()

    if args.json_file:
        with open(args.json_file) as fh:
            data = json.load(fh)
        records = data if isinstance(data, list) else [data]
    elif args.row is not None:
        df = pd.read_csv(s.CLEAN_FILE)
        row = df.iloc[args.row]
        records = [row[s.FEATURES].to_dict()]
        print(f"Actual outcome for row {args.row}: {int(row[s.TARGET])}")
    else:
        parser.error("give a JSON file or --row N")

    predictor = Predictor()
    for record, result in zip(records, predictor.predict(records), strict=True):
        print(json.dumps({"input": record, "output": result}, indent=2, default=str))


if __name__ == "__main__":
    main()
