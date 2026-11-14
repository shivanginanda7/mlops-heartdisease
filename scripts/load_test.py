"""Send realistic traffic to the API so the Grafana dashboard has data.

Patients are sampled from the cleaned UCI data, about 5% of requests are
deliberately invalid (to show the error-rate panel), and --drift shifts the
population towards older, sicker patients to move the drift signal.

Examples (from the repo root):
    python scripts/load_test.py                          # Kubernetes ingress, 2 minutes
    python scripts/load_test.py --drift --seconds 90     # simulate data drift
    python scripts/load_test.py --url http://127.0.0.1:8000 --host ""   # local uvicorn
"""
import argparse
import csv
import json
import math
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleveland_clean.csv"
FEATURES = ["age", "sex", "chest_pain", "resting_bp", "cholesterol", "fasting_sugar",
            "rest_ecg", "max_hr", "exercise_angina", "st_depression", "st_slope",
            "major_vessels", "thal"]
INTEGER = {"age", "sex", "chest_pain", "fasting_sugar", "rest_ecg", "exercise_angina",
           "st_slope", "major_vessels", "thal"}


def load_patients() -> list[dict]:
    patients = []
    with open(DATA) as fh:
        for row in csv.DictReader(fh):
            p = {}
            for f in FEATURES:
                value = row[f]
                if value == "" or math.isnan(float(value)):
                    p[f] = None
                else:
                    p[f] = int(float(value)) if f in INTEGER else float(value)
            patients.append(p)
    return patients


def drifted(p: dict) -> dict:
    """An older population with more exercise-induced ischaemia."""
    q = dict(p)
    q["age"] = min(q["age"] + 12, 100)
    q["max_hr"] = max(q["max_hr"] - 20, 60)
    q["st_depression"] = min(q["st_depression"] + 1.5, 7)
    q["exercise_angina"] = 1
    q["chest_pain"] = 4
    return q


def post(url: str, host: str, payload: dict) -> int:
    headers = {"Content-Type": "application/json"}
    if host:
        headers["Host"] = host
    req = urllib.request.Request(f"{url}/v1/predict", data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except (urllib.error.URLError, TimeoutError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic generator for the heart API")
    parser.add_argument("--url", default="http://127.0.0.1")
    parser.add_argument("--host", default="heart-api.local",
                        help="Host header (\"\" for none)")
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--rps", type=float, default=8.0, help="Requests per second")
    parser.add_argument("--drift", action="store_true", help="Send a shifted population")
    parser.add_argument("--invalid-share", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(7)
    patients = load_patients()
    statuses = Counter()
    end = time.time() + args.seconds
    print(f"Sending ~{args.rps} req/s for {args.seconds}s to {args.url} "
          f"({'drifted' if args.drift else 'normal'} population)")

    while time.time() < end:
        payload = random.choice(patients)
        if args.drift:
            payload = drifted(payload)
        if random.random() < args.invalid_share:
            payload = {**payload, "age": 5}  # rejected with 422 by the schema
        statuses[post(args.url, args.host, payload)] += 1
        time.sleep(1 / args.rps)

        sent = sum(statuses.values())
        if sent % 100 == 0:
            print(f"  {sent} requests sent: {dict(statuses)}")

    print(f"Done. Status codes: {dict(statuses)}")
    if statuses.get(0):
        print("Some requests got no answer: is `minikube tunnel` running?")


if __name__ == "__main__":
    main()
