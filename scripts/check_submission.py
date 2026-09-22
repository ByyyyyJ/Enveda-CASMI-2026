"""Validate that a proposed submission has the official sample's columns and row count."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "raw" / "sample_submission.csv")
    args = parser.parse_args()
    if args.submission.name != "submission.csv":
        raise SystemExit("The final competition artifact must be named submission.csv")
    sample = pd.read_csv(args.sample)
    submission = pd.read_csv(args.submission)
    if list(submission.columns) != list(sample.columns):
        raise SystemExit(f"Columns differ. Expected {list(sample.columns)}, got {list(submission.columns)}")
    if len(submission) != len(sample):
        raise SystemExit(f"Row count differs. Expected {len(sample)}, got {len(submission)}")
    print("Submission shape and columns match sample_submission.csv")


if __name__ == "__main__":
    main()
