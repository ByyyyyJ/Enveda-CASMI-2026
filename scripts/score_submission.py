"""Score a submission.csv against a pseudo-test solution with the official-equivalent metric.

Usage: python scripts/score_submission.py SUBMISSION.csv SOLUTION.csv [--ranks OUT.csv] [--vs OTHER_RANKS.csv]
Prints MRR@25 with a bootstrap CI and Hit@1/5/25; --ranks saves per-molecule ranks (0 = miss);
--vs compares against a saved ranks file with a paired bootstrap.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.eval.metrics import answer_ranks, paired_delta, read_submission, summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("solution")
    ap.add_argument("--ranks")
    ap.add_argument("--vs")
    a = ap.parse_args()

    sol = pd.read_csv(a.solution)
    ranks = answer_ranks(dict(zip(sol["molecule_id"], sol["smiles"])), read_submission(a.submission))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary(ranks).items()}))
    if a.ranks:
        ranks.rename_axis("molecule_id").to_csv(a.ranks)
    if a.vs:
        other = pd.read_csv(a.vs, index_col="molecule_id")["rank"]
        d = paired_delta(other, ranks)
        print("vs", a.vs, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in d.items()}))


if __name__ == "__main__":
    main()
