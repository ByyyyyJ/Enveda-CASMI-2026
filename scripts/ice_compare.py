"""Paired comparison of two ice_eval.py runs: per-molecule rank of the true structure, wins and losses.

Usage: python ice_compare.py TAG_A TAG_B [TRUTH.csv]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL  # noqa: E402

a, b = sys.argv[1], sys.argv[2]
truth = dict(pd.read_csv(sys.argv[3] if len(sys.argv) > 3 else EXTERNAL / "ice_ft/panel_truth.csv").values)


def ranks(tag):
    sc = json.load(open(EXTERNAL / f"ice_ft/scores_{tag}.json"))
    out = {}
    for mid, d in sc.items():
        if mid not in truth:
            continue
        t = d.get(truth[mid])
        if t is not None:
            out[mid] = 1 + sum(v > t for v in d.values() if v is not None)
    return pd.Series(out)


ra, rb = ranks(a), ranks(b)
r = pd.concat([ra, rb], axis=1, keys=[a, b]).dropna()
top_a, top_b = r[a] == 1, r[b] == 1
d = 1 / r[b] - 1 / r[a]
rng = np.random.default_rng(0)
boot = [d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
print(f"{len(r)} molecules | top1 {a} {top_a.mean():.3f}  {b} {top_b.mean():.3f} | "
      f"top1 gained {int((top_b & ~top_a).sum())} lost {int((top_a & ~top_b).sum())} | "
      f"rank better {int((r[b] < r[a]).sum())} worse {int((r[b] > r[a]).sum())} | "
      f"MRR diff {d.mean():+.4f} (95% CI {np.percentile(boot, 2.5):+.4f} {np.percentile(boot, 97.5):+.4f})")
