"""Leak hunt: MRR when each molecule's final 25 is sorted by a single feature (both directions, best kept).
A feature that alone beats the pipeline by a wide margin on hide runs is suspect.

Usage (WSL): python scripts/single_feature_mrr.py f0np_hide [...] [--pool-minus-hide]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL  # noqa: E402
from rerank_cv import POOL, load, rr  # noqa: E402

pool = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())
rows = []
for run in [a for a in sys.argv[1:] if not a.startswith("--")]:
    keys = pool
    if run.endswith("_hide"):
        keys = pool - set((EXTERNAL / "pseudo" / f"{run.split('_')[0]}_hide.txt").read_text().split())
    ev, X, y, groups, mols = load(run, keys)
    rng = np.random.default_rng(0).random(len(X)) * 1e-6  # random tie-break
    for c in X.columns:
        v = X[c].fillna(X[c].min() - 1).to_numpy(float)
        best = max(rr(ev, y, v + rng, mols).mean(), rr(ev, y, -v + rng, mols).mean())
        rows.append(dict(run=run, feature=c, mrr=best))
t = pd.DataFrame(rows).pivot(index="feature", columns="run", values="mrr")
t["mean"] = t.mean(1)
print(t.sort_values("mean", ascending=False).head(25).round(3).to_string())
