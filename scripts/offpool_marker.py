"""How strong is the "Engine 2 candidate outside the Engine 1 pool" marker? (docs/research-plan.md §3.1)

In a hidee1 run the hidden answer is listed by Engine 2 but is no longer an Engine 1 pool structure.
Real ChEBI / LIPID MAPS / PubChemLite structures outside COCONUT carry the same marker. Per run:
molecules with at least one such candidate, their mean count, how often the answer is one of them, and the
MRR of the rule "best such candidate first" (rest in V2 order).

Usage (WSL): python scripts/offpool_marker.py RUN:MODE[:HIDDEN_MODE] [...]
  HIDDEN_MODE names pseudo/<HIDDEN_MODE>_hide.txt, removed from the pool for that run; e.g.
  f0np_safe:f0np  f0np_hidee1:f0np:f0np  or a Kaggle test evidence copy as v2test:none
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from rerank_cv import POOL, rr  # noqa: E402

pool = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())
rows = []
for spec in sys.argv[1:]:
    run, mode, *hid = spec.split(":")
    keys = pool - set((EXTERNAL / "pseudo" / f"{hid[0]}_hide.txt").read_text().split()) if hid else pool
    ev = pd.read_parquet(EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet").reset_index(drop=True)
    off = (ev["engine2_rank"] > 0) & ~ev["metric_key"].isin(keys)
    cnt = off.groupby(ev["molecule_id"]).sum()
    row = dict(run=run, mols=len(cnt), with_marker=(cnt > 0).mean(), mean_count=cnt[cnt > 0].mean())
    if mode != "none":
        sol = pd.read_csv(EXTERNAL / "pseudo" / f"{mode}_solution.csv")
        ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
        y = (ev["metric_key"] == ev["molecule_id"].map(ans)).astype(int).to_numpy()
        mols = list(sol.molecule_id)
        b = ev["baseline_rank"].to_numpy(float)
        e2 = ev["engine2_rank"].to_numpy(float)
        best = off & (ev.assign(e=np.where(off, e2, np.inf)).groupby("molecule_id")["e"].transform("min") == e2)
        row.update(answer_marked=float((off.to_numpy() & (y == 1)).sum()) / len(mols),
                   V2=rr(ev, y, -b, mols).mean(), marked_first=rr(ev, y, -(b - 1000 * best), mols).mean())
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
