"""Scan "move the PubChem channel's best PubChem-only candidate to position k" on safe and hide runs,
and combine them at several in-pool shares p (docs/research-plan.md §3.1; the pc_last LB probe gives
p of about 0.36).

Usage (WSL): python scripts/pc_promote_scan.py [--folds 0 1 2 3 4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from mix_eval import mix_stats  # noqa: E402
from rerank_cv import rr  # noqa: E402

KS = [1, 2, 3, 4, 6]
PS = [0.6, 0.5, 0.36, 0.2]


def promote(ev, k):
    """Score that keeps V2 order but puts each molecule's best PubChem-only candidate at position k."""
    b = ev["baseline_rank"].to_numpy(float).copy()
    pc = (ev["pubchem_rank"] > 0) & (ev["main_rank"] == 0) & (ev["engine2_rank"] == 0)
    c = ev[pc]
    best = c.loc[c.groupby("molecule_id")["pubchem_rank"].idxmin()].index
    b[best] = np.minimum(b[best], k - 0.5)  # ahead of the candidate now at k, if it was lower
    return -b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    per = {}
    for kind in ("safe", "hide"):
        for f in a.folds:
            run = f"f{f}np_{kind}"
            ev = pd.read_parquet(EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet").reset_index(drop=True)
            sol = pd.read_csv(EXTERNAL / "pseudo" / f"f{f}np_solution.csv")
            ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
            y = (ev["metric_key"] == ev["molecule_id"].map(ans)).astype(int).to_numpy()
            mols = list(sol.molecule_id)
            res = {"V2": rr(ev, y, -ev["baseline_rank"].to_numpy(float), mols)}
            for k in KS:
                res[f"pc@{k}"] = rr(ev, y, promote(ev, k), mols)
            for m, r in res.items():
                per.setdefault(m, {}).setdefault(kind, []).append(r.set_axis(r.index + f"@{f}"))
    per = {m: {k: pd.concat(v) for k, v in d.items()} for m, d in per.items()}
    rows = []
    for m, d in per.items():
        row = dict(method=m, safe=d["safe"].mean(), hide=d["hide"].mean())
        for p in PS:
            _, dl, lo, hi = mix_stats(per["V2"], d, p)
            row[f"d@p={p}"] = f"{dl:+.4f} [{lo:+.3f},{hi:+.3f}]"
        rows.append(row)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
