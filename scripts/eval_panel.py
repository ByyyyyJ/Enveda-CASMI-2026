"""Score V2, main_first and re-rankers trained on the f<k>np pseudo-tests on an external panel run
(e.g. enveda-np-examples, whose chemistry matches the hidden test). docs/research-plan.md §3.1.

Usage (WSL): python scripts/eval_panel.py nplib_v2 --mode np-lib [--folds 0 1 2 3 4] [--seeds 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import build_features, main_first_order  # noqa: E402
from mix_eval import TRAIN_SETS, stack  # noqa: E402
from rerank_cv import POOL, VARIANTS, fit_predict, load, paired, rr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", required=True, help="pseudo/<mode>_solution.csv holds the answers")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    pool = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())

    ev = pd.read_parquet(EXTERNAL / "kernel_out" / a.run / "neural_candidate_features.parquet").reset_index(drop=True)
    sol = pd.read_csv(EXTERNAL / "pseudo" / f"{a.mode}_solution.csv")
    ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
    X, groups = build_features(ev, pool)
    y = (ev["metric_key"] == ev["molecule_id"].map(ans)).astype(int).to_numpy()
    mols = list(sol.molecule_id)
    print(f"{a.run}: {len(mols)} molecules, answer in final 25: {pd.Series(y).groupby(ev.molecule_id).max().sum()}")

    train = {}
    for k in a.folds:
        for kind in ("safe", "hide"):
            keys = pool
            if kind == "hide":
                keys = pool - set((EXTERNAL / "pseudo" / f"f{k}np_hide.txt").read_text().split())
            train[k, kind] = load(f"f{k}np_{kind}", keys)

    base = rr(ev, y, -ev["baseline_rank"].to_numpy(float), mols)
    res = {"V2": base, "main_first": rr(ev, y, main_first_order(ev), mols)}
    for var in ["cand", "+gate"]:
        feats = [c for g in VARIANTS[var] for c in groups[g]]
        for tname, kinds in TRAIN_SETS.items():
            Xtr, ytr, gtr = stack([train[k, t] for k in a.folds for t in kinds])
            pred, _ = fit_predict(Xtr[feats], ytr, gtr, X[feats], a.seeds)
            res[f"{var}[{tname}]"] = rr(ev, y, pred, mols)
    rows = []
    for name, r in res.items():
        d, lo, hi = paired(base, r)
        rows.append(dict(method=name, mrr=r.mean(), hit1=(r == 1).mean(), delta=d, lo=lo, hi=hi))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
