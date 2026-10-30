"""Re-ranker selection on the LB-calibrated class mix (docs/research-plan.md §3.1).

lb_mixture_fit.py found that the public LB behaves like safe 0.30 / hidee1 0.58 / hide 0.12. Leave one
fold out: train re-rankers on the other folds' runs of the chosen kinds, score every run of the held-out
fold, and report each class's MRR plus the mixture MRR and its paired delta vs V2 (bootstrap within class).

In a hidee1 or hide run the Engine 1 pool lacks the hidden answers, so in_pool is computed against that.

Usage (WSL): python scripts/lb_mix_rerank.py [--variants cand +gate] [--seeds 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import main_first_order  # noqa: E402
from mix_eval import stack  # noqa: E402
from rerank_cv import POOL, VARIANTS, fit_predict, load, rr  # noqa: E402

W = {"safe": 0.30, "hidee1": 0.58, "hide": 0.12}
TRAIN_SETS = {"safe": ["safe"], "hidee1": ["hidee1"], "e1+hide": ["hidee1", "hide"], "all": ["safe", "hidee1", "hide"]}


def mix(base, meth, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    mrr = d = 0.0
    boot = np.zeros(n_boot)
    for k, w in W.items():
        dk = (meth[k] - base[k]).to_numpy()
        mrr += w * meth[k].mean()
        d += w * dk.mean()
        boot += w * dk[rng.integers(0, len(dk), (n_boot, len(dk)))].mean(1)
    return mrr, d, np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["cand", "+gate"], choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    pool = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())

    data = {}
    for k in range(5):
        hidden = set((EXTERNAL / "pseudo" / f"f{k}np_hide.txt").read_text().split())
        for kind in W:
            run = f"f{k}np_{kind}"
            if (EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet").exists():
                data[k, kind] = load(run, pool if kind == "safe" else pool - hidden)
    folds = sorted(k for k in range(5) if all((k, t) in data for t in W))
    print("complete folds:", folds)

    per = {}
    for held in folds:
        for kind in W:
            ev, X, y, groups, mols = data[held, kind]
            res = {"V2": rr(ev, y, -ev["baseline_rank"].to_numpy(float), mols),
                   "main_first": rr(ev, y, main_first_order(ev), mols)}
            for v in a.variants:
                feats = [c for g in VARIANTS[v] for c in groups[g]]
                for tname, kinds in TRAIN_SETS.items():
                    Xtr, ytr, gtr = stack([data[f, t] for f in folds if f != held for t in kinds])
                    pred, _ = fit_predict(Xtr[feats], ytr, gtr, X[feats], a.seeds)
                    res[f"{v}[{tname}]"] = rr(ev, y, pred, mols)
            for m, r in res.items():
                per.setdefault(m, {}).setdefault(kind, []).append(r.set_axis(r.index + f"@{held}"))
        print(f"fold {held} done", flush=True)
    per = {m: {k: pd.concat(v) for k, v in d.items()} for m, d in per.items()}

    rows = []
    for m, d in per.items():
        mrr, dl, lo, hi = mix(per["V2"], d)
        rows.append(dict(method=m, **{k: d[k].mean() for k in W}, lb_mix=mrr, delta=dl, lo=lo, hi=hi))
    pd.set_option("display.width", 200)
    print(f"class weights {W}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
