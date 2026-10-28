"""Cross-fold evaluation of the top-25 re-ranker (docs/research-plan.md, E1).

Each run is one pseudo-test (kernel_out/<run>/neural_candidate_features.parquet, answers in
pseudo/<mode>_solution.csv where <mode> is the run name before '_'). Leave one run out: train on the
others, score the held-out one. Reports V2 (pipeline order), main_first and re-rankers built from
cumulative feature groups, each as MRR and paired delta vs V2 with a 95% bootstrap CI.

Usage (WSL, needs lightgbm): python scripts/rerank_cv.py f0np_safe f1np_safe [...] [--seeds 3]
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import build_features, main_first_order  # noqa: E402

POOL = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets" / "casmi26-v2-pool" / "pool_meta.parquet"
VARIANTS = {"cand": ["cand"], "+rel": ["cand", "rel"], "+agree": ["cand", "rel", "agree"],
            "+gate": ["cand", "rel", "agree", "gate"]}


def load(run, pool_keys):
    ev = pd.read_parquet(EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet")
    sol = pd.read_csv(EXTERNAL / "pseudo" / f"{run.split('_')[0]}_solution.csv")
    ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
    ev = ev.reset_index(drop=True)
    X, groups = build_features(ev, pool_keys)
    y = (ev["metric_key"] == ev["molecule_id"].map(ans)).astype(int).to_numpy()
    return ev, X, y, groups, list(sol.molecule_id)


def rr(ev, y, score, mol_ids):
    """Reciprocal rank per molecule when each molecule's rows are sorted by descending score."""
    d = pd.DataFrame({"m": ev["molecule_id"], "y": y, "s": score}).sort_values(["m", "s"], ascending=[True, False])
    d["pos"] = d.groupby("m").cumcount() + 1
    hit = d[d.y == 1].groupby("m")["pos"].min()
    return (1.0 / hit).reindex(mol_ids).fillna(0.0)


def paired(a, b, n_boot=2000):
    d = (b - a).to_numpy()
    boot = d[np.random.default_rng(0).integers(0, len(d), (n_boot, len(d)))].mean(1)
    return d.mean(), np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def fit_predict(Xtr, ytr, gtr, Xte, seeds):
    pred = np.zeros(len(Xte))
    for s in range(seeds):
        m = lgb.LGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=s)
        m.fit(Xtr, ytr, group=gtr)
        pred += m.predict(Xte)
    return pred / seeds, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    pool_keys = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())
    data = {r: load(r, pool_keys) for r in a.runs}
    groups = next(iter(data.values()))[3]
    rows, last = [], None
    for held in a.runs:
        ev, X, y, _, mols = data[held]
        base = rr(ev, y, -ev["baseline_rank"].to_numpy(float), mols)
        res = {"V2": base, "main_first": rr(ev, y, main_first_order(ev), mols)}
        tr = [r for r in a.runs if r != held]
        for name, gs in VARIANTS.items():
            feats = [c for g in gs for c in groups[g]]
            Xtr = pd.concat([data[r][1][feats] for r in tr], ignore_index=True)
            ytr = np.concatenate([data[r][2] for r in tr])
            mid = pd.concat([data[r][0]["molecule_id"] + "@" + r for r in tr], ignore_index=True)
            order = np.argsort(mid.to_numpy(), kind="stable")  # rows of a molecule must be contiguous
            gtr = mid.iloc[order].value_counts(sort=False).reindex(pd.unique(mid.iloc[order])).to_numpy()
            pred, last = fit_predict(Xtr.iloc[order], ytr[order], gtr, X[feats], a.seeds)
            res[name] = rr(ev, y, pred, mols)
        for name, r in res.items():
            d, lo, hi = paired(base, r)
            rows.append(dict(held_out=held, method=name, n=len(r), mrr=r.mean(), delta=d, lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print("\nmean over held-out runs:")
    print(out.groupby("method", sort=False)[["mrr", "delta"]].mean().round(4).to_string())
    imp = pd.Series(last.booster_.feature_importance("gain"), last.booster_.feature_name())
    print("\ntop features (last +gate model, gain):")
    print(imp.sort_values(ascending=False).head(15).round(0).to_string())


if __name__ == "__main__":
    main()
