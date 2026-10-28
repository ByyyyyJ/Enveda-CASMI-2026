"""Class-mix sensitivity of the re-ranker (docs/research-plan.md, E4/E5).

For each fold k there are two pseudo-tests on the same molecules: f<k>np_safe (answers in the pool) and
f<k>np_hide (answers hidden from the pool, so only the PubChem channel can find them). Leave one fold out:
train on the other folds, score both runs of the held-out fold. A test with share p of in-pool molecules
has MRR p * MRR(safe) + (1 - p) * MRR(hide); this is reported for several p, with a bootstrap CI of the
paired delta vs V2 (molecules resampled within each part).

In a hide run the candidate pool is the pool minus the hidden answers, so in_pool is computed against that.
Also solves for the in-pool share p at which main_first's local delta matches its public-LB delta.

Usage (WSL): python scripts/mix_eval.py [--folds 0 1 2 3 4] [--lb-delta -0.097] [--seeds 3]
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
from rerank_cv import POOL, VARIANTS, fit_predict, load, rr  # noqa: E402

MIXES = [1.0, 2 / 3, 0.5, 1 / 3, 0.0]
TRAIN_SETS = {"safe": ["safe"], "hide": ["hide"], "mix": ["safe", "hide"]}


def stack(parts):
    """Concatenate (ev, X, y) parts with rows of each molecule contiguous; return X, y, group sizes."""
    X = pd.concat([p[1] for p in parts], ignore_index=True)
    y = np.concatenate([p[2] for p in parts])
    mid = pd.concat([p[0]["molecule_id"] + "@" + str(i) for i, p in enumerate(parts)], ignore_index=True)
    order = np.argsort(mid.to_numpy(), kind="stable")
    g = mid.iloc[order].value_counts(sort=False).reindex(pd.unique(mid.iloc[order])).to_numpy()
    return X.iloc[order], y[order], g


def mix_stats(base, meth, p, n_boot=2000, seed=0):
    """MRR and paired delta (95% CI) at in-pool share p; base/meth are {'safe': rr, 'hide': rr}."""
    rng = np.random.default_rng(seed)
    mrr = sum(w * meth[k].mean() for k, w in (("safe", p), ("hide", 1 - p)) if k in meth)
    d, boot = 0.0, np.zeros(n_boot)
    for k, w in (("safe", p), ("hide", 1 - p)):
        if w == 0 or k not in meth:
            continue
        dk = (meth[k] - base[k]).to_numpy()
        d += w * dk.mean()
        boot += w * dk[rng.integers(0, len(dk), (n_boot, len(dk)))].mean(1)
    return mrr, d, np.quantile(boot, 0.025), np.quantile(boot, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lb-delta", type=float, default=-0.097, help="public-LB delta of main_first vs V2")
    ap.add_argument("--variant", default="+gate", choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    pool = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())

    data = {}  # (fold, kind) -> (ev, X, y, groups, mols)
    for k in a.folds:
        for kind in ("safe", "hide"):
            run = f"f{k}np_{kind}"
            if not (EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet").exists():
                print(f"missing {run}, skipped")
                continue
            keys = pool
            if kind == "hide":
                keys = pool - set((EXTERNAL / "pseudo" / f"f{k}np_hide.txt").read_text().split())
            data[k, kind] = load(run, keys)
    folds = sorted({k for k, _ in data})
    feats = [c for g in VARIANTS[a.variant] for c in next(iter(data.values()))[3][g]]

    per = {}  # method -> kind -> list of per-fold rr series
    for held in folds:
        for kind in ("safe", "hide"):
            if (held, kind) not in data:
                continue
            ev, X, y, _, mols = data[held, kind]
            res = {"V2": rr(ev, y, -ev["baseline_rank"].to_numpy(float), mols),
                   "main_first": rr(ev, y, main_first_order(ev), mols)}
            for tname, kinds in TRAIN_SETS.items():
                parts = [data[f, t] for f in folds if f != held for t in kinds if (f, t) in data]
                if not parts:
                    continue
                Xtr, ytr, gtr = stack(parts)
                pred, _ = fit_predict(Xtr[feats], ytr, gtr, X[feats], a.seeds)
                res[f"rerank[{tname}]"] = rr(ev, y, pred, mols)
            for m, r in res.items():
                per.setdefault(m, {}).setdefault(kind, []).append(r.set_axis(r.index + f"@{held}"))
    per = {m: {k: pd.concat(v) for k, v in d.items()} for m, d in per.items()}

    print(f"folds {folds}, variant {a.variant}")
    for kind in ("safe", "hide"):
        if kind in per["V2"]:
            print(f"  {kind}: n={len(per['V2'][kind])}, "
                  + ", ".join(f"{m} {d[kind].mean():.4f}" for m, d in per.items() if kind in d))
    rows = []
    for p in MIXES:
        for m, d in per.items():
            if p < 1 and "hide" not in d or p > 0 and "safe" not in d:
                continue
            mrr, dl, lo, hi = mix_stats(per["V2"], d, p)
            rows.append(dict(p_in_pool=round(p, 2), method=m, mrr=mrr, delta=dl, lo=lo, hi=hi))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    mf, v2 = per["main_first"], per["V2"]
    if "hide" in mf and "safe" in mf:
        ds = (mf["safe"] - v2["safe"]).mean()
        dh = (mf["hide"] - v2["hide"]).mean()
        p = (a.lb_delta - dh) / (ds - dh) if ds != dh else float("nan")
        print(f"\nmain_first delta: safe {ds:+.4f}, hide {dh:+.4f}; LB {a.lb_delta:+.4f} "
              f"-> implied in-pool share p = {p:.2f}")


if __name__ == "__main__":
    main()
