"""Fit the public LB as a mixture of local pseudo-test classes (docs/research-plan.md §3.1).

Classes: safe (answer in every store), hidee1 (answer hidden from the Engine 1 pool only), hide (hidden
from every store) and miss (answer in no candidate list: MRR 0 under every variant). For each class the
mean V2 MRR and each probe variant's delta vs V2 are measured on the f<k>np runs that exist; a grid over
the class shares (step 0.02, summing to 1) is then fitted to the LB numbers: V2 0.468 and the probe deltas.

Usage (WSL): python scripts/lb_mixture_fit.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import main_first_order  # noqa: E402
from pc_promote_scan import promote  # noqa: E402
from rerank_cv import POOL, rr  # noqa: E402

KINDS = ["safe", "hidee1", "hide"]
VARIANTS = ["pc_last", "pc_at2", "e2_last", "main_first", "e2off_first"]
LB = {"V2": 0.468, "pc_last": -0.041, "pc_at2": -0.009, "e2_last": 0.001, "main_first": -0.097,
      "e2off_first": -0.128}


def variant_scores(ev, pool_keys):
    b = ev["baseline_rank"].to_numpy(float)
    off = (ev["engine2_rank"] > 0) & ~ev["metric_key"].isin(pool_keys)
    e2 = ev["engine2_rank"].to_numpy(float)
    best = off & (ev.assign(e=np.where(off, e2, np.inf)).groupby("molecule_id")["e"].transform("min") == e2)
    inm, ine, inp = ev["main_rank"] > 0, ev["engine2_rank"] > 0, ev["pubchem_rank"] > 0
    return {"V2": -b,
            "pc_last": -(1000 * (inp & ~inm & ~ine) + b),
            "pc_at2": promote(ev, 2),
            "e2_last": -(1000 * (ine & ~inm & ~inp) + b),
            "main_first": main_first_order(ev),
            "e2off_first": -(b - 1000 * best)}


def class_stats():
    pool = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())
    out = {}
    for kind in KINDS:
        parts = []
        for k in range(5):
            p = EXTERNAL / "kernel_out" / f"f{k}np_{kind}" / "neural_candidate_features.parquet"
            if not p.exists():
                continue
            ev = pd.read_parquet(p).reset_index(drop=True)
            sol = pd.read_csv(EXTERNAL / "pseudo" / f"f{k}np_solution.csv")
            ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
            y = (ev["metric_key"] == ev["molecule_id"].map(ans)).astype(int).to_numpy()
            mols = list(sol.molecule_id)
            keys = pool if kind == "safe" else pool - set((EXTERNAL / "pseudo" / f"f{k}np_hide.txt").read_text().split())
            parts.append(pd.DataFrame({n: rr(ev, y, s, mols) for n, s in variant_scores(ev, keys).items()}).reset_index(drop=True))
        d = pd.concat(parts)
        row = {"folds": len(parts), "n": len(d), "V2": d["V2"].mean()}
        row.update({v: (d[v] - d["V2"]).mean() for v in VARIANTS})
        out[kind] = row
    out["miss"] = {"folds": 0, "n": 0, "V2": 0.0, **{v: 0.0 for v in VARIANTS}}
    return pd.DataFrame(out).T


def main():
    st = class_stats()
    print(st.round(4).to_string())
    cols = ["V2"] + VARIANTS
    M = st[cols].to_numpy(float)
    target = np.array([LB[c] for c in cols])
    grid = np.round(np.arange(0, 1.0001, 0.02), 2)
    fits = []
    for w in itertools.product(grid, repeat=3):
        if sum(w) > 1 + 1e-9:
            continue
        w = np.array([*w, 1 - sum(w)])
        pred = w @ M
        fits.append((float(np.sqrt(((pred - target) ** 2).mean())), *w, *pred))
    f = pd.DataFrame(fits, columns=["rmse", *st.index, *[f"pred_{c}" for c in cols]]).sort_values("rmse")
    print("\nLB target:", {c: LB[c] for c in cols})
    print("best mixtures:")
    print(f.head(10).round(3).to_string(index=False))
    for drop in KINDS:
        g = f[f[drop] == 0].head(1)
        print(f"\nbest without {drop}:")
        print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
