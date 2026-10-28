"""Decompose main_first into simple rule-based reorders of V2 and score each on local runs.

  pc_last    PubChem-only candidates (not in the main or Engine 2 lists) moved to the end
  e2_last    Engine-2-only candidates (in neither the main list nor PubChem) moved to the end
  nonmain_last  everything outside the main list moved to the end, V2 order kept (main_first without
             re-sorting by main rank)
  main_first everything in the main list first, in main-list order

Usage (WSL): python scripts/probe_variants.py RUN:MODE [...]   e.g. f0np_safe:f0np nplib_v2:np-lib
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import main_first_order  # noqa: E402
from rerank_cv import paired, rr  # noqa: E402

rows = []
for spec in sys.argv[1:]:
    run, mode = spec.split(":")
    ev = pd.read_parquet(EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet").reset_index(drop=True)
    sol = pd.read_csv(EXTERNAL / "pseudo" / f"{mode}_solution.csv")
    ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
    y = (ev["metric_key"] == ev["molecule_id"].map(ans)).astype(int).to_numpy()
    mols = list(sol.molecule_id)
    b = ev["baseline_rank"].to_numpy(float)
    inm, ine, inp = ev["main_rank"] > 0, ev["engine2_rank"] > 0, ev["pubchem_rank"] > 0
    scores = {
        "V2": -b,
        "pc_last": -(1000 * (inp & ~inm & ~ine) + b),
        "e2_last": -(1000 * (ine & ~inm & ~inp) + b),
        "nonmain_last": -(1000 * ~inm + b),
        "main_first": main_first_order(ev),
    }
    base = rr(ev, y, scores["V2"], mols)
    for name, s in scores.items():
        r = rr(ev, y, s, mols)
        d, lo, hi = paired(base, r)
        rows.append(dict(run=run, method=name, mrr=r.mean(), delta=d, lo=lo, hi=hi))
t = pd.DataFrame(rows)
print(t.round(4).to_string(index=False))
