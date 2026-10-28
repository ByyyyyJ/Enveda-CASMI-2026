"""Leak check for a hide run (docs/research-plan.md, E3): through which lists the hidden answers still reach
the final 25. Answers hidden from the pool should come only from the PubChem channel.

Usage (WSL): python scripts/check_hide.py f0np_hide [...]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402

for run in sys.argv[1:]:
    ev = pd.read_parquet(EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet")
    sol = pd.read_csv(EXTERNAL / "pseudo" / f"{run.split('_')[0]}_solution.csv")
    ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
    hit = ev[ev["metric_key"] == ev["molecule_id"].map(ans)]
    print(f"{run}: {hit.molecule_id.nunique()}/{len(sol)} answers in the final 25")
    for c in ["main_rank", "prefusion_rank", "engine2_rank", "pubchem_rank"]:
        print(f"  in {c}: {(hit[c] > 0).sum()}")
    print("  rank of answer:", hit["baseline_rank"].describe()[["mean", "50%"]].round(2).to_dict())
