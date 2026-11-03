"""Runner-format ICEBERG panel from a pipeline run: each molecule's leader-formula candidates in its final top 25.

Usage: python ice_panel_run.py RUN PANEL   (e.g. nplib_v2 np-lib)
Only molecules whose answer is in that group (and the group has >= 2 members) are kept, so the panel measures how
well the simulator orders the isomers the pipeline actually has to order. Writes ice_ft/panel_{RUN}.json and
ice_ft/panel_{RUN}_truth.csv (truth = the answer's SMILES as it appears in the list).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.eval.metrics import read_submission  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import formula  # noqa: E402

RUN, PANEL = sys.argv[1], sys.argv[2]
test = pd.read_parquet(EXTERNAL / f"pseudo/{PANEL}/test.parquet")
sol = pd.read_csv(EXTERNAL / f"pseudo/{PANEL}_solution.csv")
truth = dict(zip(sol.molecule_id, sol.inchikey14))
lists = read_submission(EXTERNAL / f"kernel_out/{RUN}/submission.csv")
panel, rows = [], []
for mid, cands in lists.items():
    cands = cands[:25]
    f1 = formula(cands[0])
    grp = [c for c in cands if formula(c) == f1]
    hit = [c for c in grp if ik14(c) == truth.get(mid)]
    if len(grp) < 2 or not hit:
        continue
    sp = test[test.molecule_id == mid]
    panel.append({"mid": mid, "cands": grp, "spectra": [
        {"mz": list(map(float, r.ms2_mzs)), "it": list(map(float, r.ms2_normalized_intensities)),
         "prec": float(r.precursor_mz), "adduct": r.adduct, "mode": r.ionization_mode, "instrument": r.instrument_type,
         "ce_units": r.collision_energy_orig_units, "ce_orig": r.collision_energy_orig,
         "ce_ev": None if r.collision_energy_ev is None else list(map(float, r.collision_energy_ev))}
        for r in sp.itertuples()]})
    rows.append((mid, hit[0]))
json.dump(panel, open(EXTERNAL / f"ice_ft/panel_{RUN}.json", "w"))
pd.DataFrame(rows, columns=["mid", "smiles"]).to_csv(EXTERNAL / f"ice_ft/panel_{RUN}_truth.csv", index=False)
print("molecules", len(panel), "median group", pd.Series([len(m["cands"]) for m in panel]).median(),
      "instruments", test.instrument_type.value_counts().to_dict())
