"""Two-way molecule split of enveda-np-examples for ICEBERG fine-tuning with held-out evaluation.

Writes ice_ft/np_train.parquet (np-examples spectra in the ice_ft/train.parquet layout plus `half`, A or B, by a hash
of the metric identity) and ice_ft/panel_nplib_v2_truth_{A,B}.csv (the np-lib panel truth restricted to one half).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL, RAW  # noqa: E402

half = lambda k: "A" if int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 else "B"
cols = ["inchikey14", "normalized_smiles", "adduct", "ionization_mode", "precursor_mz", "ms2_mzs",
        "ms2_normalized_intensities", "collision_energy_ev"]
d = pd.read_parquet(RAW / "train.parquet", columns=cols + ["ingest_lib"], filters=[("ingest_lib", "==", "enveda-np-examples")])
d = d[(d.ionization_mode == "positive") & d.adduct.isin(["[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M-H2O+H]+"])]
d["key"] = [ik14(s) for s in d.normalized_smiles]
d["half"] = d.key.map(half)
d["ce"] = [float(np.mean(v)) if v is not None and len(v) else np.nan for v in d.collision_energy_ev]
d = d[np.isfinite(d.ce)]
d.assign(fold=-2)[["inchikey14", "normalized_smiles", "adduct", "precursor_mz", "ce", "ms2_mzs",
                   "ms2_normalized_intensities", "fold", "half"]].reset_index(drop=True) \
    .to_parquet(EXTERNAL / "ice_ft/np_train.parquet")
print("np-examples training spectra", len(d), d.half.value_counts().to_dict(), d.adduct.value_counts().to_dict())

t = pd.read_csv(EXTERNAL / "ice_ft/panel_nplib_v2_truth.csv")
t["half"] = [half(ik14(s)) for s in t.smiles]
for h in "AB":
    t[t.half == h][["mid", "smiles"]].to_csv(EXTERNAL / f"ice_ft/panel_nplib_v2_truth_{h}.csv", index=False)
print("panel molecules per half", t.half.value_counts().to_dict())
