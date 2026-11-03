"""Replay spectra for ICEBERG fine-tuning: MassSpecGym (the msg_all checkpoint's training data) with its own tokens.

Keeps rows with a numeric collision energy and an Orbitrap / QTOF instrument; writes ice_ft/replay.parquet with the
columns of ice_ft/train.parquet plus `instr`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL  # noqa: E402

m = pd.read_csv(EXTERNAL / "massspecgym/MassSpecGym.tsv", sep="\t")
m = m[np.isfinite(m.collision_energy) & m.instrument_type.isin(["Orbitrap", "QTOF"])]
out = pd.DataFrame({
    "inchikey14": m.inchikey.str[:14], "normalized_smiles": m.smiles, "adduct": m.adduct,
    "precursor_mz": m.precursor_mz.astype(float), "ce": m.collision_energy.astype(float),
    "ms2_mzs": [np.array(s.split(","), float) for s in m.mzs],
    "ms2_normalized_intensities": [np.array(s.split(","), float) for s in m.intensities],
    "fold": -1, "instr": m.instrument_type})
out.reset_index(drop=True).to_parquet(EXTERNAL / "ice_ft/replay.parquet")
print("replay spectra", len(out), out.instr.value_counts().to_dict(), out.adduct.value_counts().to_dict())
