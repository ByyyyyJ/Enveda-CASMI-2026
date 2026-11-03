"""Data for fine-tuning ICEBERG's intensity model on timsTOF spectra and for judging it on isomer ranking.

Writes to D:/casmi-data/ice_ft/:
  panel_f0.json   runner-format molecules from identity fold 0 (enveda-180 timsTOF, positive): true structure plus
                  the NNEG most similar same-formula structures (Morgan Tanimoto) from the extended structure table
  panel_truth.csv molecule id -> true SMILES
  train.parquet   training spectra from the other folds (enveda-180 timsTOF, positive, ICEBERG adducts)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL, RAW  # noqa: E402

NMOL, NSPEC, NNEG, NTRAIN = 300, 4, 15, 120000
ICE_ADDUCTS = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M+NH4]+", "[M-2H2O+H]+"]
OUT = EXTERNAL / "ice_ft"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

st = pd.read_parquet(EXTERNAL / "isomer/structures_ext.parquet")
fp = np.load(EXTERNAL / "isomer/fp_ext.npy", mmap_mode="r")
fold = dict(zip(st.inchikey14, st.fold_identity))
cols = ["ingest_lib", "inchikey14", "normalized_smiles", "adduct", "ionization_mode", "instrument_type", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev", "collision_energy_orig",
        "collision_energy_orig_units"]
df = pd.read_parquet(RAW / "train.parquet", columns=cols,
                     filters=[("ingest_lib", "==", "enveda-180"), ("ionization_mode", "==", "positive")])
df = df[df.adduct.isin(ICE_ADDUCTS)]
df["fold"] = df.inchikey14.map(fold)
print("enveda-180 positive ICEBERG-adduct spectra", len(df), df.adduct.value_counts().to_dict(), flush=True)
print("CE units", df.collision_energy_orig_units.value_counts().head().to_dict(), flush=True)

# ---- panel: fold 0, [M+H]+ / [M+Na]+ only (the adducts the shipped model covers), so both models are comparable
p = df[(df.fold == 0) & df.adduct.isin(["[M+H]+", "[M+Na]+"])]
sid = pd.Series(np.arange(len(st)), index=st.inchikey14)
form = st.formula.to_numpy()
groups = st.groupby("formula").indices
mols = rng.permutation(p.inchikey14.unique())
panel, truth = [], []
for ik in mols:
    s = sid[ik]
    g = groups[form[s]]
    g = g[(g != s) & st.fp_ok.to_numpy()[g]]
    if len(g) < NNEG:
        continue
    a = fp[s] > 0
    b = fp[g] > 0
    sim = (b & a).sum(1) / np.maximum((b | a).sum(1), 1)
    neg = g[np.argsort(-sim)[:NNEG]]
    sp = p[p.inchikey14 == ik].head(NSPEC)
    mid = f"f0_{ik}"
    panel.append({"mid": mid, "cands": [st.smiles[s]] + st.smiles.iloc[neg].tolist(), "spectra": [
        {"mz": list(map(float, r.ms2_mzs)), "it": list(map(float, r.ms2_normalized_intensities)),
         "prec": float(r.precursor_mz), "adduct": r.adduct, "mode": 1, "instrument": r.instrument_type,
         "ce_units": r.collision_energy_orig_units, "ce_orig": r.collision_energy_orig,
         "ce_ev": None if r.collision_energy_ev is None else list(map(float, r.collision_energy_ev))}
        for r in sp.itertuples()]})
    truth.append((mid, st.smiles[s]))
    if len(panel) == NMOL:
        break
json.dump(panel, open(OUT / "panel_f0.json", "w"))
pd.DataFrame(truth, columns=["mid", "smiles"]).to_csv(OUT / "panel_truth.csv", index=False)
print("panel molecules", len(panel), "spectra", sum(len(m["spectra"]) for m in panel), flush=True)

# ---- training spectra: other folds (fold 0 and the candidate-only structures excluded)
t = df[df.fold.isin([1, 2, 3, 4])]
t = t.sample(frac=1.0, random_state=0)
t = t[t.groupby(["inchikey14", "adduct"]).cumcount() < 4].head(NTRAIN)
t = t.assign(ce=[float(np.mean(v)) if v is not None and len(v) else np.nan for v in t.collision_energy_ev])
t = t[np.isfinite(t.ce)]
t[["inchikey14", "normalized_smiles", "adduct", "precursor_mz", "ce", "ms2_mzs", "ms2_normalized_intensities", "fold"]] \
    .reset_index(drop=True).to_parquet(OUT / "train.parquet")
print("train spectra", len(t), t.adduct.value_counts().to_dict(), "structures", t.inchikey14.nunique(), flush=True)
