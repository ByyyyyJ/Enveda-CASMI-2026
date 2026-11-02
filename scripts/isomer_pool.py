"""Extend the isomer-scorer structure table with candidate-database structures (hard negatives).

Adds the structures of the pipeline's candidate pools (v2 pool, PubChemLite, ChEBI/LIPID MAPS) whose formula
occurs among the training structures and whose metric identity is not a training structure. They get
fold_identity = -1 and have no spectra, so they only ever appear as negatives.

Output: D:/casmi-data/isomer/{structures_ext.parquet, fp_ext.npy}
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.isomer import morgan_counts  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import formula  # noqa: E402

A = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets"
OUT = EXTERNAL / "isomer"
st = pd.read_parquet(OUT / "structures.parquet")
fp = np.load(OUT / "fp.npy")
known_f = set(st.formula)

pool = pd.read_parquet(A / "casmi26-v2-pool/pool_meta.parquet", columns=["smiles", "formula"])
with open(A / "casmi26-pubchemlite-cc0-candidate-fingerprints/pubchemlite_meta.pkl", "rb") as f:
    pcl = pickle.load(f)
bio = pd.read_parquet(A / "casmi-chebi-lmsd-structures/bio_structures.parquet", columns=["smiles"])
extra = pd.concat([pd.DataFrame({"smiles": pcl["smiles"]}), bio], ignore_index=True)
extra["formula"] = [formula(s) for s in extra.smiles]
cand = pd.concat([pool, extra], ignore_index=True)
cand = cand[cand.formula.isin(known_f)].drop_duplicates("smiles")
print("candidates with a training formula", len(cand), flush=True)

cand["inchikey14"] = [ik14(s) for s in cand.smiles]
cand = cand.dropna(subset=["inchikey14"]).drop_duplicates("inchikey14")
cand = cand[~cand.inchikey14.isin(set(st.inchikey14))].reset_index(drop=True)
print("new identities", len(cand), flush=True)

fps = np.zeros((len(cand), fp.shape[1]), np.uint8)
ok = np.zeros(len(cand), bool)
for i, s in enumerate(cand.smiles):
    v = morgan_counts(s)
    if v is not None:
        fps[i] = v; ok[i] = True
    if i % 100000 == 0:
        print("fp", i, flush=True)
cand = cand.assign(fold_identity=-1, fp_ok=ok)
ext = pd.concat([st, cand[["inchikey14", "smiles", "formula", "fold_identity", "fp_ok"]]], ignore_index=True)
ext.to_parquet(OUT / "structures_ext.parquet")
np.save(OUT / "fp_ext.npy", np.concatenate([fp, fps]))
print("done", len(ext), flush=True)
