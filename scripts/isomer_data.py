"""Build compact arrays for the same-formula isomer scorer.

Spectra: the top NPEAK peaks of each training spectrum (m/z, sqrt intensity), capped at CAP spectra
per structure and adduct so the big libraries do not drown the rest. Structures: Morgan count
fingerprints (radius 2, NBIT bits, counts clipped at 255). Folds come from the community
identity-disjoint split (fold_identity), so a held-out fold shares no metric identity with training.

Output: D:/casmi-data/isomer/{spectra.npz, structures.parquet, fp.npy}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL, RAW  # noqa: E402

RDLogger.DisableLog("rdApp.*")
NPEAK, CAP, NBIT = 64, 16, 2048
OUT = EXTERNAL / "isomer"
OUT.mkdir(exist_ok=True)

folds = pd.read_parquet(EXTERNAL / "casmi26-identity-disjoint-validation-folds/structures.parquet",
                        columns=["inchikey14", "smiles", "formula", "fold_identity"])
folds = folds.drop_duplicates("inchikey14").reset_index(drop=True)
print("structures", len(folds), flush=True)

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBIT)
fp = np.zeros((len(folds), NBIT), np.uint8)
ok = np.zeros(len(folds), bool)
for i, smi in enumerate(folds.smiles):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    fp[i] = np.minimum(gen.GetCountFingerprintAsNumPy(m), 255)
    ok[i] = True
    if i % 50000 == 0:
        print("fp", i, flush=True)
folds["fp_ok"] = ok
np.save(OUT / "fp.npy", fp)
folds.to_parquet(OUT / "structures.parquet")
sid = pd.Series(np.arange(len(folds)), index=folds.inchikey14)

cols = ["ingest_lib", "inchikey14", "adduct", "instrument_type", "precursor_mz", "ms2_mzs",
        "ms2_normalized_intensities", "collision_energy_ev"]
df = pd.read_parquet(RAW / "train.parquet", columns=cols)
df = df[df.inchikey14.isin(sid.index)]
df = df.sample(frac=1.0, random_state=0)
df = df[df.groupby(["inchikey14", "adduct"]).cumcount() < CAP].reset_index(drop=True)
print("spectra kept", len(df), flush=True)

n = len(df)
mz = np.zeros((n, NPEAK), np.float32)
it = np.zeros((n, NPEAK), np.float16)
for i, (m, a, p) in enumerate(zip(df.ms2_mzs, df.ms2_normalized_intensities, df.precursor_mz)):
    m = np.asarray(m, np.float32)
    a = np.asarray(a, np.float32)
    keep = m < p - 1.6
    m, a = m[keep], a[keep]
    if len(m) > NPEAK:
        top = np.argpartition(-a, NPEAK)[:NPEAK]
        m, a = m[top], a[top]
    if len(a):
        a = np.sqrt(a / a.max())
    mz[i, :len(m)] = m
    it[i, :len(m)] = a

adducts = sorted(df.adduct.unique())
ce = df.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
np.savez(OUT / "spectra.npz", mz=mz, it=it,
         prec=df.precursor_mz.to_numpy(np.float32),
         adduct=df.adduct.map({a: i for i, a in enumerate(adducts)}).to_numpy(np.int16),
         tims=(df.instrument_type == "timsTOF").to_numpy(),
         ce=ce.to_numpy(np.float32),
         sid=sid.loc[df.inchikey14].to_numpy(np.int32),
         lib=df.ingest_lib.to_numpy(str))
pd.Series(adducts).to_csv(OUT / "adducts.csv", index=False)
print("done", OUT, flush=True)
