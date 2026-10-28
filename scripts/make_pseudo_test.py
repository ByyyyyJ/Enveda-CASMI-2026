"""Build a pseudo competition directory from enveda-np-examples, for running pinned notebooks locally.

The notebook sees a directory that looks like the competition input (test.parquet,
train.parquet, sample_submission.csv). The answers are written outside it.

Modes (which training spectra the notebook may use; see docs/plan-review.md):
  np-lib  drop only the enveda-np-examples spectra. The same molecules keep their spectra from
          other instruments, which simulates class 1 (public spectra exist).
  np-nps  drop every spectrum of those molecules (matched by metric identity), which simulates
          class 2 (known structure, no public spectra). Candidate pools are left unchanged.
  f<k>np  (k = 0..4) harder set: --n natural products (LOTUS or NPAtlas) of identity fold k, not in
          MassSpecGym (ICEBERG/GLACIER training data) and not in the np panel, sampled evenly
          over PubChem-substance-count tertiles. Queries are their train spectra (at most 8 per
          molecule); every spectrum of those molecules is dropped from train, as in np-nps.
          Pair with fold-k-safe model overrides (make_overrides.py f<k>_safe).

Queries follow the official hidden-test cleaning: peaks above precursor + 2 Da are removed, and
spectra whose base peak is below 1,000 counts are dropped.

Usage: python scripts/make_pseudo_test.py MODE [--seed 0]
Writes D:/casmi-data/pseudo/<MODE>/{test.parquet,train.parquet,sample_submission.csv}
   and D:/casmi-data/pseudo/<MODE>_solution.csv (molecule_id, smiles, inchikey14).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL, RAW, SPLITS  # noqa: E402

RDLogger.DisableLog("rdApp.*")

QUERY_LIB = "enveda-np-examples"
# The hidden test's 10 adducts (the np-examples set) -> m/z offset from the neutral monoisotopic mass
H, E = 1.007825032, 0.000548580
ADDUCT_DELTA = {"[M+H]+": H - E, "[M-H]-": -H + E, "[M+CH2O2-H]-": 46.005479 - H + E,
                "[M+NH4]+": 18.034374 - E, "[M+Na]+": 22.989769 - E, "[M-H2O+H]+": -18.010565 + H - E,
                "[M+K]+": 38.963706 - E, "[M-2H2O+H]+": -2 * 18.010565 + H - E,
                "[M+Cl]-": 34.968853 + E, "[M-H2O-H]-": -18.010565 - H + E}
HIDDEN_ADDUCTS = set(ADDUCT_DELTA)
MASS_RANGE = (157.0, 1159.0)  # hidden test monoisotopic masses (official data page)
PREC_PPM = 10.0
TEST_COLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
             "base_peak_intensity", "adduct", "ionization_mode", "instrument_type", "precursor_mz",
             "collision_energy_orig", "collision_energy_ev", "collision_energy_orig_units"]


def clean_peaks(mzs, its, prec):
    keep = np.asarray(mzs) <= prec + 2.0
    return np.asarray(mzs)[keep].tolist(), np.asarray(its)[keep].tolist()


def mono_mass(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return ExactMolWt(m) if m is not None else np.nan


def fold_np_keys(train, fold, n, rng):
    """inchikey14s of the f<fold>np molecules (see module docstring)."""
    ident = pd.read_parquet(SPLITS / "structure_ik14.parquet")
    kt = EXTERNAL / "casmi26-natural-product-knowledge-table"
    npk = pd.read_parquet(kt / "natural_products.parquet",
                          columns=["scorer_key", "in_lotus", "in_npatlas", "pubchem_substances"])
    npk = npk.drop_duplicates("scorer_key")
    npk = npk[npk["in_lotus"].fillna(False) | npk["in_npatlas"].fillna(False)]
    msg = pd.read_csv(EXTERNAL / "massspecgym" / "MassSpecGym.tsv", sep="\t", usecols=["inchikey"])
    msg14 = set(msg["inchikey"].str[:14])
    panel_keys = set(pc.unique(train.filter(pc.equal(train["ingest_lib"], QUERY_LIB))["inchikey14"]).to_pylist())
    panel_ik = set(ident.loc[ident["inchikey14"].isin(panel_keys), "metric_ik14"])
    with_spectra = set(pc.unique(train["inchikey14"]).to_pylist())

    c = ident[(ident["fold_identity"] == fold) &ident["inchikey14"].isin(with_spectra)
              & ~ident["inchikey14"].isin(msg14) & ~ident["metric_ik14"].isin(panel_ik)]
    c = c.merge(npk, left_on="metric_ik14", right_on="scorer_key")
    c = c.drop_duplicates("metric_ik14").sort_values("inchikey14").reset_index(drop=True)
    mass = c["smiles"].map(mono_mass)
    c = c[mass.between(*MASS_RANGE)]
    c["tier"] = pd.qcut(c["pubchem_substances"].fillna(0).rank(method="first"), 3, labels=False)
    picked = [g.sample(n=min(len(g), n // 3), random_state=int(rng.integers(1 << 31)))
              for _, g in c.groupby("tier")]
    picked = pd.concat(picked)
    print(f"f{fold}np: {len(c)} eligible molecules, picked {len(picked)}; PubChem substances "
          f"median {picked['pubchem_substances'].median():.0f}")
    return sorted(picked["inchikey14"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["np-lib", "np-nps"] + [f"f{k}np" for k in range(5)])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=300, help="f<k>np: number of molecules")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    fold_mode = a.mode.startswith("f")

    out = EXTERNAL / "pseudo" / a.mode
    out.mkdir(parents=True, exist_ok=True)
    train = pq.read_table(RAW / "train.parquet")
    if fold_mode:
        keys_k = fold_np_keys(train, int(a.mode[1]), a.n, rng)
        is_query = pc.is_in(train["inchikey14"], value_set=pa.array(keys_k))
    else:
        is_query = pc.equal(train["ingest_lib"], QUERY_LIB)

    q = train.filter(is_query).to_pandas()
    if fold_mode:
        # only the hidden test's 10 adducts (the np-examples set; the pipeline fails on others),
        # then at most 8 spectra per molecule, like the hidden test's small groups
        q = q[q["adduct"].isin(HIDDEN_ADDUCTS)]
        mono = q["normalized_smiles"].map(mono_mass)
        expect = mono + q["adduct"].map(ADDUCT_DELTA)
        q = q[(q["precursor_mz"] - expect).abs() <= expect * PREC_PPM * 1e-6]
        q = q.sample(frac=1.0, random_state=a.seed).groupby("inchikey14").head(8)
    q = q[q["base_peak_intensity"].isna() | (q["base_peak_intensity"] >= 1000)].copy()
    cleaned = [clean_peaks(m, i, p) for m, i, p in
               zip(q["ms2_mzs"], q["ms2_normalized_intensities"], q["precursor_mz"])]
    q["ms2_mzs"] = [c[0] for c in cleaned]
    q["ms2_normalized_intensities"] = [c[1] for c in cleaned]
    q = q[q["ms2_mzs"].map(len) > 0]

    keys = sorted(q["inchikey14"].unique())
    ids = [f"m_{x:06x}" for x in rng.choice(16**6, size=len(keys), replace=False)]
    key2id = dict(zip(keys, ids))
    q["molecule_id"] = q["inchikey14"].map(key2id)
    q["spectrum_id"] = [f"s_{x:08x}" for x in rng.choice(16**8, size=len(q), replace=False)]
    q = q.sample(frac=1.0, random_state=a.seed)

    sol = (q.drop_duplicates("inchikey14")[["molecule_id", "normalized_smiles", "inchikey14"]]
           .rename(columns={"normalized_smiles": "smiles"}).sort_values("molecule_id"))
    sol.to_csv(EXTERNAL / "pseudo" / f"{a.mode}_solution.csv", index=False)

    test = pa.Table.from_pandas(q[TEST_COLS], preserve_index=False)
    pq.write_table(test, out / "test.parquet")
    pd.DataFrame({"molecule_id": sorted(ids), "smiles": "CCO"}).to_csv(
        out / "sample_submission.csv", index=False)

    if a.mode == "np-lib":
        drop = is_query
    else:  # np-nps, f<k>np: drop every spectrum of the query molecules' metric identities
        ident = pd.read_parquet(SPLITS / "structure_ik14.parquet",
                                columns=["inchikey14", "metric_ik14"])
        np_ik = set(ident.loc[ident["inchikey14"].isin(keys), "metric_ik14"])
        drop_keys = ident.loc[ident["metric_ik14"].isin(np_ik), "inchikey14"].unique().tolist()
        drop = pc.is_in(train["inchikey14"], value_set=pa.array(drop_keys))
    kept = train.filter(pc.invert(drop))
    pq.write_table(kept, out / "train.parquet")

    print(f"{a.mode}: {len(keys)} molecules, {len(q)} query spectra; "
          f"train {train.num_rows:,} -> {kept.num_rows:,} rows "
          f"({train.num_rows - kept.num_rows:,} dropped)")


if __name__ == "__main__":
    main()
