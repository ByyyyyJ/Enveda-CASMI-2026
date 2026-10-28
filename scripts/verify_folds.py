"""Check the community identity-disjoint folds against the current official metric identity.

The folds (dmitriigluzdov/casmi26-identity-disjoint-validation-folds) were keyed by tautomer
canonicalization without the stereo strip the official metric now applies first. This script
recomputes the metric identity for every structure and reports:
  - coverage of train.parquet inchikey14 values by the fold table;
  - metric identities that land in more than one fold (leaks under the current metric);
  - where the 250 enveda-np-examples structures sit.
Writes data/splits/structure_ik14.parquet (inchikey14, smiles, metric_ik14, fold_identity).

Usage: python scripts/verify_folds.py
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from casmi.paths import EXTERNAL, RAW, SPLITS  # noqa: E402

FOLDS = EXTERNAL / "casmi26-identity-disjoint-validation-folds/structures.parquet"
TRAIN = RAW / "train.parquet"
OUT = SPLITS / "structure_ik14.parquet"


def _ik14(smiles):
    from casmi.chem.canon import ik14

    return ik14(smiles)


def main():
    s = pd.read_parquet(FOLDS, columns=["inchikey14", "scorer_key", "smiles", "fold_identity",
                                        "fold_scaffold"])
    with Pool() as pool:
        s["metric_ik14"] = pool.map(_ik14, s["smiles"].tolist(), chunksize=2000)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(OUT, index=False)

    print(f"structures: {len(s):,}; unparseable: {s['metric_ik14'].isna().sum()}")
    print(f"metric_ik14 != scorer_key (stereo-strip effect): "
          f"{(s['metric_ik14'] != s['scorer_key']).sum():,}")
    for col in ("fold_identity", "fold_scaffold"):
        nfold = s.dropna(subset=["metric_ik14"]).groupby("metric_ik14")[col].nunique()
        cross = nfold[nfold > 1]
        n_struct = s["metric_ik14"].isin(cross.index).sum()
        print(f"{col}: metric identities spanning >1 fold: {len(cross):,} ({n_struct:,} structures)")

    train = pd.read_parquet(TRAIN, columns=["inchikey14", "ingest_lib"])
    keys = set(s["inchikey14"])
    missing = set(train["inchikey14"]) - keys
    print(f"train inchikey14 not in fold table: {len(missing):,}")

    np_keys = set(train.loc[train["ingest_lib"] == "enveda-np-examples", "inchikey14"])
    np_rows = s[s["inchikey14"].isin(np_keys)]
    print(f"enveda-np-examples structures: {len(np_keys)}; by fold_identity:")
    print(np_rows.drop_duplicates("inchikey14")["fold_identity"].value_counts().sort_index()
          .to_string())


if __name__ == "__main__":
    main()
