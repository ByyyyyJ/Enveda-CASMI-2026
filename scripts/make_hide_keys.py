"""Write the metric identities of a pseudo-test's answers, for the hide notebook (CASMI_HIDE_KEYS).

Usage: python scripts/make_hide_keys.py f0np [f1np ...]
Writes D:/casmi-data/pseudo/<mode>_hide.txt and reports how many answers are in the pool.
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402

POOL = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets" / "casmi26-v2-pool" / "pool_meta.parquet"


def main():
    pool = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())
    for mode in sys.argv[1:]:
        sol = pd.read_csv(EXTERNAL / "pseudo" / f"{mode}_solution.csv")
        keys = sorted({k for k in sol.smiles.map(ik14) if k})
        (EXTERNAL / "pseudo" / f"{mode}_hide.txt").write_text("\n".join(keys) + "\n")
        print(f"{mode}: {len(keys)} answer identities, {sum(k in pool for k in keys)} in the pool")


if __name__ == "__main__":
    main()
