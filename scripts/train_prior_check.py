"""Does Engine 1 prefer training structures? (docs/research-plan.md §3.1)

The GBM rankers were trained on simulations whose true structure is always a training structure, and
in the f<k>np pseudo-tests the answer is one too; hidden-test answers never are. Pool `src`: 1 training
only, 2 COCONUT only, 3 both. In hide runs (every listed candidate is wrong) a ranker without such a
prior puts training structures first about as often as they occur among the listed candidates.

Per run: share of training structures (src 1 or 3) among all candidates in the main list, among main
rank 1, and among V2 rank 1 (pipeline order), counting wrong candidates only.

Usage (WSL): python scripts/train_prior_check.py RUN:MODE [...]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402

POOL = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets" / "casmi26-v2-pool" / "pool_meta.parquet"
src = pd.read_parquet(POOL, columns=["key", "src"]).drop_duplicates("key").set_index("key")["src"]
rows = []
for spec in sys.argv[1:]:
    run, mode = spec.split(":")
    ev = pd.read_parquet(EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet")
    sol = pd.read_csv(EXTERNAL / "pseudo" / f"{mode}_solution.csv")
    ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
    ev = ev[ev.metric_key != ev.molecule_id.map(ans)].copy()
    ev["s"] = ev.metric_key.map(src)
    ev["tr"] = ev.s.isin([1, 3])
    m = ev[ev.main_rank > 0]
    top_main = m.loc[m.groupby("molecule_id").main_rank.idxmin()]
    inpool = ev[ev.s.notna()]
    top_v2 = inpool.loc[inpool.groupby("molecule_id").baseline_rank.idxmin()]
    rows.append(dict(run=run, main_all=m.tr.mean(), main_top1=top_main.tr.mean(),
                     pool_cands_all=inpool.tr.mean(), v2_top1_inpool=top_v2.tr.mean(),
                     n_top=len(top_main)))
print(pd.DataFrame(rows).round(3).to_string(index=False))
