"""Where does a pipeline run lose MRR? Per-molecule error buckets for a pseudo-test run.

Buckets (each molecule falls in exactly one):
  hit@1        answer ranked first
  isomer       answer in the top 25, top-1 guess has the answer's formula (same-formula ranking)
  formula      answer in the top 25, top-1 guess has another formula
  lost         answer not in the top 25 but in an intermediate list (Engine 2 top 40 or the
               PubChem-channel top 25): dropped by fusion / reranking
  unsurfaced   answer in the candidate pool but in no saved list: retrieval/scoring failure
  no-candidate answer not in the candidate pool (PubChem-only or truly new structures land here,
               unless the PubChem channel found them)
"Lost MRR" is the sum of (1 - reciprocal rank) over the bucket, divided by the number of molecules,
so the buckets add up to 1 - MRR.

Usage: python scripts/decompose_errors.py RUN_DIR SOLUTION.csv [--sub submission.csv] [--out per_molecule.csv]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.eval.metrics import answer_ranks, read_submission  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402

RDLogger.DisableLog("rdApp.*")
POOL = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets" / "casmi26-v2-pool" / "pool_meta.parquet"
NPK = EXTERNAL / "casmi26-natural-product-knowledge-table" / "natural_products.parquet"


def formula(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    try:
        # neutral formula without charge suffix, so salts/zwitterions written differently still match
        return CalcMolFormula(m).rstrip("+-")
    except Exception:
        return None


def list_rank(key, smiles):
    for r, s in enumerate(smiles, 1):
        if ik14(s) == key:
            return r
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("solution")
    ap.add_argument("--sub", default="submission.csv")
    ap.add_argument("--out")
    a = ap.parse_args()
    run = Path(a.run_dir)

    sol = pd.read_csv(a.solution)
    preds = read_submission(run / a.sub)
    ranks = answer_ranks(dict(zip(sol["molecule_id"], sol["smiles"])), preds)
    eng = json.loads((run / "eng_lists.json").read_text()) if (run / "eng_lists.json").exists() else {}
    pcl = json.loads((run / "pc_lists.json").read_text()) if (run / "pc_lists.json").exists() else {}

    pool_keys = set(pq.read_table(POOL, columns=["key"])["key"].to_pylist())
    npk = (pd.read_parquet(NPK, columns=["scorer_key", "pubchem_substances", "np_pathway"])
           .drop_duplicates("scorer_key").set_index("scorer_key"))

    rows = []
    for mid, smi in zip(sol["molecule_id"], sol["smiles"]):
        key, f = ik14(smi), formula(smi)
        guesses = preds.get(mid, [])[:25]
        top1_f = formula(guesses[0]) if guesses else None
        r = int(ranks[mid])
        e_r = list_rank(key, eng.get(mid, {}).get("smiles", []))
        p_r = list_rank(key, pcl.get(mid, {}).get("pc", []))
        if r == 1:
            b = "hit@1"
        elif r > 1:
            b = "isomer" if top1_f == f else "formula"
        elif e_r or p_r:
            b = "lost"
        elif key in pool_keys:
            b = "unsurfaced"
        else:
            b = "no-candidate"
        n_same_f = sum(formula(g) == f for g in guesses)
        rows.append(dict(molecule_id=mid, rank=r, rr=1 / r if r else 0.0, bucket=b, eng_rank=e_r,
                         pc_rank=p_r, in_pool=key in pool_keys, top1_formula_ok=top1_f == f,
                         n_same_formula_top25=n_same_f, formula=f,
                         pubchem_substances=npk["pubchem_substances"].get(key, np.nan),
                         np_pathway=npk["np_pathway"].get(key, None)))
    d = pd.DataFrame(rows)
    n = len(d)

    order = ["hit@1", "isomer", "formula", "lost", "unsurfaced", "no-candidate"]
    t = d.groupby("bucket").agg(n=("rr", "size"), lost_mrr=("rr", lambda x: (1 - x).sum() / n))
    t = t.reindex(order).fillna(0)
    t["share"] = t["n"] / n
    print(f"{run.name} / {a.sub}: n={n} MRR={d.rr.mean():.4f}")
    print(t.round(4).to_string())
    print(f"top-1 formula correct: {d.top1_formula_ok.mean():.3f}; answer in pool: {d.in_pool.mean():.3f}; "
          f"in Engine 2 top 40: {(d.eng_rank > 0).mean():.3f}; in PubChem-channel list: {(d.pc_rank > 0).mean():.3f}")
    if d.pubchem_substances.notna().any():
        d["pop_tier"] = pd.qcut(d.pubchem_substances.rank(method="first"), 3, labels=["low", "mid", "high"])
        print("MRR by PubChem-substance tier:", d.groupby("pop_tier", observed=True).rr.mean().round(3).to_dict())
    if d.np_pathway.notna().any():
        print("MRR by NP pathway:", d.groupby("np_pathway").rr.agg(["mean", "size"]).round(3)
              .sort_values("size", ascending=False).to_dict("index"))
    if a.out:
        d.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
