"""Chemical-space distance strata of the f<k>np pseudo-tests (docs/research-plan.md §3.1).

For each answer: max Tanimoto (Morgan r=2, 2048 bits) to the training structures outside its identity
fold, i.e. what the fold-k-safe models could have learned from ("nn_sim"), and to every structure whose
spectra the pseudo-test's train.parquet keeps, i.e. what the pipeline's analogue features can see at
run time ("nn_sim_lib"; same-fold relatives are still there). Molecules are binned by that
similarity, and per bin the rule-based variants of V2 are scored on the safe and hide runs. If the
low-similarity bin turns main_first / pc_at2 the way the public LB did, it is the better validation set.

Usage (WSL): python scripts/similarity_strata.py [--folds 0 1 2 3 4] [--by nn_sim|nn_sim_lib]
Caches the similarities in D:/casmi-data/pseudo/answer_nn_sim.csv and answer_nn_sim_lib.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.paths import EXTERNAL, SPLITS  # noqa: E402
from casmi.rerank import main_first_order  # noqa: E402
from pc_promote_scan import promote  # noqa: E402
from rerank_cv import rr  # noqa: E402

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
CACHE = EXTERNAL / "pseudo" / "answer_nn_sim.csv"
CACHE_LIB = EXTERNAL / "pseudo" / "answer_nn_sim_lib.csv"
BINS = [0, 0.4, 0.55, 0.7, 0.85, 1.01]


def fp(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return GEN.GetFingerprint(m) if m is not None else None


def similarities(folds):
    if CACHE.exists():
        return pd.read_csv(CACHE)
    st = pd.read_parquet(SPLITS / "structure_ik14.parquet", columns=["smiles", "fold_identity"])
    st["fp"] = st.smiles.map(fp)
    st = st[st.fp.notna()]
    rows = []
    for k in folds:
        lib = list(st.loc[st.fold_identity != k, "fp"])
        sol = pd.read_csv(EXTERNAL / "pseudo" / f"f{k}np_solution.csv")
        for mid, smi in zip(sol.molecule_id, sol.smiles):
            q = fp(smi)
            s = max(DataStructs.BulkTanimotoSimilarity(q, lib)) if q is not None else np.nan
            rows.append(dict(fold=k, molecule_id=mid, nn_sim=s))
        print(f"fold {k} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(CACHE, index=False)
    return out


def lib_similarities(folds):
    if CACHE_LIB.exists():
        return pd.read_csv(CACHE_LIB)
    rows = []
    for k in folds:
        tr = pd.read_parquet(EXTERNAL / "pseudo" / f"f{k}np" / "train.parquet", columns=["normalized_smiles"])
        lib = [f for f in pd.unique(tr.normalized_smiles.dropna()) for f in [fp(f)] if f is not None]
        sol = pd.read_csv(EXTERNAL / "pseudo" / f"f{k}np_solution.csv")
        for mid, smi in zip(sol.molecule_id, sol.smiles):
            q = fp(smi)
            s = max(DataStructs.BulkTanimotoSimilarity(q, lib)) if q is not None else np.nan
            rows.append(dict(fold=k, molecule_id=mid, nn_sim_lib=s))
        print(f"fold {k} done (lib {len(lib)} structures)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(CACHE_LIB, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--by", default="nn_sim", choices=["nn_sim", "nn_sim_lib"])
    a = ap.parse_args()
    sim = similarities(a.folds).merge(lib_similarities(a.folds), on=["fold", "molecule_id"])
    print(sim[["nn_sim", "nn_sim_lib"]].describe().round(3).to_string())
    sim["nn_sim"] = sim[a.by]
    rows = []
    for kind in ("safe", "hide"):
        for k in a.folds:
            run = f"f{k}np_{kind}"
            ev = pd.read_parquet(EXTERNAL / "kernel_out" / run / "neural_candidate_features.parquet").reset_index(drop=True)
            sol = pd.read_csv(EXTERNAL / "pseudo" / f"f{k}np_solution.csv")
            ans = dict(zip(sol.molecule_id, sol.smiles.map(ik14)))
            y = (ev["metric_key"] == ev["molecule_id"].map(ans)).astype(int).to_numpy()
            mols = list(sol.molecule_id)
            b = ev["baseline_rank"].to_numpy(float)
            pc = (ev["pubchem_rank"] > 0) & (ev["main_rank"] == 0) & (ev["engine2_rank"] == 0)
            r = pd.DataFrame({"V2": rr(ev, y, -b, mols),
                              "main_first": rr(ev, y, main_first_order(ev), mols),
                              "pc_last": rr(ev, y, -(1000 * pc + b), mols),
                              "pc_at2": rr(ev, y, promote(ev, 2), mols)})
            r["kind"], r["fold"], r["molecule_id"] = kind, k, r.index
            rows.append(r.reset_index(drop=True))
    d = pd.concat(rows).merge(sim, on=["fold", "molecule_id"])
    d["bin"] = pd.cut(d.nn_sim, BINS, right=False)
    for m in ["main_first", "pc_last", "pc_at2"]:
        d[m] = d[m] - d["V2"]
    t = d.groupby(["kind", "bin"], observed=True).agg(n=("V2", "size"), V2=("V2", "mean"), main_first=("main_first", "mean"),
                                                      pc_last=("pc_last", "mean"), pc_at2=("pc_at2", "mean"))
    print("\nper similarity bin (variants as delta vs V2); public LB: V2 0.468, main_first -0.097, "
          "pc_last -0.041, pc_at2 -0.009")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
