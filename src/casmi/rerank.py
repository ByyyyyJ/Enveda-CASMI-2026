"""Features for re-ranking each molecule's final 25 from the public pipeline's per-candidate evidence
(neural_candidate_features.parquet). See docs/research-plan.md.

Feature groups:
  cand    the evidence as the pipeline saved it (list ranks, simulator and neural scores)
  rel     relative to the candidate's same-formula group within the molecule: most errors are
          wrong isomers of the right formula, and absolute scores do not compare across molecules
  agree   agreement between engines on this candidate
  gate    molecule-level evidence (identical for all candidates of a molecule), from which the model
          can tell whether to trust the pool engines or the PubChem channel
NP-knowledge-table popularity is left out on purpose: the f<k>np pseudo-tests are selected through it.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")

ABSENT = 99  # list rank used when a candidate is not in that list
LISTS = ["main_rank", "prefusion_rank", "engine2_rank", "pubchem_rank"]
SCORES = ["ice", "gl", "neural_score", "neural_cos", "neural_fp"]


def formula(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return CalcMolFormula(m) if m is not None else ""


def main_first_order(ev: pd.DataFrame) -> np.ndarray:
    """Score whose descending order is the main_first reorder (postproc.main_first)."""
    mr = ev["main_rank"].where(ev["main_rank"] > 0, 10_000)
    return -(mr * 1_000 + ev["baseline_rank"]).to_numpy(float)


def build_features(ev: pd.DataFrame, pool_keys: set | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (features, groups): one row per evidence row, and feature names per group."""
    e = ev.reset_index(drop=True)
    f = pd.DataFrame(index=e.index)
    groups = {"cand": [], "rel": [], "agree": [], "gate": []}

    def add(group, name, values):
        f[name] = np.asarray(values, dtype=float)
        groups[group].append(name)

    # cand
    add("cand", "baseline_rank", e["baseline_rank"])
    for c in LISTS:
        add("cand", c, e[c].where(e[c] > 0, ABSENT))
    for c in SCORES + ["neural_std"]:
        add("cand", c, e[c])
    add("cand", "proxy_gl", e["proxy_gl"].astype(float))
    pc_only = (e["pubchem_rank"] > 0) & (e["main_rank"] == 0) & (e["engine2_rank"] == 0)
    add("cand", "pubchem_only", pc_only)
    if pool_keys is not None:
        add("cand", "in_pool", e["metric_key"].isin(pool_keys))

    # rel: same-formula group within the molecule
    form = e["smiles"].map(formula)
    key = e["molecule_id"] + "|" + form
    gm = e.groupby("molecule_id")
    gf = f.groupby(key)
    add("rel", "formula_n", key.map(key.value_counts()))
    top_form = form.loc[e.groupby("molecule_id")["baseline_rank"].idxmin()]
    top_form.index = e.loc[top_form.index, "molecule_id"]
    add("rel", "is_top1_formula", form == e["molecule_id"].map(top_form))
    for c in ["baseline_rank", "main_rank", "engine2_rank"]:
        add("rel", f"{c}_in_formula", gf[c].rank(method="min"))
    for c in SCORES:
        s = e[c]
        grp = s.groupby(key)
        add("rel", f"{c}_in_formula", grp.rank(ascending=False, method="min"))
        add("rel", f"{c}_gap_formula", grp.transform("max") - s)
        add("rel", f"{c}_z", (s - gm[c].transform("mean")) / (gm[c].transform("std") + 1e-6))

    # agree: how many engines put this candidate high
    ranks = np.stack([f[c].to_numpy() for c in ["main_rank", "engine2_rank", "pubchem_rank"]], 1)
    sim = np.stack([f[f"{c}_in_formula"].to_numpy() for c in ["ice", "gl", "neural_score"]], 1)
    add("agree", "n_lists", (ranks < ABSENT).sum(1))
    add("agree", "n_lists_top3", (ranks <= 3).sum(1))
    add("agree", "best_list_rank", ranks.min(1))
    add("agree", "n_sim_formula_top1", (sim == 1).sum(1))

    # gate: molecule-level
    i1 = e.groupby("molecule_id")["baseline_rank"].idxmin()
    first = e.loc[i1].set_index("molecule_id")
    mol = pd.DataFrame(index=first.index)
    mol["library_max"] = first["library_max"]
    mol["n_spectra"] = first["n_spectra"]
    mol["n_adducts"] = first["n_adducts"]
    mol["top1_pubchem_only"] = pc_only.loc[i1].to_numpy()
    mol["n_pubchem_only_top5"] = (pc_only & (e["baseline_rank"] <= 5)).groupby(e["molecule_id"]).sum()
    mol["main_len"] = (e["main_rank"] > 0).groupby(e["molecule_id"]).sum()
    def arg1(c):  # metric_key ranked first by list c
        s = e[e[c] > 0]
        return s.loc[s.groupby("molecule_id")[c].idxmin()].set_index("molecule_id")["metric_key"]
    m1, e1, b1 = arg1("main_rank"), arg1("engine2_rank"), first["metric_key"]
    mol["main_eq_final"] = (m1.reindex(mol.index) == b1).astype(float)
    mol["main_eq_engine2"] = (m1.reindex(mol.index) == e1.reindex(mol.index)).astype(float)
    for c in ["neural_score", "ice", "gl"]:
        top2 = e.groupby("molecule_id")[c].apply(lambda s: s.nlargest(2).to_numpy())
        mol[f"{c}_margin"] = top2.map(lambda a: a[0] - a[1] if len(a) > 1 else 0.0)
    mol["n_formulas"] = form.groupby(e["molecule_id"]).nunique()
    for c in mol.columns:
        add("gate", "mol_" + c, e["molecule_id"].map(mol[c]))
    return f, groups
