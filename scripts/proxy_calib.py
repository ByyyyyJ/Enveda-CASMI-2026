"""Proxy calibration (docs/research-plan.md, 2026-11-03): do local pseudo-tests rank pipeline variants the way
the public LB does?

From each local run of the midx479 / midx479_icetims notebooks (wsl_calib_batch.sh) this rebuilds every variant
whose public LB score is known, scores it on the pseudo-test, and compares local and LB:

  V0, V1, V2, CUSTOM (w_mid 0.75), top050 / top100 (rank-1 neural weight 0.50 / 1.00 on CUSTOM),
  uni035 (uniform weight 0.35)        leader-safe fusion of the run's own evidence (consensus.py rule)
  pub477                              fragment-check leader moves merged onto CUSTOM (pub477 merge rule)
  midx479                             the run's submission.csv
  absence                             pub477 + the absence cell of make_absence_notebook.py
  ice                                 submission.csv of the matching cal_ice_* run

Usage: python scripts/proxy_calib.py [FOLDS ...]   (default f0np..f4np)
Writes D:/casmi-data/kernel_out/proxy_calib.csv (per-molecule reciprocal ranks).
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.eval.metrics import answer_ranks, reciprocal_ranks  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402

KO = EXTERNAL / "kernel_out"
LB = {"V0": 0.465, "V1": 0.466, "V2": 0.468, "CUSTOM": 0.472, "top050": 0.448, "top100": 0.359,
      "uni035": 0.447, "pub477": 0.477, "midx479": 0.479, "absence": 0.468, "ice": 0.475}
FUSION = {"V0": (0.0, 0.0), "V1": (0.25, 0.25), "V2": (0.25, 0.50), "CUSTOM": (0.25, 0.75),
          "top050": (0.50, 0.75), "top100": (1.00, 0.75), "uni035": (0.35, 0.35)}
# parent -> child pairs that were actually compared on the LB
PAIRS = [("V2", "CUSTOM"), ("CUSTOM", "top050"), ("CUSTOM", "top100"), ("V1", "uni035"), ("CUSTOM", "pub477"),
         ("pub477", "midx479"), ("pub477", "absence"), ("midx479", "ice"), ("V0", "V2")]


def _rank(v):
    o = np.argsort(-np.asarray(v, float), kind="stable")
    r = np.empty(len(v), int)
    r[o] = np.arange(1, len(v) + 1)
    return r


def _rrf(scores, w, k=3.0):
    n = len(scores)
    if w == 0:
        return np.arange(n)
    return np.argsort(-(1 / (k + np.arange(1, n + 1)) + w / (k + _rank(scores))), kind="stable")


def fusion_lists(ev, base, w_top, w_mid):
    groups = {str(m): g.sort_values("baseline_rank", kind="stable") for m, g in ev.groupby("molecule_id", sort=False)}
    out = {}
    for mid, text in zip(base.molecule_id, base.smiles):
        g = groups[str(mid)]
        assert g.smiles.tolist() == text.split(";"), mid
        if bool(g.processed.iloc[0]) and bool(g.neural_ok.all()):
            s = g.neural_score.to_numpy(float)
            first = _rrf(s, w_top)
            if w_mid == w_top:
                order = list(first)
            else:
                lead = int(first[0])
                order = [lead] + [int(i) for i in _rrf(s, w_mid) if int(i) != lead]
        else:
            order = list(range(len(g)))
        out[str(mid)] = g.smiles.iloc[order].tolist()
    return out


def pub477_lists(phys, v2, cust):
    out = {}
    for mid in cust:
        p, v, c = phys[mid], v2[mid], cust[mid]
        if v[0] != c[0] or sorted(p) != sorted(c) or p[0] == v[0]:
            out[mid] = list(c)
        else:
            out[mid] = [p[0], v[0]] + [s for s in c if s not in (p[0], v[0])]
    return out


def read_lists(path):
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {str(m): s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}


def absence_lists(pub, fold):
    """Run the absence cell of make_absence_notebook.py on the pub477 lists of this pseudo-test."""
    from make_absence_notebook import CELL
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ids = list(pub)
        pd.DataFrame({"molecule_id": ids, "smiles": [";".join(pub[m]) for m in ids]}).to_csv(root / "submission.csv", index=False)
        g = dict(ROOT=root, COMP=EXTERNAL / "pseudo" / fold, SAMPLE_IDS=ids, NB_T0=time.time())
        exec(compile(CELL.replace("_AB_BUDGET = 0.2, 0.02, 100, 60, 10, 1500", "_AB_BUDGET = 0.2, 0.02, 100, 60, 10, 1e9"),
                     "absence_cell", "exec"), g)
        return read_lists(root / "submission.csv")


def score(lists, sol):
    r = answer_ranks(sol, lists)
    return reciprocal_ranks(r)


def run_fold(fold, kind):
    d = KO / f"cal_midx_{fold}_{kind}"
    if not (d / "submission.csv").exists():
        return None
    sol_df = pd.read_csv(EXTERNAL / "pseudo" / f"{fold}_solution.csv")
    sol = dict(zip(sol_df.molecule_id.astype(str), sol_df.smiles))
    ev = pd.read_parquet(d / "neural_candidate_features.parquet")
    base = pd.read_csv(d / "submission_baseline0465.csv", dtype=str, keep_default_na=False)
    L = {n: fusion_lists(ev, base, *w) for n, w in FUSION.items()}
    for n in ("V2", "CUSTOM"):  # the rebuilt fusion must equal the notebook's own variant files
        own = read_lists(d / "variants" / f"submission_{n}.csv")
        same = np.mean([own[m] == L[n][m] for m in own])
        print(f"  {fold}/{kind} rebuilt {n} equals notebook file in {same:.1%} of molecules")
        L[n] = own
    L["pub477"] = pub477_lists(read_lists(d / "submission_physical.csv"), L["V2"], L["CUSTOM"])
    L["midx479"] = read_lists(d / "submission.csv")
    L["absence"] = absence_lists(L["pub477"], fold)
    ice = KO / f"cal_ice_{fold}_{kind}" / "submission.csv"
    if ice.exists():
        L["ice"] = read_lists(ice)
    rows = pd.DataFrame({n: score(l, sol) for n, l in L.items()})
    rows.index.name = "molecule_id"
    return rows.reset_index().assign(fold=fold, kind=kind)


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return np.quantile(b, [0.025, 0.975])


def main():
    folds = sys.argv[1:] or [f"f{k}np" for k in range(5)]
    parts = [p for f in folds for k in ("safe", "hide") if (p := run_fold(f, k)) is not None]
    if not parts:
        sys.exit("no cal_midx_* runs found")
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(KO / "proxy_calib.csv", index=False)
    for kind, g in t.groupby("kind"):
        names = [n for n in LB if n in g and g[n].notna().all()]
        print(f"\n=== {kind}: {g.fold.nunique()} folds, {len(g)} molecules")
        m = g[names].mean()
        print(pd.DataFrame({"local": m, "LB": [LB[n] for n in names]}).round(4).T.to_string())
        rho = spearmanr(m.values, [LB[n] for n in names]).correlation
        print(f"Spearman(local, LB) over {len(names)} variants: {rho:.2f}")
        agree = 0
        for a, b in PAIRS:
            if a not in names or b not in names:
                continue
            dd = (g[b] - g[a]).to_numpy()
            lo, hi = boot_ci(dd)
            lbd = LB[b] - LB[a]
            ok = np.sign(dd.mean()) == np.sign(lbd) or (abs(lbd) < 0.004 and lo <= 0 <= hi)
            agree += ok
            print(f"  {a:>7} -> {b:<7} local {dd.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  LB {lbd:+.3f}  {'ok' if ok else 'WRONG'}")
        print(f"  direction agreement {agree}/{sum(a in names and b in names for a, b in PAIRS)}")


if __name__ == "__main__":
    main()
