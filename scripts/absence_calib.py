"""Calibrate "absence evidence" on enveda-np-examples (same instrument as the hidden test, true labels).

Absence evidence: a candidate whose structure has reference spectra in the training library, none of
which resemble the query, is probably wrong. Before using it as a rule we measure, on real timsTOF
NP spectra, how often the TRUE structure looks absent (false-fire rate) and how often wrong candidates
do (hit rate). This calibrates a threshold; it does not select a ranking model on a pseudo-test.

Queries: pseudo/np-lib/test.parquet (250 NPs, official cleaning). Candidates: final 25 of a pipeline
run on that panel. References: train.parquet rows of the candidate's InChIKey-14, excluding
enveda-np-examples, same adduct as the query spectrum. Similarity: unweighted-entropy similarity
(Li et al. 2021) with entropy-based intensity weighting, 0.02 Da.

Usage: python scripts/absence_calib.py [RUN [PANEL]] (default nplib_v2 np-lib); writes
D:/casmi-data/absence/<RUN>.parquet. On f<k>np runs only the wrong-leader statistics are meaningful: the
answers' own spectra were removed from the pseudo-test library, the hidden test's class-1 answers keep theirs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL, RAW  # noqa: E402

RDLogger.DisableLog("rdApp.*")
TOL = 0.02
MAX_PEAKS = 100
MAX_REFS = 60


def plain_ik14(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def prep(mz, it, prec):
    mz, it = np.asarray(mz, float), np.asarray(it, float)
    keep = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz < prec - 1.6)
    mz, it = mz[keep], it[keep]
    if len(mz) > MAX_PEAKS:
        top = np.argsort(it)[-MAX_PEAKS:]
        mz, it = mz[top], it[top]
    if len(mz) == 0:
        return None
    p = it / it.sum()
    h = -(p * np.log(p)).sum()
    if h < 3:  # entropy weighting from Li et al.
        p = p ** (0.25 + 0.25 * h)
        p = p / p.sum()
    o = np.argsort(mz)
    return mz[o], p[o]


def entropy_sim(a, b):
    (ma, pa), (mb, pb) = a, b
    i = np.searchsorted(mb, ma)
    pairs = []
    for k, j0 in enumerate(i):
        for j in (j0 - 1, j0):
            if 0 <= j < len(mb) and abs(mb[j] - ma[k]) <= TOL:
                pairs.append((pa[k] * pb[j], k, j))
    if not pairs:
        return 0.0
    pairs.sort(reverse=True)
    ua, ub, s = set(), set(), 0.0
    for _, k, j in pairs:
        if k in ua or j in ub:
            continue
        ua.add(k); ub.add(j)
        p, q = pa[k], pb[j]
        s += (p + q) * np.log(2) + (p + q) * np.log((p + q) / 2) - p * np.log(p) - q * np.log(q)
    return float(s / np.log(4))


def main(run, panel):
    q = pd.read_parquet(EXTERNAL / f"pseudo/{panel}/test.parquet")
    sol = pd.read_csv(EXTERNAL / f"pseudo/{panel}_solution.csv")
    truth = dict(zip(sol.molecule_id, sol.smiles.map(plain_ik14)))
    sub = pd.read_csv(EXTERNAL / "kernel_out" / run / "submission.csv")
    cands = [(mid, r, s) for mid, lst in zip(sub.molecule_id, sub.smiles)
             for r, s in enumerate(lst.split(";"), 1)]
    c = pd.DataFrame(cands, columns=["molecule_id", "rank", "smiles"])
    c["ik"] = c.smiles.map(plain_ik14)
    c["is_true"] = c.ik == c.molecule_id.map(truth)
    keys = sorted(set(c.ik.dropna()))
    print(f"{len(c)} candidates, {len(keys)} structures, true in list {c.groupby('molecule_id').is_true.any().mean():.3f}")

    cols = ["ingest_lib", "inchikey14", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    t = ds.dataset(RAW / "train.parquet").to_table(
        columns=cols, filter=ds.field("inchikey14").isin(keys) & (ds.field("ingest_lib") != "enveda-np-examples")
    ).to_pandas()
    print(f"{len(t):,} reference spectra for {t.inchikey14.nunique():,} structures")
    refs = {}
    for (ik, add), g in t.groupby(["inchikey14", "adduct"]):
        g = g.head(MAX_REFS)
        refs[(ik, add)] = [x for x in (prep(a, b, p) for a, b, p in
                                       zip(g.ms2_mzs, g.ms2_normalized_intensities, g.precursor_mz)) if x]
    n_any = t.groupby("inchikey14").size().to_dict()

    qs = {mid: [(r.adduct, prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz))
                for r in g.itertuples()] for mid, g in q.groupby("molecule_id")}
    out = []
    for row in c.itertuples():
        best, nref, nq = 0.0, 0, 0
        for add, qq in qs.get(row.molecule_id, []):
            rr = refs.get((row.ik, add), [])
            if qq is None or not rr:
                continue
            nq += 1; nref += len(rr)
            best = max(best, max(entropy_sim(qq, r) for r in rr))
        out.append((best, nref, nq, n_any.get(row.ik, 0)))
    c[["ref_max", "n_ref_same_adduct", "n_query_matched", "n_ref_any"]] = out
    dest = EXTERNAL / "absence"
    dest.mkdir(exist_ok=True)
    c.to_parquet(dest / f"{run}.parquet", index=False)

    has = c[c.n_ref_same_adduct > 0]
    print(f"\ncandidates with same-adduct references: {len(has)} ({has.is_true.sum()} true, {(~has.is_true).sum()} wrong)")
    print("ref_max quantiles  true :", has[has.is_true].ref_max.quantile([.01, .02, .05, .1, .25, .5]).round(3).tolist())
    print("ref_max quantiles  wrong:", has[~has.is_true].ref_max.quantile([.1, .25, .5, .75, .9, .95]).round(3).tolist())
    lead = c[c["rank"] == 1]
    lw = lead[(~lead.is_true) & (lead.n_ref_same_adduct > 0)]
    print(f"wrong leaders with references: {len(lw)} / {(~lead.is_true).sum()} wrong leaders")
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5):
        tr = has[has.is_true]
        wr = has[~has.is_true]
        print(f"thr {thr:.1f}: true below {(tr.ref_max < thr).mean():.3f} ({(tr.ref_max < thr).sum()}),"
              f" wrong below {(wr.ref_max < thr).mean():.3f}, wrong leaders below {(lw.ref_max < thr).sum()}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "nplib_v2"
    main(run, sys.argv[2] if len(sys.argv) > 2 else "np-lib")
