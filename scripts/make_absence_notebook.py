"""Build pub477 + absence evidence: the pub477 notebook with one extra cell after its merge cell.

Rule (pre-registered, docs/research-plan.md 2026-10-30): a candidate is flagged when its structure has
training spectra with the query's adduct and the best entropy similarity to them is below 0.2 (false-fire
rate on enveda-np-examples true answers 0.4%). Within the first 10 places flagged candidates go after the
unflagged ones, order otherwise kept. Candidate sets never change; any error restores the 0.477 file.

Usage: python scripts/make_absence_notebook.py
Writes notebooks/pub477_absence/{casmi26-pub477-absence.ipynb, kernel-metadata.json}.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "pub477"
DST = ROOT / "notebooks" / "pub477_absence"
AFTER_MARK = "# ── Variant M: rank 1 from the fragment check"

CELL = r'''# ── Absence evidence (ours): flagged candidates after unflagged ones within the first 10 ──────────────
# A candidate whose structure has training spectra with the query's adduct, none of which resemble the
# query (best entropy similarity < 0.2), is moved behind the unflagged candidates of the first 10 places.
# Threshold fixed from enveda-np-examples: 0.4% of true answers with references fall below it.
import time as _t, shutil as _sh, json as _js, traceback as _tb
import numpy as _np, pandas as _pd, pyarrow.dataset as _pds
from rdkit import Chem as _Chem, RDLogger as _RDL

ABSENCE = True
_AB_THR, _AB_TOL, _AB_MAXP, _AB_MAXR, _AB_TOP, _AB_BUDGET = 0.2, 0.02, 100, 60, 10, 1500


def _ab_ik(s):
    try:
        m = _Chem.MolFromSmiles(s)
        return _Chem.MolToInchiKey(m)[:14] if m is not None else None
    except Exception:
        return None


def _ab_prep(mz, it, prec):
    mz, it = _np.asarray(mz, float), _np.asarray(it, float)
    keep = _np.isfinite(mz) & _np.isfinite(it) & (it > 0) & (mz < float(prec) - 1.6)
    mz, it = mz[keep], it[keep]
    if len(mz) > _AB_MAXP:
        top = _np.argsort(it)[-_AB_MAXP:]
        mz, it = mz[top], it[top]
    if len(mz) == 0:
        return None
    p = it / it.sum()
    h = -(p * _np.log(p)).sum()
    if h < 3:
        p = p ** (0.25 + 0.25 * h)
        p = p / p.sum()
    o = _np.argsort(mz)
    return mz[o], p[o]


def _ab_sim(a, b):
    (ma, pa), (mb, pb) = a, b
    i = _np.searchsorted(mb, ma)
    pairs = []
    for k, j0 in enumerate(i):
        for j in (j0 - 1, j0):
            if 0 <= j < len(mb) and abs(mb[j] - ma[k]) <= _AB_TOL:
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
        s += (p + q) * _np.log(2) + (p + q) * _np.log((p + q) / 2) - p * _np.log(p) - q * _np.log(q)
    return float(s / _np.log(4))


def _ab_main():
    _RDL.DisableLog('rdApp.*')
    t0 = _t.time()
    sub_path = ROOT / 'submission.csv'
    _sh.copy2(sub_path, ROOT / 'submission_pre_absence.csv')
    sub = _pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    lists = [s.split(';') for s in sub.smiles]
    ik = {s: _ab_ik(s) for lst in lists for s in lst[:_AB_TOP]}
    keys = sorted({k for k in ik.values() if k})
    te = _pd.read_parquet(COMP / 'test.parquet', columns=['molecule_id', 'adduct', 'precursor_mz', 'ms2_mzs',
                                                           'ms2_normalized_intensities'])
    qs = {str(m): [(r.adduct, _ab_prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz))
                   for r in g.itertuples()] for m, g in te.groupby('molecule_id')}
    tr = _pds.dataset(COMP / 'train.parquet').to_table(
        columns=['ingest_lib', 'inchikey14', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'],
        filter=_pds.field('inchikey14').isin(keys)).to_pandas()
    refs = {}
    for (k, a), g in tr.groupby(['inchikey14', 'adduct']):
        g = g.head(_AB_MAXR)
        refs[(k, a)] = [x for x in (_ab_prep(m, i, p) for m, i, p in
                                    zip(g.ms2_mzs, g.ms2_normalized_intensities, g.precursor_mz)) if x]
    del tr
    new, moved, flagged, timed_out = [], 0, 0, False
    for mid, lst in zip(sub.molecule_id, lists):
        if _t.time() - t0 > _AB_BUDGET:
            timed_out = True
            new.append(';'.join(lst)); continue
        flag = []
        for s in lst[:_AB_TOP]:
            best, has = 0.0, False
            for add, q in qs.get(str(mid), []):
                rr = refs.get((ik.get(s), add), [])
                if q is None or not rr:
                    continue
                has = True
                best = max(best, max(_ab_sim(q, r) for r in rr))
            flag.append(has and best < _AB_THR)
        head = lst[:_AB_TOP]
        order = [s for s, f in zip(head, flag) if not f] + [s for s, f in zip(head, flag) if f] + lst[_AB_TOP:]
        assert sorted(order) == sorted(lst) and len(set(order)) == len(order), mid
        flagged += sum(flag); moved += order != lst
        new.append(';'.join(order))
    out = _pd.DataFrame({'molecule_id': sub.molecule_id, 'smiles': new})
    assert out.molecule_id.tolist() == SAMPLE_IDS
    assert out.smiles.str.split(';').map(len).between(1, 25).all()
    tmp = ROOT / 'submission.absence.tmp'
    out.to_csv(tmp, index=False)
    tmp.replace(sub_path)
    return dict(status='applied', molecules_reordered=moved, candidates_flagged=flagged,
                structures_with_refs=len({k for k, _ in refs}), timed_out=timed_out,
                seconds=round(_t.time() - t0, 1))


try:
    if not ABSENCE:
        ABSENCE_REPORT = dict(status='disabled')
    elif _t.time() - NB_T0 > 8.7 * 3600:
        ABSENCE_REPORT = dict(status='skipped_runtime')
    else:
        ABSENCE_REPORT = _ab_main()
except Exception as exc:
    _tb.print_exc()
    if (ROOT / 'submission_pre_absence.csv').is_file():
        _sh.copy2(ROOT / 'submission_pre_absence.csv', ROOT / 'submission.csv')
    ABSENCE_REPORT = dict(status='restored_after_error', exception=repr(exc)[:400])
(ROOT / 'absence_report.json').write_text(_js.dumps(ABSENCE_REPORT, indent=2))
final = _pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
assert list(final.columns) == ['molecule_id', 'smiles'] and final.molecule_id.tolist() == SAMPLE_IDS
assert final.smiles.str.split(';').map(lambda a: 1 <= len(a) <= 25 and len(set(a)) == len(a) and all(a)).all()
print('ABSENCE', _js.dumps(ABSENCE_REPORT))
'''


def main():
    nb = json.loads((SRC / "casmi26-pub477.ipynb").read_text(encoding="utf-8"))
    at = [i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith(AFTER_MARK)]
    assert len(at) == 1, at
    cell = dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                source=CELL.splitlines(keepends=True))
    nb["cells"].insert(at[0] + 1, cell)
    DST.mkdir(exist_ok=True)
    (DST / "casmi26-pub477-absence.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(id="wyy817/casmi26-pub477-absence", title="casmi26 pub477 absence",
                code_file="casmi26-pub477-absence.ipynb")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("inserted after cell", at[0])


if __name__ == "__main__":
    main()
