"""DreaMS embedding similarity to each candidate's own reference spectra, on enveda-np-examples.

Same set-up as absence_calib.py (queries, candidates, references), but the similarity is the cosine
between DreaMS embeddings (hengck23's minimal port of the DreaMS embedding model, weights from the
pipeline assets). Compares how well entropy and DreaMS similarity separate true from wrong candidates.

Usage: python scripts/dreams_calib.py [RUN [PANEL]]  (needs absence/<RUN>.parquet from absence_calib.py)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL, RAW  # noqa: E402

DREAMS = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets/hengck23-dreams-enveda-casmi26"
MAX_REFS = 60


def load_dreams():
    spec = importlib.util.spec_from_file_location("dreams_minimal_model", DREAMS / "dreams_minimal_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.load_model(DREAMS / "embedding_model_state_dict.pt", device="cuda")


def embed(mod, model, mzs, its, precs, batch=512):
    x = np.stack([mod.preprocess_spectrum(np.asarray(a, np.float32), np.asarray(b, np.float32), float(p))
                  for a, b, p in zip(mzs, its, precs)])
    out = [mod.embed_preprocessed(model, torch.as_tensor(x[i:i + batch]).cuda()) for i in range(0, len(x), batch)]
    e = np.concatenate(out).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main(run, panel):
    c = pd.read_parquet(EXTERNAL / "absence" / f"{run}.parquet")
    q = pd.read_parquet(EXTERNAL / f"pseudo/{panel}/test.parquet")
    keys = sorted(set(c.ik.dropna()))
    cols = ["ingest_lib", "inchikey14", "adduct", "ionization_mode", "precursor_mz", "ms2_mzs",
            "ms2_normalized_intensities"]
    t = ds.dataset(RAW / "train.parquet").to_table(
        columns=cols, filter=ds.field("inchikey14").isin(keys) & (ds.field("ingest_lib") != "enveda-np-examples")
    ).to_pandas()
    t = t.groupby(["inchikey14", "adduct"], group_keys=False).head(MAX_REFS).reset_index(drop=True)
    mod, model = load_dreams()
    te = embed(mod, model, t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz)
    qe = embed(mod, model, q.ms2_mzs, q.ms2_normalized_intensities, q.precursor_mz)
    print(f"embedded {len(t):,} references and {len(q):,} queries")

    ref_rows = {k: g.index.to_numpy() for k, g in t.groupby("inchikey14")}
    q_rows = {k: g.index.to_numpy() for k, g in q.reset_index(drop=True).groupby("molecule_id")}
    tad, tpol = t.adduct.to_numpy(), t.ionization_mode.to_numpy()
    qad, qpol = q.adduct.to_numpy(), q.ionization_mode.to_numpy()
    same_add, same_pol = [], []
    for row in c.itertuples():
        rr, qq = ref_rows.get(row.ik), q_rows.get(row.molecule_id)
        if rr is None or qq is None:
            same_add.append(np.nan); same_pol.append(np.nan); continue
        s = qe[qq] @ te[rr].T
        ma = qad[qq][:, None] == tad[rr][None, :]
        mp = qpol[qq][:, None] == tpol[rr][None, :]
        same_add.append(s[ma].max() if ma.any() else np.nan)
        same_pol.append(s[mp].max() if mp.any() else np.nan)
    c["dreams_add"], c["dreams_pol"] = same_add, same_pol
    c.to_parquet(EXTERNAL / "absence" / f"{run}.parquet", index=False)

    has = c[c.n_ref_same_adduct > 0]
    print(f"same-adduct references: {len(has)} candidates ({has.is_true.sum()} true)")
    for col in ("ref_max", "dreams_add"):
        print(f"AUC true vs wrong, {col}: {roc_auc_score(has.is_true, has[col]):.3f}")
    print("dreams_add quantiles true :", has[has.is_true].dreams_add.quantile([.01, .02, .05, .1, .25, .5]).round(3).tolist())
    print("dreams_add quantiles wrong:", has[~has.is_true].dreams_add.quantile([.1, .25, .5, .75, .9]).round(3).tolist())
    hp = c[c.dreams_pol.notna()]
    print(f"\nsame-polarity references: {len(hp)} candidates ({hp.is_true.sum()} true),"
          f" AUC {roc_auc_score(hp.is_true, hp.dreams_pol):.3f}")
    tr, wr = hp[hp.is_true], hp[~hp.is_true]
    for thr in (0.5, 0.6, 0.7, 0.75, 0.8):
        print(f"dreams_pol < {thr}: true {(tr.dreams_pol < thr).mean():.3f} ({(tr.dreams_pol < thr).sum()}),"
              f" wrong {(wr.dreams_pol < thr).mean():.3f}")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "nplib_v2"
    main(run, sys.argv[2] if len(sys.argv) > 2 else "np-lib")
