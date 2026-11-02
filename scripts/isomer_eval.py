"""Compare the isomer scorer with a pipeline's order on a pseudo-test, within the leader's formula.

Usage: python isomer_eval.py [FOLD [RUN [PANEL]]]   (defaults 0 f0np_safe f0np)
Reads D:/casmi-data/isomer/model_f{FOLD}.pt, kernel_out/{RUN}/submission.csv, pseudo/{PANEL}/test.parquet
and pseudo/{PANEL}_solution.csv. Writes the per-candidate scores to isomer/eval_{RUN}.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.chem.canon import ik14  # noqa: E402
from casmi.eval.metrics import read_submission  # noqa: E402
from casmi.isomer import Model, morgan_counts, prep_peaks  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402
from casmi.rerank import formula  # noqa: E402

FOLD = sys.argv[1] if len(sys.argv) > 1 else "0"  # model name suffix, e.g. 0 or 0_ext
RUN = sys.argv[2] if len(sys.argv) > 2 else "f0np_safe"
PANEL = sys.argv[3] if len(sys.argv) > 3 else "f0np"
DATA = EXTERNAL / "isomer"
dev = "cuda"

adducts = pd.read_csv(DATA / "adducts.csv").iloc[:, 0].tolist()
amap = {a: i for i, a in enumerate(adducts)}
model = Model(len(adducts)).to(dev)
model.load_state_dict(torch.load(DATA / f"model_f{FOLD}.pt", map_location=dev))
model.eval()

test = pd.read_parquet(EXTERNAL / f"pseudo/{PANEL}/test.parquet")
unknown = (~test.adduct.isin(amap)).sum()
if unknown:
    print("spectra with an adduct unseen in training:", unknown)
peaks = [prep_peaks(m, a, p) for m, a, p in zip(test.ms2_mzs, test.ms2_normalized_intensities, test.precursor_mz)]
with torch.no_grad():
    z = model.s(torch.from_numpy(np.stack([p[0] for p in peaks])).to(dev),
                torch.from_numpy(np.stack([p[1] for p in peaks])).to(dev),
                torch.from_numpy(test.precursor_mz.to_numpy(np.float32)).to(dev),
                torch.from_numpy(test.adduct.map(amap).fillna(0).to_numpy(np.int64)).to(dev),
                torch.from_numpy((test.instrument_type == "timsTOF").to_numpy(np.int64)).to(dev))
zmol = {m: z[ix].mean(0) for m, ix in test.groupby("molecule_id").indices.items()}

sol = pd.read_csv(EXTERNAL / f"pseudo/{PANEL}_solution.csv")
truth = dict(zip(sol.molecule_id, sol.inchikey14))
lists = read_submission(EXTERNAL / f"kernel_out/{RUN}/submission.csv")

rows = []
for mol, cands in lists.items():
    if mol not in zmol:
        continue
    for r, smi in enumerate(cands[:25], 1):
        rows.append((mol, r, smi, ik14(smi), formula(smi)))
ev = pd.DataFrame(rows, columns=["molecule_id", "rank", "smiles", "ik14", "formula"])
ev["is_true"] = ev.ik14 == ev.molecule_id.map(truth)
fps = [morgan_counts(s) for s in ev.smiles]
ok = np.array([f is not None for f in fps])
fpm = np.zeros((len(ev), 2048), np.uint8)
fpm[ok] = np.stack([f for f in fps if f is not None])
with torch.no_grad():
    m = model.m(torch.from_numpy(fpm).to(dev))
    zm = torch.stack([zmol[x] for x in ev.molecule_id])
    ev["iso"] = np.where(ok, (m * zm).sum(-1).float().cpu().numpy(), np.nan)
ev.to_parquet(DATA / f"eval_{RUN}_m{FOLD}.parquet")

lead_f = ev[ev["rank"] == 1].set_index("molecule_id").formula
ev["same_f"] = ev.formula == ev.molecule_id.map(lead_f)
g = ev[ev.same_f]
n_mol = ev.molecule_id.nunique()
hit = g.groupby("molecule_id").is_true.any()
size = g.groupby("molecule_id").size()
pool = hit[hit & (size >= 2)].index
gp = g[g.molecule_id.isin(pool)]
pipe = gp.sort_values("rank").groupby("molecule_id").head(1).is_true.sum()
mod = gp.sort_values("iso", ascending=False).groupby("molecule_id").head(1).is_true.sum()
print(f"molecules {n_mol}; answer in leader-formula group of >=2: {len(pool)}")
print(f"  first in group correct: pipeline {pipe}  isomer model {mod}")

top2 = ev[ev["rank"] <= 2].pivot(index="molecule_id", columns="rank")
same = top2[("formula", 1)] == top2[("formula", 2)]
margin = (top2[("iso", 2)] - top2[("iso", 1)])[same]
t1, t2 = top2[("is_true", 1)][same], top2[("is_true", 2)][same]
print(f"rank-1/2 same formula: {int(same.sum())} molecules; answer at 1: {int(t1.sum())}, at 2: {int(t2.sum())}")
for t in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
    sw = margin > t
    print(f"  swap if model margin > {t:.2f}: swaps {int(sw.sum())}, gains {int((sw & t2).sum())}, losses {int((sw & t1).sum())}")


def mrr_of(order_col, asc):
    rr = []
    for mol, d in ev.groupby("molecule_id"):
        d = d.copy()
        gi = d.index[d.same_f]
        d.loc[gi, "pos"] = np.sort(d.loc[gi, "rank"].to_numpy())[np.argsort(np.argsort(
            d.loc[gi, order_col].to_numpy() * (1 if asc else -1), kind="stable"))]
        d.loc[~d.same_f, "pos"] = d.loc[~d.same_f, "rank"]
        p = d.loc[d.is_true, "pos"]
        rr.append(1.0 / p.min() if len(p) else 0.0)
    return np.mean(rr)


print(f"MRR pipeline {mrr_of('rank', True):.4f}  leader-formula group reordered by model {mrr_of('iso', False):.4f}")
