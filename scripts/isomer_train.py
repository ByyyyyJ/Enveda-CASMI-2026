"""Train the same-formula isomer scorer: given a spectrum, pick its structure among same-formula isomers.

Usage: python isomer_train.py [VAL_FOLD] [EPOCHS] [TAG]
Train on the other identity folds (enveda-np-examples excluded), validate on VAL_FOLD by ranking the true
structure among every same-formula structure in the table. TAG "ext" adds the candidate-database structures
(isomer_pool.py) as negatives and draws half of each query's negatives from its most similar isomers.
Writes D:/casmi-data/isomer/model_f{VAL_FOLD}[_TAG].pt after every epoch.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.isomer import D, Model  # noqa: E402
from casmi.paths import EXTERNAL  # noqa: E402

VAL = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
K, B, SCAN = 16, 256, 64
HARD = K // 2 if TAG == "ext" else 0
DATA = EXTERNAL / "isomer"
dev = "cuda"
torch.manual_seed(0)
rng = np.random.default_rng(0)

S = np.load(DATA / "spectra.npz")
suffix = "_ext" if TAG == "ext" else ""
st = pd.read_parquet(DATA / f"structures{suffix}.parquet")
fp = torch.from_numpy(np.load(DATA / f"fp{suffix}.npy"))
n_add = len(pd.read_csv(DATA / "adducts.csv"))
sid, lib = S["sid"], S["lib"]
fold = st.fold_identity.to_numpy()
sfold = fold[sid]
ok = st.fp_ok.to_numpy()
form = st.formula.to_numpy()

okidx = np.flatnonzero(ok)
groups_all = {f: okidx[ix] for f, ix in st[ok].reset_index(drop=True).groupby("formula").indices.items()}
groups_tr = {f: ix[fold[ix] != VAL] for f, ix in groups_all.items()}

tr = np.flatnonzero((sfold != VAL) & (lib != "enveda-np-examples") & ok[sid])
tr = tr[[len(groups_tr[form[s]]) >= 2 for s in sid[tr]]]
va = np.flatnonzero((sfold == VAL) & ok[sid])
va = va[[len(groups_all[form[s]]) >= 2 for s in sid[va]]]
va = rng.choice(va, min(len(va), 20000), replace=False)
print(f"structures {len(st)}  train spectra {len(tr)}  val spectra {len(va)}", flush=True)

mz_t = torch.from_numpy(S["mz"]); it_t = torch.from_numpy(S["it"].astype(np.float32))
prec_t = torch.from_numpy(S["prec"]); add_t = torch.from_numpy(S["adduct"].astype(np.int64))
tims_t = torch.from_numpy(S["tims"].astype(np.int64))

model = Model(n_add).to(dev)


def spec(ix):
    return model.s(mz_t[ix].to(dev), it_t[ix].to(dev), prec_t[ix].to(dev), add_t[ix].to(dev), tims_t[ix].to(dev))


def struct(six):
    return model.m(fp[six].to(dev))


def tanimoto(a, b):
    """Binary Tanimoto between a (n, bits) and b (n, m, bits) on the GPU."""
    a = (a > 0).float(); b = (b > 0).float()
    inter = torch.einsum("nd,nmd->nm", a, b)
    return inter / (a.sum(-1, keepdim=True) + b.sum(-1) - inter).clamp(min=1)


def cand_batch(ix, groups):
    """Positive at column 0, then up to K same-formula isomers (HARD most similar of SCAN, rest random)."""
    c = np.zeros((len(ix), K + 1), np.int64); msk = np.zeros((len(ix), K + 1), bool)
    if HARD:
        scan = np.zeros((len(ix), SCAN), np.int64); smsk = np.zeros((len(ix), SCAN), bool)
    for r, i in enumerate(ix):
        s = sid[i]; g = groups[form[s]]; g = g[g != s]
        c[r, 0] = s
        if HARD and len(g) > K:
            g = rng.choice(g, min(len(g), SCAN), replace=False)
            scan[r, :len(g)] = g; smsk[r, :len(g)] = True
            continue
        if len(g) > K:
            g = rng.choice(g, K, replace=False)
        c[r, 1:1 + len(g)] = g; msk[r, :1 + len(g)] = True
    if HARD:
        rows = np.flatnonzero(smsk.any(1))
        if len(rows):
            sim = tanimoto(fp[c[rows, 0]].to(dev), fp[scan[rows]].to(dev)).masked_fill(~torch.from_numpy(smsk[rows]).to(dev), -1)
            order = sim.argsort(1, descending=True).cpu().numpy()
            for j, r in enumerate(rows):
                n = smsk[r].sum(); o = order[j, :n]
                pick = np.r_[o[:HARD], rng.choice(o[HARD:], K - HARD, replace=False)]
                c[r, 1:] = scan[r, pick]; msk[r, :] = True
    return torch.from_numpy(c), torch.from_numpy(msk).to(dev)


opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
steps = EPOCHS * (len(tr) // B)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)


@torch.no_grad()
def evaluate():
    """Rank the true structure among same-formula structures without spectra in training (unseen ones)."""
    model.eval(); res = []
    for b in range(0, len(va), 512):
        ix = va[b:b + 512]
        z = spec(torch.from_numpy(ix))
        for r, i in enumerate(ix):
            s = sid[i]; g = groups_all[form[s]]; g = g[(fold[g] == VAL) | (fold[g] == -1)]
            if len(g) < 2:
                continue
            sc = struct(torch.from_numpy(g)) @ z[r]
            res.append((1 + int((sc > sc[g == s][0]).sum()), len(g)))
    model.train(); a = np.array(res)
    rnd = np.mean([sum(1 / j for j in range(1, n + 1)) / n for n in a[:, 1]])
    return len(a), np.mean(a[:, 0] == 1), np.mean(1 / a[:, 0]), rnd, np.median(a[:, 1])


t0 = time.time(); step = 0
for ep in range(EPOCHS):
    perm = rng.permutation(tr)
    for b in range(0, len(perm) - B + 1, B):
        ix = perm[b:b + B]
        cands, msk = cand_batch(ix, groups_tr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = spec(torch.from_numpy(ix))
            m = struct(cands.flatten()).view(B, K + 1, D)
            iso = torch.einsum("bd,bkd->bk", z, m)
            inb = z @ m[:, 0].T  # in-batch negatives keep the space globally sensible
            sc = model.logit_scale.exp().clamp(max=100)
            loss = F.cross_entropy((sc * iso).float().masked_fill(~msk, -1e4), torch.zeros(B, dtype=torch.long, device=dev)) \
                + 0.3 * F.cross_entropy((sc * inb).float(), torch.arange(B, device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step(); sched.step(); step += 1
        if step % 2000 == 0:
            print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    n, top1, mrr, rnd, med = evaluate()
    print(f"== epoch {ep}: val (unseen negatives, n {n}, median size {med}) top1 {top1:.3f}  mrr {mrr:.3f}  "
          f"(random mrr {rnd:.3f})", flush=True)
    torch.save(model.state_dict(), DATA / f"model_f{VAL}{'_' + TAG if TAG else ''}.pt")
print("done", flush=True)
