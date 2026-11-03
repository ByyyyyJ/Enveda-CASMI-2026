"""Fine-tune ICEBERG's intensity model on timsTOF spectra (generator frozen, instrument token 'Unknown').

Usage (WSL): python ice_finetune.py [STEPS] [LR] [TAG] [REPLAY_FRAC]
REPLAY_FRAC of each batch comes from ice_ft/replay.parquet (MassSpecGym with its own instrument tokens) so the
model keeps what the msg_all checkpoint knew. Each step runs the frozen generator on a batch of training structures (as JointModel.predict_mol does), then the
intensity model with gradients, and minimises the cosine loss between its binned output and the binned true spectrum
(max intensity per 0.1 Da bucket, peaks normalised to the base peak). Saves the intensity state_dict every SAVE steps
to D:/casmi-data/ice_ft/inten_{TAG}_{step}.pt.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ice_env import EXT, load_joint, log  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
TAG = sys.argv[3] if len(sys.argv) > 3 else "ft"
REPLAY = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
NP_HALF = sys.argv[5] if len(sys.argv) > 5 else ""  # A, B or AB: np-examples halves (ice_np_split.py) to train on
B, SAVE, INSTR = 32, 1000, "Unknown"
NR = int(round(B * REPLAY))
NNP = 4 if NP_HALF else 0
dev = "cuda"

import torch  # noqa: E402  (after the runner has set up sys.path)

runner, jm = load_joint(dev)
import ms_pred.common as common  # noqa: E402
from ms_pred import nn_utils  # noqa: E402
from ms_pred.magma.fragmentation import FRAGMENT_ENGINE_PARAMS  # noqa: E402

MAXB = FRAGMENT_ENGINE_PARAMS["max_broken_bonds"]
gen, inten = jm.gen_model_obj, jm.inten_model_obj
gen.eval()
for p in gen.parameters():
    p.requires_grad_(False)
inten.train()
buckets = inten.inten_buckets.detach()

def load(name, instr=None):
    d = pd.read_parquet(f"{EXT}/ice_ft/{name}.parquet")
    d = d[[runner.smiles_ok(s)[0] for s in d.normalized_smiles]].reset_index(drop=True)
    if instr:
        d["instr"] = instr
    log(name, "spectra", len(d))
    return d


df = load("train", INSTR)
rp = load("replay") if NR else None
npx = load("np_train", INSTR) if NNP else None
if NNP:
    npx = npx[npx.half.isin(list(NP_HALF))].reset_index(drop=True)
    log("np-examples halves", NP_HALF, "spectra", len(npx))
rng = np.random.default_rng(0)


def target(mz, it):
    t = torch.zeros(len(buckets), device=dev)
    mz = torch.as_tensor(np.asarray(mz, np.float32), device=dev)
    it = torch.as_tensor(np.asarray(it, np.float32), device=dev)
    keep = mz < buckets[-1]
    idx = torch.clamp(torch.bucketize(mz[keep], buckets, right=False), max=len(buckets) - 1)
    return t.scatter_reduce(0, idx, it[keep] / it.max(), reduce="amax")


def step_batch(rows):
    smis = [common.smiles_from_inchi(common.inchi_from_smiles(common.rm_stereo(s))) for s in rows.normalized_smiles]
    adducts = rows.adduct.tolist()
    ces = rows.ce.astype(float).tolist()
    instrs = rows.instr.tolist()
    n = len(smis)
    with torch.no_grad():
        fp_, root_reprs = gen.predict_mol(root_smi=smis, collision_eng=ces, precursor_mz=[0.0] * n,
                                          instrument=instrs, adduct=adducts, threshold=0.0, device=dev,
                                          max_nodes=100, canonical_root_smi=True)
        num_frags = fp_["nfrags"]; mx = torch.max(num_frags)
        ind_maps = torch.arange(n, device=dev).repeat_interleave(num_frags)
        rep_roots, _, __ = nn_utils.slice_batched_graph(root_reprs, ind_maps)
        frag_mask = nn_utils.pack_padded_tensor(fp_["frags"], num_frags)
        frag_graphs = nn_utils.batched_mask_subgraph(
            rep_roots, nn_utils.pack_padded_tensor(frag_mask, fp_["natoms"].repeat_interleave(num_frags)),
            frag_mask.sum(dim=-1))
        broken = fp_["brokens"]
        nhs = fp_["frag_form_vecs"][:, :, common.element_to_ind["H"]]
        off = torch.tensor([(common.ion2mass[a], -common.ELECTRON_MASS) for a in adducts], device=dev)[:, :, None] + \
            (torch.arange(MAXB * 2 + 1, device=dev)[None, None, :] - MAXB) * common.CHEM_MASSES[common.element_to_ind["H"]].item()
        masses = fp_["masses_no_adduct"][:, :mx, None, None] + off[:, None, :, :]
        f = lambda x: torch.tensor(x, device=dev, dtype=torch.float)
    out = inten.forward(frag_graphs, root_reprs, ind_maps, num_frags, broken=broken,
                        collision_engs=f(ces), precursor_mzs=f(rows.precursor_mz.tolist()),
                        adducts=f([common.ion2onehot_pos[a] for a in adducts]),
                        instruments=f([common.instrument2onehot_pos[i] for i in instrs]),
                        max_add_hs=broken[:, :mx], max_remove_hs=torch.min(broken, nhs)[:, :mx], masses=masses,
                        root_forms=fp_["root_form_vec"], frag_forms=fp_["frag_form_vecs"][:, :mx])
    pred = out["output_binned"][:, 0, :]
    targ = torch.stack([target(m, i) for m, i in zip(rows.ms2_mzs, rows.ms2_normalized_intensities)])
    return (1 - torch.nn.functional.cosine_similarity(pred, targ)).mean()


opt = torch.optim.AdamW([p for p in inten.parameters() if p.requires_grad], lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
t0 = time.time(); run = []; fails = 0
order = rng.permutation(len(df)); pos = 0
for step in range(1, STEPS + 1):
    if pos + B > len(order):
        order = rng.permutation(len(df)); pos = 0
    rows = df.iloc[order[pos:pos + B - NR - NNP]]; pos += B - NR - NNP
    if NR:
        rows = pd.concat([rows, rp.iloc[rng.choice(len(rp), NR, replace=False)]], ignore_index=True)
    if NNP:
        rows = pd.concat([rows, npx.iloc[rng.choice(len(npx), NNP, replace=False)]], ignore_index=True)
    try:
        loss = step_batch(rows)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(inten.parameters(), 1.0)
        opt.step(); run.append(loss.item())
    except Exception as e:  # one bad structure must not end the run
        fails += 1; torch.cuda.empty_cache()
        if fails <= 5:
            log("batch failed", repr(e)[:200])
    sched.step()
    if step % 100 == 0:
        print(f"step {step}/{STEPS} loss {np.mean(run[-100:]):.4f} fails {fails} {time.time() - t0:.0f}s", flush=True)
    if step % SAVE == 0 or step == STEPS:
        torch.save(inten.state_dict(), f"{EXT}/ice_ft/inten_{TAG}_{step}.pt")
print("done", flush=True)
