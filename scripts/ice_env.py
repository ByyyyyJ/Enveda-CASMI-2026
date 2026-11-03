"""Load the shipped ICEBERG (ms-pred, msg_all checkpoint) in-process, with the runner's own environment setup.

Reuses ice_runner.py from the casmi26-iceberg asset (offline wheel install into SITE, shim + src on sys.path,
RDKit 2025.03.6) so fine-tuned weights stay consistent with the Kaggle inference path. Run under WSL.
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

EXT = "/mnt/d/casmi-data" if os.name != "nt" else "D:/casmi-data"
PKG = f"{EXT}/casmi26-attributed-v4g-reproduction-assets/casmi26-iceberg"
SITE = f"{EXT}/ice_site"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_runner():
    spec = importlib.util.spec_from_file_location("ice_runner", f"{PKG}/ice_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.setup_env(SimpleNamespace(pkg=PKG, site=SITE), log)
    return runner


def load_joint(dev="cuda", inten_ckpt=None):
    """-> (runner module, JointModel). inten_ckpt replaces the shipped intensity weights (a state_dict file)."""
    import torch
    runner = load_runner()
    jm = runner.load_model(PKG, dev, log)
    if inten_ckpt:
        jm.inten_model_obj.load_state_dict(torch.load(inten_ckpt, map_location=dev, weights_only=True))
    return runner, jm


if __name__ == "__main__":
    import torch
    runner, jm = load_joint()
    import rdkit
    print("rdkit", rdkit.__version__)
    for name in ("gen.pt", "inten.pt"):
        c = torch.load(f"{PKG}/ckpt/{name}", map_location="cpu", weights_only=True)
        hp = c["hyper_parameters"]
        print(name, {k: hp[k] for k in sorted(hp) if k in (
            "loss_fn", "embed_instrument", "embed_adduct", "embed_collision", "binned_targs", "include_unshifted_mz",
            "hidden_size", "learning_rate", "contr_weight", "root_encode", "mpnn_type", "encode_forms", "add_hs")})
    print("inten buckets", jm.inten_model_obj.inten_buckets.shape, jm.inten_model_obj.inten_buckets[:3].tolist(),
          jm.inten_model_obj.inten_buckets[-1].item())
    out = jm.predict_mol(["CC(=O)Oc1ccccc1C(=O)O"], collision_eng=[20.0], precursor_mz=[0.0], adduct=["[M+H]+"],
                         threshold=0.0, device="cuda", max_nodes=100, instrument=["QTOF"], binned_out=True)
    print("predict ok", out["spec"][0].shape, out["binned_spec"].shape)
