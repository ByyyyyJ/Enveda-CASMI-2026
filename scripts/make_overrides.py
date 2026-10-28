"""Build leak-removal input overrides for running the public pipeline on np-examples validation.

Override set "np_clean" (files under D:/casmi-data/overrides/np_clean/, paths mirror /kaggle/input):
  - the megayak simulated ranker rows, minus every row from the 'np' query set (Q == 0), which is
    exactly the 250 enveda-np-examples molecules the validation queries use;
  - REMOVE casmi26-fpnet-full1/fpnet_full1.pt: trained on all folds including np-examples. When it is
    missing, the pipeline falls back to the v4b fpnet_0 engine bank.
Not removable without code changes (reported as residual leakage): rank_train.npz (provenance
unclear), the ICEBERG/GLACIER MassSpecGym checkpoints, and the neural scorer.

Override set "f0_safe" (for the f0np pseudo-test, whose molecules are all in identity fold 0):
  every FPNet with the 10,226-bit public architecture is replaced by fpnet_fold0.pt (trained
  without fold 0 and the np panel): v3 fpnet_0/fpnet_1 (PubChem channel), v4b fpnet_0 and the
  engine feature models fe_A/fe_B/fe_D; v4b MANIFEST.json is rewritten with the new hashes;
  fpnet_full1 is removed. The A/B/D views become one model, so model_views features lose their
  spread, which costs some accuracy: the f0_safe score is a slightly pessimistic estimate.
  Residual leakage: the 6,930-bit Engine 2 FPNets (casmi26-fp-models-v2), the pretrained GBM
  rankers (ranker_0, ranker_v4b), feature tables of the v4b families and the neural scorer.

f1_safe..f4_safe: the same with fpnet_fold<k>.pt, for the f<k>np pseudo-tests.

Usage: python scripts/make_overrides.py [np_clean|f0_safe|f1_safe|...|f4_safe]
Writes overrides/<set>/{files...} and overrides/<set>/REMOVE.txt.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL  # noqa: E402

V4G = "datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets"
SIM = "casmi26-simulated-ranker-rows"
FULL1 = f"{V4G}/casmi26-fpnet-full1/fpnet_full1.pt"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fold_safe(k):
    """f<k>_safe. The PubChem channel's own fpnet_fold0.pt is SHA-pinned by the pipeline and stays;
    for k > 0 it has seen fold k, but that channel only scores structures outside the pool, which
    never include f<k>np answers."""
    out = EXTERNAL / "overrides" / f"f{k}_safe"
    if out.exists():
        shutil.rmtree(out)
    fold0 = EXTERNAL / "casmi26-fold-safe-fpnet" / f"fpnet_fold{k}.pt"
    src = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets"
    v4b = "casmi26-v4b-models"
    targets = ["casmi26-v3-models/fpnet_0.pt", "casmi26-v3-models/fpnet_1.pt",
               f"{v4b}/fpnet_0.pt", f"{v4b}/fe_A.pt", f"{v4b}/fe_B.pt", f"{v4b}/fe_D.pt"]
    for rel in targets:
        dst = out / V4G / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(fold0, dst)  # hard link: same drive, no copy; the layout script needs real files
        print("replace:", rel)
    man = json.loads((src / v4b / "MANIFEST.json").read_text())
    h = sha256(fold0)
    for rel in ("fpnet_0.pt", "fe_A.pt", "fe_B.pt", "fe_D.pt"):
        assert rel in man["files"], rel
        man["files"][rel] = h
    (out / V4G / v4b / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    (out / "REMOVE.txt").write_text(FULL1 + "\n")
    print("remove:", FULL1)


def main():
    if len(sys.argv) > 1 and sys.argv[1].endswith("_safe"):
        return fold_safe(int(sys.argv[1][1]))
    out = EXTERNAL / "overrides" / "np_clean"
    src = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets" / SIM
    for f in ("sim_rank_rows_nofp.npz", "sim_rank_rows_fp16k.npz", "sim_rank_rows_fppair.npz"):
        z = dict(np.load(src / f, allow_pickle=True))
        np_q = int(np.flatnonzero(z["sets"] == "np")[0])
        keep = z["Q"] != np_q
        for k in ("X", "Y", "S", "G", "Q"):
            z[k] = z[k][keep]
        dst = out / V4G / SIM / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **z)
        print(f"{f}: kept {keep.sum():,} / {keep.size:,} rows (dropped query set 'np')")
    (out / "REMOVE.txt").write_text(FULL1 + "\n")
    print("remove:", (out / "REMOVE.txt").read_text().strip())


if __name__ == "__main__":
    main()
