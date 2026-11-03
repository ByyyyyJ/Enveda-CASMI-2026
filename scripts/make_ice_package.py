"""Build the private Kaggle dataset wyy817/casmi26-iceberg-tims: the casmi26-iceberg package with timsTOF-tuned
intensity weights and timsTOF spectra sent with the 'Unknown' instrument token (the token the weights were tuned on).

Usage: python make_ice_package.py INTEN_STATE_DICT [OUT_DIR]   (default OUT_DIR D:/casmi-data/ice_tims_pkg)
Everything else (generator checkpoint, ms-pred source, shims, wheels, fuse.py) is copied unchanged.
"""
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.paths import EXTERNAL  # noqa: E402

SRC = EXTERNAL / "casmi26-attributed-v4g-reproduction-assets/casmi26-iceberg"
sd_path = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else EXTERNAL / "ice_tims_pkg"
if out.exists():
    raise SystemExit(f"{out} exists; choose a new directory")
shutil.copytree(SRC, out)

ck = torch.load(SRC / "ckpt/inten.pt", map_location="cpu", weights_only=True)
sd = torch.load(sd_path, map_location="cpu", weights_only=True)
assert set(sd) == set(ck["state_dict"]), "state_dict keys differ from the shipped intensity checkpoint"
torch.save({"hyper_parameters": ck["hyper_parameters"], "state_dict": sd}, out / "ckpt/inten.pt")

runner = (out / "ice_runner.py").read_text()
old = "def instr_token(sp):\n"
new = ("def instr_token(sp):\n"
       "    if instr_family(sp.get('instrument')) == 0:      # casmi26-iceberg-tims: weights tuned on timsTOF as 'Unknown'\n"
       "        return 'Unknown'\n")
assert runner.count(old) == 1
(out / "ice_runner.py").write_text(runner.replace(old, new))
(out / "MANIFEST.json").unlink(missing_ok=True)
(out / "dataset-metadata.json").write_text(json.dumps({
    "title": "casmi26-iceberg-tims", "id": "wyy817/casmi26-iceberg-tims",
    "licenses": [{"name": "other"}]}, indent=2))
(out / "TUNING.md").write_text(
    "ICEBERG 2.1 (ms-pred, msg_all) with the intensity model fine-tuned on enveda-180 timsTOF spectra (identity folds "
    f"1-4) mixed with MassSpecGym replay; generator unchanged. Weights: {sd_path.name}. timsTOF spectra use the "
    "'Unknown' instrument token. ms-pred is MIT (src/ms_pred/LICENSE).\n")
print("package", out)
