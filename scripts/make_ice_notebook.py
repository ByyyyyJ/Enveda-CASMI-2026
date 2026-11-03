"""Build notebooks/midx479_icetims: the midx479 (LB 0.479) notebook running ICEBERG from wyy817/casmi26-iceberg-tims.

The wrapper writes the pinned 0.465 pipeline to frozen_0465.ipynb and checks its SHA-256; right after that check this
patch points the pipeline's ICEBERG lookups at the tuned package (when it is mounted) and updates FROZEN_SHA256, which
runner.py checks again. Without the dataset mounted the notebook is the 0.479 notebook unchanged.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks/midx479"
OUT = ROOT / "notebooks/midx479_icetims"

nb = json.loads((SRC / "casmi26-midx479.ipynb").read_text(encoding="utf-8"))
CHECK = "    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == FROZEN_SHA256, 'Pinned pipeline source differs'\n"
PATCH = CHECK + (
    "    _ice_tims = glob.glob('/kaggle/input/**/casmi26-iceberg-tims/ice_runner.py', recursive=True)\n"
    "    if _ice_tims:  # timsTOF-tuned ICEBERG package (wyy817/casmi26-iceberg-tims)\n"
    "        _t = FROZEN.read_text()\n"
    "        _n = _t.count('casmi26-iceberg/**/ice_runner.py')\n"
    "        assert _n >= 2, 'ICEBERG lookups not found in the pinned pipeline'\n"
    "        FROZEN.write_text(_t.replace('casmi26-iceberg/**/ice_runner.py', 'casmi26-iceberg-tims/**/ice_runner.py'))\n"
    "        FROZEN_SHA256 = hashlib.sha256(FROZEN.read_bytes()).hexdigest()\n"
    "        print('ICEBERG: timsTOF-tuned package', _ice_tims[0], '|', _n, 'lookups redirected', flush=True)\n")
done = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if CHECK in src:
        assert "    global FROZEN\n" in src
        src = src.replace("    global FROZEN\n", "    global FROZEN, FROZEN_SHA256\n").replace(CHECK, PATCH)
        if "import glob" not in src:
            src = "import glob\n" + src
        c["source"] = src.splitlines(keepends=True)
        done += 1
assert done == 1, done
OUT.mkdir(exist_ok=True)
(OUT / "casmi26-midx479-icetims.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
meta = json.loads((SRC / "kernel-metadata.json").read_text())
meta.update(id="wyy817/casmi26-midx479-icetims", title="casmi26 midx479 icetims", code_file="casmi26-midx479-icetims.ipynb")
meta["dataset_sources"] = meta["dataset_sources"] + ["wyy817/casmi26-iceberg-tims"]
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("notebook", OUT)
