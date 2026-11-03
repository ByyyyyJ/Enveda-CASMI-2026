"""Local-validation copies of the pub477-family notebooks (midx479, midx479_icetims), for proxy calibration
(docs/research-plan.md, 2026-11-03). Never submit them.

Two local-only changes:
  1. the answer-hiding patches of make_hidden_notebook.py go into the frozen 0.465 pipeline (they do nothing
     unless CASMI_HIDE_KEYS is set), with the pinned SHA-256 updated;
  2. a cell before the final layer saves the fragment-check file as submission_physical.csv, so that pub477
     (rank 1 from that file, ranks 2-25 from CUSTOM) can be rebuilt offline from the same run.

Usage: python scripts/make_local_variant_notebook.py SRC_NOTEBOOK OUT_NOTEBOOK
"""

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hidden_notebook import PATCHES  # noqa: E402

FINAL_MARK = "LIB_MAX = {}\n# "
SAVE_CELL = """# ---- local validation only: keep the fragment-check file for offline pub477 rebuild ----
import shutil as _shl
_shl.copy2(ROOT / 'submission.csv', ROOT / 'submission_physical.csv')
print('LOCAL: saved submission_physical.csv', flush=True)
"""


def patch_frozen(nb):
    cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "FROZEN_B64 = '" in "".join(c["source"])]
    assert len(cells) == 1
    src = "".join(cells[0]["source"])
    b64 = re.search(r"FROZEN_B64 = '([^']+)'", src).group(1)
    old_sha = re.search(r"FROZEN_SHA256 = '([0-9a-f]{64})'", src).group(1)
    raw = base64.b64decode(b64)
    assert hashlib.sha256(raw).hexdigest() == old_sha
    fz = json.loads(raw)
    for anchor, patch in PATCHES:
        hits = [c for c in fz["cells"] if c["cell_type"] == "code" and anchor in "".join(c["source"])]
        assert len(hits) == 1, anchor
        s = "".join(hits[0]["source"]).replace(anchor, anchor + patch, 1)
        hits[0]["source"] = s.splitlines(keepends=True)
    new_raw = json.dumps(fz, ensure_ascii=False).encode("utf-8")
    new_sha = hashlib.sha256(new_raw).hexdigest()
    src = src.replace(b64, base64.b64encode(new_raw).decode()).replace(old_sha, new_sha)
    cells[0]["source"] = src.splitlines(keepends=True)
    return new_sha


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    nb = json.loads(src.read_text(encoding="utf-8"))
    sha = patch_frozen(nb)
    at = [i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith(FINAL_MARK)]
    assert len(at) == 1, at
    nb["cells"].insert(at[0], dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                                   source=SAVE_CELL.splitlines(keepends=True)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out, "frozen sha", sha[:12], "save cell before", at[0])


if __name__ == "__main__":
    main()
