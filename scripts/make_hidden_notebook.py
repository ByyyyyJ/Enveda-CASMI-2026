"""Local-validation-only variant of the pinned notebook that can hide answers from every candidate store
(docs/research-plan.md, E3: the "answer not in the pool" pseudo-test). Never submit it.

When the environment variable CASMI_HIDE_KEYS names a file of metric identities (tautomer-canonical
InChIKey14, one per line), the frozen 0.465 pipeline inside the notebook is patched so that:
  Engine 1   pool windows skip those structures (key2pid forgets them too; the analogue generator may
             still rebuild one from its relatives, as it could for a real new structure);
  Engine 2   its COCONUT / ChEBI / LIPID MAPS / PubChemLite / training-structure store drops them;
  PubChem    they no longer count as pool structures, so the PubChem channel may list them, and the
             best pool score (a gate input) is taken over the remaining window only.
Without CASMI_HIDE_KEYS the patches do nothing. The frozen source is re-encoded and its pinned
SHA-256 updated, so the notebook's own integrity checks still pass.

Usage: python scripts/make_hidden_notebook.py
Writes notebooks/local_hide/casmi26-local-hide.ipynb.
"""

import base64
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks" / "public_baseline" / "casmi26-baseline-v2-repro.ipynb"
OUT = ROOT / "notebooks" / "local_hide" / "casmi26-local-hide.ipynb"

HIDE_E1 = """
# ---- local validation only: hide answers from the Engine 1 pool (CASMI_HIDE_KEYS) ----
if os.environ.get('CASMI_HIDE_KEYS'):
    _hide = set(open(os.environ['CASMI_HIDE_KEYS']).read().split())
    _hide_mask = pd.Series(P.key).isin(_hide).values
    _win0 = P.window
    P.window = lambda target, ppm, _w=_win0: (lambda c: c[~_hide_mask[c]])(_w(target, ppm))
    for _k in _hide:
        P.key2pid.pop(_k, None)
    print(f'HIDE engine1: {int(_hide_mask.sum())} pool rows hidden', flush=True)
"""

HIDE_E2 = """
# ---- local validation only: hide answers from the Engine 2 store (CASMI_HIDE_KEYS) ----
if os.environ.get('CASMI_HIDE_KEYS'):
    _p = os.path.join(ENG_DIR, 'casmi_engine.py'); _s = open(_p).read()
    _old = '    P = dict(fp=fp[o], mass=mass[o], keys=keys[o], smiles=smis[o])'
    assert _s.count(_old) == 1, 'casmi_engine.build_pool changed'
    _new = ("    _hide = set(open(os.environ['CASMI_HIDE_KEYS']).read().split())\\n"
            "    _n0 = len(o); o = o[~pd.Series(keys[o]).isin(_hide).values]\\n"
            "    log(f'HIDE engine2: {_n0 - len(o)} store rows hidden')\\n" + _old)
    open(_p, 'w').write(_s.replace(_old, _new))
"""

HIDE_PC = """
# ---- local validation only: hidden answers are not pool structures for the PubChem channel ----
if os.environ.get('CASMI_HIDE_KEYS'):
    open('/kaggle/working/probe_core2.py', 'a').write(
        "\\n\\n_hide_init0 = init_worker\\n"
        "def init_worker(*a, **k):\\n"
        "    _hide_init0(*a, **k)\\n"
        "    import os\\n"
        "    _W['pool_keys'] = set(_W['pool_keys']) - set(open(os.environ['CASMI_HIDE_KEYS']).read().split())\\n")
    _s = open('/kaggle/working/pc_runner.py').read()
    _a = "    pfp = np.load(os.path.join(POOL_DIR, 'pool_fp.npy'), mmap_mode='r')\\n"
    _b = "            best = float((F @ z.astype(np.float64)).max())\\n"
    assert _s.count(_a) == 1 and _s.count(_b) == 1, 'pc_runner changed'
    _s = _s.replace(_a, _a + "    _hk = set(open(os.environ['CASMI_HIDE_KEYS']).read().split())\\n"
                    "    HIDE = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['key']).key.isin(_hk).values\\n")
    _s = _s.replace(_b, "            best = float(np.where(HIDE[a:b], -np.inf, F @ z.astype(np.float64)).max())\\n")
    open('/kaggle/working/pc_runner.py', 'w').write(_s)
    print('HIDE pubchem channel patched', flush=True)
"""

# (anchor line in a frozen code cell, patch inserted after the anchor's statement)
PATCHES = [
    ("L = _external_reference_library(_EXTERNAL_REFERENCE_STATE); P = Pool('work')\n", HIDE_E1),
    ("    open(os.path.join(ENG_DIR, _n), 'w').write(_src)\n", HIDE_E2),
    ("open('/kaggle/working/pc_runner.py', 'w').write(RUNNER)\n", HIDE_PC),
]


def main():
    nb = json.loads(BASE.read_text(encoding="utf-8"))
    cell = nb["cells"][11]
    src = "".join(cell["source"])
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
    cell["source"] = src.splitlines(keepends=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, "frozen sha", new_sha[:12])


if __name__ == "__main__":
    main()
