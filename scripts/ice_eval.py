"""Isomer ranking of ICEBERG on a runner-format panel: rank of the true structure among its candidates.

Usage (WSL): python ice_eval.py PANEL.json TRUTH.csv [--inten STATE_DICT] [--instr TOKEN] [--tag NAME]
--instr overrides the instrument token for timsTOF spectra (the runner sends 'QTOF'). Scores per candidate are written
to D:/casmi-data/ice_ft/scores_{tag}.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ice_env import EXT, load_joint, log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("panel"); ap.add_argument("truth")
ap.add_argument("--inten", default=None); ap.add_argument("--instr", default=None); ap.add_argument("--tag", default="base")
ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()

runner, jm = load_joint("cuda", a.inten)
if a.instr:
    _orig = runner.instr_token
    runner.instr_token = lambda sp: a.instr if runner.instr_family(sp.get("instrument")) == 0 else _orig(sp)

truth = dict(pd.read_csv(a.truth).values)
mols = [m for m in json.load(open(a.panel)) if m["mid"] in truth]
pred = runner.Predictor(jm, "cuda", a.bs, log)
t0 = time.time()
plans = {m["mid"]: runner.molecule_plan(m) for m in mols}
jobs = set()
for m in mols:
    jobs |= runner.molecule_jobs(plans[m["mid"]], m["cands"])
pred.predict(jobs, float("inf"))
ranks, scores = [], {}
for m in mols:
    sc = runner.score_molecule(plans[m["mid"]], m["cands"], pred.cache)
    scores[m["mid"]] = sc
    t = sc.get(truth[m["mid"]])
    if t is None:
        continue
    vals = [v for v in sc.values() if v is not None]
    ranks.append(1 + sum(v > t for v in vals))
ranks = np.array(ranks)
print(f"[{a.tag}] molecules scored {len(ranks)}  top1 {np.mean(ranks == 1):.3f}  mrr {np.mean(1 / ranks):.3f}  "
      f"mean rank {ranks.mean():.2f}  (16 candidates; random mrr {np.mean([1 / k for k in range(1, 17)]):.3f})  "
      f"{time.time() - t0:.0f}s", flush=True)
json.dump(scores, open(f"{EXT}/ice_ft/scores_{a.tag}.json", "w"))
