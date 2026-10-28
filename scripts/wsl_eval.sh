#!/usr/bin/env bash
# One local validation experiment: run the pinned notebook on a pseudo-test, keep its outputs, score.
# Usage: bash scripts/wsl_eval.sh NAME PSEUDO_MODE [OVERRIDE_SET] [NOTEBOOK]
#   NAME          output folder under D:/casmi-data/kernel_out/
#   PSEUDO_MODE   np-lib | np-nps (built by make_pseudo_test.py)
#   OVERRIDE_SET  optional, e.g. np_clean (built by make_overrides.py)
set -uo pipefail
REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
EXT=/mnt/d/casmi-data
NAME=$1; MODE=$2; OVR=${3:-}
NB=${4:-notebooks/public_baseline/casmi26-baseline-v2-repro.ipynb}
OUT=$EXT/kernel_out/$NAME
mkdir -p "$OUT"

if [ -n "$OVR" ]; then export OVERRIDES=$EXT/overrides/$OVR; else unset OVERRIDES; fi
bash "$REPO/scripts/wsl_run_notebook.sh" "$NB" "$OUT/run.log" "$EXT/pseudo/$MODE" || exit 1
cp /kaggle/working/*.csv /kaggle/working/*.json /kaggle/working/*.log /kaggle/working/*.parquet "$OUT/" 2>/dev/null
cp /kaggle/working/ice_work/ice_out.json "$OUT/ice_out.json" 2>/dev/null
cp /kaggle/working/gl_work/gl_out.json "$OUT/gl_out.json" 2>/dev/null

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate casmi2026
cd "$REPO"
for f in submission.csv submission_public0466.csv submission_baseline0465.csv; do
  [ -f "$OUT/$f" ] || continue
  echo "== $NAME / $f"
  python scripts/score_submission.py "$OUT/$f" "$EXT/pseudo/${MODE}_solution.csv" \
    --ranks "$OUT/ranks_${f%.csv}.csv" 2>&1 | grep -v Warning
done
