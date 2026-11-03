#!/usr/bin/env bash
# Two-way CV of ICEBERG fine-tuning with enveda-np-examples: train on one half (plus enveda-180 and replay),
# score the np-lib panel on the other half. Usage: bash scripts/wsl_ice_np_cv.sh TAG STEPS LR REPLAY_FRAC [HALVES]
set -euo pipefail
cd "$(dirname "$0")/.."
tag=$1; steps=$2; lr=$3; replay=$4; halves=${5:-"A B"}
PY=~/miniforge3/envs/casmi2026/bin/python
D=/mnt/d/casmi-data/ice_ft
for h in $halves; do
  other=$([ "$h" = A ] && echo B || echo A)
  $PY scripts/ice_finetune.py "$steps" "$lr" "${tag}$h" "$replay" "$h" 2>&1 | grep -E "step [0-9]*000/|done|failed|Error|Traceback" || true
  $PY scripts/ice_eval.py $D/panel_nplib_v2.json $D/panel_nplib_v2_truth_$other.csv --tag "np_${tag}$h" --instr Unknown \
      --inten $D/inten_${tag}${h}_${steps}.pt 2>&1 | grep -E '^\[|Error|Traceback' || true
done
