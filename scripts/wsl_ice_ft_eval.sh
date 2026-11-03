#!/usr/bin/env bash
# Fine-tune ICEBERG's intensity model, then judge the last checkpoint on both isomer panels (WSL).
# Usage: bash scripts/wsl_ice_ft_eval.sh TAG STEPS LR REPLAY_FRAC
set -euo pipefail
cd "$(dirname "$0")/.."
tag=$1; steps=$2; lr=$3; replay=$4
PY=~/miniforge3/envs/casmi2026/bin/python
D=/mnt/d/casmi-data/ice_ft
$PY scripts/ice_finetune.py "$steps" "$lr" "$tag" "$replay" 2>&1 | grep -E "step [0-9]*000/|done|failed|Error|Traceback" || true
ck=$D/inten_${tag}_${steps}.pt
bash scripts/wsl_ice_eval.sh f0 "$tag" --instr Unknown --inten "$ck"
bash scripts/wsl_ice_eval.sh nplib_v2 "np_$tag" --instr Unknown --inten "$ck"
