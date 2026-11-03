#!/usr/bin/env bash
# Run ice_eval.py under WSL. Usage: bash scripts/wsl_ice_eval.sh PANEL_NAME TAG [extra ice_eval args...]
# PANEL_NAME f0 -> ice_ft/panel_f0.json + panel_truth.csv; otherwise ice_ft/panel_{NAME}.json + panel_{NAME}_truth.csv
set -euo pipefail
cd "$(dirname "$0")/.."
D=/mnt/d/casmi-data/ice_ft
name=$1; tag=$2; shift 2
if [ "$name" = f0 ]; then truth=$D/panel_truth.csv; else truth=$D/panel_${name}_truth.csv; fi
~/miniforge3/envs/casmi2026/bin/python scripts/ice_eval.py "$D/panel_${name}.json" "$truth" --tag "$tag" "$@" 2>&1 \
  | grep -E '^\[|Error|Traceback' || true
