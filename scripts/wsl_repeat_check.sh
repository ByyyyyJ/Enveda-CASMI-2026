#!/usr/bin/env bash
# Determinism check: run a notebook twice locally and compare the output CSV hashes.
# Usage: bash scripts/wsl_repeat_check.sh NOTEBOOK.ipynb OUTDIR
set -uo pipefail
REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
NB=$1; OUT=$2
mkdir -p "$OUT"
for run in a b; do
  bash "$REPO/scripts/wsl_run_notebook.sh" "$NB" "$OUT/run_$run.log" || exit 1
  mkdir -p "$OUT/$run"
  cp /kaggle/working/*.csv /kaggle/working/*.json "$OUT/$run/" 2>/dev/null
done
for f in "$OUT"/a/*.csv; do
  name=$(basename "$f")
  ha=$(sha256sum < "$OUT/a/$name" | cut -c1-16)
  hb=$(sha256sum < "$OUT/b/$name" 2>/dev/null | cut -c1-16)
  [ "$ha" = "$hb" ] && s=SAME || s=DIFF
  echo "$s $name a=$ha b=$hb"
done
