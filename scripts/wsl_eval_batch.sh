#!/usr/bin/env bash
# Run several validation experiments in sequence; each argument is "NAME:PSEUDO_MODE[:OVERRIDE_SET]".
# Usage: bash scripts/wsl_eval_batch.sh f0np_base:f0np f0np_safe:f0np:f0_safe
REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
for spec in "$@"; do
  IFS=: read -r name mode ovr <<< "$spec"
  echo "##### $name ($mode${ovr:+, overrides $ovr}) $(date +%T)"
  tr -d '\r' < "$REPO/scripts/wsl_eval.sh" | bash -s -- "$name" "$mode" $ovr
done
echo "##### done $(date +%T)"
