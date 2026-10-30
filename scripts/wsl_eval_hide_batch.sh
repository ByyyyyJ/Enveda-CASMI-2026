#!/usr/bin/env bash
# "Answer not in the pool" validation runs (docs/research-plan.md, E3): like wsl_eval_batch.sh, but each
# run uses the local-only hide notebook with CASMI_HIDE_KEYS=pseudo/<MODE>_hide.txt (the pseudo-test's
# answers, built by make_hide_keys.py). Each argument is "NAME:PSEUDO_MODE:OVERRIDE_SET".
# Set CASMI_HIDE_SCOPE=e1 to hide from the Engine 1 pool only (Engine 2 store keeps the answers).
# Usage: [CASMI_HIDE_SCOPE=e1] bash scripts/wsl_eval_hide_batch.sh f0np_hide:f0np:f0_safe f1np_hide:f1np:f1_safe
REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
NB=notebooks/local_hide/casmi26-local-hide.ipynb
export CASMI_HIDE_SCOPE=${CASMI_HIDE_SCOPE:-all}
for spec in "$@"; do
  IFS=: read -r name mode ovr <<< "$spec"
  export CASMI_HIDE_KEYS=/mnt/d/casmi-data/pseudo/${mode}_hide.txt
  [ -s "$CASMI_HIDE_KEYS" ] || { echo "missing $CASMI_HIDE_KEYS"; continue; }
  echo "##### $name ($mode, overrides $ovr, scope $CASMI_HIDE_SCOPE, hidden $(wc -l < "$CASMI_HIDE_KEYS") answers) $(date +%T)"
  tr -d '\r' < "$REPO/scripts/wsl_eval.sh" | bash -s -- "$name" "$mode" "$ovr" "$NB"
  grep -h "HIDE" "/mnt/d/casmi-data/kernel_out/$name/run.log" /mnt/d/casmi-data/kernel_out/$name/*.log 2>/dev/null | sort -u | head
done
echo "##### done $(date +%T)"
