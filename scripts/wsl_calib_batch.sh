#!/usr/bin/env bash
# Proxy-calibration runs of the pub477-family local notebooks (make_local_variant_notebook.py).
# Each argument is "NB:FOLD:KIND" with NB in {midx, ice}, FOLD like f0np, KIND in {safe, hide}.
# Output folder: D:/casmi-data/kernel_out/cal_<NB>_<FOLD>_<KIND>. Uses the f<k>_safe overrides as before.
# Usage: bash scripts/wsl_calib_batch.sh midx:f0np:safe midx:f0np:hide ice:f0np:safe
REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
for spec in "$@"; do
  IFS=: read -r nbk fold kind <<< "$spec"
  case $nbk in
    midx) NB=notebooks/local_midx479/casmi26-local-midx479.ipynb; export ICE_TIMS=0 ;;
    ice)  NB=notebooks/local_midx479_icetims/casmi26-local-midx479-icetims.ipynb; export ICE_TIMS=1 ;;
    *) echo "unknown notebook $nbk"; continue ;;
  esac
  if [ "$kind" = hide ]; then
    export CASMI_HIDE_KEYS=/mnt/d/casmi-data/pseudo/${fold}_hide.txt CASMI_HIDE_SCOPE=all
  else
    unset CASMI_HIDE_KEYS CASMI_HIDE_SCOPE
  fi
  name=cal_${nbk}_${fold}_${kind}
  echo "##### $name $(date +%T)"
  tr -d '\r' < "$REPO/scripts/wsl_eval.sh" | bash -s -- "$name" "$fold" "${fold%np}_safe" "$NB"
  cp /kaggle/working/executed.ipynb "/mnt/d/casmi-data/kernel_out/$name/executed.ipynb" 2>/dev/null
  grep -ohE "(HIDE [a-z0-9 ]+:[^\"\\]*|LOCAL: [^\"\\]*|ICEBERG: [^\"\\]*)" /kaggle/working/executed.ipynb \
      /mnt/d/casmi-data/kernel_out/$name/*.log 2>/dev/null | sort -u | head
done
echo "##### done $(date +%T)"
