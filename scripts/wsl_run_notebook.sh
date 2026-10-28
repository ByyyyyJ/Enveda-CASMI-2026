#!/usr/bin/env bash
# Run a pinned Kaggle notebook locally in WSL with the Kaggle-like layout.
# Usage: [OVERRIDES=/mnt/d/casmi-data/overrides/<set>] bash scripts/wsl_run_notebook.sh NOTEBOOK.ipynb LOGFILE [COMPETITION_DATA_DIR]
set -euo pipefail
REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
NB=$1; LOG=$2; COMP=${3:-}

mkdir -p ~/kaggle_shim
tr -d '\r' < "$REPO/scripts/wsl_nvidia_smi_shim.sh" > ~/kaggle_shim/nvidia-smi
chmod +x ~/kaggle_shim/nvidia-smi
export PATH="$HOME/kaggle_shim:$PATH"

# OVERRIDES (env, optional) is passed through to the layout script.
tr -d '\r' < "$REPO/scripts/wsl_kaggle_layout.sh" | bash -s -- ${COMP:+"$COMP"} | grep -E '^(override|removed):' || true
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate casmi2026
echo "gpu name as seen by notebooks: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd "$REPO"
python scripts/run_notebook.py "$NB" --clean > "$LOG" 2>&1 || { echo "FAILED, see $LOG"; exit 1; }
echo "OK, see $LOG"
