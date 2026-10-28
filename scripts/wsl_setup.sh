#!/usr/bin/env bash
# One-time WSL2 (Ubuntu 24.04) environment for running the Kaggle pipeline locally.
# Mirrors the Kaggle image where it matters: Python 3.12, torch cu128, RDKit 2026.3.3.
set -euo pipefail

if [ ! -x "$HOME/miniforge3/bin/conda" ]; then
  cd /tmp
  curl -fsSL -o miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash miniforge.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

conda env list | grep -q '^casmi2026 ' || conda create -y -n casmi2026 python=3.12
conda activate casmi2026

REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
pip install --upgrade pip
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# Numeric stack pinned to the Kaggle image (env/kaggle_pip_freeze_20261026.txt) so local runs
# reproduce Kaggle outputs; notebook tooling (ipython, nbclient) is left unpinned.
grep -ivE "^(ipython|ipykernel|nbformat|nbclient)==" "$REPO/env/kaggle_core_pins.txt" > /tmp/pins.txt
pip install rdkit==2026.3.3 -r /tmp/pins.txt ipython ipykernel nbformat nbclient tqdm pytest kaggle

python - <<'EOF'
import torch, rdkit, lightgbm, numba, sklearn
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", float((x @ x).sum()) != 0)
print("rdkit", rdkit.__version__, "lightgbm", lightgbm.__version__, "numba", numba.__version__,
      "sklearn", sklearn.__version__)
EOF
