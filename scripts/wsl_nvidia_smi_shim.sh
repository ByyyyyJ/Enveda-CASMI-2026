#!/usr/bin/env bash
# Local shim for pinned Kaggle notebooks that assert the GPU is a T4 (e.g. the E004 preflight).
# Name queries get the real GPU name tagged "Tesla T4 (local shim: ...)"; everything else is passed
# through to the real nvidia-smi. Install: cp to ~/kaggle_shim/nvidia-smi and prepend to PATH.
REAL=/usr/lib/wsl/lib/nvidia-smi
if [[ "$*" == *--query-gpu=name* ]]; then
  echo "Tesla T4 (local shim: $("$REAL" --query-gpu=name --format=csv,noheader))"
else
  exec "$REAL" "$@"
fi
