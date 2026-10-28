#!/usr/bin/env bash
# Run a repo Python script inside WSL with the casmi2026 env (via D:/casmi-data/run_py.sh), from Git Bash.
# Usage: bash scripts/wsl_py.sh scripts/<name>.py [args...]
export MSYS_NO_PATHCONV=1
script=$1; shift
wsl.exe -d Ubuntu-24.04 -e bash /mnt/d/casmi-data/run_py.sh \
  "/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026/$script" "$@" 2>&1 | grep -v "^missing "
