#!/usr/bin/env bash
# Recreate Kaggle's /kaggle/input mount layout inside WSL with symlinks, so public notebooks run
# unchanged. Requires /kaggle to exist and be owned by the user (one-time sudo).
# Usage: bash scripts/wsl_kaggle_layout.sh [COMPETITION_DATA_DIR]
#   COMPETITION_DATA_DIR defaults to the repo's data/raw; pass a validation pseudo-test dir instead
#   to run the pipeline on held-out queries.
set -euo pipefail

REPO=/mnt/d/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026
EXT=/mnt/d/casmi-data
COMP=${1:-$REPO/data/raw}

# Directories are real and only files are symlinks (cp -rs): pathlib's rglob in Python 3.12 does
# not descend into symlinked directories, and some notebooks search with it.
rm -rf /kaggle/input
mkdir -p /kaggle/input/competitions /kaggle/input/datasets /kaggle/working
cp -rs "$COMP" /kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra

link() {  # owner slug
  mkdir -p "/kaggle/input/datasets/$1"
  cp -rs "$EXT/$2" "/kaggle/input/datasets/$1/$2"
}
link sanpier casmi-spectrum-model-v1
link sanpier catalogue-experiment-saner
for s in casmi26-fold-safe-fpnet casmi26-natural-product-knowledge-table \
         casmi26-natural-product-panel-benchmark casmi26-pubchem-popularity-prior \
         casmi26-attributed-v4g-reproduction-assets; do
  link dmitriigluzdov "$s"
done

# Optional input overrides (e.g. leak removal for validation): OVERRIDES=<dir> holds files at paths
# relative to /kaggle/input that replace the originals, plus REMOVE.txt listing paths to delete.
if [ -n "${OVERRIDES:-}" ]; then
  (cd "$OVERRIDES" && find . -type f ! -name REMOVE.txt -printf '%P\n') | while read -r rel; do
    ln -sfn "$OVERRIDES/$rel" "/kaggle/input/$rel"; echo "override: $rel"
  done
  if [ -f "$OVERRIDES/REMOVE.txt" ]; then
    tr -d '\r' < "$OVERRIDES/REMOVE.txt" | while read -r rel; do
      [ -n "$rel" ] && rm -f "/kaggle/input/$rel" && echo "removed: $rel"
    done
  fi
fi

for d in /kaggle/input/competitions/* /kaggle/input/datasets/*/*; do
  echo "$d: $(find "$d" -type l | wc -l) files"
done
