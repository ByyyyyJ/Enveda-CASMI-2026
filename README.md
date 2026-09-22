# Enveda CASMI 2026

Local, reproducible preparation workspace for the Kaggle competition **Enveda CASMI 2026 – Molecule ID From Mass Spectra**.

## Quick start

The prepared environment is named `casmi2026`. To recreate it on this Windows machine:

```powershell
conda env create -f environment.yml
conda activate casmi2026
python -m pip install uv
$env:UV_PROJECT_ENVIRONMENT = $env:CONDA_PREFIX
uv sync --system-certs --no-install-project
python -m ipykernel install --user --name casmi2026 --display-name "Python (casmi2026)"
```

Then:

1. Open a new PowerShell session so the user-level `KAGGLE_API_TOKEN` is available.
2. Activate the environment: `conda activate casmi2026`.
3. Check authentication and list official files: `python scripts/download_data.py --list`.
4. Download the current competition data: `python scripts/download_data.py`.
5. Record hashes and inspect schemas: `python scripts/verify_data.py`.
6. Open `notebooks/01_data_exploration.ipynb` for local exploration.

Raw data is intentionally stored in `data/raw/` and ignored by Git. It is inside this OneDrive-backed project by design. Do not commit credentials, downloaded data, generated artifacts, or `submission.csv`.

## Layout

- `docs/`: competition, data, and research notes in Chinese.
- `scripts/`: download, integrity verification, and submission-format checks.
- `notebooks/`: local exploration and Kaggle-compatible submission scaffolds.
- `data/raw/`: locally downloaded competition files (untracked).

See `docs/competition.md` before using external data or models; final submissions must be produced by an offline Kaggle Notebook and named `submission.csv`.
