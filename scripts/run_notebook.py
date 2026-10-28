"""Execute a Kaggle notebook unchanged inside the WSL /kaggle layout (see wsl_kaggle_layout.sh).

Usage (in WSL, casmi2026 env): python scripts/run_notebook.py NOTEBOOK.ipynb [--clean]
Runs with cwd /kaggle/working and writes the executed copy next to it as executed.ipynb.
--clean empties /kaggle/working first, as a fresh Kaggle session would be.
"""

import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

WORK = Path("/kaggle/working")


def main():
    nb_path = Path(sys.argv[1])
    if "--clean" in sys.argv and WORK.exists():
        for p in WORK.iterdir():
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    WORK.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(WORK)}})
    t0 = time.time()
    try:
        client.execute()
    finally:
        nbformat.write(nb, WORK / "executed.ipynb")
        print(f"done in {(time.time() - t0) / 60:.1f} min -> {WORK / 'executed.ipynb'}", flush=True)


if __name__ == "__main__":
    main()
