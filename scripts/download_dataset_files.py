"""Download a Kaggle dataset file by file (fallback when the whole-dataset zip endpoint 404s).

Usage: python scripts/download_dataset_files.py owner/slug DEST_DIR
Skips files already present with the expected size; unzips per-file zips Kaggle returns.
"""

import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def list_files(api, ref):
    files, token = [], None
    while True:
        res = api.dataset_list_files(ref, page_token=token, page_size=200)
        files += res.files
        token = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if not token:
            return files


def main(ref, dest):
    api = KaggleApi()
    api.authenticate()
    dest = Path(dest)
    files = list_files(api, ref)
    print(f"{ref}: {len(files)} files")
    for f in files:
        name = f.name
        target = dest / name
        size = int(getattr(f, "total_bytes", 0) or getattr(f, "totalBytes", 0) or 0)
        if target.exists() and (not size or target.stat().st_size == size):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        api.dataset_download_file(ref, name, path=str(target.parent), quiet=True)
        z = target.parent / (Path(name).name + ".zip")
        if z.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target.parent)
            z.unlink()
        print("ok", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
