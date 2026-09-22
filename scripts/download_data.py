"""Download and fingerprint the current official Enveda CASMI 2026 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(data_dir: Path) -> None:
    files = []
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "competition": COMPETITION,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
        "note": "Re-download when Kaggle announces a data revision, then compare manifests.",
    }
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def extract_archives(data_dir: Path) -> None:
    """Extract Kaggle's current competition ZIP without allowing path traversal."""
    root = data_dir.resolve()
    for archive in sorted(data_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (data_dir / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
            bundle.extractall(data_dir)
        print(f"Extracted {archive.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--list", action="store_true", help="List official files without downloading.")
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        raise SystemExit("Kaggle CLI not found. Activate casmi2026 and run `uv sync` first.")

    if args.list:
        command = ["kaggle", "competitions", "files", "-c", COMPETITION]
    else:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "kaggle",
            "competitions",
            "download",
            COMPETITION,
            "-p",
            str(args.data_dir),
        ]
    print("$", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)
    if not args.list:
        extract_archives(args.data_dir)
        write_manifest(args.data_dir)
        print(f"Wrote {args.data_dir / 'MANIFEST.json'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
