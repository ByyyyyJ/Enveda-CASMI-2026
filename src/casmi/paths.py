"""Filesystem locations. External datasets live outside the OneDrive-synced repo.

Override the external root with the CASMI_EXTERNAL environment variable.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
SPLITS = ROOT / "data/splits"
EXTERNAL = Path(os.environ.get("CASMI_EXTERNAL", "D:/casmi-data" if os.name == "nt" else "/mnt/d/casmi-data"))
