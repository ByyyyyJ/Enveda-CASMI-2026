"""Verify downloaded data and write a compact, versionable schema report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
REPORT = ROOT / "docs" / "data-dictionary.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_parquet(path: Path) -> list[str]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    lines = [f"## `{path.name}`", "", f"- 行组：{parquet.num_row_groups}", f"- 行数：{parquet.metadata.num_rows}", "", "| 字段 | Arrow 类型 |", "| --- | --- |"]
    lines.extend(f"| `{field.name}` | `{field.type}` |" for field in schema)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"Missing {data_dir}; run scripts/download_data.py first.")

    manifest_path = data_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    mismatches = []
    for item in manifest.get("files", []):
        path = data_dir / item["name"]
        if not path.exists() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            mismatches.append(item["name"])
    if mismatches:
        raise SystemExit(f"Manifest mismatch: {', '.join(mismatches)}")

    report = ["# 数据字典（自动生成）", "", "本文件由 `scripts/verify_data.py` 从当前下载的数据生成。", ""]
    for path in sorted(data_dir.glob("*.parquet")):
        report.extend(describe_parquet(path))
        report.append("")
    sample = data_dir / "sample_submission.csv"
    if sample.exists():
        frame = pd.read_csv(sample)
        report.extend(["## `sample_submission.csv`", "", f"- 行数：{len(frame)}", "", "| 字段 | pandas 类型 |", "| --- | --- |"])
        report.extend(f"| `{name}` | `{dtype}` |" for name, dtype in frame.dtypes.items())
        report.append("")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Integrity OK. Wrote {REPORT}")


if __name__ == "__main__":
    main()
