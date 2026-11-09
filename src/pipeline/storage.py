"""Deterministic frame storage for pipeline stages.

CSV, because it is the one format every environment here can read without an
extra dependency; the two functions below are the only place that knows, so
swapping in Parquet is a change to this file alone.

Determinism matters more than the format: the same frame must always produce
the same bytes, because stage fingerprints and the no-op re-run are built on
output hashes. Hence the fixed float and timestamp formats and the atomic
rename (a crash mid-write never leaves a half file that hashes as "changed").
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd

TIMESTAMP_COLUMNS = frozenset({"kickoff_utc", "record_time_utc", "snapshot_time_utc"})
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path, rows: int) -> dict[str, Any]:
    """Manifest entry for one file."""
    return {"path": str(path), "sha256": sha256_file(path), "rows": int(rows)}


def write_frame(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Write ``df`` atomically; returns its manifest entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g", date_format=_TS_FORMAT, lineterminator="\n")
    os.replace(tmp, path)
    return describe(path, len(df))


def read_frame(path: Path) -> pd.DataFrame:
    """Read a stored frame back with its timestamp columns timezone-aware."""
    df = pd.read_csv(path)
    for col in TIMESTAMP_COLUMNS & set(df.columns):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df
