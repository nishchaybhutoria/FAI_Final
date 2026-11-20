"""Shared paths and helpers for the experiment scripts.

Every script can be run from any working directory: paths are resolved from the
repository root, and ``src/`` is put on ``sys.path`` so that the package works
without installation.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

CONFIGS = ROOT / "configs"
DATA = ROOT / "results" / "data"          # recorded study data and analysis outputs
RERUN = ROOT / "results" / "rerun"        # default target for regenerated raw data
FIGURES = ROOT / "results" / "figures"
REPORT = ROOT / "report"

# Columns that measure wall-clock time; they are the only columns allowed to differ on a rerun.
TIMING = ("wall_s", "solve_ms_mean", "solve_ms_p95")


def config(name: str) -> dict:
    return tomllib.loads((CONFIGS / f"{name}.toml").read_text())


def config_sha256(name: str) -> str:
    return hashlib.sha256((CONFIGS / f"{name}.toml").read_bytes()).hexdigest()


def read_csv(path: Path, exact: bool = True) -> pd.DataFrame:
    """Read a results table.

    ``exact=True`` round-trips floats bit for bit (used for rerun comparisons and by the
    recovery analysis). ``exact=False`` uses pandas' default parser, which can differ in
    the last bit; the commitment and sweep analyses were frozen with it, so they keep it
    to reproduce their archived outputs exactly.
    """
    return pd.read_csv(path, float_precision="round_trip") if exact else pd.read_csv(path)


def write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """Deterministic gzip (no timestamp) so identical data give identical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def through_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Pass a frame through the same CSV write/read path as the saved files."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision="round_trip")


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def compare_rows(saved: pd.DataFrame, new: pd.DataFrame, keys: list[str], label: str) -> int:
    """Require ``new`` to reproduce ``saved`` exactly on every non-timing column.

    ``new`` may cover a subset of the saved rows (quick mode); every new row must
    have a saved counterpart with identical values. Returns the number of rows compared.
    """
    cols = [c for c in saved.columns if c not in TIMING]
    # Diagnostic columns exist only for some policies (e.g. the adaptive rule); when those
    # policies are not rerun the column is absent, and the recorded value is NaN for the rest.
    new = new.assign(**{c: float("nan") for c in cols if c not in new.columns})
    new = through_csv(new[cols])
    a = saved[cols].set_index(keys).sort_index()
    b = new.set_index(keys).sort_index()
    assert b.index.is_unique, f"{label}: duplicate keys in rerun"
    absent = b.index.difference(a.index)
    assert absent.empty, f"{label}: rerun produced rows that are not in the recorded data: {list(absent[:3])}"
    pd.testing.assert_frame_equal(a.loc[b.index], b, check_exact=True, check_dtype=False, check_like=True,
                                  obj=f"{label} rerun vs recorded")
    return len(b)


def run_metadata(name: str, runs: int, wall: float, **extra) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = ""
    import numpy
    import scipy
    return dict(study=name, runs=runs, wall_s=round(wall, 1), config_sha256=config_sha256(name),
                git_head=head or None, python=platform.python_version(), numpy=numpy.__version__,
                scipy=scipy.__version__, pandas=pd.__version__, machine=platform.machine(),
                cpus=os.cpu_count(), **extra)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
