"""Experiment A runner: commitment policies with instantaneous schedule installation.

Each job simulates one full stream (cell, split, replication) under every listed
policy; all policies of a job share the identical exogenous scenario (arrivals,
lanes, private information signs). Rows are one per (cell, policy, replication).

Usage (from the repository root):
    python experiments/commitment/run.py --out results/rerun/commitment            # full design
    python experiments/commitment/run.py --out results/rerun/commitment --quick    # 2 cells x 2 reps
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, default_workers, run_metadata, write_csv_gz, write_json  # noqa: E402

import pandas as pd  # noqa: E402

from lockorder.evaluation.runner import _job  # noqa: E402

NAME = "commitment"
C = config(NAME)
KEYS = ["condition", "rho", "sigma", "policy", "rep"]
QUICK = dict(rho=[0.5, 0.9], sigma=[1.0], reps=2)


def jobs(split: str, quick: bool = False) -> list[tuple]:
    s = C[split]
    rhos, sigmas, reps = (QUICK["rho"], QUICK["sigma"], QUICK["reps"]) if quick else (s["rho"], s["sigma"], s["reps"])
    return [(C["condition"], float(r), float(sg), list(s["policies"]), split, k, {})
            for r in rhos for sg in sigmas for k in range(reps)]


def run_split(split: str, workers: int, quick: bool = False) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_job, jobs(split, quick), chunksize=2):
            rows.extend(res)
    df = pd.DataFrame(rows)
    bad = df[(df.error.fillna("") != "") | (df.n_violations != 0) | (~df.drained.astype(bool))]
    assert bad.empty, f"failed or unsafe runs:\n{bad[KEYS + ['error', 'n_violations']]}"
    return df


def main(out: Path, workers: int, quick: bool = False) -> dict:
    t0 = time.perf_counter()
    n = 0
    for split in ("tune", "test"):
        df = run_split(split, workers, quick)
        write_csv_gz(df, out / f"{split}.csv.gz")
        n += len(df)
        print(f"[commitment] {split}: {len(df)} runs", flush=True)
    prov = run_metadata(NAME, n, time.perf_counter() - t0, quick=quick, workers=workers)
    write_json(out / "run_metadata.json", prov)
    print(f"[commitment] {n} runs in {prov['wall_s']:.0f} s with {workers} workers", flush=True)
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.out, a.workers, a.quick)
