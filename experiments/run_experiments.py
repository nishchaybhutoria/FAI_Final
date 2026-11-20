"""Run the report's simulation studies and check them against the recorded data.

Modes
  quick  Rerun a small deterministic subset of every study (98 simulation runs and 121 forks;
         under a minute on 4 cores) and require each regenerated row to equal the recorded
         row exactly (all columns except wall-clock timings).
  full   Regenerate every study with its original settings, seeds and replication counts
         (13,392 simulation runs plus 384 fork-sweep streams) and require the regenerated
         data to equal the recorded data exactly. Measured: 4 min 23 s with 30 workers on a
         32-thread machine, 6,940 CPU-seconds in total; estimated 30-45 minutes on 4 cores.

Regenerated data go to --out (default results/rerun/<study>/); the recorded data in
results/data/ are never overwritten. To recompute the report from regenerated data, run
    python experiments/reproduce_report.py --data results/rerun --check

Usage:
    python experiments/run_experiments.py --mode quick
    python experiments/run_experiments.py --mode full [--study recovery] [--workers 8]
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RERUN, compare_rows, default_workers, read_csv, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDIES = {"commitment": ["tune.csv.gz", "test.csv.gz"], "sweep": ["test.csv.gz"], "recovery": ["test.csv.gz"]}


def load_runner(study: str):
    # Imported by package name so that worker processes can unpickle the job functions
    # under both the fork and the spawn start methods (spawn is the default on macOS/Windows).
    return importlib.import_module(f"{study}.run")


def check(study: str, runner, out: Path, full: bool) -> dict:
    res = {}
    for f in STUDIES[study]:
        saved, new = read_csv(DATA / study / f), read_csv(out / f)
        n = compare_rows(saved, new, runner.KEYS, f"{study}/{f}")
        if full:
            assert n == len(saved), f"{study}/{f}: regenerated {n} rows, recorded {len(saved)}"
        res[f] = dict(rows_compared=n, rows_recorded=len(saved), identical=True)
        print(f"[{study}] {f}: {n} regenerated rows identical to the recorded data"
              + ("" if full else f" (recorded: {len(saved)})"), flush=True)
    return res


def main(studies: list[str], mode: str, out_root: Path, workers: int) -> dict:
    summary = {}
    for study in studies:
        runner = load_runner(study)
        out = out_root / study
        t0 = time.perf_counter()
        prov = runner.main(out, workers, quick=(mode == "quick"))
        summary[study] = dict(mode=mode, wall_s=round(time.perf_counter() - t0, 1), runs=prov["runs"],
                              check=check(study, runner, out, mode == "full"))
    write_json(out_root / f"check_{mode}.json", summary)
    print(f"All checked rows reproduced exactly ({mode} mode). Summary: {out_root / f'check_{mode}.json'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("quick", "full"), default="quick")
    ap.add_argument("--study", choices=(*STUDIES, "all"), default="all")
    ap.add_argument("--out", type=Path, default=RERUN)
    ap.add_argument("--workers", type=int, default=default_workers())
    a = ap.parse_args()
    main(list(STUDIES) if a.study == "all" else [a.study], a.mode, a.out, a.workers)
