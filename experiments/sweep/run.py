"""Experiments B and C runner: installing one revised schedule at isolated rolling decisions.

For every (cell, stream) the reference rolling simulator runs once; every rolling
decision triggered by a public information signal in [warmup, win_end) becomes a
*fork*. At each fork, every installation arm (keep, atomic, nominal direct,
best direct target, perfect-link detour, loss-aware choice) is evaluated on every
link with the same public state and packet draws (see forks.py). Cells are never pooled.

Usage (from the repository root):
    python experiments/sweep/run.py --out results/rerun/sweep            # 12 cells x 32 streams
    python experiments/sweep/run.py --out results/rerun/sweep --quick    # base cell, 2 streams
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import config, default_workers, run_metadata, write_csv_gz, write_json  # noqa: E402

import pandas as pd  # noqa: E402

from forks import evaluate  # noqa: E402
from lockorder.replacement import Channel  # noqa: E402
from lockorder.replacement.sample import Sampler  # noqa: E402
from lockorder.simulation.scenario import SimConfig, make_scenario  # noqa: E402

NAME = "sweep"
C = config(NAME)
KEYS = ["cell_rho", "cell_sigma", "cell_hf", "rep", "epoch", "d", "u", "arm"]


def cells(quick: bool = False) -> list[dict]:
    if quick:
        return [dict(C["base"])]
    return [dict(rho=r, sigma=s, hf=h) for r, s, h in itertools.product(C["rho"], C["sigma"], C["hf"])]


def stream(args):
    cell, split, rep = args
    cfg = SimConfig(rho=cell["rho"], sigma=cell["sigma"], hf=cell["hf"], warmup=C["warmup"], win_end=C["win_end"],
                    t_arr_end=C["t_arr_end"], audit=True, tie_eta=0.)
    t0 = time.perf_counter()
    sam = Sampler(make_scenario(cfg, split, rep, study=C["study"]))
    ref = sam.run()
    assert ref.summary["drained"] and not ref.violations and not ref.summary["approx"], ref.summary
    links = [Channel(d, p, p) for d, p in C["links"]]
    rows = []
    for ob in sam.observations:
        rows += evaluate(sam.sc.key, ob, links, dict(cell_rho=cell["rho"], cell_sigma=cell["sigma"], cell_hf=cell["hf"],
                                                     H_same=ob.fork.snap.H[0][0], H_cross=ob.fork.snap.H[0][1],
                                                     TL=ob.fork.snap.TL))
    assert rows, "empty stream: no silent exclusion"
    meta = dict(**cell, split=split, rep=rep, forks=len(sam.observations), wall_s=time.perf_counter() - t0,
                labels_max=ref.summary["labels_max"], n_total=ref.summary["n_total"])
    return rows, meta


def main(out: Path, workers: int, quick: bool = False) -> dict:
    jobs = [(cell, "test", rep) for cell in cells(quick) for rep in range(2 if quick else C["test"])]
    t0 = time.perf_counter()
    rows, metas = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, m in ex.map(stream, jobs):
            rows += r
            metas.append(m)
    df = pd.DataFrame(rows)
    write_csv_gz(df, out / "test.csv.gz")
    pd.DataFrame(metas).drop(columns="wall_s").to_csv(out / "test_streams.csv", index=False)
    prov = run_metadata(NAME, len(jobs), time.perf_counter() - t0, quick=quick, workers=workers,
                      forks=int(df[["cell_rho", "cell_sigma", "cell_hf", "rep", "epoch"]].drop_duplicates().shape[0]))
    write_json(out / "run_metadata.json", prov)
    print(f"[sweep] {len(jobs)} streams, {prov['forks']} forks, {len(df)} rows in {prov['wall_s']:.0f} s "
          f"with {workers} workers", flush=True)
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.out, a.workers, a.quick)
