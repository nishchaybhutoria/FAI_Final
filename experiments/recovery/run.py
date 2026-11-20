"""Experiments D and E runner: sustained traffic with safe, message-by-message installation.

Every rolling revision is installed through the acknowledged-substitution contract
on a lossy link, with 0, 1 or 2 version-safe retransmissions (configs/recovery.toml;
held-out study 5809). All arms of one (rho, rep) share the exogenous scenario.
First transmissions of every update draw from the per-vehicle stream
SeedSequence([study, split, rep, 9, vehicle]); retransmissions and queries from
stream 10, so arms with equal first attempts see equal first-attempt draws.

Usage (from the repository root):
    python experiments/recovery/run.py --out results/rerun/recovery            # 8,520 runs
    python experiments/recovery/run.py --out results/rerun/recovery --quick    # 46 runs
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, default_workers, run_metadata, write_csv_gz, write_json  # noqa: E402

import pandas as pd  # noqa: E402

from lockorder.evaluation.runner import make_policy  # noqa: E402
from lockorder.replacement import Channel  # noqa: E402
from lockorder.replacement.closed_loop import ClosedLoop  # noqa: E402
from lockorder.replacement.retry import Burst, RetryLoop  # noqa: E402
from lockorder.simulation.engine import run  # noqa: E402
from lockorder.simulation.scenario import SimConfig, make_scenario  # noqa: E402

NAME = "recovery"
C = config(NAME)
KEYS = ["block", "rho", "d", "q", "arm", "rep"]
KEEP = ("mean_delay", "p95_delay", "max_delay", "mean_delay_l0", "mean_delay_l1", "throughput", "n_cohort",
        "n_total", "drained", "violations", "approx", "fallback", "locks", "resolves", "solve_ms_mean")

# arm -> (policy, mode, budget, rule); mode "atomic" = unmodified engine (instantaneous installation),
# "orig-*" = the one-attempt closed loop (closed_loop.ClosedLoop), otherwise retry.RetryLoop.
ARMS = {
    "atomic_rolling": ("rolling", "atomic", None, None), "atomic_fcfs": ("fcfs", "atomic", None, None),
    "k0": ("rolling", "orig-direct", 0, None), "k0_log": ("rolling", "direct", 0, "blind"),
    "k1": ("rolling", "direct", 1, "blind"), "k2": ("rolling", "direct", 2, "blind"),
    "aware2": ("rolling", "direct", 2, "aware"), "oracle": ("rolling", "direct", 0, "oracle"),
    "handoff_k0": ("rolling", "orig-handoff", 0, None), "handoff_k2": ("rolling", "handoff", 2, "blind"),
    "handoff_aware2": ("rolling", "handoff", 2, "aware"),
    "freeze_k0": ("freeze_inf", "orig-direct", 0, None), "freeze_k1": ("freeze_inf", "direct", 1, "blind"),
}


def jobs(split: str = "test", quick: bool = False) -> list[dict]:
    out = []
    for blk in C["block"]:
        for rho, d, q in itertools.product(blk["rho"], blk["d"], blk["q"]):
            if quick and not (rho == .75 and q == .1 and d in (.1, 1.)):
                continue
            for rep in range(2 if quick else C[split]):
                for arm in blk["arms"]:
                    out.append(dict(block=blk["name"], rho=rho, d=d, q=q, rep=rep, arm=arm, split=split,
                                    psi=blk.get("psi")))
    return out


def job(j):
    pol, mode, K, rule = ARMS[j["arm"]]
    cfg = SimConfig(rho=j["rho"], sigma=C["sigma"], audit=True)
    sc = make_scenario(cfg, j["split"], j["rep"], study=C["study"])
    link = Channel(j["d"], 1 - j["q"], 1 - j["q"])
    t0 = time.perf_counter()
    if mode == "atomic":
        rec = run(sc, make_policy(pol, cfg))
    elif mode.startswith("orig-"):
        rec = ClosedLoop(sc, make_policy(pol, cfg), link, mode[5:]).run()
    else:
        burst = Burst(j["psi"]) if j["psi"] else None
        rec = RetryLoop(sc, make_policy(pol, cfg), link, mode, budget=K, rule=rule, burst=burst).run()
    wall = time.perf_counter() - t0
    s = rec.summary
    row = dict(j, policy=pol, mode=mode, budget=K, rule=rule, **{k: s.get(k) for k in KEEP},
               **{k: v for k, v in s.items() if (k.startswith("cl_") or k.startswith("rt_")) and k not in ("cl_mode", "rt_rule")},
               n_violations=len(rec.violations), violation_msgs="|".join(str(v[1]) for v in rec.violations[:3]),
               wall_s=wall)
    return row


def main(out: Path, workers: int, quick: bool = False) -> dict:
    js = jobs("test", quick)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(job, js, chunksize=4))
    df = pd.DataFrame(rows)
    write_csv_gz(df, out / "test.csv.gz")
    bad = df[(df.n_violations > 0) | (~df.drained.astype(bool))]
    prov = run_metadata(NAME, len(df), time.perf_counter() - t0, quick=quick, workers=workers,
                      violations_or_undrained=len(bad))
    write_json(out / "run_metadata.json", prov)
    print(f"[recovery] {len(df)} runs in {prov['wall_s']:.0f} s with {workers} workers; "
          f"violations or undrained runs: {len(bad)}", flush=True)
    assert bad.empty
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.out, a.workers, a.quick)
