"""Batch execution: policy registry, cell configuration, parallel runs.

A *job* is (condition, rho, sigma, policy, split, rep).  All policies of one
(condition, rho, sigma, split, rep) share the identical exogenous scenario.
Each finished job yields one summary row; rows are appended to a CSV.
"""
from __future__ import annotations

import dataclasses
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd

from ..models.physics import Kinematics
from ..policies import FCFS, Adaptive, ChangePenalty, Freeze, QuantileSlack, Rolling, RollingFIFO, slack_beta
from ..simulation.engine import run
from ..simulation.scenario import SimConfig, make_scenario

FREEZE_H = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, math.inf]
PENALTY_ETA = [0.25, 1.0, 4.0]
SLACK_RATIO = [1.0, 3.0, 19.0]          # c_b : c_d  -> quantiles 0.5, 0.75, 0.95

BASE_POLICIES = (["fcfs", "rolling"] + [f"freeze_{'inf' if math.isinf(h) else f'{h:g}'}" for h in FREEZE_H]
                 + [f"penalty_{e:g}" for e in PENALTY_ETA] + [f"slack_{r:g}" for r in SLACK_RATIO]
                 + ["adaptive", "adaptive_q"])
EXTRA_POLICIES = ["adaptive_whole", "adaptive_q_whole", "adaptive_sig0.5x", "adaptive_sig2x",
                  "adaptive_q_sig0.5x", "adaptive_q_sig2x"]

# physical / information conditions (RESEARCH_PLAN F, E4/E6)
CONDITIONS = {
    "default": {},
    "zero_memory": {"kin": None},               # A1
    "b1": {"kin": Kinematics(b=1.0)},
    "b6": {"kin": Kinematics(b=6.0)},
    "a2": {"kin": Kinematics(a=2.0)},
    "a6": {"kin": Kinematics(a=6.0)},
    "no_reveal": {"reveal": False},             # A3
    "hf2": {"hf": 2.0},                         # nonuniform clearances
    "late_reveal": {"reveal_offset": 2.0},      # later revelation sensitivity
    # information/motion overlap stress regime (ruling R4): long throat, fast cars,
    # signs revealed at ages 4..7 s while the head is on its 7 s free-flow approach
    "overlap": {"kin": Kinematics(a=3.0, b=3.0, V=12.0, L=60.0), "lead": 12.25, "reveal_offset": 3.0},
    "overlap_zm": {"kin": None, "lead": 12.25, "reveal_offset": 3.0},
    "overlap_b1": {"kin": Kinematics(a=3.0, b=1.0, V=12.0, L=60.0), "lead": 12.25, "reveal_offset": 3.0},
}


def make_cfg(condition: str, rho: float, sigma: float, **over) -> SimConfig:
    kw = dict(CONDITIONS[condition])
    kw.update(over)
    return SimConfig(rho=rho, sigma=sigma, **kw)


def make_policy(name: str, cfg: SimConfig):
    if name == "fcfs":
        return FCFS()
    if name == "rolling":
        return Rolling()
    if name == "rolling_fifo":
        return RollingFIFO()
    if name.startswith("freeze_"):
        H = math.inf if name.endswith("inf") else float(name.split("_")[1])
        return Freeze(H)
    if name.startswith("penalty_"):
        return ChangePenalty(float(name.split("_")[1]))
    if name.startswith("slack_"):
        ratio = float(name.split("_")[1])
        return QuantileSlack(slack_beta(cfg.sigma, cfg.K, ratio), f"{ratio:g}")
    if name.startswith("adaptive"):
        parts = name.split("_")
        q = "q" in parts
        whole = "whole" in parts
        mult = 1.0
        for p in parts:
            if p.startswith("sig"):
                mult = float(p[3:-1])
        pol = Adaptive(cfg.kin, cfg.h, cfg.sigma * mult, dq=cfg.dq, whole=whole, queue_aware=q,
                       H=cfg.H, name=name)
        return pol
    raise KeyError(name)


def _job(args) -> list[dict]:
    condition, rho, sigma, policies, split, rep, over = args
    cfg = make_cfg(condition, rho, sigma, **over)
    sc = make_scenario(cfg, split, rep)
    rows = []
    for pname in policies:
        t0 = time.perf_counter()
        try:
            rec = run(sc, make_policy(pname, cfg))
            row = dict(rec.summary)
            row["error"] = ""
            row["n_violations"] = len(rec.violations)
            row["violation_msgs"] = "|".join(str(v[1]) for v in rec.violations[:3])
        except Exception as ex:  # failures are recorded, never hidden
            row = {"error": repr(ex)[:300], "n_violations": -1}
        row.update(condition=condition, rho=rho, sigma=sigma, policy=pname, split=split, rep=rep,
                   wall_s=time.perf_counter() - t0, **{f"over_{k}": v for k, v in over.items()})
        rows.append(row)
    return rows


def run_jobs(jobs: list[tuple], out_csv: str, workers: int | None = None, chunksize: int = 4) -> pd.DataFrame:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(_job, jobs, chunksize=chunksize)):
            rows.extend(res)
            if (k + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {k + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def grid_jobs(conditions: Iterable[str], rhos, sigmas, policies, split: str, reps, over=None) -> list[tuple]:
    over = over or {}
    return [(c, float(r), float(s), list(policies), split, int(k), over)
            for c in conditions for r in rhos for s in sigmas for k in reps]
