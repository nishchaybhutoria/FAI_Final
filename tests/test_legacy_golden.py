"""Golden regression for code shared with the archived studies.

The payload below was generated from the code *before* the revision touched
``scheduling/core.py`` or ``simulation/engine.py``. Any later change to those
modules must reproduce it exactly (floats are compared bit-for-bit after a JSON
round trip). Regenerate only deliberately:

    PYTHONPATH=src python tests/test_legacy_golden.py --write
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from lockorder.evaluation.runner import make_policy
from lockorder.models.physics import Kinematics, launch_time
from lockorder.scheduling.core import Job, Plan, Snapshot, forward_times, solve_dp, validate_plan
from lockorder.simulation.engine import run
from lockorder.simulation.scenario import SimConfig, make_scenario

GOLDEN = Path(__file__).parent / "data" / "legacy_golden.json"
WALL_CLOCK = {"solve_ms_mean", "solve_ms_p95"}


def _f(x):
    """JSON-safe float (inf/nan as strings) so the comparison is exact."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def random_snapshots(n=200, seed=20261118):
    rng = np.random.default_rng(seed)
    kin = Kinematics()
    out = []
    for k in range(n):
        t = float(rng.uniform(0, 5))
        hf = 3.0 if rng.random() < 0.6 else 2.0
        H = ((hf, 3.0), (3.0, hf))
        TL = launch_time(kin) if rng.random() < 0.5 else 0.0
        lanes = ([], [])
        vid = 0
        for l in (0, 1):
            for idx in range(int(rng.integers(0, 5))):
                floor = float(t + rng.uniform(0, 12))
                job = Job(id=vid, lane=l, floor=floor)
                if idx == 0 and rng.random() < 0.5:
                    job.earliest = float(t + rng.uniform(0, 3))
                    job.latest = float(job.earliest + rng.uniform(2, 20)) if rng.random() < 0.5 else math.inf
                if rng.random() < 0.15:
                    job.need = int(rng.integers(0, 2))
                lanes[l].append(job)
                vid += 1
        last = tuple(float(t - rng.uniform(0, 4)) if rng.random() < 0.7 else -math.inf for _ in (0, 1))
        eta = 1e-6 if rng.random() < 0.5 else 0.0
        out.append(Snapshot(t=t, lanes=lanes, H=H, TL=TL, last=last, eta=eta))
    return out


def _plan_record(p):
    if p is None:
        return None
    return dict(order=list(p.order), times={str(i): _f(s) for i, s in sorted(p.times.items())},
                cost=_f(p.cost), approx=bool(p.approx))


def snapshot_payload():
    rows = []
    for snap in random_snapshots():
        ids = [j.id for q in snap.lanes for j in q]
        dp = solve_dp(snap)
        request = sorted(ids)
        fw = forward_times(snap, request)
        checks = []
        if dp is not None and dp.order:
            checks.append(len(validate_plan(snap, dp)))
            bumped = Plan(dp.order, {i: s - 0.5 for i, s in dp.times.items()}, 0.0)
            checks.append(len(validate_plan(snap, bumped)))
        rows.append(dict(dp=_plan_record(dp), forward=_plan_record(fw), checks=checks))
    return rows


def engine_payload():
    short = dict(t_arr_end=120.0, warmup=20.0, win_end=100.0, audit=True)
    cells = {
        "kin": SimConfig(rho=0.75, sigma=1.0, **short),
        "point": SimConfig(rho=0.5, sigma=1.0, kin=None, **short),
        "hf2": SimConfig(rho=0.75, sigma=0.5, hf=2.0, **short),
    }
    out = {}
    for name, cfg in cells.items():
        sc = make_scenario(cfg, "pilot", 0, study=4242)
        for pol in ("rolling", "freeze_4", "freeze_inf", "adaptive"):
            rec = run(sc, make_policy(pol, cfg))
            summ = {k: _f(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v
                    for k, v in sorted(rec.summary.items()) if k not in WALL_CLOCK}
            out[f"{name}/{pol}"] = dict(summary=summ, S=[_f(x) for x in rec.vehicles["S"]],
                                        violations=len(rec.violations))
    return out


def payload():
    return dict(snapshots=snapshot_payload(), engine=engine_payload())


def test_shared_scheduling_and_engine_outputs_are_unchanged():
    want = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(payload()))
    assert got["snapshots"] == want["snapshots"]
    assert got["engine"].keys() == want["engine"].keys()
    for key in want["engine"]:
        assert got["engine"][key] == want["engine"][key], key


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("pass --write to regenerate the golden payload deliberately")
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(payload(), indent=1, sort_keys=True) + "\n")
    print("wrote", GOLDEN)
