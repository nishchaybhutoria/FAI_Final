"""Simulate one traffic stream with a chosen coordination algorithm and installation protocol.

Algorithm (--policy): how the coordinator orders vehicles at each decision
  rolling     re-optimize the whole schedule at every event (exact label DP); orders stay flexible
  fcfs        first-come first-served: each request is locked behind earlier other-lane requests
  freeze_4    lock a vehicle's position once it is within 4 s of its estimated readiness time
  freeze_inf  lock every vehicle's position as soon as it is first scheduled
  adaptive    pairwise optimal-stopping rule (lock a pair when waiting for information no longer pays)

Installation (--install): how a new schedule reaches the vehicles
  atomic      instantaneous, all at once (idealized comparator, not a deployable protocol)
  direct      release-gated single substitutions over a lossy link (safe protocol)
  detour      as direct, but may use one temporary booking to break a release cycle

Recovery for direct/detour: --retries K (0, 1, 2) extra transmissions of an unconfirmed
update, --rule blind|aware|oracle (oracle = reliable feedback comparator).

The scenario is the held-out stream --rep of study 5809 (or --study), so e.g.
    python experiments/simulate.py --policy rolling --install direct --retries 1 --loss 0.1
reproduces the recorded row (rho 0.75, d 0.1 s, 10% loss, rep 0, arm k1) of results/data/recovery/test.csv.gz.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401  (puts src/ on sys.path)

from lockorder.evaluation.runner import make_policy  # noqa: E402
from lockorder.replacement import Channel  # noqa: E402
from lockorder.replacement.closed_loop import ClosedLoop  # noqa: E402
from lockorder.replacement.retry import RetryLoop  # noqa: E402
from lockorder.simulation.engine import run  # noqa: E402
from lockorder.simulation.scenario import SimConfig, make_scenario  # noqa: E402

POLICIES = ("rolling", "fcfs", "freeze_4", "freeze_inf", "adaptive")


def simulate(policy="rolling", install="atomic", retries=0, rule="blind", loss=0.1, slot=0.1, rho=0.75,
             sigma=1.0, rep=0, study=5809, split="test"):
    """Run one stream; returns the simulator's RunRecord (summary, per-vehicle arrays, violations)."""
    cfg = SimConfig(rho=rho, sigma=sigma, audit=True)
    sc = make_scenario(cfg, split, rep, study=study)
    pol = make_policy(policy, cfg)
    if install == "atomic":
        return run(sc, pol)
    mode = {"direct": "direct", "detour": "handoff"}[install]
    link = Channel(slot, 1 - loss, 1 - loss)
    if retries == 0 and rule == "blind":
        return ClosedLoop(sc, pol, link, mode).run()          # the one-attempt contract
    return RetryLoop(sc, pol, link, mode, budget=retries, rule=rule).run()


def describe(rec) -> dict:
    s = rec.summary
    return dict(mean_delay=s["mean_delay"], p95_delay=s["p95_delay"], throughput=s["throughput"],
                vehicles=s["n_cohort"], grants=s.get("cl_grants"), pinned=s.get("cl_pinned"),
                retransmissions=s.get("rt_retx"), violations=len(rec.violations), drained=s["drained"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=POLICIES, default="rolling")
    ap.add_argument("--install", choices=("atomic", "direct", "detour"), default="atomic")
    ap.add_argument("--retries", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--rule", choices=("blind", "aware", "oracle"), default="blind")
    ap.add_argument("--loss", type=float, default=0.1, help="loss probability per message leg")
    ap.add_argument("--slot", type=float, default=0.1, help="one-way message slot d (s)")
    ap.add_argument("--rho", type=float, default=0.75, help="offered load")
    ap.add_argument("--sigma", type=float, default=1.0, help="readiness-information noise scale (s)")
    ap.add_argument("--rep", type=int, default=0, help="stream (replication) index")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rec = simulate(a.policy, a.install, a.retries, a.rule, a.loss, a.slot, a.rho, a.sigma, a.rep)
    d = describe(rec)
    print(f"policy={a.policy} install={a.install} retries={a.retries} rule={a.rule} loss={a.loss:g} "
          f"slot={a.slot:g}s rho={a.rho:g} sigma={a.sigma:g} rep={a.rep}  ({time.perf_counter() - t0:.1f} s)")
    for k, v in d.items():
        print(f"  {k:16s} {v:.3f}" if isinstance(v, float) else f"  {k:16s} {v}")
