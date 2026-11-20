"""Closed-loop installation: safety and commitment invariants under every arm."""
import numpy as np
import pytest

from lockorder.evaluation.runner import make_policy
from lockorder.replacement import Channel
from lockorder.replacement.closed_loop import ClosedLoop
from lockorder.simulation.engine import run
from lockorder.simulation.scenario import SimConfig, make_scenario

CFG = SimConfig(rho=0.75, sigma=1.0, t_arr_end=240.0, warmup=40.0, win_end=200.0, audit=True)
ARMS = [("rolling", "direct"), ("rolling", "handoff"), ("freeze_4", "direct"), ("freeze_inf", "direct"),
        ("adaptive", "direct")]
LINKS = [Channel(0.0, 1.0, 1.0), Channel(0.1, 0.9, 0.9), Channel(0.1, 0.7, 0.7)]


@pytest.mark.parametrize("rep", [0, 1])
@pytest.mark.parametrize("link", LINKS, ids=lambda c: f"d{c.d}-p{c.u}")
def test_every_arm_is_safe_drains_and_keeps_its_commitments(rep, link):
    sc = make_scenario(CFG, "pilot", rep, study=1111)
    for pol, mode in ARMS:
        r = ClosedLoop(sc, make_policy(pol, CFG), link, mode).run()
        # audit checks the ledger after every protocol event and the plan at every
        # decision; the engine checks marks, clearances, FIFO, launch and locks at entry
        assert r.violations == [], (pol, mode, r.violations[:3])
        assert r.summary["drained"]
        assert r.summary["cl_lock_relaxed"] == 0
        assert np.all(r.vehicles["S"] >= r.vehicles["r"] - 1e-7)


def test_perfect_link_never_leaves_a_booking_unresolved():
    sc = make_scenario(CFG, "pilot", 0, study=1111)
    for pol, mode in ARMS:
        s = ClosedLoop(sc, make_policy(pol, CFG), Channel(0.0, 1.0, 1.0), mode).run().summary
        assert s["cl_pinned"] == 0 and s["cl_stranded"] == 0
        assert s["cl_grants"] == s["cl_delivered"] == s["cl_switched"] == s["cl_acks"]


def test_lossy_link_pins_vehicles_and_runs_are_deterministic():
    sc = make_scenario(CFG, "pilot", 1, study=1111)
    a = ClosedLoop(sc, make_policy("rolling", CFG), Channel(0.1, 0.8, 0.8), "handoff").run()
    b = ClosedLoop(sc, make_policy("rolling", CFG), Channel(0.1, 0.8, 0.8), "handoff").run()
    assert a.summary["cl_pinned"] > 0
    assert np.array_equal(a.vehicles["S"], b.vehicles["S"])


def test_direct_installation_cannot_beat_atomic_rolling_by_much():
    """Not a theorem (rolling is myopic), but a large gain would signal leakage."""
    for rep in range(3):
        sc = make_scenario(CFG, "pilot", rep, study=1111)
        atomic = run(sc, make_policy("rolling", CFG)).summary["mean_delay"]
        for mode in ("direct", "handoff"):
            s = ClosedLoop(sc, make_policy("rolling", CFG), Channel(0.0, 1.0, 1.0), mode).run().summary
            assert s["mean_delay"] >= atomic - 0.25
