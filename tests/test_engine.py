import dataclasses
import math

import numpy as np
import pytest

from lockorder.models.physics import Kinematics, launch_time
from lockorder.policies import FCFS, Adaptive, ChangePenalty, Freeze, QuantileSlack, Rolling
from lockorder.simulation.engine import Engine, run
from lockorder.simulation.scenario import Scenario, SimConfig, make_scenario


def _policies(cfg):
    return [FCFS(), Rolling(), Freeze(0.0), Freeze(4.0), Freeze(math.inf), ChangePenalty(1.0),
            QuantileSlack(2 * cfg.sigma, "x"), Adaptive(cfg.kin, cfg.h, cfg.sigma),
            Adaptive(cfg.kin, cfg.h, cfg.sigma, queue_aware=True),
            Adaptive(cfg.kin, cfg.h, cfg.sigma, whole=True)]


@pytest.mark.parametrize("rho,sigma,hf,kin,reveal", [
    (0.5, 1.0, 3.0, Kinematics(), True),
    (0.95, 0.5, 3.0, Kinematics(b=1.0), True),
    (0.9, 1.0, 2.0, Kinematics(a=2.0), True),
    (0.8, 1.0, 3.0, None, True),
    (0.7, 1.0, 3.0, Kinematics(), False),
])
def test_audited_runs_have_no_violations(rho, sigma, hf, kin, reveal):
    cfg = SimConfig(rho=rho, sigma=sigma, hf=hf, kin=kin, reveal=reveal, audit=True, t_arr_end=500,
                    warmup=100, win_end=400)
    sc = make_scenario(cfg, "pilot", 3)
    for pol in _policies(cfg):
        rec = run(sc, pol)
        assert rec.violations == [], (pol.name, rec.violations[:3])
        s = rec.summary
        assert s["drained"] and s["n_done"] == s["n_cohort"]
        assert s["lb_mean_delay"] <= s["mean_delay"] + 1e-9
        # executed entries respect FCFS order exactly for FCFS-type policies
        if pol.name.startswith(("fcfs", "slack")):
            S = rec.vehicles["S"]
            assert np.all(np.diff(S[np.argsort(sc.a)]) > 0)


def test_same_seed_identical_and_different_seed_distinct():
    cfg = SimConfig(rho=0.8, sigma=1.0, t_arr_end=400, warmup=50, win_end=300)
    s1 = make_scenario(cfg, "test", 1)
    s2 = make_scenario(cfg, "test", 1)
    s3 = make_scenario(cfg, "test", 2)
    assert np.array_equal(s1.a, s2.a) and np.array_equal(s1.signs, s2.signs)
    assert not np.array_equal(s1.a[:20], s3.a[:20])
    r1 = run(s1, Freeze(4.0))
    r2 = run(s2, Freeze(4.0))
    assert np.array_equal(r1.vehicles["S"], r2.vehicles["S"])
    assert r1.lock_log == r2.lock_log
    # splits are disjoint namespaces
    st = make_scenario(cfg, "tune", 1)
    assert not np.array_equal(st.a[:20], s1.a[:20])


def test_crn_across_policies_and_noise():
    cfg = SimConfig(rho=0.5, sigma=0.5, t_arr_end=300)
    a = make_scenario(cfg, "test", 4)
    b = make_scenario(dataclasses.replace(cfg, sigma=1.0), "test", 4)
    assert np.array_equal(a.a, b.a) and np.array_equal(a.signs, b.signs)


def test_no_future_information_leakage():
    """Two scenarios identical up to time T* but with different hidden futures
    must produce identical slots/locks/entries before T*."""
    cfg = SimConfig(rho=0.8, sigma=1.0, t_arr_end=300, warmup=0, win_end=300)
    sc = make_scenario(cfg, "pilot", 7)
    Tstar = 150.0
    # alter all signs whose reveal time is after T* and all births after T*
    signs2 = sc.signs.copy()
    for i in range(sc.n):
        for k in range(cfg.K):
            if sc.a[i] + (k + 1) * cfg.dq > Tstar:
                signs2[i, k] = -signs2[i, k]
    keep = sc.a <= Tstar
    a2 = np.concatenate([sc.a[keep], sc.a[~keep] + 0.37])
    lane2 = np.concatenate([sc.lane[keep], 1 - sc.lane[~keep]])
    sg2 = np.concatenate([signs2[keep], signs2[~keep]])
    o = np.lexsort((lane2, a2))
    a2, lane2, sg2 = a2[o], lane2[o], sg2[o]
    r2 = a2 + cfg.lead + cfg.sigma * sg2.sum(1)
    sc2 = Scenario(a=a2, lane=lane2, signs=sg2, r=r2, key=("alt",), cfg=cfg)
    for mk in (lambda: Rolling(), lambda: Freeze(4.0), lambda: Adaptive(cfg.kin, cfg.h, cfg.sigma)):
        e1 = Engine(sc, mk(), collect_trace=True); e1.run()
        e2 = Engine(sc2, mk(), collect_trace=True); e2.run()
        tr1 = [x for x in e1.trace if x[0] < Tstar - 1e-9]
        tr2 = [x for x in e2.trace if x[0] < Tstar - 1e-9]
        assert tr1 == tr2
        l1 = [x for x in e1.lock_log if x[0] < Tstar - 1e-9]
        l2 = [x for x in e2.lock_log if x[0] < Tstar - 1e-9]
        assert l1 == l2


def _manual_scenario(cfg, a, lane, signs):
    a = np.asarray(a, float); lane = np.asarray(lane, np.int8); signs = np.asarray(signs, np.int8)
    r = a + cfg.lead + cfg.sigma * signs.sum(1)
    return Scenario(a=a, lane=lane, signs=signs, r=r, key=("manual",), cfg=cfg)


def test_single_vehicle_perfect_information_on_time():
    cfg = SimConfig(sigma=0.0, warmup=0, win_end=100, t_arr_end=100)
    sc = _manual_scenario(cfg, [10.0], [0], [[1, 1, -1, 1]])
    rec = run(sc, Rolling())
    assert abs(rec.vehicles["S"][0] - (10.0 + cfg.lead)) < 1e-9
    assert rec.summary["mean_delay"] < 1e-9


def test_saturated_ready_stream_respects_capacity():
    cfg = SimConfig(sigma=0.0, warmup=0, win_end=500, t_arr_end=500)
    n = 40
    a = np.repeat(np.arange(n // 2) * 0.01, 2)
    lane = np.tile([0, 1], n // 2)
    sc = _manual_scenario(cfg, a, lane, np.ones((n, cfg.K)))
    rec = run(sc, Rolling())
    S = np.sort(rec.vehicles["S"])
    assert np.all(np.diff(S) >= cfg.h - 1e-9)
    assert abs(np.mean(np.diff(S)) - cfg.h) < 1e-6     # capacity 1/h attained, launch never binds


def test_no_reveal_mode_keeps_prior():
    cfg = SimConfig(sigma=1.0, reveal=False, warmup=0, win_end=100, t_arr_end=100)
    sc = _manual_scenario(cfg, [5.0], [1], [[-1, -1, -1, -1]])
    rec = run(sc, Rolling())
    # slot stays at the prior upper support a + lead + 4 sigma
    assert abs(rec.vehicles["S"][0] - (5.0 + cfg.lead + 4.0)) < 1e-9


def test_zero_memory_has_no_launch_constraint():
    cfg = SimConfig(sigma=0.0, kin=None, warmup=0, win_end=100, t_arr_end=100, hf=2.0)
    sc = _manual_scenario(cfg, [0.0, 0.0], [0, 0], np.zeros((2, 4)))
    rec = run(sc, Rolling())
    S = rec.vehicles["S"]
    assert abs(S[1] - S[0] - 2.0) < 1e-9
