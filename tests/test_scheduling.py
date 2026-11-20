import math

import numpy as np
import pytest

from lockorder.scheduling.core import (
    INF, Job, Snapshot, forward_times, solve_bruteforce, solve_dp, validate_plan,
)


def _random_snapshot(rng, n0, n1, asym=False, windows=True, locks=True, eta=0.0):
    H = ((2.0, 3.0), (3.0, 2.0)) if asym else ((3.0, 3.0), (3.0, 3.0))
    lanes = ([], [])
    nid = 0
    for l, n in ((0, n0), (1, n1)):
        base = rng.uniform(0, 5)
        for k in range(n):
            base += rng.exponential(2.0)
            jb = Job(id=nid, lane=l, floor=base)
            if k == 0 and windows and rng.random() < 0.6:
                jb.earliest = base + rng.uniform(-2, 2)
                if rng.random() < 0.5:
                    jb.latest = max(jb.floor, jb.earliest) + rng.uniform(0, 6)
            lanes[l].append(jb)
            nid += 1
    if locks:
        for l in (0, 1):
            for jb in lanes[l]:
                if rng.random() < 0.25:
                    jb.need = int(rng.integers(0, len(lanes[1 - l]) + 1))
    for l in (0, 1):
        for jb in lanes[l]:
            jb.prev_other = int(rng.integers(0, len(lanes[1 - l]) + 1))
    last = (rng.uniform(-3, 1) if rng.random() < 0.5 else -INF, rng.uniform(-3, 1) if rng.random() < 0.5 else -INF)
    return Snapshot(t=0.0, lanes=lanes, H=H, TL=2.33, last=last, eta=eta, cap=10 ** 6)


@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_dp_matches_bruteforce(asym, eta):
    rng = np.random.default_rng(11 + asym + int(eta))
    n_feas = n_inf = 0
    for trial in range(250):
        n0, n1 = int(rng.integers(0, 6)), int(rng.integers(0, 6))
        snap = _random_snapshot(rng, n0, n1, asym=asym, eta=eta)
        dp = solve_dp(snap)
        bf = solve_bruteforce(snap)
        if bf is None:
            assert dp is None
            n_inf += 1
            continue
        n_feas += 1
        assert dp is not None
        assert abs(dp.cost - bf[0]) < 1e-8, (trial, dp.cost, bf[0])
        assert validate_plan(snap, dp) == []
        fw = forward_times(snap, dp.order)
        assert fw is not None and abs(fw.cost - dp.cost) < 1e-9
        for i in dp.order:
            assert abs(fw.times[i] - dp.times[i]) < 1e-9
    assert n_feas > 100 and n_inf > 0


def test_capped_labels_never_beat_exact_and_stay_feasible():
    rng = np.random.default_rng(5)
    for _ in range(60):
        snap = _random_snapshot(rng, 6, 6, asym=True)
        exact = solve_dp(snap)
        snap.cap = 2
        cap = solve_dp(snap)
        if exact is None:
            continue
        if cap is not None:
            assert cap.cost >= exact.cost - 1e-9
            assert validate_plan(snap, cap) == []


def test_b3_single_label_counterexample():
    """RESEARCH_PLAN B3: keeping one label per (i, j, last) loses the optimum."""
    H = ((1.0, 2.0), (2.0, 1.0))
    A = [Job(id=k, lane=0, floor=0.1 * k) for k in range(4)]
    B = [Job(id=10, lane=1, floor=0.5)]
    A[1].need = 1                         # B1 before A2
    snap = Snapshot(t=0, lanes=(A, B), H=H, TL=0.0)
    dp = solve_dp(snap)
    bf = solve_bruteforce(snap)
    assert abs(bf[0] - 16.5) < 1e-12
    assert abs(dp.cost - 16.5) < 1e-12
    assert dp.order[:3] == [10, 0, 1]
    # the dominated-by-sum prefix A1 B1 A2 completes to 17
    fw = forward_times(snap, [0, 10, 1, 2, 3])
    assert abs(fw.cost - 17.0) < 1e-12
    # prefix comparison at state (2,1,last=A): sum 6 vs 6.5, last start 4 vs 3.5
    p1 = forward_times(Snapshot(t=0, lanes=(A[:2], B), H=H), [0, 10, 1])
    p2 = forward_times(Snapshot(t=0, lanes=(A[:2], B), H=H), [10, 0, 1])
    assert (p1.cost, p1.times[1]) == (6.0, 4.0)
    assert (p2.cost, p2.times[1]) == (6.5, 3.5)


def test_empty_and_single():
    snap = Snapshot(t=0, lanes=([], []), H=((3, 3), (3, 3)))
    assert solve_dp(snap).order == []
    snap = Snapshot(t=0, lanes=([Job(id=1, lane=0, floor=5.0)], []), H=((3, 3), (3, 3)), last=(-INF, 4.0))
    p = solve_dp(snap)
    assert p.order == [1] and abs(p.times[1] - 7.0) < 1e-12


def test_cyclic_locks_infeasible():
    A = [Job(id=0, lane=0, floor=0), Job(id=1, lane=0, floor=1)]
    B = [Job(id=2, lane=1, floor=0), Job(id=3, lane=1, floor=1)]
    A[0].need = 2   # B1, B2 before A1
    B[0].need = 2   # A1, A2 before B1 -> cycle
    snap = Snapshot(t=0, lanes=(A, B), H=((3, 3), (3, 3)))
    assert solve_dp(snap) is None
    assert solve_bruteforce(snap) is None


def test_deadline_infeasible_dag():
    A = [Job(id=0, lane=0, floor=0, latest=1.0)]
    B = [Job(id=1, lane=1, floor=0)]
    A[0].need = 1   # B first, then A by 1.0 is impossible with h=3
    snap = Snapshot(t=0, lanes=(A, B), H=((3, 3), (3, 3)))
    assert solve_dp(snap) is None
    A[0].need = 0
    p = solve_dp(snap)
    assert p.order == [0, 1]


def test_no_spurious_headway_between_nonconflicting_history():
    """Same-lane following headway applies separately from cross clearance."""
    H = ((2.0, 3.0), (3.0, 2.0))
    snap = Snapshot(t=0, lanes=([Job(id=1, lane=0, floor=0)], []), H=H, last=(-1.0, -INF))
    assert abs(solve_dp(snap).times[1] - 1.0) < 1e-12
    snap = Snapshot(t=0, lanes=([Job(id=1, lane=0, floor=0)], []), H=H, last=(-INF, -1.0))
    assert abs(solve_dp(snap).times[1] - 2.0) < 1e-12


from hypothesis import given, settings, strategies as st


@settings(max_examples=150, deadline=None)
@given(st.integers(0, 4), st.integers(0, 4), st.integers(0, 10 ** 6), st.booleans(), st.floats(0, 2))
def test_property_dp_equals_bruteforce(n0, n1, seed, asym, eta):
    rng = np.random.default_rng(seed)
    snap = _random_snapshot(rng, n0, n1, asym=asym, eta=eta)
    dp, bf = solve_dp(snap), solve_bruteforce(snap)
    assert (dp is None) == (bf is None)
    if bf is not None:
        assert abs(dp.cost - bf[0]) < 1e-8
        assert validate_plan(snap, dp) == []


def test_request_order_tiebreak_exact_and_prefers_fifo():
    rng = np.random.default_rng(9)
    for _ in range(80):
        snap = _random_snapshot(rng, int(rng.integers(0, 5)), int(rng.integers(0, 5)))
        for l in (0, 1):
            for jb in snap.lanes[l]:
                jb.req_other = sum(1 for o in snap.lanes[1 - l] if o.id < jb.id)
        snap.eta_req = 0.37
        dp, bf = solve_dp(snap), solve_bruteforce(snap)
        assert (dp is None) == (bf is None)
        if bf is not None:
            assert abs(dp.cost - bf[0]) < 1e-8
    # two past-due jobs: equal sum either way, FIFO tie-break picks the earlier request
    A = [Job(id=5, lane=0, floor=0.0)]
    B = [Job(id=2, lane=1, floor=0.0, req_other=0)]
    A[0].req_other = 1
    snap = Snapshot(t=0, lanes=(A, B), H=((3, 3), (3, 3)), eta_req=1e-8)
    assert solve_dp(snap).order == [2, 5]
