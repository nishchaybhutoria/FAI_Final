"""Blocked times of unrevisable bookings: exact DP, independent brute force and
validator agree; the default (no blocks) is covered by test_legacy_golden."""
import math

import numpy as np
from hypothesis import given, settings, strategies as st

from lockorder.scheduling.core import INF, Job, Plan, Snapshot, forward_times, solve_bruteforce, solve_dp, validate_plan


def snapshot(rng, n0, n1, nblocks, asym, locks):
    H = ((2.0, 3.0), (3.0, 2.0)) if asym else ((3.0, 3.0), (3.0, 3.0))
    lanes, nid = ([], []), 0
    for l, n in ((0, n0), (1, n1)):
        base = rng.uniform(0, 5)
        for k in range(n):
            base += rng.exponential(2.0)
            jb = Job(id=nid, lane=l, floor=base)
            if k == 0 and rng.random() < 0.5:
                jb.earliest = base + rng.uniform(-2, 2)
                if rng.random() < 0.4:
                    jb.latest = max(jb.floor, jb.earliest) + rng.uniform(0, 8)
            if locks and rng.random() < 0.2:
                jb.need = int(rng.integers(0, 3))
            lanes[l].append(jb)
            nid += 1
    jobs = lanes[0] + lanes[1]
    blocks = []
    for _ in range(nblocks):
        if jobs and rng.random() < 0.6:
            j = jobs[int(rng.integers(len(jobs)))]
            blocks.append((float(j.floor + rng.uniform(-4, 4)), j.lane, j.id))
        else:
            blocks.append((float(rng.uniform(0, 20)), int(rng.integers(2)), int(rng.integers(0, nid + 3))))
    last = (rng.uniform(-3, 1) if rng.random() < 0.5 else -INF, rng.uniform(-3, 1) if rng.random() < 0.5 else -INF)
    return Snapshot(t=0.0, lanes=lanes, H=H, TL=2.33, last=last, cap=10 ** 6, blocks=tuple(blocks))


@settings(max_examples=400, deadline=None)
@given(st.integers(0, 4), st.integers(0, 4), st.integers(0, 4), st.integers(0, 10 ** 6), st.booleans(), st.booleans())
def test_dp_with_blocks_equals_independent_bruteforce(n0, n1, nb, seed, asym, locks):
    snap = snapshot(np.random.default_rng(seed), n0, n1, nb, asym, locks)
    dp, bf = solve_dp(snap), solve_bruteforce(snap)
    assert (dp is None) == (bf is None)
    if bf is not None:
        assert abs(dp.cost - bf[0]) < 1e-8
        assert validate_plan(snap, dp) == []
        fw = forward_times(snap, dp.order)
        assert fw is not None and validate_plan(snap, fw) == []


def test_block_sides():
    H = ((3.0, 3.0), (3.0, 3.0))
    # cross-lane block at 10 pushes a job wanting 9 to 13
    s = Snapshot(0., ([Job(1, 0, 9.)], []), H, blocks=((10., 1, 7),))
    assert solve_dp(s).times[1] == 13.
    # cross-lane: earlier side allowed when the floor permits it
    s = Snapshot(0., ([Job(1, 0, 6.)], []), H, blocks=((10., 1, 7),))
    assert solve_dp(s).times[1] == 6.
    # same lane, job BEHIND the blocked vehicle (larger ID): must be after the block
    s = Snapshot(0., ([Job(9, 0, 2.)], []), H, blocks=((10., 0, 7),))
    assert solve_dp(s).times[9] == 13.
    # same lane, job AHEAD of the blocked vehicle: must clear it before, else infeasible
    s = Snapshot(0., ([Job(3, 0, 8.)], []), H, blocks=((10., 0, 7),))
    assert solve_dp(s) is None and solve_bruteforce(s) is None
    s = Snapshot(0., ([Job(3, 0, 7.)], []), H, blocks=((10., 0, 7),))
    assert solve_dp(s).times[3] == 7.
    # a vehicle never blocks itself
    s = Snapshot(0., ([Job(7, 0, 9.)], []), H, blocks=((10., 0, 7),))
    assert solve_dp(s).times[7] == 9.
    bad = Plan([1], {1: 11.}, 0.)
    assert validate_plan(Snapshot(0., ([Job(1, 0, 9.)], []), H, blocks=((10., 1, 7),)), bad)
    assert not math.isinf(solve_dp(Snapshot(0., ([Job(1, 0, 9.)], []), H)).times[1])
