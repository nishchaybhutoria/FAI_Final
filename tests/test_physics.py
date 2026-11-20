import math

import numpy as np
import pytest
from scipy.optimize import brentq

from lockorder.models.physics import (
    INF, Kinematics, InfeasibleArrival, d_minus, d_plus, kappa_surrogate, launch_time,
    make_profile, promotion_loss, t_minus, t_plus,
)

K = Kinematics()


def _sim_distance(v0, accel_fn, T, k, n=200000):
    """Independent explicit integration of a speed rule (midpoint)."""
    dt = T / n
    v, d = v0, 0.0
    for _ in range(n):
        u = accel_fn(v)
        v_new = min(k.V, max(0.0, v + u * dt))
        d += 0.5 * (v + v_new) * dt
        v = v_new
    return d


def test_t_minus_matches_root_solve():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.uniform(0.01, 30)
        v = rng.uniform(0, K.V)
        T = t_minus(p, v, K)
        # independent: root of d_plus(T) - p computed by brentq on a closed form
        root = brentq(lambda x: d_plus(x, v, K) - p, 0.0, 100.0, xtol=1e-14)
        assert abs(T - root) < 1e-9


def test_t_minus_matches_explicit_integration():
    p, v = 8.0, 0.0
    T = t_minus(p, v, K)
    d = _sim_distance(v, lambda _: K.a, T, K, n=100000)
    assert abs(d - p) < 1e-3
    assert abs(T - (2.0 + 2.0 / 6.0)) < 1e-12  # 6 m accel in 2 s, 2 m at 6 m/s


def test_t_plus_three_regimes():
    v = 6.0
    pb = v * v / (2 * K.b)          # 6 m
    assert t_plus(pb + 0.1, v, K) == INF
    assert abs(t_plus(pb, v, K) - v / K.b) < 1e-9
    p = 3.0
    T = t_plus(p, v, K)
    assert math.isfinite(T)
    assert abs(d_minus(T, v, K) - p) < 1e-9
    assert t_plus(5.0, 0.0, K) == INF
    assert t_plus(0.0, 3.0, K) == 0.0


def test_profile_hits_exact_target_and_respects_bounds():
    rng = np.random.default_rng(7)
    n_ok = 0
    for _ in range(500):
        p = rng.uniform(0.05, 8.0)
        v = rng.uniform(0, K.V)
        tm, tp = t_minus(p, v, K), t_plus(p, v, K)
        hi = tp if math.isfinite(tp) else tm + 20.0
        T = rng.uniform(tm, hi)
        prof = make_profile(10.0, p, v, 10.0 + T, K)
        assert 0.0 <= prof.alpha <= 1.0
        assert prof.accel_bounds_ok()
        # distance residual at arrival
        assert abs(prof.distance(10.0 + T) - p) < 1e-9
        # strictly before arrival the line is not touched; speeds within bounds
        for x in np.linspace(0, T, 41)[:-1]:
            pp, vv = prof.state(10.0 + x)
            assert pp > -1e-12
            assert -1e-12 <= vv <= K.V + 1e-12
            if x < T - 1e-6:
                assert pp > 0.0 or prof.alpha == 0.0
        # breakpoints: speed continuity check across each breakpoint
        for x in prof.breakpoints():
            _, v1 = prof.state(10.0 + max(0.0, x - 1e-7))
            _, v2 = prof.state(10.0 + min(T, x + 1e-7))
            assert abs(v1 - v2) < 1e-5
        n_ok += 1
    assert n_ok == 500


def test_profile_rejects_infeasible_targets():
    p, v = 3.0, 6.0   # cannot stop: latest touch finite
    tm, tp = t_minus(p, v, K), t_plus(p, v, K)
    with pytest.raises(InfeasibleArrival):
        make_profile(0.0, p, v, tm - 0.01, K)
    with pytest.raises(InfeasibleArrival):
        make_profile(0.0, p, v, tp + 0.01, K)


def test_earliest_time_never_decreases_along_profiles():
    rng = np.random.default_rng(3)
    for _ in range(100):
        p, v = rng.uniform(1, 8), rng.uniform(0, 6)
        tm, tp = t_minus(p, v, K), t_plus(p, v, K)
        prof = make_profile(0.0, p, v, tm + rng.uniform(0, min(6.0, tp - tm)), K)
        e_prev = tm
        for x in np.linspace(0, prof.T * 0.999, 30)[1:]:
            pp, vv = prof.state(x)
            e = x + t_minus(pp, vv, K)
            assert e >= e_prev - 1e-9
            e_prev = e


def test_promotion_loss_nonnegative_nondecreasing():
    h = 3.0
    ws = np.linspace(0, launch_time(K), 50)
    g = [promotion_loss(w, h, K) for w in ws]
    assert g[0] == 0.0
    assert all(x >= -1e-12 for x in g)
    assert all(g[i + 1] >= g[i] - 1e-9 for i in range(len(g) - 1))
    assert g[-1] > 0.0
    assert g[-1] <= launch_time(K) + 1e-9   # never worse than standing still for w
    assert kappa_surrogate(10.0, h, K) == 0.0
    assert kappa_surrogate(0.0, h, K) == pytest.approx(2 * g[-1])


def test_launch_time_below_headway_for_planned_a():
    for a in (2.0, 3.0, 6.0):
        assert launch_time(Kinematics(a=a)) < 3.0
