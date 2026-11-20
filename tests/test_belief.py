import itertools
import math

import numpy as np
from scipy import integrate, stats

from lockorder.models.belief import (
    SignProcess, discrete_hinge, gap_kernel, gaussian_hinge, rademacher_sum_pmf,
)


def test_posterior_exact_by_enumeration():
    sp = SignProcess(sigma=0.7)
    a = 5.0
    allsigns = list(itertools.product([-1, 1], repeat=sp.K))
    for k in range(sp.K + 1):
        for prefix in itertools.product([-1, 1], repeat=k):
            comps = [s for s in allsigns if s[:k] == prefix]
            marks = np.array([sp.mark(a, np.array(s)) for s in comps])
            r_hat, var, U, Ls = sp.posterior(a, list(prefix))
            assert abs(marks.mean() - r_hat) < 1e-12          # conditional mean
            assert abs(marks.var() - var) < 1e-12             # conditional variance
            assert abs(marks.max() - U) < 1e-12               # upper support
            assert abs(marks.min() - Ls) < 1e-12


def test_tower_property_and_total_variance():
    sp = SignProcess(sigma=1.0)
    allsigns = np.array(list(itertools.product([-1, 1], repeat=sp.K)))
    marks = 8.25 + allsigns.sum(1)
    for k in range(sp.K + 1):
        rh = np.array([sp.posterior(0.0, s[:k])[0] for s in allsigns])
        resid = np.array([sp.posterior(0.0, s[:k])[1] for s in allsigns])
        assert abs(rh.mean() - marks.mean()) < 1e-12
        # Var(r) = Var(E[r|F]) + E[Var(r|F)]
        assert abs(marks.var() - (rh.var() + resid.mean())) < 1e-12
        # estimate spread across paths grows, residual shrinks
        assert abs(rh.var() - k * sp.sigma ** 2) < 1e-12


def test_upper_support_nonincreasing():
    sp = SignProcess(sigma=0.5)
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = rng.choice([-1, 1], size=sp.K)
        Us = [sp.posterior(0.0, s[:k])[2] for k in range(sp.K + 1)]
        assert all(Us[i + 1] <= Us[i] + 1e-12 for i in range(sp.K))
        assert abs(Us[-1] - sp.mark(0.0, s)) < 1e-12


def test_gap_increment_doubled_variance():
    sigma = 0.5
    vals, probs = gap_kernel(3, 3, sigma)
    var = float(np.sum(probs * vals ** 2))
    assert abs(var - 2 * 3 * sigma ** 2) < 1e-12       # 2 sigma^2 per unit clock
    # Monte Carlo of independent increments
    rng = np.random.default_rng(42)
    zi = rng.choice([-1, 1], size=(200000, 3)).sum(1)
    zj = rng.choice([-1, 1], size=(200000, 3)).sum(1)
    d = sigma * (zj - zi)
    se = d.var() * math.sqrt(2 / len(d))
    assert abs(d.var() - 6 * sigma ** 2) < 4 * se


def test_rademacher_pmf():
    for n in range(7):
        vals, probs = rademacher_sum_pmf(n)
        assert abs(sum(probs) - 1) < 1e-15
        assert abs(sum(v * v * p for v, p in zip(vals, probs)) - n) < 1e-12


def test_gaussian_hinge_against_quadrature():
    h = 2.0
    for m, v in [(0.0, 1.0), (1.5, 0.3), (-3.0, 4.0), (5.0, 0.01), (2.0, 0.0)]:
        if v == 0:
            assert gaussian_hinge(h, m, v) == max(0.0, h - m)
            continue
        sd = math.sqrt(v)
        q, _ = integrate.quad(lambda x: max(0.0, h - x) * stats.norm.pdf(x, m, sd), m - 12 * sd, m + 12 * sd,
                              points=[h], limit=200)
        assert abs(gaussian_hinge(h, m, v) - q) < 1e-9


def test_discrete_hinge_matches_enumeration():
    h, step = 3.0, 0.5
    for n in range(5):
        for m in (-2.0, 0.0, 1.3, 4.0):
            allv = [sum(s) for s in itertools.product([-1, 1], repeat=n)] if n else [0]
            direct = np.mean([max(0.0, h - m - step * x) for x in allv])
            assert abs(discrete_hinge(h, m, n, step) - direct) < 1e-12


def test_brownian_bridge_semantics_gaussian():
    """Forward-generated Brownian increments: conditional law of truth given prefix."""
    rng = np.random.default_rng(5)
    sigma, Q, q_obs = 0.5, 4.0, 1.5      # total info time, residual at observation
    n = 400000
    pre = rng.normal(0, sigma * math.sqrt(Q - q_obs), size=n)
    post = rng.normal(0, sigma * math.sqrt(q_obs), size=n)
    r = pre + post
    # regress truth on estimate: slope 1, residual variance sigma^2 q_obs
    slope = np.cov(r, pre)[0, 1] / pre.var()
    resid = r - pre
    assert abs(slope - 1) < 0.01
    assert abs(resid.var() - sigma ** 2 * q_obs) < 4 * resid.var() * math.sqrt(2 / n)
