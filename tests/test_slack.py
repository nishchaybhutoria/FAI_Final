"""Quantile-slack baseline (theory/slack.py)."""
import numpy as np
from scipy import stats

from lockorder.theory.slack import generalized_quantile_discrete, minimiser_interval_discrete, slack_cost


def test_generalized_quantile_and_direct_minimisation():
    rng = np.random.default_rng(4)
    for _ in range(30):
        vals = np.sort(rng.normal(size=6).round(2))
        probs = rng.dirichlet(np.ones(6))
        cb, cd = rng.uniform(0.2, 5, 2)
        q = cb / (cb + cd)
        beta = generalized_quantile_discrete(vals, probs, q)
        grid = np.linspace(vals.min() - 1, vals.max() + 1, 4001)
        c = slack_cost(grid, vals, probs, cb, cd)
        assert slack_cost(np.array([beta]), vals, probs, cb, cd)[0] <= c.min() + 1e-9
    # atom at exactly q: interval of minimisers
    vals = np.array([0.0, 1.0]); probs = np.array([0.5, 0.5])
    lo, hi = minimiser_interval_discrete(vals, probs, 0.5)
    assert (lo, hi) == (0.0, 1.0)
    c = slack_cost(np.linspace(0, 1, 11), vals, probs, 1.0, 1.0)
    assert np.ptp(c) < 1e-12
    # Gaussian case: quantile formula
    x = np.linspace(-6, 6, 20001); p = stats.norm.pdf(x); p /= p.sum()
    beta = generalized_quantile_discrete(x, p, 0.75)
    assert abs(beta - stats.norm.ppf(0.75)) < 2e-3
