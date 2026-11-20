"""Proposition 3 (single-slot slack): newsvendor generalised quantile.

Cost  C(beta) = c_d E[(beta - eps)^+] + c_b E[(eps - beta)^+].  For c_b, c_d > 0
the subgradient condition is F(beta-) <= c_b/(c_b+c_d) <= F(beta): the set of
minimisers is the (possibly non-unique) generalised quantile interval.
"""
from __future__ import annotations

import numpy as np


def generalized_quantile_discrete(vals: np.ndarray, probs: np.ndarray, q: float) -> float:
    """Smallest beta with F(beta) >= q for a discrete law."""
    o = np.argsort(vals)
    v, p = vals[o], probs[o]
    F = np.cumsum(p)
    k = int(np.searchsorted(F, q - 1e-12))
    return float(v[min(k, len(v) - 1)])


def minimiser_interval_discrete(vals: np.ndarray, probs: np.ndarray, q: float) -> tuple[float, float]:
    o = np.argsort(vals)
    v, p = vals[o], probs[o]
    F = np.cumsum(p)
    lo = generalized_quantile_discrete(v, p, q)
    # upper end: largest beta with F(beta-) <= q
    k = int(np.searchsorted(F, q - 1e-12))
    if k < len(F) and abs(F[k] - q) < 1e-12 and k + 1 < len(v):
        return lo, float(v[k + 1])
    return lo, lo


def slack_cost(beta: np.ndarray, vals: np.ndarray, probs: np.ndarray, cb: float, cd: float) -> np.ndarray:
    beta = np.atleast_1d(beta)[:, None]
    return (cd * np.maximum(beta - vals[None, :], 0) + cb * np.maximum(vals[None, :] - beta, 0)) @ probs
