"""Information models.

Model P (executed simulator): each request i carries a not-before mark
    r_i = a_i + lead + sigma * sqrt(dq) * sum_{k=1..K} Z_ik,   Z_ik iid Rademacher,
whose signs are revealed publicly at request ages k*dq.  After k reveals the
conditional law of r_i is the revealed prefix plus an independent sum of K-k
signs, so  r_hat = E[r | F_t]  is a martingale,  Var(r | F_t) = sigma^2 (K-k) dq,
and the known upper support U = r_hat + sigma*sqrt(dq)*(K-k) is nonincreasing.

``gaussian_hinge`` is the Gaussian reference case of the hinge cost (kept with
its tests; the Gaussian two-vehicle Bellman study of the research archive is not
part of this submission).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import ndtr

SQRT2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class SignProcess:
    sigma: float = 0.5      # s / sqrt(s)
    K: int = 4              # number of sign reveals
    dq: float = 1.0         # information time per reveal (s); reveal at ages k*dq
    lead: float = 8.25      # nominal request-to-mark lead (s)

    @property
    def step(self) -> float:
        return self.sigma * math.sqrt(self.dq)

    def reveal_ages(self) -> list[float]:
        return [k * self.dq for k in range(1, self.K + 1)]

    def posterior(self, a: float, revealed: np.ndarray | list[int]) -> tuple[float, float, float, float]:
        """(r_hat, var, U, Lsup) from request time and the revealed sign prefix."""
        k = len(revealed)
        s = float(np.sum(revealed)) if k else 0.0
        r_hat = a + self.lead + self.step * s
        rem = self.K - k
        var = self.sigma ** 2 * rem * self.dq
        U = r_hat + self.step * rem
        Ls = r_hat - self.step * rem
        return r_hat, var, U, Ls

    def mark(self, a: float, signs: np.ndarray) -> float:
        return a + self.lead + self.step * float(np.sum(signs))

    def min_mark_offset(self) -> float:
        return self.lead - self.step * self.K


@lru_cache(maxsize=None)
def rademacher_sum_pmf(n: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Exact pmf of S_n = sum of n iid Rademacher signs: values -n..n step 2."""
    vals = tuple(range(-n, n + 1, 2))
    probs = tuple(math.comb(n, (v + n) // 2) / 2 ** n for v in vals)
    return vals, probs


def gap_kernel(n_i: int, n_j: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Law of the remaining increment of m = r_hat_j - r_hat_i.

    It is step*(S_j - S_i) with independent remaining sums; since -S is
    distributed as S, this equals step * S_{n_i+n_j}.  Variance = step^2 (n_i+n_j),
    i.e. the doubled intensity for equal clocks.
    """
    vals, probs = rademacher_sum_pmf(n_i + n_j)
    return step * np.array(vals, dtype=float), np.array(probs)


def discrete_hinge(h: float, m: float, n: int, step: float) -> float:
    """E[(h - (m + step*S_n))^+]."""
    vals, probs = rademacher_sum_pmf(n)
    return float(sum(p * max(0.0, h - m - step * v) for v, p in zip(vals, probs)))


def gaussian_hinge(h: float, m, v):
    """G_h(m, v) = E[(h - X)^+], X ~ N(m, v); G_h(m, 0) = (h - m)^+.  Vectorised."""
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.maximum(h - m, 0.0)
    pos = v > 0
    if np.any(pos):
        sd = np.sqrt(np.where(pos, v, 1.0))
        z = (h - m) / sd
        val = (h - m) * ndtr(z) + sd * np.exp(-0.5 * z * z) / SQRT2PI
        out = np.where(pos, val, out)
    return out if out.ndim else float(out)
