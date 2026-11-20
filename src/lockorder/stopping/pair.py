"""Finite pair stopping problem for the executed (P) information model.

A query concerns an ordered pair (first, second) of conflicting vehicles in the
current witness, with gap estimate m = r_hat_second - r_hat_first.  Future
information arrives only at known public reveal times (bounded sign
increments).  Decision epochs are "now" and every future reveal strictly
before the pair deadline t + lead.  Between reveals waiting yields no
information and the time-only surrogate kappa cannot fall, so stopping between
reveals is never strictly better than stopping at the preceding epoch.

Stop costs at an epoch with n remaining signs (both vehicles):
    keep  (first < second): E[(h - (m + X))^+] + kappa(lead_e)
    swap  (second < first): E[(h + (m + X))^+] + kappa(lead_e)
with X = step * S_n.  The last epoch must stop.  Ties: wait unless it is the
last epoch; a terminal keep/swap tie keeps the witness order.

This rule is optimal for its stated finite *pair surrogate* only, not for the
physical queue (RESEARCH_PLAN D4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..models.belief import discrete_hinge, rademacher_sum_pmf

KEEP, SWAP, WAIT = "keep", "swap", "wait"


@dataclass(frozen=True)
class PairAdvice:
    action: str
    v_keep: float
    v_swap: float
    v_wait: float          # inf if this is the last epoch
    n_epochs: int


def pair_advice(m0: float, t: float, rev_first: Sequence[float], rev_second: Sequence[float],
                step: float, h: float, lead: float, kappa: Callable[[float], float],
                tol: float = 1e-10) -> PairAdvice:
    deadline = t + max(lead, 0.0)
    rf = [x for x in rev_first if x > t + 1e-12]
    rs = [x for x in rev_second if x > t + 1e-12]
    times = [t] + sorted({x for x in rf + rs if x < deadline - 1e-12})
    E = len(times)
    rem = [sum(1 for x in rf if x > te + 1e-12) + sum(1 for x in rs if x > te + 1e-12) for te in times]
    inc = [0] + [sum(1 for x in rf + rs if abs(x - times[e]) <= 1e-12) for e in range(1, E)]
    kap = [kappa(lead - (te - t)) for te in times]
    memo: dict[tuple[int, int], float] = {}

    def stop_costs(e: int, k: int) -> tuple[float, float]:
        m = m0 + step * k
        return (discrete_hinge(h, m, rem[e], step) + kap[e],
                discrete_hinge(h, -m, rem[e], step) + kap[e])

    def value(e: int, k: int) -> float:
        key = (e, k)
        if key in memo:
            return memo[key]
        ck, cs = stop_costs(e, k)
        best = min(ck, cs)
        if e < E - 1:
            vals, probs = rademacher_sum_pmf(inc[e + 1])
            cont = sum(p * value(e + 1, k + v) for v, p in zip(vals, probs))
            best = min(best, cont)
        memo[key] = best
        return best

    ck, cs = stop_costs(0, 0)
    if E == 1:
        cw = float("inf")
        action = KEEP if ck <= cs + tol else SWAP
    else:
        vals, probs = rademacher_sum_pmf(inc[1])
        cw = sum(p * value(1, v) for v, p in zip(vals, probs))
        best_stop = min(ck, cs)
        if best_stop < cw - tol:
            action = KEEP if ck <= cs + tol else SWAP
        else:
            action = WAIT
    return PairAdvice(action=action, v_keep=ck, v_swap=cs, v_wait=cw, n_epochs=E)


def pair_advice_queue(rx: float, ry: float, T0: float, t: float, rev_x: Sequence[float],
                      rev_y: Sequence[float], step: float, h: float, lead: float,
                      kappa: Callable[[float], float], tol: float = 1e-10) -> PairAdvice:
    """Queue-aware pair surrogate (secondary variant, RESEARCH_PLAN ruling R-adaptive-Q).

    Same epochs/kappa as :func:`pair_advice`, but an order's cost is the pair's
    expected total delay when the shared resource is first free at ``T0``
    (the witness start of the job preceding the pair plus clearance):
        keep:  s_x = max(r_x, T0), s_y = max(r_y, s_x + h)
        swap:  s_y = max(r_y, T0), s_x = max(r_x, s_y + h)
    with r = estimate + step * (remaining signs).  With T0 = -inf this equals
    :func:`pair_advice` exactly.
    """
    deadline = t + max(lead, 0.0)
    fx = [x for x in rev_x if x > t + 1e-12]
    fy = [x for x in rev_y if x > t + 1e-12]
    times = [t] + sorted({x for x in fx + fy if x < deadline - 1e-12})
    E = len(times)
    rem_x = [sum(1 for x in fx if x > te + 1e-12) for te in times]
    rem_y = [sum(1 for x in fy if x > te + 1e-12) for te in times]
    inc_x = [0] + [sum(1 for x in fx if abs(x - times[e]) <= 1e-12) for e in range(1, E)]
    inc_y = [0] + [sum(1 for x in fy if abs(x - times[e]) <= 1e-12) for e in range(1, E)]
    kap = [kappa(lead - (te - t)) for te in times]
    memo: dict = {}

    def stop_costs(e: int, kx: int, ky: int) -> tuple[float, float]:
        vx, px = rademacher_sum_pmf(rem_x[e])
        vy, py = rademacher_sum_pmf(rem_y[e])
        ck = cs = 0.0
        for ax, qx in zip(vx, px):
            r_x = rx + step * (kx + ax)
            for ay, qy in zip(vy, py):
                r_y = ry + step * (ky + ay)
                w = qx * qy
                sx = max(r_x, T0); sy = max(r_y, sx + h)
                ck += w * (sx - r_x + sy - r_y)
                sy2 = max(r_y, T0); sx2 = max(r_x, sy2 + h)
                cs += w * (sx2 - r_x + sy2 - r_y)
        return ck + kap[e], cs + kap[e]

    def value(e: int, kx: int, ky: int) -> float:
        key = (e, kx, ky)
        if key in memo:
            return memo[key]
        ck, cs = stop_costs(e, kx, ky)
        best = min(ck, cs)
        if e < E - 1:
            vx, px = rademacher_sum_pmf(inc_x[e + 1])
            vy, py = rademacher_sum_pmf(inc_y[e + 1])
            cont = sum(qx * qy * value(e + 1, kx + ax, ky + ay)
                       for ax, qx in zip(vx, px) for ay, qy in zip(vy, py))
            best = min(best, cont)
        memo[key] = best
        return best

    ck, cs = stop_costs(0, 0, 0)
    if E == 1:
        return PairAdvice(KEEP if ck <= cs + tol else SWAP, ck, cs, float("inf"), 1)
    vx, px = rademacher_sum_pmf(inc_x[1])
    vy, py = rademacher_sum_pmf(inc_y[1])
    cw = sum(qx * qy * value(1, ax, ay) for ax, qx in zip(vx, px) for ay, qy in zip(vy, py))
    if min(ck, cs) < cw - tol:
        action = KEEP if ck <= cs + tol else SWAP
    else:
        action = WAIT
    return PairAdvice(action, ck, cs, cw, E)
