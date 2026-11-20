"""Commitment policies.  All share the engine, scheduler, physics and safety
layer; they differ only in (i) extra floors, (ii) the snapshot change penalty
and (iii) which cross-lane order locks they install.  Policies never draw
random numbers and only read :class:`PublicState`.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..models.physics import Kinematics, kappa_surrogate, launch_time
from ..stopping.pair import KEEP, SWAP, pair_advice, pair_advice_queue


class Policy:
    name = "base"
    eta = 0.0
    uses_ticks = False

    def floors(self, pub) -> dict:
        return {}

    def on_birth(self, pub, vid) -> list:
        return []

    def pre_locks(self, pub) -> list:
        """Locks to install before the snapshot is solved (transfer experiments only)."""
        return []

    def locks(self, pub, plan, snap) -> list:
        return []

    def diagnostics(self) -> dict:
        return {}


class Rolling(Policy):
    """Full rolling re-optimisation: no voluntary cross-lane locks."""
    name = "rolling"


class RollingFIFO(Rolling):
    """Rolling with request-order tie-breaking (post-hoc robustness variant, ruling R9):
    among snapshot plans of equal reserved-start sum, prefer fewer inversions
    relative to request order (weight 1e-8 s, below the 1e-6 s incumbent tie-break)."""
    name = "rolling_fifo"
    eta_req = 1e-8


class FCFS(Policy):
    """Global request order: each new request is locked behind every live
    request of the other lane (times stay revisable)."""
    name = "fcfs"

    def on_birth(self, pub, vid):
        other = pub.lanes[1 - pub.lane(vid)]
        others = [x for x in other if x != vid]
        return [(others[-1], vid)] if others else []


class ChangePenalty(Policy):
    def __init__(self, eta: float):
        self.eta = float(eta)
        self.name = f"penalty_{eta:g}"


class QuantileSlack(FCFS):
    """FCFS order with entry target r_hat(a_i) + beta stored as an extra floor."""

    def __init__(self, beta: float, label: str):
        self.beta = float(beta)
        self.name = f"slack_{label}"

    def floors(self, pub):
        return {x: pub.rhat_birth(x) + self.beta for q in pub.lanes for x in q}


def _vehicle_freeze_locks(pub, plan, frozen: set) -> list:
    """Lock each frozen vehicle's witnessed position relative to all other-lane
    live vehicles: (last other-lane vehicle before it, it) and (it, first
    other-lane vehicle after it).  Lane FIFO implies the rest."""
    out = []
    order = plan.order
    lane = {x: pub.lane(x) for x in order}
    for k, x in enumerate(order):
        if x not in frozen:
            continue
        l = lane[x]
        for y in reversed(order[:k]):
            if lane[y] != l:
                out.append((y, x))
                break
        for y in order[k + 1:]:
            if lane[y] != l:
                out.append((x, y))
                break
    return out


class Freeze(Policy):
    """Lock a pair once min nominal lead min(max(0, r_hat - t)) <= H, i.e. once
    either member is within H of its estimated mark."""
    uses_ticks = True

    def __init__(self, H: float):
        self.H = float(H)
        self.name = "freeze_inf" if math.isinf(H) else f"freeze_{H:g}"

    def locks(self, pub, plan, snap):
        t = pub.t
        frozen = {x for x in plan.order if max(0.0, pub.rhat(x) - t) <= self.H}
        return _vehicle_freeze_locks(pub, plan, frozen)


class Adaptive(Policy):
    """Pair-stopping commitment (RESEARCH_PLAN D4).

    For each unlocked adjacent cross-lane pair (x, y) of the current witness:
    m = r_hat_y - r_hat_x, lead u = max(0, min(s_x, s_y) - t), public future
    reveal times, kernel step sigma_lookup*sqrt(dq) and the time-only physical
    surrogate kappa(u) = 2 g(.).  KEEP => lock x before y; SWAP => deferred
    (recorded, not installed); WAIT => nothing.
    """
    uses_ticks = True

    def __init__(self, kin: Kinematics | None, h: float, sigma_lookup: float, dq: float = 1.0,
                 whole: bool = False, kappa_zero: bool = False, name: str | None = None,
                 queue_aware: bool = False, H=None):
        self.h = h
        self.queue_aware = queue_aware
        self.Hm = H if H is not None else ((h, h), (h, h))
        self.step = sigma_lookup * math.sqrt(dq)
        self.whole = whole
        if kin is None or kappa_zero:
            self.kappa = lambda u: 0.0
        else:
            Tf = launch_time(kin)
            grid = np.linspace(0.0, Tf, 401)
            vals = np.array([kappa_surrogate(u, h, kin) for u in grid])
            self.kappa = lambda u, g=grid, v=vals: float(np.interp(u, g, v)) if u < g[-1] else 0.0
        self.name = name or (("adaptive_q" if queue_aware else "adaptive") + ("_whole" if whole else ""))
        self.n_query = self.n_keep = self.n_swap = self.n_wait = 0

    def locks(self, pub, plan, snap):
        t = pub.t
        order = plan.order
        out = []
        frozen = set()
        for k in range(len(order) - 1):
            x, y = order[k], order[k + 1]
            if pub.lane(x) == pub.lane(y) or pub.locked(x, y):
                continue
            lead = max(0.0, min(plan.times[x], plan.times[y]) - t)
            if self.queue_aware:
                # resource first free for the pair: witness predecessor's start + clearance,
                # or executed history; physical earliest of a moving head is ignored (pair surrogate)
                if k > 0:
                    p = order[k - 1]
                    T0 = plan.times[p] + self.Hm[pub.lane(p)][pub.lane(x)]
                else:
                    T0 = max(snap.last[ln] + self.Hm[ln][pub.lane(x)] for ln in (0, 1))
                    T0 = max(T0, t)
                adv = pair_advice_queue(pub.rhat(x), pub.rhat(y), T0, t, pub.future_reveals(x),
                                        pub.future_reveals(y), self.step, self.h, lead, self.kappa)
            else:
                m = pub.rhat(y) - pub.rhat(x)
                adv = pair_advice(m, t, pub.future_reveals(x), pub.future_reveals(y), self.step, self.h,
                                  lead, self.kappa)
            self.n_query += 1
            if adv.action == KEEP:
                self.n_keep += 1
                if self.whole:
                    frozen.update((x, y))
                else:
                    out.append((x, y))
            elif adv.action == SWAP:
                self.n_swap += 1
                out.append((y, x))      # engine records it as deferred (contradicts witness)
            else:
                self.n_wait += 1
        if self.whole and frozen:
            out.extend(_vehicle_freeze_locks(pub, plan, frozen))
        return out

    def diagnostics(self):
        return dict(adv_query=self.n_query, adv_keep=self.n_keep, adv_swap=self.n_swap, adv_wait=self.n_wait)


def slack_beta(sigma: float, K: int, ratio: float) -> float:
    """Generalised quantile F^{-1}(c_b/(c_b+c_d)) of eps = r - r_hat(a) = sigma*S_K."""
    from ..theory.slack import generalized_quantile_discrete
    from ..models.belief import rademacher_sum_pmf
    vals, probs = rademacher_sum_pmf(K)
    q = ratio / (1.0 + ratio)
    return max(0.0, sigma * generalized_quantile_discrete(np.array(vals, float), np.array(probs), q))
