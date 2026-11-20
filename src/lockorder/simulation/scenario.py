"""Exogenous scenario generation with keyed, policy-independent randomness.

Every random quantity is drawn from a stream keyed by
(study, split, replication, stream-type, lane) through ``numpy.random.SeedSequence``.
Inter-arrival times are unit exponentials scaled by the rate, so the same
replication key yields common random numbers across loads and noise levels;
vehicle k on lane l always receives row k of that lane's sign stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models.belief import SignProcess
from ..models.physics import Kinematics

SPLITS = {"pilot": 0, "tune": 1, "test": 2, "mc": 3, "boot": 4}
SEED_BASE = {"pilot": 0, "tune": 1000, "test": 10000, "mc": 20000, "boot": 30000}
STUDY_KEY = 329


@dataclass(frozen=True)
class SimConfig:
    rho: float = 0.5
    sigma: float = 0.5
    h: float = 3.0                   # cross-lane clearance (s)
    hf: float = 3.0                  # same-lane following headway (s)
    kin: Kinematics | None = field(default_factory=Kinematics)   # None => zero-memory point queue (A1)
    K: int = 4
    dq: float = 1.0
    lead: float = 8.25
    reveal: bool = True              # False => A3: no informative signals before execution
    reveal_offset: float = 0.0       # reveal k at request age reveal_offset + k*dq (later-revelation sensitivity)
    lane_split: float = 0.5          # fraction of offered load on lane 0
    t_arr_end: float = 1020.0
    warmup: float = 300.0
    win_end: float = 900.0
    dt_tick: float = 0.5
    cap: int = 32
    tie_eta: float = 1e-6
    drain_limit: float = 4000.0
    audit: bool = False

    @property
    def sp(self) -> SignProcess:
        return SignProcess(sigma=self.sigma, K=self.K, dq=self.dq, lead=self.lead)

    @property
    def H(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.hf, self.h), (self.h, self.hf))

    @property
    def lam_total(self) -> float:
        # offered load rho = lambda_total * h (all-conflict reference headway)
        return self.rho / self.h


@dataclass
class Scenario:
    a: np.ndarray        # request times, sorted (global ID = index)
    lane: np.ndarray     # lane per vehicle
    signs: np.ndarray    # (n, K) Rademacher signs  -- PRIVATE
    r: np.ndarray        # true not-before marks     -- PRIVATE
    key: tuple
    cfg: SimConfig

    @property
    def n(self) -> int:
        return len(self.a)


def make_scenario(cfg: SimConfig, split: str, rep: int, study: int = STUDY_KEY) -> Scenario:
    seed_rep = SEED_BASE[split] + rep
    a_list, lane_list, sign_list = [], [], []
    for l in (0, 1):
        frac = cfg.lane_split if l == 0 else 1.0 - cfg.lane_split
        lam = cfg.lam_total * frac
        rng_a = np.random.default_rng(np.random.SeedSequence([study, SPLITS[split], seed_rep, 1, l]))
        rng_s = np.random.default_rng(np.random.SeedSequence([study, SPLITS[split], seed_rep, 2, l]))
        # draw unit exponentials in fixed-size blocks so the stream does not depend on lam
        times = []
        t = 0.0
        while True:
            e = rng_a.exponential(1.0, size=256)
            done = False
            for x in e:
                t += x / lam if lam > 0 else np.inf
                if t >= cfg.t_arr_end:
                    done = True
                    break
                times.append(t)
            if done:
                break
        n = len(times)
        signs = rng_s.choice(np.array([-1, 1], dtype=np.int8), size=(max(n, 1), cfg.K))[:n]
        a_list.append(np.array(times))
        lane_list.append(np.full(n, l, dtype=np.int8))
        sign_list.append(signs)
    a = np.concatenate(a_list)
    lane = np.concatenate(lane_list)
    signs = np.concatenate(sign_list) if a.size else np.zeros((0, cfg.K), dtype=np.int8)
    order = np.lexsort((lane, a))
    a, lane, signs = a[order], lane[order], signs[order]
    r = a + cfg.lead + cfg.sigma * np.sqrt(cfg.dq) * signs.sum(axis=1)
    return Scenario(a=a, lane=lane, signs=signs, r=r, key=(study, split, rep), cfg=cfg)


def offline_lower_bound(sc: Scenario, cohort: np.ndarray) -> float:
    """Mean delay lower bound for the cohort (RESEARCH_PLAN D5).

    Knows all true marks of the cohort; removes all other traffic, locks,
    launch constraints and lane order; one server with the smallest
    applicable separation.  Earliest-release-first is optimal for the sum of
    starts with identical separations.
    """
    if cohort.size == 0:
        return float("nan")
    hmin = min(sc.cfg.h, sc.cfg.hf)
    rs = np.sort(sc.r[cohort])
    s = np.empty_like(rs)
    prev = -np.inf
    for k, x in enumerate(rs):
        prev = max(x, prev + hmin)
        s[k] = prev
    return float(np.mean(s - rs))
