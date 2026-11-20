"""Hybrid event/tick simulator for the executed reservation model P.

Event batch at time t (tolerance 1e-9), processed in this order:
  1. entries of heads whose slot is due now (checked against every hard
     constraint, then the lane successor is activated at rest at the
     approach start);
  2. births (requests) and public sign reveals;
  3. one scheduling decision from the public state.
Subclasses may add timed actions (e.g. message legs of an installation
protocol) through ``next_extra_event``/``on_extra``; they run after entries and
before births. The base engine has none, so its event sequence is unchanged.
Pure ticks (every ``dt_tick``) only let policies add locks to the incumbent;
on a pure tick the feasible set can only shrink and still contains the
incumbent, so the exact snapshot optimum is unchanged (tie-break toward the
incumbent) and no re-solve is needed.

Policies only see :class:`PublicState`; the true marks and unrevealed signs
stay in the private :class:`Scenario`.
"""
from __future__ import annotations

import heapq
import math
import time as _time
from dataclasses import dataclass, field

import numpy as np

from ..models.physics import InfeasibleArrival, Profile, launch_time, make_profile, t_minus, t_plus
from ..scheduling.core import INF, Job, Plan, Snapshot, forward_times, solve_dp, validate_plan
from .scenario import Scenario, SimConfig, offline_lower_bound

ETOL = 1e-9


class PublicState:
    """Read-only public information handed to policies."""

    def __init__(self, eng: "Engine"):
        self._e = eng

    @property
    def t(self) -> float:
        return self._e.t

    @property
    def lanes(self) -> tuple[list[int], list[int]]:
        return self._e.queue

    def lane(self, vid: int) -> int:
        return int(self._e.lane[vid])

    def a(self, vid: int) -> float:
        return float(self._e.a[vid])

    def rhat(self, vid: int) -> float:
        return float(self._e.rhat[vid])

    def rhat_birth(self, vid: int) -> float:
        return float(self._e.a[vid] + self._e.cfg.lead)

    def U(self, vid: int) -> float:
        return float(self._e.U[vid])

    def n_unrevealed(self, vid: int) -> int:
        return int(self._e.cfg.K - self._e.krev[vid])

    def future_reveals(self, vid: int) -> list[float]:
        """Publicly scheduled future reveal times (empty when signals are withheld)."""
        if not self._e.cfg.reveal:
            return []
        k = int(self._e.krev[vid])
        a = float(self._e.a[vid])
        c = self._e.cfg
        return [a + c.reveal_offset + (j + 1) * c.dq for j in range(k, c.K)]

    def locked(self, before: int, after: int) -> bool:
        return self._e.is_locked(before, after)

    @property
    def cfg(self) -> SimConfig:
        return self._e.cfg


@dataclass
class RunRecord:
    key: tuple
    policy: str
    cfg: SimConfig
    vehicles: dict          # arrays per vehicle
    summary: dict
    violations: list
    lock_log: list = field(default_factory=list)


class Engine:
    def __init__(self, sc: Scenario, policy, collect_trace: bool = False):
        self.sc = sc
        self.cfg = cfg = sc.cfg
        self.policy = policy
        n = sc.n
        self.a = sc.a
        self.lane = sc.lane
        self._r = sc.r                       # private truth
        self._signs = sc.signs               # private
        self.krev = np.zeros(n, dtype=np.int16)
        self.rhat = sc.a + cfg.lead
        self.U = self.rhat + cfg.sigma * math.sqrt(cfg.dq) * cfg.K
        self.S = np.full(n, np.nan)
        self.slot = np.full(n, np.nan)
        self.first_slot = np.full(n, np.nan)
        self.inv = np.zeros(n)
        self.tchg = np.zeros(n, dtype=np.int32)
        self.tmag = np.zeros(n)
        self.act_time = np.full(n, np.nan)
        self.profile: dict[int, Profile] = {}
        self.queue: tuple[list[int], list[int]] = ([], [])
        self.last = [-INF, -INF]
        self.locks_after: dict[int, int] = {}     # after_id -> latest other-lane before_id
        self.lock_log: list = []
        self.violations: list = []
        self.incumbent: Plan | None = None
        self.t = 0.0
        self.kin = cfg.kin
        self.TL = launch_time(cfg.kin) if cfg.kin is not None else 0.0
        self.pub = PublicState(self)
        self.stats = dict(resolves=0, ticks=0, approx=0, fallback=0, dp_none=0, solve_ms=[], labels_max=0,
                          locks=0, deferred=0, peak_queue=0, profile_resets=0)
        self.collect_trace = collect_trace
        self.trace: list = []

    # ------------------------------------------------------------------ locks
    def is_locked(self, before: int, after: int) -> bool:
        """True if 'before precedes after' is implied by locks and lane FIFO."""
        lb, la = self.lane[before], self.lane[after]
        if lb == la:
            return before < after
        # after must follow some other-lane vehicle with id >= before?  check all
        # same-lane vehicles at or before 'after' in its lane queue
        q = self.queue[la]
        if after not in q:
            return False
        for x in q[: q.index(after) + 1]:
            b = self.locks_after.get(x)
            if b is not None and b >= before:
                return True
        return False

    def add_lock(self, before: int, after: int, reason: str) -> bool:
        if self.lane[before] == self.lane[after]:
            return False
        if self.is_locked(before, after):
            return False
        cur = self.locks_after.get(after)
        if cur is None or before > cur:
            self.locks_after[after] = before
        self.lock_log.append((self.t, before, after, reason))
        self.stats["locks"] += 1
        return True

    # --------------------------------------------------------------- physics
    def head_state(self, vid: int) -> tuple[float, float]:
        prof = self.profile.get(vid)
        if prof is None:
            return self.kin.L, 0.0
        return prof.state(self.t)

    def window(self, vid: int) -> tuple[float, float]:
        if self.kin is None:
            return self.t, INF
        p, v = self.head_state(vid)
        return self.t + t_minus(p, v, self.kin), self.t + t_plus(p, v, self.kin)

    # -------------------------------------------------------------- snapshot
    def build_snapshot(self) -> Snapshot:
        pol = self.policy
        extra = pol.floors(self.pub)
        inc_order = self.incumbent.order if self.incumbent is not None else []
        live = set(self.queue[0]) | set(self.queue[1])
        prev_other: dict[int, int] = {}
        seen = [0, 0]
        for x in inc_order:
            if x not in live:
                continue
            l = self.lane[x]
            prev_other[x] = seen[1 - l]
            seen[l] += 1
        lanes: tuple[list[Job], list[Job]] = ([], [])
        for l in (0, 1):
            q = self.queue[l]
            other = self.queue[1 - l]
            pos_other = {x: k for k, x in enumerate(other)}
            need_run = 0
            for k, vid in enumerate(q):
                fl = self.U[vid]
                if vid in extra:
                    fl = max(fl, extra[vid])
                jb = Job(id=vid, lane=l, floor=float(fl), prev_other=prev_other.get(vid, 0),
                         req_other=int(np.searchsorted(np.asarray(other), vid)) if other else 0)
                if k == 0:
                    e, lt = self.window(vid)
                    jb.earliest, jb.latest = e, lt
                b = self.locks_after.get(vid)
                if b is not None and b in pos_other:
                    need_run = max(need_run, pos_other[b] + 1)
                jb.need = need_run
                lanes[l].append(jb)
        eta = pol.eta + self.cfg.tie_eta
        TL = self.TL
        return Snapshot(t=self.t, lanes=lanes, H=self.cfg.H, TL=TL, last=(self.last[0], self.last[1]),
                        eta=eta, cap=self.cfg.cap, eta_req=getattr(pol, "eta_req", 0.0))

    def incumbent_order(self) -> list[int]:
        live = set(self.queue[0]) | set(self.queue[1])
        order = [x for x in (self.incumbent.order if self.incumbent else []) if x in live]
        have = set(order)
        new = sorted(x for x in live if x not in have)
        return order + new

    # ---------------------------------------------------------------- decide
    def decide(self, resolve: bool) -> None:
        for (b, a_) in self.policy.pre_locks(self.pub):
            self.add_lock(b, a_, self.policy.name + ":pre")
        snap = self.build_snapshot()
        if not snap.jobs():
            self.incumbent = Plan(order=[], times={}, cost=0.0)
            return
        plan = None
        if resolve:
            t0 = _time.perf_counter()
            dp = solve_dp(snap)
            self.stats["solve_ms"].append(1e3 * (_time.perf_counter() - t0))
            self.stats["resolves"] += 1
            inc = forward_times(snap, self.incumbent_order())
            if dp is None:
                self.stats["dp_none"] += 1
            if dp is not None:
                self.stats["labels_max"] = max(self.stats["labels_max"], dp.labels)
                if dp.approx:
                    self.stats["approx"] += 1
            if dp is not None and (inc is None or dp.cost <= inc.cost + 1e-9):
                plan = dp
            elif inc is not None:
                plan = inc
                self.stats["fallback"] += 1
            else:
                self.violations.append((self.t, "no feasible plan (dp and incumbent failed)"))
                raise RuntimeError(f"no feasible plan at t={self.t}")
        else:
            self.stats["ticks"] += 1
            plan = self.incumbent
        if self.cfg.audit:
            errs = validate_plan(snap, plan)
            if errs:
                self.violations.append((self.t, "plan: " + "; ".join(errs[:3])))
        # policy locks (must agree with this plan's order)
        pos = {x: k for k, x in enumerate(plan.order)}
        for (b, a_) in self.policy.locks(self.pub, plan, snap):
            if pos[b] < pos[a_]:
                self.add_lock(b, a_, self.policy.name)
            else:
                self.stats["deferred"] += 1
        if resolve:
            self._install(plan)
        self.incumbent = plan

    def _install(self, plan: Plan) -> None:
        old = self.incumbent
        if old is not None and old.order:
            self._count_changes(old, plan)
        for vid, s in plan.times.items():
            if np.isnan(self.first_slot[vid]):
                self.first_slot[vid] = s
            self.slot[vid] = s
        if self.kin is None:
            return
        for l in (0, 1):
            if not self.queue[l]:
                continue
            vid = self.queue[l][0]
            s = plan.times[vid]
            prof = self.profile.get(vid)
            if prof is not None and abs(prof.arrival - s) <= ETOL:
                continue
            p, v = self.head_state(vid)
            try:
                self.profile[vid] = make_profile(self.t, p, v, s, self.kin)
                self.stats["profile_resets"] += 1
            except InfeasibleArrival as ex:
                self.violations.append((self.t, f"infeasible profile for {vid}: {ex}"))
                raise

    def _count_changes(self, old: Plan, new: Plan) -> None:
        common = set(old.order) & set(new.order)
        def counts(order):
            seen = [0, 0]
            out = {}
            for x in order:
                if x in common:
                    l = self.lane[x]
                    out[x] = seen[1 - l]
                    seen[l] += 1
            return out
        co, cn = counts(old.order), counts(new.order)
        for x in common:
            d = abs(co[x] - cn[x])
            if d:
                self.inv[x] += 0.5 * d
            dt = abs(new.times[x] - old.times[x])
            if dt > 0.1:
                self.tchg[x] += 1
                self.tmag[x] += dt

    # ------------------------------------------------------------ extension
    def next_extra_event(self) -> float | None:
        """Time of the next subclass action; None in the base engine."""
        return None

    def on_extra(self, t: float) -> bool:
        """Run subclass actions due at t; True requests a re-solve at t."""
        return False

    # ------------------------------------------------------------------ run
    def _enter(self, vid: int) -> None:
        t = self.t
        l = int(self.lane[vid])
        cfg = self.cfg
        if t < self._r[vid] - 1e-7:
            self.violations.append((t, f"{vid} entered before its mark"))
        for ln in (0, 1):
            if t < self.last[ln] + cfg.H[ln][l] - 1e-7:
                self.violations.append((t, f"{vid} clearance violation vs lane {ln}"))
        b = self.locks_after.get(vid)
        if b is not None and np.isnan(self.S[b]):
            self.violations.append((t, f"{vid} entered before locked predecessor {b}"))
        if self.queue[l][0] != vid:
            self.violations.append((t, f"{vid} lane FIFO violation"))
        if self.kin is not None and not np.isnan(self.act_time[vid]) and t < self.act_time[vid] + self.TL - 1e-7:
            self.violations.append((t, f"{vid} faster than physical launch"))
        self.S[vid] = t
        self.last[l] = t
        self.queue[l].pop(0)
        self.profile.pop(vid, None)
        if self.queue[l]:
            nxt = self.queue[l][0]
            self.act_time[nxt] = t

    def run(self) -> RunRecord:
        cfg, sc = self.cfg, self.sc
        n = sc.n
        births = list(range(n))
        bi = 0
        reveals: list[tuple[float, int]] = []
        tick_k = 1
        uses_ticks = self.policy.uses_ticks
        t_stop = cfg.t_arr_end + cfg.drain_limit
        while True:
            cand = []
            if bi < n:
                cand.append(self.a[births[bi]])
            if reveals:
                cand.append(reveals[0][0])
            for l in (0, 1):
                if self.queue[l]:
                    s = self.slot[self.queue[l][0]]
                    if not np.isnan(s):
                        cand.append(s)
            nx = self.next_extra_event()
            if nx is not None:
                cand.append(nx)
            live = self.queue[0] or self.queue[1]
            if not cand:
                if bi >= n and not live:
                    break
            if uses_ticks and live:
                cand.append(tick_k * cfg.dt_tick)
            if not cand:
                break
            t = min(cand)
            if t > t_stop:
                self.violations.append((t, "drain limit exceeded"))
                break
            self.t = t
            while tick_k * cfg.dt_tick <= t + ETOL:
                tick_k += 1
            info = False
            # 1. entries
            for l in (0, 1):
                if self.queue[l]:
                    vid = self.queue[l][0]
                    if abs(self.slot[vid] - t) <= ETOL:
                        self._enter(vid)
                        info = True
            # 1b. subclass actions due now (none in the base engine)
            if self.on_extra(t):
                info = True
            # 2. births
            while bi < n and self.a[births[bi]] <= t + ETOL:
                vid = births[bi]
                l = int(self.lane[vid])
                self.queue[l].append(vid)
                if len(self.queue[l]) == 1:
                    self.act_time[vid] = t
                if cfg.reveal:
                    for k in range(cfg.K):
                        heapq.heappush(reveals, (self.a[vid] + cfg.reveal_offset + (k + 1) * cfg.dq, vid))
                for (b, a_) in self.policy.on_birth(self.pub, vid):
                    self.add_lock(b, a_, self.policy.name + ":birth")
                bi += 1
                info = True
            # reveals
            while reveals and reveals[0][0] <= t + ETOL:
                _, vid = heapq.heappop(reveals)
                if not np.isnan(self.S[vid]):
                    continue
                self.krev[vid] += 1
                k = int(self.krev[vid])
                step = cfg.sigma * math.sqrt(cfg.dq)
                self.rhat[vid] = self.a[vid] + cfg.lead + step * float(self._signs[vid, :k].sum())
                self.U[vid] = self.rhat[vid] + step * (cfg.K - k)
                info = True
            self.stats["peak_queue"] = max(self.stats["peak_queue"], len(self.queue[0]) + len(self.queue[1]))
            if self.queue[0] or self.queue[1]:
                self.decide(resolve=info or self.incumbent is None)
            elif info:
                self.incumbent = Plan(order=[], times={}, cost=0.0)
            if self.collect_trace and info:
                self.trace.append((t, [(x, self.slot[x]) for q in self.queue for x in q]))
        return self._record()

    def _record(self) -> RunRecord:
        cfg, sc = self.cfg, self.sc
        coh = np.where((sc.a >= cfg.warmup) & (sc.a < cfg.win_end))[0]
        done = ~np.isnan(self.S[coh])
        D = self.S[coh] - sc.r[coh]
        f = sc.a[coh] + self.TL
        summ = dict(n_cohort=int(coh.size), n_done=int(done.sum()), n_total=int(sc.n))
        if coh.size and done.all():
            summ.update(mean_delay=float(D.mean()), p95_delay=float(np.percentile(D, 95)),
                        max_delay=float(D.max()),
                        mean_delay_l0=float(D[sc.lane[coh] == 0].mean()) if np.any(sc.lane[coh] == 0) else np.nan,
                        mean_delay_l1=float(D[sc.lane[coh] == 1].mean()) if np.any(sc.lane[coh] == 1) else np.nan,
                        mean_S_minus_f=float(np.mean(self.S[coh] - f)),
                        inv_per_veh=float(self.inv[coh].mean()),
                        tchg_per_veh=float(self.tchg[coh].mean()),
                        tmag_per_veh=float(self.tmag[coh].mean()),
                        guard_gap=float(np.mean(self.first_slot[coh] - sc.r[coh])),
                        lb_mean_delay=offline_lower_bound(sc, coh))
        else:
            for k in ("mean_delay", "p95_delay", "max_delay", "mean_delay_l0", "mean_delay_l1",
                      "mean_S_minus_f", "inv_per_veh", "tchg_per_veh", "tmag_per_veh", "guard_gap", "lb_mean_delay"):
                summ[k] = float("nan")
        if np.all(~np.isnan(self.S)) and sc.n:
            Dall = self.S - sc.r
            summ.update(mean_delay_all=float(Dall.mean()), total_delay_all=float(Dall.sum()),
                        p95_delay_all=float(np.percentile(Dall, 95)))
        else:
            summ.update(mean_delay_all=float("nan"), total_delay_all=float("nan"), p95_delay_all=float("nan"))
        ent = self.S[(self.S >= cfg.warmup) & (self.S < cfg.win_end)]
        summ["throughput"] = float(ent.size / (cfg.win_end - cfg.warmup))
        ms = np.array(self.stats["solve_ms"]) if self.stats["solve_ms"] else np.zeros(1)
        summ.update(resolves=self.stats["resolves"], ticks=self.stats["ticks"], approx=self.stats["approx"],
                    fallback=self.stats["fallback"], dp_none=self.stats["dp_none"],
                    solve_ms_mean=float(ms.mean()), solve_ms_p95=float(np.percentile(ms, 95)),
                    labels_max=self.stats["labels_max"], locks=self.stats["locks"],
                    deferred=self.stats["deferred"], peak_queue=self.stats["peak_queue"],
                    profile_resets=self.stats["profile_resets"], violations=len(self.violations),
                    drained=bool(np.all(~np.isnan(self.S))))
        summ.update(self.policy.diagnostics())
        veh = dict(a=sc.a, lane=sc.lane, r=sc.r, S=self.S, first_slot=self.first_slot, inv=self.inv,
                   tchg=self.tchg)
        return RunRecord(key=sc.key, policy=self.policy.name, cfg=cfg, vehicles=veh, summary=summ,
                         violations=self.violations, lock_log=self.lock_log)


def run(sc: Scenario, policy, **kw) -> RunRecord:
    return Engine(sc, policy, **kw).run()
