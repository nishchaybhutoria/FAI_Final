"""Closed loop: the rolling engine with every revision installed by release-gated
single substitutions on a lossy channel (the archived direct contract, run
continuously instead of on isolated forks).

Manager state
  P[i]   slots still protected for booked live vehicle i. A grant adds its target;
         only a matching ACK retires the other slot. A lost grant and a lost ACK
         are indistinguishable, so either leaves two protected slots.
  A vehicle with two protected slots is *pinned*: it is scheduled at the later slot,
  the earlier one becomes a block for everyone else, and it is not revised again
  (the manager does not know which version it holds) until its entry is observed.

Decisions are serialized. While a transaction is pending, decisions are deferred
and one re-solve runs when it ends. A newcomer receives a reliable initial booking
that fits around every protected slot (booked vehicles pinned); this is the
archived preauthorized-arrival assumption. All later changes are revisions.

Installation modes
  direct   the policy's target, Eq. (4) closed subset of certified moves
  handoff  best of the direct menu and every admissible one-detour handoff
The atomic reference is the unmodified base engine.
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from math import fsum

import numpy as np

from ..models.physics import InfeasibleArrival, make_profile
from ..scheduling.core import Plan, forward_times, solve_dp, validate_plan
from ..simulation.engine import ETOL, Engine
from ..simulation.scenario import SPLITS
from . import handoff as H
from .model import EPS, ancestry, best_closed, compile_update, Fork
from .targets import options


@dataclass
class LoopFork(Fork):
    """A fork whose vehicles may include pinned bookings with another possible slot,
    and whose committed order locks make the conflict predicate one-sided."""
    extra: dict = field(default_factory=dict)
    locked: frozenset = frozenset()     # (before, after) pairs that every booking must respect

    def compatible(self, i, x, j, y):
        if (i, j) in self.locked:
            return y - x >= self.snap.H[self.lane[i]][self.lane[j]] - EPS
        if (j, i) in self.locked:
            return x - y >= self.snap.H[self.lane[j]][self.lane[i]] - EPS
        return super().compatible(i, x, j, y)


class ClosedLoop(Engine):
    def __init__(self, sc, policy, link, mode="direct"):
        super().__init__(sc, policy)
        if mode not in ("direct", "handoff"):
            raise ValueError(mode)
        self.link, self.mode = link, mode
        self.P: dict[int, set] = {}
        self.tx = None
        self.deferred = False
        self.epoch = 0
        self.attempt = defaultdict(int)
        self._draws = {}
        study, split, rep = sc.key
        self._key = (study, SPLITS[split], rep)
        self.pinned_since = {}
        self.cl = dict(transactions=0, deferred=0, bookings=0, grants=0, delivered=0, switched=0, acks=0,
                       pinned=0, pinned_s=0., detours=0, parks=0, returns=0, stranded=0, no_gain=0,
                       lock_relaxed=0, inserted=0, gated_bookings=0)
        self.offered = set()

    # ------------------------------------------------------------ manager view
    def compatible(self, i, x, j, y):
        a, b = int(self.lane[i]), int(self.lane[j])
        H_ = self.cfg.H
        if a != b and self.is_locked(i, j):
            return y - x >= H_[a][b] - EPS
        if a != b and self.is_locked(j, i):
            return x - y >= H_[b][a] - EPS
        if a == b:
            gap = y - x if i < j else x - y
            return gap >= max(H_[a][a], self.TL) - EPS
        return y - x >= H_[a][b] - EPS or x - y >= H_[b][a] - EPS

    def uncertain(self):
        return {i for i, s in self.P.items() if len(s) > 1}

    def known(self):
        return {i: max(s) for i, s in self.P.items()}

    def known_plan(self):
        k = self.known()
        return Plan(order=sorted(k, key=k.get), times=k, cost=fsum(k.values()))

    def manager_snapshot(self, pin_booked):
        """Planning snapshot from what the manager knows. With ``pin_booked`` every
        booked vehicle is fixed (initial insertion of newcomers, no order locks)."""
        snap = self.build_snapshot()
        unc, known = self.uncertain(), self.known()
        for job in snap.jobs():
            if job.id in unc or (pin_booked and job.id in known):
                job.floor = job.earliest = job.latest = known[job.id]
            if pin_booked:
                job.need = 0
        blocks = tuple((min(self.P[i]), int(self.lane[i]), i) for i in sorted(unc))
        return replace(snap, blocks=blocks)

    def _solve(self, snap):
        """(plan, snapshot actually solved); see the lock-relaxation note below."""
        dp = solve_dp(snap)
        self.stats["resolves"] += 1
        inc = forward_times(snap, self.incumbent_order())
        if dp is not None:
            self.stats["labels_max"] = max(self.stats["labels_max"], dp.labels)
            if dp.approx:
                self.stats["approx"] += 1
        if dp is not None and (inc is None or dp.cost <= inc.cost + 1e-9):
            return dp, snap
        if inc is not None:
            self.stats["fallback"] += 1
            return inc, snap
        if any(j.need for j in snap.jobs()):
            # After a partial installation the bookings can contradict an order lock
            # taken against an earlier target; the lock cannot be honoured, so plan
            # without locks for this decision (counted, never silent).
            self.cl["lock_relaxed"] += 1
            return self._solve(replace(snap, lanes=tuple([replace(j, need=0) for j in q] for q in snap.lanes)))
        raise RuntimeError(f"no feasible plan at t={self.t}")

    def _check_ledger(self):
        live = set(self.queue[0]) | set(self.queue[1])
        for i, s in self.P.items():
            assert i in live, ("protected slot for a vehicle that left", i)
            if np.isnan(self.slot[i]):
                continue   # gated first booking issued, not yet in effect
            assert any(abs(self.slot[i] - y) <= EPS for y in s), ("manager lost a live authorization", i)
        items = [(i, y) for i, s in self.P.items() for y in s]
        for (i, x), (j, y) in itertools.combinations(items, 2):
            if i != j:
                assert self.compatible(i, x, j, y), ("protected slots conflict", i, x, j, y)

    # --------------------------------------------------------------- decisions
    def decide(self, resolve: bool) -> None:
        if self.tx is not None:
            if resolve:
                self.deferred = True
                self.cl["deferred"] += 1
            return
        for (b, a_) in self.policy.pre_locks(self.pub):
            self.add_lock(b, a_, self.policy.name + ":pre")
        live = self.queue[0] + self.queue[1]
        if not live:
            self.incumbent = Plan(order=[], times={}, cost=0.0)
            return
        if not resolve:   # pure tick: locks against what the manager knows, no installation
            self.stats["ticks"] += 1
            if self.incumbent is not None and self.incumbent.order and all(v in self.P for v in live):
                self._policy_locks(self.incumbent, self.build_snapshot())
            return
        # A newcomer gets one chance at a release-gated booking at its optimal time;
        # afterwards it is inserted around every protected slot (FIFO by request).
        stale = [v for v in live if v not in self.P and v in self.offered]
        if stale:
            ins, _ = self._solve(self.manager_snapshot(pin_booked=True))
            for v in sorted(stale):
                self._book(v, ins.times[v])
                self.cl["inserted"] += 1
        # Commit orders against what is booked, before solving, so the target honours them.
        if self.P:
            self._policy_locks(self.known_plan(), self.build_snapshot())
        plan, snap = self._solve(self.manager_snapshot(pin_booked=False))
        if self.cfg.audit:
            errs = validate_plan(snap, plan)
            if errs:
                self.violations.append((self.t, "plan: " + "; ".join(errs[:3])))
        gated, later = {}, set()
        for q in self.queue:
            blocked = False
            for v in q:
                if v in self.P:
                    continue
                if blocked:
                    later.add(v)
                elif all(self.compatible(v, plan.times[v], j, y) for j, ys in self.P.items() for y in ys):
                    self._book(v, plan.times[v])
                else:
                    gated[v] = plan.times[v]
                    blocked = True
        self.offered |= set(gated) | later
        known = self.known()
        target = {i: plan.times[i] for i in known}
        if fsum(known.values()) - fsum(target.values()) <= EPS and not gated:
            self.cl["no_gain"] += 1
            self.incumbent = self.known_plan()
            return
        f = self._fork(snap, known, target, exclude=set(gated) | later)
        fallback = {}
        if gated:
            ins, _ = self._solve(self.manager_snapshot(pin_booked=True))
            fallback = {v: ins.times[v] for v in gated}
        best = None
        for h, T in self._candidates(f, snap, bool(gated)):
            got = self._combined(T, h, gated, fallback)
            if got is not None and (best is None or got[0] > best[0] + EPS):
                best = got
        if best is not None and best[2]:
            self._start_tx(best[1], best[2])
        elif any(v not in self.P for v in live):
            self.deferred = True   # unbooked newcomer: insert at the next decision
        self.incumbent = self.known_plan()

    def _candidates(self, f, snap, gated):
        if self.mode == "direct":
            return [(H.direct(f), f)]
        ok = lambda hh: all(f.compatible(hh.temp[0], hh.temp[1], u, y) for y, _, u in snap.blocks)
        if gated:   # newcomer targets are jointly feasible with the nominal target only
            return [(H.direct(f), f)] + [(h, f) for h in H.detours(f) if ok(h)]
        menu = options(f)
        out = [(H.direct(T, name), T) for name, T in menu.items()]
        for name, T in menu.items():
            if compile_update(T, 0.).depth.keys() != T.changed:
                out += [(h, T) for h in H.detours(T, name + "+detour") if ok(h)]
        return out

    def _combined(self, T, h, gated, fallback):
        """Move graph of handoff h plus release-gated first bookings of newcomers.
        A newcomer holds no slot, so it is never part of a cycle; its own (initial)
        booking is reliable, but it needs every conflicting protected slot retired.
        Returns (expected vehicle-seconds saved, handoff, chosen) or None."""
        moves, parents = dict(h.moves), {e: set(p) for e, p in h.parents.items()}
        for v, Tv in gated.items():
            e = (v, 0)
            moves[e] = H.Move(v, math.inf, Tv, 0)
            ps = set()
            for j, ys in self.P.items():
                if j in gated:
                    continue
                for y in ys:
                    if not self.compatible(v, Tv, j, y):
                        first = (j, 0)
                        if first not in moves or moves[first].a != y:
                            return None   # would need a slot that this handoff does not retire
                        ps.add(first)
            if h.temp is not None and not self.compatible(v, Tv, h.temp[0], h.temp[1]):
                ps.add((h.temp[0], 1))
            parents[e] = ps
        hh = H.Handoff(h.label, moves, parents, h.temp)
        depth, anc = ancestry(parents)
        d, u, w = self.link.d, self.link.u, self.link.v
        el = set()
        for e, k in sorted(depth.items(), key=lambda kv: (kv[1], kv[0])):
            if not parents[e] <= el:
                continue
            if e[0] in gated:
                tau = H.switch_time(self.t, k, d)
                ok = tau < gated[e[0]] - EPS and (self.kin is None or gated[e[0]] >= tau + self.TL - EPS)
            else:
                ok = H.certified(T, hh, depth, e, d)
            if ok:
                el.add(e)
        if not el:
            return 0., hh, set()
        wt = {}
        for e in el:
            m = moves[e]
            if e[0] in gated:
                wt[e] = (fallback[e[0]] - m.b) * (u * w) ** len(anc[e])
            else:
                wt[e] = (m.a - m.b) * u * (u * w) ** len(anc[e])
        chosen = best_closed({e: parents[e] for e in el}, wt)
        return fsum(wt[e] for e in chosen), hh, chosen

    def _policy_locks(self, plan, snap):
        """Commit an order only where every protected slot already respects it, so
        no outcome of a later installation can reverse a lock."""
        pos = {x: k for k, x in enumerate(plan.order)}
        for (b, a_) in self.policy.locks(self.pub, plan, snap):
            lb, la = int(self.lane[b]), int(self.lane[a_])
            ok = (pos[b] < pos[a_] and b in self.P and a_ in self.P and
                  all(y - x >= self.cfg.H[lb][la] - EPS for x in self.P[b] for y in self.P[a_]))
            if ok:
                self.add_lock(b, a_, self.policy.name)
            else:
                self.stats["deferred"] += 1

    def _fork(self, snap, known, target, exclude=()):
        lanes = tuple([replace(j, need=0) for j in q if j.id not in exclude] for q in snap.lanes)
        snapF = replace(snap, lanes=lanes)
        heads = {q[0].id: self.profile[q[0].id] for q in lanes if q} if self.kin is not None else {}
        self.epoch += 1
        extra = {u: {min(self.P[u])} for u in self.uncertain()}
        ids = [j.id for q in lanes for j in q]
        locked = frozenset((b, a) for b in ids for a in ids
                           if self.lane[b] != self.lane[a] and self.is_locked(b, a))
        return LoopFork(snapF, dict(known), dict(target), self.kin, heads, self.epoch, extra=extra, locked=locked)

    def _book(self, v, s):
        for j, ys in self.P.items():
            for y in ys:
                assert self.compatible(v, s, j, y), ("initial booking meets a protected slot", v, s, j, y)
        self.P[v] = {s}
        self.slot[v] = s
        if np.isnan(self.first_slot[v]):
            self.first_slot[v] = s
        if self.kin is not None and self.queue[int(self.lane[v])][0] == v:
            p, w = self.head_state(v)
            self.profile[v] = make_profile(self.t, p, w, s, self.kin)
        self.cl["bookings"] += 1

    # ------------------------------------------------------------ transactions
    def _draw(self, i):
        if i not in self._draws:
            rng = np.random.default_rng(np.random.SeedSequence([*self._key, 9, int(i)]))
            self._draws[i] = rng.random((256, 4))
        k = self.attempt[i]
        self.attempt[i] += 1
        return self._draws[i][k % 256]

    def _start_tx(self, h, chosen):
        depth, _ = ancestry(h.parents)
        t0, d = self.t, self.link.d
        K = max(depth[e] for e in chosen)
        seq = itertools.count()
        heap = []
        for k in range(K + 1):
            heapq.heappush(heap, (t0 + 2 * k * d, k, 0, next(seq), ("send", k)))
        heapq.heappush(heap, (t0 + 2 * (K + 1) * d, K + 1, 0, next(seq), ("end",)))
        self.tx = dict(h=h, chosen=set(chosen), depth=depth, heap=heap, seq=seq, released=set(), fired=set())
        self.cl["transactions"] += 1
        if h.temp is not None and any(e[0] == h.temp[0] for e in chosen):
            self.cl["detours"] += 1

    def next_extra_event(self):
        return self.tx["heap"][0][0] if self.tx is not None and self.tx["heap"] else None

    def on_extra(self, t):
        if self.tx is None:
            return False
        tx = self.tx
        resolve = False
        while tx["heap"] and tx["heap"][0][0] <= t + ETOL:
            _, rnd, _, _, ev = heapq.heappop(tx["heap"])
            if ev[0] == "send":
                self._send(tx, ev[1])
            elif ev[0] == "switch":
                self._switch(tx, ev[1], ev[2])
            elif ev[0] == "ack":
                self._ack(tx, ev[1])
            elif ev[0] == "book":
                mv = tx["h"].moves[ev[1]]
                assert self.compatible is not None and np.isnan(self.slot[mv.i])
                tx["fired"].add(ev[1])
                tx["released"].add(ev[1])
                self.P.pop(mv.i)
                self._book(mv.i, mv.b)
            else:
                resolve = self._end(tx)
                break
            if self.cfg.audit:
                self._check_ledger()
        return resolve

    def _send(self, tx, k):
        h, d, u = tx["h"], self.link.d, self.link.u
        send = self.t
        for e in sorted(e for e in tx["chosen"] if tx["depth"][e] == k):
            if not h.parents[e] <= tx["released"]:
                continue
            mv = h.moves[e]
            i = mv.i
            for j, ys in self.P.items():
                if j != i and not all(self.compatible(i, mv.b, j, y) for y in ys):
                    raise AssertionError("certified move rejected by the authorization ledger")
            if mv.a == math.inf:   # reliable, release-gated first booking of a newcomer
                assert i not in self.P and np.isnan(self.S[i])
                self.P[i] = {mv.b}
                heapq.heappush(tx["heap"], (send + d, k, 2, next(tx["seq"]), ("book", e)))
                self.cl["gated_bookings"] += 1
                continue
            if not np.isnan(self.S[i]) or i not in self.P:
                raise AssertionError("grant for a vehicle that already entered")
            self.P[i].add(mv.b)
            self.cl["grants"] += 1
            draw = self._draw(i)
            if draw[0] < u:
                self.cl["delivered"] += 1
                at = send + d
                heapq.heappush(tx["heap"], (at, k, 2, next(tx["seq"]), ("switch", e, draw)))

    def _switch(self, tx, e, draw):
        mv = tx["h"].moves[e]
        i = mv.i
        now = self.t
        ok = np.isnan(self.S[i]) and abs(self.slot[i] - mv.a) <= EPS and now < min(self.slot[i], mv.b) - EPS
        if ok and self.kin is not None and self.queue[int(self.lane[i])][0] == i:
            p, w = self.head_state(i)
            try:
                self.profile[i] = make_profile(now, p, w, mv.b, self.kin)
            except InfeasibleArrival:
                ok = False
        if not ok:
            raise AssertionError(("claimed physical certificate failed", i, now, mv))
        self.slot[i] = mv.b
        tx["fired"].add(e)
        self.cl["switched"] += 1
        if mv.leg == 1:
            self.cl["returns"] += 1
        elif tx["h"].temp is not None and i == tx["h"].temp[0]:
            self.cl["parks"] += 1
        if draw[1] < self.link.v:
            heapq.heappush(tx["heap"], (now + self.link.d * draw[3], tx["depth"][e], 3, next(tx["seq"]), ("ack", e)))

    def _ack(self, tx, e):
        mv = tx["h"].moves[e]
        tx["released"].add(e)
        self.cl["acks"] += 1
        if mv.i in self.P:
            self.P[mv.i] = {mv.b}

    def _end(self, tx):
        self.tx = None
        h = tx["h"]
        if h.temp is not None:
            k, x = h.temp
            if (k, 0) in tx["fired"] and (k, 1) not in tx["fired"]:
                self.cl["stranded"] += 1
        for i in self.uncertain():
            if i not in self.pinned_since:
                self.pinned_since[i] = self.t
                self.cl["pinned"] += 1
        self.incumbent = self.known_plan()
        live = self.queue[0] + self.queue[1]
        resolve = self.deferred or any(v not in self.P for v in live)
        self.deferred = False
        return resolve

    # ------------------------------------------------------------------ motion
    def _enter(self, vid: int) -> None:
        if vid in self.P:
            if vid in self.pinned_since:
                self.cl["pinned_s"] += self.t - self.pinned_since.pop(vid)
            del self.P[vid]
        super()._enter(vid)
        l = int(self.lane[vid])
        if self.kin is not None and self.queue[l]:
            nxt = self.queue[l][0]
            if not np.isnan(self.slot[nxt]):
                self.profile[nxt] = make_profile(self.t, self.kin.L, 0., float(self.slot[nxt]), self.kin)

    def _record(self):
        rec = super()._record()
        rec.summary.update({f"cl_{k}": v for k, v in self.cl.items()})
        rec.summary["cl_mode"] = self.mode
        return rec
