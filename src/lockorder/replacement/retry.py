"""Bounded, version-safe retransmission for release-gated reservation updates.

Protocol semantics (docs/retransmission_design.md; audit: docs/retransmission_audit.md)
---------------------------------------------------------------------------------
Rounds of length 2d start at t0 + 2rd. A message sent in round r reaches the
vehicle (if not erased) exactly at tau_r = t0 + (2r+1)d, the switch instant of
that round; a reply sent at tau_r reaches the manager before t0 + 2(r+1)d.

GRANT(gid, i, base, a -> b) attempt r   gid names the update; ``base`` is the
    vehicle version the update applies to. Every retransmission carries the same
    gid, base, a and b; only the switch instant tau_r is new.
QUERY(gid, i, base)                      asks for the vehicle's state; never moves it.
ACK(gid)       vehicle has executed gid (holds b, at version base + 1).
STATUS-VOID(gid)  vehicle holds a at version ``base`` and has voided gid.

Vehicle rules (``Agent``). On a timely GRANT for gid:
  * already executed gid      -> do nothing physically, reply ACK (duplicate)
  * gid void, or version/booking differs from (base, a), or late -> ignore (stale)
  * otherwise                 -> switch to b at tau_r, version += 1, reply ACK.
  On a QUERY: executed -> ACK; at (base, a) -> void gid, reply STATUS-VOID; else ignore.
  A voided gid is never executed afterwards. A version is never executed twice.

Manager rules. A grant's first attempt adds b to P[i] after checking b against
every slot protected for every other vehicle. Only a matching ACK retires a
(P[i] = {b}); only a matching STATUS-VOID, which can follow only a QUERY sent
after the last grant attempt, retires b (P[i] = {a}); observed entry removes
P[i]. Nothing is released on silence or elapsed time. A grant attempt is sent
only if the switch a -> b is physically certified at its own instant tau_r
(re-certified for every attempt), and never after a query for the same gid.

Retry rules (per move, ``budget`` extra transmissions):
  blind   retransmit the grant while it is certifiable; afterwards nothing
  aware   retransmit only if certifiable and the move's own change plus the
          payoff it still unlocks (descendants re-certified at the rounds they
          would now occur, one attempt each) is positive; otherwise QUERY
  oracle  one attempt, with reliable and informative feedback (the manager
          learns after one round whether the switch happened). A comparator
          isolating the cost of ambiguity; not implementable on a lossy link.
With budget 0 and rule 'blind' the executor reproduces ``closed_loop.ClosedLoop``
event for event (tests/test_retry.py, tests/test_closed_loop_golden.py).
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..models.physics import InfeasibleArrival, make_profile, t_minus, t_plus
from ..scheduling.core import Plan, validate_plan
from .closed_loop import ClosedLoop
from .handoff import switch_time
from .model import EPS, ancestry
from .protocol import Drivers

RULES = ("blind", "aware", "oracle")
RT_KEYS = tuple(["retx", "queries", "grant_legs", "query_legs", "replies_sent", "dup_grants", "dup_acks",
                 "stale", "status", "amb_s", "tx_rounds", "tx_extra_rounds"]
                + [f"mv_{s}" for s in ("done", "void", "entered", "dead", "unres_deadline", "unres_budget",
                                       "not_issued")]
                + [f"{k}_s{c}" for k in ("issued", "fail1", "rec") for c in range(4)])


@dataclass
class Agent:
    """Vehicle-side state: the only place that knows what the vehicle executed."""
    ver: int = 0
    executed: dict = field(default_factory=dict)   # gid -> (slot, round)
    void: set = field(default_factory=set)


@dataclass(frozen=True)
class Burst:
    """After an erased leg, the same leg of the next attempt of the same update is
    erased with probability ``psi`` (first attempts keep the marginal law)."""
    psi: float = 0.5


def certified_at(f, h, e, tau, prior):
    """Physical certificate of move e switching at ``tau``.

    ``prior``: (switch instant, slot) of this vehicle's own earlier moves in h
    that have taken effect. With prior instants at the depth rounds this is
    exactly ``handoff.certified``; it is evaluated afresh for every attempt.
    """
    m = h.moves[e]
    if tau >= min(m.a, m.b) - EPS:
        return False
    if f.kin is None:
        return True
    kin, i = f.kin, m.i
    prior = sorted(prior)

    def held(z):
        s = f.old[i]
        for ts, b in prior:
            if ts < z:
                s = b
        return s

    def replan(prof, start):
        for ts, b in prior:
            if start <= ts < tau + EPS:
                p, v = prof.state(ts)
                prof = make_profile(ts, p, v, b, kin)
        return prof

    p = f.pred[i]
    try:
        if p is None:
            profiles = [replan(f.heads[i], -math.inf)]
        else:
            slots = ({f.old[p]} | {mv.b for mv in h.moves.values() if mv.i == p}
                     | set(getattr(f, "extra", {}).get(p, ())))
            profiles = [replan(make_profile(z, kin.L, 0., held(z), kin), z)
                        for z in sorted(slots) if z <= tau and f.compatible(p, z, i, held(z))]
    except InfeasibleArrival:
        return False
    for prof in profiles:
        dist, v = prof.state(tau)
        if not (dist > 0 and tau + t_minus(dist, v, kin) - EPS <= m.b <= tau + t_plus(dist, v, kin) + EPS):
            return False
    return True


class Txn:
    """One serialized installation transaction with bounded retries.

    ``host`` supplies the manager ledger ``P``, ``compatible``, vehicle agents and
    physics (``vslot``, ``entered``, ``vswitch``, ``book``), certification
    (``certify``, ``gated_ok``), packet draws (``draw_first``, ``draw_retry``) and
    counters ``cl``/``rt``. Manager logic reads only replies that reached it,
    observed entries and its own ledger; vehicle logic reads only ``Agent``
    state and the physical booking. ``fired`` is ground truth for statistics.
    """

    def __init__(self, host, h, chosen, T, t0, link, budget=0, rule="blind"):
        if rule not in RULES or budget < 0 or (rule == "oracle" and budget):
            raise ValueError((rule, budget))
        self.host, self.h, self.T, self.t0, self.link = host, h, T, t0, link
        self.budget, self.rule = budget, rule
        self.chosen = set(chosen)
        self.depth, self.anc = ancestry(h.parents)
        self.desc = defaultdict(set)
        for c, a in self.anc.items():
            for e in a:
                self.desc[e].add(c)
        self.order = sorted(self.chosen, key=lambda e: (self.depth[e], e))
        self.Kd = max(self.depth[e] for e in self.chosen)
        self.state = {e: "wait" for e in self.chosen}
        self.gid, self.base, self.used = {}, {}, defaultdict(int)
        self.queried, self.stuck, self.fired = set(), set(), set()
        self.known_round = {}            # manager: round in which a confirmed move executed (from its ACK)
        self.first_round, self.slack, self.resolved_by = {}, {}, {}
        self.heap, self.seq = [], itertools.count()
        self.last_sent = -1
        self.sent_now = set()
        self._push(t0, 0, 0, ("round", 0))

    # -------------------------------------------------------------- plumbing
    def _push(self, t, rnd, prio, ev):
        heapq.heappush(self.heap, (t, rnd, prio, next(self.seq), ev))

    def next_time(self):
        return self.heap[0][0] if self.heap else None

    def tau(self, r):
        return switch_time(self.t0, r, self.link.d)

    def step(self, t, tol=1e-9):
        """Process one due event. Returns None (nothing due), 'end' or 'ok'."""
        if not self.heap or self.heap[0][0] > t + tol:
            return None
        _, rnd, _, _, ev = heapq.heappop(self.heap)
        kind = ev[0]
        if kind == "round":
            self._round(ev[1], t)
        elif kind == "deliver":
            self._deliver(t, rnd, *ev[1:])
        elif kind == "reply":
            self._reply(t, *ev[1:])
        elif kind == "book":
            e = ev[1]
            mv = self.h.moves[e]
            self.state[e] = "done"
            self.fired.add(e)
            self.host.P.pop(mv.i)
            self.host.book(mv.i, mv.b)
        else:
            return "end"
        return "ok"

    def prior(self, e):
        """Own earlier legs of this vehicle known (by ACK) to have taken effect."""
        i, leg = self.h.moves[e].i, self.h.moves[e].leg
        return [(self.tau(self.known_round[n]), mv.b) for n, mv in self.h.moves.items()
                if mv.i == i and mv.leg < leg and n in self.known_round]

    def _cert(self, e, r, prior=None):
        mv = self.h.moves[e]
        if mv.a == math.inf:
            return self.host.gated_ok(mv.i, mv.b, self.tau(r))
        return self.host.certify(self.T, self.h, e, self.tau(r), self.prior(e) if prior is None else prior)

    # ---------------------------------------------------------------- rounds
    def _round(self, r, now):
        host, h, d = self.host, self.h, self.link.d
        self.sent_now = set()
        done = {e for e, s in self.state.items() if s == "done"}
        for e in self.order:
            st, mv = self.state[e], h.moves[e]
            i = mv.i
            if st == "wait":
                if not h.parents[e] <= done:
                    continue
                if not self._cert(e, r):
                    if r == self.depth[e]:
                        raise AssertionError(("decision-time certificate not reproduced", e, r))
                    self.state[e] = "dead"     # its prerequisites confirmed too late to switch
                    continue
                if self.rule == "aware" and r > self.depth[e] and mv.a != math.inf and not self._worth(e, r):
                    self.state[e] = "dead"     # issued late, a sacrifice whose payoff no longer fits
                    continue
                for j, ys in host.P.items():
                    if j != i and not all(host.compatible(i, mv.b, j, y) for y in ys):
                        raise AssertionError("certified move rejected by the authorization ledger")
                self.sent_now.add(e)
                if mv.a == math.inf:   # reliable release-gated first booking of a newcomer
                    assert i not in host.P and np.isnan(host.vslot(i))
                    host.P[i] = {mv.b}
                    self._push(now + d, r, 2, ("book", e))
                    host.cl["gated_bookings"] += 1
                    self.state[e] = "booked"
                    continue
                if host.entered(i) or i not in host.P:
                    raise AssertionError("grant for a vehicle that already entered")
                self.gid[e] = host.new_gid()
                self.base[e] = host.mver[i]
                host.P[i].add(mv.b)
                host.ambig(i, now)
                host.cl["grants"] += 1
                self.state[e] = "out"
                self.first_round[e] = r
                self.slack[e] = self._slack(e, r)
                host.rt[f"issued_s{self.slack[e]}"] += 1
                self._transmit(e, r, now, host.draw_first(i), "grant", 0)
            elif st == "out" and e not in self.stuck and self.used[e] < self.budget:
                cert = e not in self.queried and self._cert(e, r)
                if self.rule == "blind":
                    if not cert:
                        self.stuck.add(e)      # no certifiable switch instant remains
                        continue
                    act = "grant"
                else:   # aware
                    act = "grant" if cert and self._worth(e, r) else "query"
                if act == "query":
                    self.queried.add(e)
                self.used[e] += 1
                self.sent_now.add(e)
                host.rt["retx" if act == "grant" else "queries"] += 1
                self._transmit(e, r, now, host.draw_retry(i, self, e), act, self.used[e])
        if self.sent_now:
            self.last_sent = r
        if r < self.Kd or self._alive(r):
            self._push(self.t0 + 2 * (r + 1) * d, r + 1, 0, ("round", r + 1))
        else:
            end = max(self.t0 + 2 * (self.Kd + 1) * d, self.t0 + 2 * (self.last_sent + 1) * d)
            self._push(max(end, now), max(self.Kd, self.last_sent) + 1, 0, ("end",))

    def _can_retry(self, e, r):
        if self.state[e] != "out" or e in self.stuck or self.used[e] >= self.budget:
            return False
        if self.host.entered(self.h.moves[e].i):
            return False
        # blind: only a certifiable retransmission can be sent; aware can always query
        return self.rule == "aware" or (e not in self.queried and self._cert(e, r + 1))

    def _alive(self, r):
        """Could anything be sent in round r + 1?"""
        if self.budget == 0:
            return False
        memo = {}

        def alive(e):     # may still be (or become) confirmed
            if e not in memo:
                s = self.state[e]
                if s in ("done", "booked"):
                    z = True
                elif s == "out":
                    z = e in self.sent_now or self._can_retry(e, r)
                elif s == "wait":
                    z = all(alive(p) for p in self.h.parents[e])
                else:
                    z = False
                memo[e] = z
            return memo[e]

        return any(self._can_retry(e, r) or (self.state[e] == "wait" and all(alive(p) for p in self.h.parents[e]))
                   for e in self.order)

    def _slack(self, e, r, cap=3):
        """Further rounds (0..cap) in which a retransmission would be certifiable."""
        k = 0
        while k < cap and self._cert(e, r + k + 1):
            k += 1
        return k

    def _worth(self, e, r):
        """Aware rule: own change plus the payoff it still unlocks if it takes effect in round r,
        each pending descendant re-certified at the round it would now occur, one attempt each."""
        mv = self.h.moves[e]
        own = mv.a - mv.b
        if own > EPS:
            return True
        u, v = self.link.u, self.link.v
        gain = own
        for c in self.desc[e]:
            if c not in self.chosen or self.state[c] != "wait":
                continue
            k = self.depth[c] - self.depth[e]
            mc = self.h.moves[c]
            prior = self.prior(c)
            if mc.i == mv.i:      # the vehicle's own later leg (a detour return)
                prior = prior + [(self.tau(r), mv.b)]
            if self._cert(c, r + k, prior):
                a = getattr(self.h, "fallback", {}).get(mc.i, mc.a) if mc.a == math.inf else mc.a
                gain += (a - mc.b) * v * u * (u * v) ** (k - 1)
        return gain > EPS

    # -------------------------------------------------------------- messages
    def _transmit(self, e, r, now, draw, kind, attempt):
        host, mv, g = self.host, self.h.moves[e], self.gid[e]
        host.rt[f"{kind}_legs"] += 1
        if draw[0] < self.link.u:
            host.cl["delivered"] += kind == "grant"
            host.legs[(g, "down")] = True
            self._push(now + self.link.d, r, 2, ("deliver", e, kind, draw, attempt))
            return
        host.legs[(g, "down")], host.legs[(g, "up")] = False, None
        if self.rule == "oracle":   # informative feedback: the manager learns of the erasure
            host.agent(mv.i).void.add(g)   # the grant can no longer arrive (bounded delay)
            self._push(now + self.link.d + self.link.d * draw[3], r, 3, ("reply", e, "void", g, attempt, None))

    def _deliver(self, now, r, e, kind, draw, attempt):
        host, mv = self.host, self.h.moves[e]
        i, g, base = mv.i, self.gid[e], self.base[e]
        host.legs[(g, "up")] = None
        if host.entered(i):
            host.rt["stale"] += 1
            return
        ag = host.agent(i)
        reply = None
        if kind == "grant":
            if g in ag.executed:
                host.rt["dup_grants"] += 1
                reply = "ack"
            elif g in ag.void or ag.ver != base or abs(host.vslot(i) - mv.a) > EPS or now > self.tau(r) + EPS:
                if attempt == 0:
                    raise AssertionError(("first attempt rejected", e))
                host.rt["stale"] += 1
            else:
                host.vswitch(i, mv.b, now)
                ag.ver += 1
                ag.executed[g] = (mv.b, r)
                self.fired.add(e)
                host.cl["switched"] += 1
                if mv.leg == 1:
                    host.cl["returns"] += 1
                elif self.h.temp is not None and i == self.h.temp[0]:
                    host.cl["parks"] += 1
                reply = "ack"
        else:   # query
            if g in ag.executed:
                reply = "ack"
            elif ag.ver == base and abs(host.vslot(i) - mv.a) <= EPS:
                ag.void.add(g)
                reply = "void"
            else:
                host.rt["stale"] += 1
        if reply is None:
            return
        host.rt["replies_sent"] += 1
        executed_round = ag.executed[g][1] if reply == "ack" else None   # carried by the ACK
        if self.rule == "oracle" or draw[1] < self.link.v:
            host.legs[(g, "up")] = True
            self._push(now + self.link.d * draw[3], r, 3, ("reply", e, reply, g, attempt, executed_round))
        else:
            host.legs[(g, "up")] = False

    def _reply(self, now, e, kind, g, attempt, executed_round):
        host, mv = self.host, self.h.moves[e]
        i = mv.i
        if self.gid.get(e) != g:
            host.rt["stale"] += 1
            return
        if kind == "ack":
            host.cl["acks"] += 1
            if self.state[e] != "out":
                host.rt["dup_acks"] += 1
                return
            self.state[e] = "done"
            self.resolved_by[e] = attempt
            host.mver[i] = self.base[e] + 1
            self.known_round[e] = executed_round
            if i in host.P:
                host.P[i] = {mv.b}
                host.ambig(i, now)
        else:
            host.rt["status"] += 1
            if self.state[e] != "out":
                return
            if not (e in self.queried or self.rule == "oracle"):
                raise AssertionError("void status without a query")
            self.state[e] = "void"
            self.resolved_by[e] = attempt
            if i in host.P:
                host.P[i] = {mv.a}
                host.ambig(i, now)

    def on_enter(self, i):
        for e, mv in self.h.moves.items():
            if e in self.chosen and mv.i == i and self.state[e] == "out":
                self.state[e] = "entered"

    def check_knowledge(self):
        """Manager beliefs must be true: a confirmed update was executed, a void one was not."""
        for e, s in self.state.items():
            g = self.gid.get(e)
            if g is None:
                continue
            ag = self.host.agent(self.h.moves[e].i)
            if s == "done":
                assert g in ag.executed, ("manager believes an unexecuted update", e)
            elif s == "void":
                assert g in ag.void and g not in ag.executed, ("unsound void", e)

    # --------------------------------------------------------------- outcome
    def outcome(self):
        """Counts over issued and never-issued non-newcomer moves."""
        out = defaultdict(int)
        for e in self.chosen:
            if self.h.moves[e].a == math.inf:
                continue
            s = self.state[e]
            if s == "out":
                s = "unres_deadline" if (e in self.stuck or self.used[e] < self.budget) else "unres_budget"
            elif s == "wait":
                s = "not_issued"
            out["mv_" + s] += 1
            if e in self.first_round:
                first_ok = s == "done" and self.resolved_by.get(e) == 0
                if not first_ok:
                    k = self.slack[e]
                    out[f"fail1_s{k}"] += 1
                    if self.resolved_by.get(e, 0) >= 1:
                        out[f"rec_s{k}"] += 1
        return out


class _Host:
    """Shared host plumbing (agents, versions, gids, draws, ambiguity clock)."""

    def _init_host(self):
        self.agents = defaultdict(Agent)
        self.mver = defaultdict(int)
        self._gids = itertools.count(1)
        self.rt = defaultdict(float, {k: 0. for k in RT_KEYS})
        self.amb_since = {}
        self.legs = {}

    def new_gid(self):
        return next(self._gids)

    def agent(self, i):
        return self.agents[i]

    def ambig(self, i, now):
        amb = i in self.P and len(self.P[i]) > 1
        if amb and i not in self.amb_since:
            self.amb_since[i] = now
        elif not amb and i in self.amb_since:
            self.rt["amb_s"] += now - self.amb_since.pop(i)



class RetryLoop(_Host, ClosedLoop):
    """Closed loop whose transactions use ``Txn`` (bounded retries)."""

    def __init__(self, sc, policy, link, mode="direct", budget=0, rule="blind", burst=None):
        super().__init__(sc, policy, link, mode)
        self._init_host()
        self.budget, self.rule, self.burst = budget, rule, burst
        self.label = f"{rule}{budget}"
        self._rdraws, self.rattempt = {}, defaultdict(int)
        if rule not in RULES or (rule == "oracle" and budget):
            raise ValueError((rule, budget))

    # ---------------------------------------------------------------- host API
    def vslot(self, i):
        return float(self.slot[i])

    def entered(self, i):
        return not np.isnan(self.S[i])

    def vswitch(self, i, b, now):
        ok = np.isnan(self.S[i]) and now < min(self.slot[i], b) - EPS
        if ok and self.kin is not None and self.queue[int(self.lane[i])][0] == i:
            p, w = self.head_state(i)
            try:
                self.profile[i] = make_profile(now, p, w, b, self.kin)
            except InfeasibleArrival:
                ok = False
        if not ok:
            raise AssertionError(("claimed physical certificate failed", i, now, b))
        self.slot[i] = b

    def certify(self, T, h, e, tau, prior):
        return certified_at(T, h, e, tau, prior)

    def gated_ok(self, v, Tv, tau):
        return tau < Tv - EPS and (self.kin is None or Tv >= tau + self.TL - EPS)

    def book(self, v, b):
        self._book(v, b)

    def draw_first(self, i):
        return self._draw(i)

    def draw_retry(self, i, txn, e):
        if i not in self._rdraws:
            rng = np.random.default_rng(np.random.SeedSequence([*self._key, 10, int(i)]))
            self._rdraws[i] = rng.random((256, 4))
        k = self.rattempt[i]
        self.rattempt[i] += 1
        raw = self._rdraws[i][k % 256]
        z = raw.copy()
        if self.burst is not None:
            # Correlated erasures: a leg erased in the previous attempt of this update
            # is erased again with probability psi (success iff raw >= psi).
            for col, leg in ((0, "down"), (1, "up")):
                if self.legs.get((txn.gid[e], leg)) is False:
                    z[col] = 0. if raw[col] >= self.burst.psi else 1.
        return z

    # ------------------------------------------------------------ transactions
    def _combined(self, T, h, gated, fallback):
        got = super()._combined(T, h, gated, fallback)
        if got is not None:
            got[1].fork, got[1].fallback = T, dict(fallback)
        return got

    def _start_tx(self, h, chosen):
        self.tx = Txn(self, h, chosen, h.fork, self.t, self.link, self.budget, self.rule)
        self.cl["transactions"] += 1
        if h.temp is not None and any(e[0] == h.temp[0] for e in chosen):
            self.cl["detours"] += 1

    def next_extra_event(self):
        return self.tx.next_time() if self.tx is not None else None

    def on_extra(self, t):
        if self.tx is None:
            return False
        resolve = False
        while (r := self.tx.step(t)) is not None:
            if r == "end":
                resolve = self._end_txn(self.tx)
                break
            if self.cfg.audit:
                self._check_ledger()
        return resolve

    def _end_txn(self, tx):
        h = tx.h
        if h.temp is not None:
            k, _ = h.temp
            if (k, 0) in tx.fired and (k, 1) not in tx.fired:
                self.cl["stranded"] += 1
        for key, n in tx.outcome().items():
            self.rt[key] += n
        self.rt["tx_rounds"] += max(tx.Kd, tx.last_sent) + 1
        self.rt["tx_extra_rounds"] += max(0, tx.last_sent - tx.Kd)
        self.tx = None
        for i in self.uncertain():
            if i not in self.pinned_since:
                self.pinned_since[i] = self.t
                self.cl["pinned"] += 1
        self.incumbent = self.known_plan()
        live = self.queue[0] + self.queue[1]
        resolve = self.deferred or any(v not in self.P for v in live)
        self.deferred = False
        return resolve

    def decide(self, resolve):
        super().decide(resolve)
        if self.cfg.audit and self.tx is None:
            self._check_ledger()

    def _enter(self, vid):
        if self.tx is not None:
            self.tx.on_enter(vid)
        if vid in self.amb_since:
            self.rt["amb_s"] += self.t - self.amb_since.pop(vid)
        super()._enter(vid)
        if self.cfg.audit:
            self._check_ledger()

    def _check_ledger(self):
        super()._check_ledger()
        if self.tx is not None:
            self.tx.check_knowledge()

    def _record(self):
        rec = super()._record()
        rec.summary.update({f"rt_{k}": v for k, v in self.rt.items()})
        rec.summary["rt_rule"] = self.label
        return rec


class ForkHost(_Host):
    """Runs one ``Txn`` on an isolated fork (no arrivals, no further decisions).

    ``draws(i, k)`` returns the 4 uniforms of vehicle i's k-th transmission
    (first attempts and retries share one per-vehicle sequence here).
    Physics, entries and the final independent validation reuse
    ``protocol.Drivers``.
    """

    def __init__(self, f, link, draws):
        self._init_host()
        self.f, self.link, self.draws = f, link, draws
        self.w = Drivers(f)
        self.P = {i: {s} for i, s in f.old.items()}
        self.cl = defaultdict(int)
        self.k = defaultdict(int)
        self.txn = None

    def compatible(self, i, x, j, y):
        return self.f.compatible(i, x, j, y)

    def vslot(self, i):
        return self.w.active[i]

    def entered(self, i):
        return i in self.w.entered

    def vswitch(self, i, b, now):
        if i in self.w.entered or now >= min(self.w.active[i], b) - EPS:
            raise AssertionError(("claimed physical certificate failed", i, now, b))
        try:
            self.w._retime(i, b, now)
        except InfeasibleArrival:
            raise AssertionError(("claimed physical certificate failed", i, now, b))

    def certify(self, T, h, e, tau, prior):
        return certified_at(T, h, e, tau, prior)

    def gated_ok(self, v, Tv, tau):
        raise AssertionError("no newcomers in an isolated fork")

    def book(self, v, b):
        raise AssertionError("no newcomers in an isolated fork")

    def draw_first(self, i):
        z = self.draws(i, self.k[i])
        self.k[i] += 1
        return z

    def draw_retry(self, i, txn, e):
        return self.draw_first(i)

    def advance(self, t):
        before = set(self.w.entered)
        self.w.advance(t)
        for i in sorted(set(self.w.entered) - before):
            if self.txn is not None:
                self.txn.on_enter(i)
            if i in self.amb_since:
                self.rt["amb_s"] += self.w.entered[i] - self.amb_since.pop(i)
            self.P.pop(i, None)

    def check(self):
        if self.txn is not None:
            self.txn.check_knowledge()
        for i, s in self.P.items():
            assert any(abs(self.w.active[i] - y) <= EPS for y in s), ("manager lost a live authorization", i)
        items = [(i, y) for i, s in self.P.items() for y in s]
        for (i, x), (j, y) in itertools.combinations(items, 2):
            if i != j:
                assert self.f.compatible(i, x, j, y), ("protected slots conflict", i, x, j, y)


@dataclass
class ForkResult:
    entries: dict      # actual entry times
    state: dict        # move states when the transaction ended
    P: dict            # manager's protected sets when the transaction ended
    txn: Txn
    host: ForkHost


def run_fork(f, h, chosen, link, budget=0, rule="blind", draws=None):
    """Execute ``chosen`` moves of handoff h on fork f with the retry protocol."""
    host = ForkHost(f, link, draws)
    if not chosen:
        host.advance(math.inf)
        return ForkResult(dict(host.w.entered), {}, {}, None, host)
    txn = Txn(host, h, chosen, f, f.snap.t, link, budget, rule)
    host.txn = txn
    while (t := txn.next_time()) is not None:
        host.advance(t)
        r = txn.step(t)
        if r == "end":
            break
        host.check()
    state, P_end = dict(txn.state), {i: set(s) for i, s in host.P.items()}   # at the end of the transaction
    host.txn = None
    host.advance(math.inf)
    errs = validate_plan(f.snap, Plan(sorted(host.w.entered, key=host.w.entered.get), host.w.entered, 0.))
    if errs:
        raise AssertionError(("executed hybrid failed independent validator", errs))
    return ForkResult(dict(host.w.entered), state, P_end, txn, host)
