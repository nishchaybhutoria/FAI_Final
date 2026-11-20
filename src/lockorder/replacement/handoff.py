"""Release-gated handoffs with at most one temporary booking, scored and executed
on the same lossy channel as direct substitutions.

A *move* changes one vehicle's booking from slot ``a`` to slot ``b``. A direct
target is a set of single moves. A detour splits one cycle-blocked vehicle k
into *park* (O_k -> x) and *return* (x -> T_k). Precedences are those of the
archived one-detour test (``bridge.complete``): a move may be issued only after
the moves whose old slots it conflicts with have been acknowledged.

Value (one attempt per move, release-gated, independent legs). A move fires iff
its own grant arrives and every strict ancestor's grant *and* ACK arrive, so
P(e) = u (uv)^{|anc(e)|}. A vehicle's final slot telescopes over its fired moves,
so the expected vehicle-seconds saved by a predecessor-closed set S is

    sum_{e in S} (a_e - b_e) u (uv)^{|anc(e)|},

including the loss of a park whose return never happens. Maximizing this over
closed sets is a maximum-weight closure (``model.best_closed``). A direct target
is the special case with one move per vehicle and reproduces ``targets.score``.

The archived modules (bridge, targets, model, protocol) are imported, not edited.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from ..models.physics import InfeasibleArrival, make_profile, t_minus, t_plus
from ..scheduling.core import Plan, validate_plan
from ..simulation.scenario import SPLITS
from .bridge import _times
from .model import EPS, ancestry, best_closed, compile_update
from .protocol import Drivers, Grant
from .targets import options


@dataclass(frozen=True)
class Move:
    i: int        # vehicle
    a: float      # booking held before the move
    b: float      # booking held after the move
    leg: int      # 0: first move of this vehicle, 1: return from a temporary booking


@dataclass
class Handoff:
    label: str
    moves: dict             # node -> Move; node (i,0) first move, (k,1) return
    parents: dict           # node -> set of nodes that must be acknowledged first
    temp: tuple | None      # (k, x) for a detour, None for a direct target


def direct(f, label="direct"):
    c = compile_update(f, 0.)
    moves = {(i, 0): Move(i, f.old[i], f.new[i], 0) for i in f.changed}
    return Handoff(label, moves, {(i, 0): {(j, 0) for j in ps} for i, ps in c.parents.items()}, None)


def split_graph(f, parents, k, x):
    """Same construction as the archived ``bridge.complete``; None if x conflicts
    with a vehicle that does not change."""
    D = f.changed
    if any(not f.compatible(k, x, j, f.old[j]) for j in f.old.keys() - D):
        return None
    g = {(i, int(i == k)): {(j, 0) for j in ps} for i, ps in parents.items()}
    g[(k, 0)] = set()
    g[(k, 1)].add((k, 0))
    for j in D - {k}:
        if not f.compatible(k, x, j, f.old[j]):
            g[(k, 0)].add((j, 0))
        if not f.compatible(k, x, j, f.new[j]):
            g[(j, 0)].add((k, 1))
    return g


def detours(f, label="detour"):
    """Every cycle-blocked holder k with every critical temporary time x."""
    c = compile_update(f, 0.)
    for k in sorted(f.changed - c.depth.keys()):
        for x in _times(f, k):
            g = split_graph(f, c.parents, k, x)
            if g is None:
                continue
            moves = {(i, 0): Move(i, f.old[i], f.new[i], 0) for i in f.changed if i != k}
            moves[(k, 0)] = Move(k, f.old[k], x, 0)
            moves[(k, 1)] = Move(k, x, f.new[k], 1)
            yield Handoff(f"{label}:{k}@{x:.9g}", moves, g, (k, x))


def switch_time(t0, depth, d, offset=0.):
    return t0 + offset + (2 * depth + 1) * d


def certified(f, h, depth, e, d, offset=0.):
    """Sufficient physical test for move e at its scheduled switch time.

    Generalizes ``Fork.certified`` to a vehicle whose current booking is the
    result of its own earlier move (the return from a temporary booking). Every
    launch of a staged follower compatible with the booking it then held is
    considered; a profile is replanned from the actual state at each prior switch.
    For a direct move this is exactly ``Fork.certified``.
    """
    m = h.moves[e]
    t0 = f.snap.t
    tau = switch_time(t0, depth[e], d, offset)
    if tau >= min(m.a, m.b) - EPS:
        return False
    if f.kin is None:
        return True
    kin = f.kin
    i = m.i
    prior = sorted((switch_time(t0, depth[n], d, offset), h.moves[n].b)
                   for n, mv in h.moves.items() if mv.i == i and mv.leg < m.leg)

    def held(z):   # booking at time z; entries precede message actions at equal times
        s = f.old[i]
        for ts, b in prior:
            if ts < z:
                s = b
        return s

    def replan(prof, start):   # switches at or after launch replan the moving profile
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
                     | set(getattr(f, "extra", {}).get(p, ())))   # other possible use by a pinned booking
            profiles = [replan(make_profile(z, kin.L, 0., held(z), kin), z)
                        for z in sorted(slots) if z <= tau and f.compatible(p, z, i, held(z))]
    except InfeasibleArrival:
        return False
    for prof in profiles:
        dist, v = prof.state(tau)
        if not (dist > 0 and tau + t_minus(dist, v, kin) - EPS <= m.b <= tau + t_plus(dist, v, kin) + EPS):
            return False
    return True   # never launched by tau: the protected same-lane gap covers the launch


def compile_handoff(f, h, d, offset=0.):
    depth, anc = ancestry(h.parents)
    eligible = set()
    for e, _ in sorted(depth.items(), key=lambda kv: (kv[1], kv[0])):
        if h.parents[e] <= eligible and certified(f, h, depth, e, d, offset):
            eligible.add(e)
    return depth, anc, eligible


def weights(h, anc, nodes, link):
    return {e: (h.moves[e].a - h.moves[e].b) * link.u * (link.u * link.v) ** len(anc[e]) for e in nodes}


def value(f, h, link, offset=0.):
    """(expected vehicle-seconds saved, chosen closed set, depths) for handoff h."""
    depth, anc, el = compile_handoff(f, h, link.d, offset)
    if not el:
        return 0., set(), depth
    w = weights(h, anc, el, link)
    chosen = best_closed({e: h.parents[e] for e in el}, w)
    return math.fsum(w[e] for e in chosen), chosen, depth


def eager(f, h, link, offset=0.):
    """All certified moves, with no selection: the loss-oblivious use of a witness."""
    depth, anc, el = compile_handoff(f, h, link.d, offset)
    w = weights(h, anc, el, link)
    return math.fsum(w.values()), set(el), depth


@dataclass
class Choice:
    value: float
    handoff: Handoff
    target: object      # Fork whose ``new`` is the terminal target
    chosen: set


def choose(f, link, detour=True, admissible=None):
    """Best public option: direct menu targets, then detours of any cyclic target.
    Ties keep the earlier option, so the nominal direct target is preferred.
    ``admissible(h)`` may reject a detour (e.g. one whose temporary booking would
    meet a time still protected for a vehicle outside the fork)."""
    menu = options(f)
    cands = [(direct(T, name), T) for name, T in menu.items()]
    if detour:
        for name, T in menu.items():
            if compile_update(T, 0.).depth.keys() != T.changed:
                cands += [(h, T) for h in detours(T, name + "+detour") if admissible is None or admissible(h)]
    best = None
    for h, T in cands:
        v, chosen, _ = value(T, h, link)
        if best is None or v > best.value + EPS:
            best = Choice(v, h, T, chosen)
    return best


def second_tape(key, epoch, n):
    """Independent uniforms for a return leg, keyed like the archived tape but on
    stream 8, so the direct-leg draws (and common random numbers) are unchanged."""
    study, split, rep = key
    rng = np.random.default_rng(np.random.SeedSequence([study, SPLITS[split], rep, 8, epoch]))
    return rng.random((n, 4))


class MoveLedger:
    """Manager evidence per move. A grant is checked against every slot still
    protected for every other vehicle; only a matching ACK retires a slot."""

    def __init__(self, f):
        self.f = f
        self.possible = {i: {s} for i, s in f.old.items()}
        self.issued = {}
        self.released = set()

    def issue(self, e, m, at, epoch):
        if e in self.issued:
            return None
        f = self.f
        if not all(f.compatible(m.i, m.b, j, y) for j, ys in self.possible.items() if j != m.i for y in ys):
            return None
        g = Grant(epoch, m.i, m.a, m.b, at)
        self.issued[e] = g
        self.possible[m.i].add(m.b)
        return g

    def ack(self, e, g):
        if self.issued.get(e) != g or e in self.released:
            return False
        self.possible[g.i] = {g.target}
        self.released.add(e)
        return True


class MultiDrivers(Drivers):
    """Drivers that accept successive versions of their own booking."""

    def receive(self, g, now):
        i = g.i
        if (i not in self.active or self.version[i] != g.epoch - 1 or g.base != self.active[i]
                or i in self.entered or now > g.at + EPS):
            return False
        if i in self.inbox:
            return self.inbox[i] == g
        self.inbox[i] = g
        return True


@dataclass
class Result:
    entries: dict
    fired: dict
    acked: dict
    stranded: bool


def execute(f, h, chosen, link, tape, tape2, offset=0.):
    """Release-gated execution of the chosen closed set of moves on the lossy link.
    Tape columns 0,1 delivery and 2,3 latency fractions, as in ``protocol.execute``;
    a return leg reads ``tape2``. The manager never reads driver state or draws."""
    d, u, v = link.d, link.u, link.v
    t0 = f.snap.t
    depth, _ = ancestry(h.parents)
    m = MoveLedger(f)
    w = MultiDrivers(f)
    fired, acked = {}, {}
    for k in sorted({depth[e] for e in chosen}):
        send = t0 + offset + 2 * k * d
        at = t0 + offset + (2 * k + 1) * d
        w.advance(send)
        events = []
        for e in sorted(e for e in chosen if depth[e] == k and h.parents[e] <= m.released):
            mv = h.moves[e]
            g = m.issue(e, mv, at, f.epoch + mv.leg)
            if g is None:
                raise AssertionError("certified move rejected by the authorization ledger")
            col = tape2[mv.i] if mv.leg else tape[mv.i]
            if col[0] < u:
                heapq.heappush(events, (min(at, send + d * col[2]), 0, e, g))
                heapq.heappush(events, (at, 1, e, g))
        while events:
            now, kind, e, g = heapq.heappop(events)
            w.advance(now)
            mv = h.moves[e]
            col = tape2[mv.i] if mv.leg else tape[mv.i]
            if kind == 0:
                if not w.receive(g, now):
                    raise AssertionError("valid timely reception rejected")
            elif kind == 1:
                if w.switch(g, now) != g:
                    raise AssertionError("claimed physical certificate failed")
                fired[e] = now
                if col[1] < v:
                    heapq.heappush(events, (now + d * col[3], 2, e, g))
            else:
                if not m.ack(e, g):
                    raise AssertionError("matching ACK rejected")
                acked[e] = now
            for i, s in w.active.items():
                if i not in w.entered and s not in m.possible[i]:
                    raise AssertionError("manager lost a live authorization")
    w.advance(math.inf)
    errs = validate_plan(f.snap, Plan(sorted(w.entered, key=w.entered.get), w.entered, 0.))
    if errs:
        raise AssertionError(("executed hybrid failed independent validator", errs))
    stranded = h.temp is not None and abs(w.entered[h.temp[0]] - h.temp[1]) <= EPS
    return Result(dict(w.entered), fired, acked, stranded)
