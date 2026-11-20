"""Two-lane snapshot scheduling: data contract, exact label DP, fixed-order
forward pass, brute-force reference and an independent plan validator.

A snapshot lists, per lane (FIFO by request ID), the jobs still to enter.
Job k on lane l with in-lane index idx must start at
    s_k >= floor_k                                 (mark guard / policy floor)
    s_k >= earliest_k, s_k <= latest_k             (physical window of the head)
    s_k >= s_prev_on_lane + max(H[l][l], TL)       (idx > 0: staged launch)
    s_k >= s_j + H[l_j][l]  for every job j before k, and history entries
and must follow at least ``need_k`` jobs of the other lane (order locks).
Objective: sum of starts + eta * (cross-lane inversions vs a reference order).

Optional *blocks* (time, lane, vehicle id) are times that another, unrevisable
booking may still use: every job except that vehicle must clear each block on
both sides. The default is no blocks, which leaves every result unchanged.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

INF = math.inf
TOL = 1e-7


@dataclass
class Job:
    id: int
    lane: int
    floor: float
    earliest: float = -INF
    latest: float = INF
    need: int = 0           # min number of other-lane snapshot jobs that must precede
    prev_other: int = 0     # other-lane jobs preceding it in the reference order (penalty)
    req_other: int = 0      # other-lane snapshot jobs requested before it (request-order tie-break)


@dataclass
class Snapshot:
    t: float
    lanes: tuple[list[Job], list[Job]]
    H: tuple[tuple[float, float], tuple[float, float]]
    TL: float = 0.0
    last: tuple[float, float] = (-INF, -INF)    # last executed entry per lane (history)
    eta: float = 0.0
    cap: int = 32
    eta_req: float = 0.0    # weight on inversions relative to request order (tie-break only)
    blocks: tuple = ()      # (time, lane, vehicle id): possible use of a slot by an unrevisable booking

    def jobs(self) -> list[Job]:
        return self.lanes[0] + self.lanes[1]


@dataclass
class Plan:
    order: list[int]
    times: dict[int, float]
    cost: float
    approx: bool = False
    labels: int = 0
    source: str = "dp"
    extra: dict = field(default_factory=dict)


def _gap(snap: Snapshot, a: int, b: int) -> float:
    """Required separation after an entry on lane a before an entry on lane b."""
    return max(snap.H[a][a], snap.TL) if a == b else snap.H[a][b]


def _clear_blocks(s: float, job: Job, snap: Snapshot) -> float:
    """Earliest time >= s that clears every block of another vehicle (inf if none).

    Cross-lane blocks may be cleared on either side. On the blocked vehicle's own
    lane FIFO fixes the side: a job ahead of it (smaller request ID) must clear the
    block before it, a job behind it after it. The map is nondecreasing in s, so
    label dominance and the forward pass remain exact.
    """
    moved = True
    while moved:
        moved = False
        for y, lb, vid in snap.blocks:
            if vid == job.id or (lb == job.lane and job.id < vid):
                continue
            g_after = _gap(snap, lb, job.lane)
            g_before = math.inf if lb == job.lane else _gap(snap, job.lane, lb)   # behind: only after
            if y - g_before + 1e-9 < s < y + g_after - 1e-9:
                s = y + g_after
                moved = True
    for y, lb, vid in snap.blocks:
        if lb == job.lane and job.id < vid and s > y - _gap(snap, lb, lb) + 1e-9:
            return math.inf
    return s


def _start(job: Job, idx: int, tA: float, tB: float, snap: Snapshot) -> float:
    l = job.lane
    H = snap.H
    s = max(job.floor, job.earliest, tA + H[0][l], tB + H[1][l])
    if idx > 0:
        tl = tA if l == 0 else tB
        s = max(s, tl + max(H[l][l], snap.TL))
    if snap.blocks:
        s = _clear_blocks(s, job, snap)
    return s


def solve_dp(snap: Snapshot) -> Plan | None:
    """Label-setting DP; exact when no key exceeds the label cap.

    Key (i, j): first i jobs of lane 0 and first j of lane 1 scheduled.  Label
    (cost, tA, tB): accumulated objective and last start per lane.  A label
    dominates another at the same key iff cost, tA and tB are all no larger
    (every completion is monotone in these, and the penalty for future
    appends depends only on the key).
    """
    A, B = snap.lanes
    n0, n1 = len(A), len(B)
    if n0 + n1 == 0:
        return Plan(order=[], times={}, cost=0.0)
    # label: (cost, tA, tB, parent, lane)
    root = (0.0, snap.last[0], snap.last[1], None, -1)
    layer = {(0, 0): [root]}
    approx = False
    total_labels = 1
    for depth in range(n0 + n1):
        nxt: dict[tuple[int, int], list] = {}
        for (i, j), labels in layer.items():
            for lab in labels:
                cost, tA, tB = lab[0], lab[1], lab[2]
                if i < n0 and A[i].need <= j:
                    jb = A[i]
                    s = _start(jb, i, tA, tB, snap)
                    if s <= jb.latest + TOL and s < INF:
                        c = cost + s + snap.eta * max(0, jb.prev_other - j) + snap.eta_req * max(0, jb.req_other - j)
                        nxt.setdefault((i + 1, j), []).append((c, s, tB, lab, 0))
                if j < n1 and B[j].need <= i:
                    jb = B[j]
                    s = _start(jb, j, tA, tB, snap)
                    if s <= jb.latest + TOL and s < INF:
                        c = cost + s + snap.eta * max(0, jb.prev_other - i) + snap.eta_req * max(0, jb.req_other - i)
                        nxt.setdefault((i, j + 1), []).append((c, tA, s, lab, 1))
        for key, labs in nxt.items():
            labs.sort(key=lambda x: (x[0], x[1], x[2]))
            kept = []
            for lab in labs:
                dom = False
                for o in kept:
                    if o[0] <= lab[0] + 1e-12 and o[1] <= lab[1] + 1e-12 and o[2] <= lab[2] + 1e-12:
                        dom = True
                        break
                if not dom:
                    kept.append(lab)
            if len(kept) > snap.cap:
                approx = True
                best_t = min(kept, key=lambda x: (max(x[1], x[2]), x[0]))
                kept = kept[: snap.cap - 1] + ([best_t] if best_t not in kept[: snap.cap - 1] else [kept[snap.cap - 1]])
            nxt[key] = kept
            total_labels += len(kept)
        layer = nxt
        if not layer:
            return None
    final = layer.get((n0, n1))
    if not final:
        return None
    best = min(final, key=lambda x: x[0])
    order, times = [], {}
    lab = best
    idx = [n0, n1]
    while lab[3] is not None:
        ln = lab[4]
        idx[ln] -= 1
        job = (A if ln == 0 else B)[idx[ln]]
        order.append(job.id)
        times[job.id] = lab[1] if ln == 0 else lab[2]
        lab = lab[3]
    order.reverse()
    return Plan(order=order, times=times, cost=best[0], approx=approx, labels=total_labels)


def forward_times(snap: Snapshot, order: Sequence[int]) -> Plan | None:
    """Earliest start times for a fixed lane-respecting order, or None if a
    latest window or a lock is violated."""
    pos = {jb.id: (jb, k) for lane in snap.lanes for k, jb in enumerate(lane)}
    tA, tB = snap.last
    cnt = [0, 0]
    cost = 0.0
    times = {}
    for jid in order:
        jb, idx = pos[jid]
        l = jb.lane
        if idx != cnt[l] or jb.need > cnt[1 - l]:
            return None
        s = _start(jb, idx, tA, tB, snap)
        if s > jb.latest + TOL or s == INF:
            return None
        cost += s + snap.eta * max(0, jb.prev_other - cnt[1 - l]) + snap.eta_req * max(0, jb.req_other - cnt[1 - l])
        times[jid] = s
        if l == 0:
            tA = s
        else:
            tB = s
        cnt[l] += 1
    if cnt[0] != len(snap.lanes[0]) or cnt[1] != len(snap.lanes[1]):
        return None
    return Plan(order=list(order), times=times, cost=cost, source="forward")


# ---------------------------------------------------------------- references

def _independent_eval(snap: Snapshot, order: Sequence[int]) -> float | None:
    """Pairwise evaluator written independently of ``_start``: imposes the
    separation to *every* earlier job, history, windows, launch and locks."""
    lanes = {jb.id: jb.lane for lane in snap.lanes for jb in lane}
    jobs = {jb.id: jb for lane in snap.lanes for jb in lane}
    seq = [jid for jid in order]
    lane_seen = {0: [], 1: []}
    starts: list[tuple[int, float]] = []
    total = 0.0
    for jid in seq:
        jb = jobs[jid]
        l = lanes[jid]
        lane_ids = [x.id for x in snap.lanes[l]]
        if lane_ids.index(jid) != len(lane_seen[l]):
            return None
        if len(lane_seen[1 - l]) < jb.need:
            return None
        s = max(jb.floor, jb.earliest)
        for ln in (0, 1):
            if snap.last[ln] > -INF:
                s = max(s, snap.last[ln] + snap.H[ln][l])
        for (pid, ps) in starts:
            s = max(s, ps + snap.H[lanes[pid]][l])
        if lane_seen[l]:
            s = max(s, lane_seen[l][-1][1] + snap.TL)
        # blocks, written independently of _start: push past each conflicting
        # interval until stable, then require jobs ahead of a blocked vehicle in its
        # own lane to have cleared it beforehand
        same = max(snap.H[l][l], snap.TL)
        while True:
            before = s
            for y, lb, vid in snap.blocks:
                if vid == jid:
                    continue
                if lb == l:
                    if jid > vid and s < y + same - 1e-9:
                        s = y + same
                elif y - snap.H[l][lb] + 1e-9 < s < y + snap.H[lb][l] - 1e-9:
                    s = y + snap.H[lb][l]
            if s == before:
                break
        if any(lb == l and jid < vid and s > y - same + 1e-9 for y, lb, vid in snap.blocks):
            return None
        if s > jb.latest + TOL:
            return None
        # inversions vs reference: other-lane jobs that preceded jid in reference but not yet scheduled
        inv = max(0, jb.prev_other - len(lane_seen[1 - l]))
        invr = max(0, jb.req_other - len(lane_seen[1 - l]))
        total += s + snap.eta * inv + snap.eta_req * invr
        starts.append((jid, s))
        lane_seen[l].append((jid, s))
    return total


def solve_bruteforce(snap: Snapshot) -> tuple[float, list[int]] | None:
    A = [jb.id for jb in snap.lanes[0]]
    B = [jb.id for jb in snap.lanes[1]]
    n = len(A) + len(B)
    best = None
    for posA in itertools.combinations(range(n), len(A)):
        order, ia, ib = [], 0, 0
        setA = set(posA)
        for p in range(n):
            if p in setA:
                order.append(A[ia]); ia += 1
            else:
                order.append(B[ib]); ib += 1
        c = _independent_eval(snap, order)
        if c is not None and (best is None or c < best[0] - 1e-12):
            best = (c, order)
    return best


def validate_plan(snap: Snapshot, plan: Plan, tol: float = 1e-6) -> list[str]:
    """Independent checks of a plan against every snapshot constraint."""
    errs = []
    jobs = {jb.id: jb for lane in snap.lanes for jb in lane}
    if sorted(plan.order) != sorted(jobs):
        errs.append("plan does not cover snapshot jobs exactly")
        return errs
    for l in (0, 1):
        ids = [jb.id for jb in snap.lanes[l]]
        ts = [plan.times[i] for i in ids]
        if [i for i in plan.order if jobs[i].lane == l] != ids:
            errs.append(f"lane {l} FIFO violated")
        for k in range(1, len(ts)):
            if ts[k] < ts[k - 1] + max(snap.H[l][l], snap.TL) - tol:
                errs.append(f"lane {l} launch/following violated at {ids[k]}")
    seq = sorted(plan.order, key=lambda i: plan.times[i])
    if seq != list(plan.order):
        errs.append("plan order inconsistent with its times")
    for x, i in enumerate(seq):
        jb = jobs[i]
        s = plan.times[i]
        if s < jb.floor - tol:
            errs.append(f"{i} before floor")
        if s < jb.earliest - tol or s > jb.latest + tol:
            errs.append(f"{i} outside physical window")
        for ln in (0, 1):
            if s < snap.last[ln] + snap.H[ln][jb.lane] - tol:
                errs.append(f"{i} violates clearance from history lane {ln}")
        for j in seq[:x]:
            if s < plan.times[j] + snap.H[jobs[j].lane][jb.lane] - tol:
                errs.append(f"{i} violates separation after {j}")
        before_other = sum(1 for j in seq[:x] if jobs[j].lane != jb.lane)
        if before_other < jb.need:
            errs.append(f"{i} violates order lock")
        for y, lb, vid in snap.blocks:
            if vid == i:
                continue
            if lb == jb.lane:
                g = max(snap.H[lb][lb], snap.TL)
                ok = (y - s >= g - tol) if i < vid else (s - y >= g - tol)
            else:
                ok = s - y >= snap.H[lb][jb.lane] - tol or y - s >= snap.H[jb.lane][lb] - tol
            if not ok:
                errs.append(f"{i} conflicts with block {y} of {vid}")
    return errs
