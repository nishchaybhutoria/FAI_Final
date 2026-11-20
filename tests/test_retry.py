"""Bounded retransmission: reproduction of the one-attempt baseline, safety under
every message history, the transaction-level formulas, and each message case
listed in docs/retransmission_design.md."""
from itertools import product

import numpy as np
import pytest

from lockorder.evaluation.runner import make_policy
from lockorder.models.physics import launch_time, make_profile
from lockorder.replacement import Channel, Fork
from lockorder.replacement import handoff as H
from lockorder.replacement import retry_value as RV
from lockorder.replacement.closed_loop import ClosedLoop
from lockorder.replacement.retry import Burst, RetryLoop, Txn, certified_at, run_fork
from lockorder.scheduling.core import Job, Snapshot
from lockorder.simulation.scenario import SimConfig, make_scenario

CFG = SimConfig(rho=0.75, sigma=1.0, t_arr_end=240.0, warmup=40.0, win_end=200.0, audit=True)


# ----------------------------------------------------------------- closed loop
@pytest.mark.parametrize("rep", [0, 1, 2])
@pytest.mark.parametrize("arm", [("rolling", "direct"), ("rolling", "handoff"), ("freeze_inf", "direct"),
                                 ("adaptive", "direct")])
def test_zero_budget_reproduces_the_one_attempt_loop_exactly(rep, arm):
    sc = make_scenario(CFG, "pilot", rep, study=1111)
    for d, p in [(0.1, 1.0), (0.1, 0.9), (0.05, 0.8), (0.25, 0.9)]:
        a = ClosedLoop(sc, make_policy(arm[0], CFG), Channel(d, p, p), arm[1]).run()
        b = RetryLoop(sc, make_policy(arm[0], CFG), Channel(d, p, p), arm[1], budget=0).run()
        assert np.array_equal(a.vehicles["S"], b.vehicles["S"])
        assert {k: v for k, v in a.summary.items() if k.startswith("cl_")} == \
               {k: v for k, v in b.summary.items() if k.startswith("cl_")}
        assert b.summary["rt_retx"] == b.summary["rt_queries"] == 0


RULES = [(1, "blind", None), (2, "blind", None), (2, "aware", None), (0, "oracle", None), (1, "blind", Burst(.5))]


@pytest.mark.parametrize("rep", [0, 1])
@pytest.mark.parametrize("link", [Channel(0.1, 0.9, 0.9), Channel(0.25, 0.7, 0.7), Channel(0.05, 0.8, 0.95)],
                         ids=lambda c: f"d{c.d}-u{c.u}-v{c.v}")
def test_every_retry_rule_is_safe_sound_and_drains(rep, link):
    """Audit mode checks (I1), (I2) and manager-knowledge soundness after every protocol
    event, booking and entry; the engine checks marks, clearances, FIFO, launch, locks."""
    sc = make_scenario(CFG, "pilot", rep, study=1111)
    for pol, mode in [("rolling", "direct"), ("rolling", "handoff"), ("freeze_inf", "direct")]:
        for K, rule, burst in RULES:
            r = RetryLoop(sc, make_policy(pol, CFG), link, mode, budget=K, rule=rule, burst=burst).run()
            assert r.violations == [], (pol, mode, rule, K, r.violations[:3])
            assert r.summary["drained"] and r.summary["cl_lock_relaxed"] == 0
            s = r.summary
            assert s["rt_retx"] + s["rt_queries"] <= K * s["cl_grants"]
            if rule == "oracle":
                assert s["cl_pinned"] == 0


def test_perfect_link_never_retransmits_and_every_rule_coincides():
    sc = make_scenario(CFG, "pilot", 0, study=1111)
    base = RetryLoop(sc, make_policy("rolling", CFG), Channel(0.1, 1., 1.), "direct").run()
    for K, rule, burst in RULES:
        r = RetryLoop(sc, make_policy("rolling", CFG), Channel(0.1, 1., 1.), "direct", K, rule, burst).run()
        assert np.array_equal(r.vehicles["S"], base.vehicles["S"])
        assert r.summary["rt_retx"] == r.summary["rt_queries"] == r.summary["cl_pinned"] == 0


def test_one_retry_resolves_most_ambiguity_and_is_deterministic():
    sc = make_scenario(CFG, "pilot", 0, study=1111)
    link = Channel(0.1, 0.9, 0.9)
    k0 = RetryLoop(sc, make_policy("rolling", CFG), link, "direct", 0).run().summary
    k1a = RetryLoop(sc, make_policy("rolling", CFG), link, "direct", 1).run()
    k1b = RetryLoop(sc, make_policy("rolling", CFG), link, "direct", 1).run()
    assert np.array_equal(k1a.vehicles["S"], k1b.vehicles["S"])
    assert k1a.summary["cl_pinned"] < k0["cl_pinned"]
    assert k1a.summary["rt_retx"] > 0 and k1a.summary["rt_dup_grants"] > 0   # lost ACKs were re-acknowledged


# --------------------------------------------------------------------- forks
def pair(O=(10., 14.), N=(12., 9.), floors=(10., 9.), lanes=(0, 1), h=3.):
    q = ([], [])
    for i, l in enumerate(lanes):
        q[l].append(Job(i, l, floors[i], earliest=0., latest=np.inf))
    return Fork(Snapshot(0., q, ((h, h), (h, h)), 0.), dict(enumerate(O)), dict(enumerate(N)), None, {}, 4)


def table(bits):
    """bits[i] = [(grant leg ok, reply leg ok), ...] per transmission to vehicle i."""
    def draws(i, k):
        g, a = bits[i][k]
        return np.array([0. if g else .999, 0. if a else .999, .5, .5])
    return draws


def detour(f):
    return next(h for h in H.detours(f) if h.temp == (1, 15.))


def direct_B():
    f = pair(N=(10., 13.))        # best direct target of the tight pair: B alone to a + h
    return f, H.direct(f)


def run(f, h, link, K, rule, bits):
    return run_fork(f, h, set(h.moves), link, K, rule, table(bits))


OK, LOST_GRANT, LOST_ACK = (True, True), (False, True), (True, False)


def test_lost_initial_grant_is_recovered_by_one_retry():
    f, h = direct_B()
    r = run(f, h, Channel(.1, .9, .9), 1, "blind", {1: [LOST_GRANT, OK]})
    assert r.entries[1] == 13. and r.state[(1, 0)] == "done" and r.P[1] == {13.}
    assert r.host.cl["switched"] == 1 and r.host.rt["retx"] == 1 and r.host.rt["dup_grants"] == 0


def test_lost_ack_is_recovered_by_a_duplicate_grant_without_a_second_switch():
    f, h = direct_B()
    r = run(f, h, Channel(.1, .9, .9), 1, "blind", {1: [LOST_ACK, OK]})
    assert r.entries[1] == 13. and r.state[(1, 0)] == "done"
    assert r.host.cl["switched"] == 1 and r.host.rt["dup_grants"] == 1   # executed once, re-acknowledged


def test_budget_exhaustion_leaves_both_slots_protected():
    f, h = direct_B()
    for bits, entry in [([LOST_GRANT, LOST_GRANT], 14.), ([LOST_ACK, LOST_ACK], 13.), ([LOST_GRANT, LOST_ACK], 13.)]:
        r = run(f, h, Channel(.1, .9, .9), 1, "blind", {1: bits})
        assert r.entries[1] == entry
        assert r.state[(1, 0)] == "out" and r.P[1] == {14., 13.}   # unresolved: both slots stay protected
        o = r.txn.outcome()
        assert o["fail1_s3"] == 1 and o.get("rec_s3", 0) == 0 and o["mv_unres_budget"] == 1


def test_no_release_on_silence_ledger_keeps_both_slots_until_entry():
    f, h = direct_B()
    r = run(f, h, Channel(.1, .9, .9), 0, "blind", {1: [LOST_ACK]})
    assert r.P[1] == {14., 13.} and r.entries[1] == 13.


def test_duplicate_and_stale_replies_are_idempotent():
    f, h = direct_B()
    link = Channel(.1, .9, .9)
    from lockorder.replacement.retry import ForkHost
    hst = ForkHost(f, link, table({1: [OK]}))
    txn = Txn(hst, h, set(h.moves), f, 0., link, 1, "blind")
    hst.txn = txn
    while (t := txn.next_time()) is not None:
        hst.advance(t)
        if txn.step(t) == "end":
            break
    e = (1, 0)
    g = txn.gid[e]
    assert txn.state[e] == "done" and hst.P[1] == {13.}
    txn._reply(1., e, "ack", g, 1, 0)                 # duplicate ACK (e.g. from a retry)
    assert txn.state[e] == "done" and hst.P[1] == {13.} and hst.rt["dup_acks"] == 1
    txn._reply(1., e, "ack", g + 99, 0, 0)            # ACK naming another update
    txn._reply(1., e, "void", g, 0, None)             # status after confirmation
    assert txn.state[e] == "done" and hst.P[1] == {13.} and hst.rt["stale"] == 1


def test_stale_grant_after_a_newer_version_is_rejected():
    f, h = direct_B()
    link = Channel(.1, .9, .9)
    from lockorder.replacement.retry import ForkHost
    hst = ForkHost(f, link, table({1: [OK, OK]}))
    txn = Txn(hst, h, set(h.moves), f, 0., link, 1, "blind")
    hst.txn = txn
    while (t := txn.next_time()) is not None:
        hst.advance(t)
        if txn.step(t) == "end":
            break
    ag = hst.agent(1)
    ag.ver += 1                           # the vehicle has since executed a newer update
    hst.w.active[1] = 11.
    txn.gid[(1, 0)] = 10**6               # an old grant id, not executed by this vehicle
    txn._deliver(.25, 1, (1, 0), "grant", np.array([0., 0., .5, .5]), 1)
    assert hst.w.active[1] == 11. and hst.rt["stale"] == 1 and ag.ver == 2 and 10**6 not in ag.executed


def test_physical_deadline_during_retry_blind_gives_up_aware_queries():
    f = pair(O=(10., 20.), N=(10., 13.))
    h = H.direct(f)
    link = Channel(4.5, .9, .9)           # tau_0 = 4.5 < 13 <= tau_1 = 13.5: no second switch instant
    blind = run(f, h, link, 1, "blind", {1: [LOST_GRANT, OK]})
    assert blind.state[(1, 0)] == "out" and blind.host.rt["retx"] == 0 and blind.P[1] == {20., 13.}
    o = blind.txn.outcome()
    assert o["fail1_s0"] == 1 and o["mv_unres_deadline"] == 1
    aware = run(f, h, link, 1, "aware", {1: [LOST_GRANT, OK]})
    # the query reaches the vehicle at 13.5, while it still holds 20: it voids the grant
    assert aware.state[(1, 0)] == "void" and aware.host.rt["queries"] == 1 and aware.P[1] == {20.}
    assert aware.entries[1] == 20. and aware.txn.outcome()["rec_s0"] == 1


def test_vehicle_crossing_before_its_ack_returns():
    f, h = direct_B()
    r = run(f, h, Channel(10., 1., 1.), 0, "blind", {1: [OK]})   # switch at 10, ACK due at 15
    assert r.entries[1] == 13. and r.state[(1, 0)] == "entered" and 1 not in r.P


def test_retry_while_another_vehicles_release_is_required():
    """Detour: the park's ACK is lost; A must wait for the park's confirmation, which
    comes from the retry; A is then issued one round late and the return follows."""
    f = pair()
    h = detour(f)
    r = run(f, h, Channel(.1, .9, .9), 1, "blind", {1: [LOST_ACK, OK, OK], 0: [OK]})
    assert r.entries == {0: 12., 1: 9.}
    assert r.txn.known_round[(1, 0)] == 0 and r.txn.first_round[(0, 0)] == 2


def test_temporary_booking_whose_return_message_is_lost():
    f = pair()
    h = detour(f)
    link = Channel(.1, .9, .9)
    rec = run(f, h, link, 1, "blind", {1: [OK, LOST_GRANT, OK], 0: [OK]})
    assert rec.entries == {0: 12., 1: 9.}                        # retry completes the return
    stranded = run(f, h, link, 0, "blind", {1: [OK, LOST_GRANT], 0: [OK]})
    assert stranded.entries == {0: 12., 1: 15.}                  # parked later, never returned
    assert stranded.state[(1, 1)] == "out" and stranded.P[1] == {15., 9.}


def test_retry_after_the_target_is_no_longer_reachable():
    """Physical motion: a head can still reach 13 at the first switch instant but not at
    the retry instant; the retransmission is not certified and is never sent."""
    from lockorder.models.physics import Kinematics
    kin = Kinematics()
    q = ([Job(0, 0, 0., earliest=0., latest=np.inf)], [Job(1, 1, 0., earliest=0., latest=np.inf)])
    snap = Snapshot(0., q, ((3., 3.), (3., 3.)), launch_time(kin))
    old = {0: 20., 1: 6.}
    heads = {0: make_profile(0., kin.L, 0., 20., kin), 1: make_profile(0., kin.L, 0., 6., kin)}
    # B currently arrives at 6; ask it to arrive at 5.2: reachable now, not later
    found = None
    for d in np.linspace(.05, 1.5, 60):
        f = Fork(snap, old, {0: 20., 1: 5.2}, kin, heads, 4)
        h = H.direct(f)
        e = (1, 0)
        if certified_at(f, h, e, d, []) and not certified_at(f, h, e, 3 * d, []):
            found = (f, h, d)
            break
    assert found is not None
    f, h, d = found
    r = run(f, h, Channel(d, .9, .9), 1, "blind", {1: [LOST_GRANT, OK]})
    assert r.host.rt["retx"] == 0 and r.entries[1] == 6.


# --------------------------------------------- exhaustive vs reference machine
def reference(moves, old, K, rule, d, u, v, bits, lat=.5, rounds=12):
    """Independent re-implementation of the protocol spec for a chain of moves on a
    point queue (kin None): certification is tau < min(a, b). Returns entries, P."""
    slot, P, entered = dict(old), {i: {s} for i, s in old.items()}, {}
    ver, executed, void, k = {i: 0 for i in old}, set(), set(), {i: 0 for i in old}
    n = len(moves)
    st, used, queried, base = ["wait"] * n, [0] * n, [False] * n, [0] * n
    mver = {i: 0 for i in old}

    def enter_until(t):
        for i in sorted(old, key=lambda j: slot[j]):
            if i not in entered and slot[i] <= t:
                entered[i] = slot[i]
                P.pop(i, None)
                for j, (vi, _, _) in enumerate(moves):
                    if vi == i and st[j] == "out":
                        st[j] = "entered"

    def cert(j, r):
        _, a, b = moves[j]
        return (2 * r + 1) * d < min(a, b)

    def worth(j, r):
        _, a, b = moves[j]
        w = a - b
        if w > 0:
            return True
        for c in range(j + 1, n):
            if st[c] == "wait" and cert(c, r + c - j):
                w += (moves[c][1] - moves[c][2]) * v * u * (u * v) ** (c - j - 1)
        return w > 1e-8

    for r in range(rounds):
        enter_until(2 * r * d)
        sends = []
        for j, (i, a, b) in enumerate(moves):
            if st[j] == "wait" and (j == 0 or st[j - 1] == "done"):
                if cert(j, r) and rule == "aware" and r > j and not worth(j, r):
                    st[j] = "dead"
                elif cert(j, r):
                    P[i].add(b)
                    st[j], base[j] = "out", mver[i]
                    sends.append((j, "grant"))
                else:
                    st[j] = "dead"
            elif st[j] == "out" and used[j] < K:
                c = not queried[j] and cert(j, r)
                if rule == "blind":
                    if not c:
                        used[j] = K    # stuck
                        continue
                    kind = "grant"
                else:
                    kind = "grant" if c and worth(j, r) else "query"
                queried[j] |= kind == "query"
                used[j] += 1
                sends.append((j, kind))
        tau = (2 * r + 1) * d
        enter_until(tau)
        replies = []
        for j, kind in sends:
            i, a, b = moves[j]
            g_ok, a_ok = bits[i][k[i]]
            k[i] += 1
            if not g_ok or i in entered:
                continue
            rep = None
            if kind == "grant":
                if j in executed:
                    rep = "ack"
                elif j not in void and ver[i] == base[j] and slot[i] == a:
                    slot[i], ver[i] = b, ver[i] + 1
                    executed.add(j)
                    rep = "ack"
            else:
                if j in executed:
                    rep = "ack"
                elif ver[i] == base[j] and slot[i] == a:
                    void.add(j)
                    rep = "void"
            if rep and a_ok:
                replies.append((j, rep))
        enter_until(tau + lat * d)
        for j, rep in replies:
            i, a, b = moves[j]
            if st[j] != "out":
                continue
            if rep == "ack":
                st[j], mver[i] = "done", base[j] + 1
                if i in P:
                    P[i] = {b}
            else:
                st[j] = "void"
                if i in P:
                    P[i] = {a}
    enter_until(np.inf)
    return entered


def histories(n_tx, u, v):
    """All grant/reply outcomes for n_tx[i] potential transmissions per vehicle."""
    slots = [(i, t) for i in sorted(n_tx) for t in range(n_tx[i])]
    for outcome in product(product((True, False), repeat=2), repeat=len(slots)):
        bits = {i: [None] * n_tx[i] for i in n_tx}
        p = 1.
        for (i, t), (g, a) in zip(slots, outcome):
            bits[i][t] = (g, a)
            p *= (u if g else 1 - u) * (v if a else 1 - v)
        yield bits, p


CASES = [  # (name, d): d = 0.1 gives slack everywhere; d = 1.5 leaves the return no spare round
    ("slack", .1), ("tight", 1.5)]


@pytest.mark.parametrize("name,d", CASES)
@pytest.mark.parametrize("K,rule", [(0, "blind"), (1, "blind"), (1, "aware")])
def test_detour_matches_reference_machine_on_every_history(name, d, K, rule):
    f = pair()
    h = detour(f)
    moves = [(1, 14., 15.), (0, 10., 12.), (1, 15., 9.)]
    u, v = .9, .8
    link = Channel(d, u, v)
    total = 0.
    for bits, p in histories({0: K + 1, 1: 2 * (K + 1)}, u, v):
        got = run(f, h, link, K, rule, bits)
        want = reference(moves, f.old, K, rule, d, u, v, bits)
        assert got.entries == want, (bits, got.entries, want)
        total += p * sum(f.old[i] - s for i, s in got.entries.items())
    if rule == "blind":
        R = (4, 2, 2) if name == "tight" else (99, 99, 99)
        assert total == pytest.approx(RV.pair_detour(u, v, K, R), abs=1e-12)


@pytest.mark.parametrize("K", [0, 1, 2])
@pytest.mark.parametrize("rule", ["blind", "aware"])
def test_direct_move_matches_reference_and_closed_form(K, rule):
    f, h = direct_B()
    u, v = .85, .7
    for d in (.1, 4.5):
        link = Channel(d, u, v)
        total = 0.
        for bits, p in histories({0: 0, 1: K + 1}, u, v):
            got = run(f, h, link, K, rule, bits)
            assert got.entries == reference([(1, 14., 13.)], f.old, K, rule, d, u, v, bits)
            total += p * (f.old[1] - got.entries[1])
        R = 99 if d == .1 else 0
        assert total == pytest.approx(RV.pair_direct(u, v, K, R), abs=1e-12)


# -------------------------------------------------------------------- theory
def test_single_move_probabilities_and_limits():
    u, v = .9, .9
    z = RV.single(u, v, 2)
    assert z["confirmed"] == pytest.approx(1 - .19 ** 2)          # not 1 - 0.1^2
    assert z["switch"] + z["not_switched"] == pytest.approx(1)
    assert z["ambiguous_switched"] == pytest.approx(.19 ** 2 - .01)
    # zero slack: retries change nothing; unlimited slack: product form; K = 0: Prop. 1
    assert RV.chain_effective(u, v, 2, [0, 1, 2]) == pytest.approx(RV.chain_effective(u, v, 0, [0, 1, 2]))
    assert RV.chain_effective(u, v, 2, [99] * 3)[2] == pytest.approx(RV.no_deadline(u, v, 2, 2))
    assert RV.chain_effective(u, v, 0, [99] * 3) == pytest.approx([u, u * u * v, u * (u * v) ** 2])
    # monotone in the budget, move by move
    for R in ([4, 2, 2], [1, 3, 3], [99, 99, 99]):
        prev = RV.chain_effective(u, v, 0, R)
        for K in (1, 2, 3):
            cur = RV.chain_effective(u, v, K, R)
            assert all(c >= p - 1e-15 for c, p in zip(cur, prev))
            prev = cur
    # perfect link and certain loss
    assert RV.chain_effective(1., 1., 2, [0, 1, 2]) == [1., 1., 1.]
    assert RV.chain_effective(0., 0., 5, [9] * 3) == [0., 0., 0.]


def test_blind_retry_can_lower_value_and_reverse_the_choice():
    """Counterexample (Prop. R2): with no spare round for the return, a blind retry of
    the park raises the chance of a costly park and of A's sacrifice but not of the
    return; the detour then loses to the direct target. The aware rule does not."""
    u = v = .9
    R = (4, 2, 2)
    v0, v1 = RV.pair_detour(u, v, 0, R), RV.pair_detour(u, v, 1, R)
    assert v1 < v0 and v0 > RV.pair_direct(u, v, 0, 3) and v1 < RV.pair_direct(u, v, 1, 3)
    f = pair()
    h = detour(f)
    link = Channel(1.5, u, v)
    aware = sum(p * sum(f.old[i] - s for i, s in run(f, h, link, 1, "aware", b).entries.items())
                for b, p in histories({0: 2, 1: 4}, u, v))
    assert aware == pytest.approx(v0, abs=1e-12)
    assert RV.park_threshold() == pytest.approx(.7795, abs=1e-4)
