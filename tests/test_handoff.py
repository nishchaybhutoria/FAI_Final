"""Loss-aware handoffs: exact value on the move graph, reduction to the archived
direct scorer and executor, and the tight-pair closed form."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from lockorder.models.physics import Kinematics, launch_time, make_profile
from lockorder.replacement import Channel, Fork, execute as direct_execute
from lockorder.replacement import handoff as H
from lockorder.replacement.bridge import complete
from lockorder.replacement.model import compile_update
from lockorder.replacement.sample import Sampler
from lockorder.replacement.targets import score
from lockorder.scheduling.core import Job, Snapshot
from lockorder.simulation.scenario import SimConfig, make_scenario


def pair(O=(10., 14.), N=(12., 9.), floors=(10., 9.), lanes=(0, 1), kin=None, h=3., latest=None):
    q = ([], [])
    heads = {}
    old = dict(enumerate(O))
    latest = latest or (np.inf,) * len(O)
    for i, l in enumerate(lanes):
        if not q[l] and kin:
            heads[i] = make_profile(0., kin.L, 0., old[i], kin)
        q[l].append(Job(i, l, floors[i], earliest=0., latest=latest[i]))
    return Fork(Snapshot(0., q, ((h, h), (h, h)), launch_time(kin) if kin else 0.), old, dict(enumerate(N)), kin, heads, 4)


def closed_form(u, v, h=3., delta=1., s=1.):
    return ((s + 2 * delta) * u ** 3 * v ** 2 - (h - delta - s) * (u * (1 - v) + u * v * (1 - u))
            - (2 * h - 2 * delta - s) * (u * u * v * (1 - v) + u * u * v * v * (1 - u)))


def enumerate_histories(f, h, chosen, link):
    """Exact expectation over every grant/ACK outcome of every leg."""
    legs = sorted(chosen)
    total = 0.
    for bits in product((False, True), repeat=2 * len(legs)):
        tape = np.full((len(f.old), 8), .5)
        tape2 = np.full((len(f.old), 4), .5)
        p = 1.
        for (e, kind), ok in zip(product(legs, (0, 1)), bits):
            col = tape2 if h.moves[e].leg else tape
            col[h.moves[e].i, kind] = 0. if ok else .999
            q = link.u if kind == 0 else link.v
            p *= q if ok else 1 - q
        r = H.execute(f, h, chosen, link, tape, tape2)
        total += p * sum(f.old[i] - s for i, s in r.entries.items())
    return total


@pytest.mark.parametrize("u,v,d", [(1., 1., 0.), (.9, .9, .1), (.8, .7, .1), (.95, .6, 0.)])
def test_detour_value_matches_closed_form_and_every_delivery_history(u, v, d):
    f = pair()
    link = Channel(d, u, v)
    perfect = Channel(d, 1., 1.)
    # detours that complete the target: B parks at a+2h-delta = 15, or A at a+2h+s = 17
    hs = {h.temp: h for h in H.detours(f) if H.eager(f, h, perfect)[0] == pytest.approx(3.)}
    assert {k: min(x for kk, x in hs if kk == k) for k, _ in hs} == {1: 15., 0: 17.}
    best = hs[(1, 15.)]
    if u < 1:                    # with loss the shorter displacement is strictly better
        assert H.eager(f, best, link)[0] > H.eager(f, hs[(0, 17.)], link)[0]
    ev, chosen, _ = H.eager(f, best, link)
    assert ev == pytest.approx(closed_form(u, v))
    assert enumerate_histories(f, best, chosen, link) == pytest.approx(ev)
    # the closure never does worse than sending everything, and never worse than nothing
    assert H.value(f, best, link)[0] >= max(ev, 0.) - 1e-12


def test_pair_crossover_against_best_direct_target():
    # Prop. 1: best direct value is u*s; the detour wins above the closed-form crossover.
    ps = np.linspace(.5, 1, 5001)
    gap = np.array([closed_form(p, p) - p * 1. for p in ps])
    cross = ps[np.argmax(gap > 0)]
    assert .87 < cross < .88
    f = pair()
    for p, want in [(.95, "detour"), (.8, "direct")]:
        c = H.choose(f, Channel(0., p, p))
        assert (c.handoff.temp is not None) == (want == "detour")
        assert c.value == pytest.approx(max(closed_form(p, p), p * 1.))


@pytest.mark.parametrize("s,delta", [(0., 1.), (1., 1.), (.5, 2.), (0., 2.5)])
def test_every_detour_in_the_tight_family_parks_a_vehicle_later(s, delta):
    a, h = 10., 3.
    f = pair((a, a + h + s), (a + h - delta, a - delta), (a, a - delta))
    feasible = [g for g in H.detours(f) if H.eager(f, g, Channel(0., 1., 1.))[0] == pytest.approx(s + 2 * delta)]
    assert feasible
    for g in feasible:
        k, x = g.temp
        assert x > f.old[k] + 1e-9
        assert x >= (a + 2 * h - delta if k == 1 else a + 2 * h + s) - 1e-9


def streams(n=3, study=9431):
    cfg = SimConfig(rho=.75, sigma=1., warmup=20., win_end=80., t_arr_end=90., audit=True, tie_eta=0.)
    for rep in range(n):
        sam = Sampler(make_scenario(cfg, "test", rep, study=study))
        sam.run()
        for ob in sam.observations:
            yield sam.sc.key, ob


@pytest.fixture(scope="module")
def observed():
    return list(streams())


def test_direct_special_case_reproduces_archived_score_certificate_and_execution(observed):
    links = [Channel(.1, .9, .9), Channel(0., 1., 1.), Channel(.5, .9, .9)]
    checked = 0
    for key, ob in observed:
        f = ob.fork
        tape2 = H.second_tape(key, f.epoch, len(ob.tape))
        for link in links:
            h = H.direct(f)
            c = compile_update(f, link.d)
            depth, _, el = H.compile_handoff(f, h, link.d)
            assert {e[0] for e in el} == c.eligible
            for i, k in c.depth.items():
                assert H.certified(f, h, depth, (i, 0), link.d) == f.certified(i, f.snap.t + (2 * k + 1) * link.d)
            v, chosen, _ = H.value(f, h, link)
            assert v == pytest.approx(score(f, link), abs=1e-12)
            mine = H.execute(f, h, chosen, link, ob.tape, tape2).entries
            assert mine == direct_execute(f, "scoped", link, ob.tape).entries
            checked += 1
    assert checked > 100


def test_perfect_link_value_attains_full_gain_whenever_one_detour_completes(observed):
    perfect = Channel(0., 1., 1.)
    rescued = 0
    for key, ob in observed:
        f = ob.fork
        if not f.changed:
            continue
        full = sum(f.old.values()) - sum(f.new.values())
        path = complete(f)
        c = H.choose(f, perfect)
        if path is not None:
            assert c.value >= full - 1e-9
            rescued += any(min(abs(x - f.old[i]), abs(x - f.new[i])) > 1e-8 for i, x in path)
        r = H.execute(c.target, c.handoff, c.chosen, perfect, ob.tape, H.second_tape(key, f.epoch, len(ob.tape)))
        assert sum(f.old[i] - s for i, s in r.entries.items()) == pytest.approx(c.value, abs=1e-9)
    assert rescued >= 1


def test_physical_holder_return_is_recertified_after_parking():
    kin = Kinematics()
    f = pair(latest=(12., 15.), kin=kin)
    link = Channel(.1, 1., 1.)
    hs = list(H.detours(f))
    assert hs
    for h in hs:
        depth, _, el = H.compile_handoff(f, h, link.d)
        if (h.temp[0], 1) in el:
            # a certified return is executable from the parked profile
            r = H.execute(f, h, el, link, np.zeros((2, 8)), np.zeros((2, 4)))
            assert r.entries == f.new


def test_lossy_execution_never_loses_a_live_authorization(observed):
    rng = np.random.default_rng(5)
    link = Channel(.1, .7, .7)
    for key, ob in observed[:120]:
        f = ob.fork
        c = H.choose(f, link)
        for _ in range(3):
            H.execute(c.target, c.handoff, c.chosen, link, rng.random(ob.tape.shape), rng.random((len(ob.tape), 4)))
