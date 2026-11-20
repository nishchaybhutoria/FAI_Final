"""Finite pair rule used by the information-adaptive commitment policy (stopping/pair.py)."""
import numpy as np

from lockorder.stopping.pair import KEEP, SWAP, WAIT as PWAIT, pair_advice


def test_pair_advice_last_epoch_must_commit_and_wait_with_free_info():
    kap0 = lambda u: 0.0
    # no future reveals before deadline -> commit preferred sign
    adv = pair_advice(1.0, 0.0, [], [], step=0.5, h=3.0, lead=5.0, kappa=kap0)
    assert adv.action == KEEP
    adv = pair_advice(-1.0, 0.0, [], [], step=0.5, h=3.0, lead=5.0, kappa=kap0)
    assert adv.action == SWAP
    # free information and sign uncertainty -> wait
    adv = pair_advice(0.2, 0.0, [1.0, 2.0], [1.5], step=0.5, h=3.0, lead=5.0, kappa=kap0)
    assert adv.action == PWAIT
    # huge gap: no possible sign change -> waiting only ties -> wait (tie convention)
    adv = pair_advice(10.0, 0.0, [1.0], [1.5], step=0.5, h=3.0, lead=5.0, kappa=kap0)
    assert adv.action == PWAIT
    # steeply rising cost with huge gap -> commit now
    adv = pair_advice(10.0, 0.0, [1.0], [1.5], step=0.5, h=3.0, lead=5.0, kappa=lambda u: 5.0 - u)
    assert adv.action == KEEP


def test_pair_advice_matches_bruteforce_enumeration():
    """Independent brute force over sign paths for a 2-epoch instance with zero kappa."""
    step, h = 0.5, 3.0
    m0 = 0.3
    adv = pair_advice(m0, 0.0, [1.0], [2.0], step=step, h=h, lead=10.0, kappa=lambda u: 0.0)
    # waiting until both reveals then choosing the better order with full info
    vals = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            m = m0 + step * (s2 - s1)    # second's sign raises m, first's lowers it
            vals.append(min(max(0, h - m), max(0, h + m)))
    assert abs(adv.v_wait - np.mean(vals)) < 1e-12


def test_queue_pair_reduces_to_hinge_without_queue():
    from lockorder.stopping.pair import pair_advice_queue
    rng = np.random.default_rng(3)
    for _ in range(40):
        rx, ry = rng.uniform(0, 5, 2)
        revx = sorted(rng.uniform(0, 4, int(rng.integers(0, 4))))
        revy = sorted(rng.uniform(0, 4, int(rng.integers(0, 4))))
        kap = lambda u: max(0.0, 1.0 - u)
        a = pair_advice(ry - rx, 0.0, revx, revy, 0.5, 3.0, 3.0, kap)
        b = pair_advice_queue(rx, ry, -np.inf, 0.0, revx, revy, 0.5, 3.0, 3.0, kap)
        assert a.action == b.action
        assert abs(a.v_keep - b.v_keep) < 1e-9 and abs(a.v_swap - b.v_swap) < 1e-9
        if np.isfinite(a.v_wait):
            assert abs(a.v_wait - b.v_wait) < 1e-9


def test_queue_pair_indifferent_when_both_past_due():
    from lockorder.stopping.pair import pair_advice_queue
    adv = pair_advice_queue(1.0, 2.0, 10.0, 0.0, [], [], 0.5, 3.0, 12.0, lambda u: 0.0)
    assert abs(adv.v_keep - adv.v_swap) < 1e-12 and adv.action == KEEP
