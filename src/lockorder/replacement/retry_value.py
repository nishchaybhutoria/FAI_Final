"""Transaction-level probabilities for release-gated updates with bounded retries.

Model (docs/mathematical_notes.md, section R). Rounds r = 0, 1, ...; a message
sent in round r is effective at tau_r. Each transmission of an update (grant or
retransmission) reaches the vehicle with probability u; a vehicle that receives
it replies, and the reply reaches the manager with probability v; all legs are
independent. A move is first issued in the round after all its prerequisites are
confirmed, may switch only in rounds r <= R (its last certifiable round;
certification is assumed to hold on an initial interval of rounds), and is
retransmitted blindly, at most ``K`` extra times, until confirmed.

For one move with n usable transmissions:
  P(switch)                = 1 - (1-u)^n
  P(confirmed)             = 1 - (1-uv)^n
  P(switched, unconfirmed) = (1-uv)^n - (1-u)^n        (ambiguous: lost ACKs)
  P(not switched)          = (1-u)^n                    (ambiguous unless queried)
The per-packet loss q^n does not describe either event: with u = v = 1 - q a
round fails to confirm with probability 1 - (1-q)^2, about 2q.
"""
from __future__ import annotations

import math


def usable(K, s, R):
    """Transmissions available to a move first issued in round s with last round R."""
    return max(0, min(K + 1, R - s + 1))


def single(u, v, n):
    return dict(switch=1 - (1 - u) ** n, confirmed=1 - (1 - u * v) ** n,
                ambiguous_switched=(1 - u * v) ** n - (1 - u) ** n, not_switched=(1 - u) ** n)


def chain_effective(u, v, K, R, s0=0):
    """P(move j takes effect) for a chain e_0 -> e_1 -> ... under blind retries.

    R[j] is the last round in which e_j may switch. e_0 is issued in round s0;
    e_{j+1} is issued in the round after e_j is confirmed. Exact (independent legs).
    """
    dist = {s0: 1.0}            # issue round of the current move
    out = []
    for Rj in R:
        eff, nxt = 0.0, {}
        for s, pr in dist.items():
            n = usable(K, s, Rj)
            eff += pr * (1 - (1 - u) ** n)
            for f in range(n):      # first confirmation in transmission f (round s + f)
                nxt[s + f + 1] = nxt.get(s + f + 1, 0.0) + pr * (1 - u * v) ** f * u * v
        out.append(eff)
        dist = nxt
    return out


def no_deadline(u, v, K, m):
    """Unlimited slack: P(e effective) with m strict ancestors on a chain."""
    return (1 - (1 - u) ** (K + 1)) * (1 - (1 - u * v) ** (K + 1)) ** m


def pair_detour(u, v, K, R, h=3., delta=1., s=1., x=None):
    """Tight pair (Prop. 2): B parks O_B -> x, A moves a -> a+h-delta, B returns x -> a-delta.
    Expected vehicle-seconds saved under blind retries; R = (R_park, R_A, R_return)."""
    a = 0.
    x = a + 2 * h - delta if x is None else x
    pp, pa, pr = chain_effective(u, v, K, R)
    return ((a + h + s) - x) * pp + (a - (a + h - delta)) * pa + (x - (a - delta)) * pr


def pair_direct(u, v, K, R_B, s=1.):
    """Best direct target of the tight pair: B alone to a + h, saving s."""
    return s * chain_effective(u, v, K, [R_B])[0]


def park_threshold(h=3., delta=1., s=1.):
    """Per-leg success p = u = v above which retransmitting the park pays (one attempt
    per later move, return still in time): (x - a + delta) p^4 > (x - O_B) + (h - delta) p^2."""
    x, O = 2 * h - delta, h + s
    A, B, C = x + delta, -(h - delta), -(x - O)      # A p^4 + B p^2 + C = 0
    z = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    return math.sqrt(z)
