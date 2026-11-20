"""Bounded double-integrator kinematics on a single approach.

Units: metres, seconds, m/s, m/s^2.  ``p`` is the remaining distance to the
first contact with the conflict resource, ``v >= 0`` the speed, ``a > 0`` the
maximum acceleration, ``b > 0`` the maximum braking magnitude and ``V`` the
speed cap.  All times returned here are *durations* from the current state.

The central constructive object is :class:`Profile`, the convex mixture of the
maximum-acceleration and maximum-braking speed envelopes whose distance integral
is exactly ``p`` at the requested arrival duration ``T`` (RESEARCH_PLAN B2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

INF = math.inf
EPS = 1e-9


@dataclass(frozen=True)
class Kinematics:
    a: float = 3.0      # max acceleration (m/s^2)
    b: float = 3.0      # max braking magnitude (m/s^2)
    V: float = 6.0      # speed cap (m/s)
    L: float = 8.0      # controlled approach length (m)

    def __post_init__(self) -> None:
        if not (self.a > 0 and self.b > 0 and self.V > 0 and self.L > 0):
            raise ValueError("kinematic bounds must be positive")


def t_minus(p: float, v: float, k: Kinematics) -> float:
    """Earliest duration to first contact under maximum acceleration then cruise."""
    if p <= 0.0:
        return 0.0
    a, V = k.a, k.V
    v = min(max(v, 0.0), V)
    p_a = (V * V - v * v) / (2.0 * a)
    if p <= p_a:
        return 2.0 * p / (v + math.sqrt(v * v + 2.0 * a * p))
    return (V - v) / a + (p - p_a) / V


def t_plus(p: float, v: float, k: Kinematics) -> float:
    """Latest duration to first contact (``inf`` when the car can stop short).

    If ``p > v^2/(2b)`` the vehicle can stop before the line and wait, so any
    later first touch is reachable.  If ``p == v^2/(2b) > 0`` the latest touch is
    at ``v/b`` (arriving at zero speed).  If ``p < v^2/(2b)`` the latest touch is
    the first root of ``v t - b t^2/2 = p``.  Stopping *on* the resource is not
    allowed.
    """
    if p <= 0.0:
        return 0.0
    b = k.b
    v = max(v, 0.0)
    p_b = v * v / (2.0 * b)
    if p > p_b * (1.0 + 1e-12) + 1e-12:
        return INF
    disc = v * v - 2.0 * b * p
    if disc <= 0.0:
        return v / b
    return 2.0 * p / (v + math.sqrt(disc))


def d_plus(T: float, v: float, k: Kinematics) -> float:
    """Distance covered in duration T under maximum acceleration then cruise."""
    if T <= 0.0:
        return 0.0
    a, V = k.a, k.V
    t1 = (V - v) / a
    if T <= t1:
        return v * T + 0.5 * a * T * T
    return (V * V - v * v) / (2.0 * a) + V * (T - t1)


def d_minus(T: float, v: float, k: Kinematics) -> float:
    """Distance covered in duration T under maximum braking (then rest)."""
    if T <= 0.0:
        return 0.0
    b = k.b
    t1 = v / b
    if T <= t1:
        return v * T - 0.5 * b * T * T
    return v * v / (2.0 * b)


def _v_plus(x: float, v: float, k: Kinematics) -> float:
    return min(k.V, v + k.a * x)


def _v_minus(x: float, v: float, k: Kinematics) -> float:
    return max(0.0, v - k.b * x)


class InfeasibleArrival(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    """Mixture profile from state (p0, v0) at absolute time t0 arriving at t0+T.

    speed(x) = alpha * v_plus(x) + (1-alpha) * v_minus(x), x = t - t0.
    Acceleration is alpha*a*[v_plus<V] - (1-alpha)*b*[v_minus>0], in [-b, a].
    """
    t0: float
    p0: float
    v0: float
    T: float
    alpha: float
    k: Kinematics

    @property
    def arrival(self) -> float:
        return self.t0 + self.T

    def distance(self, t: float) -> float:
        x = min(max(t - self.t0, 0.0), self.T)
        return self.alpha * d_plus(x, self.v0, self.k) + (1 - self.alpha) * d_minus(x, self.v0, self.k)

    def state(self, t: float) -> tuple[float, float]:
        """(p, v) at absolute time t (clamped to the profile duration)."""
        x = min(max(t - self.t0, 0.0), self.T)
        d = self.alpha * d_plus(x, self.v0, self.k) + (1 - self.alpha) * d_minus(x, self.v0, self.k)
        v = self.alpha * _v_plus(x, self.v0, self.k) + (1 - self.alpha) * _v_minus(x, self.v0, self.k)
        p = max(self.p0 - d, 0.0)
        if x >= self.T:
            p = 0.0
        return p, v

    def breakpoints(self) -> list[float]:
        """Elapsed times at which the acceleration changes (within [0, T])."""
        pts = {0.0, self.T}
        t1 = (self.k.V - self.v0) / self.k.a
        t2 = self.v0 / self.k.b
        for x in (t1, t2):
            if 0.0 < x < self.T:
                pts.add(x)
        return sorted(pts)

    def accel_bounds_ok(self, tol: float = 1e-9) -> bool:
        lo, hi = -self.k.b * (1 - self.alpha), self.k.a * self.alpha
        return -self.k.b - tol <= lo and hi <= self.k.a + tol


def make_profile(t0: float, p0: float, v0: float, s: float, k: Kinematics, tol: float = 1e-7) -> Profile:
    """Constructive certificate for first contact exactly at absolute time ``s``.

    Raises :class:`InfeasibleArrival` if ``s`` is outside [t0+T_-, t0+T_+]
    beyond the numerical tolerance ``tol`` (seconds).  Never clips silently
    beyond ``tol``.
    """
    T = s - t0
    if p0 <= 0.0:
        raise InfeasibleArrival("vehicle already at the line")
    tm, tp = t_minus(p0, v0, k), t_plus(p0, v0, k)
    if T < tm - tol or T > tp + tol:
        raise InfeasibleArrival(f"T={T:.9f} outside [{tm:.9f}, {tp:.9f}]")
    T = min(max(T, tm), tp)
    dp, dm = d_plus(T, v0, k), d_minus(T, v0, k)
    if dp - dm <= 1e-15:
        alpha = 1.0
    else:
        alpha = (p0 - dm) / (dp - dm)
    if alpha < -1e-6 or alpha > 1 + 1e-6:
        raise InfeasibleArrival(f"mixture weight {alpha} outside [0,1]")
    alpha = min(max(alpha, 0.0), 1.0)
    return Profile(t0=t0, p0=p0, v0=v0, T=T, alpha=alpha, k=k)


def launch_time(k: Kinematics) -> float:
    """T_-(L, 0): fastest traversal of the controlled approach from rest."""
    return t_minus(k.L, 0.0, k)


def promotion_loss(w: float, h: float, k: Kinematics) -> float:
    """Earliest-arrival loss g(w) after tracking a provisional yield slot.

    From the reference launch state (L, 0), track the slot T_f + h with the
    mixture certificate for ``w`` seconds; then g(w) = w + T_-(p_w, v_w) - T_f
    (RESEARCH_PLAN B2).  Nonnegative and nondecreasing in w; defined for
    0 <= w <= T_f + h.
    """
    Tf = launch_time(k)
    if w <= 0.0:
        return 0.0
    prof = make_profile(0.0, k.L, 0.0, Tf + h, k)
    w = min(w, Tf + h)
    p, v = prof.state(w)
    if p <= 0.0:
        return max(0.0, w - Tf)
    return max(0.0, w + t_minus(p, v, k) - Tf)


def kappa_surrogate(lead: float, h: float, k: Kinematics) -> float:
    """Time-only notification surrogate kappa(u) = 2 g(min(T_f, max(0, T_f - u)))."""
    Tf = launch_time(k)
    w = min(Tf, max(0.0, Tf - lead))
    return 2.0 * promotion_loss(w, h, k)
