"""The linac bunches an injection turn brings, and where they land in the
ring's bucket.

Linac bunch j (counted from the start of the injection) reaches the foil at
t_j = j / f_L, f_L the linac's bunch rate.  The injection turn k receives
the bunches with t_j in [t_k, t_k+1) — t_k the time of its start, counted
from the injection's start.  A bunch arriving dt after the reference
particle passes the foil (at t_k) is behind it by beta0 c dt: its zeta is
-beta0 c dt - L phi0/360, wrapped into [-L/2, L/2) with L the RF period in
zeta (C/h, h the common harmonic of the cavities with a voltage; the
circumference without RF) and phi0 the chopper phase of the first bunch
(> 0 late).  Every bucket is the same (a one-bucket model), so the wrap puts
each bunch in the tracked one.  With the RF on, the chopper keeps the bunches
within ``window_deg`` of the bucket's centre.
"""
from __future__ import annotations

import math

import numpy as np

C_LIGHT = 299792458.0


def first_bunch(t: float, f_linac_hz: float) -> int:
    """The first linac bunch arriving at time t or after: the first whole
    number at or above t f, within 1e-9 of a bunch spacing or 1e-12 of t f —
    a bunch on a turn's boundary goes to the turn that starts there, whether
    the boundary is reached as one turn's end or the next one's start (the
    two times can differ in their last bits), however many turns have run."""
    x = t * f_linac_hz
    return math.ceil(x - max(1e-9, 1e-12 * abs(x)))


def bunch_zeta(t0: float, t1: float, *, f_linac_hz: float, beta0: float, period_m: float,
               phase_deg: float = 0.0) -> np.ndarray:
    """zeta [m] of the linac bunches arriving in [t0, t1) (times from the
    injection's start), wrapped into [-period/2, period/2)."""
    j0 = first_bunch(t0, f_linac_hz)
    j1 = first_bunch(t1, f_linac_hz)
    j = np.arange(j0, j1, dtype=np.float64)
    dt = j / f_linac_hz - t0
    z = -beta0 * C_LIGHT * dt - period_m * (phase_deg / 360.0)
    return z - period_m * np.floor(z / period_m + 0.5)


def kept(zeta: np.ndarray, *, period_m: float, window_deg: float, rf_on: bool) -> np.ndarray:
    """The bunches the chopper keeps: within ``window_deg`` of the bucket's
    centre (every one without the RF, or with a window of 180 deg)."""
    if not rf_on or window_deg >= 180.0:
        return zeta
    return zeta[np.abs(zeta) <= period_m * (window_deg / 360.0)]
