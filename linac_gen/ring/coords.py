"""Exact per-particle coordinate conversions for ring mode.

Ring coordinates are canonical (the transverse ones as in MAD-X):

    x [m], px = Px/P0, y [m], py = Py/P0, zeta = s - beta0*c*t [m], delta = (P - P0)/P0

with the energy deviation ptau = (E - E0)/(P0 c), and the helpers
rpp = 1/(1 + delta) and rvv = beta/beta0.  The conversions are exact (no
paraxial or first-order shortcuts); the signs of the time coordinates are:

* HELIX linac:  dphi [deg] at the RF frequency f, > 0 means LATE; dW [MeV].
                x', y' are slopes dx/ds, dy/ds [mrad].
                zeta = -beta0 * c * dphi / (360 f).
* time lag:     t = c * dt [m], > 0 means late; pt = -dE/(P0 c)
                (in the order x, y, t, px, py, pt).
                zeta = -beta0 * t,  ptau = -pt.
* MAD-X:        T = -c * dt [m] (> 0 means early); PT = dE/(P0 c).
                zeta = beta0 * T,   ptau = PT.

Charge exchange at a stripping foil conserves velocity, not momentum: an H⁻
of kinetic energy W leaves as a proton with the same gamma (an 800 MeV H⁻
becomes a 799.1296 MeV proton) and the same direction.
"""
from __future__ import annotations

import numpy as np

from linac_gen.ring.reference import RingReference

C_LIGHT = 299792458.0


# ---------------------------------------------------------------------------
# energy variables (vectorized; beta0 is the reference velocity)
# ---------------------------------------------------------------------------
def ptau_from_delta(delta, beta0: float):
    """(E - E0)/(P0 c) from delta: sqrt((1+delta)^2 + 1/(beta0 gamma0)^2) - 1/beta0,
    written without cancellation as ((1+delta)^2 - 1)/(sqrt(...) + 1/beta0)."""
    d = np.asarray(delta, dtype=float)
    inv_b0 = 1.0 / beta0
    inv_bg2 = inv_b0 * inv_b0 - 1.0                     # 1/(beta0 gamma0)^2
    num = d * (2.0 + d)                                  # (1+d)^2 - 1
    return num / (np.sqrt((1.0 + d) ** 2 + inv_bg2) + inv_b0)


def delta_from_ptau(ptau, beta0: float):
    """delta from ptau: (1+delta)^2 = 1 + 2 ptau/beta0 + ptau^2, written as
    x/(sqrt(1+x)+1) with x = ptau^2 + 2 ptau/beta0 to avoid cancellation."""
    pt = np.asarray(ptau, dtype=float)
    x = pt * pt + 2.0 * pt / beta0
    return x / (np.sqrt(1.0 + x) + 1.0)


def rpp_from_delta(delta):
    return 1.0 / (1.0 + np.asarray(delta, dtype=float))


def rvv_from_delta(delta, beta0: float):
    """beta/beta0 = (1 + delta) / (1 + beta0 * ptau)."""
    d = np.asarray(delta, dtype=float)
    return (1.0 + d) / (1.0 + beta0 * ptau_from_delta(d, beta0))


def slopes_from_momenta(px, py, delta):
    """x' = px/pz, y' = py/pz with pz = sqrt((1+delta)^2 - px^2 - py^2)."""
    pz = np.sqrt((1.0 + np.asarray(delta, dtype=float)) ** 2
                 - np.asarray(px) ** 2 - np.asarray(py) ** 2)
    return np.asarray(px) / pz, np.asarray(py) / pz


def momenta_from_slopes(xp, yp, delta):
    """Inverse of :func:`slopes_from_momenta`: px = (1+delta) x'/sqrt(1 + x'^2 + y'^2)."""
    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    f = (1.0 + np.asarray(delta, dtype=float)) / np.sqrt(1.0 + xp * xp + yp * yp)
    return xp * f, yp * f


# ---------------------------------------------------------------------------
# HELIX linac <-> ring
# ---------------------------------------------------------------------------
def from_helix(particles, *, w_ref: float, mass: float, frequency_mhz: float,
               ring_ref: RingReference) -> np.ndarray:
    """HELIX (N, 6) [x mm, x' mrad, y mm, y' mrad, dphi deg, dW MeV] → ring (6, N).

    ``w_ref``/``mass`` describe the HELIX beam's own reference (kinetic energy,
    rest mass) — it may differ from ``ring_ref`` (e.g. an H⁻ beam at the foil
    entering a proton ring: use :func:`strip_to_proton` afterwards); the
    particles' absolute energy is preserved here, with ``mass`` as their mass.
    """
    P = np.asarray(particles, dtype=float)
    out = np.empty((6, P.shape[0]))
    w = w_ref + P[:, 5]
    pc = np.sqrt(w * (w + 2.0 * mass))
    delta = pc / ring_ref.p0c - 1.0
    xp, yp = P[:, 1] * 1e-3, P[:, 3] * 1e-3
    px, py = momenta_from_slopes(xp, yp, delta)
    out[0] = P[:, 0] * 1e-3
    out[1] = px
    out[2] = P[:, 2] * 1e-3
    out[3] = py
    dt = P[:, 4] / (360.0 * frequency_mhz * 1e6)       # delay, s (> 0 late)
    out[4] = -ring_ref.beta0 * C_LIGHT * dt
    out[5] = delta
    return out


def to_helix(coords, *, ring_ref: RingReference, w_ref: float, frequency_mhz: float) -> np.ndarray:
    """Ring (6, N) → HELIX (N, 6), relative to a HELIX reference of the SAME
    species and kinetic energy ``w_ref`` (pass ``ring_ref.kinetic0`` for the ring's)."""
    X = np.asarray(coords, dtype=float)
    delta = X[5]
    xp, yp = slopes_from_momenta(X[1], X[3], delta)
    pc = ring_ref.p0c * (1.0 + delta)
    m = ring_ref.mass
    w = pc * pc / (np.sqrt(pc * pc + m * m) + m)        # kinetic, no cancellation
    out = np.empty((X.shape[1], 6))
    out[:, 0] = X[0] * 1e3
    out[:, 1] = xp * 1e3
    out[:, 2] = X[2] * 1e3
    out[:, 3] = yp * 1e3
    dt = -X[4] / (ring_ref.beta0 * C_LIGHT)
    out[:, 4] = dt * 360.0 * frequency_mhz * 1e6
    out[:, 5] = w - w_ref
    return out


# ---------------------------------------------------------------------------
# time-lag and MAD-X coordinates <-> ring (same reference)
# ---------------------------------------------------------------------------
def from_time_lag(x, y, t, px, py, pt, *, beta0: float) -> np.ndarray:
    """Time-lag phase space (t = c dt, pt = -dE/(P0 c)) → ring (6, N), same reference."""
    ptau = -np.asarray(pt, dtype=float)
    return np.vstack([np.asarray(x, float), np.asarray(px, float), np.asarray(y, float),
                      np.asarray(py, float), -beta0 * np.asarray(t, dtype=float),
                      delta_from_ptau(ptau, beta0)])


def to_time_lag(coords, *, beta0: float):
    """Ring (6, N) → time-lag phase space (x, y, t, px, py, pt), same reference."""
    X = np.asarray(coords, dtype=float)
    return X[0], X[2], -X[4] / beta0, X[1], X[3], -ptau_from_delta(X[5], beta0)


def from_madx(x, px, y, py, t, pt, *, beta0: float) -> np.ndarray:
    """MAD-X canonical (T = -c dt, PT = dE/(P0 c)) → ring (6, N), same reference."""
    return np.vstack([np.asarray(x, float), np.asarray(px, float), np.asarray(y, float),
                      np.asarray(py, float), beta0 * np.asarray(t, dtype=float),
                      delta_from_ptau(np.asarray(pt, dtype=float), beta0)])


def to_madx(coords, *, beta0: float):
    X = np.asarray(coords, dtype=float)
    return X[0], X[1], X[2], X[3], X[4] / beta0, ptau_from_delta(X[5], beta0)


# ---------------------------------------------------------------------------
# reference change and charge exchange
# ---------------------------------------------------------------------------
def change_reference(coords, *, old: RingReference, new: RingReference) -> np.ndarray:
    """Re-express coordinates relative to a new reference momentum (same species),
    keeping each particle's momentum and arrival time: delta' = (1+delta) p0/p0' - 1,
    px' = px p0/p0', zeta' = zeta beta0'/beta0.  Returns a new array."""
    X = np.array(coords, dtype=float, copy=True)
    r = old.p0c / new.p0c
    X[5] = (1.0 + X[5]) * r - 1.0
    X[1] *= r
    X[3] *= r
    X[4] *= new.beta0 / old.beta0
    return X


def strip_to_proton(coords, *, hminus_ref: RingReference, ring_ref: RingReference) -> np.ndarray:
    """H⁻ coordinates (relative to ``hminus_ref``) → proton coordinates relative to
    ``ring_ref`` (a proton reference): same velocity and direction, so the
    momentum scales by m_p/m_H⁻.  zeta keeps the same arrival time."""
    X = np.array(coords, dtype=float, copy=True)
    k = (ring_ref.mass / hminus_ref.mass) * (hminus_ref.p0c / ring_ref.p0c)
    X[5] = (1.0 + X[5]) * k - 1.0
    X[1] *= k
    X[3] *= k
    X[4] *= ring_ref.beta0 / hminus_ref.beta0
    return X
