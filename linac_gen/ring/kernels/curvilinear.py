# Element maps ported from ImpactX (BLAST-ImpactX/impactx, development branch
# @ 472938df of 2026-05-14, version 26.04+): src/elements/ExactDrift.H,
# ExactQuad.H, ExactMultipole.H, ExactCFbend.H, DipEdge.H (linear model) and
# src/particles/integrators/Integrators.H; checked against the ImpactX 26.08
# release (tests/ring/test_curvilinear_reference.py).
#
# Copyright 2022-2026 The Regents of the University of California, through
# Lawrence Berkeley National Laboratory (subject to receipt of any required
# approvals from the U.S. Dept. of Energy).  All rights reserved.
# Authors: Chad Mitchell, Axel Huebl.  License: BSD-3-Clause-LBNL — the full
# text is in licenses/ImpactX-BSD-3-Clause-LBNL.txt.
#
# Modified for HELIX (2026): translated from C++ to numba-compatible Python on
# scalar coordinates; the maps keep ImpactX's own variables (x, px, y, py, t,
# pt) with t = c (time - reference time) and pt = -(E - E0)/(P0 c), and the
# calling node converts from and to the ring coordinates (zeta = -beta0 t,
# ptau = -pt); the sector bend (map1 of ExactCFbend) and the curvilinear
# multipole kicks evaluated in a regrouped form without cancellation (the same
# quantities), NaN where the bend map is not defined; a dipole-like
# curvilinear term (ImpactX's own formula, unused there) carries a bend field
# error.
"""Ring-mode element maps in the ImpactX conventions ("curvilinear" model).

ImpactX integrates the exact Hamiltonian of each magnet with a symplectic
splitting H = H1 + H2 (``map1``/``map2``) of order 2, 4 or 6.  Its combined-
function bend uses ideal *curvilinear* multipoles (the field expansion that
satisfies Maxwell's equations in the curved frame), which differs from the
MAD-X expansion at second order — the ``cartesian`` / ``curvilinear`` difference
of the chromaticity (docs: ring mode, element models).
"""
from __future__ import annotations

from linac_gen.ring.kernels._jit import jitable
from linac_gen.ring.kernels._math import asin, cos, cosh, log1p, nan, sin, sinh, sqrt, tan

# ImpactX's integrator constants (Integrators.H)
S4_ALPHA = -0.2599210498948732
S6_W0 = 1.31518632068391121888424973
S6_W1 = -1.17767998417887100694641568
S6_W2 = 0.23557321335935813368479318
S6_W3 = 0.78451361047755726381949763


@jitable
def exact_drift(x, px, y, py, t, pt, tau, ibeta, ibetgam2):
    """ImpactX ExactDrift (and the map1 of ExactMultipole) → (x, y, t)."""
    inv_pzden = 1.0 / sqrt((pt - ibeta) * (pt - ibeta) - ibetgam2 - px * px - py * py)
    return (x + tau * px * inv_pzden, y + tau * py * inv_pzden,
            t - tau * (ibeta + (pt - ibeta) * inv_pzden))


# ---------------------------------------------------------------------------
# ExactQuad: map1 = the linear quadrupole (on-momentum), map2 = the rest of the
# exact kinematics
# ---------------------------------------------------------------------------
@jitable
def quad_map1(x, px, y, py, t, pt, tau, g, omega, betgam2):
    sin_o = sin(omega * tau)
    cos_o = cos(omega * tau)
    sinh_o = sinh(omega * tau)
    cosh_o = cosh(omega * tau)
    slice_bg = tau / betgam2
    if g > 0.0:
        xo = cos_o * x + sin_o / omega * px
        pxo = -omega * sin_o * x + cos_o * px
        yo = cosh_o * y + sinh_o / omega * py
        pyo = omega * sinh_o * y + cosh_o * py
    elif g < 0.0:
        xo = cosh_o * x + sinh_o / omega * px
        pxo = omega * sinh_o * x + cosh_o * px
        yo = cos_o * y + sin_o / omega * py
        pyo = -omega * sin_o * y + cos_o * py
    else:
        xo = x + tau * px
        pxo = px
        yo = y + tau * py
        pyo = py
    return xo, pxo, yo, pyo, t + slice_bg * pt


@jitable
def quad_map2(x, px, y, py, t, pt, tau, beta, ibeta, ibetgam2):
    inv_pzden = 1.0 / sqrt((pt - ibeta) * (pt - ibeta) - ibetgam2 - px * px - py * py)
    return (x + tau * px * (-1.0 + inv_pzden), px, y + tau * py * (-1.0 + inv_pzden), py,
            t + tau * (-ibeta - ibetgam2 * pt + inv_pzden * (1.0 - pt * beta) * ibeta))


# ---------------------------------------------------------------------------
# ExactMultipole: map2 = the multipole kick (k_normal/k_skew in MAD-X units,
# factorial convention: dpx - i dpy = -sum (kn + i ks) (x + i y)^m / m!)
# ---------------------------------------------------------------------------
@jitable
def multipole_kick(x, y, p, o_kn, o_ks, ncoef):
    zr = 1.0                       # (x + i y)^m, by repeated multiplication
    zi = 0.0
    dpx = 0.0
    dpy = 0.0
    factorial = 1
    for m in range(ncoef):
        kn = p[o_kn + m]
        ks = p[o_ks + m]
        kr = zr * kn - zi * ks
        ki = zr * ks + zi * kn
        dpx += -1.0 * kr / factorial
        dpy += ki / factorial
        factorial *= m + 1
        nzr = zr * x - zi * y
        zi = zr * y + zi * x
        zr = nzr
    return dpx, dpy


# ---------------------------------------------------------------------------
# ExactCFbend: map1 = the exact sector bend (field = curvature), map2 = the
# ideal curvilinear multipoles
# ---------------------------------------------------------------------------
@jitable
def cfbend_map1(x, px, y, py, t, pt, tau, rc, ibeta, ibetgam2):
    """ImpactX ExactCFbend map1, the exact sector bend whose field is its
    curvature → (x, px, y, t); NaN outside the map's domain (ImpactX leaves
    such a particle unmoved with a commented-out warning; ring mode reports
    it lost).

    The same expressions as ImpactX's, regrouped so that no difference of
    terms of order rc (~40 m) is formed: ImpactX's
    x' = -rc + (rc + x) cos phi + rc (pzf + px sin phi - pzi cos phi) loses
    ~1e-14 m per call to rounding; here x' = x cos phi + rc [(pzf - pzi)
    + px sin phi + (pzi - 1)(1 - cos phi)] with each bracket evaluated from
    small quantities, and t likewise."""
    if rc == 0.0:
        inv_pzden = 1.0 / sqrt((pt - ibeta) * (pt - ibeta) - ibetgam2 - px * px - py * py)
        return (x + tau * px * inv_pzden, px, y + tau * py * inv_pzden,
                t - tau * (ibeta + (pt - ibeta) * inv_pzden))
    pperp2_m1 = pt * pt - 2.0 * ibeta * pt - py * py          # pperp^2 - 1
    pperp2 = pperp2_m1 + 1.0
    px2 = px * px
    slice_phi = tau / rc
    sin_phi = sin(slice_phi)
    cos_phi = cos(slice_phi)
    s_half = sin(0.5 * slice_phi)
    one_m_cos = 2.0 * s_half * s_half                         # 1 - cos phi
    if not pperp2 > px2:
        return nan, nan, nan, nan
    pperp = sqrt(pperp2)
    pzi = sqrt(pperp2 - px2)
    pzi_m1 = (pperp2_m1 - px2) / (pzi + 1.0)                  # pzi - 1
    # px out: px cos phi + (pzi - rho/rc) sin phi, rho/rc = 1 + x/rc
    pxout = px * cos_phi + (pzi_m1 - x / rc) * sin_phi
    px2f = pxout * pxout
    if not pperp2 > px2f:
        return nan, nan, nan, nan
    pzf = sqrt(pperp2 - px2f)
    dpz = (px2 - px2f) / (pzf + pzi)                          # pzf - pzi
    dasin = asin(px / pperp) - asin(pxout / pperp)
    theta = slice_phi + dasin
    xout = x * cos_phi + rc * (dpz + px * sin_phi + pzi_m1 * one_m_cos)
    yout = y + theta * rc * py
    # t - theta rc (pt - 1/beta) - phi rc / beta = t - rc (theta pt - (theta - phi)/beta)
    tout = t - rc * (theta * pt - ibeta * dasin)
    return xout, pxout, yout, tout


@jitable
def log1p_minus(u):
    """log(1 + u) - u, without the cancellation (a series for small u)."""
    if abs(u) < 0.05:
        # -u^2/2 + u^3/3 - ... ; 15 terms: below 1e-18 relative at |u| = 0.05
        term = -u * u
        acc = 0.0
        for k in range(2, 17):
            acc += term / k
            term = -term * u
        return acc
    return log1p(u) - u


@jitable
def cfbend_kick(x, y, rc, p, o_kn, o_ks, ncoef, m0):
    """(dpx, dpy) per unit length of the curvilinear multipoles m0..ncoef-1
    (m = 1 quadrupole ... 4 decapole; m = 0, ImpactX's "dipole-like" term, only
    for a field error).

    ImpactX's expressions in rho = 1 + u, u = x/rc, regrouped in u with
    L = log(1 + u) and Lm = L - u computed accurately: ImpactX's own
    log(rho) and (1 - rho^2 + 2 log rho)-like sums cancel to O(u) and O(u^2)
    of terms of order one, losing ~1e-12 of the kick to rounding."""
    u = x / rc
    rho = 1.0 + u
    ys = y / rc
    y2 = ys * ys
    L = log1p(u)
    Lm = log1p_minus(u)
    r = u * (2.0 + u)                         # rho^2 - 1
    u2 = u * u
    dpx = 0.0
    dpy = 0.0
    for m in range(m0, ncoef):
        kn = (rc ** m) * p[o_kn + m]
        ks = (rc ** m) * p[o_ks + m]
        n = m + 1
        if n == 1:
            dpx += -rho * kn
            dpy += ks
        elif n == 2:
            dpx += -rho * L * kn + ys * rho * ks
            dpy += ys * kn + 0.5 * r * ks
        elif n == 3:
            # 1 + 2y^2 - rho^2 + 2L = 2y^2 + 2Lm - u^2
            dpx += 0.25 * rho * (2.0 * y2 + 2.0 * Lm - u2) * kn + ys * rho * L * ks
            # 1 - 2y^2 - rho^2 + 2 rho^2 L = -2y^2 + 2Lm - u^2 + 2uL(2 + u)
            dpy += 0.5 * ys * r * kn \
                + 0.25 * (-2.0 * y2 + 2.0 * Lm - u2 + 2.0 * u * L * (2.0 + u)) * ks
        elif n == 4:
            # rho - rho^3 + (rho - 2y^2 rho + rho^3) L = rho (2Lm - u^2 + rL - 2y^2 L)
            dpx += -0.25 * rho * (2.0 * Lm - u2 + r * L - 2.0 * y2 * L) * kn \
                - 1.0 / 12.0 * ys * rho * (6.0 * Lm - 3.0 * u2 + 2.0 * y2) * ks
            # -3 + 2y^2 + 3rho^2 - 6 rho^2 L = -6Lm + 3u^2 - 6uL(2 + u) + 2y^2
            # (-1 + 4y^2 - rho^2)(rho^2 - 1) + 4 rho^2 L
            #     = 4Lm - 2u^2 + 4uL(2 + u) - r^2 + 4y^2 r
            dpy += -1.0 / 12.0 * ys * (-6.0 * Lm + 3.0 * u2 - 6.0 * u * L * (2.0 + u)
                                       + 2.0 * y2) * kn \
                - 1.0 / 16.0 * (4.0 * Lm - 2.0 * u2 + 4.0 * u * L * (2.0 + u) - r * r
                                + 4.0 * y2 * r) * ks
        elif n == 5:
            y4 = y2 * y2
            # 8y^4 - 24y^2(rho^2 - 1) + 3(-5 + 4rho^2 + rho^4) + 12(-1 + 4y^2 - 2rho^2) L
            #     = 8y^4 - 24y^2 r - 36Lm + 18u^2 + 3r^2 - 24rL + 48y^2 L
            dpx += -1.0 / 192.0 * rho * (8.0 * y4 - 24.0 * y2 * r - 36.0 * Lm + 18.0 * u2
                                         + 3.0 * r * r - 24.0 * r * L + 48.0 * y2 * L) * kn \
                + 1.0 / 12.0 * ys * rho * (-6.0 * Lm + 3.0 * u2 - 3.0 * r * L
                                           + 2.0 * y2 * L) * ks
            # (rho^2 - 1)(4y^2 - 3(1 + rho^2)) + 12 rho^2 L
            #     = 4y^2 r + 12Lm - 6u^2 - 3r^2 + 12rL
            # 3 + 8y^4 + 12rho^2 - 15rho^4 + 24y^2(rho^2 - 1) + 12rho^2(2 - 4y^2 + rho^2) L
            #     = 36Lm - 18u^2 - 15r^2 + 48rL + 12r^2 L + 8y^4 + 24y^2 r - 48y^2 (1 + r) L
            dpy += -1.0 / 48.0 * ys * (4.0 * y2 * r + 12.0 * Lm - 6.0 * u2 - 3.0 * r * r
                                       + 12.0 * r * L) * kn \
                + 1.0 / 192.0 * (36.0 * Lm - 18.0 * u2 - 15.0 * r * r + 48.0 * r * L
                                 + 12.0 * r * r * L + 8.0 * y4 + 24.0 * y2 * r
                                 - 48.0 * y2 * (1.0 + r) * L) * ks
    return dpx, dpy


# Parameter blocks: thick ImpactX element at p[o:]
CV_LENGTH = 0
CV_INTEGRATOR = 1          # 4 symp2, 5 symp4, 6 symp6
CV_NSTEPS = 2
CV_KIND = 3                # 0 ExactQuad, 1 ExactMultipole, 2 ExactCFbend
CV_G = 4                   # ExactQuad: k1;  ExactCFbend: rc;  ExactMultipole: unused
CV_NCOEF = 5               # ExactMultipole / ExactCFbend: number of coefficients
CV_M0 = 6                  # ExactCFbend: first multipole index (0 with a field error, else 1)
CV_COEF = 7                # k_normal[0..ncoef-1], then k_skew[0..ncoef-1]

KIND_QUAD = 0
KIND_MULTIPOLE = 1
KIND_CFBEND = 2


@jitable
def _map1(kind, p, o, x, px, y, py, t, pt, tau, beta, ibeta, ibetgam2, betgam2):
    if kind == KIND_QUAD:
        g = p[o + CV_G]
        omega = sqrt(abs(g))
        return quad_map1(x, px, y, py, t, pt, tau, g, omega, betgam2)
    if kind == KIND_MULTIPOLE:
        xo, yo, to = exact_drift(x, px, y, py, t, pt, tau, ibeta, ibetgam2)
        return xo, px, yo, py, to
    xo, pxo, yo, to = cfbend_map1(x, px, y, py, t, pt, tau, p[o + CV_G], ibeta, ibetgam2)
    return xo, pxo, yo, py, to


@jitable
def _map2(kind, p, o, x, px, y, py, t, pt, tau, beta, ibeta, ibetgam2):
    if kind == KIND_QUAD:
        return quad_map2(x, px, y, py, t, pt, tau, beta, ibeta, ibetgam2)
    ncoef = int(p[o + CV_NCOEF])
    o_kn = o + CV_COEF
    o_ks = o_kn + ncoef
    if kind == KIND_MULTIPOLE:
        dpx, dpy = multipole_kick(x, y, p, o_kn, o_ks, ncoef)
    else:
        dpx, dpy = cfbend_kick(x, y, p[o + CV_G], p, o_kn, o_ks, ncoef, int(p[o + CV_M0]))
    return x, px + tau * dpx, y, py + tau * dpy, t


@jitable
def thick_element(p, o, x, px, y, py, t, pt, beta, betgam2):
    """An ImpactX thick element (ExactQuad, ExactMultipole or ExactCFbend) at
    p[o:], integrated with ImpactX's symplectic integrator of order 2, 4 or 6
    → (x, px, y, py, t)."""
    kind = int(p[o + CV_KIND])
    integrator = int(p[o + CV_INTEGRATOR])
    nsteps = int(p[o + CV_NSTEPS])
    ibeta = 1.0 / beta
    ibetgam2 = 1.0 / betgam2
    dz = (p[o + CV_LENGTH] - 0.0) / float(nsteps)
    if integrator == 4:
        tau1 = dz * 0.5
        tau2 = dz
        for _j in range(nsteps):
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau1, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau2, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau1, beta, ibeta, ibetgam2, betgam2)
    elif integrator == 5:
        alpha = S4_ALPHA
        tau2 = dz / (1.0 + alpha)
        tau1 = tau2 * 0.5
        tau3 = alpha * tau1
        tau4 = (alpha - 1.0) * tau2
        for _j in range(nsteps):
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau1, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau2, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau3, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau4, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau3, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau2, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau1, beta, ibeta, ibetgam2, betgam2)
    else:
        tau1 = S6_W3 * dz / 2.0
        tau2 = S6_W3 * dz
        tau3 = (S6_W3 + S6_W2) * dz / 2.0
        tau4 = S6_W2 * dz
        tau5 = (S6_W2 + S6_W1) * dz / 2.0
        tau6 = S6_W1 * dz
        tau7 = (S6_W1 + S6_W0) * dz / 2.0
        tau8 = S6_W0 * dz
        for _j in range(nsteps):
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau1, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau2, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau3, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau4, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau5, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau6, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau7, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau8, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau7, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau6, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau5, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau4, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau3, beta, ibeta, ibetgam2, betgam2)
            x, px, y, py, t = _map2(kind, p, o, x, px, y, py, t, pt, tau2, beta, ibeta, ibetgam2)
            x, px, y, py, t = _map1(kind, p, o, x, px, y, py, t, pt, tau1, beta, ibeta, ibetgam2, betgam2)
    return x, px, y, py, t


@jitable
def dipedge_linear_coefficients(psi, rc, g, K2):
    """(R21, R43) of ImpactX's linear DipEdge (first order in the gap)."""
    R21 = tan(psi) / rc
    sp = sin(psi)
    cp = cos(psi)
    vf = (1.0 + sp * sp) / (cp * cp * cp) * g * K2 / (rc * rc)
    return R21, vf - R21
