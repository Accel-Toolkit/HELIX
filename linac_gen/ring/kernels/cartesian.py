# Element maps ported from Xsuite: xtrack v0.103.5, files
# beam_elements/elements_src/track_drift.h, track_magnet_drift.h,
# track_magnet_kick.h, track_magnet.h (integrators), track_dipole_edge_linear.h,
# track_yrotation.h, track_wedge.h, track_dipole_fringe.h, track_mult_fringe.h,
# track_srotation.h, track_xyshift.h, track_rf.h and
# particles/local_particle_custom_api.h.
#
# Copyright (c) CERN, 2021-2025 — the Xsuite authors.  Licensed under the
# Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  A copy of the License is in
# licenses/Xsuite-Apache-2.0.txt and at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# Modified for HELIX (2026): translated from C to numba-compatible Python on
# scalar coordinates; the Xsuite LocalParticle replaced by explicit arguments
# and return values; the charge/mass ratio chi dropped (every tracked particle
# has the reference species, chi = 1); the energy helpers and the straight
# exact bend's position update computed without cancellation (the same
# quantities as Xsuite's); ImpactX's symplectic integrators added to the
# magnet body (below).
#
# The symp2 / symp4 / symp6 branches of magnet_body are the step schedules of
# ImpactX's symplectic integrators (src/particles/integrators/Integrators.H,
# BLAST-ImpactX/impactx @ 472938df), translated from C++ to Python: Copyright
# 2022-2026 The Regents of the University of California, through Lawrence
# Berkeley National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy).  All rights reserved.  Authors: Chad
# Mitchell, Axel Huebl.  License: BSD-3-Clause-LBNL — the full text is in
# licenses/ImpactX-BSD-3-Clause-LBNL.txt.
"""Ring-mode element maps in the MAD-X / Xsuite conventions.

Coordinates are canonical, as in :mod:`linac_gen.ring.coords`:
x, y [m], px = Px/P0, py = Py/P0, zeta = s - beta0 c t [m], delta = (P - P0)/P0,
with the energy helpers ptau = (E - E0)/(P0 c), rpp = 1/(1 + delta) and
rvv = beta/beta0.  Lengths are in metres, magnet strengths normalized to the
reference rigidity (MAD-X units: k0 = h = 1/rho [1/m], k1 [1/m^2], ...).

Every function takes and returns plain floats, so one source runs as Python,
as multi-core CPU code and as CUDA device code (see :mod:`._jit`).  A map
whose square root goes negative (a particle at or beyond 90 degrees to the
axis) returns NaN; the tracking loop reports such a particle lost.
"""
from __future__ import annotations

from linac_gen.ring.kernels._jit import jitable
from linac_gen.ring.kernels._math import asin, atan, cos, cosh, fabs, sin, sinh, sqrt, tan

C_LIGHT = 299792458.0
PI = 3.1415926535897932384626433832795028841971693993751
DEG2RAD = 0.0174532925199432957692369076848861271344287188854

# Xsuite's integrator coefficients (from MAD-NG): Yoshida's sixth-order
# composition, seven kicks per slice (Xsuite calls it "Yoshida 4")
YD0 = 3.922568052387799819591407413100e-01
YD1 = 5.100434119184584780271052295575e-01
YD2 = -4.710533854097565531482416645304e-01
YD3 = 6.875316825251809316199569366290e-02
YK0 = 7.845136104775599639182814826199e-01
YK1 = 2.355732133593569921359289764951e-01
YK2 = -1.177679984178870098432412305556e+00
YK3 = 1.315186320683906284756403692882e+00

# integrator codes (shared with the ImpactX maps)
INT_TEAPOT = 1
INT_YOSHIDA = 2
INT_UNIFORM = 3
INT_SYMP2 = 4
INT_SYMP4 = 5
INT_SYMP6 = 6

# drift models of a thick magnet (Xsuite's numbering)
DM_NONE = -1
DM_DRIFT_EXPANDED = 0
DM_DRIFT_EXACT = 1
DM_POLAR = 2
DM_EXPANDED_CFD = 3
DM_CURVED_BEND = 4
DM_STRAIGHT_BEND = 5
DM_POLAR_YOSHIDA = 7

#: |Kx| L^2 below which :func:`expanded_cfd` takes a bend's combinations
#: divided by Kx from their series in Kx L^2: at the switch the direct forms
#: are good to 6e-13 relative (their cancellation), the series to rounding
CFD_SERIES_LIMIT = 2e-3


# ---------------------------------------------------------------------------
# energy helpers
# ---------------------------------------------------------------------------
@jitable
def energy_from_delta(delta, beta0):
    """(ptau, rpp, rvv) for the momentum deviation ``delta``."""
    inv_b0 = 1.0 / beta0
    opd = 1.0 + delta
    ptau = delta * (2.0 + delta) / (sqrt(opd * opd + inv_b0 * inv_b0 - 1.0) + inv_b0)
    return ptau, 1.0 / opd, opd / (1.0 + beta0 * ptau)


@jitable
def energy_from_ptau(ptau, beta0):
    """(delta, rpp, rvv) for the energy deviation ``ptau``."""
    u = ptau * ptau + 2.0 * ptau / beta0
    delta = u / (sqrt(1.0 + u) + 1.0)
    opd = 1.0 + delta
    return delta, 1.0 / opd, opd / (1.0 + beta0 * ptau)


# ---------------------------------------------------------------------------
# drifts
# ---------------------------------------------------------------------------
@jitable
def drift_expanded(x, px, y, py, zeta, rpp, rvv, length):
    """Paraxial drift (Xsuite ``Drift_single_particle_expanded``) → (x, y, zeta)."""
    rv0v = 1.0 / rvv
    xp = px * rpp
    yp = py * rpp
    dzeta = 1.0 - rv0v * (1.0 + (xp * xp + yp * yp) / 2.0)
    return x + xp * length, y + yp * length, zeta + length * dzeta


@jitable
def drift_exact(x, px, y, py, zeta, delta, rvv, length):
    """Exact drift (Xsuite ``Drift_single_particle_exact``) → (x, y, zeta)."""
    rv0v = 1.0 / rvv
    opd = 1.0 + delta
    one_over_pz = 1.0 / sqrt(opd * opd - px * px - py * py)
    dzeta = 1.0 - rv0v * opd * one_over_pz
    return x + px * one_over_pz * length, y + py * one_over_pz * length, zeta + dzeta * length


@jitable
def polar_drift(x, px, y, py, zeta, delta, rvv, length, h):
    """Exact drift in a frame of curvature h (Xsuite, from PTC's Sprotr and
    MAD-NG's curex_drift) → (x, px, y, zeta)."""
    opd = delta + 1.0
    pz = sqrt(opd * opd - px * px - py * py)
    rho = 1.0 / h
    ca = cos(h * length)
    sa = sin(h * length)
    sa2 = sin(0.5 * h * length)
    _pz = 1.0 / pz
    pxt = px * _pz
    _ptt = 1.0 / (ca - sa * pxt)
    pst = (x + rho) * sa * _pz * _ptt
    new_x = (x + rho * (2.0 * sa2 * sa2 + sa * pxt)) * _ptt
    new_px = ca * px + sa * pz
    new_y = y + pst * py
    delta_ell = opd * (x + rho) * sa / ca / pz / (1.0 - px * sa / ca / pz)
    return new_x, new_px, new_y, zeta + length - delta_ell / rvv


@jitable
def expanded_cfd(x, px, y, py, zeta, delta, rvv, length, k0_, k1_, h):
    """MAD-X's thick combined-function map, chromatic (k / (1 + delta)) with
    the second-order path length (Xsuite
    ``track_expanded_combined_dipole_quad_single_particle``) → (x, px, y, py, zeta).
    In a bend whose horizontal focusing Kx = k0 h + k1 nearly vanishes (a
    field index near 1), the terms divided by Kx come from their series in
    Kx L^2 (:data:`CFD_SERIES_LIMIT`): the direct forms divide a cancellation
    by Kx, and at Kx = 0 the series give the uniform-force limit, including
    the path-length terms in (h - k0)^2 and (h - k0) x' that the ported
    formula's Kx = 0 branch leaves out."""
    delta_plus_1 = delta + 1.0
    k0 = k0_ / delta_plus_1
    k1 = k1_ / delta_plus_1
    Kx = k0 * h + k1
    Ky = -k1
    if Kx > 0.0:
        sqrt_Kx = sqrt(Kx)
        Sx = sin(sqrt_Kx * length) / sqrt_Kx
        Cx = cos(sqrt_Kx * length)
    elif Kx < 0.0:
        sqrt_Kx = sqrt(-Kx)
        Sx = sinh(sqrt_Kx * length) / sqrt_Kx
        Cx = cosh(sqrt_Kx * length)
    else:
        Sx = length
        Cx = 1.0
    if Ky > 0.0:
        sqrt_Ky = sqrt(Ky)
        Sy = sin(sqrt_Ky * length) / sqrt_Ky
        Cy = cos(sqrt_Ky * length)
    elif Ky < 0.0:
        sqrt_Ky = sqrt(-Ky)
        Sy = sinh(sqrt_Ky * length) / sqrt_Ky
        Cy = cosh(sqrt_Ky * length)
    else:
        Sy = length
        Cy = 1.0
    xp = px / delta_plus_1
    yp = py / delta_plus_1
    A = -Kx * x - k0 + h
    B = xp
    C = -Ky * y
    D = yp
    x_ = x * Cx + xp * Sx
    y_ = y * Cy + yp * Sy
    px_ = (A * Sx + B * Cx) * delta_plus_1
    py_ = (C * Sy + D * Cy) * delta_plus_1
    length_ = length
    if (k0 != 0.0 or h != 0.0) and fabs(Kx) * (length * length) < CFD_SERIES_LIMIT:
        # a bend with |Kx| L^2 small or zero: c1 = (Cx - 1)/Kx,
        # s1 = (Sx - L)/Kx and q = (L - Cx Sx)/(2 Kx) by their series in
        # u = Kx L^2 (q's is s1's at 4u), and (1 - Cx^2)/Kx = Sx^2
        L2 = length * length
        u = Kx * L2
        v = 4.0 * u
        c1 = -0.5 * L2 * (
            1.0 - u / 12.0 * (1.0 - u / 30.0 * (1.0 - u / 56.0 * (1.0 - u / 90.0))))
        s1 = -(L2 * length) / 6.0 * (
            1.0 - u / 20.0 * (1.0 - u / 42.0 * (1.0 - u / 72.0 * (1.0 - u / 110.0))))
        q = (L2 * length) / 3.0 * (
            1.0 - v / 20.0 * (1.0 - v / 42.0 * (1.0 - v / 72.0 * (1.0 - v / 110.0))))
        x_ = x_ + (k0 - h) * c1
        length_ -= h * (xp * c1 - x * Sx + (h - k0) * s1)
        length_ += 0.5 * ((A * A) * q + (B * B) * (Cx * Sx + length) / 2.0 + A * B * (Sx * Sx))
    else:
        x_ = (x_ + (k0 - h) * (Cx - 1.0) / Kx if Kx != 0.0
              else x_ - (k0 - h) * 0.5 * (length * length))
        if Kx != 0.0:
            length_ -= (h * ((Cx - 1.0) * xp + Sx * A + length * (k0 - h))) / Kx
            length_ += 0.5 * (
                -((A * A) * Cx * Sx) / (2.0 * Kx)
                + ((B * B) * Cx * Sx) / 2.0
                + ((A * A) * length) / (2.0 * Kx)
                + ((B * B) * length) / 2.0
                - (A * B * (Cx * Cx)) / Kx
                + (A * B) / Kx)
        else:
            length_ += h * length * (3.0 * length * xp + 6.0 * x
                                     - (k0 - h) * (length * length)) / 6.0
            length_ += 0.5 * (B * B) * length
    if Ky != 0.0:
        length_ += 0.5 * (
            -((C * C) * Cy * Sy) / (2.0 * Ky)
            + ((D * D) * Cy * Sy) / 2.0
            + ((C * C) * length) / (2.0 * Ky)
            + ((D * D) * length) / 2.0
            - (C * D * (Cy * Cy)) / Ky
            + (C * D) / Ky)
    else:
        length_ += 0.5 * (D * D) * length
    return x_, px_, y_, py_, zeta + (length - length_ / rvv)


@jitable
def curved_exact_bend(x, px, y, py, zeta, delta, rvv, length, k0, h):
    """Exact map through a uniform field k0 in a frame of curvature h ≠ 0
    (Xsuite ``track_curved_exact_bend_single_particle``) → (x, px, y, zeta)."""
    if fabs(k0) < 1e-8:
        return polar_drift(x, px, y, py, zeta, delta, rvv, length, h)
    s = length
    opd = delta + 1.0
    A = 1.0 / sqrt(opd * opd - py * py)
    pz = sqrt(opd * opd - px * px - py * py)
    C = pz - k0 * ((1.0 / h) + x)
    new_px = px * cos(s * h) + C * sin(s * h)
    new_pz = sqrt(opd * opd - new_px * new_px - py * py)
    d_new_px_ds = C * h * cos(h * s) - h * px * sin(h * s)
    new_x = (new_pz * h - d_new_px_ds - k0) / (h * k0)
    D = asin(A * px) - asin(A * new_px)
    new_y = y + ((py * s) / (k0 / h)) + (py / k0) * D
    delta_ell = ((opd * s * h) / k0) + (opd / k0) * D
    return new_x, new_px, new_y, zeta + length - delta_ell / rvv


@jitable
def straight_exact_bend(x, px, y, py, zeta, delta, rvv, length, k0):
    """Exact map through a uniform field k0 in a straight frame (Xsuite
    ``track_straight_exact_bend_single_particle``) → (x, px, y, zeta).
    Xsuite's x' = x + (pz' - pz)/k0 is evaluated as
    x + length (px + px')/(pz' + pz) — the same quantity without the
    difference of two numbers near 1 divided by k0."""
    if fabs(k0) < 1e-8:
        nx, ny, nz = drift_exact(x, px, y, py, zeta, delta, rvv, length)
        return nx, px, ny, nz
    s = length
    opd = delta + 1.0
    A = 1.0 / sqrt(opd * opd - py * py)
    pz = sqrt(opd * opd - px * px - py * py)
    new_px = px - k0 * s
    new_pz = sqrt(opd * opd - new_px * new_px - py * py)
    new_x = x + s * (px + new_px) / (new_pz + pz)
    D = asin(A * px) - asin(A * new_px)
    new_y = y + (py / k0) * D
    delta_ell = (opd / k0) * D
    return new_x, new_px, new_y, zeta + length - delta_ell / rvv


@jitable
def magnet_drift(model, x, px, y, py, zeta, delta, rpp, rvv, length, k0, k1, h):
    """The drift part of a thick magnet (Xsuite
    ``track_magnet_drift_single_particle``; no solenoid) → (x, px, y, py, zeta)."""
    if model == DM_NONE or length == 0.0:
        return x, px, y, py, zeta
    if model == DM_DRIFT_EXPANDED:
        x, y, zeta = drift_expanded(x, px, y, py, zeta, rpp, rvv, length)
    elif model == DM_DRIFT_EXACT:
        x, y, zeta = drift_exact(x, px, y, py, zeta, delta, rvv, length)
    elif model == DM_POLAR:
        x, px, y, zeta = polar_drift(x, px, y, py, zeta, delta, rvv, length, h)
    elif model == DM_EXPANDED_CFD:
        x, px, y, py, zeta = expanded_cfd(x, px, y, py, zeta, delta, rvv, length, k0, k1, h)
    elif model == DM_CURVED_BEND:
        x, px, y, zeta = curved_exact_bend(x, px, y, py, zeta, delta, rvv, length, k0, h)
    elif model == DM_STRAIGHT_BEND:
        x, px, y, zeta = straight_exact_bend(x, px, y, py, zeta, delta, rvv, length, k0)
    elif model == DM_POLAR_YOSHIDA:
        # fourth-order Forest-Ruth of polar drifts and k0 kicks (MAD-NG coefficients)
        x, px, y, zeta = polar_drift(x, px, y, py, zeta, delta, rvv, 0.6756035959798289 * length, h)
        px = px - 1.3512071919596578 * k0 * length
        x, px, y, zeta = polar_drift(x, px, y, py, zeta, delta, rvv, -0.17560359597982889 * length, h)
        px = px - (-1.7024143839193155) * k0 * length
        x, px, y, zeta = polar_drift(x, px, y, py, zeta, delta, rvv, -0.17560359597982889 * length, h)
        px = px - 1.3512071919596578 * k0 * length
        x, px, y, zeta = polar_drift(x, px, y, py, zeta, delta, rvv, 0.6756035959798289 * length, h)
    return x, px, y, py, zeta


# ---------------------------------------------------------------------------
# kicks
# ---------------------------------------------------------------------------
@jitable
def kick_from_array(x, y, p, o_knl, o_ksl, order, inv_factorial, factor, kick_weight):
    """(dpx, dpy) of the multipole knl/ksl stored at p[o_knl..], p[o_ksl..]
    (Xsuite ``kick_simple_single_coordinates``, Horner in x + iy)."""
    index = order
    dpx_mul = p[o_knl + index] * factor * inv_factorial
    dpy_mul = p[o_ksl + index] * factor * inv_factorial
    while index > 0:
        zre = dpx_mul * x - dpy_mul * y
        zim = dpx_mul * y + dpy_mul * x
        inv_factorial *= index
        index -= 1
        this_knl = p[o_knl + index] * factor
        this_ksl = p[o_ksl + index] * factor
        dpx_mul = this_knl * inv_factorial + zre
        dpy_mul = this_ksl * inv_factorial + zim
    dpx_mul = -dpx_mul
    return kick_weight * dpx_mul, kick_weight * dpy_mul


@jitable
def kick_main(x, y, k0l, k1l, k2l, k3l, k0sl, k1sl, k2sl, k3sl, kick_weight):
    """(dpx, dpy) of the order-3 multipole (k0l..k3l, k0sl..k3sl) — Xsuite's
    "main strengths" kick, the same Horner scheme unrolled."""
    inv_factorial = 1.0 / (3 * 2)
    dpx_mul = k3l * 1.0 * inv_factorial
    dpy_mul = k3sl * 1.0 * inv_factorial
    zre = dpx_mul * x - dpy_mul * y
    zim = dpx_mul * y + dpy_mul * x
    inv_factorial *= 3
    dpx_mul = k2l * 1.0 * inv_factorial + zre
    dpy_mul = k2sl * 1.0 * inv_factorial + zim
    zre = dpx_mul * x - dpy_mul * y
    zim = dpx_mul * y + dpy_mul * x
    inv_factorial *= 2
    dpx_mul = k1l * 1.0 * inv_factorial + zre
    dpy_mul = k1sl * 1.0 * inv_factorial + zim
    zre = dpx_mul * x - dpy_mul * y
    zim = dpx_mul * y + dpy_mul * x
    inv_factorial *= 1
    dpx_mul = k0l * 1.0 * inv_factorial + zre
    dpy_mul = k0sl * 1.0 * inv_factorial + zim
    dpx_mul = -dpx_mul
    return kick_weight * dpx_mul, kick_weight * dpy_mul


# Parameter block of a thick magnet (offsets from the node's start)
MG_LENGTH = 0
MG_INTEGRATOR = 1
MG_NKICKS = 2
MG_DRIFT_MODEL = 3
MG_K0_DRIFT = 4
MG_K1_DRIFT = 5
MG_H_DRIFT = 6
MG_K0_KICK = 7
MG_K1_KICK = 8
MG_H_KICK = 9
MG_HXL = 10
MG_K0_HCORR = 11
MG_K1_HCORR = 12
MG_ROT_FRAME = 13
MG_K2 = 14
MG_K3 = 15
MG_K0S = 16
MG_K1S = 17
MG_K2S = 18
MG_K3S = 19
MG_ORDER = 20
MG_INV_FACT = 21
MG_FACTOR = 22
MG_KNL = 23            # knl[0..order], then ksl[0..order]
MG_SIZE_FIXED = 23


@jitable
def magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kick_weight):
    """One weighted kick of the thick magnet at p[o:] (Xsuite
    ``track_magnet_kick_single_particle``) → (px, py, zeta)."""
    length = p[o + MG_LENGTH]
    order = int(p[o + MG_ORDER])
    factor = p[o + MG_FACTOR]
    o_knl = o + MG_KNL
    o_ksl = o_knl + order + 1
    if order >= 0:
        dpx, dpy = kick_from_array(x, y, p, o_knl, o_ksl, order, p[o + MG_INV_FACT], factor,
                                   kick_weight)
        px += dpx
        py += dpy
    dpx, dpy = kick_main(x, y, p[o + MG_K0_KICK] * length, p[o + MG_K1_KICK] * length,
                         p[o + MG_K2] * length, p[o + MG_K3] * length,
                         p[o + MG_K0S] * length, p[o + MG_K1S] * length,
                         p[o + MG_K2S] * length, p[o + MG_K3S] * length, kick_weight)
    px += dpx
    py += dpy
    h = p[o + MG_H_KICK]
    hxl = p[o + MG_HXL]
    dpx = 0.0
    dpy = 0.0
    dzeta = 0.0
    if p[o + MG_ROT_FRAME] != 0.0:
        hl = h * length * kick_weight + hxl * kick_weight
        dpx += hl * (1.0 + delta)
        rv0v = 1.0 / rvv
        dzeta += -rv0v * hl * x
    htot = h
    if length != 0.0:
        htot += hxl / length
    # curvature terms of the Hamiltonian, H = h k0 x^2/2 + h k1 (x^3/3 - x y^2/2)
    # (MAD8 physics manual eq. 5.15)
    k0l_mult = 0.0
    if order >= 0:
        k0l_mult = p[o_knl] * factor
    dpx += -(p[o + MG_K0_HCORR] * length + k0l_mult) * kick_weight * htot * x
    k1l_mult = 0.0
    if order >= 1:
        k1l_mult = p[o_knl + 1] * factor
    k1c = p[o + MG_K1_HCORR] * length + k1l_mult
    dpx += htot * k1c * kick_weight * (-x * x + 0.5 * y * y)
    dpy += htot * k1c * kick_weight * x * y
    return px + dpx, py + dpy, zeta + dzeta


@jitable
def _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, dlength):
    return magnet_drift(int(p[o + MG_DRIFT_MODEL]), x, px, y, py, zeta, delta, rpp, rvv, dlength,
                        p[o + MG_K0_DRIFT], p[o + MG_K1_DRIFT], p[o + MG_H_DRIFT])


@jitable
def magnet_body(p, o, x, px, y, py, zeta, delta, rpp, rvv):
    """The body of the thick magnet at p[o:]: drift and kick composed by the
    chosen integrator (Xsuite ``track_magnet_body_single_particle``; the
    symplectic integrators of order 2, 4 and 6 as in ImpactX) →
    (x, px, y, py, zeta)."""
    length = p[o + MG_LENGTH]
    integrator = int(p[o + MG_INTEGRATOR])
    nk = int(p[o + MG_NKICKS])
    if (nk == 0 and p[o + MG_K0_KICK] == 0.0 and p[o + MG_K1_KICK] == 0.0
            and p[o + MG_H_KICK] == 0.0):
        return _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, length)
    if integrator == INT_TEAPOT:
        kick_weight = 1.0 / nk
        edge_drift_weight = 0.5
        inside_drift_weight = 0.0
        if nk > 1:
            edge_drift_weight = 1.0 / (2 * (1 + nk))
            inside_drift_weight = float(nk) / (float(nk * nk) - 1.0)
        x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv,
                                     edge_drift_weight * length)
        for _i in range(nk - 1):
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kick_weight)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv,
                                         inside_drift_weight * length)
        px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kick_weight)
        x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv,
                                     edge_drift_weight * length)
    elif integrator == INT_UNIFORM:
        kick_weight = 1.0 / nk
        drift_weight = kick_weight
        for _i in range(nk):
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv,
                                         0.5 * drift_weight * length)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kick_weight)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv,
                                         0.5 * drift_weight * length)
    elif integrator == INT_YOSHIDA:
        num_slices = nk // 7 + (1 if nk % 7 != 0 else 0)
        sl = length / num_slices
        kw = 1.0 / num_slices
        for _i in range(num_slices):
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD0)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kw * YK0)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD1)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kw * YK1)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD2)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kw * YK2)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD3)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kw * YK3)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD3)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kw * YK2)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD2)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kw * YK1)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD1)
            px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, kw * YK0)
            x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, sl * YD0)
    else:
        # ImpactX's symplectic integrators: map1 = the drift, map2 = the kick
        # (a kick of weight w is the force applied over w × length)
        dz = length / nk
        for _i in range(nk):
            if integrator == INT_SYMP2:
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, dz * 0.5)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, dz / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, dz * 0.5)
            elif integrator == INT_SYMP4:
                alpha = -0.2599210498948732
                tau2 = dz / (1.0 + alpha)
                tau1 = tau2 * 0.5
                tau3 = alpha * tau1
                tau4 = (alpha - 1.0) * tau2
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, tau1)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, tau2 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, tau3)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, tau4 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, tau3)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, tau2 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, tau1)
            else:
                w0 = 1.31518632068391121888424973
                w1 = -1.17767998417887100694641568
                w2 = 0.23557321335935813368479318
                w3 = 0.78451361047755726381949763
                t1 = w3 * dz / 2.0
                t2 = w3 * dz
                t3 = (w3 + w2) * dz / 2.0
                t4 = w2 * dz
                t5 = (w2 + w1) * dz / 2.0
                t6 = w1 * dz
                t7 = (w1 + w0) * dz / 2.0
                t8 = w0 * dz
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t1)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, t2 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t3)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, t4 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t5)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, t6 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t7)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, t8 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t7)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, t6 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t5)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, t4 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t3)
                px, py, zeta = magnet_kick(p, o, x, px, y, py, zeta, delta, rvv, t2 / length)
                x, px, y, py, zeta = _mdrift(p, o, x, px, y, py, zeta, delta, rpp, rvv, t1)
    return x, px, y, py, zeta


# ---------------------------------------------------------------------------
# edges and frame changes
# ---------------------------------------------------------------------------
@jitable
def edge_linear_coefficients(k, e1, e1_fd, hgap, fint):
    """(r21, r43) of the linear edge (Xsuite ``compute_dipole_edge_linear_coefficients``)."""
    corr = 2.0 * k * hgap * fint
    r21 = k * tan(e1)
    e1_v = e1 + e1_fd
    sin_e1_v = sin(e1_v)
    temp = corr / cos(e1_v) * (1.0 + sin_e1_v * sin_e1_v)
    r43 = -k * tan(e1_v - temp)
    return r21, r43


@jitable
def yrotation(x, px, y, py, zeta, ptau, beta0, sin_angle, cos_angle, tan_angle):
    """Exact rotation of the frame about y (Xsuite ``YRotation_single_particle``)
    → (x, px, y, zeta)."""
    t = zeta / beta0
    pt = ptau
    pz = sqrt(1.0 + 2.0 * pt / beta0 + pt * pt - px * px - py * py)
    ptt = 1.0 + tan_angle * px / pz
    x_hat = x / (cos_angle * ptt)
    px_hat = cos_angle * px - sin_angle * pz
    y_hat = y - tan_angle * x * py / (pz * ptt)
    t_hat = t + tan_angle * x * (1.0 / beta0 + pt) / (pz * ptt)
    return x_hat, px_hat, y_hat, t_hat * beta0


@jitable
def wedge(x, px, y, py, zeta, delta, ptau, rvv, beta0, theta, k0):
    """Exact map through the wedge of uniform dipole field between a rotated
    pole face and the sector boundary (Xsuite ``Wedge_single_particle``) →
    (x, px, y, zeta)."""
    b1 = k0
    if fabs(b1) < 10e-10:
        return yrotation(x, px, y, py, zeta, ptau, beta0, -sin(theta), cos(theta), -tan(theta))
    opd = delta + 1.0
    A = 1.0 / sqrt(opd * opd - py * py)
    pz = sqrt(opd * opd - px * px - py * py)
    new_px = px * cos(theta) + (pz - b1 * x) * sin(theta)
    new_pz = sqrt(opd * opd - new_px * new_px - py * py)
    st = sin(theta)
    new_x = x * cos(theta) + (x * px * sin(2.0 * theta) + (st * st) * (2.0 * x * pz - b1 * (x * x))) \
        / (new_pz + pz * cos(theta) - px * sin(theta))
    D = asin(A * px) - asin(A * new_px)
    delta_y = py * (theta + D) / b1
    delta_ell = opd * (theta + D) / b1
    return new_x, new_px, y + delta_y, zeta - delta_ell / rvv


@jitable
def quad_wedge(x, px, y, py, theta, k1):
    """The quadrupole field of the wedge, to first order in its angle (Xsuite
    ``Quad_wedge_single_particle``) → (px, py)."""
    b2 = k1
    new_px = px - b2 * x * x * theta + b2 * y * y / 2.0 * theta
    new_py = py + b2 * x * y * theta
    return new_px, new_py


@jitable
def dipole_fringe(x, px, y, py, zeta, delta, ptau, beta0, fint, hgap, k0):
    """Hard-edge dipole fringe with the fringe integral fint and half gap hgap
    (Xsuite ``DipoleFringe_single_particle``, from MAD-NG) → (x, y, py, zeta)."""
    if fabs(k0) < 10e-10:
        return x, y, py, zeta
    t = zeta / beta0
    pt = ptau
    fh = hgap * fint
    fsad = 1.0 / (72 * fh) if fh > 10e-10 else 0.0
    b0 = k0
    _beta = 1.0 / beta0
    dpp = (1.0 + delta) * (1.0 + delta)
    pz = sqrt(dpp - px * px - py * py)
    _pz = 1.0 / pz
    relp = 1.0 / sqrt(dpp)
    tfac = -(_beta + pt)
    c2 = b0 * fh * 2
    c3 = (b0 * b0) * fsad * relp
    xp = px / pz
    yp = py / pz
    xyp = xp * yp
    yp2 = 1.0 + yp * yp
    xp2 = xp * xp
    _yp2 = 1.0 / yp2
    fi0 = atan(xp * _yp2) - c2 * (1 + xp2 * (1 + yp2)) * pz
    cf = cos(fi0)
    co2 = b0 / (cf * cf)
    u = xp * _yp2
    co1 = co2 / (1 + u * u) * _yp2
    co3 = co2 * c2
    fi1 = co1 - co3 * 2 * xp * (1 + yp2) * pz
    fi2 = -2 * co1 * xyp * _yp2 - co3 * 2 * xp * xyp * pz
    fi3 = -co3 * (1 + xp2 * (1 + yp2))
    kx = fi1 * (1 + xp2) * _pz + fi2 * xyp * _pz - fi3 * xp
    ky = fi1 * xyp * _pz + fi2 * yp2 * _pz - fi3 * yp
    kz = (fi1 * tfac * xp * (_pz * _pz) + fi2 * tfac * yp * (_pz * _pz)
          - fi3 * tfac * _pz)
    new_y = 2 * y / (1 + sqrt(1 - 2 * ky * y))
    new_x = x + 0.5 * kx * (new_y * new_y)
    new_py = py - 4 * c3 * (new_y * new_y * new_y) - b0 * tan(fi0) * new_y
    new_t = (t + 0.5 * kz * (new_y * new_y)
             + c3 * (new_y * new_y * new_y * new_y) * (relp * relp) * tfac)
    return new_x, new_y, new_py, new_t * beta0


@jitable
def _pick4(i, a, b, c, d):
    if i == 0:
        return a
    if i == 1:
        return b
    if i == 2:
        return c
    return d


@jitable
def mult_fringe(x, px, y, py, zeta, ptau, rpp, beta0, kn0, kn1, kn2, kn3, ks0, ks1, ks2, ks3,
                p, o_knl, o_ksl, kl_order, length, is_exit, min_order):
    """Hard-edge fringe of the multipoles kn/ks (order 3) and knl/ksl at
    p[o_knl..] (Xsuite ``MultFringe_track_single_particle``, from MAD-NG) →
    (x, px, y, py, zeta)."""
    direction = -1.0 if is_exit else 1.0
    t = zeta / beta0
    pt = ptau
    rx = 1.0
    ix = 0.0
    fx = 0.0
    fxx = 0.0
    fxy = 0.0
    fy = 0.0
    fyx = 0.0
    fyy = 0.0
    order = 3 if kl_order < 3 else kl_order
    inv_factorial = 1.0
    for ii in range(order + 1):
        if ii > 1:
            inv_factorial /= ii
        component = ii + 1.0
        drx = rx
        dix = ix
        rx = drx * x - dix * y
        ix = drx * y + dix * x
        kn_total = 0.0
        ks_total = 0.0
        if ii >= min_order:
            if ii <= 3:
                kn_total += _pick4(ii, kn0, kn1, kn2, kn3) * inv_factorial
                ks_total += _pick4(ii, ks0, ks1, ks2, ks3) * inv_factorial
            if ii <= kl_order and length != 0.0:
                kn_total += p[o_knl + ii] / length * inv_factorial
                ks_total += p[o_ksl + ii] / length * inv_factorial
        nj = -direction / (4 * (component + 1))
        nf = (component + 2) / component
        kj = kn_total
        ksj = ks_total
        if ii == 0:
            u = nj * (-ksj * ix)
            v = nj * (ksj * rx)
            du = nj * (-ksj * dix)
            dv = nj * (ksj * drx)
        else:
            u = nj * (kj * rx - ksj * ix)
            v = nj * (kj * ix + ksj * rx)
            du = nj * (kj * drx - ksj * dix)
            dv = nj * (kj * dix + ksj * drx)
        dux = component * du
        dvx = component * dv
        duy = -component * dv
        dvy = component * du
        fx = fx + u * x + nf * v * y
        fy = fy + u * y - nf * v * x
        fxx = fxx + dux * x + nf * dvx * y + u
        fyy = fyy + duy * y - nf * dvy * x + u
        fxy = fxy + duy * x + nf * (dvy * y + v)
        fyx = fyx + dux * y - nf * (dvx * x + v)
    a = 1 - fxx * rpp
    b = -fyx * rpp
    c = -fxy * rpp
    d = 1 - fyy * rpp
    det = (a * d - b * c)
    new_px = (d * px - b * py) / det
    new_py = (a * py - c * px) / det
    delta_t = (1 / beta0 + pt) * (new_px * fx + new_py * fy) * (rpp * rpp * rpp)
    return x - fx * rpp, new_px, y - fy * rpp, new_py, (t + delta_t) * beta0


@jitable
def srotation(x, px, y, py, sin_z, cos_z):
    """Rotation of the frame about s (Xsuite ``SRotation_single_particle``) →
    (x, px, y, py)."""
    return (cos_z * x + sin_z * y, cos_z * px + sin_z * py,
            -sin_z * x + cos_z * y, -sin_z * px + cos_z * py)


@jitable
def xyshift(x, y, dx, dy):
    """Shift of the frame by (dx, dy) (Xsuite ``XYShift_single_particle``)."""
    return x - dx, y - dy


# ---------------------------------------------------------------------------
# RF
# ---------------------------------------------------------------------------
@jitable
def cavity_kick(zeta, ptau, beta0, p0c, voltage, frequency, lag_deg):
    """New ptau after a thin cavity of peak voltage ``voltage`` [MV for a unit
    charge, p0c in MeV], ``frequency`` [Hz] and ``lag_deg``: the energy gain is
    V sin(lag - 2 pi f zeta / (beta0 c)), relative to the reference particle
    (Xsuite ``track_rf_kick_single_particle``; px, py unchanged)."""
    tau = zeta / beta0
    energy_kick = voltage * sin(DEG2RAD * lag_deg - (2.0 * PI) / C_LIGHT * frequency * tau)
    return ptau + energy_kick / p0c
