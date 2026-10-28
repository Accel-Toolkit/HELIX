# exact_edge (below) is adapted from Xsuite: xtrack v0.103.5,
# beam_elements/elements_src/track_magnet_edge.h (track_magnet_edge_particles,
# edge models 1 and 2).  Copyright (c) CERN, 2025 — the Xsuite authors.
# Licensed under the Apache License, Version 2.0; a copy is in
# licenses/Xsuite-Apache-2.0.txt and at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.
# Modified for HELIX (2026): translated from C to numba-compatible Python on
# scalar coordinates, its parameters read from a node's parameter block; a
# choice of quadrupole wedge added (Xsuite's, or MAD8's as in MAD-X and PTC —
# quad_wedge_mad8, written from its formula) and the multipole fringe made a
# switch.  The rest of this file is HELIX's own.
"""A ring as a table of nodes, and the per-particle loop over it.

The compiler (:mod:`linac_gen.ring.compile`) turns a HELIX lattice into
``kinds[j]`` (the node kind codes below), ``offsets[j]`` (where node j's
parameters start in the flat float64 buffer ``params``) and the reference at
each node (``ref_beta0``, ``ref_p0c``, ``ref_bg2`` = (beta0 gamma0)^2).  One
function, :func:`track_node`, advances a particle through any node; the CPU
and CUDA drivers loop it over particles and nodes.

A particle is lost at an aperture node it is outside of (cause
``LOSS_APERTURE``) or at a node whose map is not defined for it — a square
root of a negative number, i.e. moving at or beyond 90 degrees to the axis
(``LOSS_INVALID``); its coordinates are then those at the entrance of that
node.
"""
from __future__ import annotations

from math import fabs, isfinite

from linac_gen.ring.kernels import cartesian as _cart
from linac_gen.ring.kernels import curvilinear as _curv
from linac_gen.ring.kernels._jit import jitable

# ---- node kinds -------------------------------------------------------------
N_MARKER = 0          # []
N_DRIFT = 1           # [length, exact (0/1)]
N_MAGNET = 2          # a thick magnet, Cartesian block (cartesian.MG_*)
N_MULTIPOLE = 3       # thin multipole: [order, inv_factorial, knl[0..order], ksl[0..order]]
N_KICK = 4            # thin dipole kick (corrector): [dpx, dpy]
N_THINLENS = 5        # thin linear lens: [1/fx, 1/fy]
N_EDGE_LINEAR = 6     # [r21, r43]
N_EDGE_EXACT = 7      # pole face, exact (edge models "full" and "exact"): EE_* below
N_CAVITY = 8          # [voltage MV, frequency Hz, lag deg]
N_SROTATION = 9       # [sin, cos]
N_XYSHIFT = 10        # [dx, dy]
N_APERTURE = 11       # AP_* below
N_MATRIX = 12         # linear map in ring coordinates: [R (36, row-major), d (6)]
N_CURVILINEAR = 13    # a thick element of the curvilinear model (curvilinear.CV_*)
N_REFSHIFT = 14       # after a cavity in a ramp: the reference's gain [dE MeV, p0c' MeV, beta0']

KIND_NAMES = {N_MARKER: "marker", N_DRIFT: "drift", N_MAGNET: "magnet",
              N_MULTIPOLE: "multipole", N_KICK: "kick", N_THINLENS: "thin lens",
              N_EDGE_LINEAR: "edge (linear)", N_EDGE_EXACT: "edge (exact)",
              N_CAVITY: "cavity", N_SROTATION: "s-rotation", N_XYSHIFT: "xy-shift",
              N_APERTURE: "aperture", N_MATRIX: "matrix", N_CURVILINEAR: "magnet (curvilinear)",
              N_REFSHIFT: "reference change"}

# ---- loss causes (stored per particle) -----------------------------------
LOSS_NONE = 0
LOSS_APERTURE = 1
LOSS_INVALID = 6      # (particles.LOSS_CAUSES numbering: 1 aperture ... 5 user, 6 invalid)

# ---- exact pole face block -------------------------------------------------
EE_IS_EXIT = 0
EE_ANGLE = 1          # face angle [rad], with its sin, cos, tan
EE_SIN = 2
EE_COS = 3
EE_TAN = 4
EE_K0_FRINGE = 5      # k0 of the dipole fringe (with knl[0]/length)
EE_K0_WEDGE = 6       # k0 of the wedge
EE_K1 = 7
EE_FINT = 8
EE_HGAP = 9
EE_QUAD_WEDGE = 10    # quadrupole wedge: 0 none, 1 Xsuite's, 2 MAD8's (MAD-X / PTC default)
EE_MULT_FRINGE = 11   # 1: with the multipole fringe (Xsuite "full")
EE_KN = 12            # kn0..kn3, ks0..ks3 of the magnet
EE_LENGTH = 20        # magnet length (for its knl/ksl)
EE_ORDER = 21
EE_KNL = 22           # knl[0..order], ksl[0..order]

# ---- aperture block ---------------------------------------------------------
AP_SHAPE = 0          # 1 circle (a = radius), 2 rectangle (a, b half widths),
#                       3 ellipse (a, b), 4 rectangle (a, b) AND ellipse (c, d),
#                       5 polygon (n vertices)
AP_A = 1
AP_B = 2
AP_C = 3
AP_D = 4
AP_X0 = 5             # centre of the aperture
AP_Y0 = 6
AP_N = 7              # polygon: number of vertices, then x[0..n-1], y[0..n-1]
AP_XY = 8


@jitable
def outside_aperture(p, o, x, y):
    """True when (x, y) is outside the aperture at p[o:]."""
    shape = int(p[o + AP_SHAPE])
    u = x - p[o + AP_X0]
    v = y - p[o + AP_Y0]
    a = p[o + AP_A]
    b = p[o + AP_B]
    if shape == 1:
        return u * u + v * v > a * a
    if shape == 2:
        return abs(u) > a or abs(v) > b
    if shape == 3:
        return (u / a) * (u / a) + (v / b) * (v / b) > 1.0
    if shape == 4:
        c = p[o + AP_C]
        d = p[o + AP_D]
        return abs(u) > a or abs(v) > b or (u / c) * (u / c) + (v / d) * (v / d) > 1.0
    if shape == 5:
        # even-odd rule: a point is inside when a ray to +x crosses the
        # boundary an odd number of times
        n = int(p[o + AP_N])
        ox = o + AP_XY
        oy = ox + n
        inside = False
        jj = n - 1
        for ii in range(n):
            xi = p[ox + ii]
            yi = p[oy + ii]
            xj = p[ox + jj]
            yj = p[oy + jj]
            if (yi > v) != (yj > v) and u < (xj - xi) * (v - yi) / (yj - yi) + xi:
                inside = not inside
            jj = ii
        return not inside
    return False


@jitable
def quad_wedge_mad8(x, px, y, py, angle, k1):
    """The quadrupole field of the wedge as MAD8 and PTC (``MAD8_WEDGE``, the
    MAD-X default) write it: the kick of H = -angle k1 (x^3/3 - x y^2) →
    (px, py)."""
    return px + angle * k1 * (x * x - y * y), py - angle * k1 * (2.0 * x * y)


@jitable
def exact_edge(p, o, x, px, y, py, zeta, delta, ptau, rpp, rvv, beta0):
    """Pole face of a bend, exact: frame rotation, hard-edge dipole fringe
    (also at a face angle of zero), optionally the multipole fringe, the
    quadrupole wedge, and the dipole wedge — the order of Xsuite's
    ``track_magnet_edge_particles`` (models 1 and 2) and of PTC's bend edges;
    the quadrupole wedge either Xsuite's or MAD8's → (x, px, y, py, zeta)."""
    is_exit = p[o + EE_IS_EXIT] != 0.0
    angle = p[o + EE_ANGLE]
    rotate = fabs(angle) > 10e-10
    s_ = p[o + EE_SIN]
    c_ = p[o + EE_COS]
    t_ = p[o + EE_TAN]
    k0f = p[o + EE_K0_FRINGE]
    k0w = p[o + EE_K0_WEDGE]
    k1 = p[o + EE_K1]
    fint = p[o + EE_FINT]
    hgap = p[o + EE_HGAP]
    quad_w = int(p[o + EE_QUAD_WEDGE])
    mult_f = p[o + EE_MULT_FRINGE] != 0.0
    order = int(p[o + EE_ORDER])
    o_knl = o + EE_KNL
    o_ksl = o_knl + order + 1
    length = p[o + EE_LENGTH]
    kn = o + EE_KN
    if not is_exit:
        if rotate:
            x, px, y, zeta = _cart.yrotation(x, px, y, py, zeta, ptau, beta0, -s_, c_, -t_)
        x, y, py, zeta = _cart.dipole_fringe(x, px, y, py, zeta, delta, ptau, beta0, fint, hgap, k0f)
        if mult_f:
            x, px, y, py, zeta = _cart.mult_fringe(
                x, px, y, py, zeta, ptau, rpp, beta0, p[kn], p[kn + 1], p[kn + 2], p[kn + 3],
                p[kn + 4], p[kn + 5], p[kn + 6], p[kn + 7], p, o_knl, o_ksl, order, length,
                False, 1)
        if rotate and quad_w == 1:
            px, py = _cart.quad_wedge(x, px, y, py, -angle, k1)
        elif rotate and quad_w == 2:
            px, py = quad_wedge_mad8(x, px, y, py, angle, k1)
        if rotate:
            x, px, y, zeta = _cart.wedge(x, px, y, py, zeta, delta, ptau, rvv, beta0, -angle, k0w)
    else:
        if rotate:
            x, px, y, zeta = _cart.wedge(x, px, y, py, zeta, delta, ptau, rvv, beta0, -angle, k0w)
        if rotate and quad_w == 1:
            px, py = _cart.quad_wedge(x, px, y, py, -angle, k1)
        elif rotate and quad_w == 2:
            px, py = quad_wedge_mad8(x, px, y, py, angle, k1)
        if mult_f:
            x, px, y, py, zeta = _cart.mult_fringe(
                x, px, y, py, zeta, ptau, rpp, beta0, p[kn], p[kn + 1], p[kn + 2], p[kn + 3],
                p[kn + 4], p[kn + 5], p[kn + 6], p[kn + 7], p, o_knl, o_ksl, order, length,
                True, 1)
        x, y, py, zeta = _cart.dipole_fringe(x, px, y, py, zeta, delta, ptau, beta0, fint, hgap, -k0f)
        if rotate:
            x, px, y, zeta = _cart.yrotation(x, px, y, py, zeta, ptau, beta0, -s_, c_, -t_)
    return x, px, y, py, zeta


@jitable
def track_node(kind, p, o, beta0, p0c, bg2, x, px, y, py, zeta, delta, ptau, rpp, rvv):
    """Advance one particle through node (kind, p[o:]) at the reference
    (beta0, p0c, (beta0 gamma0)^2) → (x, px, y, py, zeta, delta, ptau, rpp,
    rvv, loss cause)."""
    lost = LOSS_NONE
    if kind == N_DRIFT:
        L = p[o]
        if p[o + 1] != 0.0:
            x, y, zeta = _cart.drift_exact(x, px, y, py, zeta, delta, rvv, L)
        else:
            x, y, zeta = _cart.drift_expanded(x, px, y, py, zeta, rpp, rvv, L)
    elif kind == N_MAGNET:
        x, px, y, py, zeta = _cart.magnet_body(p, o, x, px, y, py, zeta, delta, rpp, rvv)
    elif kind == N_MULTIPOLE:
        order = int(p[o])
        dpx, dpy = _cart.kick_from_array(x, y, p, o + 2, o + 3 + order, order, p[o + 1], 1.0, 1.0)
        px += dpx
        py += dpy
    elif kind == N_KICK:
        px += p[o]
        py += p[o + 1]
    elif kind == N_THINLENS:
        px -= x * p[o]
        py -= y * p[o + 1]
    elif kind == N_EDGE_LINEAR:
        px += p[o] * x
        py += p[o + 1] * y
    elif kind == N_EDGE_EXACT:
        x, px, y, py, zeta = exact_edge(p, o, x, px, y, py, zeta, delta, ptau, rpp, rvv, beta0)
    elif kind == N_CAVITY:
        ptau = _cart.cavity_kick(zeta, ptau, beta0, p0c, p[o], p[o + 1], p[o + 2])
        delta, rpp, rvv = _cart.energy_from_ptau(ptau, beta0)
    elif kind == N_SROTATION:
        x, px, y, py = _cart.srotation(x, px, y, py, p[o], p[o + 1])
    elif kind == N_XYSHIFT:
        x, y = _cart.xyshift(x, y, p[o], p[o + 1])
    elif kind == N_APERTURE:
        if outside_aperture(p, o, x, y):
            lost = LOSS_APERTURE
    elif kind == N_MATRIX:
        c0 = x
        c1 = px
        c2 = y
        c3 = py
        c4 = zeta
        c5 = delta
        x = p[o + 36] + p[o] * c0 + p[o + 1] * c1 + p[o + 2] * c2 + p[o + 3] * c3 + p[o + 4] * c4 + p[o + 5] * c5
        px = p[o + 37] + p[o + 6] * c0 + p[o + 7] * c1 + p[o + 8] * c2 + p[o + 9] * c3 + p[o + 10] * c4 + p[o + 11] * c5
        y = p[o + 38] + p[o + 12] * c0 + p[o + 13] * c1 + p[o + 14] * c2 + p[o + 15] * c3 + p[o + 16] * c4 + p[o + 17] * c5
        py = p[o + 39] + p[o + 18] * c0 + p[o + 19] * c1 + p[o + 20] * c2 + p[o + 21] * c3 + p[o + 22] * c4 + p[o + 23] * c5
        zeta = p[o + 40] + p[o + 24] * c0 + p[o + 25] * c1 + p[o + 26] * c2 + p[o + 27] * c3 + p[o + 28] * c4 + p[o + 29] * c5
        delta = p[o + 41] + p[o + 30] * c0 + p[o + 31] * c1 + p[o + 32] * c2 + p[o + 33] * c3 + p[o + 34] * c4 + p[o + 35] * c5
        if delta != c5:
            ptau, rpp, rvv = _cart.energy_from_delta(delta, beta0)
    elif kind == N_CURVILINEAR:
        t = -zeta / beta0
        pt = -ptau
        x, px, y, py, t = _curv.thick_element(p, o, x, px, y, py, t, pt, beta0, bg2)
        zeta = -beta0 * t
    elif kind == N_REFSHIFT:
        # the reference particle gained dE at the cavity before: the
        # coordinates re-expressed relative to it (its momentum p0c', velocity
        # beta0'), keeping each particle's energy, transverse momentum and
        # time — ptau' = (p0c ptau - dE)/p0c', px' = px p0c/p0c',
        # zeta' = zeta beta0'/beta0 (no change at all for no gain)
        p0c_new = p[o + 1]
        if p[o] != 0.0 or p0c_new != p0c:
            beta0_new = p[o + 2]
            r = p0c / p0c_new
            ptau = (p0c * ptau - p[o]) / p0c_new
            px = px * r
            py = py * r
            zeta = zeta * (beta0_new / beta0)
            delta, rpp, rvv = _cart.energy_from_ptau(ptau, beta0_new)
    return x, px, y, py, zeta, delta, ptau, rpp, rvv, lost


@jitable
def track_particle(kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1,
                   x, px, y, py, zeta, delta, ptau, rpp, rvv):
    """Advance one particle through nodes j0..j1-1 → (x, ..., rvv, node where
    it was lost or -1, loss cause)."""
    for j in range(j0, j1):
        (x1, px1, y1, py1, zeta1, delta1, ptau1, rpp1, rvv1, lost) = track_node(
            kinds[j], p, offsets[j], ref_beta0[j], ref_p0c[j], ref_bg2[j],
            x, px, y, py, zeta, delta, ptau, rpp, rvv)
        if lost != LOSS_NONE:
            return x, px, y, py, zeta, delta, ptau, rpp, rvv, j, lost
        if not (isfinite(x1) and isfinite(px1) and isfinite(y1) and isfinite(py1)
                and isfinite(zeta1) and isfinite(delta1)):
            return x, px, y, py, zeta, delta, ptau, rpp, rvv, j, LOSS_INVALID
        x = x1
        px = px1
        y = y1
        py = py1
        zeta = zeta1
        delta = delta1
        ptau = ptau1
        rpp = rpp1
        rvv = rvv1
    return x, px, y, py, zeta, delta, ptau, rpp, rvv, -1, LOSS_NONE
