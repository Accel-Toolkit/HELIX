"""Every ring node is symplectic (L1: |J^T S J - S| <= 1e-12).

The Jacobian of each node's map is taken in the canonical coordinates
(x, px, y, py, zeta, pzeta = ptau/beta0) by central differences in 50-digit
arithmetic (the kernel sources run on mpmath, :mod:`tests.ring.mp_kernels`),
so the check sees the map itself, not float64 rounding: a symplectic map
shows ~1e-28.
"""
from __future__ import annotations

import math

import mpmath as mp
import pytest

from linac_gen.ring.compile import blocks as B
from linac_gen.ring.kernels import nodes as N
from tests.ring import mp_kernels

CART, _CURV, ND = mp_kernels.load(50)
BETA0 = mp.mpf("0.8418107060243323")
P0C = mp.mpf("1463.2960537963617")
BG2 = (P0C / mp.mpf("938.27208816")) ** 2
H = mp.mpf("1e-20")

L_B = 2.889612
ANG = L_B / 40.847086
HB = ANG / L_B
K1, K2 = 0.05410921561, 0.00546850017312


def _canonical_map(nodes):
    blocks = [(k, [mp.mpf(v) for v in blk]) for k, blk in nodes]

    def f(c):
        x, px, y, py, zeta, pzeta = c
        ptau = pzeta * BETA0
        delta, rpp, rvv = CART.energy_from_ptau(ptau, BETA0)
        for kind, p in blocks:
            (x, px, y, py, zeta, delta, ptau, rpp, rvv, _lost) = ND.track_node(
                kind, p, 0, BETA0, P0C, BG2, x, px, y, py, zeta, delta, ptau, rpp, rvv)
        return [x, px, y, py, zeta, ptau / BETA0]
    return f


def _symplectic_error(f, c0):
    n = 6
    J = mp.matrix(n, n)
    for k in range(n):
        cp = list(c0)
        cm = list(c0)
        cp[k] += H
        cm[k] -= H
        fp, fm = f(cp), f(cm)
        for i in range(n):
            J[i, k] = (fp[i] - fm[i]) / (2 * H)
    S = mp.matrix(n, n)
    for k in (0, 2, 4):
        S[k, k + 1] = 1
        S[k + 1, k] = -1
    E = J.T * S * J - S
    return float(max(abs(E[i, j]) for i in range(n) for j in range(n)))


def _point():
    delta = mp.mpf("1e-3")
    ptau, _rpp, _rvv = CART.energy_from_delta(delta, BETA0)
    return [mp.mpf("1.2e-3"), mp.mpf("2e-4"), mp.mpf("-0.9e-3"), mp.mpf("1.1e-4"),
            mp.mpf("0.05"), ptau / BETA0]


def _magnet(model, integ, nk, k0=HB, k1=K1, h=HB, **kw):
    cfg = B.magnet_config(model, k0, k1, h)
    return (N.N_MAGNET, B.magnet(L_B, integrator=B.INTEGRATORS[integ], n_kicks=nk, k2=K2,
                                 **cfg, **kw))


def _edge(is_exit, full):
    return (N.N_EDGE_EXACT, B.edge_exact(
        is_exit=is_exit, angle=ANG / 2, k0_fringe=HB, k0_wedge=HB, k1=K1, fint=0.5, hgap=0.03,
        quad_wedge=1 if full else 2, mult_fringe=full, kn=(HB, K1, K2, 0.0), length=L_B))


def _r_symp():
    """A coupled symplectic transverse map (two Twiss rotations and a skew shear)."""
    import numpy as np

    def rot(mu, beta, alpha):
        c, s_ = math.cos(mu), math.sin(mu)
        return np.array([[c + alpha * s_, beta * s_], [-(1 + alpha * alpha) / beta * s_,
                                                       c - alpha * s_]])
    R = np.eye(6)
    R[:2, :2] = rot(0.7, 12.0, -0.4)
    R[2:4, 2:4] = rot(1.9, 3.0, 0.8)
    K = np.eye(6)                     # thin skew quadrupole: px += k y, py += k x
    K[1, 2] = K[3, 0] = 0.05
    return (R @ K).tolist()


R_SYMP = _r_symp()

CASES = {
    "drift exact": [(N.N_DRIFT, B.drift(1.3, True))],
    "drift expanded": [(N.N_DRIFT, B.drift(1.3, False))],
    "bend rot-kick-rot yoshida4": [_magnet("rot-kick-rot", "yoshida4", 28)],
    "bend rot-kick-rot symp6": [_magnet("rot-kick-rot", "symp6", 3)],
    "bend rot-kick-rot teapot": [_magnet("rot-kick-rot", "teapot", 5)],
    "bend bend-kick-bend": [_magnet("bend-kick-bend", "yoshida4", 14)],
    "bend mat-kick-mat": [_magnet("mat-kick-mat", "uniform", 3)],
    # Kx = k0 h + k1 = 0 exactly (the direct Kx = 0 branch missed two path
    # terms: 8.7e-5), and field index 1 in floating point (Kx a residue)
    "bend mat-kick-mat, Kx = 0": [_magnet("mat-kick-mat", "uniform", 3, k0=0.125, k1=-0.015625,
                                          h=0.125)],
    "bend mat-kick-mat teapot, Kx = 0": [_magnet("mat-kick-mat", "teapot", 5, k0=0.125,
                                                 k1=-0.015625, h=0.125)],
    "bend mat-kick-mat, field index 1": [_magnet("mat-kick-mat", "yoshida4", 14, k1=-HB * HB)],
    "bend field error": [_magnet("rot-kick-rot", "yoshida4", 14, k0=1.02 * HB)],
    "straight bend": [_magnet("bend-kick-bend", "yoshida4", 7, k0=0.012, k1=0.0, h=0.0)],
    "quadrupole with multipoles": [(N.N_MAGNET, B.magnet(
        0.4, integrator=B.INTEGRATORS["uniform"], n_kicks=2,
        knl=[0.0, 0.0, 0.05, -0.4], **B.magnet_config("mat-kick-mat", 0.0, 0.31, 0.0)))],
    "sextupole": [(N.N_MAGNET, B.magnet(
        0.1, integrator=B.INTEGRATORS["uniform"], n_kicks=1, k2=-8.6,
        **B.magnet_config("drift-kick-drift-expanded", 0.0, 0.0, 0.0)))],
    "thin multipole": [(N.N_MULTIPOLE, B.multipole([1e-4, 0.02, -0.5, 3.0], [2e-4, -0.01, 0.3, 1.0]))],
    "kick": [(N.N_KICK, B.kick(1e-4, -2e-4))],
    "thin lens": [(N.N_THINLENS, B.thin_lens(0.3, -0.2))],
    "edge linear": [(N.N_EDGE_LINEAR, B.edge_linear(0.02, -0.019))],
    "edge exact (entry)": [_edge(False, False)],
    "edge exact (exit)": [_edge(True, False)],
    "edge full (entry)": [_edge(False, True)],
    "edge full (exit)": [_edge(True, True)],
    "cavity": [(N.N_CAVITY, B.cavity(0.2, 44.7e6, 30.0))],
    "s-rotation": [(N.N_SROTATION, B.srotation(0.3))],
    "xy-shift": [(N.N_XYSHIFT, B.xyshift(1e-3, -2e-3))],
    "matrix": [(N.N_MATRIX, B.matrix(R_SYMP))],
    "curvilinear quad": [(N.N_CURVILINEAR, B.curvilinear_quad(0.024, 0.0202, integrator=6,
                                                              nsteps=6))],
    "curvilinear multipole": [(N.N_CURVILINEAR, B.curvilinear_multipole(
        0.024, [0.0, 0.0, -9.0, 30.0], [0.0, 0.1, 0.0, 0.0], integrator=5, nsteps=5))],
    "curvilinear bend": [(N.N_CURVILINEAR, B.curvilinear_cfbend(
        L_B, 1.0 / HB, [0.0, K1, K2, 0.4, 2.0], [0.0, 0.01, 0.02, 0.0, 0.0], integrator=6,
        nsteps=6))],
    "curvilinear bend, field error": [(N.N_CURVILINEAR, B.curvilinear_cfbend(
        L_B, 1.0 / HB, [0.01 * HB, K1, K2], [0.0, 0.0, 0.0], integrator=4, nsteps=4))],
    "Booster pole face + body + pole face": [_edge(False, False),
                                             _magnet("rot-kick-rot", "yoshida4", 98),
                                             _edge(True, False)],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_node_is_symplectic(case):
    err = _symplectic_error(_canonical_map(CASES[case]), _point())
    assert err <= 1e-12, f"{case}: |J^T S J - S| = {err:.3e}"


def test_the_check_sees_a_non_symplectic_map():
    # (negative control: a thin lens whose kick depends on px is not symplectic)
    def f(c):
        x, px, y, py, zeta, pz = c
        return [x, px - 0.3 * x * (1 + px), y, py, zeta, pz]
    assert _symplectic_error(f, _point()) > 1e-6
