"""The expanded combined-function map (``bend_model="expanded"``, and every
Cartesian quadrupole) where its horizontal focusing Kx = k0 h + k1 nearly
vanishes: a bend of field index near 1.

Its terms divided by Kx — (cos sqrt(Kx) L - 1)/Kx and two companions — come
from their series in u = Kx L^2 below ``CFD_SERIES_LIMIT``.  The direct forms
divide a cancellation by Kx: at N = 1, where Kx is a rounding residue
(~1e-17), they lost every digit (the path length's dependence on x, R51,
came out 0), and the exact Kx = 0 branch left out two path-length terms (the
map was not symplectic).  Above the switch, and in quadrupoles (no
curvature, no dipole field: nothing cancels), the map is unchanged bit for
bit.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.ring.compile import blocks as B
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.kernels import cartesian as C
from linac_gen.ring.kernels import nodes as N
from linac_gen.ring.optics import one_turn_matrix, track_once
from linac_gen.ring.reference import RingReference
from tests.ring import mp_kernels
from tests.ring.kernel_harness import run

ROOT = Path(__file__).resolve().parents[2]
BETA0 = 0.8418107060243323
P0C, MASS = 1463.2960537963617, 938.27208816
LIM = C.CFD_SERIES_LIMIT
EPS = 2.220446049250313e-16


def _cfd(z, length, k0, k1, h):
    x, px, y, py, zeta, delta = z
    _ptau, _rpp, rvv = C.energy_from_delta(delta, BETA0)
    return np.array(C.expanded_cfd(x, px, y, py, zeta, delta, rvv, length, k0, k1, h))


def _k1_for(u, length, k0, h, delta):
    """k1 that puts Kx L^2 at ``u`` for a particle of momentum deviation delta
    (the map divides the strengths by 1 + delta)."""
    return (u / (length * length)) * (1.0 + delta) - k0 * h


def _symplectic_error(J):
    S = np.zeros((6, 6))
    for k in (0, 2, 4):
        S[k, k + 1] = 1.0
        S[k + 1, k] = -1.0
    return float(np.abs(J.T @ S @ J - S).max())


#: particles: (x, px, y, py, zeta, delta); zeta = 0, so its output is the change
POINTS = [np.array([1.2e-3, 2e-4, -0.9e-3, 1.1e-4, 0.0, d]) for d in (0.0, 1e-3, -1e-3)] + [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1e-3]), np.array([-2e-3, -1e-4, 1e-3, 3e-4, 0.0, 2e-3])]


@pytest.mark.parametrize("field_index", [0.999, 1.0, 1.001])
def test_bend_of_field_index_near_one_agrees_with_the_exact_model(field_index):
    # a 10-degree bend of 5 m radius: the expanded model against the exact
    # one (before: at N = 1 the expanded model gave R51 = 0, and 5.4e-6 m in x
    # and 8.8e-4 m in zeta after the bend at delta = 1e-3; 4e-10 m and
    # 2.4e-11 m at N = 0.999 and 1.001)
    ref = RingReference.from_kinetic("proton", 800.0)
    X = np.zeros((6, 3))
    X[0] = [1e-3, 0.0, 0.0]
    X[5] = [0.0, 1e-3, -1e-3]
    out = {}
    for model in ("exact", "expanded"):
        lat = Lattice()
        lat.add(Dipole("b", 10.0, 5000.0, field_index=field_index))
        rl = compile_ring(lat, ref, bend_model=model, edge_model="linear")
        out[model] = (one_turn_matrix(rl, np.zeros(6)), track_once(rl, X))
    (Re, Ye), (Rx, Yx) = out["exact"], out["expanded"]
    # the path length's dependence on x: the symplectic relation R51 = R16 R21 - R11 R26
    assert Rx[4, 0] == pytest.approx(Rx[0, 5] * Rx[1, 0] - Rx[0, 0] * Rx[1, 5], abs=1e-8)
    assert Rx[4, 0] == pytest.approx(Re[4, 0], rel=1e-7)
    assert _symplectic_error(Rx) < 1e-8
    assert np.abs(Yx[0] - Ye[0]).max() < 1e-9
    assert np.abs(Yx[4] - Ye[4]).max() < 1e-10


def test_kernel_with_kx_at_rounding_level_is_symplectic():
    # k0 = h = 0.1, k1 = -0.01: Kx = 1.7e-18 in floating point (the
    # Jacobian's symplectic error was 0.1; central differences floor at ~1e-10)
    L, k0, k1, h = 1.0, 0.1, -0.01, 0.1
    assert 0.0 < abs(k0 * h + k1) < 1e-15
    z0 = np.zeros(6)
    J = np.zeros((6, 6))
    for k in range(6):
        zp, zm = z0.copy(), z0.copy()
        zp[k] += 1e-7
        zm[k] -= 1e-7
        J[:5, k] = (_cfd(zp, L, k0, k1, h) - _cfd(zm, L, k0, k1, h)) / 2e-7
    J[5, 5] = 1.0
    assert _symplectic_error(J) < 1e-8
    # the linear map of a uniform field: a drift, the parabolic dispersion
    # k0 L^2/2, R26 = h L and R51 = R16 R21 - R11 R26 = -h L
    want = {(0, 0): 1.0, (0, 1): L, (1, 1): 1.0, (0, 5): k0 * L * L / 2.0, (1, 5): h * L,
            (4, 0): -h * L, (4, 1): -k0 * L * L / 2.0}
    for (i, j), v in want.items():
        assert J[i, j] == pytest.approx(v, abs=1e-8), (i, j)


_MP = None


def _mp_cfd(z, length, k0, k1, h, series):
    """The same map in 50-digit arithmetic: its direct forms keep 25 digits
    or more for |Kx L^2| >= 1e-24 (the truth); at Kx = 0, the series."""
    global _MP
    if _MP is None:
        _MP = mp_kernels.load(50)[0]
    _MP.CFD_SERIES_LIMIT = LIM if series else 0.0
    try:
        zz = [mp.mpf(float(v)) for v in z]
        _ptau, _rpp, rvv = _MP.energy_from_delta(zz[5], mp.mpf(BETA0))
        return _MP.expanded_cfd(*zz, rvv, mp.mpf(length), mp.mpf(k0), mp.mpf(k1), mp.mpf(h))
    finally:
        _MP.CFD_SERIES_LIMIT = LIM


U_GRID = [0.0, 1e-20, -1e-20, 1e-12, -1e-12, 1e-6, -1e-6, 1e-4, -1e-4, 0.999 * LIM, -0.999 * LIM,
          1.001 * LIM, -1.001 * LIM, 1e-2, -1e-2, 0.3, -0.3]


@pytest.mark.parametrize("length, h, k0", [(1.0, 0.1, 0.1), (0.05, 0.0349, 0.0349),
                                           (-0.3, 0.0349, 0.0349), (1.3, 0.0, 0.012),
                                           (0.7, 0.05, 0.0)])
def test_kernel_matches_a_50_digit_evaluation(length, h, k0):
    # on both sides of the switch, both signs of Kx, Kx at rounding level (a
    # negative length: the Yoshida composition's backward drift; h = 0: a
    # straight-frame dipole; k0 = 0: a curved frame without field).  Measured
    # worst: 2.6e-13 of the larger of each coordinate's input and output (the
    # rounding of k0/(1 + delta) - h, the same on both sides of the switch),
    # and the path length's rounding, 2.6e-16 of the length
    for u in U_GRID:
        for z in POINTS:
            k1 = _k1_for(u, length, k0, h, z[5])
            got = _cfd(z, length, k0, k1, h)
            kx = (k0 / (1.0 + z[5])) * h + k1 / (1.0 + z[5])
            want = _mp_cfd(z, length, k0, k1, h, series=kx == 0.0)
            for i in range(5):
                tol = 1e-12 * max(abs(z[i]), abs(float(want[i]))) + (
                    4.0 * EPS * abs(length) if i == 4 else 1e-30)
                assert abs(float(got[i] - want[i])) <= tol, (u, i, float(got[i]), float(want[i]))


def _close(a, b, z, length):
    for i in range(5):
        tol = 1e-12 * max(abs(z[i]), abs(a[i])) + (4.0 * EPS * abs(length) if i == 4 else 1e-30)
        assert abs(a[i] - b[i]) <= tol, (i, a[i], b[i])


def test_the_switch_is_continuous():
    # either side of |Kx| L^2 = CFD_SERIES_LIMIT: the map moves by no more
    # than the direct forms' own rounding there (6e-13 relative)
    for length, h, k0 in ((1.0, 0.1, 0.1), (0.4, 0.0349, 0.0349), (-0.2, 0.05, 0.06)):
        for z in POINTS:
            for u0 in (LIM, -LIM):
                du = 1e-12 * LIM
                _close(_cfd(z, length, k0, _k1_for(u0 - du, length, k0, h, z[5]), h),
                       _cfd(z, length, k0, _k1_for(u0 + du, length, k0, h, z[5]), h), z, length)
    # Kx = 0 exactly (k0 = h = 1/8, k1 = -1/64: exact for every delta) and
    # Kx a rounding residue of either sign (before: the exact zero took the
    # parabola, the residue lost the dispersion and the path length)
    for length in (1.0, -0.3):
        for z in POINTS:
            a = _cfd(z, length, 0.125, -0.015625, 0.125)
            assert (0.125 / (1.0 + z[5])) * 0.125 + (-0.015625) / (1.0 + z[5]) == 0.0
            for k1 in (-0.015625 * (1.0 + 1e-15), -0.015625 * (1.0 - 1e-15)):
                _close(a, _cfd(z, length, 0.125, k1, 0.125), z, length)


def _expanded_cfd_before(x, px, y, py, zeta, delta, rvv, length, k0_, k1_, h):
    """``expanded_cfd`` as it was before the series (verbatim)."""
    sqrt, sin, cos, sinh, cosh = math.sqrt, math.sin, math.cos, math.sinh, math.cosh
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
    x_ = (x_ + (k0 - h) * (Cx - 1.0) / Kx if Kx != 0.0
          else x_ - (k0 - h) * 0.5 * (length * length))
    length_ = length
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
        length_ += h * length * (3.0 * length * xp + 6.0 * x - (k0 - h) * (length * length)) / 6.0
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


def test_unchanged_above_the_switch_and_in_quadrupoles():
    rng = np.random.default_rng(11)
    n = 0
    for _ in range(4000):
        z = np.array([rng.normal(0, 2e-3), rng.normal(0, 2e-4), rng.normal(0, 2e-3),
                      rng.normal(0, 2e-4), rng.normal(0, 0.1), rng.normal(0, 2e-3)])
        length = float(rng.choice([1.0, -1.0])) * 10 ** rng.uniform(-2, 0.7)
        _ptau, _rpp, rvv = C.energy_from_delta(z[5], BETA0)
        if rng.random() < 0.5:                       # a bend above the switch
            h = 10 ** rng.uniform(-3, -0.5)
            k0 = h * (1.0 + rng.normal(0, 0.01))
            u = float(rng.choice([1.0, -1.0])) * LIM * 10 ** rng.uniform(1e-9, 3)
            k1 = _k1_for(u, length, k0, h, z[5])
        else:                                        # a quadrupole, any strength, also 0
            h = k0 = 0.0
            k1 = 0.0 if rng.random() < 0.1 else float(rng.choice([1.0, -1.0])) * 10 ** rng.uniform(-12, 0.5)
        got = C.expanded_cfd(*z, rvv, length, k0, k1, h)
        want = _expanded_cfd_before(*z, rvv, length, k0, k1, h)
        assert got == want, (z, length, k0, k1, h)
        n += 1
    assert n == 4000


def _n1_bend_nodes(integrator, n_kicks):
    L = 2.889612
    h = 0.0707421821963016 / L
    cfg = B.magnet_config("mat-kick-mat", h, -(h * h), h)
    return [(N.N_MAGNET, B.magnet(L, integrator=B.INTEGRATORS[integrator], n_kicks=n_kicks,
                                  k2=0.0055, **cfg))]


@pytest.mark.parametrize("integrator, n_kicks", [("uniform", 4), ("yoshida4", 14), ("teapot", 5)])
def test_compiled_kernel_agrees_with_the_python_source(integrator, n_kicks):
    pytest.importorskip("numba")
    X = np.stack(POINTS, axis=1)
    a = run(_n1_bend_nodes(integrator, n_kicks), X, beta0=BETA0, p0c=P0C, mass=MASS,
            backend="python")
    b = run(_n1_bend_nodes(integrator, n_kicks), X, beta0=BETA0, p0c=P0C, mass=MASS,
            backend="numba")
    assert np.array_equal(a, b)


def test_cuda_kernel_in_the_simulator_matches_the_cpu():
    pytest.importorskip("numba")
    code = r"""
import numpy as np
from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.kernels import cuda as kc
from linac_gen.ring.kernels.cpu import backend
from linac_gen.ring.optics import _energy
from linac_gen.ring.reference import RingReference
ref = RingReference.from_kinetic("proton", 800.0)
lat = Lattice()
for i in range(2):
    lat.add(Dipole(f"b{i}", 10.0, 5000.0, field_index=1.0))
    lat.add(Drift(f"d{i}", 500.0))
    lat.add(Quadrupole(f"q{i}", length=300.0, gradient=(-1) ** i * 5.0))
rl = compile_ring(lat, ref, bend_model="expanded", edge_model="linear")
rng = np.random.default_rng(3)
X = rng.normal(0, 1e-3, (6, 4)); X[4] *= 300
outs = []
for fn in (backend(allow_python=False).track_turns, kc.track_turns):
    Y = X.copy()
    ptau, rpp, rvv = (np.ascontiguousarray(a) for a in _energy(Y[5], ref.beta0))
    st = np.ones(4, dtype=np.int8); lt = np.full(4, -1, dtype=np.int32)
    ln = np.full(4, -1, dtype=np.int32); lc = np.zeros(4, dtype=np.int8)
    fn(Y, ptau, rpp, rvv, st, lt, ln, lc, *rl.node_table(), 0, rl.n_nodes, 1, 0, 0.0)
    outs.append(Y)
d = np.abs(outs[0] - outs[1]) / np.maximum(np.abs(outs[0]), 1e-300)
print("CUDASIM_REL_DIFF", float(np.nanmax(np.where(outs[0] == 0, 0.0, d))), flush=True)
"""
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(ROOT)})
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=900)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("CUDASIM_REL_DIFF")]
    assert lines, (out.returncode, out.stderr[-3000:])
    assert float(lines[-1].split()[1]) <= 1e-15
