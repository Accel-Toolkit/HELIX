"""Shortley-Weller Laplace solver against an exact solution.

Electrodes shaped as equipotentials of the two-term RFQ potential
U = V/2 [(x^2-y^2)/r0^2 + A I0(kr) cos kz], outer faces held at U: the
solution inside must be U itself."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import i0, i1

pytest.importorskip("pyamg")

from linac_gen.rfq_td.laplace import solve_window, stretched_axis

V2, R0, A, L = 30000.0, 5.0, 0.12, 10.0
K = np.pi / L


def U(x, y, z):
    r = np.sqrt(x * x + y * y)
    return V2 * ((x * x - y * y) / R0 ** 2 + A * i0(K * r) * np.cos(K * z))


def _sdf(X, Y, Z):
    u = U(X, Y, Z)
    fx = (V2 - u) / V2            # < 0 inside the +x electrode
    fy = (u + V2) / V2            # < 0 inside the +y electrode
    which = np.where(fx < fy, 0, 1)
    return np.minimum(fx, fy), which, np.where(which == 0, V2, -V2)


def _error(h):
    xs = stretched_axis(h, 6.0, 12.0)
    zs = np.arange(0.0, 2 * L + 0.5 * h, h)
    info = {}
    phi = solve_window(_sdf, xs, xs, zs, dirichlet=U,
                       dirichlet_faces=("x1", "y1"), info=info)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    m = np.sqrt(X * X + Y * Y) < 3.5
    return np.abs(phi - U(X, Y, Z))[m].max() / V2, info


def test_two_term_control_coarse():
    err, info = _error(0.2)
    assert info["rel_residual"] < 1e-8
    assert err < 3e-4


def test_second_order_convergence():
    e2, _ = _error(0.2)
    e1, _ = _error(0.1)
    assert 3.0 < e2 / e1 < 5.5           # O(h^2): ~4 per halving


def test_mirror_planes_and_metal_values():
    """Metal nodes carry their electrode's potential; phi is even in x and
    y by construction (quadrant grid with mirror planes)."""
    xs = stretched_axis(0.2, 6.0, 12.0)
    zs = np.arange(0.0, 2 * L + 0.1, 0.2)
    phi = solve_window(_sdf, xs, xs, zs, dirichlet=U,
                       dirichlet_faces=("x1", "y1"))
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    d, _w, V = _sdf(X, Y, Z)
    inside = d < 0
    assert inside.any()
    np.testing.assert_array_equal(phi[inside], V[inside])
    # on-axis potential is the A10 term alone
    np.testing.assert_allclose(phi[0, 0, :], U(0.0, 0.0, zs), atol=1e-3 * V2)


def test_a_solve_repeats_bit_for_bit_and_keeps_the_callers_random_state():
    """pyamg's set-up draws from numpy's global generator; the solve runs
    under linac_gen.core.amg.seeded, so two solves agree to the last bit
    whatever the caller's random state, and that state is left as it was."""
    xs = stretched_axis(0.4, 6.0, 12.0)
    zs = np.arange(0.0, 2 * L + 0.2, 0.4)

    def solve():
        return solve_window(_sdf, xs, xs, zs, dirichlet=U,
                            dirichlet_faces=("x1", "y1"))
    np.random.seed(12345)
    a = solve()
    drawn = np.random.rand(3)
    np.random.seed(12345)
    np.testing.assert_array_equal(drawn, np.random.rand(3))
    np.random.seed(999)
    np.testing.assert_array_equal(solve(), a)
