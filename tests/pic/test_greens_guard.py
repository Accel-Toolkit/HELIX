"""Guarded Green's-function spectrum: same numbers, repeatable on every grid.

numpy multiplies complex arrays with a vectorised loop unless the output's
memory touches an input's end to end, in which case it uses a scalar loop
that rounds both parts differently (no fused multiply-add).  A freshly
allocated FFT array used to land flush against the stored spectrum now and
then -- on macOS this happened for grids whose spectra are a whole number of
allocator blocks (4^3, 32^3 = 66 x 32 KiB, 64^3) -- so identical runs
differed at the 1e-13 level.  The spectrum now lives inside an allocation one
element larger on each side, in the FFT output's own memory layout, and the
product is taken in place.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.pic.poisson_solver import PoissonSolverFFT

try:
    from scipy import fft as _fft
except ImportError:                                  # pragma: no cover
    _fft = np.fft


def _solver(n, dx=(0.21, 0.17, 0.43), kind="igf"):
    n_grid = np.array([n, n, n], dtype=np.int64)
    gmin = np.array([-1.0, -2.0, -3.0])
    gmax = gmin + np.asarray(dx) * (n - 1)
    return PoissonSolverFFT(gmin, gmax, n_grid, use_gpu="cpu", green_kind=kind)


def _rho(n, seed=0):
    return np.random.default_rng(seed).random((n, n, n))


def _solve_pre_guard(solver, rho):
    """Today's solve with the spectrum as a plain array and an out-of-place
    product -- the pre-guard code path, kept here as the reference."""
    nx, ny, nz = (int(v) for v in solver.n_grid)
    dx, dy, dz = solver.dx
    G_plain = np.array(solver._G_fft, copy=True)       # fresh, unguarded copy
    padded = np.zeros((2 * nx, 2 * ny, 2 * nz))
    padded[:nx, :ny, :nz] = rho * (dx * dy * dz)
    phi = _fft.irfftn(_fft.rfftn(padded) * G_plain, s=padded.shape)[:nx, :ny, :nz]
    E = [np.zeros_like(phi) for _ in range(3)]
    E[0][1:-1] = -(phi[2:] - phi[:-2]) / (2.0 * dx)
    E[1][:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)
    E[2][:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)
    E[0][0] = -(phi[1] - phi[0]) / dx
    E[0][-1] = -(phi[-1] - phi[-2]) / dx
    E[1][:, 0] = -(phi[:, 1] - phi[:, 0]) / dy
    E[1][:, -1] = -(phi[:, -1] - phi[:, -2]) / dy
    E[2][:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / dz
    E[2][:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / dz
    return E


@pytest.mark.parametrize("n", [3, 8, 32, 48])
def test_spectrum_stored_with_guard_margins(n):
    G = _solver(n)._G_fft
    owner = G.base
    assert owner is not None and owner.ndim == 1
    assert owner.size == G.size + 2
    offset = G.__array_interface__["data"][0] - owner.__array_interface__["data"][0]
    assert offset == G.itemsize                 # one element of margin in front
    assert G.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("kind", ["igf", "point"])
def test_spectrum_values_unchanged(kind):
    s = _solver(24, kind=kind)
    G = s._build_integrated_greens() if kind == "igf" else s._build_point_greens()
    assert np.array_equal(s._G_fft, _fft.rfftn(G))


@pytest.mark.parametrize("n", [16, 24, 40, 48])
@pytest.mark.parametrize("kind", ["igf", "point"])
def test_solve_bit_identical_to_pre_guard_path(n, kind):
    s = _solver(n, kind=kind)
    rho = _rho(n, seed=n)
    got = s.solve(rho)
    want = _solve_pre_guard(s, rho)
    for g, w in zip(got, want):
        assert np.array_equal(g, w)


def test_update_grid_rebuild_keeps_guard():
    s = _solver(24)
    s.update_grid(s.grid_min, s.grid_min + 1.1 * (s.grid_max - s.grid_min))
    G = s._G_fft
    assert G.base is not None and G.base.size == G.size + 2


def _repeat_with_churn(s, rho, n_rep):
    """Solve one density n_rep times while spectrum-sized arrays come and
    go between solves, as in a tracking loop; return the differing count."""
    first = s.solve(rho)
    bad, churn = 0, []
    for i in range(n_rep):
        churn = [np.empty(s._G_fft.shape, complex) for _ in range(i % 4)]
        again = s.solve(rho)
        bad += any(not np.array_equal(a, b) for a, b in zip(first, again))
    del churn
    return bad


@pytest.mark.parametrize("n,n_rep", [(4, 600), (32, 600), (64, 40)])
def test_repeat_bitwise_with_allocation_churn(n, n_rep):
    """Grids whose spectra fill whole allocator blocks.  On the pre-guard
    code (macOS, 2026-10-10) this test saw 4^3 differ in about 80 of 600
    solves and 32^3 in about 5 of 600 (a plain repeat without the churn saw
    none).  64^3 is a regression check only: there the old code took the
    scalar loop in many solves, but a repeat comparison cannot see that
    when every solve takes the same loop."""
    s = _solver(n)
    assert _repeat_with_churn(s, _rho(n, seed=7), n_rep) == 0


def _numpy_fft_solver(n):
    """A solver whose backend uses the numpy.fft fallback (no scipy),
    which returns spectra with permuted strides."""
    s = _solver(n)
    s._backend._fft = np.fft
    s._backend._has_scipy = False
    s._precompute_greens_fft()
    return s


@pytest.mark.parametrize("n", [4, 16, 24])
def test_guard_keeps_the_fft_memory_layout(n):
    s = _numpy_fft_solver(n)
    raw = np.fft.rfftn(s._build_integrated_greens())
    assert s._G_fft.strides == raw.strides            # permuted, kept as is
    assert np.array_equal(s._G_fft, raw)
    assert s._G_fft.base.size == raw.size + 2


@pytest.mark.parametrize("n", [16, 24])
def test_numpy_fft_fallback_repeat_bitwise(n):
    """With a C-ordered guard copy the in-place product mixed layouts and
    drifted (up to 135 of 300 solves at 24^3); keeping the layout fixes it."""
    s = _numpy_fft_solver(n)
    assert _repeat_with_churn(s, _rho(n, seed=3), 300) == 0
