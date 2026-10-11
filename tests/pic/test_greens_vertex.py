"""The shared-vertex IGF build must equal the original per-corner build BIT
FOR BIT, and ``update_grid`` may keep the Green's function only when the cell
spacing is exactly unchanged.  The per-corner builder below is a literal copy
of the pre-2026-10 ``PoissonSolverFFT._build_integrated_greens`` (numpy), kept
as the reference for the identity check.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.constants import EPSILON_0, PI
from linac_gen.pic.poisson_solver import PoissonSolverFFT, _antideriv_igf


def _per_corner_igf(n_grid, d):
    """Literal copy of the original per-corner builder, with xp = numpy."""
    xp = np
    nx, ny, nz = int(n_grid[0]), int(n_grid[1]), int(n_grid[2])
    dx, dy, dz = d

    ii = xp.arange(2 * nx, dtype=xp.float64)
    jj = xp.arange(2 * ny, dtype=xp.float64)
    kk = xp.arange(2 * nz, dtype=xp.float64)
    ii = xp.where(ii < nx, ii, ii - 2 * nx)
    jj = xp.where(jj < ny, jj, jj - 2 * ny)
    kk = xp.where(kk < nz, kk, kk - 2 * nz)

    xL = (ii - 0.5) * dx
    xH = (ii + 0.5) * dx
    yL = (jj - 0.5) * dy
    yH = (jj + 0.5) * dy
    zL = (kk - 0.5) * dz
    zH = (kk + 0.5) * dz

    XL, YL, ZL = xp.meshgrid(xL, yL, zL, indexing="ij")
    XH, YH, ZH = xp.meshgrid(xH, yH, zH, indexing="ij")

    F_HHH = _antideriv_igf(xp, XH, YH, ZH)
    F_LHH = _antideriv_igf(xp, XL, YH, ZH)
    F_HLH = _antideriv_igf(xp, XH, YL, ZH)
    F_HHL = _antideriv_igf(xp, XH, YH, ZL)
    F_LLH = _antideriv_igf(xp, XL, YL, ZH)
    F_LHL = _antideriv_igf(xp, XL, YH, ZL)
    F_HLL = _antideriv_igf(xp, XH, YL, ZL)
    F_LLL = _antideriv_igf(xp, XL, YL, ZL)

    cell_vol = dx * dy * dz
    G = (
        F_HHH - F_LHH - F_HLH - F_HHL
        + F_LLH + F_LHL + F_HLL - F_LLL
    ) / cell_vol
    G *= 1.0 / (4.0 * PI * EPSILON_0)
    return G


GRIDS = [((3, 3, 3), (0.5, 0.5, 0.5)), ((4, 3, 5), (0.2, 0.3, 0.7)),
         ((8, 8, 8), (0.2, 0.3, 0.7)), ((32, 32, 32), (0.071, 0.071, 0.071)),
         ((48, 48, 48), (0.1, 0.1, 0.1)), ((48, 40, 64), (0.071, 0.083, 0.157)),
         ((64, 64, 64), (0.1, 0.12, 0.31))]


def _solver(n, d):
    n = np.asarray(n)
    return PoissonSolverFFT(np.zeros(3), np.asarray(d) * (n - 1), n, use_gpu="cpu")


@pytest.mark.parametrize("n,d", GRIDS)
def test_vertex_build_bit_identical_to_per_corner(n, d):
    s = _solver(n, d)
    assert np.array_equal(s._build_integrated_greens(), _per_corner_igf(s.n_grid, s.dx))


def test_vertex_build_bit_identical_off_centre_grid():
    """Spacing from bounds that are not multiples of the cell size."""
    s = PoissonSolverFFT(np.array([-3.71, -1.13, -9.07]), np.array([4.29, 2.87, 8.93]),
                         np.array([40, 36, 52]), use_gpu="cpu")
    assert np.array_equal(s._build_integrated_greens(), _per_corner_igf(s.n_grid, s.dx))


def _exact_solver():
    """Bounds exactly representable, so a translation keeps dx bit-exact."""
    return PoissonSolverFFT(np.zeros(3), np.array([3.0, 3.0, 4.5]), np.array([16, 16, 16]),
                            use_gpu="cpu")


def test_update_grid_keeps_green_only_for_identical_spacing(monkeypatch):
    s = _exact_solver()
    calls = []
    orig = s._precompute_greens_fft
    monkeypatch.setattr(s, "_precompute_greens_fft", lambda: (calls.append(1), orig())[1])
    G0 = s._G_fft
    # translated grid, same spacing: kept, but the bounds still move
    s.update_grid(s.grid_min + 1.25, s.grid_max + 1.25)
    assert calls == [] and s._G_fft is G0
    assert np.array_equal(s.grid_min, np.full(3, 1.25))
    # different spacing: rebuilt, and equal to a fresh solver's
    s.update_grid(np.zeros(3), np.array([0.25, 0.2, 0.3]) * 15)
    assert calls == [1]
    fresh = PoissonSolverFFT(np.zeros(3), np.array([0.25, 0.2, 0.3]) * 15, np.array([16, 16, 16]),
                             use_gpu="cpu")
    assert np.array_equal(s._G_fft, fresh._G_fft)
    assert np.array_equal(s.dx, fresh.dx)


def test_update_grid_one_ulp_spacing_change_rebuilds(monkeypatch):
    s = _exact_solver()
    calls = []
    orig = s._precompute_greens_fft
    monkeypatch.setattr(s, "_precompute_greens_fft", lambda: (calls.append(1), orig())[1])
    gmax = s.grid_max.copy()
    gmax[2] = np.nextafter(gmax[2], np.inf)
    s.update_grid(s.grid_min, gmax)
    assert calls == [1]


def test_translated_grid_solves_identically():
    rho = np.random.default_rng(2).random((16, 16, 16))
    a = _exact_solver()
    b = _exact_solver()
    G0 = b._G_fft
    b.update_grid(b.grid_min - 3.0, b.grid_max - 3.0)             # Green's function kept
    assert b._G_fft is G0
    for ea, eb in zip(a.solve(rho), b.solve(rho)):
        assert np.array_equal(ea, eb)


def test_point_kernel_update_grid_shortcut():
    s = PoissonSolverFFT(np.zeros(3), np.array([3.0, 3.0, 4.5]), np.array([12, 12, 12]),
                         use_gpu="cpu", green_kind="point")
    G0 = s._G_fft
    s.update_grid(s.grid_min + 0.5, s.grid_max + 0.5)
    assert s._G_fft is G0
    s.update_grid(np.zeros(3), np.array([3.0, 3.0, 5.5]))
    fresh = PoissonSolverFFT(np.zeros(3), np.array([3.0, 3.0, 5.5]), np.array([12, 12, 12]),
                             use_gpu="cpu", green_kind="point")
    assert np.array_equal(s._G_fft, fresh._G_fft)


def test_update_grid_retries_a_rebuild_that_raised(monkeypatch):
    """A rebuild that raises must not leave the new spacing marked as built:
    a retry with the same bounds rebuilds (adversarial review 2026-10-10)."""
    n = np.array([12, 12, 12])
    a_min, a_max, b_max = np.zeros(3), np.array([3.0, 3.0, 4.5]), np.array([2.0, 3.5, 4.0])
    s = PoissonSolverFFT(a_min, a_max, n, use_gpu="cpu")

    def boom():
        raise MemoryError("simulated failure inside the Green's-function build")

    monkeypatch.setattr(s, "_build_integrated_greens", boom)
    with pytest.raises(MemoryError):
        s.update_grid(a_min, b_max)
    monkeypatch.undo()
    s.update_grid(a_min, b_max)                       # same bounds: must rebuild now
    fresh = PoissonSolverFFT(a_min, b_max, n, use_gpu="cpu")
    assert np.array_equal(s._G_fft, fresh._G_fft)
