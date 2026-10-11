"""The symmetric (mirrored) build of the cell-integrated Green's function:
bit-identical to the exact build on the non-negative displacements, exactly
even, and equal to the exact build within that build's own rounding."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.config import SpaceChargeConfig
from linac_gen.pic import methods
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.pic.poisson_solver import PoissonSolverFFT

_GRIDS = [((48, 48, 48), (0.13, 0.11, 0.29)),
          ((32, 40, 24), (1.0, 0.7, 2.3)),
          ((5, 6, 7), (0.5, 0.25, 1.5)),
          ((16, 16, 16), (1e-4, 1e-4, 3e-4))]


def _pair(n, d):
    n, d = np.array(n), np.array(d)
    lo, hi = np.zeros(3), d * (n - 1)
    return (PoissonSolverFFT(lo, hi, n, use_gpu="cpu", green_build="exact"),
            PoissonSolverFFT(lo, hi, n, use_gpu="cpu", green_build="symmetric"))


def _mirror(k):
    """Hockney index of displacement -m for each index m (size 2k)."""
    return np.concatenate([[0], np.arange(2 * k - 1, 0, -1)])


@pytest.mark.parametrize("n,d", _GRIDS)
def test_symmetric_build_matches_exact(n, d):
    exact, sym = _pair(n, d)
    Ge = exact._build_integrated_greens()
    Gs = sym._build_integrated_greens_symmetric()
    nx, ny, nz = n
    assert Gs.shape == Ge.shape
    # the non-negative displacements are the same computation
    assert np.array_equal(Gs[:nx, :ny, :nz], Ge[:nx, :ny, :nz])
    # exactly even in each axis
    assert np.array_equal(Gs, Gs[np.ix_(_mirror(nx), _mirror(ny), _mirror(nz))])
    # elsewhere the two differ by the exact build's rounding of the 8-corner
    # difference in the far cells (4e-13 of the peak measured at 48^3)
    assert np.max(np.abs(Gs - Ge)) <= 1e-11 * np.max(np.abs(Ge))


def test_solver_fields_agree_and_default_is_exact():
    rng = np.random.default_rng(5)
    n = np.array([24, 24, 24])
    lo, hi = -np.ones(3) * 3e-3, np.ones(3) * 3e-3
    rho = rng.random(tuple(n))
    exact = PoissonSolverFFT(lo, hi, n, use_gpu="cpu")
    sym = PoissonSolverFFT(lo, hi, n, use_gpu="cpu", green_build="symmetric")
    assert exact.green_build == "exact"
    for ee, es in zip(exact.solve(rho), sym.solve(rho)):    # Ex, Ey, Ez
        ee, es = np.asarray(ee), np.asarray(es)
        assert np.max(np.abs(es - ee)) <= 1e-12 * np.max(np.abs(ee))


def test_unknown_build_is_refused():
    with pytest.raises(ValueError, match="green_build"):
        PoissonSolverFFT(np.zeros(3), np.ones(3), np.array([4, 4, 4]),
                         use_gpu="cpu", green_build="mirrored")
    with pytest.raises(ValueError, match="green_build"):
        SpaceChargeConfig(green_build="mirrored")


def test_build_choice_reaches_the_pic_solver():
    rng = np.random.default_rng(1)
    coords = rng.normal(scale=1e-3, size=(500, 3))
    for build in ("exact", "symmetric"):
        pic = PicSolver(SpaceChargeConfig(nx=8, ny=8, nz=8, use_gpu="cpu",
                                          grid_mode="adaptive",
                                          green_build=build))
        pic._setup_grid(coords)
        assert pic._solver.green_build == build


def test_defaults_per_method():
    assert SpaceChargeConfig().green_build == "exact"
    assert SpaceChargeConfig(grid_mode="adaptive").green_build == "exact"
    for name in ("cic_fixed", "tsc_fixed", "differentiable_pic"):
        assert SpaceChargeConfig(method=name).green_build == "exact"
    for name in ("cic_adaptive", "tsc_adaptive"):
        cfg = SpaceChargeConfig(method=name)
        assert cfg.green_build == "symmetric" and cfg.method == name
        # the exact build is a param of the same method
        cfg = SpaceChargeConfig(method=name, green_build="exact")
        assert cfg.method == name
    assert methods.CLASSIC["green_build"] == "exact"
