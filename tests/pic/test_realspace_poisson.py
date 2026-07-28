"""Tests for the real-space (Dirichlet) Poisson solver."""
import numpy as np
import pytest
from linac_gen.pic.realspace_poisson import PoissonSolverRealSpace


def test_zero_charge_zero_field():
    s = PoissonSolverRealSpace(
        grid_min=np.array([-1.0, -1.0, -1.0]),
        grid_max=np.array([+1.0, +1.0, +1.0]),
        n_grid=np.array([8, 8, 8], dtype=np.int64),
    )
    rho = np.zeros((8, 8, 8))
    Ex, Ey, Ez = s.solve(rho)
    np.testing.assert_allclose(Ex, 0.0, atol=1e-12)
    np.testing.assert_allclose(Ey, 0.0, atol=1e-12)
    np.testing.assert_allclose(Ez, 0.0, atol=1e-12)


def test_central_charge_radial_field():
    """A single positive charge at the centre produces an outward
    radial field on the +x axis (and zero by symmetry on y and z)."""
    nx = 17  # odd so the centre is a node
    s = PoissonSolverRealSpace(
        grid_min=np.array([-1.0, -1.0, -1.0]),
        grid_max=np.array([+1.0, +1.0, +1.0]),
        n_grid=np.array([nx, nx, nx], dtype=np.int64),
    )
    c = nx // 2
    rho = np.zeros((nx, nx, nx))
    rho[c, c, c] = 1.0e-9   # arbitrary density
    Ex, Ey, Ez = s.solve(rho)
    # Positive charge → Ex > 0 to the right of the source
    assert Ex[c + 2, c, c] > 0
    # And < 0 to the left
    assert Ex[c - 2, c, c] < 0
    # Off-axis components vanish on the x-axis
    assert abs(Ey[c + 2, c, c]) < 1e-3 * abs(Ex[c + 2, c, c])
    assert abs(Ez[c + 2, c, c]) < 1e-3 * abs(Ex[c + 2, c, c])


def test_grid_too_small_raises():
    with pytest.raises(ValueError, match="≥ 3"):
        PoissonSolverRealSpace(
            grid_min=np.array([0.0, 0.0, 0.0]),
            grid_max=np.array([1.0, 1.0, 1.0]),
            n_grid=np.array([2, 4, 4], dtype=np.int64),
        )


def test_shape_mismatch_raises():
    s = PoissonSolverRealSpace(
        grid_min=np.array([0.0, 0.0, 0.0]),
        grid_max=np.array([1.0, 1.0, 1.0]),
        n_grid=np.array([5, 5, 5], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="rho shape"):
        s.solve(np.zeros((4, 5, 5)))
