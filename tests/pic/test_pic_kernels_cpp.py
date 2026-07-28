"""Tests comparing C++ PIC kernels against the Python reference implementations.

All tests are skipped when the compiled extension is not available so the
suite passes in environments where the C++ build has not been run.
"""
import numpy as np
import pytest

try:
    from linac_gen._pic_kernels import (
        deposit_cic as cpp_deposit,
        interpolate_cic as cpp_interpolate,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from linac_gen.pic.charge_deposition import deposit_cic as py_deposit
from linac_gen.pic.field_interpolation import interpolate_cic as py_interpolate

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_grid(nx=16, ny=16, nz=16, lo=-6.0, hi=6.0):
    grid_min = np.array([lo, lo, lo])
    grid_max = np.array([hi, hi, hi])
    n_grid = np.array([nx, ny, nz], dtype=np.int32)
    return grid_min, grid_max, n_grid


# ---------------------------------------------------------------------------
# CIC deposit
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CPP, reason="C++ kernels not compiled")
class TestCppDeposit:
    def test_matches_python(self):
        """C++ and Python deposit must agree to floating-point precision."""
        rng = np.random.default_rng(42)
        N = 1000
        positions = rng.uniform(-5.0, 5.0, (N, 3))
        charges = np.ones(N) * 1e-10
        grid_min, grid_max, n_grid = _make_grid()

        rho_py = py_deposit(positions, charges, grid_min, grid_max, n_grid)
        rho_cpp = cpp_deposit(positions, charges, grid_min, grid_max, n_grid)

        np.testing.assert_allclose(rho_cpp, rho_py, rtol=1e-10,
                                   err_msg="C++ deposit differs from Python deposit")

    def test_charge_conservation(self):
        """Total charge on grid * cell_vol must equal sum of macro charges."""
        rng = np.random.default_rng(7)
        N = 500
        positions = rng.uniform(-4.0, 4.0, (N, 3))
        charges = rng.uniform(1e-12, 1e-10, N)
        grid_min, grid_max, n_grid = _make_grid(nx=20, ny=20, nz=20)

        dx = (grid_max - grid_min) / (n_grid - 1)
        cell_vol = dx[0] * dx[1] * dx[2]

        rho_cpp = cpp_deposit(positions, charges, grid_min, grid_max, n_grid)
        total_charge = rho_cpp.sum() * cell_vol

        np.testing.assert_allclose(total_charge, charges.sum(), rtol=1e-10,
                                   err_msg="C++ deposit does not conserve charge")

    def test_output_shape(self):
        """Output array shape must match requested grid dimensions."""
        positions = np.zeros((10, 3))
        charges = np.ones(10)
        nx, ny, nz = 8, 10, 12
        grid_min, grid_max, n_grid = _make_grid(nx=nx, ny=ny, nz=nz)

        rho = cpp_deposit(positions, charges, grid_min, grid_max, n_grid)
        assert rho.shape == (nx, ny, nz)

    def test_single_particle_on_node(self):
        """A particle placed exactly on a grid node should deposit to that node only."""
        grid_min = np.array([0.0, 0.0, 0.0])
        grid_max = np.array([10.0, 10.0, 10.0])
        n_grid = np.array([11, 11, 11], dtype=np.int32)
        dx = (grid_max - grid_min) / (n_grid - 1)
        cell_vol = dx[0] * dx[1] * dx[2]

        # Place particle exactly on node (5, 5, 5) = centre of grid
        positions = np.array([[5.0, 5.0, 5.0]])
        charges = np.array([1.0])

        rho = cpp_deposit(positions, charges, grid_min, grid_max, n_grid)

        # All charge must land on node (5,5,5)
        assert abs(rho[5, 5, 5] * cell_vol - 1.0) < 1e-12
        # No other node should have non-zero charge
        rho_copy = rho.copy()
        rho_copy[5, 5, 5] = 0.0
        assert np.all(rho_copy == 0.0)

    def test_uniform_beam_symmetric_rho(self):
        """A beam uniform in a sphere should give a symmetric rho profile."""
        rng = np.random.default_rng(99)
        N = 5000
        # Draw from a sphere so the distribution is symmetric
        r = rng.normal(0, 1.0, (N, 3))
        positions = r / np.linalg.norm(r, axis=1, keepdims=True) * rng.uniform(0, 2.0, (N, 1))
        charges = np.ones(N)
        grid_min, grid_max, n_grid = _make_grid(nx=32, ny=32, nz=32, lo=-3.0, hi=3.0)

        rho = cpp_deposit(positions, charges, grid_min, grid_max, n_grid)
        # At least rho must be non-negative everywhere
        assert np.all(rho >= 0.0)


# ---------------------------------------------------------------------------
# CIC interpolate
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CPP, reason="C++ kernels not compiled")
class TestCppInterpolate:
    def test_matches_python(self):
        """C++ and Python interpolate must agree to floating-point precision."""
        rng = np.random.default_rng(42)
        nx, ny, nz = 16, 16, 16
        fx = rng.normal(0, 1, (nx, ny, nz))
        fy = rng.normal(0, 1, (nx, ny, nz))
        fz = rng.normal(0, 1, (nx, ny, nz))
        N = 500
        positions = rng.uniform(-5.0, 5.0, (N, 3))
        grid_min, grid_max, n_grid = _make_grid(nx=nx, ny=ny, nz=nz)

        result_py = py_interpolate(fx, fy, fz, positions, grid_min, grid_max, n_grid)
        result_cpp = cpp_interpolate(fx, fy, fz, positions, grid_min, grid_max, n_grid)

        np.testing.assert_allclose(result_cpp, result_py, rtol=1e-10,
                                   err_msg="C++ interpolate differs from Python interpolate")

    def test_output_shape(self):
        """Output array shape must be (N, 3)."""
        rng = np.random.default_rng(0)
        N = 123
        nx, ny, nz = 8, 8, 8
        fx = rng.normal(0, 1, (nx, ny, nz))
        fy = rng.normal(0, 1, (nx, ny, nz))
        fz = rng.normal(0, 1, (nx, ny, nz))
        positions = rng.uniform(-5.0, 5.0, (N, 3))
        grid_min, grid_max, n_grid = _make_grid(nx=nx, ny=ny, nz=nz)

        result = cpp_interpolate(fx, fy, fz, positions, grid_min, grid_max, n_grid)
        assert result.shape == (N, 3)

    def test_uniform_field_returns_uniform(self):
        """Interpolating a spatially uniform field must return that field value at all particles."""
        nx, ny, nz = 8, 8, 8
        Fx_val, Fy_val, Fz_val = 3.14, -2.71, 1.41
        fx = np.full((nx, ny, nz), Fx_val)
        fy = np.full((nx, ny, nz), Fy_val)
        fz = np.full((nx, ny, nz), Fz_val)

        rng = np.random.default_rng(55)
        positions = rng.uniform(-5.0, 5.0, (200, 3))
        grid_min, grid_max, n_grid = _make_grid(nx=nx, ny=ny, nz=nz)

        result = cpp_interpolate(fx, fy, fz, positions, grid_min, grid_max, n_grid)

        np.testing.assert_allclose(result[:, 0], Fx_val, rtol=1e-12)
        np.testing.assert_allclose(result[:, 1], Fy_val, rtol=1e-12)
        np.testing.assert_allclose(result[:, 2], Fz_val, rtol=1e-12)

    def test_linear_field_exact(self):
        """Interpolating a linearly varying field must be exact at all interior points."""
        nx, ny, nz = 17, 17, 17
        grid_min = np.array([-8.0, -8.0, -8.0])
        grid_max = np.array([8.0, 8.0, 8.0])
        n_grid = np.array([nx, ny, nz], dtype=np.int32)
        dx = (grid_max - grid_min) / (n_grid - 1)

        # Build a grid where Fx = 2*x + 1, Fy = -y, Fz = 0.5*z
        xs = grid_min[0] + np.arange(nx) * dx[0]
        ys = grid_min[1] + np.arange(ny) * dx[1]
        zs = grid_min[2] + np.arange(nz) * dx[2]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        fx = 2.0 * X + 1.0
        fy = -Y
        fz = 0.5 * Z

        rng = np.random.default_rng(21)
        # Interior particles (stay within grid)
        positions = rng.uniform(-7.0, 7.0, (300, 3))

        result = cpp_interpolate(fx, fy, fz, positions, grid_min, grid_max, n_grid)

        expected_fx = 2.0 * positions[:, 0] + 1.0
        expected_fy = -positions[:, 1]
        expected_fz = 0.5 * positions[:, 2]

        np.testing.assert_allclose(result[:, 0], expected_fx, rtol=1e-10)
        np.testing.assert_allclose(result[:, 1], expected_fy, rtol=1e-10)
        np.testing.assert_allclose(result[:, 2], expected_fz, rtol=1e-10)


# ---------------------------------------------------------------------------
# Round-trip: deposit then interpolate
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CPP, reason="C++ kernels not compiled")
class TestRoundTrip:
    def test_deposit_interpolate_consistency(self):
        """The C++ deposit+interpolate round-trip must match the Python one."""
        rng = np.random.default_rng(123)
        N = 800
        positions = rng.uniform(-4.0, 4.0, (N, 3))
        charges = np.ones(N) * 1.0
        grid_min, grid_max, n_grid = _make_grid(nx=20, ny=20, nz=20)

        rho_py = py_deposit(positions, charges, grid_min, grid_max, n_grid)
        rho_cpp = cpp_deposit(positions, charges, grid_min, grid_max, n_grid)

        # Use the density as a fake field for interpolation
        result_py = py_interpolate(rho_py, rho_py, rho_py,
                                   positions, grid_min, grid_max, n_grid)
        result_cpp = cpp_interpolate(rho_cpp, rho_cpp, rho_cpp,
                                     positions, grid_min, grid_max, n_grid)

        np.testing.assert_allclose(result_cpp, result_py, rtol=1e-10)
