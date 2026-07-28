"""Tests for CIC field interpolation from grid to particles."""
import numpy as np
import pytest
from linac_gen.pic.field_interpolation import interpolate_cic


@pytest.fixture
def simple_grid():
    """A 5x5x5 grid from 0 to 4 mm in each dimension."""
    return dict(
        grid_min=np.array([0.0, 0.0, 0.0]),
        grid_max=np.array([4.0, 4.0, 4.0]),
        n_grid=np.array([5, 5, 5], dtype=np.int64),
    )


class TestOutputShape:
    def test_single_particle(self, simple_grid):
        """Interpolation for 1 particle returns (1, 3)."""
        fx = np.ones((5, 5, 5))
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        pos = np.array([[2.0, 2.0, 2.0]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        assert result.shape == (1, 3)

    def test_multiple_particles(self, simple_grid):
        """Interpolation for N particles returns (N, 3)."""
        fx = np.ones((5, 5, 5))
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        pos = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        assert result.shape == (3, 3)

    def test_empty_particles(self, simple_grid):
        """No particles returns (0, 3)."""
        fx = np.ones((5, 5, 5))
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        pos = np.zeros((0, 3))
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        assert result.shape == (0, 3)


class TestUniformField:
    def test_uniform_field_x(self, simple_grid):
        """A uniform field in x returns the same value for all particles."""
        val = 42.0
        fx = np.full((5, 5, 5), val)
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        rng = np.random.default_rng(99)
        pos = rng.uniform(0.0, 4.0, size=(50, 3))
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        np.testing.assert_allclose(result[:, 0], val, rtol=1e-12)
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(result[:, 2], 0.0, atol=1e-14)

    def test_uniform_field_all_components(self, simple_grid):
        """Uniform fields in all directions are interpolated correctly."""
        fx = np.full((5, 5, 5), 1.0)
        fy = np.full((5, 5, 5), 2.0)
        fz = np.full((5, 5, 5), 3.0)
        pos = np.array([[2.0, 2.0, 2.0], [0.5, 0.5, 0.5], [3.5, 3.5, 3.5]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        np.testing.assert_allclose(result[:, 0], 1.0, rtol=1e-12)
        np.testing.assert_allclose(result[:, 1], 2.0, rtol=1e-12)
        np.testing.assert_allclose(result[:, 2], 3.0, rtol=1e-12)


class TestExactAtGridNode:
    def test_particle_at_grid_node(self, simple_grid):
        """Particle exactly at a grid node gets the exact field at that node."""
        fx = np.zeros((5, 5, 5))
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        # Set a specific value at node (2,2,2)
        fx[2, 2, 2] = 100.0
        fy[2, 2, 2] = 200.0
        fz[2, 2, 2] = 300.0

        pos = np.array([[2.0, 2.0, 2.0]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        np.testing.assert_allclose(result[0, 0], 100.0, rtol=1e-12)
        np.testing.assert_allclose(result[0, 1], 200.0, rtol=1e-12)
        np.testing.assert_allclose(result[0, 2], 300.0, rtol=1e-12)

    def test_particle_at_cell_center_averages_corners(self, simple_grid):
        """Particle at cell center averages the 8 corner values equally."""
        # Cell (1,1,1)-(2,2,2), center at (1.5, 1.5, 1.5)
        # Grid spacing = 1.0, so cell center is at position (1.5, 1.5, 1.5)
        fx = np.zeros((5, 5, 5))
        # Set all 8 corners to different values
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    fx[1 + di, 1 + dj, 1 + dk] = float(di * 4 + dj * 2 + dk)

        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        pos = np.array([[1.5, 1.5, 1.5]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        # At cell center, all 8 weights are 1/8, so result = mean of 8 corners
        expected = np.mean([di * 4 + dj * 2 + dk
                           for di in [0, 1] for dj in [0, 1] for dk in [0, 1]])
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-12)


class TestLinearField:
    def test_linear_field_exact_interpolation(self):
        """CIC (trilinear) interpolation is exact for linear fields."""
        # f(x,y,z) = 2*x + 3*y + 5*z should be interpolated exactly
        grid_min = np.array([0.0, 0.0, 0.0])
        grid_max = np.array([10.0, 10.0, 10.0])
        n_grid = np.array([11, 11, 11], dtype=np.int64)
        dx = 10.0 / 10.0  # 1.0

        # Build field on grid
        fx = np.zeros((11, 11, 11))
        fy = np.zeros((11, 11, 11))
        fz = np.zeros((11, 11, 11))
        for i in range(11):
            for j in range(11):
                for k in range(11):
                    x = i * dx
                    y = j * dx
                    z = k * dx
                    fx[i, j, k] = 2 * x + 3 * y + 5 * z

        # Test at random positions inside the grid
        rng = np.random.default_rng(42)
        pos = rng.uniform(0.0, 10.0, size=(100, 3))
        result = interpolate_cic(fx, fy, fz, pos,
                                 grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

        expected = 2 * pos[:, 0] + 3 * pos[:, 1] + 5 * pos[:, 2]
        np.testing.assert_allclose(result[:, 0], expected, rtol=1e-12)


class TestBoundaryParticles:
    def test_particle_at_lower_boundary(self, simple_grid):
        """Particle at grid_min should not crash and give reasonable values."""
        fx = np.ones((5, 5, 5))
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        pos = np.array([[0.0, 0.0, 0.0]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        np.testing.assert_allclose(result[0, 0], 1.0, rtol=1e-12)

    def test_particle_at_upper_boundary(self, simple_grid):
        """Particle at grid_max should not crash and give reasonable values."""
        fx = np.ones((5, 5, 5))
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        pos = np.array([[4.0, 4.0, 4.0]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        np.testing.assert_allclose(result[0, 0], 1.0, rtol=1e-12)

    def test_particle_outside_grid_clamped(self, simple_grid):
        """Particles outside the grid should be clamped and not crash."""
        fx = np.ones((5, 5, 5))
        fy = np.zeros((5, 5, 5))
        fz = np.zeros((5, 5, 5))
        pos = np.array([[-100.0, 50.0, 200.0]])
        result = interpolate_cic(fx, fy, fz, pos, **simple_grid)
        assert result.shape == (1, 3)
        # Since uniform field, even clamped should give 1.0
        np.testing.assert_allclose(result[0, 0], 1.0, rtol=1e-12)


class TestConsistencyWithDeposition:
    def test_interpolate_at_deposition_point(self):
        """Depositing a delta charge and interpolating at the same point
        should give back the self-field from the node where charge was placed."""
        from linac_gen.pic.charge_deposition import deposit_cic as deposit
        grid_min = np.array([0.0, 0.0, 0.0])
        grid_max = np.array([4.0, 4.0, 4.0])
        n_grid = np.array([5, 5, 5], dtype=np.int64)

        # Place particle at a grid node
        pos = np.array([[2.0, 2.0, 2.0]])
        charges = np.array([1.0])
        rho = deposit(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

        # Use rho as a "field" and interpolate back
        result = interpolate_cic(rho, rho, rho, pos,
                                 grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

        # At node (2,2,2), the deposited value should be rho[2,2,2]
        expected = rho[2, 2, 2]
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-12)

    def test_deposit_interpolate_momentum_conservation(self):
        """For a symmetric pair of particles, the net interpolated force
        should be zero (Newton's third law in discretized form)."""
        grid_min = np.array([-5.0, -5.0, -5.0])
        grid_max = np.array([5.0, 5.0, 5.0])
        n_grid = np.array([11, 11, 11], dtype=np.int64)

        # Symmetric pair of particles along x
        pos = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        charges = np.array([1.0, 1.0])

        from linac_gen.pic.charge_deposition import deposit_cic as deposit
        from linac_gen.pic.poisson_solver import PoissonSolverFFT

        rho = deposit(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)
        solver = PoissonSolverFFT(grid_min, grid_max, n_grid)
        Ex, Ey, Ez = solver.solve(rho)

        fields = interpolate_cic(Ex, Ey, Ez, pos,
                                 grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

        # Sum of forces should be near zero (momentum conservation).
        # The x-component cancels by symmetry but the residual is bounded
        # by float64 round-off at the per-particle field magnitude
        # (~3e9 V/m here), i.e. ~1e-6 in absolute terms.  y and z pick up
        # coarse-grid finite-difference asymmetries up to ~1e-5.
        scale_x = max(np.max(np.abs(fields[:, 0])), 1.0)
        np.testing.assert_allclose(fields[:, 0].sum(), 0.0, atol=1e-12 * scale_x)
        np.testing.assert_allclose(fields[:, 1].sum(), 0.0, atol=1e-4)
        np.testing.assert_allclose(fields[:, 2].sum(), 0.0, atol=1e-4)

        # The two particles should see opposite forces in x
        assert fields[0, 0] > 0, "Right particle should be pushed rightward"
        assert fields[1, 0] < 0, "Left particle should be pushed leftward"
