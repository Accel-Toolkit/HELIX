"""Tests for Cloud-In-Cell (CIC) charge deposition."""
import numpy as np
import pytest
from linac_gen.pic.charge_deposition import deposit_cic


@pytest.fixture
def simple_grid():
    """A 5x5x5 grid from -10 to 10 mm in each dimension."""
    return dict(
        grid_min=np.array([-10.0, -10.0, -10.0]),
        grid_max=np.array([10.0, 10.0, 10.0]),
        n_grid=np.array([5, 5, 5], dtype=np.int64),
    )


def test_output_shape(simple_grid):
    """Output density grid has shape (nx, ny, nz)."""
    pos = np.array([[0.0, 0.0, 0.0]])
    charges = np.array([1.0])
    rho = deposit_cic(pos, charges, **simple_grid)
    assert rho.shape == (5, 5, 5)


def test_single_particle_at_grid_center_equal_weights(simple_grid):
    """Particle at cell center deposits 1/8 to each of the 8 surrounding nodes."""
    # Grid: 5 points from -10 to 10, spacing = 5.0
    # Cell centers at -7.5, -2.5, 2.5, 7.5
    # Place particle at center of cell (1,1,1) which is at (-2.5, -2.5, -2.5)
    pos = np.array([[-2.5, -2.5, -2.5]])
    charges = np.array([1.0])
    rho = deposit_cic(pos, charges, **simple_grid)

    # Particle is at the center of cell (1,1,1)-(2,2,2)
    # fx = fy = fz = 0.5, so each of 8 corners gets weight = 0.125
    # But rho is charge density = deposited_charge / cell_volume
    dx = 20.0 / 4.0  # = 5.0
    cell_vol = dx ** 3  # = 125.0

    for di in [0, 1]:
        for dj in [0, 1]:
            for dk in [0, 1]:
                expected = 0.125 * 1.0 / cell_vol
                assert abs(rho[1 + di, 1 + dj, 1 + dk] - expected) < 1e-14, \
                    f"Node ({1+di},{1+dj},{1+dk}) has rho={rho[1+di,1+dj,1+dk]}, expected {expected}"


def test_charge_conservation(simple_grid):
    """Total deposited charge must equal sum of input charges."""
    rng = np.random.default_rng(42)
    n = 100
    pos = rng.uniform(-9.0, 9.0, size=(n, 3))
    charges = rng.uniform(0.5, 2.0, size=n)
    rho = deposit_cic(pos, charges, **simple_grid)

    dx = 20.0 / 4.0
    cell_vol = dx ** 3
    total_deposited = rho.sum() * cell_vol
    np.testing.assert_allclose(total_deposited, charges.sum(), rtol=1e-12)


def test_particle_at_grid_node():
    """Particle exactly at a grid node deposits 100% to that node."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([4.0, 4.0, 4.0])
    n_grid = np.array([5, 5, 5], dtype=np.int64)

    # Place particle at node (2, 2, 2) which is at position (2.0, 2.0, 2.0)
    pos = np.array([[2.0, 2.0, 2.0]])
    charges = np.array([1.0])
    rho = deposit_cic(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

    dx = 4.0 / 4.0  # = 1.0
    cell_vol = dx ** 3
    expected = 1.0 / cell_vol

    assert abs(rho[2, 2, 2] - expected) < 1e-14
    # All other nodes should be zero
    rho[2, 2, 2] = 0.0
    assert abs(rho.sum()) < 1e-14


def test_particles_outside_grid_clamped():
    """Particles outside the grid boundary are clamped, not crashed."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([10.0, 10.0, 10.0])
    n_grid = np.array([5, 5, 5], dtype=np.int64)

    # Particle far outside the grid
    pos = np.array([[-100.0, 50.0, 200.0]])
    charges = np.array([1.0])
    # Should not raise
    rho = deposit_cic(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)
    assert rho.shape == (5, 5, 5)

    # Charge conservation still holds
    dx = 10.0 / 4.0
    cell_vol = dx ** 3
    total = rho.sum() * cell_vol
    np.testing.assert_allclose(total, 1.0, rtol=1e-12)


def test_symmetric_distribution_symmetric_density():
    """A symmetric particle distribution must produce a symmetric density."""
    grid_min = np.array([-5.0, -5.0, -5.0])
    grid_max = np.array([5.0, 5.0, 5.0])
    n_grid = np.array([11, 11, 11], dtype=np.int64)

    # Place 8 particles symmetrically around origin
    d = 1.5
    positions = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                positions.append([sx * d, sy * d, sz * d])
    pos = np.array(positions)
    charges = np.ones(8)

    rho = deposit_cic(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

    # Check symmetry: rho[i,j,k] == rho[10-i,j,k] (mirror in x)
    np.testing.assert_allclose(rho, rho[::-1, :, :], atol=1e-14)
    # Mirror in y
    np.testing.assert_allclose(rho, rho[:, ::-1, :], atol=1e-14)
    # Mirror in z
    np.testing.assert_allclose(rho, rho[:, :, ::-1], atol=1e-14)


def test_two_particles_additive():
    """Charge from two particles at same location should add."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([4.0, 4.0, 4.0])
    n_grid = np.array([5, 5, 5], dtype=np.int64)

    pos_one = np.array([[2.0, 2.0, 2.0]])
    pos_two = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

    rho_one = deposit_cic(pos_one, np.array([1.0]),
                          grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)
    rho_two = deposit_cic(pos_two, np.array([1.0, 1.0]),
                          grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

    np.testing.assert_allclose(rho_two, 2.0 * rho_one, atol=1e-14)


def test_different_charges():
    """Particles with different charge values deposit proportionally."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([4.0, 4.0, 4.0])
    n_grid = np.array([5, 5, 5], dtype=np.int64)

    pos = np.array([[1.5, 1.5, 1.5]])

    rho_q1 = deposit_cic(pos, np.array([1.0]),
                         grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)
    rho_q3 = deposit_cic(pos, np.array([3.0]),
                         grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

    np.testing.assert_allclose(rho_q3, 3.0 * rho_q1, atol=1e-14)


def test_particle_at_lower_boundary():
    """Particle exactly at grid_min is handled correctly."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([4.0, 4.0, 4.0])
    n_grid = np.array([5, 5, 5], dtype=np.int64)

    pos = np.array([[0.0, 0.0, 0.0]])
    charges = np.array([1.0])
    rho = deposit_cic(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

    dx = 1.0
    cell_vol = dx ** 3
    assert abs(rho[0, 0, 0] - 1.0 / cell_vol) < 1e-14


def test_particle_at_upper_boundary():
    """Particle exactly at grid_max is handled correctly (clamped to last valid cell)."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([4.0, 4.0, 4.0])
    n_grid = np.array([5, 5, 5], dtype=np.int64)

    pos = np.array([[4.0, 4.0, 4.0]])
    charges = np.array([1.0])
    rho = deposit_cic(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)

    dx = 1.0
    cell_vol = dx ** 3
    # Clamped to cell (3,3,3)-(4,4,4) with fx=fy=fz=1.0
    # All weight goes to node (4,4,4)
    assert abs(rho[4, 4, 4] - 1.0 / cell_vol) < 1e-14


def test_empty_particles():
    """No particles should give zero density."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([4.0, 4.0, 4.0])
    n_grid = np.array([5, 5, 5], dtype=np.int64)

    pos = np.zeros((0, 3))
    charges = np.zeros(0)
    rho = deposit_cic(pos, charges, grid_min=grid_min, grid_max=grid_max, n_grid=n_grid)
    assert rho.shape == (5, 5, 5)
    assert abs(rho.sum()) < 1e-15


# ---------------------------------------------------------------------------
# TSC tests
# ---------------------------------------------------------------------------
class TestTSCWeights:
    def test_partition_of_unity(self):
        """1-D TSC weights sum to 1 for any fractional position."""
        from linac_gen.pic.charge_deposition import _tsc_weights_1d
        for f in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            wm, w0, wp = _tsc_weights_1d(np.array([f]))
            np.testing.assert_allclose(float((wm + w0 + wp)[0]), 1.0, atol=1e-12)

    def test_centered_particle_max_central_weight(self):
        """Particle at f=0.5 (cell centre) has w_zero=3/4, w_±=1/8 each."""
        from linac_gen.pic.charge_deposition import _tsc_weights_1d
        wm, w0, wp = _tsc_weights_1d(np.array([0.5]))
        np.testing.assert_allclose(float(w0[0]), 0.75, atol=1e-12)
        np.testing.assert_allclose(float(wm[0]), 0.125, atol=1e-12)
        np.testing.assert_allclose(float(wp[0]), 0.125, atol=1e-12)


class TestTSCDeposit:
    def test_total_charge_conserved(self):
        from linac_gen.pic.charge_deposition import deposit_tsc
        gmin = np.array([-5.0, -5.0, -5.0])
        gmax = np.array([+5.0, +5.0, +5.0])
        ng = np.array([16, 16, 16], dtype=np.int64)
        dx = (gmax - gmin) / (ng - 1)
        cell_vol = dx.prod()
        rng = np.random.default_rng(7)
        N = 200
        pos = rng.uniform(-3.0, 3.0, size=(N, 3))
        q = np.full(N, 0.5)
        rho = deposit_tsc(pos, q, gmin, gmax, ng) * cell_vol
        np.testing.assert_allclose(rho.sum(), N * 0.5, atol=1e-12)

    def test_27_cell_stencil(self):
        """A particle off-centre deposits into 27 cells (3×3×3)."""
        from linac_gen.pic.charge_deposition import deposit_tsc
        gmin = np.array([-5.0, -5.0, -5.0])
        gmax = np.array([+5.0, +5.0, +5.0])
        ng = np.array([11, 11, 11], dtype=np.int64)
        # Particle slightly inside one cell — should hit 27 cells.
        pos = np.array([[0.21, 0.34, -0.17]])
        q = np.array([1.0])
        rho = deposit_tsc(pos, q, gmin, gmax, ng)
        nonzero = np.count_nonzero(rho)
        assert nonzero == 27, f"TSC deposit should fill 27 cells, got {nonzero}"
