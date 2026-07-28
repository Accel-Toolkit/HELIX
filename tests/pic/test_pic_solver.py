"""Tests for the PIC solver integration (PicSolver)."""
import numpy as np
import pytest
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.constants import EPSILON_0, PI, E_CHARGE, C_LIGHT


def _make_beam(n=100, energy=3.0, current=0.0, sigma_x=1.0, sigma_y=1.0,
               sigma_dphi=5.0, sigma_dw=0.01, frequency=352.21):
    """Create a beam with Gaussian-distributed particles for testing."""
    ref = ReferenceParticle(species=PROTON, w_kin=energy, frequency=frequency)
    beam = Beam(ref=ref, n_particles=n, current=current)
    rng = np.random.default_rng(42)
    beam.particles[:, 0] = rng.normal(0.0, sigma_x, n)      # x (mm)
    beam.particles[:, 1] = rng.normal(0.0, 0.1, n)          # xp (mrad)
    beam.particles[:, 2] = rng.normal(0.0, sigma_y, n)      # y (mm)
    beam.particles[:, 3] = rng.normal(0.0, 0.1, n)          # yp (mrad)
    beam.particles[:, 4] = rng.normal(0.0, sigma_dphi, n)   # dphi (deg)
    beam.particles[:, 5] = rng.normal(0.0, sigma_dw, n)     # dW (MeV)
    return beam


@pytest.fixture
def sc_config():
    """Small grid space-charge config for fast tests."""
    return SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0)


class TestPicSolverInit:
    def test_creation(self, sc_config):
        """PicSolver can be created from a SpaceChargeConfig."""
        pic = PicSolver(sc_config)
        assert pic is not None
        assert pic.config is sc_config


class TestNoCurrentNoKick:
    def test_zero_current_leaves_particles_unchanged(self, sc_config):
        """When beam.current=0, kick() is a no-op."""
        beam = _make_beam(n=50, current=0.0)
        particles_before = beam.particles.copy()
        pic = PicSolver(sc_config)
        pic.kick(beam, ds=10.0)
        np.testing.assert_array_equal(beam.particles, particles_before)

    def test_single_alive_particle_no_kick(self, sc_config):
        """With fewer than 2 alive particles, kick() is a no-op."""
        beam = _make_beam(n=5, current=10.0)
        # Kill all but one
        for i in range(1, 5):
            beam.lost[i] = True
        particles_before = beam.particles.copy()
        pic = PicSolver(sc_config)
        pic.kick(beam, ds=10.0)
        np.testing.assert_array_equal(beam.particles, particles_before)


class TestRepulsiveKick:
    def test_beam_expands_after_kick(self, sc_config):
        """A bunched beam with positive current should expand after a kick."""
        beam = _make_beam(n=200, current=20.0, sigma_x=1.0, sigma_y=1.0)
        sigma_x_before = np.std(beam.particles[beam.alive_mask, 0])
        sigma_xp_before = np.std(beam.particles[beam.alive_mask, 1])

        pic = PicSolver(sc_config)
        # Apply a large kick so the effect is visible
        pic.kick(beam, ds=100.0)

        # The momentum spread should increase (repulsive kicks push outward)
        sigma_xp_after = np.std(beam.particles[beam.alive_mask, 1])
        assert sigma_xp_after > sigma_xp_before, \
            f"xp spread should increase: before={sigma_xp_before}, after={sigma_xp_after}"

    def test_kick_magnitude_scales_with_current(self, sc_config):
        """Doubling the current should approximately double the kick."""
        beam1 = _make_beam(n=100, current=10.0)
        beam2 = _make_beam(n=100, current=20.0)  # same particles, double current
        particles_orig = beam1.particles.copy()

        pic1 = PicSolver(sc_config)
        pic2 = PicSolver(sc_config)

        pic1.kick(beam1, ds=10.0)
        pic2.kick(beam2, ds=10.0)

        # Change in xp for each beam
        dxp1 = beam1.particles[:, 1] - particles_orig[:, 1]
        dxp2 = beam2.particles[:, 1] - particles_orig[:, 1]

        # dxp2 should be ~2x dxp1 (same grid, same geometry, double charge)
        # Allow some tolerance since grid setup may differ slightly
        ratio = np.std(dxp2) / np.std(dxp1)
        np.testing.assert_allclose(ratio, 2.0, rtol=0.05)


class TestKickSymmetry:
    def test_symmetric_beam_symmetric_kicks(self, sc_config):
        """For a beam centered at origin, the mean kick should be near zero."""
        beam = _make_beam(n=500, current=10.0)
        particles_before = beam.particles.copy()

        pic = PicSolver(sc_config)
        pic.kick(beam, ds=10.0)

        dxp = beam.particles[:, 1] - particles_before[:, 1]
        dyp = beam.particles[:, 3] - particles_before[:, 3]

        # Mean kick should be near zero for a symmetric beam
        # (finite N means some asymmetry, but should be small compared to spread)
        assert abs(np.mean(dxp)) < 0.3 * np.std(dxp), \
            "Mean xp kick should be small compared to spread"
        assert abs(np.mean(dyp)) < 0.3 * np.std(dyp), \
            "Mean yp kick should be small compared to spread"


class TestChargeConservation:
    def test_deposited_charge_matches_beam(self, sc_config):
        """Total charge deposited on the grid equals beam charge per bunch."""
        from linac_gen.pic.coordinates import beam_to_spatial
        from linac_gen.pic.lorentz_boost import boost_to_rest_frame
        from linac_gen.pic.charge_deposition import deposit_cic

        beam = _make_beam(n=200, current=10.0)
        coords_lab = beam_to_spatial(beam)
        gamma = beam.ref.gamma
        coords_rest = boost_to_rest_frame(coords_lab, gamma)

        # Compute expected charge per macro-particle
        current_A = beam.current * 1e-3
        freq_Hz = beam.ref.frequency * 1e6
        charge_per_bunch = current_A / freq_Hz
        macro_charge = charge_per_bunch / beam.n_particles
        charges = np.full(beam.n_alive, macro_charge)

        # Set up grid
        std = np.std(coords_rest, axis=0)
        mean = np.mean(coords_rest, axis=0)
        half_size = 4.0 * std
        half_size = np.maximum(half_size, 1e-6)
        grid_min = mean - half_size
        grid_max = mean + half_size
        n_grid = np.array([16, 16, 16], dtype=np.int64)
        dx = (grid_max - grid_min) / (n_grid - 1).astype(float)
        cell_vol = dx[0] * dx[1] * dx[2]

        rho = deposit_cic(coords_rest, charges, grid_min, grid_max, n_grid)
        total_deposited = rho.sum() * cell_vol
        total_expected = charges.sum()

        np.testing.assert_allclose(total_deposited, total_expected, rtol=1e-10)


class TestKickPhysics:
    def test_transverse_kick_direction(self, sc_config):
        """Space charge defocuses: kicks correlate positively with displacement."""
        # For a self-consistent PIC solve the sum of internal kicks is zero
        # (Newton's third law). What must be positive is the correlation
        # between a particle's displacement from the charge centroid and its
        # kick: particles on the +x side get +dxp, particles on the -x side
        # get -dxp -- that is what "defocusing" means.
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        beam = Beam(ref=ref, n_particles=50, current=10.0)
        rng = np.random.default_rng(55)
        beam.particles[:, 0] = rng.uniform(0.5, 2.0, 50)
        beam.particles[:, 2] = rng.normal(0.0, 0.5, 50)
        beam.particles[:, 4] = rng.normal(0.0, 3.0, 50)
        beam.particles[:, 1] = 0.0
        beam.particles[:, 3] = 0.0
        beam.particles[:, 5] = 0.0

        x = beam.particles[:, 0].copy()
        xp_before = beam.particles[:, 1].copy()
        pic = PicSolver(sc_config)
        pic.kick(beam, ds=50.0)
        dxp = beam.particles[:, 1] - xp_before

        x_rel = x - np.mean(x)
        # Defocusing => positive correlation between x_rel and dxp.
        corr = np.corrcoef(x_rel, dxp)[0, 1]
        assert corr > 0.5, f"x vs dxp correlation should be strongly positive, got {corr:.3f}"

    def test_longitudinal_kick_exists(self, sc_config):
        """The longitudinal kick (dW) should be nonzero for a bunched beam."""
        beam = _make_beam(n=200, current=20.0, sigma_dphi=10.0)
        dw_before = beam.particles[:, 5].copy()

        pic = PicSolver(sc_config)
        pic.kick(beam, ds=50.0)

        ddw = beam.particles[:, 5] - dw_before
        # Should have some nonzero longitudinal kicks
        assert np.std(ddw) > 0, "Longitudinal kicks should be nonzero"


class TestGridSetup:
    def test_fixed_grid_reuses_solver(self, sc_config):
        """In 'fixed' grid_mode, the solver is created once and reused."""
        beam = _make_beam(n=50, current=10.0)
        pic = PicSolver(sc_config)
        pic.kick(beam, ds=1.0)
        solver_first = pic._solver
        pic.kick(beam, ds=1.0)
        solver_second = pic._solver
        assert solver_first is solver_second, "Fixed grid should reuse solver"

    def test_adaptive_grid_reuses_solver_with_updated_greens(self):
        """In adaptive grid_mode the solver instance is reused across kicks
        (avoids per-kick backend re-init), but the Green's function FFT is
        rebuilt in place to follow the changing beam σ.
        """
        config = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0,
                                   grid_mode="adaptive")
        beam = _make_beam(n=50, current=10.0)
        pic = PicSolver(config)
        pic.kick(beam, ds=1.0)
        solver_first = pic._solver
        Gfft_first_id = id(solver_first._G_fft)
        # Perturb the beam so std differs on the second kick → grid extent
        # changes → Green's function must be rebuilt.
        beam.particles[:, 0] *= 2.0
        pic.kick(beam, ds=1.0)
        solver_second = pic._solver
        assert solver_first is solver_second, \
            "Adaptive grid should reuse the solver instance"
        assert id(solver_second._G_fft) != Gfft_first_id, \
            "Green's function FFT should have been rebuilt for the new extent"


class TestDsScaling:
    def test_kick_scales_with_ds(self, sc_config):
        """The kick magnitude should scale linearly with ds."""
        beam1 = _make_beam(n=100, current=10.0)
        beam2 = _make_beam(n=100, current=10.0)
        orig = beam1.particles.copy()

        pic1 = PicSolver(sc_config)
        pic2 = PicSolver(sc_config)
        pic1.kick(beam1, ds=10.0)
        pic2.kick(beam2, ds=20.0)

        dxp1 = beam1.particles[:, 1] - orig[:, 1]
        dxp2 = beam2.particles[:, 1] - orig[:, 1]

        ratio = np.std(dxp2) / np.std(dxp1)
        np.testing.assert_allclose(ratio, 2.0, rtol=0.01)


class TestSimulationWiring:
    def test_simulation_accepts_space_charge_config(self):
        """Simulation.run() wires PicSolver when sc_config is provided."""
        from linac_gen.core.simulation import Simulation
        from linac_gen.core.lattice import Lattice
        from linac_gen.elements.drift import Drift

        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        beam = Beam(ref=ref, n_particles=10, current=5.0)
        rng = np.random.default_rng(99)
        beam.particles[:, 0] = rng.normal(0.0, 1.0, 10)
        beam.particles[:, 2] = rng.normal(0.0, 1.0, 10)
        beam.particles[:, 4] = rng.normal(0.0, 5.0, 10)

        lattice = Lattice()
        lattice.add(Drift(name="d1", length=10.0))

        sc = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0)
        sim = Simulation(lattice, beam, space_charge=sc)
        # Should not raise
        results = sim.run()
        assert results is not None
