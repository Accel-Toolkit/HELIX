"""Tests for BeamConfig factory (create_beam)."""
import math
import numpy as np
import pytest
from linac_gen.core.config import BeamConfig
from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
from linac_gen.distributions.factory import create_beam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config(**kwargs) -> BeamConfig:
    """Return a BeamConfig with small n_particles for fast tests."""
    defaults = dict(
        species="proton",
        energy=3.0,
        frequency=352.21,
        current=0.0,
        n_particles=1000,
        distribution="waterbag",
        emit_nx=0.25,
        alpha_x=0.0,
        beta_x=0.1,
        emit_ny=0.25,
        alpha_y=0.0,
        beta_y=0.1,
        emit_z=0.3,
        alpha_z=0.0,
        beta_z=1.0,
    )
    defaults.update(kwargs)
    return BeamConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic creation
# ---------------------------------------------------------------------------

class TestCreateBeamBasic:
    def test_returns_beam_instance(self):
        beam = create_beam(_default_config())
        assert isinstance(beam, Beam)

    def test_particle_count_matches_config(self):
        cfg = _default_config(n_particles=500)
        beam = create_beam(cfg)
        assert beam.n_particles == 500

    def test_particles_shape(self):
        cfg = _default_config(n_particles=200)
        beam = create_beam(cfg)
        assert beam.particles.shape == (200, 6)

    def test_particles_dtype(self):
        cfg = _default_config(n_particles=200)
        beam = create_beam(cfg)
        assert beam.particles.dtype == np.float64

    def test_all_particles_alive_initially(self):
        cfg = _default_config(n_particles=300)
        beam = create_beam(cfg)
        assert beam.n_alive == 300


# ---------------------------------------------------------------------------
# Deviations (centered near zero)
# ---------------------------------------------------------------------------

class TestParticlesAreDeviations:
    """Check that particles represent deviations centred near zero.

    Tolerance is 5% of the column sigma (relative) to avoid dependence on the
    absolute units, while still catching a distribution that is badly offset.
    With N=5000 the expected statistical fluctuation in the mean is
    sigma/sqrt(N) ~ 1.4% of sigma, so 5% gives a comfortable 3.5-sigma margin.
    """

    def _assert_means_near_zero(self, beam, *, rel_tol=0.05):
        stds = np.std(beam.particles, axis=0)
        means = np.mean(beam.particles, axis=0)
        for col, (m, s) in enumerate(zip(means, stds)):
            if s < 1e-30:
                continue
            assert abs(m) < rel_tol * s, (
                f"Column {col} mean {m:.4f} is more than {rel_tol*100:.0f}% "
                f"of sigma ({s:.4f})"
            )

    # A fixed seed makes these statistical checks deterministic.  Without
    # one, create_beam draws fresh OS entropy each run, so the sample mean
    # occasionally exceeds the 5%-of-sigma tolerance (~0.2% of runs across
    # the 6 phase-space columns) — a spurious full-suite failure.
    def test_mean_near_zero_all_columns(self):
        """Generated particles are deviations: mean should be near zero."""
        cfg = _default_config(n_particles=5000, distribution="waterbag")
        beam = create_beam(cfg, seed=1234)
        self._assert_means_near_zero(beam)

    def test_gaussian_mean_near_zero(self):
        cfg = _default_config(n_particles=5000, distribution="gaussian")
        beam = create_beam(cfg, seed=1234)
        self._assert_means_near_zero(beam)

    def test_kv_mean_near_zero(self):
        cfg = _default_config(n_particles=5000, distribution="kv")
        beam = create_beam(cfg, seed=1234)
        self._assert_means_near_zero(beam)


# ---------------------------------------------------------------------------
# Distribution types
# ---------------------------------------------------------------------------

class TestDistributionTypes:
    @pytest.mark.parametrize("dist", ["gaussian", "waterbag", "kv", "parabolic", "uniform"])
    def test_supported_distribution(self, dist):
        cfg = _default_config(n_particles=200, distribution=dist)
        beam = create_beam(cfg)
        assert beam.n_particles == 200

    def test_unknown_distribution_raises(self):
        cfg = _default_config(distribution="foobar")
        with pytest.raises((ValueError, KeyError)):
            create_beam(cfg)


# ---------------------------------------------------------------------------
# Species lookup
# ---------------------------------------------------------------------------

class TestSpeciesLookup:
    def test_proton_species(self):
        cfg = _default_config(species="proton")
        beam = create_beam(cfg)
        assert beam.species == PROTON

    def test_deuteron_species(self):
        cfg = _default_config(species="deuteron")
        beam = create_beam(cfg)
        assert beam.species == DEUTERON

    def test_h_minus_species(self):
        cfg = _default_config(species="H-")
        beam = create_beam(cfg)
        assert beam.species == H_MINUS

    def test_unknown_species_raises(self):
        cfg = _default_config(species="antiproton")
        with pytest.raises((ValueError, KeyError)):
            create_beam(cfg)


# ---------------------------------------------------------------------------
# Reference particle
# ---------------------------------------------------------------------------

class TestReferenceParticle:
    def test_ref_energy_matches_config(self):
        cfg = _default_config(energy=5.0)
        beam = create_beam(cfg)
        assert abs(beam.ref.w_kin - 5.0) < 1e-9

    def test_ref_frequency_matches_config(self):
        cfg = _default_config(frequency=704.42)
        beam = create_beam(cfg)
        assert abs(beam.ref.frequency - 704.42) < 1e-9

    def test_ref_beta_gamma_positive(self):
        cfg = _default_config(energy=3.0)
        beam = create_beam(cfg)
        assert beam.ref.bg > 0.0


# ---------------------------------------------------------------------------
# Geometric emittance conversion
# ---------------------------------------------------------------------------

class TestGeometricEmittanceConversion:
    """emit_geo = emit_n / (beta * gamma).

    The factory must convert normalised emittance (emit_nx, emit_ny) to
    geometric before calling the generator.  We verify by checking the
    RMS emittance of the resulting beam is consistent with the conversion.
    """

    def _expected_emit_geo(self, emit_n: float, energy: float,
                           species_mass: float) -> float:
        gamma = 1.0 + energy / species_mass
        beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
        return emit_n / (beta * gamma)

    def test_emittance_conversion_x(self):
        from linac_gen.diagnostics.moments import compute_emittance
        cfg = _default_config(
            n_particles=20000,
            distribution="waterbag",
            emit_nx=0.25,
            alpha_x=0.0,
            beta_x=0.1,
        )
        beam = create_beam(cfg)
        emit_measured = compute_emittance(beam.particles, plane="x")
        emit_expected = self._expected_emit_geo(0.25, 3.0, PROTON.mass)
        assert abs(emit_measured - emit_expected) / emit_expected < 0.15, (
            f"x emittance {emit_measured:.6f} vs expected geo {emit_expected:.6f}"
        )

    def test_emittance_conversion_y(self):
        from linac_gen.diagnostics.moments import compute_emittance
        cfg = _default_config(
            n_particles=20000,
            distribution="waterbag",
            emit_ny=0.30,
            alpha_y=0.0,
            beta_y=0.1,
        )
        beam = create_beam(cfg)
        emit_measured = compute_emittance(beam.particles, plane="y")
        emit_expected = self._expected_emit_geo(0.30, 3.0, PROTON.mass)
        assert abs(emit_measured - emit_expected) / emit_expected < 0.15, (
            f"y emittance {emit_measured:.6f} vs expected geo {emit_expected:.6f}"
        )

    def test_bg_value_correct_proton_3mev(self):
        """Sanity-check beta*gamma for 3 MeV proton."""
        cfg = _default_config(energy=3.0, species="proton")
        beam = create_beam(cfg)
        bg_ref = beam.ref.bg
        bg_expected = self._expected_emit_geo(1.0, 3.0, PROTON.mass)
        # bg = beta*gamma; emit_geo = emit_n / bg, so bg = 1/emit_geo for emit_n=1
        gamma = 1.0 + 3.0 / PROTON.mass
        beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
        assert abs(bg_ref - beta * gamma) < 1e-9


# ---------------------------------------------------------------------------
# Current stored on beam
# ---------------------------------------------------------------------------

class TestCurrent:
    def test_current_stored(self):
        cfg = _default_config(current=60.0)
        beam = create_beam(cfg)
        assert abs(beam.current - 60.0) < 1e-9


def test_create_beam_from_file(tmp_path):
    """source='file' should load particles from distribution file."""
    # Create a small distribution file
    fpath = tmp_path / "test_dist.dat"
    fpath.write_text(
        "# w_kin_ref: 3.000000 MeV\n"
        "# phi_ref: 0.000000 deg\n"
        " 0.1  0.2  0.3  0.4  -5.0  3.001\n"
        " 0.5  0.6  0.7  0.8  -2.0  2.999\n"
    )
    config = BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=10.0,
        source="file", distribution_file=str(fpath),
    )
    beam = create_beam(config)
    assert beam.n_particles == 2
    assert beam.current == 10.0
    # Check deviations: dW = W_abs - w_kin_ref
    assert abs(beam.particles[0, 5] - 0.001) < 1e-6


def test_create_beam_file_missing_path():
    """source='file' without distribution_file should raise."""
    config = BeamConfig(source="file", distribution_file=None)
    with pytest.raises(ValueError, match="distribution_file"):
        create_beam(config)
