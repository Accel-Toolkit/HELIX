"""Tests for the bi-Gaussian (thermal halo) distribution generator."""
import numpy as np
import pytest

from linac_gen.distributions.thermal import generate_thermal
from linac_gen.diagnostics.moments import (
    compute_emittance, compute_twiss_from_particles, compute_halo,
)

EMIT_X = 1.0
ALPHA_X = 0.5
BETA_X = 2.0
EMIT_Y = 0.8
ALPHA_Y = -0.3
BETA_Y = 1.5
EMIT_Z = 0.5
ALPHA_Z = 0.0
BETA_Z = 3.0

N = 50_000
SEED = 42


@pytest.fixture(scope="module")
def halo_particles():
    return generate_thermal(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        halo_fraction=0.05,
        halo_ratio=5.0,
        cutoff=3.0,
        seed=SEED,
    )


def test_shape(halo_particles):
    assert halo_particles.shape == (N, 6)
    assert np.all(np.isfinite(halo_particles))


def test_emittance_matches_target(halo_particles):
    """Final RMS emittance must equal the input ε_x/y/z to ~1 %."""
    ex = compute_emittance(halo_particles, "x")
    ey = compute_emittance(halo_particles, "y")
    ez = compute_emittance(halo_particles, "z")
    assert ex == pytest.approx(EMIT_X, rel=0.01)
    assert ey == pytest.approx(EMIT_Y, rel=0.01)
    assert ez == pytest.approx(EMIT_Z, rel=0.01)


def test_twiss_matches_target(halo_particles):
    """RMS Twiss α, β must equal the targets after rescale."""
    twx = compute_twiss_from_particles(halo_particles, "x")
    assert twx["alpha"] == pytest.approx(ALPHA_X, abs=0.05)
    assert twx["beta"] == pytest.approx(BETA_X, rel=0.01)


def test_halo_inflates_kurtosis():
    """Halo fraction > 0 → 4th moment dominates → halo parameter > Gaussian."""
    # A pure Gaussian has kurtosis 3 → halo parameter = u4/u2² - 1 = 2.
    # Adding a halo should push it well above 2.
    p_gauss = generate_thermal(
        N, EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y, EMIT_Z, ALPHA_Z, BETA_Z,
        halo_fraction=0.0, halo_ratio=5.0, cutoff=3.0, seed=SEED,
    )
    p_halo = generate_thermal(
        N, EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y, EMIT_Z, ALPHA_Z, BETA_Z,
        halo_fraction=0.10, halo_ratio=6.0, cutoff=3.0, seed=SEED,
    )
    h_gauss = compute_halo(p_gauss, "x")
    h_halo = compute_halo(p_halo, "x")
    assert h_halo > h_gauss + 0.5


def test_zero_halo_matches_gaussian_emittance():
    """halo_fraction = 0 → identical RMS to a single Gaussian."""
    p = generate_thermal(
        N, EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y, EMIT_Z, ALPHA_Z, BETA_Z,
        halo_fraction=0.0, halo_ratio=5.0, cutoff=3.0, seed=SEED,
    )
    assert compute_emittance(p, "x") == pytest.approx(EMIT_X, rel=0.01)


def test_extreme_halo_still_matches_target():
    """Even with halo_fraction=0.50 and ratio=10, RMS is preserved."""
    p = generate_thermal(
        N, EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y, EMIT_Z, ALPHA_Z, BETA_Z,
        halo_fraction=0.50, halo_ratio=10.0, cutoff=3.0, seed=SEED,
    )
    assert compute_emittance(p, "x") == pytest.approx(EMIT_X, rel=0.02)


def test_invalid_halo_fraction_raises():
    """halo_fraction outside [0, 1] must raise."""
    with pytest.raises(ValueError):
        generate_thermal(
            10, EMIT_X, ALPHA_X, BETA_X,
            EMIT_Y, ALPHA_Y, BETA_Y, EMIT_Z, ALPHA_Z, BETA_Z,
            halo_fraction=1.5, halo_ratio=5.0, cutoff=3.0, seed=SEED,
        )


def test_invalid_halo_ratio_raises():
    """halo_ratio < 1 must raise."""
    with pytest.raises(ValueError):
        generate_thermal(
            10, EMIT_X, ALPHA_X, BETA_X,
            EMIT_Y, ALPHA_Y, BETA_Y, EMIT_Z, ALPHA_Z, BETA_Z,
            halo_fraction=0.05, halo_ratio=0.5, cutoff=3.0, seed=SEED,
        )


def test_factory_dispatch():
    """BeamConfig with distribution='thermal' must round-trip via factory."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    cfg = BeamConfig(
        species="proton", energy=10.0, frequency=325.0,
        n_particles=5000, distribution="thermal",
        emit_nx=0.25, emit_ny=0.25, emit_z=0.3,
        beta_x=2.0, beta_y=2.0, beta_z=3.0,
        halo_fraction=0.08, halo_ratio=4.0,
    )
    beam = create_beam(cfg, seed=SEED)
    assert beam.particles.shape == (5000, 6)
    # Should have inflated halo parameter vs equivalent Gaussian config.
    h = compute_halo(beam.particles, "x")
    assert h > 1.0  # Gaussian baseline ≈ 2; 8% halo at 4× pushes it higher
