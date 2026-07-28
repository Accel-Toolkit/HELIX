"""Tests for Gaussian distribution generator."""
import numpy as np
import pytest
from linac_gen.distributions.gaussian import generate_gaussian
from linac_gen.diagnostics.moments import (
    compute_emittance,
    compute_twiss_from_particles,
    compute_halo,
)

# Common Twiss parameters for tests
EMIT_X = 1.0   # mm.mrad
ALPHA_X = 0.5
BETA_X = 2.0

EMIT_Y = 0.8   # mm.mrad
ALPHA_Y = -0.3
BETA_Y = 1.5

EMIT_Z = 0.5   # deg.MeV
ALPHA_Z = 0.0
BETA_Z = 3.0

N = 50_000
SEED = 42


@pytest.fixture(scope="module")
def particles():
    return generate_gaussian(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        cutoff=3.0,
        seed=SEED,
    )


def test_shape(particles):
    assert particles.shape == (N, 6)


def test_dtype(particles):
    assert particles.dtype == np.float64


def test_mean_near_zero(particles):
    """Mean of each coordinate should be near zero (centered distribution)."""
    means = np.mean(particles, axis=0)
    for col, mean in enumerate(means):
        assert abs(mean) < 0.05, (
            f"Column {col} mean {mean:.4f} too far from zero"
        )


def test_rms_emittance_x(particles):
    """RMS emittance in x should match input within 5%."""
    emit = compute_emittance(particles, plane="x")
    assert abs(emit - EMIT_X) / EMIT_X < 0.05, (
        f"x emittance {emit:.4f} vs expected {EMIT_X}"
    )


def test_rms_emittance_y(particles):
    """RMS emittance in y should match input within 5%."""
    emit = compute_emittance(particles, plane="y")
    assert abs(emit - EMIT_Y) / EMIT_Y < 0.05, (
        f"y emittance {emit:.4f} vs expected {EMIT_Y}"
    )


def test_rms_emittance_z(particles):
    """RMS emittance in z should match input within 5%."""
    emit = compute_emittance(particles, plane="z")
    assert abs(emit - EMIT_Z) / EMIT_Z < 0.05, (
        f"z emittance {emit:.4f} vs expected {EMIT_Z}"
    )


def test_twiss_alpha_x(particles):
    """Alpha_x should match input within 10%."""
    twiss = compute_twiss_from_particles(particles, plane="x")
    alpha = twiss["alpha"]
    # Use absolute tolerance for values near zero as well
    tol = max(0.1 * abs(ALPHA_X), 0.1)
    assert abs(alpha - ALPHA_X) < tol, (
        f"x alpha {alpha:.4f} vs expected {ALPHA_X}"
    )


def test_twiss_beta_x(particles):
    """Beta_x should match input within 10%."""
    twiss = compute_twiss_from_particles(particles, plane="x")
    beta = twiss["beta"]
    assert abs(beta - BETA_X) / BETA_X < 0.10, (
        f"x beta {beta:.4f} vs expected {BETA_X}"
    )


def test_twiss_alpha_z(particles):
    """Alpha_z should match input (zero) within absolute tolerance."""
    twiss = compute_twiss_from_particles(particles, plane="z")
    alpha = twiss["alpha"]
    assert abs(alpha - ALPHA_Z) < 0.1, (
        f"z alpha {alpha:.4f} vs expected {ALPHA_Z}"
    )


def test_twiss_beta_z(particles):
    """Beta_z should match input within 10%."""
    twiss = compute_twiss_from_particles(particles, plane="z")
    beta = twiss["beta"]
    assert abs(beta - BETA_Z) / BETA_Z < 0.10, (
        f"z beta {beta:.4f} vs expected {BETA_Z}"
    )


def test_no_particles_beyond_cutoff(particles):
    """No particle coordinate should exceed cutoff * theoretical_sigma by
    more than the per-plane Cholesky-rescaling slop (~1/√N relative).

    The cutoff is applied in terms of the *theoretical* (untruncated) sigma,
    which equals sqrt(beta * emit) for position and sqrt(gamma_t * emit) for
    angle.  The post-rejection sample is then linearly rescaled per plane so
    its empirical covariance exactly matches the requested Σ — this can
    nudge a particle that was at exactly cutoff·σ to ~cutoff·σ × (1+ε)
    where ε ~ 1/√N.  At N=10 000 that's ~1 % relative slop; we allow 3 %.
    """
    cutoff = 3.0
    # Theoretical sigma per coordinate (from Twiss, pre-truncation)
    gamma_x = (1.0 + ALPHA_X ** 2) / BETA_X
    gamma_y = (1.0 + ALPHA_Y ** 2) / BETA_Y
    gamma_z = (1.0 + ALPHA_Z ** 2) / BETA_Z

    theoretical_sigmas = [
        np.sqrt(BETA_X * EMIT_X),    # col 0: x
        np.sqrt(gamma_x * EMIT_X),   # col 1: xp
        np.sqrt(BETA_Y * EMIT_Y),    # col 2: y
        np.sqrt(gamma_y * EMIT_Y),   # col 3: yp
        np.sqrt(BETA_Z * EMIT_Z),    # col 4: dphi
        np.sqrt(gamma_z * EMIT_Z),   # col 5: dW
    ]

    for col in range(6):
        col_data = particles[:, col]
        sigma_th = theoretical_sigmas[col]
        max_val = np.max(np.abs(col_data))
        assert max_val <= cutoff * sigma_th * 1.03, (
            f"Column {col}: max|val|={max_val:.4f} exceeds "
            f"{cutoff}*sigma_th×1.03={cutoff * sigma_th * 1.03:.4f}"
        )


def test_halo_x_near_gaussian(particles):
    """Halo parameter H in x should be near 2.0 (Gaussian kurtosis-1 value),
    within 0.5 tolerance (truncation reduces it slightly)."""
    h = compute_halo(particles, plane="x")
    assert abs(h - 2.0) < 0.5, (
        f"x halo parameter {h:.4f} not near Gaussian value 2.0"
    )


def test_halo_y_near_gaussian(particles):
    """Halo parameter H in y should be near 2.0."""
    h = compute_halo(particles, plane="y")
    assert abs(h - 2.0) < 0.5, (
        f"y halo parameter {h:.4f} not near Gaussian value 2.0"
    )


def test_reproducibility(particles):
    """Same seed should produce identical results."""
    particles2 = generate_gaussian(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        cutoff=3.0,
        seed=SEED,
    )
    np.testing.assert_array_equal(particles, particles2)


def test_different_seeds_differ():
    """Different seeds should produce different results."""
    p1 = generate_gaussian(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=1,
    )
    p2 = generate_gaussian(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=2,
    )
    assert not np.allclose(p1, p2)


def test_tighter_cutoff_reduces_halo():
    """Tighter cutoff should reduce halo below 2.0 more than 3-sigma cutoff."""
    p_tight = generate_gaussian(
        20_000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        cutoff=2.0,
        seed=SEED,
    )
    h_tight = compute_halo(p_tight, plane="x")
    assert h_tight < 2.0, f"Tight cutoff halo {h_tight:.4f} should be below 2.0"
