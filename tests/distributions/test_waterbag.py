"""Tests for Waterbag distribution generator."""
import numpy as np
import pytest
from linac_gen.distributions.waterbag import generate_waterbag
from linac_gen.diagnostics.moments import (
    compute_emittance,
    compute_twiss_from_particles,
    compute_halo,
)

# Common Twiss parameters
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
def particles():
    return generate_waterbag(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=SEED,
    )


def test_shape(particles):
    assert particles.shape == (N, 6)


def test_dtype(particles):
    assert particles.dtype == np.float64


def test_mean_near_zero(particles):
    """Waterbag distribution should be centered at zero."""
    means = np.mean(particles, axis=0)
    for col, mean in enumerate(means):
        assert abs(mean) < 0.05, (
            f"Column {col} mean {mean:.4f} too far from zero"
        )


def test_rms_emittance_x(particles):
    """RMS emittance in x should match input within 10%."""
    emit = compute_emittance(particles, plane="x")
    assert abs(emit - EMIT_X) / EMIT_X < 0.10, (
        f"x emittance {emit:.4f} vs expected {EMIT_X}"
    )


def test_rms_emittance_y(particles):
    """RMS emittance in y should match input within 10%."""
    emit = compute_emittance(particles, plane="y")
    assert abs(emit - EMIT_Y) / EMIT_Y < 0.10, (
        f"y emittance {emit:.4f} vs expected {EMIT_Y}"
    )


def test_rms_emittance_z(particles):
    """RMS emittance in z should match input within 10%."""
    emit = compute_emittance(particles, plane="z")
    assert abs(emit - EMIT_Z) / EMIT_Z < 0.10, (
        f"z emittance {emit:.4f} vs expected {EMIT_Z}"
    )


def test_twiss_alpha_x(particles):
    """Alpha_x from waterbag should match input within 10%."""
    twiss = compute_twiss_from_particles(particles, plane="x")
    alpha = twiss["alpha"]
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


def test_halo_x_near_waterbag(particles):
    """Halo parameter H in x should be near 1.4 (6D waterbag value),
    within 0.15 tolerance.

    For a uniform 6D hyperellipsoid (waterbag), the marginal distribution of
    any single coordinate u has:
        E[u^4] / E[u^2]^2 - 1 = (2*d+2)/(d+4) - 1 = 1.4  (d=6)
    """
    h = compute_halo(particles, plane="x")
    expected = 1.4
    assert abs(h - expected) < 0.15, (
        f"x halo {h:.4f} not near 6D waterbag value {expected:.4f}"
    )


def test_halo_y_near_waterbag(particles):
    """Halo parameter H in y should be near 1.4 (6D waterbag value)."""
    h = compute_halo(particles, plane="y")
    expected = 1.4
    assert abs(h - expected) < 0.15, (
        f"y halo {h:.4f} not near 6D waterbag value {expected:.4f}"
    )


def test_halo_z_near_waterbag(particles):
    """Halo parameter H in z should be near 1.4 (6D waterbag value)."""
    h = compute_halo(particles, plane="z")
    expected = 1.4
    assert abs(h - expected) < 0.15, (
        f"z halo {h:.4f} not near 6D waterbag value {expected:.4f}"
    )


def test_finite_extent(particles):
    """Waterbag particles should have finite extent (no extremely large values)."""
    # For a proper waterbag, sigma in each coord = sqrt(emit * beta / (n_dim+2)) * some factor
    # The key check: no particles should exceed ~5 sigma
    for col in range(6):
        col_data = particles[:, col]
        sigma = np.std(col_data)
        max_val = np.max(np.abs(col_data))
        # Waterbag has a hard boundary; just ensure nothing is unreasonably large
        assert max_val < 5.0 * sigma, (
            f"Column {col}: max|val|={max_val:.4f} suspiciously large vs sigma={sigma:.4f}"
        )


def test_reproducibility(particles):
    """Same seed gives identical results."""
    particles2 = generate_waterbag(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=SEED,
    )
    np.testing.assert_array_equal(particles, particles2)


def test_different_seeds_differ():
    """Different seeds produce different results."""
    p1 = generate_waterbag(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=1,
    )
    p2 = generate_waterbag(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=2,
    )
    assert not np.allclose(p1, p2)
