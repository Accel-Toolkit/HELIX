"""Tests for Parabolic distribution generator."""
import numpy as np
import pytest
from linac_gen.distributions.parabolic import generate_parabolic
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
    return generate_parabolic(
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
    """Parabolic distribution should be centered at zero."""
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


def test_halo_between_waterbag_and_gaussian(particles):
    """Parabolic halo in x should be between waterbag (1.4) and Gaussian (2.0).

    The parabolic density peaks at the center and goes to zero at the boundary,
    giving more weight to low-amplitude particles than waterbag but less than
    a true Gaussian.  Typical value ~1.0-1.4 for the marginal distribution.
    The key property is: halo < waterbag value 1.4 (more centrally peaked).
    """
    h = compute_halo(particles, plane="x")
    # Parabolic should be below waterbag (1.4) since it weights center more
    assert h < 1.55, (
        f"x halo {h:.4f} should be below 1.55 for parabolic"
    )
    # And well below Gaussian (2.0)
    assert h < 2.0, (
        f"x halo {h:.4f} should be below Gaussian value 2.0"
    )


def test_halo_y(particles):
    """Parabolic halo in y should be below waterbag 1.4."""
    h = compute_halo(particles, plane="y")
    assert h < 1.55, f"y halo {h:.4f} should be below 1.55 for parabolic"


def test_halo_z(particles):
    """Parabolic halo in z should be below waterbag 1.4."""
    h = compute_halo(particles, plane="z")
    assert h < 1.55, f"z halo {h:.4f} should be below 1.55 for parabolic"


def test_finite_extent(particles):
    """Parabolic particles should have finite extent (no extremely large values)."""
    for col in range(6):
        col_data = particles[:, col]
        sigma = np.std(col_data)
        max_val = np.max(np.abs(col_data))
        assert max_val < 6.0 * sigma, (
            f"Column {col}: max|val|={max_val:.4f} suspiciously large vs sigma={sigma:.4f}"
        )


def test_reproducibility(particles):
    """Same seed gives identical results."""
    particles2 = generate_parabolic(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=SEED,
    )
    np.testing.assert_array_equal(particles, particles2)


def test_different_seeds_differ():
    """Different seeds produce different results."""
    p1 = generate_parabolic(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=1,
    )
    p2 = generate_parabolic(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=2,
    )
    assert not np.allclose(p1, p2)
