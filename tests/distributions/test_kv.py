"""Tests for KV distribution generator."""
import numpy as np
import pytest
from linac_gen.distributions.kv import generate_kv
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
    return generate_kv(
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
    """KV distribution should be centered at zero."""
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
    """Alpha_x should match input within tolerance."""
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


def test_halo_x_near_kv(particles):
    """Halo parameter H in x should be near 1.0 (KV 4D-surface distribution),
    within 0.2 tolerance.

    For the KV distribution (uniform on the surface of a 4D hyperellipsoid),
    the marginal distribution of any single transverse coordinate follows a
    semicircle distribution with:
        E[u^4] / E[u^2]^2 - 1 = (1/8) / (1/4)^2 - 1 = 2 - 1 = 1.0
    """
    h = compute_halo(particles, plane="x")
    expected = 1.0
    assert abs(h - expected) < 0.2, (
        f"x halo {h:.4f} not near KV value {expected:.4f}"
    )


def test_halo_y_near_kv(particles):
    """Halo parameter H in y should be near 1.0 (KV 4D-surface value)."""
    h = compute_halo(particles, plane="y")
    expected = 1.0
    assert abs(h - expected) < 0.2, (
        f"y halo {h:.4f} not near KV value {expected:.4f}"
    )


def test_transverse_4d_constraint(particles):
    """For KV, all transverse particles lie on a 4D hyperellipsoid surface.
    The normalized radius should be constant (all equal to 1) for transverse coords."""
    # Normalize each coordinate by its sigma
    x = particles[:, 0]
    xp = particles[:, 1]
    y = particles[:, 2]
    yp = particles[:, 3]

    # Compute sigma for each coordinate
    sx = np.std(x)
    sxp = np.std(xp)
    sy = np.std(y)
    syp = np.std(yp)

    # Normalized coordinates
    xn = x / sx
    xpn = xp / sxp
    yn = y / sy
    ypn = yp / syp

    # The sum of squares should be approximately constant for KV
    r2 = xn**2 + xpn**2 + yn**2 + ypn**2
    # For KV on 4D surface, variance should be very small
    # (all on same ellipsoid after normalization by sigma)
    r2_std = np.std(r2)
    r2_mean = np.mean(r2)
    # Coefficient of variation should be small (< 50%)
    assert r2_std / r2_mean < 0.5, (
        f"r2 coefficient of variation {r2_std/r2_mean:.4f} too large for KV"
    )


def test_reproducibility(particles):
    """Same seed gives identical results."""
    particles2 = generate_kv(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=SEED,
    )
    np.testing.assert_array_equal(particles, particles2)


def test_different_seeds_differ():
    """Different seeds produce different results."""
    p1 = generate_kv(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=1,
    )
    p2 = generate_kv(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=2,
    )
    assert not np.allclose(p1, p2)
