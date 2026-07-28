"""Tests for Uniform distribution generator (wrapper around waterbag)."""
import numpy as np
import pytest
from linac_gen.distributions.uniform import generate_uniform
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
    return generate_uniform(
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
    """Uniform distribution should be centered at zero."""
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


def test_halo_near_waterbag(particles):
    """Uniform distribution halo should match waterbag (both are uniform 6D fills)."""
    h = compute_halo(particles, plane="x")
    expected = 1.4
    assert abs(h - expected) < 0.2, (
        f"x halo {h:.4f} not near 6D waterbag value {expected:.4f}"
    )


def test_identical_to_waterbag_given_same_seed():
    """generate_uniform with same seed should produce identical results to generate_waterbag."""
    p_uniform = generate_uniform(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=SEED,
    )
    p_waterbag = generate_waterbag(
        1000,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=SEED,
    )
    np.testing.assert_array_equal(p_uniform, p_waterbag)


def test_reproducibility(particles):
    """Same seed gives identical results."""
    particles2 = generate_uniform(
        N,
        EMIT_X, ALPHA_X, BETA_X,
        EMIT_Y, ALPHA_Y, BETA_Y,
        EMIT_Z, ALPHA_Z, BETA_Z,
        seed=SEED,
    )
    np.testing.assert_array_equal(particles, particles2)
