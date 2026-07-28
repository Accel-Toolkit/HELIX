# tests/diagnostics/test_moments.py
import numpy as np
import pytest
from linac_gen.diagnostics.moments import (
    compute_moments, compute_emittance, compute_twiss_from_particles, compute_halo,
)


def _make_gaussian_beam(n=10000, sigma_x=1.0, sigma_xp=0.5):
    """Helper: uncorrelated Gaussian in x-x', zero in other planes."""
    rng = np.random.default_rng(42)
    particles = np.zeros((n, 6))
    particles[:, 0] = rng.normal(0, sigma_x, n)
    particles[:, 1] = rng.normal(0, sigma_xp, n)
    return particles


def test_compute_moments_shape():
    p = _make_gaussian_beam()
    m = compute_moments(p)
    assert m["sigma_matrix"].shape == (6, 6)
    assert m["mean"].shape == (6,)


def test_compute_moments_sigma_x():
    p = _make_gaussian_beam(n=100000, sigma_x=2.0)
    m = compute_moments(p)
    assert abs(m["sigma_x"] - 2.0) < 0.05  # 2.5% tolerance for 100k particles


def test_compute_emittance_uncorrelated():
    """For uncorrelated x-x', emit = sigma_x * sigma_xp."""
    p = _make_gaussian_beam(n=100000, sigma_x=1.0, sigma_xp=0.5)
    emit = compute_emittance(p, plane="x")
    assert abs(emit - 0.5) < 0.02  # sigma_x * sigma_xp = 1.0 * 0.5


def test_compute_emittance_zero_for_empty_plane():
    """Planes with all zeros should give zero emittance."""
    p = _make_gaussian_beam()
    emit_y = compute_emittance(p, plane="y")
    assert emit_y < 1e-20


def test_compute_twiss_from_particles():
    p = _make_gaussian_beam(n=100000, sigma_x=2.0, sigma_xp=0.5)
    tw = compute_twiss_from_particles(p, plane="x")
    assert tw["emittance"] > 0
    # For uncorrelated beam: alpha ~ 0, beta = sigma_x^2 / emit
    assert abs(tw["alpha"]) < 0.1
    expected_beta = 4.0 / tw["emittance"]  # sigma_x^2 = 4.0
    assert abs(tw["beta"] - expected_beta) < 0.5
    # Courant-Snyder identity: beta * gamma_t - alpha^2 = 1
    cs = tw["beta"] * tw["gamma_t"] - tw["alpha"]**2
    assert abs(cs - 1.0) < 0.01


def test_compute_halo_gaussian():
    """Gaussian distribution: halo parameter H ~ 2."""
    p = _make_gaussian_beam(n=100000)
    H = compute_halo(p, plane="x")
    assert abs(H - 2.0) < 0.3  # should be ~2 for Gaussian


def test_compute_moments_empty_array():
    """Empty particle array should return zeros, not crash."""
    p = np.zeros((0, 6))
    m = compute_moments(p)
    assert m["sigma_x"] == 0.0
    assert m["sigma_matrix"].shape == (6, 6)


def test_compute_emittance_empty_array():
    p = np.zeros((0, 6))
    assert compute_emittance(p, "x") == 0.0


def test_compute_halo_empty_array():
    p = np.zeros((0, 6))
    assert compute_halo(p, "x") == 0.0
