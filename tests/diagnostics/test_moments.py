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


# ----------------------------------------------------------- ellipse_points
@pytest.mark.parametrize("alpha,beta,eps", [(0.0, 2.0, 1.0), (1.3, 0.4, 2.5), (-2.0, 10.0, 0.1)])
def test_ellipse_points_courant_snyder(alpha, beta, eps):
    """Every point of the drawn ellipse satisfies γu² + 2αuu′ + βu′² = ε
    about the centre; the curve is closed, has area πε and the extents
    √(εβ), √(εγ); the sign convention matches compute_twiss_from_particles."""
    from linac_gen.diagnostics.moments import ellipse_points
    cx, cy = 1.5, -0.7
    u, up = ellipse_points(alpha, beta, eps, centre=(cx, cy), n=200)
    assert u.shape == up.shape == (201,)
    gamma = (1 + alpha * alpha) / beta
    du, dup = u - cx, up - cy
    inv = gamma * du * du + 2 * alpha * du * dup + beta * dup * dup
    assert np.allclose(inv, eps, rtol=1e-12, atol=1e-12 * eps)
    assert u[0] == pytest.approx(u[-1]) and up[0] == pytest.approx(up[-1])
    area = 0.5 * abs(np.dot(u[:-1], np.roll(up[:-1], -1)) - np.dot(up[:-1], np.roll(u[:-1], -1)))
    assert area == pytest.approx(np.pi * eps, rel=1e-3)
    assert du.max() == pytest.approx(np.sqrt(eps * beta), rel=1e-6)
    assert dup.max() == pytest.approx(np.sqrt(eps * gamma), rel=5e-4)   # extremum between samples
    # alpha > 0 (converging): at the largest u the slope u′ is negative
    k = int(np.argmax(du))
    if alpha > 0:
        assert dup[k] < 0
    elif alpha < 0:
        assert dup[k] > 0


def test_ellipse_points_matches_particle_twiss_sign():
    """The ellipse of (α, β, ε) estimated from particles encloses the
    1-σ shape of those particles: same sign convention end to end."""
    from linac_gen.diagnostics.moments import compute_twiss_from_particles, ellipse_points
    p = _make_gaussian_beam(n=20000)
    p[:, 1] = p[:, 1] - 0.8 * p[:, 0]            # correlated: alpha > 0
    tw = compute_twiss_from_particles(p, "x")
    assert tw["alpha"] > 0
    u, up = ellipse_points(tw["alpha"], tw["beta"], tw["emittance"], n=400)
    # the principal axis of the particle cloud and of the ellipse agree
    w_p, v_p = np.linalg.eigh(np.cov(p[:, 0], p[:, 1]))
    w_e, v_e = np.linalg.eigh(np.cov(u, up))
    assert abs(np.dot(v_p[:, 1], v_e[:, 1])) == pytest.approx(1.0, abs=1e-2)


def test_ellipse_points_degenerate_inputs():
    from linac_gen.diagnostics.moments import ellipse_points
    for a, b, e in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, -1.0, 1.0), (0.0, 1.0, -1.0), (0.0, np.nan, 1.0)):
        u, up = ellipse_points(a, b, e)
        assert u.size == 0 and up.size == 0
