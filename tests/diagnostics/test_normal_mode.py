"""Tests for the normal-mode (eigen-) emittance helper."""
import numpy as np
import pytest

from linac_gen.diagnostics.recorder import _normal_mode_emittances


def test_uncoupled_matches_2d_emittances():
    """For an uncoupled Σ block, ε_1, ε_2 reduce to ε_x, ε_y.

    Σ_x = ε_x · [[β, −α], [−α, γ]] with α=0, β=1, γ=1 gives det=ε_x² so
    the 2-D emittance ε_x = √det(Σ_x) is recovered.  Same for y.
    """
    eps_x = 1.5
    eps_y = 0.7
    sigma = np.diag([eps_x, eps_x, eps_y, eps_y])  # β=γ=1, α=0
    e1, e2 = _normal_mode_emittances(sigma)
    np.testing.assert_allclose(sorted([e1, e2]), sorted([eps_x, eps_y]),
                               rtol=1e-10)


def test_zero_determinant_returns_zero():
    """Singular 4×4 (e.g. all zeros) should return (0, 0) without crashing."""
    e1, e2 = _normal_mode_emittances(np.zeros((4, 4)))
    assert e1 == 0.0
    assert e2 == 0.0


def test_invariant_under_symplectic_rotation():
    """A solenoid-like x–y rotation must leave the normal modes unchanged.

    Constructs an uncoupled Σ, then applies the 4×4 symplectic rotation
    R(θ) corresponding to an x–y rotation by θ.  ε_1, ε_2 must match.
    """
    eps_x = 2.0
    eps_y = 0.3
    # α=0, β=γ=1 Twiss → diag(ε, ε, ε, ε) form
    sigma_uncoupled = np.diag([eps_x, eps_x, eps_y, eps_y])
    e1_u, e2_u = _normal_mode_emittances(sigma_uncoupled)

    theta = 0.42
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, 0, s, 0],
        [0, c, 0, s],
        [-s, 0, c, 0],
        [0, -s, 0, c],
    ])
    sigma_coupled = R @ sigma_uncoupled @ R.T
    e1_c, e2_c = _normal_mode_emittances(sigma_coupled)
    np.testing.assert_allclose(e1_c, e1_u, rtol=1e-9)
    np.testing.assert_allclose(e2_c, e2_u, rtol=1e-9)
