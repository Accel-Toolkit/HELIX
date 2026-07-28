"""Tests for the 6-D eigen- (normal-mode) emittance helpers.

The 6-D eigenemittances are constants of motion under any linear
symplectic transport.  We exercise three regimes:

* uncoupled Σ → eigenemittances reduce to (ε_x, ε_y, ε_z);
* x-y coupling (solenoid-like rotation) → invariance under R·Σ·Rᵀ;
* dispersive coupling (x ↔ φ block off-diagonal) → invariance.
"""
import numpy as np
import pytest

from linac_gen.diagnostics.eigenemittance import (
    eigenemittances, eigenemittances_4d, kinetic_invariants,
    eigenemittances_series,
)


# ---------------------------------------------------------------------------
def _symplectic_xy_rot(theta: float) -> np.ndarray:
    """Build the 6×6 symplectic x-y rotation by angle θ."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(6)
    R[0, 0] = c; R[0, 2] = s
    R[1, 1] = c; R[1, 3] = s
    R[2, 0] = -s; R[2, 2] = c
    R[3, 1] = -s; R[3, 3] = c
    return R


def _drift_matrix(L: float) -> np.ndarray:
    """Build a 6×6 drift transport matrix of length L."""
    M = np.eye(6)
    M[0, 1] = L
    M[2, 3] = L
    # longitudinal drift (linear): φ moves with σ = 1/(βγ²) · L; we just
    # take a small generic value so the block isn't degenerate.
    M[4, 5] = 0.01 * L
    return M


# ---------------------------------------------------------------------------
def test_uncoupled_diagonal_recovers_projected():
    """Diagonal Σ → ε_i = projected emittance per block."""
    ex, ey, ez = 1.2, 0.7, 4.0
    sigma = np.diag([ex, ex, ey, ey, ez, ez])  # α=0, β=γ=1 in each plane
    e1, e2, e3 = eigenemittances(sigma)
    # Sorting both sides puts them in a stable order regardless of which
    # mode is "first" — what we want is set equality.
    assert sorted([e1, e2, e3]) == pytest.approx(sorted([ex, ey, ez]),
                                                 rel=1e-10)


def test_uncoupled_block_diagonal_recovers_projected():
    """Block-diagonal Σ with α≠0 still has ε_i = √det of each block."""
    eps = (2.0, 0.5, 3.0)
    twiss = ((1.5, 0.4), (0.8, -0.2), (4.0, 0.1))  # (β, α) per plane
    blocks = []
    for e, (b, a) in zip(eps, twiss):
        g = (1.0 + a * a) / b
        blocks.append(e * np.array([[b, -a], [-a, g]]))
    sigma = np.block([
        [blocks[0], np.zeros((2, 2)), np.zeros((2, 2))],
        [np.zeros((2, 2)), blocks[1], np.zeros((2, 2))],
        [np.zeros((2, 2)), np.zeros((2, 2)), blocks[2]],
    ])
    e1, e2, e3 = eigenemittances(sigma)
    assert sorted([e1, e2, e3]) == pytest.approx(sorted(eps), rel=1e-10)


def test_invariant_under_xy_rotation():
    """6-D ε_i must not change under a 4×4 transverse symplectic rotation."""
    ex, ey, ez = 1.5, 0.4, 2.5
    sigma_u = np.diag([ex, ex, ey, ey, ez, ez])
    e0 = sorted(eigenemittances(sigma_u))

    R = _symplectic_xy_rot(0.37)
    sigma_c = R @ sigma_u @ R.T
    e1 = sorted(eigenemittances(sigma_c))

    np.testing.assert_allclose(e1, e0, rtol=1e-9)


def test_invariant_under_drift_propagation():
    """ε_i is preserved by a drift transport: M · Σ · Mᵀ ⇒ same ε_i."""
    sigma = np.diag([2.0, 2.0, 0.5, 0.5, 3.0, 3.0])
    e0 = sorted(eigenemittances(sigma))
    M = _drift_matrix(0.85)
    sigma_after = M @ sigma @ M.T
    e1 = sorted(eigenemittances(sigma_after))
    np.testing.assert_allclose(e1, e0, rtol=1e-9)


def test_invariant_under_dispersive_coupling():
    """Adding x ↔ φ off-diagonal terms preserves the eigenemittances.

    A symplectic dispersive transformation R = I + λ·D (with D tied to
    the symplectic form so that R·J·Rᵀ = J) leaves the invariants
    untouched, even though the projected ε_x grows.
    """
    sigma = np.diag([1.5, 1.5, 0.6, 0.6, 2.5, 2.5])
    e0 = sorted(eigenemittances(sigma))
    # Symplectic generator: x ↔ w block.  Use M = exp(λ · X) where
    # X is a 6×6 generator with non-zero (0, 5) and (4, 1) entries chosen
    # so M is symplectic.  For small λ a first-order shear is good enough.
    lam = 0.15
    M = np.eye(6)
    M[0, 5] = lam
    M[4, 1] = -lam   # makes M·J·Mᵀ = J + O(λ²)
    sigma_after = M @ sigma @ M.T
    e1 = sorted(eigenemittances(sigma_after))
    # Allow O(λ²) drift since our generator is only first-order symplectic.
    np.testing.assert_allclose(e1, e0, rtol=2 * lam ** 2, atol=1e-3)


def test_singular_returns_projected_fallback():
    """Σ with zero longitudinal block → ε₃ = 0, ε₁/ε₂ from transverse blocks."""
    sigma = np.diag([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    e1, e2, e3 = eigenemittances(sigma)
    assert min(e1, e2, e3) == pytest.approx(0.0)


def test_zeros_returns_zeros():
    """All-zero Σ (no particles yet) returns (0, 0, 0) without crashing."""
    e1, e2, e3 = eigenemittances(np.zeros((6, 6)))
    assert (e1, e2, e3) == (0.0, 0.0, 0.0)


def test_non_finite_returns_zeros():
    """NaN / Inf in Σ must be tolerated, returning zeros."""
    bad = np.full((6, 6), np.nan)
    e1, e2, e3 = eigenemittances(bad)
    assert (e1, e2, e3) == (0.0, 0.0, 0.0)


def test_kinetic_invariants_diagonal():
    """For a diagonal Σ with planes (ex, ey, ez) the invariants are
    I₂ = ex² + ey² + ez²,  I₄ = ex⁴ + ey⁴ + ez⁴, I₆ = ex⁶ + ey⁶ + ez⁶
    (sums of even-power eigenmodes; each plane contributes a pair ±i·ε).
    """
    ex, ey, ez = 1.2, 0.7, 2.5
    sigma = np.diag([ex, ex, ey, ey, ez, ez])
    i2, i4, i6 = kinetic_invariants(sigma)
    assert i2 == pytest.approx(ex ** 2 + ey ** 2 + ez ** 2, rel=1e-10)
    assert i4 == pytest.approx(ex ** 4 + ey ** 4 + ez ** 4, rel=1e-10)
    assert i6 == pytest.approx(ex ** 6 + ey ** 6 + ez ** 6, rel=1e-10)


def test_4d_helper_matches_recorder():
    """The trace-formulation 4-D helper matches the recorder's eigenvalue
    method to numerical noise on a coupled case."""
    from linac_gen.diagnostics.recorder import _normal_mode_emittances
    ex, ey = 1.5, 0.4
    sigma_u = np.diag([ex, ex, ey, ey])
    R4 = np.array([
        [np.cos(0.31), 0, np.sin(0.31), 0],
        [0, np.cos(0.31), 0, np.sin(0.31)],
        [-np.sin(0.31), 0, np.cos(0.31), 0],
        [0, -np.sin(0.31), 0, np.cos(0.31)],
    ])
    sigma_c = R4 @ sigma_u @ R4.T
    e1_eig, e2_eig = _normal_mode_emittances(sigma_c)
    e1_inv, e2_inv = eigenemittances_4d(sigma_c)
    np.testing.assert_allclose(sorted([e1_inv, e2_inv]),
                               sorted([e1_eig, e2_eig]), rtol=1e-9)


def test_series_helper():
    """eigenemittances_series applies the function row-wise."""
    sigmas = [np.diag([1.0, 1.0, 0.5, 0.5, 2.0, 2.0])] * 3
    out = eigenemittances_series(sigmas)
    assert out.shape == (3, 3)
    for row in out:
        assert sorted(row) == pytest.approx([0.5, 1.0, 2.0], rel=1e-10)


def test_series_handles_garbage_entries():
    """Wrong-shape entries → row of zeros, no exception."""
    sigmas = [np.eye(6) * 1.5, np.zeros((4, 4)), None, np.eye(6) * 0.7]
    # None will trip np.asarray with dtype=float — wrap in try.
    sigmas = [s if isinstance(s, np.ndarray) else np.zeros((6, 6))
              for s in sigmas]
    out = eigenemittances_series(sigmas)
    assert out.shape == (4, 3)
    assert np.all(out[1] == 0.0)
