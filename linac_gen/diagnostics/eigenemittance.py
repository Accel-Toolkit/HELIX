"""6-D eigen- (normal-mode) emittances of the beam Σ-matrix.

The three eigenemittances ε_1, ε_2, ε_3 are constants of motion under
*any* linear symplectic transport — including the dispersive, x-y
coupled, and longitudinal-coupled systems where the projected
emittances ε_x, ε_y, ε_z visibly oscillate or grow.  In an uncoupled
lattice they reduce to (ε_x, ε_y, ε_z); under coupling they remain
invariant.

Implementation follows Balandin et al., *"On the calculation of
generalized 4-D and 6-D eigenemittances"*, IPAC 2013, arXiv:1305.1532
(also used by the IMPACT-X reference code in
``EmittanceInvariants.cpp``).  The recipe is

    S₁    = Σ · J                    (6×6, J the symplectic form)
    Sₙ    = (S₁)ⁿ
    I₂    = -tr(S₂) / 2
    I₄    = +tr(S₄) / 2
    I₆    = -tr(S₆) / 2
    cubic = x³ - I₂ · x² + (I₂² - I₄)/2 · x - (I₂³/6 - I₂·I₄/2 + I₆/3) = 0
    (ε_i)² = roots of the cubic, then ε_i = √|root|

This avoids the conjugate-pair extraction step from a direct eigenvalue
solve on (J·Σ), which is numerically fragile when the modes have very
different magnitudes (e.g. ε_x = ε_y ≪ ε_z in a long-bunched proton
linac).  The cubic in (ε²) only depends on three traces — well-behaved
even when det(Σ) underflows.

The 4-D variant (transverse only) for use with the recorder's
``emit_n1`` / ``emit_n2`` is provided as :func:`eigenemittances_4d`
for completeness; it returns the same numbers as the existing
``_normal_mode_emittances`` but using the trace formulation.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Symplectic forms (Σ-matrix block convention: x, x', y, y', φ, w)
_J6 = np.array(
    [
        [0,  1, 0,  0, 0,  0],
        [-1, 0, 0,  0, 0,  0],
        [0,  0, 0,  1, 0,  0],
        [0,  0, -1, 0, 0,  0],
        [0,  0, 0,  0, 0,  1],
        [0,  0, 0,  0, -1, 0],
    ],
    dtype=float,
)

_J4 = _J6[0:4, 0:4].copy()


# ---------------------------------------------------------------------------
def kinetic_invariants(sigma: np.ndarray) -> tuple[float, float, float]:
    """Return the three Balandin invariants (I₂, I₄, I₆) of a 6-D Σ.

    These are the coefficients used to build the cubic whose roots are
    the squared eigenemittances.  Returned as floats.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (6, 6):
        raise ValueError(f"sigma must be 6x6, got {sigma.shape}")
    s1 = sigma @ _J6
    s2 = s1 @ s1
    s4 = s2 @ s2
    s6 = s4 @ s2
    i2 = -0.5 * float(np.trace(s2))
    i4 = +0.5 * float(np.trace(s4))
    i6 = -0.5 * float(np.trace(s6))
    return i2, i4, i6


# ---------------------------------------------------------------------------
def eigenemittances(sigma: np.ndarray) -> tuple[float, float, float]:
    """Return the three normal-mode emittances (ε₁, ε₂, ε₃) of a 6-D Σ.

    Output ordering matches IMPACT-X: roots are sorted ascending, then
    indexed so that for an *uncoupled* Σ the result reduces to
    (ε_x, ε_y, ε_z) — i.e. ε₁ ↔ x, ε₂ ↔ y, ε₃ ↔ z.

    Returns ``(0.0, 0.0, 0.0)`` for a degenerate Σ (e.g. before any
    particles have been recorded).
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (6, 6):
        raise ValueError(f"sigma must be 6x6, got {sigma.shape}")
    if not np.all(np.isfinite(sigma)):
        return 0.0, 0.0, 0.0
    # Singular Σ → fall back to the 2-D projected emittances per block.
    if np.linalg.det(sigma) <= 0.0:
        return _projected_fallback(sigma)
    # Cubic in (ε²):  ε⁶ - I₂·ε⁴ + (I₂² - I₄)/2 · ε² - (I₂³/6 - I₂·I₄/2 + I₆/3) = 0
    #
    # A diverged envelope can leave Σ finite but so large that these
    # 6th-power trace invariants overflow float64 — the cubic is then
    # meaningless.  Catch that and fall back to the projected emittances
    # (2×2 determinants only, always finite), so a blown-up run still
    # finishes with a (huge) number instead of crashing the simulation.
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            i2, i4, i6 = kinetic_invariants(sigma)
            a = 1.0
            b = -i2
            c = 0.5 * (i2 * i2 - i4)
            d = -(i2 ** 3) / 6.0 + 0.5 * i2 * i4 - i6 / 3.0
    except (OverflowError, FloatingPointError):
        return _projected_fallback(sigma)
    if not all(np.isfinite(v) for v in (b, c, d)):
        return _projected_fallback(sigma)
    roots = np.roots([a, b, c, d])
    # Symplectic invariants give *real* roots in the symplectic case;
    # numerical noise can produce tiny imaginary parts — discard them.
    real = np.real(roots)
    real = np.sort(real)
    # IMPACT-X ordering: e1 = √|real[1]|, e2 = √|real[2]|, e3 = √|real[0]|
    # (so the smallest root, typically ≈ ε_z² for tall longitudinal blocks,
    # comes out as the third eigenemittance rather than the first).
    if real.size < 3:
        return 0.0, 0.0, 0.0
    e1 = float(np.sqrt(abs(real[1])))
    e2 = float(np.sqrt(abs(real[2])))
    e3 = float(np.sqrt(abs(real[0])))
    return e1, e2, e3


# ---------------------------------------------------------------------------
def eigenemittances_4d(sigma4: np.ndarray) -> tuple[float, float]:
    """Trace-formulation 4-D eigenemittances (ε₁, ε₂) ordered ε₁ ≥ ε₂.

    Equivalent to the recorder's existing ``_normal_mode_emittances`` but
    obtained from invariants of the 4×4 Σ-block:

        I₂_4D = -tr((Σ·J₄)²) / 2
        I₄_4D = +tr((Σ·J₄)⁴) / 2
        ε₁,₂² = (I₂ ± √(I₂² - 4·I₄·something)) / 2  — i.e. roots of the
                quadratic in ε² built from {I₂_4D, det(Σ_4D)}.

    The quadratic shortcut: for a 4-D symplectic problem the two
    invariants of (J·Σ_4D) are
        ε₁² + ε₂² = -tr((J·Σ_4D)²) / 2
        ε₁²·ε₂²  = det(Σ_4D)
    so the squared emittances are roots of t² - I₂·t + det = 0.
    """
    sigma4 = np.asarray(sigma4, dtype=float)
    if sigma4.shape != (4, 4):
        raise ValueError(f"sigma4 must be 4x4, got {sigma4.shape}")
    det4 = float(np.linalg.det(sigma4))
    if det4 <= 0.0:
        # Degenerate — fall back to projected emittances.
        ex_sq = float(np.linalg.det(sigma4[0:2, 0:2]))
        ey_sq = float(np.linalg.det(sigma4[2:4, 2:4]))
        e1 = float(np.sqrt(max(ex_sq, 0.0)))
        e2 = float(np.sqrt(max(ey_sq, 0.0)))
        if e2 > e1:
            e1, e2 = e2, e1
        return e1, e2
    s1 = sigma4 @ _J4
    s2 = s1 @ s1
    i2 = -0.5 * float(np.trace(s2))
    disc = i2 * i2 - 4.0 * det4
    disc = max(disc, 0.0)
    sqd = float(np.sqrt(disc))
    e1_sq = 0.5 * (i2 + sqd)
    e2_sq = 0.5 * (i2 - sqd)
    e1 = float(np.sqrt(max(e1_sq, 0.0)))
    e2 = float(np.sqrt(max(e2_sq, 0.0)))
    if e2 > e1:
        e1, e2 = e2, e1
    return e1, e2


# ---------------------------------------------------------------------------
def _projected_fallback(sigma: np.ndarray) -> tuple[float, float, float]:
    """ε from the three 2×2 diagonal blocks (used when Σ is singular)."""
    ex = float(np.sqrt(max(np.linalg.det(sigma[0:2, 0:2]), 0.0)))
    ey = float(np.sqrt(max(np.linalg.det(sigma[2:4, 2:4]), 0.0)))
    ez = float(np.sqrt(max(np.linalg.det(sigma[4:6, 4:6]), 0.0)))
    return ex, ey, ez


# ---------------------------------------------------------------------------
def eigenemittances_series(sigma_matrices) -> np.ndarray:
    """Apply :func:`eigenemittances` to a sequence of 6×6 Σ-matrices.

    Returns an array of shape ``(N, 3)`` with columns ``(ε₁, ε₂, ε₃)``.
    Empty / non-6×6 entries map to ``(0, 0, 0)``.
    """
    n = len(sigma_matrices)
    out = np.zeros((n, 3), dtype=float)
    for i, s in enumerate(sigma_matrices):
        s = np.asarray(s, dtype=float)
        if s.shape != (6, 6):
            continue
        out[i, :] = eigenemittances(s)
    return out
