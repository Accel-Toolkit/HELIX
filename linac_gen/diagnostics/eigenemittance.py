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


# ---------------------------------------------------------------------------
# Canonical (normalised) intrinsic emittances
# ---------------------------------------------------------------------------
# The recorded Σ is in mixed units (mm, mrad, mm, mrad, deg, MeV) with
# geometric angles.  Its three conjugate pairs then carry DIFFERENT
# phase-space area scales, so the eigenvalues of Σ·J (what
# :func:`eigenemittances` returns) are invariants only while the
# longitudinal plane is decoupled from x and y.  The functions below first
# map Σ to normalised canonical coordinates, all in mm and mrad,
#
#     (x, βγ·x', y, βγ·y', c·Δt, −1000·ΔE/mc²)
#     Δt  = Δφ / (360·f)          →  c·Δt [mm]   = Δφ · λ/360
#     ΔE  = ΔW                    →  [mrad]       = −ΔW · 1000/mc²
#
# where every pair has the same symplectic weight, then take the
# symplectic eigenvalues.  The minus sign is not cosmetic: Δφ > 0 is a
# LATE particle (Δφ ∝ +Δt) and the conjugate of t is −E, so (Δφ, ΔW) is
# anti-canonical.  Uncoupled it does not matter (|ε| per plane); with
# x–ΔE or x–Δφ coupling the unsigned form gives wrong modes — pinned by
# the dipole-matrix test (R51/R52 vs R16/R26).  Uncoupled, the three modes reduce exactly to
# the recorder's ε_nx, ε_ny and ε_nz (ε_nz = ε_z,mm·mrad · βγ — the longitudinal
# scale λ·1000/(360·mc²) is independent of β and γ).  They are invariant
# under any linear symplectic transport, including dispersive x–ΔE and
# RF x–Δφ coupling, acceleration and FREQ jumps.  The momenta are the
# mechanical ones (x' = p_x/p_z scaled by the reference βγ): inside a
# solenoid field the true canonical P_x = p_x + qA_x differs, so the modes
# are invariants outside solenoid fields only (√det of the 4-D block is
# not affected — that change of variables has unit determinant).

_PLANES = ("x", "y", "z")


def canonical_scale(bg: float, freq_mhz: float, mass_mev: float) -> np.ndarray:
    """Diagonal D (length 6) taking the recorded Σ to normalised canonical
    units: Σ_c = D·Σ·D.  Returns NaNs when any input is non-physical."""
    if not (bg > 0.0 and freq_mhz > 0.0 and mass_mev > 0.0):
        return np.full(6, np.nan)
    from linac_gen.core.constants import C_LIGHT
    lam_mm = C_LIGHT / (freq_mhz * 1e6) * 1e3
    return np.array([1.0, bg, 1.0, bg, lam_mm / 360.0, -1e3 / mass_mev])


def canonical_eigenemittances(sigma: np.ndarray, bg: float, freq_mhz: float,
                              mass_mev: float) -> dict:
    """Normalised canonical intrinsic emittances of one recorded 6×6 Σ.

    Returns a dict (all emittances in mm·mrad, normalised):

    ``modes``     (ε₁, ε₂, ε₃) labelled by dominant plane — ``modes[0]`` is
                  the x-like mode, ``[1]`` y-like, ``[2]`` z-like (assigned
                  by the largest total share of each mode's symplectic
                  area per plane; near-equal modes may swap labels)
    ``share``     3×3 array, ``share[i, k]`` = fraction of mode i in plane k
    ``proj``      projected (ε_nx, ε_ny, ε_nz) of the same Σ
    ``e4d``       √det Σ_c,4×4 = (βγ)²·ε_4D            [(mm·mrad)²]
    ``e6d``       √det Σ_c = ε₁ε₂ε₃                    [(mm·mrad)³]
    ``coupling6`` ε_nx·ε_ny·ε_nz / (ε₁ε₂ε₃) ≥ 1 (1 = no coupling)
    ``coupling4`` ε_nx·ε_ny / ε_4D,n ≥ 1 (transverse only)

    Every value is NaN when Σ is not positive definite or a scale input is
    non-physical (e.g. a record after the whole beam is lost).
    """
    nan3 = np.full(3, np.nan)
    out = {"modes": nan3.copy(), "share": np.full((3, 3), np.nan),
           "proj": nan3.copy(), "e4d": np.nan, "e6d": np.nan,
           "coupling6": np.nan, "coupling4": np.nan}
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (6, 6) or not np.all(np.isfinite(sigma)):
        return out
    d = canonical_scale(bg, freq_mhz, mass_mev)
    if not np.all(np.isfinite(d)):
        return out
    sc = sigma * np.outer(d, d)
    proj = np.array([np.linalg.det(sc[k:k + 2, k:k + 2]) for k in (0, 2, 4)])
    det4 = float(np.linalg.det(sc[:4, :4]))
    det6 = float(np.linalg.det(sc))
    if det4 > 0.0 and np.all(proj[:2] > 0.0):
        out["e4d"] = float(np.sqrt(det4))
        out["coupling4"] = float(np.sqrt(proj[0] * proj[1]) / out["e4d"])
    if det6 <= 0.0 or np.any(proj <= 0.0):
        return out
    out["proj"] = np.sqrt(proj)
    out["e6d"] = float(np.sqrt(det6))
    out["coupling6"] = float(np.prod(out["proj"]) / out["e6d"])
    # Σ_c·J = A·diag(ε_k J₂)·A⁻¹ with A symplectic: its eigenvalues are
    # ±iε_k and the eigenvector of +iε_k is (a_2k − i·a_2k+1) for the
    # mode's pair of columns of A.  Im(v̄_2j·v_2j+1) is that mode's share
    # of symplectic area in plane j; the shares of one mode sum to 1.
    lam, vec = np.linalg.eig(sc @ _J6)
    up = np.argsort(-lam.imag)[:3]            # the three +iε_k
    eps = lam.imag[up]
    if np.any(eps <= 0.0):
        return out
    share = np.empty((3, 3))
    for i, col in enumerate(up):
        v = vec[:, col]
        a = np.array([(np.conj(v[2 * j]) * v[2 * j + 1]).imag for j in range(3)])
        tot = a.sum()
        share[i] = a / tot if tot != 0.0 else np.nan
    # Label modes by plane: the permutation maximising the summed share.
    import itertools
    best = max(itertools.permutations(range(3)),
               key=lambda p: sum(share[p[k], k] for k in range(3)))
    out["modes"] = np.array([eps[best[k]] for k in range(3)])
    out["share"] = np.array([share[best[k]] for k in range(3)])
    return out


def canonical_eigenemittances_series(results) -> dict | None:
    """:func:`canonical_eigenemittances` along a run.

    ``results`` needs per-step ``sigma_matrix``, ``ref_beta``, ``ref_gamma``
    and ``ref_frequency`` (MHz) plus ``mass_mev`` — recorded by both the
    envelope solver and the multi-particle recorder, and kept by the HDF5
    results file.  Records of a DC (unbunched) beam carry no longitudinal
    emittance, so their 6-D quantities are NaN (``e4d`` is kept).

    Returns a dict of arrays aligned with ``results.s`` — ``eps_x``,
    ``eps_y``, ``eps_z`` (labelled modes), ``proj_x``, ``proj_y``,
    ``proj_z``, ``e4d``, ``e6d``, ``coupling6``, ``coupling4`` — or None
    when the inputs are missing or misaligned.
    """
    def arr(name):
        v = getattr(results, name, None)
        if v is None:
            return np.zeros(0)
        try:
            return np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            return np.zeros(0)

    S = arr("sigma_matrix")
    b, g, f = arr("ref_beta"), arr("ref_gamma"), arr("ref_frequency")
    m = float(getattr(results, "mass_mev", 0.0) or 0.0)
    if not m:
        w = arr("ref_w_kin")
        if w.size and g.size and g[0] > 1.0 + 1e-12:
            m = float(w[0] / (g[0] - 1.0))
    n = S.shape[0] if S.ndim == 3 else 0
    if n == 0 or S.shape[1:] != (6, 6) or not (b.size == g.size == f.size == n):
        return None
    dc = arr("continuous_at")
    dc = dc.astype(bool) if dc.size == n else np.zeros(n, dtype=bool)
    return _canonical_batch(S, b * g, f, m, dc)


def _canonical_batch(S, bg, f, m, dc) -> dict:
    """Vectorised :func:`canonical_eigenemittances` over a (n, 6, 6) stack
    (same NaN rules; ``dc`` marks records whose 6-D outputs are NaN)."""
    import itertools
    n = S.shape[0]
    keys = ("eps_x", "eps_y", "eps_z", "proj_x", "proj_y", "proj_z",
            "e4d", "e6d", "coupling6", "coupling4")
    out = {k: np.full(n, np.nan) for k in keys}
    ok_in = (np.all(np.isfinite(S.reshape(n, 36)), axis=1) & (bg > 0.0)
             & (f > 0.0) & (m > 0.0))
    if not ok_in.any():
        return out
    from linac_gen.core.constants import C_LIGHT
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(f > 0.0, C_LIGHT / (f * 1e6) * 1e3, np.nan)
        d = np.stack([np.ones(n), bg, np.ones(n), bg, lam / 360.0,
                      np.full(n, -1e3 / m if m > 0.0 else np.nan)], axis=1)
    idx = np.flatnonzero(ok_in)
    sc = S[idx] * d[idx, :, None] * d[idx, None, :]
    proj = np.stack([np.linalg.det(sc[:, k:k + 2, k:k + 2]) for k in (0, 2, 4)],
                    axis=1)
    det4 = np.linalg.det(sc[:, :4, :4])
    det6 = np.linalg.det(sc)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok4 = (det4 > 0.0) & (proj[:, 0] > 0.0) & (proj[:, 1] > 0.0)
        e4 = np.sqrt(np.where(ok4, det4, np.nan))
        out["e4d"][idx] = e4
        out["coupling4"][idx] = np.sqrt(proj[:, 0] * proj[:, 1]) / e4
    ok6 = (det6 > 0.0) & np.all(proj > 0.0, axis=1) & ~dc[idx]
    if not ok6.any():
        return out
    i6, sc6 = idx[ok6], sc[ok6]
    pr = np.sqrt(proj[ok6])
    e6 = np.sqrt(det6[ok6])
    lam_, vec = np.linalg.eig(sc6 @ _J6)
    up = np.argsort(-lam_.imag, axis=1)[:, :3]
    eps = np.take_along_axis(lam_.imag, up, axis=1)            # (k, 3)
    v = np.take_along_axis(vec, up[:, None, :], axis=2)         # (k, 6, 3)
    a = (np.conj(v[:, 0::2, :]) * v[:, 1::2, :]).imag           # (k, plane, mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.transpose(a / a.sum(axis=1, keepdims=True), (0, 2, 1))
    perms = list(itertools.permutations(range(3)))
    score = np.stack([sum(share[:, p[k], k] for k in range(3)) for p in perms],
                     axis=1)
    best = np.asarray(perms)[np.nanargmax(np.nan_to_num(score, nan=-np.inf), axis=1)]
    modes = np.take_along_axis(eps, best, axis=1)
    good = np.all(eps > 0.0, axis=1)
    for k, pl in enumerate("xyz"):
        out[f"eps_{pl}"][i6] = np.where(good, modes[:, k], np.nan)
        out[f"proj_{pl}"][i6] = pr[:, k]
    out["e6d"][i6] = e6
    out["coupling6"][i6] = np.prod(pr, axis=1) / e6
    return out
