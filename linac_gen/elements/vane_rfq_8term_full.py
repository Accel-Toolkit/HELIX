"""Full 8-term Crandall multipole expansion for VaneRFQ (M3.3 milestone).

Engaged via ``field_model="8term_full"``.  This is the canonical
Toutatis-equivalent analytic form: each term in the expansion
individually satisfies ``∇²V = 0`` so there is no boundary noise or
grid quantisation as in the laplace2d / laplace3d numerical paths.

Potential expansion (Crandall LA-11968-MS Eq. 2-10):

  V(r,θ,z) = (V_amp/2) [
      A_{1,0} · (r/r₀)² · cos(2θ)
    + A_{3,0} · (r/r₀)⁶ · cos(6θ)
    + A_{0,1} · I_0(kr) · cos(kz')
    + A_{2,1} · I_4(kr) · cos(4θ) · cos(kz')
    + A_{1,2} · I_2(2kr) · cos(2θ) · cos(2kz')
    + A_{3,2} · I_6(2kr) · cos(6θ) · cos(2kz')
    + A_{0,3} · I_0(3kr) · cos(3kz')
    + A_{2,3} · I_4(3kr) · cos(4θ) · cos(3kz')
  ]

with ``k = π/L``, ``z' = z − z_cell_start``.

Coefficient strategy
--------------------
``solve_cell_coeffs_dat(m, A10, r0, L)`` — analytic 2-term Crandall
formulas (Wangler RFQ §8.1, Crandall LA-11968-MS Eq. 2-7) computed
from the .dat-supplied cell parameters.  Higher modes (3,0), (1,2),
(0,3) etc. are set to zero — that's the canonical 2-term Crandall
limit.  This is the safe, well-defined coefficient source.

``solve_cell_coeffs_bc(...)`` — solves an over-determined BC system
on the actual .vane aperture profile (16 eqs × 8 unknowns).  Useful
for non-Crandall vane shapes but ill-conditioned for matcher / type-3
cells where the .vane profile doesn't match the basis ansatz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import iv


# Order of the 8 (n,p) modes used throughout this module.
MODES: Tuple[Tuple[int, int], ...] = (
    (1, 0),  # DC quadrupole          → (r/r₀)² cos(2θ)
    (3, 0),  # DC 12-pole             → (r/r₀)⁶ cos(6θ)
    (0, 1),  # accel. 1st harmonic    → I_0(kr) cos(kz)
    (2, 1),  # octupole 1st harmonic  → I_4(kr) cos(4θ) cos(kz)
    (1, 2),  # RF quad 2nd harmonic   → I_2(2kr) cos(2θ) cos(2kz)
    (3, 2),  # RF 12-pole 2nd         → I_6(2kr) cos(6θ) cos(2kz)
    (0, 3),  # accel. 3rd harmonic    → I_0(3kr) cos(3kz)
    (2, 3),  # octupole 3rd harmonic  → I_4(3kr) cos(4θ) cos(3kz)
)


@dataclass
class CellCoeffs:
    """8-term Crandall coefficients for a single RFQ cell."""

    A: np.ndarray          # shape (8,)  coefficients in MODES order
    r0_mm: float           # mm   reference aperture
    k_per_mm: float        # 1/mm   π/L
    L_mm: float            # mm   cell length
    z_start_mm: float      # mm   cell start (element-local z)
    V_amp_volts: float     # V   inter-vane voltage at cell midpoint
    cond_number: float     # condition number of the BC system
    residual: float        # max BC residual (V on vane − V_target) / V_amp


# ------------------------------------------------------------------
# Basis function evaluation
# ------------------------------------------------------------------
def _basis_at(x_mm: float, y_mm: float, z_rel_mm: float,
              k_per_mm: float, r0_mm: float) -> np.ndarray:
    """Return the 8 basis-function values at (x, y, z_rel).

    Vector-valued — order matches :data:`MODES`.
    """
    r2 = x_mm * x_mm + y_mm * y_mm
    r = np.sqrt(r2)

    # Azimuthal cos(2nθ) via Cartesian → Chebyshev recursion to avoid
    # arctan2 round-off near r=0.
    if r2 > 0.0:
        c2 = (x_mm * x_mm - y_mm * y_mm) / r2
        s2 = 2.0 * x_mm * y_mm / r2
    else:
        c2 = 1.0
        s2 = 0.0
    c4 = 2.0 * c2 * c2 - 1.0
    c6 = 4.0 * c2 * c2 * c2 - 3.0 * c2

    kr = k_per_mm * r
    cos_kz  = np.cos(k_per_mm * z_rel_mm)
    cos_2kz = np.cos(2.0 * k_per_mm * z_rel_mm)
    cos_3kz = np.cos(3.0 * k_per_mm * z_rel_mm)

    rho2 = r2 / (r0_mm * r0_mm) if r0_mm > 0 else 0.0
    rho6 = rho2 * rho2 * rho2

    return np.array([
        rho2 * c2,                                   # (1,0)
        rho6 * c6,                                   # (3,0)
        float(iv(0, kr)) * cos_kz,                   # (0,1)
        float(iv(4, kr)) * c4 * cos_kz,              # (2,1)
        float(iv(2, 2.0 * kr)) * c2 * cos_2kz,       # (1,2)
        float(iv(6, 2.0 * kr)) * c6 * cos_2kz,       # (3,2)
        float(iv(0, 3.0 * kr)) * cos_3kz,            # (0,3)
        float(iv(4, 3.0 * kr)) * c4 * cos_3kz,       # (2,3)
    ])


# ------------------------------------------------------------------
# Coefficient solver — analytic 2-term from cell .dat parameters
# ------------------------------------------------------------------
def solve_cell_coeffs_dat(z_start_mm: float, z_end_mm: float,
                          modulation: float, A10_dat: float,
                          r0_mm: float, V_amp_volts: float,
                          S_sign: float = -1.0,
                          ) -> CellCoeffs:
    """Crandall 2-term coefficients from cell .dat parameters.

    Solves the analytic 2-term BC system at the *idealised* vane apex
    points (x-vane apex at (a,0) at z=0, y-vane apex at (0,ma) at z=0)
    using:

      A_{0,1} = (m²−1) / (m² · I₀(ka) + I₀(mka))
      A_{1,0} = m · (I₀(ka) + I₀(mka)) / (m² · I₀(ka) + I₀(mka))

    where ``a = r₀/√m`` and ``k = π/L``.  All other 8-term coefficients
    set to zero — this is the standard 2-term limit.

    Notes
    -----
    The .dat-supplied ``A10`` is used as a sanity check: in the ideal
    Crandall geometry, ``A_{0,1} == A10``.  We do *not* re-derive
    ``A_{0,1}`` from the formula — we trust the .dat (which may carry
    designer-specified values that differ slightly from the analytic
    expression because of finite-Tc corrections in RFQUIK's design).
    Instead we derive ``A_{1,0}`` from the .dat's ``A10`` to maintain
    self-consistency:

      A_{1,0} = m · (1 − A10 · I₀(ka))    (rearrangement of the BC)

    This collapses to the M1 short form (1 − A10) only in the m → 1
    limit (matcher).  For m > 1 the full Crandall coefficient is
    typically 30 – 70 % stronger than M1's short form — exactly the
    effect that closes the σ_x gap toward TraceWin's reference.
    """
    L = z_end_mm - z_start_mm
    if L <= 0:
        raise ValueError(f"cell length must be positive; got {L} mm")
    k = np.pi / L

    # 2-term Crandall coefficients (Crandall LA-11968-MS Eq. 2-7).
    if abs(modulation - 1.0) < 1e-9 or r0_mm <= 0:
        # Pure DC quadrupole — matcher / m=1 limit.
        A10_coeff = 1.0
        A01_coeff = float(A10_dat)  # typically 0 for matcher
    else:
        a_mm = r0_mm / float(np.sqrt(modulation))
        ka = k * a_mm
        I0_ka  = float(iv(0, ka))
        I0_mka = float(iv(0, modulation * ka))
        # A_{0,1} from analytic formula; trust the .dat A10 if given.
        A01_analytic = (modulation * modulation - 1.0) / (
            modulation * modulation * I0_ka + I0_mka)
        A01_coeff = float(A10_dat) if A10_dat else A01_analytic
        # A_{1,0} from BC consistency: m · (1 − A_{0,1} · I₀(ka)).
        A10_coeff = modulation * (1.0 - A01_coeff * I0_ka)

    A_coeffs = np.zeros(8)
    A_coeffs[0] = A10_coeff   # (1, 0)
    A_coeffs[2] = A01_coeff   # (0, 1)
    # All others stay at 0 (canonical 2-term limit).

    # TraceWin AG-focusing sign convention.  M1's RfqCell folds an
    # ``S`` factor into ``rf_quad_part``: for accelerating cells
    # ``S = −sign(cell_type)``; for matcher / shaper cells (Type ±3,
    # ±4) ``S`` is determined by the *adjacent* cell's sign — see
    # :meth:`RfqCell._type_coeffs` (manual images 281–290).  The caller
    # is expected to supply the same ``S_sign`` so the AG alternation
    # matches M1 cell-for-cell.  Sign flip applies to the cos(2nθ)
    # modes with ``n`` odd (n=1, 3).
    A_coeffs[0] *= S_sign    # (1, 0)
    A_coeffs[1] *= S_sign    # (3, 0)
    A_coeffs[4] *= S_sign    # (1, 2)
    A_coeffs[5] *= S_sign    # (3, 2)

    return CellCoeffs(
        A=A_coeffs, r0_mm=float(r0_mm), k_per_mm=k, L_mm=L,
        z_start_mm=z_start_mm, V_amp_volts=float(V_amp_volts),
        cond_number=1.0, residual=0.0,
    )


# ------------------------------------------------------------------
# Vectorised basis matrix + surface fit (vane8t, 2026-07-30)
# ------------------------------------------------------------------
def basis_matrix(pts_mm: np.ndarray, k_per_mm: float,
                 r0_mm: float) -> np.ndarray:
    """(N, 8) basis-function matrix at points (x, y, z_rel) [mm].

    Vectorised twin of :func:`_basis_at` (same MODES order, same
    Chebyshev azimuthal recursion)."""
    x = pts_mm[:, 0]
    y = pts_mm[:, 1]
    z = pts_mm[:, 2]
    r2 = x * x + y * y
    r = np.sqrt(r2)
    safe = r2 > 0.0
    c2 = np.where(safe, (x * x - y * y) / np.where(safe, r2, 1.0), 1.0)
    c4 = 2.0 * c2 * c2 - 1.0
    c6 = 4.0 * c2 ** 3 - 3.0 * c2
    kr = k_per_mm * r
    cz1 = np.cos(k_per_mm * z)
    cz2 = np.cos(2.0 * k_per_mm * z)
    cz3 = np.cos(3.0 * k_per_mm * z)
    rho2 = r2 / (r0_mm * r0_mm)
    return np.column_stack([
        rho2 * c2,
        rho2 ** 3 * c6,
        iv(0, kr) * cz1,
        iv(4, kr) * c4 * cz1,
        iv(2, 2.0 * kr) * c2 * cz2,
        iv(6, 2.0 * kr) * c6 * cz2,
        iv(0, 3.0 * kr) * cz3,
        iv(4, 3.0 * kr) * c4 * cz3,
    ])


def fit_cell_multipoles(pts_mm: np.ndarray, v_norm: np.ndarray,
                        z_start_mm: float, length_mm: float,
                        r0_mm: float, V_amp_volts: float) -> CellCoeffs:
    """Least-squares 8-term fit to TRUE electrode-surface points.

    ``pts_mm`` are (x, y, z_local) boundary samples from
    :mod:`linac_gen.elements.rfq_vane_surface` (the Tc tip arcs — NOT
    just the four apexes, which carry only two-term information);
    ``v_norm`` is the normalised electrode potential (±1 ≡ ±V/2).
    Returns a :class:`CellCoeffs` with conditioning diagnostics.

    .. warning:: DIAGNOSTIC USE ONLY — never feed the result into
       tracking.  Surface points sit OUTSIDE the expansion's
       convergence radius (r > a), so while the leading modes are
       stable, the degenerate higher modes come back as wild
       unregularised extrapolations (|A| up to ~10²-10³ on long PXIE
       cells).  The interior-field route (FD solve + cylinder Fourier
       projection, see the vane-field campaign record in tests/rfq/
       test_vane_campaign.py) is the valid extraction; this fitter
       exists to quantify the leading-mode surface content (the quad
       reduction that motivated the TW calibration).
    """
    if length_mm <= 0:
        raise ValueError("cell length must be positive")
    k = np.pi / length_mm
    M = basis_matrix(pts_mm, k, r0_mm)
    A, _res, _rank, sv = np.linalg.lstsq(M, v_norm, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    res_max = float(np.max(np.abs(M @ A - v_norm)))
    return CellCoeffs(A=A, r0_mm=float(r0_mm), k_per_mm=k,
                      L_mm=float(length_mm), z_start_mm=float(z_start_mm),
                      V_amp_volts=float(V_amp_volts),
                      cond_number=cond, residual=res_max)


# ------------------------------------------------------------------
# Coefficient solver — BC fit on .vane geometry (legacy/diagnostic)
# ------------------------------------------------------------------
def solve_cell_coeffs(z_start_mm: float, z_end_mm: float,
                      vane: object, V_amp_volts: float,
                      n_z_samples: int = 4) -> CellCoeffs:
    """Solve the 8 multipole coefficients for one RFQ cell.

    Parameters
    ----------
    z_start_mm, z_end_mm
        Cell start and end (element-local z, mm).
    vane
        VaneGeometry instance — used for per-z aperture lookups.
    V_amp_volts
        Inter-vane voltage (V₁ − V₂); typically +V on vanes 1,3 and
        −V on vanes 2,4 with ``V = V_amp / 2``.
    n_z_samples
        Number of z-positions inside the cell at which to enforce BCs
        (default 4 → 16 equations × 8 unknowns).
    """
    L = z_end_mm - z_start_mm
    if L <= 0:
        raise ValueError(f"cell length must be positive; got {L} mm")
    k = np.pi / L

    # z-sample positions: spread inside (0, L) avoiding the endpoints.
    # Even spacing inside each quarter: 0.125L, 0.375L, 0.625L, 0.875L.
    z_offsets_rel = (np.arange(n_z_samples) + 0.5) / n_z_samples * L

    # Use the cell-mean of √(a₁·a₂) across samples as r₀ reference.
    z_m = (z_start_mm + z_offsets_rel) * 1e-3
    a1_m = np.interp(z_m, vane.z, vane.aperture_v1)   # m
    a2_m = np.interp(z_m, vane.z, vane.aperture_v2)
    a3_m = np.interp(z_m, vane.z, vane.aperture_v3)
    a4_m = np.interp(z_m, vane.z, vane.aperture_v4)
    r0_mm = float(np.sqrt(np.mean(a1_m) * np.mean(a2_m)) * 1000.0)

    # Vane voltages at samples (V).
    v1 = np.interp(z_m, vane.z, vane.voltage_v1)
    v2 = np.interp(z_m, vane.z, vane.voltage_v2)
    v3 = np.interp(z_m, vane.z, vane.voltage_v3)
    v4 = np.interp(z_m, vane.z, vane.voltage_v4)
    V_half = 0.5 * V_amp_volts

    # Build the 4*n_z × 8 BC system: rows are [vane1,vane2,vane3,vane4]
    # at each z-sample.  RHS is V_vane / (V_amp/2).
    n_rows = 4 * n_z_samples
    M = np.zeros((n_rows, 8))
    rhs = np.zeros(n_rows)
    for i, z_rel in enumerate(z_offsets_rel):
        a1 = float(a1_m[i] * 1000.0)  # mm
        a2 = float(a2_m[i] * 1000.0)
        a3 = float(a3_m[i] * 1000.0)
        a4 = float(a4_m[i] * 1000.0)
        # Vane 1: (+a1, 0)
        M[4 * i + 0, :] = _basis_at(+a1, 0.0, z_rel, k, r0_mm)
        rhs[4 * i + 0] = float(v1[i]) / V_half if V_half != 0.0 else 0.0
        # Vane 2: (0, +a2)
        M[4 * i + 1, :] = _basis_at(0.0, +a2, z_rel, k, r0_mm)
        rhs[4 * i + 1] = float(v2[i]) / V_half if V_half != 0.0 else 0.0
        # Vane 3: (−a3, 0)
        M[4 * i + 2, :] = _basis_at(-a3, 0.0, z_rel, k, r0_mm)
        rhs[4 * i + 2] = float(v3[i]) / V_half if V_half != 0.0 else 0.0
        # Vane 4: (0, −a4)
        M[4 * i + 3, :] = _basis_at(0.0, -a4, z_rel, k, r0_mm)
        rhs[4 * i + 3] = float(v4[i]) / V_half if V_half != 0.0 else 0.0

    # Least-squares solve (16 eqs > 8 unknowns).
    A_coeffs, residuals, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float('inf')
    pred = M @ A_coeffs
    res_max = float(np.max(np.abs(pred - rhs)))

    return CellCoeffs(
        A=A_coeffs, r0_mm=r0_mm, k_per_mm=k, L_mm=L,
        z_start_mm=z_start_mm, V_amp_volts=V_amp_volts,
        cond_number=cond, residual=res_max,
    )


# ------------------------------------------------------------------
# On-axis derivatives — what the envelope tracker actually needs
# ------------------------------------------------------------------
def K_xx_axis(coeffs: CellCoeffs, z_mm: float) -> float:
    """Return ``∂²V/∂x²(0,0,z) / 1`` (V/mm²) at element-local z.

    Computed analytically from the 8-term basis at the on-axis limit.
    The (3,0), (2,1), (3,2), (2,3) modes vanish to second order at
    r=0; only (1,0), (0,1), (1,2), (0,3) contribute.
    """
    z_rel = z_mm - coeffs.z_start_mm
    if z_rel < 0.0 or z_rel > coeffs.L_mm:
        return 0.0
    k = coeffs.k_per_mm
    A10 = coeffs.A[0]
    A01 = coeffs.A[2]
    A12 = coeffs.A[4]
    A03 = coeffs.A[6]
    half = 0.5 * coeffs.V_amp_volts

    # Hessian contributions at r=0:
    #   ∂²/∂x²[(r/r₀)²cos(2θ)] = +2/r₀²
    #   ∂²/∂x²[I_0(kr)cos(kz)] at r=0  = (k²/2) cos(kz)
    #   ∂²/∂x²[I_2(2kr)cos(2θ)cos(2kz)] at r=0  = +k² cos(2kz)
    #   ∂²/∂x²[I_0(3kr)cos(3kz)] at r=0  = (9k²/2) cos(3kz)
    return half * (
        A10 * (2.0 / (coeffs.r0_mm * coeffs.r0_mm))
        + A01 * (0.5 * k * k) * np.cos(k * z_rel)
        + A12 * (k * k) * np.cos(2.0 * k * z_rel)
        + A03 * (4.5 * k * k) * np.cos(3.0 * k * z_rel)
    )


def K_yy_axis(coeffs: CellCoeffs, z_mm: float) -> float:
    """``∂²V/∂y²(0,0,z)`` (V/mm²) — sign-flipped (1,0) and (1,2) vs K_xx."""
    z_rel = z_mm - coeffs.z_start_mm
    if z_rel < 0.0 or z_rel > coeffs.L_mm:
        return 0.0
    k = coeffs.k_per_mm
    A10 = coeffs.A[0]
    A01 = coeffs.A[2]
    A12 = coeffs.A[4]
    A03 = coeffs.A[6]
    half = 0.5 * coeffs.V_amp_volts
    return half * (
        A10 * (-2.0 / (coeffs.r0_mm * coeffs.r0_mm))
        + A01 * (0.5 * k * k) * np.cos(k * z_rel)
        + A12 * (-(k * k)) * np.cos(2.0 * k * z_rel)
        + A03 * (4.5 * k * k) * np.cos(3.0 * k * z_rel)
    )


def Ez_axis(coeffs: CellCoeffs, z_mm: float) -> float:
    """``E_z(0,0,z)`` static amplitude (V/mm) at element-local z.

    Static = before the global sin(ωt+φ_s) RF time factor.  Equals
    ``-∂V/∂z`` of the on-axis potential ``(V/2)·[A_{0,1}cos(kz')
    + A_{0,3}cos(3kz')]``.
    """
    z_rel = z_mm - coeffs.z_start_mm
    if z_rel < 0.0 or z_rel > coeffs.L_mm:
        return 0.0
    k = coeffs.k_per_mm
    A01 = coeffs.A[2]
    A03 = coeffs.A[6]
    half = 0.5 * coeffs.V_amp_volts
    return half * (
        A01 * k * np.sin(k * z_rel)
        + A03 * 3.0 * k * np.sin(3.0 * k * z_rel)
    )
