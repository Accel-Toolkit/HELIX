# linac_gen/tracking/envelope.py
"""RMS envelope tracking with analytical 3D-ellipsoid space charge.

The 6x6 beam sigma matrix propagates through linear elements as:
    Sigma_new = M @ Sigma_old @ M.T

Space charge is a thin defocusing lens built from the **uniform triaxial
ellipsoid** model (Lapostolle/Wangler).  Using Q_bunch = I/f_RF and the
rms-equivalent semi-axes (√5·σ_x, √5·σ_y, √5·γ·σ_z) it computes Maxwell
depolarization factors M_x, M_y, M_z via the Carlson elliptic integral
R_D, and applies linear kicks in all three planes — including
longitudinal, which the earlier 2D Sacherer formulation lacked.
"""
import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.special import elliprd

from linac_gen.core.constants import EPSILON_0, E_CHARGE, C_LIGHT, PI
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.base import (
    TransferMapElement, ThinKickElement, PassiveElement, FieldMapElement,
)
from linac_gen.tracking.matrix_tracking import get_element_matrix


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EnvelopeResults:
    """Output of the envelope solver: RMS beam parameters vs. longitudinal position."""
    s: List[float] = field(default_factory=list)
    sigma_x: List[float] = field(default_factory=list)
    sigma_y: List[float] = field(default_factory=list)
    sigma_phi: List[float] = field(default_factory=list)
    sigma_w: List[float] = field(default_factory=list)
    emit_x: List[float] = field(default_factory=list)        # mm.mrad
    emit_y: List[float] = field(default_factory=list)        # mm.mrad
    emit_z: List[float] = field(default_factory=list)        # deg.MeV (native)
    emit_z_mmmrad: List[float] = field(default_factory=list) # emit_z re-expressed in mm.mrad
    emit_4d: List[float] = field(default_factory=list)       # √det(σ_4D) — coupling-invariant transverse area (TraceWin εt)
    # 6-D normal-mode (eigen-) emittances of Σ — invariants of any
    # symplectic transport.  In uncoupled motion they reduce to (ε_x,
    # ε_y, ε_z); under x-y / dispersive coupling they remain constant
    # while the projections oscillate.  Computed via the Balandin
    # I₂/I₄/I₆ trace formula.  Units: mm·mrad for ε₁, ε₂, deg·MeV for ε₃.
    emit_e1: List[float] = field(default_factory=list)
    emit_e2: List[float] = field(default_factory=list)
    emit_e3: List[float] = field(default_factory=list)
    alpha_x: List[float] = field(default_factory=list)
    beta_x: List[float] = field(default_factory=list)
    alpha_y: List[float] = field(default_factory=list)
    beta_y: List[float] = field(default_factory=list)
    # Longitudinal Twiss from the (Δφ deg, ΔW MeV) Σ-block — HELIX-
    # internal convention (α_z = −Σ45/ε_z; a late particle has Δφ>0,
    # so TW's α_z is the NEGATIVE of this — appendix E).  β_z is in
    # deg/MeV at the LOCAL machine clock (scales f_new/f_old at FREQ,
    # like emit_z; α_z is invariant across a jump).
    alpha_z: List[float] = field(default_factory=list)
    beta_z: List[float] = field(default_factory=list)
    ref_w_kin: List[float] = field(default_factory=list)
    ref_beta: List[float] = field(default_factory=list)
    ref_gamma: List[float] = field(default_factory=list)
    ref_frequency: List[float] = field(default_factory=list)  # MHz, per-step
    sigma_matrix: List = field(default_factory=list)     # list of (6, 6) arrays
    element_names: List[str] = field(default_factory=list)
    # Record index of the row holding element j's EXIT state.  With
    # record_substeps off this is simply j+1 (row 0 = INPUT); with
    # substeps on, elements contribute a variable number of interior
    # rows and this mapping is the ONLY correct way to translate
    # element spans → record spans (see analysis.phase_advance.
    # element_record_span).
    element_exit_idx: List[int] = field(default_factory=list)
    # --- Phase probe (populated only when EnvelopeSolver(phase_probe=
    # True)) -----------------------------------------------------------
    # element_maps_dep[j]: the EXACT composite 6×6 map the solver applied
    # to Σ across element j — the ordered product of every Strang
    # half-map, SC kick, freq-jump D, edge, etc., at the σ/energy the
    # run used.  element_maps_bare[j]: the same product with SC kicks
    # replaced by identity, accumulated in the SAME walk (identical
    # slices and reference trajectory — never a separate I=0 run).
    # Foil diffusion is additive (not a matrix); its mean map is the
    # identity in both.  Per-slice history (probe_*) enables monodromy
    # assembly over arbitrary spans and substep-resolution phase
    # integration: one entry per applied map, in application order.
    element_maps_dep: List = field(default_factory=list)   # (6,6) per element
    element_maps_bare: List = field(default_factory=list)  # (6,6) per element
    probe_elem_idx: List[int] = field(default_factory=list)  # element of each slice
    probe_is_sc: List[bool] = field(default_factory=list)    # SC kick slice?
    probe_s: List[float] = field(default_factory=list)       # ref.s after slice [mm]
    probe_M: List = field(default_factory=list)              # (6,6) slice maps
    current_mA: float = 0.0                              # SC beam current at run time
    continuous: bool = False                              # True if DC (2-D SC), False if bunched
    mass_mev: float = 0.0                                # rest mass in MeV (for downstream analyzers)
    # First moment (beam centroid) per record row, RELATIVE to the
    # reference particle: 6-vector [x mm, x' mrad, y mm, y' mrad,
    # Δφ deg, ΔW MeV] — the same convention as the MP recorder's
    # ``centroid`` so envelope and MP orbits are directly comparable.
    # Propagated c → Mc through the same matrices Σ uses, plus steerer
    # kicks and dx/dy/tilt misalignment feed-down; linear space charge
    # does NOT move the centroid (own-field is centred on the beam —
    # same approximation as TraceWin).  len(centroid) == len(s) at
    # every record cadence (an EMPTY list = the result carries no first
    # moment, e.g. backtrack shells).  Foil caveat: the envelope books
    # the mean ionisation loss on the REFERENCE while MP shifts the
    # particles' dW, so downstream of a Foil the RELATIVE c[5] differs
    # from MP's by dE_mean (absolute mean energy agrees).
    centroid: List = field(default_factory=list)         # (6,) arrays


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def compute_perveance(current_mA: float, charge: int,
                      mass_MeV: float, beta: float, gamma: float) -> float:
    """Generalized perveance K (dimensionless).

    K = q * I / (2*pi * eps0 * m0*c^3 * beta^3 * gamma^3)

    Parameters
    ----------
    current_mA : float
        Beam current in milli-Amperes.
    charge : int
        Charge state (units of elementary charge, sign ignored — always positive).
    mass_MeV : float
        Rest mass energy m0*c^2 in MeV.
    beta : float
        Relativistic beta of the synchronous particle.
    gamma : float
        Relativistic gamma of the synchronous particle.

    Returns
    -------
    float
        Dimensionless perveance K >= 0.
    """
    if current_mA == 0.0:
        return 0.0
    I_A = abs(current_mA) * 1e-3  # Amperes
    q = abs(charge) * E_CHARGE    # Coulombs
    # m0*c^3 in SI: kg * m^3/s^3 = (m0*c^2 in Joules) * c = mass_MeV * 1e6 * E_CHARGE * C_LIGHT
    m0c3 = mass_MeV * 1e6 * E_CHARGE * C_LIGHT
    K = q * I_A / (2.0 * PI * EPSILON_0 * m0c3 * beta**3 * gamma**3)
    return K


def _sigma_to_twiss(sigma: np.ndarray, plane: str) -> dict:
    """Extract Twiss parameters from a 6x6 sigma matrix for a given plane.

    Parameters
    ----------
    sigma : np.ndarray, shape (6, 6)
    plane : str, one of 'x', 'y', 'z'

    Returns
    -------
    dict with keys 'alpha', 'beta', 'emit'
    """
    col_map = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
    i, j = col_map[plane]
    s11 = sigma[i, i]
    s12 = sigma[i, j]
    s22 = sigma[j, j]
    emit = math.sqrt(max(s11 * s22 - s12 * s12, 0.0))
    if emit < 1e-30:
        return {"alpha": 0.0, "beta": 0.0, "emit": 0.0}
    beta = s11 / emit
    alpha = -s12 / emit
    return {"alpha": alpha, "beta": beta, "emit": emit}


def _build_sigma_matrix(alpha_x: float, beta_x: float, emit_x: float,
                        alpha_y: float, beta_y: float, emit_y: float,
                        alpha_z: float, beta_z: float, emit_z: float,
                        disp_x: float = 0.0, disp_xp: float = 0.0,
                        disp_y: float = 0.0, disp_yp: float = 0.0) -> np.ndarray:
    """Build a 6x6 sigma matrix from Twiss parameters.

    Each 2x2 transverse block follows:
        [[beta * emit,   -alpha * emit],
         [-alpha * emit,  gamma_t * emit]]
    where gamma_t = (1 + alpha^2) / beta.

    ``disp_x`` / ``disp_xp`` / ``disp_y`` / ``disp_yp`` (default 0) seed an
    input dispersion — a coordinate becomes ``q = q_betatron + D_q · ΔW``,
    which adds ``D_q·D_r·σ_WW`` cross-terms.  With all four zero the result
    is the plain block-diagonal matrix (unchanged behaviour).
    """
    sigma = np.zeros((6, 6))

    def _fill_block(i, j, alpha, beta, emit):
        gamma_t = (1.0 + alpha * alpha) / beta
        sigma[i, i] = beta * emit
        sigma[i, j] = -alpha * emit
        sigma[j, i] = -alpha * emit
        sigma[j, j] = gamma_t * emit

    _fill_block(0, 1, alpha_x, beta_x, emit_x)  # x plane
    _fill_block(2, 3, alpha_y, beta_y, emit_y)  # y plane
    _fill_block(4, 5, alpha_z, beta_z, emit_z)  # z plane

    # Optional input dispersion: q_total = q_betatron + D_q · ΔW (index 5).
    if disp_x or disp_xp or disp_y or disp_yp:
        s_ww = sigma[5, 5]
        eta = {0: disp_x, 1: disp_xp, 2: disp_y, 3: disp_yp}
        for i, di in eta.items():
            if di == 0.0:
                continue
            sigma[i, 5] += di * s_ww
            sigma[5, i] += di * s_ww
            for j, dj in eta.items():
                sigma[i, j] += di * dj * s_ww
    return sigma


def _ellipsoid_depolarization_factors(a: float, b: float, c: float) -> tuple:
    """Maxwell depolarization factors (M_x, M_y, M_z) of a uniform triaxial
    ellipsoid with semi-axes (a, b, c).

    Definition (with a, b, c > 0):
        M_i = (abc / 2) · ∫₀^∞ dt / [(t + r_i²)^(3/2) · √(t + r_j²)(t + r_k²)]

    Satisfies M_x + M_y + M_z = 1.  Implemented via Carlson's symmetric
    elliptic integral R_D:
        M_i = (abc / 3) · R_D(r_j², r_k², r_i²).

    Analytic closed forms are used for the near-axisymmetric and
    near-spherical limits for numerical robustness.
    """
    if a <= 0 or b <= 0 or c <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    # Sphere (within 1 ppm): isotropic
    if abs(a - b) < 1e-6 * a and abs(b - c) < 1e-6 * b:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    a2, b2, c2 = a * a, b * b, c * c
    abc = a * b * c
    Mx = (abc / 3.0) * elliprd(b2, c2, a2)
    My = (abc / 3.0) * elliprd(c2, a2, b2)
    Mz = (abc / 3.0) * elliprd(a2, b2, c2)
    return float(Mx), float(My), float(Mz)


def _sc_kick_matrix_3d(
    current_mA: float, charge_state: int, mass_MeV: float,
    beta: float, gamma: float, frequency_MHz: float,
    sigma_x_mm: float, sigma_y_mm: float, sigma_phi_deg: float,
    ds_mm: float,
    bunch_frequency_MHz: float | None = None,
) -> np.ndarray:
    """Thin-lens SC kick matrix for a bunched, uniform-density 3D ellipsoid.

    Input geometry:
        - Transverse σ_x, σ_y are LAB-frame sizes in mm (unchanged by the
          z-boost).
        - σ_φ is the LAB-frame longitudinal rms in degrees of RF phase.  We
          convert to σ_z_lab [mm] via z = -Δφ · β · λ_mm / 360 and then
          elongate by γ to reach the rest-frame length scale (σ_z_rest =
          γ · σ_z_lab).

    Physics (rest-frame uniform ellipsoid, semi-axes a = √5·σ_x_rest etc):
        E_i_rest(r_i) = (3Q / 4πε₀·abc) · M_i · r_i_rest

    Kicks (derivations in pic_solver.py):
        Δx'[rad]  = q · E_x_rest · Δs / (β²γ²·mc²)
        Δy'       = same form with M_y
        ΔW[J]     = q · E_z_rest · Δs          (no γ factor)

    In the code's natural units (x [mm], x' [mrad], Δφ [deg], W [MeV]) the
    (mm, mrad) and (m, rad) numerical values coincide; we do the MeV
    conversion explicitly.  z_rest is expressed as γ·(-β·λ/360)·Δφ so the
    longitudinal block couples Δφ → ΔW as a pure 2×2 kick.

    Returns a 6×6 thin-lens matrix; the split-operator tracker applies
    it at the mid-point of each sub-step.
    """
    M_sc = np.eye(6)
    if (current_mA == 0.0 or frequency_MHz <= 0.0
            or beta <= 0.0 or gamma <= 0.0 or mass_MeV <= 0.0
            or sigma_x_mm <= 0.0 or sigma_y_mm <= 0.0 or sigma_phi_deg <= 0.0):
        return M_sc

    # Bunch charge [C].  Q is the charge in ONE bunch — at a FREQ jump
    # (e.g. MEBT 162.5 MHz → SSR1 325 MHz) the cavity sees the bunch every
    # other RF period, so Q must use the bunch repetition frequency, NOT
    # the local cavity frequency.  When bunch_frequency_MHz is None, fall
    # back to frequency_MHz (single-frequency lattice).
    f_bunch = bunch_frequency_MHz if bunch_frequency_MHz else frequency_MHz
    Q = (abs(current_mA) * 1e-3) / (f_bunch * 1e6)
    q_abs = abs(charge_state) * E_CHARGE  # only magnitude matters here

    # RF wavelength [mm] used for σ_φ → σ_z conversion.  σ_φ is in degrees
    # of phase relative to the CURRENT ref.frequency (= frequency_MHz —
    # σ is rescaled at FREQ jumps so this stays consistent).
    wavelength_mm = C_LIGHT / (frequency_MHz * 1e6) * 1000.0
    sigma_z_lab_mm = sigma_phi_deg * beta * wavelength_mm / 360.0
    sigma_z_rest_mm = gamma * sigma_z_lab_mm

    # RMS-equivalent ellipsoid semi-axes, in metres (rest frame)
    a_m = math.sqrt(5.0) * sigma_x_mm   * 1e-3
    b_m = math.sqrt(5.0) * sigma_y_mm   * 1e-3
    c_m = math.sqrt(5.0) * sigma_z_rest_mm * 1e-3
    Mx, My, Mz = _ellipsoid_depolarization_factors(a_m, b_m, c_m)

    ds_m = ds_mm * 1e-3
    mc2_J = mass_MeV * 1e6 * E_CHARGE   # rest energy in joules
    abc = a_m * b_m * c_m

    # Transverse kick coefficient (mrad / mm is numerically rad/m):
    #   Δx' = 3·q·Q·M_x·Δs / (4πε₀·abc·β²γ²·mc²) · x
    k_perp_common = 3.0 * q_abs * Q * ds_m / (
        4.0 * PI * EPSILON_0 * abc * beta * beta * gamma * gamma * mc2_J
    )
    M_sc[1, 0] = k_perp_common * Mx
    M_sc[3, 2] = k_perp_common * My

    # Longitudinal kick: ΔW[MeV] per Δφ[deg]
    #   ΔW[J] = q · E_z_rest · Δs with E_z_rest = 3Q·Mz·z_rest/(4πε₀·abc)
    #   z_rest[m]  = γ · z_lab     = γ · (−β·λ/360) · Δφ[deg]   (λ in m)
    #   ΔW[MeV]     = ΔW[J] / (1e6 · e)
    # Substituting gives a coefficient that's negative (Δφ>0 is behind → loses W).
    wavelength_m = wavelength_mm * 1e-3
    M_sc[5, 4] = -3.0 * q_abs * Q * Mz * gamma * beta * wavelength_m * ds_m / (
        4.0 * PI * EPSILON_0 * abc * 360.0 * 1e6 * E_CHARGE
    )
    return M_sc


def _sc_kick_matrix_2d_dc(
    current_mA: float, charge_state: int, mass_MeV: float,
    beta: float, gamma: float,
    sigma_x_mm: float, sigma_y_mm: float,
    ds_mm: float,
) -> np.ndarray:
    """Thin-lens 2-D analytic SC kick matrix for a continuous (DC) beam.

    Longitudinal block is identity (no SC force along s for an
    unbunched beam).  Transverse kick uses the TraceWin continuous-beam
    formula E_x = I · x / (2πε₀·β·c·a·(a+b)) with semi-axes a, b = 2σ.

    Returns a 6×6 thin-lens matrix.  Matches the per-particle kick that
    ``pic_solver.kick_continuous_2d`` applies to a multi-particle beam
    so the envelope and MP tracks of the same DC beam agree.
    """
    M_sc = np.eye(6)
    if current_mA == 0.0 or beta <= 0.0 or gamma <= 0.0 or mass_MeV <= 0.0:
        return M_sc
    if sigma_x_mm <= 0.0 or sigma_y_mm <= 0.0 or ds_mm <= 0.0:
        return M_sc

    I_A = abs(current_mA) * 1e-3
    q_abs = abs(charge_state) * E_CHARGE
    sigma_x_m = sigma_x_mm * 1e-3
    sigma_y_m = sigma_y_mm * 1e-3
    a_m = 2.0 * sigma_x_m
    b_m = 2.0 * sigma_y_m
    ab_sum = a_m + b_m
    v_m_s = beta * C_LIGHT
    # Uniform elliptic-cylinder SC field (Wiedemann §3.3):
    #   E_x = I · x / (π·ε₀·v·a·(a+b)),  same for y with b.
    factor = I_A / (PI * EPSILON_0 * v_m_s * ab_sum)          # V/m per m
    k_x = factor / a_m
    k_y = factor / b_m
    mc2_J = mass_MeV * 1e6 * E_CHARGE
    ds_m = ds_mm * 1e-3
    pre = q_abs * ds_m / (beta * beta * gamma * mc2_J)
    # x'[rad] = pre · k_x · x[m]; converting positions and angles to mm/mrad
    # preserves the numerical value (1 rad/m ≡ 1 mrad/mm).
    M_sc[1, 0] = pre * k_x
    M_sc[3, 2] = pre * k_y
    return M_sc


# ---------------------------------------------------------------------------
# Envelope solver
# ---------------------------------------------------------------------------

def _fitted_matrix_slice_at(element, ref, ds_mm: float,
                             z_from_mm: float) -> np.ndarray:
    """Call ``element.fitted_matrix_slice`` with a z-cursor when supported.

    M3 envelope-mode surrogate seam: if a surrogate is registered for
    this element (by ``element.name``) and the surrogate's prediction
    is in-scope, the slice goes through the surrogate; on
    :class:`OutOfScopeError` the call falls back to the wrapped RK4
    path so the surrogate can NEVER silently use bad outputs.

    Multi-cell elements (:class:`VaneRFQ`, :class:`RfqCell`) accept
    ``_z_from_mm`` so the bundle matrix is computed at the correct
    element-local z; without it every bundle restarts at z=0 and applies
    the first cell's kick over and over (σ blows up by 10⁵³).  Elements
    that don't take the kwarg (FieldMap1D / FieldMap3D) fall back to the
    unkeyed call, which is correct for them since the field they
    integrate is z-aware via β/γ/ref.s rather than via this argument.
    """
    # M3 hook: route through a registered surrogate if available.
    # Import lazily to avoid a circular import at module load.
    from linac_gen.surrogates import registry as _surr_reg
    from linac_gen.surrogates.base import OutOfScopeError as _OOS
    surr = _surr_reg.get_for_element(element)
    if surr is not None:
        try:
            return surr.fitted_matrix_slice(ref, ds_mm)
        except _OOS:
            pass   # OOD -> fall through to the wrapped RK4 below.

    try:
        return element.fitted_matrix_slice(ref, ds_mm, _z_from_mm=z_from_mm)
    except TypeError:
        return element.fitted_matrix_slice(ref, ds_mm)


def _rescale_sigma_for_freq_jump(sigma: np.ndarray, ratio: float) -> np.ndarray:
    """Rescale Σ across an RF-frequency jump: exact ``Σ → D Σ Dᵀ`` with
    ``D = diag(1, 1, 1, 1, ratio, 1)``, ``ratio = f_new / f_old``.

    The longitudinal coordinate is phase in degrees at the *current* RF
    frequency, ``Δφ = 360·f·Δt``.  A frequency jump preserves the physical
    time deviation ``Δt``, so every particle's ``Δφ`` scales by ``ratio`` —
    hence the ENTIRE phase row and column of Σ scale by ``ratio``:

    * ``σ_φ²``            (``[4,4]``)      → × ``ratio²``  — the size the SC
      kick reads as ``σ_z``; getting this wrong halves ``σ_z`` at a jump;
    * ``⟨Δφ·ΔW⟩``         (``[4,5]/[5,4]``) → × ``ratio``  — longitudinal Twiss;
    * ``⟨Δφ·x⟩ …⟨Δφ·y'⟩`` (``[4,0:4]``)    → × ``ratio``  — the transverse↔
      longitudinal cross-moments.

    For an **uncoupled** beam the cross-moments are exactly zero, so this is
    bit-identical to rescaling the longitudinal 2×2 block alone (``0·ratio =
    0``); the diagonal/longitudinal operations are written verbatim so that
    bit-identity holds at any ``ratio``.  The cross-moment terms matter only
    for a transversely–longitudinally **coupled** beam, where they keep the
    6-D eigenemittances consistent across the jump.  Projected σ / ε / Twiss
    and the space-charge kick — which read only the diagonal RMS sizes — are
    unchanged either way.
    """
    sigma = sigma.copy()
    sigma[4, 0:4] *= ratio         # phase↔(x, x', y, y') cross-moments (row)
    sigma[0:4, 4] *= ratio         # (column)
    sigma[4, 4] *= ratio * ratio   # σ_φ² (SC-critical longitudinal size)
    sigma[4, 5] *= ratio           # phase↔energy (longitudinal Twiss)
    sigma[5, 4] *= ratio
    return sigma


class EnvelopeSolver:
    """RMS envelope tracking with analytical space charge.

    The 6x6 beam sigma matrix is propagated through the lattice using:
        Sigma_new = M @ Sigma_old @ M.T
    with an optional space-charge thin-lens kick per element.

    Parameters
    ----------
    lattice : Lattice
        The lattice to track through.
    ref : ReferenceParticle
        Initial reference particle state. A copy is advanced internally.
    initial : dict
        Must contain: alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z,
        emit_x, emit_y, emit_z.
    current : float
        Beam current in mA (default 0 = space charge off).
    """

    def __init__(self, lattice, ref: ReferenceParticle, initial: dict,
                 current: float = 0.0,
                 progress_callback=None, should_abort=None,
                 record_substeps: bool = False,
                 phase_probe: bool = False,
                 initial_sigma=None,
                 bunch_frequency: float | None = None,
                 sc_factor: float = 1.0,
                 rfq_dc_envelope: bool = False):
        self.lattice = lattice
        self.initial = initial
        self.current = current
        # TraceWin-compatible DC envelope: when True, a continuous beam
        # is NEVER flipped to bunched at an RF bunching element — the
        # 2-D DC space charge is kept through the RFQ, exactly as
        # TraceWin's envelope mode does (its ENV+SC export carries no
        # longitudinal envelope at all: σφ ≡ σ_dp/p ≡ σ_W ≡ 0 through
        # the RFQ).  Default False = HELIX's own model: seed a uniform
        # 103.9° bunch at the first RF element and evolve it — more
        # physical, but NOT comparable to a TraceWin ENV+SC export
        # (measured on the PXIE 2-solenoid LEBT+RFQ line at 5 mA:
        # bunched 27-64 % σ deviation inside the RFQ vs the TW export,
        # DC-through-RFQ 4-6 % full-line — see the manual).  Only
        # meaningful together with ``initial["continuous"]=True``.
        self._rfq_dc_envelope = bool(rfq_dc_envelope)
        # Phase probe: when True, every matrix the solver applies to Σ
        # is also accumulated into per-element composite maps (SC-
        # inclusive AND bare) plus a per-slice history — the exact
        # tangent map of this run, at the σ/energy it used.  Purely
        # additive: σ propagation itself is untouched (bit-identical
        # with the probe on or off).
        self._phase_probe = bool(phase_probe)
        self._probe_dep = None      # running per-element accumulators
        self._probe_bare = None
        # Optional full 6×6 entrance covariance.  Overrides the Twiss-
        # dict Σ construction — required to seed coupled states (e.g. a
        # recorded mid-lattice Σ with solenoid x–y cross terms, which a
        # block-diagonal Twiss dict cannot represent).  Default None ⇒
        # legacy behaviour, bit-identical.
        self._initial_sigma = (None if initial_sigma is None
                               else np.asarray(initial_sigma, float).copy())
        # Optional bunch repetition frequency override [MHz].  A period-
        # scoped rerun that starts after a FREQ jump would otherwise
        # derive _bunch_freq from the local (post-jump) ref and halve
        # the bunch charge Q = I/f_bunch.  Default None ⇒ ref.frequency
        # (legacy).
        self._bunch_freq_override = (None if bunch_frequency is None
                                     else float(bunch_frequency))
        # Work on a copy of the reference particle so the original is not mutated
        self._ref = ref.copy()
        # ``TrackState`` carries SET_*/ADJUST_* run-time mutations.  Bound
        # to the working copy of ref so SetBeamEnergy / SetBeamE0P0 flow
        # through.  Imported lazily to avoid a circular import at module
        # load time (envelope.py is one of the heaviest modules).
        from linac_gen.core.track_state import TrackState
        self.track_state = TrackState(ref=self._ref)
        # Progress / cancellation hooks — see Tracker for the contract.
        self._progress_callback = progress_callback
        self._should_abort = should_abort
        self.aborted = False
        # When True, σ is recorded at every internal slice / bundle boundary
        # inside drifts and field maps — produces a smooth σ(s) curve at
        # the cost of more records.  Default is off for back-compat.
        self._record_substeps = record_substeps
        self._results = None  # set in run()
        # First-moment (centroid) state — seeded in run() from
        # initial["centroid"]; None until then so helpers no-op safely.
        self._c = None
        # SPACE_CHARGE_COMP factor: 1.0 = full SC, 0.0 = neutralized.
        # Mirrors the Tracker's _sc_factor.  Updated by passive elements
        # carrying a .factor attribute (SpaceChargeComp) as the envelope
        # walks the lattice.  Applied as a multiplier on the SC current
        # inside _sc_matrix.  The constructor value seeds mid-lattice
        # (period-scoped) reruns with the compensation state captured at
        # the period entry; default 1.0 = legacy full SC.
        self._sc_factor = float(sc_factor)
        # DC / continuous-beam flag.  True → skip longitudinal σ setup,
        # use 2-D analytic DC SC kick, auto-flip to bunched at the first
        # RF element (mirrors the Tracker's behaviour).  User signals it
        # via ``initial["continuous"]=True``; ``initial["emit_z"]`` etc.
        # are then allowed to be zero / missing.
        self._continuous = bool(initial.get("continuous", False))
        # Bunch repetition frequency [MHz] — fixed by the upstream RFQ /
        # buncher and constant downstream of it.  Used by the 3-D SC kick
        # for the bunch-charge Q = I / f_bunch (independent of cavity
        # frequency, which can differ at FREQ jumps).  Initialised from
        # the input ref.frequency unless explicitly overridden (period-
        # scoped reruns starting after a FREQ jump MUST pass the
        # original bunch frequency).
        self._bunch_freq = (self._bunch_freq_override
                            if self._bunch_freq_override is not None
                            else float(self._ref.frequency))

    def run(self) -> EnvelopeResults:
        """Track envelope through lattice and return results."""
        if self._initial_sigma is not None:
            # Full entrance covariance supplied — required for coupled
            # seeds (solenoid cross terms) that a Twiss dict cannot
            # express.  The DC-transition longitudinal seeding below is
            # skipped implicitly (caller owns the full Σ).
            if self._initial_sigma.shape != (6, 6):
                raise ValueError("initial_sigma must be a (6, 6) array")
            sigma = self._initial_sigma.copy()
        else:
            # Initialise sigma matrix.  For continuous beams the longitudinal
            # 2×2 block of Σ is set to zero (no bunch structure yet); the
            # user's ``initial`` may omit or zero out α_z, β_z, ε_z.
            emit_z_init = float(self.initial.get("emit_z", 0.0) or 0.0)
            alpha_z_init = float(self.initial.get("alpha_z", 0.0) or 0.0)
            beta_z_init = float(self.initial.get("beta_z", 1.0) or 1.0)
            if self._continuous:
                emit_z_init = 0.0
                alpha_z_init = 0.0
                beta_z_init = 1.0   # just to keep _build_sigma_matrix happy
            sigma = _build_sigma_matrix(
                self.initial["alpha_x"], self.initial["beta_x"], self.initial["emit_x"],
                self.initial["alpha_y"], self.initial["beta_y"], self.initial["emit_y"],
                alpha_z_init, beta_z_init, emit_z_init,
                disp_x=float(self.initial.get("disp_x", 0.0) or 0.0),
                disp_xp=float(self.initial.get("disp_xp", 0.0) or 0.0),
                disp_y=float(self.initial.get("disp_y", 0.0) or 0.0),
                disp_yp=float(self.initial.get("disp_yp", 0.0) or 0.0),
            )

        results = EnvelopeResults()
        results.current_mA = float(self.current)
        results.continuous = bool(self._continuous)
        results.mass_mev = float(getattr(getattr(self._ref, "species", None), "mass", 0.0))
        self._results = results     # so sub-step helpers can append
        # First-moment (centroid) seed: 6-vector relative to the
        # reference, from ``initial["centroid"]`` (mm/mrad/deg/MeV) —
        # zeros for an on-axis launch.  Propagated alongside Σ at every
        # matrix site via _probe_push's choke point.
        seed_c = self.initial.get("centroid")
        self._c = (np.zeros(6) if seed_c is None
                   else np.asarray(seed_c, dtype=float).reshape(6).copy())
        # Record initial state
        self._record(results, sigma, self._ref, element_name="INPUT")

        elements = self.lattice.elements
        n_el = len(elements)
        for i, element in enumerate(elements):
            if self._should_abort is not None and self._should_abort():
                self.aborted = True
                return results
            # DC → bunched transition.  When the envelope first sees an
            # RF element we flip to bunched and seed the longitudinal
            # block of Σ with a "uniform over one RF period" equivalent
            # so the cavity's longitudinal kick receives a physical
            # input (σ_φ = 180°/√3 for a uniform distribution; σ_W uses
            # whatever the DC energy spread was, encoded via the
            # ``dc_energy_spread_keV`` → initial["dc_energy_spread_MeV"]
            # conversion below).
            if self._continuous and not self._rfq_dc_envelope:
                from linac_gen.tracking.tracker import _is_rf_bunching_element
                if _is_rf_bunching_element(element):
                    self._continuous = False
                    # σ_φ of a uniform dist on [-180°, +180°] is 180°/√3 ≈ 103.92°
                    sigma[4, 4] = (180.0 / math.sqrt(3.0)) ** 2
                    sigma[4, 5] = sigma[5, 4] = 0.0
                    sw_keV = float(self.initial.get("dc_energy_spread_keV", 0.0))
                    sw_MeV = sw_keV * 1e-3
                    sigma[5, 5] = sw_MeV * sw_MeV
                    # A DC beam is phase-uniform: its mean phase is 0 by
                    # construction (MP fills uniform [-180°, 180°)), so a
                    # seeded c[4] is meaningless upstream of bunching —
                    # zero it at the same boundary Σ is re-seeded at.
                    # The mean energy offset c[5] stays (physical).
                    if self._c is not None:
                        self._c[4] = 0.0
            if self._phase_probe:
                # Fresh per-element accumulators; every matrix applied
                # to Σ inside _propagate_element left-multiplies them
                # via _probe_push.
                self._probe_dep = np.eye(6)
                self._probe_bare = np.eye(6)
                self._probe_elem_i = i
            sigma = self._propagate_element(element, sigma)
            self._record(results, sigma, self._ref,
                         element_name=getattr(element, "name", "?"))
            # Row index of this element's exit — substep helpers may
            # have appended interior rows before this one.
            results.element_exit_idx.append(len(results.s) - 1)
            if self._phase_probe:
                results.element_maps_dep.append(self._probe_dep)
                results.element_maps_bare.append(self._probe_bare)
            if self._progress_callback is not None:
                try:
                    self._progress_callback(self._ref.s, i + 1, n_el)
                except Exception:
                    pass

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_push(self, M: np.ndarray, *, is_sc: bool = False) -> None:
        """Accumulate one applied map into the phase probe.

        Called at EVERY site that multiplies Σ by a matrix (Strang
        half-maps, SC kicks, freq-jump D, edges, thin kicks…) — the
        probe therefore captures the run's exact tangent map, slice by
        slice, at the σ/energy the run used.  SC kicks are skipped in
        the bare accumulator (identical slices, SC → identity — never a
        separate I=0 run, which would use different FieldMap slicing).
        Foil diffusion is additive, not a matrix, and must NOT call
        this (its mean map is the identity).

        This method is ALSO the centroid choke point: because every
        matrix applied to Σ flows through here, the first moment gets
        the identical map (``c → M c``) in one place — including the
        freq-jump D (c[4] scales with the coordinate-unit change).  SC
        kicks are skipped: linear space charge is centred on the beam
        and does not move the centroid (TraceWin's approximation too).
        """
        c = getattr(self, "_c", None)
        if c is not None and not is_sc:
            self._c = M @ c
        if not self._phase_probe or self._probe_dep is None:
            return
        self._probe_dep = M @ self._probe_dep
        if not is_sc:
            self._probe_bare = M @ self._probe_bare
        res = self._results
        if res is not None:
            res.probe_elem_idx.append(int(getattr(self, "_probe_elem_i", -1)))
            res.probe_is_sc.append(bool(is_sc))
            res.probe_s.append(float(self._ref.s))
            res.probe_M.append(np.array(M, dtype=float, copy=True))

    def _record(self, results: EnvelopeResults, sigma: np.ndarray,
                ref: ReferenceParticle,
                element_name: str = "INPUT") -> None:
        """Append current state to results.

        ``element_name`` tags the entry with the element whose exit this
        record corresponds to (``"INPUT"`` for the pre-lattice record).
        """
        twiss_x = _sigma_to_twiss(sigma, "x")
        twiss_y = _sigma_to_twiss(sigma, "y")
        twiss_z = _sigma_to_twiss(sigma, "z")

        results.s.append(ref.s)
        results.sigma_x.append(math.sqrt(max(sigma[0, 0], 0.0)))
        results.sigma_y.append(math.sqrt(max(sigma[2, 2], 0.0)))
        results.sigma_phi.append(math.sqrt(max(sigma[4, 4], 0.0)))
        results.sigma_w.append(math.sqrt(max(sigma[5, 5], 0.0)))
        results.sigma_matrix.append(sigma.copy())
        results.element_names.append(element_name)
        results.emit_x.append(twiss_x["emit"])
        results.emit_y.append(twiss_y["emit"])
        results.emit_z.append(twiss_z["emit"])
        # Re-express longitudinal emittance in mm.mrad for plot consistency.
        from linac_gen.diagnostics.recorder import _convert_emit_z_to_mmmrad
        results.emit_z_mmmrad.append(
            _convert_emit_z_to_mmmrad(twiss_z["emit"], ref)
        )
        # 4-D coupling-invariant transverse emittance (TraceWin εt).
        # In an uncoupled lattice this equals emit_x · emit_y; under
        # solenoid coupling the 2-D projections wobble while √det(σ_4D)
        # remains constant.
        det4 = float(np.linalg.det(sigma[:4, :4]))
        results.emit_4d.append(math.sqrt(max(det4, 0.0)))
        # 6-D eigenemittances (Balandin) — invariants of any symplectic
        # transport.  Cheap to compute (3 trace evaluations) so always on.
        from linac_gen.diagnostics.eigenemittance import eigenemittances
        ee1, ee2, ee3 = eigenemittances(sigma)
        results.emit_e1.append(ee1)
        results.emit_e2.append(ee2)
        results.emit_e3.append(ee3)
        results.alpha_x.append(twiss_x["alpha"])
        results.beta_x.append(twiss_x["beta"])
        results.alpha_y.append(twiss_y["alpha"])
        results.beta_y.append(twiss_y["beta"])
        results.alpha_z.append(twiss_z["alpha"])
        results.beta_z.append(twiss_z["beta"])
        results.ref_w_kin.append(ref.w_kin)
        results.ref_beta.append(ref.beta)
        results.ref_gamma.append(ref.gamma)
        results.ref_frequency.append(float(getattr(ref, "frequency", 0.0)))
        c = getattr(self, "_c", None)
        # No centroid state = shell solvers (backtrack) calling _record
        # without run(): append NOTHING.  A short/empty list is the
        # honest "carries no first moment" signal consumers key on
        # (len(centroid) == len(s) gates); a zeros row would masquerade
        # as a genuinely on-axis orbit.
        if c is not None:
            results.centroid.append(c.copy())

    def _advance_ref(self, element) -> None:
        """Advance the reference particle past one element."""
        from linac_gen.elements.base import FieldMapElement as FME, ThinKickElement as TKE
        if isinstance(element, FME):
            element.advance_ref(self._ref)
        elif isinstance(element, TKE):
            self._ref.s += element.length  # 0 for thin lens
            element.advance_ref(self._ref)
        else:
            self._ref.s += element.length
            if element.length > 0:
                self._ref.phi_s += (
                    360.0 * element.length
                    / (self._ref.beta * self._ref.wavelength)
                )

    def _propagate_element(self, element, sigma: np.ndarray) -> np.ndarray:
        """Propagate sigma matrix through one element, with optional SC
        kick + per-element misalignment wrap (M8).

        ``tilt_deg`` rotates the (x, x', y, y') Σ-block around the
        longitudinal axis -- mirrors :func:`linac_gen.tracking.tracker.
        Tracker._track_element` (lines ~196-254) so envelope mode gets
        the same misalignment treatment as the MP tracker.

        ``dx`` / ``dy`` are no-ops for Σ (invariant under rigid
        translation about the centroid) but NOT for the first moment:
        the centroid enters the element frame shifted, so a focusing
        element feeds the offset down as an effective dipole kick —
        net effect c_out = M c_in + (I − M)·Δ.  The wrapper below
        replicates the MP tracker's exact sequence
        (Tracker._track_element: translate in → rotate in → element →
        rotate out → translate out), so envelope-vs-MP centroid parity
        holds for misaligned linear elements.

        Per-particle (x, x') rotation in the MP tracker uses the same
        :func:`linac_gen.elements.mixins.Misalignment.tilt_rotation_matrix`
        coefficients, so env-vs-MP σ_x parity holds for small linear
        elements.
        """
        # Mirror the MP tracker's exclusions: PassiveElements reuse
        # dx/dy as geometry (Aperture half-widths — tracker zeroes them)
        # and Multipole self-handles its misalignment inside the kick —
        # wrapping either would double-apply or corrupt.
        from linac_gen.elements.multipole import Multipole
        _skip_wrap = isinstance(element, (PassiveElement, Multipole))
        tilt = 0.0 if _skip_wrap else \
            float(getattr(element, "tilt_deg", 0.0) or 0.0)
        dx = 0.0 if _skip_wrap else \
            float(getattr(element, "dx", 0.0) or 0.0)
        dy = 0.0 if _skip_wrap else \
            float(getattr(element, "dy", 0.0) or 0.0)
        has_tilt = abs(tilt) > 1e-12
        has_shift = (abs(dx) > 1e-15 or abs(dy) > 1e-15) \
            and getattr(self, "_c", None) is not None
        if has_shift:
            # Translate the centroid into the element frame (Σ invariant).
            self._c[0] -= dx
            self._c[2] -= dy
        if has_tilt:
            from linac_gen.elements.mixins import Misalignment
            # ENTRY rotation = tilt_rotation_matrix(+tilt), matching the
            # MP tracker's per-particle formula (tracker.py:330-341:
            # x_new = c·x + s·y, y_new = −s·x + c·y with θ = +tilt).
            # The previous R(−tilt)/R(+tilt) pair modelled −tilt — a
            # latent sign bug invisible to symmetric Σ's and to
            # sign-symmetric Monte-Carlo tilt distributions, but fatal
            # for the centroid (and for coupled Σ seeds).
            R_in = Misalignment.tilt_rotation_matrix(tilt)
            sigma = R_in @ sigma @ R_in.T
            if getattr(self, "_c", None) is not None:
                self._c = R_in @ self._c
        sigma = self._propagate_element_aligned(element, sigma)
        if has_tilt:
            from linac_gen.elements.mixins import Misalignment
            R_out = Misalignment.tilt_rotation_matrix(-tilt)
            sigma = R_out @ sigma @ R_out.T
            if getattr(self, "_c", None) is not None:
                self._c = R_out @ self._c
        if has_shift:
            self._c[0] += dx
            self._c[2] += dy
        return sigma

    def _propagate_element_aligned(self, element, sigma: np.ndarray) -> np.ndarray:
        """Propagate sigma matrix through one element, with optional SC kick."""
        # LatticeCommand: mutate run-time state (energy / phase shift /
        # sync mode), then no-op on σ.  Imported here to keep this module's
        # top-level imports light.  A command that changes the machine RF
        # frequency (``Freq``) redefines the Δφ coordinate's degrees, so σ
        # gets the exact freq-jump rescale at the card position — the same
        # ``D Σ Dᵀ`` the per-element jump applies (which then no-ops).
        from linac_gen.elements.lattice_commands import LatticeCommand
        if isinstance(element, LatticeCommand):
            f_old = float(getattr(self._ref, "frequency", 0.0) or 0.0)
            element.apply_command(self.track_state)
            f_new = float(getattr(self._ref, "frequency", 0.0) or 0.0)
            if f_old > 0 and f_new > 0 and f_new != f_old:
                ratio = f_new / f_old
                sigma = _rescale_sigma_for_freq_jump(sigma, ratio)
                # The rescale is exactly D Σ Dᵀ with
                # D = diag(1,1,1,1,ratio,1) — a coordinate-unit change that
                # belongs in BOTH the depressed and bare probe maps AND in
                # the centroid (c[4] is in degrees of the machine clock).
                # Push unconditionally: the probe body self-guards, the
                # centroid choke point must always see D.
                D = np.eye(6)
                D[4, 4] = ratio
                self._probe_push(D)
            return sigma

        # For passive elements (apertures, markers, SC comp): no optics change,
        # but SpaceChargeComp carries a .factor attribute that toggles SC
        # compensation — mirror the MP tracker by updating _sc_factor here.
        if isinstance(element, PassiveElement):
            if hasattr(element, "factor"):
                self._sc_factor = 1.0 - float(element.factor)
            # Edge is the lone PassiveElement carrying a real linear map
            # (zero-length pole-face rotation around bends).  Apply it to σ
            # so RBENDs get their full edge+body+edge cumulative.
            if hasattr(element, "transfer_matrix"):
                M = element.transfer_matrix(self._ref)
                self._probe_push(M)
                return M @ sigma @ M.T
            return sigma

        # Stripper / scattering foil: the MEAN map is the identity
        # (kick_matrix), but the foil is NOT a no-op at second-moment
        # level — a zero-mean stochastic kick is a diffusion process.
        # Exact moment update for kick moments D:
        #     Σ ← M Σ Mᵀ + D,   M = I,
        #     D = diag(0, θ₀², 0, θ₀², 0, σ_E²)
        # (θ₀ = Highland rms plane angle [mrad], σ_E = Bohr straggling
        # [MeV] — from Foil.envelope_diffusion), plus the Bethe-Bloch
        # mean loss applied to the reference energy.  This mirrors the
        # MP tracker, where apply_kick draws exactly these moments per
        # particle; there the mean loss lives in the particles' dw
        # (ref unchanged), while a centred-Σ description must carry it
        # on the reference — equivalent to first order (⟨ΔE⟩/W ~ 1e-6
        # for the PIP-II BTL stripper).  Imported lazily like the other
        # element types in this method.
        from linac_gen.elements.foil import Foil
        if isinstance(element, Foil):
            D, dE_mean = element.envelope_diffusion(self._ref)
            sigma = sigma + D
            self._advance_ref(element)   # thin element: s += 0
            if dE_mean > 0.0:
                self._ref.w_kin = self._ref.w_kin - dE_mean
            return sigma

        # FREQ-jump handling: if this element is an RF cavity whose
        # frequency differs from ref.frequency, rescale σ so σ_φ stays in
        # degrees-at-ref.frequency (see _rescale_sigma_for_freq_jump).
        # Without it a 162.5 → 325 MHz boundary leaves σ_φ at the old freq
        # while ref.frequency / cavity matrices / SC kicks run at the new
        # freq, halving σ_z in the SC kick (~1.2 MeV loss over SSR1 cm1).
        # Use the element's EFFECTIVE frequency (incl. frequency_offset
        # cavity errors) — particle tracking rescales by the effective
        # value, so the envelope must too or error studies diverge.
        elem_freq = (getattr(element, "effective_frequency", 0.0)
                     or getattr(element, "frequency", 0.0) or 0.0)
        if (elem_freq > 0
                and elem_freq != self._ref.frequency
                and getattr(element, "_has_electric_channel", lambda: True)()):
            ratio = elem_freq / self._ref.frequency
            sigma = _rescale_sigma_for_freq_jump(sigma, ratio)
            # The rescale is exactly D Σ Dᵀ with D = diag(1,1,1,1,ratio,1)
            # — a coordinate-unit change for the probe maps AND the
            # centroid's Δφ component.  Unconditional push: the probe
            # body self-guards, the centroid choke point must see D.
            D = np.eye(6)
            D[4, 4] = ratio
            self._probe_push(D)
            self._ref.frequency = elem_freq

        # Thin correctors (Steerer, Multipole): their kick_matrix drops
        # the constant dipole component (Steerer's is identity, Multipole
        # omits k0L) and any nonlinearity — pure first-moment effects.
        # Recover them EXACTLY by evaluating the element's own apply_kick
        # on a one-particle "beam" at the centroid and adding the
        # non-linear remainder (full kick minus the kick_matrix's linear
        # prediction); the linear part then arrives via the normal M·c
        # choke point, so nothing is double-counted.  EXACTNESS
        # PRECONDITION: apply_kick must change ANGLES only, with the
        # kick_matrix's non-identity entries confined to angle-rows ×
        # position-columns — then K = km−I is nilpotent with
        # K@Δfull = 0 and the composition error K@Δfull − K²c vanishes
        # identically.  True for Steerer and Multipole; a future
        # corrector whose kick touches positions or ΔW silently
        # degrades this to first order — extend the recipe, don't
        # just add the class below.  Multipole also self-handles
        # dx/dy/tilt inside apply_kick — which is why the misalignment
        # wrapper excludes it.
        from linac_gen.elements.multipole import Multipole
        from linac_gen.elements.steerer import Steerer
        if isinstance(element, (Steerer, Multipole)) and \
                getattr(self, "_c", None) is not None:
            from linac_gen.core.beam import Beam
            mini = Beam(ref=self._ref, n_particles=1, current=0.0)
            mini.particles[0, :] = self._c
            element.apply_kick(mini)
            full_delta = mini.particles[0] - self._c
            km = element.kick_matrix(self._ref)
            lin_delta = (km - np.eye(6)) @ self._c
            self._c = self._c + (full_delta - lin_delta)

        # FieldMapElements are stateful (slice cursor, resolved βg≤0
        # geometry, SET_SYNC_PHASE cache).  The SC and substep branches
        # reset before their own walks; the pure-linear I=0 path below uses
        # the full-element fitted matrix directly, so reset here too — a
        # reused lattice object otherwise replays the previous run's cursor
        # (second full call starts at z=length: no gap ever fires).
        if isinstance(element, FieldMapElement):
            element.reset_run_state()

        # NOTE: the full-element 6x6 matrix is computed lazily on the
        # branches that consume it (pure-linear I=0 path; thin-element
        # branch inside _propagate_with_sc).  The SC and substep walks
        # rebuild transport from half-maps / sub-slice Jacobians, so
        # requesting the full matrix up front spent a surrogate MLP
        # query (plus its scope check) that was then discarded.

        # --- substep recording for FieldMap without SC -------------------
        # When record_substeps is on and current=0, mirror the SC loop but
        # skip the SC kick so σ(s) is captured at every bundle boundary.
        if (self.current == 0.0
                and (self._record_substeps
                     or getattr(element, "interior_markers", None))
                and isinstance(element, FieldMapElement)
                and element.length > 0):
            # interior_markers (SHIFT_IN_FIELD_MAP diagnostics inside a
            # SUPERPOSE container) need the sub-stepped walk so a row can
            # be recorded AT the marker even when substep recording is
            # otherwise off.
            sigma = self._propagate_fieldmap_substeps_noSC(element, sigma)
            return sigma

        # --- substep recording for other transfer-map elements -----------
        # Solenoids and similar TransferMapElement subclasses have a
        # length-parameterised matrix (transfer_matrix(ref, ds=...)).
        # Slice them into n_steps so coupling-driven ε_x / ε_y excursions
        # inside the body are visible alongside the MP curves.
        if (self.current == 0.0
                and self._record_substeps
                and isinstance(element, TransferMapElement)
                and not isinstance(element, FieldMapElement)
                and element.length > 0
                and hasattr(element, "transfer_matrix")):
            sigma = self._propagate_tme_substeps_noSC(element, sigma)
            return sigma

        if self.current == 0.0:
            # Pure linear propagation (no SC)
            M = get_element_matrix(element, self._ref)
            self._probe_push(M)
            sigma = M @ sigma @ M.T
            self._advance_ref(element)
        else:
            # Split-operator: half external map, SC kick, half external map
            sigma = self._propagate_with_sc(element, sigma)
            # For FieldMap with non-zero length, _propagate_with_sc has
            # advanced self._ref inline (bundle-by-bundle) so SC kicks see
            # the locally-correct β.  For other paths it has not, so
            # advance here.
            is_fieldmap_sc_inline = (
                isinstance(element, FieldMapElement)
                and element.length > 0
            )
            if not is_fieldmap_sc_inline:
                self._advance_ref(element)
        return sigma

    def _propagate_tme_substeps_noSC(self, element, sigma: np.ndarray) -> np.ndarray:
        """Sub-step a TransferMapElement (e.g. Solenoid) when record_substeps
        is on and SC is off, recording σ at each slice boundary.  Without
        this, the solenoid's full 4×4 coupled rotation is applied in one
        matrix multiply and the transient ε_x / ε_y peaks inside the body
        are not sampled.
        """
        n_native = max(int(getattr(element, "n_steps", 1)), 1)
        # Target ~50/m record density to make σ(s) curves comparable to
        # TraceWin envelope output. Honour element-native n_steps when it
        # already exceeds the target (e.g. dipoles/quads with n_steps=5).
        target_n = max(1, int(np.ceil(element.length * 50.0 / 1000.0)))
        n = max(n_native, target_n)
        ds_mm = element.length / n
        for _ in range(n):
            M_slice = element.transfer_matrix(self._ref, ds=ds_mm)
            self._probe_push(M_slice)
            sigma = M_slice @ sigma @ M_slice.T
            # Advance ref over this slice using the same length recipe as
            # _advance_ref but parameterised on ds_mm.
            self._ref.s += ds_mm
            if ds_mm > 0:
                self._ref.phi_s += (
                    360.0 * ds_mm
                    / (self._ref.beta * self._ref.wavelength)
                )
            self._record(self._results, sigma, self._ref,
                         element_name=getattr(element, "name", "?"))
        return sigma

    def _propagate_fieldmap_substeps_noSC(self, element, sigma: np.ndarray) -> np.ndarray:
        """Sub-step through a FieldMap element with no SC, recording σ at
        each bundle boundary so σ(s) curves match TraceWin envelope
        resolution.  Uses the same slicing the SC path uses so matrices
        compose consistently.
        """
        # Canonical cadence — shared with the MP tracker (and the
        # backtracker) so matrices, SC kick planes and diagnostics
        # compose consistently across modes.  NOTE: this also brings the
        # RfqCell n_steps floor to the envelope path, which the old
        # inline copy was missing (violating this method's own "same
        # slicing as the SC path" contract for RFQ cells).
        from linac_gen.tracking.tracker import field_map_step_counts
        n_int, n_sc, ds_mm, sc_every, n_bundles = field_map_step_counts(
            self.lattice, element)
        element.reset_run_state()
        z_cursor = 0.0
        _interior = sorted(getattr(element, "interior_markers", []) or [],
                           key=lambda t: t[0])
        _mk_i = 0

        def _adv_transport(step):
            # Advance *step* mm, split at interior markers so each
            # SHIFT_IN_FIELD_MAP row carries the exact s = s_entry + dz
            # and the Σ AT that z (matrix composition makes the split
            # mathematically identical; the no-marker path is one
            # unsplit advance of exactly *step*, bit-identical to the
            # pre-marker code).  Pass z_cursor so multi-cell elements
            # (e.g. VaneRFQ) compute the matrix at the correct
            # element-local z; without this, every bundle restarts from
            # z=0 and applies cell[0] kicks repeatedly, blowing σ up
            # to 10⁵³.
            nonlocal sigma, z_cursor, _mk_i
            remaining = step
            while (_mk_i < len(_interior)
                   and _interior[_mk_i][0] <= z_cursor + remaining + 1e-9):
                piece = _interior[_mk_i][0] - z_cursor
                if piece > 1e-9:
                    M = _fitted_matrix_slice_at(element, self._ref,
                                                piece, z_cursor)
                    self._probe_push(M)
                    sigma = M @ sigma @ M.T
                    element.advance_ref_over(self._ref, z_cursor,
                                             z_cursor + piece)
                    z_cursor += piece
                    remaining -= piece
                self._record(self._results, sigma, self._ref,
                             element_name=_interior[_mk_i][1].name)
                _mk_i += 1
            if remaining > 1e-9:
                M = _fitted_matrix_slice_at(element, self._ref,
                                            remaining, z_cursor)
                self._probe_push(M)
                sigma = M @ sigma @ M.T
                element.advance_ref_over(self._ref, z_cursor,
                                         z_cursor + remaining)
                z_cursor += remaining

        for _ in range(n_bundles):
            _adv_transport(sc_every * ds_mm)
            if self._record_substeps:
                self._record(self._results, sigma, self._ref,
                             element_name=getattr(element, "name", "?"))
        leftover = n_int - n_bundles * sc_every
        if leftover > 0:
            _adv_transport(leftover * ds_mm)
        # Same rationale as FieldMap.advance_ref: only RF cavities should
        # carry their frequency over.  Magnetic-only maps (solenoids) hold
        # the parser default frequency as inert metadata; clobbering ref
        # would silently change downstream wavelength.
        if (getattr(element, "frequency", 0.0) > 0
                and element.frequency != self._ref.frequency
                and getattr(element, "_has_electric_channel", lambda: True)()):
            self._ref.frequency = element.frequency
        return sigma

    def _sc_matrix(self, sigma: np.ndarray, ds_mm: float) -> np.ndarray:
        """Pick the right SC kick matrix for the current beam type.

        - ``self._continuous == True`` → 2-D DC kick (longitudinal block
          is identity; σ_x, σ_y read from the current Σ).
        - Bunched → 3-D uniform-ellipsoid Ferrario kick.

        Applies ``self._sc_factor`` to the current so SPACE_CHARGE_COMP
        neutralization regions correctly suppress the kick.
        """
        sx = math.sqrt(max(sigma[0, 0], 0.0))
        sy = math.sqrt(max(sigma[2, 2], 0.0))
        I_eff = self.current * self._sc_factor
        if I_eff == 0.0:
            return np.eye(6)
        if self._continuous:
            return _sc_kick_matrix_2d_dc(
                current_mA=I_eff,
                charge_state=self._ref.species.charge,
                mass_MeV=self._ref.species.mass,
                beta=self._ref.beta, gamma=self._ref.gamma,
                sigma_x_mm=sx, sigma_y_mm=sy, ds_mm=ds_mm,
            )
        sphi = math.sqrt(max(sigma[4, 4], 0.0))
        return _sc_kick_matrix_3d(
            current_mA=I_eff,
            charge_state=self._ref.species.charge,
            mass_MeV=self._ref.species.mass,
            beta=self._ref.beta, gamma=self._ref.gamma,
            frequency_MHz=self._ref.frequency,
            sigma_x_mm=sx, sigma_y_mm=sy, sigma_phi_deg=sphi,
            ds_mm=ds_mm,
            bunch_frequency_MHz=self._bunch_freq,
        )

    def _propagate_with_sc(self, element,
                           sigma: np.ndarray) -> np.ndarray:
        """Propagate sigma with SC using sub-stepped split-operator.

        Matches the multi-particle tracker: n_steps sub-steps per element,
        each with half-map / SC-kick / half-map.  The full-element matrix
        is fetched only on the thin-element branch that actually applies
        it (the extended-element branches walk half-maps / sub-slices).
        """
        total_length = element.length  # mm

        if isinstance(element, TransferMapElement) and total_length > 0:
            # For drifts / transfer-map elements, use step_config's SC cadence
            # (drifts default to element.n_steps=1, which makes envelope apply
            # just one SC kick per drift element — much coarser than the MP
            # tracker's step_config-based cadence, producing tighter waists
            # through long SC-active drifts).
            cfg = getattr(self.lattice, "step_config", None)
            if cfg is not None:
                n_steps = cfg.sc_steps_for_length_mm(total_length)
            else:
                n_steps = max(element.n_steps, 1)
            ds = total_length / n_steps  # mm per sub-step

            for istep in range(n_steps):
                # Half-step map
                M_half = element.transfer_matrix(self._ref, ds=ds / 2.0)
                self._probe_push(M_half)
                sigma = M_half @ sigma @ M_half.T
                # SC kick at midpoint — 2-D DC or 3-D Ferrario depending
                # on the beam state.
                M_sc = self._sc_matrix(sigma, ds_mm=ds)
                self._probe_push(M_sc, is_sc=True)
                sigma = M_sc @ sigma @ M_sc.T
                # Second half-step
                self._probe_push(M_half)
                sigma = M_half @ sigma @ M_half.T
                # Record σ at step boundary when substeps enabled.  Ref
                # doesn't advance inside this loop; we temporarily bump it
                # so the recorded s reflects the actual sub-step position.
                if self._record_substeps and self._results is not None:
                    saved_s = self._ref.s
                    self._ref.s = saved_s + (istep + 1) * ds
                    self._record(self._results, sigma, self._ref,
                                 element_name=getattr(element, "name", "?"))
                    self._ref.s = saved_s

        elif isinstance(element, FieldMapElement) and total_length > 0:
            # Mirror the TransferMapElement Strang-split but use a numerical
            # Jacobian over a sub-slice of the field map instead of a
            # closed-form transfer matrix.
            cfg = getattr(self.lattice, "step_config", None)
            if cfg is not None:
                n_int = cfg.integration_steps_for_length_mm(total_length)
                n_sc  = cfg.sc_steps_for_length_mm(total_length)
            else:
                n_int = max(element.n_steps, 1)
                n_sc  = n_int

            # Auto-refine magnetic-only 1-D field maps (solenoids).  At
            # strong B + low β the first-order thin-slice integrator used
            # inside track_rk4 (via fitted_matrix_slice) needs ~5000 steps
            # per metre to match TraceWin — the default 100/m gives ~10 %
            # transverse-matrix error that compounds across multiple
            # cryomodule solenoids to a 10 % σ_x overshoot at HWR exit.
            # RF cavities (any element with an E channel) don't need this.
            from linac_gen.tracking.tracker import _is_magnetic_only_1d_fieldmap
            if _is_magnetic_only_1d_fieldmap(element):
                min_int = int((5000.0 * total_length * 1e-3) + 0.5)
                if min_int > n_int:
                    n_int = min_int
                    # Leave n_sc at its configured cadence — sc_every grows
                    # as n_int // n_sc below, which is what we want.  An
                    # earlier version forced n_sc up to n_int here; that
                    # collapsed sc_every to 1 and applied ~100× too many SC
                    # kicks inside each solenoid, blowing up σ and ε with
                    # SC on even though the no-SC case looked fine.
            # Cavity floor: parser-assigned n_steps matches the field-map
            # grid; without this, PARTRAN_STEP=100/m drops cavity transit-
            # time integrals.  See tracker._track_field_map.
            from linac_gen.elements.field_map import FieldMap as _FM1D
            from linac_gen.elements.field_map_3d import FieldMap3D as _FM3D
            from linac_gen.elements.superposed_field_map import (
                SuperposedFieldMap as _SFM,
            )
            if isinstance(element, (_FM1D, _FM3D, _SFM)) \
                    and element.n_steps > n_int:
                n_int = element.n_steps

            ds_mm    = total_length / n_int      # mm per integration sub-step
            sc_every = max(1, n_int // n_sc)     # integration steps per SC kick
            n_bundles = n_int // sc_every        # number of full SC bundles
            # Reset per-run integrator state before the clean pass.
            element.reset_run_state()
            # z-cursor within the element (0 = entrance, total_length = exit).
            z_cursor = 0.0

            # Per-element ``.scc`` table: when present, overrides the global
            # SPACE_CHARGE_COMP factor inside this field map (manual: local
            # comp = Ki·scc(z), active SC = max(0, 1 − Ki·scc(z))).
            fd = getattr(element, "field_data", None)
            scc_prof = getattr(fd, "scc_profile", None) if fd is not None else None
            scc_scale = getattr(fd, "scc_scale", 0.0) if fd is not None else 0.0
            scc_mode = getattr(fd, "scc_mode", 0) if fd is not None else 0
            scc_active = (scc_prof is not None and scc_scale != 0.0
                          and scc_mode == 0)

            _interior = sorted(getattr(element, "interior_markers", [])
                               or [], key=lambda t: t[0])
            _mk_i = 0

            def _adv_transport(step):
                # Pure transport advance of *step* mm, split at interior
                # markers so each SHIFT_IN_FIELD_MAP row carries the
                # exact s = s_entry + dz and the Σ AT that z.  The
                # Strang SC-kick cadence is untouched (matrix
                # composition makes the split identical); the no-marker
                # path is one unsplit advance of exactly *step*,
                # bit-identical to the pre-marker code.  Pass z_cursor
                # so multi-cell elements (e.g. VaneRFQ) compute the
                # matrix at the correct element-local z; otherwise every
                # bundle restarts at z=0.
                nonlocal sigma, z_cursor, _mk_i
                remaining = step
                while (_mk_i < len(_interior)
                       and (_interior[_mk_i][0]
                            <= z_cursor + remaining + 1e-9)):
                    piece = _interior[_mk_i][0] - z_cursor
                    if piece > 1e-9:
                        M = _fitted_matrix_slice_at(element, self._ref,
                                                    piece, z_cursor)
                        self._probe_push(M)
                        sigma = M @ sigma @ M.T
                        element.advance_ref_over(self._ref, z_cursor,
                                                 z_cursor + piece)
                        z_cursor += piece
                        remaining -= piece
                    if self._results is not None:
                        self._record(self._results, sigma, self._ref,
                                     element_name=_interior[_mk_i][1].name)
                    _mk_i += 1
                if remaining > 1e-9:
                    M = _fitted_matrix_slice_at(element, self._ref,
                                                remaining, z_cursor)
                    self._probe_push(M)
                    sigma = M @ sigma @ M.T
                    element.advance_ref_over(self._ref, z_cursor,
                                             z_cursor + remaining)
                    z_cursor += remaining

            for _ in range(n_bundles):
                bundle_ds = sc_every * ds_mm     # mm covered by this bundle
                half_ds = bundle_ds / 2.0
                # First half of the bundle transport (matrix uses ref at
                # bundle entrance); the ref advances entrance → bundle
                # midpoint so the SC kick below sees the locally-correct
                # β, γ for accelerating elements (HWR cavities gain
                # ~30 % β per 250 mm).
                _adv_transport(half_ds)
                # SC kick at bundle midpoint (2-D DC or 3-D Ferrario).
                if scc_active:
                    scc_z = float(np.interp(
                        z_cursor, scc_prof[:, 0], scc_prof[:, 1]))
                    saved = self._sc_factor
                    self._sc_factor = max(0.0, 1.0 - scc_scale * scc_z)
                    M_sc = self._sc_matrix(sigma, ds_mm=bundle_ds)
                    self._sc_factor = saved
                else:
                    M_sc = self._sc_matrix(sigma, ds_mm=bundle_ds)
                self._probe_push(M_sc, is_sc=True)
                sigma = M_sc @ sigma @ M_sc.T
                # Second half of the bundle transport (matrix uses ref at
                # bundle midpoint — the locally-correct β for this half).
                _adv_transport(half_ds)
                # Record σ at bundle exit when substep recording is enabled.
                if self._record_substeps and self._results is not None:
                    self._record(self._results, sigma, self._ref,
                                 element_name=getattr(element, "name", "?"))

            # Trailing remainder (integration steps not covered by full bundles)
            leftover_steps = n_int - n_bundles * sc_every
            if leftover_steps > 0:
                _adv_transport(leftover_steps * ds_mm)
            # Frequency carry-over (mirrors the last line of advance_ref so
            # downstream ref.phi_s maths stays consistent).  Magnetic-only
            # maps must NOT update ref.frequency (see _has_electric_channel).
            if (getattr(element, "frequency", 0.0) > 0
                    and element.frequency != self._ref.frequency
                    and getattr(element, "_has_electric_channel", lambda: True)()):
                self._ref.frequency = element.frequency

        else:
            # ThinKick / zero-length / other passive: full map, no SC sub-stepping
            M_full = get_element_matrix(element, self._ref)
            self._probe_push(M_full)
            sigma = M_full @ sigma @ M_full.T

        return sigma
