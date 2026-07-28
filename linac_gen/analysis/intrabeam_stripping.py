"""Intra-beam stripping (IBS) loss analyzer for H⁻ linacs.

Implements the closed-form rate from Lebedev et al., "Intrabeam Stripping
in H⁻ Linacs", arXiv:1207.5492 / LINAC2010 (FERMILAB-CONF-10-382-AD).

Eq. (1) — single-electron stripping cross-section σ_H(β):

    σ_H(β) = 120·a₀² · (α/(β+α))² · (β-β_m)⁶/((β-β_m)⁶+β_m⁶) · ln(3.2·((β+α)/α)²)

Eq. (6) — closed-form approximation of the form factor F(a, b, c):

    F(a,b,c) = 1 + (2-√3)/(√3·(√3-1)) · ((a+b+c)/√(a²+b²+c²) - 1)

Eq. (7) — lab-frame relative loss rate per metre travelled:

    1   dN     N · σ_max · √(γ²θ_x² + γ²θ_y² + θ_s²)
    -- ---- = ─────────────────────────────────────── · F(γθ_x, γθ_y, θ_s)
    N   ds          8π² · σ_x σ_y σ_s · γ²

where θ_s is the relative rms momentum spread dp/p (paper convention).

Eq. (8) — non-constant cross-section override (used outside the σ_H
plateau, e.g. PIP-II's superconducting linac):

    σ_max → σ_H(0.72·β_m + 2β·√(γ²θ_x² + γ²θ_y² + θ_s²))

Two corrections relative to Ostiguy's `intrabeam_stripping3.py` (2010,
python-3 port 2022) are adopted by default:

  * F is evaluated on the velocity spreads (γθ_x, γθ_y, θ_s), as the
    paper specifies — not on the bunch sizes.  Ostiguy acknowledged the
    bug in a 2022 email but the script we have still uses sizes.
  * θ_s = dp/p (paper).  Ostiguy's script substitutes θ_s = dv/v on
    physical grounds (factor 1/γ²); reproduce by passing
    ``theta_z_convention="dv_over_v"``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


# Physical constants (cgs for σ_H, then converted to mm² for the rate).
_A0_CM = 0.529e-8         # Bohr radius [cm]
_ALPHA_FS = 1.0 / 137.0   # fine-structure constant
_BETA_M = 7.5e-5          # cross-section vanishes below this β
_E_CHARGE = 1.602176634e-19   # Coulombs
_C_LIGHT = 2.99792458e8       # m/s


# ---------------------------------------------------------------------------
# Building blocks (Eqs. 1 and 6).
# ---------------------------------------------------------------------------
def sigma_h(beta) -> np.ndarray:
    """H⁻ single-electron stripping cross-section in cm² (Lebedev Eq. 1).

    Vectorised: ``beta`` may be a scalar or array.  Returns an array of
    the same shape, in cm².
    """
    b = np.asarray(beta, dtype=float)
    # NB: Lebedev Eq. (1) prints 240·a0²·…·ln(1.79·X); the 120·…·ln(3.2·X²)
    # form below is the IDENTICAL expression (1.79² = 3.2041, and squaring the
    # log absorbs the factor 2: 240 = 120·2).  Do NOT "fix" 120 → 240 without
    # also changing ln(3.2·X²) → ln(1.79·X).
    val = 120.0 * _A0_CM ** 2
    val = val * (_ALPHA_FS / (b + _ALPHA_FS)) ** 2
    val = val * (b - _BETA_M) ** 6 / ((b - _BETA_M) ** 6 + _BETA_M ** 6)
    val = val * np.log(3.2 * ((b + _ALPHA_FS) / _ALPHA_FS) ** 2)
    return val


def form_factor(a, b, c) -> np.ndarray:
    """Closed-form approximation to the form factor F(a, b, c) (Lebedev Eq. 6).

    Scale-invariant: depends only on the ratios of (a, b, c).  Returns its
    minimum of 1.0 when at most one argument is nonzero (i.e. two of the
    three are zero), and peaks at 2/√3 ≈ 1.1547 when all three are equal.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    denom = np.sqrt(a * a + b * b + c * c)
    safe = np.where(denom > 0.0, denom, 1.0)
    factor = (2.0 - np.sqrt(3.0)) / (np.sqrt(3.0) * (np.sqrt(3.0) - 1.0))
    val = 1.0 + factor * ((a + b + c) / safe - 1.0)
    return np.where(denom > 0.0, val, 1.0)


# ---------------------------------------------------------------------------
# Analyzer entry point.
# ---------------------------------------------------------------------------
@dataclass
class IbsResult:
    """Per-step IBS quantities along the linac.

    All s-indexed arrays share the same length as ``s``.  The scalars
    record the inputs that drove the calculation.
    """
    s: np.ndarray                   # m (lab-frame longitudinal position)
    loss_rate_per_m: np.ndarray     # (1/N) dN/ds  in units of [1/m]
    power_loss_per_m_W: np.ndarray  # local power loss [W/m]
    integral_loss: np.ndarray       # cumulative ∫ rate ds  (dimensionless)
    integral_power_loss_W: np.ndarray  # cumulative ∫ power_loss_per_m ds [W]
    theta_x: np.ndarray             # rms x' [rad], lab frame
    theta_y: np.ndarray             # rms y' [rad], lab frame
    theta_s: np.ndarray             # dp/p (or dv/v if requested), lab frame
    sigma_x_mm: np.ndarray
    sigma_y_mm: np.ndarray
    sigma_z_mm: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray
    sigma_strip_cm2: np.ndarray     # σ used at each step (constant or β-dependent)
    form_factor: np.ndarray
    duty_factor: float
    current_mA: float
    n_per_bunch: float
    species: str
    theta_z_convention: str         # "dp_over_p" or "dv_over_v"
    non_constant_xs: bool


def _theta_z(sigma_w_mev: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
             mass_mev: float, convention: str) -> np.ndarray:
    """Convert σ_W (MeV) to the longitudinal velocity-coordinate spread.

    dp/p = σ_W / (β² γ m_0)  (relativistic momentum, m_0 in MeV).
    dv/v = dp/p / γ²         (Ostiguy override).
    """
    sigma_w = np.asarray(sigma_w_mev, dtype=float)
    denom = (beta ** 2) * gamma * mass_mev
    safe = np.where(denom > 0.0, denom, 1.0)
    theta_dp_p = np.where(denom > 0.0, sigma_w / safe, 0.0)
    if convention == "dv_over_v":
        return theta_dp_p / np.where(gamma > 0.0, gamma * gamma, 1.0)
    return theta_dp_p


def _sigma_z_from_phi(sigma_phi_deg: np.ndarray, beta: np.ndarray,
                      ref_freq_mhz: np.ndarray) -> np.ndarray:
    """σ_z [mm] from σ_φ [deg], using the per-step β and RF frequency.

    Same recipe used by EnvelopeTriple — phase spread × β × λ_RF / 360.
    """
    sigma_phi = np.asarray(sigma_phi_deg, dtype=float)
    f_hz = np.asarray(ref_freq_mhz, dtype=float) * 1.0e6
    safe_f = np.where(f_hz > 0.0, f_hz, 1.0)
    wl_mm = (_C_LIGHT / safe_f) * 1.0e3   # mm
    return np.where(f_hz > 0.0,
                    sigma_phi * beta * wl_mm / 360.0,
                    0.0)


def ibs_loss(results, beam_config, *,
             duty_factor: float | None = None,
             current_mA: float | None = None,
             non_constant_xs: bool = False,
             theta_z_convention: str = "dp_over_p",
             _legacy_F_args: bool = False) -> IbsResult:
    """Compute IBS loss curves for an H⁻ tracking run.

    Parameters
    ----------
    results
        A finished tracking ``Results`` object (must have ``s``,
        ``sigma_x``, ``sigma_y``, ``sigma_phi``, ``sigma_w``,
        ``sigma_matrix``, ``ref_beta``, ``ref_gamma``, ``ref_w_kin``,
        ``mass_mev``).
    beam_config
        The ``BeamConfig`` used for the run.  Provides species, peak
        current, bunch frequency.
    duty_factor : float, optional
        Power-loss scaling factor (0–1).  Defaults to
        ``beam_config.duty_cycle / 100``.  Only multiplies the W/m
        curves; the fractional loss rate is independent of duty.
    current_mA : float, optional
        Peak beam current.  Defaults to ``beam_config.current``.
    non_constant_xs : bool
        If True, use Eq. (8) per-step β-dependent σ_H instead of a
        constant σ_max.  Recommended outside the σ_H plateau (PIP-II SC
        section: β ≳ 0.5).
    theta_z_convention : {"dp_over_p", "dv_over_v"}
        Longitudinal coordinate inside the radical.  Default is the
        paper's dp/p; pass "dv_over_v" to reproduce Ostiguy's override.
    _legacy_F_args : bool
        Internal regression flag.  When True, evaluate the form factor
        on bunch sizes (γσ_x, γσ_y, σ_z) instead of velocity spreads
        (γθ_x, γθ_y, θ_s).  Kept *only* to reproduce Ostiguy's known
        bug in unit tests; do not enable from production code.

    Returns
    -------
    IbsResult
    """
    if getattr(beam_config, "species", None) != "H-":
        raise ValueError(
            f"IBS analysis applies only to H- beams; got species="
            f"{beam_config.species!r}"
        )
    if theta_z_convention not in ("dp_over_p", "dv_over_v"):
        raise ValueError(
            f"theta_z_convention must be 'dp_over_p' or 'dv_over_v', "
            f"got {theta_z_convention!r}"
        )

    if duty_factor is None:
        duty_factor = float(getattr(beam_config, "duty_cycle", 100.0)) / 100.0
    if current_mA is None:
        current_mA = float(beam_config.current)

    f_bunch_hz = float(beam_config.frequency) * 1.0e6
    if f_bunch_hz <= 0.0:
        raise ValueError("BeamConfig.frequency must be > 0 (MHz)")
    n_per_bunch = (current_mA * 1.0e-3) / (_E_CHARGE * f_bunch_hz)

    # Per-step arrays.
    s_m = np.asarray(results.s, dtype=float) / 1000.0  # mm → m
    sigma_x = np.asarray(results.sigma_x, dtype=float)        # mm
    sigma_y = np.asarray(results.sigma_y, dtype=float)        # mm
    sigma_phi = np.asarray(results.sigma_phi, dtype=float)    # deg
    sigma_w = np.asarray(results.sigma_w, dtype=float)        # MeV
    gamma = np.asarray(results.ref_gamma, dtype=float)
    beta = np.asarray(results.ref_beta, dtype=float)
    w_kin_mev = np.asarray(results.ref_w_kin, dtype=float)
    mass_mev = float(getattr(results, "mass_mev", 0.0))
    if mass_mev <= 0.0:
        raise ValueError(
            "Results.mass_mev must be set (rest mass in MeV); your "
            "tracking run may have skipped recording species mass."
        )

    # Reference RF frequency (per-step) used to convert σ_φ → σ_z.  This
    # mutates at every FREQ-card boundary in the lattice — the recorder
    # captures the local value at each diagnostic step.  Older runs (pre
    # ref_frequency support), or results loaded from formats that flattened
    # the list to a scalar, fall back to the constant bunch frequency.
    ref_freq_attr = getattr(results, "ref_frequency", None)
    if ref_freq_attr is None:
        ref_freq_arr = np.full_like(gamma, float(beam_config.frequency))
    else:
        ref_freq_arr = np.atleast_1d(np.asarray(ref_freq_attr, dtype=float))
        if ref_freq_arr.size == 0:
            ref_freq_arr = np.full_like(gamma, float(beam_config.frequency))
        elif ref_freq_arr.size != gamma.size:
            # Scalar broadcast or length mismatch (older runs): use the
            # first finite value if available, otherwise BeamConfig.
            seed = (float(ref_freq_arr.flat[0])
                    if ref_freq_arr.size and ref_freq_arr.flat[0] > 0.0
                    else float(beam_config.frequency))
            ref_freq_arr = np.full_like(gamma, seed)
    # Some early-run records may carry 0.0 (the recorder's float fallback)
    # if the reference particle hadn't been set yet — patch those to the
    # bunch frequency so σ_z stays well-defined.
    ref_freq_mhz = np.where(ref_freq_arr > 0.0, ref_freq_arr,
                            float(beam_config.frequency))

    sigma_z = _sigma_z_from_phi(sigma_phi, beta, ref_freq_mhz)         # mm

    # Lab-frame angular spreads from the σ-matrix (mrad).
    sm = getattr(results, "sigma_matrix", None)
    if sm is None:
        raise ValueError(
            "Results.sigma_matrix is required to extract σ_x' and σ_y'."
        )
    sm_arr = np.asarray(sm, dtype=float)            # (N, 6, 6)
    if sm_arr.ndim != 3 or sm_arr.shape[0] != s_m.size:
        raise ValueError(
            f"Results.sigma_matrix has shape {sm_arr.shape}; expected "
            f"(N, 6, 6) with N = {s_m.size} matching len(s)."
        )
    theta_x_rad = np.sqrt(np.maximum(sm_arr[:, 1, 1], 0.0)) * 1.0e-3
    theta_y_rad = np.sqrt(np.maximum(sm_arr[:, 3, 3], 0.0)) * 1.0e-3
    theta_s = _theta_z(sigma_w, gamma, beta, mass_mev, theta_z_convention)

    # Velocity-spread radical (units of c).
    v_rel = np.sqrt(
        (gamma * theta_x_rad) ** 2
        + (gamma * theta_y_rad) ** 2
        + theta_s ** 2
    )

    # Stripping cross-section (cm²) → mm² for the rate.
    if non_constant_xs:
        sigma_strip_cm2 = sigma_h(0.72 * _BETA_M + 2.0 * beta * v_rel)
    else:
        sigma_strip_cm2 = np.full_like(beta, sigma_h(4.0e-4))
    sigma_strip_mm2 = sigma_strip_cm2 * 1.0e2

    # Form factor — paper-correct arguments by default.
    if _legacy_F_args:
        F = form_factor(gamma * sigma_x, gamma * sigma_y, sigma_z)
    else:
        F = form_factor(gamma * theta_x_rad, gamma * theta_y_rad, theta_s)

    # Lebedev Eq. (7).  Inputs in mm; result in mm⁻¹ → multiply by 1000.
    safe_g2 = np.where(gamma > 0.0, gamma * gamma, 1.0)
    denom = (8.0 * pi ** 2) * sigma_x * sigma_y * sigma_z * safe_g2
    safe_d = np.where(denom > 0.0, denom, 1.0)
    rate_per_mm = np.where(
        denom > 0.0,
        n_per_bunch * sigma_strip_mm2 * v_rel * F / safe_d,
        0.0,
    )
    loss_rate_per_m = rate_per_mm * 1.0e3

    # Power loss [W/m] = rate · DF · I_peak[A] · W_kin[J] / e
    # (= rate · DF · I_peak[A] · W_kin[eV])
    power_loss_per_m_W = (
        loss_rate_per_m * duty_factor * (current_mA * 1.0e-3) * w_kin_mev * 1.0e6
    )

    # Cumulative integrals.
    integral_loss = np.concatenate(([0.0], _cumtrapz(loss_rate_per_m, s_m)))
    integral_power_loss_W = np.concatenate(
        ([0.0], _cumtrapz(power_loss_per_m_W, s_m))
    )

    return IbsResult(
        s=s_m,
        loss_rate_per_m=loss_rate_per_m,
        power_loss_per_m_W=power_loss_per_m_W,
        integral_loss=integral_loss,
        integral_power_loss_W=integral_power_loss_W,
        theta_x=theta_x_rad,
        theta_y=theta_y_rad,
        theta_s=theta_s,
        sigma_x_mm=sigma_x,
        sigma_y_mm=sigma_y,
        sigma_z_mm=sigma_z,
        gamma=gamma,
        beta=beta,
        sigma_strip_cm2=sigma_strip_cm2,
        form_factor=F,
        duty_factor=float(duty_factor),
        current_mA=float(current_mA),
        n_per_bunch=float(n_per_bunch),
        species=beam_config.species,
        theta_z_convention=theta_z_convention,
        non_constant_xs=bool(non_constant_xs),
    )


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of ``y`` over ``x``; returns one
    fewer point than its inputs (consistent with scipy's signature)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return np.zeros(0, dtype=float)
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return np.cumsum(avg * dx)
