"""Magnetic-stripping (Lorentz-stripping) loss analyzer for H⁻ linacs.

Implements the Lorentz / magnetic-stripping fractional loss rate from
Folsom, Eshraqi, Blaskovic-Kraljevic, Gålnander, *"Stripping mechanisms
and remediation for H⁻ beams"*, Phys. Rev. Accel. Beams **24**, 074201
(2021), arXiv:2103.16195, Eq. (5):

    1   dN     |B_⊥|         A_2
    -- ---- = ──────  · exp(─ ──────────── )           [1/m]
    N   ds     A_1            γ β c |B_⊥|

with the Lorentz transformation Eq. (4) supplying the rest-frame
electric field:

    |E*| = γ β c |B_⊥|                                  [V/m]

**Only the field perpendicular to v strips** (Eq. 4 transforms B_⊥ only),
so per element we build

    |B_⊥| = sqrt( B_transverse² + (B_axial · θ_rms)² )

where B_transverse is the field already ⊥ to the axis (dipole bend field,
quad G·r, field-map B_r/B_x/B_y) and B_axial is the on-axis longitudinal
field (a solenoid's B_z, field-map F_z) which is ∥ to v on axis and only
couples through the rms beam divergence θ_rms (from the σ-matrix; v×B → 0
for a paraxial beam).  Treating a solenoid's axial B_z as B_⊥ directly
over-states its stripping by orders of magnitude, which is why Folsom 2021
/ TraceWin model only dipoles and quads.

and empirical constants (Folsom 2021 Eq. (5); same fit used by TraceWin
and PyORBIT, originally from Stinson 1969 / Scherk 1979 refits):

    A_1 = 3.073 × 10⁻⁶ s · V / m
    A_2 = 4.414 × 10⁹ V / m

Sister to ``intrabeam_stripping.py``: this is a pure post-tracking
analyzer.  It walks the recorded ``s``-grid, looks up the element a
given step belongs to via ``results.element_names``, dispatches on
element type to extract the design |B|, evaluates Eq. (5) per step, and
returns smooth loss-rate / power-loss / cumulative curves for the GUI's
``_MagStripPopup`` and any CLI consumer.

Mode A only — per-particle in-track Monte-Carlo loss is out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Folsom 2021 Eq. (5) empirical constants.
A1_SVPM = 3.073e-6      # s · V / m
A2_VPM  = 4.414e9       # V / m

# Speed of light, m/s.
_C = 2.99792458e8
# Electron charge, C.
_E = 1.602176634e-19


# ---------------------------------------------------------------------------
# Result dataclass — mirrors IbsResult shape so the GUI plumbing is symmetric.
# ---------------------------------------------------------------------------
@dataclass
class MagStripResult:
    """Per-step magnetic-stripping quantities along the linac.

    ``s`` is in metres (lab-frame longitudinal).  All ``s``-indexed
    arrays share its length.  Scalars record the inputs that drove the
    calculation so the GUI can echo them back into its summary line.
    """
    s: np.ndarray                       # m
    B_T: np.ndarray                     # design |B| at each step [T]
    E_star_V_per_m: np.ndarray          # γβc|B| [V/m]
    loss_rate_per_m: np.ndarray         # dN/N/ds [1/m]   (Folsom Eq. 5)
    power_loss_per_m_W: np.ndarray      # dP/ds [W/m]
    integral_loss: np.ndarray           # ∫ rate ds (dimensionless)
    integral_power_loss_W: np.ndarray   # ∫ power_loss ds (W)
    gamma: np.ndarray
    beta: np.ndarray
    sigma_x_mm: np.ndarray
    sigma_y_mm: np.ndarray
    element_names: List[str]            # name of dominant element per step
    element_kinds: List[str]            # kind label per step ("dipole", "quad", ...)
    duty_factor: float
    current_mA: float
    species: str
    # Constants used (echoed for traceability).
    A1: float = A1_SVPM
    A2: float = A2_VPM
    quad_b_scale: str = "2sigma"
    fieldmap_sample: str = "max"
    theta_rms: Optional[np.ndarray] = None   # rms divergence used for axial B⊥


# ---------------------------------------------------------------------------
# Element-type B-field dispatch.
#
# Returns (|B| in Tesla, kind-label).  Returns 0.0 for elements that don't
# carry a magnetic field (drifts, bare RF cavities, edges, passive markers).
# ---------------------------------------------------------------------------
def _design_b_for_element(element, brho: float,
                          sigma_x_mm: float, sigma_y_mm: float,
                          quad_b_scale: str,
                          fieldmap_sample: str,
                          theta_rms: float = 1.0e-3) -> tuple[float, str]:
    """Return (|B_⊥|, kind) for one element at a given (σ_x, σ_y, brho).

    ``brho`` is the local magnetic rigidity at the recorder step in
    T·m (= |p/q|).  σ_x, σ_y in mm.  ``theta_rms`` is the rms beam
    divergence [rad] used to project an *axial* field (solenoid) onto the
    stripping-relevant perpendicular direction.  All conventions match the
    rest of the codebase.
    """
    cls = type(element).__name__

    # Dipole — B = brho / |ρ|, transverse (⊥ to v).  ρ stored in mm → m.
    if cls == "Dipole":
        rho_m = abs(float(getattr(element, "rho", 0.0))) * 1.0e-3
        if rho_m <= 0.0:
            return 0.0, "dipole"
        return abs(brho) / rho_m, "dipole"

    # Solenoid — on-axis field is AXIAL (∥ v); only B_z·θ_rms is perpendicular.
    if cls == "Solenoid":
        b_axial = abs(float(getattr(element, "field", 0.0)))
        return b_axial * float(theta_rms), "solenoid"

    # Quadrupole — B at a representative radius.  In a quad, |B(r)| = G·r
    # is purely transverse; particles near the axis see almost no B.
    # Three modes:
    #   "1sigma"  — B at √(σx² + σy²)
    #   "2sigma"  — B at 2·√(σx² + σy²)   (default — typical halo radius)
    #   "pole"    — B at the aperture (pole-tip; conservative upper bound)
    if cls == "Quadrupole":
        G = abs(float(getattr(element, "gradient", 0.0)))
        if G == 0.0:
            return 0.0, "quad"
        if quad_b_scale == "pole":
            r_mm = abs(float(getattr(element, "aperture", 0.0)))
            if r_mm <= 0.0:
                # No aperture set — fall back to 2σ.
                r_mm = 2.0 * float(np.hypot(sigma_x_mm, sigma_y_mm))
        elif quad_b_scale == "1sigma":
            r_mm = float(np.hypot(sigma_x_mm, sigma_y_mm))
        else:  # "2sigma" (default) and any unknown string
            r_mm = 2.0 * float(np.hypot(sigma_x_mm, sigma_y_mm))
        r_m = r_mm * 1.0e-3
        return G * r_m, "quad"

    # FieldMap (1-D / 2-D cyl / 1-D quad-gradient) — pull max |B| from any
    # magnetic channel in field_data.  Apply the element's `kb` scale
    # so the bundled value reflects the as-tracked field.
    if cls in ("FieldMap", "FieldMap3D"):
        return _design_b_for_fieldmap(element, sigma_x_mm, sigma_y_mm,
                                      fieldmap_sample, theta_rms), "fieldmap"

    # Everything else (Drift, RfCavity, Edge, DipoleEdge, PassiveElement, …):
    # no magnetic field that contributes to Lorentz stripping at lowest order.
    return 0.0, "passive"


def _design_b_for_fieldmap(element, sigma_x_mm: float, sigma_y_mm: float,
                           sample: str, theta_rms: float = 1.0e-3) -> float:
    """Return a representative |B_⊥| in Tesla for a FIELD_MAP element.

    Splits each magnetic channel into the **transverse** components
    (Fx/Fy/Fr — already ⊥ to v) and the **axial** component (Fz — ∥ to v
    on axis, which only strips through the beam divergence ``theta_rms``):

        |B_⊥| = sqrt( B_transverse² + (B_axial · θ_rms)² )

    so a solenoid imported as an on-axis B_z(z) map no longer masquerades
    as a full perpendicular field.

    `sample="max"`  — peak over the entire grid (default).
    `sample="on_axis"` — restrict the axial channel to the r=0 line.

    For a 1-D quad-gradient (geometry=9), Fz holds G(z) [T/m] and the
    representative B is G · 2σ (transverse, matching the hard-edge quad).
    """
    fd = getattr(element, "field_data", None)
    if fd is None or not getattr(fd, "channels", None):
        return 0.0

    # Late import: avoid pulling tracewin_geom into module load if a user
    # imports this analyzer without ever opening a FIELD_MAP.
    from linac_gen.io.tracewin_geom import Channel

    kb = float(getattr(element, "kb", 1.0)) * float(getattr(element, "scale", 1.0))
    if kb == 0.0:
        return 0.0

    # Walk every magnetic channel; geometry=9 (1-D G(z)) is special-cased.
    b_max = 0.0
    for ch_kind, ch_data in fd.channels.items():
        if not getattr(ch_kind, "is_magnetic", False):
            continue

        geom = getattr(ch_data, "geometry", 0)

        if geom == 9:
            # G(z) in T/m → representative B = G · 2σ_xy (transverse).
            Fz = ch_data.Fz
            if Fz is None:
                continue
            G_max = float(np.max(np.abs(np.asarray(Fz, dtype=float))))
            r_m = 2.0 * float(np.hypot(sigma_x_mm, sigma_y_mm)) * 1.0e-3
            b_here = G_max * r_m
        else:
            # Transverse components (Fx/Fy/Fr) are already ⊥ to v; the axial
            # Fz is ∥ to v on axis and only strips through θ_rms.
            trans_sq = None
            for name in ("Fx", "Fy", "Fr", "Fq"):
                arr = getattr(ch_data, name, None)
                if arr is not None:
                    a = np.asarray(arr, dtype=float)
                    trans_sq = a * a if trans_sq is None else trans_sq + a * a
            b_trans = (float(np.sqrt(np.max(trans_sq)))
                       if trans_sq is not None else 0.0)

            b_axial = 0.0
            fz = getattr(ch_data, "Fz", None)
            if fz is not None:
                fz = np.asarray(fz, dtype=float)
                if sample == "on_axis" and geom in (4, 5) and fz.ndim == 2:
                    r_axis = getattr(ch_data, "r", None)
                    z_axis = getattr(ch_data, "z", None)
                    if (r_axis is not None and z_axis is not None
                            and fz.shape == (len(r_axis), len(z_axis))):
                        fz = fz[0, :]   # r=0 line
                b_axial = float(np.max(np.abs(fz)))

            if b_trans == 0.0 and b_axial == 0.0:
                continue
            b_here = float(np.hypot(b_trans, b_axial * float(theta_rms)))

        b_max = max(b_max, b_here * kb)

    return b_max


# ---------------------------------------------------------------------------
# Analyzer entry point.
# ---------------------------------------------------------------------------
def magnetic_stripping_loss(
    results,
    lattice,
    beam_config,
    *,
    duty_factor: Optional[float] = None,
    current_mA: Optional[float] = None,
    quad_b_scale: str = "2sigma",
    fieldmap_sample: str = "max",
    default_div_rad: float = 1.0e-3,
) -> MagStripResult:
    """Compute Lorentz-stripping loss curves for an H⁻ tracking run.

    Parameters
    ----------
    results
        A finished tracking ``Results`` (envelope or multiparticle).
        Must expose ``s`` (mm), ``ref_gamma``, ``ref_beta``,
        ``ref_w_kin``, ``sigma_x``, ``sigma_y``, ``element_names``,
        and ``mass_mev``.
    lattice
        The lattice tracked through.  Used to resolve element-name →
        element-instance for B-field dispatch.
    beam_config
        ``BeamConfig`` driving the run.  Must be H⁻ (raises otherwise).
    duty_factor
        Power-loss scale factor (0–1).  Defaults to
        ``beam_config.duty_cycle / 100``.
    current_mA
        Peak beam current.  Defaults to ``beam_config.current``.
    quad_b_scale
        How to evaluate |B| inside a quadrupole: ``"2sigma"`` (default,
        representative halo), ``"1sigma"``, or ``"pole"`` (aperture / pole
        tip — conservative upper bound).
    fieldmap_sample
        How to evaluate |B| inside a ``FieldMap`` / ``FieldMap3D``:
        ``"max"`` (peak over full grid, default) or ``"on_axis"`` (peak
        along r = 0).

    Returns
    -------
    MagStripResult
    """
    species = getattr(beam_config, "species", None)
    if species != "H-":
        raise ValueError(
            f"Magnetic-stripping analysis applies only to H- beams; "
            f"got species={species!r}"
        )

    if duty_factor is None:
        duty_factor = float(getattr(beam_config, "duty_cycle", 100.0)) / 100.0
    if current_mA is None:
        current_mA = float(beam_config.current)

    s_m = np.asarray(results.s, dtype=float) / 1000.0          # mm → m
    sigma_x = np.asarray(results.sigma_x, dtype=float)         # mm
    sigma_y = np.asarray(results.sigma_y, dtype=float)         # mm
    gamma = np.asarray(results.ref_gamma, dtype=float)
    beta = np.asarray(results.ref_beta, dtype=float)
    w_kin_mev = np.asarray(results.ref_w_kin, dtype=float)
    elem_names = list(getattr(results, "element_names", []))
    mass_mev = float(getattr(results, "mass_mev", 0.0))
    if mass_mev <= 0.0:
        raise ValueError(
            "Results.mass_mev must be set (rest mass in MeV); your "
            "tracking run may have skipped recording species mass."
        )

    n = s_m.size
    if (sigma_x.size != n or sigma_y.size != n or gamma.size != n
            or beta.size != n or len(elem_names) != n):
        raise ValueError(
            f"Result-array length mismatch (s={n}, sigma_x={sigma_x.size}, "
            f"sigma_y={sigma_y.size}, gamma={gamma.size}, beta={beta.size}, "
            f"element_names={len(elem_names)})"
        )

    # Resolve element-name → element-instance once.  Lattices can have
    # duplicate names; fall back to the first match (loss is per-element-
    # *type* design field, not per-instance position, so this is fine).
    name_to_elem = {}
    for elem in lattice.elements:
        name = getattr(elem, "name", None)
        if name and name not in name_to_elem:
            name_to_elem[name] = elem

    # Local magnetic rigidity per step: brho [T·m] = γβ · m₀c / |q|.
    # In useful units, with |q| = z_q · e and m₀c² = mass_MeV·MeV:
    #     brho_T_m = γβ · mass_MeV · 1e6 / (c · z_q)
    # since the electron-charge factor cancels between p [eV/c] and q [e].
    abs_q_e = max(abs(float(beam_config_charge_e(beam_config))), 1e-12)
    brho_arr = gamma * beta * mass_mev * 1.0e6 / (_C * abs_q_e)

    # Per-step rms divergence θ_rms [rad] used to project axial fields onto
    # the perpendicular (stripping) direction.  Prefer the σ-matrix
    # (σ_x' = √Σ₂₂, σ_y' = √Σ₄₄, stored in mrad); fall back to a default for
    # runs/results that didn't record it.
    theta_rms = np.full(n, float(default_div_rad), dtype=float)
    sm = getattr(results, "sigma_matrix", None)
    if sm is not None:
        sm_arr = np.asarray(sm, dtype=float)
        if sm_arr.ndim == 3 and sm_arr.shape == (n, 6, 6):
            txp = np.sqrt(np.maximum(sm_arr[:, 1, 1], 0.0)) * 1.0e-3   # rad
            typ = np.sqrt(np.maximum(sm_arr[:, 3, 3], 0.0)) * 1.0e-3
            th = np.hypot(txp, typ)
            theta_rms = np.where(th > 0.0, th, float(default_div_rad))

    # Per-step |B_⊥| in Tesla via element dispatch.
    B_T = np.zeros(n, dtype=float)
    kinds: List[str] = []
    for i in range(n):
        elem = name_to_elem.get(elem_names[i])
        if elem is None:
            kinds.append("unknown")
            continue
        b, kind = _design_b_for_element(
            elem, float(brho_arr[i]), float(sigma_x[i]), float(sigma_y[i]),
            quad_b_scale, fieldmap_sample, float(theta_rms[i]),
        )
        B_T[i] = b
        kinds.append(kind)

    # Folsom Eq. (4): |E*| = γβc|B|.
    E_star = gamma * beta * _C * B_T

    # Folsom Eq. (5): rate = |B|/A1 · exp(-A2/E*).
    # Guard E* = 0 (B = 0 elements: drifts, RF cavities, passive markers)
    # — exp(-A2/0) is taken as zero and the prefactor multiplies through.
    safe = E_star > 0.0
    rate = np.zeros(n, dtype=float)
    if np.any(safe):
        rate[safe] = (B_T[safe] / A1_SVPM) * np.exp(
            -A2_VPM / E_star[safe]
        )

    # Power loss per metre [W/m] = rate · DF · I_peak[A] · W_kin[eV].
    # I_peak in mA; W_kin in MeV; one factor of 1e-3 (mA → A) and one of
    # 1e6 (MeV → eV) leaves a net factor of 1e3.
    power_W_per_m = rate * duty_factor * current_mA * w_kin_mev * 1.0e3

    # Cumulative integrals.
    integral_loss = np.concatenate(([0.0], _cumtrapz(rate, s_m)))
    integral_power = np.concatenate(([0.0], _cumtrapz(power_W_per_m, s_m)))

    return MagStripResult(
        s=s_m,
        B_T=B_T,
        E_star_V_per_m=E_star,
        loss_rate_per_m=rate,
        power_loss_per_m_W=power_W_per_m,
        integral_loss=integral_loss,
        integral_power_loss_W=integral_power,
        gamma=gamma,
        beta=beta,
        sigma_x_mm=sigma_x,
        sigma_y_mm=sigma_y,
        element_names=elem_names,
        element_kinds=kinds,
        duty_factor=float(duty_factor),
        current_mA=float(current_mA),
        species=species,
        A1=A1_SVPM,
        A2=A2_VPM,
        quad_b_scale=quad_b_scale,
        fieldmap_sample=fieldmap_sample,
        theta_rms=theta_rms,
    )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------
def beam_config_charge_e(beam_config) -> float:
    """Return the species charge in units of e (signed)."""
    species = getattr(beam_config, "species", "H-")
    # Keep the table local to avoid pulling linac_gen.core.particle here.
    if species in ("H-",):
        return -1.0
    if species in ("proton", "deuteron"):
        return 1.0
    # Unknown — assume |q| = 1 e (most common for accelerated species).
    return 1.0


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of ``y`` over ``x``.

    Returns one fewer point than its inputs, matching scipy's signature.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return np.zeros(0, dtype=float)
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return np.cumsum(avg * dx)
