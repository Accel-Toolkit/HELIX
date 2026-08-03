"""Exact TraceWin RFQ_CELL per-step coefficients — ground-truth calibrated.

Source: the TraceWin manual's "RFQ cell" transfer-matrix annex
(``TraceWIn_Tools/tracewin.htm``, formulas embedded as images
``tracewin_fichiers/image254-284.png`` — every formula below carries its
image number), with the transcription ambiguities RESOLVED numerically
against TraceWin's own per-cell matrices (``Transfer_matrix1.dat`` of the
PXIE LEBT+RFQ project: cumulative 6x6 at each element, per-cell via
M_n·M_{n-1}⁻¹, 203 RFQ cells).  Final state (2026-07-30 vane-field
campaign, see ``tests/rfq``): median per-cell relative error 1.25 %,
mean 2.1 %, 200/203 cells within 10 %, worst 0.17; envelope σ vs the
TW ENV export 1.1 % (x) / 1.7 % (y) — against mean ABSOLUTE error ~70
(wrong focusing sign) for the pre-2026-07 model.  NOTE: the reference
was produced with the Toutatis VANE tables (RFQ_GEOM active in the
project decks); the residual beyond the exact annex algorithm is
carried by the smooth TW calibration below (see its cautions).

The per-cell algorithm ("Transport through a RFQ cell", images 280-284):

    dz  = L/N                                   [280]
    t_s = dz/(2 β_in c) ;  z_s = dz/2           [281, 242]
    for i in 0..N-1:
        γ*_I = γ*_O                             [245]
        γ_{i+1} = γ_i + |q|πA10·V/(2L·mc²) · sin(ωt_s+φ0)
                                 · sin(πz_s/L) · dz          [282]
        γ*_O = γ_{i+1};  β_{i+1} = sqrt(1-γ_{i+1}⁻²)         [247]
        γ_s = (γ_{i+1}+γ_i)/2;  β_s from γ_s                 [283, 249]
        apply DKD substep matrices (below)      [250, 268, 284]
        t_s += dz/(β_{i+1} c);  z_s += dz       [252, 253]

Substep matrices (drift-kick-drift, images 258/263/264):

    M_x = D(dz/2) · [[1,0],[k̂_x1, K2]] · D(dz/2)
    M_y = D(dz/2) · [[1,0],[k̂_y1, K2]] · D(dz/2)
    M_z = Dz(dz/2γ*_O²) · [[1,0],[K1, K2]] · Dz(dz/2γ*_I²)

with, writing  pref = |q|·dz / (γ_s β_s² · 2mc²)  and  φ = ωt_s + φ0:

    k̂_x1 = -pref·( sin φ · 2S·(V/r0²)·C1  -  cos φ ·(π/L)²·(A10·V/2)·C2 )
    k̂_y1 = -pref·(-sin φ · 2S·(V/r0²)·C1  -  cos φ ·(π/L)²·(A10·V/2)·C2 )
    K1   = -pref·(π/L)²·A10·V·C3·cos φ                       [259]
    K2   = 1 - pref·A10·V·(π/L)·C3·sin φ                     [260; sin —
           K2 is the momentum ratio and follows the gain phase, see the
           determinant-invariant note in step_kicks]

GROUND-TRUTH RESOLUTIONS vs the manual images (the annex images 265/266
show cos(ωt_s+φ0) on the whole transverse bracket and an A01 factor on
the quadrupole term; both are contradicted by TraceWin's own matrices):

1. The electric-quadrupole term oscillates as sin(ωt_s+φ0) — in phase
   with the vane POTENTIAL, like E_z [257] — not cos.  (cos gave mean
   per-cell error 71; sin with the resolutions below gives 0.86.)  With
   φ0 ≈ -90° this makes the transverse gradient change sign mid-cell:
   each cell is an FD doublet — exactly the diagonal-swap structure of
   consecutive TW per-cell matrices.
2. The quadrupole strength is V/r0² exactly — NO A01 = (1-A10)
   reduction (TW's R0 "vane average radius" is by definition the radius
   where the quad term is V/r0²; fitted scale = 1.98±0.02 ≈ 2 across
   the full A10 ∈ [0, 0.69] range with pref's 1/2).
3. The RF-defocus term keeps the cos-phased single component
   (π/L)²·(A10·V/2)·C2 — the transverse counterpart of the
   longitudinal gradient K1.  HISTORY NOTE: with the initial
   (erroneous) cos-phased K2, the 203-cell fit demanded a spurious
   second sin-phased cos(πz/L) component (b=½) to patch the
   gentle-buncher cells; fixing K2's phase (resolution 5) removed the
   need entirely — the pure annex form then wins outright (mean error
   0.033, buncher cells 0.034, was 0.24 with the compensating pair).
   A cautionary example of two wrong terms fitting better than one.
4. The overall transverse sign: S = +sign(type) for ±2 accelerating
   cells composes correctly with the -pref convention above (the
   manual's S = -sign(type) [271] with its cos-phase produced the
   opposite per-cell M[1,0] sign — the 2026-04 audit's finding).
5. K2's unreadable middle glyph in image260 is a factor ONE (factor 2
   doubled the longitudinal diagonal error), and its time factor is
   sin(ωt_s+φ0) — K2 is the linearised momentum ratio p_in/p_out, so
   it follows the energy-gain phase.  The cumulative determinant
   invariant Π det(2×2) = p_in/p_out over the whole RFQ decides this
   unambiguously (see step_kicks); the per-cell max-element fit alone
   cannot (the per-cell difference is ~0.7 %, inside its noise).

Cell-type table (C1, C2, S, C3), z ∈ [0, L], arg = πz/L:

  ±2 accelerating [269, 270, 271; C3: 261]:
        C1 = 1;  C2 = sin(arg);  S = ±1;  C3 = sin(arg)
  +3 front-end   [272, 273, 274]:
        C1 = ¼[3cos(u) + cos(3u)],  u = arg/2 - π/2   (≡ sin³(arg/2))
        C2 = 0;  S = sign(type[n+1]);  C3 = sin(arg)
  -3 front-end   [275, 273, 276]:
        C1 = ¾[cos(v) + cos(3v)/3],  v = arg/2        (≡ cos³(arg/2);
        image275's middle sign is '+' — ground truth 2e-4 vs TW)
        C2 = 0;  S = sign(type[n-1]);  C3 = sin(arg)
  +4 transcell   [269, 277, 274; C3: 262]:
        C1 = 1;  C2 = ½[cos(arg)+1];  S = -sign(type[n+1]); C3 = ½sin(arg)
  -4 transcell   [278, 279, 276; C3: 262]:
        C1 = 1;  C2 = -½[cos(arg)-1]; S = -sign(type[n-1]); C3 = ½sin(arg)

(±4's S keeps the manual's own minus sign — OPPOSITE to ±2/±3, which
carry the calibrated global flip; the PXIE -4 transcell's plane-swap
vs TW decided this, see type_coeffs.)

(S entries already include resolution 4: they are the manual's
-sign(type[..]) composed with the calibrated global sign flip.)

Known residual vs TW (documented, not hidden): the -3 end cell and a
few shaper cells (φs ≈ -84°, m ≈ 1.1, PXIE ~cells 67-80) sit at
10-22 % relative on near-cancelling matrix elements (their F/D halves
nearly balance); plausibly vane-table effects.  Composed-envelope
agreement (1.1/1.7 % mean vs the vane-based export with the smooth TW
calibration; the export's own-matrix floor is ~4 %) is the acceptance
gate.
"""
from __future__ import annotations

import numpy as np


def type_coeffs(cell_type: int, type_prev: int, type_next: int,
                z_over_L: float) -> tuple[float, float, float, float]:
    """(C1, C2, S, C3) for one substep at fractional position z/L.

    ``S`` carries the calibrated overall sign (module docstring, item 4).
    Unknown cell types return all-zero coefficients (pure drift).
    """
    arg = np.pi * z_over_L
    st = 1.0 if cell_type > 0 else -1.0
    at = abs(int(cell_type))
    if at == 2:
        return 1.0, float(np.sin(arg)), st, float(np.sin(arg))
    if at == 3:
        # +3: C1 = ¼[3cos u + cos 3u] ≡ sin³(arg/2)  (ramp 0 → 1);
        # -3: C1 = cos³(arg/2)                        (ramp 1 → 0).
        # The -3 form is the triple-angle IDENTITY ¾[cos v + cos(3v)/3]
        # — image275's middle sign is '+' (the first transcription read
        # '−', a 22 % error on the PXIE exit cell; cos³ matches TW's
        # per-cell matrix to 2e-4, see tests/rfq).
        if st > 0:
            c1 = np.sin(0.5 * arg) ** 3
            s = 1.0 if type_next > 0 else -1.0
        else:
            c1 = np.cos(0.5 * arg) ** 3
            s = 1.0 if type_prev > 0 else -1.0
        return float(c1), 0.0, s, float(np.sin(arg))
    if at == 4:
        # ±4 keeps the manual's own -sign(type[n±1]) — OPPOSITE to the
        # ±3 convention above.  Ground truth: with correct neighbour
        # wiring the PXIE -4 transcell (neighbours -2/-3) needs S=+1;
        # +sign gave an exactly plane-swapped transverse matrix vs TW
        # (x-block ↔ y-block to 4 digits, found 2026-07-30).
        if st > 0:
            c2 = 0.5 * (np.cos(arg) + 1.0)
            s = -1.0 if type_next > 0 else 1.0
        else:
            c2 = -0.5 * (np.cos(arg) - 1.0)
            s = -1.0 if type_prev > 0 else 1.0
        return 1.0, float(c2), s, float(0.5 * np.sin(arg))
    return 0.0, 0.0, 0.0, 0.0


def vane_apertures(r0_mm: float, A10: float, length_mm: float,
                   cell_type: int, z_local_mm: float,
                   n_iter: int = 3) -> tuple[float, float]:
    """(x_lim, y_lim) in mm — the vane-tip distances at position z.

    Solves the two-term equipotential tip condition U(r, θ=0/90°, z) =
    ±V/2 with the calibrated A01 = 1 convention:

        x_lim² / r0² = 1 − s·A10·I0(k·x_lim)·cos(πz/L)
        y_lim² / r0² = 1 + s·A10·I0(k·y_lim)·cos(πz/L)

    with k = π/L and s = −sign(type), by fixed-point iteration on the
    Bessel factor.  Validated directly against the PXIE ``pxie-rfq.vane``
    tip table (2026-07-30): mean error 0.03–0.14 % across cells spanning
    A10 = 0.01–0.69 — the exact condition needs NO empirical factor
    (DYNAC's 0.7523 patches its parabolic profile approximation, not
    the physics).

    For ±3 front-end / exit cells the real vanes FLARE far beyond r0
    (the PXIE radial-matching section opens to ≈3.5·r0); callers skip
    transverse loss checks there rather than clip with a wrongly tight
    bore.
    """
    from scipy.special import i0
    if length_mm <= 0 or r0_mm <= 0:
        return r0_mm, r0_mm
    s = -1.0 if cell_type > 0 else 1.0
    c = float(np.cos(np.pi * z_local_mm / length_mm))
    k = np.pi / length_mm
    term = s * A10 * c
    x = r0_mm * np.sqrt(max(1.0 - term, 0.05))
    y = r0_mm * np.sqrt(max(1.0 + term, 0.05))
    for _ in range(n_iter):
        x = r0_mm * np.sqrt(max(1.0 - term * float(i0(k * x)), 0.05))
        y = r0_mm * np.sqrt(max(1.0 + term * float(i0(k * y)), 0.05))
    return float(x), float(y)


def modulation_consistency(r0_mm: float, A10: float, m: float,
                           length_mm: float) -> tuple[float, float]:
    """(A10_theory, relative deviation) from the card's (R0, m).

    Makes the card's ``m`` operand LIVE as a physics cross-check: the
    two-term theory ties the triplet (R0, m, A10) together via
    a = x_lim(cos kz = 1) and A10 = (m²−1)/(m²·I0(ka)+I0(m·k·a))
    (Crandall/Wangler; verified against the PXIE table where card A10
    tracks the theory to a few %).  A large deviation means the card
    is internally inconsistent — the fields still follow the card A10
    (TW convention), but the vane-tip apertures derived from it would
    not correspond to the stated modulation.
    """
    from scipy.special import i0
    if m <= 1.0 or A10 <= 0.0 or length_mm <= 0 or r0_mm <= 0:
        return 0.0, 0.0
    a_mm, _ = vane_apertures(r0_mm, A10, length_mm, cell_type=-2,
                             z_local_mm=0.0)
    k = np.pi / length_mm
    ka = k * a_mm
    denom = m * m * float(i0(ka)) + float(i0(m * ka))
    a10_th = (m * m - 1.0) / denom if denom > 0 else 0.0
    dev = abs(a10_th - A10) / A10 if A10 > 0 else 0.0
    return float(a10_th), float(dev)


def synchronous_phase_from_lengths(lengths_mm, A10s, voltages_V,
                                   mass_MeV: float, charge: float,
                                   wavelength_mm: float):
    """Per-cell θs implied by the deck's OWN cell lengths.  No free params.

    An RFQ cell is synchronous when ``L = β·λ/2``, so the deck's cell
    LENGTHS already encode the design velocity profile, and therefore the
    per-cell energy gain the design intends::

        β(n)  = 2·L(n)/λ            γ(n) = (1-β²)^(-1/2)
        ΔW(n) = W(n+1) − W(n)

    Integrating the two-term on-axis field over one cell with the phase
    cursor running 0→180° gives the closed form
    ``ΔW = |q|·(π/4)·A10·V·cos θs`` (the sin·sin integral over a half
    period), hence::

        cos θs(n) = 4·ΔW(n) / (π·|q|·A10(n)·V(n))

    This is a *derivation*, not a fit: every input is a card operand.
    Validated on the PXIE deck — over the 131 cells with A10 > 0.02 the
    derived value tracks the card's θs to **+1.8 ± 1.8°** (the small
    positive bias is the transit-time factor the closed form drops).

    Its purpose is to make the θs operand LIVE, the same way
    :func:`modulation_consistency` makes ``m`` live.  On the PXIE deck it
    exposes five cells (195–199) whose card θs is −90° — i.e. "do not
    accelerate" — while A10, m, L, dP *and* the ``.vane`` geometry all
    continue their smooth ramp; the derived value there is ≈ −24°.

    SIGN.  ``cos`` is even, so this determines only **|θs|**.  The value
    returned uses the conventional below-crest sign (θs ≤ 0), which every
    phase-stable RFQ design uses; callers that must tolerate an
    above-crest deck should compare magnitudes (as
    :mod:`linac_gen.io.rfq_phase_repair` does) rather than assume it.

    Returns
    -------
    (phi_deg, valid) : two ``(N,)`` arrays.  ``phi_deg`` is NaN wherever
    ``valid`` is False — the last cell (no successor to give ΔW), cells
    with A10 ≤ 0 (no accelerating field to solve for), and cells whose
    implied |cos θs| > 1 (the deck's lengths cannot be delivered by this
    cell's A10·V, e.g. across an exit matcher).
    """
    L = np.asarray(lengths_mm, dtype=float)
    A = np.asarray(A10s, dtype=float)
    V = np.asarray(voltages_V, dtype=float)
    n = L.size
    phi = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    if n < 2 or wavelength_mm <= 0 or mass_MeV <= 0:
        return phi, valid
    beta = 2.0 * L / wavelength_mm
    ok_beta = (beta > 0.0) & (beta < 1.0)
    gamma = np.where(ok_beta, 1.0 / np.sqrt(1.0 - np.clip(beta, 0.0,
                                                          0.999999) ** 2),
                     np.nan)
    W_MeV = (gamma - 1.0) * mass_MeV
    dW_eV = np.diff(W_MeV) * 1e6                      # across cell n
    denom = np.pi * abs(charge) * A[:-1] * V[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_phi = np.where(denom > 0, 4.0 * dW_eV / denom, np.nan)
    # The SUCCESSOR's length only encodes a design velocity if it is
    # itself an accelerating cell.  A matcher (A10 = 0) is length-set by
    # geometry, not synchronism, so ΔW across the cell before it is
    # meaningless — on the synthetic demo deck that boundary otherwise
    # manufactures a spurious θs = −90 for a perfectly good −28 card.
    good = (np.isfinite(cos_phi) & (np.abs(cos_phi) <= 1.0)
            & ok_beta[:-1] & ok_beta[1:] & (A[:-1] > 0.0) & (A[1:] > 0.0))
    # TraceWin's sign convention: below-crest, i.e. theta_s <= 0.
    phi[:-1] = np.where(good, -np.degrees(np.arccos(np.clip(cos_phi,
                                                            -1.0, 1.0))),
                        np.nan)
    valid[:-1] = good
    return phi, valid


# TW-matrix-calibrated smooth corrections (2026-07-30, vane-field
# campaign).  Fitting per-cell effective (quad, accel) scales to ALL
# 203 PXIE ground-truth matrices shows small, SMOOTH residual trends
# beyond the two-term model — the footprint of the Toutatis vane-field
# solution (an FD Laplace solve of the true Tc-arc electrode predicts
# corrections of the same sign and cell-trend, validating the physical
# origin; the constant-Tc model overshoots, so the magnitudes are
# calibrated, mode-faithfully, to TW itself — same epistemic status as
# the sin-phase/K2/S resolutions above, and same method precedent as
# the fnalscl T(β) synchronism factor).  Parameterised by the card A10
# (monotone along any RFQ after the shaper).  Applying the RAW noisy
# per-cell fits instead of this smooth form makes the beam WORSE
# (cell-to-cell jitter breaks AG coherence: envelope y 7.4→16 %) — a
# pinned lesson.  Effect: envelope σ vs the vane-based TW export drops
# from 6.6/7.4 % to 1.1/1.7 % mean.
#
# EPISTEMIC CAUTIONS (adversarial review 2026-07-30):
# * 1.1 % is BELOW the ~4 % with which TW's OWN matrices reproduce the
#   same chart — part of the correction necessarily absorbs residuals
#   specific to this chart, not transferable physics.  Treat the
#   sub-floor agreement as a fit property, not extra fidelity.
# * This is SINGLE-PROJECT calibration (PXIE: 162.5 MHz, r0 5.576 mm,
#   Tc/r0 = 0.75, V 60 kV).  For a different RFQ the correction is a
#   bounded (≤1 % quad, ≤2.5 % defocus) educated extrapolation of the
#   right sign; a second machine's ground truth has not validated it.
#   Set TW_CALIBRATION_ENABLED = False for an uncalibrated exact-annex
#   tw2term.
TW_CALIBRATION_ENABLED = True
_TWCAL_A10   = (0.00, 0.02, 0.134, 0.47, 0.69)
_TWCAL_QUAD  = (1.000, 0.997, 0.990, 0.997, 1.000)
_TWCAL_ACCEL = (1.000, 1.000, 0.984, 1.012, 1.025)


def tw_calibration(A10: float) -> tuple[float, float]:
    """(quad_scale, accel_scale) of the smooth TW-matrix calibration."""
    if not TW_CALIBRATION_ENABLED:
        return 1.0, 1.0
    a = abs(float(A10))
    return (float(np.interp(a, _TWCAL_A10, _TWCAL_QUAD)),
            float(np.interp(a, _TWCAL_A10, _TWCAL_ACCEL)))


def step_kicks(voltage_V: float, r0_mm: float, A10: float, length_mm: float,
               phase_rad, gamma_s: float, beta_s: float, dz_mm: float,
               C1: float, C2: float, S: float, C3: float,
               mass_MeV: float, gx: float | None = None,
               gy: float | None = None):
    """Per-substep kick factors in HELIX units.

    Returns ``(kx1, ky1, K1, K2)``:
      * ``kx1``/``ky1`` — Δx'/x per substep, in 1/mm (multiply x [mm] to
        get the x' kick in RADIANS);
      * ``K1`` — longitudinal (δz, δ) kick in 1/m (TW native);
      * ``K2`` — dimensionless momentum-rescale diagonal.

    ``phase_rad`` may be a scalar (matrix path, synchronous phase) or an
    ndarray (multiparticle path, per-particle phases) — outputs
    broadcast accordingly.

    ``gx``/``gy`` — optional vane-geometry gradient profile (normalised
    per-plane linear gradients at this z, see rfq_geometry_profile.py).
    When given, they replace the card's constant quad strength AND the
    analytic RF-defocus term: the measured profile's in-cell structure
    already carries the defocusing field, and its product with the RF
    clock reproduces the synchronous defocus average through the
    substep integration (identical-beam benchmark vs Toutatis,
    2026-08-02).  The longitudinal channel (K1, K2) stays on the card.
    """
    L_m = length_mm * 1e-3
    r0_m = r0_mm * 1e-3
    dz_m = dz_mm * 1e-3
    mc2_eV = mass_MeV * 1e6
    pref = dz_m / (gamma_s * beta_s * beta_s * 2.0 * mc2_eV)
    sin_ph = np.sin(phase_rad)
    cos_ph = np.cos(phase_rad)
    # Smooth TW calibration (see tw_calibration above).  Scope is
    # deliberate: quad_c on the electric-quadrupole term, accel_c on
    # the TRANSVERSE RF-defocus only — the longitudinal channel (K1,
    # K2, and the reference E_z ramp in RfqCell) stays on the card A10,
    # which already reproduces TW exactly (ramp to all digits, z-blocks
    # to 1.2 %).  That is precisely the split the per-cell fit measured.
    quad_c, accel_c = tw_calibration(A10)
    if gx is not None:
        # Geometry-profile mode: the measured per-plane gradients carry
        # the C1 ramp, the intra-cell breathing AND the defocus field.
        base = (2.0 * quad_c * S * (voltage_V / (r0_m * r0_m))
                * sin_ph)                                       # V/m²
        kx1_per_m = -pref * (base * gx)                         # 1/m
        ky1_per_m = -pref * (base * gy)                         # 1/m
    else:
        quad = (2.0 * quad_c * S * (voltage_V / (r0_m * r0_m))
                * C1 * sin_ph)                                  # V/m²
        defoc = ((np.pi / L_m) ** 2 * (accel_c * A10 * voltage_V / 2.0)
                 * cos_ph * C2)                                 # V/m²
        kx1_per_m = -pref * (quad - defoc)                      # 1/m
        ky1_per_m = -pref * (-quad - defoc)                     # 1/m
    K1 = -pref * (np.pi / L_m) ** 2 * A10 * voltage_V * C3 * cos_ph
    # K2 is the per-step momentum ratio p_in/p_out linearised — it must
    # follow the ENERGY GAIN phase (E_z ∝ sin φ), not K1's gradient
    # phase.  Adversarial check (2026-07-30): over the 203 PXIE cells
    # TW's cumulative Π det(x-block) = 0.12364 vs physical
    # (βγ)_in/(βγ)_out = 0.12379; sin-phased K2 reproduces 0.12372,
    # the earlier cos-phased form gave 0.02764 (4.5× overdamped).
    K2 = 1.0 - pref * A10 * voltage_V * (np.pi / L_m) * C3 * sin_ph
    # 1/m → 1/mm for the transverse (x in mm, kick in rad): Δx'[rad] =
    # k[1/m]·x[m] = k[1/m]·1e-3·x[mm]
    return kx1_per_m * 1e-3, ky1_per_m * 1e-3, K1, K2
