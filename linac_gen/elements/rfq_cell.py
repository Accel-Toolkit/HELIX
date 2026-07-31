"""TraceWin-compatible RFQ_CELL element.

Implements the per-cell numerical integration documented in the TraceWin
user manual ("RFQ cell" / "Transport through a RFQ cell" section, with
equations rendered as images 265–301).  The model is *not* the textbook
Crandall 8-term potential expansion; rather, TraceWin truncates the
multipole expansion and tracks each cell as

  • an on-axis longitudinal sinusoidal field
        E_z(z, t) = (π · A₁₀ · V) / (2 L) · sin(π z / L) · sin(ω t_s + φ_s)
  • a per-Type linear transverse focusing kick parametrised by three
    numbers C₁(z), C₂(z), S — given by the manual for each cell-type
    flag (±2 = accelerating, ±3 = front-end / shaper, ±4 = transcell).

For each particle the cell is divided into N substeps; each substep is a
Strang Drift–Kick–Drift (manual image 269) with the kick amplitude

  k_x = −(|q|·dz / (γ·β²·m c²)) · [ cos(ω t_s + φ_s)·S·(V/2)·A·C₁(z)
                                    − (π/L)² · (A₁₀·V/2)·C₂(z) ]
  k_y = −(|q|·dz / (γ·β²·m c²)) · [−cos(ω t_s + φ_s)·S·(V/2)·A·C₁(z)
                                    − (π/L)² · (A₁₀·V/2)·C₂(z) ]

(reading from images 277, 278; the quadrupole part flips sign between x
and y, the RF defocusing is common-mode).  The DC quadrupole coefficient
``A`` is *not* a separate user input in TraceWin; we use the standard
two-term approximation ``A = (1 − A₁₀)/R₀²`` (Wangler eq. 8.31 limit;
also matches Crandall's RFQUIK convention when no vane geometry is
supplied).  Power users can override this via the ``A_quad`` constructor
keyword.

Reference particle update per substep (manual images 265, 267, 268):

  W ← W + |q|·dz·E_z(z, t_s)
  Φ ← Φ + dz · 2π / (β·λ)
  z ← z + dz

where β, γ are recomputed after each W update so the longitudinal field
phasing self-consistently tracks the changing velocity.

This module honours the same conventions as the rest of LG: lengths in
mm, angles in mrad, kinetic energies in MeV.
"""
from __future__ import annotations

import numpy as np

from linac_gen.core.constants import C_LIGHT, E_CHARGE
from linac_gen.elements.base import FieldMapElement


class RfqCell(FieldMapElement):
    """One RFQ cell, tracked à la TraceWin's ``RFQ_CELL``.

    Parameters
    ----------
    name : str
    voltage_V : float
        Inter-vane voltage V in volts.  TraceWin's ``V`` operand.
    r0_mm : float
        Vane mean radius R₀ in millimetres.
    A10 : float
        Longitudinal Fourier coefficient (TraceWin's acceleration parameter).
        Dimensionless; sets the magnitude of E_z via
        ``E_z(z,t) = π A₁₀ V / (2 L) · sin(πz/L) · sin(ω t + φ_s)``.
    modulation : float
        Vane modulation factor m (≥ 1).  Used to derive the DC quadrupole
        coefficient via the two-term approximation when ``A_quad`` is None.
    length_mm : float
        Cell length L (mm).  TraceWin allows arbitrary L; no check vs
        β·λ/2 is enforced because the manual does not enforce it.
    phi_s_deg : float
        Synchronous phase φ_s in degrees.
    cell_type : int
        ±2 (accelerating), ±3 (front-end / shaper), ±4 (transcell).  The
        sign selects which neighbour cell the transverse model couples to.
    Tc_mm : float, optional
        Transverse curvature (TraceWin's ``Tc``).  Currently unused —
        accepted for parser compatibility.
    dP_deg : float, optional
        Output-phase shift (TraceWin's ``dP``).  Applied to the running
        synchronous phase at the cell exit.
    n_steps : int
        Sub-steps per cell.  When ``None`` (default), auto-selects
        ``max(20, ceil(L_mm / 0.1))`` so every cell gets ≤0.1 mm per
        substep regardless of length — required because RFQ cells span
        ~7 mm (front-end) to ~60 mm (output) within a single design,
        and a fixed ``n_steps`` either over-resolves the short cells or
        leaves the long cells with multi-percent integration error.
        Pass an explicit integer to override.
    type_prev, type_next : int
        Neighbouring cell types (used by S = −sign(type[n±1])).  Set by
        the parser when chaining a sequence of ``RFQ_CELL`` lines.  Both
        default to ``cell_type`` for an isolated cell.
    A_quad : float, optional
        Override for the DC quadrupole coefficient.  When None, defaults
        to the two-term approximation ``(1 − A₁₀) / R₀²`` (units 1/mm²).
    aperture : float
        Scalar aperture radius (mm) consumed by the tracker's generic
        end-of-element check; 0 disables THAT check.  NOTE: under
        ``field_model="tw2term"`` transverse losses are governed by the
        per-substep vane-tip profile x_lim(z)/y_lim(z) REGARDLESS of
        this attribute (legacy models keep their historical no-loss
        behaviour).
    field_model : str
        Physics-model selector.

        ``"tw2term"`` (DEFAULT since 2026-07-30, Phase 5 of the RFQ
        overhaul) — the exact TraceWin annex per-step algorithm with
        every transcription ambiguity resolved against TW's own per-cell
        matrices (``linac_gen/elements/rfq_coefficients.py`` carries the
        full derivation and calibration record; median per-cell relative
        error ~3 % over the 203 PXIE cells — against a VANE-based
        reference, see the coefficients module).  Differences vs the
        legacy model: sin-phased quadrupole of strength V/R₀² (no
        (1−A₁₀) reduction), cos-phased RF defocus
        (π/L)²(A₁₀V/2)·sin(πz/L), exact
        ±3/±4 C₁/C₂ forms, per-substep synchronous-γ advance, TW's K₂
        momentum rescale in the matrix path, and — critically for
        multiparticle capture — the per-particle longitudinal phase
        slip that the legacy track path lacked entirely (without it a
        DC beam can never bunch: exit σ_φ stayed at the injected 104°).

        ``"2term"`` (legacy fallback) keeps ``A_quad = (1−A₁₀)/R₀²``
        and ``S = −sign(Type)`` — the empirically-calibrated 2026-04
        path, kept bit-identical; its multiparticle path has no phase
        slip and no losses.  Diagnostic-only opt-ins:

          * ``"crand_x"`` — uses Crandall X = (m²−1)/(m²·I₀(ka)+I₀(mka))
            with ``a = R₀/√m``, ``k = π/L`` (Crandall LANL LA-11968-MS
            Eq. 2-4) AND flips ``S`` for ±2 cells.  Per-cell calibration
            probe (``diag_rfq_single_cell_calib.py`` V_CRAND_X) shows
            mean |Δ_4×4| = 25.8 vs TW Transfer_matrix1.dat (3× better
            than 2term).  But the envelope solve diverges: the smaller
            X_crand strength + S flip pushes outside the Mathieu first
            stability band → σ blowup to ~10² mm by RFQ exit (verified
            ``diag_rfq_env_crand_x.py``, 2026-04-28).
          * ``"crand_x_noflip"`` — Crandall X without S flip.  Same
            instability as ``crand_x`` (just swaps which plane blows up).
          * ``"pdf_2term"`` — TraceWin manual page 147 LITERAL: factor
            ``|q|·dz/(γ_s β_s²·2mc²)`` (note the /2 — LG default omits
            it), AG term ``S·V/r_0²·(1-A10)·C_1``, defoc term
            ``(π/L)²·A_10·V·C_2/4`` (LG uses V/2), C_2 = sin(πz/L) for
            ±2 (LG uses cos), S = -sign(Type).  This is the formula set
            extracted from ``Tracewin_code/tracewin.pdf`` p.147 with
            PDF text extraction.  Per-cell calibration (V_PDF, V_PDF_FLIP):
            mean |Δ_4×4| = 71-52 vs LG's 70 — same ballpark per cell.
            Envelope blows up to σ ~10³ mm (verified
            ``diag_rfq_env_crand_x.py`` 2026-04-28): the literal formula
            with C_2=sin doesn't have LG's calibrated cell-integrated
            cancellation, so AG balance breaks → Mathieu instability.

        All three opt-ins are kept for diagnostic comparison only.  Until
        the unbridgeable gap between manual formulas and TW matrix output
        is resolved (likely requires TraceWin source-code inspection),
        do NOT use them for production envelope or MP runs.
    """

    def __init__(self,
                 name: str,
                 voltage_V: float,
                 r0_mm: float,
                 A10: float,
                 modulation: float,
                 length_mm: float,
                 phi_s_deg: float,
                 cell_type: int,
                 Tc_mm: float = 0.0,
                 dP_deg: float = 0.0,
                 n_steps: int | None = None,
                 type_prev: int | None = None,
                 type_next: int | None = None,
                 A_quad: float | None = None,
                 aperture: float = 0.0,
                 field_model: str = "tw2term"):
        # Auto-pick n_steps when not supplied: target ≤0.1 mm per substep,
        # never less than 20 substeps for very short cells.  See class
        # docstring for the rationale.
        if n_steps is None:
            n_steps = max(20, int(np.ceil(float(length_mm) / 0.1)))
        super().__init__(name=name, length=float(length_mm),
                         aperture=aperture, n_steps=int(n_steps))
        self.voltage_V = float(voltage_V)
        self.r0_mm = float(r0_mm)
        self.A10 = float(A10)
        self.modulation = float(modulation)
        self.phi_s_deg = float(phi_s_deg)
        self.cell_type = int(cell_type)
        self.Tc_mm = float(Tc_mm)
        self.dP_deg = float(dP_deg)
        self.type_prev = int(type_prev) if type_prev is not None else self.cell_type
        self.type_next = int(type_next) if type_next is not None else self.cell_type
        # DC quadrupole coefficient.  TraceWin computes this internally
        # from m, R₀ via the standard 2-term Bessel-function approximation;
        # in the small-modulation limit (m → 1) this collapses to
        # A_quad ≈ 1/R₀² with a (1 − A₁₀) prefactor for the residual
        # acceleration efficiency.  Power users with vane geometry can
        # override.  Units: 1/mm².
        if A_quad is None:
            denom = self.r0_mm * self.r0_mm
            self._A_quad = max(0.0, (1.0 - self.A10)) / denom if denom > 0 else 0.0
        else:
            self._A_quad = float(A_quad)
        # Opt-in alternative DC quad coefficient based on the classical
        # Crandall X = (m²−1)/(m²·I₀(ka)+I₀(mka)) with a = R₀/√m, k = π/L
        # (Crandall LANL LA-11968-MS Eq. 2-4).  Per-cell calibration probe
        # (diag_rfq_single_cell_calib.py V_CRAND_X, 2026-04-28) shows this
        # form gives mean per-cell |Δ_4×4| = 25.8 vs TWOTERM matrix output
        # (3× better than the (1-A10)/R₀² short form).  Combined with the
        # +sign(Type) S convention for ±2 cells, the composed RFQ matrix
        # M[0,0] = +3.25 vs TW −0.05 (closer than any other variant tried).
        # ``field_model`` selector is opt-in; default ``"2term"`` keeps the
        # existing (1-A10)/R₀² short form and S = -sign(Type) bit-for-bit.
        if field_model not in {"2term", "tw2term", "crand_x",
                               "crand_x_noflip", "pdf_2term"}:
            raise ValueError(f"field_model must be one of "
                             f"{{'2term','tw2term','crand_x',"
                             f"'crand_x_noflip','pdf_2term'}}, "
                             f"got {field_model!r}")
        self.field_model = field_model
        if field_model in ("crand_x", "crand_x_noflip"):
            from scipy.special import iv as _iv
            m = self.modulation
            if self.length > 0 and m >= 1.0 and self.r0_mm > 0:
                k_pi = np.pi / self.length
                a_v = self.r0_mm / np.sqrt(m)
                ka = k_pi * a_v
                mka = m * ka
                denom_X = m * m * float(_iv(0, ka)) + float(_iv(0, mka))
                self._X_crand = (m * m - 1.0) / denom_X if denom_X > 0 else 0.0
                self._A_quad_eff = self._X_crand / (self.r0_mm * self.r0_mm)
            else:
                self._X_crand = max(0.0, 1.0 - self.A10)
                self._A_quad_eff = self._A_quad
        else:
            self._X_crand = max(0.0, 1.0 - self.A10)
            self._A_quad_eff = self._A_quad
        # Physics sanity warnings (Phase 3, 2026-07-30): the card's
        # ``m`` operand becomes LIVE as a consistency check against
        # (R0, A10) via the two-term closure, and m > 3.2 flags the
        # documented breakdown of TraceWin's tabulated-A10 treatment
        # (CEA forum).  Warnings only — the fields always follow the
        # card A10, exactly as TraceWin does.
        import warnings as _warnings
        if self.modulation > 3.2:
            _warnings.warn(
                f"{name}: vane modulation m = {self.modulation:.3f} > 3.2 "
                "— the two-term potential (and TraceWin's own tabulated "
                "A10 treatment) is documented to break down here; "
                "results are unreliable.", stacklevel=2)
        if self.A10 > 0.02 and self.modulation > 1.0:
            from linac_gen.elements.rfq_coefficients import (
                modulation_consistency,
            )
            a10_th, dev = modulation_consistency(
                self.r0_mm, self.A10, self.modulation, self.length)
            if dev > 0.15:
                _warnings.warn(
                    f"{name}: card A10 = {self.A10:.4f} deviates "
                    f"{dev*100:.0f} % from the two-term value "
                    f"{a10_th:.4f} implied by (R0 = {self.r0_mm} mm, "
                    f"m = {self.modulation}) — the card triplet is "
                    "internally inconsistent; fields follow the card "
                    "A10.", stacklevel=2)
        # Substep cursor — set to 0 at element entry (advance_ref / track
        # share this convention with the rest of LG's field-map elements).
        self._step_idx = 0
        # Per-run cursors (re-armed at each cell entry; initialised here
        # so an out-of-order first call degrades gracefully instead of
        # raising AttributeError — adversarial hardening 2026-07-30).
        self._z_cursor_mm = 0.0
        self._tw_ts_deg = 0.0
        self._adv_cell_phi_deg = 0.0
        # Cell-local synchronous phase, in degrees, advanced per substep
        # by 360·dz/(β·λ).  TraceWin's E_z(z,t) = (πA₁₀V/2L)·sin(πz/L)·
        # sin(ω t_s + φ_s) uses t_s **relative to cell entrance** (so that
        # at z=L/2 with L=βλ/2 the argument is π/2 + φ_s and the peak
        # gain is cos(φ_s)).  We must NOT use the global ref.phi_s here,
        # which accumulates across the whole lattice and would put each
        # cell at an arbitrary running phase.
        self._cell_phi_deg: float = 0.0

    # ------------------------------------------------------------------
    # Per-cell type C₁(z), C₂(z), S coefficients (manual images 281–290).
    # z_local is the substep midpoint coordinate measured from the cell
    # entrance, in the same units as ``self.length`` (mm).
    # ------------------------------------------------------------------
    def _type_coeffs(self, z_local_mm: float) -> tuple[float, float, float]:
        L = self.length
        if L <= 0:
            return 1.0, 0.0, 0.0
        arg = np.pi * z_local_mm / L
        sign_type = 1 if self.cell_type > 0 else -1
        abs_type = abs(self.cell_type)

        if abs_type == 2:
            # Accelerating cell.  Manual: C₁=1 (img 281), C₂=cos(πz/L)
            # (img 282 — small-PNG OCR ambiguous between sin/cos; cos
            # is the physically-correct radial-defocusing envelope from
            # ∂E_z/∂z and the sin form blows up to 10²⁰ mm at mid-RFQ),
            # S=−sign(Type) (img 283).
            #
            # 2026-04-28: per-cell matrix audit vs TraceWin TWOTERM
            # Transfer_matrix1.dat (LEBT+RFQ, .vane disabled) shows the
            # current `S = -sign_type + (1-A10)/r₀²` formula gives the
            # OPPOSITE per-cell M[1,0] sign from TraceWin (LG: focus,
            # TW: defocus for type -2).  However, every attempted fix —
            # flip S, scale magnitude, swap C_2 sin/cos — that brings
            # the per-cell matrix closer to TW destabilises the
            # envelope, because the cell-to-cell AG cancellation
            # requires this specific (S, magnitude, C_2) combination.
            # See diag_rfq_single_cell_calib.py: V7, V13, V15_a30 all
            # blow up envelope σ_x_exit from 1.5 mm to 8-10 mm.  The
            # coordinated formula fix needs the leading common-mode
            # term from manual image 276/277 (suppressed prefactor)
            # which we cannot read at the available resolution.  Until
            # higher-res images of img 276/277 are available, the
            # current calibration is the empirical stability ceiling.
            C1 = 1.0
            C2 = float(np.cos(arg))
            S = -sign_type
        elif abs_type == 3:
            # Front-end / shaper.  Manual: C₂=0 (img 285) — no spatial
            # RF-defocusing inside ±3 cells; the kick comes purely from
            # the C₁ quadrupole term.  C₁ is a Bessel-like normalised
            # expression (img 284 for +3, img 287 for −3) which we don't
            # decode in full here; placeholder C₁=1 captures the leading
            # behaviour and the cell's own dW from on-axis E_z still
            # tracks correctly via _Ez_onaxis.  S references the *next*
            # cell type for +3 (img 286) and the *previous* for −3
            # (img 288).
            C1 = 1.0
            C2 = 0.0
            if sign_type > 0:
                S = -1 if self.type_next > 0 else +1
            else:
                S = -1 if self.type_prev > 0 else +1
        elif abs_type == 4:
            # Transcell.  Manual differentiates +4 vs −4:
            #   +4: C₁=1 (img 281), C₂=½·[cos(πz/L)+1] (img 289),
            #        S=−sign(Type[n+1]) (img 286).
            #   −4: C₁=1 (img 290), C₂=½·[1−cos(πz/L)] (img 291),
            #        S=−sign(Type[n−1]) (img 288).
            C1 = 1.0
            if sign_type > 0:
                C2 = 0.5 * (float(np.cos(arg)) + 1.0)
                S  = -1 if self.type_next > 0 else +1
            else:
                C2 = 0.5 * (1.0 - float(np.cos(arg)))
                S  = -1 if self.type_prev > 0 else +1
        else:
            # Unknown cell type — pass-through (no transverse kick).
            C1 = 0.0
            C2 = 0.0
            S = 0.0
        return C1, C2, S

    # ------------------------------------------------------------------
    # On-axis longitudinal field (manual image 268).
    # ------------------------------------------------------------------
    def _Ez_onaxis(self, z_local_mm: float, phase_rad: float) -> float:
        L = self.length
        if L <= 0:
            return 0.0
        # E_z(z, t) = (π A₁₀ V) / (2 L) · sin(π z / L) · sin(ω t_s + φ_s)
        # V is the inter-vane voltage (volts), L is mm; the prefactor has
        # units V/mm.  The caller multiplies by dz [m] · |q| [e] to get
        # dW in eV (then 1e-6 → MeV) so we keep V in volts and L in mm
        # and return the field in V/mm.
        return ((np.pi * self.A10 * self.voltage_V) / (2.0 * L)
                * np.sin(np.pi * z_local_mm / L)
                * np.sin(phase_rad))

    # ------------------------------------------------------------------
    # FieldMapElement contract
    # ------------------------------------------------------------------
    def track_rk4(self, beam, ds: float) -> None:
        """Advance the beam by one slice of length ds (mm).

        The slice corresponds to a single substep of TraceWin's per-cell
        Strang DKD; the lattice tracker breaks the full cell into
        ``self.n_steps`` slices and calls this once per slice.
        """
        if self.field_model == "tw2term":
            self._track_tw2term(beam, ds)
            return
        ref = beam.ref
        ds_m = ds * 1e-3
        ds_half_m = 0.5 * ds_m

        # Accumulated z cursor — NOT ``_step_idx * ds``: the tracker's
        # trailing substeps (n_int % sc_every != 0) arrive with a
        # DIFFERENT ds than the paired ds/2 calls, and the multiply-out
        # form then lands past the cell end (adversarial finding
        # 2026-07-30; ncells.py documents the same rule).  Identical
        # values whenever every call shares one ds.
        if self._step_idx == 0:
            self._cell_phi_deg = 0.0
            self._z_cursor_mm = 0.0
        z_local_mm = self._z_cursor_mm + ds * 0.5
        self._z_cursor_mm += ds
        self._step_idx += 1

        beta = ref.beta
        gamma = ref.gamma
        mass_MeV = ref.species.mass
        charge = ref.species.charge      # signed, in units of e
        wavelength_mm = ref.wavelength   # mm

        # ---- Reference advance (on-axis E_z, midpoint) ----------------
        if wavelength_mm > 0:
            # Cell-local synchronous-phase advance up to the substep midpoint.
            dphi_to_mid = 180.0 * ds * 0.5 / (beta * wavelength_mm)
        else:
            dphi_to_mid = 0.0
        # Cell-local Φ at the substep midpoint, plus the cell's φ_s offset.
        phi_s_mid_deg = self._cell_phi_deg + dphi_to_mid + self.phi_s_deg
        phi_s_mid_rad = np.deg2rad(phi_s_mid_deg)

        E_z = self._Ez_onaxis(z_local_mm, phi_s_mid_rad)   # V/mm
        # Manual image 265: dW = |q| · dz · E_z.  TraceWin uses the
        # *absolute value* of the species charge here so the user-supplied
        # φ_s gives acceleration (cos φ_s > 0) for protons and H- alike.
        # The species sign convention is absorbed into the user's choice
        # of V; flipping the species sign does NOT flip the physical
        # acceleration for the same input numbers.
        q_abs = abs(charge)
        dW_ref_MeV = q_abs * E_z * ds * 1e-6
        ref.w_kin += dW_ref_MeV
        ref.s += ds
        # Advance both the global ref phase (used by downstream elements)
        # and the cell-local phase (used by *this* cell's next substep).
        if wavelength_mm > 0:
            full_step_dphi = 360.0 * ds / (beta * wavelength_mm)
            ref.phi_s += full_step_dphi
            self._cell_phi_deg += full_step_dphi

        alive = beam.alive_mask
        if not np.any(alive):
            # Still advance the ref so downstream cells stay synchronised.
            return

        # ---- Half-drift transverse (Strang) ---------------------------
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_half_m

        # ---- Per-particle longitudinal kick ---------------------------
        # Particle's instantaneous phase = sync phase + Δφ (deg → rad).
        dphi_deg = beam.particles[alive, 4]
        # Re-evaluate E_z at each particle's own phase so off-bunch
        # particles see different longitudinal fields.
        phi_part_rad = phi_s_mid_rad + np.deg2rad(dphi_deg)
        E_z_part = ((np.pi * self.A10 * self.voltage_V) / (2.0 * self.length)
                    * np.sin(np.pi * z_local_mm / self.length)
                    * np.sin(phi_part_rad))
        # Same |q| convention as the reference (manual image 265).
        dW_part_MeV = q_abs * E_z_part * ds * 1e-6
        beam.particles[alive, 5] += dW_part_MeV - dW_ref_MeV

        # Adiabatic damping: as p_z grows the geometric x' shrinks.
        if beta > 0 and gamma > 0 and mass_MeV > 0:
            damp = 1.0 / (1.0 + dW_part_MeV / (beta * beta * gamma * mass_MeV))
            beam.particles[alive, 1] *= damp
            beam.particles[alive, 3] *= damp

        # ---- Per-particle transverse kick (manual images 277, 278) ----
        # k_x = -[ cos(ω t + φ_s) · S · (V/2) · A · C₁
        #         - (π/L)² · (A₁₀ V / 2) · C₂ ] · (|q| ds / γ β² m c²)
        # k_y mirrors with quadrupole sign flipped.  The RF time factor
        # cos(ω t + φ_s) is **per-particle** — each particle uses its
        # own phase φ_s + δφ.  For a DC beam this matters: particles at
        # different phases see opposite-sign quadrupole kicks, and the
        # AG focusing only emerges from the bunch that the RFQ self-
        # bunches longitudinally.  Treating all particles with the
        # synchronous-phase cos factor gave σ blow-up by ~30× through
        # the RFQ.
        C1, C2, S = self._type_coeffs(z_local_mm)
        # crand_x model: flip S on ±2 to match TraceWin's TWOTERM matrix
        # output (per-cell calibration probe — V13/V_CRAND_X variant).
        if self.field_model == "crand_x" and abs(self.cell_type) == 2:
            S = -S    # ``crand_x_noflip`` deliberately keeps LG's S convention
        cos_phase_part = np.cos(phi_part_rad)   # per-particle, length = #alive
        if self.field_model == "pdf_2term":
            # PDF page 147 LITERAL: factor /2mc² (not /mc²), V·A_10/4 in defoc
            # (not V·A_10/2), C_2 = sin(πz/L) for ±2 (not cos), S = -sign(type).
            r02 = self.r0_mm * self.r0_mm
            A01 = max(0.0, 1.0 - self.A10)
            if abs(self.cell_type) == 2:
                C2_eff = float(np.sin(np.pi * z_local_mm / self.length))
            else:
                C2_eff = C2
            ag = S * self.voltage_V / r02 * A01 * C1                    # V/mm²
            defoc_pdf = ((np.pi / self.length) ** 2
                         * self.A10 * self.voltage_V * 0.25 * C2_eff)   # V/mm²
            if beta > 0 and gamma > 0 and mass_MeV > 0:
                factor_pdf = abs(charge) * ds / (gamma * beta * beta * 2.0 * mass_MeV * 1e6)
                kx_per_x = -factor_pdf * cos_phase_part * (ag - defoc_pdf)
                ky_per_y = -factor_pdf * cos_phase_part * (-ag - defoc_pdf)
                xs_mm = beam.particles[alive, 0]
                ys_mm = beam.particles[alive, 2]
                beam.particles[alive, 1] += kx_per_x * xs_mm * 1e3
                beam.particles[alive, 3] += ky_per_y * ys_mm * 1e3
        else:
            # Volts per mm² — A_quad has units 1/mm² and (π/L)² has 1/mm².
            rf_quad_part = cos_phase_part * S * (self.voltage_V * 0.5) * self._A_quad_eff * C1   # V/mm²
            rf_defoc     = (np.pi / self.length) ** 2 * (self.A10 * self.voltage_V * 0.5) * C2  # V/mm²

            # Kick coefficient: |q| ds / (γ β² m c²)
            if beta > 0 and gamma > 0 and mass_MeV > 0:
                factor = abs(charge) * ds / (gamma * beta * beta * mass_MeV * 1e6)
                # rf_quad_part is per-particle; rf_defoc is scalar (z-dependent only).
                kx_per_x = -factor * (rf_quad_part - rf_defoc)   # 1/mm  (per-particle vector)
                ky_per_y = -factor * (-rf_quad_part - rf_defoc)  # 1/mm

                xs_mm = beam.particles[alive, 0]
                ys_mm = beam.particles[alive, 2]
                beam.particles[alive, 1] += kx_per_x * xs_mm * 1e3   # rad → mrad
                beam.particles[alive, 3] += ky_per_y * ys_mm * 1e3

        # ---- Half-drift transverse (Strang) ---------------------------
        beam.particles[alive, 0] += beam.particles[alive, 1] * ds_half_m
        beam.particles[alive, 2] += beam.particles[alive, 3] * ds_half_m

    # ------------------------------------------------------------------
    # tw2term multiparticle path — the exact annex per-step algorithm
    # (see rfq_coefficients module docstring for the derivation).
    # ------------------------------------------------------------------
    def _track_tw2term(self, beam, ds: float) -> None:
        from linac_gen.elements.rfq_coefficients import (step_kicks,
                                                         type_coeffs)
        ref = beam.ref
        mass = ref.species.mass
        q_abs = abs(ref.species.charge)
        wl = ref.wavelength
        beta_in, gamma_in = ref.beta, ref.gamma
        if self.length <= 0 or beta_in <= 0:
            ref.s += ds
            return
        # Accumulated z cursor — see the legacy-path comment: the
        # tracker's trailing substeps may use a different ds.
        if self._step_idx == 0:
            self._z_cursor_mm = 0.0
        z_local_mm = self._z_cursor_mm + ds * 0.5
        self._z_cursor_mm += ds
        # Annex t_s cursor in RF degrees: initialised at the FIRST
        # substep midpoint with the CELL-entry velocity [image 281],
        # then advanced one full step with the post-step velocity
        # [image 252].  (Recomputing the half-step with the running β —
        # the legacy bookkeeping — integrates to the historical −0.12 %
        # ramp error; the annex cursor reproduces TW's 1.955717 MeV on
        # the PXIE deck to all printed digits.)
        if self._step_idx == 0:
            self._tw_ts_deg = (180.0 * ds / (beta_in * wl)) if wl > 0 else 0.0
        self._step_idx += 1
        phi_s_mid_rad = np.deg2rad(self._tw_ts_deg + self.phi_s_deg)

        # ---- reference update (identical physics to the legacy path —
        # the on-axis ramp is verified to −0.12 % vs TraceWin) ----------
        E_z = self._Ez_onaxis(z_local_mm, phi_s_mid_rad)
        dW_ref_MeV = q_abs * E_z * ds * 1e-6
        ref.w_kin += dW_ref_MeV
        ref.s += ds
        beta_out, gamma_out = ref.beta, ref.gamma
        if wl > 0:
            # annex image 252 advances t_s with the POST-step velocity
            full = 360.0 * ds / (beta_out * wl)
            ref.phi_s += full
            self._tw_ts_deg += full

        alive = beam.alive_mask
        if not np.any(alive):
            return
        P = beam.particles
        ds_half_mm = 0.5 * ds

        # ---- entry half-drift: transverse + LONGITUDINAL PHASE SLIP --
        # (the slip was entirely absent from the legacy track path — a
        # ΔW≠0 particle then never rotates in phase and a DC beam can
        # never bunch; measured exit σ_φ stayed at the injected 104°.)
        P[alive, 0] += P[alive, 1] * ds_half_mm * 1e-3
        P[alive, 2] += P[alive, 3] * ds_half_mm * 1e-3
        if wl > 0:
            r45_in = -360.0 * ds_half_mm / (beta_in ** 3 * gamma_in ** 3
                                            * mass * wl)
            P[alive, 4] += r45_in * P[alive, 5]

        # ---- midpoint kick, per-particle phases ----------------------
        phi_part = phi_s_mid_rad + np.deg2rad(P[alive, 4])
        E_z_part = ((np.pi * self.A10 * self.voltage_V)
                    / (2.0 * self.length)
                    * np.sin(np.pi * z_local_mm / self.length)
                    * np.sin(phi_part))
        dW_part_MeV = q_abs * E_z_part * ds * 1e-6
        P[alive, 5] += dW_part_MeV - dW_ref_MeV

        gamma_s = 0.5 * (gamma_in + gamma_out)
        beta_s = float(np.sqrt(max(1.0 - gamma_s ** -2, 0.0)))
        C1, C2, S, C3 = type_coeffs(self.cell_type, self.type_prev,
                                    self.type_next,
                                    z_local_mm / self.length)
        kx1, ky1, _K1, _K2 = step_kicks(
            self.voltage_V, self.r0_mm, self.A10, self.length,
            phi_part, gamma_s, beta_s, ds, C1, C2, S, C3, mass)
        xs_mm = P[alive, 0]
        ys_mm = P[alive, 2]
        P[alive, 1] += kx1 * xs_mm * 1e3          # rad → mrad
        P[alive, 3] += ky1 * ys_mm * 1e3
        # Adiabatic damping — the physical per-particle p-ratio (each
        # particle's own energy gain), the nonlinear generalisation of
        # TW's linear K₂ diagonal.
        if beta_s > 0 and gamma_s > 0 and mass > 0:
            damp = 1.0 / (1.0 + dW_part_MeV
                          / (beta_s * beta_s * gamma_s * mass))
            P[alive, 1] *= damp
            P[alive, 3] *= damp

        # ---- exit half-drift (post-kick kinematics) ------------------
        P[alive, 0] += P[alive, 1] * ds_half_mm * 1e-3
        P[alive, 2] += P[alive, 3] * ds_half_mm * 1e-3
        if wl > 0:
            r45_out = -360.0 * ds_half_mm / (beta_out ** 3
                                             * gamma_out ** 3 * mass * wl)
            P[alive, 4] += r45_out * P[alive, 5]

        # ---- LOSSES (tw2term only — the legacy path keeps its
        # historical no-loss behavior).  Two physical criteria per
        # substep, checked at the substep exit:
        #   * transverse: outside the actual vane tip x_lim(z)/y_lim(z)
        #     from the two-term equipotential condition (validated to
        #     0.03-0.14 % against the PXIE .vane table) — skipped in ±3
        #     front-end/exit cells whose real vanes flare to ~3.5·r0;
        #   * longitudinal: total kinetic energy below zero
        #     (back-accelerated junk — unphysical to keep tracking).
        # The DYNAC-style ±π phase-window kill is deliberately NOT
        # applied: with the phase slip active, uncaptured particles
        # drift in φ and are removed by the vanes on physical grounds.
        from linac_gen.elements.rfq_coefficients import vane_apertures
        alive_idx = np.where(alive)[0]
        if alive_idx.size:
            bad = P[alive_idx, 5] < -ref.w_kin          # W_total < 0
            if abs(self.cell_type) != 3:
                z_exit = min(z_local_mm + 0.5 * ds, self.length)
                x_lim, y_lim = vane_apertures(self.r0_mm, self.A10,
                                              self.length, self.cell_type,
                                              z_exit)
                bad |= (np.abs(P[alive_idx, 0]) > x_lim) \
                    | (np.abs(P[alive_idx, 2]) > y_lim)
            for pid in alive_idx[bad]:
                beam.record_loss(int(pid), ref.s, self.name)

    # ------------------------------------------------------------------
    # tw2term matrix path — exact annex loop; longitudinal built in TW
    # native (δz, δ) then converted to HELIX (Δφ deg, ΔW MeV) once per
    # slice with entry/exit kinematics.
    # ------------------------------------------------------------------
    def _fitted_matrix_slice_tw2term(self, ref, ds_mm: float,
                                     z_from: float) -> np.ndarray:
        from linac_gen.elements.rfq_coefficients import (step_kicks,
                                                         type_coeffs)
        if ds_mm <= 0 or self.length <= 0:
            return np.eye(6)
        n_full = max(1, self.n_steps)
        native_ds = self.length / n_full
        n_sub = max(1, int(round(ds_mm / native_ds)))
        dz = ds_mm / n_sub

        mass = ref.species.mass
        q_abs = abs(ref.species.charge)
        wl = ref.wavelength
        gamma_i = ref.gamma
        beta_i = ref.beta
        if beta_i <= 0 or mass <= 0:
            return np.eye(6)
        gamma_entry, beta_entry = gamma_i, beta_i

        # Annex t_s cursor (RF degrees) at the first substep midpoint of
        # this slice.  z<z_from history approximated with the slice-entry
        # β (same approximation as the legacy path; exact for whole-cell
        # slices, which is how the matrix path is normally called).
        if wl > 0 and beta_i > 0:
            ts_deg = (360.0 * z_from + 180.0 * dz) / (beta_i * wl)
        else:
            ts_deg = 0.0

        Mx = np.eye(2)
        My = np.eye(2)
        Mz = np.eye(2)                       # TW native (δz[m], δ=dp/p)
        for i in range(n_sub):
            z_local = z_from + (i + 0.5) * dz
            ph = np.deg2rad(ts_deg + self.phi_s_deg)

            # synchronous-γ advance across the substep (annex image 282)
            dgam = q_abs * self._Ez_onaxis(z_local, ph) * dz * 1e-6 / mass
            gamma_o = gamma_i + dgam
            beta_o = float(np.sqrt(max(1.0 - gamma_o ** -2, 0.0)))
            gamma_s = 0.5 * (gamma_i + gamma_o)
            beta_s = float(np.sqrt(max(1.0 - gamma_s ** -2, 0.0)))

            C1, C2, S, C3 = type_coeffs(self.cell_type,
                                        self.type_prev,
                                        self.type_next,
                                        z_local / self.length)
            kx1, ky1, K1, K2 = step_kicks(
                self.voltage_V, self.r0_mm, self.A10, self.length,
                float(ph), gamma_s, beta_s, dz, C1, C2, S, C3, mass)

            dh = dz * 0.5e-3                 # mm per mrad (≡ m per rad)
            Dh = np.array([[1.0, dh], [0.0, 1.0]])
            Mx = Dh @ np.array([[1.0, 0.0], [kx1 * 1e3, K2]]) @ Dh @ Mx
            My = Dh @ np.array([[1.0, 0.0], [ky1 * 1e3, K2]]) @ Dh @ My
            dz_m = dz * 1e-3
            Dzi = np.array([[1.0, dz_m / (2.0 * gamma_i ** 2)],
                            [0.0, 1.0]])
            Dzo = np.array([[1.0, dz_m / (2.0 * gamma_o ** 2)],
                            [0.0, 1.0]])
            Mz = Dzo @ np.array([[1.0, 0.0], [K1, K2]]) @ Dzi @ Mz

            gamma_i, beta_i = gamma_o, beta_o
            if wl > 0 and beta_o > 0:
                ts_deg += 360.0 * dz / (beta_o * wl)

        # (δz[m], δ) → (Δφ[deg], ΔW[MeV]):  Δφ = −360·δz·1e3/(β·λ[mm]),
        # ΔW = δ·β²γ·mc².  Entry kinematics on the way in, exit on the
        # way out (T_out · Mz · T_in⁻¹) — this asymmetry carries the
        # adiabatic part of TW's K₂ into HELIX coordinates exactly.
        M = np.eye(6)
        M[0:2, 0:2] = Mx
        M[2:4, 2:4] = My
        if wl > 0:
            T_in = np.array([[-360.0e3 / (beta_entry * wl), 0.0],
                             [0.0, beta_entry ** 2 * gamma_entry * mass]])
            T_out = np.array([[-360.0e3 / (beta_i * wl), 0.0],
                              [0.0, beta_i ** 2 * gamma_i * mass]])
            M[4:6, 4:6] = T_out @ Mz @ np.linalg.inv(T_in)
        else:
            M[4:6, 4:6] = Mz
        return M

    # ------------------------------------------------------------------
    def advance_ref(self, ref) -> None:
        """Advance the reference particle through the full cell.

        Mirrors ``track_rk4`` for ref-only physics: the on-axis E_z uses
        a *cell-local* synchronous phase that resets at cell entrance so
        the synchronous condition `E_z ∝ cos(φ_s)` at mid-cell holds for
        L = β·λ/2.
        """
        n = self.n_steps
        ds = self.length / n
        cell_phi_deg = 0.0
        # tw2term: annex-exact t_s cursor — half-step frozen at the
        # CELL-entry velocity (image 281), full steps with the post-step
        # velocity (image 252).  The legacy bookkeeping (running-β
        # half-step) integrates to a −0.12 % ramp error on the PXIE
        # deck; the annex cursor reproduces TW's 1.955717 MeV exactly.
        tw = self.field_model == "tw2term"
        if tw and ref.wavelength > 0 and ref.beta > 0:
            cell_phi_deg = 180.0 * ds / (ref.beta * ref.wavelength)
        for i in range(n):
            z_local = (i + 0.5) * ds
            if tw:
                phi_s_mid_deg = cell_phi_deg + self.phi_s_deg
            else:
                if ref.wavelength > 0:
                    dphi_mid = (180.0 * ds * 0.5
                                / (ref.beta * ref.wavelength))
                else:
                    dphi_mid = 0.0
                phi_s_mid_deg = cell_phi_deg + dphi_mid + self.phi_s_deg
            E_z = self._Ez_onaxis(z_local, np.deg2rad(phi_s_mid_deg))
            # Manual image 265: dW = |q|·dz·E_z (see track_rk4 comment).
            ref.w_kin += abs(ref.species.charge) * E_z * ds * 1e-6
            ref.s += ds
            if ref.wavelength > 0:
                full = 360.0 * ds / (ref.beta * ref.wavelength)
                ref.phi_s += full
                cell_phi_deg += full
        # Apply the user-requested output-phase shift to the *global*
        # ref phase (it propagates to the next element).
        ref.phi_s += self.dP_deg

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        """Advance ``ref`` through a sub-range of the cell.

        Mirrors ``advance_ref`` (full-cell ref-only physics) but only for
        the slice [z_from_mm, z_to_mm].  Used by EnvelopeSolver's no-SC
        substep loop so β / γ / W / Φ stay in sync with the slice cursor.

        Cell-local synchronous-phase accumulator ``_adv_cell_phi_deg`` is
        carried on the instance and reset at z_from_mm == 0 so consecutive
        bundles compose into the same E_z(z, t_s) sampling that the full
        ``advance_ref`` produces.
        """
        length_slice = z_to_mm - z_from_mm
        if length_slice <= 0.0:
            return
        tw = self.field_model == "tw2term"
        n = max(1, self.n_steps)
        native_ds = self.length / n
        n_sub = max(1, int(round(length_slice / native_ds)))
        ds = length_slice / n_sub
        if z_from_mm == 0.0:
            # tw2term: annex t_s cursor pre-loaded with the frozen
            # entry-β half step (see advance_ref); legacy: zero.
            if tw and ref.wavelength > 0 and ref.beta > 0:
                self._adv_cell_phi_deg = (180.0 * ds
                                          / (ref.beta * ref.wavelength))
            else:
                self._adv_cell_phi_deg = 0.0
        cell_phi_deg = float(self._adv_cell_phi_deg)
        for i in range(n_sub):
            z_local = z_from_mm + (i + 0.5) * ds
            if tw:
                phi_s_mid_deg = cell_phi_deg + self.phi_s_deg
            else:
                if ref.wavelength > 0:
                    dphi_mid = (180.0 * ds * 0.5
                                / (ref.beta * ref.wavelength))
                else:
                    dphi_mid = 0.0
                phi_s_mid_deg = cell_phi_deg + dphi_mid + self.phi_s_deg
            E_z = self._Ez_onaxis(z_local, np.deg2rad(phi_s_mid_deg))
            ref.w_kin += abs(ref.species.charge) * E_z * ds * 1e-6
            ref.s += ds
            if ref.wavelength > 0:
                full = 360.0 * ds / (ref.beta * ref.wavelength)
                ref.phi_s += full
                cell_phi_deg += full
        self._adv_cell_phi_deg = cell_phi_deg
        # Apply the user-requested output-phase shift on the last slice.
        if abs(z_to_mm - self.length) < 1e-9:
            ref.phi_s += self.dP_deg

    def fitted_matrix(self, ref):
        """6×6 transfer matrix for the full cell at the entry-side ref.

        Implements the manual's per-substep Strang DKD as a product of
        substep matrices, evaluated at the SYNCHRONOUS-particle phase.
        Uses ``self.n_steps`` substeps over ``self.length`` and a
        cell-local phase cursor that ramps from 0 at entry to ~180° at
        exit (the synchronous half-cycle for a cell of length β·λ/2).
        """
        return self.fitted_matrix_slice(ref, self.length, _z_from_mm=0.0)

    def fitted_matrix_slice(self, ref, ds_mm, _z_from_mm: float | None = None):
        """Substep transfer matrix for a sub-range of the cell.

        Builds the per-substep DKD product matching ``track_rk4`` but
        evaluated at sync phase (no per-particle δφ).  The state vector
        order is (x, x'[mrad], y, y'[mrad], Δφ[deg], ΔW[MeV]).

        Parameters
        ----------
        ref : ReferenceParticle
            Entry-side reference (β, γ, λ at slice start).
        ds_mm : float
            Slice length in mm.
        _z_from_mm : float, optional
            Position of slice entry within the cell, in mm.  Used by
            EnvelopeSolver's no-SC bundle loop so the substep grid lines
            up with the cell-local phase cursor.  Default ``None`` =
            slice covers the full cell starting at z=0.
        """
        if ds_mm <= 0:
            return np.eye(6)
        z_from = 0.0 if _z_from_mm is None else float(_z_from_mm)
        if self.field_model == "tw2term":
            return self._fitted_matrix_slice_tw2term(ref, ds_mm, z_from)

        # Native substep size for this cell (matches advance_ref / track_rk4).
        n_full = max(1, self.n_steps)
        native_ds = self.length / n_full
        n_sub = max(1, int(round(ds_mm / native_ds)))
        ds = ds_mm / n_sub
        ds_m = ds * 1e-3
        ds_half_m = 0.5 * ds_m

        # Cell-local phase cursor at slice entry.  We need the same value
        # the ref-advance uses; advance_ref_over stores it in
        # ``_adv_cell_phi_deg`` after each call, but during fitted_matrix
        # we may be invoked before any advance happened.  Recompute from
        # z_from using the entry-side β as a first approximation (good to
        # ~0.1 % per cell and irrelevant for the matrix as long as the
        # phase progression is consistent across the slice).
        wl_mm = ref.wavelength
        if wl_mm > 0 and ref.beta > 0:
            cell_phi_deg = 360.0 * z_from / (ref.beta * wl_mm)
        else:
            cell_phi_deg = 0.0

        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass
        wl = ref.wavelength
        q_abs = abs(ref.species.charge)

        # Drift half-step matrix, in (x, x', y, y', Δφ, ΔW) coordinates with
        # x' in mrad and Δφ in deg.  R45 is the longitudinal slip coefficient
        # (deg / MeV) for a half-step; the per-substep total slip is twice
        # this, applied as a single coupled term.
        if wl > 0 and beta > 0 and gamma > 0 and mass > 0:
            r45_full = -360.0 * ds / (beta**3 * gamma**3 * mass * wl)  # deg / MeV
        else:
            r45_full = 0.0
        # Half-drift uses ds/2; in our units x' is mrad, x is mm, so the
        # x↔x' coupling per half-drift is (ds/2)[m] · 1e3[mrad/rad] / 1
        # = ds/2 [mm/mrad·1] →  M[0,1] = ds_half_m·1e3 [mm per mrad]
        # — but actually ds_half_m·1e3 = ds_half (in mm).  Cleaner:
        d_half_xprime = ds * 0.5            # mm per mrad (since x'/1000 is rad, x' [mrad] · ds [mm] /1000 = mm; but we want full mm coupling so use ds/2 in mm directly).
        # ↑ derivation: x_out = x + (ds/2)·(x'/1000) [mm·rad] · 1e3 = x + (ds/2)·x'[mrad]/1000 ... no
        # let's redo: if x in mm and x' in mrad, then x' [mrad] = 1e-3·x' [rad].
        # over distance L[mm], dx = L · x'[rad] = L·1e-3·x'[mrad] (mm).
        # So M[0,1] = L · 1e-3 (with L in mm).  Half-drift = (ds_mm/2)·1e-3.
        d_half_xprime = ds * 0.5e-3   # mm per mrad

        Mh = np.eye(6)
        Mh[0, 1] = d_half_xprime
        Mh[2, 3] = d_half_xprime
        Mh[4, 5] = 0.5 * r45_full

        M_total = np.eye(6)
        for i in range(n_sub):
            z_local = z_from + (i + 0.5) * ds
            # Synchronous-particle phase at substep midpoint
            if wl > 0 and beta > 0:
                dphi_to_mid = 180.0 * ds * 0.5 / (beta * wl)
            else:
                dphi_to_mid = 0.0
            phi_s_mid_rad = np.deg2rad(cell_phi_deg + dphi_to_mid + self.phi_s_deg)

            # Transverse coefficients (manual images 281-291)
            C1, C2, S = self._type_coeffs(z_local)
            if self.field_model == "crand_x" and abs(self.cell_type) == 2:
                S = -S
            cos_phase = float(np.cos(phi_s_mid_rad))
            if self.field_model == "pdf_2term":
                # PDF page 147 literal — see track_rk4 for derivation.
                r02 = self.r0_mm * self.r0_mm
                A01 = max(0.0, 1.0 - self.A10)
                if abs(self.cell_type) == 2:
                    C2_eff = float(np.sin(np.pi * z_local / self.length))
                else:
                    C2_eff = C2
                ag = S * self.voltage_V / r02 * A01 * C1
                defoc_pdf = ((np.pi / self.length) ** 2
                             * self.A10 * self.voltage_V * 0.25 * C2_eff)
                if beta > 0 and gamma > 0 and mass > 0:
                    factor_pdf = q_abs * ds / (gamma * beta * beta * 2.0 * mass * 1e6)
                    kx_per_x = -factor_pdf * cos_phase * (ag - defoc_pdf)
                    ky_per_y = -factor_pdf * cos_phase * (-ag - defoc_pdf)
                else:
                    kx_per_x = 0.0
                    ky_per_y = 0.0
            else:
                rf_quad   = cos_phase * S * (self.voltage_V * 0.5) * self._A_quad_eff * C1     # V/mm²
                rf_defoc  = (np.pi / self.length) ** 2 * (self.A10 * self.voltage_V * 0.5) * C2  # V/mm²

                if beta > 0 and gamma > 0 and mass > 0:
                    factor = q_abs * ds / (gamma * beta * beta * mass * 1e6)
                    kx_per_x = -factor * (rf_quad - rf_defoc)   # 1/mm
                    ky_per_y = -factor * (-rf_quad - rf_defoc)
                else:
                    kx_per_x = 0.0
                    ky_per_y = 0.0

            # Longitudinal kick: δW kick proportional to δφ (small-Δφ
            # linearisation of dW = q·E_z(z, φ_s+Δφ)·dz).
            # ∂E_z/∂φ |_{φ_s+sync_advance} = (πA10V/2L)·sin(πz/L)·cos(φ_s_mid)
            # δW contribution (MeV) = q·(πA10V/2L)·sin(πz/L)·cos(φ_s_mid)·dz·1e-6·δφ[rad]
            # Δφ in degrees ⇒ multiply by π/180.
            if self.length > 0:
                Ez_amp_per_phi = ((np.pi * self.A10 * self.voltage_V) / (2.0 * self.length)
                                   * np.sin(np.pi * z_local / self.length)
                                   * cos_phase)  # V/mm  (per radian δφ)
                k54 = q_abs * Ez_amp_per_phi * ds * 1e-6 * (np.pi / 180.0)  # MeV per deg
            else:
                k54 = 0.0

            # Build kick matrix
            Mk = np.eye(6)
            # x' kick: x' [mrad] += kx_per_x[1/mm]·x[mm]·1e3[mrad/rad]
            Mk[1, 0] = kx_per_x * 1e3
            Mk[3, 2] = ky_per_y * 1e3
            Mk[5, 4] = k54

            M_sub = Mh @ Mk @ Mh
            M_total = M_sub @ M_total

            # Advance cell-local phase cursor to next substep
            if wl > 0 and beta > 0:
                cell_phi_deg += 360.0 * ds / (beta * wl)

        return M_total
