"""TraceWin ``NCELLS`` ("Cavity multi-gap") element.

TraceWin models a multi-cell RF cavity analytically as **a set of thin RF gaps,
one gap at the middle of each cell** (TraceWin manual, "Cavity multi-gap"
section).  ``NCells`` reproduces that model directly: the cavity is expanded, at
track time, into ``n_cells`` drift-gap-drift segments, each gap applying the same
Wangler thin-gap kick that :class:`~linac_gen.elements.rf_gap.RFGap` implements.

Why an element (not a parse-time expansion into Drift+RFGap):  real decks
(e.g. ``fnalscl.dat``) use **absolute phase** (``P=1``) — the card's ``θs`` grows
down the linac to track the beam's accumulated time-of-flight, keeping the *true*
synchronous phase ~constant.  Recovering that synchronous phase needs the running
RF clock ``ref.phi_s`` **at track time**, which the parser (a static lattice with
no energy tracking) does not have.  Same argument for ``βg ≤ 0`` (cell length set
by the running velocity).  So the phase/geometry must be resolved during
tracking — hence a runtime element mirroring
:class:`~linac_gen.elements.vane_rfq.VaneRFQ` (multi-cell-in-one-element) with
:class:`~linac_gen.elements.rf_gap.RFGap` per-gap physics.

Units follow the rest of ``linac_gen``: lengths **mm**, angles/phase **deg**,
kinetic energy **MeV**, gap voltage **MV**, frequency **MHz**.  Energy gain uses
the **|q|** (abs-charge) convention TraceWin/RfqCell use (not RFGap's signed
charge), so the deck's θs maps directly and H⁻ and protons accelerate alike;
the absolute-phase convention is validated against TraceWin (see below).

TraceWin ``NCELLS`` operands (in order):
``mode Nc βg EoT θs R P kEoTi kEoTo dzi dzo [βs Ts kT's k²T''s Ti kT'i k²T''i To kT'o k²T''o]``
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from linac_gen.core.constants import C_LIGHT
from linac_gen.elements.base import FieldMapElement


# --------------------------------------------------------------------------- #
# ABSOLUTE-phase (P=1) convention, VALIDATED against a TraceWin ground-truth   #
# run of fnalscl.dat (Tracewin_code/piplatticetracewin/):                      #
#                                                                             #
#   φ_gap = ref.phi_s − θs         (the phase the beam SEES: ω·t − θs)         #
#   dW    = |q| · V · T · cos(φ)   (the |q| convention TraceWin/RfqCell use)   #
#                                                                             #
# TraceWin's absolute θs is set so the seen phase (ref.phi_s − θs) holds the   #
# design synchronous phase down the whole linac.  Getting BOTH the sign and    #
# the |q| right is essential: HELIX then reproduces TraceWin's 116.1 → 404.8   #
# MeV (→ 404.6) AND its bounded envelope (rms ≈ 5 mm) in envelope and MP.      #
# The sign of φ matters beyond the energy (cos is even): sin φ sets the RF     #
# defocus and the longitudinal focusing, so ``phi_s − θs`` (not ``θs − phi_s``)#
# is the physical one — the other blows the transverse envelope up.            #
# --------------------------------------------------------------------------- #


def _sync_factor(beta: float, beta_g: float) -> float:
    """Geometric synchronism (transit-time) factor of a π-mode cell,
    UNIFORM-field profile — the classic Panofsky/DTL form with the full
    cell as the gap:  T(β) = sin(u)/u,  u = (π/2)·βg/β  (T = 2/π at
    synchronism).

    Returned NORMALISED to 1 at β = βg, because the deck's ``EoT`` is
    the effective at-synchronism field (E₀T): the card's energy gain is
    kept as-given and only the RELATIVE β-dependence enters.

    Reverse-engineered from TraceWin's own per-element transfer-matrix
    export of fnalscl.dat (2026-07-21): with this factor's W-derivative
    in the gap map, all four longitudinal matrix elements agree with
    TraceWin to ≤0.04% per cavity (they differed by up to 1.8%
    without it); the half-sine cell profile closes only half the gap.
    TraceWin applies this internally even when the card carries no
    βs/Ts tail.
    """
    if beta <= 0 or beta_g <= 0:
        return 1.0
    u = (np.pi / 2.0) * beta_g / beta
    t = np.sin(u) / u
    return float(t / (2.0 / np.pi))


def _w_kin_from_beta(beta: float, mass_MeV: float) -> float:
    """Kinetic energy (MeV) for a given β (used to seed βg<0 cavities)."""
    beta = min(max(abs(beta), 0.0), 0.999999999)
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    return (gamma - 1.0) * mass_MeV


@dataclass(frozen=True)
class TTFSet:
    """One transit-time-factor triple (SUPERFISH ``T``, ``T'``, ``T''`` about βs).

    ``kTp`` is the card's ``kT'`` ( = SUPERFISH T' × (−2π) ), ``k2Tpp`` its
    ``k²T''`` ( = SUPERFISH T'' × (−4π²) ).  See :func:`_ttf_value`.
    """
    Ts: float
    kTp: float
    k2Tpp: float


@dataclass(frozen=True)
class TTFTable:
    """Optional βs≠0 transit-time-factor tail (middle / input / output gaps)."""
    beta_s: float
    middle: TTFSet
    input: TTFSet
    output: TTFSet


@dataclass(frozen=True)
class _Gap:
    """A single resolved cell-gap: position, signed voltage (MV), gap class."""
    z_center_mm: float
    voltage_mv: float          # σ_k · EoT · Lc · eotl_scale  (polarity baked in)
    kind: str                  # "input" | "middle" | "output"


def parse_ttf_tail(tail: list[str]) -> Optional[TTFTable]:
    """Parse the optional ``βs Ts kT's k²T''s Ti … To …`` operand tail.

    Returns ``None`` when the tail is empty or ``βs == 0`` (the entire
    ``fnalscl.dat`` case — no β-dependent transit-time correction).
    """
    if not tail:
        return None
    vals = [float(t) for t in tail]
    beta_s = vals[0]
    if beta_s == 0.0:
        return None

    def triple(i: int) -> TTFSet:
        g = vals[i:i + 3] + [0.0, 0.0, 0.0]
        return TTFSet(Ts=g[0], kTp=g[1], k2Tpp=g[2])

    middle = triple(1)                       # Ts, kT's, k²T''s
    inp = triple(4) if len(vals) > 4 else middle   # Ti, kT'i, k²T''i
    out = triple(7) if len(vals) > 7 else middle   # To, kT'o, k²T''o
    return TTFTable(beta_s=beta_s, middle=middle, input=inp, output=out)


def _ttf_value(s: TTFSet, beta: float, beta_s: float) -> float:
    """Transit-time factor T(β) via the standard Wangler expansion about βs.

    Wave number ``k = 2π/(β·λ_rf)`` (λ_rf cancels between k and ks, so the
    expansion variable is ``u = k/ks − 1 = βs/β − 1``):

        T(β) = Ts + kT'·u + ½·k²T''·u²

    RECONSTRUCTED from the standard Wangler transit-time expansion (manual p.202;
    the manual's own ``T(β)`` figure renders blank in our copy).  NOT validatable
    against ``fnalscl.dat`` (every card there is βs=0).  Only reached when a
    βs≠0 TTF tail is present.
    """
    if beta <= 0 or beta_s <= 0:
        return s.Ts
    u = beta_s / beta - 1.0
    return s.Ts + s.kTp * u + 0.5 * s.k2Tpp * u * u


class NCells(FieldMapElement):
    """TraceWin ``NCELLS`` multi-gap cavity, tracked as a set of thin gaps.

    Parameters
    ----------
    name : str
    mode : int
        0 = 2π, 1 = π, 2 = π&2π (sets cell length / gap polarity).
    n_cells : int
    beta_g : float
        Geometric β.  ``>0``: all cells share length from βg.  ``=0``: cell
        length from the running beam velocity (resolved at track time).
        ``<0``: seeded at ``|βg|`` and ramped cell-to-cell by the cavity's own
        acceleration.
    eot_v_per_m : float
        Effective gap field EoT (V/m).  Per-cell gap voltage = EoT·Lc.
    theta_s_deg : float
        RF phase θs at the first gap (deg).
    aperture_mm, p_flag, k_eot_i, k_eot_o, dz_i_mm, dz_o_mm
        TraceWin operands 6-11 (aperture; phase mode 0/1/2; input/output field
        corrections; first/last gap displacements).
    frequency_mhz : float
        Cavity RF frequency, sourced from the lattice ``FREQ`` state.
    sync_phase : bool
        ``True`` when a ``SET_SYNC_PHASE`` preceded the card → θs is the
        *synchronous* phase (calibrated), not the absolute phase.
    ttf : TTFTable, optional
        βs≠0 transit-time-factor tail (see :func:`parse_ttf_tail`).
    n_steps : int, optional
        Integration/SC sub-step count.  Defaults to ``2·n_cells`` (≈2 SC kicks
        per cell).
    """

    # ``βg≤0`` cavities can't know their length until the beam arrives, but the
    # tracker sizes its slice count from ``element.length`` BEFORE tracking.  A
    # deliberate OVER-estimate (β=0.9, above any hadron-linac NCELLS velocity)
    # guarantees the tracker's slices span the real cavity so no cell-gap is
    # skipped in MP mode; ``track_rk4`` caps drifts at the lazily-resolved true
    # length so there is never phantom drift past the cavity.  SC-kick cadence
    # is coarser for βg≤0 (flagged); reference/envelope (advance_ref) is exact.
    _PROVISIONAL_BETA = 0.9

    def __init__(self, name: str, *,
                 mode: int, n_cells: int, beta_g: float,
                 eot_v_per_m: float, theta_s_deg: float,
                 aperture_mm: float = 0.0, p_flag: int = 0,
                 k_eot_i: float = 0.0, k_eot_o: float = 0.0,
                 dz_i_mm: float = 0.0, dz_o_mm: float = 0.0,
                 frequency_mhz: float = 0.0, sync_phase: bool = False,
                 ttf: Optional[TTFTable] = None,
                 n_steps: Optional[int] = None):
        self.mode = int(mode)
        self.n_cells = int(n_cells)
        self.beta_g = float(beta_g)
        self.eot_v_per_m = float(eot_v_per_m)
        self.theta_s_deg = float(theta_s_deg)
        self.p_flag = int(p_flag)
        self.k_eot_i = float(k_eot_i)
        self.k_eot_o = float(k_eot_o)
        self.dz_i_mm = float(dz_i_mm)
        self.dz_o_mm = float(dz_o_mm)
        self.frequency_mhz = float(frequency_mhz)
        self.sync_phase = bool(sync_phase)
        # ERROR_CAV hooks (TraceWin cavity tolerance errors), mutated by
        # ErrorStudy on lattice copies.  Consumed at USE time — every gap
        # kick/matrix/probe amplitude scales by (1 + voltage_rel) and the
        # applied RF phase shifts by phase_offset (deg) — so mutations on
        # an already-constructed element (cached gap lists) take effect.
        self.voltage_rel = 0.0
        self.phase_offset = 0.0
        self._ttf = ttf
        self.beta_s = ttf.beta_s if ttf is not None else 0.0

        if n_steps is None:
            n_steps = max(2 * self.n_cells, 2)

        # Resolve geometry now iff βg>0 (fixed length); else provisional length
        # for step-count sizing, real cells built lazily at track entry.
        self._gaps: Optional[list[_Gap]] = None
        if self.beta_g > 0:
            self._gaps = self._build_gaps_fixed(self.beta_g)
            length = sum(self._cell_lengths(self.beta_g))
        else:
            length = self.n_cells * self._interior_cell_length(self._PROVISIONAL_BETA)

        super().__init__(name=name, length=float(length),
                         aperture=float(aperture_mm), n_steps=int(n_steps))

        # Per-run integrator state (reset in reset_run_state).  ``_z_cursor`` is
        # the accumulated PHYSICAL position (mm from entry, capped at length);
        # ``_next_gap`` is the index of the next un-applied gap.  Position is
        # tracked by accumulation — NEVER reconstructed as ``_step_idx*ds`` —
        # so mixed step sizes (SC half-steps, trailing remainders) stay in sync,
        # and each gap fires exactly once regardless of slice-boundary alignment.
        self._step_idx = 0
        self._z_cursor = 0.0
        self._next_gap = 0
        self._phi_s_at_entrance = 0.0
        self._phi_s_at_gap1 = 0.0
        self._sync_offset_deg: Optional[float] = None

    @property
    def frequency(self) -> float:
        """RF frequency (MHz) — alias of ``frequency_mhz`` so NCELLS reads like
        the other RF elements (RFGap/FieldMap carry ``.frequency``), e.g. for the
        GUI's running-frequency validation and diagnostics."""
        return self.frequency_mhz

    @frequency.setter
    def frequency(self, value: float) -> None:
        self.frequency_mhz = float(value)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _wavelength_mm(self) -> float:
        """RF wavelength λ = c/f (mm) at the cavity frequency."""
        if self.frequency_mhz <= 0:
            return 0.0
        return C_LIGHT / (self.frequency_mhz * 1e6) * 1000.0

    def _interior_cell_length(self, beta: float) -> float:
        """Interior-cell length Lc for the given β (mm)."""
        lam = self._wavelength_mm()
        if self.mode == 0:            # 2π
            return beta * lam
        if self.mode == 1:            # π
            return 0.5 * beta * lam
        return beta * lam             # 2 (π&2π) interior cells are βλ

    def _cell_kind(self, k: int) -> str:
        if self.n_cells == 1:
            return "middle"
        if k == 0:
            return "input"
        if k == self.n_cells - 1:
            return "output"
        return "middle"

    def _cell_polarity(self, k: int) -> int:
        """Standing-wave field sign σ_k for cell k (0-indexed)."""
        if self.mode == 1:            # π: adjacent cells flip
            return 1 if (k % 2 == 0) else -1
        # mode 0 (2π): same sign everywhere.
        # mode 2 (π&2π): FLAG — the exact polarity pattern renders blank in the
        # manual; no deck exercises it.  Treated as same-sign pending validation.
        return 1

    def _cell_dims(self, k: int, beta: float) -> tuple[float, float, float]:
        """(Le, Ls, Lc) for cell k (mm): entrance drift, exit drift, length."""
        lam = self._wavelength_mm()
        kind = self._cell_kind(k)
        if self.mode == 2 and kind in ("input", "output"):
            lc = 0.75 * beta * lam            # FLAG: m=2 end cell, unvalidated
        else:
            lc = self._interior_cell_length(beta)
        half = 0.5 * lc
        le, ls = half, half
        if kind == "input":
            le, ls = half + self.dz_i_mm, half - self.dz_i_mm
        elif kind == "output":
            le, ls = half + self.dz_o_mm, half - self.dz_o_mm
        return le, ls, lc

    def _cell_lengths(self, beta: float) -> list[float]:
        return [self._cell_dims(k, beta)[2] for k in range(self.n_cells)]

    def _ttf_ratio(self, kind: str) -> float:
        """End-cell field ratio Ti/Ts (input) or To/Ts (output) at βs; 1 if no TTF."""
        if self._ttf is None:
            return 1.0
        ts = self._ttf.middle.Ts
        if ts == 0.0:
            return 1.0
        if kind == "input":
            return self._ttf.input.Ts / ts
        if kind == "output":
            return self._ttf.output.Ts / ts
        return 1.0

    def _eotl_scale(self, kind: str) -> float:
        """Per-cell field scale: (1+kEoTi)·(Ti/Ts) input, (1+kEoTo)·(To/Ts) output."""
        if kind == "input":
            return (1.0 + self.k_eot_i) * self._ttf_ratio("input")
        if kind == "output":
            return (1.0 + self.k_eot_o) * self._ttf_ratio("output")
        return 1.0

    def _ttf_correction(self, beta: float, kind: str) -> float:
        """β-dependent T(β)/T(βs) correction (1.0 unless a βs≠0 tail is present)."""
        if self._ttf is None or self.beta_s <= 0:
            return 1.0
        tset = {"input": self._ttf.input,
                "middle": self._ttf.middle,
                "output": self._ttf.output}[kind]
        t_ref = tset.Ts
        if t_ref == 0.0:
            return 1.0
        return _ttf_value(tset, beta, self.beta_s) / t_ref

    def _v_of(self, gap: "_Gap") -> float:
        """Gap kick amplitude incl. the ERROR_CAV relative amplitude error."""
        return gap.voltage_mv * (1.0 + self.voltage_rel)

    def _gap_voltage_mv(self, k: int, lc_mm: float) -> float:
        kind = self._cell_kind(k)
        pol = self._cell_polarity(k)
        return (pol * self.eot_v_per_m * (lc_mm * 1e-3)
                * self._eotl_scale(kind) * 1e-6)

    def _build_gaps_fixed(self, beta: float) -> list[_Gap]:
        """Gap list for a fixed β (βg>0): geometry independent of dynamics."""
        gaps = []
        z = 0.0
        for k in range(self.n_cells):
            le, ls, lc = self._cell_dims(k, beta)
            gaps.append(_Gap(z_center_mm=z + le,
                             voltage_mv=self._gap_voltage_mv(k, lc),
                             kind=self._cell_kind(k)))
            z += lc
        return gaps

    def _build_gaps_running(self, ref) -> list[_Gap]:
        """Gap list for βg≤0: cell length follows the running velocity.

        A self-consistent reference forward-pass (drift-gap-drift, ref only)
        determines each cell's β and hence its length.  FLAG: this path is not
        validatable against ``fnalscl.dat`` (all βg>0); gated behind synthetic
        βg≤0 tests only.
        """
        rf = ref.copy()
        if self.frequency_mhz > 0:
            rf.frequency = self.frequency_mhz
        if self.beta_g < 0:
            rf.w_kin = _w_kin_from_beta(abs(self.beta_g), rf.species.mass)
        # Pin the gap-1 phase reference from the ENTRY state so this forward
        # pass is deterministic (independent of any prior instance state) — MP
        # and advance_ref must resolve identical geometry.  Uses the first
        # cell's Le at the entry β so the build is gap-1-referenced like the walk.
        le0 = self._cell_dims(0, rf.beta)[0]
        if rf.beta > 0 and rf.wavelength > 0:
            self._phi_s_at_gap1 = rf.phi_s + 360.0 * le0 / (rf.beta * rf.wavelength)
        else:
            self._phi_s_at_gap1 = rf.phi_s
        gaps = []
        z = 0.0
        for k in range(self.n_cells):
            le, ls, lc = self._cell_dims(k, rf.beta)
            gap = _Gap(z_center_mm=z + le,
                       voltage_mv=self._gap_voltage_mv(k, lc),
                       kind=self._cell_kind(k))
            gaps.append(gap)
            # advance the local reference across the cell to ramp β
            self._ref_drift(rf, le)
            phi = self._phi_gap_rad(rf, gap)
            rf.w_kin += (abs(rf.species.charge) * self._v_of(gap)
                         * self._ttf_correction(rf.beta, gap.kind) * np.cos(phi))
            self._ref_drift(rf, ls)
            z += lc
        self.length = z
        return gaps

    def _ensure_gaps(self, ref) -> list[_Gap]:
        if self._gaps is None:
            self._gaps = self._build_gaps_running(ref)
        return self._gaps

    # ------------------------------------------------------------------ #
    # Phase (the crux — one place; sign locked by fnalscl validation)
    # ------------------------------------------------------------------ #
    def _phi_gap_rad(self, ref, gap: _Gap) -> float:
        """Synchronous phase (rad) seen by the gap, per the P flag.

        TraceWin's θs is the phase **at the first gap** (not the element
        entrance), so the relative/sync clocks are referenced to gap 1
        (``_phi_s_at_gap1`` = entrance clock + the Le drift to the first gap),
        making ``φ(gap 1) = θs`` exactly.

        * SET_SYNC_PHASE : θs is the synchronous phase (calibrated offset ψ).
          Both the ψ probe and this application are GAP-1-referenced (unlike
          FieldMap, which references the element entrance) — the two frames
          must match or ψ double-counts the entrance→gap-1 drift (~90° in
          π-mode).
        * P=1 absolute   : ``ref.phi_s − θs`` — the phase the beam SEES against
          the global RF clock; θs grows down the linac so this holds the design
          synchronous phase.  Validated against TraceWin (see the module header).
        * P=0/2 relative : θs + (running clock − gap-1 clock) → θs at gap 1.
        """
        theta = self.theta_s_deg + self.phase_offset
        phi_run = ref.phi_s
        if self.sync_phase:
            off = self._sync_offset_deg or 0.0
            return np.deg2rad(theta - off + (phi_run - self._phi_s_at_gap1))
        if self.p_flag == 1:
            return np.deg2rad(phi_run - theta)
        return np.deg2rad(theta + (phi_run - self._phi_s_at_gap1))

    def _calibrate_sync_phase(self, ref_entry) -> None:
        """SET_SYNC_PHASE calibration: find ψ so θs is the synchronous phase.

        Iterates ψ = arctan2(Im V, Re V) with the same self-consistent
        β-evolving voltage integral as :meth:`FieldMap._calibrate_sync_phase`.
        """
        if not self.sync_phase or self._sync_offset_deg is not None:
            return
        theta = self.theta_s_deg
        psi = 0.0
        for _ in range(30):
            re_v, im_v = self._probe_voltage_integral(ref_entry, theta - psi)
            psi_new = float(np.rad2deg(np.arctan2(im_v, re_v)))
            if abs(psi_new - psi) < 0.005:
                psi = psi_new
                break
            psi = 0.3 * psi + 0.7 * psi_new
        self._sync_offset_deg = float(psi)

    def _probe_voltage_integral(self, ref_entry, phi_input_deg: float):
        """Σ q·V·e^{i·φ_gen} over the gaps with self-consistent β (for calibration).

        The running phase is referenced to GAP 1 (φ_run = 0 at the first gap
        center), matching :meth:`_phi_gap_rad`'s ``phi_run − _phi_s_at_gap1``
        application frame.  An entrance-referenced probe would fold the
        entrance→gap-1 drift advance (90° for π-mode, Le = βλ/4) into the
        calibrated ψ, which the application then subtracts a second time —
        putting the beam ~90° off the requested synchronous phase.
        """
        rf = ref_entry.copy()
        if self.frequency_mhz > 0:
            rf.frequency = self.frequency_mhz
        rf.phi_s = 0.0
        gaps = self._gaps if self._gaps is not None else self._build_gaps_fixed(
            self.beta_g if self.beta_g > 0 else max(rf.beta, 1e-6))
        phi_in = np.deg2rad(phi_input_deg)
        re_v = im_v = 0.0
        z = 0.0
        phi_gap1 = None
        for gap in gaps:
            self._ref_drift(rf, gap.z_center_mm - z)
            if phi_gap1 is None:
                phi_gap1 = rf.phi_s      # gap-1 clock: the reference zero
            q = abs(rf.species.charge)
            ttf = self._ttf_correction(rf.beta, gap.kind)
            amp = q * self._v_of(gap) * ttf
            phi_run = np.deg2rad(rf.phi_s - phi_gap1)
            re_v += amp * np.cos(phi_run)
            im_v += amp * np.sin(phi_run)
            rf.w_kin += amp * np.cos(phi_in + phi_run)
            z = gap.z_center_mm
        return re_v, im_v

    # ------------------------------------------------------------------ #
    # Drift + gap primitives (mirror Drift.track / RFGap.apply_kick)
    # ------------------------------------------------------------------ #
    def _drift_matrix(self, ref, L_mm: float) -> np.ndarray:
        M = np.eye(6)
        L_m = L_mm * 1e-3
        M[0, 1] = L_m
        M[2, 3] = L_m
        wl = ref.wavelength
        if wl > 0 and ref.beta > 0 and ref.gamma > 0:
            M[4, 5] = -360.0 * L_mm / (ref.beta ** 3 * ref.gamma ** 3
                                       * ref.species.mass * wl)
        return M

    def _ref_drift(self, ref, L_mm: float) -> None:
        if L_mm == 0.0:
            return
        ref.s += L_mm
        if ref.wavelength > 0 and ref.beta > 0:
            ref.phi_s += 360.0 * L_mm / (ref.beta * ref.wavelength)

    def _drift_beam(self, beam, L_mm: float) -> None:
        if L_mm == 0.0:
            return
        ref = beam.ref
        M = self._drift_matrix(ref, L_mm)
        self._ref_drift(ref, L_mm)
        alive = beam.alive_mask
        if np.any(alive):
            beam.particles[alive] = (M @ beam.particles[alive].T).T

    # ------------------------------------------------------------------ #
    # Geometric synchronism factor (TraceWin-internal T(β), 2026-07-21)
    # ------------------------------------------------------------------ #
    #: escape hatch: set False to recover the pre-2026-07 ideal thin-gap
    #: model (T ≡ 1, no W-dependent synchronism) for comparison studies.
    synchronism: bool = True

    #: validity clamp for the per-gap synchronism δ.  The factor is a
    #: perturbative TW-matrix correction (validated regime ~1e-3/gap on
    #: the matched fnalscl deck); when β falls toward βg/2, T̂→0 and the
    #: log-derivative diverges (δ can reach −800 while staying finite,
    #: silently poisoning the envelope with ~1e5 matrix entries).  Past
    #: this bound the cavity is grossly off-synchronism — far outside
    #: the regime the factor was reverse-engineered in — so the matrix
    #: falls back to the stock thin-gap form and warns once.
    _SYNC_DELTA_MAX = 0.1
    _sync_clamp_warned: bool = False

    def _sync_delta(self, ref, gap: _Gap) -> float:
        """δ = |q|·V·T·cos(φ)·d(ln T̂)/dW at the synchronous energy — the
        linearised W-feedback of the synchronism factor.  Enters the gap
        map as M[5,5] = 1+δ with the det-preserving M[4,4] = 1/(1+δ)
        (mirrors TraceWin's determinant-normalised gap matrix).

        Disabled when the card carries a βs≠0 TTF tail: the tail IS
        TraceWin's explicit T(β) for that cavity (the manual's k₁/k₂
        formulas use the tail's kT′/T), so the geometric factor must
        not stack on top of it."""
        if (not self.synchronism or self.beta_g <= 0
                or self._ttf is not None):
            return 0.0
        mass = ref.species.mass
        if mass <= 0:
            return 0.0
        h = 1e-3
        def _th(w):
            g = 1.0 + w / mass
            b = float(np.sqrt(max(1.0 - 1.0 / g**2, 1e-12)))
            return _sync_factor(b, self.beta_g)
        t0 = _th(ref.w_kin)
        if t0 <= 0:
            return 0.0
        dln = (_th(ref.w_kin + h) - _th(ref.w_kin - h)) / (2.0 * h * t0)
        V = self._v_of(gap)
        ttf = self._ttf_correction(ref.beta, gap.kind)
        phi = self._phi_gap_rad(ref, gap)
        delta = abs(ref.species.charge) * V * ttf * np.cos(phi) * dln
        if abs(delta) > self._SYNC_DELTA_MAX:
            if not self._sync_clamp_warned:
                self._sync_clamp_warned = True
                warnings.warn(
                    f"NCells '{self.name}': beam is grossly off the "
                    f"cavity's geometric beta (synchronism delta = "
                    f"{delta:.3g} per gap, validated regime ~1e-3) — "
                    f"the TW-matrix synchronism factor is outside its "
                    f"validity range here and is being skipped for "
                    f"this cavity (stock thin-gap matrix used).",
                    stacklevel=2)
            return 0.0
        return delta

    def _apply_gap(self, beam, gap: _Gap) -> None:
        """Thin-gap kick — energy gain, adiabatic damping, RF defocus.

        Mirrors :meth:`RFGap.apply_kick` but with the |q| (abs-charge)
        convention TraceWin uses (like RfqCell) so the deck's θs phases map
        directly and an H⁻ beam accelerates at the same numbers as a proton.
        Uses the gap's signed voltage (polarity baked in), the β-dependent
        T(β), and the phase from :meth:`_phi_gap_rad`.
        """
        ref = beam.ref
        charge = abs(ref.species.charge)
        mass = ref.species.mass
        V = self._v_of(gap)
        ttf = self._ttf_correction(ref.beta, gap.kind)
        phi = self._phi_gap_rad(ref, gap)

        bg_old = ref.bg
        ref.w_kin += charge * V * ttf * np.cos(phi)
        bg_new = ref.bg

        alive = beam.alive_mask
        if not np.any(alive):
            return
        dphi = beam.particles[alive, 4]
        phi_i = phi + dphi * (np.pi / 180.0)
        # MODE-FAITHFUL design (2026-07-21, measured, not assumed):
        # TraceWin's own two engines disagree here — its MATRIX engine
        # carries the geometric synchronism factor (see
        # _gap_kick_matrix), its PARTRAN tracker does not.  Three MP
        # runs against TW's partran reference decided the tracking
        # form: stock thin-gap kick 1.4%/4.1% (sigma_phi deviation
        # below/above 30 m), + synchronism factor with phase
        # contraction 2.4%/3.9%, + factor alone 5.8%/33%.  The stock
        # kick IS TraceWin-partran-faithful; the synchronism lives in
        # the matrix/envelope path only, mirroring TW's architecture.
        beam.particles[alive, 5] += charge * V * ttf * (
            np.cos(phi_i) - np.cos(phi))

        damping = bg_old / bg_new
        beam.particles[alive, 1] *= damping
        beam.particles[alive, 3] *= damping

        wl = ref.wavelength
        if mass > 0 and bg_new > 0 and wl > 0:
            k_rf = -np.pi * charge * V * ttf * np.sin(phi) / (mass * bg_new ** 3 * wl)
            beam.particles[alive, 1] += k_rf * beam.particles[alive, 0] * 1e3
            beam.particles[alive, 3] += k_rf * beam.particles[alive, 2] * 1e3

    # ------------------------------------------------------------------ #
    # Entry bookkeeping
    # ------------------------------------------------------------------ #
    def _snapshot_entry(self, ref) -> None:
        """Record the entrance clock and the gap-1 clock.

        ``_phi_s_at_gap1`` = entrance phi_s + the Le drift to the first gap, so
        the relative/sync phase references gap 1 (φ(gap 1) = θs), matching
        TraceWin's "phase at the first gap" definition.
        """
        self._phi_s_at_entrance = ref.phi_s
        gaps = self._ensure_gaps(ref)
        if gaps and ref.beta > 0 and ref.wavelength > 0:
            self._phi_s_at_gap1 = (ref.phi_s + 360.0 * gaps[0].z_center_mm
                                   / (ref.beta * ref.wavelength))
        else:
            self._phi_s_at_gap1 = ref.phi_s
        # SET_SYNC_PHASE calibration lives HERE — the one choke point every
        # tracking mode passes through (track_rk4 via _on_entry, advance_ref,
        # advance_ref_over, fitted_matrix, fitted_matrix_slice).  Previously
        # only the track/advance paths calibrated, so matrix-mode/envelope
        # first-bundle matrices were built with ψ=0 while the ref advance
        # used the calibrated ψ.
        if self.sync_phase and self._sync_offset_deg is None:
            self._calibrate_sync_phase(ref)

    def _on_entry(self, beam) -> None:
        """First-slice setup: FREQ-jump rescale, entrance snapshot, geometry,
        SET_SYNC_PHASE calibration."""
        ref = beam.ref
        f = self.frequency_mhz
        if f > 0 and f != ref.frequency and ref.frequency > 0:
            ratio = f / ref.frequency
            beam.particles[beam.alive_mask, 4] *= ratio
            ref.frequency = f
        self._snapshot_entry(ref)
        if self.sync_phase and self._sync_offset_deg is None:
            self._calibrate_sync_phase(ref)

    # ------------------------------------------------------------------ #
    # FieldMapElement contract — multi-particle
    # ------------------------------------------------------------------ #
    def track_rk4(self, beam, ds: float) -> None:
        """Advance one integration slice of length ``ds`` (mm).

        Drifts to each un-applied gap centre reached within the slice, applies
        the thin-gap kick, then drifts the remainder.  Position is the
        accumulated ``_z_cursor`` (never ``_step_idx*ds``) and gaps fire by
        index, so this stays correct for any — possibly mixed — ``ds`` (the SC
        half-steps and trailing remainders the tracker uses) and for a gap that
        lands exactly on a slice boundary.  Drifts are capped at the resolved
        length so βg≤0 provisional lengths add no phantom tail.
        """
        if self._step_idx == 0:
            self._on_entry(beam)
            self._z_cursor = 0.0
            self._next_gap = 0
        self._step_idx += 1
        target = min(self._z_cursor + ds, self.length)
        gaps = self._gaps
        while (self._next_gap < len(gaps)
               and gaps[self._next_gap].z_center_mm <= target + 1e-9):
            gap = gaps[self._next_gap]
            self._drift_beam(beam, gap.z_center_mm - self._z_cursor)
            self._apply_gap(beam, gap)
            self._z_cursor = gap.z_center_mm
            self._next_gap += 1
        if target > self._z_cursor:
            self._drift_beam(beam, target - self._z_cursor)
            self._z_cursor = target

    # ------------------------------------------------------------------ #
    # FieldMapElement contract — reference / envelope / matrix
    # ------------------------------------------------------------------ #
    def advance_ref(self, ref) -> None:
        """Advance the reference particle through the whole cavity (envelope/matrix)."""
        if self.frequency_mhz > 0 and ref.frequency > 0 and self.frequency_mhz != ref.frequency:
            ref.frequency = self.frequency_mhz
        self._snapshot_entry(ref)
        gaps = self._gaps
        if self.sync_phase and self._sync_offset_deg is None:
            self._calibrate_sync_phase(ref)
        z = 0.0
        for gap in gaps:
            self._ref_drift(ref, gap.z_center_mm - z)
            ttf = self._ttf_correction(ref.beta, gap.kind)
            ref.w_kin += abs(ref.species.charge) * self._v_of(gap) * ttf * np.cos(
                self._phi_gap_rad(ref, gap))
            z = gap.z_center_mm
        self._ref_drift(ref, self.length - z)

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        """Advance the reference over a sub-range (envelope SC bundle loop).

        The envelope drives the position via explicit contiguous [z_from, z_to]
        ranges.  The half-open rule ``z_from < z_center <= z_to`` fires each gap
        in exactly one bundle — the SAME rule ``fitted_matrix_slice`` uses for
        the same range, so the reference advance and the transfer matrix stay
        consistent bundle-for-bundle (and boundary-aligned gaps fire once).
        """
        if z_to_mm <= z_from_mm:
            return
        if z_from_mm == 0.0:
            if self.frequency_mhz > 0 and ref.frequency > 0 and self.frequency_mhz != ref.frequency:
                ref.frequency = self.frequency_mhz
            self._snapshot_entry(ref)
            if self.sync_phase and self._sync_offset_deg is None:
                self._calibrate_sync_phase(ref)
        gaps = self._ensure_gaps(ref)
        cursor = z_from_mm
        target = min(z_to_mm, self.length)
        for gap in gaps:
            if z_from_mm < gap.z_center_mm <= target:
                self._ref_drift(ref, gap.z_center_mm - cursor)
                ttf = self._ttf_correction(ref.beta, gap.kind)
                ref.w_kin += abs(ref.species.charge) * self._v_of(gap) * ttf * np.cos(
                    self._phi_gap_rad(ref, gap))
                cursor = gap.z_center_mm
        if target > cursor:
            self._ref_drift(ref, target - cursor)

    def _gap_kick_matrix(self, ref, gap: _Gap) -> np.ndarray:
        """Linearized 6×6 for one gap (mirrors RFGap.kick_matrix), no ref mutation.

        |q| (abs-charge) convention, matching :meth:`_apply_gap`.
        """
        M = np.eye(6)
        charge = abs(ref.species.charge)
        mass = ref.species.mass
        V = self._v_of(gap)
        ttf = self._ttf_correction(ref.beta, gap.kind)
        phi = self._phi_gap_rad(ref, gap)
        bg_old = ref.bg
        rc = ref.copy()
        rc.w_kin += charge * V * ttf * np.cos(phi)
        bg_new = rc.bg
        wl = rc.wavelength
        damping = bg_old / bg_new
        M[1, 1] = damping
        M[3, 3] = damping
        if mass > 0 and bg_new > 0 and wl > 0:
            k_rf = -np.pi * charge * V * ttf * np.sin(phi) / (mass * bg_new ** 3 * wl)
            M[1, 0] = k_rf * 1e3
            M[3, 2] = k_rf * 1e3
        M[5, 4] = -charge * V * ttf * np.sin(phi) * (np.pi / 180.0)
        # W-dependent synchronism factor (see _sync_delta): asymmetric
        # diagonal terms with the determinant kept exactly 1.  The
        # envelope engine pushes the beam CENTROID through this same
        # matrix, so with a nonzero longitudinal centroid offset the
        # envelope centroid contracts by ~delta per gap relative to MP
        # tracking (which is partran-faithful and carries no factor) —
        # the same deliberate two-engine split as the sigma's; only the
        # second moments are validated against the TW matrix export.
        delta = self._sync_delta(ref, gap)
        if delta != 0.0:
            M[5, 5] = 1.0 + delta
            M[4, 4] = 1.0 / (1.0 + delta)
        return M

    def fitted_matrix_slice(self, ref, ds_mm: float, _z_from_mm=None) -> np.ndarray:
        """6×6 map for a slice (envelope SC).

        The envelope passes an explicit z-cursor ``_z_from_mm``; the half-open
        gap rule ``z_from < z_center <= z_from+ds`` matches ``advance_ref_over``
        so the matrix and the reference advance stay consistent per bundle (and
        a boundary-aligned gap fires once).  ``ref`` is the slice-entry state
        and is copied — never mutated.
        """
        if ds_mm <= 0:
            return np.eye(6)
        if _z_from_mm is None:
            # standalone / sequential (e.g. fitted_matrix): accumulate the cursor
            if self._step_idx == 0:
                self._snapshot_entry(ref)
                self._z_cursor = 0.0
            self._step_idx += 1
            z_from = self._z_cursor
            self._z_cursor = z_from + ds_mm
        else:
            z_from = float(_z_from_mm)
            if z_from == 0.0:
                self._snapshot_entry(ref)
        gaps = self._gaps if self._gaps is not None else self._ensure_gaps(ref)
        target = min(z_from + ds_mm, self.length)
        rc = ref.copy()
        M = np.eye(6)
        cursor = z_from
        for gap in gaps:
            if z_from < gap.z_center_mm <= target:
                seg = gap.z_center_mm - cursor
                M = self._drift_matrix(rc, seg) @ M
                self._ref_drift(rc, seg)
                M = self._gap_kick_matrix(rc, gap) @ M
                rc.w_kin += abs(rc.species.charge) * self._v_of(gap) * self._ttf_correction(
                    rc.beta, gap.kind) * np.cos(self._phi_gap_rad(rc, gap))
                cursor = gap.z_center_mm
        if target > cursor:
            M = self._drift_matrix(rc, target - cursor) @ M
        return M

    def fitted_matrix(self, ref) -> np.ndarray:
        """Full-cavity 6×6 (envelope/matrix) on a fresh cursor.

        Saves/restores ALL per-run cursor + phase state so a mid-sequence call
        (e.g. a diagnostic probe) can't corrupt an in-progress slice walk.
        """
        saved = (self._step_idx, self._z_cursor, self._next_gap,
                 self._phi_s_at_entrance, self._phi_s_at_gap1,
                 self._sync_offset_deg)
        self._step_idx = 0
        self._next_gap = 0
        M = self.fitted_matrix_slice(ref, self.length)
        (self._step_idx, self._z_cursor, self._next_gap,
         self._phi_s_at_entrance, self._phi_s_at_gap1,
         self._sync_offset_deg) = saved
        return M

    def reset_run_state(self) -> None:
        self._step_idx = 0
        self._z_cursor = 0.0
        self._next_gap = 0
        self._phi_s_at_entrance = 0.0
        self._phi_s_at_gap1 = 0.0
        self._sync_offset_deg = None
        if self.beta_g <= 0:
            self._gaps = None          # β-dependent geometry: re-resolve per run
            # Restore the PROVISIONAL over-estimate length.  Run 1 resolved
            # ``self.length`` to the true (energy-dependent) value; if a
            # re-run at a different energy sized its slice count from that
            # stale length, the tracker's Σds could stop short of the new
            # cavity and the tail gaps would never fire (measured −6 MeV on
            # a 45→235 MeV reuse).  The over-estimate guarantees coverage;
            # track paths cap at the re-resolved true length as on run 1.
            self.length = float(
                self.n_cells
                * self._interior_cell_length(self._PROVISIONAL_BETA))

    def _has_electric_channel(self) -> bool:
        """NCELLS is always an RF (electric) cavity — used by bunching/SC hooks."""
        return True
