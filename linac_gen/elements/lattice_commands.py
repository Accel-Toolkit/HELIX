"""TraceWin SET / ADJUST commands as first-class lattice elements.

Each ``SET_*`` and ``ADJUST_*`` directive in a TraceWin .dat file is parsed
into a subclass of :class:`LatticeCommand`.  Two distinct subclass
families:

* **Run-time-active** (``SetSyncPhase``, ``SetBeamPhaseError``,
  ``SetBeamE0P0``, ``SetBeamEnergy``, ``SetGaussianCutOff``) override
  :meth:`LatticeCommand.apply_command` to mutate the live
  :class:`linac_gen.core.track_state.TrackState` when the tracking loop
  reaches them.
* **Passive constraint / variable markers** (``SetTwiss``, ``SetPosition``,
  …, ``Adjust``, ``AdjustSteerer``, ``AdjustBeam*``) carry their typed
  arguments but inherit the no-op ``apply_command`` default.  The matcher
  engine (``linac_gen.matching``) collects them into Variables /
  Constraints; the deterministic tracker walks past them.

Every subclass implements :meth:`to_tracewin_args` so the writer can
round-trip the command back to a .dat file losslessly.

PassiveElement contract: ``apply(beam) -> None`` is implemented as a no-op
on the base class — commands never touch the beam state directly; they
only mutate ``TrackState`` (active commands) or are introspected by the
matcher (passive).
"""
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

from linac_gen.elements.base import PassiveElement


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class LatticeCommand(PassiveElement):
    """Base class for every TraceWin SET_* / ADJUST_* command.

    Subclasses set the class variable ``KEYWORD`` to the .dat keyword
    they emit (e.g. ``"SET_SYNC_PHASE"``) and override
    :meth:`to_tracewin_args` to return the positional args list.

    Active subclasses also override :meth:`apply_command`.
    """

    KEYWORD: ClassVar[str] = "LATTICE_COMMAND"  # subclass overrides

    def __init__(self, name: str):
        super().__init__(name=name)

    # ----- PassiveElement contract --------------------------------------
    def apply(self, beam) -> None:  # noqa: D401 — explicit no-op
        """No-op.  Lattice commands never mutate the beam directly."""
        return None

    # ----- LatticeCommand-specific hooks --------------------------------
    def apply_command(self, track_state) -> None:
        """Mutate the run-time tracking state.  Default no-op."""
        return None

    def to_tracewin_args(self) -> List[str]:
        """Positional args (as strings) for the .dat round-trip writer."""
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    """Format a number with enough precision for a parse → write → parse
    round-trip; pass through non-numeric strings unchanged."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.10g}"
    return str(v)


# ---------------------------------------------------------------------------
# RUN-TIME ACTIVE commands
# ---------------------------------------------------------------------------
class Freq(LatticeCommand):
    """``FREQ`` — switch the machine RF clock at this lattice position.

    TraceWin semantics: the RF reference frequency that the running clock
    (``ref.phi_s``) uses changes AT THE CARD, not at the next RF element.
    With an input beam defined at a different frequency (e.g. the PIP-II
    HB650 example: beam 804.6 MHz, lattice ``FREQ 804.96``), switching only
    at the first cavity leaves the clock running slow over the entrance
    section — a ~0.5° arrival-phase error that seeds a spurious relative
    synchrotron oscillation against TraceWin (validated on ``fnalscl``).

    ``apply_command`` converts the machine clock and the beam's longitudinal
    phase coordinate to the new frequency's degrees, time-continuously:

    * ``ref.phi_s   *= f_new/f_old``  (same instant, new degrees)
    * ``ref.frequency = f_new``
    * MP beams: ``Δφ *= f_new/f_old`` (mirrors the per-element FREQ-jump
      rescale in ``NCells._on_entry`` / ``FieldMap``, which stays as a
      fallback for hand-built lattices and becomes a no-op after this).

    The envelope σ-matrix rescale (exact ``D Σ Dᵀ``) is applied by
    ``EnvelopeSolver`` when it observes ``ref.frequency`` change across
    ``apply_command`` — σ is a propagation-loop local there.  The bunch
    repetition frequency used for space-charge Q = I/f_bunch is pinned at
    beam/solver creation and intentionally NOT touched.  The matrix-mode
    path keeps its own frequency handling.
    """

    KEYWORD = "FREQ"

    def __init__(self, name: str, frequency_mhz: float = 0.0):
        super().__init__(name=name)
        self.frequency_mhz = float(frequency_mhz)

    def apply_command(self, track_state) -> None:
        ref = getattr(track_state, "ref", None)
        if ref is None or self.frequency_mhz <= 0:
            return
        old = float(getattr(ref, "frequency", 0.0) or 0.0)
        if old <= 0:
            ref.frequency = self.frequency_mhz
            return
        if old == self.frequency_mhz:
            return
        ratio = self.frequency_mhz / old
        ref.phi_s = ref.phi_s * ratio
        ref.frequency = self.frequency_mhz
        beam = getattr(track_state, "beam", None)
        if (beam is not None
                and getattr(beam, "particles", None) is not None
                and not getattr(beam, "continuous", False)):
            beam.particles[beam.alive_mask, 4] *= ratio

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.frequency_mhz)]


class SetSyncPhase(LatticeCommand):
    """``SET_SYNC_PHASE`` — interpret following cavity phases as φ_s.

    Sticky from the position where it appears until the end of the lattice
    (or until a hypothetical reset; TraceWin offers no explicit reset).
    """

    KEYWORD = "SET_SYNC_PHASE"

    def __init__(self, name: str):
        super().__init__(name=name)

    def apply_command(self, track_state) -> None:
        track_state.sync_phase_mode = True

    def to_tracewin_args(self) -> List[str]:
        return []


class SetBeamPhaseError(LatticeCommand):
    """``SET_BEAM_PHASE_ERROR Dp(deg) RandomFlag``.

    ``Dp = 0`` clears the accumulated phase shift; otherwise adds Dp to the
    running offset that following RF cavities see.  ``RandomFlag = 1``
    (draw a random offset) is not yet implemented.
    """

    KEYWORD = "SET_BEAM_PHASE_ERROR"

    def __init__(self, name: str, dphi_deg: float, random_flag: int = 0):
        super().__init__(name=name)
        self.dphi_deg = float(dphi_deg)
        self.random_flag = int(random_flag)

    def apply_command(self, track_state) -> None:
        if self.random_flag:
            raise NotImplementedError(
                "SET_BEAM_PHASE_ERROR with RandomFlag=1 (random draw) "
                "is not implemented; only deterministic Dp is supported."
            )
        if self.dphi_deg == 0.0:
            track_state.phase_ref_shift = 0.0
        else:
            track_state.phase_ref_shift += self.dphi_deg

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.dphi_deg), _fmt(self.random_flag)]


class SetBeamE0P0(LatticeCommand):
    """``SET_BEAM_E0_P0 k DeltaE(MeV) DeltaPhi(deg) ke kp``.

    When ``ke != 0``: shifts the reference particle's kinetic energy by
    ``DeltaE``.  When ``kp != 0``: shifts the running phase offset by
    ``DeltaPhi``.  ``k`` is the matching-criterion weight (used by the
    matcher; ignored by the deterministic tracker).
    """

    KEYWORD = "SET_BEAM_E0_P0"

    def __init__(self, name: str,
                 k: int = 0,
                 dE_MeV: float = 0.0, dphi_deg: float = 0.0,
                 ke: int = 0, kp: int = 0):
        super().__init__(name=name)
        self.k = int(k)
        self.dE_MeV = float(dE_MeV)
        self.dphi_deg = float(dphi_deg)
        self.ke = int(ke)
        self.kp = int(kp)

    def apply_command(self, track_state) -> None:
        if self.ke and track_state.ref is not None:
            track_state.ref.w_kin = track_state.ref.w_kin + self.dE_MeV
        if self.kp:
            track_state.phase_ref_shift += self.dphi_deg

    def to_tracewin_args(self) -> List[str]:
        return [
            _fmt(self.k), _fmt(self.dE_MeV), _fmt(self.dphi_deg),
            _fmt(self.ke), _fmt(self.kp),
        ]


class SetBeamEnergy(LatticeCommand):
    """``SET_BEAM_ENERGY k Ei(MeV)`` — set the reference energy at this
    position.  Used by TraceWin both as a matching criterion and as a
    runtime injection; we honour the latter and ignore ``k``."""

    KEYWORD = "SET_BEAM_ENERGY"

    def __init__(self, name: str, k: int = 0, energy_MeV: float = 0.0):
        super().__init__(name=name)
        self.k = int(k)
        self.energy_MeV = float(energy_MeV)

    def apply_command(self, track_state) -> None:
        if track_state.ref is not None:
            track_state.ref.w_kin = self.energy_MeV

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.energy_MeV)]


class SetGaussianCutOff(LatticeCommand):
    """``SET_GAUSSIAN_CUT_OFF sigma`` — sets the Gaussian-error cutoff
    used by error-study commands.  No effect on a deterministic track."""

    KEYWORD = "SET_GAUSSIAN_CUT_OFF"

    def __init__(self, name: str, sigma: float = 4.0):
        super().__init__(name=name)
        self.sigma = float(sigma)

    def apply_command(self, track_state) -> None:
        track_state.error_cutoff_sigma = self.sigma

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.sigma)]


# ---------------------------------------------------------------------------
# PASSIVE constraint markers (no run-time effect; consumed by matcher)
# ---------------------------------------------------------------------------
class SetTwiss(LatticeCommand):
    """``SET_TWISS family α_x β_x α_y β_y α_z β_z kax kbx kay kby kaz kbz``.

    Imposes Twiss parameters at the output of the following element of
    the named family.  k-flags (1 / 0) declare which parameters are part
    of the matching criterion.
    """

    KEYWORD = "SET_TWISS"

    def __init__(self, name: str, family: str = "",
                 alpha_x: float = 0.0, beta_x: float = 0.0,
                 alpha_y: float = 0.0, beta_y: float = 0.0,
                 alpha_z: float = 0.0, beta_z: float = 0.0,
                 kax: int = 0, kbx: int = 0,
                 kay: int = 0, kby: int = 0,
                 kaz: int = 0, kbz: int = 0):
        super().__init__(name=name)
        self.family = str(family)
        self.alpha_x = float(alpha_x); self.beta_x = float(beta_x)
        self.alpha_y = float(alpha_y); self.beta_y = float(beta_y)
        self.alpha_z = float(alpha_z); self.beta_z = float(beta_z)
        self.kax = int(kax); self.kbx = int(kbx)
        self.kay = int(kay); self.kby = int(kby)
        self.kaz = int(kaz); self.kbz = int(kbz)

    def to_tracewin_args(self) -> List[str]:
        return [
            self.family,
            _fmt(self.alpha_x), _fmt(self.beta_x),
            _fmt(self.alpha_y), _fmt(self.beta_y),
            _fmt(self.alpha_z), _fmt(self.beta_z),
            _fmt(self.kax), _fmt(self.kbx),
            _fmt(self.kay), _fmt(self.kby),
            _fmt(self.kaz), _fmt(self.kbz),
        ]


class SetPosition(LatticeCommand):
    """``SET_POSITION k x(mm) x'(mrad) y(mm) y'(mrad)``."""

    KEYWORD = "SET_POSITION"

    def __init__(self, name: str, k: float = 0.0,
                 x_mm: float = 0.0, xp_mrad: float = 0.0,
                 y_mm: float = 0.0, yp_mrad: float = 0.0):
        super().__init__(name=name)
        self.k = float(k)
        self.x_mm = float(x_mm); self.xp_mrad = float(xp_mrad)
        self.y_mm = float(y_mm); self.yp_mrad = float(yp_mrad)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.x_mm), _fmt(self.xp_mrad),
                _fmt(self.y_mm), _fmt(self.yp_mrad)]


class SetAchromat(LatticeCommand):
    """``SET_ACHROMAT k f1 f2 plane`` — make preceding line achromatic."""

    KEYWORD = "SET_ACHROMAT"

    def __init__(self, name: str, k: int = 0,
                 f1: int = 0, f2: int = 0, plane: int = 0):
        super().__init__(name=name)
        self.k = int(k); self.f1 = int(f1); self.f2 = int(f2)
        self.plane = int(plane)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.f1), _fmt(self.f2), _fmt(self.plane)]


class SetSize(LatticeCommand):
    """``SET_SIZE k x(mm) y(mm) phi(deg)|z(mm) k2``."""

    KEYWORD = "SET_SIZE"

    def __init__(self, name: str, k: float = 0.0,
                 x_mm: float = 0.0, y_mm: float = 0.0,
                 phi_or_z: float = 0.0, k2: int = 0):
        super().__init__(name=name)
        self.k = float(k)
        self.x_mm = float(x_mm); self.y_mm = float(y_mm)
        self.phi_or_z = float(phi_or_z); self.k2 = int(k2)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.x_mm), _fmt(self.y_mm),
                _fmt(self.phi_or_z), _fmt(self.k2)]


class SetSizeMax(LatticeCommand):
    """``SET_SIZE_MAX k N x(mm) y(mm) phi/z k2`` — upper bound over N elems."""

    KEYWORD = "SET_SIZE_MAX"

    def __init__(self, name: str, k: float = 0.0, n_elems: int = 1,
                 x_mm: float = 0.0, y_mm: float = 0.0,
                 phi_or_z: float = 0.0, k2: int = 0):
        super().__init__(name=name)
        self.k = float(k); self.n_elems = int(n_elems)
        self.x_mm = float(x_mm); self.y_mm = float(y_mm)
        self.phi_or_z = float(phi_or_z); self.k2 = int(k2)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.n_elems),
                _fmt(self.x_mm), _fmt(self.y_mm),
                _fmt(self.phi_or_z), _fmt(self.k2)]


class SetSizeMin(LatticeCommand):
    """``SET_SIZE_MIN k N x(mm) y(mm) phi/z k2`` — lower bound."""

    KEYWORD = "SET_SIZE_MIN"

    def __init__(self, name: str, k: float = 0.0, n_elems: int = 1,
                 x_mm: float = 0.0, y_mm: float = 0.0,
                 phi_or_z: float = 0.0, k2: int = 0):
        super().__init__(name=name)
        self.k = float(k); self.n_elems = int(n_elems)
        self.x_mm = float(x_mm); self.y_mm = float(y_mm)
        self.phi_or_z = float(phi_or_z); self.k2 = int(k2)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.n_elems),
                _fmt(self.x_mm), _fmt(self.y_mm),
                _fmt(self.phi_or_z), _fmt(self.k2)]


class SetBeamPhaseAdv(LatticeCommand):
    """``SET_BEAM_PHASE_ADV k N μ_x(deg) μ_y(deg) μ_z(deg)``."""

    KEYWORD = "SET_BEAM_PHASE_ADV"

    def __init__(self, name: str, k: float = 0.0, n_elems: int = 1,
                 mu_x_deg: float = 0.0, mu_y_deg: float = 0.0,
                 mu_z_deg: float = 0.0):
        super().__init__(name=name)
        self.k = float(k); self.n_elems = int(n_elems)
        self.mu_x_deg = float(mu_x_deg)
        self.mu_y_deg = float(mu_y_deg)
        self.mu_z_deg = float(mu_z_deg)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.n_elems),
                _fmt(self.mu_x_deg), _fmt(self.mu_y_deg),
                _fmt(self.mu_z_deg)]


class SetSeparation(LatticeCommand):
    """``SET_SEPARATION k Sx Sy``."""

    KEYWORD = "SET_SEPARATION"

    def __init__(self, name: str, k: float = 0.0,
                 sx: float = 0.0, sy: float = 0.0):
        super().__init__(name=name)
        self.k = float(k); self.sx = float(sx); self.sy = float(sy)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.k), _fmt(self.sx), _fmt(self.sy)]


class SetAdv(LatticeCommand):
    """``SET_ADV kxot kyot`` — zero-current phase-advance law setter."""

    KEYWORD = "SET_ADV"

    def __init__(self, name: str, kxot: float = 0.0, kyot: float = 0.0):
        super().__init__(name=name)
        self.kxot = float(kxot); self.kyot = float(kyot)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.kxot), _fmt(self.kyot)]


class MinEmitGrowth(LatticeCommand):
    """``MIN_EMIT_GROWTH plane weight`` — penalise end-of-line emittance
    growth in one plane.

    Residual is one-sided: ``max(0, ε_out − ε_in)`` so a drop below the
    seed beam's emittance (e.g. from a coupling-resonance crossing) is
    not penalised; recovery up to ε_in is free.

    ``plane`` is ``'X'`` | ``'Y'`` | ``'Z'`` (case-insensitive).
    """

    KEYWORD = "MIN_EMIT_GROWTH"

    def __init__(self, name: str, plane: str = "X", weight: float = 1.0):
        super().__init__(name=name)
        p = str(plane).upper().strip()
        if p not in ("X", "Y", "Z"):
            raise ValueError(
                f"MIN_EMIT_GROWTH plane must be X|Y|Z, got {plane!r}"
            )
        self.plane = p
        self.weight = float(weight)

    def to_tracewin_args(self) -> List[str]:
        return [self.plane, _fmt(self.weight)]


class MinEmit4DGrowth(LatticeCommand):
    """``MIN_EMIT_4D_GROWTH weight tol_4d tol_z`` -- emittance-exchange-aware
    end-of-line emittance-growth constraint.

    Two residuals (one per row) are produced from the envelope results:

    * ``r_4d = max(0, ε_4D_norm_out  - tol_4d · ε_4D_norm_in)``
      where ``ε_4D_norm = sqrt(det(Σ_4x4_transverse)) · (β γ)²`` -- the
      coupled 4-D transverse normalised RMS emittance.  Invariant under
      x-y exchange, so the matcher is free to trade transverse area
      between the two transverse planes (e.g. through a solenoid
      rotation or coupling resonance) without paying a cost.
    * ``r_z  = max(0, ε_z_norm_out  - tol_z  · ε_z_norm_in)``
      where ``ε_z_norm = emit_z · (β γ)`` -- the normalised longitudinal
      RMS emittance.  Same one-sided semantics as MIN_EMIT_GROWTH.

    The tolerance factors are *multiplicative* (e.g. ``tol_4d = 1.10``
    permits 10 % 4-D growth before the residual becomes positive).
    Default ``1.0`` for both -- strict, every drop of growth penalised.

    Use this instead of three separate ``MIN_EMIT_GROWTH X|Y|Z`` cards
    when the lattice naturally exchanges emittance between planes
    (e.g. solenoid + RF cavity cryomodule sections) and the strict
    per-axis constraint would force the matcher onto a structural
    floor.  Keep MIN_EMIT_GROWTH for problems where each plane has a
    hard independent target.
    """

    KEYWORD = "MIN_EMIT_4D_GROWTH"

    def __init__(self, name: str, weight: float = 1.0,
                 tol_4d: float = 1.0, tol_z: float = 1.0):
        super().__init__(name=name)
        self.weight = float(weight)
        self.tol_4d = float(tol_4d)
        self.tol_z = float(tol_z)
        if self.tol_4d < 1.0 or self.tol_z < 1.0:
            raise ValueError(
                f"MIN_EMIT_4D_GROWTH tolerances must be >= 1.0 "
                f"(1.0 = strict, 1.1 = allow 10% growth); got "
                f"tol_4d={tol_4d}, tol_z={tol_z}"
            )

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.weight), _fmt(self.tol_4d), _fmt(self.tol_z)]


class MinTransmission(LatticeCommand):
    """``MIN_TRANSMISSION threshold_pct weight`` — end-of-line
    transmission floor.

    Without this constraint the matcher's emittance metrics
    (``MIN_EMIT_4D_GROWTH`` etc.) are mathematically gameable in MP
    mode: ε is computed from *alive* particles only, so a trial that
    scrapes off the halo on an aperture has *smaller measured* ε in
    the survivors → lower cost → preferred.  Adding this card produces
    an explicit penalty proportional to how far below ``threshold_pct``
    the final transmission falls, so loss-inducing trials cost
    optimiser budget instead of being rewarded.

    Residual is one-sided: ``r = max(0, threshold_pct − T_final) /
    100`` scaled by ``weight`` (divide by 100 so a 1% loss below
    threshold produces a residual of order 0.01 — comparable to the
    other constraint residuals).

    Envelope mode caveat: env tracks the RMS Σ matrix, not individual
    particles, so apertures are no-ops and transmission is always
    100%.  The constraint evaluator returns a zero residual in env
    mode and prints a one-time warning telling the user to switch to
    MP cost-solver if the loss check matters.

    Recommended threshold: 99.0–99.9% for high-power linacs (above
    which any loss is operationally unacceptable); weight: 10–100×
    the per-plane emittance weight (loss is a hard physics constraint,
    emittance is a soft optimisation target).
    """

    KEYWORD = "MIN_TRANSMISSION"

    def __init__(self, name: str, threshold_pct: float = 99.0,
                 weight: float = 1.0):
        super().__init__(name=name)
        self.threshold_pct = float(threshold_pct)
        self.weight = float(weight)
        if not 0.0 <= self.threshold_pct <= 100.0:
            raise ValueError(
                f"MIN_TRANSMISSION threshold_pct must be in [0, 100]; "
                f"got {threshold_pct}"
            )

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.threshold_pct), _fmt(self.weight)]


class SetKeOutMin(LatticeCommand):
    """``SET_KE_OUT_MIN energy_MeV weight`` — output kinetic-energy floor.

    Residual is one-sided: ``max(0, E_floor − W_kin,out)`` so detuning a
    cavity off-crest to gain transverse emittance no longer comes for
    free.  Recommended weight: ~10× the per-plane emittance weights.
    """

    KEYWORD = "SET_KE_OUT_MIN"

    def __init__(self, name: str, energy_mev: float = 0.0,
                 weight: float = 1.0):
        super().__init__(name=name)
        self.energy_mev = float(energy_mev)
        self.weight = float(weight)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.energy_mev), _fmt(self.weight)]


# ---------------------------------------------------------------------------
# ADJUST-family (matcher variables)
# ---------------------------------------------------------------------------
class Adjust(LatticeCommand):
    """``ADJUST N v n vmin vmax start_step kn``.

    ``N`` (family or section), ``v`` (parameter index per the manual),
    ``n`` (link group), ``vmin`` / ``vmax`` (bounds), ``start_step``
    (initial step the optimiser may take), ``kn`` (legacy, unused).

    ``start_step`` is parsed and round-tripped but NOT consumed by any
    HELIX optimiser — they choose their own initial step sizes, so the
    value has no effect on matching results.  It is kept so decks
    written for TraceWin survive a load/save cycle unchanged.
    """

    KEYWORD = "ADJUST"

    def __init__(self, name: str, target: str = "",
                 param_idx: int = 0, link_group: int = 0,
                 vmin: float = 0.0, vmax: float = 0.0,
                 start_step: float = 0.0, kn: int = 0):
        super().__init__(name=name)
        self.target = str(target)
        self.param_idx = int(param_idx)
        self.link_group = int(link_group)
        self.vmin = float(vmin); self.vmax = float(vmax)
        self.start_step = float(start_step); self.kn = int(kn)

    def to_tracewin_args(self) -> List[str]:
        return [self.target, _fmt(self.param_idx),
                _fmt(self.link_group),
                _fmt(self.vmin), _fmt(self.vmax),
                _fmt(self.start_step), _fmt(self.kn)]


class AdjustSteerer(LatticeCommand):
    """``ADJUST_STEERER N vmax first_step``."""

    KEYWORD = "ADJUST_STEERER"

    def __init__(self, name: str, diag_n: int = 0,
                 vmax: float = 0.0, first_step: float = 0.0):
        super().__init__(name=name)
        self.diag_n = int(diag_n)
        self.vmax = float(vmax); self.first_step = float(first_step)

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.diag_n), _fmt(self.vmax), _fmt(self.first_step)]


class AdjustSteererBx(AdjustSteerer):
    """``ADJUST_STEERER_BX N vmax first_step`` — Bx knob only.

    A B_x field kicks **y′**: this card authorizes the ``bx_l`` knob,
    i.e. VERTICAL-plane correction.  (The old docstring said
    "horizontal", contradicting the kick physics in steerer.py and the
    matcher/correction plane mapping.)
    """

    KEYWORD = "ADJUST_STEERER_BX"


class AdjustSteererBy(AdjustSteerer):
    """``ADJUST_STEERER_BY N vmax first_step`` — By knob only.

    A B_y field kicks **x′**: this card authorizes the ``by_l`` knob,
    i.e. HORIZONTAL-plane correction.
    """

    KEYWORD = "ADJUST_STEERER_BY"


class _AdjustBeamFlagBase(LatticeCommand):
    """Shared base for ``ADJUST_BEAM_*`` commands that take a diag-number
    plus a tail of integer flags."""

    KEYWORD = "ADJUST_BEAM_BASE"
    _N_FLAGS: ClassVar[int] = 0

    def __init__(self, name: str, diag_n: int = 0, *flags):
        super().__init__(name=name)
        self.diag_n = int(diag_n)
        flags = list(flags) + [0] * (self._N_FLAGS - len(flags))
        self.flags = [int(f) for f in flags[: self._N_FLAGS]]

    def to_tracewin_args(self) -> List[str]:
        return [_fmt(self.diag_n)] + [_fmt(f) for f in self.flags]


class AdjustBeamTwiss(_AdjustBeamFlagBase):
    """``ADJUST_BEAM_TWISS N AlpX_flag BetX_flag AlpY_flag BetY_flag
    AlpZ_flag BetZ_flag``.  Flags: 1=adjust, 2=couple to previous axis,
    0=skip."""

    KEYWORD = "ADJUST_BEAM_TWISS"
    _N_FLAGS = 6


class AdjustBeamCentroid(_AdjustBeamFlagBase):
    """``ADJUST_BEAM_CENTROID N X Xp Y Yp Z Zp`` (six flags)."""

    KEYWORD = "ADJUST_BEAM_CENTROID"
    _N_FLAGS = 6


class AdjustBeamEmit(_AdjustBeamFlagBase):
    """``ADJUST_BEAM_EMIT N Ex Ey Ez``."""

    KEYWORD = "ADJUST_BEAM_EMIT"
    _N_FLAGS = 3


class AdjustBeamCurrent(_AdjustBeamFlagBase):
    """``ADJUST_BEAM_CURRENT N I_flag``."""

    KEYWORD = "ADJUST_BEAM_CURRENT"
    _N_FLAGS = 1


# ---------------------------------------------------------------------------
# Public registry — keyword → class
# ---------------------------------------------------------------------------
COMMAND_CLASSES: dict = {
    cls.KEYWORD: cls for cls in (
        SetSyncPhase, SetBeamPhaseError, SetBeamE0P0, SetBeamEnergy,
        SetGaussianCutOff,
        SetTwiss, SetPosition, SetAchromat,
        SetSize, SetSizeMax, SetSizeMin,
        SetBeamPhaseAdv, SetSeparation, SetAdv,
        MinEmitGrowth, MinEmit4DGrowth, MinTransmission, SetKeOutMin,
        Adjust, AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
        AdjustBeamTwiss, AdjustBeamCentroid, AdjustBeamEmit,
        AdjustBeamCurrent,
    )
}

__all__ = [
    "LatticeCommand",
    "Freq",
    "SetSyncPhase", "SetBeamPhaseError", "SetBeamE0P0", "SetBeamEnergy",
    "SetGaussianCutOff",
    "SetTwiss", "SetPosition", "SetAchromat",
    "SetSize", "SetSizeMax", "SetSizeMin",
    "SetBeamPhaseAdv", "SetSeparation", "SetAdv",
    "MinEmitGrowth", "MinEmit4DGrowth", "MinTransmission", "SetKeOutMin",
    "Adjust", "AdjustSteerer", "AdjustSteererBx", "AdjustSteererBy",
    "AdjustBeamTwiss", "AdjustBeamCentroid", "AdjustBeamEmit",
    "AdjustBeamCurrent",
    "COMMAND_CLASSES",
]
