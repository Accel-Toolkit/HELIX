# linac_gen/tracking/backtrack.py
"""Backward transport of a particle distribution through a lattice range.

Walks a beam from the EXIT of ``elements[end]`` to the ENTRANCE of
``elements[start]`` by applying, in reverse element/sub-step order, the
exact algebraic inverse of every forward operation the multi-particle
tracker would perform.

Physics formulation
-------------------
*Algebraic undo, no coordinate flip.*  For each forward operation
``v_out = Φ(v_in)`` the backward walk applies ``Φ⁻¹``.  This is
equivalent to the textbook time-reversal recipe (flip the momentum-like
coordinates with ``F = diag(+1,−1,+1,−1,−1,+1)``, track forward through
the reversed lattice, flip back): time-reversal-even maps satisfy
``M⁻¹ = F·M·F``.  The undo formulation is used because solenoids and
dipoles then need no field-sign bookkeeping and every mid-walk
diagnostic (σ, ε, Twiss) is directly physical.

Matrix inverses use :func:`numpy.linalg.inv` — HELIX matrices in
(mm, mrad, deg, MeV) units are *not* symplectic (RF adiabatic damping
puts ``bg_old/bg_new`` on the diagonal), so the cheap symplectic
inverse would be silently wrong for every accelerating element.

Reference-particle strategy
---------------------------
``advance_ref`` is ``+=``-hardcoded throughout HELIX and is never
called in reverse.  Instead :func:`build_replay_table` replays the
design reference FORWARD over ``elements[0..end]`` once (the same idiom
``compute_transfer_matrix`` uses, plus the tracker's LatticeCommand /
SpaceChargeComp / DC-transition bookkeeping) and records a
:class:`BoundaryState` at every element boundary.  The backward walk
*sets* the beam's reference from the table at each boundary.  This also
lets SET_SYNC_PHASE cavities calibrate with correct forward semantics
(the calibration is left on the element and reused).

Space charge
------------
The forward tracker applies Strang bundles ``half-map → SC(+ds) →
half-map``; the SC kick depends only on particle positions, which the
inverse of the trailing half-map recovers exactly.  The backward bundle
is therefore ``inv(M₂) → SC kick SUBTRACTED → inv(M₁)``.  Subtraction
is implemented kernel-agnostically by flipping the momentum-like
columns (x′, y′, dW) around the ordinary forward kick call — no SC
kernel needs to know about direction.

PIC grid caveat: with ``grid_mode="fixed"`` the forward solver freezes
its grid on the run's FIRST kick, and a fresh backward solver cannot
reproduce those bounds (they derive from the very distribution being
reconstructed).  Pass the forward run's own ``pic_solver`` to reuse its
frozen grid (exact undo), or the backward walk falls back to per-kick
adaptive grids with a warning.  With ``grid_mode="adaptive"`` forward
and backward evaluate every kick on identical positions → identical
grids → exact subtraction (round-trip ~1e-6, float accumulation only).
The analytic DC kernels are stateless and always undo exactly.

Invertibility limits (documented, guarded)
------------------------------------------
* Aperture losses are non-invertible: the backward walk applies NO
  cuts; a warning notes the reconstruction covers survivors only.
* Field maps (``field_map_mode="rk4"``, the default since 2026-07-11)
  are undone EXACTLY: each forward kick–drift call is closed-form
  invertible, and a zero-particle replay of the literal tracker
  schedule supplies bit-identical per-call entry references and
  SET_SYNC_PHASE calibration.  Round trips close at float round-off
  (~1e-13 over the full MEBT).  ``field_map_mode="linear"`` keeps the
  v1 inverted fitted-matrix path (nonlinear-in-cavity effects not
  undone, ~2 % closure through strong bunchers); RfqCell / VaneRFQ and
  surrogate-driven elements always use it (no exact inverse exists —
  warned for surrogates).
* Quadrupole g3–g6 thin-multipole content is undone exactly (the kicks
  are position-only and never move positions, so re-applying with
  ``fraction=−0.5`` at the same positions subtracts them IEEE-exactly).
* The envelope backward walk keeps the linear per-slice matrices — a
  Σ propagates by matrices by construction.
* DC↔bunched: a bunched beam inverts through RF elements exactly (the
  forward tracker flips ``continuous`` only for DC beams, so a bunched
  beam sees no transition).  ``allow_dc_crossing=True`` asserts the
  forward beam was DC upstream of the first bunching element — the
  backward walk then flips ``beam.continuous`` there, mirroring the
  forward flip, so the space-charge undo dispatches to the DC kernel
  upstream.  A beam still flagged continuous downstream of a bunching
  element is refused as inconsistent input.
* LatticeCommands are skipped (their effects live in the replay table).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.track_state import TrackState
from linac_gen.diagnostics.recorder import DiagnosticRecorder
from linac_gen.elements.base import (
    FieldMapElement, PassiveElement, ThinKickElement, TransferMapElement,
)
from linac_gen.elements.lattice_commands import LatticeCommand
from linac_gen.tracking.tracker import (
    _is_rf_bunching_element, field_map_step_counts,
)

__all__ = [
    "BacktrackWarning", "BoundaryState", "build_replay_table",
    "backtrack_distribution", "backtrack_envelope",
]

# Beam coordinate columns that behave like momenta under time reversal —
# flipped around a forward SC kick call to turn it into an exact
# subtraction (positions are untouched by SC kicks).
_MOMENTUM_COLS = (1, 3, 5)          # x', y', dW


class BacktrackWarning(UserWarning):
    """Non-fatal physics caveats raised during a backward walk."""


@dataclass(frozen=True)
class BoundaryState:
    """Design reference state at one element boundary.

    ``index`` is the boundary number: boundary ``i`` is the ENTRANCE of
    ``elements[i]``; a table over ``elements[0..end]`` has ``end + 2``
    rows, the last being the exit of ``elements[end]``.
    """
    index: int
    s: float                 # mm
    w_kin: float             # MeV
    phi_s: float             # deg
    frequency: float         # MHz
    sc_factor: float         # active SC fraction in force at this boundary
    rf_seen: bool            # an RF bunching element lies upstream (≤ here)
    element_name: str        # element STARTING at this boundary ("EXIT" last)

    def make_ref(self, species) -> ReferenceParticle:
        ref = ReferenceParticle(species=species, w_kin=self.w_kin,
                                frequency=self.frequency)
        ref.s = self.s
        ref.phi_s = self.phi_s
        return ref


@dataclass(frozen=True)
class _FmStep:
    """One forward ``track_rk4`` call of a field map, as the tracker
    schedules it: the call's ``_step_idx``, its per-call ``ds`` (``ds/2``
    inside Strang bundles, full ``ds`` for trailing sub-steps), a COPY of
    the reference state at the call's entrance, and — for step 0 only —
    the FREQ-jump rescale ratio the forward pass applied."""
    step_idx: int
    ds: float
    ref_before: ReferenceParticle
    freq_ratio: float | None = None


def _replay_field_map_ref_states(lattice, element, ref) -> list[_FmStep]:
    """Drive ``ref`` through ``element`` on the LITERAL tracker schedule.

    A zero-particle beam is pushed through the exact per-call sequence of
    ``Tracker._track_field_map`` (``sc_every`` × ``track_rk4(ds/2)``,
    twice per bundle, plus trailing full-``ds`` calls — from the shared
    ``field_map_step_counts``), snapshotting ``ref.copy()`` before each
    call.  ``track_rk4`` early-outs on the empty alive mask only AFTER
    the reference advance and the step-0 entry operations (frequency
    propagation, SET_SYNC_PHASE calibration), so the resulting reference
    states, ψ calibration and ``_phi_s_at_entrance`` are BIT-IDENTICAL
    to a forward tracker run — with zero changes to forward code.

    ``ref`` is mutated to the element-exit state (like ``advance_ref``);
    the returned list carries the per-call entry snapshots.  The
    element's run state is reset first and the calibration is left in
    place for :meth:`FieldMap.untrack_rk4` to reuse.
    """
    from linac_gen.core.beam import Beam
    from linac_gen.tracking.tracker import field_map_step_counts

    n_int, n_sc, ds, sc_every, n_bundles = field_map_step_counts(
        lattice, element)

    element.reset_run_state()
    ghost = Beam(ref=ref, n_particles=1, current=0.0)
    ghost.lost[:] = True                       # zero alive → ref-only walk

    # The forward step-0 FREQ-jump rescale fires when the cavity's
    # frequency differs from the INCOMING ref frequency.
    eff_freq = element.effective_frequency
    ratio = None
    if (eff_freq > 0 and eff_freq != ref.frequency
            and element._has_electric_channel()):
        ratio = eff_freq / ref.frequency

    steps: list[_FmStep] = []

    def _call(ds_call: float) -> None:
        idx = len(steps)
        steps.append(_FmStep(
            step_idx=idx, ds=ds_call, ref_before=ref.copy(),
            freq_ratio=(ratio if idx == 0 else None),
        ))
        element.track_rk4(ghost, ds_call)

    for _b in range(n_bundles):
        for _k in range(sc_every):
            _call(ds / 2)
        for _k in range(sc_every):
            _call(ds / 2)
    for _ in range(n_int - n_bundles * sc_every):
        _call(ds)
    return steps


def build_replay_table(lattice, entrance_ref: ReferenceParticle, *,
                       end: int,
                       field_map_ref_mode: str = "advance_ref",
                       ) -> list[BoundaryState]:
    """Forward-replay the design reference over ``elements[0..end]``.

    Mirrors ``compute_transfer_matrix``'s replay (matrix_tracking.py)
    plus the tracker bookkeeping it lacks: LatticeCommands are applied
    to a throwaway :class:`TrackState` bound to the replay ref (so
    ``SET_BEAM_ENERGY`` etc. shape the table exactly as they shape a
    forward run), ``SpaceChargeComp`` updates the active SC fraction,
    and the first RF bunching element flips ``rf_seen``.  Field maps get
    ``reset_run_state()`` first so SET_SYNC_PHASE calibrates with
    forward semantics; the calibration is left on the element for the
    backward walk to reuse.

    ``field_map_ref_mode`` selects how FieldMap/FieldMap3D advance the
    replay reference:

    * ``"advance_ref"`` (default, the v1 behaviour) — the element's own
      full-``ds`` on-axis integrator.  Differs from the tracker's
      half-step schedule by ~2e-6 relative through strong bunchers.
    * ``"tracker"`` — the zero-particle literal-schedule replay
      (:func:`_replay_field_map_ref_states`): boundary states become
      BIT-IDENTICAL to a forward tracker run.  Used by the exact
      (``field_map_mode="rk4"``) backward walk, where that 2e-6 would
      dominate the closure.
    """
    elements = lattice.elements
    ref = entrance_ref.copy()
    state = TrackState(ref=ref)
    sc_factor = 1.0
    rf_seen = False

    table: list[BoundaryState] = []
    for i in range(end + 1):
        element = elements[i]
        table.append(BoundaryState(
            index=i, s=ref.s, w_kin=ref.w_kin, phi_s=ref.phi_s,
            frequency=ref.frequency, sc_factor=sc_factor,
            rf_seen=rf_seen,
            element_name=getattr(element, "name", "?"),
        ))
        if _is_rf_bunching_element(element):
            rf_seen = True

        if isinstance(element, LatticeCommand):
            element.apply_command(state)
            continue
        if isinstance(element, PassiveElement):
            if hasattr(element, "factor"):
                sc_factor = 1.0 - element.factor
            continue
        if isinstance(element, FieldMapElement):
            from linac_gen.elements.field_map import FieldMap as _FM1D
            from linac_gen.elements.field_map_3d import FieldMap3D as _FM3D
            from linac_gen.elements.superposed_field_map import (
                SuperposedFieldMap as _SFM,
            )
            if (field_map_ref_mode == "tracker"
                    and isinstance(element, (_FM1D, _FM3D, _SFM))):
                # Literal tracker schedule — boundary states bit-identical
                # to a forward run (resets + recalibrates internally).
                _replay_field_map_ref_states(lattice, element, ref)
            else:
                element.reset_run_state()
                element.advance_ref(ref)
        elif isinstance(element, ThinKickElement):
            element.advance_ref(ref)
        else:
            length = getattr(element, "length", 0.0) or 0.0
            if length > 0:
                ref.s += length
                ref.phi_s += 360.0 * length / (ref.beta * ref.wavelength)

    table.append(BoundaryState(
        index=end + 1, s=ref.s, w_kin=ref.w_kin, phi_s=ref.phi_s,
        frequency=ref.frequency, sc_factor=sc_factor, rf_seen=rf_seen,
        element_name="EXIT",
    ))
    return table


def _invert(M: np.ndarray, element_name: str) -> np.ndarray:
    """Inverse of a 6×6 transfer matrix, with a conditioning guard."""
    cond = np.linalg.cond(M)
    if not np.isfinite(cond) or cond > 1e12:
        raise np.linalg.LinAlgError(
            f"transfer matrix of element '{element_name}' is "
            f"near-singular (cond={cond:.3e}) — cannot backtrack "
            "through it.")
    return np.linalg.inv(M)


class _Backtracker:
    """Backward mirror of :class:`~linac_gen.tracking.tracker.Tracker`.

    Bundle structures, SC kick planes and sub-step cadences replicate
    the forward tracker byte-for-byte (via the shared
    ``field_map_step_counts`` helper) so that undoing composes against
    exactly the operations a forward run performed.
    """

    def __init__(self, lattice, beam, table, *, start, end,
                 pic_solver=None, recorder=None, record_substeps=False,
                 progress_callback=None, should_abort=None,
                 dc_upstream=False, field_map_mode="rk4"):
        self.lattice = lattice
        self.beam = beam
        self.table = table
        self.start = start
        self.end = end
        # Same "off" sentinel as Tracker: explicit no-SC intent —
        # disables the DC 2-D kick too (which a bare None does not).
        self._sc_explicitly_off = (pic_solver == "off")
        self.pic_solver = None if self._sc_explicitly_off else pic_solver
        self.recorder = recorder or DiagnosticRecorder()
        self._record_substeps = record_substeps
        self._progress_callback = progress_callback
        self._should_abort = should_abort
        self._dc_upstream = dc_upstream
        self._sc_factor = 1.0
        self.aborted = False
        if field_map_mode not in ("rk4", "linear"):
            raise ValueError(
                f"field_map_mode must be 'rk4' or 'linear', "
                f"got {field_map_mode!r}")
        self._field_map_mode = field_map_mode

    # ------------------------------------------------------------------
    def run(self) -> DiagnosticRecorder:
        # Same refusal as backtrack_distribution, repeated here because
        # this class is also driven directly (tests, Simulation
        # internals) and the fold is no more invertible on that path.
        if getattr(self.beam, "periodic_phase", False) \
                and getattr(self.beam, "bunch_train", False):
            raise ValueError(
                "the supplied beam was tracked with periodic phase "
                "coordinates — the fold discards which bunch of the "
                "train each particle landed in and cannot be undone.")
        elements = self.lattice.elements
        species = self.beam.ref.species
        n_total = self.end - self.start + 1

        exit_state = self.table[self.end + 1]
        self.recorder.record(self.beam, exit_state.s, element_name="INPUT")

        for done, i in enumerate(range(self.end, self.start - 1, -1)):
            if self._should_abort is not None and self._should_abort():
                self.aborted = True
                break
            element = elements[i]
            entry, exit_ = self.table[i], self.table[i + 1]
            # SC strength in force INSIDE this element is the one at its
            # entrance boundary (SpaceChargeComp acts from its position
            # onward in a forward run).
            self._sc_factor = entry.sc_factor
            self._untrack_element(element, entry, exit_, species)
            # Restore the design entrance reference for this element —
            # never advance a ref backwards.
            self.beam.ref = entry.make_ref(species)
            # DC-upstream assertion: mirror the forward tracker's
            # continuous→bunched flip.  Forward, the flag flipped just
            # BEFORE the first bunching element tracked, so backward the
            # beam becomes DC at that element's entrance boundary
            # (entry.rf_seen is False there for the first time).
            if self._dc_upstream and not entry.rf_seen \
                    and not self.beam.continuous:
                self.beam.continuous = True
                # Mirror the forward marker too.  Leaving it set gave a
                # beam that was simultaneously continuous AND a bunch
                # train, and a later forward run of that object would
                # have folded a genuinely DC distribution — the fold
                # tests only the marker, never ``continuous``.
                self.beam.bunch_train = False
                self.beam.bunch_train_frequency = 0.0
            if self._progress_callback is not None:
                try:
                    self._progress_callback(entry.s, done + 1, n_total)
                except Exception:
                    pass
            self.recorder.record(self.beam, entry.s,
                                 element_name=getattr(element, "name", "?"))

        _reverse_recorder_in_place(self.recorder)
        self.recorder.direction = "backward"
        self.recorder.backtrack_range = (self.start, self.end)
        return self.recorder

    # ------------------------------------------------------------------
    def _untrack_element(self, element, entry, exit_, species) -> None:
        if isinstance(element, LatticeCommand):
            # Ref-state effects live in the replay table — but Freq is the
            # one command that also mutates the BEAM (Δφ *= f_new/f_old at
            # the card), which a ref-only table cannot carry.  Undo it here
            # or the exact-closure contract silently degrades on every
            # FREQ-jump deck (×2 Δφ error across a 162.5→325 boundary).
            from linac_gen.elements.lattice_commands import Freq as _Freq
            if (isinstance(element, _Freq)
                    and entry.frequency > 0 and exit_.frequency > 0
                    and exit_.frequency != entry.frequency
                    and not getattr(self.beam, "continuous", False)):
                alive = self.beam.alive_mask
                self.beam.particles[alive, 4] *= (
                    entry.frequency / exit_.frequency)
            return
        # Misalignment wrap: the forward pre-transform (translate −dx,
        # rotate +θ) and post-transform (rotate −θ, translate +dx) are
        # mutual inverses, so the backward wrap is IDENTICAL — same pre,
        # inverse body, same post.
        from linac_gen.elements.multipole import Multipole
        if isinstance(element, (PassiveElement, Multipole)):
            # Multipole self-handles dx/dy/tilt inside its kick — the
            # forward tracker no longer wraps it, so neither do we
            # (symmetry keeps the exact-closure contract).
            dx = dy = tilt_deg = 0.0
        else:
            dx = getattr(element, "dx", 0.0)
            dy = getattr(element, "dy", 0.0)
            tilt_deg = getattr(element, "tilt_deg", 0.0)
        self._pre_transform(dx, dy, tilt_deg)
        try:
            self._untrack_body(element, entry, exit_, species)
        finally:
            self._post_transform(dx, dy, tilt_deg)

    def _untrack_body(self, element, entry, exit_, species) -> None:
        from linac_gen.elements.drift import Drift

        if isinstance(element, ThinKickElement):
            # beam.ref still holds the element's EXIT state here.
            element.inverse_kick(self.beam, entry.make_ref(species))
            return
        if isinstance(element, FieldMapElement):
            self._untrack_field_map(element, entry, species)
            return
        if isinstance(element, TransferMapElement):
            if isinstance(element, Drift):
                self._untrack_drift(element, entry, species)
            else:
                self._untrack_transfer_map(element, entry, species)
            return
        if isinstance(element, PassiveElement):
            # Aperture: NO cuts backward (losses are non-invertible).
            # SpaceChargeComp: handled via the table.  Edge is passive
            # but carries a matrix:
            if hasattr(element, "transfer_matrix"):
                ref_entry = entry.make_ref(species)
                M = element.transfer_matrix(ref_entry)
                self._apply_matrix(_invert(np.asarray(M, dtype=float),
                                           element.name))
            return
        raise NotImplementedError(
            f"element type {type(element).__name__} cannot be "
            "backtracked yet")

    # ------------------------------------------------------------------
    def _untrack_drift(self, element, entry, species) -> None:
        """Inverse of Tracker._track_drift: remainder first, then the
        Strang bundles in reverse — [inv(half), SC⁻, inv(half)]."""
        cfg = getattr(self.lattice, "step_config", None)
        n_int = (cfg.integration_steps_for_length_mm(element.length)
                 if cfg is not None else 2)
        n_sc = (cfg.sc_steps_for_length_mm(element.length)
                if cfg is not None else n_int)
        ds = element.length / n_int
        sc_every = max(1, n_int // n_sc)
        n_bundles = n_int // sc_every
        bundle_len = sc_every * ds

        ref_entry = entry.make_ref(species)   # β constant through a drift
        M_half_inv = _invert(np.asarray(
            element.transfer_matrix(ref_entry, ds=bundle_len / 2),
            dtype=float), element.name)

        remainder_len = element.length - n_bundles * bundle_len
        if remainder_len > 0:
            M_rem = np.asarray(
                element.transfer_matrix(ref_entry, ds=remainder_len),
                dtype=float)
            self._apply_matrix(_invert(M_rem, element.name))
        for _ in range(n_bundles):
            self._apply_matrix(M_half_inv)
            self._apply_sc_kick_negated(bundle_len)
            self._apply_matrix(M_half_inv)

    def _untrack_transfer_map(self, element, entry, species) -> None:
        """Inverse of Tracker._track_transfer_map (n=2 split-operator).

        Forward per bundle: ``track(L/4)`` → SC(L/2) → ``track(L/4)``,
        where a Quadrupole's ``track`` is K½ → M(L/4) → K½ (the g3–g6
        thin-multipole split).  The kicks depend only on positions and
        never modify them, and Bρ is constant through the magnet, so
        re-applying each kick with ``fraction=−0.5`` at the same
        positions is an IEEE-exact subtraction — the quarter undo is
        K⁻ → inv(M) → K⁻, making the g3–g6 content EXACTLY undone
        (v1 warned and dropped it)."""
        n = 2
        ds = element.length / n
        ref_entry = entry.make_ref(species)
        M_q_inv = _invert(np.asarray(
            element.transfer_matrix(ref_entry, ds=ds / 2), dtype=float),
            element.name)
        kick = getattr(element, "_apply_higher_multipole_kick", None)

        def _k_neg() -> None:
            # No-op for non-Quadrupoles and for g3..g6 == 0 (the kick
            # itself early-returns) — the g-free path stays bit-identical
            # to v1.
            if kick is not None:
                kick(self.beam, ds_slice=ds / 2, fraction=-0.5)

        for _ in range(n):
            _k_neg()
            self._apply_matrix(M_q_inv)
            _k_neg()
            self._apply_sc_kick_negated(ds)
            _k_neg()
            self._apply_matrix(M_q_inv)
            _k_neg()

    def _untrack_field_map(self, element, entry, species) -> None:
        """Dispatch a field map to the exact (rk4) or linear inverse.

        The exact path applies to FieldMap / FieldMap3D in
        ``field_map_mode="rk4"`` (the default).  Everything else — the
        explicit ``"linear"`` mode, RfqCell / VaneRFQ (no per-step
        inverse yet), and surrogate-registered elements (a neural
        surrogate has no algebraic inverse) — takes the v1 linear path.
        """
        from linac_gen.elements.field_map import FieldMap as _FM1D
        from linac_gen.elements.field_map_3d import FieldMap3D as _FM3D
        from linac_gen.elements.superposed_field_map import (
            SuperposedFieldMap as _SFM,
        )

        use_rk4 = (self._field_map_mode == "rk4"
                   and isinstance(element, (_FM1D, _FM3D, _SFM)))
        if self._field_map_mode == "rk4" and not use_rk4:
            # Exact mode was requested but this element type has no exact
            # inverse (NCells / RfqCell / VaneRFQ) — warn once per element
            # so the "~1e-13 closure" promise never fails silently (an
            # NCELLS cavity leaves ~5%/35% of the input Δφ/ΔW spreads as
            # linear-inverse residuals).
            warned = getattr(self, "_warned_linear_fallback", None)
            if warned is None:
                warned = self._warned_linear_fallback = set()
            if element.name not in warned:
                warned.add(element.name)
                warnings.warn(
                    f"'{element.name}' ({type(element).__name__}) has no "
                    "exact backward integrator — falling back to the "
                    "LINEAR field-map inverse for this element; "
                    "longitudinal closure is degraded.",
                    BacktrackWarning, stacklevel=4)
        if use_rk4:
            from linac_gen.surrogates import registry as _surr_reg
            if (_surr_reg.is_mp_enabled()
                    and _surr_reg.get_for_element(element)
                    is not None):
                warnings.warn(
                    f"'{element.name}' is driven by a surrogate in "
                    "forward MP runs — a surrogate has no algebraic "
                    "inverse, falling back to the LINEAR field-map "
                    "inverse for this element.",
                    BacktrackWarning, stacklevel=4)
                use_rk4 = False
        if use_rk4:
            self._untrack_field_map_rk4(element, entry, species)
        else:
            self._untrack_field_map_linear(element, entry, species)

    def _untrack_field_map_rk4(self, element, entry, species) -> None:
        """EXACT inverse of Tracker._track_field_map.

        A zero-particle replay reconstructs the per-call entry reference
        states on the literal tracker schedule (and re-establishes the
        SET_SYNC_PHASE calibration on the element); each forward
        ``track_rk4`` call is then undone in exact reverse order via the
        element's closed-form ``untrack_rk4``.  The SC undo is byte-
        identical to the linear path: same bundle boundaries, same
        ``bundle_len``, same ``.scc`` interpolation — only the half-
        bundle CONTENT changed from inverted fitted matrices to exact
        per-step inverses.  Round-trip residual: float round-off
        (~1e-12 per element), vs ~1e-2 for the linear fit through
        strong bunchers.
        """
        n_int, n_sc, ds, sc_every, n_bundles = field_map_step_counts(
            self.lattice, element)
        bundle_len = sc_every * ds

        # .scc local compensation profile — mirror the forward tracker.
        fd = getattr(element, "field_data", None)
        scc_prof = getattr(fd, "scc_profile", None) if fd is not None else None
        scc_scale = getattr(fd, "scc_scale", 0.0) if fd is not None else 0.0
        scc_mode = getattr(fd, "scc_mode", 0) if fd is not None else 0
        scc_active = (scc_prof is not None and scc_scale != 0.0
                      and scc_mode == 0)

        steps = _replay_field_map_ref_states(
            self.lattice, element, entry.make_ref(species))

        def _undo(st: _FmStep) -> None:
            element.untrack_rk4(self.beam, st.ds, st.ref_before,
                                st.step_idx, freq_ratio=st.freq_ratio)

        # Trailing full-ds calls first (they were forward-last).
        n_bundle_calls = 2 * sc_every * n_bundles
        for st in reversed(steps[n_bundle_calls:]):
            _undo(st)
        for b in reversed(range(n_bundles)):
            first = steps[b * 2 * sc_every: b * 2 * sc_every + sc_every]
            second = steps[b * 2 * sc_every + sc_every:
                           (b + 1) * 2 * sc_every]
            for st in reversed(second):
                _undo(st)
            if scc_active:
                z_mid = (b + 0.5) * bundle_len
                scc_z = float(np.interp(z_mid, scc_prof[:, 0],
                                        scc_prof[:, 1]))
                local_factor = max(0.0, 1.0 - scc_scale * scc_z)
                saved = self._sc_factor
                self._sc_factor = local_factor
                self._apply_sc_kick_negated(bundle_len)
                self._sc_factor = saved
            else:
                self._apply_sc_kick_negated(bundle_len)
            for st in reversed(first):
                _undo(st)

    def _untrack_field_map_linear(self, element, entry, species) -> None:
        """Inverse of Tracker._track_field_map using inverted per-slice
        fitted (linear) matrices — v1 fidelity, exactly the envelope
        solver's slicing.  Ops are built FORWARD (so SET_SYNC_PHASE
        calibration and the ref evolution are the replay's), then
        applied in reverse: inv(M₂) → SC⁻(bundle) → inv(M₁)."""
        from linac_gen.tracking.envelope import _fitted_matrix_slice_at

        n_int, n_sc, ds, sc_every, n_bundles = field_map_step_counts(
            self.lattice, element)
        bundle_len = sc_every * ds

        # .scc local compensation profile — mirror the forward tracker.
        fd = getattr(element, "field_data", None)
        scc_prof = getattr(fd, "scc_profile", None) if fd is not None else None
        scc_scale = getattr(fd, "scc_scale", 0.0) if fd is not None else 0.0
        scc_mode = getattr(fd, "scc_mode", 0) if fd is not None else 0
        scc_active = (scc_prof is not None and scc_scale != 0.0
                      and scc_mode == 0)

        # ---- forward pass: build the op stream at replay-consistent refs
        element.reset_run_state()
        ref = entry.make_ref(species)
        z = 0.0
        bundle_ops = []                  # (M1, M2, z_mid)
        for b in range(n_bundles):
            half = bundle_len / 2
            M1 = np.asarray(_fitted_matrix_slice_at(element, ref, half, z),
                            dtype=float)
            element.advance_ref_over(ref, z, z + half)
            M2 = np.asarray(_fitted_matrix_slice_at(element, ref, half,
                                                    z + half), dtype=float)
            element.advance_ref_over(ref, z + half, z + bundle_len)
            bundle_ops.append((M1, M2, (b + 0.5) * bundle_len))
            z += bundle_len
        remainder = element.length - n_bundles * bundle_len
        M_rem = None
        if remainder > 1e-12:
            M_rem = np.asarray(_fitted_matrix_slice_at(element, ref,
                                                       remainder, z),
                               dtype=float)

        # ---- backward pass
        if M_rem is not None:
            self._apply_matrix(_invert(M_rem, element.name))
        for M1, M2, z_mid in reversed(bundle_ops):
            self._apply_matrix(_invert(M2, element.name))
            if scc_active:
                scc_z = float(np.interp(z_mid, scc_prof[:, 0],
                                        scc_prof[:, 1]))
                local_factor = max(0.0, 1.0 - scc_scale * scc_z)
                saved = self._sc_factor
                self._sc_factor = local_factor
                self._apply_sc_kick_negated(bundle_len)
                self._sc_factor = saved
            else:
                self._apply_sc_kick_negated(bundle_len)
            self._apply_matrix(_invert(M1, element.name))

    # ------------------------------------------------------------------
    def _apply_matrix(self, M: np.ndarray) -> None:
        alive = self.beam.alive_mask
        self.beam.particles[alive, :6] = self.beam.particles[alive, :6] @ M.T

    def _apply_sc_kick_negated(self, ds_mm: float) -> None:
        """Subtract the forward SC kick.

        SC kicks write only to the momentum-like columns (x′, y′ and —
        for the 3-D PIC — dW) as functions of positions alone.  Flipping
        those columns around the ordinary FORWARD kick therefore turns
        addition into exact subtraction, independent of kernel
        internals: −(−p + Δ) = p − Δ.
        """
        if ds_mm <= 0 or self._sc_factor <= 0:
            return
        if self._sc_explicitly_off:
            return
        if not getattr(self.beam, "continuous", False) \
                and self.pic_solver is None:
            return
        p = self.beam.particles
        for c in _MOMENTUM_COLS:
            p[:, c] *= -1.0
        try:
            self._forward_sc_kick(ds_mm)
        finally:
            for c in _MOMENTUM_COLS:
                p[:, c] *= -1.0

    def _forward_sc_kick(self, ds_mm: float) -> None:
        """Verbatim mirror of Tracker._apply_sc_kick's dispatch."""
        if getattr(self.beam, "continuous", False):
            cfg = getattr(self.pic_solver, "config", None)
            kind = getattr(cfg, "dc_kernel", "uniform") if cfg else "uniform"
            if kind == "gaussian":
                from linac_gen.pic.pic_solver import kick_continuous_2d_gauss
                kick_continuous_2d_gauss(self.beam, ds_mm * self._sc_factor)
            elif kind == "pic2d":
                from linac_gen.pic.pic_solver import kick_continuous_2d_pic
                nx = int(getattr(cfg, "nx", 128))
                ny = int(getattr(cfg, "ny", 128))
                ext = float(getattr(cfg, "grid_extent", 6.0))
                kick_continuous_2d_pic(self.beam, ds_mm * self._sc_factor,
                                       nx=nx, ny=ny, grid_extent=ext)
            else:
                from linac_gen.pic.pic_solver import kick_continuous_2d
                kick_continuous_2d(self.beam, ds_mm * self._sc_factor)
        elif self.pic_solver:
            self.pic_solver.kick(self.beam, ds_mm * self._sc_factor)

    # ------------------------------------------------------------------
    def _pre_transform(self, dx, dy, tilt_deg) -> None:
        alive = self.beam.alive_mask
        if dx or dy:
            self.beam.particles[alive, 0] -= dx
            self.beam.particles[alive, 2] -= dy
        if tilt_deg:
            theta = math.radians(tilt_deg)
            c, s = math.cos(theta), math.sin(theta)
            x = self.beam.particles[alive, 0].copy()
            y = self.beam.particles[alive, 2].copy()
            xp = self.beam.particles[alive, 1].copy()
            yp = self.beam.particles[alive, 3].copy()
            self.beam.particles[alive, 0] = c * x + s * y
            self.beam.particles[alive, 2] = -s * x + c * y
            self.beam.particles[alive, 1] = c * xp + s * yp
            self.beam.particles[alive, 3] = -s * xp + c * yp

    def _post_transform(self, dx, dy, tilt_deg) -> None:
        alive = self.beam.alive_mask
        if tilt_deg:
            theta = math.radians(tilt_deg)
            c, s = math.cos(theta), math.sin(theta)
            x = self.beam.particles[alive, 0].copy()
            y = self.beam.particles[alive, 2].copy()
            xp = self.beam.particles[alive, 1].copy()
            yp = self.beam.particles[alive, 3].copy()
            self.beam.particles[alive, 0] = c * x - s * y
            self.beam.particles[alive, 2] = s * x + c * y
            self.beam.particles[alive, 1] = c * xp - s * yp
            self.beam.particles[alive, 3] = s * xp + c * yp
        if dx or dy:
            self.beam.particles[alive, 0] += dx
            self.beam.particles[alive, 2] += dy


# ---------------------------------------------------------------------------
def _reverse_recorder_in_place(rec: DiagnosticRecorder) -> None:
    """Reverse every per-step series so consumers see increasing s.

    Applies to every list attribute whose length equals ``len(rec.s)``
    (the per-step series), plus per-axis density row lists.
    """
    n = len(rec.s)
    if n == 0:
        return
    for name, val in list(vars(rec).items()):
        if isinstance(val, list) and len(val) == n:
            setattr(rec, name, list(reversed(val)))
        elif isinstance(val, dict):
            for k, rows in val.items():
                if isinstance(rows, list) and len(rows) == n:
                    val[k] = list(reversed(rows))


# ---------------------------------------------------------------------------
def backtrack_distribution(lattice, beam,
                           entrance_ref: ReferenceParticle, *,
                           start: int = 0, end: int | None = None,
                           sc_config=None, pic_solver=None,
                           recorder: DiagnosticRecorder | None = None,
                           record_substeps: bool = False,
                           progress_callback=None, should_abort=None,
                           allow_dc_crossing: bool = False,
                           approximate_backtracking: bool = False,
                           energy_tolerance: float = 1e-3,
                           energy_hard_limit: float = 5e-2,
                           field_map_mode: str = "rk4"
                           ) -> DiagnosticRecorder:
    """Backtrack ``beam`` from the exit of ``elements[end]`` to the
    entrance of ``elements[start]``.

    Parameters
    ----------
    beam :
        The distribution AT THE EXIT PLANE (e.g. loaded from a measured
        ``.dst`` — ``beam.ref`` then carries the file's energy).  On
        return it holds the reconstructed upstream distribution with
        ``beam.ref`` set to the design entrance-of-``start`` state.
    entrance_ref :
        The DESIGN reference at the lattice entrance (element 0) — the
        machine's fields and phases are evaluated on its forward replay.
    sc_config / pic_solver :
        Space-charge configuration.  Prefer passing the FORWARD run's
        ``pic_solver`` instance when undoing a run you just made — with
        ``grid_mode="fixed"`` its frozen grid is then reused and the SC
        undo is exact.  When only ``sc_config`` is given and it has
        ``grid_mode="fixed"``, the backward solver falls back to
        per-kick adaptive grids (warned): the forward frozen grid
        derives from the distribution being reconstructed and cannot be
        rebuilt in advance.
    energy_tolerance / energy_hard_limit :
        Relative |ΔW|/W between the supplied beam's reference energy and
        the design exit energy: warn above the tolerance, refuse above
        the hard limit (linearised maps and SET_SYNC_PHASE calibrations
        are design-energy-specific).

    Returns
    -------
    DiagnosticRecorder
        Reversed to increasing s (index 0 = reconstructed entrance,
        last = the supplied exit distribution, tagged ``"INPUT"``), with
        ``.direction == "backward"`` and ``.backtrack_range``.
    """
    elements = lattice.elements
    if end is None:
        end = len(elements) - 1
    if not (0 <= start <= end < len(elements)):
        raise ValueError(
            f"invalid backtrack range [start={start}, end={end}] for a "
            f"lattice of {len(elements)} elements (need "
            "0 <= start <= end < n)")

    # The exact per-step inverse needs boundary refs bit-identical to
    # the forward tracker; the linear mode keeps the cheaper advance_ref
    # replay (v1 behaviour, pinned by its own tests).
    table = build_replay_table(
        lattice, entrance_ref, end=end,
        field_map_ref_mode=("tracker" if field_map_mode == "rk4"
                            else "advance_ref"))

    # ---- guards ---------------------------------------------------------
    skipped_cmds = [getattr(e, "name", type(e).__name__)
                    for e in elements[start:end + 1]
                    if isinstance(e, LatticeCommand)]
    if skipped_cmds:
        warnings.warn(
            f"backtrack skips {len(skipped_cmds)} lattice command(s) "
            f"({', '.join(skipped_cmds[:5])}{'…' if len(skipped_cmds) > 5 else ''}) — "
            "their effects are honoured through the forward replay "
            "table only.", BacktrackWarning, stacklevel=2)

    has_aperture = any(
        (getattr(e, "aperture", 0.0) or 0.0) > 0
        for e in elements[start:end + 1]
        if not isinstance(e, LatticeCommand))
    lost = beam.n_particles - beam.n_alive
    if has_aperture or lost > 0:
        warnings.warn(
            "aperture losses are non-invertible — the backward walk "
            "applies no cuts and the reconstruction is valid for the "
            f"surviving particles only ({lost} already lost in the "
            "supplied distribution).", BacktrackWarning, stacklevel=2)

    # Periodic phase coordinates are NOT invertible.  The forward fold
    # discards which bunch of the train a particle ended up in, so the
    # backward walk cannot know whether to put it back at Δφ, Δφ ± P,
    # Δφ ± 2P …  Reconstructing it would silently miss by whole bunch
    # spacings and quietly break the machine-precision closure contract
    # this module exists to provide.  Refuse rather than approximate.
    # The test is the MARKER, not the flag: ``bunch_train`` is set only
    # by a forward run that actually bunched this beam, so it is the
    # thing that says "these coordinates have been folded".  Testing the
    # flag alone refused design-mode backtracking of a freshly generated
    # beam that had never been tracked at all — including the shipped
    # LEBT+RFQ examples, which now enable the flag.  The other hazard,
    # backtracking a .dst exported FROM a folded run, is caught at the
    # CLI where the file source is known (``cli/backtrack.py``).
    if getattr(beam, "periodic_phase", False) \
            and getattr(beam, "bunch_train", False):
        raise ValueError(
            "the supplied beam was tracked with periodic phase "
            "coordinates (BeamConfig.periodic_phase) — the phase fold "
            "is non-invertible (it discards which bunch of the train "
            "each particle landed in), so backtracking it would be "
            "wrong by whole multiples of the bunch spacing.  Re-run the "
            "forward pass with periodic_phase=False to backtrack.")

    # DC↔bunched.  The forward tracker flips `continuous` only for DC
    # beams, so a bunched exit beam inverts through RF elements with no
    # transition at all — the default walk stays bunched everywhere.
    # `allow_dc_crossing=True` asserts the forward beam WAS DC upstream
    # of the first bunching element: the walk then flips the flag there
    # (mirroring the forward flip position) so the SC undo uses the DC
    # kernel upstream.
    crosses_dc = table[start].rf_seen != table[end + 1].rf_seen
    if getattr(beam, "continuous", False) and table[end + 1].rf_seen:
        raise ValueError(
            "the supplied beam is flagged continuous (DC) but the range "
            "end lies downstream of an RF bunching element — a forward "
            "run would have bunched it there.  Check the beam/lattice "
            "pairing (or clear beam.continuous).")
    dc_upstream = False
    if allow_dc_crossing and crosses_dc \
            and not getattr(beam, "continuous", False):
        dc_upstream = True
        warnings.warn(
            "backtracking across the DC↔bunched transition — "
            "beam.continuous is set upstream of the first RF bunching "
            "element and the space-charge undo switches to the DC "
            "kernel there, mirroring the forward flip.",
            BacktrackWarning, stacklevel=2)
    elif crosses_dc and sc_config is not None:
        warnings.warn(
            "the range crosses the first RF bunching element with space "
            "charge enabled — the undo assumes the beam was BUNCHED "
            "upstream too (3-D PIC kernel).  If the forward beam was DC "
            "there (LEBT-style front end), pass allow_dc_crossing=True.",
            BacktrackWarning, stacklevel=2)

    if sc_config is not None and getattr(sc_config, "csr_enabled", False):
        # CSR kicks cannot be undone (no backward CSR model): the
        # reconstruction is APPROXIMATE.  Refuse unless the caller
        # opts in — a warn-and-skip used to let the degraded result
        # pass as an exact undo.
        if not approximate_backtracking:
            raise ValueError(
                "the forward run applied CSR kicks, which have no "
                "backward model — the reconstruction would be "
                "approximate.  Pass approximate_backtracking=True to "
                "accept a CSR-skipped (approximate) undo."
            )
        warnings.warn(
            "approximate_backtracking=True: CSR kicks are SKIPPED on "
            "the backward walk — the reconstructed distribution is "
            "approximate.", BacktrackWarning, stacklevel=2)

    # ---- energy re-centring on the design exit reference ----------------
    design_exit = table[end + 1]
    w_meas = beam.ref.w_kin
    rel = abs(w_meas - design_exit.w_kin) / max(abs(design_exit.w_kin), 1e-12)
    if rel > energy_hard_limit:
        raise ValueError(
            f"supplied beam reference energy {w_meas:.6f} MeV deviates "
            f"{rel * 100:.2f} % from the design exit energy "
            f"{design_exit.w_kin:.6f} MeV — beyond energy_hard_limit "
            f"({energy_hard_limit * 100:.1f} %).  The machine maps are "
            "design-energy-specific; check the lattice/beam pairing or "
            "raise the limit explicitly.")
    if rel > energy_tolerance:
        warnings.warn(
            f"supplied beam energy deviates {rel * 100:.3f} % from the "
            "design exit energy — re-centring dW on the design "
            "reference (maps stay design-energy).",
            BacktrackWarning, stacklevel=2)
    if w_meas != design_exit.w_kin:
        alive = beam.alive_mask
        beam.particles[alive, 5] += (w_meas - design_exit.w_kin)
    beam.ref = design_exit.make_ref(beam.ref.species)

    # ---- PIC solver -------------------------------------------------------
    # Prefer the forward run's solver: with grid_mode="fixed" its frozen
    # grid is exactly the one the forward kicks used.
    pic = pic_solver
    # A forward HaloPicSolver must NOT drive the backward walk (the
    # shipped Simulation.run_backtrack path passes it here directly):
    # the learned corrector is stateful (kick index, anchor cadence,
    # alpha gate) and has no backward model, so reusing it mid-state is
    # a silent approximate undo.  Same contract as CSR: refuse unless
    # the caller opts in, then swap to a fresh plain PicSolver.
    import sys as _sys
    _ml = _sys.modules.get("linac_gen.pic.ml.solver")
    if (_ml is not None and pic is not None and not isinstance(pic, str)
            and isinstance(pic, getattr(_ml, "HaloPicSolver", ()))):
        if not approximate_backtracking:
            raise ValueError(
                "the forward run used sc_backend='halo' (learned defect "
                "corrector), which has no backward model — the backward "
                "walk would reuse the stateful forward solver and the "
                "reconstruction would be approximate.  Pass "
                "approximate_backtracking=True to accept a plain-PIC "
                "(approximate) undo."
            )
        warnings.warn(
            "approximate_backtracking=True: sc_backend='halo' is not "
            "supported on the backward walk — using a fresh plain "
            "PicSolver (no anchors, no corrector) for the backward SC "
            "kicks (approximate undo).", BacktrackWarning, stacklevel=2)
        import dataclasses
        from linac_gen.pic.pic_solver import PicSolver
        _cfg = getattr(pic, "config", None) or sc_config
        if _cfg is not None:
            if getattr(_cfg, "grid_mode", "fixed") == "fixed" \
                    and not getattr(beam, "continuous", False):
                _cfg = dataclasses.replace(_cfg, grid_mode="adaptive")
            _cfg = dataclasses.replace(_cfg, sc_backend="numpy")
            pic = PicSolver(_cfg)
        else:
            pic = None
    if pic is None and sc_config is not None:
        from linac_gen.pic.pic_solver import PicSolver
        # The frozen-grid concern only applies to the 3-D PIC path; the
        # analytic DC kernels are stateless (grid-free) either way.
        if getattr(sc_config, "grid_mode", "fixed") == "fixed" \
                and not getattr(beam, "continuous", False):
            import dataclasses
            warnings.warn(
                "sc_config uses grid_mode='fixed', but the forward run's "
                "frozen PIC grid cannot be rebuilt from the exit "
                "distribution — the backward walk uses per-kick ADAPTIVE "
                "grids instead (approximate SC undo).  For an exact undo "
                "pass the forward run's pic_solver, or track forward "
                "with grid_mode='adaptive'.", BacktrackWarning,
                stacklevel=2)
            sc_config = dataclasses.replace(sc_config, grid_mode="adaptive")
        if getattr(sc_config, "sc_backend", "numpy") == "halo":
            # The learned-defect (halo) backend has no backward model:
            # swapping in the plain PicSolver changes the SC physics of
            # the undo.  Refuse unless the caller opts into an
            # approximate reconstruction (same contract as CSR above —
            # a warn-and-swap used to let the degraded result pass as
            # an exact undo).
            if not approximate_backtracking:
                raise ValueError(
                    "the forward run used sc_backend='halo' (learned "
                    "defect corrector), which has no backward model — "
                    "the backward SC kicks would use the plain "
                    "PicSolver and the reconstruction would be "
                    "approximate.  Pass approximate_backtracking=True "
                    "to accept the plain-PIC (approximate) undo."
                )
            warnings.warn(
                "approximate_backtracking=True: sc_backend='halo' is "
                "not supported on the backtrack rebuild path — using "
                "the plain PicSolver (no anchors, no corrector) for "
                "the backward SC kicks (approximate undo).",
                BacktrackWarning, stacklevel=2)
        pic = PicSolver(sc_config)

    walker = _Backtracker(
        lattice, beam, table, start=start, end=end, pic_solver=pic,
        recorder=recorder, record_substeps=record_substeps,
        progress_callback=progress_callback, should_abort=should_abort,
        dc_upstream=dc_upstream, field_map_mode=field_map_mode)
    return walker.run()


# ---------------------------------------------------------------------------
def backtrack_envelope(lattice, entrance_ref: ReferenceParticle,
                       exit_twiss: dict, *, start: int = 0,
                       end: int | None = None, current: float = 0.0,
                       progress_callback=None, should_abort=None):
    """Backtrack an RMS envelope: Σ_entry = M⁻¹ · Σ · M⁻ᵀ per element.

    Walks the 6×6 sigma matrix from the exit of ``elements[end]`` to the
    entrance of ``elements[start]`` using the inverse of each element's
    linear matrix (``get_element_matrix`` dispatch — the same matrices
    the forward envelope solver composes), with the reference restored
    from the forward replay table at every boundary.

    Parameters
    ----------
    exit_twiss :
        Twiss at the EXIT plane: ``alpha_x, beta_x, emit_x, alpha_y,
        beta_y, emit_y, alpha_z, beta_z, emit_z`` (+ optional
        ``disp_x/disp_xp/disp_y/disp_yp``).  Same conventions as the
        forward solver's ``initial`` dict (β in mm/mrad·π, ε in mm·mrad
        transverse / deg·MeV longitudinal).
    current :
        Must be 0 in v1.  Backward envelope transport with space charge
        is ill-posed as a marching problem (the SC kick at each step
        depends on the unknown upstream σ); use
        ``linac_gen.matching.periodic.find_sc_matched_input_twiss`` —
        forward shooting — for the SC-consistent reconstruction.

    Returns
    -------
    EnvelopeResults
        Reversed to increasing s (index 0 = reconstructed entrance
        state), tagged ``.direction = "backward"`` and
        ``.backtrack_range = (start, end)``.
    """
    if current and current > 0.0:
        raise ValueError(
            "backtrack_envelope does not support space charge (the "
            "backward SC kick depends on the unknown upstream sigma).  "
            "Use matching.periodic.find_sc_matched_input_twiss — "
            "forward shooting — for an SC-consistent input match, or "
            "backtrack at current=0.")
    elements = lattice.elements
    if end is None:
        end = len(elements) - 1
    if not (0 <= start <= end < len(elements)):
        raise ValueError(
            f"invalid backtrack range [start={start}, end={end}] for a "
            f"lattice of {len(elements)} elements (need "
            "0 <= start <= end < n)")

    from linac_gen.tracking.envelope import (
        EnvelopeResults, EnvelopeSolver, _build_sigma_matrix,
    )
    from linac_gen.tracking.matrix_tracking import get_element_matrix

    table = build_replay_table(lattice, entrance_ref, end=end)
    species = entrance_ref.species

    sigma = _build_sigma_matrix(
        exit_twiss["alpha_x"], exit_twiss["beta_x"], exit_twiss["emit_x"],
        exit_twiss["alpha_y"], exit_twiss["beta_y"], exit_twiss["emit_y"],
        float(exit_twiss.get("alpha_z", 0.0) or 0.0),
        float(exit_twiss.get("beta_z", 1.0) or 1.0),
        float(exit_twiss.get("emit_z", 0.0) or 0.0),
        disp_x=float(exit_twiss.get("disp_x", 0.0) or 0.0),
        disp_xp=float(exit_twiss.get("disp_xp", 0.0) or 0.0),
        disp_y=float(exit_twiss.get("disp_y", 0.0) or 0.0),
        disp_yp=float(exit_twiss.get("disp_yp", 0.0) or 0.0),
    )

    # Shell solver: reuses the forward solver's rich _record (Twiss,
    # eigen-emittances, εt, …) without running it.
    exit_ref = table[end + 1].make_ref(species)
    shell = EnvelopeSolver(lattice, exit_ref, dict(exit_twiss), current=0.0)
    results = EnvelopeResults()
    results.current_mA = 0.0
    results.continuous = False
    results.mass_mev = float(getattr(species, "mass", 0.0))
    shell._results = results
    shell._record(results, sigma, exit_ref, element_name="INPUT")

    n_total = end - start + 1
    for done, i in enumerate(range(end, start - 1, -1)):
        if should_abort is not None and should_abort():
            break
        element = elements[i]
        ref_entry = table[i].make_ref(species)
        M = np.asarray(get_element_matrix(element, ref_entry.copy()),
                       dtype=float)
        M_inv = _invert(M, getattr(element, "name", "?"))
        sigma = M_inv @ sigma @ M_inv.T
        shell._record(results, sigma, ref_entry,
                      element_name=getattr(element, "name", "?"))
        if progress_callback is not None:
            try:
                progress_callback(ref_entry.s, done + 1, n_total)
            except Exception:
                pass

    _reverse_recorder_in_place(results)   # generic per-step list reversal
    results.direction = "backward"
    results.backtrack_range = (start, end)
    return results
