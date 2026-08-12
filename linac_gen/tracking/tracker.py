# linac_gen/tracking/tracker.py
"""Multi-particle Tracker: dispatches elements and records diagnostics."""
import warnings

import numpy as np
from linac_gen.core.track_state import TrackState
from linac_gen.elements.base import TransferMapElement, ThinKickElement, FieldMapElement, PassiveElement
from linac_gen.elements.lattice_commands import LatticeCommand
from linac_gen.elements.sc_grid import ScGridDirective
from linac_gen.diagnostics.recorder import DiagnosticRecorder


class PeriodicPhaseWarning(UserWarning):
    """``periodic_phase`` was requested where folding is not meaningful.

    The fold moves a particle by one bunch spacing, which is only a
    real place for it to be when the local RF frequency is a whole
    multiple of the bunch repetition rate.  At a genuine sub-harmonic
    the neighbouring bunch is somewhere else entirely, so the fold is
    skipped and this is raised once.

    Note the failure mode is PLACEMENT, not force: the fold period is
    360·round(ratio), a whole number of local degrees, and cos/sin are
    exactly 360°-periodic in those.  The force only goes wrong for
    folds that a LATER frequency change rescales into a non-period —
    that case raises ``ValueError`` instead, because it cannot be
    undone.
    """


class NoSpaceChargeWarning(UserWarning):
    """A bunched beam carrying current is tracked with NO space-charge
    solver — every SC kick is silently omitted.

    Only raw API calls can reach this state: ``Simulation(lattice, beam)``
    without ``space_charge=`` (or ``Tracker`` without ``pic_solver=``).
    The CLI and GUI always build a solver when the beam current is > 0.
    """


def _is_magnetic_only_fieldmap(element) -> bool:
    """True for any static-B-only field map — solenoid-type element.

    Covers both representations our parser produces:
      * ``FieldMap`` with ``geometry == 1`` (TraceWin ``FIELD_MAP 10``,
        1-D on-axis, ``.bsz``).
      * ``FieldMap3D`` with ``geometry == 7`` (TraceWin ``FIELD_MAP 70``,
        3-D Cartesian triplet, ``.bsx/.bsy/.bsz``).

    Solenoid-type elements need much finer integration sub-stepping than
    the default 100 steps/m at strong B and low β (first-order thin-slice
    integrator has O((K·ds)²) error per sub-step).  RF cavities — which
    carry electric channels — are fine at the default.
    """
    try:
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        from linac_gen.elements.superposed_field_map import (
            SuperposedFieldMap,
        )
    except Exception:
        return False
    if isinstance(element, SuperposedFieldMap):
        # A cluster is solenoid-type iff EVERY child is.
        return all(_is_magnetic_only_fieldmap(child)
                   for _z0, child in element.children)
    if not isinstance(element, (FieldMap, FieldMap3D)):
        return False
    fd = getattr(element, "field_data", None)
    if fd is None or not fd.channels:
        return False
    ok_geoms = {1, 7}     # 1-D on-axis or 3-D Cartesian
    for ch_enum, ch in fd.channels.items():
        if ch_enum.is_electric:
            return False
        if getattr(ch, "geometry", None) not in ok_geoms:
            return False
    return True


# Backward-compatible alias for the narrower old predicate name.
_is_magnetic_only_1d_fieldmap = _is_magnetic_only_fieldmap


def _is_rf_bunching_element(element) -> bool:
    """True if *element* is an RF bunching element — an RFGap, an RFQ
    cell, or a FIELD_MAP carrying an electric channel.  The tracker uses
    this to transition a continuous beam to bunched at the first such
    element.  Pure magnetic FIELD_MAPs (solenoids), zero-length RFGaps
    with voltage=0, and 1-D quadrupole-gradient maps (geometry digit 9)
    don't count.
    """
    try:
        from linac_gen.elements.rf_gap import RFGap
    except Exception:
        RFGap = None
    if RFGap is not None and isinstance(element, RFGap) \
            and abs(getattr(element, "voltage", 0.0) or 0.0) > 0:
        return True
    # RFQ cells are buncher-accelerators by construction (TraceWin RFQ_CELL
    # carries a non-zero V and a vane-tip modulation that bunches
    # longitudinally).  RfqCell is a FieldMapElement subclass but NOT a
    # FieldMap/FieldMap3D, so the channel-based detection below skips it;
    # treat it explicitly here.
    try:
        from linac_gen.elements.rfq_cell import RfqCell
    except Exception:
        RfqCell = None
    if RfqCell is not None and isinstance(element, RfqCell) \
            and abs(getattr(element, "voltage_V", 0.0) or 0.0) > 0:
        return True
    # VaneRFQ is a WHOLE RFQ in one element — same physics as the RfqCell
    # chain it replaces, but it carries its voltage per CellSpan rather
    # than on the element, so the getattr above finds nothing.  Without
    # this branch a DC beam tracked through a VaneRFQ never flips to
    # bunched and the longitudinal dynamics never engage.
    try:
        from linac_gen.elements.vane_rfq import VaneRFQ
    except Exception:
        VaneRFQ = None
    if VaneRFQ is not None and isinstance(element, VaneRFQ) \
            and any(abs(c.voltage_V or 0.0) > 0 for c in element.cells):
        return True
    # NCELLS multi-gap cavities are buncher-accelerators (non-zero EoT).
    # Like RfqCell they subclass FieldMapElement but not FieldMap/FieldMap3D,
    # so the channel-based detection below skips them — treat explicitly.
    try:
        from linac_gen.elements.ncells import NCells
    except Exception:
        NCells = None
    if NCells is not None and isinstance(element, NCells) \
            and abs(getattr(element, "eot_v_per_m", 0.0) or 0.0) > 0:
        return True
    try:
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        from linac_gen.elements.superposed_field_map import (
            SuperposedFieldMap,
        )
    except Exception:
        return False
    if isinstance(element, SuperposedFieldMap):
        # A cluster bunches iff ANY child does (electric non-quad-grad
        # channel).
        return any(_is_rf_bunching_element(child)
                   for _z0, child in element.children)
    if not isinstance(element, (FieldMap, FieldMap3D)):
        return False
    fd = getattr(element, "field_data", None)
    if fd is None:
        return False
    for ch_enum, ch in fd.channels.items():
        if ch_enum.is_electric and int(getattr(ch, "geometry", 0) or 0) != 9:
            return True
    return False


def _is_time_varying_rf_buncher(element) -> bool:
    """True only for a genuinely TIME-VARYING RF buncher.

    Deliberately narrower than :func:`_is_rf_bunching_element`, which
    accepts any ELECTRIC channel and therefore also matches a static
    field — an einzel lens or DC extraction column (``Channel.STAT_E``,
    ``is_electric`` but not ``is_rf``).  A static electrostatic element
    cannot bunch anything, so a beam that meets one is still DC and
    must never be marked a bunch train: folding it would compress a
    genuinely uniform ±180° phase distribution into a fictitious
    bucket.

    Used ONLY to set ``Beam.bunch_train``.  The looser predicate keeps
    driving ``beam.continuous`` so this change cannot alter any
    existing run.
    """
    try:
        from linac_gen.elements.rf_gap import RFGap
    except Exception:
        RFGap = None
    if RFGap is not None and isinstance(element, RFGap) \
            and abs(getattr(element, "voltage", 0.0) or 0.0) > 0:
        return True
    try:
        from linac_gen.elements.rfq_cell import RfqCell
    except Exception:
        RfqCell = None
    if RfqCell is not None and isinstance(element, RfqCell) \
            and abs(getattr(element, "voltage_V", 0.0) or 0.0) > 0:
        return True
    try:
        from linac_gen.elements.ncells import NCells
    except Exception:
        NCells = None
    if NCells is not None and isinstance(element, NCells) \
            and abs(getattr(element, "eot_v_per_m", 0.0) or 0.0) > 0:
        return True
    try:
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D
        from linac_gen.elements.superposed_field_map import (
            SuperposedFieldMap,
        )
    except Exception:
        return False
    if isinstance(element, SuperposedFieldMap):
        return any(_is_time_varying_rf_buncher(child)
                   for _z0, child in element.children)
    if not isinstance(element, (FieldMap, FieldMap3D)):
        return False
    fd = getattr(element, "field_data", None)
    if fd is None:
        return False
    for ch_enum, ch in fd.channels.items():
        if ch_enum.is_electric and ch_enum.is_rf \
                and int(getattr(ch, "geometry", 0) or 0) != 9:
            return True
    return False


def field_map_step_counts(lattice, element):
    """Canonical sub-step cadence for a field-map element.

    Returns ``(n_int, n_sc, ds_mm, sc_every, n_bundles)`` — the SINGLE
    source of truth shared by the multi-particle tracker, the envelope
    solver and the backtracker, so all three walk a field map with
    byte-identical slicing (matrices, SC kick planes and diagnostics
    then compose consistently across modes).

    Rules (in order, each with its physics rationale at the original
    call site in ``Tracker._track_field_map``):

    1. Base counts from ``lattice.step_config`` (PARTRAN_STEP), falling
       back to the element's own ``n_steps``.
    2. Magnetic-only 1-D maps (cryomodule solenoids) get a 5000 steps/m
       floor on the integration count — first-order thin-slice error at
       low β/strong B.  ``n_sc`` deliberately stays at its configured
       cadence (forcing it up collapsed ``sc_every`` to 1 and applied
       ~100× too many PIC kicks).
    3. ``RfqCell`` honours its own auto-selected ``n_steps`` floor
       (0.1 mm per substep) — PARTRAN_STEP at 100/m gives 1 substep on
       a 7 mm cell and a ~5 % synchronous-W shortfall.
    4. RF FIELD_MAP cavities (1-D and 3-D) honour the parser-set
       ``n_steps`` (the map's own grid resolution) — undercutting it
       coarsens the transit-time integral (~130 keV bias on SSR1).
    """
    cfg = getattr(lattice, "step_config", None)
    if cfg is not None:
        n_int = cfg.integration_steps_for_length_mm(element.length)
        n_sc = cfg.sc_steps_for_length_mm(element.length)
    else:
        n_int = max(int(getattr(element, "n_steps", 1) or 1), 1)
        n_sc = n_int

    if _is_magnetic_only_1d_fieldmap(element):
        min_int = int((5000.0 * element.length * 1e-3) + 0.5)
        if min_int > n_int:
            n_int = min_int

    # RFQ: honour the element's own substep count.  VaneRFQ sizes itself
    # at ~2× the .vane slice resolution (~0.13 mm on PXIE); left to a
    # coarse PARTRAN_STEP it gets ~10 mm slices, the RK4 push goes
    # unstable and drives the reference energy negative ("math domain
    # error" out of sqrt).  Same floor as the RfqCell chain it replaces.
    from linac_gen.elements.rfq_cell import RfqCell
    from linac_gen.elements.vane_rfq import VaneRFQ
    if isinstance(element, (RfqCell, VaneRFQ)) and element.n_steps > n_int:
        n_int = element.n_steps

    # NCELLS: honour the element's own ~2-per-cell floor so the SC-kick cadence
    # isn't undercut by a coarse PARTRAN_STEP (the interior gap kicks are exact
    # regardless; this only sets how often space charge is applied).
    from linac_gen.elements.ncells import NCells
    if isinstance(element, NCells) and element.n_steps > n_int:
        n_int = element.n_steps

    from linac_gen.elements.field_map import FieldMap as _FM1D
    from linac_gen.elements.field_map_3d import FieldMap3D as _FM3D
    from linac_gen.elements.superposed_field_map import (
        SuperposedFieldMap as _SFM,
    )
    if isinstance(element, (_FM1D, _FM3D, _SFM)) and element.n_steps > n_int:
        n_int = element.n_steps

    ds_mm = element.length / n_int
    sc_every = max(1, n_int // n_sc)
    n_bundles = n_int // sc_every
    return n_int, n_sc, ds_mm, sc_every, n_bundles


class Tracker:
    """Iterates over lattice elements and dispatches to the correct tracking strategy."""

    def __init__(self, lattice, beam, pic_solver=None, recorder=None,
                 snapshot_locations=None, snapshot_every_n=None,
                 snapshot_elements=None,
                 record_substeps=False,
                 progress_callback=None, should_abort=None,
                 csr_kicker=None,
                 element_entry_hook=None, element_exit_hook=None):
        self.lattice = lattice
        self.beam = beam
        # ``pic_solver="off"`` is the explicit no-space-charge sentinel:
        # identical physics to None, but it declares intent and
        # suppresses NoSpaceChargeWarning.
        if isinstance(pic_solver, str) and pic_solver != "off":
            raise ValueError(
                f"pic_solver={pic_solver!r}: the only string sentinel "
                f"is \"off\" (lowercase); pass a PIC solver instance "
                f"to enable space charge")
        self._sc_explicitly_off = (pic_solver == "off")
        self.pic_solver = None if self._sc_explicitly_off else pic_solver
        # Optional 1-D steady-state CSR kicker — applied inside Dipole
        # elements when present.  None ⇒ CSR disabled.
        self._csr_kicker = csr_kicker
        self.recorder = recorder or DiagnosticRecorder()
        self._sc_factor = 1.0  # space charge compensation factor
        self._snapshot_locations = set(snapshot_locations or [])
        self._snapshot_every_n = snapshot_every_n
        # Snapshot at named elements — robust (name match, not fragile
        # float-s equality like snapshot_locations).  This is what the GUI
        # "snapshot at elements" control feeds.
        self._snapshot_elements = set(snapshot_elements or [])
        # When True, record diagnostics at every internal sub-step bundle
        # inside long drifts and field maps — produces TraceWin-like
        # fine-resolution σ(s) curves at the cost of ~50× more records.
        self._record_substeps = record_substeps
        # Multibunch train hooks (opt-in; default None = zero-cost).
        # ``element_entry_hook(element, index, beam)`` fires immediately
        # before an element is tracked; ``element_exit_hook`` immediately
        # after its physics (before the exit-plane aperture check, so the
        # hook sees the charge that actually traversed the element).
        # Used by linac_gen.train to couple bunches through cavity state.
        self._element_entry_hook = element_entry_hook
        self._element_exit_hook = element_exit_hook
        # Multibunch M6 hybrid replay: per-element static HOM kick
        # attributes (``hom_kick_x``/``hom_kick_y``, mrad — see
        # linac_gen/train/replay.py).  Populated once per run() by a
        # lattice scan; None (every normal run — the attributes default
        # to 0.0) costs one is-None check per element in the loop.
        self._hom_kick_map = None
        # Progress / cancellation hooks.  ``progress_callback`` (if given)
        # is invoked with ``(s_mm, element_index, n_elements)`` after each
        # element so the GUI can update a progress bar using the real
        # longitudinal position.  ``should_abort`` is polled between
        # elements and, if it returns True, the tracker returns early with
        # whatever diagnostics it has collected so far.
        self._progress_callback = progress_callback
        self._should_abort = should_abort
        self.aborted = False
        # ``TrackState`` carries run-time state (energy, phase shift, sync
        # mode, error cutoff) that ``LatticeCommand.apply_command`` mutates
        # at its position in the lattice.  Bind to the live beam ref so
        # ``SetBeamEnergy`` etc. flow straight through to the tracker.
        self.track_state = TrackState(ref=self.beam.ref, beam=self.beam)
        self._warned_no_sc = False
        # Periodic phase coordinates: read once so the hot path (called
        # per element AND per SC kick) is a plain attribute test.
        self._periodic_phase = bool(getattr(self.beam, "periodic_phase",
                                            False))
        self._phase_fold_warned = False
        self._n_phase_folded = 0
        self._pending_bunch_train = False
        # Stamp the run so downstream consumers (notably the GUI display
        # fold) can tell folded results from unfolded ones instead of
        # guessing from a checkbox.
        try:
            self.recorder.periodic_phase = self._periodic_phase
        except Exception:                                   # noqa: BLE001
            pass        # custom/null recorders need not carry provenance
        if self._periodic_phase:
            self._check_periodic_phase_compatible()

    def _warn_if_sc_missing(self, where: str) -> None:
        """Warn (once per run) when a bunched beam carries current but no
        space-charge solver exists — the run omits SC physics entirely.

        Suppressed when the caller passed the explicit ``"off"``
        sentinel: intent declared, no footgun."""
        if self._warned_no_sc or self._sc_explicitly_off:
            return
        current = float(getattr(self.beam, "current", 0.0) or 0.0)
        if current <= 0:
            return
        self._warned_no_sc = True
        warnings.warn(
            f"Tracking a bunched beam carrying {current:g} mA with NO "
            f"space-charge solver ({where}) — all space-charge kicks are "
            "OMITTED. Pass space_charge=SpaceChargeConfig(...) to "
            "Simulation (or pic_solver=... to Tracker); pass "
            "space_charge=\"off\" / pic_solver=\"off\" (or set the beam "
            "current to 0) if tracking without space charge is intended.",
            NoSpaceChargeWarning, stacklevel=3)

    def _apply_train_hom_kick(self, kx: float, ky: float) -> None:
        """Consume the M6-replay ``hom_kick_x``/``hom_kick_y`` element
        attributes: one static transverse kick (mrad) to every alive
        particle at element entry — the replay image of
        ``HomManager.entry_hook``'s live kick, using the SAME guarded
        array ops so a replayed bunch stays bit-identical to the tracked
        bunch it reconstructs."""
        beam = self.beam
        alive = beam.alive_mask
        p = beam.particles
        if kx != 0.0:
            p[alive, 1] += kx
        if ky != 0.0:
            p[alive, 3] += ky

    def run(self) -> DiagnosticRecorder:
        """Track beam through entire lattice."""
        if self.pic_solver is None \
                and not getattr(self.beam, "continuous", False):
            self._warn_if_sc_missing("from the start of the run")
        # Record initial state
        self.recorder.record(self.beam, self.beam.ref.s, element_name="INPUT")
        n_el = len(self.lattice.elements)

        # M6 hybrid replay: collect nonzero static HOM-kick attributes
        # (one O(n_elements) getattr scan; zero float work for normal
        # runs, whose attributes are all at the 0.0 class default).
        _hk = {}
        for _i, _el in enumerate(self.lattice.elements):
            _kx = float(getattr(_el, "hom_kick_x", 0.0) or 0.0)
            _ky = float(getattr(_el, "hom_kick_y", 0.0) or 0.0)
            if _kx != 0.0 or _ky != 0.0:
                _hk[_i] = (_kx, _ky)
        self._hom_kick_map = _hk or None

        for i, element in enumerate(self.lattice.elements):
            if self._should_abort is not None and self._should_abort():
                self.aborted = True
                return self.recorder
            # DC → bunched transition: the first RF bunching element seen
            # by a continuous beam flips the flag.  From this element on,
            # SC dispatches to the normal PIC path and longitudinal
            # dynamics are honoured.  The uniform-phase particles already
            # carry the right longitudinal coordinates for the cavity to
            # bunch them physically — no coordinate remapping needed.
            if getattr(self.beam, "continuous", False) \
                    and _is_rf_bunching_element(element):
                self.beam.continuous = False
                # This beam was injected DC and has just been bunched:
                # from here on it is ONE PERIOD OF A BUNCH TRAIN.  Only
                # such a beam may use periodic phase coordinates — a
                # beam born bunched never gets the marker, so it can
                # never be folded.  The stricter predicate excludes
                # STATIC electric elements (einzel lens, DC extraction
                # column), which satisfy the looser flip test above but
                # cannot bunch anything.
                if _is_time_varying_rf_buncher(element):
                    # Defer by one element: the bunch repetition rate is
                    # the RF clock of the element doing the bunching,
                    # which for an RFGap/FieldMap is only installed on
                    # ``ref`` once that element has been tracked.
                    self._pending_bunch_train = True
                if self.pic_solver is None:
                    # The built-in continuous-beam 2-D analytic kick no
                    # longer applies past this point — without a solver
                    # the rest of the line runs with no SC at all.
                    self._warn_if_sc_missing(
                        f"after the DC-to-bunched transition at "
                        f"'{getattr(element, 'name', element)}'")
            if self._element_entry_hook is not None:
                self._element_entry_hook(element, i, self.beam)
            if self._hom_kick_map is not None:
                _kk = self._hom_kick_map.get(i)
                if _kk is not None:
                    self._apply_train_hom_kick(*_kk)
            self._track_element(element)
            if self._element_exit_hook is not None:
                self._element_exit_hook(element, i, self.beam)
            if self._pending_bunch_train:
                # The bunching element has now run, so ``ref.frequency``
                # holds ITS clock — that, not the beam-config frequency,
                # is the rate at which bunches are produced.  Pinning it
                # here is what makes the fold period right for a beam
                # created at one frequency and bunched at another (a
                # 162.5 MHz config bunched by a 325 MHz gap would
                # otherwise fold with a 720° period and leave every
                # satellite in place, silently).  ``bunch_frequency`` is
                # deliberately NOT touched: it sets the space-charge
                # Q = I/f and changing it would move existing results.
                self._pending_bunch_train = False
                self.beam.bunch_train = True
                self.beam.bunch_train_frequency = float(
                    self.beam.ref.frequency)
            self._check_aperture(element)
            s = self.beam.ref.s
            if self._progress_callback is not None:
                try:
                    self._progress_callback(s, i + 1, n_el)
                except Exception:
                    # Progress hook failures must never break the run.
                    pass
            # When _record_substeps is on, drifts / field maps already
            # recorded their end state; avoid logging a duplicate row.
            substep_already_recorded = (
                self._record_substeps
                and isinstance(element, (TransferMapElement, FieldMapElement))
                and getattr(element, "length", 0.0) > 0
            )
            if not substep_already_recorded:
                self.recorder.record(self.beam, s, element_name=element.name)
            # Row index of this element's exit — with substeps the last
            # row came from the element's own interior recording.
            self.recorder.element_exit_idx.append(len(self.recorder.s) - 1)
            # Snapshot triggers: Marker(snapshot=True), snapshot_locations
            # (by s), snapshot_elements (by name), snapshot_every_n
            should_snap = (
                (hasattr(element, 'snapshot') and element.snapshot)
                or s in self._snapshot_locations
                or element.name in self._snapshot_elements
                or (self._snapshot_every_n and (i + 1) % self._snapshot_every_n == 0)
            )
            if should_snap:
                self.recorder.save_snapshot(self.beam, s)

        return self.recorder

    def _track_element(self, element) -> None:
        # LatticeCommand mutations happen *before* any optics change.
        # The PassiveElement branch below is a separate no-op call to
        # ``element.apply(beam)`` (also harmless), but ``apply_command``
        # is what touches ``track_state``.
        if isinstance(element, LatticeCommand):
            element.apply_command(self.track_state)
            # Fall through to PassiveElement dispatch (which is a no-op)
            # so the existing element-end recording logic still fires.

        # Stamp the element's entrance s so the aperture-profile check
        # inside ``_check_aperture`` can compute a particle's z-offset
        # inside this element without walking the whole lattice.
        element._s_entry = self.beam.ref.s
        # Alignment offsets only apply to elements that support misalignment
        # (quads, solenoids, RF cavities).  Passive elements like Aperture
        # reuse the attribute names ``dx`` / ``dy`` for their own geometry
        # (half-widths / radii) — we must NOT treat those as beam shifts.
        # Multipole handles its OWN dx/dy/tilt inside apply_kick (the
        # manual's contract: kick evaluated at (x−dx, y−dy) in the tilted
        # frame) — wrapping it here DOUBLE-applied both offset and tilt
        # (a 1 mm offset kicked like 2 mm; a 45° skew tracked as 90°).
        # The envelope solver has always excluded it for this reason.
        from linac_gen.elements.multipole import Multipole
        if isinstance(element, (PassiveElement, Multipole)):
            dx = dy = 0.0
            tilt_deg = 0.0
        else:
            dx = getattr(element, 'dx', 0.0)
            dy = getattr(element, 'dy', 0.0)
            tilt_deg = getattr(element, 'tilt_deg', 0.0)
        alive = self.beam.alive_mask
        # Pre-element transform: translate to element-centred frame, then
        # rotate by -tilt_deg around z so the element body operates in its
        # own (rotated) frame.  Rotates both positions (x, y) and angles
        # (x', y').  Tracker convention: particles[:,0]=x, [:,1]=x', etc.
        if dx or dy:
            self.beam.particles[alive, 0] -= dx
            self.beam.particles[alive, 2] -= dy
        if tilt_deg:
            import math as _math
            theta = _math.radians(tilt_deg)
            c, s = _math.cos(theta), _math.sin(theta)
            x = self.beam.particles[alive, 0].copy()
            y = self.beam.particles[alive, 2].copy()
            xp = self.beam.particles[alive, 1].copy()
            yp = self.beam.particles[alive, 3].copy()
            self.beam.particles[alive, 0] = c * x + s * y
            self.beam.particles[alive, 2] = -s * x + c * y
            self.beam.particles[alive, 1] = c * xp + s * yp
            self.beam.particles[alive, 3] = -s * xp + c * yp

        if isinstance(element, TransferMapElement):
            from linac_gen.elements.drift import Drift
            from linac_gen.elements.matrix_element import MatrixElement
            if isinstance(element, Drift):
                self._track_drift(element)
            elif isinstance(element, MatrixElement):
                self._track_matrix(element)
            else:
                self._track_transfer_map(element)
        elif isinstance(element, ThinKickElement):
            element.apply_kick(self.beam)
        elif isinstance(element, FieldMapElement):
            self._track_field_map(element)
        elif isinstance(element, PassiveElement):
            element.apply(self.beam)
            # Update SC compensation factor if this is a SpaceChargeComp element
            if hasattr(element, 'factor'):
                self._sc_factor = 1.0 - element.factor
            elif isinstance(element, ScGridDirective):
                # HELIX_SC_GRID card: retarget the 3-D bunched PIC grid
                # extent from here on (e.g. linac at the default, BTL at
                # 20 sigma in one combined deck).
                if self.pic_solver is not None and hasattr(
                        self.pic_solver, "set_grid_extent"):
                    self.pic_solver.set_grid_extent(element.extent_sigma)
                elif not getattr(self, "_warned_sc_grid_inert", False):
                    self._warned_sc_grid_inert = True
                    import warnings
                    warnings.warn(
                        f"HELIX_SC_GRID {element.extent_sigma:g} at "
                        f"'{element.name}' has NO effect: no 3-D bunched "
                        "PIC solver is active in this run (the card does "
                        "not apply to DC/continuous SC models or "
                        "envelope runs).", stacklevel=2)

        # Post-element transform: inverse rotation, then restore offset.
        if tilt_deg:
            import math as _math
            theta = _math.radians(tilt_deg)
            c, s = _math.cos(theta), _math.sin(theta)
            alive = self.beam.alive_mask
            x = self.beam.particles[alive, 0].copy()
            y = self.beam.particles[alive, 2].copy()
            xp = self.beam.particles[alive, 1].copy()
            yp = self.beam.particles[alive, 3].copy()
            self.beam.particles[alive, 0] = c * x - s * y
            self.beam.particles[alive, 2] = s * x + c * y
            self.beam.particles[alive, 1] = c * xp - s * yp
            self.beam.particles[alive, 3] = s * xp + c * yp
        if dx or dy:
            alive = self.beam.alive_mask
            self.beam.particles[alive, 0] += dx
            self.beam.particles[alive, 2] += dy
        # Reporting hook: every recorded row, snapshot, .dst export and
        # aperture-loss coordinate downstream of here sees folded
        # phases, and SC-off runs are covered (they never reach the
        # kick hook below).
        self._fold_phase()

    # ------------------------------------------------------------------
    # Periodic phase coordinates (opt-in; see BeamConfig.periodic_phase)
    # ------------------------------------------------------------------
    def _fold_phase(self) -> None:
        """Fold Δφ into ONE BUNCH SPACING about the synchronous particle.

        No-op unless the run opted in AND the beam is genuinely a bunch
        train (DC-injected then bunched).  The period is the bunch
        spacing expressed in LOCAL RF degrees,

            P = 360 · f_local / f_bunch,

        so it is 360° at the RFQ's own frequency and 720° after a
        162.5 → 325 MHz jump.  ``f_bunch`` is
        ``Beam.bunch_train_frequency`` — the clock of the element that
        actually bunched the beam, pinned at the transition — and NOT
        ``bunch_frequency``, which comes from the beam config and only
        coincides when the config frequency equals the buncher's.
        Because the FREQ rescale multiplies Δφ and P by the same ratio,
        the fold and the rescale commute exactly; P is therefore
        recomputed from the LIVE ``ref.frequency`` every time and never
        cached.

        Physics: the RF kicks depend on Δφ only through cos/sin, whose
        period is 360° in LOCAL degrees, and the fold shifts by
        360·round(ratio) — so the force a folded particle then feels is
        exactly unchanged, always.  What the integer test protects is
        MEANING: the shift only lands the particle on a real
        neighbouring bunch when the local frequency is a whole multiple
        of the bunch rate.  Harmonic linacs qualify; a sub-harmonic
        buncher does not, and is skipped with a single warning.

        The one case where the force does go wrong is a fold that a
        LATER frequency change rescales by a non-integer factor, at
        which point it is no longer a whole RF period.  That is not
        recoverable, so it raises rather than warns.

        Only particles that actually left the bucket are touched: the
        blanket ``(φ+P/2) % P − P/2`` is not exactly the identity in
        floating point and drifts ~1e-13 per application, which over a
        2872-element line accumulated to 8e-11 deg on a beam that
        should not have moved at all.
        """
        beam = self.beam
        if not self._periodic_phase or not getattr(beam, "bunch_train",
                                                   False):
            return
        f_local = float(beam.ref.frequency)
        # The BUNCH REPETITION RATE — the clock of the element that
        # actually bunched this beam, captured at the transition.  NOT
        # ``bunch_frequency``, which comes from the beam config and is
        # only the same number when the config frequency happens to
        # match the buncher's.
        f_bunch = float(getattr(beam, "bunch_train_frequency", 0.0) or 0.0)
        # np.isfinite, not just > 0: a NaN frequency passes ``<= 0`` and
        # would then blow up inside round() with an opaque
        # "cannot convert float NaN to integer".
        if not (np.isfinite(f_local) and np.isfinite(f_bunch)
                and f_local > 0.0 and f_bunch > 0.0):
            return
        ratio = f_local / f_bunch
        # Tolerance 1e-3, NOT machine epsilon.  The fold period is
        # 360·round(ratio) — a whole number of LOCAL RF degrees — so the
        # RF force it produces is exactly unchanged whatever the
        # residual: cos has period 360° in local degrees, full stop.
        # What the residual costs is placement: the particle lands
        # 360·|ratio − round(ratio)| degrees off the neighbouring
        # bunch's true centre, i.e. ≤ 0.36° here, against a bunch that
        # is several degrees wide.  Machine-epsilon strictness instead
        # made the feature hostage to cavity detuning — an error study
        # perturbing a cavity by a few kHz (Δratio ~ 6e-5) would have
        # tripped it — while a genuine sub-harmonic sits 0.5 away and
        # is still caught with three orders of magnitude to spare.
        if abs(ratio - round(ratio)) > 1e-3 or round(ratio) < 1:
            # Skipping is only safe while NOTHING has been folded yet.
            # Once folds are baked in, every downstream RF element
            # rescales them by f_new/f_old, and a non-integer ratio
            # means those already-applied 360° shifts are no longer
            # whole RF periods — the beam is silently corrupted by up
            # to the full crest-to-trough cavity voltage.  There is no
            # way to undo them, so fail loudly instead of continuing.
            if self._n_phase_folded:
                raise ValueError(
                    f"periodic_phase: the RF clock moved to "
                    f"{f_local:g} MHz, which is not an integer multiple "
                    f"of the {f_bunch:g} MHz bunch repetition rate "
                    f"(ratio {ratio:.6f}), AFTER "
                    f"{self._n_phase_folded} phase folds had already "
                    "been applied.  Those folds are rescaled by every "
                    "downstream RF element and are no longer whole RF "
                    "periods, so the run is not recoverable.  Re-run "
                    "with periodic_phase=False, or with a lattice whose "
                    "frequencies are all harmonics of the buncher.")
            if not self._phase_fold_warned:
                self._phase_fold_warned = True
                warnings.warn(
                    f"periodic_phase: local RF frequency {f_local:g} MHz "
                    f"is not an integer multiple of the bunch repetition "
                    f"rate {f_bunch:g} MHz (ratio {ratio:.4f}) — one "
                    "bunch spacing is not a whole number of local RF "
                    "degrees here, so a fold would move a particle to a "
                    "phase where no bunch exists.  Skipped.",
                    PeriodicPhaseWarning, stacklevel=2)
            return
        period = 360.0 * round(ratio)
        half = 0.5 * period
        phi = beam.particles[:, 4]
        need = beam.alive_mask & (np.abs(phi) > half)
        n = int(np.count_nonzero(need))
        if not n:
            return
        beam.particles[need, 4] = (phi[need] + half) % period - half
        # Running tally of bucket crossings — the honest measure of how
        # much of the run this feature actually changed.  Read it off the
        # tracker after a run; it is 0 whenever the fold never fired.
        self._n_phase_folded += n

    def _check_periodic_phase_compatible(self) -> None:
        """Refuse the combinations where folding is provably wrong.

        CSR is the one longitudinal model in HELIX that is NOT
        360°-periodic: ``csr.py`` converts Δφ to metres, then derives
        the histogram range, bin width and smoothing scale from
        ``z.min()``/``z.max()`` of the whole ensemble.  Folding
        therefore changes the wake felt by particles it never touched
        (measured: 5 % of σ_W on the unfolded core), and it teleports a
        head particle to the tail, flipping the sign of its wake.
        There is no cheap fix — a train's CSR needs the real
        multi-bunch geometry — so the two features are declared
        mutually exclusive rather than silently combined.
        """
        cfg = getattr(self.pic_solver, "config", None)
        if cfg is not None and getattr(cfg, "csr_enabled", False):
            raise ValueError(
                "periodic_phase and csr_enabled cannot be used together: "
                "the CSR wake is computed from the ensemble's absolute "
                "longitudinal extent (z from Δφ, then a histogram over "
                "z.min()..z.max()), so it is not periodic in the bunch "
                "spacing and the fold changes the force on particles it "
                "never moved.  Disable one of the two.")
        # A hand-supplied 6×6 (Elegant EMATRIX) applies column 4
        # LINEARLY: any M[i,4] != 0 for i != 4 turns a fold into a
        # deterministic shift of M[i,4]·P in coordinate i.  Every
        # HELIX-generated matrix has column 4 = [0,0,0,0,1,0], so this
        # only ever fires on an imported one.
        try:
            from linac_gen.elements.matrix_element import MatrixElement
        except Exception:                                   # noqa: BLE001
            return
        bad = []
        for el in getattr(self.lattice, "elements", ()):
            if not isinstance(el, MatrixElement):
                continue
            m = getattr(el, "matrix", None)
            if m is None:
                continue
            m = np.asarray(m)
            if m.shape != (6, 6):
                continue
            col = np.delete(m[:, 4], 4)
            if np.any(np.abs(col) > 0):
                bad.append(getattr(el, "name", "?"))
        if bad:
            raise ValueError(
                f"periodic_phase cannot be used with a MATRIX element "
                f"whose column 4 couples Δφ into another coordinate "
                f"({', '.join(bad[:5])}): that coupling is linear in "
                "Δφ, not 360°-periodic, so folding a particle by one "
                "bunch spacing P shifts its output by M[i,4]·P.  Use "
                "periodic_phase=False for this lattice.")

    def _record_substep(self, element) -> None:
        """Record one substep row, folded first.

        Without the fold a substep-resolved σ_φ(s) of a flagged run
        would still spike wherever a particle is transiently outside
        its bucket mid-element — precisely the artefact the flag exists
        to remove.  The whole-element hook cannot cover this because it
        runs after the element finishes.  A no-op when the flag is off.
        """
        self._fold_phase()
        self.recorder.record(self.beam, self.beam.ref.s,
                             element_name=element.name)

    def _apply_sc_kick(self, ds_mm: float) -> None:
        """Dispatch to the correct SC kick based on beam type.

        - Continuous (DC) beam → 2-D analytic line-charge kick.
        - Bunched beam → 3-D PIC kick via ``self.pic_solver`` if one was
          configured, else no kick.
        """
        # Correctness hook: fold BEFORE the guards below, so the solver
        # can never see an unwrapped train — this is the single choke
        # point through which all five SC call sites pass (including the
        # torch PIC solver, which bypasses pic.coordinates).
        self._fold_phase()
        if ds_mm <= 0 or self._sc_factor <= 0:
            return
        if self._sc_explicitly_off:
            # "off" means OFF for every beam type — the DC branch used
            # to ignore the sentinel and apply the 2-D kick anyway.
            return
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

    def _track_matrix(self, element) -> None:
        """Apply a MatrixElement's explicit map ONCE.

        A general 6x6 cannot be decomposed into the 2-sub-step grid
        ``_track_transfer_map`` uses, so the matrix is applied whole and
        any space-charge kick is taken over the element length afterward
        (a no-op for the common thin, length-0 case).
        """
        ds = element.length
        element.track(self.beam, ds=ds)
        if ds:
            self._apply_sc_kick(ds)
        self._check_aperture(element)

    def _track_transfer_map(self, element) -> None:
        """Split-operator on a 2-sub-step grid (TraceWin "2 steps" convention)."""
        n = 2
        ds = element.length / n
        for _ in range(n):
            element.track(self.beam, ds=ds / 2)
            self._apply_sc_kick(ds)
            self._apply_csr_kick(element, ds)
            element.track(self.beam, ds=ds / 2)
            self._check_aperture(element)
            if self._record_substeps:
                self._record_substep(element)

    def _apply_csr_kick(self, element, ds_mm: float) -> None:
        """Apply a 1-D steady-state CSR energy kick — Dipole elements only.

        No-op unless a CsrKicker was configured (``csr_enabled``) and the
        element being tracked is a :class:`Dipole`.  CSR in straight
        elements (drifts, quads) is zero by construction, so the
        isinstance gate is the physics, not just an optimisation.
        """
        if self._csr_kicker is None:
            return
        from linac_gen.elements.dipole import Dipole
        if isinstance(element, Dipole):
            self._csr_kicker.apply(self.beam, element, ds_mm)

    def _track_field_map(self, element) -> None:
        """Field-map: integration sub-steps per PARTRAN_STEP.step1, SC per step2.

        Uses Strang-split bundles: for each group of ``sc_every`` RK4 sub-steps
        we do RK4(ds·sc_every/2) → SC(ds·sc_every) → RK4(ds·sc_every/2) so the
        SC kick is placed at the midpoint of the bundle (symmetric 2nd order).

        Per-element ``ki``: TraceWin's manual defines Ki as the normalisation
        factor for an external ``FileName.scc`` Z-vs-Scc table.  When ``Ki>0``
        and a ``.scc`` file is present, the reader loads it onto
        ``field_data.scc_profile`` (and ``scc_scale = Ki``); the per-bundle
        loop below interpolates ``Scc(z_local)`` and overrides the running
        ``_sc_factor`` for that kick to ``max(0, 1 − Ki·Scc(z_local))``.
        When no .scc file is present, Ki is a no-op and the global
        ``SPACE_CHARGE_COMP`` marker controls compensation as usual.
        """
        # M7 hook: if MP-engagement is on AND a surrogate is registered
        # for this element, route the per-substep `track_rk4` calls
        # through the surrogate's hybrid linear-anchor + RK4-residual
        # path.  Per-element decisions (length, n_steps, isinstance
        # type checks, .scc/.field_data attributes) still come from
        # the original lattice element -- only the per-substep tracking
        # call is swapped, mirroring the envelope hook's pattern.
        # If the surrogate's ref input is OOD it transparently falls
        # back to the wrapped element inside its own track_rk4.
        from linac_gen.surrogates import registry as _surr_reg
        _surr = (_surr_reg.get_for_element(element)
                 if _surr_reg.is_mp_enabled() else None)
        effective_element = _surr if _surr is not None else element

        # Reset per-run integrator state before this pass.  Beyond the
        # sub-step cursor (without which a re-used cavity samples past the
        # end of its map and silently becomes a drift), the SET_SYNC_PHASE
        # calibration cache (`_sync_offset_deg`) is tuned to the reference
        # β at cavity entry of the *previous* run; if the previous run was
        # the envelope solver and the present run is multi-particle, the
        # MP pass would inherit env's calibration and mis-phase the cavity.
        effective_element.reset_run_state()
        # Canonical cadence — shared with EnvelopeSolver and the
        # backtracker so all modes slice field maps identically (rules
        # and rationale live on field_map_step_counts).
        n_int, n_sc, ds, sc_every, n_bundles = field_map_step_counts(
            self.lattice, element)
        # Per-slice diagnostic recording inside field maps.  When
        # ``record_substeps`` is on the tracker records at every
        # integration sub-step (every ``ds``), so 1-D / 3-D solenoids
        # refined to 5000 steps/m contribute ~2000 data points per
        # element instead of the ~20-per-bundle the old implementation
        # produced.  That matches the user's "every field-map point"
        # resolution and gives smooth σ(s) curves inside solenoids.
        # Per-element ``.scc`` table: when present, overrides the global
        # SPACE_CHARGE_COMP factor inside this field map.  Manual: the
        # local space-charge compensation is ``Ki × scc(z_local)`` and
        # the active SC fraction is ``1 - Ki × scc(z_local)``.
        fd = getattr(element, "field_data", None)
        scc_prof = getattr(fd, "scc_profile", None) if fd is not None else None
        scc_scale = getattr(fd, "scc_scale", 0.0) if fd is not None else 0.0
        scc_mode = getattr(fd, "scc_mode", 0) if fd is not None else 0
        scc_active = (scc_prof is not None and scc_scale != 0.0
                      and scc_mode == 0)

        # SHIFT_IN_FIELD_MAP interior diagnostics (SUPERPOSE containers):
        # record one row per marker at the first sub-step boundary ≥ dz,
        # attributed to the MARKER's name (accuracy = the walk's per-call
        # ds).  Zero overhead when no markers exist.
        _interior = sorted(getattr(element, "interior_markers", []) or [],
                           key=lambda t: t[0])
        if _interior:
            _mk_s_entry = getattr(element, "_s_entry", self.beam.ref.s)
            _mk_state = {"i": 0}

            def _record_marker_crossings():
                z_loc = self.beam.ref.s - _mk_s_entry
                while (_mk_state["i"] < len(_interior)
                       and _interior[_mk_state["i"]][0] <= z_loc + 1e-9):
                    dz_mk, mk = _interior[_mk_state["i"]]
                    self.recorder.record(self.beam, _mk_s_entry + dz_mk,
                                         element_name=mk.name)
                    _mk_state["i"] += 1
        else:
            def _record_marker_crossings():
                return None

        for b in range(n_bundles):
            # First half of the bundle — record after each full ds
            # (= 2 half-steps) so samples are evenly spaced.  The inner
            # loop does ds/2 per track_rk4 call; pair them up.
            for k in range(sc_every):
                effective_element.track_rk4(self.beam, ds / 2)
                _record_marker_crossings()
                # Second half of the pair happens in the "second half of
                # bundle" loop below, after the SC kick.
            if scc_active:
                z_local_mm = (b + 0.5) * sc_every * ds
                scc_z = float(np.interp(
                    z_local_mm, scc_prof[:, 0], scc_prof[:, 1]))
                local_factor = max(0.0, 1.0 - scc_scale * scc_z)
                saved = self._sc_factor
                self._sc_factor = local_factor
                self._apply_sc_kick(sc_every * ds)
                self._sc_factor = saved
            else:
                self._apply_sc_kick(sc_every * ds)
            # Second half of the bundle — record after each full ds slice.
            for k in range(sc_every):
                effective_element.track_rk4(self.beam, ds / 2)
                _record_marker_crossings()
                if self._record_substeps:
                    # k-th pair is now complete (one ds/2 before SC, one
                    # ds/2 after).  Record at the slice's exit position.
                    self._record_substep(element)
            self._check_aperture(element)
        # Any trailing sub-steps (n_int not a multiple of sc_every) get RK4 only
        for _ in range(n_int - n_bundles * sc_every):
            effective_element.track_rk4(self.beam, ds)
            _record_marker_crossings()
            if self._record_substeps:
                self._record_substep(element)
        if n_int > n_bundles * sc_every:
            self._check_aperture(element)

    def _track_drift(self, element) -> None:
        """Drift integration on the global step1/step2 grid.

        Drifts have an exact analytic map, so sub-stepping only matters for
        SC kick placement.  We use a Strang bundle: drift(bundle/2) →
        SC(bundle) → drift(bundle/2) per group of ``sc_every`` sub-steps.
        """
        cfg = getattr(self.lattice, "step_config", None)
        n_int = (cfg.integration_steps_for_length_mm(element.length)
                 if cfg is not None else 2)
        n_sc = (cfg.sc_steps_for_length_mm(element.length)
                if cfg is not None else n_int)
        ds = element.length / n_int
        sc_every = max(1, n_int // n_sc)
        n_bundles = n_int // sc_every
        bundle_len = sc_every * ds
        for b in range(n_bundles):
            element.track(self.beam, ds=bundle_len / 2)
            self._apply_sc_kick(bundle_len)
            element.track(self.beam, ds=bundle_len / 2)
            self._check_aperture(element)
            if self._record_substeps:
                self._record_substep(element)
        remainder_len = element.length - n_bundles * bundle_len
        if remainder_len > 0:
            element.track(self.beam, ds=remainder_len)
            self._check_aperture(element)
            if self._record_substeps:
                self._record_substep(element)

    def _check_aperture(self, element) -> None:
        """Mark alive particles outside the element's aperture as lost.

        Loss s-position is ``beam.ref.s`` at call time (the end of the
        just-completed sub-step, so sub-step granularity applies); the
        recorded (x, y) is the particle's position at that s.

        Shape selection follows the TraceWin DRIFT convention:
          * FIELD_MAP with a z-varying profile loaded from ``.ouv``
            (Ka=1): per-slice half-widths ``rx(s_local)``, ``ry(s_local)``
            linearly interpolated from the .ouv table.  Elliptical-shape
            check ``(x/rx)² + (y/ry)² > 1`` (the TraceWin "FIELD_MAP
            flag = 1" behaviour, manual §"Aperture map").
          * ``element.aperture_y`` missing or <= 0 → **circular** pipe
            of radius ``element.aperture``.
          * ``element.aperture_y`` > 0 → **rectangular** pipe with
            half-widths (``element.aperture``, ``element.aperture_y``).
        Richer shapes (elliptical, pepperpot, …) live on the dedicated
        :class:`linac_gen.elements.aperture.Aperture` element.
        """
        # ------------------------------------------------------------------
        # Per-z pipe-radius profile from an .ouv aperture file (Ka=1 on
        # the FIELD_MAP card).  Takes precedence over the flat
        # ``element.aperture`` scalar when present.
        # ------------------------------------------------------------------
        fd = getattr(element, "field_data", None)
        profile = getattr(fd, "pipe_radius_profile", None) if fd is not None else None
        if profile is not None:
            # s-local is the particle's z offset from the element entrance.
            # beam.ref.s is the global s (mm) at the end of the current
            # sub-step.  ``element._s_entry`` is set by _track_element so
            # we can compute the local z without knowing the lattice layout.
            alive_idx = np.where(self.beam.alive_mask)[0]
            if len(alive_idx) == 0:
                return
            s_entry = getattr(element, "_s_entry", self.beam.ref.s - element.length)
            s_local_mm = max(0.0, min(element.length,
                                      self.beam.ref.s - s_entry))
            z_tab_mm, rx_tab_mm, ry_tab_mm = profile
            rx_mm = float(np.interp(s_local_mm, z_tab_mm, rx_tab_mm))
            ry_mm = float(np.interp(s_local_mm, z_tab_mm, ry_tab_mm))
            x = self.beam.particles[alive_idx, 0]
            y = self.beam.particles[alive_idx, 2]
            # Elliptical pipe (TraceWin FIELD_MAP flag=1 default).
            if rx_mm > 0 and ry_mm > 0:
                lost = (x / rx_mm) ** 2 + (y / ry_mm) ** 2 > 1.0
            else:
                lost = np.zeros_like(x, dtype=bool)
            s = self.beam.ref.s
            for idx in alive_idx[lost]:
                self.beam.record_loss(int(idx), s, element.name)
            return

        if element.aperture <= 0:
            return
        alive_idx = np.where(self.beam.alive_mask)[0]
        if len(alive_idx) == 0:
            return
        x = self.beam.particles[alive_idx, 0]
        y = self.beam.particles[alive_idx, 2]
        ap_y = getattr(element, "aperture_y", None)
        if ap_y is not None and ap_y > 0.0:
            lost = (np.abs(x) > element.aperture) | (np.abs(y) > ap_y)
        else:
            lost = x ** 2 + y ** 2 > element.aperture ** 2
        s = self.beam.ref.s
        for idx in alive_idx[lost]:
            self.beam.record_loss(int(idx), s, element.name)
