"""Phase resolution helpers shared by RF cavity / field-map tracking.

Routes through :class:`linac_gen.core.track_state.TrackState` so that
``SetSyncPhase`` and ``SetBeamPhaseError`` lattice commands take effect at
their position in the lattice.

Most call sites currently read ``elem.phase`` directly.  Migrating them is
incremental — switching to ``effective_phase`` is a no-op when no
``LatticeCommand`` is in play (default ``TrackState`` leaves phases
unchanged).
"""
from __future__ import annotations


def effective_phase(elem, track_state) -> float:
    """Return the phase (degrees) that ``elem`` should use given the
    current run-time ``track_state``.

    Rules
    -----
    * ``track_state.phase_ref_shift`` is added to ``elem.phase``.
    * ``track_state.sync_phase_mode`` is **advisory**: callers that
      distinguish RF phase from synchronous phase honour it; for now the
      RF gap and field-map kick sites simply add the shift.  The deeper
      φ_RF ↔ φ_s conversion is deferred to a follow-up that touches the
      kick math directly.

    A ``None`` track_state behaves as the default (no shift), so this
    helper is safe to call from code paths that don't yet build a
    ``TrackState``.
    """
    base = float(getattr(elem, "phase", 0.0))
    if track_state is None:
        return base
    return base + float(getattr(track_state, "phase_ref_shift", 0.0))
