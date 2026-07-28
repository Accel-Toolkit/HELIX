"""Mutable per-run state object propagated through tracking loops.

Linac_Gen tracking loops (``Simulation.run`` / ``Simulation.run_envelope``
/ ``EnvelopeSolver.run``) walk the lattice element by element.  A handful of
TraceWin lattice commands (``SET_SYNC_PHASE``, ``SET_BEAM_PHASE_ERROR``,
``SET_BEAM_E0_P0``, ``SET_BEAM_ENERGY``, …) need to mutate the beam state
*at their position in the file*, not as a global pre-run option.  Those
commands are represented as :class:`linac_gen.elements.lattice_commands.LatticeCommand`
elements; their ``apply_command`` hook receives this object.

Note: this is **independent** of the parser-level ``pending_sync_phase``
one-shot flag in ``tracewin_parser.py``.  The parser flag bakes
``p_flag=1`` into the next FIELD_MAP at construction time.  ``TrackState``
is a *runtime* flag consulted at every RF cavity / field-map kick.  The
two coexist; phasing them out is a follow-up.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackState:
    """Live tracking state mutated by ``LatticeCommand`` instances.

    Attributes
    ----------
    ref :
        The :class:`linac_gen.core.reference.ReferenceParticle` for the run.
        ``SetBeamEnergy`` / ``SetBeamE0P0`` mutate ``ref.w_kin`` directly.
    beam :
        The live multi-particle :class:`~linac_gen.core.beam.Beam`, bound by
        the MP tracker only (``None`` in envelope/matrix modes).  ``Freq``
        rescales the particles' Δφ coordinate through it at an RF-frequency
        change.
    phase_ref_shift :
        Accumulated phase offset (degrees) applied to every following RF
        cavity reference phase.  ``SET_BEAM_PHASE_ERROR`` adds; ``Dp = 0``
        clears.
    sync_phase_mode :
        When ``True``, RF cavity ``phase`` arguments are interpreted as
        synchronous phase φ_s rather than φ_RF.  Sticky until the end of
        the lattice.
    error_cutoff_sigma :
        Optional Gaussian-error cutoff (× σ) declared by
        ``SET_GAUSSIAN_CUT_OFF``.  Linac_Gen reads this when running
        Monte-Carlo error studies; the deterministic tracker ignores it.
    user :
        Free-form scratch dict for downstream consumers.
    """

    ref: object = None
    beam: object = None
    phase_ref_shift: float = 0.0
    sync_phase_mode: bool = False
    error_cutoff_sigma: Optional[float] = None
    user: dict = field(default_factory=dict)
