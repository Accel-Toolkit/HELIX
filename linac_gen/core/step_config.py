"""Global integration / space-charge step density, matching TraceWin's PARTRAN_STEP.

TraceWin specifies two numbers: ``step1`` and ``step2``, both in units of
*steps per metre*.  ``step1`` drives the sub-step count used to integrate
DRIFT and FIELD_MAP elements; ``step2`` sets how often a space-charge kick
is applied inside those elements.  All other elements (QUAD, BEND,
SOLENOID, GAP, ...) are tracked in exactly 2 integration sub-steps with
one space-charge kick at the mid-plane, regardless of this config.

``drift_single_push`` (default on): a DRIFT has an exact transfer map, so
its ``step1`` sub-steps only matter for what happens *between* them —
space-charge kicks, sub-step diagnostics, the bunch-train phase fold.
When none of those is active the multiparticle tracker (and the
backtracker) apply the map once; a particle that leaves the pipe inside
the drift is located analytically on its straight line (exact ``s`` and
wall coordinates) instead of at the end of the sub-step that first saw it
outside.  Set it to ``False`` to keep the sub-stepped walk in every case
(identical to HELIX ≤ 1.9.1).

``rfq_solver`` (default ``"auto"``): how multiparticle runs track an RFQ.
``"auto"`` tracks the whole RFQ in time through the 3-D field of its vanes
(:mod:`linac_gen.rfq_td`) whenever that solver can run the lattice and
beam, and otherwise one ``RfqCell`` at a time along z, with a notice that
says why (it also keeps the per-cell model when a run sets that model up
explicitly: ``Simulation(rfq_geometry=...)`` other than ``"auto"``, or a
vane profile armed with ``apply_rfq_geometry``); ``"time_domain"`` always
uses the time-domain solver (a case it cannot run is an error); ``"cells"``
always uses the per-cell model.
Envelope runs and backtracking use the per-cell model in every case.

``rfq_stats_accelerated`` (default on): with the time-domain solver, a
particle that leaves the RFQ unaccelerated (it gains less than half the
reference particle's energy gain) stays in the beam -- transmission and the
exported distribution are unchanged -- but the recorded statistics (rms
sizes, emittances, energy spread, Twiss, density) leave it out from the cell
where it falls behind the bunch, so they describe the accelerated beam.  Off:
every surviving particle counts, as in HELIX <= 1.19.
"""
from dataclasses import dataclass
from typing import ClassVar
import math

#: the values of ``StepConfig.rfq_solver``, the default first
RFQ_SOLVERS = ("auto", "cells", "time_domain")


@dataclass(frozen=True)
class StepConfig:
    """Steps-per-metre for integration and space-charge kicks."""
    integration_steps_per_metre: float = 100.0  # step1
    sc_steps_per_metre: float = 50.0            # step2
    # One push per field-free drift when nothing sits between the sub-steps
    # (see the module docstring).  Kept LAST so positional StepConfig(a, b)
    # construction keeps working.
    drift_single_push: bool = True
    # RFQ solver for multiparticle runs (module docstring): "auto" (the
    # time-domain solver whenever it can run, else the per-cell model),
    # "time_domain" or "cells".  Appended after drift_single_push so
    # positional construction keeps working.
    rfq_solver: str = "auto"
    rfq_td_steps_per_period: int = 48
    rfq_td_sc_every: int = 4
    # time-domain RFQ: statistics of the accelerated beam (module docstring)
    rfq_stats_accelerated: bool = True

    # Lower bounds so that very short drifts still get at least one
    # half-kick split and one SC call.  ClassVar so they remain true
    # class constants instead of becoming constructor arguments.
    MIN_INTEGRATION_STEPS: ClassVar[int] = 2
    MIN_SC_STEPS: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.integration_steps_per_metre <= 0.0:
            raise ValueError(
                "integration_steps_per_metre must be > 0, "
                f"got {self.integration_steps_per_metre}"
            )
        if self.sc_steps_per_metre <= 0.0:
            raise ValueError(
                f"sc_steps_per_metre must be > 0, got {self.sc_steps_per_metre}"
            )
        # numpy.bool_ / JSON ints arrive from project files and provenance
        # readers; store a plain bool (frozen dataclass -> object.__setattr__)
        object.__setattr__(self, "drift_single_push", bool(self.drift_single_push))
        solver = str(self.rfq_solver).strip().lower().replace("-", "_")
        if solver not in RFQ_SOLVERS:
            raise ValueError(
                "rfq_solver must be 'auto', 'cells' or 'time_domain', got "
                f"{self.rfq_solver!r}")
        object.__setattr__(self, "rfq_solver", solver)
        steps = int(self.rfq_td_steps_per_period)
        if steps < 8:
            raise ValueError("rfq_td_steps_per_period must be >= 8, got "
                             f"{self.rfq_td_steps_per_period!r}")
        object.__setattr__(self, "rfq_td_steps_per_period", steps)
        every = int(self.rfq_td_sc_every)
        if every < 1:
            raise ValueError("rfq_td_sc_every must be >= 1, got "
                             f"{self.rfq_td_sc_every!r}")
        object.__setattr__(self, "rfq_td_sc_every", every)
        object.__setattr__(self, "rfq_stats_accelerated",
                           bool(self.rfq_stats_accelerated))

    def integration_steps_for_length_mm(self, length_mm: float) -> int:
        """Number of integration sub-steps for a drift / field map of this length.

        Returns at least :attr:`MIN_INTEGRATION_STEPS` even for zero or
        negative ``length_mm``.
        """
        n = int(math.ceil(length_mm * 1e-3 * self.integration_steps_per_metre))
        return max(n, self.MIN_INTEGRATION_STEPS)

    def sc_steps_for_length_mm(self, length_mm: float) -> int:
        """Number of space-charge kicks for a drift / field map of this length.

        Returns at least :attr:`MIN_SC_STEPS` even for zero or negative
        ``length_mm``.
        """
        n = int(math.ceil(length_mm * 1e-3 * self.sc_steps_per_metre))
        return max(n, self.MIN_SC_STEPS)


def rfq_fields(cfg) -> dict:
    """The RFQ solver fields of a StepConfig (defaults when ``cfg`` is None
    or lacks them) -- for code that builds a fresh StepConfig for a run on
    a lattice whose solver choice it must keep:
    ``replace(fresh, **rfq_fields(lattice.step_config))``."""
    d = StepConfig()
    return dict(
        rfq_solver=getattr(cfg, "rfq_solver", d.rfq_solver),
        rfq_td_steps_per_period=getattr(cfg, "rfq_td_steps_per_period",
                                        d.rfq_td_steps_per_period),
        rfq_td_sc_every=getattr(cfg, "rfq_td_sc_every", d.rfq_td_sc_every))
