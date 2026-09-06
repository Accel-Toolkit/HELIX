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
"""
from dataclasses import dataclass
from typing import ClassVar
import math


@dataclass(frozen=True)
class StepConfig:
    """Steps-per-metre for integration and space-charge kicks."""
    integration_steps_per_metre: float = 100.0  # step1
    sc_steps_per_metre: float = 50.0            # step2
    # One push per field-free drift when nothing sits between the sub-steps
    # (see the module docstring).  Kept LAST so positional StepConfig(a, b)
    # construction keeps working.
    drift_single_push: bool = True

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
