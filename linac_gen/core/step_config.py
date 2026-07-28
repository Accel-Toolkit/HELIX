"""Global integration / space-charge step density, matching TraceWin's PARTRAN_STEP.

TraceWin specifies two numbers: ``step1`` and ``step2``, both in units of
*steps per metre*.  ``step1`` drives the sub-step count used to integrate
DRIFT and FIELD_MAP elements; ``step2`` sets how often a space-charge kick
is applied inside those elements.  All other elements (QUAD, BEND,
SOLENOID, GAP, ...) are tracked in exactly 2 integration sub-steps with
one space-charge kick at the mid-plane, regardless of this config.
"""
from dataclasses import dataclass
from typing import ClassVar
import math


@dataclass(frozen=True)
class StepConfig:
    """Steps-per-metre for integration and space-charge kicks."""
    integration_steps_per_metre: float = 100.0  # step1
    sc_steps_per_metre: float = 50.0            # step2

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
