"""Element failure analysis: impact/criticality sweeps + fault recovery.

Put any active element (cavity / quad / solenoid / dipole) into a failure state
(OFF / cavity DETUNE / magnet PARTIAL), sweep over single elements, all pairs,
or user-defined sets, rank scenarios by beam impact, and — MYRRHA / LightWin
style — re-tune neighbouring elements to recover the beam.

Failure injection rides the elements' additive error slots via
``ScanPoint.element_overrides``; the sweep reuses ``parallel.scan_pool``;
recovery reuses ``matching.match``.
"""
from linac_gen.failures.compensation import (
    CompensationConfig, CompensationResult, compensate, select_zone)
from linac_gen.failures.criticality import criticality_score
from linac_gen.failures.element_filter import (
    ALL_TYPES, classify, failable_elements, valid_kinds)
from linac_gen.failures.failure_mode import (
    FailureKind, FailureMode, can_fail, is_cavity)
from linac_gen.failures.scenario import FailureScenario, enumerate_scenarios
from linac_gen.failures.study import (
    FailureStudy, FailureStudyResults, ScenarioImpact)

__all__ = [
    "FailureKind", "FailureMode", "can_fail", "is_cavity",
    "classify", "failable_elements", "valid_kinds", "ALL_TYPES",
    "FailureScenario", "enumerate_scenarios",
    "criticality_score",
    "FailureStudy", "FailureStudyResults", "ScenarioImpact",
    "CompensationConfig", "CompensationResult", "compensate", "select_zone",
]
