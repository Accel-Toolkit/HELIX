"""Shared plain-Python test helpers (no Qt, no fixtures).

Used by the run-current provenance tests (tests/tracking/
test_run_current_stamp.py) and the orbit-correction dead-beam tests —
both probe getattr-based readers that must distinguish "field absent"
from "field is None" from "field is 0.0".
"""
from types import SimpleNamespace


def stub_results(**fields):
    """A SimpleNamespace standing in for a results object
    (DiagnosticRecorder / EnvelopeResults / _LoadedResults).

    Only the passed fields exist, so ``getattr(obj, name, default)``
    readers see exactly the attribute surface the test declares.
    """
    return SimpleNamespace(**fields)
