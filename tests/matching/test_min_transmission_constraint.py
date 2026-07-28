"""Tests for the MIN_TRANSMISSION constraint.

Covers:
* Parser/writer round-trip (the card serialises cleanly).
* Constraint evaluator: residual=0 above threshold, residual>0 below.
* collect_constraints picks up the card.
* Envelope mode is inert (no transmission tracked) and warns once.
* Weight=0 drops the constraint.
* threshold_pct out of [0, 100] is rejected at construction.
"""
from __future__ import annotations

import io
from contextlib import redirect_stderr

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.lattice_commands import MinTransmission
from linac_gen.matching.constraints import (
    collect_constraints, _make_min_transmission_evaluator,
)


class _FakeResults:
    """Minimal stand-in for DiagnosticRecorder / EnvelopeResults --
    just enough fields for the evaluator to read."""

    def __init__(self, transmission):
        self.transmission = list(transmission) if transmission is not None else None


# ----------------------------------------------------------------------
# Evaluator unit tests -- residual semantics
# ----------------------------------------------------------------------
def test_min_transmission_residual_zero_when_above_threshold():
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)
    ev = _make_min_transmission_evaluator(cmd)
    res = ev(_FakeResults([99.5]), lattice=None)
    assert res.shape == (1,)
    assert res[0] == 0.0


def test_min_transmission_residual_zero_when_exactly_at_threshold():
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)
    ev = _make_min_transmission_evaluator(cmd)
    res = ev(_FakeResults([99.0]), lattice=None)
    assert res[0] == 0.0


def test_min_transmission_residual_positive_when_below_threshold():
    """1% below threshold with weight=1 -> residual 0.01 (1/100 scaling)."""
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)
    ev = _make_min_transmission_evaluator(cmd)
    res = ev(_FakeResults([98.0]), lattice=None)
    assert res[0] == pytest.approx(0.01, abs=1e-12)


def test_min_transmission_evaluator_returns_raw_unweighted_residual():
    """The evaluator itself is weight-free: Constraint.evaluate() applies
    the weight exactly once.  (Previously the evaluator also multiplied
    by weight, making the effective end-to-end weight k² -- fixed
    2026-07-10.)"""
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=50.0)
    ev = _make_min_transmission_evaluator(cmd)
    res = ev(_FakeResults([98.0]), lattice=None)
    assert res[0] == pytest.approx(0.01, abs=1e-12)


def test_min_transmission_weight_scales_residual_linearly():
    """Through Constraint.evaluate() -- the path the matcher actually
    uses -- weight=50 makes the 1% loss residual 0.5 (50x the unit
    weight, linear in k, not k^2)."""
    from linac_gen.matching.constraints import Constraint
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=50.0)
    con = Constraint(label="MT",
                     evaluator=_make_min_transmission_evaluator(cmd),
                     weight=cmd.weight, source=None)
    res = con.evaluate(_FakeResults([98.0]), lattice=None)
    assert res[0] == pytest.approx(0.5, abs=1e-12)


def test_min_transmission_uses_final_value_only():
    """The constraint cares about end-of-line transmission, not
    intermediate dips (some lattice models record per-element losses
    that recover at exit; the residual must read the LAST value)."""
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)
    ev = _make_min_transmission_evaluator(cmd)
    res = ev(_FakeResults([100.0, 50.0, 99.9]), lattice=None)
    assert res[0] == 0.0      # end-of-line is 99.9, above threshold


# ----------------------------------------------------------------------
# Envelope inertia + one-time warning
# ----------------------------------------------------------------------
def test_min_transmission_inert_when_results_lack_transmission():
    """Envelope results don't carry transmission unless the parameter-
    scan synth populates it.  Evaluator must return [0.0] silently."""
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)
    ev = _make_min_transmission_evaluator(cmd)
    res = ev(_FakeResults(None), lattice=None)
    assert res.shape == (1,)
    assert res[0] == 0.0


def test_min_transmission_warns_once_in_env_mode():
    """Stderr warning fires the first time env results are seen, so
    the user knows the constraint isn't doing anything."""
    # Reset the module-level "already-warned" sentinel for this test
    import linac_gen.matching.constraints as _cm
    _cm._MIN_TRANSMISSION_ENV_WARNED = False
    cmd = MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)
    ev = _make_min_transmission_evaluator(cmd)
    buf = io.StringIO()
    with redirect_stderr(buf):
        ev(_FakeResults(None), lattice=None)
        ev(_FakeResults(None), lattice=None)
        ev(_FakeResults(None), lattice=None)
    text = buf.getvalue()
    # Warning fires exactly once, even though the evaluator was
    # called three times.
    assert text.count("MIN_TRANSMISSION") == 1
    assert "INERT" in text


# ----------------------------------------------------------------------
# collect_constraints integration
# ----------------------------------------------------------------------
def test_collect_constraints_picks_up_min_transmission():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Quadrupole("Q1", length=100.0, gradient=5.0, aperture=10.0))
    lat.add(MinTransmission("MT", threshold_pct=99.5, weight=10.0))
    constraints = collect_constraints(lat)
    labels = [c.label for c in constraints]
    assert any("MIN_TRANSMISSION:99.5" in lbl for lbl in labels)


def test_collect_constraints_skips_min_transmission_when_weight_zero():
    """Weight=0 should drop the constraint entirely (consistent with
    every other ε / KE constraint in the codebase)."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(MinTransmission("MT", threshold_pct=99.0, weight=0.0))
    constraints = collect_constraints(lat)
    assert not any("MIN_TRANSMISSION" in c.label for c in constraints)


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad_thresh", [-1.0, -0.1, 100.1, 200.0])
def test_min_transmission_rejects_threshold_outside_0_100(bad_thresh):
    with pytest.raises(ValueError, match="threshold_pct"):
        MinTransmission(name="MT", threshold_pct=bad_thresh, weight=1.0)


@pytest.mark.parametrize("good_thresh", [0.0, 50.0, 99.9, 100.0])
def test_min_transmission_accepts_threshold_in_0_100(good_thresh):
    cmd = MinTransmission(name="MT", threshold_pct=good_thresh, weight=1.0)
    assert cmd.threshold_pct == good_thresh


# ----------------------------------------------------------------------
# .dat round-trip
# ----------------------------------------------------------------------
def test_min_transmission_parser_writer_round_trip(tmp_path):
    """Parse a .dat carrying MIN_TRANSMISSION, write it back, re-parse,
    confirm the parameters survived."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    src = tmp_path / "src.dat"
    src.write_text(
        "DRIFT 200 10 0 0 0\n"
        "QUAD 100 5.0 10 0 0 0 0 0 0\n"
        "DRIFT 200 10 0 0 0\n"
        "MIN_TRANSMISSION 98.5 25.0\n"
        "END\n"
    )
    lat, meta = parse_tracewin(str(src))
    assert meta.get("warnings") in (None, [])

    mt = [e for e in lat.elements if isinstance(e, MinTransmission)]
    assert len(mt) == 1
    assert mt[0].threshold_pct == pytest.approx(98.5)
    assert mt[0].weight == pytest.approx(25.0)

    out = tmp_path / "out.dat"
    write_tracewin(lat, str(out))
    lat2, _ = parse_tracewin(str(out))
    mt2 = [e for e in lat2.elements if isinstance(e, MinTransmission)]
    assert len(mt2) == 1
    assert mt2[0].threshold_pct == pytest.approx(98.5)
    assert mt2[0].weight == pytest.approx(25.0)
