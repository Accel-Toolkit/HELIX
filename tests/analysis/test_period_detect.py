"""Tests for the auto period detector."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.analysis.period_detect import detect_periods


def _fodo() -> Lattice:
    """Build a FODO with 3 cells of (D, QF, D, QD) — no LATTICE card."""
    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0, aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0, aperture=10.0))
    return lat


def test_detect_fodo_no_card():
    lat = _fodo()
    periods = detect_periods(lat)
    # First entry must be the type-sequence cell — 4 elements × 3 reps.
    assert periods[0].source == "type_sequence"
    assert periods[0].inner_period_length == 4
    assert periods[0].n_repeats == 3


def test_detect_aperiodic_falls_back():
    """Random-ish lattice with no clear repeat returns only the fallback."""
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0, aperture=10.0))
    lat.add(Drift(name="D2", length=200.0, aperture=10.0))
    lat.add(RFGap(name="G1", voltage=0.5, phase=-30.0, frequency=162.5))
    lat.add(Drift(name="D3", length=300.0, aperture=10.0))
    periods = detect_periods(lat)
    # Only "(whole lattice)" — no repeat possible.
    sources = {p.source for p in periods}
    assert "fallback" in sources
    assert "type_sequence" not in sources


def test_detect_lattice_card_bracket(tmp_path):
    """Explicit LATTICE 4 0 ... LATTICE_END recovers n_cells = 4."""
    text = textwrap.dedent("""\
        FREQ 162.5
        LATTICE 4 0
        DRIFT 100 10
        QUAD 50 10 10
        DRIFT 100 10
        QUAD 50 -10 10
        DRIFT 100 10
        QUAD 50 10 10
        DRIFT 100 10
        QUAD 50 -10 10
        DRIFT 100 10
        QUAD 50 10 10
        DRIFT 100 10
        QUAD 50 -10 10
        DRIFT 100 10
        QUAD 50 10 10
        DRIFT 100 10
        QUAD 50 -10 10
        LATTICE_END
        END
    """)
    fp = tmp_path / "card.dat"
    fp.write_text(text)
    lat, _ = parse_tracewin(str(fp))
    periods = detect_periods(lat)
    cards = [p for p in periods if p.source == "lattice_card"]
    assert cards, "LATTICE-card detector should fire"
    assert cards[0].n_repeats == 4
    # Each repeat is 4 significant elements (D Q D Q).
    assert cards[0].inner_period_length == 4


def test_qf_qd_not_collapsed_to_n_eq_one():
    """The (class, length, key_param) hash must distinguish QF from QD;
    a naive class-only hash would call this an n=1 periodic lattice."""
    lat = _fodo()
    periods = detect_periods(lat)
    p0 = periods[0]
    # n=1 would mean every element is "the same" — the gradient sign must
    # break that.  Demand at least 2 elements per cell.
    assert p0.inner_period_length >= 2


def test_periodic_structure_slice_mirrors_first_cell():
    lat = _fodo()
    p = detect_periods(lat)[0]
    cell = lat.elements[p.start:p.inner_slice_end]
    assert len(cell) == 4
    assert [type(e).__name__ for e in cell] == [
        "Drift", "Quadrupole", "Drift", "Quadrupole"
    ]
