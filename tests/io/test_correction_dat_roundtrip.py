"""Round-trip parser/writer test for the correction-demo .dat.

Verifies:

1. ``correction_demo.dat`` parses to four ``AdjustSteerer`` instances
   with the right ``diag_n`` / ``vmax`` / ``first_step``.
2. The ``DIAG_POSITION`` cards land as ``Marker(is_bpm=True)``.
3. Writing the parsed lattice back out and re-parsing produces a
   structurally-equivalent lattice (same element types in order, same
   ADJUST_STEERER fields).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from linac_gen.elements.lattice_commands import (
    AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
)
from linac_gen.elements.marker import Marker
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin

DEMO_DAT = Path(__file__).resolve().parents[2] / "examples" / "correction_demo" / "correction_demo.dat"


def _adjusters(lattice):
    return [e for e in lattice.elements
            if isinstance(e, (AdjustSteerer, AdjustSteererBx, AdjustSteererBy))]


def _bpms(lattice):
    return [e for e in lattice.elements
            if isinstance(e, Marker) and getattr(e, "is_bpm", False)]


def test_parses_four_adjust_steerer_cards():
    lat, _meta = parse_tracewin(str(DEMO_DAT))
    cards = _adjusters(lat)
    assert len(cards) == 4
    for n, card in enumerate(cards, start=1):
        assert isinstance(card, AdjustSteerer)
        assert card.diag_n == n
        assert card.vmax == pytest.approx(0.02, abs=1e-6)
        assert card.first_step == pytest.approx(1e-4, abs=1e-9)


def test_parses_four_bpms():
    lat, _meta = parse_tracewin(str(DEMO_DAT))
    bpms = _bpms(lat)
    assert len(bpms) == 4
    # Ordered top-to-bottom in the lattice.
    names = [b.name for b in bpms]
    assert names == sorted(names)


def test_round_trip_preserves_structure():
    lat1, _ = parse_tracewin(str(DEMO_DAT))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, encoding="utf-8",
    ) as tf:
        path = tf.name
    try:
        write_tracewin(lat1, path)
        lat2, _ = parse_tracewin(path)
    finally:
        os.unlink(path)

    cards1 = _adjusters(lat1)
    cards2 = _adjusters(lat2)
    assert len(cards1) == len(cards2) == 4
    for c1, c2 in zip(cards1, cards2):
        assert type(c1) is type(c2)
        assert c1.diag_n == c2.diag_n
        assert c1.vmax == pytest.approx(c2.vmax, abs=1e-9)
        assert c1.first_step == pytest.approx(c2.first_step, abs=1e-9)

    # BPM count preserved.
    assert len(_bpms(lat1)) == len(_bpms(lat2)) == 4

    # Steerer count preserved.
    from linac_gen.elements.steerer import Steerer
    s1 = [e for e in lat1.elements if isinstance(e, Steerer)]
    s2 = [e for e in lat2.elements if isinstance(e, Steerer)]
    assert len(s1) == len(s2) == 4


def test_demo_bpms_sit_after_defocusing_quad():
    """Deck-conditioning pin (2026-09 dead-beam fix): a BPM 50 mm of
    drift downstream of its steerer is an ill-conditioned lever arm
    (nulling 1.8 mm took -35 mrad and killed the beam).  Each demo BPM
    must sit directly after its cell's defocusing quad, downstream of
    the partner steerer."""
    from linac_gen.elements.lattice_commands import LatticeCommand
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.steerer import Steerer

    lat, _meta = parse_tracewin(str(DEMO_DAT))
    els = list(lat.elements)
    bpms = _bpms(lat)
    assert len(bpms) == 4
    for bpm in bpms:
        i = els.index(bpm)
        prev = next(p for p in reversed(els[:i])
                    if not isinstance(p, LatticeCommand))
        assert isinstance(prev, Quadrupole), (
            f"{bpm.name}: element before it is {type(prev).__name__}")
        assert prev.gradient < 0, (
            f"{bpm.name}: preceding quad {prev.name} is not defocusing")
    for idx, e in enumerate(els):
        if isinstance(e, AdjustSteerer):
            steer = next(c for c in els[idx + 1:] if isinstance(c, Steerer))
            bpm = bpms[e.diag_n - 1]
            assert els.index(bpm) > els.index(steer), (
                f"{e.name}: BPM {bpm.name} not downstream of {steer.name}")
