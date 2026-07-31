"""Semantic element classification, anchored to the REAL MEBT-to-Foil
deck — the numbers the operator counts in the raw file.

The decisive case: 12 HB650 β=0.92 cavities are parked at ke=kb=0.
The historical ``ke != 0`` rule (categorize_fieldmap) files them as
solenoids (123); the channel-based classifier must report 135.
"""
from __future__ import annotations

import os

import pytest

from linac_gen.lattice_semantics import classify_element, summarize_lattice

_DECK = os.path.join(os.path.dirname(__file__), "..", "..",
                     "examples", "MEBT_To_Foil", "mebt_to_foil.dat")


@pytest.fixture(scope="module")
def deck():
    if not os.path.isfile(_DECK):
        pytest.skip("MEBT_To_Foil example deck not present")
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _meta = parse_tracewin(_DECK)
    return lat


def test_real_deck_operator_counts(deck):
    s = summarize_lattice(deck)
    assert s["n_elements"] == 2872
    assert abs(s["length_m"] - 544.438) < 0.01
    assert s["cavities"] == 135                  # NOT 123: 12 parked
    assert s["cavities_powered"] == 123          # cavities are still
    assert s["cavities_parked_zero_amplitude"] == 12   # cavities
    assert s["solenoids"] == 37                  # 29 geom-70 + 8 geom-10
    assert s["quads"] == 124
    assert s["dipoles"] == 36
    assert s["correctors"] == 77                 # 19 steerers + 58 X/YCOR
    assert s["bpms"] == 119
    assert s["kind_counts"]["foil"] == 1
    assert s["rf_sections_mhz"] == [162.5, 325.0, 650.0]


def test_channel_rule_beats_ke_rule(deck):
    """Documents the +12 divergence from categorize_fieldmap: the
    channel rule must classify every ke-rule cavity as a cavity, PLUS
    the parked ones."""
    from linac_gen.matching.variables import categorize_fieldmap
    ke_cavities = set()
    ch_cavities = set()
    for i, e in enumerate(deck.elements):
        if type(e).__name__ not in ("FieldMap", "FieldMap3D"):
            continue
        if categorize_fieldmap(e) == "cavity":
            ke_cavities.add(i)
        if classify_element(e)["kind"] == "cavity":
            ch_cavities.add(i)
    assert ke_cavities <= ch_cavities            # never lose a cavity
    assert len(ch_cavities - ke_cavities) == 12  # the parked HB650s


def test_fieldmap_solenoid_vs_cavity_facts(deck):
    kinds = {}
    for e in deck.elements:
        if type(e).__name__ in ("FieldMap", "FieldMap3D"):
            c = classify_element(e)
            key = (c["kind"], c["dims"])
            kinds[key] = kinds.get(key, 0) + 1
    assert kinds[("cavity", "3D")] == 135
    assert kinds[("solenoid", "3D")] == 29
    assert kinds[("solenoid", "1D")] == 8


def test_classifier_handles_bare_elements():
    """No field_data, no geom, unknown classes — never raises."""
    class _Weird:
        name = "X"
    out = classify_element(_Weird())
    assert out["kind"] == "other"

    class _Cmd:
        KEYWORD = "SET_THING"
        name = "SET_1"
    assert classify_element(_Cmd())["kind"] == "command"
