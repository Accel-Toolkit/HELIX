# tests/matching/test_adjust_family_semantics.py
"""Integer ADJUST targets: TraceWin family semantics vs legacy index.

TraceWin's ``ADJUST N v`` ties the variable to diagnostic family N and
varies the NEXT element after the card.  HELIX historically read the
integer as a 1-based element index.  The resolver disambiguates by the
deck itself: if any ``DIAG_POSITION N`` marker declares family N, the
TW semantics win; otherwise the legacy index behavior is preserved.
"""
import os

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.matching.variables import (MatchingConfigError,
                                          collect_variables)

FNALSCL = os.path.join(os.path.dirname(__file__), "..", "..",
                       "examples", "piplattice", "fnalscl.dat")


def _beam_cfg():
    return BeamConfig()


def test_family_target_binds_next_element_when_diag_declared():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Marker("BPM1", is_bpm=True, diag_family=7,
                   x_target_mm=0.1, y_target_mm=0.0))
    lat.add(Adjust("A1", target="7", param_idx=2))
    lat.add(Quadrupole("Q1", 85.3, -8.0))
    vs = collect_variables(lat, _beam_cfg())
    assert len(vs) == 1
    assert vs[0].target.name == "Q1" and vs[0].attr == "gradient"


def test_index_fallback_without_diag_markers():
    # No DIAG_POSITION family declared -> integer stays a 1-based index
    # (pinned legacy behavior, see test_variables.py).
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Quadrupole("Q1", 85.3, -8.0))
    lat.add(Adjust("A1", target="2", param_idx=2))
    lat.add(Quadrupole("Q2", 85.3, 8.0))
    vs = collect_variables(lat, _beam_cfg())
    assert len(vs) == 1
    assert vs[0].target.name == "Q1"     # 2nd element, NOT the next one


def test_family_without_following_element_warns_and_skips():
    lat = Lattice()
    lat.add(Marker("BPM1", is_bpm=True, diag_family=3))
    lat.add(Adjust("A1", target="3", param_idx=2))   # nothing follows
    with pytest.warns(UserWarning, match="no element follows"):
        vs = collect_variables(lat, _beam_cfg())
    assert vs == []


def test_steerer_planes_from_param_idx():
    lat = Lattice()
    lat.add(Marker("BPM1", is_bpm=True, diag_family=5, x_target_mm=0.0))
    lat.add(Adjust("A1", target="5", param_idx=1))
    lat.add(Steerer("S1"))
    lat.add(Adjust("A2", target="5", param_idx=2))
    lat.add(Steerer("S2"))
    vs = collect_variables(lat, _beam_cfg())
    assert [(v.target.name, v.attr) for v in vs] == \
        [("S1", "bx_l"), ("S2", "by_l")]


def test_fnalscl_collects_30_variables():
    if not os.path.exists(FNALSCL):
        pytest.skip("fnalscl.dat not present")
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(FNALSCL)
    vs = collect_variables(lat, _beam_cfg())
    assert len(vs) == 30
    attrs = [(type(v.target).__name__, v.attr) for v in vs]
    assert attrs.count(("Steerer", "bx_l")) == 1     # D01T horizontal trim
    assert attrs.count(("Steerer", "by_l")) == 1     # D01T vertical trim
    assert attrs.count(("Quadrupole", "gradient")) == 28
    # seeds come from the deck values
    sx = next(v for v in vs if v.attr == "bx_l")
    assert sx.x0 == pytest.approx(4.56e-5)
