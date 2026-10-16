"""Matched input Twiss of an RFQ (``find_rfq_periods`` /
``find_rfq_matched_input_twiss``) and the ``twiss`` CLI on RFQ decks.

The RFQ's input is matched to its first regular focusing period (two
consecutive cells of opposite sign), carried back through the radial
matching section to the RFQ entrance; with space charge from a DC input
beam.  Decks: the public demo RFQ (examples/rfq_demo) and variants built
from its cards.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching import periodic
from linac_gen.matching.periodic import (
    RfqPeriod, find_fodo_cells, find_matched_input_twiss,
    find_rfq_matched_input_twiss, find_rfq_periods, find_sc_matched_input_twiss,
)
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss, propagate_twiss, reference_at,
)

ROOT = Path(__file__).parents[2]
DEMO_DAT = ROOT / "examples" / "rfq_demo" / "rfq_demo.dat"
DEMO_PROJ = ROOT / "examples" / "rfq_demo" / "rfq_demo.lgproj"

pytestmark = pytest.mark.skipif(not DEMO_DAT.exists(),
                                reason="examples/rfq_demo missing")


def _cards():
    return [ln for ln in DEMO_DAT.read_text().splitlines()
            if ln.startswith("RFQ_CELL")]


def _with_type(card, t):
    f = card.split()
    f[7] = str(t)                    # RFQ_CELL V r0 A10 m L phis TYPE ...
    return " ".join(f)


def _deck(tmp_path, body, name="deck.dat"):
    p = tmp_path / name
    p.write_text("\n".join(["FREQ 352.21", *body, "END"]) + "\n")
    return _parse(p)


def _parse(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(path))
    return lat


def _ref(freq=352.21):
    return ReferenceParticle(species=PROTON, w_kin=0.075, frequency=freq)


TW_KEYS = ("alpha_x", "beta_x", "alpha_y", "beta_y")


# ---------------------------------------------------------------------------
# find_rfq_periods
# ---------------------------------------------------------------------------
def test_demo_period_is_first_regular_pair():
    lat = _parse(DEMO_DAT)
    periods = find_rfq_periods(lat)
    assert len(periods) == 1
    p = periods[0]
    assert (p.rfq_start, p.start, p.end) == (1, 2, 3)
    assert lat.elements[2].cell_type == -lat.elements[3].cell_type
    assert "RFQ_002 + RFQ_003" in p.label


def test_period_skips_transition_cell_and_lattice_card(tmp_path):
    c = _cards()
    body = [c[0], _with_type(c[1], 4), "LATTICE 2 0", *c[2:-1],
            "LATTICE_END", c[-1]]
    lat = _deck(tmp_path, body)
    (p,) = find_rfq_periods(lat)
    names = [e.name for e in lat.elements]
    assert names[p.rfq_start] == "RFQ_001"
    assert (names[p.start], names[p.end]) == ("RFQ_003", "RFQ_004")
    assert type(lat.elements[p.start - 1]).__name__ == "Marker"   # LATTICE


def test_same_sign_cells_are_no_period(tmp_path):
    c = _cards()
    body = [c[0]] + [_with_type(x, 2) for x in c[1:-1]] + [c[-1]]
    assert find_rfq_periods(_deck(tmp_path, body)) == []


def test_no_rfq_no_period():
    assert find_rfq_periods(_parse(ROOT / "examples" / "fodo_cell.dat")) == []


def test_two_rfqs_two_periods(tmp_path):
    c = _cards()[:30]
    lat = _deck(tmp_path, [*c, "DRIFT 50 20", *c])
    periods = find_rfq_periods(lat)
    assert len(periods) == 2
    assert periods[1].rfq_start > periods[0].end
    assert periods[1].start - periods[1].rfq_start == 1


# ---------------------------------------------------------------------------
# reference_at — the prefix replay shared with compute_transfer_matrix
# ---------------------------------------------------------------------------
def test_reference_at_matches_transfer_matrix_prefix():
    lat = _parse(DEMO_DAT)
    from linac_gen.core.lattice import Lattice
    sub = Lattice()
    for e in lat.elements[2:]:
        sub.add(e)
    full = compute_transfer_matrix(lat, _ref(), start=2, end=7)
    part = compute_transfer_matrix(sub, reference_at(lat, _ref(), 2),
                                   start=0, end=5)
    assert np.array_equal(full, part)


# ---------------------------------------------------------------------------
# find_rfq_matched_input_twiss — zero current
# ---------------------------------------------------------------------------
def test_matched_input_at_entrance_equals_back_propagation():
    lat = _parse(DEMO_DAT)
    tw = find_rfq_matched_input_twiss(lat, _ref())
    ref_tw = find_matched_input_twiss(lat, _ref(), 2, 3)
    for k in TW_KEYS:
        assert tw[k] == pytest.approx(ref_tw[k], rel=1e-12, abs=1e-15)
    assert tw["kind"] == "rfq" and tw["at_lattice_entrance"] is True
    assert (tw["rfq_start"], tw["cell_start"], tw["cell_end"]) == (1, 2, 3)
    assert tw["rfq_name"] == "RFQ_001" and tw["converged"] is True
    assert tw["beta_x"] > 0 and tw["beta_y"] > 0
    assert 0.0 < tw["mu_x"] < 180.0


def test_matched_input_reaches_the_periodic_twiss():
    """The defining property: carried forward through the radial matching
    section, the result is the period's own periodic Twiss."""
    lat = _parse(DEMO_DAT)
    tw = find_rfq_matched_input_twiss(lat, _ref())
    m_front = compute_transfer_matrix(lat, _ref(), start=1, end=1)
    m_cell = compute_transfer_matrix(lat, _ref(), start=2, end=3)
    for plane, (i, j) in (("x", (0, 1)), ("y", (2, 3))):
        a, b = propagate_twiss(m_front[i:j + 1, i:j + 1],
                               tw[f"alpha_{plane}"], tw[f"beta_{plane}"])
        per = compute_twiss(m_cell, plane)
        assert a == pytest.approx(per["alpha"], abs=1e-9)
        assert b == pytest.approx(per["beta"], rel=1e-9)


def test_upstream_line_is_not_back_propagated(tmp_path):
    """An RFQ behind a line: same Twiss at the RFQ entrance, flagged as not
    the lattice-entrance beam."""
    bare = find_rfq_matched_input_twiss(_deck(tmp_path, _cards(), "a.dat"),
                                        _ref())
    lat = _deck(tmp_path, ["DRIFT 100 20", "QUAD 50 2 20", "DRIFT 30 20",
                           *_cards()], "b.dat")
    tw = find_rfq_matched_input_twiss(lat, _ref())
    for k in TW_KEYS:
        assert tw[k] == bare[k]
    assert tw["at_lattice_entrance"] is False
    assert tw["upstream_mm"] == pytest.approx(180.0)
    assert tw["rfq_start"] == 4
    assert find_fodo_cells(lat) == []          # FODO detection unchanged


def test_zero_length_lens_upstream_is_not_the_entrance(tmp_path):
    lat = _deck(tmp_path, ["EDGE 10 1000 50 0.45 2.8 50 0", *_cards()])
    tw = find_rfq_matched_input_twiss(lat, _ref())
    assert tw["upstream_mm"] == 0.0
    assert tw["at_lattice_entrance"] is False


def test_freq_card_before_the_rfq_sets_its_frequency(tmp_path):
    """Matrix-mode replay ignores FREQ; the RFQ is matched at the
    frequency of the last FREQ card before it."""
    lat = _deck(tmp_path, _cards())             # FREQ 352.21 first
    at_card = find_rfq_matched_input_twiss(lat, _ref(freq=176.105))
    at_beam = find_rfq_matched_input_twiss(lat, _ref(freq=352.21))
    for k in TW_KEYS:
        assert at_card[k] == pytest.approx(at_beam[k], rel=1e-12)


def test_no_period_raises():
    with pytest.raises(ValueError, match="no RFQ period"):
        find_rfq_matched_input_twiss(_parse(ROOT / "examples" / "fodo_cell.dat"),
                                     _ref())


# ---------------------------------------------------------------------------
# find_rfq_matched_input_twiss — space charge (DC input beam)
# ---------------------------------------------------------------------------
def _demo_emittances():
    from linac_gen.cli.common import build_ref, load_input
    from linac_gen.distributions.factory import geometric_emittances
    _lat, cfg, _ = load_input(str(DEMO_PROJ))
    ex, ey, _ = geometric_emittances(cfg, build_ref(cfg).bg)
    return ex, ey


def test_sc_match_uses_a_dc_seed(monkeypatch):
    lat = _parse(DEMO_DAT)
    ex, ey = _demo_emittances()
    seen = {}
    real = periodic.find_sc_matched_input_twiss

    def spy(*args, **kwargs):
        seen["base"] = dict(args[5])
        seen["sc_factor"] = kwargs.get("sc_factor")
        seen["max_iter"] = kwargs.get("max_iter")
        return real(*args, **kwargs)

    monkeypatch.setattr(periodic, "find_sc_matched_input_twiss", spy)
    tw = find_rfq_matched_input_twiss(lat, _ref(), current=15.0,
                                      emit_x=ex, emit_y=ey)
    assert seen["base"]["continuous"] is True
    assert seen["base"]["emit_x"] == pytest.approx(ex)
    assert seen["base"]["emit_y"] == pytest.approx(ey)
    assert seen["sc_factor"] == 1.0 and seen["max_iter"] == 200
    assert tw["converged"] is True and tw["current"] == 15.0


def test_sc_match_differs_from_zero_current_and_bunched_seed():
    """Space charge changes the match; a bunched seed (the old dialog's)
    would silently return about the zero-current answer."""
    lat = _parse(DEMO_DAT)
    ex, ey = _demo_emittances()
    zc = find_rfq_matched_input_twiss(lat, _ref())
    sc = find_rfq_matched_input_twiss(lat, _ref(), current=15.0,
                                      emit_x=ex, emit_y=ey)
    bunched = find_sc_matched_input_twiss(
        lat, _ref(), 2, 3, 15.0,
        {"alpha_z": 0.0, "beta_z": 1.0, "emit_x": ex, "emit_y": ey,
         "emit_z": 0.0}, max_iter=200)
    assert abs(sc["alpha_y"] - zc["alpha_y"]) > 0.05 * abs(zc["alpha_y"])
    assert abs(sc["beta_y"] - zc["beta_y"]) > 0.03 * zc["beta_y"]
    assert abs(bunched["alpha_y"] - zc["alpha_y"]) < 0.01 * abs(zc["alpha_y"])


def test_sc_match_keeps_upstream_compensation(tmp_path, monkeypatch):
    lat = _deck(tmp_path, ["DRIFT 100 20", "SPACE_CHARGE_COMP 0.9",
                           *_cards()])
    ex, ey = _demo_emittances()
    seen = {}
    real = periodic.find_sc_matched_input_twiss

    def spy(*args, **kwargs):
        seen["sc_factor"] = kwargs.get("sc_factor")
        return real(*args, **kwargs)

    monkeypatch.setattr(periodic, "find_sc_matched_input_twiss", spy)
    find_rfq_matched_input_twiss(lat, _ref(), current=15.0, emit_x=ex,
                                 emit_y=ey)
    assert seen["sc_factor"] == pytest.approx(0.1)


def test_sc_match_needs_emittances():
    with pytest.raises(ValueError, match="emit_x and emit_y"):
        find_rfq_matched_input_twiss(_parse(DEMO_DAT), _ref(), current=5.0)


def test_explicit_period():
    lat = _parse(DEMO_DAT)
    p = RfqPeriod(rfq_start=1, start=4, end=5, label="RFQ: later period")
    tw = find_rfq_matched_input_twiss(lat, _ref(), p)
    first = find_rfq_matched_input_twiss(lat, _ref())
    assert (tw["cell_start"], tw["cell_end"]) == (4, 5)
    # the cells change slowly: a later period gives nearly the same input
    for k in TW_KEYS:
        assert tw[k] == pytest.approx(first[k], rel=0.03, abs=0.01)


# ---------------------------------------------------------------------------
# twiss CLI on RFQ decks
# ---------------------------------------------------------------------------
def _cli(capsys, *argv):
    from linac_gen.__main__ import main
    rc = main(["twiss", *argv])
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli_default_is_the_rfq_period(capsys):
    rc, out, _ = _cli(capsys, str(DEMO_PROJ), "-q")
    assert rc == 0
    nums = [float(v) for v in out.split()]
    tw = find_rfq_matched_input_twiss(_parse(DEMO_DAT), _ref())
    assert nums == pytest.approx([tw[k] for k in TW_KEYS], rel=1e-7)
    rc, out, _ = _cli(capsys, str(DEMO_PROJ))
    assert rc == 0 and "at the RFQ entrance (RFQ_001, element 1)" in out
    assert "RFQ period RFQ_002 + RFQ_003" in out


def test_cli_list_cells_shows_rfq_period(capsys):
    rc, out, _ = _cli(capsys, str(DEMO_PROJ), "--list-cells")
    assert rc == 0
    assert "1 cell candidate(s)" in out and "RFQ period" in out


def test_cli_notes_rfq_behind_a_line(tmp_path, capsys):
    deck = tmp_path / "line.dat"
    deck.write_text("\n".join(["FREQ 352.21", "DRIFT 100 20", *_cards(),
                               "END"]) + "\n")
    rc, out, _ = _cli(capsys, str(deck), "--energy", "0.075", "--species",
                      "proton", "--freq", "352.21")
    assert rc == 0 and "starts 0.1000 m into the lattice" in out


def test_cli_space_charge(capsys):
    ex, ey = _demo_emittances()
    rc, out, _ = _cli(capsys, str(DEMO_PROJ), "-q", "--current", "15")
    assert rc == 0
    nums = [float(v) for v in out.split()]
    tw = find_rfq_matched_input_twiss(_parse(DEMO_DAT), _ref(), current=15.0,
                                      emit_x=ex, emit_y=ey)
    assert nums == pytest.approx([tw[k] for k in TW_KEYS], rel=1e-7)


def test_cli_space_charge_needs_emittances_on_a_bare_deck(capsys):
    bare = [str(DEMO_DAT), "--energy", "0.075", "--species", "proton",
            "--freq", "352.21", "-q"]
    rc, _, err = _cli(capsys, *bare, "--current", "15")
    assert rc == 2 and "needs the beam emittances" in err
    rc, out, _ = _cli(capsys, *bare, "--current", "15", "--emit-nx", "0.2",
                      "--emit-ny", "0.2")
    assert rc == 0 and len(out.split()) == 4


def test_cli_current_refused_in_whole_mode(capsys):
    rc, _, err = _cli(capsys, str(DEMO_PROJ), "--mode", "whole",
                      "--current", "5")
    assert rc == 2 and "cell mode" in err


def test_cli_no_cell_message(capsys):
    rc, out, _ = _cli(capsys, str(ROOT / "examples" / "matching_demo.dat"),
                      "--list-cells", "--energy", "3", "--species", "proton")
    assert rc == 0 and out.startswith("no cell detected")


# ---------------------------------------------------------------------------
# RF-phase sensitivity: a DC beam enters at every phase
# ---------------------------------------------------------------------------
def _long_rms_deck(tmp_path, factor):
    c = _cards()
    f = c[0].split()
    f[5] = f"{float(f[5]) * factor:.4f}"          # radial matching cell length
    return _deck(tmp_path, [" ".join(f), *c[1:]], f"rms{factor}.dat")


def test_short_radial_matching_section_is_flagged():
    """The demo's radial matching section is one 5.4 mm cell (0.5 focusing
    periods): its synchronous-particle match does not suit the DC beam."""
    tw = find_rfq_matched_input_twiss(_parse(DEMO_DAT), _ref())
    assert tw["input_periods"] == pytest.approx(0.5, abs=0.01)
    assert tw["phase_sensitivity"] > 1.0
    assert "depends on the RF phase" in tw["caution"]


def test_long_radial_matching_section_is_phase_independent(tmp_path):
    """Lengthened to 4 focusing periods the same RFQ's match no longer
    depends on the RF phase, and becomes nearly round."""
    tw = find_rfq_matched_input_twiss(_long_rms_deck(tmp_path, 8), _ref())
    assert tw["input_periods"] == pytest.approx(4.0, abs=0.01)
    assert tw["phase_sensitivity"] < periodic.RFQ_PHASE_SENSITIVITY_LIMIT
    assert tw["caution"] is None
    assert tw["alpha_x"] == pytest.approx(tw["alpha_y"], rel=0.05)
    assert tw["beta_x"] == pytest.approx(tw["beta_y"], rel=0.05)


def test_phase_probe_leaves_the_lattice_untouched():
    lat = _parse(DEMO_DAT)
    before = [e.phi_s_deg for e in lat.elements if hasattr(e, "phi_s_deg")]
    find_rfq_matched_input_twiss(lat, _ref())
    assert before == [e.phi_s_deg for e in lat.elements
                      if hasattr(e, "phi_s_deg")]


def test_cli_prints_the_caution(capsys):
    rc, out, _ = _cli(capsys, str(DEMO_PROJ))
    assert rc == 0 and "caution: the match depends on the RF phase" in out
    rc, out, err = _cli(capsys, str(DEMO_PROJ), "-q")
    assert rc == 0 and len(out.split()) == 4
    assert "warning: the match depends on the RF phase" in err


def test_cli_bunched_space_charge_needs_a_project(capsys):
    """A FODO cell's space charge depends on the bunch length, which a bare
    deck does not carry."""
    rc, _, err = _cli(capsys, str(ROOT / "examples" / "fodo_cell.dat"),
                      "--energy", "3", "--species", "proton", "--current",
                      "5", "--emit-nx", "0.2", "--emit-ny", "0.2", "-q")
    assert rc == 2 and "longitudinal emittance" in err
    proj = ROOT / "examples" / "batch_test" / "fodo.lgproj"
    if proj.exists():
        rc, out, _ = _cli(capsys, str(proj), "--current", "5", "-q")
        assert rc == 0 and len(out.split()) == 4


def test_cli_manual_rfq_range_uses_a_dc_seed(capsys):
    """--cell-start/--cell-end inside an RFQ with --current: DC beam, the
    same answer as the RFQ period when the RFQ starts the lattice."""
    ex, ey = _demo_emittances()
    rc, out, _ = _cli(capsys, str(DEMO_PROJ), "--cell-start", "2",
                      "--cell-end", "3", "--current", "15", "-q")
    assert rc == 0
    nums = [float(v) for v in out.split()]
    tw = find_rfq_matched_input_twiss(_parse(DEMO_DAT), _ref(), current=15.0,
                                      emit_x=ex, emit_y=ey)
    assert nums == pytest.approx([tw[k] for k in TW_KEYS], rel=1e-4)
