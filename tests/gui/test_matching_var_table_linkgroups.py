"""Matching-tab Variables table vs. link groups (display-only defect).

``MatchResult.variables`` has one entry per ADJUST DoF while
``x0``/``x_final`` have one entry per optimiser COLUMN (linked
variables share a column).  The table fill in ``_on_match_finished``
used a positional ``zip(variables, x0, x_final)`` which truncated to
the shortest list: on the linked demo deck row 1 got no
QTableWidgetItem at all, and on a mixed deck the follower row showed
the NEXT column's values while the tail row stayed blank.

These tests drive the real slot (``_on_match_finished`` — the
``finished_with`` target of ``_MatchWorker``) with the real engine on
four regimes: linked, mixed, unlinked (byte-identical rendering guard)
and the ADJUST_BEAM_TWISS demo whose flag=1 knobs carry singleton link
groups (must never be tagged ``(link n)``).
"""
from __future__ import annotations

import copy
from pathlib import Path

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching import match

from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
from linac_gen_gui.interphase.tabs.matching_tab import MatchingTab

REPO = Path(__file__).resolve().parents[2]
LINKED_DECK = REPO / "examples" / "matching_demo.dat"
TWISS_DECK = REPO / "examples" / "twiss_matching_demo.dat"

# Linked pair (link_group=1) plus one independent ADJUST → 3 DoF, 2 columns.
MIXED_TEXT = (
    "ADJUST QUAD 2 1 -30 30 0.5 0\n"
    "DRIFT 100 30\n"
    "QUAD 80 5 20\n"
    "ADJUST QUAD 2 1 -30 30 0.5 0\n"
    "DRIFT 200 30\n"
    "QUAD 80 -5 20\n"
    "ADJUST QUAD 2 0 -30 30 0.5 0\n"
    "DRIFT 100 30\n"
    "QUAD 80 3 20\n"
    "DRIFT 100 30\n"
    "SET_SIZE 1 4 0 0 0\n"
    "END\n"
)

# Same two quads, link_group=0 → 2 DoF, 2 columns (pre-fix rendering
# must be preserved verbatim).
UNLINKED_TEXT = (
    "ADJUST QUAD 2 0 -30 30 0.5 0\n"
    "DRIFT 100 30\n"
    "QUAD 80 5 20\n"
    "ADJUST QUAD 2 0 -30 30 0.5 0\n"
    "DRIFT 200 30\n"
    "QUAD 80 -5 20\n"
    "DRIFT 100 30\n"
    "SET_SIZE 1 4 0 0 0\n"
    "END\n"
)


def _render(qapp, dat_path, max_iter=200):
    """Parse a deck, run the real matcher, and push the result through
    the app-level ``_on_match_finished`` slot; return the result and
    the raw QTableWidgetItems of the Variables table."""
    lat, meta = parse_tracewin(str(dat_path))
    assert meta["warnings"] == []
    st = AppState()
    st.set_lattice(lat, str(dat_path))
    beam_tab = BeamTab(st)
    tab = MatchingTab(st, beam_tab)
    cfg = copy.deepcopy(st.beam_config)
    lat2 = copy.deepcopy(lat)
    res = match(lat2, cfg, max_iter=max_iter, algorithm="least_squares")
    tab._on_match_finished(lat2, cfg, res)  # the worker's finished_with slot
    T = tab._aa_var_table
    cells = [[T.item(r, c) for c in range(T.columnCount())]
             for r in range(T.rowCount())]
    return res, cells, tab, beam_tab


def test_linked_demo_deck_fills_every_row(qapp):
    res, cells, tab, _bt = _render(qapp, LINKED_DECK)
    # Both ADJUST cards share link_group=1 → one optimiser column.
    assert res.x0.shape == (1,)
    assert len(res.variables) == 2
    assert tab._aa_var_table.rowCount() == 2
    for r in range(2):
        for c in range(4):
            assert cells[r][c] is not None, f"row {r} col {c} has no item"
    for r in range(2):
        assert cells[r][0].text().startswith(res.variables[r].label)
        assert "(link 1)" in cells[r][0].text()
        assert cells[r][1].text() == f"{float(res.x0[0]):.6g}"
        assert cells[r][2].text() == f"{float(res.x_final[0]):.6g}"
        assert cells[r][3].text() == "[-30, 30]"
    # Apply/Save path untouched: the stored result is the object itself.
    assert tab._aa_result[2] is res


def test_mixed_deck_follower_and_independent_rows(qapp, tmp_path):
    deck = tmp_path / "mixed.dat"
    deck.write_text(MIXED_TEXT)
    res, cells, tab, _bt = _render(qapp, deck)
    assert res.x0.shape == (2,)
    assert tab._aa_var_table.rowCount() == 3
    for r in range(3):
        for c in range(4):
            assert cells[r][c] is not None, f"row {r} col {c} has no item"
    # Rows 0 and 1 (linked pair) both show column 0's values.
    for r in (0, 1):
        assert "(link 1)" in cells[r][0].text()
        assert cells[r][1].text() == f"{float(res.x0[0]):.6g}"
        assert cells[r][2].text() == f"{float(res.x_final[0]):.6g}"
    # Row 2 (independent) shows ITS OWN column, unannotated.
    assert cells[2][0].text() == "QUAD_003.gradient"
    assert cells[2][1].text() == f"{float(res.x0[1]):.6g}"
    assert cells[2][2].text() == f"{float(res.x_final[1]):.6g}"


def test_unlinked_deck_renders_identically(qapp, tmp_path):
    """Second-regime identity guard: passes before AND after the fix —
    an unlinked deck must render exactly as it always did."""
    deck = tmp_path / "unlinked.dat"
    deck.write_text(UNLINKED_TEXT)
    res, cells, tab, _bt = _render(qapp, deck)
    T = tab._aa_var_table
    assert [T.horizontalHeaderItem(c).text() for c in range(4)] == \
        ["Variable", "Initial", "Matched", "Bounds"]
    assert T.rowCount() == 2
    for i, var in enumerate(res.variables):
        texts = [cells[i][c].text() for c in range(4)]
        assert texts == [
            var.label,
            f"{float(res.x0[i]):.6g}",
            f"{float(res.x_final[i]):.6g}",
            f"[{var.vmin:.4g}, {var.vmax:.4g}]",
        ]
        assert "(link" not in texts[0]


def test_twiss_deck_singleton_groups_never_annotated(qapp):
    """ADJUST_BEAM_TWISS flag=1 knobs carry singleton link groups
    (1000000001.., variables.py) — one member per column, so no row may
    be tagged '(link n)' and every row shows its own column."""
    res, cells, tab, _bt = _render(qapp, TWISS_DECK, max_iter=100)
    assert len(res.variables) == 4
    assert res.x0.shape == (4,)
    assert tab._aa_var_table.rowCount() == 4
    for i, var in enumerate(res.variables):
        assert var.link_group != 0  # the singleton groups are real
        assert cells[i][0] is not None
        assert cells[i][0].text() == var.label
        assert "(link" not in cells[i][0].text()
        assert cells[i][1].text() == f"{float(res.x0[i]):.6g}"
        assert cells[i][2].text() == f"{float(res.x_final[i]):.6g}"
