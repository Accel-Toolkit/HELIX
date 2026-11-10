"""The Reliability Study's Apply refuses a bend setting: a campaign run before bends left the
compensator families re-tuned bend angles, which move the reference path."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")

DECK = """FREQ 352.21
DRIFT 100 20
QUAD 100 8 20
DRIFT 100 20
EDGE 5 1000 30 0.45 2.8 20 0
BEND 10 1000 0 20 0
EDGE 5 1000 30 0.45 2.8 20 0
DRIFT 100 20
QUAD 100 -8 20
DRIFT 100 20
END
"""


def test_apply_refuses_a_bend_setting_from_an_older_campaign(win, tmp_path):
    from linac_gen.core.config import BeamConfig
    deck = tmp_path / "line.dat"
    deck.write_text(DECK)
    win.open_lattice(deck)
    win.state.set_beam_config(BeamConfig(species="proton", energy=3.0, frequency=352.21))
    win._toolbar.open_reliability_requested.emit()
    dlg = win._reliability_dlg
    lat = win.state.lattice
    bend = next(e for e in lat.elements if type(e).__name__ == "Dipole")
    quad = next(e for e in lat.elements if type(e).__name__ == "Quadrupole")
    angle, gradient = bend.angle, quad.gradient
    dlg._comp_rows = [{"case_id": "S2_QUAD_001", "strategy": "k_out_of_n",
                       "settings": json.dumps({f"{quad.name}.gradient": gradient * 1.1,
                                               f"{bend.name}.angle": angle * 1.2})}]
    dlg._comp_table.setRowCount(1)
    dlg._comp_table.setCurrentCell(0, 0)
    dlg._apply_allowed = lambda: True
    n = len(win.message_boxes)
    dlg._apply_settings()
    assert len(win.message_boxes) == n + 1, win.message_boxes[n:]
    assert bend.name in win.message_boxes[-1][2] and "bend" in win.message_boxes[-1][2]
    assert bend.angle == angle and quad.gradient == gradient      # nothing applied
