"""Beam tab keeps project beam values exactly.

The QDoubleSpinBox fields rounded every value to 3-6 decimals: a project
beam of 4.84235 mA / ε = 0.2052723 reached the form as 4.842 / 0.205272,
and pressing Apply wrote the rounded values back into the run.  Qt rounds
at every decimal setting, so the physics boxes now remember the exact
value until the user edits it.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import pytest

pytest.importorskip("PyQt6")

# SCL-revised beam (TraceWin PIP_II_FDR_v_2.ini, beam 1) + a full-precision
# energy as read from a .dst header
_VALUES = dict(
    species="H-", energy=2.1226235573291987, frequency=162.5,
    current=4.84235, n_particles=10000,
    emit_nx=0.2052723, alpha_x=-0.44807844, beta_x=0.5135438,
    emit_ny=0.1972638, alpha_y=-0.40463024, beta_y=0.51570757,
    emit_z=0.06519386, alpha_z=-0.97440478, beta_z=478.72045,
    centroid_x=6.4222938e-10, centroid_xp=-1.9474942e-08,
    centroid_dphi=8.5718424e-08,
    disp_x=0.123456789, disp_xp=-1.000000001,
)


def test_project_beam_survives_the_form_exactly(qapp):
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        cfg = BeamConfig(**_VALUES)
        tab.set_beam_config(cfg)
        out = asdict(tab._build_cfg())
        for k, v in _VALUES.items():
            assert out[k] == v, (k, out[k], v)
        # shown without trailing zeros
        assert tab._current.cleanText() == "4.84235"
        assert tab._emit_nx.cleanText() == "0.2052723"
    finally:
        tab.close()


def test_user_edits_still_win(qapp):
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        tab.set_beam_config(BeamConfig(**_VALUES))
        tab._current.stepBy(1)                    # one single-step (0.01)
        assert tab._build_cfg().current == pytest.approx(4.85235, abs=1e-12)
        tab._current.selectAll()                  # the number, not the suffix
        tab._current.lineEdit().insert("5.5")
        tab._current.interpretText()
        assert tab._build_cfg().current == 5.5
    finally:
        tab.close()


def test_typing_the_displayed_number_replaces_a_hidden_value(qapp):
    """3e-12 displays as "0"; typing 0 must give 0, not the hidden 3e-12."""
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        tab.set_beam_config(BeamConfig(**dict(_VALUES, centroid_x=3e-12)))
        assert tab._build_cfg().centroid_x == 3e-12
        from PyQt6.QtTest import QTest
        box = tab._cx
        box.selectAll()
        QTest.keyClicks(box.lineEdit(), "0")      # real key events
        box.interpretText()
        assert tab._build_cfg().centroid_x == 0.0
    finally:
        tab.close()
