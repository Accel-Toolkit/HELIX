"""GUI round-trip of the BeamConfig dispersion fields (2026-07-11).

Guards the parity requirement: without BeamTab set/get support the new
``disp_*`` fields would be silently dropped on every project save/load
cycle; without the MatchingDialog Apply wiring the matched dispersion
would never reach the beam setup.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("PyQt6")


def _cfg(**kw):
    from linac_gen.core.config import BeamConfig
    base = dict(species="proton", energy=3.0, frequency=162.5,
                current=0.0, n_particles=1000)
    base.update(kw)
    return BeamConfig(**base)


def test_beam_tab_disp_round_trip(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    try:
        cfg_in = _cfg(disp_x=3.25, disp_xp=-1.5, disp_y=0.75, disp_yp=0.0625)
        tab.set_beam_config(cfg_in)
        cfg_out = tab._build_cfg()
        assert cfg_out.disp_x == pytest.approx(3.25)
        assert cfg_out.disp_xp == pytest.approx(-1.5)
        assert cfg_out.disp_y == pytest.approx(0.75)
        assert cfg_out.disp_yp == pytest.approx(0.0625)
        # Legacy config without the attributes → zeros, no crash.
        legacy = _cfg()
        for k in ("disp_x", "disp_xp", "disp_y", "disp_yp"):
            delattr(legacy, k) if hasattr(legacy, k) else None
        tab.set_beam_config(_cfg())
        assert tab._build_cfg().disp_x == 0.0
    finally:
        tab.deleteLater()


def test_matching_dialog_apply_pushes_dispersion(qapp, mini_lattice):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    from linac_gen_gui.dialogs.matching_dialog import MatchingDialog

    st = AppState()
    st.set_lattice(mini_lattice, "/tmp/disp_apply.dat")
    beam_tab = BeamTab(st)
    dlg = MatchingDialog(mini_lattice, beam_tab)
    try:
        dlg._matched_twiss = {
            "alpha_x": 1.0, "beta_x": 2.0, "alpha_y": -0.5, "beta_y": 3.0,
            "disp_x": 4.5, "disp_xp": -0.25, "disp_y": 0.0,
            "disp_yp": float("nan"),   # integer-tune sentinel
        }
        beam_tab._disp_yp.setValue(9.0)   # pre-existing value must survive NaN
        dlg._apply_matched()
        assert beam_tab._alpha_x.value() == pytest.approx(1.0)
        assert beam_tab._disp_x.value() == pytest.approx(4.5)
        assert beam_tab._disp_xp.value() == pytest.approx(-0.25)
        assert beam_tab._disp_y.value() == pytest.approx(0.0)
        # NaN must NOT be written into the spinbox.
        assert beam_tab._disp_yp.value() == pytest.approx(9.0)
        assert not math.isnan(beam_tab._disp_yp.value())
    finally:
        dlg.deleteLater()
        beam_tab.deleteLater()
