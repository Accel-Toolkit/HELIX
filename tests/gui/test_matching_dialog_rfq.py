"""Matching dialog on an RFQ deck: the RFQ period is offered first and
selected, the matched input is computed at the RFQ entrance (space charge
in a worker thread), Apply is offered only when the RFQ starts the deck,
and on a beam read from a file Apply ticks the Beam tab's re-shape option.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).parents[2]
DEMO_DAT = ROOT / "examples" / "rfq_demo" / "rfq_demo.dat"
DEMO_PROJ = ROOT / "examples" / "rfq_demo" / "rfq_demo.lgproj"

pytestmark = pytest.mark.skipif(not DEMO_DAT.exists(),
                                reason="examples/rfq_demo missing")


def _demo():
    from linac_gen.cli.common import load_input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, cfg, _ = load_input(str(DEMO_PROJ))
    return lat, cfg


def _deck(tmp_path, prefix):
    from linac_gen.io.tracewin_parser import parse_tracewin
    cards = [ln for ln in DEMO_DAT.read_text().splitlines()
             if ln.startswith("RFQ_CELL")]
    p = tmp_path / "line_rfq.dat"
    p.write_text("\n".join(["FREQ 352.21", *prefix, *cards, "END"]) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(p))
    return lat


def _setup(lattice, cfg):
    from linac_gen_gui.dialogs.matching_dialog import MatchingDialog
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    st = AppState()
    st.set_lattice(lattice, "/tmp/rfq_match.dat")
    tab = BeamTab(st)
    tab.set_beam_config(cfg)
    return tab, MatchingDialog(lattice, tab)


def _api(lattice, cfg, **kw):
    from linac_gen.cli.common import build_ref
    from linac_gen.matching.periodic import find_rfq_matched_input_twiss
    return find_rfq_matched_input_twiss(lattice, build_ref(cfg), **kw)


def test_rfq_deck_opens_on_its_period(qapp):
    lat, cfg = _demo()
    tab, dlg = _setup(lat, cfg)
    try:
        assert dlg._mode_combo.currentIndex() == 1
        assert dlg._cell_combo.isEnabled()
        kind, period = dlg._cell_combo.currentData()
        assert kind == "rfq" and (period.start, period.end) == (2, 3)
        assert "matched at the RFQ entrance" in dlg._cell_combo.currentText()
        assert not dlg._cell_note.isHidden()
        assert dlg._max_iter_spin.value() >= 200
    finally:
        dlg.deleteLater()
        tab.deleteLater()


def test_zero_current_match_and_apply(qapp):
    lat, cfg = _demo()
    tab, dlg = _setup(lat, cfg)
    try:
        dlg._compute_periodic()
        tw = _api(lat, cfg)
        assert dlg._alpha_y_label.text() == f"{tw['alpha_y']:.4f}"
        assert dlg._beta_x_label.text() == f"{tw['beta_x']:.4f}"
        assert dlg._apply_periodic_btn.isEnabled()
        assert "CAUTION: the match depends on the RF phase" in \
            dlg._log.toPlainText()                 # the demo's short RMS
        dlg._apply_matched()
        assert tab._alpha_x.value() == pytest.approx(tw["alpha_x"], abs=1e-6)
        assert tab._beta_y.value() == pytest.approx(tw["beta_y"], abs=1e-5)
        assert tab._build_cfg().reshape_to_twiss is False   # generated beam
    finally:
        dlg.deleteLater()
        tab.deleteLater()


def test_rfq_behind_a_line_is_a_target_not_applied(qapp, tmp_path):
    lat = _deck(tmp_path, ["DRIFT 100 20"])
    _lat, cfg = _demo()
    tab, dlg = _setup(lat, cfg)
    try:
        alpha_before = tab._alpha_x.value()
        dlg._compute_periodic()
        assert not dlg._apply_periodic_btn.isEnabled()
        assert "target" in dlg._apply_periodic_btn.toolTip()
        dlg._apply_matched()                       # refuses
        assert tab._alpha_x.value() == alpha_before
        assert "Not applied" in dlg._log.toPlainText()
    finally:
        dlg.deleteLater()
        tab.deleteLater()


def _wait_worker(dlg, timeout=120.0):
    from PyQt6.QtCore import QCoreApplication
    t0 = time.time()
    while dlg._worker is not None and time.time() - t0 < timeout:
        dlg._worker.wait(50)
        QCoreApplication.processEvents()
    assert dlg._worker is None, "space-charge worker did not finish"


def test_space_charge_match_runs_in_a_worker(qapp):
    lat, cfg = _demo()
    tab, dlg = _setup(lat, cfg)
    try:
        dlg._compute_sc_matched()
        assert dlg._worker is not None
        assert not dlg._sc_match_btn.isEnabled()
        assert not dlg._buttons.isEnabled()
        dlg.reject()                                # ignored while busy
        assert "Still computing" in dlg._log.toPlainText()
        _wait_worker(dlg)
        from linac_gen.distributions.factory import geometric_emittances
        from linac_gen.cli.common import build_ref
        ex, ey, _ = geometric_emittances(cfg, build_ref(cfg).bg)
        tw = _api(lat, cfg, current=cfg.current, emit_x=ex, emit_y=ey)
        assert dlg._sc_alpha_y_label.text() == f"{tw['alpha_y']:.4f}"
        assert dlg._sc_beta_y_label.text() == f"{tw['beta_y']:.4f}"
        assert dlg._sc_converged_label.text() == "Converged (RFQ period)"
        assert dlg._apply_sc_btn.isEnabled() and dlg._buttons.isEnabled()
        assert dlg._busy.isHidden()
    finally:
        dlg.deleteLater()
        tab.deleteLater()


def _dst(tmp_path):
    from linac_gen.io.tracewin_dst import write_dst
    rng = np.random.default_rng(3)
    p = np.zeros((2000, 6))
    p[:, :4] = rng.normal(size=(2000, 4)) * [0.8, 12.0, 0.8, 12.0]
    p[:, 4] = rng.uniform(-180, 180, 2000)
    path = tmp_path / "in.dst"
    write_dst(str(path), p, current_mA=15.0, frequency_MHz=352.21,
              mass_MeV=938.272088, w_kin_ref=0.075)
    return path


def test_apply_on_a_file_beam_ticks_reshape(qapp, tmp_path):
    from dataclasses import replace
    lat, cfg = _demo()
    file_cfg = replace(cfg, source="file", distribution_file=str(_dst(tmp_path)))
    tab, dlg = _setup(lat, file_cfg)
    try:
        assert not tab._reshape_cb.isHidden()
        assert not tab._reshape_cb.isChecked()
        dlg._compute_periodic()
        dlg._apply_matched()
        assert tab._reshape_cb.isChecked()
        out = tab.get_beam_config()
        assert out.reshape_to_twiss is True
        assert "re-shape to α, β" in dlg._log.toPlainText()
        tw = _api(lat, cfg)
        assert out.alpha_y == pytest.approx(tw["alpha_y"], abs=1e-6)
    finally:
        dlg.deleteLater()
        tab.deleteLater()


def test_beam_tab_reshape_checkbox(qapp, tmp_path):
    from dataclasses import replace
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    _lat, cfg = _demo()
    tab = BeamTab(AppState())
    try:
        assert tab._reshape_cb.isHidden()                 # generated beam
        assert tab.set_reshape_to_twiss(True) is False
        file_cfg = replace(cfg, source="file",
                           distribution_file=str(_dst(tmp_path)),
                           reshape_to_twiss=True)
        tab.set_beam_config(file_cfg)
        assert not tab._reshape_cb.isHidden() and tab._reshape_cb.isChecked()
        assert tab._build_cfg().reshape_to_twiss is True
        tab._clear_file_source()
        assert tab._reshape_cb.isHidden() and not tab._reshape_cb.isChecked()
        assert tab._build_cfg().reshape_to_twiss is False
        tab.set_beam_config(file_cfg)
        tab._reset()
        assert not tab._reshape_cb.isChecked()
        assert tab._build_cfg().reshape_to_twiss is False
    finally:
        tab.deleteLater()


def test_fodo_lattice_items_unchanged(qapp):
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.cli.common import build_ref
    from linac_gen.matching.periodic import find_matched_input_twiss
    lat = Lattice()
    for c in range(4):
        lat.add(Quadrupole(f"QF_{c}", 100.0, gradient=10.0, aperture=20.0))
        lat.add(Drift(f"D1_{c}", 200.0, aperture=20.0))
        lat.add(Quadrupole(f"QD_{c}", 100.0, gradient=-10.0, aperture=20.0))
        lat.add(Drift(f"D2_{c}", 200.0, aperture=20.0))
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21)
    tab, dlg = _setup(lat, cfg)
    try:
        assert dlg._mode_combo.currentIndex() == 0          # as before
        assert dlg._cell_combo.itemText(0).startswith("FODO cell 0:")
        dlg._mode_combo.setCurrentIndex(1)
        dlg._compute_periodic()
        kind, (cs, ce) = dlg._cell_combo.currentData()
        tw = find_matched_input_twiss(lat, build_ref(cfg), cs, ce)
        assert kind == "fodo"
        assert dlg._beta_x_label.text() == f"{tw['beta_x']:.4f}"
        assert dlg._apply_periodic_btn.isEnabled()
    finally:
        dlg.deleteLater()
        tab.deleteLater()
