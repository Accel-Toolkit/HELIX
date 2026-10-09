"""Clicking a legend entry hides that curve AND refits the view to the
curves that remain (pyqtgraph itself never re-runs auto-range on a
visibility change, and auto-range is off after any manual zoom).

Pins: left click toggles and refits (y always, x only when the x axis is
not linked), a second click restores, a right click does nothing, a manual
zoom is overridden by the next legend click, symmetric ±bands follow their
legend entry, ignoreBounds items and fixed-scale plots toggle without a
refit, and a real popup (BMAG along s) refits through the real legend.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


class _Click:
    def __init__(self, button):
        self._b = button

    def button(self):
        return self._b

    def accept(self):
        pass


def _left():
    from PyQt6.QtCore import Qt
    return _Click(Qt.MouseButton.LeftButton)


def _right():
    from PyQt6.QtCore import Qt
    return _Click(Qt.MouseButton.RightButton)


def _sample(leg, name):
    for sample, label in leg.items:
        if label.text == name:
            return sample
    raise KeyError(name)


def _vb(p):
    return p.getPlotItem().getViewBox()


def _ymax(p):
    _vb(p).updateAutoRange()
    return _vb(p).viewRange()[1][1]


def _two_curve_plot(qapp):
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.plot_style import add_legend, curve_pen, style_plot
    p = pg.PlotWidget()
    style_plot(p, "y", "")
    leg = add_legend(p)
    x = np.linspace(0.0, 10.0, 50)
    small = p.plot(x, np.sin(x), pen=curve_pen("#22d3ee"), name="small")
    big = p.plot(x, 100.0 * np.cos(x), pen=curve_pen("#f4a259"), name="big")
    p.resize(400, 300); p.show(); qapp.processEvents()
    return p, leg, small, big


def test_click_hides_and_refits_then_restores(qapp):
    p, leg, small, big = _two_curve_plot(qapp)
    try:
        assert _ymax(p) >= 90.0
        _sample(leg, "big").mouseClickEvent(_left())
        assert not big.isVisible() and small.isVisible()
        assert _vb(p).viewRange()[1][1] <= 1.5
        assert _vb(p).state["autoRange"][1]
        _sample(leg, "big").mouseClickEvent(_left())
        assert big.isVisible()
        assert _vb(p).viewRange()[1][1] >= 90.0
    finally:
        p.close(); p.deleteLater()


def test_manual_zoom_is_overridden_by_legend_click(qapp):
    p, leg, small, big = _two_curve_plot(qapp)
    try:
        p.setYRange(0.0, 50.0)
        assert _vb(p).state["autoRange"][1] is False
        _sample(leg, "big").mouseClickEvent(_left())
        assert _vb(p).viewRange()[1][1] <= 1.5
        assert _vb(p).state["autoRange"][1]
        assert _vb(p).state["autoRange"][0]                        # x not linked: re-enabled too
    finally:
        p.close(); p.deleteLater()


def test_right_click_does_nothing(qapp):
    p, leg, small, big = _two_curve_plot(qapp)
    try:
        before = _vb(p).viewRange()
        _sample(leg, "big").mouseClickEvent(_right())
        assert big.isVisible()
        assert _vb(p).viewRange() == before
    finally:
        p.close(); p.deleteLater()


def test_xlinked_slave_refits_y_only(qapp):
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.plot_style import add_legend, curve_pen, style_plot
    master = pg.PlotWidget(); style_plot(master, "m", "")
    master.plot(np.linspace(0, 5, 20), np.zeros(20), pen=curve_pen("#22d3ee"), name="m")
    p, leg, small, big = _two_curve_plot(qapp)
    try:
        p.setXLink(master)
        qapp.processEvents()
        x_state = _vb(p).state["autoRange"][0]
        master_x = _vb(master).viewRange()[0]
        _sample(leg, "big").mouseClickEvent(_left())
        assert _vb(p).viewRange()[1][1] <= 1.5                    # y refit
        assert _vb(p).state["autoRange"][0] == x_state           # x auto state untouched
        assert _vb(master).viewRange()[0] == master_x            # master x untouched
    finally:
        p.close(); p.deleteLater(); master.close(); master.deleteLater()


def test_symmetric_band_follows_its_legend_entry(qapp):
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.envelope_plot import _make_symmetric_band, set_band_data
    from linac_gen_gui.interphase.plots.plot_style import add_legend, curve_pen, style_plot
    p = pg.PlotWidget(); style_plot(p, "y", ""); leg = add_legend(p)
    x = np.linspace(0.0, 10.0, 50)
    p.plot(x, np.sin(x), pen=curve_pen("#22d3ee"), name="small")
    pos, neg, fill = _make_symmetric_band(p, "#f4a259", name="band")
    set_band_data(pos, neg, x, 100.0 + 0.0 * x)
    p.resize(400, 300); p.show(); qapp.processEvents()
    try:
        assert _ymax(p) >= 90.0
        _sample(leg, "band").mouseClickEvent(_left())
        assert not pos.isVisible() and not neg.isVisible() and not fill.isVisible()
        assert _vb(p).viewRange()[1][1] <= 1.5
        _sample(leg, "band").mouseClickEvent(_left())
        assert pos.isVisible() and neg.isVisible() and fill.isVisible()
        assert _vb(p).viewRange()[1][1] >= 90.0
    finally:
        p.close(); p.deleteLater()


def test_ignore_bounds_item_toggles_without_refit(qapp):
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.plot_style import HelixItemSample, curve_pen, style_plot
    p = pg.PlotWidget(); style_plot(p, "y", "")
    x = np.linspace(0.0, 10.0, 50)
    p.plot(x, np.sin(x), pen=curve_pen("#22d3ee"), name="small")
    ring = pg.PlotDataItem(x, 100.0 * np.cos(x), pen=curve_pen("#f4a259"), name="ring")
    p.getPlotItem().addItem(ring, ignoreBounds=True)
    leg = pg.LegendItem(offset=(8, 4), sampleType=HelixItemSample)
    leg.setParentItem(_vb(p)); leg.addItem(ring, "ring")
    p.resize(400, 300); p.show(); qapp.processEvents()
    try:
        _vb(p).updateAutoRange()
        before = _vb(p).viewRange(); state = _vb(p).state["autoRange"]
        assert before[1][1] <= 1.5                                # ring never drove the range
        _sample(leg, "ring").mouseClickEvent(_left())
        assert not ring.isVisible()
        assert _vb(p).viewRange() == before and _vb(p).state["autoRange"] == state
    finally:
        p.close(); p.deleteLater()


def test_fixed_range_opt_out(qapp):
    from linac_gen_gui.interphase.plots.plot_style import fix_plot_range
    p, leg, small, big = _two_curve_plot(qapp)
    try:
        fix_plot_range(p)
        p.setYRange(0.0, 1.05, padding=0)
        before = _vb(p).viewRange()
        _sample(leg, "big").mouseClickEvent(_left())
        assert not big.isVisible()
        assert _vb(p).viewRange() == before
        assert _vb(p).state["autoRange"][1] is False
    finally:
        p.close(); p.deleteLater()


# ------------------------------------------------------------ real popup
def _noop(*_a, **_k):
    return None


def test_bmag_popup_refits_after_hiding_z(qapp, gui_message_boxes):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    from linac_gen.elements.drift import Drift
    from linac_gen.tracking.envelope import EnvelopeResults, _build_sigma_matrix
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    rt._COMPANION_PROBE.clear()
    lat = Lattice(); lat.add(Drift(name="D", length=300.0, aperture=10.0))
    cfg = BeamConfig(species="H-", energy=2.12, frequency=162.5, n_particles=10)

    def sig(bz, bx=1.0):
        return _build_sigma_matrix(0.0, bx, 1.0, 0.0, 1.0, 1.0, 0.0, bz, 0.1)
    rec = DiagnosticRecorder(); rec.s = [0.0, 150.0, 300.0]
    # z badly mismatched downstream (BMAG_z 1 → 30); x slightly (1 → 1.02) so the
    # remaining range is not degenerate (a zero-height range keeps its old span)
    rec.sigma_matrix = [sig(2.0), sig(60.0, 1.1), sig(120.0, 1.2)]
    rec.element_names = ["INPUT", "D", "D"]
    env = EnvelopeResults(); env.s = [0.0, 150.0, 300.0]; env.sigma_matrix = [sig(2.0)] * 3
    env.element_exit_idx = [1]
    st = AppState(); st.set_lattice(lat); st.set_beam_config(cfg); st.set_results(rec)
    rt._COMPANION_PROBE.update({"sig": rt._probe_sig(lat, cfg), "results": env})
    tab = ResultsTab(st, _noop, _noop, _noop)
    try:
        tab._open_popup("bmag_s")
        dlg = tab._popups["bmag_s"]
        vb = dlg._plot.getPlotItem().getViewBox(); vb.updateAutoRange()
        assert vb.viewRange()[1][1] >= 10.0                       # BMAG_z ≈ 30 sets the scale
        leg = dlg._plot.getPlotItem().legend
        _sample(leg, "BMAG_z").mouseClickEvent(_left())
        assert not dlg._cz.isVisible()
        assert 0.9 <= vb.viewRange()[1][0] and vb.viewRange()[1][1] <= 1.5   # x ≈ 1.00–1.02 and the 1-line remain
        names = {str(c.name) for pn in dlg._collect_panels() for c in (getattr(pn, "curves", None) or [])}
        assert "BMAG_z" in names                                  # export unaffected by visibility
        assert gui_message_boxes == []
    finally:
        for d in tab._popups.values():
            d.close()
        tab.deleteLater(); rt._COMPANION_PROBE.clear()
