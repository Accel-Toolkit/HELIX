"""Plots fit all their data whenever they are drawn, on x and y.

pyqtgraph switches a view's auto-range off whenever its range is set: by
code, by a mouse zoom or pan, and by the menu's "View All" itself, which
fits once.  The Interphase plots live for the whole session, so a frame once
fitted by hand used to stay frozen for every later run.  Pinned here:

* View All, the "A" button and the x "Auto" option (and its percentage) fit
  AND keep fitting; on a lower panel they act on the leader's x and never
  fit the follower's; on a fixed view a designed axis is fitted once and an
  undesigned one keeps fitting, a fixed lower panel leaves its leader alone
  and a fixed leader keeps its x frame;
* rearm_autorange: x on leaders, off on followers (switched off first, so
  a pending follower update cannot reach a re-armed leader), y on all; the
  plots of layout widgets included, colour bars left alone; designed frames
  restored axis by axis, the other axes armed;
* the lattice strip never drives its data plot (a lattice edit used to
  switch the plot's x auto-range off and clip the start of the lattice),
  frames a plot without data on the whole lattice (also after a new
  lattice is loaded, and with a cursor line in the plot), and scales to
  metres for plots in m;
* the Results tab, through its real refresh paths: opening a popup, a new
  run and the selectors above a plot re-fit; clicking an open popup's card
  and a live match preview keep the zoom, the end of the preview re-fits;
  a lattice edit leaves the data plot fitting; the transmission panel
  keeps its 0-100 % frame; SCC, footprint and the Partran comparison re-fit
  their new results;
* the other long-lived plots re-fit for a new result and keep a zoom across
  live updates (parameter scan, failure study, study panels, reliability,
  train summary);
* end to end through the real window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

_REPO = Path(__file__).resolve().parents[2]


def _ps():
    # plot_style first: it patches pyqtgraph's View All / "A" button slots,
    # which PyQt binds when each plot is built.
    from linac_gen_gui.interphase.plots import plot_style
    return plot_style


@pytest.fixture(autouse=True)
def _close_new_windows(qapp):
    """Close the top-level widgets a test leaves behind."""
    before = set(map(id, qapp.topLevelWidgets()))
    yield
    for w in qapp.topLevelWidgets():
        if id(w) not in before:
            try:
                w.close()
                w.deleteLater()
            except RuntimeError:
                pass
    qapp.processEvents()


def _vb(p):
    return p.getPlotItem().getViewBox()


def _auto(p):
    a = _vb(p).autoRangeEnabled()
    return (bool(a[0]), bool(a[1]))


def _fit(p):
    _vb(p).updateAutoRange()
    return _vb(p).viewRange()


def _show(qapp, *widgets, size=(500, 240)):
    for w in widgets:
        w.resize(*size)
        w.show()
    qapp.processEvents()


def _plot(qapp, x, y):
    _ps()
    import pyqtgraph as pg
    p = pg.PlotWidget()
    c = p.plot(np.asarray(x, float), np.asarray(y, float))
    _show(qapp, p)
    return p, c


def _pair(qapp):
    """A top plot and a lower panel that follows its x, in one container."""
    _ps()
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QVBoxLayout, QWidget
    box = QWidget()
    lay = QVBoxLayout(box)
    top, low = pg.PlotWidget(), pg.PlotWidget()
    top.plot(np.arange(11.0), np.arange(11.0))
    low.plot(np.arange(11.0), 5.0 * np.arange(11.0))
    low.setXLink(top)
    lay.addWidget(top)
    lay.addWidget(low)
    _show(qapp, box, size=(500, 480))
    # Linking a view to a leader that already holds data switches the
    # leader's x auto-range off (pyqtgraph syncs the follower and forwards
    # that back); the app re-arms before drawing, as here.
    _ps().rearm_autorange(box)
    assert _auto(top) == (True, True) and _auto(low) == (False, True)
    return box, top, low


# ---------------------------------------------------------------- View All
def test_view_all_fits_and_keeps_fitting(qapp):
    p, c = _plot(qapp, np.arange(11.0), np.arange(11.0))
    _vb(p).setRange(xRange=(2, 3), yRange=(2, 3))            # a zoom
    assert _auto(p) == (False, False)
    _vb(p).menu.viewAll.trigger()
    assert _auto(p) == (True, True)                          # pyqtgraph: (False, False)
    c.setData(np.arange(101.0), 10.0 * np.arange(101.0))     # the next run, ×10
    (x0, x1), (y0, y1) = _fit(p)
    assert x0 <= 0.0 and x1 >= 100.0 and y0 <= 0.0 and y1 >= 1000.0


def _trigger(low, how):
    menu = _vb(low).menu
    if how == "view_all":
        menu.viewAll.trigger()
    elif how == "a_button":
        btn = low.getPlotItem().autoBtn
        btn.clicked.emit(btn)
    elif how == "x_auto":
        menu.ctrl[0].autoRadio.click()
    else:                                                    # "Auto %" spin
        spin = menu.ctrl[0].autoPercentSpin
        spin.setValue(90 if spin.value() != 90 else 80)


@pytest.mark.parametrize("how", ["view_all", "a_button", "x_auto", "x_auto_pct"])
def test_a_lower_panel_refits_through_its_leader(qapp, how):
    _box, top, low = _pair(qapp)
    _vb(top).setXRange(2.0, 3.0)
    _vb(low).setYRange(0.0, 1.0)
    assert not _auto(top)[0]
    _trigger(low, how)
    assert _auto(top)[0]                       # the leader's x fits again
    assert not _auto(low)[0]                   # a follower never fits x itself
    if how in ("view_all", "a_button"):
        assert _auto(low)[1]
    assert _vb(low).linkedView(0) is _vb(top)
    (x0, x1), _ = _fit(top)
    assert x0 <= 0.0 and x1 >= 10.0


def test_designed_frames_fit_once_and_come_back(qapp):
    ps = _ps()
    p, c = _plot(qapp, np.arange(11.0), 3.0 * np.arange(11.0))
    ps.fix_plot_range(p, yrange=(0.0, 1.05))
    y0, y1 = _vb(p).viewRange()[1]
    assert y0 <= 0.0 and 1.05 <= y1 < 1.2
    _vb(p).setXRange(2.0, 3.0)                 # a zoom on the undesigned x
    _vb(p).menu.viewAll.trigger()
    assert _auto(p) == (True, False)           # y fitted once, x keeps fitting
    assert _vb(p).viewRange()[1][1] >= 30.0
    _vb(p).setXRange(2.0, 3.0)
    ps.rearm_autorange(p)                      # the next new plot
    y0, y1 = _vb(p).viewRange()[1]
    assert y0 <= 0.0 and 1.05 <= y1 < 1.2      # the y frame is back
    assert _auto(p)[0]                         # and x fits its data again
    c.setData(np.arange(-7.0, -3.0), np.full(4, 0.5))   # e.g. a pressure scan
    (x0, x1), _ = _fit(p)
    assert x0 <= -7.0 and x1 >= -4.0


def test_a_fixed_view_without_a_recorded_frame(qapp):
    """Plots that set their frame at every redraw (tune depression, halo
    map, ORM trims): View All fits once, re-arming arms both axes (the
    redraw then sets the frame again)."""
    ps = _ps()
    p, _c = _plot(qapp, np.arange(11.0), np.arange(11.0))
    ps.fix_plot_range(p)
    _vb(p).setRange(xRange=(2, 3), yRange=(2, 3))
    _vb(p).menu.viewAll.trigger()
    assert _auto(p) == (False, False)
    (x0, x1), (y0, y1) = _vb(p).viewRange()
    assert x0 <= 0.0 and x1 >= 10.0 and y1 >= 10.0
    ps.rearm_autorange(p)
    assert _auto(p) == (True, True)


def test_a_fixed_lower_panel_fits_y_only(qapp):
    ps = _ps()
    _box, top, low = _pair(qapp)
    ps.fix_plot_range(low, yrange=(0.0, 102.0))
    x_before = list(_vb(top).viewRange()[0])
    assert _auto(top)[0]
    _vb(low).menu.viewAll.trigger()
    assert _auto(top)[0] and list(_vb(top).viewRange()[0]) == x_before
    y0, y1 = _vb(low).viewRange()[1]
    assert 50.0 <= y1 < 102.0                  # its data (0…50), once
    _vb(top).setXRange(2.0, 3.0)
    _vb(low).menu.ctrl[0].autoRadio.click()    # x "Auto" on it: leader untouched
    assert not _auto(top)[0]


def test_a_fixed_leader_keeps_its_x_frame(qapp):
    ps = _ps()
    _box, top, low = _pair(qapp)
    ps.fix_plot_range(top, xrange=(3.0, 4.0))
    _vb(low).menu.viewAll.trigger()
    assert not _auto(top)[0]
    x0, x1 = _vb(top).viewRange()[0]
    assert 2.9 <= x0 and x1 <= 4.1
    assert _auto(low)[1]


# ---------------------------------------------------------------- re-arming
def test_rearm_arms_leaders_and_disarms_followers(qapp):
    ps = _ps()
    box, top, low = _pair(qapp)
    _vb(top).setXRange(2.0, 3.0)
    _vb(low).setYRange(0.0, 1.0)
    # a follower left fitting x (pyqtgraph's own "A" or a plot transform),
    # with its update still pending
    _vb(low).enableAutoRange(x=True)
    assert _auto(low)[0]
    ps.rearm_autorange(box)
    assert _auto(top) == (True, True)
    assert _auto(low) == (False, True)
    qapp.processEvents()
    assert _auto(top)[0]                       # the follower did not drive it off


def test_rearm_reaches_layout_plots_but_not_colour_bars(qapp):
    ps = _ps()
    import pyqtgraph as pg
    rng = np.random.default_rng(0)
    panel = ps.DensityPanel(title="x-x'")
    panel.set_data(rng.normal(size=2000), rng.normal(size=2000))
    glw = pg.GraphicsLayoutWidget()            # a colour bar as its own layout item
    plot = glw.addPlot(row=0, col=0)
    plot.plot(np.arange(5.0), np.arange(5.0))
    cbar = pg.ColorBarItem(values=(0.0, 1.0), interactive=False)
    glw.addItem(cbar, row=0, col=1)
    _show(qapp, panel, glw)
    vb = panel._plot.getViewBox()
    vb.setRange(xRange=(0.0, 0.1), yRange=(0.0, 0.1))
    plot.getViewBox().setRange(xRange=(0.0, 0.1), yRange=(0.0, 0.1))
    cbar_before = tuple(cbar.getViewBox().autoRangeEnabled())
    assert cbar.getViewBox() not in ps._views_under(glw)
    ps.rearm_autorange(panel)
    ps.rearm_autorange(glw)
    assert tuple(bool(a) for a in vb.autoRangeEnabled()) == (True, True)
    assert tuple(bool(a) for a in plot.getViewBox().autoRangeEnabled()) == (True, True)
    assert tuple(cbar.getViewBox().autoRangeEnabled()) == cbar_before


# ---------------------------------------------------------------- lattice strip
def _lattice(n=5, length=1000.0):
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    lat = Lattice()
    for i in range(n):
        lat.add(Drift(name=f"D{i}", length=length / n))
    return lat


def test_a_lattice_edit_never_clips_the_data_plot(qapp):
    _ps()
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.lattice_strip import make_lattice_strip
    host = pg.PlotWidget()
    host.plot(np.linspace(0.0, 1000.0, 11), np.arange(11.0))
    strip = make_lattice_strip(None, host, _lattice(length=1000.0))
    _show(qapp, host, strip)
    _ps().rearm_autorange(host)
    strip.set_lattice(_lattice(length=5000.0))   # state.lattice_changed
    qapp.processEvents()
    assert _auto(host)[0]                        # before: off, and clipped at the left
    (x0, x1), _ = _fit(host)
    assert x0 <= 0.0 and x1 >= 1000.0


@pytest.mark.parametrize("cursor", [None, 0.0, 1200.0])
def test_the_strip_frames_a_plot_without_data_on_the_lattice(qapp, cursor):
    _ps()
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.lattice_strip import make_lattice_strip
    host = pg.PlotWidget()
    if cursor is not None:                       # e.g. EnvelopeTriple's s-cursor
        host.addItem(pg.InfiniteLine(pos=cursor, angle=90))
    strip = make_lattice_strip(None, host, _lattice(length=5000.0))
    _show(qapp, host, strip)
    x0, x1 = _vb(host).viewRange()[0]
    assert x0 == pytest.approx(0.0, abs=1e-9) and x1 == pytest.approx(5000.0)
    _ps().rearm_autorange(host)                  # the data arrives: a new plot
    host.plot(np.linspace(0.0, 800.0, 9), np.ones(9))
    (x0, x1), _ = _fit(host)
    # the data (and a cursor, which pyqtgraph counts) set the frame, not
    # the 5000 mm lattice
    assert 800.0 <= x1 < max(1000.0, (cursor or 0.0) * 1.2)


def test_the_strip_scales_to_metres(qapp):
    _ps()
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QGraphicsRectItem
    from linac_gen_gui.interphase.plots.lattice_strip import make_lattice_strip
    host = pg.PlotWidget()
    strip = make_lattice_strip(None, host, _lattice(n=5, length=5000.0),
                               scale=1e-3)
    rects = [it for it in strip._items if isinstance(it, QGraphicsRectItem)]
    assert rects[1].rect().x() == pytest.approx(1.0)        # 1000 mm → 1 m
    assert rects[1].rect().width() == pytest.approx(1.0)
    assert _vb(host).viewRange()[0][1] == pytest.approx(5.0)
    assert "1000.0–2000.0 mm" in rects[1].toolTip()          # tooltips stay in mm


# ---------------------------------------------------------------- Results tab
def _noop(*_a, **_k):
    return None


@pytest.fixture
def results_tab(qapp, gui_message_boxes):
    _ps()
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen.cli import common
    from linac_gen_gui.interphase.state import AppState
    lat, cfg, _conv = common.load_input(str(_REPO / "examples" / "fodo_cell.dat"))
    res = common.run_envelope_sim(lat, cfg)
    st = AppState()
    st.set_lattice(lat)
    st.set_beam_config(cfg)
    tab = rt.ResultsTab(st, _noop, _noop, _noop)
    st.set_results(res)
    yield tab, st, res, lat
    for d in tab._popups.values():
        d.close()
    tab.deleteLater()
    qapp.processEvents()


def _open(tab, key):
    tab._open_popup(key)
    return tab._popups[key]


def test_a_new_run_refits_an_open_popup(qapp, results_tab):
    tab, st, res, _lat = results_tab
    dlg = _open(tab, "emit")
    assert _auto(dlg._px) == (True, True)
    _vb(dlg._px).setRange(xRange=(100.0, 200.0), yRange=(0.0, 1e-3))
    st.set_results(res)                          # a new run
    assert _auto(dlg._px) == (True, True)
    (x0, x1), _ = _fit(dlg._px)
    assert x0 <= 0.0 and x1 >= 1600.0


def test_reopening_refits_and_a_card_click_keeps_the_zoom(qapp, results_tab):
    tab, _st, _res, _lat = results_tab
    dlg = _open(tab, "emit")
    _vb(dlg._px).setXRange(100.0, 200.0)
    _open(tab, "emit")                           # its card clicked again, still open
    assert not _auto(dlg._px)[0]
    dlg.close()
    _vb(dlg._px).setXRange(100.0, 200.0)
    _open(tab, "emit")                           # reopened: a new plot
    assert _auto(dlg._px) == (True, True)


def test_a_live_preview_keeps_the_zoom_and_its_end_refits(qapp, results_tab):
    tab, _st, res, _lat = results_tab
    dlg = _open(tab, "emit")
    dlg._live_cb.setChecked(True)
    _vb(dlg._px).setXRange(100.0, 200.0)
    tab.preview_refresh(res, 3)
    assert not _auto(dlg._px)[0]                 # live iterates keep the zoom
    tab.end_preview()
    assert _auto(dlg._px) == (True, True)        # the committed result: a new plot


def test_a_lattice_edit_leaves_open_popups_fitting(qapp, results_tab):
    tab, st, _res, lat = results_tab
    dlg = _open(tab, "emit")
    qapp.processEvents()
    st.lattice_changed.emit(lat)                 # every edit, undo, match apply…
    qapp.processEvents()
    assert _auto(dlg._px)[0]
    (x0, x1), _ = _fit(dlg._px)
    assert x0 <= 0.0 and x1 >= 1600.0


def test_a_new_lattice_frames_open_popups_on_it(qapp, results_tab):
    tab, st, _res, _lat = results_tab
    dlg = _open(tab, "emit")
    qapp.processEvents()
    st.set_lattice(_lattice(n=10, length=10_000.0))   # results are cleared too
    qapp.processEvents()
    x0, x1 = _vb(dlg._px).viewRange()[0]
    assert x0 == pytest.approx(0.0, abs=1e-6) and x1 == pytest.approx(10_000.0)


def test_an_empty_popup_with_a_cursor_shows_the_lattice(qapp, gui_message_boxes):
    _ps()
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen_gui.interphase.state import AppState
    st = AppState()
    st.set_lattice(_lattice(n=5, length=5000.0))
    tab = rt.ResultsTab(st, _noop, _noop, _noop)
    try:
        dlg = _open(tab, "rms")                  # EnvelopeTriple: cursors at s = 0
        qapp.processEvents()
        x0, x1 = _vb(dlg._triple._p_x).viewRange()[0]
        assert x0 == pytest.approx(0.0, abs=1e-6) and x1 == pytest.approx(5000.0)
    finally:
        for d in tab._popups.values():
            d.close()
        tab.deleteLater()


def test_the_lattice_parameter_popup_leaves_auto_range_on(qapp, results_tab):
    tab, _st, _res, _lat = results_tab
    dlg = _open(tab, "quad_grad")
    assert _auto(dlg._plot) == (True, True)      # before: a one-time fit, then off


def test_the_transmission_panel_keeps_its_frame(qapp, results_tab):
    tab, st, res, _lat = results_tab
    dlg = _open(tab, "energy")
    # 0 … 102 % with pyqtgraph's default padding (which depends on the
    # plot's pixel height, as before this change)
    y0, y1 = _vb(dlg._pt).viewRange()[1]
    assert y0 <= 0.0 and 102.0 <= y1 < 120.0
    _vb(dlg._pt).menu.viewAll.trigger()          # fitted once (y only: it follows W)
    assert _auto(dlg._pw)[0] and _auto(dlg._pt) == (False, False)
    _vb(dlg._pt).setYRange(40.0, 60.0)           # a zoom on the panel
    st.set_results(res)                          # a new run: the frame is back
    y0, y1 = _vb(dlg._pt).viewRange()[1]
    assert y0 <= 0.0 and 102.0 <= y1 < 120.0
    assert _auto(dlg._pw) == (True, True)


def test_metre_plots_get_a_metre_strip(qapp, results_tab):
    tab, _st, _res, _lat = results_tab
    for key in ("ibs", "magstrip"):
        dlg = _open(tab, key)
        assert dlg._lattice_strip._scale == pytest.approx(1e-3)
        x0, x1 = _vb(dlg._p_rate).viewRange()[0]
        assert x1 < 10.0                         # 1.6 m, not 1600 "m"


def test_a_selector_refits_its_popup(qapp, results_tab):
    tab, _st, _res, _lat = results_tab
    dlg = _open(tab, "ibs")
    _vb(dlg._p_rate).setXRange(0.1, 0.2)
    dlg._duty_spin.setValue(dlg._duty_spin.value() / 2.0 or 1.0)   # another input
    assert _auto(dlg._p_rate)[0]


def test_scc_footprint_and_comparison_refit_their_new_results(qapp, results_tab):
    tab, _st, _res, _lat = results_tab
    scc = _open(tab, "scc")
    for p in (scc._plot_phi, scc._plot_tau, scc._plot_loss, scc._plot_fc):
        _vb(p).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
    rows = [{"pressure_mbar": p, "mean_fc": 0.5, "phi_min_V": -10.0,
             "tau_scc_us": 20.0, "transmission_gas_pct": 99.0}
            for p in (1e-7, 1e-6, 1e-5, 1e-4)]
    scc._render_scan(rows)                       # x: log10 p ∈ [-7, -4]
    for p in (scc._plot_phi, scc._plot_tau, scc._plot_loss):
        assert _auto(p) == (True, True)
    assert _auto(scc._plot_fc)[0]                # f_c: x fits, y keeps 0 … 1.05
    (x0, x1), (y0, y1) = _fit(scc._plot_fc)
    assert x0 <= -7.0 and x1 >= -4.0 and y0 <= 0.0 and y1 >= 1.05

    fp = _open(tab, "footprint")
    _vb(fp._plot).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
    fp._done({"qx": [0.20, 0.21], "qy": [0.25, 0.26], "ax_sigma": [0.0, 1.0],
              "ay_sigma": [0.0, 1.0], "mu_x_core_deg": 72.0,
              "mu_y_core_deg": 90.0, "model": "test"})
    assert _auto(fp._plot) == (True, True)

    pt = _open(tab, "partran")
    _vb(pt._p_abs).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
    pt._axis.setCurrentIndex((pt._axis.currentIndex() + 1) % pt._axis.count())
    assert _auto(pt._p_abs) == (True, True)
    _vb(pt._p_abs).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
    pt._on_clear()
    assert _auto(pt._p_abs) == (True, True)


# ---------------------------------------------------------------- other plots
class _StubSignal:
    def connect(self, *_a, **_k):
        pass


class _StubWorker:
    def __init__(self, *_a, **_k):
        for name in ("progress", "point_done", "finished_ok", "failed", "finished"):
            setattr(self, name, _StubSignal())

    def start(self):
        pass

    def isRunning(self):
        return False


def test_a_parameter_scan_refits_at_its_start_and_keeps_a_zoom(
        qapp, mini_lattice, monkeypatch, gui_message_boxes):
    _ps()
    import linac_gen_gui.interphase.dialogs.parameter_scan as psd
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.state import AppState
    monkeypatch.setattr(psd, "_ScanWorker", _StubWorker)
    st = AppState()
    st.set_lattice(mini_lattice)
    st.set_beam_config(BeamConfig(species="H-", energy=2.1, frequency=162.5,
                                  n_particles=10))
    dlg = psd.ParameterScanDialog(None, st)
    try:
        _show(qapp, dlg)
        dlg._element_combo.setCurrentIndex(1)    # Q1
        dlg._min_spin.setValue(1.0)
        dlg._max_spin.setValue(2.0)
        _vb(dlg._plot).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
        dlg._run_scan()                          # the real start of a scan
        assert gui_message_boxes == []
        assert _auto(dlg._plot) == (True, True)
        dlg._on_point_done(0, 1.0, 2.0)
        dlg._on_point_done(1, 2.0, 2.0)          # a flat line so far
        (_x, (y0, y1)) = _fit(dlg._plot)
        assert y0 < 2.0 < y1                     # still visible
        _vb(dlg._plot).setXRange(1.0, 1.5)       # the user zooms during the scan
        dlg._on_point_done(2, 3.0, 5.0)
        assert not _auto(dlg._plot)[0]           # the zoom is kept
    finally:
        dlg.close()
        dlg.deleteLater()


def test_a_new_failure_study_refits_its_plots(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.failure_study_tab import FailureStudyTab
    tab = FailureStudyTab(AppState())
    try:
        _vb(tab._bar).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
        tab._run_combo = "singles"
        tab._on_prepared(["A", "B"], 2)
        assert _auto(tab._bar) == (True, True)
        assert _auto(tab._heat) == (True, True)
    finally:
        tab.deleteLater()


def test_study_panels_refit_for_a_new_choice_not_for_a_new_run(qapp, monkeypatch):
    _ps()
    from linac_gen_gui.interphase.panels.study_plots import StudyAnalysisPanel
    panel = StudyAnalysisPanel()
    try:
        monkeypatch.setattr(panel.model, "load", lambda *_a, **_k: None)
        panel.load_study("unused")               # the study is opened first
        for sub, poke in ((panel.p1d, lambda s: s.logx.toggle()),
                          (panel.p2d, lambda s: s.z.currentTextChanged.emit("q")),
                          (panel.overlay, lambda s: s.qty.setCurrentIndex(
                              (s.qty.currentIndex() + 1) % s.qty.count()))):
            _vb(sub.plot).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
            panel.load_study("unused", new_plot=False)   # a finished study run
            assert not _auto(sub.plot)[0]                # keeps the zoom
            poke(sub)                                     # the user picks again
            assert _auto(sub.plot) == (True, True)
            _vb(sub.plot).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
            panel.load_study("unused")                    # a study opened
            assert _auto(sub.plot) == (True, True)
    finally:
        panel.deleteLater()


def test_reliability_bars_refit_for_new_results(qapp):
    _ps()
    import pyqtgraph as pg
    from linac_gen_gui.interphase.dialogs import reliability_dialog as rd
    owner = type("_Owner", (), {})()
    owner._bar = pg.PlotWidget()
    _vb(owner._bar).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
    faults = [{"criticality": c, "bracket": "rephased", "case_id": f"c{i}",
               "elements": f"E{i}"} for i, c in enumerate((3.0, 1.0, 2.0))]
    rd.ReliabilityStudyDialog._render_bar(owner, faults)
    assert _auto(owner._bar) == (True, True)
    _x, (y0, y1) = _vb(owner._bar).viewRange()
    assert y0 <= 0.0 and y1 >= 3.0


def test_the_train_summary_refits_for_new_results(qapp):
    _ps()
    from linac_gen_gui.interphase.dialogs.train_dialog import TrainSummaryDialog
    dlg = TrainSummaryDialog()
    try:
        _vb(dlg._plot).setRange(xRange=(5.0, 6.0), yRange=(5.0, 6.0))
        dlg.set_results(None)
        assert _auto(dlg._plot) == (True, True)
    finally:
        dlg.deleteLater()


# ---------------------------------------------------------------- end to end
def test_end_to_end_through_the_window(win, mini_lattice):
    """The real window: two envelope runs through the run slot, a popup
    opened as its card opens it, a zoom, a lattice edit, View All on a
    lower panel."""
    win.state.set_lattice(mini_lattice, None)
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    tab = win.results_tab
    tab._open_popup("emit")
    dlg = tab._popups["emit"]
    win.pump(0.2)
    assert _auto(dlg._px) == (True, True)
    _vb(dlg._px).setRange(xRange=(100.0, 150.0), yRange=(0.0, 1e-6))   # a zoom
    win._run_envelope()                                                # a new run
    win.wait_worker(win._envelope_worker)
    win.pump(0.2)
    assert _auto(dlg._px) == (True, True)
    (x0, x1), _ = _fit(dlg._px)
    assert x0 <= 0.0 and x1 >= 500.0                                   # the whole 500 mm
    win.state.lattice_changed.emit(win.state.lattice)                  # an edit
    win.pump(0.2)
    assert _auto(dlg._px)[0]
    (x0, x1), _ = _fit(dlg._px)
    assert x0 <= 0.0 and x1 >= 500.0
    _vb(dlg._px).setXRange(100.0, 150.0)
    _vb(dlg._py).menu.viewAll.trigger()                                # lower panel
    assert _auto(dlg._px)[0] and not _auto(dlg._py)[0]
    assert _vb(dlg._py).linkedView(0) is _vb(dlg._px)
