"""A ring run in the Results tab: the selector "Run shown: Linac run | Ring
run", the tiles that show the beam turn by turn (x = turn, no lattice strip,
nothing of a linac run left), the tiles disabled for a ring run (with their
reason, fed nothing), the RING RUN tiles, switching back to the linac run
(axes, strips and boxes as they were), the bend tiles' rigidity, Import of a
ring results file, the assistant's ``open_plot``, the KPIs; the Ring tab's
live strip (the wheel scrolls the page) and its turn label in a ramp."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
FODO = ROOT / "examples" / "fodo_cell.dat"


@pytest.fixture(scope="module")
def ring_run():
    """A small ring run of the simple Booster (RF on): 300 particles, 20 turns,
    snapshots at 0, 10 and 20, one tune probe."""
    from linac_gen.cli.common import load_lattice
    from linac_gen.core.config import BeamConfig
    from linac_gen.ring import RingConfig, RingSimulation
    lat = load_lattice(str(FIXTURE))
    b = BeamConfig(species="proton", energy=800.0, frequency=44.7044096, current=0.0,
                   emit_nx=1.5, emit_ny=1.5, emit_z=5.0, beta_z=100.0, cutoff=4.0)
    return RingSimulation(lat, b, RingConfig(n_turns=20, n_particles=300, source="matched",
                                             tune_probes=1,
                                             snapshot_turns=[0, 10, 20])).run()


def _plot_items(dlg):
    import pyqtgraph as pg
    items = [pw.getPlotItem() for pw in dlg.findChildren(pg.PlotWidget)]
    for glw in dlg.findChildren(pg.GraphicsLayoutWidget):
        items += [it for it in glw.ci.items if isinstance(it, pg.PlotItem)]
    return items


def _plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _strips(dlg):
    from linac_gen_gui.interphase.plots.lattice_strip import LatticeStripWidget
    return dlg.findChildren(LatticeStripWidget)


def _idle(win, dlg):
    """Wait for a popup's workers to end.  (Not ``wait_popup_idle``: a
    dispersion popup keeps its ``_pending_key`` after its model is drawn, so
    that would wait its whole timeout.)"""
    assert win.pump_until(lambda: not dlg._live_workers(), 120_000)
    win.pump(0.3)


def _show_ring(win, res):
    """Ring results set, the ring run shown (on its own only in ring mode)."""
    win.state.set_ring_results(copy.deepcopy(res))
    win.results_tab.show_ring_run()
    win.pump(0.2)
    return win.results_tab


def _no_refresh_failure(win):
    """No popup's refresh raised (the Results tab shows the error in a message
    box — stubbed in the tests — and opens the popup anyway)."""
    bad = [b for b in win.message_boxes if "refresh failed" in str(b)]
    assert not bad, bad


def _linac_run(win):
    win.open_lattice(FODO)
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    assert win.state.results is not None


# ------------------------------------------------------------------ the tiles
def test_every_turn_tile_opens_on_a_turn_axis(win, ring_run):
    from linac_gen_gui.interphase.tabs.results_tab import _TURN_TILES, _is_s_title
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    assert rt._source == "ring" and not rt._run_combo.isHidden()
    assert not rt._ring_banner.isHidden() and "20 turn(s)" in rt._ring_banner.text()
    turns = np.asarray(ring_run.turns, dtype=float)
    for key in sorted(_TURN_TILES):
        card = rt._card_of(key)
        assert card.isEnabled(), key
        assert rt.open_plot(key), key
        dlg = rt._popups[key]
        win.wait_popup_idle(dlg)
        assert dlg._x_domain == "turn", key
        assert "ring run, turn by turn" in dlg.windowTitle(), key
        # no lattice strip, also after its box is clicked
        assert all(s.isHidden() for s in _strips(dlg)), key
        live = getattr(dlg, "_live_cb", None)
        assert live is None or live.isHidden(), key
        for name in ("_chk_lat", "_chk_ap"):
            box = getattr(dlg, name, None)
            if box is not None:
                assert box.isHidden() and not box.isEnabled(), (key, name)
                dlg.set_lattice_strip_visible(True)
                assert all(s.isHidden() for s in _strips(dlg)), key
        import pyqtgraph as pg
        from linac_gen_gui.interphase.plots.lattice_strip import LatticeStripWidget
        # (the s cursors; the hidden lattice strip's element markers aside)
        cursors = [it for pw in dlg.findChildren(pg.PlotWidget)
                   if not isinstance(pw, LatticeStripWidget)
                   for it in pw.getPlotItem().getViewBox().addedItems
                   if isinstance(it, pg.InfiniteLine) and it.angle == 90]
        if key == "rms":
            assert len(cursors) == 3 and not any(c.isVisible() for c in cursors)
        # every s axis says "turn"; every curve is turn by turn
        n_turn_axes, n_turn_curves = 0, 0
        import pyqtgraph as pg
        for pi in _plot_items(dlg):
            text = _plain(pi.getAxis("bottom").labelText)
            assert not _is_s_title(text), (key, text)
            if text == "turn":
                n_turn_axes += 1
                for it in pi.getViewBox().addedItems:
                    if isinstance(it, pg.PlotDataItem):
                        # (the whole curve: what is drawn is thinned and clipped)
                        x = it.xData
                        if x is not None and len(x):
                            assert x.min() >= -0.5 and x.max() <= 20.5, key
                            n_turn_curves += np.array_equal(x, turns)
                            assert it.opts["autoDownsample"] and it.opts["clipToView"], key
                            solid = pg.mkPen(it.opts.get("pen")).style() == Qt.PenStyle.SolidLine
                            assert (it.curve.opts["segmentedLineMode"] == "on") == solid, key
        if key != "sigma":
            assert n_turn_axes >= 1 and n_turn_curves >= 1, key
    _no_refresh_failure(win)
    # the Σ window: x = the turns themselves (not turns / 1000), its export too
    t = rt._popups["sigma"]._terms()
    assert t.domain == "turn" and np.array_equal(t.s, turns)


def test_disabled_tiles_carry_their_reason_and_get_nothing(win, ring_run):
    from linac_gen_gui.interphase.tabs.results_tab import _RING_OFF
    _linac_run(win)
    rt = win.results_tab
    peak = rt._card_of("peak")
    assert peak.isEnabled() and peak._value_lbl.text() != "—"
    rt.open_plot("peak")
    assert rt._popups["peak"].isVisible()
    _show_ring(win, ring_run)
    for key, why in _RING_OFF.items():
        card = rt._card_of(key)
        if card is None:
            continue
        assert not card.isEnabled(), key
        assert card.toolTip() == f"Not for a ring run.  {why}", key
        assert card._value_lbl.text() == "—", key
        assert card.graphicsEffect() is not None, key
    # (a popup of a tile disabled now is closed)
    assert not rt._popups["peak"].isVisible()
    assert not rt.open_plot("peak")
    assert rt._hint.text().endswith("turn by turn for the ring run shown.")
    # back to the linac run: the tiles as they were
    rt._run_combo.setCurrentIndex(0)
    assert rt._source == "linac"
    assert rt._hint.text().endswith("the underlying quantity along s.")
    assert peak.isEnabled() and peak.graphicsEffect() is None
    assert "Not for a ring run" not in peak.toolTip()
    assert peak._value_lbl.text() != "—"
    # the IBS tile follows the beam in the linac view only
    rt._run_combo.setCurrentIndex(1)
    win.state.beam_config_changed.emit(win.state.beam_config)
    assert not rt._card_of("ibs").isEnabled()
    assert rt._card_of("ibs").toolTip().startswith("Not for a ring run.")


def test_ring_and_back_restores_axes_strips_and_boxes(win, ring_run):
    _linac_run(win)
    rt = win.results_tab
    keys = ("rms", "emit", "long_twiss", "divergence", "dispersion", "sigma", "centroid")

    def drawing(dlg):
        # (how each curve is drawn: a turn axis thins it, s as before)
        boxes = [getattr(dlg, n, None) for n in ("_chk_lat", "_chk_ap", "_live_cb")]
        return [(it.opts["autoDownsample"], it.opts["downsample"], it.opts["clipToView"],
                 it.curve.opts["segmentedLineMode"])
                for pi in _plot_items(dlg) for it in pi.listDataItems()] + [
            (b.isEnabled(), b.isHidden(), b.isChecked()) for b in boxes if b is not None]
    before = {}
    for key in keys:
        assert rt.open_plot(key)
        dlg = rt._popups[key]
        _idle(win, dlg)
        before[key] = ([(pi.getAxis("bottom").labelText, pi.getAxis("bottom").labelUnits)
                        for pi in _plot_items(dlg)],
                       [s.isHidden() for s in _strips(dlg)], dlg.windowTitle(), drawing(dlg))
    disp = rt._popups["dispersion"]
    disp._chk_model.setChecked(True)
    _idle(win, disp)
    model = disp._cx_m.getData()[0]
    assert model is not None and len(model) > 0
    def legend(plot):
        return [label.text for _sample, label in plot.getPlotItem().legend.items]
    legends = (legend(disp._px), legend(disp._py))
    assert any("model" in t for t in legends[0])
    _show_ring(win, ring_run)
    assert not disp._chk_model.isChecked() and not disp._chk_model.isEnabled()
    assert disp._chk_model.isHidden()
    assert not any("model" in t for t in legend(disp._px) + legend(disp._py))
    rt._run_combo.setCurrentIndex(0)
    for key in keys:
        dlg = rt._popups[key]
        _idle(win, dlg)
        now = ([(pi.getAxis("bottom").labelText, pi.getAxis("bottom").labelUnits)
                for pi in _plot_items(dlg)],
               [s.isHidden() for s in _strips(dlg)], dlg.windowTitle(), drawing(dlg))
        assert now == before[key], key
        assert dlg._x_domain == "s", key
    assert disp._chk_model.isChecked() and disp._chk_model.isEnabled()
    assert not disp._chk_model.isHidden()
    assert (legend(disp._px), legend(disp._py)) == legends
    assert np.array_equal(disp._cx_m.getData()[0], model)
    # (the linac run's own data again)
    s = np.asarray(win.state.results.s, dtype=float)
    x, _y = rt._popups["emit"]._cx.getData()
    assert np.array_equal(x, s)


def test_a_lattice_edit_keeps_the_turn_axis(win, ring_run):
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    assert rt.open_plot("rms")
    dlg = rt._popups["rms"]
    # (an edit emits lattice_changed with the ring run still shown)
    win.state.lattice_changed.emit(win.state.lattice)
    win.pump(0.2)
    assert rt._source == "ring" and all(s.isHidden() for s in _strips(dlg))


def test_ring_tiles_show_the_run(win, ring_run):
    from linac_gen_gui.interphase.tabs.ring_popups import POPUPS
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    res = win.state.ring_results
    want = {"ring_ramp": False, "ring_long": True, "ring_losses": False, "ring_tunes": True,
            "ring_phase": True, "ring_optics": True, "ring_injection": False}
    assert set(want) == set(POPUPS)
    for key, ok in want.items():
        card = rt._card_of(key)
        assert card.isEnabled() == ok, key
        assert rt.open_plot(key) == ok, key
    _no_refresh_failure(win)
    assert "No ramp" in rt._card_of("ring_ramp").toolTip()
    assert "No particle was lost" in rt._card_of("ring_losses").toolTip()
    assert "No foil" in rt._card_of("ring_injection").toolTip()
    # the bunch in eV s, turn by turn
    b = rt._popups["ring_long"]
    assert np.array_equal(b.c_el.yData, np.asarray(res.emit_long_evs))
    assert np.array_equal(b.c_el.xData, np.asarray(res.turns, dtype=float))
    assert b.p_bins.isHidden()                         # (no space charge)
    # the periodic optics along s, from the run's own optics table
    o = rt._popups["ring_optics"]
    t = res.optics_table
    assert np.array_equal(o.c_bx.xData, t["s"]) and np.array_equal(o.c_bx.yData, t["betx"])
    assert np.array_equal(o.c_dx.yData, t["dx"])
    assert f"{res.optics['qx']:.6f}" in o.text.text()
    assert o._lattice_strip is not None and not o._lattice_strip.isHidden()
    # the tune diagram: framed on its points, with the resonance lines
    tp = rt._popups["ring_tunes"]
    assert len(tp.s_fp.data) == 1 and len(tp._lines) > 0
    (x0, x1), (y0, y1) = tp.p_fp.getPlotItem().getViewBox().viewRange()
    qx, qy = tp.s_fp.data["x"][0], tp.s_fp.data["y"][0]
    assert x0 < qx < x1 and y0 < qy < y1
    # phase space: all the particles of the last snapshot as densities
    ph = rt._popups["ring_phase"]
    assert ph.snap_combo.currentText() == "20"
    img = ph.d_x._image.image
    assert img is not None and img.sum() > 0
    assert ph.c_line.yData is not None and len(ph.c_line.yData) > 0
    # leaving the ring run closes the RING RUN windows; open_plot comes back to it
    rt._run_combo.setCurrentIndex(0)
    assert not ph.isVisible() and rt._source == "linac"
    assert rt.open_plot("ring_phase") and rt._source == "ring"


def test_open_plot_of_a_ring_tile_without_a_ring_run(win):
    win.open_lattice(FIXTURE)
    rt = win.results_tab
    assert win.state.ring_results is None and rt._run_combo.isHidden()
    assert not rt.open_plot("ring_phase")
    assert rt._source == "linac"


def test_tunes_window_marks_an_analytic_shift_within_half_an_integer(qapp):
    from types import SimpleNamespace

    from linac_gen_gui.interphase.tabs.ring_popups import RingTunesPopup
    ring = SimpleNamespace(
        tunes={"qx": 0.69, "qy": 0.79}, optics={"eigen_tunes_6d": [0.71, 0.81, 0.02]},
        probes={"tunes": np.array([[0.69, 0.79, np.nan]])}, ramp=None,
        meta={"config": {"tune_probes": 1},
              "space_charge": {"bunched": True, "laslett_gaussian": [-0.05, -0.04]}})
    d = RingTunesPopup(None)
    d.refresh(SimpleNamespace(is_ring_view=True, ring=ring))
    assert d.s_bare.data["x"][0] == 0.71 and d.s_bare.data["y"][0] == 0.81
    assert d.s_shift.data["x"][0] == pytest.approx(0.66, abs=1e-12)
    assert d.s_shift.data["y"][0] == pytest.approx(0.77, abs=1e-12)
    assert "dQx -0.02000  dQy -0.02000" in d.tunes_text.text()
    d.deleteLater()


def test_resonance_lines_cross_the_frame():
    from linac_gen_gui.interphase.tabs.ring_popups import resonance_lines
    lines = resonance_lines(0.6, 0.8, 0.7, 0.9, 4)
    for xa, ya, xb, yb, k in lines:
        assert 1 <= k <= 4
        for x, y in ((xa, ya), (xb, yb)):
            assert 0.6 - 1e-9 <= x <= 0.8 + 1e-9 and 0.7 - 1e-9 <= y <= 0.9 + 1e-9

    def has(k, f):
        # (the end points are rounded to 12 decimals)
        return any(kk == k and abs(f(xa, ya)) < 1e-9 and abs(f(xb, yb)) < 1e-9
                   for xa, ya, xb, yb, kk in lines)
    assert has(2, lambda x, y: x - y)                  # Qx - Qy = 0
    assert has(3, lambda x, y: 3 * x - 2)              # 3 Qx = 2
    assert has(4, lambda x, y: 2 * x + 2 * y - 3)      # 2 Qx + 2 Qy = 3
    assert has(4, lambda x, y: 4 * x - 3)              # 4 Qx = 3
    # (no integer p: Qx + Qy runs from 1.3 to 1.7 here — no second-order sum line)
    assert not has(2, lambda x, y: x + y - 1.5)
    # each line once, with its lowest order (2 Qx - 2 Qy = 0 is Qx - Qy = 0)
    keyed = [(round(xa, 9), round(ya, 9), round(xb, 9), round(yb, 9))
             for xa, ya, xb, yb, _k in lines]
    assert len(set(keyed)) == len(keyed)


# --------------------------------------------------- KPIs, state, bend tiles
def test_the_kpis_of_a_ring_run_and_no_stale_value(win, ring_run):
    from linac_gen.ring.view import RingResultsView
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    kpi = RingResultsView(win.state.ring_results).kpi

    def text(w):
        from PyQt6.QtWidgets import QLabel
        return w.findChild(QLabel, "value").text()
    assert text(rt._k_sx) == f"{kpi['sigma_x_mm']:.3f}"
    assert text(rt._k_trans) == f"{kpi['transmission']:.2f}"
    assert text(rt._k_ex) == f"{kpi['ex_growth']:.2f}×"
    # (a linac run without a transmission shows "—", not the ring run's value)
    from types import SimpleNamespace
    rt._refresh(SimpleNamespace(s=[0.0, 1.0], sigma_x=[1.0, 2.0], sigma_y=[1.0, 2.0],
                                sigma_phi=[1.0, 1.0], emit_x=[1.0, 1.0]))
    assert text(rt._k_trans) == "—" and text(rt._k_loss) == "—"


def test_a_ring_run_leaves_the_linac_results_alone(win, ring_run):
    _linac_run(win)
    before = win.state.results
    _show_ring(win, ring_run)
    assert win.state.results is before


def test_bend_tiles_use_the_ring_runs_rigidity(win, ring_run):
    from linac_gen_gui.interphase.tabs.results_tab import _brho_for_element

    from linac_gen.elements.dipole import Dipole
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    res = win.state.ring_results
    bend = next(e for e in win.state.lattice.elements if isinstance(e, Dipole))
    view = rt._shown_results()
    brho = _brho_for_element(win.state, bend, rt._shown_results)
    assert brho == pytest.approx(view.ref_bg[0] * res.meta["mass_mev"] / 299.792458, rel=1e-15)
    # (the run at 800 MeV: B rho = p / c)
    m = res.meta["mass_mev"]
    assert brho == pytest.approx(np.sqrt(800.0 ** 2 + 2 * 800.0 * m) / 299.792458, rel=1e-12)
    rt._run_combo.setCurrentIndex(0)
    assert _brho_for_element(win.state, bend, rt._shown_results) == \
        _brho_for_element(win.state, bend)


# ------------------------------------------------------------------ Import
def test_import_of_a_ring_results_file(win, ring_run, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.tabs import results_tab as RT

    from linac_gen.ring.results import save_ring_hdf5
    win.open_lattice(FIXTURE)
    path = tmp_path / "x_ring.h5"
    save_ring_hdf5(ring_run, path)
    monkeypatch.setattr(RT.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    rt = win.results_tab
    # refused while a ring run is going
    rt.ring_running_provider = lambda: True
    win.message_boxes.clear()
    rt._import_results()
    assert win.state.ring_results is None
    assert any("Ring run in progress" in str(b) for b in win.message_boxes)
    rt.ring_running_provider = lambda: False
    rt._import_results()
    res = win.state.ring_results
    assert res is not None and res.meta["dump_path"] == str(path)
    assert rt._source == "ring" and np.array_equal(res.cov, ring_run.cov)
    assert rt._popups == {} or all(not d.isVisible() for d in rt._popups.values())
    assert rt.open_plot("ring_optics")


# ---------------------------------------------------------------- Ring tab
def test_the_live_strip_never_takes_the_wheel(qapp):
    # pyqtgraph's axes accept a wheel event even with the mouse off, so the
    # page it sits on never scrolls (a plain plot, the control); the strip's
    # plots ignore it over the plot and over the axis, and leave their view.
    # (A wheel event sent by a program does not go on to the parent widgets in
    # Qt 6 — a real one does once ignored — so "ignored" is what is checked.)
    import pyqtgraph as pg
    from linac_gen_gui.interphase.panels.ring_plots import RingLiveStrip, _plot
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication

    def wheel(plot, where):
        pi = plot.getPlotItem()
        if where == "axis":
            pos = plot.mapFromScene(pi.mapToScene(pi.getAxis("left").geometry().center()))
        else:
            vb = pi.getViewBox()
            pos = plot.mapFromScene(vb.mapToScene(vb.boundingRect().center()))
        ev = QWheelEvent(QPointF(pos), QPointF(plot.viewport().mapToGlobal(QPoint(pos))),
                         QPoint(0, 0), QPoint(0, -240), Qt.MouseButton.NoButton,
                         Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(plot.viewport(), ev)
        qapp.processEvents()
        return ev.isAccepted()

    control = _plot("y", cls=pg.PlotWidget)
    control.getPlotItem().setMouseEnabled(x=False, y=False)
    control.resize(500, 300)
    control.show()
    qapp.processEvents()
    assert wheel(control, "axis"), "pyqtgraph no longer takes the wheel over an axis"
    control.deleteLater()

    strip = RingLiveStrip()
    strip.resize(1200, strip.HEIGHT_PX)
    strip.show()
    strip.show_live({"start": 0, "turns": np.arange(5), "n_alive": np.full(5, 10),
                     "cov": np.tile(np.eye(6) * 1e-6, (5, 1, 1)), "beta0gamma0": 1.6,
                     "p0c_mev": 1500.0})
    for _ in range(5):
        qapp.processEvents()
    for plot in (strip.p_surv, strip.p_emit, strip.p_long):
        vb = plot.getPlotItem().getViewBox()
        vb.disableAutoRange()
        before = [list(r) for r in vb.viewRange()]
        for where in ("plot", "axis"):
            assert not wheel(plot, where), where
            assert [list(r) for r in vb.viewRange()] == before, where
    strip.deleteLater()


def test_the_turn_label_shows_a_ramps_gamma(win):
    from PyQt6.QtCore import QObject, pyqtSignal

    class Fake(QObject):
        live = pyqtSignal(object)
        progress_turn = pyqtSignal(int, int)

        def isRunning(self):                     # noqa: N802 — a QThread's name
            return False

    win.open_lattice(FIXTURE)
    act = win._toolbar._ring_action
    if not act.isChecked():
        act.trigger()
    win.pump(0.1)
    tab = win.ring_tab
    fake = Fake()
    fake.live.connect(tab._on_live)
    fake.progress_turn.connect(tab._on_progress_turn)
    tab._worker = fake
    fake.progress_turn.emit(5, 100)
    assert tab._turn_label.text() == "turn 5 of 100"
    cov = np.tile(np.eye(6) * 1e-6, (2, 1, 1))
    fake.live.emit({"start": 0, "turns": np.array([0, 4]), "n_alive": np.array([10, 10]),
                    "cov": cov, "beta0gamma0": np.array([1.5, 1.6]),
                    "p0c_mev": np.array([1400.0, 1500.0]), "ramp": True})
    assert tab._turn_label.text() == (f"turn 5 of 100; the reference's gamma "
                                      f"{np.sqrt(1 + 1.6 ** 2):.6f} (turn 4)")
    fake.progress_turn.emit(6, 100)
    assert tab._turn_label.text().startswith("turn 6 of 100; the reference's gamma")
    # (at fixed energy: the turn alone)
    tab._set_running(False)
    fake.live.emit({"start": 2, "turns": np.array([5]), "n_alive": np.array([10]),
                    "cov": cov[:1], "beta0gamma0": 1.6, "p0c_mev": 1500.0, "ramp": False})
    fake.progress_turn.emit(7, 100)
    assert tab._turn_label.text() == "turn 7 of 100"
    tab._worker = None


# --------------------------------------------------------------- headless
def test_sigma_terms_keep_the_linac_path(ring_run):
    from types import SimpleNamespace

    from linac_gen.analysis.sigma_terms import sigma_terms
    from linac_gen.ring.view import RingResultsView
    view = RingResultsView(ring_run)
    t = sigma_terms(view)
    assert t.domain == "turn" and np.array_equal(t.s, np.asarray(view.s))
    lin = SimpleNamespace(s=[0.0, 1000.0, 2500.0],
                          sigma_matrix=[np.eye(6) * (k + 1) for k in range(3)],
                          element_names=["a", "b", "c"], element_exit_idx=[0, 1, 2],
                          ref_frequency=[162.5] * 3, ref_beta=[0.1] * 3, ref_gamma=[1.005] * 3)
    t = sigma_terms(lin)
    assert t.domain == "s"
    assert np.array_equal(t.s, np.asarray(lin.s) * 1e-3)


def test_a_saved_turn_plot_keeps_every_point(win, ring_run):
    # (a turn axis draws each curve thinned and clipped to the view: a saved
    # file still gets every row)
    from linac_gen_gui.interphase.tabs.results_tab import _curves_from_plotitem
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    assert rt.open_plot("emit_n")
    dlg = rt._popups["emit_n"]
    item = dlg._px.getPlotItem()
    item.getViewBox().setXRange(0.0, 5.0, padding=0.0)       # (most of it out of view)
    win.pump(0.2)
    curves = _curves_from_plotitem(item)
    assert curves and all(len(c.x) == len(ring_run.turns) for c in curves)


def test_the_sigma_window_saves_turn_by_turn(win, ring_run):
    # (its own exporter: every term against the turn, the matrix at a turn)
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    assert rt.open_plot("sigma")
    panels = rt._popups["sigma"]._collect_panels()
    terms = [p for p in panels if p.curves and p.xlabel != "row"]   # (not the Σ table)
    assert len(terms) == 21 and all(p.xlabel == "turn" for p in terms)
    assert np.array_equal(terms[0].curves[0].x, np.asarray(ring_run.turns, dtype=float))
    assert any(p.label == f"Σ at turn {int(ring_run.turns[-1])}" for p in panels)


# ------------------------------------------------- reviews: linac unchanged
def test_the_assistants_catalogue_has_ring_plots_only_with_ring_mode(win, ring_run):
    # (a linac session resolves plot names as before ring mode: "optics" or
    # "ramp" match nothing there, not a RING RUN tile that cannot open)
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    rt = win.results_tab

    def ring_keys():
        return {k for k, _l in rt.plot_catalog() if k.startswith("ring_")}
    assert ring_keys() == set()
    assert not [k for k, _l in ResultsTab.plot_catalog(None) if k.startswith("ring_")]
    win.open_lattice(FIXTURE)
    act = win._toolbar._ring_action
    act.trigger()                                          # ring mode on
    win.pump(0.1)
    assert win.state.ring_mode and "ring_optics" in ring_keys()
    act.trigger()                                          # off again
    win.pump(0.1)
    assert not win.state.ring_mode and ring_keys() == set()
    win.state.set_ring_results(copy.deepcopy(ring_run))    # (an imported ring run)
    assert "ring_optics" in ring_keys()


def test_the_assistant_panel_reads_the_catalogue_when_asked():
    from types import SimpleNamespace

    from linac_gen_gui.interphase.dialogs.assistant_panel import AssistantPanel
    cat = [("rms", "RMS σ (x · y · z)")]
    panel = SimpleNamespace(_app=SimpleNamespace(result_plot_catalog=lambda: list(cat)))
    prop = AssistantPanel.plot_catalog
    assert prop.fget(panel) == cat
    cat.append(("ring_optics", "Ring optics · periodic β, D"))
    assert prop.fget(panel)[-1][0] == "ring_optics"         # (read when asked)
    prop.fset(panel, [("phase", "Phase space (4-panel)")])
    assert prop.fget(panel) == [("phase", "Phase space (4-panel)")]


def test_a_round_trip_keeps_the_zoom_and_the_sigma_position(win, ring_run):
    import pyqtgraph as pg
    from linac_gen_gui.interphase.plots.lattice_strip import LatticeStripWidget
    _linac_run(win)
    rt = win.results_tab
    assert rt.open_plot("rms") and rt.open_plot("sigma")
    rms, sig = rt._popups["rms"], rt._popups["sigma"]
    _idle(win, rms)
    vb = next(pw for pw in rms.findChildren(pg.PlotWidget)
              if not isinstance(pw, LatticeStripWidget)).getPlotItem().getViewBox()
    s = np.asarray(win.state.results.s, dtype=float)
    vb.setXRange(float(s[len(s) // 4]), float(s[len(s) // 2]), padding=0.0)
    zoom = [list(r) for r in vb.viewRange()]
    sig._slider.setValue(3)
    assert sig._row == 3
    _show_ring(win, ring_run)
    assert vb.viewRange()[0] != zoom[0]                    # (the turn axis: its frame)
    assert sig._row == len(ring_run.turns) - 1            # (a new axis: its end)
    rt._run_combo.setCurrentIndex(0)
    win.pump(0.2)
    assert np.allclose(np.ravel(vb.viewRange()), np.ravel(zoom), rtol=1e-12, atol=0)
    assert sig._row == 3
    # (a new linac run: a new plot, its frame and the end row)
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    assert not np.allclose(vb.viewRange()[0], zoom[0])


def test_a_switch_closes_only_what_the_run_shown_cannot_show(win, ring_run):
    # ring results present, the linac run shown, the window of a linac tile
    # disabled for its own reason open (as IBS after a species change): a new
    # lattice drops the ring run — the window stays, as before ring mode
    _linac_run(win)
    rt = win.results_tab
    win.state.set_ring_results(copy.deepcopy(ring_run))
    rt._run_combo.setCurrentIndex(0)
    assert rt._source == "linac" and rt.open_plot("rms")
    rms = rt._popups["rms"]
    card = rt._card_of("rms")
    card.setEnabled(False)
    try:
        win.state.set_ring_results(None)
        assert rt._source == "linac" and rms.isVisible()
    finally:
        card.setEnabled(True)
    # (while the ring run is shown, the windows of the tiles off for it close)
    assert rt.open_plot("peak")
    _show_ring(win, ring_run)
    assert rt._source == "ring" and not rt._popups["peak"].isVisible()


def test_a_switch_ends_a_match_preview_title(win, ring_run):
    _linac_run(win)
    rt = win.results_tab
    assert rt.open_plot("quad_grad")
    d = rt._popups["quad_grad"]
    d._previewing = True
    d.setWindowTitle(f"{d._raw_title}  —  LIVE match iter 7")
    _show_ring(win, ring_run)
    assert not d._previewing and d.windowTitle() == f"{d._raw_title}  —  Ctrl+S to save"


def test_the_line_density_uses_the_runs_cells(qapp):
    # (the run's line-density cells start at -period/2: an odd number of bins
    # is half a bin off a grid at 0)
    from types import SimpleNamespace

    from linac_gen_gui.interphase.tabs.ring_popups import RingPhaseSpacePopup
    d = RingPhaseSpacePopup(None)
    res = SimpleNamespace(meta={"zeta_wrap_m": 1.0,
                                "space_charge": {"z_bins_used": [[0, 4], [10, 5]]}})
    z = np.random.default_rng(1).uniform(-0.29, 0.29, 4000)
    for t, dz in ((5, 0.25), (20, 0.2)):
        d._line_density(res, z, t)
        edges = np.asarray(d.c_line.xData)
        k = (edges + 0.5) / dz
        assert np.allclose(k, np.round(k), atol=1e-9), (t, edges[:3])
        assert np.diff(edges) == pytest.approx(np.full(edges.size - 1, dz))
        lam = np.asarray(d.c_line.yData)
        assert float(np.sum(lam * np.diff(edges))) == pytest.approx(1.0)   # (1/m, per particle)
    d.deleteLater()



# ------------------------------------------------ reviews: GUI lifecycle
def test_a_refused_ring_tile_switches_nothing(win, ring_run):
    # (a run at fixed energy: no ramp — "Ring ramp" refused before the view or
    # any window changes)
    _linac_run(win)
    rt = win.results_tab
    win.state.set_ring_results(copy.deepcopy(ring_run))
    assert rt._source == "linac" and rt.open_plot("peak")
    assert not rt.open_plot("ring_ramp")
    assert rt._source == "linac" and rt._popups["peak"].isVisible()


def test_new_ring_results_are_shown_in_ring_mode_only(win, ring_run):
    _linac_run(win)
    rt = win.results_tab
    win.state.set_ring_results(copy.deepcopy(ring_run))  # (ring mode off: a run stopped by
    assert rt._source == "linac"                          #  switching it off, for example)
    assert not rt._run_combo.isHidden()                    # (offered)
    act = win._toolbar._ring_action
    act.trigger()                                          # ring mode on
    win.pump(0.1)
    win.state.set_ring_results(copy.deepcopy(ring_run))
    assert rt._source == "ring"


def test_a_project_without_its_lattice_drops_the_ring_run(win, ring_run):
    win.open_lattice(FIXTURE)
    act = win._toolbar._ring_action
    act.trigger()
    win.pump(0.1)
    win.state.set_ring_results(copy.deepcopy(ring_run))
    rt = win.results_tab
    assert rt._source == "ring"
    data = win._collect_project_dict()
    data["lattice_path"] = str(FIXTURE.parent / "no_such_ring.madx")
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    assert win.state.ring_results is None and rt._source == "linac"
    assert win.ring_tab._summary.text() == "No ring run yet."


def test_the_live_strip_shows_a_ring_filling_up(qapp):
    from linac_gen_gui.interphase.panels.ring_plots import RingLiveStrip
    strip = RingLiveStrip()
    cov = np.tile(np.eye(6) * 1e-6, (3, 1, 1))
    strip.show_live({"start": 0, "turns": np.arange(3), "n_alive": np.array([10, 20, 30]),
                     "cov": cov, "beta0gamma0": 1.6, "p0c_mev": 1500.0})
    assert strip.p_surv.getAxis("left").labelText == "circulating (% of the start)"
    assert list(strip.c_surv.yData) == [100.0, 200.0, 300.0]
    assert strip.p_surv.getPlotItem().getViewBox().autoRangeEnabled()[1]
    strip.show_live({"start": 0, "turns": np.arange(3), "n_alive": np.array([10, 10, 9]),
                     "cov": cov, "beta0gamma0": 1.6, "p0c_mev": 1500.0})
    assert strip.p_surv.getAxis("left").labelText == "survival (%)"
    strip.deleteLater()


def test_every_turn_tile_card_shows_a_value(win, ring_run):
    from linac_gen_gui.interphase.tabs.results_tab import _TURN_TILES
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    empty = [k for k in sorted(_TURN_TILES) if rt._card_of(k)._value_lbl.text() == "—"]
    assert empty == [], empty


def test_a_lattice_edit_leaves_the_ring_runs_tiles(win, ring_run, monkeypatch):
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    fed = []
    for key in ("rms", "ring_long", "quad_grad"):
        card = rt._card_of(key)
        orig = card.set_results
        monkeypatch.setattr(card, "set_results",
                            lambda r, _k=key, _o=orig: (fed.append(_k), _o(r))[1])
    win.state.lattice_changed.emit(win.state.lattice)
    win.pump(0.1)
    assert "quad_grad" in fed and "rms" not in fed and "ring_long" not in fed


def test_dashed_curves_keep_their_dashes_on_a_turn_axis(win, ring_run):
    win.open_lattice(FIXTURE)
    rt = _show_ring(win, ring_run)
    assert rt.open_plot("energy")
    dashed = [it for pi in _plot_items(rt._popups["energy"]) for it in pi.listDataItems()
              if __import__("pyqtgraph").mkPen(it.opts.get("pen")).style()
              == Qt.PenStyle.DashLine]
    assert dashed and all(it.curve.opts["segmentedLineMode"] != "on" for it in dashed)


def test_the_rfq_panels_stay_off_for_a_ring_run(win, ring_run, monkeypatch):
    win.open_lattice(FIXTURE)
    rt = win.results_tab
    monkeypatch.setattr(rt, "_lattice_has_rfq", lambda: True)
    rt._apply_tile_filter("")
    rfq = [c for c in rt._cards if c._key.startswith("rfq_")]
    assert rfq and not any(c.isHidden() for c in rfq)
    _show_ring(win, ring_run)
    assert all(c.isHidden() and not c.isEnabled() for c in rfq)
    rt._run_combo.setCurrentIndex(0)
    assert all(c.isEnabled() and not c.isHidden() for c in rfq)
