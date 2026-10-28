"""Results tab "Σ matrix along s": the 21 independent terms of Σ along s
(rms values on the diagonal, correlations or covariances below it), the
whole matrix at a chosen position, and one term full width.

Pins: the tile opens a non-modal popup (also from the toolbar action); the
plotted values are those of linac_gen.analysis.sigma_terms; the Quantity
and Basis choices change titles, values and frames and are remembered; the
position moves the cursors, the colour matrix, the Σ table and the plane
summary, also by a click on a panel; a double-click shows the term full
width; no results / no Σ / continuous rows are reported, not drawn; the
export names every curve; the tile shows the largest coupling without a
trend; real FODO runs (multiparticle and envelope) plot the run's own Σ.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")

FODO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))
RMS = np.array([1.2, 2.5, 0.8, 1.7, 9.0, 4e-3])


def _noop(*_a, **_k):
    return None


def _data(curve):
    """The curve's full rows (the panels draw downsampled: getData() is
    what is on screen)."""
    return curve.xData, curve.yData


def _sigma(rng):
    a = rng.normal(size=(6, 6))
    c = a @ a.T + 6 * np.eye(6)
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d) * np.outer(RMS, RMS)


def _run(n=5, seed=1, continuous_at=None, sigma=True):
    """A DiagnosticRecorder-like multiparticle run with known Σ."""
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    rng = np.random.default_rng(seed)
    rec = DiagnosticRecorder()
    rec.s = [float(v) for v in np.linspace(0.0, 2000.0, n)]
    rec.sigma_matrix = [_sigma(rng) for _ in range(n)] if sigma else []
    rec.element_names = [f"E{k}" for k in range(n)]
    rec.ref_beta = [0.0667] * n
    rec.ref_gamma = [1.0022] * n
    rec.ref_frequency = [162.5] * n
    rec.mass_mev = 939.2942
    rec.transmission = [100.0] * n
    rec.continuous_at = list(continuous_at) if continuous_at is not None else [False] * n
    return rec


def _tab(results):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    st = AppState()
    tab = ResultsTab(st, _noop, _noop, _noop)
    st.set_results(results)
    return st, tab


def _open(tab):
    assert tab.open_plot("sigma")
    return tab._popups["sigma"]


def test_tile_opens_the_popup_with_the_21_terms(qapp, gui_message_boxes):
    from linac_gen.analysis import sigma_terms as stm
    from linac_gen_gui.interphase.tabs.sigma_popup import SigmaTermsPopup
    rec = _run()
    _st, tab = _tab(rec)
    try:
        dlg = _open(tab)
        assert isinstance(dlg, SigmaTermsPopup) and not dlg.isModal()
        assert gui_message_boxes == []
        assert len(dlg._plots) == 21 and set(dlg._plots) == set(stm.TERMS)
        assert dlg._plots[(0, 0)].titleLabel.text == "σ(x) [mm]"
        assert dlg._plots[(1, 0)].titleLabel.text == "r(x, x′)"
        assert dlg._plots[(5, 4)].titleLabel.text == "r(Δφ, ΔW)"
        t = stm.sigma_terms(rec, "ours")
        for (i, j), curve in dlg._curves.items():
            x, y = _data(curve)
            np.testing.assert_allclose(x, t.s)
            want = t.rms[:, i] if i == j else t.corr[:, i, j]
            np.testing.assert_allclose(y, want, rtol=1e-12)
        # correlation panels keep a −1…1 frame, rms panels auto-range
        assert dlg._plots[(1, 0)].getViewBox()._helix_frame[1] == (-1.05, 1.05)
        assert getattr(dlg._plots[(0, 0)].getViewBox(), "_helix_frame", None) is None
        assert "Multiparticle run" in dlg._note.text()
        # starts at the end of the lattice
        assert dlg._slider.maximum() == 4 and dlg._slider.value() == 4
        assert "s = 2.0000 m" in dlg._pos_label.text() and "E4" in dlg._pos_label.text()
    finally:
        tab.deleteLater()


def test_quantity_and_basis(qapp):
    from linac_gen.analysis import sigma_terms as stm
    rec = _run()
    _st, tab = _tab(rec)
    try:
        dlg = _open(tab)
        dlg._quantity.setCurrentIndex(dlg._quantity.findData("cov"))
        t = stm.sigma_terms(rec, "ours")
        _x, y = _data(dlg._curves[(3, 0)])
        np.testing.assert_allclose(y, t.sigma[:, 3, 0], rtol=1e-12)
        assert dlg._plots[(3, 0)].titleLabel.text == "⟨x y′⟩ [mm·mrad]"
        assert getattr(dlg._plots[(3, 0)].getViewBox(), "_helix_frame", None) is None
        _x, yw = _data(dlg._curves[(5, 5)])
        np.testing.assert_allclose(yw, RMS[5] * 1e3, rtol=1e-12)    # keV
        dlg._basis.setCurrentIndex(dlg._basis.findData("tracewin"))
        tw = stm.sigma_terms(rec, "tracewin")
        _x, yz = _data(dlg._curves[(4, 0)])
        np.testing.assert_allclose(yz, tw.sigma[:, 4, 0], rtol=1e-12)
        assert dlg._plots[(4, 4)].titleLabel.text == "σ(z) [mm]"
        assert dlg._plots[(5, 4)].titleLabel.text == "⟨z δ⟩ [mm·‰]"
        assert "opposite sign" in dlg._note.text()
        dlg._quantity.setCurrentIndex(dlg._quantity.findData("corr"))
        _x, rz = _data(dlg._curves[(4, 0)])
        _x0 = stm.sigma_terms(rec, "ours")
        np.testing.assert_allclose(rz, -_x0.corr[:, 4, 0], rtol=1e-12)  # z = −k·Δφ
    finally:
        tab.deleteLater()


def test_choices_are_remembered(qapp):
    from linac_gen_gui.interphase.tabs.sigma_popup import SigmaTermsPopup
    _st, tab = _tab(_run())
    try:
        dlg = _open(tab)
        dlg._quantity.setCurrentIndex(dlg._quantity.findData("cov"))
        dlg._basis.setCurrentIndex(dlg._basis.findData("tracewin"))
        again = SigmaTermsPopup(tab, tab.state)
        assert again._quantity.currentData() == "cov"
        assert again._basis.currentData() == "tracewin"
        dlg._quantity.setCurrentIndex(dlg._quantity.findData("corr"))
        dlg._basis.setCurrentIndex(dlg._basis.findData("ours"))
        again.deleteLater()
    finally:
        tab.deleteLater()


def test_position_moves_cursors_matrix_table_and_summary(qapp):
    from linac_gen.analysis import sigma_terms as stm
    rec = _run()
    _st, tab = _tab(rec)
    try:
        dlg = _open(tab)
        t = stm.sigma_terms(rec, "ours")
        dlg._slider.setValue(2)
        assert all(c.value() == pytest.approx(t.s[2]) for c in dlg._cursors)
        assert dlg._mat_text[(1, 0)].toPlainText() == f"{t.corr[2, 1, 0]:+.2f}"
        assert dlg._mat_text[(0, 1)].toPlainText() == f"{t.corr[2, 0, 1]:+.2f}"
        assert float(dlg._table.item(3, 1).text()) == pytest.approx(
            t.sigma[2, 3, 1], rel=1e-3)
        assert "x: σ(x)" in dlg._summary.text() and "deg·MeV" in dlg._summary.text()
        img = dlg._mat_img.image
        assert img.shape == (6, 6, 4)
        # each cell drawn in its own colour (the rect once covered the
        # matrix with the first cell's colour)
        qapp.processEvents()
        q = dlg._mat_gl.grab().toImage()
        vb = dlg._mat_plot.getViewBox()
        for (i, j) in ((1, 0), (3, 2)):
            pt = vb.mapViewToScene(pg_point(j + 0.5, i + 0.5))
            got = q.pixelColor(int(pt.x()), int(pt.y()))
            want = dlg._cmap.map((t.corr[2, i, j] + 1.0) / 2.0, mode="byte")
            assert abs(got.red() - int(want[0])) <= 3 and \
                abs(got.blue() - int(want[2])) <= 3, (i, j, got.getRgb(), want)
        # a click on a panel moves the position
        p = dlg._plots[(2, 1)]
        dlg._grid.resize(1000, 800)
        qapp.processEvents()
        vb = p.getViewBox()
        target = vb.mapViewToScene(pg_point(float(t.s[1]), 0.0))

        class _Ev:
            def button(self):
                from PyQt6.QtCore import Qt
                return Qt.MouseButton.LeftButton

            def scenePos(self):
                return target

            def double(self):
                return False

        dlg._on_grid_click(_Ev())
        assert dlg._slider.value() == 1
    finally:
        tab.deleteLater()


def pg_point(x, y):
    from PyQt6.QtCore import QPointF
    return QPointF(x, y)


def test_full_width_view(qapp):
    from linac_gen.analysis import sigma_terms as stm
    rec = _run()
    _st, tab = _tab(rec)
    try:
        dlg = _open(tab)
        dlg._show_focus(3, 0)
        assert dlg._stack.currentIndex() == 1
        assert dlg._focus_title.text() == "r(x, y′)"
        _x, y = _data(dlg._focus_curve)
        np.testing.assert_allclose(y, stm.sigma_terms(rec).corr[:, 3, 0], rtol=1e-12)
        assert dlg._lattice_strip is not None
        dlg._show_grid()
        assert dlg._stack.currentIndex() == 0
        assert _data(dlg._focus_curve)[0] is None or len(_data(dlg._focus_curve)[0]) == 0
    finally:
        tab.deleteLater()


def test_no_results_no_sigma_and_continuous_rows(qapp):
    from linac_gen_gui.interphase.tabs import sigma_popup as sp
    st, tab = _tab(None)
    try:
        dlg = _open(tab)
        assert dlg._note.text() == sp.NO_RESULTS
        assert all(_data(c)[0] is None or len(_data(c)[0]) == 0 for c in dlg._curves.values())
        st.set_results(_run(sigma=False))
        dlg.refresh(st.results)
        assert dlg._note.text() == sp.NO_SIGMA
        st.set_results(_run(continuous_at=[True, True, False, False, False]))
        dlg.refresh(st.results)
        assert "continuous up to s = 0.500 m" in dlg._note.text()
        _x, y = _data(dlg._curves[(4, 4)])
        assert np.isnan(y[:2]).all() and np.isfinite(y[2:]).all()
        _x, y = _data(dlg._curves[(5, 5)])
        assert np.isfinite(y).all()                       # ΔW kept
        dlg.refresh(None)
        assert dlg._note.text() == sp.NO_RESULTS and dlg._pos_label.text() == ""
    finally:
        tab.deleteLater()


def test_hint_when_only_element_exits_are_recorded(qapp):
    rec = _run()
    rec.element_exit_idx = [1, 2, 3, 4]                  # 5 rows: input + 4 exits
    _st, tab = _tab(rec)
    try:
        dlg = _open(tab)
        assert "Record per-sub-step" in dlg._note.text()
        rec2 = _run(n=40)
        rec2.element_exit_idx = [10, 20, 30, 39]          # rows inside elements
        dlg.refresh(rec2)
        assert "Record per-sub-step" not in dlg._note.text()
    finally:
        tab.deleteLater()


def test_export_names_every_curve(qapp):
    _st, tab = _tab(_run())
    try:
        dlg = _open(tab)
        panels = dlg._collect_panels()
        names = {c.name for pn in panels for c in (pn.curves or [])}
        assert {"σ(x) [mm]", "r(x, x′)", "r(Δφ, ΔW)", "σ(ΔW) [keV]"} <= names
        assert sum(1 for n in names if n.startswith("r(")) == 15
        # full rows, whatever the panels draw
        assert all(len(c.x) == 5 for pn in panels[:21] for c in pn.curves)
        # the Σ table at the position, one column per coordinate
        assert any({"x (mm)", "ΔW (keV)"} <= {c.name for c in pn.curves}
                   for pn in panels)
    finally:
        tab.deleteLater()


def test_tile_shows_largest_coupling_without_trend(qapp):
    rec = _run()
    rec.sigma_matrix = [np.diag(RMS ** 2) for _ in range(5)]
    S = rec.sigma_matrix[3]
    S[0, 3] = S[3, 0] = 0.6 * RMS[0] * RMS[3]
    _st, tab = _tab(rec)
    try:
        card = next(c for c in tab._cards if c._key == "sigma")
        assert card._title_lbl.text() == "Σ matrix along s"
        assert card._value_lbl.text() == "0.00"
        assert card._trend_lbl.text() == ""
        tip = card.toolTip()
        assert "largest between planes  0.600" in tip and "at s = 1.500 m" in tip
    finally:
        tab.deleteLater()


def test_toolbar_action_and_real_runs(win):
    from linac_gen.analysis import sigma_terms as stm
    win.open_lattice(FODO)
    win.beam_tab._npart.setValue(400)
    win.beam_tab._apply(); win.pump()
    win._run_mp(); win.wait_worker(win._mp_worker)
    win._open_sigma_matrix()                               # the toolbar action
    dlg = win.results_tab._popups["sigma"]
    win.pump(0.2)
    assert dlg.isVisible() and not dlg.isModal()
    rec = win.state.results
    t = stm.sigma_terms(rec, "ours")
    S = np.asarray(rec.sigma_matrix, float)
    d = np.sqrt(np.einsum("nii->ni", S))
    _x, y = _data(dlg._curves[(1, 0)])
    np.testing.assert_allclose(y, S[:, 1, 0] / (d[:, 0] * d[:, 1]), rtol=1e-12)
    _x, y = _data(dlg._curves[(0, 0)])
    np.testing.assert_allclose(y, d[:, 0], rtol=1e-12)
    # a FODO does not couple the planes: only sampling noise in a 400-particle run
    _x, c = _data(dlg._curves[(2, 0)])
    assert np.nanmax(np.abs(c)) < 0.3
    assert "Multiparticle run" in dlg._note.text()
    # an envelope run of the same lattice: exactly uncoupled
    win._run_envelope(); win.wait_worker(win._envelope_worker); win.pump(0.2)
    _x, c = _data(dlg._curves[(2, 0)])
    assert np.nanmax(np.abs(np.asarray(c, float))) == 0.0
    assert "Envelope run" in dlg._note.text()
    assert len(t) > 2
    assert win.message_boxes == []
