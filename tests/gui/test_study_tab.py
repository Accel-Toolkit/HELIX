"""Param Study tab: construction, live run count, worker teardown."""
from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("PyQt6")

FODO = """DRIFT 100 20 0 0 0
QUAD 80 8.0 20 0 0 0 0 0 0
DRIFT 200 20 0 0 0
QUAD 80 -8.0 20 0 0 0 0 0 0
DRIFT 100 20 0 0 0
END
"""


@pytest.fixture()
def tab(qapp, tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.study_tab import StudyTab

    deck = tmp_path / "fodo.dat"
    deck.write_text(FODO)
    state = AppState()
    lat, _ = parse_tracewin(str(deck))
    state.set_lattice(lat, path=str(deck))
    t = StudyTab(state)
    yield t
    t.deleteLater()


def _add_quad_param(tab) -> None:
    # pick the first QUAD entry in the element combo (skip "Beam")
    for i in range(tab._elem_combo.count()):
        if "Quadrupole" in tab._elem_combo.itemText(i):
            tab._elem_combo.setCurrentIndex(i)
            break
    for j in range(tab._attr_combo.count()):
        if tab._attr_combo.itemData(j) == "gradient":
            tab._attr_combo.setCurrentIndex(j)
            break
    tab._on_add_param()


class TestRunCount:
    def test_empty_shows_dash(self, tab):
        tab._refresh_run_count()
        assert tab._run_count.text() == "—"

    def test_grid_and_repeats(self, tab):
        _add_quad_param(tab)
        assert tab._run_count.text() == "5"          # default n=5
        tab._repeats.setValue(3)
        assert tab._run_count.text() == "15"

    def test_selector_uses_index_grammar(self, tab):
        _add_quad_param(tab)
        sel = tab._ptable.item(0, 0).text()
        assert sel.startswith("@") and sel.endswith(".gradient")

    def test_beam_pseudo_element(self, tab):
        tab._elem_combo.setCurrentIndex(0)           # "Beam"
        assert tab._attr_combo.count() > 0
        names = [tab._attr_combo.itemData(j)
                 for j in range(tab._attr_combo.count())]
        assert "current" in names


class TestWorkerTeardown:
    def test_shutdown_begin_joins(self, qapp, tmp_path, tab):
        """A running study worker must stop within the shutdown budget."""
        from linac_gen.study.engine import StudyManager
        from linac_gen.study.spec import ParamSpec, StudySpec
        from linac_gen_gui.interphase.tabs.study_tab import _StudyWorker

        deck = tmp_path / "fodo.dat"                 # from the fixture
        spec = StudySpec(
            name="teardown", input=str(deck), mode="envelope",
            strategy="grid",
            parameters=[ParamSpec(selector="@2.gradient",
                                  start=7.0, stop=9.0, n=3)],
            beam={"energy": 2.1, "current": 0.0, "n_particles": 300})
        StudyManager.create(tmp_path / "sd", spec)
        w = _StudyWorker(str(tmp_path / "sd"), max_workers=1)
        tab._worker = w
        w.start()
        workers = tab.shutdown_begin()
        assert workers == [w]
        assert w.wait(15000), "study worker did not stop in time"

    def test_idle_shutdown_is_empty(self, tab):
        assert tab.shutdown_begin() == []


class TestAnalysisHelpers:
    """Pure helpers of the analysis panel (no Qt needed beyond import)."""

    def _col(self, rec, name):
        if name in rec["metrics"]:
            return rec["metrics"].get(name)
        return rec["obs"].get(name)

    def _rec(self, g, cur, trans, seed=42):
        return {"params": {"g": g, "cur": cur},
                "metrics": {"transmission": trans}, "obs": {},
                "status": "ok"}

    def test_aggregate_1d_repeats_to_error_bars(self, qapp):
        from linac_gen_gui.interphase.panels.study_plots import (
            aggregate_1d)
        recs = [self._rec(1.0, 0.0, 90.0), self._rec(1.0, 0.0, 92.0),
                self._rec(2.0, 0.0, 80.0)]
        out = aggregate_1d(recs, "g", "transmission", None, self._col)
        xv, ym, ys, n = out[None]
        assert list(xv) == [1.0, 2.0]
        assert ym[0] == 91.0 and n[0] == 2 and ys[0] == 1.0
        assert ym[1] == 80.0 and n[1] == 1

    def test_aggregate_1d_group_by(self, qapp):
        from linac_gen_gui.interphase.panels.study_plots import (
            aggregate_1d)
        recs = [self._rec(1.0, 0.0, 90.0), self._rec(1.0, 5.0, 70.0)]
        out = aggregate_1d(recs, "g", "transmission", "cur", self._col)
        assert set(out) == {0.0, 5.0}

    def test_detect_grid_full_and_holed(self, qapp):
        import numpy as np

        from linac_gen_gui.interphase.panels.study_plots import (
            detect_grid)
        recs = [self._rec(x, y, x * 10 + y)
                for x in (1.0, 2.0) for y in (0.0, 5.0)]
        xu, yu, Z = detect_grid(recs, "g", "cur", "transmission",
                                self._col)
        assert list(xu) == [1.0, 2.0] and list(yu) == [0.0, 5.0]
        assert Z[0, 0] == 10.0 and Z[1, 1] == 25.0
        # a hole (missing cell) must demote to scatter (None)
        assert detect_grid(recs[:-1], "g", "cur", "transmission",
                           self._col) is None

    def test_cell_edges_follow_the_true_values(self, qapp):
        import numpy as np

        from linac_gen_gui.interphase.panels.study_plots import cell_edges
        np.testing.assert_allclose(cell_edges([1.0, 2.0, 3.0]),
                                   [0.5, 1.5, 2.5, 3.5])
        np.testing.assert_allclose(cell_edges([1.0, 2.0, 4.0, 8.0]),
                                   [0.5, 1.5, 3.0, 6.0, 10.0])
        np.testing.assert_allclose(cell_edges([7.0]), [6.5, 7.5])

    def _panel(self, recs):
        from types import SimpleNamespace

        from linac_gen_gui.interphase.panels.study_plots import _Plot2DPanel
        model = SimpleNamespace(
            param_names=["g", "cur"], value_columns=lambda: ["transmission"],
            ok_records=lambda: recs, column=self._col)
        panel = _Plot2DPanel(model)
        panel.refresh_choices()
        img = [it for it in panel.plot.getPlotItem().items
               if type(it).__name__ == "ImageItem"][0]
        return panel, img

    def test_uneven_scan_cells_sit_at_their_values(self, qapp):
        """g = 1, 2, 4, 8 used to be drawn evenly spaced (at 1, 3.33,
        5.67, 8) under an axis reading 1, 2, 4, 8."""
        recs = [self._rec(g, c, g * 10 + c)
                for g in (1.0, 2.0, 4.0, 8.0) for c in (0.0, 5.0)]
        panel, img = self._panel(recs)
        rect = img.mapRectToParent(img.boundingRect())
        assert rect.left() == 0.5 and rect.right() == 10.0
        data = img.image                         # (x pixels, y pixels)

        def z_at(g, row):
            ix = int((g - rect.left()) / rect.width() * data.shape[0])
            return data[ix, row]
        assert z_at(1.6, 0) == 20.0              # inside the g = 2 cell
        assert z_at(2.9, 0) == 20.0
        assert z_at(3.1, 0) == 40.0              # g = 4 cell starts at 3
        assert z_at(9.9, 1) == 85.0

    def test_even_scan_draws_exactly_as_before(self, qapp):
        import numpy as np
        recs = [self._rec(g, c, g * 10 + c)
                for g in (1.0, 2.0, 3.0) for c in (0.0, 5.0)]
        panel, img = self._panel(recs)
        rect = img.mapRectToParent(img.boundingRect())
        # the pre-fix placement: one pixel per cell, rect grown by half a
        # step on each side
        assert (rect.left(), rect.right()) == (0.5, 3.5)
        assert (rect.top(), rect.bottom()) == (-2.5, 7.5)
        np.testing.assert_array_equal(
            img.image, np.array([[10.0, 15.0], [20.0, 25.0], [30.0, 35.0]]))


# ---------------------------------------------------------------------------
# QSettings persistence (sandboxed store via HELIX_QSETTINGS_DIR)
# ---------------------------------------------------------------------------
def _fresh_tab(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.study_tab import StudyTab
    deck = tmp_path / "fodo.dat"
    if not deck.exists():
        deck.write_text(FODO)
    state = AppState()
    lat, _ = parse_tracewin(str(deck))
    state.set_lattice(lat, path=str(deck))
    return StudyTab(state)


def test_persistence_restores_root_and_last_study(qapp, tmp_path):
    from linac_gen_gui.interphase.tabs.study_tab import (
        _S_LAST, _S_ROOT, _settings)
    st = _settings()
    st.remove(_S_ROOT)
    st.remove(_S_LAST)
    try:
        # a real (finished) study dir to remember: build via the engine
        from linac_gen.study import ParamSpec, StudySpec, StudyManager
        spec = StudySpec(
            name="persisted", input=str(tmp_path / "fodo.dat"),
            parameters=[ParamSpec(selector="@2.gradient", start=6.0,
                                  stop=10.0, n=2)])
        (tmp_path / "fodo.dat").write_text(FODO)
        mgr = StudyManager.create(tmp_path / "persisted", spec)
        mgr.run(serial=True)

        t1 = _fresh_tab(tmp_path)
        t1._folder.setText(str(tmp_path))
        t1._study_dir = str(tmp_path / "persisted")
        t1._remember()
        t1.deleteLater()

        t2 = _fresh_tab(tmp_path)
        assert t2._folder.text() == str(tmp_path)
        assert t2._study_dir == str(tmp_path / "persisted")
        assert "persisted" in t2._status.text()
        t2.deleteLater()
    finally:
        st.remove(_S_ROOT)
        st.remove(_S_LAST)


def test_persistence_clears_vanished_study(qapp, tmp_path):
    from linac_gen_gui.interphase.tabs.study_tab import (
        _S_LAST, _S_ROOT, _settings)
    st = _settings()
    st.setValue(_S_LAST, str(tmp_path / "gone"))
    st.remove(_S_ROOT)
    try:
        t = _fresh_tab(tmp_path)
        assert t._study_dir is None
        assert st.value(_S_LAST) is None
        t.deleteLater()
    finally:
        st.remove(_S_LAST)
