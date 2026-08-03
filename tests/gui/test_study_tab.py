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
