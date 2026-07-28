"""Headless tests for the Failure Study tab (construction, populate, worker)."""
from __future__ import annotations

import copy

import pytest

PyQt6 = pytest.importorskip("PyQt6")

from linac_gen.core.config import BeamConfig
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.failure_study_tab import (
    FailureStudyTab, _FailureWorker)


def test_failure_tab_constructs(qapp):
    tab = FailureStudyTab(AppState())          # no lattice yet
    assert tab._elem_list.count() == 0
    tab.deleteLater()


def test_failure_tab_populates_and_mode_fields(qapp, mini_lattice):
    state = AppState()
    state.set_lattice(mini_lattice, path=None)
    tab = FailureStudyTab(state)
    names = {tab._elem_list.item(i).text().split("  [")[0]
             for i in range(tab._elem_list.count())}
    assert {"Q1", "G1"} <= names               # quad + cavity; drifts excluded

    tab._mode.setCurrentText("off")
    assert not tab._amp.isEnabled() and not tab._phase.isEnabled()
    tab._mode.setCurrentText("detune")
    assert tab._amp.isEnabled() and tab._phase.isEnabled()
    tab._mode.setCurrentText("partial")
    assert tab._amp.isEnabled() and not tab._phase.isEnabled()
    tab.deleteLater()


def test_worker_runs_serial(qapp, mini_lattice):
    from linac_gen.failures import CompensationConfig, FailureKind
    beam = BeamConfig(species="proton", energy=2.5, frequency=162.5,
                      current=0.0, n_particles=200, distribution="waterbag",
                      emit_nx=0.25, beta_x=1.0, emit_ny=0.25, beta_y=1.0,
                      emit_z=0.3, beta_z=10.0)
    worker = _FailureWorker(
        copy.deepcopy(mini_lattice), beam,
        types={"cavity", "quad", "solenoid", "dipole"},
        kind=FailureKind.OFF, amp=1.0, phase=0.0, combination="single",
        custom_sets=[], only_names=None, mode="envelope", workers=1,
        compensate=False, comp_cfg=CompensationConfig(), n_compensate=0)

    seen = {"done": None, "failed": None, "scn": 0}
    worker.scenario_done.connect(lambda i, im: seen.__setitem__("scn", seen["scn"] + 1))
    worker.done.connect(lambda r: seen.__setitem__("done", r))
    worker.failed.connect(lambda m: seen.__setitem__("failed", m))

    worker.run()                                # synchronous (not start())

    assert seen["failed"] is None, seen["failed"]
    res = seen["done"]
    assert res is not None
    assert len(res.impacts) == 2                # Q1 + G1
    assert seen["scn"] == 2
    assert res.ranking == sorted(
        range(len(res.impacts)),
        key=lambda i: res.impacts[i].criticality, reverse=True)


def test_on_run_empty_selection_means_all(qapp, mini_lattice, monkeypatch):
    """Regression: an empty element selection must mean 'all' (only_names=None),
    not an empty filter (only_names=[]) which would enumerate zero scenarios."""
    # Stub the thread launch: _on_run builds the worker (setting _only_names) and
    # the table header synchronously; we assert on those without spawning a real
    # QThread (a leaked worker segfaults the next GUI test at teardown).
    monkeypatch.setattr(_FailureWorker, "start", lambda self: None)
    state = AppState()
    state.set_lattice(mini_lattice, path=None)
    state.set_beam_config(BeamConfig(
        species="proton", energy=2.5, frequency=162.5, current=0.0,
        n_particles=80, distribution="waterbag", emit_nx=0.25, beta_x=1.0,
        emit_ny=0.25, beta_y=1.0, emit_z=0.3, beta_z=10.0))
    tab = FailureStudyTab(state)
    tab._elem_list.clearSelection()                 # select nothing
    tab._forward.setCurrentText("envelope")
    tab._combo.setCurrentText("single")
    tab._on_run()                                   # builds worker; start() stubbed
    assert tab._worker is not None
    # the fix: empty selection -> None (all), never []
    assert tab._worker._only_names is None
    headers = [tab._table.horizontalHeaderItem(c).text()
               for c in range(tab._table.columnCount())]
    assert "loss [%]" in headers
    tab.deleteLater()


def test_on_run_construction_failure_restores_buttons(qapp, mini_lattice,
                                                      monkeypatch):
    """Regression: buttons used to flip BEFORE the worker was constructed
    — a constructor failure left Run disabled / Stop enabled with no
    worker to stop, wedging the tab."""
    def _boom(*a, **k):
        raise RuntimeError("constructor exploded")
    monkeypatch.setattr(
        "linac_gen_gui.interphase.tabs.failure_study_tab._FailureWorker",
        _boom)
    errors = []
    from linac_gen_gui.interphase.tabs import failure_study_tab as fstab
    monkeypatch.setattr(
        fstab.QMessageBox, "critical",
        staticmethod(lambda *a, **k: errors.append(a)))

    state = AppState()
    state.set_lattice(mini_lattice, path=None)
    state.set_beam_config(BeamConfig(
        species="proton", energy=2.5, frequency=162.5, current=0.0,
        n_particles=80, distribution="waterbag", emit_nx=0.25, beta_x=1.0,
        emit_ny=0.25, beta_y=1.0, emit_z=0.3, beta_z=10.0))
    tab = FailureStudyTab(state)
    tab._forward.setCurrentText("envelope")
    tab._on_run()

    assert errors                      # failure surfaced, not swallowed
    assert tab._worker is None
    assert tab._run_btn.isEnabled()    # tab NOT wedged
    assert not tab._stop_btn.isEnabled()
    tab.deleteLater()


def test_heatmap_labels_shortened_and_thinned(qapp):
    """Axis ticks must not overlap: a common prefix (FMAP_) is stripped and
    crowded axes are thinned."""
    F = FailureStudyTab
    # common prefix stripped at its last separator -> short, unique labels
    assert F._short_labels([f"FMAP_{i:03d}" for i in range(1, 5)]) == \
        ["001", "002", "003", "004"]
    # no common prefix -> unchanged (element types must stay distinguishable)
    assert F._short_labels(["QUAD_001", "SOL_001"]) == ["QUAD_001", "SOL_001"]
    assert F._short_labels(["Q1", "G1"]) == ["Q1", "G1"]
    # thinning: 60 labels at step 3 -> 20 ticks; small N keeps all
    assert len(F._thinned_ticks([str(i) for i in range(60)], 3)[0]) == 20
    assert len(F._thinned_ticks(["a", "b", "c"], 1)[0]) == 3


def test_pairs_heatmap_renders(qapp, mini_lattice):
    """Regression: the pair heatmap must render with the view auto-ranged to the
    full N×N matrix (the ImageItem previously stayed zoomed to a 1×1 corner and
    looked empty), with element-name ticks and a colour scale."""
    from PyQt6.QtCore import Qt
    from linac_gen.failures import CompensationConfig, FailureKind
    beam = BeamConfig(species="proton", energy=2.5, frequency=162.5, current=0.0,
                      n_particles=120, distribution="waterbag", emit_nx=0.25,
                      beta_x=1.0, emit_ny=0.25, beta_y=1.0, emit_z=0.3, beta_z=10.0)
    state = AppState()
    state.set_lattice(mini_lattice, path=None)
    state.set_beam_config(beam)
    tab = FailureStudyTab(state)
    tab._run_combo = "pairs"                    # _on_run would set this
    worker = _FailureWorker(
        copy.deepcopy(mini_lattice), beam,
        types={"cavity", "quad", "solenoid", "dipole"},
        kind=FailureKind.OFF, amp=1.0, phase=0.0, combination="pairs",
        custom_sets=[], only_names=None, mode="envelope", workers=1,
        compensate=False, comp_cfg=CompensationConfig(), n_compensate=0)
    for sig, slot in ((worker.prepared, tab._on_prepared),
                      (worker.progress, tab._on_progress),
                      (worker.scenario_done, tab._on_scenario_done),
                      (worker.done, tab._on_done)):
        sig.connect(slot, Qt.ConnectionType.DirectConnection)
    worker.run()                                # synchronous; spawns no thread

    img = tab._heat_img.image
    assert img is not None
    n = img.shape[0]
    assert n == 2                               # Q1 + G1
    # the fix: the view covers the whole matrix, not a 1×1 corner
    (x0, x1), (y0, y1) = tab._heat.getViewBox().viewRange()
    assert x0 <= 0.1 and x1 >= n - 0.1, (x0, x1)
    assert y0 <= 0.1 and y1 >= n - 0.1, (y0, y1)
    # element-name ticks on both axes
    ticks = tab._heat.getAxis("bottom")._tickLevels
    assert ticks and {t[1] for t in ticks[0]} == {"Q1", "G1"}
    # all 3 scenarios present (2 singles + 1 pair); heatmap symmetric
    assert tab._table.rowCount() == 3
    assert tab._heat_cbar is None or tab._heat_cbar.levels()[1] >= 0.0
    # must NOT be aspect-locked (else a square image sits as a band in the
    # wide/short pane instead of filling it)
    assert not tab._heat.getViewBox().state["aspectLocked"]
    tab.deleteLater()


def test_bar_yaxis_autoscales(qapp, mini_lattice):
    """Regression: the criticality bar must scale its Y-axis to the data. It
    previously stayed at [0,1], clipping every bar > 1 so they looked constant."""
    from linac_gen.failures.study import FailureStudyResults, ScenarioImpact
    from linac_gen.failures.scenario import FailureScenario
    from linac_gen.failures.failure_mode import FailureMode, FailureKind

    state = AppState()
    state.set_lattice(mini_lattice, path=None)
    state.set_beam_config(BeamConfig())
    tab = FailureStudyTab(state)

    def mk(name, crit):
        scn = FailureScenario(((name, FailureMode(FailureKind.OFF)),),
                              label=f"{name}:off")
        return ScenarioImpact(
            scenario=scn,
            metrics={"transmission": 100.0, "emit_nx": 0.25, "emit_ny": 0.25,
                     "emit_nz": 1.0},
            d_transmission=0.0, emit_growth_x=0.0, emit_growth_y=0.0,
            emit_growth_z=0.0, d_energy_mev=0.0, beam_lost=False, criticality=crit)

    impacts = [mk("A", 2.5), mk("B", 1.7), mk("C", 0.3)]
    res = FailureStudyResults(
        baseline={"transmission": 100.0, "ref_w_kin": 10.0}, impacts=impacts,
        ranking=[0, 1, 2], element_names=["A", "B", "C"], mode="envelope",
        pair_matrix=None)
    tab._on_done(res)
    y1 = tab._bar.getViewBox().viewRange()[1][1]
    assert y1 >= 2.5, y1            # covers the tallest bar (old bug: stuck at 1.0)
    tab.deleteLater()


def test_bar_fills_incrementally(qapp, mini_lattice):
    """The criticality bar must fill as scenarios complete (not only at the
    end) so long runs show progress."""
    from linac_gen.failures.study import ScenarioImpact
    from linac_gen.failures.scenario import FailureScenario
    from linac_gen.failures.failure_mode import FailureMode, FailureKind

    state = AppState()
    state.set_lattice(mini_lattice, path=None)
    state.set_beam_config(BeamConfig())
    tab = FailureStudyTab(state)
    tab._run_combo = "single"

    def mk(name, c):
        scn = FailureScenario(((name, FailureMode(FailureKind.OFF)),),
                              label=f"{name}:off")
        return ScenarioImpact(
            scenario=scn,
            metrics={"transmission": None, "emit_nx": 0.25, "emit_ny": 0.25,
                     "emit_nz": 1.0},
            d_transmission=None, emit_growth_x=0.0, emit_growth_y=0.0,
            emit_growth_z=0.0, d_energy_mev=0.1, beam_lost=False, criticality=c)

    bars = lambda: [it for it in tab._bar.plotItem.items
                    if it.__class__.__name__ == "BarGraphItem"]
    tab._on_prepared(["A", "B"], 2)
    assert bars() == []                                # cleared at run start
    tab._on_scenario_done(0, mk("A", 0.3))
    assert len(bars()) == 1                            # a bar after one scenario
    tab._on_scenario_done(1, mk("B", 2.1))
    assert tab._bar.getViewBox().viewRange()[1][1] >= 2.1   # y-range tracks new max
    # bars are labelled by element so they map back to the table by name
    ticks = tab._bar.getAxis("bottom")._tickLevels
    assert ticks and {t[1] for t in ticks[0]} == {"A", "B"}
    tab.deleteLater()
