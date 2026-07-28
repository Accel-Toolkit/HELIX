"""Phase Space popup: the 'Beam parameters' toggle swaps the density
grid for a full parameter table of the selected distribution."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _recorder_with_snapshots():
    from linac_gen.core.beam import Beam
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    rec = DiagnosticRecorder()
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
    rng = np.random.default_rng(0)
    rec.s = [0.0, 150.0, 300.0, 400.0]
    rec.element_names = ["INPUT", "Q1", "Q2", "D3"]
    for s in (150.0, 300.0):
        b = Beam(ref=ref, n_particles=300, current=0.0)
        b.particles[:] = rng.normal(0, 0.3, (300, 6))
        rec.save_snapshot(b, s)
    exit_beam = Beam(ref=ref, n_particles=300, current=0.0)
    exit_beam.particles[:] = rng.normal(0, 0.5, (300, 6))
    rec.beam = exit_beam
    return rec


@pytest.fixture()
def state_with_results(qapp):
    from linac_gen_gui.interphase.state import AppState
    st = AppState()
    st.set_results(_recorder_with_snapshots())
    return st


def _cell(dlg, name):
    """Value string of the table row whose parameter cell endswith name."""
    tbl = dlg._table
    for r in range(tbl.rowCount()):
        it = tbl.item(r, 0)
        if it is not None and it.text().strip() == name:
            return tbl.item(r, 1).text()
    raise AssertionError(f"row '{name}' not found")


def test_toggle_swaps_plots_for_table_with_correct_values(
        state_with_results):
    from linac_gen.diagnostics.moments import (
        compute_moments, compute_twiss_from_particles,
    )
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup

    dlg = _PhaseSpacePopup(None, state_with_results)
    try:
        dlg.refresh(state_with_results.results)
        assert dlg._table.isHidden() is True        # hidden by default
        dlg._params_btn.setChecked(True)            # -> _toggle_view
        assert dlg._table.isHidden() is False
        assert dlg._plots_box.isHidden() is True
        assert dlg._table.rowCount() > 30

        q = state_with_results.results.beam.alive_particles
        mom = compute_moments(q)
        twx = compute_twiss_from_particles(q, "x")
        assert float(_cell(dlg, "σ_x")) == pytest.approx(
            mom["sigma_x"], rel=1e-4)
        assert float(_cell(dlg, "α_x")) == pytest.approx(
            twx["alpha"], rel=1e-4)
        assert float(_cell(dlg, "ε_x (geometric)")) == pytest.approx(
            twx["emittance"], rel=1e-4)
        # ref-dependent rows present (exit beam carries its ref)
        assert float(_cell(dlg, "W_kin")) == pytest.approx(2.12)
        assert _cell(dlg, "species") == "H-"

        # toggling back restores the plots
        dlg._params_btn.setChecked(False)
        assert dlg._plots_box.isHidden() is False
        assert dlg._table.isHidden() is True
    finally:
        dlg.deleteLater()


def test_table_follows_location_selector(state_with_results):
    from linac_gen.diagnostics.moments import compute_moments
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup

    dlg = _PhaseSpacePopup(None, state_with_results)
    try:
        dlg.refresh(state_with_results.results)
        dlg._params_btn.setChecked(True)
        exit_sigma = float(_cell(dlg, "σ_x"))
        idx = dlg._location.findData(300.0)
        dlg._location.setCurrentIndex(idx)          # triggers _redraw
        snap_sigma = float(_cell(dlg, "σ_x"))
        expect = compute_moments(
            state_with_results.results.alive_at(300.0))["sigma_x"]
        assert snap_sigma == pytest.approx(expect, rel=1e-4)
        assert snap_sigma != pytest.approx(exit_sigma, rel=1e-3)
        # snapshot ref travels with the snapshot -> W_kin row present
        assert float(_cell(dlg, "W_kin")) == pytest.approx(2.12)
    finally:
        dlg.deleteLater()


def test_refresh_none_with_table_visible_is_safe(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup

    st = AppState()
    dlg = _PhaseSpacePopup(None, st)
    try:
        dlg._params_btn.setChecked(True)
        dlg.refresh(None)                           # must not raise
        assert dlg._table.rowCount() >= 1
    finally:
        dlg.deleteLater()


def test_toggle_off_clears_table_so_exports_never_stale(
        state_with_results):
    """Adversarial-review finding: _collect_panels discovers hidden
    QTableWidgets, so a populated table left behind after toggling back
    to plots would attach STALE (wrong-location) data to a plots-view
    export.  Toggle-off must clear the rows; the empty table is then
    skipped and plots-view exports contain no table panel."""
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup

    dlg = _PhaseSpacePopup(None, state_with_results)
    try:
        dlg.refresh(state_with_results.results)
        dlg._params_btn.setChecked(True)            # fill at exit
        assert dlg._table.rowCount() > 30
        dlg._params_btn.setChecked(False)           # back to plots
        assert dlg._table.rowCount() == 0
        # change location in plots view — no stale table in the export
        idx = dlg._location.findData(300.0)
        dlg._location.setCurrentIndex(idx)
        curve_names = [c.name for p in dlg._collect_panels()
                       for c in getattr(p, "curves", []) or []]
        assert not any("value" in str(n).lower() for n in curve_names), \
            curve_names
    finally:
        dlg.deleteLater()


def test_ctrl_s_discovers_the_table(state_with_results):
    """_collect_panels auto-discovers QTableWidget children, so the
    parameters table exports with the universal Ctrl+S path."""
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup

    dlg = _PhaseSpacePopup(None, state_with_results)
    try:
        dlg.refresh(state_with_results.results)
        dlg._params_btn.setChecked(True)
        panels = dlg._collect_panels()
        curve_names = [c.name for p in panels
                       for c in getattr(p, "curves", []) or []]
        # the table's numeric "Value" column must appear as a curve
        assert any("value" in str(n).lower() for n in curve_names), \
            curve_names
    finally:
        dlg.deleteLater()
