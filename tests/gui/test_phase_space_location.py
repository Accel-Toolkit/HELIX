"""Phase Space popup: the location dropdown lists captured snapshots and
selecting one plots that distribution (not just the exit beam)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _recorder_with_snapshots():
    """A real DiagnosticRecorder carrying two snapshots + the exit beam,
    the same shape the GUI receives from Simulation.run()."""
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.core.beam import Beam
    rec = DiagnosticRecorder()
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
    rng = np.random.default_rng(0)
    # two snapshots at s = 150, 300 mm, labelled Q1 / Q2
    rec.s = [0.0, 150.0, 300.0, 400.0]
    rec.element_names = ["INPUT", "Q1", "Q2", "D3"]
    for s in (150.0, 300.0):
        b = Beam(ref=ref, n_particles=200, current=0.0)
        b.particles[:] = rng.normal(0, 0.3, (200, 6))
        rec.save_snapshot(b, s)
    # exit beam
    exit_beam = Beam(ref=ref, n_particles=200, current=0.0)
    exit_beam.particles[:] = rng.normal(0, 0.3, (200, 6))
    rec.beam = exit_beam
    return rec


@pytest.fixture()
def state_with_results(qapp):
    from linac_gen_gui.interphase.state import AppState
    st = AppState()
    st.set_results(_recorder_with_snapshots())
    return st


def test_location_combo_lists_exit_and_snapshots(state_with_results):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, state_with_results)
    try:
        dlg.refresh(state_with_results.results)
        labels = [dlg._location.itemText(i)
                  for i in range(dlg._location.count())]
        assert labels[0].startswith("exit")
        assert dlg._location.count() == 3            # exit + 2 snapshots
        assert any("Q1" in t for t in labels)
        assert any("Q2" in t for t in labels)
        # userData: exit -> None, snapshots -> s-position
        assert dlg._location.itemData(0) is None
        assert set(dlg._location.itemData(i) for i in (1, 2)) == {150.0, 300.0}
    finally:
        dlg.deleteLater()


def test_selecting_snapshot_plots_it(state_with_results):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, state_with_results)
    try:
        dlg.refresh(state_with_results.results)
        # select the Q2 snapshot (s=300) and confirm the source resolves to
        # that snapshot's particles, not the exit beam
        idx = dlg._location.findData(300.0)
        assert idx > 0
        dlg._location.setCurrentIndex(idx)             # triggers _redraw
        chosen = dlg._current_particles()
        expect = state_with_results.results.alive_at(300.0)
        assert chosen.shape == expect.shape
        np.testing.assert_array_equal(chosen, expect)
    finally:
        dlg.deleteLater()


def test_default_is_exit_beam(state_with_results):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    dlg = _PhaseSpacePopup(None, state_with_results)
    try:
        dlg.refresh(state_with_results.results)
        # index 0 = exit (final) -> uses results.beam, not a snapshot
        assert dlg._location.currentData() is None
        chosen = dlg._current_particles()
        np.testing.assert_array_equal(
            chosen, state_with_results.results.beam.alive_particles)
    finally:
        dlg.deleteLater()


def test_no_snapshots_leaves_only_exit(qapp):
    """A run with no snapshots: dropdown has only 'exit (final)'."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.core.beam import Beam
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseSpacePopup
    rec = DiagnosticRecorder()
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
    rec.beam = Beam(ref=ref, n_particles=50, current=0.0)
    st = AppState(); st.set_results(rec)
    dlg = _PhaseSpacePopup(None, st)
    try:
        dlg.refresh(rec)
        assert dlg._location.count() == 1
        assert dlg._location.itemData(0) is None
    finally:
        dlg.deleteLater()
