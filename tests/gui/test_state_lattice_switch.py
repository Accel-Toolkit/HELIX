"""Regression tests: switching lattices must invalidate stale run state.

Covers two related bugs:

* ``AppState.set_lattice`` used to leave ``_results`` (and the error/
  failure study bags) from the previous lattice in place — every
  results consumer then rendered old physics against the new lattice.
* A retired envelope/MP worker could still deliver a queued
  ``finished_ok`` emission after ``disconnect()`` (the event is posted
  before the disconnect); the app-side handlers now drop emissions
  whose sender is not the *current* worker.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal  # noqa: E402


def test_set_lattice_clears_results_before_signals(qapp, mini_lattice):
    from linac_gen_gui.interphase.state import AppState

    st = AppState()
    st.set_results(object())
    st.error_study_results = object()
    st.failure_study_results = object()

    seen_at_lattice_changed = []
    st.lattice_changed.connect(
        lambda _lat: seen_at_lattice_changed.append(st.results))
    results_emissions = []
    st.results_changed.connect(results_emissions.append)

    st.set_lattice(mini_lattice, "/tmp/switch.dat")

    # Results dropped before ANY consumer ran.
    assert st.results is None
    assert all(r is None for r in seen_at_lattice_changed)
    # And broadcast so results views clear themselves.
    assert results_emissions and results_emissions[-1] is None
    # Study bags are lattice-scoped too.
    assert st.error_study_results is None
    assert st.failure_study_results is None
    # The path argument still round-trips (matching-Apply regression).
    assert st.lattice_path == "/tmp/switch.dat"


def test_stale_worker_emission_is_dropped(qapp, mini_lattice):
    """An emission from a worker that is no longer current must not
    overwrite state.results (queued-delivery hole after disconnect)."""
    from linac_gen_gui.interphase.app import InterphaseWindow

    class _FakeWorker(QObject):
        finished_ok = pyqtSignal(object)

    win = InterphaseWindow()
    try:
        win.state.set_lattice(mini_lattice, None)

        stale = _FakeWorker()
        stale.finished_ok.connect(win._env_done)
        # The window's current envelope worker is None (or another object)
        # — the stale sender must be ignored wholesale.
        marker = object()
        stale.finished_ok.emit(marker)
        assert win.state.results is not marker
        assert win.state.results is None
    finally:
        win.close()
        win.deleteLater()
