"""Regression tests: closing the window stops background workers.

closeEvent used to only run the unsaved-changes prompt — live QThreads
were then destroyed by Qt on the way out, aborting the process
("QThread: Destroyed while thread is still running").
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.close()
    w.deleteLater()


def test_close_joins_running_envelope_worker(win, mini_lattice):
    """A real envelope run in flight at close time must be stopped and
    joined before the close is accepted."""
    win.state.set_lattice(mini_lattice, None)
    win._run_envelope()
    worker = win._envelope_worker
    assert worker is not None and worker.isRunning()

    win.close()

    assert not worker.isRunning()          # joined, not abandoned
    from linac_gen_gui.interphase import app as app_mod
    assert app_mod._SHUTTING_DOWN is True  # teardown latch set
    assert app_mod._PARKED_WORKERS == []   # nothing needed parking


def test_close_collects_tab_workers(win, mini_lattice, monkeypatch):
    """shutdown_begin() hooks are consulted on every worker-owning tab."""
    calls = []
    for tab_name in ("matching_tab", "convergence_tab", "errors_tab",
                     "failures_tab", "surrogates_tab", "results_tab",
                     "lattice_tab"):
        tab = getattr(win, tab_name)
        monkeypatch.setattr(
            tab, "shutdown_begin",
            lambda name=tab_name: (calls.append(name), [])[1],
            raising=False)

    win.state.set_lattice(mini_lattice, None)
    win.close()
    assert set(calls) == {"matching_tab", "convergence_tab", "errors_tab",
                          "failures_tab", "surrogates_tab", "results_tab",
                          "lattice_tab"}


class _StubbornWorker:
    """Looks like a running worker whose wait() always times out."""

    def __init__(self):
        from PyQt6.QtCore import QObject, pyqtSignal

        class _Sig(QObject):
            finished = pyqtSignal()
        self._sig = _Sig()
        self.finished = self._sig.finished
        self.running = True
        self.stop_requested = False

    def isRunning(self):
        return self.running

    def request_stop(self):
        self.stop_requested = True

    def requestInterruption(self):
        pass

    def wait(self, _ms):
        return False

    def disconnect(self):
        pass


def test_retire_timeout_parks_worker(win):
    """Review finding: _retire_worker's timeout path used to let the
    caller rebind the only reference to a live QThread → GC destroys a
    running thread → qFatal abort.  Stragglers are now parked."""
    from linac_gen_gui.interphase import app as app_mod

    w = _StubbornWorker()
    win._retire_worker(w, timeout_ms=1)
    assert w.stop_requested
    assert w in app_mod._PARKED_WORKERS
    # Thread eventually exits → pruned.
    w.running = False
    w.finished.emit()
    assert w not in app_mod._PARKED_WORKERS


def test_detach_nulls_worker_attributes(win):
    """Review finding: a FINISHED worker kept in _envelope_worker still
    passed the sender-identity guard, so its queued finished_ok (posted
    before the lattice switch) delivered stale results onto the new
    lattice.  Detach now nulls the attributes."""
    w = _StubbornWorker()
    w.running = False                      # already finished
    win._envelope_worker = w
    win._detach_workers_for_new_lattice()
    assert win._envelope_worker is None
    assert win._mp_worker is None


def test_new_window_resets_shutdown_latch(qapp):
    from linac_gen_gui.interphase import app as app_mod
    from linac_gen_gui.interphase.app import InterphaseWindow

    app_mod._SHUTTING_DOWN = True
    w = InterphaseWindow()
    try:
        assert app_mod._SHUTTING_DOWN is False
    finally:
        w.close()
        w.deleteLater()
