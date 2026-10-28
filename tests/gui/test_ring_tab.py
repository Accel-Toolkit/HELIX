"""Ring mode in the GUI: Simulate → Ring Mode adds the Ring tab (a linac
session never sees it), a run of 10 turns goes through the real worker, Stop
ends a run between turns, and a project keeps a "ring" section only once ring
mode has been used."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

FIXTURE = Path(__file__).resolve().parents[1] / "ring" / "fixtures" / "simple_booster.madx"


def _ring_on(win):
    act = win._toolbar._ring_action
    if not act.isChecked():
        act.trigger()                     # the real menu action
    win.pump(0.1)
    tab = win.ring_tab
    # (a run takes the Beam tab's beam: 300 of its particles keep the runs
    # quick — its form only, the project's ring settings stay as the tests
    # expect)
    win.beam_tab._npart.setValue(300)
    return tab


def test_ring_mode_adds_and_hides_the_ring_tab(win):
    from linac_gen_gui.interphase.state import TABS
    act = win._toolbar._ring_action
    assert act.isCheckable() and not act.isChecked()
    assert win.ring_tab is None and win._tabs.count() == len(TABS)
    tab = _ring_on(win)
    assert win.state.ring_mode and tab is not None
    i = win._tabs.indexOf(tab)
    assert i == len(TABS) and win._tabs.isTabVisible(i) and win._tabs.currentIndex() == i
    act.trigger()                         # off again: the tab is hidden, not destroyed
    win.pump(0.1)
    assert not win.state.ring_mode and not win._tabs.isTabVisible(i)
    assert win._tabs.count() == len(TABS) + 1 and win.ring_tab is tab


def test_a_ten_turn_run_through_the_worker(win):
    win.open_lattice(FIXTURE)
    lat = win.state.lattice
    before = [(type(e).__name__, e.name, e.length, getattr(e, "frequency", None))
              for e in lat.elements]
    tab = _ring_on(win)
    assert "Closes on itself" in tab._verdict.text() and "RF harmonic 84" in tab._verdict.text()
    tab._turns.setValue(10)
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res is not None and res.turns_done == 10 and not res.truncated
    assert "10 of 10 turn(s)" in tab._summary.text()
    # the run is at the deck's beam (800 MeV protons), the Beam tab's linac
    # default generated there: the Booster's tunes, a finite beam
    assert "Reference: proton at 800 MeV" in tab._verdict.text()
    assert res.optics["qx"] % 1 == pytest.approx(0.71, abs=1e-3)
    assert res.optics["qy"] % 1 == pytest.approx(0.81, abs=1e-3)
    assert (res.sigma[:, :4] < 1.0).all() and list(res.n_alive) == [300] * 11
    assert tab._progress.value() == 100 and tab._run_btn.isEnabled()
    # the run used a copy: the window's lattice is the same object, unchanged
    assert win.state.lattice is lat
    assert [(type(e).__name__, e.name, e.length, getattr(e, "frequency", None))
            for e in lat.elements] == before


def test_stop_ends_a_run_between_turns(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(5_000_000)
    tab._run_btn.click()
    assert win.pump_until(lambda: tab._progress.value() >= 0 and
                          tab._turn_label.text().startswith("turn"), 60_000)
    tab._stop_btn.click()
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res is not None and res.truncated and 0 < len(res.turns) < 5_000_000
    assert "stopped after" in tab._summary.text()


def test_shutdown_joins_a_running_ring_worker(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(5_000_000)
    tab._run_btn.click()
    win.pump(0.2)
    workers = tab.shutdown_begin()
    assert workers and all(w.wait(10_000) for w in workers)


def test_a_project_keeps_ring_mode_only_once_it_was_used(win):
    win.open_lattice(FIXTURE)
    assert "ring" not in win._collect_project_dict()      # a linac session
    tab = _ring_on(win)
    tab._turns.setValue(7)
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    data = win._collect_project_dict()
    assert data["ring"]["enabled"] is True and data["ring"]["config"]["n_turns"] == 7
    # a project without a "ring" section switches ring mode off …
    plain = {k: v for k, v in data.items() if k != "ring"}
    win._apply_project_dict(plain, silent=True)
    assert not win.state.ring_mode and "ring" not in win._collect_project_dict()
    # … and one with it switches it back on, with its settings
    win._apply_project_dict(data, silent=True)
    assert win.state.ring_mode and win.ring_tab._turns.value() == 7


def test_the_window_starts_without_ring_mode(tmp_path):
    # a linac session never imports ring mode or numba: the Ring tab and the
    # ring worker load them only when ring mode is used (fresh process,
    # sandboxed settings, offscreen)
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    code = ("import os, sys\n"
            "from PyQt6.QtWidgets import QApplication\n"
            "app = QApplication([])\n"
            "from linac_gen_gui.interphase.app import InterphaseWindow\n"
            "InterphaseWindow._launch_update_check = lambda self: None\n"
            "w = InterphaseWindow()\n"
            "bad = sorted(m for m in sys.modules if m.startswith('linac_gen.ring') or m == 'numba')\n"
            "print('IMPORTED:' + ','.join(bad), flush=True)\n"
            "os._exit(0)\n")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.pathsep.join([str(root), str(root / "gui")]),
                "QT_QPA_PLATFORM": "offscreen", "HELIX_QSETTINGS_DIR": str(tmp_path),
                "HELIX_ASSIST_NO_PREWARM": "1", "OMP_NUM_THREADS": "1"})
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                         text=True, env=env, timeout=300)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("IMPORTED:")]
    assert lines, (out.returncode, out.stderr[-2000:])
    assert lines[-1] == "IMPORTED:", f"window start-up imported ring-mode modules: {lines[-1]}"


def test_a_new_lattice_during_a_run_keeps_no_old_results(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(5_000_000)
    tab._run_btn.click()
    win.pump(0.3)
    worker = tab._worker
    win.state.mark_project_clean()        # (no "unsaved changes" prompt in the way)
    win.open_lattice(Path(__file__).resolve().parents[2] / "examples" / "fodo_cell.dat")
    # (another lattice stops the run, and the Run page says why)
    assert tab._worker is None and worker._stop_event.is_set()
    assert tab._turn_label.text() == "Stopped: the lattice was replaced."
    assert worker.wait(10_000)
    win.pump(0.5)
    assert win.state.ring_results is None and tab._run_btn.isEnabled()
    assert tab._turn_label.text() == "Stopped: the lattice was replaced."


def test_an_edit_during_a_run_leaves_the_run_going(win, monkeypatch):
    # the command bus sends the same lattice again for every edit (the
    # inspector, undo and redo, the assistant's changes; here a value set to
    # itself): the run goes on — on its own copy — and its results arrive;
    # the optics and the ramp preview, of the lattice as it was, go
    import threading

    from linac_gen_gui.interphase import workers
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    reached, go_on = threading.Event(), threading.Event()
    real = workers.RingWorker._on_turn

    def held(self, turn, n):
        reached.set()
        go_on.wait(60)                    # (the run waits at its first turn)
        real(self, turn, n)

    monkeypatch.setattr(workers.RingWorker, "_on_turn", held)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(4)
    tab._run_btn.click()
    worker = tab._worker
    try:
        assert reached.wait(120)
        tab._optics = {"shown": True}
        tab._programs._preview = {"shown": True}
        el = win.state.lattice.elements[0]
        old = getattr(el, "aperture", None)
        win.state.bus.do(ParamChangeCommand(el, "aperture", old, old))
        win.pump(0.2)
        assert tab._worker is worker and worker.isRunning()
        assert not worker._stop_event.is_set() and tab._stop_btn.isEnabled()
        assert tab._optics is None and tab._programs._preview is None
        win.state.bus.undo()              # (the same lattice again)
        win.pump(0.1)
        assert tab._worker is worker and not worker._stop_event.is_set()
    finally:
        go_on.set()
    win.wait_worker(worker)
    res = win.state.ring_results
    assert res is not None and res.turns_done == 4 and not res.truncated
    assert "4 of 4 turn(s)" in tab._summary.text() and tab._run_btn.isEnabled()


def test_a_stop_before_the_first_turn_keeps_the_last_results(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(5)
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    first = win.state.ring_results
    assert first.turns_done == 5
    tab._turns.setValue(1000)
    tab._run_btn.click()
    tab._stop_btn.click()                 # (before the thread reaches turn 1)
    win.wait_worker(tab._worker)
    assert win.state.ring_results is first


def test_ring_settings_mark_the_project_changed(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    win.state.mark_project_clean()
    tab._turns.setValue(77)
    assert win.state.project_dirty
    # (with the Run page's start: always saved — an earlier version's project
    # without it started from a beam matched to the ring)
    assert win._collect_project_dict()["ring"]["config"] == {"n_turns": 77, "source": "beam"}


def test_invalid_ring_settings_in_a_project_are_reported(win):
    win.open_lattice(FIXTURE)
    data = win._collect_project_dict()
    data["ring"] = {"enabled": True, "config": {"n_turns": "abc"}}
    warns = win._apply_project_dict(data, silent=True)
    assert any("ring settings are not valid" in w for w in warns)
    assert win.state.ring_mode and win.ring_tab is not None
    assert win.ring_tab._turns.value() == 10


def test_the_assistant_does_not_reach_the_hidden_ring_tab(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    assert "Ring" in win.assistant_tab_labels()
    win._toolbar._ring_action.trigger()          # off: the tab is hidden
    win.pump(0.1)
    assert "Ring" not in win.assistant_tab_labels()
    assert win.show_tab("Ring") is None and win._tabs.currentWidget() is not tab
