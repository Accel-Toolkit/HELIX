"""Numerics tab "CPU threads": every CPU by default; a choice applies at
once and is saved for this computer, not in the project; the saved choice
applies at the next start; HELIX_THREADS at launch wins; and a run started
from the toolbar applies the setting inside its own worker thread."""
from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("PyQt6")

from tests.helpers import kept_thread_setting  # noqa: E402

KEY = "compute/threads"
FODO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))


@pytest.fixture
def settings():
    from linac_gen_gui.interphase.app_settings import make_settings
    s = make_settings("HELIX", "linac_gen_gui")
    s.remove(KEY)
    yield s
    s.remove(KEY)


@pytest.fixture
def threads(settings, monkeypatch):
    with kept_thread_setting() as t:
        t._setting = t._requested = t._env_before = None
        monkeypatch.delenv(t.ENV, raising=False)
        # as the GUI starts: its own OMP_NUM_THREADS cap, marked as such
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.setenv(t.GUI_OMP_CAP, "1")
        yield t


def _tab():
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.convergence_tab import ConvergenceTab
    return ConvergenceTab(AppState())


def test_default_is_every_cpu(qapp, threads):
    tab = _tab()
    box, n = tab._cpu_threads, threads.available_threads()
    assert box.itemText(0) == f"All ({n})" and box.currentIndex() == 0
    assert [box.itemData(i) for i in range(box.count())] == list(range(n + 1))
    assert threads.get_threads() == n
    assert "identical" in box.toolTip()
    tab.deleteLater()


def test_a_choice_applies_at_once_and_is_saved(qapp, threads, settings):
    tab = _tab()
    box = tab._cpu_threads
    box.setCurrentIndex(box.findData(3))
    assert threads.get_threads() == 3 and settings.value(KEY) == "3"
    box.setCurrentIndex(0)
    assert threads.requested_threads() is None
    assert threads.get_threads() == threads.available_threads()
    assert settings.value(KEY) == "all"
    tab.deleteLater()


def test_the_saved_choice_applies_at_start(qapp, threads, settings):
    settings.setValue(KEY, "2")
    tab = _tab()
    assert threads.get_threads() == 2 and tab._cpu_threads.currentData() == 2
    tab.deleteLater()


def test_an_unreadable_saved_choice_means_every_cpu(qapp, threads, settings):
    settings.setValue(KEY, "lots")
    tab = _tab()
    assert tab._cpu_threads.currentIndex() == 0
    assert threads.get_threads() == threads.available_threads()
    tab.deleteLater()


def test_helix_threads_at_launch_wins(qapp, threads, settings, monkeypatch):
    settings.setValue(KEY, "2")
    monkeypatch.setenv(threads.ENV, "5")
    tab = _tab()
    assert threads.get_threads() == 5 and tab._cpu_threads.currentData() == 5
    assert f"{threads.ENV}=5" in tab._cpu_threads.toolTip()
    tab.deleteLater()


def test_a_users_omp_num_threads_at_launch(qapp, threads, settings,
                                           monkeypatch):
    """Started with the user's own OMP_NUM_THREADS (no GUI mark): HELIX
    follows it, above the saved choice, and says so."""
    settings.setValue(KEY, "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv(threads.GUI_OMP_CAP)
    tab = _tab()
    assert threads.get_threads() == 3 and tab._cpu_threads.currentData() == 3
    assert "OMP_NUM_THREADS=3" in tab._cpu_threads.toolTip()
    tab.deleteLater()


def test_the_guis_own_omp_cap_is_ignored(qapp, threads):
    tab = _tab()
    assert threads.get_threads() == threads.available_threads()
    assert tab._cpu_threads.currentIndex() == 0
    tab.deleteLater()


def test_more_than_this_computer_has(qapp, threads, settings):
    n = threads.available_threads() + 3
    settings.setValue(KEY, str(n))
    tab = _tab()
    assert tab._cpu_threads.currentData() == n and threads.get_threads() == n
    tab.deleteLater()


def test_run_from_the_toolbar_applies_it_in_the_worker(qapp, win, threads,
                                                       monkeypatch):
    """Toolbar MP run through the real slot: the setting is applied in the
    run's worker thread, at the chosen count, and stays out of the
    project file."""
    from linac_gen.parallel import threads as mod
    seen = []
    real = mod.apply_in_this_thread

    def spy():
        n = real()
        seen.append((threading.current_thread() is threading.main_thread(), n))
        return n
    monkeypatch.setattr(mod, "apply_in_this_thread", spy)
    box = win.convergence_tab._cpu_threads
    box.setCurrentIndex(box.findData(2))
    win.open_lattice(FODO)
    win.beam_tab._npart.setValue(300)
    win.beam_tab._apply()
    win.pump()
    win._run_mp()
    win.wait_worker(win._mp_worker)
    assert type(win.state.results).__name__ == "DiagnosticRecorder"
    assert seen and all(not on_main and n == 2 for on_main, n in seen), seen
    assert win.state.results.threads_info["threads"] == 2   # in the worker

    def keys(d):
        for k, v in d.items():
            yield k
            if isinstance(v, dict):
                yield from keys(v)
    assert not [k for k in keys(win._collect_project_dict(None))
                if "thread" in str(k).lower()]          # not in the project
