"""Regression tests: orbit correction runs in a worker on a snapshot and
applies its kicks to the live lattice as one undoable command.

The old handler ran the full correction INLINE on the GUI thread
(freezing the app) while mutating the live lattice in place with no
undo and originally no dirty flag.
"""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("PyQt6")

from linac_gen.core.config import BeamConfig  # noqa: E402
from linac_gen.core.lattice import Lattice  # noqa: E402
from linac_gen.elements.drift import Drift  # noqa: E402
from linac_gen.elements.lattice_commands import AdjustSteerer  # noqa: E402
from linac_gen.elements.marker import Marker  # noqa: E402
from linac_gen.elements.steerer import Steerer  # noqa: E402


def _pair_lattice() -> Lattice:
    lat = Lattice()
    lat.add(Drift("D_pre", 100.0))
    lat.add(AdjustSteerer("ADJ_1", diag_n=1, vmax=0.0, first_step=1e-4))
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D_post", 100.0))
    lat.add(Marker("BPM_1", is_bpm=True))
    lat.add(Drift("D_after", 100.0))
    return lat


def _beam_cfg() -> BeamConfig:
    from dataclasses import replace
    return replace(BeamConfig(), n_particles=100)


@pytest.fixture()
def tab_env(qapp, monkeypatch):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs import lattice_tab as lt

    # No modal dialogs in tests.
    monkeypatch.setattr(lt.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(lt.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    st = AppState()
    st.set_lattice(_pair_lattice(), None)
    st.set_beam_config(_beam_cfg())
    tab = lt.LatticeTab(st)
    yield st, tab, lt
    tab.deleteLater()


def _run_worker_sync(lt, lattice, cfg):
    w = lt._CorrectionWorker(copy.deepcopy(lattice), copy.deepcopy(cfg))
    out = {}
    w.finished_ok.connect(lambda r: out.setdefault("res", r))
    w.failed.connect(lambda m: out.setdefault("fail", m))
    w.cancelled.connect(lambda: out.setdefault("cancelled", True))
    w.run()
    return out


def test_correction_applies_kicks_via_undoable_command(tab_env):
    st, tab, lt = tab_env
    out = _run_worker_sync(lt, st.lattice, st.beam_config)
    assert "res" in out, out
    res = out["res"]
    assert res["n_pairs"] == 1 and res["kicks"]

    tab._corr_lattice_at_launch = st.lattice
    tab._on_correction_done(res)

    steerer = next(e for e in st.lattice.elements if isinstance(e, Steerer))
    k = res["kicks"]["STEER_1"]
    assert steerer.bx_l == pytest.approx(k["bx_l"])
    assert steerer.by_l == pytest.approx(k["by_l"])
    # One undoable step; dirty flagged through the bus.
    assert st.bus.dirty
    assert st.bus.can_undo
    st.bus.undo()
    assert steerer.bx_l == 0.0 and steerer.by_l == 0.0


def test_stale_correction_result_is_discarded(tab_env):
    st, tab, lt = tab_env
    out = _run_worker_sync(lt, st.lattice, st.beam_config)
    res = out["res"]

    tab._corr_lattice_at_launch = object()   # lattice changed mid-run
    tab._on_correction_done(res)

    steerer = next(e for e in st.lattice.elements if isinstance(e, Steerer))
    assert steerer.bx_l == 0.0 and steerer.by_l == 0.0
    assert not st.bus.can_undo


def test_lattice_switch_mid_correction_restarts_instead_of_cancelling(
        tab_env, monkeypatch):
    """Review finding: the auto-correct-on-load hook fired while a
    manual correction ran was swallowed by the 'second click = cancel'
    branch, so the NEW lattice was never corrected."""
    st, tab, lt = tab_env
    monkeypatch.setattr(lt._CorrectionWorker, "start", lambda self: None)

    class _FakeRunning:
        def __init__(self, lattice):
            self._live_lattice_at_launch = lattice
            self.stopped = False

        def isRunning(self):
            return True

        def request_stop(self):
            self.stopped = True

        def requestInterruption(self):
            pass

    old_lattice = st.lattice
    prev = _FakeRunning(old_lattice)
    tab._corr_worker = prev

    # Same lattice → the click is a user cancel: no relaunch.
    tab._on_correct_orbit()
    assert prev.stopped
    assert tab._corr_worker is prev

    # Lattice replaced (project load) → stale run cancelled AND a fresh
    # worker launched for the new lattice.
    prev2 = _FakeRunning(object())        # launched against another lattice
    tab._corr_worker = prev2
    tab._on_correct_orbit()
    assert prev2.stopped
    assert isinstance(tab._corr_worker, lt._CorrectionWorker)
    assert tab._corr_worker._live_lattice_at_launch is st.lattice


def test_predecessor_signals_are_ignored(tab_env):
    """A relaunch-cancelled predecessor finishing late must not touch
    the UI or apply kicks — handlers check the sender identity."""
    from PyQt6.QtCore import QObject, pyqtSignal

    st, tab, lt = tab_env

    class _OldWorker(QObject):
        finished_ok = pyqtSignal(object)
        _live_lattice_at_launch = None

    old = _OldWorker()
    old._live_lattice_at_launch = st.lattice
    old.finished_ok.connect(tab._on_correction_done)
    tab._corr_worker = object()           # a different, current worker

    out = _run_worker_sync(lt, st.lattice, st.beam_config)
    old.finished_ok.emit(out["res"])      # late delivery from the old one

    steerer = next(e for e in st.lattice.elements
                   if isinstance(e, Steerer))
    assert steerer.bx_l == 0.0            # nothing applied
    assert not st.bus.can_undo


def test_correction_worker_cancel(tab_env):
    st, tab, lt = tab_env
    w = lt._CorrectionWorker(copy.deepcopy(st.lattice),
                             copy.deepcopy(st.beam_config))
    out = {}
    w.finished_ok.connect(lambda r: out.setdefault("res", r))
    w.cancelled.connect(lambda: out.setdefault("cancelled", True))
    w.request_stop()
    w.run()
    assert out == {"cancelled": True}
