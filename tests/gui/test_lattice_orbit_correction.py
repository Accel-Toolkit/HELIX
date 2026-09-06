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


def test_beam_lost_result_applies_no_kicks_and_warns(tab_env, monkeypatch):
    """D2: on ``status == "beam_lost"`` the GUI must refuse to apply the
    beam-killing kicks — nothing pushed to the bus, warning names the
    element."""
    st, tab, lt = tab_env
    captured = {}
    monkeypatch.setattr(
        lt.QMessageBox, "warning",
        staticmethod(lambda parent, title, text, *a, **k:
                     captured.setdefault("text", text)))

    res = {
        "kicks": {"STEER_1": {"bx_l": 1e-3, "by_l": 1e-3}},
        "history": [{"iter": 1, "rms_orbit_mm": float("nan"),
                     "n_saturated": 0, "n_dead_bpms": 1,
                     "transmission_pct": 0.0, "beam_lost_at": "D_post",
                     "stop_reason": "beam_lost"}],
        "method": "one_to_one", "n_pairs": 1,
        "status": "beam_lost", "converged": False,
        "beam_lost_at": "D_post",
    }
    tab._corr_lattice_at_launch = st.lattice
    tab._on_correction_done(res)

    steerer = next(e for e in st.lattice.elements if isinstance(e, Steerer))
    assert steerer.bx_l == 0.0 and steerer.by_l == 0.0
    assert not st.bus.can_undo
    assert not st.bus.dirty
    assert "D_post" in captured["text"]
    assert "No kicks" in captured["text"]


def test_worker_beam_lost_end_to_end_leaves_steerers_untouched(tab_env):
    """Through the real worker: the original demo geometry produces a
    beam-lost result and the done handler applies nothing."""
    import numpy as np

    from tests.errors.test_correction_dead_beam import (
        _demo_cfg, _original_demo_lattice, _plant_demo_misalignments)

    st, tab, lt = tab_env
    lat = _original_demo_lattice()
    _plant_demo_misalignments(lat)
    st.set_lattice(lat, None)
    st.set_beam_config(_demo_cfg(300))

    out = _run_worker_sync(lt, st.lattice, st.beam_config)
    assert "res" in out, out
    assert out["res"]["status"] == "beam_lost"
    assert out["res"]["beam_lost_at"]

    tab._corr_lattice_at_launch = st.lattice
    tab._on_correction_done(out["res"])
    for e in st.lattice.elements:
        if isinstance(e, Steerer):
            assert e.bx_l == 0.0 and e.by_l == 0.0
    assert not st.bus.can_undo


def test_downstream_loss_is_named_in_converged_summary(tab_env, monkeypatch):
    """Downstream-only loss (after the last BPM): the run converges and
    the kicks apply, but the summary must NAME the loss element —
    ``status: "converged"`` with ``beam_lost_at`` non-None used to be
    completely silent about the dead beam."""
    st, tab, lt = tab_env
    captured = {}
    monkeypatch.setattr(
        lt.QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k:
                     captured.setdefault("text", text)))

    res = {
        "kicks": {"STEER_1": {"bx_l": 1e-4, "by_l": -1e-4}},
        "history": [{"iter": 1, "rms_orbit_mm": 0.01,
                     "n_saturated": 0, "n_dead_bpms": 0,
                     "transmission_pct": 0.0, "beam_lost_at": "COLL_KILL",
                     "stop_reason": "converged"}],
        "method": "one_to_one", "n_pairs": 1,
        "status": "converged", "converged": True,
        "beam_lost_at": "COLL_KILL",
    }
    tab._corr_lattice_at_launch = st.lattice
    tab._on_correction_done(res)

    # Kicks ARE applied (status vocabulary unchanged — only reporting).
    from linac_gen.elements.steerer import Steerer
    steerer = next(e for e in st.lattice.elements if isinstance(e, Steerer))
    assert steerer.bx_l == pytest.approx(1e-4)
    assert st.bus.can_undo
    # ... and the loss is visible in the summary.
    assert "COLL_KILL" in captured["text"]
    assert "Status: converged" in captured["text"]


def test_beam_lost_svd_advice_is_method_aware(tab_env, monkeypatch):
    """When the FAILED method was already svd, the warning must not
    advise "switch method to SVD"; for one_to_one it still does."""
    st, tab, lt = tab_env
    captured = {}
    monkeypatch.setattr(
        lt.QMessageBox, "warning",
        staticmethod(lambda parent, title, text, *a, **k:
                     captured.__setitem__("text", text)))

    def _res(method):
        return {
            "kicks": {"STEER_1": {"bx_l": 1e-3, "by_l": 1e-3}},
            "history": [{"iter": 1, "rms_orbit_mm": float("nan"),
                         "n_saturated": 0, "n_dead_bpms": 1,
                         "transmission_pct": 0.0, "beam_lost_at": "D_post",
                         "stop_reason": "beam_lost"}],
            "method": method, "n_pairs": 1,
            "status": "beam_lost", "converged": False,
            "beam_lost_at": "D_post",
        }

    tab._corr_lattice_at_launch = st.lattice
    tab._on_correction_done(_res("svd"))
    assert "switch method to SVD" not in captured["text"]
    assert "D_post" in captured["text"]

    tab._on_correction_done(_res("one_to_one"))
    assert "switch method to SVD" in captured["text"]


def test_downstream_loss_end_to_end_through_real_worker(tab_env, monkeypatch):
    """End-to-end through the REAL worker: a killer aperture strictly
    after the last BPM converges (alive BPM readings) yet reports the
    loss element in the summary popup; no-loss runs stay silent."""
    st, tab, lt = tab_env
    captured = {}
    monkeypatch.setattr(
        lt.QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k:
                     captured.__setitem__("text", text)))

    # Regime 1 — no loss anywhere: summary carries no loss warning.
    out = _run_worker_sync(lt, st.lattice, st.beam_config)
    assert out["res"]["beam_lost_at"] is None
    tab._corr_lattice_at_launch = st.lattice
    tab._on_correction_done(out["res"])
    assert "beam lost" not in captured["text"]
    assert "Status: converged" in captured["text"]
    st.bus.undo()

    # Regime 2 — killer aperture after the last BPM.
    lat = _pair_lattice()
    lat.add(Drift("COLL_KILL", 100.0, aperture=1e-6))
    st.set_lattice(lat, None)
    out2 = _run_worker_sync(lt, st.lattice, st.beam_config)
    res2 = out2["res"]
    assert res2["status"] == "converged"
    assert res2["beam_lost_at"] == "COLL_KILL"
    tab._corr_lattice_at_launch = st.lattice
    tab._on_correction_done(res2)
    assert "COLL_KILL" in captured["text"]
    assert "Status: converged" in captured["text"]
