"""The coupled per-cell walks run in a worker, memoised, for both popups.

Pins the 2026-09-08 change that took ``coupled_phase_advance_per_cell`` and
``coupled_beam_phase_advance_per_cell_via_M`` off the GUI thread in the
tune-depression and phase-advance popups (``_CoupledWalkMixin`` /
``_CoupledWalkWorker`` in results_tab.py).  On a solenoid channel
(transversely coupled, no field maps) the walk takes milliseconds, so the
lifecycle regressions below slow it down deliberately -- compute, then sleep
inside the worker -- to hold a "walk in flight" long enough to act during
it.  Repo idiom: ``worker.wait(...)`` + ``qapp.processEvents()``.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from linac_gen.cli.common import _envelope_initial, build_ref, load_lattice
from linac_gen.core.config import BeamConfig
from linac_gen.analysis.phase_advance import (
    coupled_beam_phase_advance_per_cell_via_M, coupled_phase_advance_per_cell,
    run_phase_probe,
)
import linac_gen.analysis.phase_advance as pa

TWO_PERIODS = """; two periods: a LATTICE bracket of three solenoid cells, plus the whole-lattice fallback
FREQ 162.5
LATTICE 2 0
SOLENOID 150.0 0.25 30.0
DRIFT 80.0 40.0
SOLENOID 150.0 0.25 30.0
DRIFT 80.0 40.0
SOLENOID 150.0 0.25 30.0
DRIFT 80.0 40.0
LATTICE_END
DRIFT 50.0 40.0
END
"""


def _state_with_run(path="examples/solenoid_channel.dat"):
    from linac_gen_gui.interphase.state import AppState
    lat = load_lattice(path)
    cfg = BeamConfig(species="H-", energy=2.1226695, frequency=162.5, current=5.0)
    ref = build_ref(cfg)
    res = run_phase_probe(lat, ref, _envelope_initial(cfg, ref), current=cfg.current)
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_beam_config(cfg)
    state.set_results(res)
    return state, lat, ref, res


def _settle(qapp, pop, timeout_s=120.0):
    """Pump the loop until the popup has nothing pending and no worker runs."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        qapp.processEvents()
        if not pop._busy():
            qapp.processEvents(); qapp.processEvents()
            if not pop._busy():
                return
        time.sleep(0.02)
    raise AssertionError(f"popup still busy: pending={pop._coupled_pending_key!r} "
                         f"workers={[type(w).__name__ for w in pop._live_workers()]}")


def _warm_all_periods(qapp, pop):
    """The coupled block runs only once the period's σ₀ walk has landed;
    visit every period so later flips reach the coupled walk at once."""
    for i in range(pop._combo.count()):
        pop._combo.setCurrentIndex(i); _settle(qapp, pop)
    pop._combo.setCurrentIndex(0); _settle(qapp, pop)


def _pump(qapp, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        qapp.processEvents(); time.sleep(0.01)


@pytest.fixture
def rt():
    from linac_gen_gui.interphase.tabs import results_tab as mod
    mod._COUPLED_PER_CELL.clear(); mod._COUPLED_PENDING.clear(); mod._COUPLED_FAILED.clear()
    yield mod
    for w in list(mod._COUPLED_PENDING.values()) + [
            z for z in mod._ZOMBIE_WORKERS if isinstance(z, mod._CoupledWalkWorker)]:
        w.request_stop(); w.wait(30000)
    mod._COUPLED_PER_CELL.clear(); mod._COUPLED_PENDING.clear(); mod._COUPLED_FAILED.clear()


class _Counting:
    """Count the coupled walk and record the threads it ran on; optionally
    hold the worker inside the walk for ``delay`` seconds (after computing)
    so a test can act while the walk is in flight."""
    def __init__(self, monkeypatch, delay=0.0, delay_bpc=0.0):
        self.calls, self.threads, self.delay = 0, [], delay
        orig = pa.coupled_phase_advance_per_cell
        def _hold(seconds):
            t0 = time.time()
            while time.time() - t0 < seconds:
                time.sleep(0.02)
        def spy(*a, **k):
            self.calls += 1
            self.threads.append(threading.current_thread())
            out = orig(*a, **k)
            if self.delay:
                _hold(self.delay)
            return out
        monkeypatch.setattr(pa, "coupled_phase_advance_per_cell", spy)
        self.orig = orig
        if delay_bpc:
            # after a results change only the depressed walk recomputes (cpc is
            # results-independent and stays memoised), so hold THAT one
            orig_b = pa.coupled_beam_phase_advance_per_cell_via_M
            def spy_b(*a, **k):
                out = orig_b(*a, **k); _hold(delay_bpc); return out
            monkeypatch.setattr(pa, "coupled_beam_phase_advance_per_cell_via_M", spy_b)


class _FakeWorker(QObject):
    """A running, unstoppable worker (templates: test_results_struct_worker)."""
    finished = pyqtSignal()
    def __init__(self):
        super().__init__(); self.stopped = False; self._waiters = set()
    def isRunning(self): return True
    def request_stop(self): self.stopped = True
    def requestInterruption(self): pass
    def wait(self, _ms): return False


# ---------------------------------------------------------------- basics

def test_walk_runs_off_the_gui_thread_and_plots_the_synchronous_numbers(qapp, rt, monkeypatch):
    state, lat, ref, res = _state_with_run()
    c = _Counting(monkeypatch)
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)
    assert c.calls >= 1
    assert all(t is not threading.main_thread() for t in c.threads), "walk ran on the GUI thread"
    period = pop._periods[pop._combo.currentIndex()]
    cpc = c.orig(lat, ref.copy(), period)
    bpc = coupled_beam_phase_advance_per_cell_via_M(lat, ref.copy(), period, res)
    eta_I = bpc["mu_I_deg"] / cpc["mu_I_deg"]
    _x, y = pop._rows["x"]["curve"].getData()
    assert y is not None and len(y) == np.isfinite(eta_I).sum()
    np.testing.assert_array_equal(np.asarray(y), eta_I[np.isfinite(eta_I)])
    assert "η_I_med" in pop._info.text() and "computing" not in pop._info.text()
    pop.close()


def test_memo_serves_a_second_refresh_and_the_other_popup(qapp, rt, monkeypatch):
    state, *_ = _state_with_run()
    c = _Counting(monkeypatch)
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)
    assert c.calls == 1
    pop.refresh_with_state(); _settle(qapp, pop)
    assert c.calls == 1                                   # memo hit
    other = rt._PhaseAdvancePopup(None, state); other.show(); _settle(qapp, other)
    assert c.calls == 1                                   # shared memo
    assert "eigenmode tunes: μ_I=" in other._info.text()
    pop.close(); other.close()


def test_two_visible_popups_start_one_worker(qapp, rt, monkeypatch):
    state, *_ = _state_with_run()
    started = []
    real = rt._CoupledWalkWorker
    class Recording(real):
        def start(self):
            started.append(self); super().start()
    monkeypatch.setattr(rt, "_CoupledWalkWorker", Recording)
    a = rt._TuneDepressionPopup(None, state); b = rt._PhaseAdvancePopup(None, state)
    a.show(); b.show()
    _settle(qapp, a); _settle(qapp, b)
    assert len(started) == 1, f"{len(started)} workers for one key"
    assert "η_I_med" in a._info.text() and "eigenmode tunes: μ_I=" in b._info.text()
    a.close(); b.close()


def test_cancelled_worker_emits_nothing(qapp, rt):
    state, lat, ref, res = _state_with_run()
    from linac_gen.analysis.period_detect import detect_periods
    period = detect_periods(lat)[0]
    w = rt._CoupledWalkWorker(lat, ref.copy(), period, ("k",), res, gen=rt._COUPLED_GEN[0])
    got = []
    w.finished_signal.connect(lambda *a: got.append(("ok", a)))
    w.failed_signal.connect(lambda *a: got.append(("fail", a)))
    w.request_stop()
    w.run()                                    # synchronous, house pattern
    qapp.processEvents()
    assert got == [] and ("k",) not in rt._COUPLED_PER_CELL


# ------------------------------------------------------ lifecycle regressions

def test_results_replaced_does_not_serve_the_old_bpc(qapp, rt):
    state, lat, ref, res = _state_with_run()
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)
    key = next(iter(rt._COUPLED_PER_CELL))
    cpc, bpc, res_obj, gen = rt._COUPLED_PER_CELL[key]
    assert res_obj is res and bpc is not None
    res2 = run_phase_probe(lat, ref, _envelope_initial(state.beam_config, ref),
                           current=state.beam_config.current)
    state.set_results(res2); qapp.processEvents()
    cpc2, bpc2, res_obj2, gen2 = rt._COUPLED_PER_CELL[key]
    assert cpc2 is cpc and gen2 > gen and (bpc2 is None or res_obj2 is res2)
    _settle(qapp, pop)
    assert rt._COUPLED_PER_CELL[key][2] is res2
    pop.close()


def test_in_place_lattice_edit_during_a_walk_is_not_memoised_as_current(qapp, rt, monkeypatch):
    """lattice_changed is re-emitted with the SAME object for in-place edits;
    a walk started before the edit must not land as the post-edit result."""
    state, lat, ref, res = _state_with_run()
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)   # σ₀ warm
    c = _Counting(monkeypatch, delay=1.5)
    pop._recompute(); _pump(qapp, 0.3)                       # slow walk in flight
    assert pop._busy()
    sol = next(e for e in lat.elements if type(e).__name__ == "Solenoid")
    attr = "field" if hasattr(sol, "field") else "b_field"
    setattr(sol, attr, getattr(sol, attr) * 1.3)
    state.lattice_changed.emit(lat)                            # in-place edit mid-walk
    _settle(qapp, pop, timeout_s=180)
    key = pop._struct_cache_key(lat, pop._periods[pop._combo.currentIndex()],
                                pop._state_ref() if hasattr(pop, "_state_ref") else ref)
    keys = list(rt._COUPLED_PER_CELL)
    assert keys, "nothing memoised after the edit"
    memo_cpc = rt._COUPLED_PER_CELL[keys[-1]][0]
    period = pop._periods[pop._combo.currentIndex()]
    true_cpc = c.orig(lat, ref.copy(), period)
    np.testing.assert_array_equal(memo_cpc["mu_I_deg"], true_cpc["mu_I_deg"])
    assert "η_I_med" in pop._info.text()
    pop.close()


def test_close_then_reopen_during_a_walk_does_not_stall(qapp, rt, monkeypatch):
    state, *_ = _state_with_run()
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)
    c = _Counting(monkeypatch, delay=3.0)
    pop._recompute(); _pump(qapp, 0.3)
    w1 = pop._coupled_worker
    assert w1 is not None and w1.isRunning()
    pop.close(); qapp.processEvents()
    assert w1._stop_event.is_set()
    assert w1 not in rt._COUPLED_PENDING.values()             # a dying worker is never joined
    pop.show()
    _settle(qapp, pop, timeout_s=180)
    assert not pop._busy() and "η_I_med" in pop._info.text()
    assert c.calls >= 2                                        # a fresh walk served the reopen


def test_period_change_during_a_walk_dispatches_the_new_period(qapp, rt, monkeypatch, tmp_path):
    deck = tmp_path / "two_periods.dat"; deck.write_text(TWO_PERIODS, encoding="utf-8")
    state, lat, ref, res = _state_with_run(str(deck))
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)
    assert pop._combo.count() >= 2
    _warm_all_periods(qapp, pop)
    c = _Counting(monkeypatch, delay=1.5)
    pop._recompute(); _pump(qapp, 0.3)
    assert pop._busy()
    pop._combo.setCurrentIndex(1)                              # change selection mid-walk
    _settle(qapp, pop, timeout_s=180)
    assert pop._coupled_pending_key is None
    assert "η_I_med" in pop._info.text()
    period = pop._periods[1]
    key = pop._struct_cache_key(lat, period, ref)
    assert key in rt._COUPLED_PER_CELL
    pop.close()


def test_period_change_during_the_sigma0_walk_does_not_stall(qapp, rt, monkeypatch, tmp_path):
    """The σ₀ struct worker's ready slot refreshes only for its own key; a
    period change while it runs used to leave the popup on 'Computing σ₀…'
    for good (and one loop turn later isRunning() could still be True, so a
    deferred re-check did not help either).  The popup now follows the
    worker's finished signal and dispatches the pending key."""
    deck = tmp_path / "two_periods.dat"; deck.write_text(TWO_PERIODS, encoding="utf-8")
    state, lat, ref, res = _state_with_run(str(deck))
    orig = pa.structure_phase_advance
    def slow(*a, **k):
        out = orig(*a, **k); time.sleep(1.5); return out
    monkeypatch.setattr(pa, "structure_phase_advance", slow)
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _pump(qapp, 0.3)
    assert pop._worker is not None and pop._worker.isRunning()          # σ₀ walk for period 0
    pop._combo.setCurrentIndex(1)                                         # move away mid-walk
    _settle(qapp, pop, timeout_s=180)
    assert pop._pending_key is None and pop._coupled_pending_key is None
    assert "η_I_med" in pop._info.text(), pop._info.text()
    pop.close()


def test_results_change_then_period_flip_stops_and_tracks_the_first_worker(qapp, rt, monkeypatch, tmp_path):
    deck = tmp_path / "two_periods.dat"; deck.write_text(TWO_PERIODS, encoding="utf-8")
    state, lat, ref, res = _state_with_run(str(deck))
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)
    _warm_all_periods(qapp, pop)
    c = _Counting(monkeypatch, delay=1.5, delay_bpc=1.5)
    res2 = run_phase_probe(lat, ref, _envelope_initial(state.beam_config, ref),
                           current=state.beam_config.current)
    state.set_results(res2); _pump(qapp, 0.3)                  # bpc walk for period 0 in flight
    w1 = pop._coupled_worker
    assert w1 is not None and w1.isRunning()
    pop._combo.setCurrentIndex(1)                              # a second worker for period 1
    _pump(qapp, 0.2)
    w2 = pop._coupled_worker
    assert w2 is not None and w2 is not w1
    assert w1._stop_event.is_set(), "orphaned worker was not told to stop"
    assert w1 not in rt._COUPLED_PENDING.values()
    assert (w1 in rt._ZOMBIE_WORKERS) or not w1.isRunning()   # kept alive until it exits
    tab = rt.ResultsTab.__new__(rt.ResultsTab); tab._popups = {"tune_depr": pop}
    out = rt.ResultsTab.shutdown_begin(tab)
    assert w2 in out
    for w in (w1, w2):
        w.wait(60000)
    _settle(qapp, pop, timeout_s=180)
    pop.close()


def test_phase_popup_first_then_tune_popup_joins_and_gets_its_bpc(qapp, rt, monkeypatch):
    state, *_ = _state_with_run()
    tune = rt._TuneDepressionPopup(None, state); tune.show(); _settle(qapp, tune)
    tune.close(); rt._COUPLED_PER_CELL.clear()                 # σ₀ warm, coupled memo cold
    c = _Counting(monkeypatch, delay=1.5)
    phase = rt._PhaseAdvancePopup(None, state); phase.show()
    _pump(qapp, 0.5)                                            # cpc-only worker in flight
    pw = getattr(phase, "_coupled_worker", None)
    tune.show(); _pump(qapp, 0.2)
    if pw is not None and pw.isRunning():
        assert getattr(tune, "_coupled_worker", None) is pw    # joined, not duplicated
    _settle(qapp, phase, timeout_s=180); _settle(qapp, tune, timeout_s=180)
    assert c.calls == 1                                         # cpc computed exactly once
    assert "η_I_med" in tune._info.text()
    assert "eigenmode tunes: μ_I=" in phase._info.text()
    phase.close(); tune.close()


def test_failed_walk_is_reported_once_and_not_respawned(qapp, rt, monkeypatch):
    state, *_ = _state_with_run()
    calls = {"n": 0}
    def boom(*a, **k):
        calls["n"] += 1; raise ValueError("synthetic walk failure")
    monkeypatch.setattr(pa, "coupled_phase_advance_per_cell", boom)
    pop = rt._TuneDepressionPopup(None, state); pop.show(); _settle(qapp, pop)
    assert "coupled η failed: synthetic walk failure" in pop._info.text(), pop._info.text()
    pop.refresh_with_state(); _settle(qapp, pop)
    assert calls["n"] == 1                                      # memoised failure, no respawn
    pop.close()


# ------------------------------------------------------------- teardown

def test_close_stops_an_owned_worker_and_parks_a_stubborn_one(qapp, rt):
    popup = rt._PopupPlot(None, "t")
    fake = _FakeWorker(); fake._waiters.add(id(popup))
    popup._coupled_worker = fake
    popup.close(); qapp.processEvents()
    assert fake.stopped and fake in rt._ZOMBIE_WORKERS
    rt._ZOMBIE_WORKERS.remove(fake)
    # a worker another popup still waits on is left alone
    popup2 = rt._PopupPlot(None, "t2"); fake2 = _FakeWorker()
    fake2._waiters.update({id(popup2), 12345})
    popup2._coupled_worker = fake2
    popup2.close(); qapp.processEvents()
    assert not fake2.stopped and fake2 not in rt._ZOMBIE_WORKERS


def test_shutdown_begin_returns_popup_and_registry_workers(qapp, rt):
    dlg = rt._PopupPlot(None, "t"); fake = _FakeWorker(); dlg._coupled_worker = fake
    tab = rt.ResultsTab.__new__(rt.ResultsTab); tab._popups = {"tune_depr": dlg}
    stray = _FakeWorker(); rt._COUPLED_PENDING[("stray",)] = stray
    out = rt.ResultsTab.shutdown_begin(tab)
    assert fake in out and fake.stopped and stray in out and stray.stopped
    rt._COUPLED_PENDING.clear()


def test_cancel_background_walks_stops_everything(qapp, rt):
    dlg = rt._PopupPlot(None, "t"); fake = _FakeWorker(); dlg._worker = fake
    tab = rt.ResultsTab.__new__(rt.ResultsTab); tab._popups = {"phase_adv": dlg}
    stray = _FakeWorker(); rt._COUPLED_PENDING[("stray",)] = stray
    rt.ResultsTab.cancel_background_walks(tab)
    assert fake.stopped and stray.stopped and not rt._COUPLED_PENDING
