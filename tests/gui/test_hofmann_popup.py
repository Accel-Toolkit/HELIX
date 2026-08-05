"""Hofmann growth-chart popup (results_tab._HofmannPopup).

Covers the REAL entry seam (ResultsTab._open_popup) plus the compute
path: button → _HofmannChartWorker → heatmap + classified cells, the
DC-refusal rendering, the legacy-bands toggle, and worker teardown on
close-during-compute.
"""
from __future__ import annotations

import time

import numpy as np

from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.matching.periodic import find_periodic_twiss
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix,
    compute_twiss,
)
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.results_tab import _HofmannPopup

_noop = lambda *a, **k: None                                 # noqa: E731


def _bunched_probe_run(current=5.0):
    """FODO+buncher cell (stable in all three planes) with a phase probe."""
    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(Quadrupole(name="QF", length=40.0, gradient=+50.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(RFGap(name="G", voltage=0.10, phase=-90.0, frequency=162.5))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(Quadrupole(name="QD", length=40.0, gradient=-50.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    M = compute_transfer_matrix(lat, ref, start=0, end=6)
    twz = compute_twiss(M, "z")
    init = dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=2.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=2.0,
                alpha_z=twz["alpha"], beta_z=twz["beta"], emit_z=0.1)
    res = EnvelopeSolver(lat, ref.copy(), init, current=current,
                         phase_probe=True).run()
    return lat, res


def _dc_probe_run():
    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=20.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    init = dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=15.0,
                         phase_probe=True).run()
    return lat, res


def _wait_worker(qapp, pop, timeout_s: float = 120.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        qapp.processEvents()
        w = pop._worker
        if w is not None and not w.isRunning():
            qapp.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("Hofmann chart worker did not finish in time")


def test_real_open_popup_computes_chart(qapp):
    """End-to-end through ResultsTab._open_popup: open → trajectory drawn;
    Compute → heatmap set, cells classified, flags in the info line."""
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    lat, res = _bunched_probe_run()
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(res)
    tab = ResultsTab(state, _noop, _noop, _noop)
    tab._open_popup("hofmann")
    pop = tab._popups.get("hofmann")
    assert isinstance(pop, _HofmannPopup)

    x, y = pop._traj.getData()
    assert x is not None and len(x) > 0          # trajectory on open

    pop._steps.setValue(60)
    pop._btn.click()
    assert pop._worker is not None
    _wait_worker(qapp, pop)

    assert pop._payload is not None
    tab_out = pop._payload["tab"]
    assert tab_out["reason"] is None
    assert pop._image.image is not None          # heatmap rendered
    assert len(pop._scatter.points()) > 0        # classified cells
    txt = pop._info.text()
    assert "flags" in txt and "solver" in txt
    pop.close()


def test_dc_results_render_reason_not_crash(qapp):
    lat, res = _dc_probe_run()
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(res)
    pop = _HofmannPopup(None, state)
    pop.refresh(res)
    pop._btn.click()
    assert pop._worker is not None
    _wait_worker(qapp, pop)
    assert pop._payload is not None
    assert pop._payload["tab"]["reason"] is not None
    assert "DC" in pop._info.text()
    assert len(pop._scatter.points()) == 0
    pop.close()


def test_legacy_bands_toggle(qapp):
    lat, res = _bunched_probe_run()
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(res)
    pop = _HofmannPopup(None, state)
    pop.refresh(res)
    assert not pop._band_items                   # off by default
    pop._chk_bands.setChecked(True)
    qapp.processEvents()
    assert pop._band_items                       # heuristic overlay drawn
    pop._chk_bands.setChecked(False)
    qapp.processEvents()
    assert not pop._band_items
    pop.close()


def test_close_during_compute_stops_worker(qapp):
    lat, res = _bunched_probe_run()
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(res)
    pop = _HofmannPopup(None, state)
    pop.refresh(res)
    pop._steps.setValue(300)                     # ~11 s — surely mid-flight
    pop._btn.click()
    w = pop._worker
    assert w is not None and w.isRunning()
    pop.close()                                  # closeEvent stops/parks it
    assert w.wait(30000)
    assert not w.isRunning()


def test_results_change_invalidates_payload(qapp):
    """A new run is new physics: the chart/table computed for the old
    results must be dropped when state.results_changed fires
    (adversarial-review finding F1)."""
    lat, res = _bunched_probe_run(current=5.0)
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(res)
    pop = _HofmannPopup(None, state)
    pop.refresh(res)
    pop._btn.click()
    _wait_worker(qapp, pop)
    assert pop._payload is not None
    _, res2 = _bunched_probe_run(current=0.0)     # different physics
    state.set_results(res2)
    qapp.processEvents()
    assert pop._payload is None                   # stale payload dropped
    assert len(pop._scatter.points()) == 0
    pop.close()


def test_period_change_invalidates_payload(qapp):
    """Switching periods must not show a chart computed for another one."""
    lat, res = _bunched_probe_run()
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(res)
    pop = _HofmannPopup(None, state)
    pop.refresh(res)
    pop._steps.setValue(60)
    pop._btn.click()
    _wait_worker(qapp, pop)
    assert pop._payload is not None
    # Inject a second (whole-lattice) period and switch to it.
    pop._periods.append(PeriodicStructure(
        start=0, end=len(lat.elements), inner_period_length=7,
        inner_slice_end=7, n_repeats=3, label="alt", source="manual"))
    pop._combo.blockSignals(True)
    pop._combo.addItem("alt")
    pop._combo.blockSignals(False)
    pop._combo.setCurrentIndex(pop._combo.count() - 1)
    qapp.processEvents()
    assert pop._payload is None                  # stale payload dropped
    assert len(pop._scatter.points()) == 0
    pop.close()
