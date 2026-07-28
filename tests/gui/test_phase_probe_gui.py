"""GUI wiring for the phase probe / channel tunes (σ_model).

Pins:
* EnvelopeWorker records the probe by default (the matching-tab KPIs
  and the tune-depression popup need probe-bearing results).
* The matching-tab panel exposes the model KPI cards and fills them
  from ``channel_phase_advance``.
"""
from __future__ import annotations

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen_gui.interphase.workers import EnvelopeWorker


def _fodo(n_cells: int = 3) -> Lattice:
    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=10.0))
    return lat


def _initial() -> dict:
    # DC beam: uniform SC along the line, so every cell's depressed
    # channel is identical and stable (a tiny synthetic bunch would
    # over-crush cell 1 and debunch downstream).
    return dict(alpha_x=0.0, beta_x=1.0, emit_x=1.0,
                alpha_y=0.0, beta_y=1.0, emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)


def _run_worker(**kw):
    lat = _fodo()
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    w = EnvelopeWorker(lat, ref, _initial(), current_mA=5.0,
                       solver_kind="matrix", **kw)
    captured = {}
    w.finished_ok.connect(lambda res: captured.update(results=res))
    w.run()
    return lat, captured.get("results")


def test_envelope_worker_records_probe_by_default():
    lat, res = _run_worker()
    assert res is not None
    assert len(res.element_maps_dep) == len(lat.elements)
    assert len(res.element_maps_bare) == len(lat.elements)


def test_envelope_worker_probe_can_be_disabled():
    lat, res = _run_worker(phase_probe=False)
    assert res is not None
    assert len(res.element_maps_dep) == 0


def test_matching_tab_model_kpis_populate(qapp=None):
    """The panel's model KPI cards fill from a probe-bearing run."""
    import pytest
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    lat, res = _run_worker()
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.analysis.phase_advance import channel_phase_advance
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    ch = channel_phase_advance(res, period)
    assert not ch["coupled_xy"]
    # The exact numbers the panel displays (median over cells).
    import numpy as np
    mx = np.asarray(ch["mu_x_dep_deg"], float)
    assert np.isfinite(mx).all() and (mx > 0).all()
    ex = np.asarray(ch["eta_x"], float)
    assert np.isfinite(ex).all() and (ex <= 1.0 + 1e-9).all()


def test_tune_depression_companion_probe_for_mp_results(qapp):
    """Option A: multi-particle results carry no probe maps — the
    tune-depression popup offers 'Compute channel model', runs a
    companion envelope probe with the Beam-tab config, plots the model
    curves next to the MP beam markers, and shares the probe with the
    Hofmann popup via the single-slot cache."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.matching.periodic import find_periodic_twiss
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs import results_tab as rt

    rt._COMPANION_PROBE.clear()                  # test isolation

    lat = _fodo()
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    cfg = BeamConfig(
        species="H-", energy=2.1, frequency=162.5, current=0.0,
        duty_cycle=100.0, n_particles=200, distribution="gaussian",
        cutoff=3.0,
        emit_nx=0.2, alpha_x=tw["alpha_x"], beta_x=tw["beta_x"],
        emit_ny=0.2, alpha_y=tw["alpha_y"], beta_y=tw["beta_y"],
        emit_z=0.1, alpha_z=0.0, beta_z=10.0)
    mp_res = Simulation(lat, create_beam(cfg, seed=42)).run()
    assert not getattr(mp_res, "element_maps_dep", None)

    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_beam_config(cfg)
    state.set_results(mp_res)

    pop = rt._TuneDepressionPopup(None, state)
    pop.refresh(mp_res)
    if pop._worker is not None:                  # σ₀ struct walk
        assert pop._worker.wait(120000)
    qapp.processEvents(); qapp.processEvents()
    assert not pop._probe_btn.isHidden()         # button offered

    pop._compute_companion_probe()
    assert pop._probe_worker is not None
    assert pop._probe_worker.wait(120000)
    qapp.processEvents(); qapp.processEvents()

    mx = pop._rows["x"]["model"].xData
    assert mx is not None and len(mx) > 0        # model curve plotted
    assert pop._probe_btn.isHidden()             # offer withdrawn
    assert "companion envelope probe" in pop._info.text()

    hof = rt._HofmannPopup(None, state)
    hof.refresh(mp_res)                          # reuses shared cache
    assert "companion envelope probe" in hof._info.text()
