"""The Numerics tab's RFQ solver choice (StepConfig.rfq_solver): project
round trip, the live lattice, and the GUI run paths that read it."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

_PROJ = "linac_gen_project"


@pytest.fixture()
def win(qapp, mini_lattice):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    w.state.set_lattice(mini_lattice)
    yield w
    w.close()
    w.deleteLater()


#: combo index of each StepConfig.rfq_solver value
_IDX = {"auto": 0, "cells": 1, "time_domain": 2}


def _settings(st):
    return (st.rfq_solver, st.rfq_td_steps_per_period, st.rfq_td_sc_every)


def test_default_is_auto(win):
    ct = win.convergence_tab
    assert ct._rfq_solver.count() == 3
    assert ct.current_rfq_solver() == "auto"
    assert ct._rfq_td_steps.isEnabled()        # auto may run the time domain
    assert _settings(ct.current_step_config()) == ("auto", 48, 4)


@pytest.mark.parametrize("value", ["auto", "cells", "time_domain"])
def test_each_choice_round_trips(win, value):
    ct = win.convergence_tab
    win._apply_project_dict({"__kind__": _PROJ,
                             "convergence": {"rfq_solver": value}},
                            silent=True)
    assert ct._rfq_solver.currentIndex() == _IDX[value]
    assert ct.current_rfq_solver() == value
    assert ct._rfq_td_steps.isEnabled() is (value != "cells")
    assert win._collect_project_dict([], None)["convergence"]["rfq_solver"] \
        == value
    assert win.state.lattice.step_config.rfq_solver == value


def test_round_trips_through_the_project(win):
    ct = win.convergence_tab
    warns = win._apply_project_dict(
        {"__kind__": _PROJ,
         "convergence": {"rfq_solver": "time_domain",
                         "rfq_td_steps_per_period": 96,
                         "rfq_td_sc_every": 2}}, silent=True)
    assert not [w for w in warns if "RFQ" in w]
    assert ct.current_rfq_solver() == "time_domain"
    assert ct._rfq_td_steps.isEnabled() and ct._rfq_td_steps.value() == 96
    assert _settings(ct.current_step_config()) == ("time_domain", 96, 2)
    conv = win._collect_project_dict([], None)["convergence"]
    assert (conv["rfq_solver"], conv["rfq_td_steps_per_period"],
            conv["rfq_td_sc_every"]) == ("time_domain", 96, 2)
    # the live lattice follows the loaded project at once
    assert _settings(win.state.lattice.step_config) == ("time_domain", 96, 2)


def test_a_project_without_the_keys_gets_the_default(win):
    """Never the previous project's choice (as for space charge)."""
    ct = win.convergence_tab
    win._apply_project_dict({"__kind__": _PROJ,
                             "convergence": {"rfq_solver": "time_domain"}},
                            silent=True)
    assert ct.current_rfq_solver() == "time_domain"
    win._apply_project_dict({"__kind__": _PROJ,
                             "convergence": {"step1_per_m": 100.0}},
                            silent=True)
    assert _settings(ct.current_step_config()) == ("auto", 48, 4)


def test_an_invalid_project_value_is_reported(win):
    ct = win.convergence_tab
    warns = win._apply_project_dict(
        {"__kind__": _PROJ, "convergence": {"rfq_solver": "zdomain"}},
        silent=True)
    assert any("RFQ solver" in w and "auto" in w for w in warns)
    assert ct.current_rfq_solver() == "auto"


def test_the_choice_reaches_the_lattice_and_survives_an_edit(win):
    """A lattice edit re-syncs the tab from lattice.step_config, so the
    choice must already be there."""
    ct = win.convergence_tab
    ct._rfq_solver.setCurrentIndex(_IDX["time_domain"])
    ct._rfq_td_steps.setValue(72)
    assert ct._rfq_td_steps.isEnabled()
    assert _settings(win.state.lattice.step_config) == ("time_domain", 72, 4)
    win.state.lattice_changed.emit(win.state.lattice)
    assert ct.current_rfq_solver() == "time_domain"
    assert ct._rfq_td_steps.value() == 72
    # the step densities of the lattice are untouched by the RFQ choice
    st = win.state.lattice.step_config
    assert (st.integration_steps_per_metre, st.sc_steps_per_metre) \
        == (100.0, 50.0)
    ct._rfq_solver.setCurrentIndex(_IDX["cells"])
    assert win.state.lattice.step_config.rfq_solver == "cells"
    assert not ct._rfq_td_steps.isEnabled()
    ct._rfq_solver.setCurrentIndex(_IDX["auto"])
    assert win.state.lattice.step_config.rfq_solver == "auto"
    assert ct._rfq_td_steps.isEnabled()


def test_a_new_or_reloaded_lattice_keeps_the_choice(win):
    """A lattice file never carries the RFQ solver (a run setting): a new
    or re-loaded lattice takes the tab's choice -- it used to reset it to
    the per-cell model, so a lattice re-applied after choosing the time
    domain ran per cell."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    ct = win.convergence_tab
    ct._rfq_solver.setCurrentIndex(_IDX["time_domain"])
    ct._rfq_td_steps.setValue(60)
    lat = Lattice()
    lat.add(Drift("D", length=10.0))
    assert lat.step_config.rfq_solver == "auto"
    win.state.set_lattice(lat)
    assert ct.current_rfq_solver() == "time_domain"
    assert _settings(lat.step_config) == ("time_domain", 60, 4)
    # the step densities still come from the lattice (PARTRAN_STEP)
    import dataclasses
    lat2 = Lattice()
    lat2.add(Drift("D", length=10.0))
    lat2.step_config = dataclasses.replace(lat2.step_config,
                                           integration_steps_per_metre=37.0)
    win.state.set_lattice(lat2)
    assert ct._fixed_step1.value() == 37.0
    assert lat2.step_config.rfq_solver == "time_domain"
    # a project without the keys still shows the default
    win._apply_project_dict({"__kind__": _PROJ, "convergence": {}},
                            silent=True)
    assert ct.current_rfq_solver() == "auto"


def test_run_paths_read_the_choice(win):
    """Matching, the convergence scan's pool points and the Study tab."""
    from linac_gen_gui.interphase.tabs.convergence_tab import _ScanWorker
    ct = win.convergence_tab
    ct._rfq_solver.setCurrentIndex(_IDX["time_domain"])
    ct._rfq_td_steps.setValue(64)
    _sc, step = win.matching_tab._build_mp_configs()
    assert _settings(step) == ("time_domain", 64, 4)
    fields = _ScanWorker._scan_rfq_fields(
        type("W", (), {"lattice": win.state.lattice})())
    assert fields == dict(rfq_solver="time_domain",
                          rfq_td_steps_per_period=64, rfq_td_sc_every=4)
    num = win.study_tab._draft_spec().numerics
    assert (num["rfq_solver"], num["rfq_td_steps"]) == ("time_domain", 64)
    # the default needs no key; an explicit per-cell choice is carried
    # (the default is auto, so dropping it would run the time domain)
    ct._rfq_td_steps.setValue(48)
    ct._rfq_solver.setCurrentIndex(_IDX["auto"])
    assert "rfq_solver" not in win.study_tab._draft_spec().numerics
    ct._rfq_solver.setCurrentIndex(_IDX["cells"])
    assert win.study_tab._draft_spec().numerics["rfq_solver"] == "cells"
    _sc, step = win.matching_tab._build_mp_configs()
    assert step.rfq_solver == "cells"


def test_mp_worker_shows_the_solver_progress_lines(qapp, mini_lattice,
                                                   monkeypatch):
    """The field-map build reports on the linac_gen.rfq_td logger; the run
    worker forwards those lines (status bar) only while it runs."""
    import logging

    from linac_gen.core import simulation
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    from linac_gen_gui.interphase import workers

    def run(self):
        logging.getLogger("linac_gen.rfq_td.build").info(
            "time-domain RFQ field map: window %d/%d", 3, 84)
        return self._real_run()

    monkeypatch.setattr(simulation.Simulation, "_real_run",
                        simulation.Simulation.run, raising=False)
    monkeypatch.setattr(simulation.Simulation, "run", run)
    log = logging.getLogger("linac_gen.rfq_td")
    level, handlers = log.level, list(log.handlers)
    beam = create_beam(BeamConfig(species="proton", energy=3.0,
                                  frequency=352.21, current=0.0,
                                  n_particles=20), seed=1)
    w = workers.MultiparticleWorker(mini_lattice, beam, "off")
    got, done = [], []
    w.status.connect(got.append)
    w.finished_ok.connect(done.append)
    w._run_locked()                     # in this thread: direct delivery
    assert done
    assert got == ["time-domain RFQ field map: window 3/84"]
    assert (log.level, log.handlers) == (level, handlers)


def test_a_step_count_beyond_the_control_is_reported(win):
    ct = win.convergence_tab
    warns = win._apply_project_dict(
        {"__kind__": _PROJ,
         "convergence": {"rfq_solver": "time_domain",
                         "rfq_td_steps_per_period": 5000}}, silent=True)
    assert any("RFQ time steps" in w for w in warns)
    assert ct._rfq_td_steps.value() == ct._rfq_td_steps.maximum()
    assert win.state.lattice.step_config.rfq_td_steps_per_period \
        == ct._rfq_td_steps.maximum()
