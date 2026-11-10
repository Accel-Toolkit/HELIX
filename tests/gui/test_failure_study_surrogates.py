"""The GUI's studies and the registered surrogates.

The Failure Study tab runs the registered surrogates only when the design lattice with them
agrees with the native run (multiparticle runs native), and says what it did; the Study tab
and a reliability campaign run native and say so.  The Error Study tab leaves that to its
study (``ErrorStudy`` decides by its forward model: tests/surrogates/
test_envelope_error_study.py)."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.surrogates import registry
from tests.surrogates.accuracy_helpers import W, native, off, pinned
from tests.surrogates.test_frequency_jump import F_NEW, _magnet

PyQt6 = pytest.importorskip("PyQt6")      # the GUI modules are imported where used


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


def _line():
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="Q1", length=200.0, gradient=2.0, aperture=10.0))
    lat.add(Drift(name="D2", length=100.0, aperture=10.0))
    lat.add(_magnet())
    lat.add(Drift(name="D3", length=100.0, aperture=10.0))
    return lat


def _beam():
    return BeamConfig(species="proton", energy=W, frequency=F_NEW, current=0.0, n_particles=200,
                      distribution="waterbag", emit_nx=0.25, beta_x=1.0, emit_ny=0.25, beta_y=1.0,
                      emit_z=0.3, beta_z=10.0)


def _run(lat, mode="envelope", compensate=False, cost="envelope"):
    from linac_gen_gui.interphase.tabs.failure_study_tab import _FailureWorker

    from linac_gen.failures import CompensationConfig, FailureKind
    worker = _FailureWorker(
        copy.deepcopy(lat), _beam(), types={"quad"}, kind=FailureKind.OFF, amp=1.0, phase=0.0,
        combination="single", custom_sets=[], only_names=None, mode=mode, workers=1,
        compensate=compensate, comp_cfg=CompensationConfig(cost_solver=cost, max_iter=3),
        n_compensate=1)
    seen = {}
    worker.done.connect(lambda r: seen.__setitem__("done", r))
    worker.failed.connect(lambda m: seen.__setitem__("failed", m))
    worker.run()                                    # synchronous
    assert "failed" not in seen, seen.get("failed")
    return seen["done"]


def _status_of(results):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.failure_study_tab import FailureStudyTab
    tab = FailureStudyTab(AppState())
    tab._on_done(results)
    text = tab._status.text()
    tab.deleteLater()
    return text


def test_a_study_whose_design_lattice_agrees_uses_the_surrogates(qapp):
    lat = _line()
    el = lat.elements[3]
    surr = pinned(el, native(el))
    registry.register(surr)
    got = _run(lat)
    assert got.surrogates.use and got.surrogates.note.startswith("surrogates used")
    # the design check's answers, then the baseline's and each scenario's (one map each)
    assert surr.nn_calls == got.surrogates.nn_calls + 1 + len(got.impacts)
    assert got.surrogates.note in _status_of(got)


def test_a_study_whose_design_lattice_disagrees_runs_native(qapp):
    lat = _line()
    native_study = _run(lat)
    el = lat.elements[3]
    surr = pinned(el, off(native(el), 0, 0, 1.02))
    registry.register(surr)
    got = _run(lat)
    assert not got.surrogates.use and got.surrogates.note.startswith("surrogates not used")
    assert surr.nn_calls == got.surrogates.nn_calls      # the design check asked them, not the study
    assert got.baseline == native_study.baseline
    assert [im.metrics for im in got.impacts] == [im.metrics for im in native_study.impacts]
    assert got.surrogates.note in _status_of(got)


def test_a_multiparticle_study_runs_native(qapp):
    lat = _line()
    el = lat.elements[3]
    registry.register(pinned(el, native(el)))
    got = _run(lat, mode="mp")
    assert not got.surrogates.use and "multiparticle" in got.surrogates.note


@pytest.mark.parametrize("cost", ["envelope", "mp"])
def test_a_compensation_runs_native(qapp, cost):
    lat = _line()
    el = lat.elements[3]
    surr = pinned(el, native(el))
    registry.register(surr)
    got = _run(lat, compensate=True, cost=cost)
    assert got.surrogates.use                       # the sweep itself: envelope, checked
    assert not got.compensation_surrogates.use
    # the design check's answers, the baseline's and the one scenario's: none of the match
    assert surr.nn_calls == got.surrogates.nn_calls + 1 + len(got.impacts)
    assert "compensation: surrogates not used: a compensation runs native" in _status_of(got)


def test_a_study_without_surrogates_says_nothing_of_them(qapp):
    got = _run(_line())
    assert got.surrogates.note == "" and not got.surrogates.use
    assert "surrogates" not in _status_of(got)


def test_the_status_wraps_instead_of_widening_the_column(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.failure_study_tab import FailureStudyTab
    tab = FailureStudyTab(AppState())
    assert tab._status.wordWrap()
    tab.deleteLater()


# ---------------------------------------------------------------------------
# the other studies of the GUI: the Error Study tab leaves it to its study; the Study tab
# and a reliability campaign run native, and say so
# ---------------------------------------------------------------------------
def _registered():
    el = _magnet()
    registry.register(pinned(el, native(el)))


class _Spy:
    """Records whether the registry was suspended while the study ran."""

    def __init__(self, result=None):
        self.suspended, self.result = None, result

    def run(self, *a, **kw):
        self.suspended = registry.is_suspended()
        return self.result


def test_the_error_study_tab_leaves_the_surrogates_to_its_study(qapp):
    """The study decides by its forward model (an envelope one may use them, a multiparticle
    one runs native); the tab's worker neither suspends the registry nor touches the
    results."""
    from linac_gen_gui.interphase.tabs.error_study_tab import _StudyWorker
    _registered()
    results = SimpleNamespace(n_seeds=0)
    spy = _Spy(results)
    worker = _StudyWorker(spy, n_seeds=1)
    got = {}
    worker.done.connect(lambda r, stopped: got.__setitem__("r", r))
    worker.run()
    assert spy.suspended is False and not registry.is_suspended()
    assert got["r"] is results and vars(results) == {"n_seeds": 0}


def test_the_study_tab_runs_native(qapp, monkeypatch):
    from linac_gen_gui.interphase.tabs.study_tab import _StudyWorker

    from linac_gen.study import engine
    _registered()
    spy = _Spy()
    monkeypatch.setattr(engine.StudyManager, "load", classmethod(lambda cls, d: spy))
    worker = _StudyWorker("unused", max_workers=1)
    worker.run()
    assert spy.suspended is True
    assert worker.surrogates.note == "surrogates not used: a parameter study runs native"


def test_a_reliability_campaign_runs_native(qapp, monkeypatch, tmp_path):
    from linac_gen_gui.interphase.dialogs.reliability_dialog import _CampaignWorker

    from linac_gen.reliability import campaign
    _registered()
    spy = _Spy({})
    (tmp_path / "campaign.json").write_text("{}")
    monkeypatch.setattr(campaign.ReliabilityCampaign, "load", classmethod(lambda cls, d: spy))
    worker = _CampaignWorker(str(tmp_path), spec=None, legs=None, max_workers=1)
    logs = []
    worker.log.connect(logs.append)
    worker.run()
    assert spy.suspended is True
    assert "surrogates not used: a reliability campaign runs native" in logs
