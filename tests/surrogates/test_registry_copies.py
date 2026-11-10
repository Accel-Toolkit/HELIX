"""A copy of the lattice decides for itself which surrogates engage.

The engagement decision is pinned on each element for the run.  The pin went with a deep
copy, so an error or failure study's copies of the lattice used the surrogate bound to
the original element: a cavity switched off in the copy still answered with the design
map, and an ADJUST on the copy was read from the original.
"""
import copy
import pickle

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.drift import Drift
from linac_gen.io.tracewin_geom import Channel
from linac_gen.surrogates import registry
from linac_gen.tracking.envelope import EnvelopeSolver
from tests.surrogates.test_frequency_jump import F_NEW, _lattice, _map
from tests.surrogates.test_surrogate_provenance import INIT, _at, _surrogate

SWEPT = dict(params=("phase",), lo=(-180.0,), hi=(180.0,), means=(-30.0,))


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


def _line():
    cav = _map(Channel.RF_E, F_NEW, p_flag=1)
    cav.name = "CAV"
    return _lattice(Drift("D0", 100.0), cav, Drift("D1", 100.0)), cav


def _registered(cav, **kw):
    surr = _surrogate(cav, copy.deepcopy(cav).fitted_matrix(_at(F_NEW)), train_frequency_mhz=F_NEW, **kw)
    registry.register(surr)
    return surr


def _sigma(lat):
    r = EnvelopeSolver(lat, _at(F_NEW), dict(INIT), current=0.0).run()
    return np.asarray(r.sigma_x), np.asarray(r.sigma_phi)


def test_a_copy_with_a_cavity_switched_off_runs_the_native_cavity():
    lat, cav = _line()
    surr = _registered(cav)
    _sigma(lat)                                   # the original pins its decision
    assert surr.nn_calls == 1
    off = copy.deepcopy(lat)
    off.elements[1].voltage_rel = -1.0            # as a failure study switches it off
    got = _sigma(off)
    registry.clear()
    want = _sigma(copy.deepcopy(off))
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    assert surr.nn_calls == 1


def test_a_copy_carries_no_pin():
    lat, cav = _line()
    surr = _registered(cav)
    assert registry.get_for_element(cav) is surr
    assert copy.deepcopy(cav)._surr_binding is None
    assert pickle.loads(pickle.dumps(cav))._surr_binding is None
    shallow = copy.copy(cav)                     # shares the pin, which is not its own
    assert registry.get_for_element(shallow) is not surr


def test_a_copy_engages_a_view_that_follows_its_own_adjust():
    lat, cav = _line()
    surr = _registered(cav, **SWEPT)
    twin = copy.deepcopy(lat).elements[1]
    view = registry.get_for_element(twin)
    assert view is not surr and view._wrapped is twin and view.mlp is surr.mlp
    twin.phase = 10.0                             # an ADJUST on the copy, mid-run
    assert registry.get_for_element(twin) is view             # the decision holds
    assert view._make_input(_at(F_NEW))[3] == 10.0            # and reads the copy
    assert surr._make_input(_at(F_NEW))[3] == cav.phase == -30.0


def test_the_decision_holds_through_an_adjust_on_the_element():
    lat, cav = _line()
    surr = _registered(cav, **SWEPT)
    assert registry.get_for_element(cav) is surr
    cav.phase = 12.0
    assert registry.get_for_element(cav) is surr


def test_a_view_counts_its_queries_with_the_surrogate():
    lat, cav = _line()
    surr = _registered(cav)
    view = registry.get_for_element(copy.deepcopy(cav))
    view.fitted_matrix(_at(F_NEW))
    assert view.nn_calls == surr.nn_calls == 1
    assert view.metadata is surr.metadata          # and shares the load's checks


def test_a_failure_study_with_surrogates_gives_the_native_impacts():
    from linac_gen.failures.failure_mode import FailureKind
    from linac_gen.failures.scenario import enumerate_scenarios
    from linac_gen.failures.study import FailureStudy

    def study():
        lat, cav = _line()
        cfg = BeamConfig(species="proton", energy=5.0, frequency=F_NEW, n_particles=10, current=0.0,
                         emit_nx=0.2, alpha_x=0.0, beta_x=1.0, emit_ny=0.2, alpha_y=0.0, beta_y=1.0,
                         emit_z=0.2, alpha_z=0.0, beta_z=1.0)
        scn, n2c, names = enumerate_scenarios(lat, kind=FailureKind.OFF, combination="single")
        return lat, cav, FailureStudy(lattice=lat, beam_config=cfg, mode="envelope"), (scn, names, n2c)

    lat, cav, fs, args = study()
    surr = _registered(cav)
    with_surr = fs.run(*args, combination="single")
    assert surr.nn_calls >= 1                    # the baseline used it
    registry.clear()
    _lat, _cav, fs2, args2 = study()
    native = fs2.run(*args2, combination="single")
    for a, b in zip(with_surr.impacts, native.impacts, strict=True):
        for k in ("emit_nx", "emit_ny", "emit_nz", "ref_w_kin"):
            assert a.metrics.get(k) == b.metrics.get(k), (a.scenario, k)


def test_a_copys_queries_count_with_the_surrogate_and_share_its_network():
    """A deep copy used to copy the pinned surrogate too, network and all, so the copy's
    queries went to a counter nobody reads (the compare report, the CLI)."""
    lat, cav = _line()
    surr = _registered(cav)
    _sigma(lat)
    twin = copy.deepcopy(lat)
    _sigma(twin)
    assert surr.nn_calls == 2
    assert registry.get_for_element(twin.elements[1]).mlp is surr.mlp
