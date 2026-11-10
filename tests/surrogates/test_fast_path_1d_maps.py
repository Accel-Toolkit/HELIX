"""The multiparticle fast path asks no weights for a map it cannot push.

The fast path samples a field map's 3-D channel interpolators; a 1-D map has none, so the
fast path hands its traversal to the native element before asking the weights anything.  An
answer asked for and never used would be counted (``nn_calls`` and the thread's tally, which the
studies report), and the run would read as pushed through the weights where the native element
pushed every call.  So in a fast-path run the weights of a 1-D map are not asked at all: a
setting they hold that has changed is not counted as a refusal, and weights from a file are not
put through their accuracy check (both still happen where the weights are asked, as in an
envelope run).
"""
import copy

import numpy as np
import pytest

from linac_gen.elements.drift import Drift
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import tally
from linac_gen.tracking.tracker import Tracker
from tests.surrogates.accuracy_helpers import W, from_file, native, off, pinned
from tests.surrogates.test_frequency_jump import F_NEW, _beam, _cavity, _cavity_1d, _lattice, _ref


@pytest.fixture(autouse=True)
def _flags():
    mp, fast = registry.is_mp_enabled(), registry.is_fast_path_enabled()
    registry.clear()
    yield
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(fast)
    registry.clear()


@pytest.mark.parametrize("make, pushes", [(lambda: _cavity_1d(p_flag=1), False),
                                          (lambda: _cavity(p_flag=1), True)],
                         ids=["1-D map", "3-D map"])
def test_the_fast_path_asks_the_weights_only_for_a_map_it_can_push(make, pushes):
    el = make()
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    reference = _beam(_ref(F_NEW, W))
    Tracker(copy.deepcopy(lat), reference).run()        # native
    surr = pinned(el, native(el), w=(W - 0.1, W + 0.5))
    registry.register(surr)
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    with tally() as counted:
        Tracker(lat, beam).run()
    assert surr.nn_calls == counted.calls == (1 if pushes else 0)
    assert np.array_equal(beam.particles, reference.particles) is not pushes


@pytest.mark.parametrize("case", ["file weights that fail their check", "a held setting changed"])
def test_a_fast_path_run_neither_checks_nor_refuses_a_1d_maps_weights(case, tmp_path):
    """The weights of a 1-D map are not asked: weights from a file that their accuracy check
    would refuse are not checked, and a setting they hold that has changed is not counted as a
    refusal; the native element pushes every call."""
    el = _cavity_1d(p_flag=1)
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    if case.startswith("file"):
        surr = from_file(tmp_path, el, off(native(el), factor=1.5), w=(W - 0.1, W + 0.5))
        assert surr.metadata.accuracy_due and surr.metadata.accuracy_check is None
    else:
        surr = pinned(el, native(el), w=(W - 0.1, W + 0.5))
        el.scale = 1.3                      # held by the weights (not an input): a query refuses
    reference = _beam(_ref(F_NEW, W))
    Tracker(copy.deepcopy(lat), reference).run()        # native
    registry.register(surr)
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    with tally() as counted:
        Tracker(lat, beam).run()
    assert surr.nn_calls == counted.calls == 0
    assert surr.refusals == {} and counted.refused == {}
    assert surr.metadata.accuracy_check is None
    assert np.array_equal(beam.particles, reference.particles)
