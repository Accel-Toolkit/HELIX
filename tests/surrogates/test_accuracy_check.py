"""Weights from a file answer only within the accuracy limits of their native element.

Shipped weights may predate today's physics, or be roughly trained: some are off their
element by more than 10 % in a block of the map, and a run used them all the same.  At their
first use after a load inside their range, a file's weights are compared with the native
element at the input of that use and at 16 points of their training range; the largest
2x2-block error (in units where a 1 m drift reads the same in every plane, each block over the
largest native block norm in its row and column) must be within 3e-3, and the transverse and
longitudinal determinants within 1e-3 of the native map's.  Weights that fail go native in
every run, say by how much once per load, and still answer in the compare view.  A study uses
surrogates only when its design lattice with them agrees with the native run.
"""
import copy
import threading
import warnings

import numpy as np
import pytest

from linac_gen.elements.drift import Drift
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import (
    OutOfScopeError,
    SurrogateFieldMap,
    accuracy_refusals_suspended,
    block_error,
    determinant_error,
    drift_scale_z,
)
from linac_gen.surrogates.training import load_surrogate, save_surrogate
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.torch_matrices import RefKinematics
from linac_gen.tracking.torch_tracking import element_matrix_torch
from tests.surrogates.accuracy_helpers import (
    INIT,
    W,
    exact,
    from_file,
    meta,
    mlp,
    native,
    off,
    pinned,
)
from tests.surrogates.test_frequency_jump import (
    F_NEW,
    F_OLD,
    _cavity,
    _cavity_1d,
    _lattice,
    _magnet,
    _ref,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


class _no_accuracy_warning(warnings.catch_warnings):
    """Records the warnings of a block and checks none is the accuracy check's (other
    warnings, a library's deprecation, are not this test's business)."""

    def __init__(self):
        super().__init__(record=True)

    def __enter__(self):
        self._seen = super().__enter__()
        warnings.simplefilter("always")
        return self._seen

    def __exit__(self, *exc):
        super().__exit__(*exc)
        hits = [str(w.message) for w in self._seen if "accuracy limits" in str(w.message)]
        assert not hits, hits
        return False


def _count_native(monkeypatch, el):
    calls = []
    real = type(el).fitted_matrix

    def counted(self, ref):
        calls.append(1)
        return real(self, ref)

    monkeypatch.setattr(type(el), "fitted_matrix", counted)
    return calls


def _refused(surr, ref):
    with (pytest.warns(UserWarning, match="not within the accuracy limits"),
          pytest.raises(OutOfScopeError, match="Retrain it")):
        surr.fitted_matrix(ref)


def _answers(surr, ref):
    with _no_accuracy_warning():
        out = surr.fitted_matrix(ref)
    assert surr.metadata.accuracy_check == "", surr.metadata.accuracy_check
    return out


# ---------------------------------------------------------------------------
# the limits and the metric
# ---------------------------------------------------------------------------
def test_the_limits():
    from linac_gen.surrogates.study_check import DESIGN_TOLERANCE
    assert (SurrogateFieldMap.ACCURACY_TOLERANCE, SurrogateFieldMap.ACCURACY_DET_TOLERANCE,
            SurrogateFieldMap.ACCURACY_POINTS, DESIGN_TOLERANCE) == (3e-3, 1e-3, 16, 5e-3)


def test_the_block_metric_sees_the_longitudinal_plane():
    """In (deg, MeV) the Δφ–ΔW term (75 deg/MeV here) outweighs the rest of the block: a
    magnet's map given a longitudinal focusing term and a Δφ-Δφ term 7.5 % off, with its
    determinant kept, reads 1e-3 there, within the limit.  In units where a 1 m drift is the
    same in every plane it reads 0.27."""
    el = _magnet()
    M = native(el)
    r = _ref(F_NEW, W)
    s = drift_scale_z(r.beta, r.gamma, r.species.mass, r.wavelength)
    bad = M.copy()
    bad[5, 4] = 1e-3
    bad[4, 4] = 1.0 + M[4, 5] * bad[5, 4]          # det of the longitudinal block kept
    assert determinant_error(M, bad) < 1e-12
    assert block_error(M, bad, 1.0) < 3e-3           # what (deg, MeV) would accept
    assert block_error(M, bad, s) > 0.1


def test_the_drift_scale_is_a_drifts_term():
    """The scale is |M45| of a 1 m drift: the native map of a field-free stretch."""
    r = _ref(F_NEW, W)
    s = drift_scale_z(r.beta, r.gamma, r.species.mass, r.wavelength)
    M = np.asarray(Drift("D", 1000.0).transfer_matrix(r.copy()))
    assert abs(abs(M[4, 5]) / s - 1.0) < 1e-9


def test_a_block_is_measured_against_the_largest_block_of_its_row_and_column():
    """A solenoid turned near 90° has nearly empty diagonal blocks: against their norm, a
    0.02 % change of its field would read percent-level; against its coupling blocks it reads
    as small as it is.  An uncoupled map keeps each block's own norm."""
    c, s = np.cos(np.radians(89.9)), np.sin(np.radians(89.9))
    K = np.array([[1.0, 0.3], [-0.5, 0.85]])
    M = np.eye(6)
    M[0:2, 0:2] = M[2:4, 2:4] = c * K
    M[0:2, 2:4], M[2:4, 0:2] = s * K, -s * K
    near = M.copy()
    near[0:4, 0:4] *= 1.0002
    assert block_error(M, near, 400.0) < 3e-4
    uncoupled = np.eye(6)
    uncoupled[0:2, 0:2] = K
    bumped = uncoupled.copy()
    bumped[0, 0] *= 1.01                              # 1 % of M11 over its block's norm
    assert block_error(uncoupled, bumped, 400.0) == pytest.approx(0.01 / np.linalg.norm(K))


def test_a_map_that_cannot_be_taken_is_no_match():
    """NaN in either map, anywhere, is outside every limit (Python's max(0.0, nan) is 0.0, and
    LAPACK gives a determinant of 0 for some NaN placements)."""
    M = native(_magnet())
    for i, j in ((2, 3), (0, 0), (1, 1), (5, 5), (0, 4)):
        bad = M.copy()
        bad[i, j] = np.nan
        for a, b in ((M, bad), (bad, M)):
            assert np.isnan(block_error(a, b, 400.0)) and np.isnan(determinant_error(a, b))


def test_the_determinants_are_those_of_the_4x4_and_the_longitudinal_block():
    M = native(_magnet())
    stretched = M.copy()
    stretched[4:, 4:] *= 1.002                      # every entry 0.2 %, its det 0.4 %
    assert block_error(M, stretched, 400.0) < 3e-3
    assert determinant_error(M, stretched) == pytest.approx(1.002 ** 2 - 1.0, rel=1e-9)
    coupled = M.copy()
    coupled[0, 4] = 1e-2                            # outside both blocks: no determinant moves
    assert determinant_error(M, coupled) == 0.0 and block_error(M, coupled, 400.0) > 3e-3


def test_a_rotation_about_the_axis_moves_no_determinant():
    """A solenoid's planes couple: a rotation in x-y keeps the 4x4 determinant (the 4-D
    emittance) and changes each plane's 2x2 one, which is no invariant there."""
    M = native(_magnet())
    c, s = np.cos(0.1), np.sin(0.1)
    R = np.eye(6)
    R[0:2, 0:2] = R[2:4, 2:4] = c * np.eye(2)
    R[0:2, 2:4], R[2:4, 0:2] = s * np.eye(2), -s * np.eye(2)
    assert determinant_error(M, R @ M) < 1e-12
    assert abs(np.linalg.det((R @ M)[0:2, 0:2]) / np.linalg.det(M[0:2, 0:2]) - 1.0) > 1e-3


# ---------------------------------------------------------------------------
# weights from a file
# ---------------------------------------------------------------------------
def test_weights_within_the_limits_answer(tmp_path):
    el = _magnet()
    M = native(el)
    surr = from_file(tmp_path, el, M)
    assert surr.metadata.accuracy_due and surr.metadata.accuracy_check is None
    np.testing.assert_array_equal(_answers(surr, _ref(F_NEW, W)), M)
    assert surr.nn_calls == 1
    m = surr.metadata
    assert m.accuracy_block < 1e-6 and m.accuracy_det < 1e-6 and m.accuracy_seconds > 0.0


@pytest.mark.parametrize("make_bad, block_over, det_over", [
    (lambda M: off(M, 0, 0, 1.01), True, True),
    (lambda M: off(M, 0, 1, 1.05), True, False),
    (lambda M: np.where(np.arange(36).reshape(6, 6) >= 28, M * 1.002, M), False, True),
])
def test_weights_off_their_element_go_native_and_say_by_how_much(tmp_path, monkeypatch,
                                                                  make_bad, block_over, det_over):
    el = _magnet()
    surr = from_file(tmp_path, el, make_bad(native(el)))
    with (pytest.warns(UserWarning, match="not within the accuracy limits"),
          pytest.raises(OutOfScopeError, match="Retrain it") as e):
        surr.fitted_matrix(_ref(F_NEW, W))
    m = surr.metadata
    assert (m.accuracy_block > 3e-3, m.accuracy_det > 1e-3) == (block_over, det_over)
    assert f"largest 2x2-block error {m.accuracy_block:.2g}" in str(e.value)
    assert f"determinant error {m.accuracy_det:.2g}" in str(e.value)
    assert surr.nn_calls == 0
    calls = _count_native(monkeypatch, el)
    with _no_accuracy_warning():              # said once per load
        for _ in range(3):
            with pytest.raises(OutOfScopeError):
                surr.fitted_matrix(_ref(F_NEW, W))
    assert calls == []                              # the verdict is kept


def test_the_check_runs_once_per_load_for_every_view_and_copy(tmp_path, monkeypatch):
    el = _magnet()
    surr = from_file(tmp_path, el, native(el))
    calls = _count_native(monkeypatch, el)
    surr.fitted_matrix(_ref(F_NEW, W))
    assert len(calls) == 1 + SurrogateFieldMap.ACCURACY_POINTS
    surr.bound_to(copy.deepcopy(el)).fitted_matrix(_ref(F_NEW, W))
    copy.deepcopy(surr).fitted_matrix(_ref(F_NEW, W))       # a study's copy of the lattice
    surr.fitted_matrix(_ref(F_NEW, W))
    assert len(calls) == 1 + SurrogateFieldMap.ACCURACY_POINTS
    assert surr.nn_calls == 4


def test_a_copy_of_a_lattice_holding_the_weights_shares_their_load(tmp_path):
    el = _magnet()
    surr = from_file(tmp_path, el, native(el))
    lat = _lattice(Drift("D0", 100.0), surr, Drift("D1", 100.0))
    twin = copy.deepcopy(lat)
    copied = twin.elements[1]
    assert isinstance(copied, SurrogateFieldMap) and copied is not surr
    assert copied.metadata is surr.metadata and copied.mlp is surr.mlp
    assert copied._wrapped is not el and copied._calls is surr._calls


def test_concurrent_first_uses_check_once(tmp_path, monkeypatch):
    el = _magnet()
    surr = from_file(tmp_path, el, native(el))
    calls = _count_native(monkeypatch, el)
    go = threading.Barrier(3)

    def first_use():
        go.wait()
        surr.bound_to(copy.deepcopy(el)).fitted_matrix(_ref(F_NEW, W))

    threads = [threading.Thread(target=first_use) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 + SurrogateFieldMap.ACCURACY_POINTS and surr.nn_calls == 3


def test_a_file_loaded_again_is_checked_again_and_says_so_again(tmp_path):
    """Under the default filter, as the GUI and the CLI run (pytest.warns would show every
    warning whatever the code does)."""
    el = _magnet()
    d = tmp_path / "w"
    save_surrogate(mlp(), meta(el, off(native(el))), d)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("default")
        for _ in range(2):
            mlp_, m = load_surrogate(d)
            with pytest.raises(OutOfScopeError):
                SurrogateFieldMap(el, mlp_, m).fitted_matrix(_ref(F_NEW, W))
    assert sum("accuracy limits" in str(w.message) for w in seen) == 2


def test_the_points_cover_the_training_range(tmp_path, monkeypatch):
    """16 points of a Latin hypercube over the energies trained, besides the first use's."""
    el = _magnet()
    surr = from_file(tmp_path, el, native(el), w=(4.0, 6.0))
    seen = []
    real = type(el).fitted_matrix
    monkeypatch.setattr(type(el), "fitted_matrix",
                        lambda self, ref: (seen.append(ref.w_kin), real(self, ref))[1])
    _refused(surr, _ref(F_NEW, W))                  # a constant map is off at other energies
    assert seen[0] == W and len(seen) == 17
    bins = np.floor((np.array(seen[1:]) - 4.0) / (2.0 / 16)).astype(int)
    assert sorted(bins) == list(range(16))          # one point in each sixteenth


def test_a_native_failure_in_the_range_refuses_and_the_run_goes_on(tmp_path, monkeypatch):
    """Where the native element cannot be taken, the weights cannot be shown accurate: they
    are refused, and the query that ran the check falls back as any refusal does."""
    el = _magnet()
    surr = from_file(tmp_path, el, native(el), w=(4.0, 6.0))
    real = type(el).fitted_matrix

    def failing(self, ref):
        if abs(ref.w_kin - W) > 1e-12:
            raise ValueError("no map here")
        return real(self, ref)

    monkeypatch.setattr(type(el), "fitted_matrix", failing)
    _refused(surr, _ref(F_NEW, W))
    assert surr.metadata.accuracy_block == np.inf


def test_weights_built_in_memory_are_not_checked(monkeypatch):
    el = _magnet()
    bad = off(native(el))
    surr = pinned(el, bad)
    calls = _count_native(monkeypatch, el)
    np.testing.assert_array_equal(surr.fitted_matrix(_ref(F_NEW, W)), bad)
    assert calls == [] and surr.nn_calls == 1 and surr.metadata.accuracy_check is None


# ---------------------------------------------------------------------------
# the native maps of the check are the training's targets
# ---------------------------------------------------------------------------
def test_exact_weights_over_a_range_and_a_swept_input_pass():
    """A network exact over 4-6 MeV and a scale of 0.9-1.1 answers; one that did not learn
    the scale is refused, though exact at the scale it is used at."""
    el = _magnet()
    rng = (("scale", 0.9, 1.1),)
    np.testing.assert_allclose(_answers(exact(el, params=rng), _ref(F_NEW, W)), native(el),
                               rtol=1e-12, atol=1e-12)
    _refused(exact(copy.deepcopy(el), params=rng, blind=("scale",)), _ref(F_NEW, W))


def test_the_network_is_asked_at_each_points_beta_and_gamma():
    """β and γ follow each point's energy: a network that reads its γ is exact too."""
    el = _magnet()
    _answers(exact(el, from_gamma=True), _ref(F_NEW, W))


def test_exact_weights_of_a_calibrated_cavity_pass():
    """A p_flag 1 cavity calibrates at its entrance: each point of the check starts afresh."""
    el = _cavity_1d(p_flag=1)
    surr = exact(el, w=(4.5, 5.5), params=(("phase", -40.0, -20.0),))
    _answers(surr, _ref(F_NEW, W))


def test_exact_weights_first_used_away_from_the_training_phase_pass():
    """A p_flag 0 cavity's phase input moves with the running phase: the check takes its native
    maps at the training phase, with the input as the network sees it."""
    el = _cavity(p_flag=0)
    surr = exact(el, params=(("phase", -60.0, 0.0),))
    ref = _ref(F_NEW, W)
    ref.phi_s = 20.0
    _answers(surr, ref)


@pytest.mark.parametrize("knob, value", [("phase_offset", 5.0), ("voltage_rel", 0.05)])
def test_exact_weights_of_an_element_with_an_error_pass(knob, value):
    """The error is folded into an input (the phase, the scale): the native maps of the check
    are the design element's, not the error counted twice."""
    el = _cavity(p_flag=0)
    setattr(el, knob, value)
    surr = exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1)))
    _answers(surr, _ref(F_NEW, W))


@pytest.mark.parametrize("shift", ["rf_shift", "rf_rephase"])
def test_exact_weights_under_a_beam_phase_error_pass(shift):
    """SET_BEAM_PHASE_ERROR's shifts are folded into the phase input of a map whose field
    follows the running phase, so the native maps of the check are taken without them."""
    el = _cavity(p_flag=0)
    surr = exact(el, params=(("phase", -60.0, 0.0),))
    ref = _ref(F_NEW, W)
    setattr(ref, shift, 10.0)
    _answers(surr, ref)


@pytest.mark.parametrize("clock", [F_OLD, 2 * F_NEW])
def test_a_magnets_weights_are_checked_as_they_answer_at_another_clock(tmp_path, clock):
    """A magnet's Δφ row is in degrees of its training clock and converted to the clock it is
    asked at (as the torch arm does): weights exact at their own clock pass at another one,
    in both arms."""
    el = _magnet(F_NEW)
    surr = from_file(tmp_path, el, native(el, f=F_NEW), name=f"m{clock:g}")
    got = _answers(surr, _ref(clock, W))
    np.testing.assert_allclose(got, native(el, f=clock), rtol=1e-12, atol=1e-12)
    twin = from_file(tmp_path, copy.deepcopy(el), native(el, f=F_NEW), name=f"t{clock:g}")
    with _no_accuracy_warning():
        torch_map = element_matrix_torch(twin, RefKinematics.from_reference(_ref(clock, W)))
    np.testing.assert_allclose(torch_map.detach().numpy(), native(el, f=clock), rtol=1e-12,
                               atol=1e-12)


# ---------------------------------------------------------------------------
# runs, the compare view, the torch arm, the backtracker
# ---------------------------------------------------------------------------
def _line(el):
    return _lattice(Drift("D0", 100.0), el, Drift("D1", 100.0))


def _sigma(lat):
    res = EnvelopeSolver(lat, _ref(F_NEW, W), dict(INIT), current=0.0).run()
    return np.asarray(res.sigma_matrix)


def test_a_run_with_refused_weights_is_the_native_run(tmp_path):
    el = _magnet()
    lat = _line(el)
    native_run = _sigma(copy.deepcopy(lat))
    surr = from_file(tmp_path, el, off(native(el)))
    registry.register(surr)
    with pytest.warns(UserWarning, match="not within the accuracy limits"):
        got = _sigma(lat)
    np.testing.assert_array_equal(got, native_run)
    assert surr.nn_calls == 0


def test_a_run_with_accurate_weights_uses_them(tmp_path):
    el = _magnet()
    surr = from_file(tmp_path, el, native(el))
    registry.register(surr)
    _sigma(_line(el))
    assert surr.nn_calls == 1


def test_the_compare_view_still_shows_refused_weights_and_times_the_run_alone(tmp_path):
    from linac_gen.surrogates.compare import compare_envelope
    el = _magnet()
    lat = _line(el)
    surr = from_file(tmp_path, el, off(native(el), 0, 1, 1.5))
    with pytest.warns(UserWarning, match="not within the accuracy limits"):
        report = compare_envelope(lat, _ref(F_NEW, W), dict(INIT), surrogates=[surr])
    assert report.nn_calls == 1 and report.end_of_line["rel_diff_x"] > 1e-3
    assert any("it answered in this compare" in n for n in report.notes)
    checked = surr.metadata.accuracy_seconds
    assert any(f"leaves out {checked:.2g} s" in n for n in report.notes)
    assert report.wall_surrogate_s < checked        # the 17 native maps are not its time
    registry.register(surr)                         # outside the compare: native
    _sigma(lat)
    assert surr.nn_calls == 1


def test_a_compare_where_refused_weights_do_not_answer_says_so(tmp_path):
    from linac_gen.surrogates.compare import compare_envelope
    el = _magnet()
    lat = _line(el)
    surr = from_file(tmp_path, el, off(native(el)))
    registry.register(surr)
    with pytest.warns(UserWarning):
        _sigma(lat)                                 # refused in a 0 mA run
    report = compare_envelope(lat, _ref(F_NEW, W), dict(INIT), current=5.0, surrogates=[surr])
    assert report.nn_calls == 0
    assert any("did not answer in this compare either" in n for n in report.notes)
    assert not any("leaves out" in n for n in report.notes)


@pytest.fixture
def fast_path():
    mp, fast = registry.is_mp_enabled(), registry.is_fast_path_enabled()
    registry.set_fast_path_enabled(True)
    yield
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(fast)


def test_a_multiparticle_compare_still_shows_refused_weights(tmp_path, fast_path):
    from linac_gen.surrogates.compare import compare_mp
    from tests.surrogates.test_frequency_jump import _beam
    el = _magnet()
    lat = _line(el)
    surr = from_file(tmp_path, el, off(native(el), 0, 1, 1.5))
    with pytest.warns(UserWarning, match="not within the accuracy limits"):
        report = compare_mp(lat, _beam(_ref(F_NEW, W)), surrogates=[surr])
    assert surr.nn_calls == 1 and report.end_of_line["rel_diff_sigma_x"] > 1e-3
    assert any("it answered in this compare" in n for n in report.notes)


def test_the_suspension_is_this_threads():
    el = _magnet()
    m = meta(el, off(native(el)))
    m.accuracy_due, m.accuracy_check, m.accuracy_block, m.accuracy_det = True, "refused", 1.0, 1.0
    surr = SurrogateFieldMap(el, mlp(), m)
    seen = {}

    def other():
        try:
            surr.fitted_matrix(_ref(F_NEW, W))
            seen["other"] = "answered"
        except OutOfScopeError:
            seen["other"] = "refused"

    with accuracy_refusals_suspended():
        surr.fitted_matrix(_ref(F_NEW, W))           # answers here
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert seen["other"] == "refused"
    with pytest.raises(OutOfScopeError):
        surr.fitted_matrix(_ref(F_NEW, W))


def test_the_torch_arm_refuses_weights_off_their_element(tmp_path):
    el = _magnet()
    kin = RefKinematics.from_reference(_ref(F_NEW, W))
    good = from_file(tmp_path, el, native(el), name="good")
    assert np.allclose(element_matrix_torch(good, kin).detach().numpy(), native(el))
    assert good.metadata.accuracy_check == ""
    bad = from_file(tmp_path, copy.deepcopy(el), off(native(el)), name="bad")
    with (pytest.warns(UserWarning),
          pytest.raises(ValueError, match="not within the accuracy limits")):
        element_matrix_torch(bad, kin)


def test_the_backtracker_inverts_a_refused_set_exactly(tmp_path, fast_path):
    """The forward run had the native element in place of refused weights, so the backtracker
    uses its exact inverse and says nothing of a surrogate."""
    from linac_gen.tracking.backtrack import BacktrackWarning, backtrack_distribution
    from linac_gen.tracking.tracker import Tracker
    from tests.surrogates.test_frequency_jump import _beam
    el = _magnet()
    lat = _line(el)
    surr = from_file(tmp_path, el, off(native(el)))
    registry.register(surr)
    registry.set_mp_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    with pytest.warns(UserWarning, match="not within the accuracy limits"):
        Tracker(lat, beam).run()
    assert surr.nn_calls == 0
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        backtrack_distribution(lat, beam, _ref(F_NEW, W))
    assert not [w for w in seen if issubclass(w.category, BacktrackWarning)
                and "driven by a surrogate" in str(w.message)]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# the registry's suspension and the design check of a study
# ---------------------------------------------------------------------------
def test_a_suspended_registry_engages_nothing_in_this_thread_and_pins_nothing():
    el = _magnet()
    surr = pinned(el, native(el))
    registry.register(surr)
    other = copy.deepcopy(el)
    assert registry.get_for_element(el) is surr          # decided, and pinned on el
    seen = {}
    with registry.suspended():
        assert registry.get_for_element(el) is None      # the pin does not outrank it
        assert registry.get_for_element(other) is None
        assert getattr(other, "_surr_binding", None) is None
        t = threading.Thread(target=lambda: seen.setdefault("other", registry.get_for_element(el)))
        t.start()
        t.join()
    assert seen["other"] is surr
    assert registry.get_for_element(el) is surr


def _beam_config(**kw):
    from linac_gen.core.config import BeamConfig
    kw = {"species": "proton", "energy": W, "frequency": F_NEW, "current": 0.0,
          "n_particles": 16, **kw}
    return BeamConfig(**kw)


def test_the_design_check_with_nothing_registered_says_nothing():
    from linac_gen.surrogates.study_check import check_design, runs_native
    check = check_design(_line(_magnet()), _beam_config())
    assert not check.use and check.note == ""
    assert runs_native("x").note == ""


def test_a_multiparticle_study_runs_native():
    from linac_gen.surrogates.study_check import check_design
    el = _magnet()
    registry.register(pinned(el, native(el)))
    check = check_design(_line(el), _beam_config(), mode="mp")
    assert not check.use and "multiparticle" in check.note


def test_a_study_uses_surrogates_that_agree_on_its_design_lattice():
    from linac_gen.surrogates.study_check import check_design
    el = _magnet()
    registry.register(pinned(el, native(el)))
    check = check_design(_line(el), _beam_config())
    assert check.use and check.nn_calls == 1 and check.worst < 1e-9
    assert check.note.startswith("surrogates used")


@pytest.mark.parametrize("i, j, keys", [(0, 0, ("eps_x",)), (5, 5, ("sigma_w", "eps_z"))])
def test_a_study_runs_native_when_its_design_lattice_disagrees(i, j, keys):
    """Transversely and longitudinally (the bias that started this was longitudinal)."""
    from linac_gen.cli.common import run_envelope_sim
    from linac_gen.surrogates.study_check import check_design, study_surrogates
    el = _magnet()
    lat = _line(el)
    registry.register(pinned(el, off(native(el), i, j, 1.02)))
    check = check_design(lat, _beam_config())
    assert not check.use and check.worst_key in keys
    assert all(check.diffs[k] > 5e-3 for k in keys)
    assert check.note.startswith("surrogates not used") and check.worst_key in check.note
    with study_surrogates(check):
        inside = run_envelope_sim(copy.deepcopy(lat), _beam_config())
    registry.clear()
    native_run = run_envelope_sim(copy.deepcopy(lat), _beam_config())
    np.testing.assert_array_equal(np.asarray(inside.sigma_matrix),
                                  np.asarray(native_run.sigma_matrix))


def test_a_design_check_that_meets_a_nan_runs_native(monkeypatch):
    from linac_gen.surrogates import study_check
    el = _magnet()
    registry.register(pinned(el, native(el)))
    real = study_check.exit_values
    legs = []

    def nan_in_the_second(res):
        out = real(res)
        legs.append(1)
        if len(legs) == 2:
            out["sigma_w"] = float("nan")
        return out

    monkeypatch.setattr(study_check, "exit_values", nan_in_the_second)
    check = study_check.check_design(_line(el), _beam_config())
    assert not check.use and check.worst_key == "sigma_w" and check.worst == np.inf


def test_a_study_with_space_charge_runs_native():
    """At a current the envelope slices the map, so no network answers on the line."""
    from linac_gen.surrogates.study_check import check_design
    el = _magnet()
    registry.register(pinned(el, native(el)))
    check = check_design(_line(el), _beam_config(current=5.0))
    assert not check.use and check.nn_calls == 0 and "none answered" in check.note


# ---------------------------------------------------------------------------
# the Surrogates tab
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_tab_shows_each_sets_verdict(qapp, tmp_path):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
    state = AppState()
    tab = SurrogatesTab(state)
    els = [_magnet() for _ in range(3)]
    good = from_file(tmp_path, els[0], native(els[0]), name="a")
    bad = from_file(tmp_path, els[1], off(native(els[1])), name="b")
    memory = pinned(els[2], native(els[2]))
    for key, s in (("A", good), ("B", bad), ("C", memory)):
        tab._trained[key] = (s, tmp_path, s.metadata)
    tab._refresh_table()
    assert [tab._table.item(r, 5).text() for r in range(3)] == ["not checked yet",
                                                                "not checked yet", "—"]
    good.fitted_matrix(_ref(F_NEW, W))
    with pytest.warns(UserWarning), pytest.raises(OutOfScopeError):
        bad.fitted_matrix(_ref(F_NEW, W))
    state.set_running(True)
    state.set_running(False)                        # a run ended: its checks are shown
    assert tab._table.item(0, 5).text().startswith("within limits (")
    assert tab._table.item(1, 5).text().startswith("refused (")
    assert "Retrain it" in tab._table.item(1, 5).toolTip()
    third = from_file(tmp_path, els[2], native(els[2]), name="c")
    tab._trained["C"] = (third, tmp_path, third.metadata)
    tab._refresh_table()
    third.fitted_matrix(_ref(F_NEW, W))             # a study of another tab checked it
    assert tab._table.item(2, 5).text() == "not checked yet"
    tab.show()                                      # shown: the verdicts as they stand
    assert tab._table.item(2, 5).text().startswith("within limits (")
    tab.hide()
    use = tab._table.cellWidget(1, 3)
    assert use.isEnabled()                          # it can be ticked, and compared
    use.setChecked(True)
    assert "accuracy check refused" in tab._status.text()
    tab.deleteLater()


def test_the_tabs_single_training_takes_the_files_metadata(qapp, tmp_path):
    """As the batch path: weights it has just written are checked like any file's."""
    from types import SimpleNamespace

    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    from linac_gen.core.lattice import Lattice
    el = _magnet()
    lat = Lattice()
    lat.add(el)
    state = AppState()
    state.set_lattice(lat, None)
    tab = SurrogatesTab(state)
    d = tmp_path / "MAP"
    in_memory = meta(el, native(el))
    save_surrogate(mlp(), in_memory, d)
    tab._worker = SimpleNamespace(out_dir=d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab._on_finished(in_memory)
    surr, _dir, m = tab._trained[el.name]
    assert m is surr.metadata and m is not in_memory and m.accuracy_due
    tab.deleteLater()
