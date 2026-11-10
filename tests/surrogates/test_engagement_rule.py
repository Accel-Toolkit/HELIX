"""A surrogate answers for its element as the element stands.

The registry recognised an element by its settings too, so an error study's seed (an error
drawn on the element), a parameter scan's point or a deck edited since never met the weights.
It now recognises an element by its structure alone (class, length, field file, geometry,
aperture, steps, p_flag, ki, ka); the weights check the element when they are asked: its
structure still the one they were built on, their inputs read live (inside their range), the
error knobs folded in, the misalignments applied around it, and every other setting equal to
what the weights hold within 1e-9 (recorded in training before the samples, or, for a file
from before, taken from the element it is first built on), a setting the map cannot feel
aside; else the native element answers and the reason is counted, in a per-thread tally too.
A compensation runs native; a match (or a Pareto exploration) runs an element native from
start to end when its variable moves a setting the weights hold fixed or leaves their range;
a parameter scan of such a setting runs the element native.
"""
import contextlib
import copy
import json
import warnings

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import OutOfScopeError, SurrogateFieldMap, settings_of
from linac_gen.surrogates.training import load_surrogate, save_surrogate
from linac_gen.tracking.envelope import EnvelopeSolver
from tests.surrogates.accuracy_helpers import INIT, W, exact, meta, mlp, native, pinned
from tests.surrogates.test_frequency_jump import F_NEW, _cavity, _lattice, _magnet, _ref


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


def _line(el):
    return _lattice(Drift("D0", 100.0), el, Drift("D1", 100.0))


def _sigma(lat):
    return np.asarray(EnvelopeSolver(lat, _ref(F_NEW, W), dict(INIT), current=0.0).run()
                      .sigma_matrix)


# ---------------------------------------------------------------------------
# the lookup: the element's structure
# ---------------------------------------------------------------------------
def test_a_copy_set_otherwise_is_the_same_element_and_answers_exactly():
    """Another value of an input, an error on it, a misalignment: the same element, the
    network asked at the inputs the copy stands at (the error folded into the scale)."""
    el = _cavity(p_flag=0)
    surr = exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1)))
    registry.register(surr)
    twin = copy.deepcopy(el)
    twin.phase, twin.voltage_rel, twin.dx = -20.0, 0.02, 0.1
    view = registry.get_for_element(twin)
    assert view is not None and view._wrapped is twin and view.metadata is surr.metadata
    got = view.fitted_matrix(_ref(F_NEW, W))
    assert surr.nn_calls == 1
    np.testing.assert_allclose(got, native(twin), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("attr, value", [("length", 150.0), ("aperture", 12.0),
                                         ("n_steps", 7), ("p_flag", 1),
                                         ("field_file", "other.edz"), ("geom", 7700),
                                         ("ki", 0.5), ("ka", 0)])
def test_another_structure_is_another_element(attr, value):
    el = _magnet()
    registry.register(pinned(el, native(el)))
    other = copy.deepcopy(el)
    setattr(other, attr, value)
    with pytest.warns(UserWarning, match="DIFFERENT element"):
        assert registry.get_for_element(other) is None


# ---------------------------------------------------------------------------
# the settings, at each query
# ---------------------------------------------------------------------------
def test_the_settings_are_those_the_weights_hold_fixed():
    el = _cavity(p_flag=0)
    el.superpose_z0 = 0.0                            # how the parser saw the card written
    held = settings_of(el, {"phase"})
    assert {"ke", "kb", "scale", "frequency", "sync_phase_pin", "integrator_kind",
            "interp_kind"} <= set(held)
    assert not ({"phase", "voltage_rel", "phase_offset", "frequency_offset", "dx", "dy", "dz",
                 "tilt_deg", "pitch_deg", "yaw_deg", "hom_kick_x", "hom_kick_y", "name", "length",
                 "field_file", "geom", "aperture", "n_steps", "p_flag", "ki", "ka",
                 "superpose_z0"} & set(held))


def test_a_setting_the_weights_hold_fixed_refuses_by_name_and_is_counted():
    el = _magnet()
    surr = pinned(el, native(el))                   # built in memory: settings from el
    assert surr.metadata.settings_snapshot
    registry.register(surr)
    twin = copy.deepcopy(el)
    twin.kb = 1.3
    view = registry.get_for_element(twin)
    with pytest.raises(OutOfScopeError, match=r"its kb is 1\.3, 1\.0 when its weights were "
                                              r"loaded, and kb is not an input"):
        view.fitted_matrix(_ref(F_NEW, W))
    assert surr.refusals == {"its kb changed": 1} == view.refusals
    twin.kb = 1.0                                   # set back: it answers again
    view.fitted_matrix(_ref(F_NEW, W))
    assert surr.nn_calls == 1


def test_a_run_on_an_element_set_otherwise_is_the_native_run():
    el = _magnet()
    lat = _line(el)
    registry.register(pinned(el, native(el)))
    twin = copy.deepcopy(lat)
    twin.elements[1].kb = 1.3
    got = _sigma(twin)
    registry.clear()
    np.testing.assert_array_equal(got, _sigma(copy.deepcopy(twin)))


def test_training_records_the_settings_and_a_refusal_says_so(tmp_path):
    from linac_gen.surrogates.training import train_surrogate_for_element
    el = _magnet()
    d = tmp_path / "MAP"
    train_surrogate_for_element(
        element=el, ref_template=_ref(F_NEW, W), n_samples=16, ref_w_kin_range=(4.0, 6.0),
        param_ranges={"scale": (0.9, 1.1)}, hidden_dims=(4,), epochs=1, out_dir=d,
        element_key="MAP")
    written = json.loads((d / "metadata.json").read_text())["train_settings"]
    assert written == settings_of(el, {"scale"}) and "scale" not in written and "kb" in written
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp_, m = load_surrogate(d)
    assert m.train_settings == written and not m.settings_snapshot
    m.accuracy_due = False                          # not what this test is about
    twin = copy.deepcopy(el)
    twin.kb = 1.3
    with pytest.raises(OutOfScopeError, match="1.0 in training"):
        SurrogateFieldMap(twin, mlp_, m).fitted_matrix(_ref(F_NEW, W))


def test_a_file_from_before_takes_the_settings_of_the_element_it_is_built_on(tmp_path):
    el = _magnet()
    d = tmp_path / "old"
    save_surrogate(mlp(), meta(el, native(el)), d)
    data = json.loads((d / "metadata.json").read_text())
    data.pop("train_settings")
    (d / "metadata.json").write_text(json.dumps(data))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp_, m = load_surrogate(d)
    assert m.train_settings is None
    SurrogateFieldMap(el, mlp_, m)
    assert m.settings_snapshot and m.train_settings == settings_of(el, set())


def test_an_ncells_setting_runs_native():
    """An ``NCELLS`` (a surrogate only the Python API makes) set otherwise than its weights hold
    is the native element's, not a wrong answer."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.ncells import NCells
    el = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=1, frequency_mhz=325.0)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=325.0)
    surr = pinned(el, np.asarray(copy.deepcopy(el).fitted_matrix(ref.copy())), train_frequency=325.0)
    surr.fitted_matrix(ref.copy())
    twin = copy.deepcopy(el)
    twin.eot_v_per_m = 2.1e6
    with pytest.raises(OutOfScopeError, match="its eot_v_per_m is"):
        surr.bound_to(twin).fitted_matrix(ref.copy())


# ---------------------------------------------------------------------------
# what a study does with it
# ---------------------------------------------------------------------------
def test_error_study_seeds_use_the_weights_and_answer_exactly():
    """Errors drawn on the element are the seed's: the network answers each seed with the
    errors folded in, and the line is the native line's."""
    el = _cavity(p_flag=0)
    lat = _line(el)
    surr = exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1)))
    registry.register(surr)
    rng = np.random.default_rng(3)
    seeds = []
    for _ in range(4):
        s = copy.deepcopy(lat)
        cav = s.elements[1]
        cav.phase_offset, cav.voltage_rel = rng.normal(0, 1.0), rng.normal(0, 0.01)
        cav.dx, cav.tilt_deg = rng.normal(0, 0.2), rng.normal(0, 0.1)
        seeds.append(s)
    got = [_sigma(copy.deepcopy(s)) for s in seeds]
    assert surr.nn_calls == len(seeds) and surr.refusals == {}
    registry.clear()
    for g, s in zip(got, seeds, strict=True):
        np.testing.assert_allclose(g, _sigma(copy.deepcopy(s)), rtol=1e-9, atol=1e-12)


def test_a_scan_answers_at_every_value_of_an_input_and_goes_native_for_a_setting():
    """(The phase input of a map whose field follows the running phase takes the reference phase
    at its entrance, about 20 deg here, behind the drift: the range is wide enough for it.)"""
    el = _cavity(p_flag=0)
    lat = _line(el)
    surr = exact(el, params=(("phase", -120.0, 60.0),))
    registry.register(surr)
    for value in (-50.0, -30.0, -10.0):              # the first is not the element's own
        point = copy.deepcopy(lat)
        point.elements[1].phase = value
        _sigma(point)
    assert surr.nn_calls == 3
    for value in (0.9, 1.1):                         # scale: held fixed by these weights
        point = copy.deepcopy(lat)
        point.elements[1].scale = value
        _sigma(point)
    assert surr.nn_calls == 3 and surr.refusals == {"its scale changed": 2}


def test_a_compensation_runs_native(monkeypatch):
    from linac_gen.failures import compensation
    seen = {}
    monkeypatch.setattr(compensation, "_compensate",
                        lambda *a, **k: seen.setdefault("suspended", registry.is_suspended()))
    el = _magnet()
    registry.register(pinned(el, native(el)))
    compensation.compensate(None, None, None, {}, {}, compensation.CompensationConfig())
    assert seen["suspended"] is True and not registry.is_suspended()


def _bcfg():
    return BeamConfig(species="proton", energy=W, frequency=F_NEW, n_particles=10,
                      distribution="waterbag", emit_nx=0.25, beta_x=2.0, emit_ny=0.25, beta_y=2.0,
                      emit_z=0.3, beta_z=10.0)


def _match_line(el):
    lat = _lattice(Drift("D0", 100.0))
    lat.add(Adjust("A1", target="MAP", param_idx=5, link_group=0, vmin=0.5, vmax=1.5,
                   start_step=0.05))
    lat.add(el)
    lat.add(Drift("D1", 200.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=1.0, y_mm=0.0, phi_or_z=0.0))
    return lat


def test_a_match_moving_a_setting_the_weights_hold_runs_that_element_native():
    """Its kb is the variable and not an input: the element is native from the first
    evaluation to the last, and its decision is as it was after the match."""
    from linac_gen.matching.engine import match
    el = _magnet()
    lat = _match_line(el)
    surr = pinned(el, native(el))
    registry.register(surr)
    with pytest.warns(UserWarning, match="'MAP' runs native in this match: its kb is a variable"):
        match(lat, _bcfg(), max_iter=4)
    assert surr.nn_calls == 0 and surr.refusals == {}
    el.kb = 1.0                                      # the matcher moved it
    assert registry.get_for_element(el) is surr      # and nothing of the match stays


def test_a_match_moving_an_input_uses_the_weights():
    from linac_gen.matching.engine import match
    el = _magnet()
    lat = _match_line(el)
    surr = exact(el, params=(("kb", 0.5, 1.5),))
    registry.register(surr)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        match(lat, _bcfg(), max_iter=4)
    assert not [w for w in seen if "runs native in this match" in str(w.message)]
    assert surr.nn_calls > 1 and surr.refusals == {}


# ---------------------------------------------------------------------------
# the Parameter Scan dialog
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _scan(lattice, values, param="phase", convergence=None):
    from types import SimpleNamespace

    from linac_gen_gui.interphase.dialogs.parameter_scan import _ScanWorker
    cfg = BeamConfig(species="proton", energy=W, frequency=F_NEW, current=0.0, n_particles=10)
    worker = _ScanWorker(SimpleNamespace(lattice=lattice, beam_config=cfg), lattice.elements[1],
                         param, np.asarray(values, dtype=float), "env", "sigma_x", "last",
                         convergence=convergence)
    got = {}
    worker.finished_ok.connect(lambda xs, ys: got.setdefault("ys", ys))
    worker.failed.connect(lambda m: got.setdefault("failed", m))
    worker.run()                                     # synchronous
    assert "failed" not in got, got.get("failed")
    return worker, got["ys"]


def test_a_parameter_scan_checks_its_design_lattice_and_uses_the_weights(qapp):
    el = _cavity(p_flag=0)
    lat = _line(el)
    surr = exact(el, params=(("phase", -60.0, 0.0),))
    registry.register(surr)
    worker, ys = _scan(lat, (-50.0, -40.0, -30.0))
    assert worker.surrogates.use and np.all(np.isfinite(ys))
    assert surr.nn_calls == worker.surrogates.nn_calls + 3


def test_a_parameter_scan_whose_design_lattice_disagrees_runs_native(qapp):
    """(A magnet scanned in dx: weights with no input would answer at every point.)"""
    el = _magnet()
    lat = _line(el)
    surr = pinned(el, np.eye(6))                     # nothing like the magnet
    registry.register(surr)
    worker, ys = _scan(lat, (0.0, 0.1, 0.2), param="dx")
    assert not worker.surrogates.use and worker.surrogates.note.startswith("surrogates not used")
    assert surr.nn_calls == worker.surrogates.nn_calls == 1   # the design check, not the scan


def test_the_scan_says_on_its_status_line_what_its_points_did_with_the_surrogates(qapp):
    """The status line counts the network answers over the points (not the design check's
    verdict alone), the tooltip says the rest."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from linac_gen_gui.interphase.dialogs.parameter_scan import ParameterScanDialog
    from PyQt6.QtWidgets import QLabel

    from linac_gen.surrogates.base import Tally
    counted = Tally()
    counted.calls, counted.refused = 1, {"its scale changed": 2}
    cases = [
        (SimpleNamespace(use=True, note="surrogates used: ..."), counted, "",
         " · surrogates: 1 network answers, 2 traversals native",
         "surrogates used: ...; over the scan's points 1 network answers, native: its scale "
         "changed 2"),
        (SimpleNamespace(use=True, note="surrogates used: ..."), Tally(), "its weights hold x",
         " · surrogates: 0 network answers",
         "surrogates used: ...; over the scan's points 0 network answers; its weights hold x"),
        (SimpleNamespace(use=False, note="surrogates not used: ..."), None, "",
         " · surrogates not used", "surrogates not used: ..."),
        (SimpleNamespace(use=False, note=""), None, "", "", ""),
    ]
    for check, tally_, held, shown, tip in cases:
        dialog = SimpleNamespace(_run_btn=MagicMock(), _stop_btn=MagicMock(),
                                 _progress=MagicMock(), _status=QLabel(),
                                 _worker=SimpleNamespace(surrogates=check, counted=tally_,
                                                         held_note=held))
        ParameterScanDialog._on_scan_done(dialog, np.array([1.0, 2.0]), np.array([0.5, 0.6]))
        assert dialog._status.text() == "done · 2/2 points" + shown
        assert dialog._status.toolTip() == tip


def test_a_scan_of_a_structure_or_a_held_setting_runs_the_element_native(qapp):
    """A length the weights were not built at is another element, and a setting they hold is
    one they cannot follow: the element runs native at every point, the first one included,
    so the scan is the native scan; the scan says so."""
    from linac_gen_gui.interphase.dialogs.parameter_scan import surrogates_status
    el = _cavity(p_flag=0)
    lat = _line(el)
    registry.register(exact(el, params=(("phase", -120.0, 60.0),)))
    for param, values in (("length", (200.0, 190.0, 180.0)), ("scale", (1.0, 0.98, 1.02))):
        worker, ys = _scan(lat, values, param=param)
        assert worker.surrogates.use and worker.counted.calls == 0
        assert worker.held_note == f"its weights hold {param} fixed, so MAP runs native at " \
                                   "every point"
        short, whole = surrogates_status(worker.surrogates, worker.counted, worker.held_note)
        assert short == " · surrogates: 0 network answers" and whole.endswith(worker.held_note)
        with registry.suspended():
            _, want = _scan(lat, values, param=param)
        np.testing.assert_array_equal(ys, want)


def test_a_scan_with_the_sacherer_envelope_runs_native_and_a_failed_check_does_not_stop_it(
        qapp, monkeypatch):
    from linac_gen.surrogates import study_check
    el = _magnet()
    lat = _line(el)
    registry.register(pinned(el, native(el)))
    worker, ys = _scan(lat, (0.0, 0.1), param="dx", convergence={"env_solver": "sacherer"})
    assert np.all(np.isfinite(ys)) and not worker.surrogates.use
    assert worker.surrogates.note == ("surrogates not used: the Sacherer envelope does not use "
                                      "them")

    def broken(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(study_check, "check_design", broken)
    worker, ys = _scan(lat, (0.0, 0.1), param="dx")
    assert np.all(np.isfinite(ys)) and not worker.surrogates.use
    assert worker.surrogates.note == ("surrogates not used: the check of the design lattice "
                                      "failed (RuntimeError: boom)")


# ---------------------------------------------------------------------------
# the structure, at each query
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attr, value", [("length", 150.0), ("n_steps", 7), ("p_flag", 1)])
def test_an_element_whose_structure_changed_since_is_refused(attr, value):
    """The lookup decides once per run; the weights refuse the element once it is no longer
    the one they were built on (an edit, a scan of its length)."""
    el = _magnet()
    surr = pinned(el, native(el))
    registry.register(surr)
    assert registry.get_for_element(el) is surr
    surr.fitted_matrix(_ref(F_NEW, W))
    setattr(el, attr, value)
    assert registry.get_for_element(el) is surr                   # decided for the run
    with pytest.raises(OutOfScopeError, match=rf"its {attr} is {value!r}, .* when its weights "
                                              "were built on it"):
        surr.fitted_matrix(_ref(F_NEW, W))
    assert surr.refusals == {f"its {attr} changed": 1} and surr.nn_calls == 1
    twin = copy.deepcopy(el)                                       # a copy of the edited one
    with pytest.raises(OutOfScopeError, match=f"its {attr} is"):
        registry.get_for_element(twin).fitted_matrix(_ref(F_NEW, W))


def test_another_class_is_another_element():
    from tests.surrogates.test_frequency_jump import _cavity_1d
    el = _cavity()
    registry.register(pinned(el, native(el)))
    with pytest.warns(UserWarning, match="DIFFERENT element"):
        assert registry.get_for_element(_cavity_1d()) is None


# ---------------------------------------------------------------------------
# the settings: what is held, and how closely
# ---------------------------------------------------------------------------
def test_a_pinned_ncells_calibration_is_a_setting_the_weights_hold():
    """A calibration pinned after training (a reliability pin, a multibunch replay) is not
    the one each training sample took afresh: the native element answers."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.ncells import NCells
    el = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=1, frequency_mhz=325.0, sync_phase=True)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=325.0)
    surr = pinned(el, np.asarray(copy.deepcopy(el).fitted_matrix(ref.copy())), train_frequency=325.0)
    surr.fitted_matrix(ref.copy())
    twin = copy.deepcopy(el)
    twin.sync_phase_pin = 12.0
    with pytest.raises(OutOfScopeError, match="its sync_phase_pin is 12.0, None"):
        surr.bound_to(twin).fitted_matrix(ref.copy())


def test_an_ncells_error_on_its_phase_is_folded_into_its_input():
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.ncells import NCells
    el = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=0, frequency_mhz=325.0)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=325.0)
    surr = exact(el, params=(("theta_s_deg", -60.0, 0.0),), template=ref)
    twin = copy.deepcopy(el)
    twin.phase_offset = 2.0
    got = surr.bound_to(twin).fitted_matrix(ref.copy())
    assert surr.nn_calls == 1 and surr.refusals == {}
    np.testing.assert_allclose(got, np.asarray(copy.deepcopy(twin).fitted_matrix(ref.copy())),
                               rtol=1e-9, atol=1e-12)


def test_an_input_named_otherwise_than_its_attribute_is_an_input():
    from linac_gen.elements.ncells import NCells
    el = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=0, frequency_mhz=325.0)
    assert "frequency_mhz" not in settings_of(el, {"frequency"})
    assert "frequency_mhz" in settings_of(el, set())
    surr = pinned(el, np.eye(6), params=(("frequency", 320.0, 330.0),), train_frequency=325.0)
    assert surr.follows("frequency_mhz") and surr.follows("frequency")


def test_how_the_card_was_written_is_not_a_setting():
    """The parser marks a map written after a ``SUPERPOSE_MAP 0`` (the same physics)."""
    el = _magnet()
    surr = pinned(el, native(el))
    twin = copy.deepcopy(el)
    twin.superpose_z0 = 0.0
    surr.bound_to(twin).fitted_matrix(_ref(F_NEW, W))
    assert surr.nn_calls == 1 and surr.refusals == {}


def test_a_setting_written_and_read_back_is_the_same_setting():
    """A deck keeps ten digits; a held setting a match left at full precision is still held
    after a save and a reload, and a real change still refuses."""
    el = _magnet()
    el.kb = 0.31038271604621
    surr = pinned(el, native(el))
    twin = copy.deepcopy(el)
    twin.kb = float(f"{el.kb:.10g}")                               # as tracewin_writer writes it
    assert twin.kb != el.kb
    surr.bound_to(twin).fitted_matrix(_ref(F_NEW, W))
    assert surr.nn_calls == 1
    twin.kb = el.kb * (1.0 + 3e-9)
    with pytest.raises(OutOfScopeError, match="its kb is"):
        surr.bound_to(twin).fitted_matrix(_ref(F_NEW, W))
    from linac_gen.surrogates.base import _same_setting
    assert _same_setting(2, 2.0) and _same_setting(2, 2.0 + 1e-10)    # an int held, a float now
    assert not _same_setting(2, 2.0 + 1e-6)
    assert not _same_setting(True, 1.5) and _same_setting("kd", "kd")


def test_a_setting_a_static_map_cannot_feel_is_not_held():
    """A static magnetic map has no RF field to phase and no electric field to scale: its
    phase, its ke and its frequency (the clock is converted) do not refuse, and the answer is
    the native map; its kb does."""
    el = _magnet()
    surr = exact(el)
    twin = copy.deepcopy(el)
    twin.phase, twin.ke = 40.0, 3.0
    got = surr.bound_to(twin).fitted_matrix(_ref(F_NEW, W))
    np.testing.assert_allclose(got, native(twin), rtol=1e-9, atol=1e-12)
    twin.frequency = 162.5
    got = surr.bound_to(twin).fitted_matrix(_ref(162.5, W))
    np.testing.assert_allclose(got, native(twin, f=162.5), rtol=1e-9, atol=1e-12)
    assert surr.refusals == {}
    assert all(surr.follows(a) for a in ("phase", "ke", "frequency")) and not surr.follows("kb")
    twin.kb = 1.3
    with pytest.raises(OutOfScopeError, match="its kb is 1.3"):
        surr.bound_to(twin).fitted_matrix(_ref(162.5, W))


def test_the_field_map_integrator_is_a_setting_the_weights_hold(monkeypatch):
    from linac_gen.elements.field_map_3d import FieldMap3D
    el = _magnet()
    surr = pinned(el, native(el))
    other = "dkd" if FieldMap3D.integrator_kind != "dkd" else "kd"
    monkeypatch.setattr(FieldMap3D, "integrator_kind", other)
    with pytest.raises(OutOfScopeError, match=f"its integrator_kind is '{other}'"):
        surr.fitted_matrix(_ref(F_NEW, W))


def test_follows_is_what_the_folds_take():
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.ncells import NCells
    cav = _cavity(p_flag=0)
    assert pinned(cav, np.eye(6), params=(("phase", -60.0, 0.0),)).follows("phase_offset")
    assert not pinned(cav, np.eye(6), params=(("ke", 0.5, 1.5),)).follows("phase_offset")
    assert pinned(cav, np.eye(6), params=(("scale", 0.9, 1.1),)).follows("voltage_rel")
    assert not pinned(cav, np.eye(6), params=(("phase", -60.0, 0.0),)).follows("voltage_rel")
    assert not pinned(cav, np.eye(6), params=(("phase", -60.0, 0.0),)).follows("frequency_offset")
    mag = _magnet()
    assert pinned(mag, np.eye(6)).follows("phase_offset")          # nothing RF to move
    nc = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=0, frequency_mhz=325.0)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=325.0)
    nsurr = pinned(nc, np.asarray(copy.deepcopy(nc).fitted_matrix(ref.copy())),
                   params=(("theta_s_deg", -60.0, 0.0),), train_frequency=325.0)
    assert nsurr.follows("phase_offset") and not nsurr.follows("voltage_rel")


def test_an_rf_shift_no_fold_takes_refuses():
    """An element the folds do not cover has no input for the RF shift of
    SET_BEAM_PHASE_ERROR: it runs native."""
    from linac_gen.elements.base import FieldMapElement

    class Gap(FieldMapElement):
        def __init__(self):
            super().__init__(name="GAP", length=10.0, aperture=10.0, n_steps=1)

        def track_rk4(self, beam, ds):
            return None

        def fitted_matrix(self, ref):
            return np.eye(6)

        def fitted_matrix_slice(self, ref, ds_mm):
            return np.eye(6)

    surr = pinned(Gap(), np.eye(6))
    ref = _ref(F_NEW, W)
    surr.fitted_matrix(ref)
    ref.rf_shift = 2.0
    with pytest.raises(OutOfScopeError, match="the RF shift of SET_BEAM_PHASE_ERROR moves"):
        surr.fitted_matrix(ref)


# ---------------------------------------------------------------------------
# training: the targets and the settings
# ---------------------------------------------------------------------------
def test_training_takes_its_targets_as_a_query_folds_the_error_knobs(tmp_path):
    """The targets are the design element's map (its error knobs at zero), as a query folds
    the knobs into the inputs; the element itself is left as it was."""
    from linac_gen.surrogates.training import _at_element_frequency, generate_training_data
    el = _cavity(p_flag=0)
    el.phase_offset, el.voltage_rel = 10.0, 0.05
    X, Y, _info = generate_training_data(el, _ref(F_NEW, W), n_samples=3,
                                         ref_w_kin_range=(4.0, 6.0),
                                         param_ranges={"phase": (-60.0, 0.0)}, seed=1)
    design = copy.deepcopy(el)
    design.phase_offset = design.voltage_rel = 0.0
    for x, y in zip(X, Y, strict=True):
        design.phase = float(x[3])
        design.reset_run_state()
        ref = _ref(F_NEW, float(x[0]))
        _at_element_frequency(design, ref)
        np.testing.assert_allclose(y, np.asarray(design.fitted_matrix(ref)).ravel(),
                                   rtol=1e-12, atol=1e-14)
    assert (el.phase_offset, el.voltage_rel) == (10.0, 0.05)


def test_training_records_the_settings_before_its_samples(tmp_path):
    """The GUI leaves the element open to edits during a training: the held settings are the
    ones the samples were taken at."""
    from linac_gen.surrogates.training import train_surrogate_for_element
    el = _magnet()

    def edit(info):
        el.kb = 1.3
    _mlp, m = train_surrogate_for_element(
        element=el, ref_template=_ref(F_NEW, W), n_samples=16, ref_w_kin_range=(4.0, 6.0),
        param_ranges={"scale": (0.9, 1.1)}, hidden_dims=(4,), epochs=1, out_dir=tmp_path / "M",
        element_key="MAP", progress_callback=edit)
    assert el.kb == 1.3 and m.train_settings["kb"] == 1.0


def test_register_multi_says_which_names_run_native(monkeypatch, tmp_path, capsys):
    from linac_gen.io import tracewin_parser
    from linac_gen.surrogates import cli
    a, b = _magnet(), _magnet()
    a.name, b.name = "M1", "M2"
    b.kb = 1.3
    lat = _lattice(a, b)
    m = meta(a, native(a))
    m.train_settings = settings_of(a, set())
    save_surrogate(mlp(3), m, tmp_path / "w")
    deck = tmp_path / "deck.dat"
    deck.write_text("END\n")
    monkeypatch.setattr(tracewin_parser, "parse_tracewin", lambda path: (lat, None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli.main(["register-multi", "--source", str(tmp_path / "w"), "--lattice", str(deck),
                       "--names", "M1", "M2"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "OK   M1" in err and "WARN M2" in err and "its kb is 1.3, 1.0 in training" in err
    assert "1 run native" in err


# ---------------------------------------------------------------------------
# counting: the copies' refusals, and per thread
# ---------------------------------------------------------------------------
def test_a_copy_with_its_cavity_switched_off_is_refused_by_the_folds():
    """Its weights take no scale: the switched-off copy is the native cavity's, refused as an
    error knob the folds cannot take (the lookup itself finds the same element)."""
    el = _cavity(p_flag=0)
    surr = exact(el, params=(("phase", -120.0, 60.0),))
    registry.register(surr)
    twin = copy.deepcopy(el)
    twin.voltage_rel = -1.0
    view = registry.get_for_element(twin)
    assert view is not None
    with pytest.raises(OutOfScopeError, match="its voltage_rel is -1"):
        view.fitted_matrix(_ref(F_NEW, W))
    assert surr.refusals == {"error knob": 1} and surr.nn_calls == 0


def test_misalignments_are_applied_around_the_weights():
    """dy, dz, pitch, yaw and the HOM kicks, as dx and the tilt: the copy answers with the
    native line's result."""
    el = _cavity(p_flag=0)
    lat = _line(el)
    surr = exact(el, params=(("phase", -120.0, 60.0),))
    registry.register(surr)
    seed = copy.deepcopy(lat)
    cav = seed.elements[1]
    cav.dy, cav.dz, cav.pitch_deg, cav.yaw_deg = 0.1, 0.2, 0.05, 0.05
    cav.hom_kick_x, cav.hom_kick_y = 1e-4, -1e-4
    got = _sigma(copy.deepcopy(seed))
    assert surr.nn_calls == 1 and surr.refusals == {}
    registry.clear()
    np.testing.assert_allclose(got, _sigma(copy.deepcopy(seed)), rtol=1e-9, atol=1e-12)


def test_a_tally_counts_its_own_thread_only():
    import threading

    from linac_gen.surrogates.base import tally
    el = _magnet()
    surr = pinned(el, native(el))
    other = threading.Thread(target=lambda: [surr.fitted_matrix(_ref(F_NEW, W))
                                             for _ in range(3)])
    with tally() as outer, tally() as t:
        surr.fitted_matrix(_ref(F_NEW, W))
        other.start()
        other.join()
        el.kb = 1.3
        with pytest.raises(OutOfScopeError):
            surr.fitted_matrix(_ref(F_NEW, W))
    assert (t.calls, t.refused) == (1, {"its kb changed": 1}) == (outer.calls, outer.refused)
    assert surr.nn_calls == 4


def test_the_compare_report_lists_the_traversals_left_native():
    from linac_gen.surrogates.compare import compare_envelope
    el = _magnet()
    lat = _line(el)
    surr = pinned(el, native(el))
    el.kb = 1.3                                      # after the weights were built
    rep = compare_envelope(lat, _ref(F_NEW, W), dict(INIT), surrogates=[surr])
    assert rep.nn_calls == 0
    assert "traversals the weights left to the native element: its kb changed 1" in rep.notes
    assert not any("never queried" in n for n in rep.notes)


# ---------------------------------------------------------------------------
# the backtracker: the exact inverse wherever the forward push was native
# ---------------------------------------------------------------------------
@pytest.fixture
def mp_flags():
    mp, fast = registry.is_mp_enabled(), registry.is_fast_path_enabled()
    yield
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(fast)


@pytest.mark.parametrize("fast, kb", [(False, 1.0), (True, 1.3)])
def test_the_backtracker_inverts_a_copy_left_native_exactly(mp_flags, fast, kb):
    """A copy (an error seed) the forward run pushed through its native element — the fast
    path off, or a setting its weights hold changed — is undone exactly, with no word of a
    surrogate."""
    from linac_gen.tracking.backtrack import backtrack_distribution
    from linac_gen.tracking.tracker import Tracker
    from tests.surrogates.test_frequency_jump import _beam
    el = _magnet()
    lat = _line(el)
    surr = pinned(el, native(el))
    registry.register(surr)
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(fast)
    seed = copy.deepcopy(lat)
    seed.elements[1].kb = kb
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(seed, beam).run()
    assert surr.nn_calls == 0
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        backtrack_distribution(seed, beam, _ref(F_NEW, W))
    assert not [w for w in seen if "driven by a surrogate" in str(w.message)]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-10)
    assert surr.nn_calls == 0 and sum(surr.refusals.values()) == (1 if fast else 0)


def test_the_backtracker_undoes_a_copy_its_weights_pushed_exactly(mp_flags):
    """The fast path's push is a matrix slice per call: the backtracker undoes it with the
    inverse of each slice, from the record the forward run left on the beam."""
    from linac_gen.tracking.backtrack import backtrack_distribution
    from linac_gen.tracking.tracker import Tracker
    from tests.surrogates.test_frequency_jump import _beam
    el = _magnet()
    lat = _line(el)
    surr = pinned(el, native(el))
    registry.register(surr)
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    seed = copy.deepcopy(lat)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(seed, beam).run()
    calls = surr.nn_calls
    assert calls >= 1
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        backtrack_distribution(seed, beam, _ref(F_NEW, W))
    assert not [w for w in seen if "surrogate" in str(w.message)]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-10)
    assert surr.nn_calls == calls                    # the backtracker's question is not counted


# ---------------------------------------------------------------------------
# matching: one model per run
# ---------------------------------------------------------------------------
def test_a_match_whose_bounds_leave_the_training_range_runs_that_element_native():
    from linac_gen.matching.engine import match
    el = _magnet()
    lat = _match_line(el)                            # ADJUST kb over 0.5..1.5
    surr = exact(el, params=(("kb", 0.9, 1.1),))
    registry.register(surr)
    with pytest.warns(UserWarning, match="'MAP' runs native in this match: its kb is a variable, "
                                         "between bounds that leave its weights' training range"):
        match(lat, _bcfg(), max_iter=4)
    assert surr.nn_calls == 0 and surr.refusals == {}


def test_a_match_that_mixed_the_network_and_the_native_element_says_so():
    """A phase input the running phase moves too is checked as the match runs: weights that
    answered some evaluations and left their range at others are named after it."""
    from linac_gen.matching.engine import _surrogates_decided_once
    el = _cavity(p_flag=0)
    lat = _lattice(Drift("D0", 100.0))
    lat.add(Adjust("A1", target="MAP", param_idx=2, link_group=0, vmin=-50.0, vmax=-10.0,
                   start_step=1.0))
    lat.add(el)
    lat.add(Drift("D1", 200.0))
    surr = exact(el, params=(("phase", -60.0, 0.0),))
    registry.register(surr)

    def run(lattice, beam_cfg):
        for phase in (-30.0, 170.0):
            el.phase = phase
            with contextlib.suppress(OutOfScopeError):
                surr.fitted_matrix(_ref(F_NEW, W))
        return "done"
    with pytest.warns(UserWarning, match="this match left its weights' training range in 1 "
                                         "evaluations, which ran it native, and used the "
                                         "weights in 1"):
        assert _surrogates_decided_once(run)(lat, _bcfg()) == "done"


def test_a_pareto_exploration_decides_the_surrogates_once():
    pytest.importorskip("pymoo")
    from linac_gen.matching import multiobjective as mo
    assert mo.pareto_optimize.__wrapped__ is mo._pareto_optimize
    assert mo.pareto_optimize.__name__ == "pareto_optimize"
    el = _magnet()
    lat = _match_line(el)
    surr = pinned(el, native(el))
    registry.register(surr)
    with pytest.warns(UserWarning, match="'MAP' runs native in this pareto_optimize: its kb is a "
                                         "variable, not an input of its weights"):
        mo.pareto_optimize(lat, _bcfg(), ["exit_sigma_x", "exit_sigma_y"], pop_size=4, n_gen=1,
                           seed=0)
    assert surr.nn_calls == 0 and surr.refusals == {}


def test_a_match_warns_of_a_card_it_skips_once():
    from linac_gen.matching.engine import match
    el = _magnet()
    lat = _lattice(Drift("D0", 100.0))
    lat.add(Adjust("A0", target="NOPE", param_idx=5, link_group=1, vmin=0.5, vmax=1.5,
                   start_step=0.05))
    lat.add(Adjust("A1", target="MAP", param_idx=5, link_group=0, vmin=0.5, vmax=1.5,
                   start_step=0.05))
    lat.add(el)
    lat.add(Drift("D1", 200.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=1.0, y_mm=0.0, phi_or_z=0.0))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        match(lat, _bcfg(), max_iter=2)
    assert len([w for w in seen if "skipping ADJUST" in str(w.message)]) == 1
