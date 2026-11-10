"""A surrogate at an RF frequency jump.

A cavity whose frequency differs from the reference's makes the jump on entry
when no FREQ card did it first: Δφ takes the factor f_new/f_old and the reference
moves to the new clock.  Two seams missed that.

* The multiparticle fast path pushes the particles with a matrix and advances the
  reference itself, with no frequency handling, so the clock stayed at the old
  frequency through the cavity and after it.  It now hands such a traversal to the
  native element.
* The training data computed each target at the template's frequency, so for a
  cavity at another frequency the target held the jump factor, which the envelope
  applies again at its seam.  Targets are now computed at the cavity's own frequency,
  and ``SurrogateFieldMap.fitted_matrix`` applies the factor when asked at another
  clock: the same contract as the native element's.
"""
import copy

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.lattice_commands import Freq
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel
from linac_gen.surrogates import registry, training
from linac_gen.surrogates.base import MlpHead, Scope, SurrogateFieldMap, SurrogateMetadata
from linac_gen.surrogates.training import generate_training_data
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, reference_at
from linac_gen.tracking.tracker import Tracker

F_OLD = 162.5
F_NEW = 325.0


def _map(channel, freq, L=200.0, p_flag=0):
    nz, n = 11, 3
    x = np.linspace(-10.0, 10.0, n)
    z = np.linspace(0.0, L, nz)
    fd = FieldMapData(z=z, frequency=freq)
    fd.channels[channel] = FieldChannel(
        geometry=7, x=x, y=x, z=z, Fx=np.zeros((n, n, nz)), Fy=np.zeros((n, n, nz)),
        Fz=np.full((n, n, nz), 1.0))
    return FieldMap3D(name="MAP", length=L, field_data=fd, scale=1.0, phase=-30.0,
                      frequency=freq, aperture=30.0, n_steps=20, p_flag=p_flag)


def _cavity(freq=F_NEW, p_flag=0):
    return _map(Channel.RF_E, freq, p_flag=p_flag)


def _cavity_1d(freq=F_NEW, p_flag=0):
    z = np.linspace(0.0, 200.0, 101)
    fd = FieldMapData(z=z, Ez=np.sin(np.pi * z / 200.0), symmetry="1d")
    return FieldMap("MAP", length=200.0, field_data=fd, scale=1.0, phase=-30.0, frequency=freq,
                    aperture=30.0, n_steps=100, p_flag=p_flag)


def _magnet(freq=F_NEW):
    return _map(Channel.STAT_B, freq)


def _ref(f=F_OLD, w=5.0):
    return ReferenceParticle(species=PROTON, w_kin=w, frequency=f)


def _lattice(*elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    return lat


def _beam(ref):
    b = Beam(ref=ref.copy(), n_particles=64, current=0.0)
    b.particles[:] = (np.random.default_rng(1).normal(size=(64, 6))
                      * [0.5, 0.5, 0.5, 0.5, 2.0, 1e-4])
    return b


def _pinned(el, M, w_lo=2.0):
    """A real surrogate whose output is pinned to ``M`` (output std = 0)."""
    mlp = MlpHead(input_dim=3, output_dim=36, hidden_dims=(4,))
    meta = SurrogateMetadata(
        element_key=el.name, element_class=type(el).__name__,
        architecture={"input_dim": 3, "output_dim": 36, "hidden_dims": [4],
                      "activation": "silu", "param_names": []},
        scope=Scope(input_names=["w_kin", "beta", "gamma"], input_lo=np.array([w_lo, 0.0, 1.0]),
                    input_hi=np.array([10.0, 1.0, 100.0])),
        input_norm={"mean": [0.0] * 3, "std": [1.0] * 3},
        output_norm={"mean": [float(v) for v in M.ravel()], "std": [0.0] * 36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="lh", created_iso="")
    return SurrogateFieldMap(el, mlp, meta)


@pytest.fixture
def fast_path():
    mp, fast = registry.is_mp_enabled(), registry.is_fast_path_enabled()
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    yield
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(fast)


def _run(make_line, ref, *, surrogate=True, w_lo=2.0):
    """Run the line; ``surrogate``: with its map's surrogate registered, pinned to the
    map's matrix at the reference it meets, on its own clock."""
    lat = _lattice(*make_line())
    surr = None
    if surrogate:
        i, el = next((i, e) for i, e in enumerate(lat.elements) if isinstance(e, FieldMap3D))
        at_map = reference_at(lat, ref, i)
        if el._has_electric_channel():
            at_map.frequency = el.effective_frequency
        surr = _pinned(el, copy.deepcopy(el).fitted_matrix(at_map), w_lo=w_lo)
        registry.register(surr)
    beam = _beam(ref)
    Tracker(lat, beam).run()
    registry.clear()
    return beam, surr


def _assert_same_as_native(beam, native):
    np.testing.assert_array_equal(beam.particles, native.particles)
    assert beam.ref.frequency == native.ref.frequency
    assert beam.ref.phi_s == native.ref.phi_s
    assert beam.ref.w_kin == native.ref.w_kin


@pytest.mark.parametrize("p_flag", [0, 1], ids=["absolute", "calibrated"])
def test_a_jump_without_a_card_goes_to_the_native_cavity(fast_path, p_flag):
    def line():
        return [Drift("D0", 50.0), _cavity(p_flag=p_flag), Drift("D1", 50.0)]
    native, _ = _run(line, _ref(), surrogate=False)
    beam, surr = _run(line, _ref())
    assert native.ref.frequency == F_NEW
    _assert_same_as_native(beam, native)
    assert surr.nn_calls == 0


def test_a_frequency_offset_goes_to_the_native_cavity(fast_path):
    """A cavity error: the map's nominal frequency equals the clock, its effective
    frequency does not."""
    def line():
        cav = _cavity()
        cav.frequency_offset = 0.5
        return [Drift("D0", 50.0), cav, Drift("D1", 50.0)]
    native, _ = _run(line, _ref(f=F_NEW), surrogate=False)
    beam, surr = _run(line, _ref(f=F_NEW))
    assert native.ref.frequency == F_NEW + 0.5
    _assert_same_as_native(beam, native)
    assert surr.nn_calls == 0


@pytest.mark.parametrize("ref_mhz, cards", [(F_NEW, False), (F_OLD, True)],
                         ids=["same_clock", "card_before_the_cavity"])
def test_no_jump_left_keeps_the_fast_path(fast_path, ref_mhz, cards):
    """After a card there is nothing left for the cavity to do."""
    def line():
        return ([Drift("D0", 50.0)] + ([Freq("F", F_NEW)] if cards else [])
                + [_cavity(), Drift("D1", 50.0)])
    native, _ = _run(line, _ref(f=ref_mhz), surrogate=False)
    beam, surr = _run(line, _ref(f=ref_mhz))
    assert surr.nn_calls >= 1
    assert beam.ref.frequency == native.ref.frequency == F_NEW
    assert beam.ref.phi_s == pytest.approx(native.ref.phi_s, rel=1e-3)
    assert np.std(beam.particles[:, 4]) == pytest.approx(np.std(native.particles[:, 4]), rel=0.01)


def test_a_magnet_carries_its_frequency_as_metadata_only(fast_path):
    """No electric channel: no jump, so the fast path stays and the clock stays."""
    def line():
        return [Drift("D0", 50.0), _magnet(), Drift("D1", 50.0)]
    native, _ = _run(line, _ref(), surrogate=False)
    beam, surr = _run(line, _ref())
    assert surr.nn_calls >= 1
    assert beam.ref.frequency == native.ref.frequency == F_OLD


@pytest.mark.parametrize("p_flag", [0, 1], ids=["absolute", "calibrated"])
def test_a_cavity_entered_out_of_scope_stays_native(fast_path, p_flag):
    """The energy comes into scope part way: the set-up of the fast path is for the
    entrance, and would restart the map's cursor and entrance phase there."""
    def line():
        return [_cavity(p_flag=p_flag)]
    native, _ = _run(line, _ref(f=F_NEW), surrogate=False)
    assert native.ref.w_kin > 5.0
    beam, surr = _run(line, _ref(f=F_NEW), w_lo=(5.0 + native.ref.w_kin) / 2)
    _assert_same_as_native(beam, native)
    assert surr.nn_calls == 0


def test_a_refused_traversal_does_not_stick_to_the_next_one(fast_path):
    lat = _lattice(_cavity())
    el = lat.elements[0]
    surr = _pinned(el, copy.deepcopy(el).fitted_matrix(_ref(f=F_NEW)))
    registry.register(surr)
    Tracker(lat, _beam(_ref())).run()
    assert surr.nn_calls == 0
    Tracker(lat, _beam(_ref(f=F_NEW))).run()
    assert surr.nn_calls >= 1


# ------------------------------------------------------------------ training
def _direct(el, w, scale, f):
    e = copy.deepcopy(el)
    e.scale = scale
    e.reset_run_state()
    return np.asarray(e.fitted_matrix(_ref(f=f, w=w)), dtype=np.float64).reshape(36)


@pytest.mark.parametrize("make, target_mhz", [(_cavity, F_NEW), (_cavity_1d, F_NEW),
                                               (_magnet, F_OLD)],
                         ids=["cavity", "cavity_1d", "magnet"])
def test_training_targets_are_the_map_at_its_own_clock(make, target_mhz):
    """The template sits at 162.5 MHz.  A 325 MHz cavity's target holds no factor 2.
    A magnet's frequency is metadata, so its target stays on the template's clock
    (its Δφ row in those degrees, the limit the manual states)."""
    el = make()
    X, Y, _info = generate_training_data(element=el, ref_template=_ref(), n_samples=4,
                                         ref_w_kin_range=(4.0, 6.0),
                                         param_ranges={"scale": (0.9, 1.1)}, seed=3)
    for x, y in zip(X, Y, strict=True):
        np.testing.assert_array_equal(y, _direct(el, x[0], x[3], target_mhz))
    assert np.allclose(Y.reshape(-1, 6, 6)[:, 4, 4], 1.0, atol=0.2)


def test_the_worker_path_builds_the_same_target(monkeypatch):
    el = _cavity()
    for key, value in (("element", copy.deepcopy(el)), ("ref_template", _ref()),
                       ("param_names", ["scale"]), ("counters", None), ("wid", None)):
        monkeypatch.setitem(training._WORKER_STATE, key, value)
    _i, _b, _g, y = training._worker_compute_one((0, 5.0, np.array([1.0])))
    np.testing.assert_array_equal(y, _direct(el, 5.0, 1.0, F_NEW))


@pytest.mark.parametrize("make", [_cavity, _cavity_1d], ids=["3d", "1d"])
@pytest.mark.parametrize("p_flag", [0, 1], ids=["absolute", "calibrated"])
def test_a_trained_target_through_the_seam_is_the_native_envelope(make, p_flag):
    """The whole chain at a jump: the training target, in a surrogate, through the
    envelope's I = 0 seam, against a native run.  It would be off by the factor if
    the target held it."""
    ref = _ref()
    init = dict(alpha_x=0.0, beta_x=2.0, emit_x=1.0, alpha_y=0.0, beta_y=2.0, emit_y=1.0,
                alpha_z=0.0, beta_z=10.0, emit_z=0.3)

    def lat():
        return _lattice(make(p_flag=p_flag), Drift("D1", 50.0))
    _x, Y, _i = generate_training_data(element=make(p_flag=p_flag), ref_template=ref,
                                       n_samples=1, ref_w_kin_range=(5.0, 5.0),
                                       param_ranges={}, seed=1)
    native = EnvelopeSolver(lat(), ref.copy(), dict(init), current=0.0).run()
    line = lat()
    surr = _pinned(line.elements[0], Y[0].reshape(6, 6))
    registry.register(surr)
    got = EnvelopeSolver(line, ref.copy(), dict(init), current=0.0).run()
    assert surr.nn_calls == 1
    np.testing.assert_allclose(np.asarray(got.sigma_matrix[-1]), np.asarray(native.sigma_matrix[-1]),
                               rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("make", [_cavity, _cavity_1d, _magnet], ids=["3d", "1d", "magnet"])
def test_a_surrogate_asked_at_another_clock_gives_the_native_matrix(make):
    """A surrogate is a drop-in for its element, in a lattice too, where no seam acts."""
    ref = _ref()
    el = make()
    own = _ref(f=F_NEW) if el._has_electric_channel() else _ref()
    surr = _pinned(el, copy.deepcopy(el).fitted_matrix(own))
    np.testing.assert_allclose(surr.fitted_matrix(ref), copy.deepcopy(el).fitted_matrix(ref),
                               rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(compute_transfer_matrix(_lattice(surr), ref),
                               copy.deepcopy(el).fitted_matrix(ref), rtol=1e-9, atol=1e-10)
    # at its own clock nothing is added
    np.testing.assert_array_equal(surr.fitted_matrix(own),
                                  np.asarray(surr.metadata.output_norm["mean"]).reshape(6, 6))


def test_a_surrogate_of_ncells_takes_the_factor_as_the_cavity_does():
    """Python API only: the Surrogates tab does not offer NCELLS.  The target is on the
    cavity's clock, and asked at another one the surrogate applies the factor, so the
    native matrix and the surrogate's agree."""
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.ncells import NCells
    from linac_gen.surrogates.training import _at_element_frequency

    el = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=1, frequency_mhz=325.0)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=162.5)
    r = ref.copy()
    _at_element_frequency(el, r)
    assert r.frequency == 325.0
    target = el.fitted_matrix(r)
    assert np.linalg.det(target[4:6, 4:6]) == pytest.approx(1.0, rel=1e-9)
    surr = _pinned(el, target)
    np.testing.assert_allclose(surr.fitted_matrix(ref), el.fitted_matrix(ref), rtol=1e-12, atol=1e-12)
