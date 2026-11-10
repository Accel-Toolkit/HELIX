"""What a surrogate was trained at, and what it does when asked at something else.

Weights hold a map at one clock, one species and one reference phase, and take none of
them as an input.  New files record the clock and the phase.  A map asked at another
clock is converted where it can be, and goes to the native element where it cannot.
Files from before hold no record, and are checked by what must be true of any map: on
its own clock a longitudinal determinant of 1, and for a magnet a drift's Δφ–ΔW term.
"""
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Freq
from linac_gen.surrogates import registry, training
from linac_gen.surrogates.base import (
    PHASE_TOLERANCE_DEG,
    MlpHead,
    OutOfScopeError,
    Scope,
    SurrogateFieldMap,
    SurrogateMetadata,
    species_name_of,
)
from linac_gen.surrogates.training import (
    find_cached_surrogate,
    generate_training_data,
    load_surrogate,
    save_surrogate,
    train_surrogate_for_element,
)
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import reference_at
from linac_gen.tracking.torch_matrices import RefKinematics
from linac_gen.tracking.torch_tracking import element_matrix_torch
from linac_gen.tracking.tracker import Tracker
from tests.surrogates.test_frequency_jump import (
    F_NEW,
    F_OLD,
    _beam,
    _cavity,
    _lattice,
    _magnet,
    _ref,
)

INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2,
            alpha_z=0.0, beta_z=1.0, emit_z=0.2)


def _meta(el, M, *, params=(), lo=(), hi=(), means=(), **kw):
    """Metadata whose map is pinned to ``M`` (output std = 0), in scope for 2 to 10 MeV."""
    names = ["w_kin", "beta", "gamma", *params]
    return SurrogateMetadata(
        element_key=el.name, element_class=type(el).__name__,
        architecture={"input_dim": len(names), "output_dim": 36, "hidden_dims": [4],
                      "activation": "silu", "param_names": list(params)},
        scope=Scope(input_names=names, input_lo=np.array([2.0, 0.0, 1.0, *lo]),
                    input_hi=np.array([10.0, 1.0, 100.0, *hi])),
        input_norm={"mean": [0.0, 0.0, 0.0, *means], "std": [1.0] * len(names)},
        output_norm={"mean": [float(v) for v in np.asarray(M).ravel()], "std": [0.0] * 36},
        training_seed=0, n_samples=0, epochs=0, val_mape=0.0,
        helix_commit_sha="", lattice_hash="lh", created_iso="", **kw)


def _mlp(meta):
    return MlpHead(input_dim=meta.architecture["input_dim"], output_dim=36, hidden_dims=(4,))


def _surrogate(el, M, **kw):
    meta = _meta(el, M, **kw)
    return SurrogateFieldMap(el, _mlp(meta), meta)


def _save(path, el, M, *, drop=("train_frequency_mhz", "train_phi_s_deg"), **kw):
    """A file on disk; by default one from before these fields existed.  It records the
    species ``_at`` gives unless told otherwise: a file that records none counts as H⁻."""
    kw.setdefault("species_name", PROTON.name)
    kw.setdefault("species_mass_mev", {PROTON.name: PROTON.mass, H_MINUS.name: H_MINUS.mass}.get(
        kw["species_name"], 0.0))
    meta = _meta(el, M, **kw)
    save_surrogate(_mlp(meta), meta, path)
    data = json.loads((path / "metadata.json").read_text())
    for key in drop:
        del data[key]
    (path / "metadata.json").write_text(json.dumps(data))
    return path


def _loaded(path, el):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp, meta = load_surrogate(path)
    # pinned maps, not their element's: what these tests check comes before the accuracy
    # check, which has its own tests (test_accuracy_check.py)
    meta.accuracy_due = False
    return SurrogateFieldMap(el, mlp, meta)


def _kin(ref):
    return RefKinematics.from_reference(ref)


def _z_block(det):
    M = np.eye(6)
    M[4, 4] = det
    return M


def _at(f=F_OLD, w=5.0, phi_s=0.0, species=PROTON):
    r = ReferenceParticle(species=species, w_kin=w, frequency=f)
    r.phi_s = phi_s
    return r


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------
def test_what_a_file_records_comes_back(tmp_path):
    path = _save(tmp_path, _cavity(), np.eye(6), drop=(),
                 train_frequency_mhz=F_NEW, train_phi_s_deg=12.5)
    _mlp_, meta = load_surrogate(path)
    assert meta.train_frequency_mhz == F_NEW and meta.train_phi_s_deg == 12.5
    assert meta.legacy_weights is False and meta.stale_factor is None


def test_a_file_from_before_still_loads_and_is_marked(tmp_path):
    _mlp_, meta = load_surrogate(_save(tmp_path, _cavity(), np.eye(6)))
    assert meta.train_frequency_mhz == 0.0 and meta.train_phi_s_deg == 0.0
    assert meta.legacy_weights is True and meta.stale_factor is None


def test_a_file_with_the_field_at_zero_counts_as_from_before(tmp_path):
    """What a hand-built metadata saved through the Python API writes."""
    _mlp_, meta = load_surrogate(_save(tmp_path, _cavity(), np.eye(6), drop=()))
    assert meta.legacy_weights is True


def test_what_is_worked_out_on_load_is_not_saved():
    d = _meta(_cavity(), np.eye(6), train_frequency_mhz=F_NEW, legacy_weights=True,
              stale_factor=2.0).to_json()
    assert d["train_frequency_mhz"] == F_NEW
    assert "legacy_weights" not in d and "stale_factor" not in d


def test_training_records_the_targets_clock_and_phase(tmp_path):
    """A cavity's own clock whatever the template's; a magnet's is the template's.  The phase is the
    one the targets are taken at, on their clock: 33° on the template's 162.5 MHz is 66° on the
    cavity's 325 MHz, where the switch puts it as a FREQ card would."""
    ref = _ref(F_OLD)
    ref.phi_s = 33.0
    kw = dict(n_samples=8, ref_w_kin_range=(4.0, 6.0), param_ranges={"phase": (-40.0, -20.0)},
              hidden_dims=(4,), epochs=1, seed=0)
    _m, cavity = train_surrogate_for_element(_cavity(F_NEW), ref, out_dir=tmp_path / "c", **kw)
    assert cavity.train_frequency_mhz == F_NEW and cavity.train_phi_s_deg == 33.0 * F_NEW / F_OLD
    _m, magnet = train_surrogate_for_element(_magnet(F_NEW), ref, out_dir=tmp_path / "m", **kw)
    assert magnet.train_frequency_mhz == F_OLD
    assert load_surrogate(tmp_path / "c")[1].legacy_weights is False


# ---------------------------------------------------------------------------
# item 1: a file from before, whose targets hold the jump
# ---------------------------------------------------------------------------
def _stale_target(monkeypatch):
    """What training produced before: the template's clock, so the jump in the target."""
    monkeypatch.setattr(training, "_at_element_frequency", lambda element, ref: None)
    _X, Y, _info = generate_training_data(_cavity(F_NEW), _ref(F_OLD), n_samples=4,
                                          ref_w_kin_range=(4.5, 5.5), param_ranges={})
    monkeypatch.undo()
    return Y.mean(axis=0).reshape(6, 6)


def test_a_file_whose_targets_hold_the_jump_is_found(tmp_path, monkeypatch):
    M = _stale_target(monkeypatch)
    path = _save(tmp_path, _cavity(F_NEW), M)
    with pytest.warns(UserWarning, match="stale"):
        _mlp_, meta = load_surrogate(path)
    assert meta.stale_factor == pytest.approx(F_NEW / F_OLD, rel=0.02)


def test_a_stale_surrogate_gives_way_to_the_native_cavity(tmp_path, monkeypatch):
    """Registered as the GUI would: the run equals the run with no surrogate."""
    M = _stale_target(monkeypatch)
    el = _cavity(F_NEW, p_flag=1)       # so that nothing but the staleness refuses

    def line():
        return _lattice(Drift("D", 50.0), el, Drift("E", 50.0))
    base = EnvelopeSolver(line(), _ref(F_OLD), dict(INIT), current=0.0).run()
    surr = _loaded(_save(tmp_path, el, M), el)
    registry.register(surr)
    got = EnvelopeSolver(line(), _ref(F_OLD), dict(INIT), current=0.0).run()
    registry.clear()
    assert surr.nn_calls == 0
    assert np.asarray(got.sigma_matrix).tobytes() == np.asarray(base.sigma_matrix).tobytes()


@pytest.mark.parametrize("det, stale", [(1.0, False), (1.04, False), (1.06, True), (1.24, True),
                                        (2.0, True), (0.96, False), (0.94, True), (0.5, True), (4.0, True)])
def test_the_threshold_on_each_side(tmp_path, det, stale):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _mlp_, meta = load_surrogate(_save(tmp_path, _cavity(), _z_block(det)))
    assert (meta.stale_factor is not None) == stale


def test_a_new_file_is_not_put_to_the_test(tmp_path):
    _mlp_, meta = load_surrogate(_save(tmp_path, _cavity(), _z_block(2.0), drop=(),
                                       train_frequency_mhz=F_NEW))
    assert meta.stale_factor is None


def test_the_shipped_weights_are_not_stale():
    root = Path(training.__file__).parent / "weights"
    files = sorted(root.glob("*/*/metadata.json"))
    if not files:
        pytest.skip("no shipped weights")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metas = [load_surrogate(f.parent)[1] for f in files]
    assert [m.element_key for m in metas if m.stale_factor is not None] == []
    # a file that records no clock was put to the stale test as it loaded; one trained since
    # records its clock, its training recipe and its element's structure
    assert all(m.legacy_weights or (m.train_recipe is not None and m.train_structure is not None)
               for m in metas)


def test_the_torch_arm_refuses_a_stale_surrogate(tmp_path):
    el = _cavity(F_NEW)
    surr = _loaded(_save(tmp_path, el, _z_block(2.0)), el)
    with pytest.raises(ValueError, match="stale|determinant"):
        element_matrix_torch(surr, _kin(_at(F_NEW)))


def test_metadata_built_in_memory_is_not_judged():
    """A pinned map in a test, or anyone's by hand: it says nothing about its training."""
    el = _cavity(F_NEW)
    surr = _surrogate(el, _z_block(2.0))
    surr.fitted_matrix(_at(F_NEW, phi_s=77.0))
    assert surr.nn_calls == 1


def test_a_loaded_file_goes_through_the_cache_route_too(tmp_path):
    el = _cavity(F_OLD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cached = find_cached_surrogate(_save(tmp_path, el, _z_block(2.0)))
    assert cached is not None and cached[1].stale_factor == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# item 2: a magnet's clock
# ---------------------------------------------------------------------------
def _native(el, ref):
    return copy.deepcopy(el).fitted_matrix(ref.copy())


@pytest.mark.parametrize("clock", [F_OLD, F_NEW, 650.0])
def test_a_magnet_is_converted_to_the_clock_it_is_asked_at(clock):
    el = _magnet()
    M0 = _native(el, _at(F_OLD))
    surr = _surrogate(el, M0, train_frequency_mhz=F_OLD)
    got = surr.fitted_matrix(_at(clock))
    np.testing.assert_allclose(got, _native(el, _at(clock)), rtol=1e-9, atol=1e-9)
    assert got[4, 5] == pytest.approx(M0[4, 5] * clock / F_OLD, rel=1e-13)


def test_at_its_own_clock_a_magnet_is_untouched_to_the_last_bit():
    el = _magnet()
    M0 = _native(el, _at(F_OLD))
    surr = _surrogate(el, M0, train_frequency_mhz=F_OLD)
    want = _surrogate(el, M0).fitted_matrix(_at(F_OLD))
    assert surr.fitted_matrix(_at(F_OLD)).tobytes() == want.tobytes()


def test_a_cavity_takes_the_jump_not_the_magnets_conversion():
    el = _cavity(F_NEW)
    M = _native(el, _at(F_NEW))
    surr = _surrogate(el, M, train_frequency_mhz=F_NEW)
    D = np.eye(6)
    D[4, 4] = F_NEW / F_OLD
    np.testing.assert_allclose(surr.fitted_matrix(_at(F_OLD)), M @ D, rtol=1e-6, atol=1e-6)


def test_a_magnet_built_in_memory_is_left_alone():
    el = _magnet()
    M0 = _native(el, _at(F_OLD))
    assert _surrogate(el, M0).fitted_matrix(_at(F_NEW))[4, 5] == pytest.approx(M0[4, 5], rel=1e-6)


def test_the_drift_formula_is_the_drifts():
    el = _magnet()
    surr = _surrogate(el, np.eye(6), legacy_weights=True)
    r = _at(F_OLD)
    drift = Drift("d", el.length).transfer_matrix(r)[4, 5]
    assert surr._m45_refusal(drift, r.beta, r.gamma, r.species.mass, r.wavelength) is None
    assert surr._m45_refusal(2.0 * drift, r.beta, r.gamma, r.species.mass, r.wavelength) is not None


@pytest.mark.parametrize("scale, ok", [(1.0, True), (1.2, True), (0.85, True),
                                       (1.3, False), (0.75, False), (2.0, False), (0.5, False)])
def test_a_magnet_from_before_goes_native_at_another_clock(tmp_path, scale, ok):
    """The term shows the clock it was trained on, with no clock assumed."""
    el = _magnet()
    M = _native(el, _at(F_OLD))
    M[4, 5] *= scale
    surr = _loaded(_save(tmp_path, el, M), el)
    if ok:
        surr.fitted_matrix(_at(F_OLD))
    else:
        with pytest.raises(OutOfScopeError, match="another"):
            surr.fitted_matrix(_at(F_OLD))
        with pytest.raises(ValueError, match="another"):
            element_matrix_torch(surr, _kin(_at(F_OLD)))


def test_a_magnet_from_before_trained_at_162_goes_native_at_325(tmp_path):
    el = _magnet()
    surr = _loaded(_save(tmp_path, el, _native(el, _at(F_OLD))), el)
    surr.fitted_matrix(_at(F_OLD))
    with pytest.raises(OutOfScopeError):
        surr.fitted_matrix(_at(F_NEW))


def test_the_torch_arm_converts_a_magnet_too():
    el = _magnet()
    M0 = _native(el, _at(F_OLD))
    surr = _surrogate(el, M0, train_frequency_mhz=F_OLD)
    got = element_matrix_torch(surr, _kin(_at(F_NEW))).detach().numpy()
    np.testing.assert_allclose(got, surr.fitted_matrix(_at(F_NEW)), rtol=1e-12, atol=1e-12)
    # at its own clock, λ → f and back must not move a bit
    assert surr._magnet_clock_ratio(2.99792458e8 / (_at(F_OLD).wavelength * 1e3)) == 1.0


def test_a_cavity_that_runs_on_the_clock_answers_only_on_its_own(tmp_path):
    """No frequency of its own: its field turns at the clock, so its map holds the one it
    was trained on."""
    el = _cavity(0.0, p_flag=1)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD)
    surr.fitted_matrix(_at(F_OLD))
    element_matrix_torch(surr, _kin(_at(F_OLD)))
    with pytest.raises(OutOfScopeError, match="runs on the clock"):
        surr.fitted_matrix(_at(F_NEW))
    with pytest.raises(ValueError, match="runs on the clock"):
        element_matrix_torch(surr, _kin(_at(F_NEW)))
    # a file from before records no clock, and is not judged on it
    _loaded(_save(tmp_path, el, np.eye(6)), el).fitted_matrix(_at(F_NEW))
    # nor is a cavity with a frequency of its own, which makes its own jump
    _surrogate(_cavity(F_OLD, p_flag=1), np.eye(6), train_frequency_mhz=F_OLD).fitted_matrix(_at(F_NEW))


# ---------------------------------------------------------------------------
# item 2: the species
# ---------------------------------------------------------------------------
def test_a_map_trained_for_one_species_does_not_answer_for_another():
    el = _magnet()
    surr = _surrogate(el, np.eye(6), species_name=H_MINUS.name)
    surr.fitted_matrix(_at(species=H_MINUS))
    with pytest.raises(OutOfScopeError, match="trained for"):
        surr.fitted_matrix(_at(species=PROTON))
    with pytest.raises(ValueError, match="trained for"):
        element_matrix_torch(surr, _kin(_at(species=PROTON)))


def test_a_file_with_no_species_is_not_judged_on_it():
    surr = _surrogate(_magnet(), np.eye(6))
    surr.fitted_matrix(_at(species=PROTON))
    surr.fitted_matrix(_at(species=H_MINUS))


def _no_species(path, el):
    """A file that records no species, as the Surrogates tab and the CLI wrote them until
    species were recorded: they trained H⁻ only."""
    return _loaded(_save(path, el, np.eye(6), drop=(), train_frequency_mhz=F_OLD,
                         species_name="", species_mass_mev=0.0), el)


def test_a_file_that_records_no_species_counts_as_h_minus(tmp_path):
    surr = _no_species(tmp_path, _magnet())
    assert surr.metadata.species_name == "" and surr.metadata.trained_species == H_MINUS.name
    surr.fitted_matrix(_at(species=H_MINUS))
    element_matrix_torch(surr, _kin(_at(species=H_MINUS)))
    with pytest.raises(OutOfScopeError, match="records no species"):
        surr.fitted_matrix(_at(species=PROTON))
    with pytest.raises(ValueError, match="records no species"):
        element_matrix_torch(surr, _kin(_at(species=PROTON)))
    # the assumption is made at each load and never written back
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["species_name"] == "" and "species_assumed" not in saved


def test_a_file_that_records_no_species_says_what_it_counts_as(tmp_path):
    el = _magnet()
    _save(tmp_path, el, np.eye(6), drop=(), train_frequency_mhz=F_OLD, species_name="",
          species_mass_mev=0.0)
    with pytest.warns(UserWarning, match="counts as trained for H⁻"):
        load_surrogate(tmp_path)


def test_a_file_that_records_a_species_is_not_assumed_anything(tmp_path):
    surr = _loaded(_save(tmp_path, _magnet(), np.eye(6), drop=(), train_frequency_mhz=F_OLD), _magnet())
    assert surr.metadata.species_assumed == "" and surr.metadata.trained_species == PROTON.name
    surr.fitted_matrix(_at(species=PROTON))


def test_the_row_of_a_file_with_no_species_follows_the_beam(qapp, tmp_path):
    meta = _no_species(tmp_path, _magnet()).metadata
    tab = _tab(_beam_config("H-"))
    text, use = _row(tab, meta)
    assert use.isEnabled()
    tab.state.set_beam_config(_beam_config("proton"))
    assert tab._table.item(0, 1).text() == "trained for H-"
    assert not tab._table.cellWidget(0, 3).isEnabled()


def test_a_species_is_known_from_its_mass_and_charge():
    assert species_name_of(PROTON.mass, PROTON.charge) == PROTON.name
    assert species_name_of(H_MINUS.mass, H_MINUS.charge) == H_MINUS.name
    assert species_name_of(PROTON.mass, -1) is None
    assert species_name_of(100.0, 1) is None


# ---------------------------------------------------------------------------
# item 7: the running phase
# ---------------------------------------------------------------------------
PHASE = dict(params=("phase",), lo=(-60.0,), hi=(0.0,), means=(-30.0,))


def _phase_input(surr, r):
    return float(surr._make_input(r)[3])


@pytest.mark.parametrize("phi_s", [40.0, 137.0, 134823.893])
def test_a_map_at_p_flag_0_depends_on_phase_plus_phi_s_alone(phi_s):
    """What the fix rests on, against the native map."""
    a = _cavity(F_OLD, p_flag=0)
    b = _cavity(F_OLD, p_flag=0)
    b.phase = a.phase + phi_s
    np.testing.assert_allclose(a.fitted_matrix(_at(F_OLD, phi_s=phi_s)), b.fitted_matrix(_at(F_OLD)),
                               rtol=1e-9, atol=1e-9)
    far = _cavity(F_OLD, p_flag=0).fitted_matrix(_at(F_OLD))
    assert np.abs(a.fitted_matrix(_at(F_OLD, phi_s=phi_s)) - far).max() > 1e-5


def test_at_p_flag_1_a_map_does_not_depend_on_phi_s():
    np.testing.assert_allclose(_cavity(F_OLD, p_flag=1).fitted_matrix(_at(F_OLD, phi_s=137.0)),
                               _cavity(F_OLD, p_flag=1).fitted_matrix(_at(F_OLD)),
                               rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("phi_s, shift", [
    (100.0, 0.0), (110.0, 10.0), (90.0, -10.0),
    (100.0 + 190.0, -170.0), (100.0 - 190.0, 170.0), (100.0 + 360.0, 0.0),
    (100.0 + 3600.0 + 5.0, 5.0)])
def test_the_phase_input_follows_the_shift_from_the_training(phi_s, shift):
    el = _cavity(F_OLD, p_flag=0)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0, **PHASE)
    assert _phase_input(surr, _at(F_OLD, phi_s=phi_s)) == pytest.approx(el.phase + shift, abs=1e-9)


def test_a_field_map_whose_p_flag_is_not_1_follows_it_too():
    """As its native map does: every p_flag but 1 is relative to the running phase."""
    a, b = _cavity(F_OLD, p_flag=2), _cavity(F_OLD, p_flag=2)
    b.phase = a.phase + 40.0
    np.testing.assert_allclose(a.fitted_matrix(_at(F_OLD, phi_s=40.0)), b.fitted_matrix(_at(F_OLD)),
                               rtol=1e-9, atol=1e-9)
    surr = _surrogate(a, np.eye(6), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0, **PHASE)
    assert _phase_input(surr, _at(F_OLD, phi_s=110.0)) == pytest.approx(a.phase + 10.0, abs=1e-9)


def test_an_ncells_keeps_its_own_convention():
    """Its p_flag 0 is relative to its entrance, not to the running phase: not shifted."""
    from linac_gen.elements.ncells import NCells
    el = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=0, frequency_mhz=F_OLD)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0)
    surr.fitted_matrix(_at(F_OLD, phi_s=137.0))
    assert surr.nn_calls == 1


def test_the_shifted_phase_is_turned_back_into_the_scope():
    """The map is periodic in the phase: a sweep over the whole circle answers 250° as -110°."""
    el = _cavity(F_OLD, p_flag=0)
    el.phase = 150.0
    whole = dict(params=("phase",), lo=(-180.0,), hi=(180.0,), means=(0.0,))
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD, train_phi_s_deg=0.0, **whole)
    assert _phase_input(surr, _at(F_OLD, phi_s=100.0)) == pytest.approx(-110.0, abs=1e-9)
    surr.fitted_matrix(_at(F_OLD, phi_s=100.0))
    assert surr.nn_calls == 1


def test_at_the_training_phase_the_input_is_untouched_to_the_last_bit():
    el = _cavity(F_OLD, p_flag=0)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0, **PHASE)
    assert _phase_input(surr, _at(F_OLD, phi_s=100.0)) == el.phase


@pytest.mark.parametrize("make", [lambda: _cavity(F_OLD, p_flag=1), lambda: _magnet(F_OLD)],
                         ids=["p_flag_1", "magnet"])
def test_nothing_else_has_its_input_moved(make):
    el = make()
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD, **PHASE)
    assert _phase_input(surr, _at(F_OLD, phi_s=137.0)) == el.phase


@pytest.mark.parametrize("delta, ok", [(0.0, True), (0.005, True), (360.005, True), (0.02, False), (90.0, False)])
def test_with_its_phase_not_swept_a_cavity_answers_only_at_its_training_phase(delta, ok):
    el = _cavity(F_OLD, p_flag=0)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0)
    assert PHASE_TOLERANCE_DEG == 0.01
    if ok:
        surr.fitted_matrix(_at(F_OLD, phi_s=100.0 + delta))
    else:
        with pytest.raises(OutOfScopeError, match="running phase"):
            surr.fitted_matrix(_at(F_OLD, phi_s=100.0 + delta))


def test_a_cavity_from_before_is_taken_as_trained_at_phase_0(tmp_path):
    """Which is what every template used.  Out of its phase scope it goes native."""
    el = _cavity(F_OLD, p_flag=0)
    surr = _loaded(_save(tmp_path, el, np.eye(6), **PHASE), el)
    surr.fitted_matrix(_at(F_OLD, phi_s=0.0))
    with pytest.raises(OutOfScopeError):
        surr.fitted_matrix(_at(F_OLD, phi_s=137.0))
    calibrated = _cavity(F_OLD, p_flag=1)
    _loaded(_save(tmp_path / "x", calibrated, np.eye(6), **PHASE), calibrated).fitted_matrix(
        _at(F_OLD, phi_s=137.0))


def test_a_surrogate_trained_at_its_phase_follows_it_as_the_native_map_does(tmp_path):
    """A real training.  It tracks the native map when the running phase moves, where
    a surrogate that ignores the move would stay at the map for the old one."""
    el = _cavity(F_OLD, p_flag=0)
    ref = _ref(F_OLD)
    ref.phi_s = 137.0
    mlp, meta = train_surrogate_for_element(
        el, ref, n_samples=400, ref_w_kin_range=(4.9, 5.1), param_ranges={"phase": (-60.0, 0.0)},
        hidden_dims=(32, 32), epochs=400, lr=3e-3, batch_size=64, seed=0, out_dir=tmp_path / "s")
    surr = SurrogateFieldMap(el, mlp, meta)
    at, moved = _at(F_OLD, phi_s=137.0), _at(F_OLD, phi_s=137.0 + 20.0)
    native_at, native_moved = _native(el, at), _native(el, moved)
    got = surr.fitted_matrix(moved)
    assert np.abs(got - native_moved).max() < 0.25 * np.abs(native_at - native_moved).max()


def _passes(el, ref, surr):
    """Run a line through the fast path; how many pushes went to the native map."""
    calls = []
    real = el.track_rk4
    el.track_rk4 = lambda beam, ds: (calls.append(ds), real(beam, ds))[1]
    lat = _lattice(el)
    registry.register(surr)
    Tracker(lat, _beam(ref)).run()
    registry.clear()
    return calls


@pytest.fixture
def fast_path():
    mp, fast = registry.is_mp_enabled(), registry.is_fast_path_enabled()
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    yield
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(fast)


def test_the_fast_path_stays_on_through_the_whole_cavity(fast_path):
    """Its scope check runs at every substep.  On the running phase, the input walks
    out of a ±30° sweep part way through and the rest goes native."""
    el = _cavity(F_OLD, p_flag=0)
    r = _at(F_OLD, phi_s=100.0)
    surr = _surrogate(el, _native(el, r), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0, **PHASE)
    assert _passes(el, r, surr) == []
    assert surr.nn_calls == 1


def test_a_new_traversal_reads_the_phase_again(fast_path):
    el = _cavity(F_OLD, p_flag=0)
    r = _at(F_OLD, phi_s=100.0)
    surr = _surrogate(el, _native(el, r), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0, **PHASE)
    assert _passes(el, r, surr) == []
    surr.reset_run_state()
    # 120° off the training's: out of the ±30° sweep, so all of it goes native
    assert _passes(el, _at(F_OLD, phi_s=220.0), surr)


def test_a_traversal_out_of_scope_does_not_hold_back_the_next(fast_path):
    """The entrance phase is read again on each traversal: one 120° off the training's
    goes native, and the next, at the training's phase, takes the fast path all through."""
    el = _cavity(F_OLD, p_flag=0)
    r = _at(F_OLD, phi_s=100.0)
    surr = _surrogate(el, _native(el, r), train_frequency_mhz=F_OLD, train_phi_s_deg=100.0, **PHASE)
    assert _passes(el, _at(F_OLD, phi_s=220.0), surr)
    surr.reset_run_state()
    assert _passes(el, r, surr) == []


def test_the_torch_arm_refuses_a_cavity_that_follows_the_running_phase():
    el = _cavity(F_OLD, p_flag=0)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_OLD, **PHASE)
    with pytest.raises(ValueError, match="running phase"):
        element_matrix_torch(surr, _kin(_at(F_OLD)))
    calibrated = _cavity(F_OLD, p_flag=1)
    element_matrix_torch(_surrogate(calibrated, np.eye(6), train_frequency_mhz=F_OLD, **PHASE),
                         _kin(_at(F_OLD)))


# ---------------------------------------------------------------------------
# the tab
# ---------------------------------------------------------------------------
@pytest.fixture
def qapp():
    PyQt6 = pytest.importorskip("PyQt6")
    assert PyQt6
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _tab(beam=None):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
    state = AppState()
    tab = SurrogatesTab(state)
    if beam is not None:
        state.set_beam_config(beam)
    return tab


def _row(tab, meta, name="MAP"):
    tab._trained[name] = (SurrogateFieldMap(_cavity(), _mlp(meta), meta), Path("."), meta)
    tab._refresh_table()
    return tab._table.item(0, 1).text(), tab._table.cellWidget(0, 3)


def test_a_stale_row_says_so_and_cannot_be_used(qapp):
    tab = _tab()
    meta = _meta(_cavity(), np.eye(6))
    meta.stale_factor = 2.0
    text, use = _row(tab, meta)
    assert text == "stale, retrain"
    assert not use.isEnabled()
    tab._set_all_use(True)
    assert not use.isChecked()
    assert registry.get(meta.lattice_hash, "MAP") is None


def test_a_good_row_is_as_before(qapp):
    tab = _tab()
    text, use = _row(tab, _meta(_cavity(), np.eye(6)))
    assert text == "0.00e+00"
    assert use.isEnabled()


def test_a_row_that_cannot_be_used_is_not_left_in_use(qapp):
    """A registration made before, under the same name, goes when the row cannot be used."""
    tab = _tab()
    good = _meta(_cavity(), np.eye(6))
    registry.register(SurrogateFieldMap(_cavity(), _mlp(good), good))
    try:
        assert registry.get(good.lattice_hash, "MAP") is not None
        stale = _meta(_cavity(), np.eye(6))
        stale.stale_factor = 2.0
        _text, use = _row(tab, stale)
        assert registry.get(stale.lattice_hash, "MAP") is None
        assert not use.isChecked() and not use.isEnabled()
    finally:
        registry.clear()


def test_compare_skips_a_row_that_cannot_be_used(qapp, monkeypatch):
    from linac_gen_gui.interphase.tabs import surrogates_tab as mod

    from linac_gen.core.config import BeamConfig
    shown = []
    monkeypatch.setattr(mod.QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[1:3])))

    def worker(*args, **kw):
        raise AssertionError("a comparison started")
    monkeypatch.setattr(mod, "_MpCompareWorker", worker)
    monkeypatch.setattr(mod, "_EnvCompareWorker", worker)
    tab = _tab()
    tab.state.set_lattice(_lattice(_cavity()))
    tab.state.set_beam_config(BeamConfig(species="H-", energy=5.0, frequency=F_OLD, n_particles=16))
    meta = _meta(_cavity(), np.eye(6))
    meta.stale_factor = 2.0
    _row(tab, meta)
    tab._on_compare_clicked("MAP")
    assert "cannot be compared: stale, retrain" in tab._status.text()
    tab._on_mp_compare_clicked()
    assert "nothing to compare" in tab._status.text()
    assert shown == [] and not tab.state.running


def test_the_status_line_counts_the_stale_ones(qapp, tmp_path, monkeypatch):
    from linac_gen.core.lattice import Lattice
    tab = _tab()
    el = _cavity(F_NEW)
    root = tmp_path / "weights"
    _save(root / el.name, el, _z_block(2.0))
    monkeypatch.setattr(tab, "_weights_root", lambda lattice_hash=None: root)
    lat = Lattice()
    lat.add(el)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab._discover_cached_for_lattice(lat)
    assert "auto-loaded 1 cached" in tab._status.text()
    assert "1 stale, retrain" in tab._status.text()


def _beam_config(species="proton", energy=5.0, frequency=F_OLD, **kw):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species=species, energy=energy, frequency=frequency, n_particles=16, **kw)


def test_a_row_for_another_species_follows_the_beam(qapp):
    tab = _tab(_beam_config("H-"))
    meta = _meta(_cavity(), np.eye(6), species_name="H-")
    text, use = _row(tab, meta)
    assert use.isEnabled() and text == "0.00e+00"
    tab.state.set_beam_config(_beam_config("proton"))
    assert tab._table.item(0, 1).text() == "trained for H-"
    assert not tab._table.cellWidget(0, 3).isEnabled()


def test_the_template_is_the_projects_beam(qapp):
    r = _tab(_beam_config("proton", energy=30.0, frequency=325.0))._ref_template()
    assert r.species.name == PROTON.name and r.w_kin == 30.0 and r.frequency == 325.0


def test_with_no_beam_the_template_is_the_old_one_and_says_so(qapp):
    tab = _tab()
    r = tab._ref_template()
    assert r.species.name == H_MINUS.name and r.w_kin == 2.12 and r.frequency == 162.5
    assert "no beam" in tab._status.text()


def test_the_template_takes_a_dst_beams_energy_and_clock(qapp, tmp_path):
    from linac_gen.io.tracewin_dst import write_dst
    p = tmp_path / "b.dst"
    parts = np.zeros((8, 6))
    parts[:, 0] = np.linspace(-1.0, 1.0, 8)
    write_dst(str(p), parts, current_mA=1.0, frequency_MHz=325.0, mass_MeV=PROTON.mass, w_kin_ref=7.0)
    r = _tab(_beam_config(source="file", distribution_file=str(p)))._ref_template()
    assert r.frequency == 325.0 and r.w_kin == pytest.approx(7.0, rel=1e-9)


def test_a_beam_that_cannot_be_built_gives_no_template_and_says_why(qapp):
    tab = _tab(_beam_config(source="file", distribution_file="/nonexistent/x.dst"))
    assert tab._ref_template() is None
    assert "not usable" in tab._status.text()


def test_the_template_makes_no_particles_and_leaves_the_beam_alone(qapp, tmp_path, monkeypatch):
    """The reference a run gets, read from the config and a .dst's header; making the
    beam wrote the file's Twiss into the shared config and cost a beam's memory."""
    import copy

    from linac_gen.distributions import factory
    from linac_gen.io.tracewin_dst import write_dst
    p = tmp_path / "b.dst"
    parts = np.random.default_rng(3).normal(size=(500, 6)) * [1.0, 1.0, 1.0, 1.0, 5.0, 0.01]
    write_dst(str(p), parts, current_mA=1.0, frequency_MHz=325.0, mass_MeV=PROTON.mass, w_kin_ref=7.0)
    configs = [_beam_config(source="file", distribution_file=str(p)),
               _beam_config("deuteron", energy=30.0, frequency=325.0),
               _beam_config("H-", energy=2.1, continuous=True)]
    wants = [factory.create_beam(copy.deepcopy(cfg), seed=42).ref for cfg in configs]

    def made_a_beam(*args, **kwargs):
        raise AssertionError("made a beam")
    monkeypatch.setattr(factory, "create_beam", made_a_beam)
    for cfg, want in zip(configs, wants, strict=True):
        tab = _tab(cfg)
        before = copy.deepcopy(cfg)
        got = tab._ref_template()
        assert (got.species.name, got.w_kin, got.frequency) == (want.species.name, want.w_kin, want.frequency)
        assert tab.state.beam_config == before


def test_the_cached_envelope_follows_the_beam(qapp, monkeypatch):
    from linac_gen.tracking import envelope as env_mod
    tab = _tab()
    tab.state.set_lattice(_lattice(Drift("D", 100.0)))
    runs = []
    real = env_mod.EnvelopeSolver.run

    def spy(self):
        runs.append(1)
        return real(self)
    monkeypatch.setattr(env_mod.EnvelopeSolver, "run", spy)
    a = _at(F_OLD, w=5.0)
    assert tab._get_or_compute_envelope(a) is tab._get_or_compute_envelope(a)
    assert len(runs) == 1
    for other in (_at(F_OLD, w=6.0), _at(F_NEW, w=5.0), _at(F_OLD, w=5.0, species=H_MINUS)):
        assert tab._get_or_compute_envelope(other) is not tab._get_or_compute_envelope(a)
    assert len(tab._envelope_cache) == 1          # one beam at a time


def test_the_rows_of_an_element_with_interior_markers(qapp):
    """Markers inside a map add rows even with sub-steps off, so ``idx`` is not its row."""
    from linac_gen_gui.interphase.tabs.surrogates_tab import _element_rows

    from linac_gen.core.step_config import StepConfig
    from tests.tracking.test_envelope_reference_matches_tracker import _sfm

    def lat():
        line = _lattice(Drift("D0", 100.0), _sfm(markers=True), Drift("D1", 100.0))
        line.step_config = StepConfig(100.0, 50.0)
        return line
    ref = _at(352.21, w=3.0)
    res = EnvelopeSolver(lat(), ref.copy(), dict(INIT), current=0.0).run()
    for idx in range(3):
        row_in, row_out = _element_rows(res, idx)
        assert res.ref_w_kin[row_in] == reference_at(lat(), ref, idx).w_kin
        assert res.ref_w_kin[row_out] == reference_at(lat(), ref, idx + 1).w_kin
    assert res.ref_w_kin[2] != reference_at(lat(), ref, 2).w_kin    # the row the old index read


def test_a_result_without_the_list_uses_the_element_index(qapp):
    from types import SimpleNamespace

    from linac_gen_gui.interphase.tabs.surrogates_tab import _element_rows
    res = SimpleNamespace(ref_w_kin=[1.0, 2.0, 3.0])
    assert _element_rows(res, 1) == (1, 2)
    assert _element_rows(res, 2) == (2, 2)


@pytest.mark.parametrize("species, suffix", [("H-", ""), ("proton", "_proton"), ("deuteron", "_deuteron")])
def test_each_species_has_its_own_weights_folder(qapp, species, suffix):
    tab = _tab(_beam_config(species))
    assert tab._weights_root("0123456789abcdef0123").name == "0123456789abcdef" + suffix


def test_with_no_beam_the_weights_folder_is_the_plain_one(qapp):
    assert _tab()._weights_root("0123456789abcdef0123").name == "0123456789abcdef"


def test_a_proton_retrain_leaves_the_h_minus_files_alone(qapp, tmp_path, monkeypatch):
    """Through the worker the tab uses, on a proton project."""
    from linac_gen_gui.interphase.tabs.surrogates_tab import _TrainWorker
    monkeypatch.chdir(tmp_path)
    el = _cavity(F_OLD)
    line = _lattice(Drift("D0", 100.0), el, Drift("D1", 100.0))
    h_minus = _tab(_beam_config("H-"))._weights_dir_for_element(el.name, "ab" * 32)
    h_minus.mkdir(parents=True)
    (h_minus / "weights.pt").write_bytes(b"H- weights")
    tab = _tab(_beam_config("proton", energy=5.0))
    tab.state.set_lattice(line)
    template = tab._ref_template()
    res = tab._get_or_compute_envelope(template)
    assert res is not None
    out = tab._weights_dir_for_element(el.name, "ab" * 32)
    assert out != h_minus
    worker = _TrainWorker(
        element=el, ref_template=tab._at_element_phase(template, el, res), out_dir=out,
        lattice_hash="ab" * 32,
        options=dict(n_samples=16, ref_w_kin_range=(4.5, 5.5),
                     param_ranges={"phase": (-40.0, -20.0)}, hidden_dims=(4,), epochs=1,
                     lr=1e-3, batch_size=8, seed=0, val_frac=0.2, n_workers=None))
    done = []
    worker.finished_ok.connect(done.append)
    worker.run()
    meta = load_surrogate(out)[1]
    assert meta.species_name == PROTON.name
    assert meta.train_frequency_mhz == F_OLD
    assert meta.train_phi_s_deg == reference_at(line, template, 1).phi_s
    assert (h_minus / "weights.pt").read_bytes() == b"H- weights"


class _Stop(Exception):
    pass


class _AcceptingDialog:
    """The train dialog, accepted with its options."""
    def __init__(self, *args, **kwargs):
        pass

    def exec(self):
        from PyQt6.QtWidgets import QDialog
        return QDialog.DialogCode.Accepted

    def get_options(self):
        return {"n_samples": 16, "epochs": 1}


def _routes_line(monkeypatch, tmp_path):
    """A proton project whose cavity is entered at a phase other than 0, and a worker that
    records what it is given and goes no further."""
    from linac_gen_gui.interphase.tabs import surrogates_tab as mod
    monkeypatch.chdir(tmp_path)                 # no cached weights to be asked about
    got = {}

    def worker(**kw):
        got.update(kw)
        raise _Stop
    monkeypatch.setattr(mod, "_TrainWorker", worker)
    el = _cavity(F_OLD)
    line = _lattice(Drift("D0", 100.0), el, Drift("D1", 100.0))
    tab = _tab(_beam_config("proton", energy=5.0))
    tab.state.set_lattice(line)
    return mod, tab, el, line, got


def test_a_beam_that_cannot_be_built_stops_training(qapp, tmp_path, monkeypatch):
    """Rather than train H⁻ into the project's folder, and say nothing."""
    mod, tab, el, line, got = _routes_line(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "_TrainDialog", _AcceptingDialog)
    shown = []
    monkeypatch.setattr(mod.QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[1:3])))
    tab.state.set_beam_config(_beam_config(source="file", distribution_file=str(tmp_path / "missing.dst")))
    tab._on_train_clicked()
    assert got == {} and shown and "not usable" in shown[0][1]


def test_the_train_button_writes_into_the_folder_of_the_species_it_trains(qapp, tmp_path, monkeypatch):
    mod, tab, el, line, got = _routes_line(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "_TrainDialog", _AcceptingDialog)
    with pytest.raises(_Stop):
        tab._on_train_clicked()
    assert got["ref_template"].species.name == PROTON.name
    assert Path(got["out_dir"]).parent.name.endswith("_proton")


def test_a_batch_writes_into_the_folder_of_the_species_it_trains(qapp, tmp_path, monkeypatch):
    """Not into that of whatever beam is set when an element's turn comes."""
    mod, tab, el, line, got = _routes_line(monkeypatch, tmp_path)
    controller = mod._BatchTrainController(
        elements=[el], ref_template=tab._ref_template(), lattice=line, lattice_hash="ab" * 32,
        shared_opts={"n_samples": 16, "epochs": 1}, tab=tab)
    tab.state.set_beam_config(_beam_config("H-"))
    with pytest.raises(_Stop):
        controller.start()
    assert Path(got["out_dir"]).parent.name == "abababababababab_proton"


def test_a_batch_loads_each_element_back_from_where_its_worker_wrote(qapp, tmp_path, monkeypatch):
    mod, tab, el, line, got = _routes_line(monkeypatch, tmp_path)
    controller = mod._BatchTrainController(
        elements=[el], ref_template=tab._ref_template(), lattice=line, lattice_hash="ab" * 32,
        shared_opts={"n_samples": 16, "epochs": 1}, tab=tab)
    with pytest.raises(_Stop):
        controller.start()
    out = Path(got["out_dir"])
    _save(out, el, np.eye(6), drop=(), species_name=PROTON.name, train_frequency_mhz=F_OLD)  # the worker's
    tab.state.set_beam_config(_beam_config("H-"))
    monkeypatch.setattr(controller, "_train_next", lambda: None)
    controller._on_element_done(None)
    assert tab._trained[el.name][1] == out


def test_discovery_follows_the_species_of_the_beam(qapp, tmp_path, monkeypatch):
    """In the folder the tab picks itself, also when a project sets its lattice before its
    beam, as opening one does; and a change of beam that keeps the species keeps the rows."""
    monkeypatch.chdir(tmp_path)
    deck = tmp_path / "line.dat"
    deck.write_text("DRIFT 100 20\nEND\n")         # any file: its hash names the folder
    el = _cavity(F_OLD, p_flag=1)
    h16 = registry.hash_lattice_file(str(deck))[:16]
    root = Path("linac_gen/surrogates/weights")
    _save(root / h16 / el.name, el, np.eye(6), drop=(), species_name=H_MINUS.name, train_frequency_mhz=F_OLD)
    _save(root / f"{h16}_proton" / el.name, el, np.eye(6), drop=(), species_name=PROTON.name,
          train_frequency_mhz=F_OLD)
    tab = _tab(_beam_config("H-"))
    tab.state.set_lattice(_lattice(el), str(deck))
    assert tab._trained[el.name][2].species_name == H_MINUS.name
    assert "; " not in tab._status.text()             # nothing that cannot be used
    tab.state.set_beam_config(_beam_config("proton"))
    assert tab._trained[el.name][2].species_name == PROTON.name
    assert tab._table.cellWidget(0, 3).isEnabled()
    tab._table.cellWidget(0, 3).setChecked(True)
    tab.state.set_beam_config(_beam_config("proton", energy=6.0))
    assert registry.get("lh", el.name) is not None    # the same species: the rows and ticks stay
    registry.clear()
    tab.state.set_beam_config(_beam_config("deuteron"))
    assert tab._trained == {} and "no cached surrogates" in tab._status.text()


def test_the_tab_says_when_it_cannot_find_the_entrance_phase(qapp):
    """Only for a cavity whose field follows that phase, where it matters."""
    tab = _tab()
    el = _cavity(F_OLD, p_flag=0)
    tab.state.set_lattice(_lattice(Drift("D0", 100.0), el))
    assert tab._at_element_phase(_at(F_OLD), el, None).phi_s == 0.0
    assert "not found" in tab._status.text()
    tab._status.setText("")
    tab._at_element_phase(_at(F_OLD), _cavity(F_OLD, p_flag=1), None)
    assert tab._status.text() == ""


def test_training_starts_at_the_phase_the_element_is_entered_at(qapp):
    tab = _tab()
    el = _cavity(F_OLD)
    line = _lattice(Drift("D0", 100.0), el, Drift("D1", 100.0))
    tab.state.set_lattice(line)
    ref = _at(F_OLD)
    res = EnvelopeSolver(line, ref.copy(), dict(INIT), current=0.0).run()
    got = tab._at_element_phase(ref, el, res)
    assert got.phi_s == reference_at(line, ref, 1).phi_s != 0.0
    assert ref.phi_s == 0.0                 # the shared template stays as it was
    assert tab._at_element_phase(ref, el, None).phi_s == 0.0


def test_the_train_button_starts_at_the_phase_the_element_is_entered_at(qapp, tmp_path, monkeypatch):
    """Through the button's slot, with its dialog accepted."""
    mod, tab, el, line, got = _routes_line(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "_TrainDialog", _AcceptingDialog)
    assert tab._elem_combo.count() == 1
    with pytest.raises(_Stop):
        tab._on_train_clicked()
    template = tab._ref_template()
    assert got["element"] is el
    assert got["ref_template"].phi_s == reference_at(line, template, 1).phi_s != 0.0


def test_batch_training_starts_each_element_at_its_phase(qapp, tmp_path, monkeypatch):
    mod, tab, el, line, got = _routes_line(monkeypatch, tmp_path)
    template = tab._ref_template()
    controller = mod._BatchTrainController(
        elements=[el], ref_template=template, lattice=line, lattice_hash="ab" * 32,
        shared_opts={"n_samples": 16, "epochs": 1}, tab=tab)
    with pytest.raises(_Stop):
        controller.start()
    assert got["element"] is el
    assert got["ref_template"].phi_s == reference_at(line, template, 1).phi_s != 0.0
    assert template.phi_s == 0.0


# ---------------------------------------------------------------------------
# a cavity that switches the clock itself
# ---------------------------------------------------------------------------

def test_asked_at_another_clock_the_phase_is_read_on_the_cavitys():
    """The cavity switches the clock on entry, which moves the reference phase as a FREQ card does, and a surrogate
    reads its phase input there.  Trained at 100° on its own clock with its phase not swept, it answers at 50° on a
    clock of half its frequency (100° on its own), and not at 100° there."""
    el = _cavity(F_NEW, p_flag=0)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_NEW, train_phi_s_deg=100.0)
    surr.fitted_matrix(_at(F_OLD, phi_s=50.0))
    assert surr.nn_calls == 1
    with pytest.raises(OutOfScopeError):
        surr.fitted_matrix(_at(F_OLD, phi_s=100.0))



def test_asked_at_another_clock_the_phase_input_is_read_on_the_cavitys():
    """With its phase swept, the input itself: 50° on half its clock is 100° on its own, the training's, so the
    phase input is the cavity's phase and in scope."""
    el = _cavity(F_NEW, p_flag=0)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_NEW, train_phi_s_deg=100.0, **PHASE)
    surr.fitted_matrix(_at(F_OLD, phi_s=50.0))
    assert surr.nn_calls == 1


def test_a_frequency_offset_does_not_move_the_phase_a_surrogate_reads():
    """An LLRF offset is not a change of clock: the phase stays where it was, 500° here, the training's."""
    el = _cavity(F_NEW, p_flag=0)
    el.frequency_offset = 0.1                              # 100 kHz: 0.15° at 500° were it a change of clock
    # the offset changes the map, so the weights must have swept the frequency to answer
    swept = dict(params=("frequency",), lo=(F_NEW - 1.0,), hi=(F_NEW + 1.0,), means=(F_NEW,))
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_NEW, train_phi_s_deg=500.0, **swept)
    surr.fitted_matrix(_at(F_NEW, phi_s=500.0))
    assert surr.nn_calls == 1

def test_the_template_carries_the_clock_its_phase_is_read_on(qapp):
    """Behind a FREQ card the envelope reads the entrance phase on the card's clock.  The template takes that clock
    with it, so the training does not move a phase that is already on the cavity's clock."""
    tab = _tab()
    el = _cavity(F_NEW)
    line = _lattice(Drift("D0", 100.0), Freq("FREQ", F_NEW), el, Drift("D1", 100.0))
    tab.state.set_lattice(line)
    ref = _at(F_OLD)
    res = EnvelopeSolver(line, ref.copy(), dict(INIT), current=0.0).run()
    got = tab._at_element_phase(ref, el, res)
    entry = reference_at(line, ref, 2)
    assert got.frequency == entry.frequency == F_NEW
    assert got.phi_s == pytest.approx(entry.phi_s, rel=1e-12) and got.phi_s != 0.0
    assert ref.frequency == F_OLD and ref.phi_s == 0.0      # the shared template stays as it was
