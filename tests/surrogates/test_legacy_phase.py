"""A file from before the training phase was recorded is checked against its native element once.

Such a file counts as trained at reference phase 0, which the Surrogates tab and the CLI
used.  One trained through the Python API at another phase, for a cavity whose field
follows the running phase, then had its phase input shifted from the wrong origin.  At
its first use after loading, the map is scanned over its phase sweep against the native
element: if phase 0 is not where it fits, it goes native and the message names the
phase it fits at, which can be declared in its metadata.json.
"""
import copy
import json

import numpy as np
import pytest

from linac_gen.io.tracewin_geom import Channel
from linac_gen.surrogates.base import OutOfScopeError, SurrogateFieldMap
from linac_gen.surrogates.training import load_surrogate, train_surrogate_for_element
from tests.surrogates.test_frequency_jump import F_NEW, _map
from tests.surrogates.test_surrogate_provenance import _at

PHASE_SWEEP = (-120.0, 60.0)


def _cavity():
    return _map(Channel.RF_E, F_NEW, p_flag=0)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Weights of one cavity trained at reference phase 20 deg and at 0 deg."""
    out = {}
    for phi in (0.0, 20.0):
        d = tmp_path_factory.mktemp(f"phi{phi:g}")
        train_surrogate_for_element(
            element=_cavity(), ref_template=_at(F_NEW, phi_s=phi), n_samples=300,
            ref_w_kin_range=(4.0, 6.0), param_ranges={"phase": PHASE_SWEEP}, hidden_dims=(32, 32),
            epochs=300, lr=3e-3, batch_size=64, seed=0, out_dir=d, element_key="MAP")
        out[phi] = d
    return out


def _as_before(d, *, declare=None):
    """The file as one from before the clock and the phase were recorded (or with the phase
    declared by hand)."""
    m = d / "metadata.json"
    data = json.loads(m.read_text())
    data.pop("train_frequency_mhz", None)
    data.pop("train_phi_s_deg", None)
    if declare is not None:
        data["train_phi_s_deg"] = declare
    m.write_text(json.dumps(data))
    return d


def _loaded(d, el):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp, meta = load_surrogate(d)
    # networks trained in seconds: the phase check under test comes before the accuracy check,
    # which has its own tests (test_accuracy_check.py)
    meta.accuracy_due = False
    return SurrogateFieldMap(el, mlp, meta)


def _copy_of(d, tmp_path):
    import shutil
    dst = tmp_path / d.name
    shutil.copytree(d, dst)
    return dst


def test_a_file_from_the_api_at_another_phase_goes_native_and_says_where_it_fits(trained, tmp_path):
    surr = _loaded(_as_before(_copy_of(trained[20.0], tmp_path)), _cavity())
    assert surr.metadata.legacy_weights
    with pytest.raises(OutOfScopeError, match=r"fits best near a reference phase of") as e:
        surr.fitted_matrix(_at(F_NEW, w=5.0))
    fit = float(str(e.value).split("near a reference phase of ")[1].split("°")[0])
    assert abs(fit - 20.0) <= 5.0, str(e.value)
    assert "if you know it" in str(e.value) and "train_phi_s_deg" in str(e.value)


def test_the_check_runs_once_per_load(trained, tmp_path, monkeypatch):
    el = _cavity()
    surr = _loaded(_as_before(_copy_of(trained[20.0], tmp_path)), el)
    calls = []
    real = type(el).fitted_matrix

    def counted(self, ref):
        calls.append(1)
        return real(self, ref)

    monkeypatch.setattr(type(el), "fitted_matrix", counted)
    for _ in range(3):
        with pytest.raises(OutOfScopeError):
            surr.fitted_matrix(_at(F_NEW, w=5.0))
    assert len(calls) == 1


def test_a_declared_phase_is_believed(trained, tmp_path):
    surr = _loaded(_as_before(_copy_of(trained[20.0], tmp_path), declare=20.0), _cavity())
    surr.fitted_matrix(_at(F_NEW, w=5.0))
    assert surr.nn_calls == 1


def test_a_file_trained_at_phase_0_answers(trained, tmp_path):
    el = _cavity()
    surr = _loaded(_as_before(_copy_of(trained[0.0], tmp_path)), el)
    ref = _at(F_NEW, w=5.0, phi_s=15.0)
    got = surr.fitted_matrix(ref.copy())
    assert surr.nn_calls == 1
    native = copy.deepcopy(el).fitted_matrix(ref.copy())
    assert np.abs(got - native).max() < 0.1 * np.abs(native).max()


def test_a_file_that_records_its_phase_is_not_checked(trained, tmp_path, monkeypatch):
    el = _cavity()
    surr = _loaded(_copy_of(trained[20.0], tmp_path), el)
    assert not surr.metadata.legacy_weights
    calls = []
    real = type(el).fitted_matrix
    monkeypatch.setattr(type(el), "fitted_matrix", lambda self, ref: (calls.append(1), real(self, ref))[1])
    surr.fitted_matrix(_at(F_NEW, w=5.0, phi_s=20.0))
    assert calls == [] and surr.nn_calls == 1


def test_a_roughly_trained_file_at_phase_0_is_not_refused(tmp_path):
    """Trained as the Surrogates tab and the CLI train by default (200 samples, 40 epochs):
    its network's own error looked like a phase shift, and nearly half the checks refused
    it, naming a phase to declare that was wrong."""
    from tests.surrogates.test_frequency_jump import _cavity_1d
    for seed in range(2):
        d = tmp_path / f"s{seed}"
        train_surrogate_for_element(
            element=_cavity_1d(p_flag=0), ref_template=_at(F_NEW, phi_s=0.0), n_samples=200,
            ref_w_kin_range=(4.0, 6.0), param_ranges={"ke": (0.8, 1.2), "phase": (-50.0, -10.0)},
            hidden_dims=(64, 64), epochs=40, seed=seed, out_dir=d, element_key="MAP")
        _as_before(d)
        for w in (4.5, 5.0, 5.5):
            for phi in (-15.0, 0.0, 15.0):
                _loaded(d, _cavity_1d(p_flag=0)).fitted_matrix(_at(F_NEW, w=w, phi_s=phi))


def test_a_best_fit_at_the_edge_of_the_sweep_names_no_phase(tmp_path):
    """Trained 40 deg off with a narrow sweep: the input it should take is outside the
    sweep, so the best fit is at its edge and no phase is named."""
    d = tmp_path / "narrow"
    train_surrogate_for_element(
        element=_cavity(), ref_template=_at(F_NEW, phi_s=40.0), n_samples=400,
        ref_w_kin_range=(4.0, 6.0), param_ranges={"phase": (-50.0, -10.0)}, hidden_dims=(64, 64),
        epochs=300, lr=3e-3, batch_size=64, seed=0, out_dir=d, element_key="MAP")
    surr = _loaded(_as_before(d), _cavity())
    with pytest.raises(OutOfScopeError, match="edge of its phase sweep") as e:
        surr.fitted_matrix(_at(F_NEW, w=5.0, phi_s=15.0))
    assert "near a reference phase" not in str(e.value)
