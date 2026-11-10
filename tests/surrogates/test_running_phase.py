"""Which surrogates follow the running phase, and how their inputs follow it.

A field map's field follows the running phase where the RF drives it: an RF channel,
electric or magnetic, with a non-zero amplitude.  A static channel does not, nor an RF
channel switched off (ke or kb = 0), whose map is then the same at every phase.
"""
import copy

import numpy as np
import pytest

from linac_gen.io.tracewin_geom import Channel
from linac_gen.surrogates.base import OutOfScopeError
from linac_gen.tracking.torch_tracking import element_matrix_torch
from tests.surrogates.test_frequency_jump import F_NEW, _map
from tests.surrogates.test_surrogate_provenance import _at, _kin, _surrogate

PHASES = (0.0, 40.0, 137.0)


def _native(el, ref):
    return copy.deepcopy(el).fitted_matrix(ref.copy())


def _cavity(channel=Channel.RF_E, *, ke=1.0, kb=1.0, p_flag=0):
    el = _map(channel, F_NEW, p_flag=p_flag)
    el.ke, el.kb = ke, kb
    return el


def _phase_free(el):
    """A surrogate pinned to the native map at phase 0, on the map's own clock, phase not swept."""
    return _surrogate(el, _native(el, _at(F_NEW)), train_frequency_mhz=F_NEW, train_phi_s_deg=0.0)


def test_a_cavity_switched_off_is_the_same_at_every_phase():
    el = _cavity(ke=0.0)
    for phi in PHASES:
        np.testing.assert_array_equal(_native(el, _at(F_NEW, phi_s=phi)), _native(el, _at(F_NEW)))


@pytest.mark.parametrize("phi", PHASES)
def test_a_cavity_switched_off_answers_at_any_phase(phi):
    el = _cavity(ke=0.0)
    surr = _phase_free(el)
    ref = _at(F_NEW, phi_s=phi)
    np.testing.assert_allclose(surr.fitted_matrix(ref.copy()), _native(el, ref), rtol=0, atol=1e-12)
    got = element_matrix_torch(surr, _kin(ref)).detach().numpy()
    np.testing.assert_allclose(got, _native(el, ref), rtol=0, atol=1e-12)


def test_a_static_field_is_the_same_at_every_phase_and_answers():
    el = _cavity(Channel.STAT_E)
    for phi in PHASES:
        np.testing.assert_allclose(_native(el, _at(F_NEW, phi_s=phi)), _native(el, _at(F_NEW)),
                                   rtol=0, atol=1e-14)
    surr = _phase_free(el)
    for phi in PHASES:
        surr.fitted_matrix(_at(F_NEW, phi_s=phi))
        element_matrix_torch(surr, _kin(_at(F_NEW, phi_s=phi)))


def test_a_driven_cavity_still_goes_native_off_its_phase():
    surr = _phase_free(_cavity(ke=1.0))
    surr.fitted_matrix(_at(F_NEW))
    with pytest.raises(OutOfScopeError, match="follows the running phase"):
        surr.fitted_matrix(_at(F_NEW, phi_s=40.0))
    with pytest.raises(ValueError, match="follows the running phase"):
        element_matrix_torch(surr, _kin(_at(F_NEW)))


def test_an_rf_magnetic_field_follows_the_running_phase():
    """Natively its map depends on phase + phi_s only, as an electric one's does."""
    el = _cavity(Channel.RF_B, kb=1.0)
    a = copy.deepcopy(el)
    a.phase = el.phase + 40.0
    np.testing.assert_allclose(_native(el, _at(F_NEW, phi_s=40.0)), _native(a, _at(F_NEW)),
                               rtol=0, atol=1e-12)
    assert np.abs(_native(el, _at(F_NEW, phi_s=40.0)) - _native(el, _at(F_NEW))).max() > 1e-9


def test_an_rf_magnetic_field_takes_the_phase_shift():
    el = _cavity(Channel.RF_B, kb=1.0)
    surr = _surrogate(el, _native(el, _at(F_NEW)), train_frequency_mhz=F_NEW, train_phi_s_deg=0.0,
                      params=("phase",), lo=(-180.0,), hi=(180.0,), means=(el.phase,))
    x = surr._make_input(_at(F_NEW, phi_s=40.0))
    assert x[3] == pytest.approx(el.phase + 40.0, abs=1e-12)
    with pytest.raises(ValueError, match="follows the running phase"):
        element_matrix_torch(surr, _kin(_at(F_NEW)))


# ---------------------------------------------------------------------------
# NCELLS: P=1 fires each gap at phi_s - theta_s
# ---------------------------------------------------------------------------
F_NC = 162.5
PHI_T = 100.0           # the reference phase a surrogate below is trained at


def _ncells(p_flag=1, theta=-30.0, sync=False, eot=2.0e6):
    from linac_gen.elements.ncells import NCells
    return NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=eot, theta_s_deg=theta,
                  aperture_mm=15.0, p_flag=p_flag, frequency_mhz=F_NC, sync_phase=sync)


def _nc_native(theta=-30.0, phi=PHI_T, **kw):
    return _native(_ncells(theta=theta, **kw), _at(F_NC, phi_s=phi))


@pytest.mark.parametrize("delta", [40.0, -75.0, 250.0])
def test_a_p1_cavity_depends_on_phi_s_minus_theta(delta):
    """M(theta, phi_t + delta) = M(theta - wrap180(delta), phi_t)."""
    turned = ((delta + 180.0) % 360.0) - 180.0
    np.testing.assert_allclose(_nc_native(-30.0, PHI_T + delta), _nc_native(-30.0 - turned, PHI_T),
                               rtol=0, atol=1e-12)
    assert np.abs(_nc_native(phi=PHI_T + delta) - _nc_native()).max() > 1e-6


@pytest.mark.parametrize("kw", [dict(p_flag=0), dict(p_flag=2), dict(p_flag=1, sync=True)])
def test_p0_p2_and_sync_do_not_depend_on_phi_s(kw):
    np.testing.assert_allclose(_nc_native(phi=PHI_T + 40.0, **kw), _nc_native(**kw), rtol=0, atol=1e-12)


def _nc_surrogate(el, *, swept=True):
    params = dict(params=("theta_s_deg",), lo=(-180.0,), hi=(180.0,), means=(-30.0,)) if swept else {}
    return _surrogate(el, np.eye(6), train_frequency_mhz=F_NC, train_phi_s_deg=PHI_T, **params)


@pytest.mark.parametrize("theta, phi, want", [(-30.0, PHI_T + 40.0, -70.0),
                                              (-30.0, PHI_T + 250.0, 80.0),
                                              (170.0, PHI_T - 20.0, -170.0)])
def test_theta_follows_the_running_phase_backwards(theta, phi, want):
    surr = _nc_surrogate(_ncells(theta=theta))
    assert surr._make_input(_at(F_NC, phi_s=phi))[3] == pytest.approx(want, abs=1e-9)


def test_theta_not_swept_answers_only_at_the_training_phase():
    surr = _nc_surrogate(_ncells(), swept=False)
    surr.fitted_matrix(_at(F_NC, phi_s=PHI_T))
    with pytest.raises(OutOfScopeError, match="follows the running phase"):
        surr.fitted_matrix(_at(F_NC, phi_s=PHI_T + 0.02))
    with pytest.raises(ValueError, match="follows the running phase"):
        element_matrix_torch(_nc_surrogate(_ncells()), _kin(_at(F_NC, phi_s=PHI_T)))


@pytest.mark.parametrize("kw", [dict(p_flag=0), dict(p_flag=2), dict(p_flag=1, sync=True),
                                dict(p_flag=1, eot=0.0)])
def test_the_others_are_not_shifted(kw):
    surr = _nc_surrogate(_ncells(**kw))
    assert surr._make_input(_at(F_NC, phi_s=PHI_T + 40.0))[3] == -30.0


def test_a_p1_surrogate_asked_off_its_phase_answers_for_the_shifted_theta():
    """Exactly the answer it gives at its training phase for theta - delta."""
    from linac_gen.surrogates.base import SurrogateFieldMap
    surr = _nc_surrogate(_ncells(theta=-30.0))
    meta = surr.metadata
    meta.output_norm = {"mean": [0.0] * 36, "std": [1.0] * 36}      # an output that moves
    surr = SurrogateFieldMap(_ncells(theta=-30.0), surr.mlp, meta)
    shifted = SurrogateFieldMap(_ncells(theta=-70.0), surr.mlp, meta)
    np.testing.assert_array_equal(surr.fitted_matrix(_at(F_NC, phi_s=PHI_T + 40.0)),
                                  shifted.fitted_matrix(_at(F_NC, phi_s=PHI_T)))
    assert not np.array_equal(surr.fitted_matrix(_at(F_NC, phi_s=PHI_T + 40.0)),
                              surr.fitted_matrix(_at(F_NC, phi_s=PHI_T)))


def test_a_trained_p1_surrogate_follows_the_running_phase(tmp_path):
    """Asked 40 deg past its training phase, its error is a small part of how far the native
    map moves, both in the network's own units (each entry over its training spread): it
    ignored phi_s, so its error was the whole movement."""
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.surrogates.training import train_surrogate_for_element
    el = _ncells()
    template = _at(F_NC, phi_s=PHI_T)
    mlp, meta = train_surrogate_for_element(
        element=el, ref_template=template, n_samples=400, ref_w_kin_range=(4.0, 6.0),
        param_ranges={"theta_s_deg": (-120.0, 60.0)}, hidden_dims=(32, 32), epochs=400,
        lr=3e-3, batch_size=64, seed=0, out_dir=tmp_path / "nc", element_key="NC")
    surr = SurrogateFieldMap(el, mlp, meta)
    ref = _at(F_NC, w=5.0, phi_s=PHI_T + 40.0)
    std = np.asarray(meta.output_norm["std"]).reshape(6, 6)
    moved = np.linalg.norm((_native(el, ref) - _native(el, _at(F_NC, w=5.0, phi_s=PHI_T))) / std)
    err = np.linalg.norm((surr.fitted_matrix(ref.copy()) - _native(el, ref)) / std)
    assert err < 0.25 * moved, (err, moved)
