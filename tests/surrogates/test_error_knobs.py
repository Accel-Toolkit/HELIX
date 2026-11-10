"""A surrogate answers for the element as an error study leaves it, or goes native.

The weights hold the map of the design element.  An error study, or SET_BEAM_PHASE_ERROR,
moves the element's phase (phase_offset, the RF shift), its amplitude (voltage_rel) or
its frequency (frequency_offset); the surrogate answered as if nothing had moved.  Each
is the same map as a design element with an input moved (checked natively below), so it
is folded into that input, and refused when that input was not swept in training.
"""
import copy

import numpy as np
import pytest

from linac_gen.io.tracewin_geom import Channel
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import OutOfScopeError
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.torch_matrices import RefKinematics
from linac_gen.tracking.torch_tracking import element_matrix_torch
from tests.surrogates.test_frequency_jump import (
    F_NEW,
    FieldChannel,
    FieldMap3D,
    FieldMapData,
    _lattice,
    _map,
)
from tests.surrogates.test_running_phase import F_NC, PHI_T, _ncells
from tests.surrogates.test_surrogate_provenance import INIT, _at, _kin, _surrogate


def _native(el, ref):
    return copy.deepcopy(el).fitted_matrix(ref.copy())


def _with(el, **kw):
    e = copy.deepcopy(el)
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def _cavity(p_flag=1):
    return _map(Channel.RF_E, F_NEW, p_flag=p_flag)


def _two_channels():
    """An RF electric channel and a static magnetic one: ke and kb both act."""
    nz, n, L = 11, 3, 200.0
    x, z = np.linspace(-10.0, 10.0, n), np.linspace(0.0, L, nz)
    fd = FieldMapData(z=z, frequency=F_NEW)
    for ch, fz in ((Channel.RF_E, 1.0), (Channel.STAT_B, 0.5)):
        fd.channels[ch] = FieldChannel(geometry=7, x=x, y=x, z=z, Fx=np.zeros((n, n, nz)),
                                       Fy=np.zeros((n, n, nz)), Fz=np.full((n, n, nz), fz))
    return FieldMap3D(name="MAP", length=L, field_data=fd, scale=1.0, phase=-30.0, frequency=F_NEW,
                      aperture=30.0, n_steps=20, p_flag=1, ke=1.0, kb=0.8)


def _ref(phi=0.0, shift=0.0, rephase=0.0, f=F_NEW):
    r = _at(f, phi_s=phi)
    r.rf_shift, r.rf_rephase = shift, rephase
    return r


def same(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def differ(a, b):
    assert np.abs(np.asarray(a) - np.asarray(b)).max() > 1e-6


# ---------------------------------------------------------------------------
# natively, each moved element is a design element with an input moved
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p_flag", [0, 1])
def test_a_phase_offset_adds_to_the_phase(p_flag):
    el = _cavity(p_flag)
    same(_native(_with(el, phase_offset=7.0), _ref()), _native(_with(el, phase=el.phase + 7.0), _ref()))


def test_the_rf_shift_adds_to_the_phase_of_a_map_that_follows_the_running_phase():
    el = _cavity(p_flag=0)
    same(_native(el, _ref(shift=5.0)), _native(_with(el, phase=el.phase + 5.0), _ref()))
    differ(_native(el, _ref(shift=5.0)), _native(el, _ref()))


def test_at_p_flag_1_the_rf_shift_comes_after_the_calibration():
    """So it is not a phase change, which the calibration would follow."""
    el = _cavity(p_flag=1)
    moved = _native(el, _ref(shift=5.0))
    differ(moved, _native(el, _ref()))
    assert np.abs(moved - _native(_with(el, phase=el.phase + 5.0), _ref())).max() > 1e-6


def test_a_voltage_error_scales_scale_or_both_ke_and_kb():
    el = _cavity()
    same(_native(_with(el, voltage_rel=0.1), _ref()), _native(_with(el, scale=1.1), _ref()))
    two = _two_channels()
    moved = _native(_with(two, voltage_rel=0.1), _ref())
    same(moved, _native(_with(two, ke=two.ke * 1.1, kb=two.kb * 1.1), _ref()))
    differ(moved, _native(_with(two, ke=two.ke * 1.1), _ref()))          # ke alone misses kb


def test_the_clock_correction_moves_the_running_phase():
    el = _cavity(p_flag=0)
    same(_native(el, _ref(rephase=6.0)), _native(el, _ref(phi=6.0)))


def test_an_ncells_fires_at_theta_plus_its_offset_and_the_shift():
    p1, p0 = _ncells(p_flag=1), _ncells(p_flag=0)
    at = _ref(PHI_T, f=F_NC)
    same(_native(_with(p1, phase_offset=7.0), at), _native(_with(p1, theta_s_deg=-23.0), at))
    same(_native(p1, _ref(PHI_T, shift=5.0, f=F_NC)), _native(_with(p1, theta_s_deg=-35.0), at))
    same(_native(p0, _ref(PHI_T, shift=5.0, f=F_NC)), _native(_with(p0, theta_s_deg=-25.0), at))
    same(_native(p1, _ref(PHI_T, rephase=6.0, f=F_NC)), _native(_with(p1, theta_s_deg=-36.0), at))


def test_under_set_sync_phase_an_offset_is_not_a_theta():
    sync = _ncells(p_flag=1, sync=True)
    at = _ref(PHI_T, f=F_NC)
    differ(_native(_with(sync, phase_offset=7.0), at), _native(_with(sync, theta_s_deg=-23.0), at))


# ---------------------------------------------------------------------------
# folded into the inputs
# ---------------------------------------------------------------------------
SWEPT = dict(params=("scale", "phase", "frequency"), lo=(0.5, -180.0, F_NEW - 1.0),
             hi=(1.5, 180.0, F_NEW + 1.0), means=(1.0, -30.0, F_NEW))


def _inputs(el, ref, **swept):
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_NEW, **(swept or SWEPT))
    return surr._make_input(ref)[3:]


def test_the_knobs_of_a_field_map_are_folded():
    el = _with(_cavity(), phase_offset=7.0, voltage_rel=0.1, frequency_offset=0.05)
    np.testing.assert_allclose(_inputs(el, _ref()), [1.1, -30.0 + 7.0, F_NEW + 0.05], rtol=0, atol=1e-12)


def test_a_map_the_rf_does_not_drive_takes_no_phase_or_frequency_knob():
    el = _with(_cavity(), ke=0.0, phase_offset=7.0, frequency_offset=0.05)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_NEW)          # nothing swept
    surr.fitted_matrix(_ref(shift=5.0))
    assert surr.nn_calls == 1


def test_a_voltage_error_goes_into_ke_and_kb_when_scale_was_not_swept():
    el = _with(_two_channels(), voltage_rel=0.1)
    kk = dict(params=("ke", "kb"), lo=(0.0, 0.0), hi=(2.0, 2.0), means=(1.0, 0.8))
    np.testing.assert_allclose(_inputs(el, _ref(), **kk), [el.ke * 1.1, el.kb * 1.1], rtol=0, atol=1e-12)


def test_the_running_phase_and_the_clock_correction_add_on_top():
    el = _with(_cavity(p_flag=0), phase_offset=7.0)
    got = _surrogate(el, np.eye(6), train_frequency_mhz=F_NEW, train_phi_s_deg=0.0, **SWEPT)._make_input(
        _ref(phi=20.0, shift=5.0, rephase=6.0))
    assert got[4] == pytest.approx(-30.0 + 7.0 + 5.0 + 20.0 + 6.0, abs=1e-12)


@pytest.mark.parametrize("p_flag, want", [(1, -30.0 + 7.0 - 5.0), (0, -30.0 + 7.0 + 5.0)])
def test_an_ncells_offset_and_shift_go_into_theta(p_flag, want):
    el = _with(_ncells(p_flag=p_flag), phase_offset=7.0)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_NC, train_phi_s_deg=PHI_T,
                      params=("theta_s_deg",), lo=(-180.0,), hi=(180.0,), means=(-30.0,))
    assert surr._make_input(_ref(PHI_T, shift=5.0, f=F_NC))[3] == pytest.approx(want, abs=1e-12)


def test_a_phase_moved_past_the_scope_is_turned_back():
    el = _with(_cavity(), phase=170.0, phase_offset=20.0)
    assert _inputs(el, _ref())[1] == pytest.approx(-170.0, abs=1e-12)


def test_no_knob_leaves_the_inputs_as_they_were():
    np.testing.assert_array_equal(_inputs(_cavity(), _ref()), [1.0, -30.0, F_NEW])


# ---------------------------------------------------------------------------
# refused when the input was not swept
# ---------------------------------------------------------------------------
REFUSED = [
    (lambda: _with(_cavity(), phase_offset=7.0), {}, _ref(), "'phase' was not swept"),
    (lambda: _cavity(p_flag=0), {}, _ref(shift=5.0), "'phase' was not swept"),
    (lambda: _cavity(), SWEPT, _ref(shift=5.0), "after its p_flag 1 calibration"),
    (lambda: _with(_cavity(), voltage_rel=0.1), {}, _ref(), "voltage_rel"),
    (lambda: _with(_two_channels(), voltage_rel=0.1),
     dict(params=("ke",), lo=(0.0,), hi=(2.0,), means=(1.0,)), _ref(), "voltage_rel"),
    (lambda: _with(_cavity(), frequency_offset=0.05), {}, _ref(), "'frequency' was not swept"),
]


@pytest.mark.parametrize("make, swept, ref, words", REFUSED)
def test_a_knob_whose_input_was_not_swept_is_refused(make, swept, ref, words):
    surr = _surrogate(make(), np.eye(6), train_frequency_mhz=F_NEW, **swept)
    with pytest.raises(OutOfScopeError, match=words):
        surr.fitted_matrix(ref.copy())
    with pytest.raises(ValueError, match=words):
        element_matrix_torch(surr, _kin(ref))


@pytest.mark.parametrize("make, words", [
    (lambda: _with(_ncells(), voltage_rel=0.1), "gap voltages"),
    (lambda: _with(_ncells(sync=True), phase_offset=7.0), "SET_SYNC_PHASE"),
])
def test_what_no_ncells_input_stands_for_is_refused(make, words):
    surr = _surrogate(make(), np.eye(6), train_frequency_mhz=F_NC, train_phi_s_deg=PHI_T,
                      params=("theta_s_deg",), lo=(-180.0,), hi=(180.0,), means=(-30.0,))
    with pytest.raises(OutOfScopeError, match=words):
        surr.fitted_matrix(_ref(PHI_T, f=F_NC))


def test_the_torch_arm_reads_the_rf_shift_from_its_kinematics():
    kin = RefKinematics.from_reference(_ref(shift=5.0, rephase=6.0))
    assert (kin.rf_shift, kin.rf_rephase) == (5.0, 6.0)
    surr = _surrogate(_cavity(p_flag=0), np.eye(6), train_frequency_mhz=F_NEW, **SWEPT)
    assert surr._param_values(ref=kin)[1] == pytest.approx(-25.0, abs=1e-12)
    with pytest.raises(ValueError, match="after its p_flag 1 calibration"):
        element_matrix_torch(_surrogate(_cavity(), np.eye(6), train_frequency_mhz=F_NEW, **SWEPT), kin)


# ---------------------------------------------------------------------------
# end to end: an envelope run with a cavity error and its surrogate engaged
# ---------------------------------------------------------------------------
def test_an_error_the_weights_cannot_take_runs_the_native_cavity():
    el = _cavity()
    design = _native(el, _ref())
    surr = _surrogate(el, design, train_frequency_mhz=F_NEW)
    el.voltage_rel = 0.1                                     # as an error study leaves it
    registry.clear()
    registry.register(surr)
    try:
        with_surr = EnvelopeSolver(_lattice(el), _ref(), dict(INIT), current=0.0).run()
    finally:
        registry.clear()
    native = EnvelopeSolver(_lattice(copy.deepcopy(el)), _ref(), dict(INIT), current=0.0).run()
    np.testing.assert_allclose(with_surr.sigma_x, native.sigma_x, rtol=1e-12)
    np.testing.assert_allclose(with_surr.sigma_phi, native.sigma_phi, rtol=1e-12)
    assert surr.nn_calls == 0


def test_the_torch_arm_checks_the_scope_at_the_folded_input():
    """An NCELLS P=0 takes the RF shift into theta: 5 deg moves it out of a narrow sweep."""
    surr = _surrogate(_ncells(p_flag=0), np.eye(6), train_frequency_mhz=F_NC, train_phi_s_deg=PHI_T,
                      params=("theta_s_deg",), lo=(-35.0,), hi=(-28.0,), means=(-30.0,))
    element_matrix_torch(surr, _kin(_ref(PHI_T, f=F_NC)))
    with pytest.raises(ValueError, match="outside its trained scope"):
        element_matrix_torch(surr, RefKinematics.from_reference(_ref(PHI_T, shift=5.0, f=F_NC)))


def test_a_rephased_seed_shifts_theta_by_where_the_clock_correction_puts_it():
    """SET_BEAM_PHASE_ERROR with Dp = 0 rephases a seed back to the error-free arrival; the
    native map is then the error-free one, and so is the surrogate's input (it was shifted
    by the seed's raw phase)."""
    el = _ncells(p_flag=1)
    surr = _surrogate(el, np.eye(6), train_frequency_mhz=F_NC, train_phi_s_deg=PHI_T,
                      params=("theta_s_deg",), lo=(-180.0,), hi=(180.0,), means=(-30.0,))
    seed = _ref(PHI_T + 15.0, rephase=-15.0, f=F_NC)
    assert surr._make_input(seed)[3] == pytest.approx(-30.0, abs=1e-12)
    same(_native(el, seed), _native(el, _ref(PHI_T, f=F_NC)))


def test_a_pinned_calibration_is_refused():
    """In train mode, or under a frozen reliability pin, a p_flag 1 cavity keeps a calibration
    taken once; the weights, sampled with a fresh calibration at each input, hold another map
    for a moved input, and for another entrance energy without any error."""
    el = _cavity()
    design = _native(el, _ref())
    el.sync_phase_pin = _probe_pin(el)
    differ(_native(_with(el, voltage_rel=0.3), _ref()), _native(_with(_cavity(), scale=1.3), _ref()))
    slower = _ref()
    slower.w_kin = 4.5
    differ(_native(el, slower), _native(_cavity(), slower))
    for moved in (_with(el, voltage_rel=0.3), el):
        surr = _surrogate(moved, design, train_frequency_mhz=F_NEW, **SWEPT)
        with pytest.raises(OutOfScopeError, match="pinned"):
            surr.fitted_matrix(_ref())


def _probe_pin(el):
    """The design pass's calibration, as train mode pins it."""
    e = copy.deepcopy(el)
    e.fitted_matrix(_ref())
    return e._sync_offset_deg
