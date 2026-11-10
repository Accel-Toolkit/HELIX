"""A surrogate placed in the lattice runs as the element it stands for.

The compare flows put a ``SurrogateFieldMap`` in the lattice in place of its element.  It
did not carry its element's frequency, so the envelope made no jump to a cavity's own
clock there, and the reference walked the map's full steps instead of the tracker's
schedule, which the native element follows.
"""
import copy

import numpy as np
import pytest

from linac_gen.elements.drift import Drift
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking.envelope import EnvelopeSolver
from tests.surrogates.test_frequency_jump import F_NEW, F_OLD, _lattice, _map
from tests.surrogates.test_surrogate_provenance import INIT, _at, _surrogate


def _ncells():
    from linac_gen.elements.ncells import NCells
    return NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                  aperture_mm=15.0, p_flag=0, frequency_mhz=F_NEW)


def _cavity():
    return _map(Channel.RF_E, F_NEW, p_flag=1)


def _pinned(el):
    """A surrogate pinned to the native map on the cavity's clock, at the beam it will meet."""
    from linac_gen.tracking.matrix_tracking import reference_at
    at = reference_at(_lattice(Drift("D0", 100.0), copy.deepcopy(el)), _at(F_OLD), 1)
    own = at.copy()
    own.switch_clock(F_NEW)
    return _surrogate(el, copy.deepcopy(el).fitted_matrix(own.copy()), train_frequency_mhz=F_NEW,
                      train_phi_s_deg=own.phi_s)


def _run(el, current):
    lat = _lattice(Drift("D0", 100.0), el, Drift("D1", 100.0))
    return EnvelopeSolver(lat, _at(F_OLD), dict(INIT), current=current).run()


@pytest.mark.parametrize("current", [1e-9, 5.0])
@pytest.mark.parametrize("make", [_ncells, _cavity], ids=["ncells", "field_map"])
def test_a_substituted_cavity_at_another_clock_runs_as_its_element(make, current):
    native = _run(make(), current)
    surr = _run(_pinned(make()), current)
    for k in ("sigma_x", "sigma_y", "sigma_phi", "ref_w_kin", "ref_frequency"):
        np.testing.assert_allclose(np.asarray(getattr(surr, k)), np.asarray(getattr(native, k)),
                                   rtol=1e-9, atol=1e-12, err_msg=k)


def test_the_reference_walks_the_trackers_schedule_through_a_substituted_map():
    """The energy at the exit is the native element's to the last bit (the map's full steps
    gave another)."""
    native = _run(_cavity(), 0.0)
    surr = _run(_pinned(_cavity()), 0.0)
    assert np.asarray(surr.ref_w_kin)[-1] == np.asarray(native.ref_w_kin)[-1]
    assert np.asarray(surr.ref_phi_s)[-1] == np.asarray(native.ref_phi_s)[-1]


def test_a_substituted_magnet_keeps_the_clock():
    mag = _map(Channel.STAT_B, F_NEW)
    surr = _surrogate(mag, copy.deepcopy(mag).fitted_matrix(_at(F_OLD)), train_frequency_mhz=F_OLD)
    res = _run(surr, 0.0)
    assert np.all(np.asarray(res.ref_frequency) == F_OLD)
    assert surr.frequency == F_NEW and surr.effective_frequency == F_NEW
    assert not surr._has_electric_channel()


def test_the_walk_is_the_native_elements_with_the_fast_path_on():
    """With the multiparticle fast path engaged the surrogate's own steps advance the
    reference analytically, the same steps as the element's: the walk ends where the
    element's does."""
    from linac_gen.surrogates import registry

    def on_the_clock():                        # the fast path hands a jump to the element
        return _map(Channel.RF_E, F_OLD, p_flag=1)

    native = _run(on_the_clock(), 0.0)
    cav = on_the_clock()
    pinned = _surrogate(cav, copy.deepcopy(cav).fitted_matrix(_at(F_OLD)), train_frequency_mhz=F_OLD)
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    try:
        surr = _run(pinned, 0.0)
    finally:
        registry.set_mp_enabled(False)
        registry.set_fast_path_enabled(False)
    assert np.asarray(surr.ref_w_kin)[-1] == np.asarray(native.ref_w_kin)[-1]


def test_a_substituted_map_is_classified_and_stepped_as_its_element():
    from linac_gen.core.step_config import StepConfig
    from linac_gen.tracking.tracker import (
        _is_magnetic_only_fieldmap,
        _is_rf_bunching_element,
        _is_time_varying_rf_buncher,
        field_map_step_counts,
    )
    cav = _cavity()
    lat = _lattice(cav)
    lat.step_config = StepConfig(integration_steps_per_metre=20.0, sc_steps_per_metre=10.0)
    surr = _pinned(_cavity())
    assert field_map_step_counts(lat, surr) == field_map_step_counts(lat, cav)     # the map's floor
    assert _is_rf_bunching_element(surr) and _is_time_varying_rf_buncher(surr)
    mag = _map(Channel.STAT_B, F_NEW)
    assert _is_magnetic_only_fieldmap(mag)
    assert _is_magnetic_only_fieldmap(_surrogate(mag, np.eye(6), train_frequency_mhz=F_OLD))


def test_the_backtrackers_replay_walks_a_substituted_map_as_its_element():
    from linac_gen.tracking.backtrack import build_replay_table
    native = build_replay_table(_lattice(Drift("D0", 100.0), _cavity(), Drift("D1", 100.0)), _at(F_OLD),
                                end=2)
    surr = build_replay_table(_lattice(Drift("D0", 100.0), _pinned(_cavity()), Drift("D1", 100.0)),
                              _at(F_OLD), end=2)
    for k in ("w_kin", "phi_s", "frequency", "s"):
        assert [getattr(r, k) for r in surr] == [getattr(r, k) for r in native], k


def test_a_substituted_map_keeps_its_elements_aperture_map_and_compensation_table():
    """The multiparticle tracker reads a field map's aperture map (.ouv) and space-charge
    compensation table (.scc) from its field data, which a surrogate placed in the lattice
    does not have of its own: it checked the scalar aperture and applied full space charge."""
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.simulation import Simulation
    from tests.surrogates.test_frequency_jump import _beam

    def cav():
        el = _map(Channel.RF_E, F_OLD, p_flag=1)
        z = np.array([0.0, 200.0])
        el.field_data.pipe_radius_profile = (z, np.array([1.2, 1.2]), np.array([1.2, 1.2]))
        el.field_data.scc_profile = np.array([[0.0, 1.0], [200.0, 1.0]])
        el.field_data.scc_scale, el.field_data.scc_mode = 0.5, 0
        return el

    def run(el):
        lat = _lattice(Drift("D0", 50.0, aperture=30.0), el, Drift("D1", 50.0, aperture=30.0))
        beam = _beam(_at(F_OLD))
        beam.current = 5.0
        sc = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0, use_gpu="cpu", grid_mode="adaptive")
        Simulation(lat, beam, space_charge=sc).run()
        return beam

    native = run(cav())
    c = cav()
    surr = run(_surrogate(c, copy.deepcopy(c).fitted_matrix(_at(F_OLD)), train_frequency_mhz=F_OLD))
    assert native.n_alive < native.n_particles                 # the map cuts
    np.testing.assert_array_equal(surr.alive_mask, native.alive_mask)
    np.testing.assert_array_equal(surr.particles[surr.alive_mask], native.particles[native.alive_mask])
