"""The torch path takes the clock jump of a surrogated cavity at another frequency.

The numpy surrogate answers at another clock as the native element does: the jump to its
own clock on entry, then the map (M D).  The torch arm gave the map alone, and the torch
walks went on at the old clock past it, so a line through such a cavity differed from the
numpy one and the gradient matcher refused it.
"""
import copy

import numpy as np
import pytest
import torch

from linac_gen.core.config import BeamConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.tracewin_geom import Channel
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
from linac_gen.tracking.torch_step_tracker import track_beam_torch_stepwise
from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch, element_matrix_torch
from tests.surrogates.test_frequency_jump import F_NEW, F_OLD, _lattice, _map
from tests.surrogates.test_surrogate_provenance import _at, _kin, _surrogate


def _off_cavity():
    """A cavity at 325 MHz switched off (ke = 0): no gain, so the gradient matcher takes it,
    and no running-phase dependence, so the torch arm answers at any phase."""
    el = _map(Channel.RF_E, F_NEW, p_flag=0)
    el.ke = 0.0
    return el


def _surr(el):
    """Pinned to the native map on its own clock."""
    return _surrogate(el, copy.deepcopy(el).fitted_matrix(_at(F_NEW)), train_frequency_mhz=F_NEW)


def test_the_torch_arm_makes_the_jump():
    el = _off_cavity()
    surr = _surr(el)
    ref = _at(F_OLD)
    got = element_matrix_torch(surr, _kin(ref)).detach().numpy()
    np.testing.assert_allclose(got, surr.fitted_matrix(ref.copy()), rtol=0, atol=1e-12)
    np.testing.assert_allclose(got, copy.deepcopy(el).fitted_matrix(ref.copy()), rtol=0, atol=1e-12)
    # on its own clock nothing moves, bit for bit
    np.testing.assert_array_equal(element_matrix_torch(surr, _kin(_at(F_NEW))).detach().numpy(),
                                  surr.fitted_matrix(_at(F_NEW)))


def _line(cavity):
    return _lattice(Drift("D0", 100.0), cavity, Drift("D1", 150.0))


def test_the_torch_line_goes_on_at_the_cavitys_clock():
    el = _off_cavity()
    ref = _at(F_OLD)
    torch_line = compute_transfer_matrix_torch(_line(_surr(el)), ref.copy()).detach().numpy()
    np.testing.assert_allclose(torch_line, compute_transfer_matrix(_line(_surr(el)), ref.copy()),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(torch_line, compute_transfer_matrix(_line(copy.deepcopy(el)), ref.copy()),
                               rtol=1e-9, atol=1e-9)


def test_the_stepwise_torch_tracker_too():
    el = _off_cavity()
    ref = _at(F_OLD)
    X = np.random.default_rng(3).normal(size=(16, 6)) * [0.5, 0.5, 0.5, 0.5, 2.0, 1e-4]
    got = track_beam_torch_stepwise(_line(_surr(el)), ref.copy(), torch.as_tensor(X)).detach().numpy()
    want = (compute_transfer_matrix(_line(copy.deepcopy(el)), ref.copy()) @ X.T).T
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_the_gradient_matcher_agrees_with_least_squares():
    from linac_gen.matching import match

    def build():
        lat = _lattice(Drift("D1", length=200.0, aperture=30.0))
        lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0, vmin=-30, vmax=30,
                       start_step=0.5))
        lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0, aperture=30.0))
        lat.add(Drift("D2", length=100.0, aperture=30.0))
        lat.add(_surr(_off_cavity()))
        lat.add(Drift("D3", length=200.0, aperture=30.0))
        lat.add(SetSize("CSET", k=1.0, x_mm=3.0, y_mm=0.0, phi_or_z=2.0))
        return lat

    cfg = BeamConfig(species="proton", energy=5.0, frequency=F_OLD, n_particles=10,
                     distribution="waterbag", emit_nx=0.25, alpha_x=0.0, beta_x=2.0, emit_ny=0.25,
                     alpha_y=0.0, beta_y=2.0, emit_z=0.3, alpha_z=0.0, beta_z=10.0)
    grad = match(build(), cfg, algorithm="gradient", max_iter=60)
    lsq = match(build(), cfg, algorithm="least_squares", max_iter=200)
    assert grad.x_final[0] == pytest.approx(lsq.x_final[0], rel=1e-4)
    assert grad.cost == pytest.approx(lsq.cost, rel=1e-6, abs=1e-12)


def test_the_exit_half_kick_of_space_charge_is_on_the_cavitys_clock():
    """With space charge the stepwise tracker kicks half before and half after the cavity's
    map; past the map the beam is on the cavity's clock.  A FREQ card in front of the
    cavity, where it makes no jump, gives the same beam."""
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.elements.lattice_commands import Freq
    el = _off_cavity()
    X = np.random.default_rng(4).normal(size=(2000, 6)) * [1.0, 1.0, 1.0, 1.0, 3.0, 1e-3]
    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0, use_gpu="cpu", grid_mode="adaptive")
    q = (5e-3 / (F_OLD * 1e6)) / len(X)
    jump = track_beam_torch_stepwise(_line(_surr(el)), _at(F_OLD), torch.as_tensor(X), sc_cfg=cfg,
                                     macro_charge=q).detach().numpy()
    card = _lattice(Drift("D0", 100.0), Freq("F", frequency_mhz=F_NEW), _surr(el), Drift("D1", 150.0))
    want = track_beam_torch_stepwise(card, _at(F_OLD), torch.as_tensor(X), sc_cfg=cfg,
                                     macro_charge=q).detach().numpy()
    np.testing.assert_allclose(jump, want, rtol=1e-9, atol=1e-12 * np.abs(want).max())


def test_the_torch_arm_reads_the_rf_shift_on_the_cavitys_clock():
    """An NCELLS P=0 takes the RF shift into theta, counted on its own clock as the numpy
    path counts it after the jump."""
    from linac_gen.elements.ncells import NCells
    from linac_gen.surrogates.base import SurrogateFieldMap
    from linac_gen.tracking.torch_matrices import RefKinematics
    el = NCells("NC", mode=1, n_cells=6, beta_g=0.2, eot_v_per_m=2.0e6, theta_s_deg=-30.0,
                aperture_mm=15.0, p_flag=0, frequency_mhz=F_NEW)
    base = _surrogate(el, np.eye(6), train_frequency_mhz=F_NEW, params=("theta_s_deg",),
                      lo=(-180.0,), hi=(180.0,), means=(-30.0,))
    meta = base.metadata
    meta.output_norm = {"mean": np.eye(6).ravel().tolist(), "std": [0.01] * 36}   # moves with theta
    surr = SurrogateFieldMap(el, base.mlp, meta)
    ref = _at(F_OLD)
    ref.rf_shift = 4.0
    got = element_matrix_torch(surr, RefKinematics.from_reference(ref)).detach().numpy()
    np.testing.assert_allclose(got, surr.fitted_matrix(ref.copy()), rtol=0, atol=1e-12)
