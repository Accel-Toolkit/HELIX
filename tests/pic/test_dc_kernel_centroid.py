"""DC space-charge kernels: self-field is centroid-centred (claim 4) and
the TSC deposit convention is pinned (claim 3).  2026-07-25 review round."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.pic import pic_solver as ps
from linac_gen.pic.charge_deposition import deposit_tsc


def _dc_beam(dx_mm: float):
    cfg = BeamConfig(species="H-", energy=0.030, frequency=162.5,
                     current=5.0, n_particles=20000, continuous=True,
                     emit_nx=0.2, alpha_x=0.0, beta_x=0.5,
                     emit_ny=0.2, alpha_y=0.0, beta_y=0.5,
                     emit_z=0.06, alpha_z=0.0, beta_z=500.0)
    b = create_beam(cfg, seed=3)
    b.particles[:, 0] += dx_mm
    return b


@pytest.mark.parametrize("kick_name", ["kick_continuous_2d",
                                       "kick_continuous_2d_gauss",
                                       "kick_continuous_2d_pic"])
def test_displaced_dc_beam_has_no_coherent_self_kick(kick_name):
    """A beam's own field cannot deflect its centroid (momentum
    conservation).  Before 2026-07-25 the two ANALYTIC kernels evaluated
    the field about the axis, giving a 5 mm-displaced beam a coherent
    self-kick 1.4-1.7x the incoherent rms."""
    kick = getattr(ps, kick_name)
    b = _dc_beam(5.0)
    xp0 = b.particles[:, 1].copy()
    kick(b, 1.0)
    d = b.particles[:, 1] - xp0
    coherent = abs(d.mean())
    rms = max(d.std(), 1e-30)
    assert coherent / rms < 0.02, \
        f"{kick_name}: coherent self-kick {coherent:.3e} = " \
        f"{coherent / rms:.2f}x rms (momentum not conserved)"


def test_displaced_equals_centred_kick_field():
    """Stronger invariance for the linear kernel: the kick field of a
    displaced beam is the centred beam's field translated with it."""
    b0, b5 = _dc_beam(0.0), _dc_beam(5.0)
    xp0 = b0.particles[:, 1].copy(); xp5 = b5.particles[:, 1].copy()
    ps.kick_continuous_2d(b0, 1.0); ps.kick_continuous_2d(b5, 1.0)
    np.testing.assert_allclose(b5.particles[:, 1] - xp5,
                               b0.particles[:, 1] - xp0,
                               rtol=0, atol=1e-12)


def test_tsc_deposit_convention_pinned():
    """Characterisation test for the DOCUMENTED TSC convention: deposit
    centroid sits exactly dx/2 low at every in-cell offset (cancels with
    the matching gather; density maps only).  A deliberate future fix of
    deposit+gather together should update this pin in the same commit."""
    gmin = np.array([0.0, 0.0, 0.0]); gmax = np.array([16.0] * 3)
    ng = np.array([17, 17, 17])                 # dx = 1.0, nodes at ints
    nodes = np.arange(17.0)
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        rho = deposit_tsc(np.array([[8.0 + f, 8.0, 8.0]]),
                          np.array([1.0]), gmin, gmax, ng)
        cx = (rho.sum(axis=(1, 2)) * nodes).sum() / rho.sum()
        assert cx - (8.0 + f) == pytest.approx(-0.5, abs=1e-12)
