"""M7 — parity + autograd tests for the torch TSC (Triangular Shape Cloud)
deposit/gather kernels, and the SC kick with ``kernel='tsc'``.
"""
import copy

import numpy as np
import torch

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.pic.charge_deposition import deposit_tsc
from linac_gen.pic.field_interpolation import interpolate_tsc
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.pic.torch.deposition import deposit_tsc_torch
from linac_gen.pic.torch.interpolation import interpolate_tsc_torch
from linac_gen.pic.torch.solver import TorchPicSolver

GMIN = np.array([-3.0, -2.0, -5.0])
GMAX = np.array([3.0, 2.0, 5.0])


def _setup(n=16, npart=200, seed=0):
    """Random positions well inside the grid (TSC needs a +/-1 margin)."""
    rng = np.random.default_rng(seed)
    ng = np.array([n, n, n], dtype=np.int64)
    pos = rng.uniform(GMIN * 0.6, GMAX * 0.6, size=(npart, 3))
    q = rng.uniform(0.5, 1.5, size=npart)
    return ng, pos, q


def test_tsc_deposit_parity():
    """deposit_tsc_torch reproduces the numpy TSC deposit."""
    ng, pos, q = _setup(seed=0)
    rho_n = deposit_tsc(pos, q, GMIN, GMAX, ng)
    rho_t = deposit_tsc_torch(pos, q, GMIN, GMAX, ng).numpy()
    np.testing.assert_allclose(rho_t, rho_n, rtol=1e-12,
                               atol=1e-12 * np.abs(rho_n).max())


def test_tsc_interpolate_parity():
    """interpolate_tsc_torch reproduces the numpy TSC gather."""
    ng, pos, q = _setup(seed=1)
    n = int(ng[0])
    rng = np.random.default_rng(2)
    fx, fy, fz = (rng.random((n, n, n)) for _ in range(3))
    e_n = interpolate_tsc(fx, fy, fz, pos, GMIN, GMAX, ng)
    e_t = interpolate_tsc_torch(fx, fy, fz, pos, GMIN, GMAX, ng).numpy()
    np.testing.assert_allclose(e_t, e_n, rtol=1e-12,
                               atol=1e-12 * np.abs(e_n).max())


def test_tsc_charge_conservation():
    """TSC's 27 weights sum to 1 — total deposited charge is preserved."""
    ng, pos, q = _setup(seed=3)
    rho = deposit_tsc_torch(pos, q, GMIN, GMAX, ng)
    dx = (GMAX - GMIN) / (ng - 1)
    cell_vol = float(dx[0] * dx[1] * dx[2])
    np.testing.assert_allclose(float(rho.sum()) * cell_vol,
                               float(q.sum()), rtol=1e-12)


def test_tsc_deposit_differentiable():
    """Gradients flow through the TSC deposit w.r.t. particle positions."""
    ng, pos, q = _setup(n=12, npart=40, seed=4)
    p = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    qt = torch.tensor(q, dtype=torch.float64)
    deposit_tsc_torch(p, qt, GMIN, GMAX, ng).pow(2).sum().backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0


def _make_beam(n=200, seed=42):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=10.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, 1.0, n)
    beam.particles[:, 1] = rng.normal(0.0, 0.1, n)
    beam.particles[:, 2] = rng.normal(0.0, 1.0, n)
    beam.particles[:, 3] = rng.normal(0.0, 0.1, n)
    beam.particles[:, 4] = rng.normal(0.0, 5.0, n)
    beam.particles[:, 5] = rng.normal(0.0, 0.01, n)
    return beam


def test_sc_kick_tsc_parity():
    """The torch SC kick with kernel='tsc' reproduces the numpy PicSolver."""
    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0,
                            use_gpu="cpu", kernel="tsc",
                            grid_mode="adaptive")
    beam_np = _make_beam()
    beam_t = copy.deepcopy(beam_np)
    before = beam_np.particles.copy()
    PicSolver(cfg).kick(beam_np, ds=50.0)
    TorchPicSolver(cfg).kick(beam_t, ds=50.0)
    d_np = beam_np.particles - before
    d_t = beam_t.particles - before
    for col in (1, 3, 5):
        scale = max(float(np.abs(d_np[:, col]).max()), 1e-30)
        np.testing.assert_allclose(d_t[:, col], d_np[:, col],
                                   rtol=1e-9, atol=1e-9 * scale)
