"""M3 — the key validation: the torch PIC space-charge kick must reproduce
the numpy ``PicSolver.kick`` (both FP64 CPU) to ~1e-9, across a spread of
beam conditions, and be autograd-differentiable.
"""
import copy

import numpy as np
import pytest
import torch

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.pic.torch.sc_kick import torch_pic_sc_kick
from linac_gen.pic.torch.solver import TorchPicSolver


def _make_beam(n=200, energy=3.0, current=10.0, species=PROTON,
               sigma_x=1.0, sigma_y=1.0, sigma_dphi=5.0, sigma_dw=0.01,
               offset_x=0.0, frequency=352.21, seed=42):
    ref = ReferenceParticle(species=species, w_kin=energy, frequency=frequency)
    beam = Beam(ref=ref, n_particles=n, current=current)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(offset_x, sigma_x, n)
    beam.particles[:, 1] = rng.normal(0.0, 0.1, n)
    beam.particles[:, 2] = rng.normal(0.0, sigma_y, n)
    beam.particles[:, 3] = rng.normal(0.0, 0.1, n)
    beam.particles[:, 4] = rng.normal(0.0, sigma_dphi, n)
    beam.particles[:, 5] = rng.normal(0.0, sigma_dw, n)
    return beam


# FP64 CPU on the numpy side too, so the parity comparison is meaningful
# (the "auto" backend would pick FP32 MPS on this machine).
def _cfg():
    # grid_mode="adaptive": the torch backend is adaptive-only and warns
    # on fixed-grid configs (AdaptiveOnlyBackendWarning, honesty round).
    return SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0,
                             use_gpu="cpu", grid_mode="adaptive")


_CASES = {
    "round":      dict(sigma_x=1.0, sigma_y=1.0),
    "asymmetric": dict(sigma_x=1.5, sigma_y=0.6),
    "offset":     dict(offset_x=0.8),
    "h_minus":    dict(species=H_MINUS, energy=2.5),
    "high_gamma": dict(energy=469.65),
    "low_gamma":  dict(energy=1.5),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_torch_vs_numpy_kick_parity(name):
    """TorchPicSolver.kick reproduces PicSolver.kick to ~1e-9."""
    cfg = _cfg()
    beam_np = _make_beam(**_CASES[name])
    beam_t = copy.deepcopy(beam_np)
    before = beam_np.particles.copy()

    PicSolver(cfg).kick(beam_np, ds=50.0)
    TorchPicSolver(cfg).kick(beam_t, ds=50.0)

    d_np = beam_np.particles - before
    d_t = beam_t.particles - before
    # Columns 0, 2, 4 are untouched; 1, 3, 5 carry the kick.
    for col in (1, 3, 5):
        scale = max(float(np.abs(d_np[:, col]).max()), 1e-30)
        np.testing.assert_allclose(d_t[:, col], d_np[:, col],
                                   rtol=1e-9, atol=1e-9 * scale)
    np.testing.assert_array_equal(beam_t.particles[:, [0, 2, 4]],
                                  before[:, [0, 2, 4]])


def test_ds_scaling():
    """The kick scales linearly with the step length."""
    cfg = _cfg()
    b1 = _make_beam()
    b2 = copy.deepcopy(b1)
    before = b1.particles.copy()
    TorchPicSolver(cfg).kick(b1, ds=10.0)
    TorchPicSolver(cfg).kick(b2, ds=20.0)
    d1 = b1.particles[:, 1] - before[:, 1]
    d2 = b2.particles[:, 1] - before[:, 1]
    np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-9,
                               atol=1e-9 * np.abs(d1).max())


def test_zero_current_noop():
    """Zero current is a no-op."""
    cfg = _cfg()
    beam = _make_beam(current=0.0)
    before = beam.particles.copy()
    TorchPicSolver(cfg).kick(beam, ds=50.0)
    np.testing.assert_array_equal(beam.particles, before)


def test_kick_is_differentiable():
    """Gradients flow through the kick w.r.t. the particle coordinates."""
    cfg = _cfg()
    beam = _make_beam()
    particles = torch.tensor(beam.particles, dtype=torch.float64,
                             requires_grad=True)
    macro_charge = ((beam.current * 1e-3) / (beam.bunch_frequency * 1e6)
                    / beam.n_particles)
    out = torch_pic_sc_kick(
        particles, cfg, ds_mm=50.0,
        gamma=float(beam.ref.gamma), beta=float(beam.ref.beta),
        mass_mev=float(beam.ref.species.mass),
        wavelength_mm=float(beam.ref.wavelength),
        macro_charge=macro_charge,
    )
    out.pow(2).sum().backward()
    assert particles.grad is not None
    assert torch.isfinite(particles.grad).all()
    assert particles.grad.abs().sum() > 0


def test_simulation_runs_with_torch_backend():
    """Simulation wires TorchPicSolver when sc_backend='torch'."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.elements.drift import Drift

    lattice = Lattice()
    lattice.add(Drift(name="d1", length=50.0))
    beam = _make_beam(n=120, current=15.0)
    sc = SpaceChargeConfig(nx=12, ny=12, nz=12, grid_extent=4.0,
                           use_gpu="cpu", sc_backend="torch",
                           grid_mode="adaptive")
    results = Simulation(lattice, beam, space_charge=sc).run()
    assert results is not None
