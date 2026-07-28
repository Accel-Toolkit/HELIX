"""M1 — parity + autograd tests for the torch IGF Poisson solver.

The torch solver must reproduce the numpy ``PoissonSolverFFT`` to
floating-point round-off (both FP64 CPU), and gradients must flow through
the solve.
"""
import numpy as np
import torch

from linac_gen.pic.poisson_solver import PoissonSolverFFT
from linac_gen.pic.torch.greens import build_integrated_greens_torch
from linac_gen.pic.torch.poisson import TorchPoissonSolverFFT


def _grid(n):
    gmin = np.array([-3.0, -2.0, -5.0])
    gmax = np.array([3.0, 2.0, 5.0])
    ng = np.array([n, n, n], dtype=np.int64)
    return gmin, gmax, ng


def test_igf_green_tensor_parity():
    """Torch IGF Green tensor matches the numpy builder."""
    gmin, gmax, ng = _grid(16)
    nps = PoissonSolverFFT(gmin, gmax, ng, use_gpu="cpu", green_kind="igf")
    g_np = nps._build_integrated_greens()
    dx = (gmax - gmin) / (ng - 1)
    g_t = build_integrated_greens_torch((16, 16, 16), dx).numpy()
    np.testing.assert_allclose(g_t, g_np, rtol=1e-10,
                               atol=1e-10 * np.abs(g_np).max())


def _assert_solve_parity(green_kind, n, seed):
    gmin, gmax, ng = _grid(n)
    rho = np.random.default_rng(seed).random((n, n, n))
    nps = PoissonSolverFFT(gmin, gmax, ng, use_gpu="cpu",
                           green_kind=green_kind)
    ex_n, ey_n, ez_n = nps.solve(rho)
    tps = TorchPoissonSolverFFT(gmin, gmax, ng, green_kind=green_kind)
    ex_t, ey_t, ez_t = tps.solve(torch.as_tensor(rho, dtype=torch.float64))
    for got, want in ((ex_t, ex_n), (ey_t, ey_n), (ez_t, ez_n)):
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-9,
                                   atol=1e-9 * np.abs(want).max())


def test_solve_parity_igf():
    """Torch solve(rho) matches numpy for the IGF kernel."""
    _assert_solve_parity("igf", 24, seed=0)


def test_solve_parity_point():
    """Torch solve(rho) matches numpy for the point-source kernel."""
    _assert_solve_parity("point", 20, seed=1)


def test_autograd_through_solve():
    """Gradients flow through the solve w.r.t. the charge density."""
    gmin, gmax, ng = _grid(16)
    rho = torch.rand(16, 16, 16, dtype=torch.float64, requires_grad=True)
    tps = TorchPoissonSolverFFT(gmin, gmax, ng, green_kind="igf")
    ex, ey, ez = tps.solve(rho)
    (ex.pow(2).sum() + ey.pow(2).sum() + ez.pow(2).sum()).backward()
    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()
    assert rho.grad.abs().sum() > 0


def test_autograd_through_grid_spacing():
    """The Green's function is differentiable w.r.t. the grid spacing."""
    dx = torch.tensor([0.4, 0.3, 0.6], dtype=torch.float64, requires_grad=True)
    g = build_integrated_greens_torch((12, 12, 12), dx)
    g.sum().backward()
    assert dx.grad is not None
    assert torch.isfinite(dx.grad).all()
    assert dx.grad.abs().sum() > 0
