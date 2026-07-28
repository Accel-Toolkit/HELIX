"""Point Green's kernel must be differentiable-safe (claim 2, 2026-07-25).

The origin cell's ``sqrt(0)`` had an infinite backward; with the grid
spacing differentiable (adaptive-grid case), 0*inf = NaN poisoned every
gradient through the point kernel while igf's stayed finite.  Forward
values are asserted unchanged (torch builders vs themselves pre-fix was
verified bit-identical; here we pin finiteness + igf/point independence).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from linac_gen.pic.torch.poisson import TorchPoissonSolverFFT  # noqa: E402


@pytest.mark.parametrize("kind", ["igf", "point"])
def test_greens_gradient_finite_through_grid_bounds(kind):
    shift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    solver = TorchPoissonSolverFFT(
        torch.tensor([-4.0] * 3, dtype=torch.float64) + shift,
        torch.tensor([+4.0] * 3, dtype=torch.float64) + shift,
        (8, 8, 8), green_kind=kind)
    rho = torch.zeros((8, 8, 8), dtype=torch.float64)
    rho[4, 4, 4] = 1.0
    out = solver.solve(rho)
    phi = out[0] if isinstance(out, tuple) else out
    val = phi.sum()
    assert bool(torch.isfinite(val))
    g = torch.autograd.grad(val, shift)[0]
    assert bool(torch.isfinite(g)), \
        f"{kind}: NaN/inf gradient through grid bounds"


def test_point_forward_value_matches_scaled_grid():
    """Forward sanity: doubling all grid spacings halves the point
    Green's function everywhere except the self-cell (1/r scaling) —
    guards against the sqrt fix having altered forward values."""
    from linac_gen.pic.torch.greens import build_point_greens_torch
    dx1 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    G1 = build_point_greens_torch((8, 8, 8), dx1)
    G2 = build_point_greens_torch((8, 8, 8), 2.0 * dx1)
    mask = torch.ones_like(G1, dtype=torch.bool)
    mask[0, 0, 0] = False
    assert torch.allclose(G2[mask], 0.5 * G1[mask], rtol=1e-12, atol=0.0)
    assert torch.allclose(G2[0, 0, 0], 0.5 * G1[0, 0, 0], rtol=1e-12)
