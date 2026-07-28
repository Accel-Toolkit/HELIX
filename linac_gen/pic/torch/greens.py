"""Torch (autograd) Green's functions for the differentiable PIC Poisson solver.

A direct PyTorch translation of the numpy builders in
:mod:`linac_gen.pic.poisson_solver` — the Integrated Green Function (Qiang
et al., PRSTAB 9, 044204, 2006) and the legacy point-source kernel. The
math is identical; only the array module changes, so the torch result
matches the numpy result to floating-point round-off.

The builders are differentiable with respect to the grid spacing ``dx`` —
needed because the differentiable space-charge kick sizes its grid from the
(differentiable) beam distribution.
"""
from __future__ import annotations

import torch

from linac_gen.core.constants import EPSILON_0, PI

__all__ = [
    "torch_antideriv_igf",
    "build_integrated_greens_torch",
    "build_point_greens_torch",
]


def torch_antideriv_igf(x: torch.Tensor, y: torch.Tensor,
                        z: torch.Tensor) -> torch.Tensor:
    """Closed-form antiderivative ``F`` with ``d3 F / (dx dy dz) = 1/r``.

    Torch mirror of :func:`linac_gen.pic.poisson_solver._antideriv_igf` — the
    Cheetah atan + asinh formulation, odd under ``(x,y,z) -> (-x,-y,-z)`` so
    the Hockney kernel stays reflection-symmetric.
    """
    r = torch.sqrt(x * x + y * y + z * z)
    # Corners sit at +/- half-integer multiples of (dx, dy, dz), so x, y, z
    # are never simultaneously zero; ``eps`` is only a defensive guard and
    # matches the numpy implementation bit-for-bit.
    eps = 1e-300
    return (
        -0.5 * z * z * torch.arctan(x * y / (z * r + eps))
        - 0.5 * y * y * torch.arctan(x * z / (y * r + eps))
        - 0.5 * x * x * torch.arctan(y * z / (x * r + eps))
        + y * z * torch.arcsinh(x / (torch.sqrt(y * y + z * z) + eps))
        + x * z * torch.arcsinh(y / (torch.sqrt(x * x + z * z) + eps))
        + x * y * torch.arcsinh(z / (torch.sqrt(x * x + y * y) + eps))
    )


def _wrapped_index(n: int, dtype, device) -> torch.Tensor:
    """Doubled-grid index coordinate: 0..n-1, then -n..-1 (Hockney wrap)."""
    idx = torch.arange(2 * n, dtype=dtype, device=device)
    return torch.where(idx < n, idx, idx - 2 * n)


def build_integrated_greens_torch(n_grid, dx, *,
                                  dtype: torch.dtype = torch.float64,
                                  device=None) -> torch.Tensor:
    """Cell-integrated Green's function on the doubled (2n) grid.

    Torch mirror of ``PoissonSolverFFT._build_integrated_greens``. ``n_grid``
    is the (nx, ny, nz) physical grid size; ``dx`` is the (3,) cell spacing
    (may carry ``requires_grad``). Returns a ``(2nx, 2ny, 2nz)`` tensor.
    """
    nx, ny, nz = (int(n) for n in n_grid)
    dx = torch.as_tensor(dx, dtype=dtype, device=device)
    dxx, dyy, dzz = dx[0], dx[1], dx[2]

    ii = _wrapped_index(nx, dtype, device)
    jj = _wrapped_index(ny, dtype, device)
    kk = _wrapped_index(nz, dtype, device)

    # Cell-corner positions — low and high in each axis.
    xL, xH = (ii - 0.5) * dxx, (ii + 0.5) * dxx
    yL, yH = (jj - 0.5) * dyy, (jj + 0.5) * dyy
    zL, zH = (kk - 0.5) * dzz, (kk + 0.5) * dzz

    XL, YL, ZL = torch.meshgrid(xL, yL, zL, indexing="ij")
    XH, YH, ZH = torch.meshgrid(xH, yH, zH, indexing="ij")

    # 8-point inclusion-exclusion of the antiderivative over each cell.
    G = (
        torch_antideriv_igf(XH, YH, ZH)
        - torch_antideriv_igf(XL, YH, ZH)
        - torch_antideriv_igf(XH, YL, ZH)
        - torch_antideriv_igf(XH, YH, ZL)
        + torch_antideriv_igf(XL, YL, ZH)
        + torch_antideriv_igf(XL, YH, ZL)
        + torch_antideriv_igf(XH, YL, ZL)
        - torch_antideriv_igf(XL, YL, ZL)
    ) / (dxx * dyy * dzz)
    return G * (1.0 / (4.0 * PI * EPSILON_0))


def build_point_greens_torch(n_grid, dx, *,
                             dtype: torch.dtype = torch.float64,
                             device=None) -> torch.Tensor:
    """Point-source ``1/(4 pi eps0 r)`` Green's function with the
    self-potential regulariser at ``r = 0``.

    Torch mirror of ``PoissonSolverFFT._build_point_greens``.
    """
    nx, ny, nz = (int(n) for n in n_grid)
    dx = torch.as_tensor(dx, dtype=dtype, device=device)
    dxx, dyy, dzz = dx[0], dx[1], dx[2]

    ii = _wrapped_index(nx, dtype, device)
    jj = _wrapped_index(ny, dtype, device)
    kk = _wrapped_index(nz, dtype, device)
    II, JJ, KK = torch.meshgrid(ii, jj, kk, indexing="ij")
    # The origin cell has arg == 0; sqrt(0) has an infinite backward and
    # 0·inf = NaN poisons every gradient that flows through the grid
    # spacing (adaptive-grid case: dxx/dyy/dzz are differentiable).
    # Masking AFTER the sqrt cannot save it — guard the argument itself:
    # the origin cell's r is replaced by the where() below anyway, so the
    # forward values are bit-identical for every cell (2026-07-25 review:
    # point-kernel gradients were NaN while igf's were finite).
    arg = (II * dxx) ** 2 + (JJ * dyy) ** 2 + (KK * dzz) ** 2
    origin = arg > 0
    r = torch.sqrt(torch.where(origin, arg, torch.ones_like(arg)))

    coul = 1.0 / (4.0 * PI * EPSILON_0)
    self_pot = coul / (0.5 * torch.sqrt(dxx * dxx + dyy * dyy + dzz * dzz))
    # Select on the PRE-guard mask (``origin``): after the sqrt guard the
    # origin cell's r is 1.0, so testing ``r > 0`` here would hand it the
    # coul/r branch instead of the self-potential.
    return torch.where(origin, coul / r, self_pot * torch.ones_like(r))
