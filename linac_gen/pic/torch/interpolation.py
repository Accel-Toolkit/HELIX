"""Torch (autograd) field-interpolation kernel — Cloud-In-Cell gather.

Torch mirror of :func:`linac_gen.pic.field_interpolation.interpolate_cic`,
the reverse of :func:`linac_gen.pic.torch.deposition.deposit_cic_torch`. It
reuses the same CIC stencil, so the deposit/gather pair is adjoint — no
spurious self-force. Differentiable in the grid fields and the positions.
"""
from __future__ import annotations

import torch

from linac_gen.pic.torch.deposition import cic_cells, tsc_cells

__all__ = ["interpolate_cic_torch", "interpolate_tsc_torch"]


def interpolate_cic_torch(field_x, field_y, field_z, positions,
                          grid_min, grid_max, n_grid, *,
                          dtype: torch.dtype = torch.float64,
                          device=None) -> torch.Tensor:
    """Cloud-In-Cell gather of the three grid fields to particle positions.

    ``field_*`` are ``(nx, ny, nz)`` grids, ``positions`` is ``(N, 3)``.
    Returns ``(N, 3)`` interpolated ``[Fx, Fy, Fz]`` per particle.
    """
    positions = torch.as_tensor(positions, dtype=dtype, device=device)
    if positions.shape[0] == 0:
        return torch.zeros((0, 3), dtype=dtype, device=device)

    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)
    field_x = torch.as_tensor(field_x, dtype=dtype, device=device)
    field_y = torch.as_tensor(field_y, dtype=dtype, device=device)
    field_z = torch.as_tensor(field_z, dtype=dtype, device=device)

    cix, ciy, ciz, weights = cic_cells(positions, grid_min, grid_max, n_grid,
                                       dtype=dtype, device=device)
    ex = (weights * field_x[cix, ciy, ciz]).sum(dim=1)           # (N,)
    ey = (weights * field_y[cix, ciy, ciz]).sum(dim=1)
    ez = (weights * field_z[cix, ciy, ciz]).sum(dim=1)
    return torch.stack([ex, ey, ez], dim=1)                      # (N, 3)


def interpolate_tsc_torch(field_x, field_y, field_z, positions,
                          grid_min, grid_max, n_grid, *,
                          dtype: torch.dtype = torch.float64,
                          device=None) -> torch.Tensor:
    """Triangular Shape Cloud gather (27-cell) — the reverse of
    :func:`linac_gen.pic.torch.deposition.deposit_tsc_torch`.

    Same signature as :func:`interpolate_cic_torch`; shares the TSC stencil
    with the deposit so the pair stays adjoint.
    """
    positions = torch.as_tensor(positions, dtype=dtype, device=device)
    if positions.shape[0] == 0:
        return torch.zeros((0, 3), dtype=dtype, device=device)

    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)
    field_x = torch.as_tensor(field_x, dtype=dtype, device=device)
    field_y = torch.as_tensor(field_y, dtype=dtype, device=device)
    field_z = torch.as_tensor(field_z, dtype=dtype, device=device)

    cix, ciy, ciz, weights = tsc_cells(positions, grid_min, grid_max, n_grid,
                                       dtype=dtype, device=device)
    ex = (weights * field_x[cix, ciy, ciz]).sum(dim=1)
    ey = (weights * field_y[cix, ciy, ciz]).sum(dim=1)
    ez = (weights * field_z[cix, ciy, ciz]).sum(dim=1)
    return torch.stack([ex, ey, ez], dim=1)
