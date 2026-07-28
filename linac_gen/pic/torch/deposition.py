"""Torch (autograd) charge-deposition kernel — Cloud-In-Cell.

Torch mirror of :func:`linac_gen.pic.charge_deposition.deposit_cic`.
Differentiable with respect to particle positions (through the continuous
CIC weights) and per-particle charges. The integer cell index is detached:
gradient flows through the trilinear weights, not the cell selection — the
standard behaviour for a differentiable PIC deposit.
"""
from __future__ import annotations

import torch

__all__ = ["deposit_cic_torch", "cic_cells",
           "deposit_tsc_torch", "tsc_cells"]

# The 8 corner offsets of a cell, in the same order numpy iterates them.
_CIC_OFFSETS = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))

# The 27 cell offsets for the Triangular Shape Cloud stencil.
_TSC_OFFSETS = tuple((ox, oy, oz) for ox in (-1, 0, 1)
                     for oy in (-1, 0, 1) for oz in (-1, 0, 1))


def cic_cells(positions: torch.Tensor, grid_min: torch.Tensor,
              grid_max: torch.Tensor, n_grid, *,
              dtype: torch.dtype, device):
    """Shared CIC stencil for deposit and gather.

    Returns ``(cix, ciy, ciz, weights)`` — the first three are ``(N, 8)``
    long index tensors for the 8 cell corners, ``weights`` is the matching
    ``(N, 8)`` trilinear weight tensor (differentiable in ``positions``).
    Deposit and gather share this so they use identical shape functions.
    """
    nx, ny, nz = (int(n) for n in n_grid)
    n_t = torch.tensor([nx, ny, nz], dtype=dtype, device=device)
    dx = (grid_max - grid_min) / (n_t - 1.0)
    pos_norm = (positions - grid_min) / dx                       # (N, 3)

    base = torch.floor(pos_norm).long()                          # detached int
    nmax = torch.tensor([nx - 2, ny - 2, nz - 2], dtype=torch.long,
                        device=device)
    zero = torch.zeros((), dtype=torch.long, device=device)
    base = torch.minimum(torch.maximum(base, zero), nmax)        # clamp [0,n-2]

    frac = torch.clamp(pos_norm - base.to(dtype), 0.0, 1.0)      # (N, 3)
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    wcx = torch.stack([1.0 - fx, fx], dim=1)                     # (N, 2)
    wcy = torch.stack([1.0 - fy, fy], dim=1)
    wcz = torch.stack([1.0 - fz, fz], dim=1)

    offs = torch.tensor(_CIC_OFFSETS, dtype=torch.long, device=device)  # (8,3)
    corner = base.unsqueeze(1) + offs.unsqueeze(0)               # (N, 8, 3)
    ox, oy, oz = offs[:, 0], offs[:, 1], offs[:, 2]              # (8,)
    weights = wcx[:, ox] * wcy[:, oy] * wcz[:, oz]              # (N, 8)
    return corner[:, :, 0], corner[:, :, 1], corner[:, :, 2], weights


def deposit_cic_torch(positions, charges, grid_min, grid_max, n_grid, *,
                      dtype: torch.dtype = torch.float64,
                      device=None) -> torch.Tensor:
    """Deposit particle charges onto a 3-D grid by Cloud-In-Cell weighting.

    ``positions`` is ``(N, 3)`` in mm, ``charges`` is ``(N,)``. Returns the
    ``(nx, ny, nz)`` charge-density tensor (charge / cell volume).
    """
    nx, ny, nz = (int(n) for n in n_grid)
    positions = torch.as_tensor(positions, dtype=dtype, device=device)
    charges = torch.as_tensor(charges, dtype=dtype, device=device)
    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)

    rho = torch.zeros((nx, ny, nz), dtype=dtype, device=device)
    if positions.shape[0] == 0:
        return rho

    n_t = torch.tensor([nx, ny, nz], dtype=dtype, device=device)
    dx = (grid_max - grid_min) / (n_t - 1.0)
    cell_vol = dx[0] * dx[1] * dx[2]

    cix, ciy, ciz, weights = cic_cells(positions, grid_min, grid_max, n_grid,
                                       dtype=dtype, device=device)
    vals = (weights * charges.unsqueeze(1)).reshape(-1)          # (8N,)
    rho.index_put_((cix.reshape(-1), ciy.reshape(-1), ciz.reshape(-1)),
                   vals, accumulate=True)
    return rho / cell_vol


# ---------------------------------------------------------------------------
# Triangular Shape Cloud (27-cell, quadratic) — smoother than CIC
# ---------------------------------------------------------------------------
def _tsc_weights_1d(f: torch.Tensor) -> torch.Tensor:
    """1-D TSC weights as an ``(N, 3)`` tensor ``[w_minus, w_zero, w_plus]``.

    Mirrors ``linac_gen.pic.charge_deposition._tsc_weights_1d``; the three
    weights sum to 1 for any fractional offset ``f``.
    """
    w_zero = 0.75 - (f - 0.5) ** 2
    w_plus = 0.5 * f * f
    w_minus = 0.5 * (1.0 - f) ** 2
    return torch.stack([w_minus, w_zero, w_plus], dim=1)


def tsc_cells(positions: torch.Tensor, grid_min: torch.Tensor,
              grid_max: torch.Tensor, n_grid, *,
              dtype: torch.dtype, device):
    """Shared 27-cell TSC stencil for deposit and gather.

    Returns ``(cix, ciy, ciz, weights)`` — the first three are ``(N, 27)``
    long index tensors, ``weights`` the matching ``(N, 27)`` quadratic
    weight tensor (differentiable in ``positions``).
    """
    nx, ny, nz = (int(n) for n in n_grid)
    n_t = torch.tensor([nx, ny, nz], dtype=dtype, device=device)
    dx = (grid_max - grid_min) / (n_t - 1.0)
    pos_norm = (positions - grid_min) / dx

    base = torch.floor(pos_norm).long()
    # TSC reaches the +/-1 neighbours, so clamp one cell tighter: [1, n-2].
    nmax = torch.tensor([nx - 2, ny - 2, nz - 2], dtype=torch.long,
                        device=device)
    one = torch.ones((), dtype=torch.long, device=device)
    base = torch.minimum(torch.maximum(base, one), nmax)

    frac = torch.clamp(pos_norm - base.to(dtype), 0.0, 1.0)
    w1d_x = _tsc_weights_1d(frac[:, 0])                          # (N, 3)
    w1d_y = _tsc_weights_1d(frac[:, 1])
    w1d_z = _tsc_weights_1d(frac[:, 2])

    offs = torch.tensor(_TSC_OFFSETS, dtype=torch.long, device=device)  # (27,3)
    corner = base.unsqueeze(1) + offs.unsqueeze(0)               # (N, 27, 3)
    widx = offs + 1                       # offset -1/0/+1 -> weight index 0/1/2
    weights = (w1d_x[:, widx[:, 0]] * w1d_y[:, widx[:, 1]]
               * w1d_z[:, widx[:, 2]])                           # (N, 27)
    return corner[:, :, 0], corner[:, :, 1], corner[:, :, 2], weights


def deposit_tsc_torch(positions, charges, grid_min, grid_max, n_grid, *,
                      dtype: torch.dtype = torch.float64,
                      device=None) -> torch.Tensor:
    """Deposit particle charges by Triangular Shape Cloud (27-cell) weighting.

    Torch mirror of :func:`linac_gen.pic.charge_deposition.deposit_tsc`;
    smoother than CIC (~4x less grid noise). Same signature as
    :func:`deposit_cic_torch`.
    """
    nx, ny, nz = (int(n) for n in n_grid)
    positions = torch.as_tensor(positions, dtype=dtype, device=device)
    charges = torch.as_tensor(charges, dtype=dtype, device=device)
    grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
    grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)

    rho = torch.zeros((nx, ny, nz), dtype=dtype, device=device)
    if positions.shape[0] == 0:
        return rho

    n_t = torch.tensor([nx, ny, nz], dtype=dtype, device=device)
    dx = (grid_max - grid_min) / (n_t - 1.0)
    cell_vol = dx[0] * dx[1] * dx[2]

    cix, ciy, ciz, weights = tsc_cells(positions, grid_min, grid_max, n_grid,
                                       dtype=dtype, device=device)
    vals = (weights * charges.unsqueeze(1)).reshape(-1)          # (27N,)
    rho.index_put_((cix.reshape(-1), ciy.reshape(-1), ciz.reshape(-1)),
                   vals, accumulate=True)
    return rho / cell_vol
