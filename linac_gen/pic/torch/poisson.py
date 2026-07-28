"""Differentiable (PyTorch autograd) FFT Poisson solver — open boundary.

A torch mirror of :class:`linac_gen.pic.poisson_solver.PoissonSolverFFT`:
zero-pads the charge density onto a doubled grid, convolves with the
free-space Green's function via :mod:`torch.fft`, and returns
``E = -grad(phi)`` by finite differences. Every operation is
autograd-capable, so gradients flow through the solve with respect to the
charge density and the grid spacing.

FP64 on CPU — required for parity with the numpy solver and clean gradients.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from linac_gen.pic.torch.greens import (
    build_integrated_greens_torch,
    build_point_greens_torch,
)

__all__ = ["TorchPoissonSolverFFT"]


class TorchPoissonSolverFFT:
    """Differentiable 3-D FFT Poisson solver (open BC, Hockney's method)."""

    def __init__(self, grid_min, grid_max, n_grid, *,
                 green_kind: str = "igf",
                 dtype: torch.dtype = torch.float64, device=None):
        if green_kind not in ("igf", "point"):
            raise ValueError(
                f"green_kind must be 'igf' or 'point', got {green_kind!r}")
        self.dtype = dtype
        self.device = device
        self.green_kind = green_kind

        self.grid_min = torch.as_tensor(grid_min, dtype=dtype, device=device)
        self.grid_max = torch.as_tensor(grid_max, dtype=dtype, device=device)
        self.n_grid = tuple(int(n) for n in n_grid)
        self._n = torch.tensor([float(v) for v in self.n_grid],
                               dtype=dtype, device=device)
        self.dx = (self.grid_max - self.grid_min) / (self._n - 1.0)
        self._build_greens()

    # ------------------------------------------------------------------
    def _build_greens(self) -> None:
        builder = (build_integrated_greens_torch if self.green_kind == "igf"
                   else build_point_greens_torch)
        G = builder(self.n_grid, self.dx, dtype=self.dtype, device=self.device)
        self._G_fft = torch.fft.rfftn(G)

    def update_grid(self, grid_min, grid_max) -> None:
        """Re-target the solver to a new grid extent (rebuilds the Green's
        function). ``n_grid`` and ``green_kind`` are unchanged."""
        self.grid_min = torch.as_tensor(grid_min, dtype=self.dtype,
                                        device=self.device)
        self.grid_max = torch.as_tensor(grid_max, dtype=self.dtype,
                                        device=self.device)
        self.dx = (self.grid_max - self.grid_min) / (self._n - 1.0)
        self._build_greens()

    # ------------------------------------------------------------------
    def solve(self, rho: torch.Tensor):
        """Solve ``nabla^2 phi = -rho/eps0``; return ``(Ex, Ey, Ez)`` tensors.

        ``rho`` is an ``(nx, ny, nz)`` charge-density tensor; the returned
        fields are ``(nx, ny, nz)`` tensors on the physical grid.
        """
        nx, ny, nz = self.n_grid
        dxx, dyy, dzz = self.dx[0], self.dx[1], self.dx[2]
        cell_vol = dxx * dyy * dzz

        rho = torch.as_tensor(rho, dtype=self.dtype, device=self.device)
        # Zero-pad (rho * cell_vol) onto the doubled grid, original block in
        # the low corner — F.pad order is (z_lo, z_hi, y_lo, y_hi, x_lo, x_hi).
        padded = F.pad(rho * cell_vol, (0, nz, 0, ny, 0, nx))

        phi_padded = torch.fft.irfftn(
            torch.fft.rfftn(padded) * self._G_fft,
            s=(2 * nx, 2 * ny, 2 * nz),
        )
        phi = phi_padded[:nx, :ny, :nz]

        # E = -grad(phi): central differences in the interior, one-sided at
        # the two boundary planes — built out-of-place via cat (autograd-safe),
        # matching the numpy solver's boundary treatment exactly.
        ex = torch.cat([
            -(phi[1:2, :, :] - phi[0:1, :, :]) / dxx,
            -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dxx),
            -(phi[-1:, :, :] - phi[-2:-1, :, :]) / dxx,
        ], dim=0)
        ey = torch.cat([
            -(phi[:, 1:2, :] - phi[:, 0:1, :]) / dyy,
            -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dyy),
            -(phi[:, -1:, :] - phi[:, -2:-1, :]) / dyy,
        ], dim=1)
        ez = torch.cat([
            -(phi[:, :, 1:2] - phi[:, :, 0:1]) / dzz,
            -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dzz),
            -(phi[:, :, -1:] - phi[:, :, -2:-1]) / dzz,
        ], dim=2)
        return ex, ey, ez
