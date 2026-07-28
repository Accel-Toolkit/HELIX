"""Differentiable (PyTorch autograd) PIC space-charge kick.

One space-charge kick over a step ``ds``, computed entirely with torch ops
so gradients flow through it. Numerically reproduces
:meth:`linac_gen.pic.pic_solver.PicSolver.kick` (FP64 CPU) to ~1e-9.

Pipeline (mirrors ``PicSolver.kick``): phase-space -> spatial coords ->
Lorentz boost to the rest frame -> grid sized from the distribution ->
CIC/TSC deposit -> IGF FFT Poisson solve -> gather -> momentum kick.
"""
from __future__ import annotations

import torch

from linac_gen.pic.torch.deposition import deposit_cic_torch, deposit_tsc_torch
from linac_gen.pic.torch.interpolation import (
    interpolate_cic_torch, interpolate_tsc_torch,
)
from linac_gen.pic.torch.poisson import TorchPoissonSolverFFT

__all__ = ["torch_pic_sc_kick"]


def torch_pic_sc_kick(particles, cfg, *, ds_mm, gamma, beta, mass_mev,
                      wavelength_mm, macro_charge, charge_state: float = 1.0,
                      dtype: torch.dtype = torch.float64,
                      device=None) -> torch.Tensor:
    """Apply one differentiable PIC space-charge kick.

    Parameters
    ----------
    particles : (N, 6) tensor
        ``[x mm, x' mrad, y mm, y' mrad, dphi deg, dW MeV]``.
    cfg : SpaceChargeConfig
        Provides ``nx/ny/nz``, ``grid_extent``, ``kernel``, ``green_kind``.
    ds_mm : float
        Step length in mm.
    gamma, beta, mass_mev, wavelength_mm : float
        Reference-particle kinematics.
    charge_state : float, default 1.0
        Test-particle charge-state magnitude |Z| in units of e; scales
        the kick (the deposited bunch charge I/f fixes the source field).
    macro_charge : float or tensor
        Charge per macro-particle, in Coulomb.

    Returns
    -------
    (N, 6) tensor
        A NEW tensor: ``particles`` with the kick added to columns 1, 3, 5
        (x', y', dW); columns 0, 2, 4 unchanged. The input is not mutated.
    """
    particles = torch.as_tensor(particles, dtype=dtype, device=device)
    n = particles.shape[0]
    if n < 2 or float(macro_charge) == 0.0:
        return particles

    kernel = getattr(cfg, "kernel", "cic")
    if kernel not in ("cic", "tsc"):
        raise NotImplementedError(
            f"torch SC backend: kernel={kernel!r} is not supported "
            f"(use 'cic' or 'tsc')")
    deposit_fn = deposit_tsc_torch if kernel == "tsc" else deposit_cic_torch
    interpolate_fn = (interpolate_tsc_torch if kernel == "tsc"
                      else interpolate_cic_torch)
    n_grid = (int(cfg.nx), int(cfg.ny), int(cfg.nz))
    green_kind = getattr(cfg, "green_kind", "igf")

    # Phase space -> spatial [x, y, z]; boost the longitudinal axis to the
    # rest frame (z_rest = gamma * z_lab), exactly as PicSolver.kick.
    x = particles[:, 0]
    y = particles[:, 2]
    dphi = particles[:, 4]
    z_lab = -dphi * (beta * wavelength_mm / 360.0)
    z_rest = z_lab * gamma
    coords = torch.stack([x, y, z_rest], dim=1)                  # (N, 3)

    # Grid sized from the (differentiable) distribution — matches
    # PicSolver._setup_grid. numpy uses population std, so correction=0.
    mean = coords.mean(dim=0)
    std = coords.std(dim=0, correction=0)
    half = torch.clamp(cfg.grid_extent * std, min=1e-6)
    grid_min = mean - half
    grid_max = mean + half

    charges = torch.ones(n, dtype=dtype, device=device) * macro_charge
    rho = deposit_fn(coords, charges, grid_min, grid_max, n_grid,
                     dtype=dtype, device=device)
    solver = TorchPoissonSolverFFT(grid_min, grid_max, n_grid,
                                   green_kind=green_kind, dtype=dtype,
                                   device=device)
    ex, ey, ez = solver.solve(rho)
    e = interpolate_fn(ex, ey, ez, coords, grid_min, grid_max, n_grid,
                       dtype=dtype, device=device)               # (N, 3)
    # The solver works on a millimetre grid, so E_solver = E_SI / 1e6.
    e_si = e * 1.0e6                                             # V/m

    ds_m = ds_mm * 1.0e-3
    # |Z| scales the kick on each test particle (mirrors the numpy PIC
    # kick; no-op for |Z| = 1 species, fixed 2026-07-10).
    z_state = abs(charge_state)
    factor_t = (z_state * ds_m * 1.0e3
                / (mass_mev * 1.0e6 * beta * beta * gamma * gamma))
    factor_z = z_state * ds_m * 1.0e-6

    zero = torch.zeros(n, dtype=dtype, device=device)
    delta = torch.stack([
        zero,
        factor_t * e_si[:, 0],   # x' kick (mrad)
        zero,
        factor_t * e_si[:, 1],   # y' kick (mrad)
        zero,
        factor_z * e_si[:, 2],   # dW kick (MeV)
    ], dim=1)
    return particles + delta
