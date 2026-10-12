"""Differentiable (PyTorch autograd) PIC space-charge kick.

One space-charge kick over a step ``ds``, computed entirely with torch ops
so gradients flow through it. Numerically reproduces
:meth:`linac_gen.pic.pic_solver.PicSolver.kick` (FP64 CPU) to ~1e-9.

Pipeline (mirrors ``PicSolver.kick``): phase-space -> spatial coords ->
Lorentz boost to the rest frame -> grid sized from the distribution ->
CIC/TSC deposit -> IGF FFT Poisson solve -> gather -> momentum kick.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import torch

from linac_gen.pic.torch.deposition import deposit_cic_torch, deposit_tsc_torch
from linac_gen.pic.torch.interpolation import (
    interpolate_cic_torch, interpolate_tsc_torch,
)
from linac_gen.pic.torch.poisson import TorchPoissonSolverFFT
from linac_gen.parallel.threads import sync_torch as _sync_torch

# a run that loads torch here keeps its CPU work on one thread
_sync_torch()

__all__ = ["torch_pic_sc_kick"]

# Solvers whose Green's function was built for a given grid spacing.  Used
# only for plain tensors (see _cacheable): TorchPoissonSolverFFT.solve()
# reads nothing but (n_grid, dx) and the Green's function and keeps no
# per-call state, so a solver built for the same exact spacing returns the
# same fields.  Bounded by entry count and by the bytes its Green's
# functions hold; keyed on the device the spacing actually lives on;
# inference-mode tensors are kept apart from normal ones.
# clear_solver_cache() (also TorchPicSolver.free_memory()) empties it.
_SOLVER_CACHE: "OrderedDict[tuple, tuple[TorchPoissonSolverFFT, int]]" = OrderedDict()
_SOLVER_CACHE_MAX = 4
_SOLVER_CACHE_MAX_BYTES = 128 * 2**20
_SOLVER_LOCK = threading.Lock()


def clear_solver_cache() -> None:
    """Drop every cached solver and the Green's-function memory it holds."""
    with _SOLVER_LOCK:
        _SOLVER_CACHE.clear()


def _cacheable(grid_min, grid_max) -> bool:
    """True only for plain tensors: no reverse- or forward-mode gradient,
    not wrapped by a torch.func transform, not being traced or compiled.
    In any of those a reused solver would carry the wrong (or no) tangent,
    or be frozen into the graph as a constant."""
    if torch.is_grad_enabled() and (grid_min.requires_grad
                                    or grid_max.requires_grad):
        return False
    if torch.jit.is_tracing():
        return False
    compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if compiling is not None and compiling():
        return False
    from torch.autograd import forward_ad
    if any(forward_ad.unpack_dual(g).tangent is not None
           for g in (grid_min, grid_max)):
        return False
    wrapped = getattr(getattr(torch._C, "_functorch", None),
                      "is_functorch_wrapped_tensor", None)
    if wrapped is None:                 # cannot tell: build fresh, as before
        return False
    return not (wrapped(grid_min) or wrapped(grid_max))


def _solver_for(grid_min, grid_max, n_grid, green_kind, dtype, device):
    if not _cacheable(grid_min, grid_max):
        return TorchPoissonSolverFFT(grid_min, grid_max, n_grid,
                                     green_kind=green_kind, dtype=dtype,
                                     device=device)
    n = tuple(int(v) for v in n_grid)
    n_t = torch.tensor([float(v) for v in n], dtype=dtype, device=device)
    dx = (grid_max - grid_min) / (n_t - 1.0)        # the solver's expression
    key = (n, tuple(dx.tolist()), green_kind, str(dtype), str(dx.device),
           torch.is_inference_mode_enabled())
    with _SOLVER_LOCK:
        hit = _SOLVER_CACHE.get(key)
        if hit is not None:
            _SOLVER_CACHE.move_to_end(key)
            return hit[0]
    solver = TorchPoissonSolverFFT(grid_min, grid_max, n_grid,
                                   green_kind=green_kind, dtype=dtype,
                                   device=device)
    nbytes = solver._G_fft.numel() * solver._G_fft.element_size()
    if nbytes <= _SOLVER_CACHE_MAX_BYTES:
        with _SOLVER_LOCK:
            _SOLVER_CACHE[key] = (solver, nbytes)
            total = sum(b for _, b in _SOLVER_CACHE.values())
            while (len(_SOLVER_CACHE) > _SOLVER_CACHE_MAX
                   or total > _SOLVER_CACHE_MAX_BYTES):
                _, (_, b) = _SOLVER_CACHE.popitem(last=False)
                total -= b
    return solver


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
    solver = _solver_for(grid_min, grid_max, n_grid, green_kind, dtype, device)
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
