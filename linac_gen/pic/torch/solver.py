"""Torch space-charge backend for the ``kick(beam, ds)`` contract.

``TorchPicSolver`` matches the interface of
:class:`linac_gen.pic.pic_solver.PicSolver` but computes the bunched-beam
kick with the differentiable torch PIC. Selected via
``SpaceChargeConfig.sc_backend = "torch"``.

NOT a drop-in replacement for fixed-grid runs: the torch kick re-fits
its grid from the live distribution on EVERY kick (mean ± grid_extent·σ)
and ignores ``grid_mode`` — a ``grid_mode="fixed"`` configuration runs a
different numerical model under this backend (a one-shot
:class:`AdaptiveOnlyBackendWarning` says so at construction).

Inside the numpy multi-particle tracker the result is written back as a
numpy array (gradients are not propagated there — the numpy tracker is not
an autograd graph). End-to-end differentiability is provided separately by
the torch step tracker.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from linac_gen.pic.torch.sc_kick import torch_pic_sc_kick

__all__ = ["AdaptiveOnlyBackendWarning", "TorchPicSolver"]


class AdaptiveOnlyBackendWarning(UserWarning):
    """The torch SC backend honours only adaptive gridding — a
    ``grid_mode="fixed"`` config silently became a per-kick adaptive
    grid before this warning existed (PRAB review finding)."""


class TorchPicSolver:
    """Differentiable-torch PIC space-charge backend (adaptive-grid only)."""

    def __init__(self, config):
        self.config = config
        if getattr(config, "grid_mode", "fixed") == "fixed":
            warnings.warn(
                "sc_backend='torch' is ADAPTIVE-ONLY: the grid is "
                "re-fitted from the live distribution every kick and "
                "grid_mode='fixed' is not honoured — this run uses a "
                "different numerical model than the numpy fixed-grid "
                "solver.  Set grid_mode='adaptive' to silence this "
                "warning, or use the numpy backend for fixed grids.",
                AdaptiveOnlyBackendWarning, stacklevel=3)

    def kick(self, beam, ds: float) -> None:
        """Apply one space-charge kick over step ``ds`` (mm); beam in-place."""
        if beam.current <= 0 or beam.n_alive < 2:
            return
        alive_idx = np.where(beam.alive_mask)[0]
        particles = torch.as_tensor(beam.particles[alive_idx],
                                    dtype=torch.float64)
        ref = beam.ref
        macro_charge = ((beam.current * 1e-3)
                        / (beam.bunch_frequency * 1e6)
                        / beam.n_particles)
        kicked = torch_pic_sc_kick(
            particles, self.config,
            ds_mm=float(ds),
            gamma=float(ref.gamma),
            beta=float(ref.beta),
            mass_mev=float(ref.species.mass),
            wavelength_mm=float(ref.wavelength),
            macro_charge=macro_charge,
            charge_state=float(abs(ref.species.charge)),
        )
        beam.particles[alive_idx] = kicked.detach().numpy()

    def free_memory(self) -> None:
        """Interface parity with ``PicSolver`` — nothing to release."""
