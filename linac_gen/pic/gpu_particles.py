"""The default engine's charge deposit and field gather on an NVIDIA GPU.

With the CUDA backend (``use_gpu="cuda"``, or ``"auto"``/``"gpu"`` finding
cupy) the Poisson solve runs on the GPU, while the deposit and the gather
run on the CPU threads: the charge grid goes to the GPU and the potential
comes back at every kick.  With ``SpaceChargeConfig.gpu_particles`` (or
``LINAC_GEN_GPU_PARTICLES=1``; ``=0`` turns it off whatever the config
says) they run on the GPU as well: the particles go over once per kick, the
deposit, the solve and the field differences stay there, and the fields at
the particles come back.

The deposit and the gather are the torch engine's (linac_gen.pic.torch), on
the GPU in float64.  A run repeats bit for bit: torch's CUDA deposit sums
every cell in sorted order (see linac_gen.pic.torch.devices).  They use the
same cells and weights as the CPU kernels but add in another order, so
results differ from the CPU's in the last digits -- as the GPU solve
already does.  Arrays pass between torch and cupy without copies (DLPack).
Memory: the deposit builds every particle's stencil at once, about 0.6 kB
(CIC) or 1.6 kB (TSC) per particle besides torch's sort buffers.

Not used with the MPS backend (float32) or the CPU backend, nor with field
reuse (``reuse_every > 1``), whose kicks keep the CPU kernels; a request
that cannot be met is said once in the log.  The same code runs on the CPU
(numpy for cupy, torch's CPU tensors), which is how it is tested where
there is no NVIDIA GPU (tests/pic/test_gpu_particles.py); on a CUDA
machine, scripts/verify_cuda.py checks it against the CPU and itself.
"""
from __future__ import annotations

import logging
import os

import numpy as np

#: Environment switch: "1"/"true"/"on" asks for it, "0"/"false"/"off"
#: refuses it, whatever SpaceChargeConfig.gpu_particles says.
ENV = "LINAC_GEN_GPU_PARTICLES"

_log = logging.getLogger(__name__)
_warned: set[str] = set()


def warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        _log.warning(msg, *args)


def requested(config) -> bool:
    """True when the environment, else the config, asks for it."""
    env = os.environ.get(ENV, "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return bool(getattr(config, "gpu_particles", False))


def unavailable_reason(solver) -> str | None:
    """None when ``solver``'s FFTs run on the CUDA backend and torch can run
    on a CUDA device; else why the deposit and gather stay on the CPU."""
    backend = getattr(getattr(solver, "_backend", None), "name", None)
    if backend != "gpu":                            # cupy; not MPS, not CPU
        return (f"the PIC backend is {backend or 'not set up'}, not the "
                f"CUDA one (cupy)")
    try:
        from linac_gen.pic.torch.devices import cuda_available
    except ImportError:                             # pragma: no cover
        return "torch is not installed"
    if not cuda_available():
        return "torch has no usable CUDA device (a CPU-only torch build?)"
    return None


def available(solver) -> bool:
    return unavailable_reason(solver) is None


def rest_frame_field(solver, coords, charges, grid_min, grid_max, n_grid,
                     kernel: str = "cic", device="cuda") -> np.ndarray:
    """Deposit, Poisson solve and gather on ``device``; returns the field at
    every particle, ``(N, 3)``, on the host, in solver units -- what
    PicSolver._rest_frame_field returns."""
    import torch

    from linac_gen.pic.torch.deposition import (deposit_cic_torch,
                                                deposit_tsc_torch)
    from linac_gen.pic.torch.interpolation import (interpolate_cic_torch,
                                                   interpolate_tsc_torch)
    deposit, gather = ((deposit_tsc_torch, interpolate_tsc_torch)
                       if kernel == "tsc" else
                       (deposit_cic_torch, interpolate_cic_torch))
    n_grid = tuple(int(v) for v in n_grid)
    as_t = dict(dtype=torch.float64, device=device)
    pos = torch.as_tensor(np.ascontiguousarray(coords), **as_t)
    q = torch.as_tensor(np.ascontiguousarray(charges), **as_t)
    gmin = torch.as_tensor(np.asarray(grid_min, dtype=np.float64), **as_t)
    gmax = torch.as_tensor(np.asarray(grid_max, dtype=np.float64), **as_t)
    xp = solver._backend.xp
    with torch.no_grad():
        rho = deposit(pos, q, gmin, gmax, n_grid, device=device)
        fields = solver.solve_device(xp.from_dlpack(rho))
        ex, ey, ez = (torch.from_dlpack(f) for f in fields)
        e = gather(ex, ey, ez, pos, gmin, gmax, n_grid, device=device)
    return e.cpu().numpy()
