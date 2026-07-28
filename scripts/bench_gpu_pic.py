"""Benchmark PIC Poisson solve for CPU vs GPU at 48³, 64³, 96³, 128³.

Prints wall time per backend and the relative error between the two.  On a
host without cupy the GPU rows report "skipped".

Run from the repository root:
    PYTHONPATH=. python3 scripts/bench_gpu_pic.py
"""
from __future__ import annotations

import os
import time

import numpy as np

# Clear any forcing env var so each solver picks its own backend.
os.environ.pop("LINAC_GEN_USE_GPU", None)

from linac_gen.pic.poisson_solver import PoissonSolverFFT


rng = np.random.default_rng(99)

print(f"{'grid':>6}  {'backend':>6}  {'time [ms]':>10}  {'max|Δ| vs CPU':>14}")
for n in (48, 64, 96, 128):
    gmin = np.array([-1., -1., -1.])
    gmax = np.array([+1., +1., +1.])
    ng   = np.array([n, n, n])
    rho  = rng.standard_normal((n, n, n))

    cpu = PoissonSolverFFT(gmin, gmax, ng, use_gpu="cpu")
    _   = cpu.solve(rho)                     # warm-up
    t0 = time.perf_counter()
    for _ in range(3):
        Ex_cpu, Ey_cpu, Ez_cpu = cpu.solve(rho)
    cpu_ms = (time.perf_counter() - t0) * 1000 / 3
    print(f"{n:>4}³  {'cpu':>6}  {cpu_ms:>10.2f}  {'—':>14}")

    try:
        gpu = PoissonSolverFFT(gmin, gmax, ng, use_gpu="gpu")
    except RuntimeError as exc:
        print(f"{n:>4}³  {'gpu':>6}  {'skipped':>10}  {'—':>14}   ({exc})")
        continue
    _ = gpu.solve(rho)                       # warm-up (JIT + mempool)
    t0 = time.perf_counter()
    for _ in range(3):
        Ex_g, Ey_g, Ez_g = gpu.solve(rho)
    gpu_ms = (time.perf_counter() - t0) * 1000 / 3
    max_d = max(
        float(np.max(np.abs(Ex_cpu - Ex_g))),
        float(np.max(np.abs(Ey_cpu - Ey_g))),
        float(np.max(np.abs(Ez_cpu - Ez_g))),
    )
    scale = float(np.max(np.abs(Ex_cpu)))
    print(
        f"{n:>4}³  {'gpu':>6}  {gpu_ms:>10.2f}  "
        f"{max_d/max(scale,1.0):>14.3e}   speed-up {cpu_ms/gpu_ms:.1f}×"
    )
    gpu.free_memory()
