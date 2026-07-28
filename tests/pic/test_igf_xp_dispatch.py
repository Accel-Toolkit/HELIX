"""IGF (and update_grid + solve) parity between numpy and cupy backends.

Skipped when cupy or a CUDA device is not available.
"""
import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from linac_gen.pic.poisson_solver import PoissonSolverFFT


def _has_device() -> bool:
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_device(),
                                reason="No CUDA device visible")


def test_igf_xp_dispatch_numpy_cupy_close():
    """The Green's function built natively on each backend must agree to
    ~1e-12 relative — cupy/numpy transcendentals can drift at FP64 fpu."""
    n = np.array([32, 32, 32])
    gmin = np.array([-1.5, -2.1, -0.8])
    gmax = np.array([+1.5, +2.1, +0.8])

    cpu = PoissonSolverFFT(gmin, gmax, n, use_gpu="cpu", green_kind="igf")
    gpu = PoissonSolverFFT(gmin, gmax, n, use_gpu="gpu", green_kind="igf")

    G_cpu = cpu._G_fft  # numpy complex array
    G_gpu = cupy.asnumpy(gpu._G_fft)  # bring to host for compare

    rel = np.max(np.abs(G_cpu - G_gpu)) / max(np.max(np.abs(G_cpu)), 1.0)
    assert rel < 1e-12, f"IGF FFT cupy/numpy parity: max rel diff {rel:.3e}"
    gpu.free_memory()


def test_update_grid_gpu_memory_bounded():
    """Calling update_grid many times must not leak GPU memory.  cupy's
    default mempool will grow as needed, then plateau when allocations get
    reused.  A 200-iteration loop on a small grid should stabilise well
    below 100 MiB peak usage (one G_fft + transients ≈ tens of MiB)."""
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()

    n = np.array([32, 32, 32])
    solver = PoissonSolverFFT(np.array([-1., -1., -1.]),
                              np.array([+1., +1., +1.]),
                              n, use_gpu="gpu", green_kind="igf")

    rng = np.random.default_rng(0)
    peaks = []
    for i in range(200):
        rand_min = rng.uniform(-3, -0.5, size=3)
        rand_max = rng.uniform(0.5, 3, size=3)
        solver.update_grid(rand_min, rand_max)
        if i % 20 == 0:
            peaks.append(pool.used_bytes())

    peak_mib = max(peaks) / (1024 ** 2)
    assert peak_mib < 100, f"GPU memory peak {peak_mib:.1f} MiB exceeds budget"
    # Also assert no monotonic growth: late-iteration usage shouldn't be
    # significantly larger than mid-iteration usage.
    mid = peaks[len(peaks) // 2]
    late = peaks[-1]
    assert late <= mid * 1.2 + 1e6, \
        f"GPU memory growing: mid={mid/1e6:.1f}MB late={late/1e6:.1f}MB"
    solver.free_memory()


def test_update_grid_gpu_solve_matches_fresh():
    """End-to-end: re-targeted GPU solver should give identical fields to
    a fresh GPU solver on the same grid + ρ (within FP rounding)."""
    rng = np.random.default_rng(11)
    rho = rng.standard_normal((32, 32, 32))
    n = np.array([32, 32, 32])
    A_min = np.array([-1.0, -1.0, -1.0])
    A_max = np.array([+1.0, +1.0, +1.0])
    B_min = np.array([-2.0, -1.5, -3.0])
    B_max = np.array([+2.0, +1.5, +3.0])

    A = PoissonSolverFFT(A_min, A_max, n, use_gpu="gpu")
    B = PoissonSolverFFT(B_min, B_max, n, use_gpu="gpu")
    A.update_grid(B_min, B_max)

    Ea = A.solve(rho)
    Eb = B.solve(rho)
    for a, b, name in zip(Ea, Eb, ("Ex", "Ey", "Ez")):
        scale = max(float(np.max(np.abs(a))), 1.0)
        diff = float(np.max(np.abs(a - b)))
        assert diff < 1e-9 * scale, \
            f"{name}: re-target vs fresh diff {diff:.3e} scale {scale:.3e}"
    A.free_memory()
    B.free_memory()
