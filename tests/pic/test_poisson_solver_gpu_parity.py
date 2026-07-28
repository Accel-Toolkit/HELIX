"""CPU vs GPU numerical parity for PoissonSolverFFT.

Skipped when cupy is not importable or no CUDA device is visible.
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


@pytest.mark.parametrize("n", [32, 64])
def test_cpu_gpu_parity(n):
    """CPU and GPU FFT paths must agree to ~1e-9 relative on the same ρ.

    The Green's function is now built on each backend's native array
    module (numpy for CPU, cupy for GPU) rather than always on CPU and
    transferred.  This avoids a per-kick H→D copy in adaptive mode.  The
    cost is that cupy's ``arctan/arcsinh`` differ from numpy's by
    ~1e-15 relative; through the IGF inclusion–exclusion + FFT + finite
    differences this amplifies to ~1e-10 in the field magnitude.  That is
    still ~6 orders of magnitude below typical PIC discretisation noise.
    """
    rng = np.random.default_rng(17)
    rho = rng.standard_normal((n, n, n))
    gmin = np.array([-1., -1., -1.])
    gmax = np.array([+1., +1., +1.])
    ng   = np.array([n, n, n])

    cpu = PoissonSolverFFT(gmin, gmax, ng, use_gpu="cpu")
    gpu = PoissonSolverFFT(gmin, gmax, ng, use_gpu="gpu")

    Excpu, Eycpu, Ezcpu = cpu.solve(rho)
    Exgpu, Eygpu, Ezgpu = gpu.solve(rho)

    for a, b, name in [(Excpu, Exgpu, "Ex"),
                        (Eycpu, Eygpu, "Ey"),
                        (Ezcpu, Ezgpu, "Ez")]:
        diff = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(a)))
        assert diff < 1e-9 * max(scale, 1.0), (
            f"{name}: max|Δ|={diff:.3e} scale={scale:.3e}"
        )
    gpu.free_memory()
