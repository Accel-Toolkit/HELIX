"""MPS (torch / Apple Silicon Metal) backend tests.

Skipped automatically on hosts without an MPS device (Linux, NVIDIA, etc.),
so the rest of the suite remains portable.

FP32 round-off (~1e-7 relative) is the expected accuracy floor — Apple
Silicon GPUs do not support FP64 in hardware.
"""
import numpy as np
import pytest

from linac_gen.pic import gpu_backend
from linac_gen.pic.poisson_solver import PoissonSolverFFT


mps_only = pytest.mark.skipif(
    gpu_backend._try_import_torch_mps() is None,
    reason="torch MPS device not available on this host",
)


@mps_only
def test_mps_probe_returns_torch():
    """_try_import_torch_mps() succeeds on MPS-capable hosts."""
    torch_mod = gpu_backend._try_import_torch_mps()
    assert torch_mod is not None
    assert torch_mod.backends.mps.is_available()


@mps_only
def test_force_mps_returns_gpu_name(monkeypatch):
    """use_gpu='mps' selects 'gpu' from select_backend()."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    assert gpu_backend.select_backend("mps") == "gpu"


@mps_only
def test_force_mps_picks_mps_backend_class(monkeypatch):
    """get_backend('mps') instantiates _MpsBackend."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    gpu_backend._BANNER_EMITTED = False
    be = gpu_backend.get_backend("mps")
    assert isinstance(be, gpu_backend._MpsBackend)
    assert be.name == "mps"


@mps_only
def test_auto_never_picks_mps_even_when_available(monkeypatch):
    """POLICY REVERSAL (2026-07-11): on hosts with MPS but no cupy, auto
    mode picks the float64 CPU backend, NOT the float32 MPS backend.  The
    old auto→MPS behavior silently ran all PIC space charge in float32 on
    Apple Silicon (and measured no faster than the CPU FFT).  MPS now
    requires explicit opt-in: use_gpu='mps' or 'gpu'."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    gpu_backend._BANNER_EMITTED = False
    be = gpu_backend.get_backend("auto")
    assert isinstance(be, gpu_backend._CpuBackend)


@mps_only
def test_explicit_gpu_still_reaches_mps(monkeypatch):
    """use_gpu='gpu' (explicit any-GPU) falls back to MPS when cupy is
    absent — the opt-in path survives the auto-policy reversal."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    gpu_backend._BANNER_EMITTED = False
    with pytest.warns(UserWarning, match="float32"):
        be = gpu_backend.get_backend("gpu")
    assert isinstance(be, gpu_backend._MpsBackend)


@mps_only
def test_force_cuda_without_cupy_raises(monkeypatch):
    """use_gpu='cuda' must raise when cupy is missing, even with MPS available."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    with pytest.raises(RuntimeError, match="cupy"):
        gpu_backend.select_backend("cuda")


@mps_only
def test_force_mps_without_torch_raises(monkeypatch):
    """use_gpu='mps' must raise when torch/MPS is missing."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_torch_mps", lambda: None)
    with pytest.raises(RuntimeError, match="torch"):
        gpu_backend.select_backend("mps")


@mps_only
def test_env_var_mps_selects_mps(monkeypatch):
    """LINAC_GEN_USE_GPU=mps env override picks the MPS backend."""
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "mps")
    gpu_backend._BANNER_EMITTED = False
    be = gpu_backend.get_backend("auto")  # arg ignored — env wins
    assert isinstance(be, gpu_backend._MpsBackend)


@mps_only
def test_mps_rfftn_round_trip_matches_numpy():
    """rfftn(x) → irfftn → reconstructs x within FP32 round-off."""
    be = gpu_backend._MpsBackend()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, 8, 8))
    y = be.rfftn(be.to_device(x))
    x_back = be.to_host(be.irfftn(y, s=x.shape))
    # FP32 round-off floor; loosen tolerance vs. the FP64 CPU equivalent.
    assert np.allclose(x_back, x, atol=1e-5, rtol=1e-5)


@mps_only
def test_mps_poisson_matches_cpu_within_fp32_floor():
    """Full PoissonSolverFFT.solve produces same fields on CPU and MPS.

    Tolerance is set to FP32 round-off (~1e-6 rel) since the Metal hardware
    does not support FP64.  This is the accuracy floor we accept for PIC
    Poisson on Apple Silicon.
    """
    gmin = np.array([-1.0, -1.0, -1.0])
    gmax = np.array([ 1.0,  1.0,  1.0])
    n    = np.array([16, 16, 16])
    rng  = np.random.default_rng(42)
    rho  = rng.standard_normal((16, 16, 16)) * 1e-6
    rho -= rho.mean()  # neutralise net charge

    gpu_backend._BANNER_EMITTED = False
    solver_cpu = PoissonSolverFFT(gmin, gmax, n, use_gpu="cpu")
    Ex_c, Ey_c, Ez_c = solver_cpu.solve(rho)

    gpu_backend._BANNER_EMITTED = False
    solver_mps = PoissonSolverFFT(gmin, gmax, n, use_gpu="mps")
    Ex_m, Ey_m, Ez_m = solver_mps.solve(rho)

    for name, A, B in (("Ex", Ex_c, Ex_m), ("Ey", Ey_c, Ey_m), ("Ez", Ez_c, Ez_m)):
        denom   = max(float(np.abs(A).max()), 1e-300)
        rel_max = float(np.abs(A - B).max()) / denom
        assert rel_max < 5e-6, (
            f"{name}: MPS deviates from CPU by {rel_max:.2e} (FP32 floor "
            f"expected ~6e-7; >5e-6 means the FFT path has a real bug)"
        )
