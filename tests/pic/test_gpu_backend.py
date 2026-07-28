"""Backend-selection logic for the PIC GPU/CPU switch."""
import pytest

from linac_gen.pic import gpu_backend


def test_select_backend_defaults_to_cpu_when_no_gpu_present(monkeypatch):
    """Without cupy or MPS available and no env override, backend is CPU."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    # Force both GPU probes to fail even if a GPU is present on this host.
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    monkeypatch.setattr(gpu_backend, "_try_import_torch_mps", lambda: None)
    assert gpu_backend.select_backend("auto") == "cpu"


def test_env_var_overrides_argument_cpu(monkeypatch):
    """LINAC_GEN_USE_GPU=0 forces CPU even if argument says 'gpu'."""
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "0")
    assert gpu_backend.select_backend("gpu") == "cpu"


def test_forced_gpu_without_any_gpu_raises(monkeypatch):
    """use_gpu='gpu' must raise when neither cupy nor MPS is available."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    monkeypatch.setattr(gpu_backend, "_try_import_torch_mps", lambda: None)
    with pytest.raises(RuntimeError, match="cupy/CUDA nor torch/MPS"):
        gpu_backend.select_backend("gpu")


def test_invalid_mode_raises(monkeypatch):
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    with pytest.raises(ValueError, match="auto/cpu/gpu/cuda/mps"):
        gpu_backend.select_backend("maybe")


def test_cpu_backend_rfftn_matches_numpy(monkeypatch):
    """CPU backend must reproduce numpy.fft.rfftn exactly."""
    import numpy as np
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    be = gpu_backend.get_backend("cpu")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, 8, 8))
    y_ref = np.fft.rfftn(x)
    y_be  = be.to_host(be.rfftn(be.to_device(x)))
    assert np.allclose(y_be, y_ref, atol=1e-12, rtol=1e-12)


def test_fft_workers_single_threaded_off_main_thread(monkeypatch):
    """Regression: scipy.fft's multi-threaded pocketfft pool is unsafe on a
    secondary thread — it raised 'Work item submitted after shutdown' during
    a multiparticle+SC run driven from the assistant's background job pool.
    Off the main thread the CPU backend must fall back to a single worker;
    on the main thread it keeps all cores; an explicit env override wins."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)

    assert gpu_backend._CpuBackend._workers() == -1          # main thread
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(gpu_backend._CpuBackend._workers).result() == 1

    # explicit override is honoured on every thread
    monkeypatch.setenv("LINAC_GEN_FFT_WORKERS", "3")
    assert gpu_backend._CpuBackend._workers() == 3
    with ThreadPoolExecutor(max_workers=1) as ex:
        assert ex.submit(gpu_backend._CpuBackend._workers).result() == 3


def test_cpu_backend_rfftn_workers_bit_identical():
    """The single-thread fallback must not change the transform: scipy.fft
    is bit-identical for workers=1 vs workers=-1 (so the fix is numerically
    inert — the only difference a real run shows is BLAS thread noise)."""
    pytest.importorskip("scipy")
    import numpy as np
    from scipy import fft as sfft
    x = np.random.default_rng(3).standard_normal((32, 32, 32))
    a = sfft.rfftn(x, workers=1)
    b = sfft.rfftn(x, workers=-1)
    assert np.array_equal(a, b)


def test_grid_cells_argument_is_ignored(monkeypatch):
    """select_backend no longer switches on grid size: ``auto`` means GPU
    if cupy/CUDA is available, else CPU.  The ``grid_cells`` arg is kept
    for API compatibility but has no effect on the decision — the earlier
    "tiny grids stay on CPU" heuristic didn't match the measured crossover
    on WSL2 + Ada (PCIe transfer cost scales with grid size, so tiny grids
    actually win on GPU on that hardware)."""
    class _FakeCupy:
        pass
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    # Regardless of grid size, "auto" + available GPU picks GPU.
    assert gpu_backend.select_backend("auto", grid_cells=32 ** 3) == "gpu"
    assert gpu_backend.select_backend("auto", grid_cells=64 ** 3) == "gpu"
    assert gpu_backend.select_backend("auto", grid_cells=128 ** 3) == "gpu"


def test_auto_never_selects_mps(monkeypatch):
    """``auto`` with ONLY MPS available must fall back to CPU.  The FP32
    MPS backend requires explicit opt-in — before 2026-07-11 auto-MPS
    silently ran all PIC space charge in float32 on Apple Silicon."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    monkeypatch.setattr(gpu_backend, "_try_import_torch_mps", lambda: object())
    assert gpu_backend.select_backend("auto") == "cpu"


def test_explicit_gpu_accepts_mps_fallback(monkeypatch):
    """Explicit ``gpu`` (any-GPU request) still accepts MPS when cupy is
    absent — the opt-in path stays functional."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    monkeypatch.setattr(gpu_backend, "_try_import_torch_mps", lambda: object())
    assert gpu_backend.select_backend("gpu") == "gpu"
    assert gpu_backend._pick_gpu_backend_class("gpu") is gpu_backend._MpsBackend


def test_env_var_gpu_with_only_mps_selects_gpu(monkeypatch):
    """LINAC_GEN_USE_GPU=1 is an explicit any-GPU request → MPS allowed."""
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "1")
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    monkeypatch.setattr(gpu_backend, "_try_import_torch_mps", lambda: object())
    assert gpu_backend.select_backend("auto") == "gpu"


def test_mps_backend_construction_warns():
    """Constructing the MPS backend must emit the float32 precision warning."""
    if gpu_backend._try_import_torch_mps() is None:
        pytest.skip("torch MPS not available on this host")
    with pytest.warns(UserWarning, match="float32"):
        gpu_backend._MpsBackend()


def test_auto_solver_is_float64_without_cuda():
    """End-to-end policy guard: a default-constructed (use_gpu='auto')
    Poisson solver on a host without CUDA must compute in float64 —
    superposition holds to float64 round-off, not the ~1e-7 of MPS FP32."""
    if gpu_backend._try_import_cupy() is not None:
        pytest.skip("cupy present — auto legitimately picks CUDA here")
    import numpy as np
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    rng = np.random.default_rng(3)
    gmin = np.array([-1.0] * 3)
    gmax = np.array([1.0] * 3)
    ng = np.array([16, 16, 16])
    r1 = rng.standard_normal(tuple(ng))
    r2 = rng.standard_normal(tuple(ng))
    s = PoissonSolverFFT(gmin, gmax, ng)          # defaults: use_gpu="auto"
    E1 = s.solve(r1)[0]
    E2 = s.solve(r2)[0]
    E12 = s.solve(r1 + r2)[0]
    err = np.abs(E12 - (E1 + E2)).max() / np.abs(E12).max()
    assert err < 1e-12


def test_backend_banner_logs_once(caplog, monkeypatch):
    """INFO-level backend announcement must be emitted at most once per session."""
    import logging
    import numpy as np
    from linac_gen.pic.poisson_solver import PoissonSolverFFT

    gpu_backend._BANNER_EMITTED = False   # reset for the test
    with caplog.at_level(logging.INFO, logger="linac_gen.pic"):
        PoissonSolverFFT(np.array([-1.] * 3), np.array([1.] * 3),
                         np.array([8, 8, 8]), use_gpu="cpu")
        PoissonSolverFFT(np.array([-1.] * 3), np.array([1.] * 3),
                         np.array([8, 8, 8]), use_gpu="cpu")
    banners = [r for r in caplog.records if "PicSolver: using" in r.getMessage()]
    assert len(banners) == 1, f"expected one banner, got {len(banners)}"
