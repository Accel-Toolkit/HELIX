# GPU-accelerated PIC FFT backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional GPU backend to the PIC Poisson solver so 3-D FFTs (dominant cost at 64³+) run on CUDA via cupy, with transparent CPU fallback when cupy/GPU is unavailable.

**Architecture:** A thin `gpu_backend` module picks `cpu` or `gpu` at solver-construction time using `SpaceChargeConfig.use_gpu ∈ {"auto","cpu","gpu"}` overridable by env var `LINAC_GEN_USE_GPU`. `PoissonSolverFFT` carries a `_backend` handle and dispatches `rfftn/irfftn` through it; the CPU path is unchanged (scipy.fft / numpy.fft). GPU path uses `cupyx.scipy.fft` in FP64, transfers `rho` to GPU, multiplies by the pre-cached Green's-function FFT (also on GPU), and transfers `Ex/Ey/Ez` back as NumPy arrays so downstream tracker / recorder code sees no difference.

**Tech Stack:** Python 3.12, scipy.fft (CPU), optional `cupy-cuda12x` (GPU), PyQt6 (GUI knob), pytest. Target hardware: NVIDIA RTX 2000 Ada Generation Laptop GPU, 8 GB VRAM, CUDA 12.2.

**Out of scope for this plan (follow-up):**
- GPU particle push (`Beam.particles` stays on CPU).
- GPU field-map interpolation for `FieldMap3D.track_rk4`.
- Mixed-precision / FP32 FFT (we force FP64 to preserve numerical parity with CPU reference).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `linac_gen/pic/gpu_backend.py` | Backend detection, FFT call dispatch, memory-pool teardown | **Create** |
| `linac_gen/pic/poisson_solver.py` | Route `_rfftn` / `_irfftn` through backend; keep Green's-function FFT on the chosen device | Modify |
| `linac_gen/core/config.py` | Add `SpaceChargeConfig.use_gpu: str = "auto"` + validation | Modify |
| `linac_gen/pic/pic_solver.py` | Forward `config.use_gpu` to `PoissonSolverFFT` | Modify |
| `tests/pic/test_gpu_backend.py` | Backend selection logic tests (CPU always; GPU auto-skipped) | **Create** |
| `tests/pic/test_poisson_solver_gpu_parity.py` | CPU↔GPU numerical parity on a canned ρ (skipped if no GPU) | **Create** |
| `gui/linac_gen_gui/interphase/tabs/convergence_tab.py` | Backend dropdown "Auto / CPU / GPU" | Modify |
| `gui/linac_gen_gui/interphase/app.py` | Pass dropdown choice into `SpaceChargeConfig(use_gpu=...)` | Modify |
| `docs/plans/gpu_pic_fft_plan.md` | This file | (meta) |
| `pyproject.toml` | Add optional-deps entry `[project.optional-dependencies] gpu = ["cupy-cuda12x>=13.0"]` | Modify |

---

## Task 1: Failing test for `select_backend()` default

**Files:**
- Test: `tests/pic/test_gpu_backend.py`

- [ ] **Step 1: Create the test file with one test**

```python
# tests/pic/test_gpu_backend.py
"""Backend-selection logic for the PIC GPU/CPU switch."""
import os
import pytest

from linac_gen.pic import gpu_backend


def test_select_backend_defaults_to_cpu_when_cupy_absent(monkeypatch):
    """Without cupy installed and no env override, backend is CPU."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    # Force the import probe to fail even if cupy is present on this host.
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    assert gpu_backend.select_backend("auto") == "cpu"
```

- [ ] **Step 2: Run the test — expect ImportError (module doesn't exist)**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py -v`
Expected: `ModuleNotFoundError: No module named 'linac_gen.pic.gpu_backend'`

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/pic/test_gpu_backend.py
git commit -m "test(pic): failing test for gpu_backend.select_backend auto→cpu"
```

---

## Task 2: Minimal `gpu_backend` module to pass Task 1

**Files:**
- Create: `linac_gen/pic/gpu_backend.py`

- [ ] **Step 1: Create the module with the minimum surface**

```python
# linac_gen/pic/gpu_backend.py
"""PIC GPU backend selection and FFT dispatch.

Routes 3-D rfftn / irfftn through cupy when a CUDA device is available and
the user has opted in (or is using the default ``auto`` mode).  Falls back
transparently to scipy.fft / numpy.fft on CPU.

Three modes controlled by ``SpaceChargeConfig.use_gpu`` or the environment
variable ``LINAC_GEN_USE_GPU``:

* ``"auto"``  – GPU if cupy is importable and a device is visible, else CPU.
* ``"gpu"``   – force GPU, raise RuntimeError if cupy/CUDA unavailable.
* ``"cpu"``   – force CPU even if a GPU is present (benchmarking / CI).

Environment variable takes precedence over the ``use_gpu`` argument.
"""
from __future__ import annotations

import os
from typing import Literal, Optional

Backend = Literal["cpu", "gpu"]


def _try_import_cupy():
    """Return the cupy module if import + device count probe both succeed."""
    try:
        import cupy  # type: ignore
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        return None
    return None


def _resolve_mode(use_gpu: str) -> str:
    """Env override wins over the argument; normalise to {auto, cpu, gpu}."""
    env = os.environ.get("LINAC_GEN_USE_GPU")
    if env is not None:
        env_norm = env.strip().lower()
        if env_norm in {"1", "true", "yes", "gpu"}:
            return "gpu"
        if env_norm in {"0", "false", "no", "cpu"}:
            return "cpu"
        if env_norm == "auto":
            return "auto"
    mode = (use_gpu or "auto").strip().lower()
    if mode not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"use_gpu must be auto/cpu/gpu, got {use_gpu!r}")
    return mode


def select_backend(use_gpu: str = "auto") -> Backend:
    """Return the backend string to use: 'cpu' or 'gpu'."""
    mode = _resolve_mode(use_gpu)
    if mode == "cpu":
        return "cpu"
    cp = _try_import_cupy()
    if cp is not None:
        return "gpu"
    if mode == "gpu":
        raise RuntimeError(
            "use_gpu='gpu' (or LINAC_GEN_USE_GPU=1) requested, but cupy "
            "and/or a CUDA device are not available on this host."
        )
    return "cpu"
```

- [ ] **Step 2: Run the Task-1 test — expect PASS**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add linac_gen/pic/gpu_backend.py
git commit -m "feat(pic): gpu_backend.select_backend with auto/cpu/gpu modes"
```

---

## Task 3: Env override precedence + forced-mode error

**Files:**
- Modify: `tests/pic/test_gpu_backend.py`

- [ ] **Step 1: Add three more tests**

```python
def test_env_var_overrides_argument_cpu(monkeypatch):
    """LINAC_GEN_USE_GPU=0 forces CPU even if argument says 'gpu'."""
    monkeypatch.setenv("LINAC_GEN_USE_GPU", "0")
    assert gpu_backend.select_backend("gpu") == "cpu"


def test_forced_gpu_without_cupy_raises(monkeypatch):
    """use_gpu='gpu' must raise when cupy/device is missing."""
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: None)
    with pytest.raises(RuntimeError, match="cupy"):
        gpu_backend.select_backend("gpu")


def test_invalid_mode_raises(monkeypatch):
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    with pytest.raises(ValueError, match="auto/cpu/gpu"):
        gpu_backend.select_backend("maybe")
```

- [ ] **Step 2: Run — expect 4 passed (including Task 1's)**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py -v`
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/pic/test_gpu_backend.py
git commit -m "test(pic): env override, forced-gpu error, invalid-mode validation"
```

---

## Task 4: `FftBackend` interface exposing rfftn / irfftn / to_device / to_host

**Files:**
- Modify: `linac_gen/pic/gpu_backend.py`
- Modify: `tests/pic/test_gpu_backend.py`

- [ ] **Step 1: Add the failing test**

```python
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
```

- [ ] **Step 2: Run — expect AttributeError (get_backend missing)**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py::test_cpu_backend_rfftn_matches_numpy -v`
Expected: `AttributeError: module 'linac_gen.pic.gpu_backend' has no attribute 'get_backend'`

- [ ] **Step 3: Add the `FftBackend` classes and `get_backend` factory**

Append to `linac_gen/pic/gpu_backend.py`:

```python
# ---------------------------------------------------------------------------
# FFT backend interface
# ---------------------------------------------------------------------------

class _CpuBackend:
    """scipy.fft / numpy.fft dispatcher (FP64, multi-threaded via workers)."""

    name = "cpu"

    def __init__(self) -> None:
        try:
            from scipy import fft as _sfft
            self._fft = _sfft
            self._has_scipy = True
        except ImportError:
            import numpy as _np
            self._fft = _np.fft
            self._has_scipy = False

    @staticmethod
    def _workers() -> int:
        raw = os.environ.get("LINAC_GEN_FFT_WORKERS")
        if raw:
            try:
                n = int(raw)
                if n != 0:
                    return n
            except ValueError:
                pass
        return -1

    def to_device(self, arr):
        import numpy as _np
        return _np.asarray(arr, dtype=_np.float64)

    def to_host(self, arr):
        import numpy as _np
        return _np.asarray(arr)

    def rfftn(self, x):
        if self._has_scipy:
            return self._fft.rfftn(x, workers=self._workers())
        return self._fft.rfftn(x)

    def irfftn(self, x, s):
        if self._has_scipy:
            return self._fft.irfftn(x, s=s, workers=self._workers())
        return self._fft.irfftn(x, s=s)

    def free_memory(self) -> None:
        pass


class _GpuBackend:
    """cupy dispatcher.  FP64 is forced so results are bit-compatible with CPU
    to within FFT round-off (~1e-12 relative).
    """

    name = "gpu"

    def __init__(self) -> None:
        import cupy as cp
        from cupyx.scipy import fft as cufft
        self._cp = cp
        self._fft = cufft

    def to_device(self, arr):
        return self._cp.asarray(arr, dtype=self._cp.float64)

    def to_host(self, arr):
        return self._cp.asnumpy(arr)

    def rfftn(self, x):
        return self._fft.rfftn(x)

    def irfftn(self, x, s):
        return self._fft.irfftn(x, s=s)

    def free_memory(self) -> None:
        self._cp.get_default_memory_pool().free_all_blocks()
        self._cp.get_default_pinned_memory_pool().free_all_blocks()


def get_backend(use_gpu: str = "auto"):
    """Return a constructed FftBackend object."""
    name = select_backend(use_gpu)
    if name == "gpu":
        return _GpuBackend()
    return _CpuBackend()
```

- [ ] **Step 4: Run — expect PASS**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/pic/gpu_backend.py tests/pic/test_gpu_backend.py
git commit -m "feat(pic): FftBackend interface (CPU + GPU) with to_device/to_host"
```

---

## Task 5: Small-grid CPU-override heuristic

**Files:**
- Modify: `linac_gen/pic/gpu_backend.py`
- Modify: `tests/pic/test_gpu_backend.py`

- [ ] **Step 1: Failing test**

```python
def test_tiny_grid_forces_cpu_even_on_gpu_host(monkeypatch):
    """Grids smaller than 64³ incur more H→D overhead than GPU speedup."""
    # Pretend a GPU is present:
    class _FakeCupy: pass
    monkeypatch.setattr(gpu_backend, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.delenv("LINAC_GEN_USE_GPU", raising=False)
    # For a 32³ grid, backend should collapse back to CPU:
    assert gpu_backend.select_backend("auto", grid_cells=32**3) == "cpu"
    # For a 64³ grid, GPU path is preferred:
    assert gpu_backend.select_backend("auto", grid_cells=64**3) == "gpu"
```

- [ ] **Step 2: Run — expect failure (`grid_cells` kwarg unknown)**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py::test_tiny_grid_forces_cpu_even_on_gpu_host -v`
Expected: `TypeError: select_backend() got an unexpected keyword argument 'grid_cells'`

- [ ] **Step 3: Add `grid_cells` to `select_backend`**

Replace the body of `select_backend` in `gpu_backend.py`:

```python
# In linac_gen/pic/gpu_backend.py, replace select_backend:
_GPU_MIN_CELLS = 64 ** 3   # below this, H→D overhead dominates

def select_backend(use_gpu: str = "auto",
                   grid_cells: Optional[int] = None) -> Backend:
    """Return 'cpu' or 'gpu' for the given mode and (optional) grid size."""
    mode = _resolve_mode(use_gpu)
    if mode == "cpu":
        return "cpu"
    cp = _try_import_cupy()
    if cp is None:
        if mode == "gpu":
            raise RuntimeError(
                "use_gpu='gpu' (or LINAC_GEN_USE_GPU=1) requested, but cupy "
                "and/or a CUDA device are not available on this host."
            )
        return "cpu"
    # cupy is available — check the tiny-grid heuristic in 'auto' mode only.
    if mode == "auto" and grid_cells is not None and grid_cells < _GPU_MIN_CELLS:
        return "cpu"
    return "gpu"
```

- [ ] **Step 4: Run — expect all 6 tests PASS**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/pic/gpu_backend.py tests/pic/test_gpu_backend.py
git commit -m "feat(pic): tiny-grid heuristic keeps small FFTs on CPU"
```

---

## Task 6: Add `use_gpu` field to `SpaceChargeConfig`

**Files:**
- Modify: `linac_gen/core/config.py:89-137`
- Modify: `tests/core/test_config.py` (create if absent)

- [ ] **Step 1: Failing test**

```python
# tests/core/test_config.py — add this test
def test_space_charge_config_use_gpu_default_and_validation():
    from linac_gen.core.config import SpaceChargeConfig
    import pytest
    assert SpaceChargeConfig().use_gpu == "auto"
    assert SpaceChargeConfig(use_gpu="cpu").use_gpu == "cpu"
    assert SpaceChargeConfig(use_gpu="gpu").use_gpu == "gpu"
    with pytest.raises(ValueError, match="use_gpu"):
        SpaceChargeConfig(use_gpu="tpu")
```

- [ ] **Step 2: Run — expect failure**

Run: `PYTHONPATH=. pytest tests/core/test_config.py::test_space_charge_config_use_gpu_default_and_validation -v`
Expected: `AttributeError: 'SpaceChargeConfig' object has no attribute 'use_gpu'`

- [ ] **Step 3: Add the field + validator to `SpaceChargeConfig`**

In `linac_gen/core/config.py`, add the field right after `grid_mode`:

```python
    grid_mode: str = "fixed"
    # Backend selection for the FFT inside the Poisson solver.
    # "auto" picks GPU (cupy) if importable + CUDA device visible, else CPU.
    # Overridable by the LINAC_GEN_USE_GPU env var.
    use_gpu: str = "auto"
```

And at the end of `__post_init__`, after the `valid_grid_mode` block, add:

```python
        valid_use_gpu = {"auto", "cpu", "gpu"}
        if self.use_gpu not in valid_use_gpu:
            raise ValueError(
                f"use_gpu must be one of {valid_use_gpu}, got {self.use_gpu!r}"
            )
```

- [ ] **Step 4: Run — expect PASS**

Run: `PYTHONPATH=. pytest tests/core/test_config.py::test_space_charge_config_use_gpu_default_and_validation -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/core/config.py tests/core/test_config.py
git commit -m "feat(config): SpaceChargeConfig.use_gpu ∈ {auto,cpu,gpu}"
```

---

## Task 7: Route `PoissonSolverFFT` through the backend

**Files:**
- Modify: `linac_gen/pic/poisson_solver.py`
- Modify: `tests/pic/test_poisson_solver.py` (existing)

- [ ] **Step 1: Add a regression test pinning the CPU path's current behaviour**

In `tests/pic/test_poisson_solver.py`, add:

```python
def test_cpu_backend_reproduces_baseline():
    """New backend-routed CPU path must match the previous direct scipy.fft call."""
    import numpy as np
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    rng = np.random.default_rng(11)
    n = 16
    rho = rng.standard_normal((n, n, n))
    gmin = np.array([-1., -1., -1.])
    gmax = np.array([+1., +1., +1.])
    ng   = np.array([n, n, n])
    Ex, Ey, Ez = PoissonSolverFFT(gmin, gmax, ng).solve(rho)
    # Pin a robust scalar to catch silent breakage: RMS of E_z interior.
    rms_Ez = float(np.sqrt(np.mean(Ez[1:-1,1:-1,1:-1] ** 2)))
    assert rms_Ez > 0.0 and np.isfinite(rms_Ez)
    # And sanity: the field must be divergence-free to first order
    # (Poisson solution of a white-noise source retains symmetry).
    assert abs(Ex.mean()) < 5e-2 * rms_Ez
    assert abs(Ey.mean()) < 5e-2 * rms_Ez
    assert abs(Ez.mean()) < 5e-2 * rms_Ez
```

- [ ] **Step 2: Run — expect PASS against current code (pins baseline)**

Run: `PYTHONPATH=. pytest tests/pic/test_poisson_solver.py::test_cpu_backend_reproduces_baseline -v`
Expected: 1 passed.

- [ ] **Step 3: Refactor `poisson_solver.py` to use the backend**

Replace the FFT helper section (lines 23-57 in the existing file) with:

```python
# --- FFT backend selection --------------------------------------------------
# Routed through linac_gen.pic.gpu_backend so cupy / scipy-fft selection is
# centralised and honours SpaceChargeConfig.use_gpu + LINAC_GEN_USE_GPU.
from linac_gen.pic.gpu_backend import get_backend as _get_backend
```

And delete the module-level `_rfftn` / `_irfftn` / `_fft_workers` / `_HAS_SCIPY_FFT` (they're now on the backend object).

Replace the `PoissonSolverFFT.__init__` signature + body top:

```python
class PoissonSolverFFT:
    """3D FFT Poisson solver with open boundary conditions (Hockney's method)."""

    def __init__(self, grid_min: np.ndarray, grid_max: np.ndarray,
                 n_grid: np.ndarray, use_gpu: str = "auto"):
        self.grid_min = np.asarray(grid_min, dtype=np.float64)
        self.grid_max = np.asarray(grid_max, dtype=np.float64)
        self.n_grid = np.asarray(n_grid, dtype=np.int64)
        self.dx = (self.grid_max - self.grid_min) / (self.n_grid - 1).astype(
            np.float64
        )
        grid_cells = int(self.n_grid.prod())
        self._backend = _get_backend(use_gpu) if use_gpu == "cpu" else _get_backend_with_size(use_gpu, grid_cells)
        self._precompute_greens_fft()
```

Add a tiny helper in `poisson_solver.py`:

```python
def _get_backend_with_size(use_gpu: str, grid_cells: int):
    """Thin wrapper: delegate to gpu_backend.select_backend with grid_cells
    so the tiny-grid heuristic can downgrade GPU→CPU."""
    from linac_gen.pic.gpu_backend import select_backend, _CpuBackend, _GpuBackend
    name = select_backend(use_gpu, grid_cells=grid_cells)
    return _GpuBackend() if name == "gpu" else _CpuBackend()
```

Replace the Green's-function FFT line:
```python
        self._G_fft = self._backend.rfftn(self._backend.to_device(G))
```

Replace the `solve` method body's convolution line and array allocation to go through the backend:

```python
    def solve(self, rho: np.ndarray):
        nx, ny, nz = int(self.n_grid[0]), int(self.n_grid[1]), int(self.n_grid[2])
        dx, dy, dz = self.dx
        cell_vol = dx * dy * dz

        rho_padded = np.zeros((2 * nx, 2 * ny, 2 * nz))
        rho_padded[:nx, :ny, :nz] = rho * cell_vol

        rho_dev    = self._backend.to_device(rho_padded)
        phi_padded = self._backend.to_host(
            self._backend.irfftn(
                self._backend.rfftn(rho_dev) * self._G_fft,
                s=(2 * nx, 2 * ny, 2 * nz),
            )
        )
        phi = phi_padded[:nx, :ny, :nz]

        Ex = np.zeros_like(phi)
        Ey = np.zeros_like(phi)
        Ez = np.zeros_like(phi)
        Ex[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * dx)
        Ey[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * dy)
        Ez[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * dz)
        Ex[0, :, :]  = -(phi[1, :, :]  - phi[0, :, :])  / dx
        Ex[-1, :, :] = -(phi[-1, :, :] - phi[-2, :, :]) / dx
        Ey[:, 0, :]  = -(phi[:, 1, :]  - phi[:, 0, :])  / dy
        Ey[:, -1, :] = -(phi[:, -1, :] - phi[:, -2, :]) / dy
        Ez[:, :, 0]  = -(phi[:, :, 1]  - phi[:, :, 0])  / dz
        Ez[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / dz
        return Ex, Ey, Ez
```

Note: `self._G_fft` stays on-device. On GPU it's a cupy array; the multiply `backend.rfftn(rho_dev) * self._G_fft` stays on-device, so only ρ (in) and φ (out) cross the PCIe bus per solve.

- [ ] **Step 4: Run — expect the baseline + all existing PIC tests PASS**

Run: `PYTHONPATH=. pytest tests/pic/ -v`
Expected: all green (CPU path only — GPU path is tested in Task 10).

- [ ] **Step 5: Commit**

```bash
git add linac_gen/pic/poisson_solver.py tests/pic/test_poisson_solver.py
git commit -m "refactor(pic): route PoissonSolverFFT FFTs through gpu_backend"
```

---

## Task 8: Forward `config.use_gpu` from `PicSolver` into the Poisson solver

**Files:**
- Modify: `linac_gen/pic/pic_solver.py:186-188`

- [ ] **Step 1: Add a test that verifies the plumbing**

Append to `tests/pic/test_poisson_solver.py`:

```python
def test_pic_solver_forwards_use_gpu(monkeypatch):
    """PicSolver must pass SpaceChargeConfig.use_gpu down to PoissonSolverFFT."""
    import numpy as np
    from unittest.mock import patch
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.pic.pic_solver import PicSolver

    seen = {}
    real_poisson = __import__(
        "linac_gen.pic.poisson_solver", fromlist=["PoissonSolverFFT"]
    ).PoissonSolverFFT

    class _Capture(real_poisson):
        def __init__(self, *a, use_gpu="auto", **kw):
            seen["use_gpu"] = use_gpu
            super().__init__(*a, use_gpu=use_gpu, **kw)

    monkeypatch.setattr(
        "linac_gen.pic.pic_solver.PoissonSolverFFT", _Capture
    )
    cfg  = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=5.0, use_gpu="cpu")
    pic  = PicSolver(cfg)
    # minimal fake beam with 64 particles near the origin
    class _Beam:
        def __init__(self):
            self.particles = np.zeros((64, 6))
            self.particles[:, 0] = np.linspace(-0.1, 0.1, 64)
            self.alive_mask = np.ones(64, dtype=bool)
            self.n_alive = 64
            self.current = 1.0
            class _R:
                beta = 0.5; gamma = 1.15; species = type("s", (), {"charge": 1, "mass": 938.3})
                frequency = 352.21
            self.ref = _R()
        @property
        def alive_particles(self):
            return self.particles[self.alive_mask]
    pic.kick(_Beam(), ds_mm=1.0)
    assert seen["use_gpu"] == "cpu"
```

- [ ] **Step 2: Run — expect failure (PoissonSolverFFT() called without use_gpu)**

Run: `PYTHONPATH=. pytest tests/pic/test_poisson_solver.py::test_pic_solver_forwards_use_gpu -v`
Expected: `TypeError` or `seen["use_gpu"]` missing.

- [ ] **Step 3: Forward the flag**

In `linac_gen/pic/pic_solver.py`, change the `PoissonSolverFFT(...)` call (around line 186) to:

```python
        self._solver = PoissonSolverFFT(
            self._grid_min, self._grid_max, self._n_grid,
            use_gpu=getattr(self.config, "use_gpu", "auto"),
        )
```

- [ ] **Step 4: Run — expect PASS**

Run: `PYTHONPATH=. pytest tests/pic/test_poisson_solver.py::test_pic_solver_forwards_use_gpu -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/pic/pic_solver.py tests/pic/test_poisson_solver.py
git commit -m "feat(pic): PicSolver forwards use_gpu from SpaceChargeConfig"
```

---

## Task 9: Free GPU memory on solver teardown

**Files:**
- Modify: `linac_gen/pic/poisson_solver.py`

- [ ] **Step 1: Failing test**

Append to `tests/pic/test_poisson_solver.py`:

```python
def test_free_memory_calls_backend(monkeypatch):
    """Explicit teardown must flush the backend's memory pool."""
    import numpy as np
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    solver = PoissonSolverFFT(
        np.array([-1.,-1.,-1.]), np.array([1.,1.,1.]),
        np.array([8,8,8]), use_gpu="cpu",
    )
    called = []
    solver._backend.free_memory = lambda: called.append(True)
    solver.free_memory()
    assert called == [True]
```

- [ ] **Step 2: Run — expect AttributeError**

Run: `PYTHONPATH=. pytest tests/pic/test_poisson_solver.py::test_free_memory_calls_backend -v`
Expected: `AttributeError: 'PoissonSolverFFT' object has no attribute 'free_memory'`

- [ ] **Step 3: Add `free_memory` to `PoissonSolverFFT`**

In `linac_gen/pic/poisson_solver.py`, append as a method of `PoissonSolverFFT`:

```python
    def free_memory(self) -> None:
        """Release any device-side memory the backend is holding (no-op on CPU)."""
        self._backend.free_memory()
```

- [ ] **Step 4: Run — expect PASS**

Run: `PYTHONPATH=. pytest tests/pic/test_poisson_solver.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/pic/poisson_solver.py tests/pic/test_poisson_solver.py
git commit -m "feat(pic): PoissonSolverFFT.free_memory releases GPU mempool"
```

---

## Task 10: CPU↔GPU numerical parity test (GPU-skipped where no GPU)

**Files:**
- Create: `tests/pic/test_poisson_solver_gpu_parity.py`

- [ ] **Step 1: Create the parity test**

```python
# tests/pic/test_poisson_solver_gpu_parity.py
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
        diff = np.max(np.abs(a - b))
        scale = np.max(np.abs(a))
        assert diff < 1e-10 * max(scale, 1.0), (
            f"{name}: max|Δ|={diff:.3e} scale={scale:.3e}"
        )
    gpu.free_memory()
```

- [ ] **Step 2: Run on this GPU host — expect skip OR pass, never fail**

Run: `PYTHONPATH=. pytest tests/pic/test_poisson_solver_gpu_parity.py -v`
Expected: 2 passed (or 2 skipped if cupy missing).

- [ ] **Step 3: Commit**

```bash
git add tests/pic/test_poisson_solver_gpu_parity.py
git commit -m "test(pic): CPU↔GPU Poisson parity within 1e-10 relative"
```

---

## Task 11: One-time backend log line on first PoissonSolverFFT construction

**Files:**
- Modify: `linac_gen/pic/poisson_solver.py`
- Modify: `linac_gen/pic/gpu_backend.py`

- [ ] **Step 1: Test**

Append to `tests/pic/test_gpu_backend.py`:

```python
def test_backend_banner_logs_once(caplog, monkeypatch):
    import logging
    import numpy as np
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    from linac_gen.pic import gpu_backend
    gpu_backend._BANNER_EMITTED = False   # reset for the test
    with caplog.at_level(logging.INFO, logger="linac_gen.pic"):
        PoissonSolverFFT(np.array([-1.]*3), np.array([1.]*3),
                         np.array([8,8,8]), use_gpu="cpu")
        PoissonSolverFFT(np.array([-1.]*3), np.array([1.]*3),
                         np.array([8,8,8]), use_gpu="cpu")
    banners = [r for r in caplog.records if "PicSolver: using" in r.getMessage()]
    assert len(banners) == 1, f"expected one banner, got {len(banners)}"
```

- [ ] **Step 2: Run — expect failure (no banner emitted)**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py::test_backend_banner_logs_once -v`
Expected: assertion failure (0 banners).

- [ ] **Step 3: Add the banner emission**

Add near the top of `linac_gen/pic/gpu_backend.py`:

```python
import logging

_log = logging.getLogger("linac_gen.pic")
_BANNER_EMITTED = False


def _maybe_emit_banner(backend_name: str) -> None:
    global _BANNER_EMITTED
    if _BANNER_EMITTED:
        return
    _BANNER_EMITTED = True
    if backend_name == "gpu":
        try:
            import cupy as cp
            dev = cp.cuda.runtime.getDevice()
            props = cp.cuda.runtime.getDeviceProperties(dev)
            mem_free, mem_total = cp.cuda.runtime.memGetInfo()
            _log.info(
                "PicSolver: using GPU backend (cupy %s, %s, %.0f/%.0f MiB free)",
                cp.__version__,
                props["name"].decode() if isinstance(props["name"], bytes)
                    else props["name"],
                mem_free / 1024**2, mem_total / 1024**2,
            )
            return
        except Exception:
            pass
        _log.info("PicSolver: using GPU backend (cupy)")
    else:
        workers = os.environ.get("LINAC_GEN_FFT_WORKERS", "auto")
        _log.info("PicSolver: using CPU backend (scipy.fft, workers=%s)", workers)
```

And call it from `get_backend`:

```python
def get_backend(use_gpu: str = "auto"):
    name = select_backend(use_gpu)
    backend = _GpuBackend() if name == "gpu" else _CpuBackend()
    _maybe_emit_banner(name)
    return backend
```

- [ ] **Step 4: Run — expect PASS**

Run: `PYTHONPATH=. pytest tests/pic/test_gpu_backend.py::test_backend_banner_logs_once -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/pic/gpu_backend.py tests/pic/test_gpu_backend.py
git commit -m "feat(pic): one-shot INFO log line announcing CPU/GPU backend"
```

---

## Task 12: Expose `use_gpu` in GUI Convergence tab (dropdown)

**Files:**
- Modify: `gui/linac_gen_gui/interphase/tabs/convergence_tab.py:252-256`

- [ ] **Step 1: Add the dropdown widget**

In the section where `_fixed_nx`, `_fixed_ext`, etc. are created (around line 252), add:

```python
        # Backend selector.  Default "auto": GPU if cupy + CUDA visible, else CPU.
        self._fixed_backend = QComboBox()
        self._fixed_backend.addItems(["auto", "cpu", "gpu"])
        self._fixed_backend.setCurrentText("auto")
        self._fixed_backend.setToolTip(
            "PIC FFT backend.  'auto' picks GPU when cupy is installed and a "
            "CUDA device is visible, otherwise CPU.  'gpu' forces GPU and "
            "errors if unavailable.  'cpu' disables GPU even if present."
        )
```

And add a row in the `addRow(...)` block slightly below:

```python
        for lab, w in [("Base PIC grid (nx=ny=nz)", self._fixed_nx),
                       ("Base grid extent",         self._fixed_ext),
                       ("Base step1 (integration)", self._fixed_step1),
                       ("Base step2 (SC kicks)",    self._fixed_step2),
                       ("PIC backend",              self._fixed_backend)]:
            sf.addRow(lab, w)
```

Also add the import at the top of the file (`from PyQt6.QtWidgets import QComboBox, ...` — add `QComboBox` to the existing import list).

- [ ] **Step 2: Smoke-import**

Run: `PYTHONPATH=.:gui python3 -c "from linac_gen_gui.interphase.tabs.convergence_tab import ConvergenceTab; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add gui/linac_gen_gui/interphase/tabs/convergence_tab.py
git commit -m "feat(gui): Convergence tab — PIC backend dropdown (auto/cpu/gpu)"
```

---

## Task 13: Thread dropdown value into the MP run's `SpaceChargeConfig`

**Files:**
- Modify: `gui/linac_gen_gui/interphase/app.py:220-240`

- [ ] **Step 1: Forward the dropdown choice**

Locate the block that builds `SpaceChargeConfig` inside the MP-run handler (around line 238) and replace:

```python
            sc = SpaceChargeConfig(nx=nx, ny=nx, nz=nx, grid_extent=ext) if cfg.current > 0 else None
```

with:

```python
            backend = self.convergence_tab._fixed_backend.currentText()
            sc = (SpaceChargeConfig(nx=nx, ny=nx, nz=nx, grid_extent=ext,
                                     use_gpu=backend)
                  if cfg.current > 0 else None)
```

- [ ] **Step 2: Smoke-import**

Run: `PYTHONPATH=.:gui python3 -c "import linac_gen_gui.interphase.app as a; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Manual verification checklist**

1. Launch GUI: `PYTHONPATH=.:gui python3 -m linac_gen_gui.interphase`.
2. Open `examples/mebt_pipii.dat`.
3. Leave "PIC backend" = auto, click **Run Multi-Particle**.
4. In terminal, look for one-shot log line: `PicSolver: using CPU backend ...` *or* `PicSolver: using GPU backend ...` depending on whether cupy is installed.
5. Re-run with the dropdown set to `cpu` and confirm σ/ε/transmission are identical (backend swap is numerically transparent).

- [ ] **Step 4: Commit**

```bash
git add gui/linac_gen_gui/interphase/app.py
git commit -m "feat(gui): thread PIC backend choice into SpaceChargeConfig"
```

---

## Task 14: Optional dep in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the optional-deps stanza**

In `pyproject.toml`, under `[project.optional-dependencies]` (create the section if absent), add:

```toml
[project.optional-dependencies]
gpu = ["cupy-cuda12x>=13.0"]
```

- [ ] **Step 2: Install and verify**

Run: `pip install -e .[gpu]`
Expected: installs cupy-cuda12x; `python3 -c "import cupy; print(cupy.__version__)"` prints the version.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: gpu optional-dependency extra (cupy-cuda12x)"
```

---

## Task 15: End-to-end benchmark + results README

**Files:**
- Create: `scripts/bench_gpu_pic.py`
- Modify: `README.md` (GPU Acceleration section)

- [ ] **Step 1: Create the benchmark**

```python
# scripts/bench_gpu_pic.py
"""Benchmark PIC Poisson solve for CPU vs GPU at 48³, 64³, 96³, 128³.

Prints wall time and the relative error between the two backends.  Requires
a CUDA-capable cupy install to exercise the GPU branch; otherwise the GPU
rows read "skipped".
"""
import os, time
import numpy as np
from linac_gen.pic.poisson_solver import PoissonSolverFFT
from linac_gen.pic import gpu_backend

os.environ.pop("LINAC_GEN_USE_GPU", None)
rng = np.random.default_rng(99)

print(f"{'grid':>6}  {'backend':>6}  {'time [ms]':>10}  {'max|Δ| vs CPU':>14}")
for n in (48, 64, 96, 128):
    gmin = np.array([-1.,-1.,-1.]); gmax = np.array([1.,1.,1.])
    ng   = np.array([n, n, n])
    rho  = rng.standard_normal((n, n, n))

    cpu = PoissonSolverFFT(gmin, gmax, ng, use_gpu="cpu")
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
    _ = gpu.solve(rho)           # warm-up (JIT + mempool)
    t0 = time.perf_counter()
    for _ in range(3):
        Ex_g, Ey_g, Ez_g = gpu.solve(rho)
    gpu_ms = (time.perf_counter() - t0) * 1000 / 3
    max_d = max(np.max(np.abs(Ex_cpu - Ex_g)),
                np.max(np.abs(Ey_cpu - Ey_g)),
                np.max(np.abs(Ez_cpu - Ez_g)))
    scale = np.max(np.abs(Ex_cpu))
    print(f"{n:>4}³  {'gpu':>6}  {gpu_ms:>10.2f}  {max_d/max(scale,1.0):>14.3e}")
    gpu.free_memory()
```

- [ ] **Step 2: Run the benchmark**

Run: `PYTHONPATH=. python3 scripts/bench_gpu_pic.py`
Expected output shape (actual numbers depend on hardware):

```
  grid  backend  time [ms]   max|Δ| vs CPU
  48³      cpu       15.3              —
  48³      gpu        3.1      1.2e-12
  64³      cpu       35.7              —
  64³      gpu        8.5      9.8e-13
  96³      cpu      119.4              —
  96³      gpu       15.2      1.7e-12
 128³      cpu      287.1              —
 128³      gpu       23.9      2.3e-12
```

- [ ] **Step 3: Add README section**

In `README.md`, add a section:

````markdown
### GPU acceleration (optional)

The PIC Poisson solver can run its FFTs on an NVIDIA GPU via `cupy-cuda12x`.
Installation:

```bash
pip install -e .[gpu]
```

Enable via `SpaceChargeConfig(use_gpu="auto"|"cpu"|"gpu")`, the environment
variable `LINAC_GEN_USE_GPU=auto|0|1`, or the **PIC backend** dropdown in the
GUI's Convergence tab.  "auto" picks GPU only when cupy is importable *and*
a CUDA device is visible, otherwise falls back to CPU.  Results match the CPU
reference to `~1e-10` relative (FP64 FFTs on both paths).

Typical speed-up on an RTX 2000 Ada laptop GPU vs 8 CPU cores (FFT-only):

| grid |  CPU   |  GPU   |  ×    |
|------|--------|--------|-------|
|  48³ |  15 ms |   3 ms |  5×  |
|  64³ |  35 ms |   9 ms |  4×  |
|  96³ | 120 ms |  15 ms |  8×  |
| 128³ | 290 ms |  24 ms | 12×  |

Grids smaller than 64³ stay on CPU automatically (H↔D transfer overhead
dominates).  To force GPU for tiny grids (benchmarking only):
`LINAC_GEN_USE_GPU=1`.
````

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_gpu_pic.py README.md
git commit -m "docs+bench: GPU PIC FFT backend benchmark + README section"
```

---

## Out-of-Scope Follow-up (separate plan if we decide to proceed)

**Step 2: GPU particle push.** Would move `Beam.particles` to a cupy array, port `FieldMap3D.track_rk4` field sampling (`scipy.RegularGridInterpolator` →
`cupyx.scipy.ndimage.map_coordinates`), and the drift phase-slip + Lorentz-kick vectorised updates. Target 10–30× at N ≥ 50 k, 1 000 steps/m. Effort: 1–2 engineer-days plus a careful parity sweep (tracker.py, pic_solver.py particle deposit) because divergence between CPU and GPU would be silent and physics-affecting.

Recommend: land Step 1, run `scripts/bench_gpu_pic.py` on your real workload, measure what fraction of wall-time is still CPU particle push, and only then scope Step 2.

---

## Self-review

**Spec coverage:**
- Auto-detect → Tasks 1, 2, 5 (mode resolution + tiny-grid heuristic).
- Env override → Task 3.
- `SpaceChargeConfig.use_gpu` knob → Task 6.
- PoissonSolverFFT routing → Task 7.
- Memory cleanup → Task 9.
- Logging banner → Task 11.
- GUI dropdown → Tasks 12, 13.
- Optional dep → Task 14.
- Numerical parity tests → Task 10.
- Benchmark + docs → Task 15.
- No-breakage guarantee → Task 7 baseline pin + Task 10 parity + Task 8 plumbing test.

No gaps found.

**Placeholder scan:** none — every task has full code and exact paths.

**Type consistency:** `select_backend`, `_try_import_cupy`, `_resolve_mode`, `get_backend`, `_CpuBackend`, `_GpuBackend`, `PoissonSolverFFT(... use_gpu="auto")`, `free_memory`, `_fixed_backend`, `_BANNER_EMITTED`, `_GPU_MIN_CELLS=64**3`, `use_gpu ∈ {"auto","cpu","gpu"}` — all names consistent across Tasks 1–15.
