"""PIC GPU backend selection and FFT dispatch.

Routes 3-D rfftn / irfftn through a GPU backend when one is available, and
falls back transparently to scipy.fft / numpy.fft on CPU.

Two GPU backends are supported:

* **cupy** (CUDA, FP64) — NVIDIA hardware, bit-compatible with CPU to FFT
  round-off (~1e-12 rel).
* **torch MPS** (Apple Silicon, FP32) — Metal GPU on M1/M2/M3 etc.  FP64
  is not supported by the Metal hardware, so FFT round-off is ~1e-7 rel.

Modes controlled by ``SpaceChargeConfig.use_gpu`` or the env var
``LINAC_GEN_USE_GPU``:

* ``"auto"``  – cupy (FP64) if available, else CPU.  The FP32 MPS backend
  is **never** chosen automatically: silently downgrading a physics code
  to float32 is unacceptable, and measured on Apple Silicon MPS is no
  faster than the multithreaded CPU FFT at PIC-typical grids (32³–64³).
* ``"gpu"``   – force any GPU (cupy preferred, MPS fallback with a
  precision warning), raise if none available.
* ``"cuda"``  – force cupy/CUDA, raise if cupy unavailable.
* ``"mps"``   – force torch/MPS (FP32, warns), raise if MPS unavailable.
* ``"cpu"``   – force CPU even if a GPU is present.

Environment variable takes precedence over the ``use_gpu`` argument.
"""
from __future__ import annotations

import logging
import os
from typing import Literal, Optional

Backend = Literal["cpu", "gpu"]

_log = logging.getLogger("linac_gen.pic")

# Set to True after the first backend announcement is emitted.  Tests reset
# this to False to exercise the banner again.
_BANNER_EMITTED: bool = False


def _try_import_cupy():
    """Return the cupy module if import + device count probe both succeed."""
    try:
        import cupy  # type: ignore
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        return None
    return None


def _try_import_torch_mps():
    """Return the torch module if torch is importable and MPS is available."""
    try:
        import torch  # type: ignore
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            return torch
    except Exception:
        return None
    return None


_VALID_MODES = {"auto", "cpu", "gpu", "cuda", "mps"}


_ENV_OVERRIDE_WARNED = False


def _env_mode() -> str | None:
    """Normalised LINAC_GEN_USE_GPU value, or None when unset/unrecognised."""
    env = os.environ.get("LINAC_GEN_USE_GPU")
    if env is None:
        return None
    env_norm = env.strip().lower()
    if env_norm in {"1", "true", "yes", "gpu"}:
        return "gpu"
    if env_norm in {"0", "false", "no", "cpu"}:
        return "cpu"
    if env_norm in {"auto", "cuda", "mps"}:
        return env_norm
    return None


def _resolve_mode(use_gpu: str) -> str:
    """Env override wins over the argument; normalise to {auto, cpu, gpu, cuda, mps}.

    The argument is ALWAYS validated (a junk ``use_gpu`` raises
    ``ValueError`` even when the env var would decide the outcome) —
    an invalid config should never be masked by the environment."""
    env_mode = _env_mode()
    # str() so non-string junk raises the documented ValueError below
    # instead of AttributeError on .strip().
    mode = str(use_gpu or "auto").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"use_gpu must be auto/cpu/gpu/cuda/mps, got {use_gpu!r}"
        )
    if env_mode is not None:
        if env_mode != mode:
            # An env var silently changing the configured backend is a
            # reproducibility hazard — say so ONCE, loudly.
            global _ENV_OVERRIDE_WARNED
            if not _ENV_OVERRIDE_WARNED:
                _ENV_OVERRIDE_WARNED = True
                _log.warning(
                    "LINAC_GEN_USE_GPU=%s OVERRIDES the configured "
                    "use_gpu=%r — the effective SC backend differs from "
                    "the project/config setting.",
                    os.environ.get("LINAC_GEN_USE_GPU"), use_gpu,
                )
        return env_mode
    return mode


def effective_backend_info(use_gpu: str = "auto") -> dict:
    """Provenance summary of the SC-backend resolution — what was
    requested, whether the environment overrode it, and the effective
    FFT worker setting.  Cheap: no GPU backend is constructed."""
    env_raw = os.environ.get("LINAC_GEN_USE_GPU")
    try:
        resolved = _resolve_mode(use_gpu)
    except ValueError:
        resolved = "invalid"
    return {
        "requested": str(use_gpu),
        "resolved_mode": resolved,
        "env_override": env_raw if env_raw is not None else "",
        "fft_workers": os.environ.get("LINAC_GEN_FFT_WORKERS", ""),
    }


def select_backend(use_gpu: str = "auto",
                   grid_cells: Optional[int] = None) -> Backend:
    """Return the backend string to use: 'cpu' or 'gpu'.

    ``auto`` simply means *GPU if available, else CPU*.  Previous versions
    shipped a ``grid_cells``-based crossover heuristic, but measurements on
    WSL2 showed the crossover runs the *opposite* direction from the
    classical "small grids stay on CPU" rule (PCIe transfer cost scales
    with grid size, so tiny grids actually win on GPU on this hardware),
    so we no longer try to pick for the user.  The ``grid_cells`` arg is
    retained for API compatibility but is ignored.

    Parameters
    ----------
    use_gpu : {"auto", "cpu", "gpu", "cuda", "mps"}
        User/config selection.  ``auto`` picks cupy/CUDA (FP64) when
        available, else CPU — never the FP32 MPS backend.  ``gpu``
        requires any GPU (cupy preferred, MPS accepted with a precision
        warning).  ``cuda``/``mps`` force the corresponding backend.
        Env var ``LINAC_GEN_USE_GPU`` overrides.
    grid_cells : int, optional
        Ignored — kept so existing callers don't break.
    """
    del grid_cells  # intentionally unused — backend choice is by mode only
    mode = _resolve_mode(use_gpu)
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if _try_import_cupy() is None:
            raise RuntimeError(
                "use_gpu='cuda' requested, but cupy and/or a CUDA device are "
                "not available on this host."
            )
        return "gpu"
    if mode == "mps":
        if _try_import_torch_mps() is None:
            raise RuntimeError(
                "use_gpu='mps' requested, but torch with an MPS device is not "
                "available on this host."
            )
        return "gpu"
    # auto / gpu: prefer cupy (FP64).  ``auto`` NEVER selects the FP32 MPS
    # backend — a silent float32 downgrade produced ~1e-7 field errors on
    # Apple Silicon for months without anyone noticing (found 2026-07-11),
    # and MPS measures no faster than the multithreaded CPU FFT at
    # PIC-typical grids anyway.  Explicit "gpu" accepts MPS as a fallback;
    # explicit "mps" forces it — both warn at backend construction.
    if _try_import_cupy() is not None:
        return "gpu"
    if mode == "gpu":
        if _try_import_torch_mps() is not None:
            return "gpu"
        raise RuntimeError(
            "use_gpu='gpu' (or LINAC_GEN_USE_GPU=1) requested, but neither "
            "cupy/CUDA nor torch/MPS are available on this host."
        )
    return "cpu"


# ---------------------------------------------------------------------------
# FFT backend interface
# ---------------------------------------------------------------------------

class _CpuBackend:
    """scipy.fft / numpy.fft dispatcher (FP64, multi-threaded via workers)."""

    name = "cpu"

    def __init__(self) -> None:
        import numpy as _np
        self.xp = _np
        try:
            from scipy import fft as _sfft
            self._fft = _sfft
            self._has_scipy = True
        except ImportError:
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
        # scipy.fft's multi-threaded pocketfft uses a *global* thread pool
        # that is unsafe to drive from a secondary thread: when the pool is
        # torn down elsewhere in the process it raises "Work item submitted
        # after shutdown" mid-run.  Off the main thread (GUI run-worker,
        # assistant job pool) force a single thread — the transform is
        # bit-identical regardless of worker count and the cost is
        # negligible at PIC grid sizes.  An explicit env override wins.
        import threading
        if threading.current_thread() is not threading.main_thread():
            return 1
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
        self.xp = cp
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


class _MpsBackend:
    """torch.mps dispatcher for Apple Silicon GPUs.

    Metal hardware does not support FP64, so FFTs run in FP32.  Round-off is
    therefore ~1e-7 relative (vs. ~1e-12 for the cupy backend) — fine for PIC
    where macroparticle shot noise (~1/sqrt(N)) dominates anyway, but worth
    keeping in mind for regression tests.

    ``xp`` is kept as numpy: the Green's-function build in PoissonSolverFFT
    uses numpy slicing / where / meshgrid plus elementwise math, and Apple
    unified memory means the host→device copy at FFT time is essentially free
    (no PCIe).  Only the FFT itself and the cached Green's-function spectrum
    live on the GPU.
    """

    name = "mps"

    def __init__(self) -> None:
        import numpy as _np
        import torch
        import warnings as _warnings
        # This is the only backend that does NOT compute in float64.  It is
        # reachable only by explicit opt-in (use_gpu="mps"/"gpu"), but make
        # the precision cost loud every time one is constructed.
        _warnings.warn(
            "PIC MPS backend: Metal has no float64 — space-charge FFTs run "
            "in float32 (~1e-7 relative field error; results are not "
            "float64-reproducible).  Use use_gpu='cpu' or 'cuda' for full "
            "precision.",
            UserWarning,
            stacklevel=3,
        )
        self._np = _np
        self._torch = torch
        self._dev = torch.device("mps")
        # PyTorch's MPS FFT backend internally allocates an empty
        # output tensor and lets it auto-resize to the result shape;
        # this emits a UserWarning starting with PyTorch 2.x and will
        # hard-error in some future release.  The FFT itself works
        # correctly today; suppress just this specific message so the
        # PIC log isn't littered.  Revisit when PyTorch flips the
        # default and we genuinely have to find a workaround (likely
        # downgrading to CPU FFT on MPS for the affected shapes).
        _warnings.filterwarnings(
            "ignore",
            message=r"An output with one or more elements was resized.*",
            category=UserWarning,
        )
        # Build Green's function and the padded ρ buffer on host with numpy.
        # We pay one host→device copy per ``rfftn`` call but get zero PCIe
        # cost on unified memory, so this is cheaper than maintaining a full
        # numpy-on-torch shim for the xp namespace.
        self.xp = _np

    def to_device(self, arr):
        # FP32 numpy on host; the actual MPS transfer happens inside rfftn.
        return self._np.asarray(arr, dtype=self._np.float32)

    def to_host(self, arr):
        # Accept either a torch tensor (FFT output) or numpy array.
        if hasattr(arr, "detach"):
            return arr.detach().to("cpu").numpy().astype(
                self._np.float64, copy=False
            )
        return self._np.asarray(arr, dtype=self._np.float64)

    def _to_mps(self, x):
        if isinstance(x, self._np.ndarray):
            return self._torch.from_numpy(
                x.astype(self._np.float32, copy=False)
            ).to(self._dev)
        # Already a torch tensor — make sure it's on MPS.
        if x.device != self._dev:
            return x.to(self._dev)
        return x

    def rfftn(self, x):
        t = self._to_mps(x)
        return self._torch.fft.rfftn(t)

    def irfftn(self, x, s):
        t = self._to_mps(x)
        # torch.fft.irfftn expects a sequence for ``s``.
        return self._torch.fft.irfftn(t, s=tuple(int(v) for v in s))

    def free_memory(self) -> None:
        try:
            self._torch.mps.empty_cache()
        except (AttributeError, RuntimeError):
            pass


def _maybe_emit_banner(backend_name: str) -> None:
    """Emit the one-shot INFO banner announcing the chosen backend."""
    global _BANNER_EMITTED
    if _BANNER_EMITTED:
        return
    _BANNER_EMITTED = True
    if backend_name == "gpu":
        try:
            import cupy as cp
            dev = cp.cuda.runtime.getDevice()
            props = cp.cuda.runtime.getDeviceProperties(dev)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            mem_free, mem_total = cp.cuda.runtime.memGetInfo()
            _log.info(
                "PicSolver: using GPU backend "
                "(cupy %s, %s, %.0f/%.0f MiB free)",
                cp.__version__, name,
                mem_free / 1024 ** 2, mem_total / 1024 ** 2,
            )
            return
        except Exception:
            _log.info("PicSolver: using GPU backend (cupy)")
            return
    if backend_name == "mps":
        try:
            import torch
            _log.info(
                "PicSolver: using MPS backend (torch %s, Apple Silicon Metal GPU, FP32)",
                torch.__version__,
            )
            return
        except Exception:
            _log.info("PicSolver: using MPS backend (torch)")
            return
    workers = os.environ.get("LINAC_GEN_FFT_WORKERS", "auto")
    _log.info(
        "PicSolver: using CPU backend (scipy.fft, workers=%s)", workers,
    )


def _pick_gpu_backend_class(use_gpu: str):
    """Internal: pick the concrete GPU backend class.

    Honors explicit ``cuda``/``mps`` modes; ``gpu`` falls back from cupy to
    MPS.  Assumes :func:`select_backend` already returned ``'gpu'`` — which
    for ``auto`` can only mean cupy is available (auto never selects MPS).
    """
    mode = _resolve_mode(use_gpu)
    if mode == "cuda":
        return _GpuBackend
    if mode == "mps":
        return _MpsBackend
    # gpu (explicit any-GPU request): prefer cupy, fall back to MPS.
    # auto: select_backend only returns 'gpu' when cupy exists, so the
    # fallback below is unreachable for auto — kept for the 'gpu' mode.
    if _try_import_cupy() is not None:
        return _GpuBackend
    return _MpsBackend


def get_backend(use_gpu: str = "auto"):
    """Return a constructed FftBackend object.

    Returns one of ``_CpuBackend``, ``_GpuBackend`` (cupy), or ``_MpsBackend``
    (torch MPS), chosen via :func:`select_backend` plus
    :func:`_pick_gpu_backend_class`.
    """
    name = select_backend(use_gpu)
    if name == "gpu":
        cls = _pick_gpu_backend_class(use_gpu)
        backend = cls()
    else:
        backend = _CpuBackend()
    _maybe_emit_banner(backend.name)
    return backend
