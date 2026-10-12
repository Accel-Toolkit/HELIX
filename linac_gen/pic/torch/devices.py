"""Where the torch space-charge code runs.

:func:`torch_device` turns the ``use_gpu`` setting into the device of the
torch space-charge engine (``sc_backend="torch"``): an NVIDIA GPU, in
float64, only when one is asked for -- ``"cuda"`` or ``"gpu"``, or the
``LINAC_GEN_USE_GPU`` environment variable -- and torch can run on it.
``"auto"`` keeps the CPU: a GPU's results differ from the CPU's in the last
digits, so moving a run there is the user's decision.  Apple GPUs (MPS) have
no float64; asking for ``"mps"`` warns and keeps the CPU.

A GPU run repeats bit for bit without any switch: on CUDA torch's
``index_put_(accumulate=True)`` -- the charge deposit -- always sums each
cell in sorted order (aten/src/ATen/native/TensorAdvancedIndexing.cpp,
``index_put_with_sort_stub``), and the kick uses no operation with a
non-repeatable CUDA version.  scripts/verify_cuda.py checks it on a GPU.
"""
from __future__ import annotations

import logging
import threading

import torch

from linac_gen.parallel.threads import sync_torch as _sync_torch

# a run that loads torch here keeps its CPU work on one thread
_sync_torch()

_log = logging.getLogger(__name__)
_warned: set[str] = set()
_lock = threading.Lock()
_usable: bool | None = None         # torch can run a kernel on the GPU


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        _log.warning(msg, *args)


def cuda_available() -> bool:
    """True when torch sees a CUDA device and can run on it: a torch build
    without kernels for the card (an old GPU with a new CUDA wheel) reports
    the device but fails at the first operation -- tried once, here."""
    global _usable
    with _lock:
        if _usable is None:
            try:
                if not torch.cuda.is_available():
                    _usable = False
                else:
                    x = torch.ones(2, dtype=torch.float64, device="cuda")
                    _usable = float((x + x).sum().item()) == 4.0
            except Exception as exc:                # noqa: BLE001
                _usable = False
                _warn_once("cuda-broken", "torch sees a CUDA device but "
                           "cannot run on it (%s): the GPU is not used",
                           exc)
        return _usable


def torch_device(use_gpu: str = "auto") -> torch.device:
    """The device for ``use_gpu`` (see the module docstring)."""
    from linac_gen.pic.gpu_backend import _resolve_mode
    mode = _resolve_mode(use_gpu)           # validates; the env var wins
    if mode in ("cuda", "gpu"):
        if cuda_available():
            return torch.device("cuda")
        _warn_once("no-cuda", "use_gpu=%r: torch has no usable CUDA "
                   "device, so the torch space-charge engine runs on the "
                   "CPU", use_gpu)
    elif mode == "mps":
        _warn_once("mps", "the torch space-charge engine computes in "
                   "float64, which Apple GPUs (MPS) do not have: it runs "
                   "on the CPU")
    return torch.device("cpu")


def free_gpu_memory() -> None:
    """Hand the memory torch's CUDA cache holds back to the device (cupy's
    cache is the cupy backend's business)."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                               # noqa: BLE001
        pass
