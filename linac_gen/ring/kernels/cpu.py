"""The ring kernels on the CPU: numba-compiled, serial or multi-core.

:func:`backend` returns the compiled drivers of :mod:`.drivers` (built on
first use, once per process; the first build takes a few seconds), or the
same functions as plain Python when numba is not installed (then ring mode
works, slowly — ``RingBackend.compiled`` says which).

numba's multi-core launches use its own ``workqueue`` threading layer unless
NUMBA_THREADING_LAYER says otherwise: its OpenMP layer would load a second
OpenMP runtime next to the one torch brings into a HELIX process (two
runtimes abort the process).  The workqueue layer is not re-entrant, so
parallel launches are serialized by :data:`PARALLEL_LOCK`.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

PARALLEL_LOCK = threading.Lock()
_BUILD_LOCK = threading.Lock()
_BACKEND = None


@dataclass(frozen=True)
class RingBackend:
    compiled: bool                  # False: plain Python (numba not installed)
    track_turns: Callable           # serial
    track_turns_parallel: Callable  # multi-core (== track_turns without numba)
    track_probe: Callable           # optics probe: no apertures, optional record
    track_turns_tbt: Callable       # tune probes: turn by turn, serial


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def backend(*, allow_python: bool = True) -> RingBackend:
    """The compiled drivers (built once per process)."""
    global _BACKEND
    if _BACKEND is not None and (_BACKEND.compiled or allow_python):
        return _BACKEND
    with _BUILD_LOCK:
        if _BACKEND is None:
            _BACKEND = _build()
    if not _BACKEND.compiled and not allow_python:
        raise RuntimeError("ring mode needs numba for this — install it with "
                           "`pip install numba` (or the helix [ring] extra)")
    return _BACKEND


def _build() -> RingBackend:
    from linac_gen.ring.kernels import drivers
    if not numba_available():
        return RingBackend(False, drivers.track_turns, drivers.track_turns, drivers.track_probe,
                           drivers.track_turns_tbt)
    import numba
    if not os.environ.get("NUMBA_THREADING_LAYER"):
        numba.config.THREADING_LAYER = "workqueue"
    # (numpy's error model: a division by exactly zero gives inf or NaN —
    # the particle is then lost as invalid — instead of raising, as on CUDA)
    serial = numba.njit(cache=False, error_model="numpy")(drivers.track_turns)
    parallel = numba.njit(cache=False, parallel=True, error_model="numpy")(drivers.track_turns)
    probe = numba.njit(cache=False, error_model="numpy")(drivers.track_probe)
    tbt = numba.njit(cache=False, error_model="numpy")(drivers.track_turns_tbt)
    return RingBackend(True, serial, parallel, probe, tbt)


def set_threads(n: int | None) -> int:
    """Use ``n`` CPU threads for the multi-core kernels (None: all); returns
    the number in use."""
    import numba
    nmax = numba.config.NUMBA_NUM_THREADS
    numba.set_num_threads(nmax if n is None else max(1, min(int(n), nmax)))
    return numba.get_num_threads()
