"""Threaded 3-D real FFTs: scipy's one-thread result, bit for bit, on any
number of threads, and never scipy's own FFT thread pool.

A 3-D transform is three passes of 1-D transforms along one axis.  Each pass
is cut into slabs across another axis, and each slab is one scipy call with
``workers=1`` on a HELIX thread pool (one slab in the calling thread); the
passes come in pocketfft's order -- rfftn: r2c along the last axis, then c2c
along axis 0, then axis 1; irfftn: c2c along axis 0, then axis 1 (unscaled),
then c2r along the last axis with the whole 1/N, which pocketfft applies as
one multiplication of each output value.

Why not scipy's own threading (``workers > 1``):

* its pool is shut down and restarted around every ``fork()``; a thread that
  submits work in between gets "Work item submitted after shutdown", and
  rarer interleavings strand a work item or leave one that runs after its
  caller has gone (pocketfft ``thread_pool``, read 2026-10-11) -- a GUI run
  is a worker thread while the main thread may start subprocesses (the July
  2026 failure, commit 7bc605a);
* its result can depend on the worker count.  pocketfft transforms the lines
  of a call in vector batches (two lines at a time with NEON or SSE2) and an
  odd last line on its own, and some builds round the two paths differently
  (scipy 1.18.1's macOS arm64 wheel does; 1.17 does not).  A thread's share
  of the lines then decides which lines go alone.

The slabs here start on multiples of 8 lines (``ALIGN``; batches are 2, 4 or
8 lines wide), so every slab batches its lines exactly as the one-thread call
does and the odd last line stays last: the result is scipy's one-thread
result for any slab count, in every build.  Pinned by
tests/pic/test_fft_split.py.
"""
from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

#: Transforms with fewer grid points run as one scipy call on one thread:
#: splitting them costs more than it saves.
MIN_POINTS = 1 << 17

#: Slab boundaries fall on multiples of this many lines (a multiple of every
#: vector batch width pocketfft uses for doubles: 2, 4, 8).
ALIGN = 8

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pool_size = 0


def _executor(n: int) -> ThreadPoolExecutor:
    """A pool with at least ``n`` threads."""
    global _pool, _pool_size
    with _lock:
        if _pool is None or _pool_size < n:
            # The old pool is not shut down -- another thread may still be
            # submitting to it -- and its threads end when it is collected.
            _pool = ThreadPoolExecutor(max_workers=n,
                                       thread_name_prefix="helix-fft")
            _pool_size = n
        return _pool


def _after_fork_in_child() -> None:
    # The pool's threads do not exist in a forked child, and the lock may
    # have been held by one of the parent's threads.
    global _pool, _pool_size, _lock
    _pool, _pool_size, _lock = None, 0, threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _slabs(n: int, k: int, lines: int = 1) -> list[tuple[int, int]]:
    """Up to ``k`` slabs of ``range(n)`` planes of ``lines`` lines each,
    every boundary but the last on a multiple of :data:`ALIGN` lines."""
    step = ALIGN // math.gcd(int(lines), ALIGN)      # planes per aligned unit
    units = -(-n // step)
    k = max(1, min(int(k), units))
    b = [min(n, step * (units * i // k)) for i in range(k + 1)]
    b[-1] = n
    return [(b[i], b[i + 1]) for i in range(k) if b[i + 1] > b[i]]


def _each(fn, n: int, k: int, lines: int) -> None:
    """``fn(a, b)`` for every aligned slab of ``range(n)`` planes (``lines``
    lines each) cut up to ``k`` ways; returns when all have finished (they
    write into arrays the caller owns)."""
    slabs = _slabs(n, k, lines)
    if len(slabs) == 1:
        fn(*slabs[0])
        return
    pool = _executor(len(slabs) - 1)
    futures = [pool.submit(fn, a, b) for a, b in slabs[1:]]
    error = None
    try:
        fn(*slabs[0])
    except BaseException as exc:                    # noqa: BLE001
        error = exc
    for f in futures:
        try:
            f.result()
        except BaseException as exc:                # noqa: BLE001
            if error is None:
                error = exc
    if error is not None:
        raise error


def _store(view: np.ndarray, result: np.ndarray) -> None:
    """``view[...] = result`` unless scipy already wrote into ``view``."""
    if (result.__array_interface__["data"][0]
            != view.__array_interface__["data"][0]):
        view[...] = result


def rfftn(x, workers: int) -> np.ndarray:
    """``scipy.fft.rfftn(x)`` on ``workers`` threads, bit for bit."""
    from scipy import fft as sfft

    x = np.asarray(x)
    if (workers <= 1 or x.ndim != 3 or x.dtype != np.float64
            or x.size < MIN_POINTS):
        return sfft.rfftn(x, workers=1)
    nx, ny, nz = x.shape
    out = np.empty((nx, ny, nz // 2 + 1), dtype=np.complex128)

    def last(a, b):                     # r2c along axis 2
        out[a:b] = sfft.rfft(x[a:b], axis=2, workers=1)

    def first(a, b):                    # c2c along axis 0
        v = out[:, a:b]
        _store(v, sfft.fft(v, axis=0, overwrite_x=True, workers=1))

    def middle(a, b):                   # c2c along axis 1
        v = out[a:b]
        _store(v, sfft.fft(v, axis=1, overwrite_x=True, workers=1))

    m = nz // 2 + 1
    _each(last, nx, workers, ny)        # lines of plane i: (i, j), ny of them
    _each(first, ny, workers, m)        # lines of plane j: (j, k), m of them
    _each(middle, nx, workers, m)       # lines of plane i: (i, k), m of them
    return out


def irfftn(X, s, workers: int) -> np.ndarray:
    """``scipy.fft.irfftn(X, s=s)`` on ``workers`` threads, bit for bit."""
    from scipy import fft as sfft

    X = np.asarray(X)
    s = tuple(int(v) for v in s)
    if (workers <= 1 or X.ndim != 3 or len(s) != 3
            or X.dtype != np.complex128
            or X.shape != (s[0], s[1], s[2] // 2 + 1)
            or s[0] * s[1] * s[2] < MIN_POINTS):
        return sfft.irfftn(X, s=s, workers=1)
    nx, ny, nz = s
    tmp = np.array(X, order="C", copy=True)        # X stays as it was
    out = np.empty(s, dtype=np.float64)
    # pocketfft's normalisation factor, computed as pocketfft computes it
    fct = np.float64(np.longdouble(1) / np.longdouble(nx * ny * nz))

    def first(a, b):                    # c2c along axis 0, unscaled
        v = tmp[:, a:b]
        _store(v, sfft.ifft(v, axis=0, norm="forward", overwrite_x=True,
                            workers=1))

    def middle(a, b):                   # c2c along axis 1, unscaled
        v = tmp[a:b]
        _store(v, sfft.ifft(v, axis=1, norm="forward", overwrite_x=True,
                            workers=1))

    def last(a, b):                     # c2r along axis 2, then the 1/N
        r = sfft.irfft(tmp[a:b], n=nz, axis=2, norm="forward", workers=1)
        np.multiply(r, fct, out=out[a:b])

    m = nz // 2 + 1
    _each(first, ny, workers, m)
    _each(middle, nx, workers, m)
    _each(last, nx, workers, ny)
    return out
