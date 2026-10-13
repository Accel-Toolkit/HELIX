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

import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_log = logging.getLogger(__name__)

#: Transforms with fewer grid points run as one scipy call on one thread:
#: splitting them costs more than it saves.
MIN_POINTS = 1 << 17

#: Slab boundaries fall on multiples of this many lines (a multiple of every
#: vector batch width pocketfft uses for doubles: 2, 4, 8).
ALIGN = 8

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pool_size = 0
_local = threading.local()          # .pooled: this is one of the pool's


def _mark_pooled() -> None:
    _local.pooled = True


def _in_pool() -> bool:
    return getattr(_local, "pooled", False)


def _executor(n: int) -> ThreadPoolExecutor:
    """A pool with at least ``n`` threads."""
    global _pool, _pool_size
    with _lock:
        if _pool is None or _pool_size < n:
            # The old pool is not shut down -- another thread may still be
            # submitting to it -- and its threads end when it is collected.
            _pool = ThreadPoolExecutor(max_workers=n,
                                       thread_name_prefix="helix-fft",
                                       initializer=_mark_pooled)
            _pool_size = n
        return _pool


def _after_fork_in_child() -> None:
    # The pool's threads do not exist in a forked child, and the locks may
    # have been held by one of the parent's threads.
    global _pool, _pool_size, _lock, _prune_lock
    _pool, _pool_size, _lock = None, 0, threading.Lock()
    _prune_lock = threading.Lock()
    for key in [k for k, v in _prunes.items() if v is _BUILDING]:
        del _prunes[key]                # its builder is not in the child


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
    run_groups(fn, [[slab] for slab in _slabs(n, k, lines)])


def _run_group(fn, group) -> None:
    for a, b in group:
        fn(a, b)


def run_pieces(fn, pieces, workers: int, *, enter=None,
               caller: bool = True) -> None:
    """``fn(a, b)`` for every piece ``(a, b)``, on up to ``workers`` threads
    -- the calling thread among them unless ``caller`` is false -- each
    taking the next piece as it finishes one, so slower cores take fewer.
    A pool thread runs its pieces inside ``enter()`` when given (a context
    manager factory).  Returns when all have finished.  After an error or an
    interrupt no further piece starts; the threads finish the pieces they
    are on, and then the interrupt, else the first error, is raised.  Called
    from one of the pool's own threads (a piece that splits again), all the
    pieces run in that thread: no pool task ever waits for another."""
    pieces = list(pieces)
    workers = max(1, min(int(workers), len(pieces)))
    if _in_pool() or (workers == 1 and caller):
        if caller or enter is None:
            _run_group(fn, pieces)
        else:
            with enter():
                _run_group(fn, pieces)
        return
    lock = threading.Lock()
    queue = iter(pieces)
    errors: list[BaseException] = []

    def drain():
        try:
            while True:
                with lock:
                    piece = None if errors else next(queue, None)
                if piece is None:
                    return
                fn(*piece)
        except BaseException as exc:                # noqa: BLE001
            with lock:
                errors.append(exc)

    def pooled():
        if enter is None:
            drain()
        else:
            with enter():
                drain()

    helpers = workers - 1 if caller else workers
    pool = _executor(helpers)
    futures = [pool.submit(pooled) for _ in range(helpers)]
    if caller:
        drain()
    _wait_all(futures, errors, lock)


def _wait_all(futures, errors, lock) -> None:
    """Wait for every future, even when interrupted while waiting (the
    interrupt goes on ``errors``, which stops the threads taking pieces);
    then raise an interrupt first, else the first error."""
    for f in futures:
        while True:
            try:
                f.result()
                break
            except BaseException as exc:            # noqa: BLE001
                with lock:
                    errors.append(exc)
                if f.done():                        # the task's own error
                    break
    if errors:
        stop = [e for e in errors if not isinstance(e, Exception)]
        raise (stop or errors)[0]


def run_groups(fn, groups) -> None:
    """``fn(a, b)`` for every piece ``(a, b)`` of every group, one group per
    thread: the first in the calling thread, the others on the pool.
    Returns when all have finished; an error is raised after that (an
    interrupt first).  Called from one of the pool's own threads, all the
    groups run in that thread."""
    groups = [g for g in groups if g]
    if not groups:
        return
    if len(groups) == 1 or _in_pool():
        for g in groups:
            _run_group(fn, g)
        return
    pool = _executor(len(groups) - 1)
    futures = [pool.submit(_run_group, fn, g) for g in groups[1:]]
    errors: list[BaseException] = []
    try:
        _run_group(fn, groups[0])
    except BaseException as exc:                    # noqa: BLE001
        errors.append(exc)
    _wait_all(futures, errors, threading.Lock())


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


def irfftn(X, s, workers: int, overwrite_x: bool = False) -> np.ndarray:
    """``scipy.fft.irfftn(X, s=s)`` on ``workers`` threads, bit for bit.
    ``overwrite_x``: the caller gives ``X`` up, so a C-contiguous ``X`` is
    transformed in place instead of in a copy (the same passes on the same
    layout; ``X`` then holds intermediate values)."""
    from scipy import fft as sfft

    X = np.asarray(X)
    s = tuple(int(v) for v in s)
    if (workers <= 1 or X.ndim != 3 or len(s) != 3
            or X.dtype != np.complex128
            or X.shape != (s[0], s[1], s[2] // 2 + 1)
            or s[0] * s[1] * s[2] < MIN_POINTS):
        return sfft.irfftn(X, s=s, workers=1)
    nx, ny, nz = s
    if overwrite_x and X.flags.c_contiguous and X.flags.writeable:
        tmp = X
    else:
        tmp = np.array(X, order="C", copy=True)    # X stays as it was
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


# ---------------------------------------------------------------------------
# Pruned transforms for the Poisson solve's doubled grids.
#
# PoissonSolverFFT.solve transforms a charge that is zero outside the corner
# [:c0, :c1, :c2] of its doubled (2c0, 2c1, 2c2) grid, and keeps only that
# corner of the inverse.  A line whose input is all zeros transforms to zeros
# of a fixed sign pattern, and an output line nobody reads need not be
# computed: the forward transform skips 3/4 of its first pass and 1/2 of its
# second, the inverse 1/2 of its second and 3/4 of its last.
#
# Every line that is computed is transformed as in scipy's one-thread call.
# What decides a line's bits is only whether pocketfft takes it in a vector
# batch or alone (the lines at the end of a call that do not fill a batch).
# Every call here has a multiple of 8 lines -- J = ceil8(c1) rows, I =
# ceil8(c0) planes -- and stops before the full call's last plane, where its
# lone lines are, so its lines are batched as they are there.  The skipped
# lines' outputs, which later passes read, are the full transform's own,
# captured once per shape by transforming zeros the same way.  A self-check
# against the full transform (random data, raw bits) turns pruning off for a
# shape where they differ.  LINAC_GEN_PRUNED_FFT=0 turns it off.

#: The doubled grids pruned: from MIN_POINTS up to this many points (a beam
#: grid of up to about 100^3).  A new size's first solve also transforms
#: zeros and a random charge in full, once, to capture and check (about
#: three of its spectra in memory for a moment); larger grids keep the full
#: transforms, as the halo solver's fine grids (up to 192^3) do.
MAX_PRUNE_POINTS = 1 << 23


def _ceil8(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


class _Prune:
    """One doubled shape and charge corner: J, I, the full transform's
    outputs for the skipped lines, and the self-check's verdict ``ok``.
    ``fill_a`` (first pass, planes >= c0, rows < J) and ``fill_b`` (second
    pass, rows >= J) are kept as one line and one row-slab when every line
    (row) of the region holds the same values -- lines of zeros transform
    alike, signed zeros included -- else whole.  A new one is complete and
    off (``ok`` False) until its build says otherwise."""
    __slots__ = ("shape", "corner", "J", "I", "fill_a", "fill_b", "ok")

    def __init__(self, shape, corner):
        self.shape, self.corner = shape, corner
        self.J = self.I = 0
        self.fill_a = self.fill_b = None
        self.ok = False


_prunes: dict = {}
_prune_lock = threading.Lock()
_BUILDING = object()        # a thread is building this key's _Prune


def _prune_for(shape, corner):
    """The checked _Prune for a doubled ``shape`` and charge ``corner``
    (built on first use in this process), or None where pruning does not
    apply, failed its self-check, or is being built by another thread (the
    caller then runs the full transform; nobody waits for a build).  A
    build cut short -- an interrupt, or a MemoryError, which may pass --
    leaves no trace, and a later call builds again; any other error turns
    pruning off for the shape."""
    key = (tuple(int(v) for v in shape), tuple(int(v) for v in corner))
    mine = False
    built = None
    try:
        with _prune_lock:
            p = _prunes.get(key)
            if p is None:
                mine = True             # set first: the finally then cleans up
                _prunes[key] = _BUILDING
        if not mine:
            return None if p is _BUILDING or not p.ok else p
        try:
            built = _build_prune(*key)
        except MemoryError:
            return None
        except Exception as exc:                    # noqa: BLE001
            _log.warning("pruned FFT unavailable for the doubled grid %s "
                         "(%s); using the full transform", key[0], exc)
            built = _Prune(*key)
        return built if built.ok else None
    finally:
        if mine:
            with _prune_lock:
                if built is None:       # cut short: a later call builds it
                    if _prunes.get(key) is _BUILDING:
                        del _prunes[key]
                else:
                    _prunes[key] = built


def _compact(region, axis):
    """``region`` as its first slab along ``axis`` when every slab is the
    same bit for bit (an assignment broadcasts it back), else whole; a
    read-only copy either way."""
    first = np.take(region, [0], axis=axis)
    rb = np.ascontiguousarray(region).view(np.uint64)
    fb = np.ascontiguousarray(first).view(np.uint64)
    out = first if bool((rb == fb).all()) else np.array(region)
    out.flags.writeable = False
    return out


def _build_prune(shape, corner) -> _Prune:
    from scipy import fft as sfft

    p = _Prune(shape, corner)
    if len(shape) != 3 or len(corner) != 3 or min(corner) < 1:
        return p
    n0, n1, n2 = shape
    c0, c1, c2 = corner
    p.J, p.I = _ceil8(c1), _ceil8(c0)
    if (shape != (2 * c0, 2 * c1, 2 * c2) or p.J >= n1 or p.I >= n0
            or not MIN_POINTS <= n0 * n1 * n2 <= MAX_PRUNE_POINTS):
        return p
    # the full transform's passes over zeros, each one call over every line
    # (the second in place: its bits do not depend on that)
    o = sfft.rfft(np.zeros(shape), axis=2, workers=1)
    p.fill_a = _compact(_compact(o[c0:, :p.J], 1), 0)   # (1|n0-c0, 1|J, m)
    o = sfft.fft(o, axis=0, overwrite_x=True, workers=1)
    p.fill_b = _compact(o[:, p.J:], 1)              # (n0, 1|n1-J, m)
    del o
    # self-check: scipy's one-thread transforms of a padded random charge,
    # on this thread (no pool: a caller in the pool never waits on it),
    # each array let go as soon as it has been compared
    rng = np.random.default_rng(20261012)
    x = np.zeros(shape)
    x[:c0, :c1, :c2] = rng.standard_normal((c0, c1, c2))
    full = sfft.rfftn(x, workers=1)
    ok = np.array_equal(_rfftn_pruned(x, p, 1).view(np.uint64),
                        full.view(np.uint64))
    del x
    if ok:
        back = sfft.irfftn(full, s=shape, workers=1)[:c0, :c1, :c2].copy()
        got = _irfftn_pruned(full, p, 1, overwrite_x=True)[:, :c1, :c2]
        ok = np.array_equal(np.ascontiguousarray(got).view(np.uint64),
                            back.view(np.uint64))
    del full
    if not ok:
        _log.warning("pruned FFT differs from the full transform for the "
                     "doubled grid %s; using the full transform", shape)
    p.ok = ok
    return p


def _pruning_on() -> bool:
    return os.environ.get("LINAC_GEN_PRUNED_FFT", "1") != "0"


def rfftn_padded(x, corner, workers: int):
    """``scipy.fft.rfftn(x)``, bit for bit, on ``workers`` threads, for a
    C-contiguous float64 ``x`` that is zero outside ``x[:c0, :c1, :c2]`` of a
    doubled (2c0, 2c1, 2c2) grid -- the zeros are not read; or None where
    pruning does not apply (the caller then runs :func:`rfftn`)."""
    x = np.asarray(x)
    if (not _pruning_on() or x.ndim != 3 or x.dtype != np.float64
            or not x.flags.c_contiguous):
        return None
    p = _prune_for(x.shape, corner)
    return None if p is None else _rfftn_pruned(x, p, workers)


def irfftn_corner(X, s, corner, workers: int, overwrite_x: bool = False):
    """An array whose ``[:c0, :c1, :c2]`` is that corner of
    ``scipy.fft.irfftn(X, s=s)``, bit for bit, on ``workers`` threads; or
    None where pruning does not apply (the caller then runs :func:`irfftn`).
    ``overwrite_x`` as for :func:`irfftn`."""
    X = np.asarray(X)
    s = tuple(int(v) for v in s)
    if (not _pruning_on() or X.ndim != 3 or len(s) != 3
            or X.dtype != np.complex128
            or X.shape != (s[0], s[1], s[2] // 2 + 1)):
        return None
    p = _prune_for(s, corner)
    return None if p is None else _irfftn_pruned(X, p, workers, overwrite_x)


def _rfftn_pruned(x, p: _Prune, workers: int) -> np.ndarray:
    from scipy import fft as sfft

    n0, n1, n2 = p.shape
    c0, J = p.corner[0], p.J
    m = n2 // 2 + 1
    out = np.empty((n0, n1, m), dtype=np.complex128)

    def last(a, b):                     # r2c along axis 2: planes < c0, rows < J
        out[a:b, :J] = sfft.rfft(x[a:b, :J], axis=2, workers=1)

    def first(a, b):                    # c2c along axis 0: rows < J
        v = out[:, a:b]
        _store(v, sfft.fft(v, axis=0, overwrite_x=True, workers=1))

    def middle(a, b):                   # c2c along axis 1: every plane
        v = out[a:b]
        _store(v, sfft.fft(v, axis=1, overwrite_x=True, workers=1))

    _each(last, c0, workers, J)
    out[c0:, :J] = p.fill_a             # transforms of zero lines
    _each(first, J, workers, m)
    out[:, J:] = p.fill_b               # transforms of zero lines
    _each(middle, n0, workers, m)
    return out


def _irfftn_pruned(X, p: _Prune, workers: int, overwrite_x: bool) -> np.ndarray:
    from scipy import fft as sfft

    n0, n1, n2 = p.shape
    c0, J, I = p.corner[0], p.J, p.I
    m = n2 // 2 + 1
    if overwrite_x and X.flags.c_contiguous and X.flags.writeable:
        tmp = X
    else:
        tmp = np.array(X, order="C", copy=True)    # X stays as it was
    out = np.empty((c0, J, n2), dtype=np.float64)
    # pocketfft's normalisation factor, computed as pocketfft computes it
    fct = np.float64(np.longdouble(1) / np.longdouble(n0 * n1 * n2))

    def first(a, b):                    # c2c along axis 0, unscaled: every line
        v = tmp[:, a:b]
        _store(v, sfft.ifft(v, axis=0, norm="forward", overwrite_x=True,
                            workers=1))

    def middle(a, b):                   # c2c along axis 1, unscaled: planes < I
        v = tmp[a:b]
        _store(v, sfft.ifft(v, axis=1, norm="forward", overwrite_x=True,
                            workers=1))

    def last(a, b):                     # c2r along axis 2: planes < c0, rows < J
        r = sfft.irfft(tmp[a:b, :J], n=n2, axis=2, norm="forward", workers=1)
        np.multiply(r, fct, out=out[a:b])

    _each(first, n1, workers, m)
    _each(middle, I, workers, m)
    _each(last, c0, workers, J)
    return out
