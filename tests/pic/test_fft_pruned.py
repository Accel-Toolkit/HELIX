"""The Poisson solve's pruned FFT pair (linac_gen.pic.fft_split.rfftn_padded,
irfftn_corner) gives scipy's one-thread transforms bit for bit: the forward
transform of a charge that is zero outside the corner of its doubled grid,
and that corner of the inverse.  Lines of zeros and unread lines are skipped;
every computed call has a multiple of 8 lines, so each line is batched as in
the full call -- the rule that matters in builds that round batched and lone
lines differently (scipy 1.18.1 on macOS arm64; run this file there too)."""
from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

from linac_gen.pic import fft_split

sfft = pytest.importorskip("scipy.fft")

CORNERS = [(32, 32, 32), (48, 40, 56), (33, 33, 33), (35, 41, 29),
           (26, 26, 26), (57, 31, 23), (8, 120, 80), (100, 9, 40)]


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _padded(corner, seed, holes=False):
    rng = np.random.default_rng(seed)
    x = np.zeros(tuple(2 * c for c in corner))
    block = rng.standard_normal(corner)
    if holes:                           # zero lines inside the corner too
        block[rng.random(corner) < 0.5] = 0.0
        block[:, 3] = 0.0
    x[:corner[0], :corner[1], :corner[2]] = block
    return x


@pytest.mark.parametrize("corner", CORNERS, ids=lambda c: "x".join(map(str, c)))
@pytest.mark.parametrize("holes", [False, True])
def test_pruned_pair_equals_scipy(corner, holes):
    x = _padded(corner, 1, holes)
    s = x.shape
    full = sfft.rfftn(x, workers=1)
    back = sfft.irfftn(full, s=s, workers=1)[:corner[0], :corner[1], :corner[2]]
    for w in (1, 2, 3, 5, 8, 14):
        X = fft_split.rfftn_padded(x, corner, w)
        assert X is not None
        assert np.array_equal(_bits(X), _bits(full)), w
        Y = fft_split.irfftn_corner(full, s, corner, w)
        assert np.array_equal(_bits(Y[:, :corner[1], :corner[2]]), _bits(back)), w
    kept = full.copy()
    Y = fft_split.irfftn_corner(full, s, corner, 4, overwrite_x=True)
    assert np.array_equal(_bits(Y[:, :corner[1], :corner[2]]), _bits(back))
    assert not np.array_equal(_bits(full), _bits(kept))     # used in place


def test_pruning_declines_where_it_cannot_apply(monkeypatch):
    x = _padded((32, 32, 32), 2)
    assert fft_split.rfftn_padded(np.zeros((32, 32, 32)), (16, 16, 16), 4) is None  # < MIN_POINTS
    assert fft_split.rfftn_padded(np.zeros((64, 64, 62)), (32, 32, 32), 4) is None  # not doubled
    # ceil8(4) = 8 reaches the doubled size: nothing to skip on that axis
    assert fft_split.rfftn_padded(np.zeros((8, 400, 400)), (4, 200, 200), 4) is None
    assert fft_split.rfftn_padded(np.zeros((400, 8, 400)), (200, 4, 200), 4) is None
    assert fft_split.rfftn_padded(np.asfortranarray(x), (32, 32, 32), 4) is None
    assert fft_split.rfftn_padded(x.astype(np.float32), (32, 32, 32), 4) is None
    X = sfft.rfftn(x, workers=1)
    assert fft_split.irfftn_corner(X.astype(np.complex64), x.shape, (32, 32, 32), 4) is None
    monkeypatch.setenv("LINAC_GEN_PRUNED_FFT", "0")
    assert fft_split.rfftn_padded(x, (32, 32, 32), 4) is None
    assert fft_split.irfftn_corner(X, x.shape, (32, 32, 32), 4) is None


def test_a_failed_self_check_turns_pruning_off(monkeypatch, caplog):
    corner = (40, 40, 40)
    monkeypatch.setattr(fft_split, "_prunes", {})
    real = fft_split._rfftn_pruned

    def off_by_one_bit(x, p, w):
        out = real(x, p, w)
        out.view(np.uint64).flat[7] ^= np.uint64(1)
        return out
    monkeypatch.setattr(fft_split, "_rfftn_pruned", off_by_one_bit)
    x = _padded(corner, 3)
    with caplog.at_level(logging.WARNING, logger=fft_split.__name__):
        assert fft_split.rfftn_padded(x, corner, 4) is None
        assert fft_split.rfftn_padded(x, corner, 4) is None       # decided once
    assert sum("pruned FFT differs" in r.message for r in caplog.records) == 1


def test_fills_are_the_full_transform_of_zeros():
    """The skipped lines get the full transform's outputs for lines of zeros
    (read-only, captured per shape)."""
    corner = (48, 48, 48)
    p = fft_split._prune_for(tuple(2 * c for c in corner), corner)
    assert p is not None and p.J == 48 and p.I == 48
    assert not p.fill_a.flags.writeable and not p.fill_b.flags.writeable
    # kept as one line and one row-slab (lines of zeros transform alike)
    assert p.fill_a.shape == (1, 1, 49) and p.fill_b.shape == (96, 1, 49)
    z = sfft.rfft(np.zeros(p.shape), axis=2, workers=1)
    a = np.broadcast_to(p.fill_a, z[48:, :48].shape)
    assert np.array_equal(_bits(a), _bits(z[48:, :48]))
    zb = sfft.fft(z, axis=0, workers=1)[:, 48:]
    assert np.array_equal(_bits(np.broadcast_to(p.fill_b, zb.shape)), _bits(zb))


def test_concurrent_first_use_builds_one_prune(monkeypatch):
    corner = (36, 36, 36)
    monkeypatch.setattr(fft_split, "_prunes", {})
    builds = []
    real = fft_split._build_prune
    monkeypatch.setattr(fft_split, "_build_prune",
                        lambda *a: (builds.append(1), real(*a))[1])
    x = _padded(corner, 4)
    want = sfft.rfftn(x, workers=1)
    out, errors = [], []

    def go():
        try:
            out.append(fft_split.rfftn_padded(x, corner, 3))
        except BaseException as exc:            # noqa: BLE001
            errors.append(exc)
    ts = [threading.Thread(target=go) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # one build; the threads that came while it ran got None (the caller
    # then runs the full transform), nobody waited for it
    assert not errors and len(builds) == 1
    assert all(o is None or np.array_equal(_bits(o), _bits(want)) for o in out)
    assert np.array_equal(_bits(fft_split.rfftn_padded(x, corner, 3)), _bits(want))


def test_fork_resets_the_prune_lock(monkeypatch):
    for name in ("_pool", "_pool_size", "_lock", "_prune_lock"):
        # restored afterwards: the hook drops this process's pool reference
        monkeypatch.setattr(fft_split, name, getattr(fft_split, name))
    old = fft_split._prune_lock
    old.acquire()
    try:
        fft_split._after_fork_in_child()
        assert fft_split._prune_lock is not old
        assert fft_split._prune_lock.acquire(blocking=False)
        fft_split._prune_lock.release()
    finally:
        old.release()


def test_solves_inside_the_pool_never_wait_for_a_build(monkeypatch):
    """Solves run as pieces on the FFT pool while the calling thread builds
    the same shape's pruning: the pool's solves take the full transform and
    nothing waits on anything (a build used to hold a lock the pool's
    solves needed, while waiting for the pool)."""
    import faulthandler
    import time
    from linac_gen.pic.poisson_solver import PoissonSolverFFT
    from tests.helpers import kept_thread_setting
    monkeypatch.setattr(fft_split, "_prunes", {})
    with kept_thread_setting() as threads:
        threads.set_threads(4)
        solvers = [PoissonSolverFFT(np.full(3, -1.0), np.full(3, 1.0),
                                    np.array([34, 34, 34]), use_gpu="cpu")
                   for _ in range(4)]
        rho = np.random.default_rng(5).normal(size=(34, 34, 34))
        want = solvers[0].solve(rho)              # a build ran here
        monkeypatch.setattr(fft_split, "_prunes", {})
        out = {}

        def piece(a, b):
            if fft_split._in_pool():
                time.sleep(0.02)                  # the caller builds first
            out[a] = solvers[a].solve(rho)
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            fft_split.run_pieces(piece, [(i, i + 1) for i in range(4)], 4)
        finally:
            faulthandler.cancel_dump_traceback_later()
    for got in out.values():
        for g, w in zip(got, want):
            assert np.array_equal(_bits(g), _bits(w))


@pytest.mark.parametrize("error", [KeyboardInterrupt, MemoryError, ValueError])
def test_a_build_cut_short_leaves_no_trace(error, monkeypatch):
    """An interrupt or a MemoryError during a build leaves no marker, so a
    later call builds again (a MemoryError meanwhile gives the full
    transform); any other error turns pruning off for the shape."""
    corner = (38, 38, 38)
    shape = tuple(2 * c for c in corner)
    monkeypatch.setattr(fft_split, "_prunes", {})
    real = fft_split._build_prune
    calls = []

    def failing(*a):
        calls.append(1)
        raise error("cut short")
    monkeypatch.setattr(fft_split, "_build_prune", failing)
    x = _padded(corner, 7)
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            fft_split.rfftn_padded(x, corner, 2)
    else:
        assert fft_split.rfftn_padded(x, corner, 2) is None
    key = (shape, corner)
    if error is ValueError:
        assert fft_split._prunes[key].ok is False       # off for good
    else:
        assert key not in fft_split._prunes              # no trace left
        monkeypatch.setattr(fft_split, "_build_prune", real)
        assert np.array_equal(_bits(fft_split.rfftn_padded(x, corner, 2)),
                              _bits(sfft.rfftn(x, workers=1)))
    assert len(calls) == 1
