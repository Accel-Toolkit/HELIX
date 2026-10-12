"""On the CPU the integrated Green's function is built in pieces of whole x
planes on HELIX's threads (``_greens_exact_cpu``, ``_greens_symmetric_cpu``
in linac_gen.pic.poisson_solver).  The kernel must be the one-call build's
BIT FOR BIT for any thread count and any cut into pieces.  The one-call
builders below are literal copies of the numpy code before the change
(2026-10-12), kept as the reference.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from linac_gen.core.constants import EPSILON_0, PI
from linac_gen.pic import fft_split
from linac_gen.pic import poisson_solver as ps
from linac_gen.pic.poisson_solver import PoissonSolverFFT, _antideriv_igf
from tests.helpers import kept_thread_setting


def _one_call_exact(n_grid, d):
    xp = np
    nx, ny, nz = int(n_grid[0]), int(n_grid[1]), int(n_grid[2])
    dx, dy, dz = d
    vx = (xp.arange(-nx, nx + 1, dtype=xp.float64) - 0.5) * dx
    vy = (xp.arange(-ny, ny + 1, dtype=xp.float64) - 0.5) * dy
    vz = (xp.arange(-nz, nz + 1, dtype=xp.float64) - 0.5) * dz
    F = _antideriv_igf(xp, vx[:, None, None], vy[None, :, None],
                       vz[None, None, :])
    H, L = slice(1, None), slice(None, -1)
    cell_vol = dx * dy * dz
    G = (
        F[H, H, H] - F[L, H, H] - F[H, L, H] - F[H, H, L]
        + F[L, L, H] + F[L, H, L] + F[H, L, L] - F[L, L, L]
    ) / cell_vol
    G *= 1.0 / (4.0 * PI * EPSILON_0)
    return xp.roll(G, (-nx, -ny, -nz), axis=(0, 1, 2))


def _one_call_symmetric(n_grid, d):
    xp = np
    nx, ny, nz = int(n_grid[0]), int(n_grid[1]), int(n_grid[2])
    dx, dy, dz = d
    vx = (xp.arange(0, nx + 2, dtype=xp.float64) - 0.5) * dx
    vy = (xp.arange(0, ny + 2, dtype=xp.float64) - 0.5) * dy
    vz = (xp.arange(0, nz + 2, dtype=xp.float64) - 0.5) * dz
    F = _antideriv_igf(xp, vx[:, None, None], vy[None, :, None],
                       vz[None, None, :])
    H, L = slice(1, None), slice(None, -1)
    cell_vol = dx * dy * dz
    G = (
        F[H, H, H] - F[L, H, H] - F[H, L, H] - F[H, H, L]
        + F[L, L, H] + F[L, H, L] + F[H, L, L] - F[L, L, L]
    ) / cell_vol
    G *= 1.0 / (4.0 * PI * EPSILON_0)
    mx = xp.concatenate([xp.arange(nx), xp.arange(nx, 0, -1)])
    my = xp.concatenate([xp.arange(ny), xp.arange(ny, 0, -1)])
    mz = xp.concatenate([xp.arange(nz), xp.arange(nz, 0, -1)])
    return G[xp.ix_(mx, my, mz)]


REFERENCE = {"exact": _one_call_exact, "symmetric": _one_call_symmetric}

# (n_grid, grid_min, grid_max): tiny, odd and prime axes, unequal cells,
# bounds that are not multiples of the cell size
GRIDS = [
    ((2, 2, 2), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    ((3, 5, 7), (-1.3e-3, -0.7e-3, -2.1e-3), (1.1e-3, 0.9e-3, 1.7e-3)),
    ((13, 17, 11), (-3.71, -1.13, -9.07), (4.29, 2.87, 8.93)),
    ((16, 16, 32), (-2e-3, -2e-3, -9e-3), (2e-3, 2e-3, 9e-3)),
    ((32, 32, 32), (-1.5, -2.1, -0.8), (1.5, 2.1, 0.8)),
    ((47, 53, 41), (-1.3e-3, -0.7e-3, -2.1e-3), (1.1e-3, 0.9e-3, 1.7e-3)),
]


def _build(solver, build):
    return (solver._build_integrated_greens() if build == "exact"
            else solver._build_integrated_greens_symmetric())


def _bits(a):
    return np.ascontiguousarray(a).view(np.int64)


@pytest.mark.parametrize("build", ["exact", "symmetric"])
@pytest.mark.parametrize("n,lo,hi", GRIDS)
def test_the_threaded_build_is_the_one_call_build(n, lo, hi, build,
                                                  monkeypatch):
    s = PoissonSolverFFT(np.array(lo), np.array(hi), np.array(n),
                         use_gpu="cpu", green_build=build)
    ref = REFERENCE[build](s.n_grid, s.dx)
    with kept_thread_setting() as threads:
        for t in (1, 3, 8):
            threads.set_threads(t)
            # one plane per piece, an odd size, the default
            for points in ((1, 1), (777, 777), ps.GREEN_PIECE_POINTS):
                monkeypatch.setattr(ps, "GREEN_PIECE_POINTS", points)
                G = _build(s, build)
                assert G.shape == ref.shape and G.dtype == ref.dtype
                assert np.array_equal(_bits(G), _bits(ref)), (t, points)


def test_the_pieces_are_whole_planes_cut_by_the_grid_alone(monkeypatch):
    # about GREEN_PIECES pieces, within the point limits, whole planes
    monkeypatch.setattr(ps, "GREEN_PIECES", 6)
    monkeypatch.setattr(ps, "GREEN_PIECE_POINTS", (1000, 5000))
    assert ps._step(97, 97 * 97) == 1       # a plane is more than a piece
    assert ps._step(60, 100) == 10          # 6000 points: 6 pieces
    assert ps._step(6, 100) == 10           # at least 1000 points
    assert ps._step(600, 100) == 50         # at most 5000 points
    assert ps._pieces(10, 33, 10) == [(10, 20), (20, 30), (30, 33)]
    assert ps._pieces(5, 5, 10) == []
    with kept_thread_setting() as threads:   # the thread count plays no part
        threads.set_threads(1)
        one = ps._step(50, 77)
        threads.set_threads(7)
        assert ps._step(50, 77) == one


@pytest.mark.parametrize("build", ["exact", "symmetric"])
def test_the_pieces_are_shared_out_among_the_threads(build, monkeypatch):
    seen = []
    real = fft_split.run_pieces

    def spy(fn, pieces, workers, **kw):
        ran = []
        lock = threading.Lock()

        def record(a, b):
            with lock:
                ran.append((a, b, threading.get_ident()))
            return fn(a, b)
        real(record, pieces, workers, **kw)
        seen.append((list(pieces), workers, ran))
    monkeypatch.setattr(fft_split, "run_pieces", spy)
    monkeypatch.setattr(ps, "GREEN_PIECE_POINTS", (1 << 12, 1 << 12))
    s = PoissonSolverFFT(np.full(3, -1.0), np.full(3, 1.0),
                         np.array([24, 24, 24]), use_gpu="cpu",
                         green_build=build)
    with kept_thread_setting() as threads:
        threads.set_threads(4)
        seen.clear()
        _build(s, build)
    assert len(seen) == 2                   # the vertices, then the cells
    for pieces, workers, ran in seen:
        assert workers == 4 and len(pieces) >= 4
        # every plane once, in runs that do not depend on the threads, and
        # every piece run exactly once
        ordered = sorted(pieces)
        assert ordered[0][0] == 0
        assert all(a[1] == b[0] for a, b in zip(ordered, ordered[1:]))
        assert sorted((a, b) for a, b, _ in ran) == ordered


def test_run_pieces_runs_each_piece_once_on_at_most_the_threads():
    ran, threads_seen = [], set()
    lock = threading.Lock()

    def fn(a, b):
        with lock:
            ran.append((a, b))
            threads_seen.add(threading.get_ident())
    pieces = [(a, a + 1) for a in range(50)]
    fft_split.run_pieces(fn, pieces, 3)
    assert sorted(ran) == pieces
    assert 1 <= len(threads_seen) <= 3
    ran.clear()
    fft_split.run_pieces(fn, [], 3)         # nothing to do
    fft_split.run_pieces(fn, pieces[:1], 8)
    assert ran == [(0, 1)]


def test_run_pieces_stops_after_an_error_and_raises_it():
    started = []
    lock = threading.Lock()

    def fn(a, b):
        with lock:
            started.append(a)
        if a == 2:
            raise ZeroDivisionError("piece 2")
    with pytest.raises(ZeroDivisionError, match="piece 2"):
        fft_split.run_pieces(fn, [(a, a + 1) for a in range(1000)], 4)
    assert 2 in started and len(started) < 1000


def test_a_worker_thread_and_the_main_thread_give_the_same_spectrum():
    """update_grid from a GUI-style worker thread on 4 threads keeps the
    spectrum the main thread gets on one."""
    n = np.array([40, 36, 52])
    lo, hi = np.array([-3.7, -1.1, -9.0]), np.array([4.3, 2.9, 8.9])
    with kept_thread_setting() as threads:
        threads.set_threads(1)
        a = PoissonSolverFFT(lo, hi, n, use_gpu="cpu")
        a.update_grid(lo * 1.01, hi)
        threads.set_threads(4)
        b = PoissonSolverFFT(lo, hi, n, use_gpu="cpu")
        t = threading.Thread(target=b.update_grid, args=(lo * 1.01, hi))
        t.start()
        t.join()
    assert b.n_green_builds == a.n_green_builds == 2
    assert np.array_equal(_bits(a._G_fft), _bits(b._G_fft))


def test_a_piece_that_fails_fails_the_build_and_the_rebuild_is_retried(
        monkeypatch):
    s = PoissonSolverFFT(np.full(3, -1.0), np.full(3, 1.0),
                         np.array([24, 24, 24]), use_gpu="cpu")
    before = s._G_fft.copy()
    real = ps._antideriv_into
    calls = []
    lock = threading.Lock()

    def flaky(*args):
        with lock:
            calls.append(None)
            k = len(calls)
        if k == 3:
            raise FloatingPointError("piece 3")
        return real(*args)
    monkeypatch.setattr(ps, "_antideriv_into", flaky)
    monkeypatch.setattr(ps, "GREEN_PIECE_POINTS", (1 << 10, 1 << 10))
    with kept_thread_setting() as threads:
        threads.set_threads(4)
        with pytest.raises(FloatingPointError, match="piece 3"):
            s.update_grid(np.full(3, -2.0), np.full(3, 2.0))
        assert np.array_equal(s._G_fft, before)     # the old kernel kept
        monkeypatch.setattr(ps, "_antideriv_into", real)
        s.update_grid(np.full(3, -2.0), np.full(3, 2.0))   # retried
    fresh = PoissonSolverFFT(np.full(3, -2.0), np.full(3, 2.0),
                             np.array([24, 24, 24]), use_gpu="cpu")
    assert np.array_equal(s._G_fft, fresh._G_fft)


def test_run_pieces_runs_pieces_on_several_threads_at_once():
    """Two pieces that each wait for the other finish only if two threads
    run them at the same time."""
    barrier = threading.Barrier(2, timeout=30)
    fft_split.run_pieces(lambda a, b: barrier.wait(), [(0, 1), (1, 2)], 2)


def test_the_threaded_build_keeps_the_caller_s_numpy_error_handling():
    """numpy keeps its error handling per thread: the pieces on the pool's
    threads follow the caller's."""
    import warnings
    s = PoissonSolverFFT(np.full(3, -1.0), np.full(3, 1.0),
                         np.array([24, 24, 24]), use_gpu="cpu")
    s.dx = np.array([s.dx[0], s.dx[1], 0.0])   # a flat axis: 0/0 in pieces
    with kept_thread_setting() as threads:
        threads.set_threads(4)
        with np.errstate(all="raise"), pytest.raises(FloatingPointError):
            s._build_integrated_greens()
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("error")
            s._build_integrated_greens()


def test_an_interrupt_stops_the_pieces_and_wins_over_an_error():
    started = []
    lock = threading.Lock()

    def fn(a, b):
        with lock:
            started.append(a)
        if a == 1:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        fft_split.run_pieces(fn, [(a, a + 1) for a in range(1000)], 4)
    assert len(started) < 1000
    # an interrupt is raised before an error found earlier
    with pytest.raises(KeyboardInterrupt):
        fft_split._wait_all([], [ValueError("x"), KeyboardInterrupt()],
                            threading.Lock())


def test_a_piece_that_splits_again_does_not_wait_on_the_pool():
    """Pieces that call run_pieces themselves (the pool's threads run the
    inner pieces in place) finish instead of deadlocking."""
    inner_done = []
    lock = threading.Lock()

    def inner(a, b):
        with lock:
            inner_done.append(a)

    def outer(a, b):
        fft_split.run_pieces(inner, [(10 * a + k, 10 * a + k + 1)
                                     for k in range(10)], 4)
    t = threading.Thread(target=fft_split.run_pieces,
                         args=(outer, [(a, a + 1) for a in range(8)], 4))
    t.start()
    t.join(60)
    assert not t.is_alive()
    assert sorted(inner_done) == list(range(80))
