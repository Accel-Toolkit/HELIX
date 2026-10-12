"""Every HELIX process pool hands each worker its share of the thread
setting, max(1, threads // running workers), through its initializer -- so
N workers never run N teams of every CPU -- and each initializer applies
it in the worker."""
from __future__ import annotations

import pytest

from linac_gen.parallel.threads import init_worker
from tests.helpers import kept_thread_setting


class _Captured(Exception):
    pass


def _capture(monkeypatch, module):
    """Replace the module's ProcessPoolExecutor: record its keyword
    arguments, then stop before any process starts."""
    seen = {}

    class _Pool:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            raise _Captured
    monkeypatch.setattr(module, "ProcessPoolExecutor", _Pool)
    return seen


def test_scan_pool(monkeypatch):
    from linac_gen.parallel import scan_pool
    seen = _capture(monkeypatch, scan_pool)
    pts = [scan_pool.ScanPoint(lattice_path="x.dat", beam_config={}, nx=8,
                               grid_extent=4.0, step1=10.0, step2=5.0,
                               use_gpu="cpu") for _ in range(3)]
    with kept_thread_setting() as threads:
        threads.set_threads(12)
        with pytest.raises(_Captured):
            scan_pool.run_scan_points(pts, max_workers=6)
    assert seen["initializer"] is init_worker
    assert seen["initargs"] == (4,)          # 12 threads, 3 points running


def test_reliability_pool(monkeypatch):
    from linac_gen.reliability import comp_pool
    seen = _capture(monkeypatch, comp_pool)
    with kept_thread_setting() as threads:
        threads.set_threads(12)
        with pytest.raises(_Captured):
            comp_pool.run_jobs(abs, [1, 2], max_workers=4)
    assert seen["initializer"] is init_worker
    assert seen["initargs"] == (6,)          # 12 threads, 2 jobs running


def test_error_ensemble(monkeypatch):
    from linac_gen.errors import ensemble
    seen = _capture(monkeypatch, ensemble)
    with kept_thread_setting() as threads:
        threads.set_threads(12)
        with pytest.raises(_Captured):
            ensemble.run_ensemble(object(), [1, 2, 3], design={"x": 1},
                                  workers=8)
        assert seen["initializer"] is ensemble._init_worker
        assert seen["initargs"] == (4,)      # 12 threads, 3 seeds running
        ensemble._init_worker(3)
        assert threads.get_threads() == 3


def test_cmaes_worker():
    from linac_gen.core.lattice import Lattice
    from linac_gen.matching import engine
    try:
        with kept_thread_setting() as threads:
            engine._cmaes_pool_init(Lattice(), None, [], {}, False,
                                    "envelope", None, 5)
            assert threads.get_threads() == 5
    finally:
        engine._PARALLEL_STATE.clear()


def test_surrogate_training_worker():
    from linac_gen.surrogates import training
    try:
        with kept_thread_setting() as threads:
            training._worker_init(object(), None, [], threads=3)
            assert threads.get_threads() == 3
    finally:
        training._WORKER_STATE.clear()
