"""The one HELIX thread setting (linac_gen.parallel.threads): where the
number comes from, that it reaches the compiled kernels' teams from any
thread and torch's count in a run's own thread, the share a pool worker
gets, and BLAS on one thread while runs -- even overlapping ones -- last."""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import kept_thread_setting

CSRC = Path(__file__).resolve().parents[2] / "linac_gen" / "csrc"


@pytest.fixture
def threads(monkeypatch):
    """No set_threads() in force, no HELIX_THREADS, no OMP_NUM_THREADS."""
    with kept_thread_setting() as t:
        t._setting = t._requested = t._env_before = None
        monkeypatch.delenv(t.ENV, raising=False)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.delenv(t.GUI_OMP_CAP, raising=False)
        monkeypatch.delenv("LINAC_GEN_FFT_WORKERS", raising=False)
        yield t


@pytest.mark.parametrize("value", ["all", "ALL", " all ", 0, "0", None])
def test_all_means_every_cpu(threads, value):
    assert threads.parse_threads(value) is None
    assert threads.set_threads(value) == threads.available_threads()
    assert threads.requested_threads() is None
    assert os.environ[threads.ENV] == "all"


@pytest.mark.parametrize("value", [3, "3", " 3 ", np.int64(3), 1024])
def test_a_count(threads, value):
    n = int(str(value).strip())
    assert threads.parse_threads(value) == n
    assert threads.set_threads(value) == n
    assert threads.get_threads() == n
    assert threads.requested_threads() == n
    assert os.environ[threads.ENV] == str(n)       # inherited by children


@pytest.mark.parametrize("value", [-1, "-2", "x", "", 2.5, True, "3.0", 1025,
                                   "100000"])
def test_rejected(threads, value):
    with pytest.raises(ValueError, match="positive integer|at most 1024"):
        threads.parse_threads(value)
    with pytest.raises(ValueError):
        threads.set_threads(value)


def test_where_the_number_comes_from(threads, monkeypatch, caplog):
    n_cpu = threads.available_threads()
    assert (threads.get_threads(), threads.thread_source()) == (n_cpu, "all")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")      # a user's OpenMP cap
    assert (threads.get_threads(), threads.thread_source()) == (3, "omp")
    monkeypatch.setenv("OMP_NUM_THREADS", "4,2")    # nested: the outer level
    assert threads.get_threads() == 4
    monkeypatch.setenv(threads.GUI_OMP_CAP, "1")    # the GUI's own cap
    assert (threads.get_threads(), threads.thread_source()) == (n_cpu, "all")
    monkeypatch.delenv(threads.GUI_OMP_CAP)
    monkeypatch.setenv(threads.ENV, "2")            # beats OMP_NUM_THREADS
    assert (threads.get_threads(), threads.thread_source()) == (2, "env")
    assert threads.requested_threads() == 2
    monkeypatch.setenv(threads.ENV, "all")
    assert (threads.get_threads(), threads.thread_source()) == (n_cpu, "env")
    threads.set_threads(5)                          # beats the environment
    assert (threads.get_threads(), threads.thread_source()) == (5, "set")
    threads.reset_threads()
    monkeypatch.delenv("OMP_NUM_THREADS")
    monkeypatch.setenv(threads.ENV, "lots")         # unusable: every CPU
    threads._warned.discard("env")
    with caplog.at_level(logging.WARNING, logger=threads.__name__):
        assert threads.get_threads() == n_cpu
        assert threads.get_threads() == n_cpu
    assert threads.thread_source() == "all"
    assert sum("ignoring HELIX_THREADS" in r.message
               for r in caplog.records) == 1        # said once
    monkeypatch.delenv(threads.ENV)
    monkeypatch.setenv("OMP_NUM_THREADS", "junk")   # unusable: ignored
    assert threads.thread_source() == "all"


def test_reset_undoes_set_threads(threads, monkeypatch):
    monkeypatch.setenv(threads.ENV, "2")
    threads.set_threads(6)
    threads.set_threads(5)
    assert os.environ[threads.ENV] == "5"
    assert threads.environment_without_mirror()[threads.ENV] == "2"
    threads.reset_threads()
    assert os.environ[threads.ENV] == "2"            # as before set_threads
    assert (threads.get_threads(), threads.thread_source()) == (2, "env")
    monkeypatch.delenv(threads.ENV)
    threads.set_threads(3)
    assert threads.ENV not in threads.environment_without_mirror()
    threads.reset_threads()
    assert threads.ENV not in os.environ
    assert threads.get_threads() == threads.available_threads()


def test_fft_threads(threads, monkeypatch):
    threads.set_threads(3)
    assert threads.fft_threads() == 3
    monkeypatch.setenv("LINAC_GEN_FFT_WORKERS", "2")
    assert threads.fft_threads() == 2
    monkeypatch.setenv("LINAC_GEN_FFT_WORKERS", "-1")
    assert threads.fft_threads() == (os.cpu_count() or 1)
    monkeypatch.setenv("LINAC_GEN_FFT_WORKERS", "junk")
    assert threads.fft_threads() == 3


def _loaded_kernels(threads):
    pytest.importorskip("linac_gen.pic.pic_solver")    # torch first
    import linac_gen.elements.field_map_3d  # noqa: F401
    mods = [sys.modules[m] for m in threads.KERNEL_MODULES if m in sys.modules]
    # thread control and OpenMP (a serial build always runs one thread)
    mods = [m for m in mods if hasattr(m, "set_num_threads")
            and getattr(m, "_openmp_enabled", False)]
    if not mods:
        pytest.skip("no compiled kernel with thread control and OpenMP")
    return mods


def test_kernel_teams_follow_the_setting_in_a_one_thread_worker(threads):
    """A GUI run is a worker thread whose OpenMP default is 1 (the GUI caps
    OMP_NUM_THREADS).  The kernels' teams must follow the setting there --
    the regression this change exists to fix."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    mods = _loaded_kernels(threads)
    if not all(hasattr(m, "_team_size") for m in mods):
        pytest.skip("kernels without the team-size probe")
    seen = {}

    def worker():
        with threadpoolctl.threadpool_limits(1, user_api="openmp"):
            for m in mods:
                m.set_num_threads(0)                    # the OpenMP default
            seen["default"] = [m._team_size() for m in mods]
            threads.set_threads(3)
            seen["set"] = [m._team_size() for m in mods]
            seen["reported"] = [m.get_num_threads() for m in mods]
    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert seen["default"] == [1] * len(mods)       # what the GUI used to get
    assert seen["set"] == [3] * len(mods)
    assert seen["reported"] == [3] * len(mods)
    threads.set_threads("all")
    assert [m.get_num_threads() for m in mods] == \
        [threads.available_threads()] * len(mods)


def test_every_parallel_region_names_its_team():
    """Each `#pragma omp parallel` in the kernels carries
    num_threads(...): a region without it would fall back to the calling
    thread's OpenMP default -- one thread in a GUI worker."""
    regions = []
    for src in ("pic_kernels.cpp", "tsc_kernels.cpp", "fieldmap_kernels.cpp",
                "helix_threads.h"):
        text = (CSRC / src).read_text()
        text = re.sub(r"\\\n\s*", " ", text)        # join continued pragmas
        for line in text.splitlines():
            if re.match(r"\s*#\s*pragma\s+omp\s+parallel\b", line):
                regions.append((src, line.strip()))
    assert len(regions) >= 8        # 4 CIC, 1 TSC, 2 field-map, 1 probe
    missing = [r for r in regions if "num_threads(" not in r[1]]
    assert not missing, missing


def test_kernel_rejects_a_negative_count(threads):
    mods = _loaded_kernels(threads)
    with pytest.raises(ValueError, match=">= 0"):
        mods[0].set_num_threads(-1)


def test_a_kernel_without_thread_control_is_left_alone(threads, monkeypatch,
                                                        caplog):
    old = types.ModuleType("linac_gen._tsc_kernels")   # a pre-2026-10 build
    monkeypatch.setitem(sys.modules, "linac_gen._tsc_kernels", old)
    threads._warned.discard("linac_gen._tsc_kernels")
    with caplog.at_level(logging.WARNING, logger=threads.__name__):
        threads.set_threads(2)
        threads.set_threads(3)
    assert sum("built before thread control" in r.message
               for r in caplog.records) == 1
    assert threads.kernel_threads()["_tsc_kernels"].startswith("OpenMP default")


def test_a_run_sets_torch_in_its_own_thread(threads):
    """torch keeps a count per thread: a thread that used torch before the
    setting changed keeps its old count until a run applies the setting."""
    torch = pytest.importorskip("torch")
    threads.set_threads(2)
    changed, out = threading.Event(), {}

    def run():
        torch.ones(8).sum()                         # torch at 2 in this thread
        out["before"] = torch.get_num_threads()
        changed.wait(30)
        out["stale"] = torch.get_num_threads()      # still 2
        with threads.run_scope() as info:
            out["run"] = torch.get_num_threads()
            out["info"] = info
    t = threading.Thread(target=run)
    t.start()
    while "before" not in out and t.is_alive():
        t.join(0.01)
    threads.set_threads(5)                          # from the main thread
    changed.set()
    t.join()
    assert out["before"] == 2 and out["stale"] == 2
    assert out["run"] == 5 and out["info"]["torch"] == 5


def test_run_scope_reports_what_the_run_uses(threads, monkeypatch):
    threads.set_threads(3)
    with threads.run_scope() as info:
        # a kernel module the run imports (here: appears) is listed too
        mod = types.ModuleType("linac_gen._tsc_kernels")
        mod.set_num_threads = lambda n: None
        mod.get_num_threads = lambda: 3
        monkeypatch.setitem(sys.modules, "linac_gen._tsc_kernels", mod)
    assert info["kernels"]["_tsc_kernels"] == 3
    assert info["threads"] == 3 and info["source"] == "set"
    assert info["fft"] == 3
    assert all(v == 3 for v in info["kernels"].values()
               if isinstance(v, int)) or not info["kernels"]


def test_in_run_scope_attaches_what_the_run_used(threads):
    threads.set_threads(2)

    class Result:
        pass

    @threads.in_run_scope
    def run():
        return Result()
    assert run().threads_info["threads"] == 2
    assert threads.in_run_scope(lambda: 7)() == 7   # a plain value passes


def test_pool_worker_share(threads):
    threads.set_threads(14)
    assert [threads.worker_threads(n) for n in (1, 3, 14, 100, 0)] == \
        [14, 4, 1, 1, 14]
    threads.init_worker(4)
    assert threads.get_threads() == 4


def _blas():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    counts = [d["num_threads"] for d in threadpoolctl.threadpool_info()
              if d["user_api"] == "blas"]
    if not counts:
        pytest.skip("no BLAS library visible to threadpoolctl")
    return counts


@pytest.fixture
def blas_two():
    """BLAS at 2 threads for the test, so a limit to 1 is visible."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    _blas()
    with threadpoolctl.threadpool_limits(2, user_api="blas"):
        yield


def test_run_scope_keeps_blas_on_one_thread(threads, blas_two):
    before = _blas()
    assert all(n == 2 for n in before)
    with threads.run_scope() as info:
        assert _blas() == [1] * len(before) and info["blas"] == 1
        with threads.run_scope():                   # nested: harmless
            assert _blas() == [1] * len(before)
        assert _blas() == [1] * len(before)
    assert _blas() == before                        # restored
    with pytest.raises(RuntimeError):
        with threads.run_scope():
            raise RuntimeError("a failing run")
    assert _blas() == before                        # restored after an error


def test_overlapping_runs_keep_the_limit_until_the_last_ends(threads,
                                                            blas_two):
    """Two runs in two threads (the assistant runs two jobs at once): the
    first to end must not lift the limit under the second."""
    before = _blas()
    a_in, a_out, b_in, b_done = (threading.Event() for _ in range(4))
    seen = {}

    def run_a():
        with threads.run_scope():
            a_in.set()
            b_in.wait(30)
        a_out.set()

    def run_b():
        a_in.wait(30)
        with threads.run_scope():
            b_in.set()
            a_out.wait(30)
            seen["b_after_a"] = _blas()             # A has ended
        b_done.set()
    ta, tb = threading.Thread(target=run_a), threading.Thread(target=run_b)
    ta.start(); tb.start(); ta.join(); tb.join()
    assert seen["b_after_a"] == [1] * len(before)
    assert _blas() == before


def test_package_exports():
    import linac_gen
    from linac_gen.parallel import threads
    assert linac_gen.set_threads is threads.set_threads
    assert linac_gen.get_threads is threads.get_threads
