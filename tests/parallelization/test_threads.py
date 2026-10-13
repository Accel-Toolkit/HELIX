"""The one HELIX thread setting (linac_gen.parallel.threads): where the
number comes from, that it reaches the compiled kernels' teams from any
thread, the share a pool worker gets, and torch and BLAS on one thread while
runs -- even nested or overlapping ones -- last."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from tests.helpers import kept_thread_setting

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "linac_gen" / "csrc"


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
    # CIC 4, TSC 7 (gather, the threaded deposit's one region, the torch
    # engine's 5), field map 3 (3-D sampling, the 3-D step, the 1-D step),
    # the probe 1
    assert len(regions) >= 15
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


def test_a_run_keeps_torch_on_one_thread(threads):
    """torch splits its sums by thread, so in a run's thread torch's CPU work
    runs on one thread whatever torch's own count and the setting; the run's
    end restores torch's count (a nested run leaves that to the outer one).
    set_threads leaves torch alone."""
    torch = pytest.importorskip("torch")
    out = {}

    def run():
        torch.set_num_threads(2)
        threads.set_threads(5)
        out["set"] = torch.get_num_threads()
        with threads.run_scope() as info:
            out["run"] = torch.get_num_threads()
            with threads.run_scope() as inner:
                pass
            out["nested"] = torch.get_num_threads()
        out["after"] = torch.get_num_threads()
        out["info"], out["inner"] = info, inner
    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert out["set"] == 2
    assert out["run"] == out["nested"] == 1
    assert out["info"]["torch"] == out["inner"]["torch"] == 1
    assert out["after"] == 2
    assert out["info"]["threads"] == 5


def test_sync_torch_outside_a_run_does_nothing(threads):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(3)
    threads.sync_torch()
    assert torch.get_num_threads() == 3


def _torch_count_in_a_new_thread(torch):
    """torch's default for a thread that has not used torch yet."""
    out = []
    t = threading.Thread(target=lambda: out.append(torch.get_num_threads()))
    t.start()
    t.join()
    return out[0]


def test_torch_loaded_during_a_run(threads, monkeypatch):
    """A run that is the first to load torch: the module that loads it
    calls sync_torch, and the run's end restores torch's count, in this
    thread and as torch's default for new threads."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(3)
    monkeypatch.delitem(sys.modules, "torch")       # not loaded yet
    with threads.run_scope() as info:
        assert info["torch"] is None
        monkeypatch.setitem(sys.modules, "torch", torch)   # the run loads it
        threads.sync_torch()
        inside = torch.get_num_threads()
        during = _torch_count_in_a_new_thread(torch)
    assert inside == 1 and info["torch"] == 1 and during == 1
    assert torch.get_num_threads() == 3
    assert _torch_count_in_a_new_thread(torch) == 3


def test_the_blas_limit_leaves_openmp_alone(threads):
    """Lifting the BLAS limit restores the BLAS libraries only: a limiter
    over every library also resets OpenMP's count in the lifting thread,
    torch's among them."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(3)
    limiter = threads._limit_blas()
    if limiter is None:
        pytest.skip("threadpoolctl unavailable")
    torch.set_num_threads(2)                        # changed during the run
    limiter.restore_original_limits()
    assert torch.get_num_threads() == 2


def test_every_module_that_loads_torch_holds_it_in_runs():
    """Every linac_gen module that imports torch when it loads calls
    sync_torch at load time too: whichever is the first to load torch
    inside a run then puts that run's torch work on one thread."""
    import ast
    missing = []
    for path in sorted((ROOT / "linac_gen").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack, loads, calls = list(tree.body), False, False
        while stack:                    # module level, try/if blocks included
            node = stack.pop()
            if isinstance(node, ast.Import):
                loads |= any(a.name.split(".")[0] == "torch"
                             for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                loads |= (node.module or "").split(".")[0] == "torch"
            elif (isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call)
                  and getattr(node.value.func, "id", None)
                  in ("sync_torch", "_sync_torch")):
                calls = True
            elif isinstance(node, (ast.Try, ast.If)):
                stack += node.body + node.orelse
                stack += getattr(node, "finalbody", [])
                for h in getattr(node, "handlers", []):
                    stack += h.body
        if loads and not calls:
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, missing


def _fresh_python(code):
    """Run ``code`` in a new interpreter asking torch for 3 threads by
    default; returns the words of its line that starts with TORCH."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("HELIX_THREADS", "HELIX_GUI_OMP_CAP",
                        "LINAC_GEN_FFT_WORKERS")}
    # torch's default count: OMP_NUM_THREADS, then MKL_NUM_THREADS (the
    # GUI's start-up code, imported by other tests, sets the latter to 1);
    # with Intel MKL (torch's Linux and Windows builds) MKL's own count,
    # which stays within the physical cores unless MKL_DYNAMIC is off
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "3"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                         cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300)
    line = [s for s in out.stdout.splitlines() if s.startswith("TORCH")]
    assert line, out.stdout + out.stderr
    return line[0].split()[1:]


def _torch_default():
    """torch's count in such an interpreter right after loading torch (3
    where torch honours the request; 1 on a one-core CI runner, where
    restoring it cannot be told from leaving it)."""
    return _fresh_python(
        "import torch; print('TORCH', torch.get_num_threads())")[0]


def test_a_fresh_process_loads_torch_in_its_first_run():
    """The CLI's case: nothing loads torch before the first run, whose
    space-charge solver loads it -- on one thread from then on, recorded,
    and torch's own count back afterwards."""
    pytest.importorskip("torch")
    during, after = _fresh_python("""
        import sys
        from linac_gen.core.config import SpaceChargeConfig
        from linac_gen.core.simulation import Simulation
        from tests.test_simulation_e2e import _fodo_lattice, _gaussian_beam
        assert "torch" not in sys.modules
        sim = Simulation(_fodo_lattice(),
                         _gaussian_beam(n=400, current=5.0, seed=1),
                         space_charge=SpaceChargeConfig(
                             nx=8, ny=8, nz=8, grid_extent=4.0,
                             sc_backend="torch"))
        rec = sim.run()
        import torch
        print("TORCH", rec.threads_info["torch"], torch.get_num_threads())
    """)
    assert during == "1"
    assert after == _torch_default()   # torch's own count back


def test_an_envelope_run_that_loads_torch():
    """An envelope run meeting a field-map element loads torch through
    the ML-surrogate registry: on one thread too."""
    pytest.importorskip("torch")
    during, after = _fresh_python("""
        import sys
        from linac_gen.core.lattice import Lattice
        from linac_gen.core.particle import H_MINUS
        from linac_gen.core.reference import ReferenceParticle
        from linac_gen.elements.drift import Drift
        from linac_gen.elements.ncells import NCells
        from linac_gen.tracking.envelope import EnvelopeSolver
        assert "torch" not in sys.modules
        lat = Lattice()
        lat.add(NCells("c", mode=1, n_cells=6, beta_g=0.5, eot_v_per_m=5e6,
                       theta_s_deg=0.0, aperture_mm=15, p_flag=0,
                       frequency_mhz=650.0))
        lat.add(Drift("d", 100.0))
        ref = ReferenceParticle(species=H_MINUS, w_kin=130.0,
                                frequency=650.0)
        twiss = dict(alpha_x=0.0, beta_x=1.0, emit_x=1e-6, alpha_y=0.0,
                     beta_y=1.0, emit_y=1e-6, alpha_z=0.0, beta_z=1.0,
                     emit_z=1e-6)
        res = EnvelopeSolver(lat, ref, twiss, current=5.0).run()
        assert "torch" in sys.modules
        import torch
        print("TORCH", res.threads_info["torch"], torch.get_num_threads())
    """)
    assert during == "1"
    assert after == _torch_default()   # torch's own count back


def test_error_study_runs_as_one_run(threads):
    """ErrorStudy.run: its seeds run inside one run scope, recorded on
    the results and on every seed's recorder."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.errors.error_model import ErrorStudy
    threads.set_threads(2)
    lat = Lattice()
    lat.add(Drift(name="D1", length=20.0, aperture=50.0))
    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5, current=0.0,
                     n_particles=200, emit_nx=0.2, beta_x=0.5, emit_ny=0.2,
                     beta_y=0.5, emit_z=0.06, beta_z=500.0)
    res = ErrorStudy(lat, cfg, n_seeds=2).run()
    assert res.threads_info["threads"] == 2
    assert len(res._recorders) == 2
    assert all(r.threads_info is res.threads_info for r in res._recorders)


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
    for name, v in info["kernels"].items():
        mod = sys.modules.get(f"linac_gen.{name}")
        if isinstance(v, int) and name != "_tsc_kernels":
            # a module built without OpenMP always runs one thread
            assert v == (3 if getattr(mod, "_openmp_enabled", False) else 1)


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
