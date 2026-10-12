"""One CPU-thread setting for every HELIX run: GUI, CLI and API.

The setting is the number of threads for

* the space-charge FFT (:class:`linac_gen.pic.gpu_backend._CpuBackend`),
* the compiled kernels' OpenMP loops -- the CIC charge deposit, the CIC and
  TSC field gathers, the field-map sampler and kick-drift step
  (``_pic_kernels``, ``_tsc_kernels``, ``_fieldmap_kernels``; the TSC
  deposit runs on one thread),
* torch's CPU thread pool (the torch space-charge paths).

It applies whatever the space-charge backend: on CUDA or MPS the FFT runs on
the GPU and the threads serve everything that stays on the CPU.

BLAS stays on one thread.  HELIX multiplies 6x6 and N x 6 matrices, too
small to split, and idle threaded-OpenBLAS workers keep spinning against the
threads above.  The GUI caps it at start-up; a run elsewhere limits it for
its duration (:func:`run_scope`).

Every result is the same for any thread count: the FFT computes each line
on its own whatever the split, the gathers and the field-map step compute
each particle on its own, the CIC deposit sums every cell in particle order
and the TSC deposit runs on one thread (tests/pic/test_deposit_threads.py,
tests/pic/test_fft_split.py, tests/pic/test_thread_independence.py,
tests/elements/test_fieldmap_step.py).

Where the number comes from, first match wins: :func:`set_threads` (the GUI
"CPU threads" control, the CLI ``--threads`` option, scripts), the
``HELIX_THREADS`` environment variable, ``OMP_NUM_THREADS`` (unless the GUI
set it -- its own cap for third-party code, marked by ``HELIX_GUI_OMP_CAP``),
every CPU the process may use.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import threading
import time

_log = logging.getLogger(__name__)

#: Environment variable read when :func:`set_threads` was not called; also
#: what :func:`set_threads` writes, so processes started later inherit it.
ENV = "HELIX_THREADS"

#: Set to "1" by the GUI when it sets OMP_NUM_THREADS itself (a cap for
#: third-party code in GUI threads): HELIX's own threads then ignore it.
GUI_OMP_CAP = "HELIX_GUI_OMP_CAP"

#: The largest count accepted (a typo such as --threads 100000 would ask
#: the OpenMP runtime for teams it cannot start).
MAX_THREADS = 1024

#: The compiled modules whose OpenMP loops take the setting.
KERNEL_MODULES = ("linac_gen._pic_kernels", "linac_gen._tsc_kernels",
                  "linac_gen._fieldmap_kernels")

#: Limit BLAS to one thread for the duration of a run (:func:`run_scope`).
LIMIT_BLAS_IN_RUNS = True

_lock = threading.Lock()
_setting: int | None = None        # None: not set here (environment, else all)
_requested: int | None = None      # what set_threads asked: None = "all"
_env_before: tuple | None = None   # (HELIX_THREADS,) before set_threads
_warned: set[str] = set()


def available_threads() -> int:
    """The number of CPUs this process may run on."""
    count = getattr(os, "process_cpu_count", None)       # Python 3.13+
    if count is not None:
        n = count()
        if n:
            return int(n)
    try:
        return len(os.sched_getaffinity(0))              # Linux
    except (AttributeError, OSError):
        return int(os.cpu_count() or 1)


def parse_threads(value) -> int | None:
    """``"all"``, ``0`` or ``None`` -> ``None`` (every CPU); an integer from
    1 to :data:`MAX_THREADS` (or its string) -> that integer.  Anything else
    raises :class:`ValueError`."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("all", "0"):
            return None
        try:
            value = int(text)
        except ValueError:
            raise ValueError(f"threads must be 'all' or a positive integer, "
                             f"got {value!r}") from None
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise ValueError(f"threads must be 'all' or a positive integer, "
                         f"got {value!r}")
    n = int(value)
    if n == 0:
        return None
    if n < 0:
        raise ValueError(f"threads must be 'all' or a positive integer, "
                         f"got {value!r}")
    if n > MAX_THREADS:
        raise ValueError(f"threads must be at most {MAX_THREADS}, got {n}")
    return n


def _resolve(n: int | None) -> int:
    return available_threads() if n is None else n


def _from_environment() -> tuple[str, int | None]:
    """(source, count asked for) from the environment."""
    raw = os.environ.get(ENV)
    if raw:
        try:
            return "env", parse_threads(raw)
        except ValueError:
            _warn_once("env", "ignoring %s=%r: not 'all' or an integer from "
                       "1 to %d", ENV, raw, MAX_THREADS)
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw and os.environ.get(GUI_OMP_CAP) != "1":
        try:
            # a list ("4,2") gives the outermost level
            n = parse_threads(raw.split(",")[0])
            if n is not None:
                return "omp", n
        except ValueError:
            pass
    return "all", None


def get_threads() -> int:
    """The thread count HELIX uses now."""
    if _setting is not None:
        return _setting
    return _resolve(_from_environment()[1])


def thread_source() -> str:
    """Where :func:`get_threads` takes its number from: ``"set"``,
    ``"env"`` (HELIX_THREADS), ``"omp"`` (OMP_NUM_THREADS) or ``"all"``."""
    if _setting is not None:
        return "set"
    return _from_environment()[0]


def requested_threads() -> int | None:
    """The count asked for -- ``None`` when it is "all" (every CPU) -- as
    opposed to :func:`get_threads`, the count in use."""
    if _setting is not None:
        return _requested
    return _from_environment()[1]


def fft_threads() -> int:
    """Threads for a CPU FFT: ``LINAC_GEN_FFT_WORKERS`` when set (scipy's
    convention, -1 = every CPU), else the setting.  (Off the main thread a
    transform smaller than ``fft_split.MIN_POINTS`` runs on one.)"""
    raw = os.environ.get("LINAC_GEN_FFT_WORKERS")
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n > 0:
            return n
        if n < 0:
            return max(1, (os.cpu_count() or 1) + 1 + n)
    return get_threads()


def set_threads(value="all") -> int:
    """Use ``value`` threads (``"all"``: every CPU) for HELIX in this process
    and in the processes it starts from now on; returns the count.

    Takes effect at once: the compiled kernels at their next call, the FFT
    at its next transform, torch in this thread (and in a run started
    later).  The results do not depend on the count.
    """
    n = parse_threads(value)
    global _setting, _requested, _env_before
    with _lock:
        if _env_before is None:
            _env_before = (os.environ.get(ENV),)
        _setting = _resolve(n)
        _requested = n
        os.environ[ENV] = "all" if n is None else str(n)
        count = _setting
    sync_kernels()
    _set_torch_threads(count)
    return count


def reset_threads() -> None:
    """Undo :func:`set_threads`: ``HELIX_THREADS`` back to what it was before
    the first call, and the count back to the environment (else every CPU)."""
    global _setting, _requested, _env_before
    with _lock:
        _setting = _requested = None
        if _env_before is not None:
            (env,) = _env_before
            if env is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = env
            _env_before = None
    sync_kernels()
    _set_torch_threads(get_threads())


def environment_without_mirror() -> dict:
    """``os.environ`` as it would be without :func:`set_threads`'s copy in
    ``HELIX_THREADS`` -- for a process that must start afresh (the GUI
    relaunching itself after an update)."""
    env = dict(os.environ)
    if _env_before is not None:
        (orig,) = _env_before
        if orig is None:
            env.pop(ENV, None)
        else:
            env[ENV] = orig
    return env


def sync_kernels() -> int:
    """Hand the current count to every compiled kernel module already
    imported; returns the count.  Importers call this right after loading a
    kernel module (they cannot be imported from here: on macOS their OpenMP
    symbols bind to the libomp that torch has loaded, so torch must come
    first)."""
    n = get_threads()
    for name in KERNEL_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        setter = getattr(mod, "set_num_threads", None)
        if setter is None:
            _warn_once(name, "%s was built before thread control "
                       "(2026-10-11): its loops use the OpenMP default "
                       "(1 thread in the GUI), and the CIC deposit's last "
                       "bits depend on the thread count.  Rebuild it.", name)
            continue
        setter(n)
    return n


def kernel_threads() -> dict:
    """Each loaded kernel module's thread count."""
    out = {}
    for name in KERNEL_MODULES:
        mod = sys.modules.get(name)
        if mod is not None:
            getter = getattr(mod, "get_num_threads", None)
            out[name.rsplit(".", 1)[1]] = (
                int(getter()) if getter is not None
                else "OpenMP default (built before thread control)")
    return out


def apply_in_this_thread() -> int:
    """Bring the calling thread up to the setting; returns the count.

    The kernels and the FFT read the setting process-wide, but torch keeps
    an OpenMP thread count per thread, so a run started in a new thread (a
    GUI worker) sets torch's count in that thread."""
    n = sync_kernels()
    _set_torch_threads(n)
    return n


def _set_torch_threads(n: int) -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.get_num_threads() != n:
            torch.set_num_threads(n)
    except Exception as exc:                       # noqa: BLE001
        _warn_once("torch", "could not set torch's thread count: %s", exc)


def _torch_threads():
    torch = sys.modules.get("torch")
    return int(torch.get_num_threads()) if torch is not None else None


@contextlib.contextmanager
def run_scope():
    """For one run: the setting applied in the calling thread, BLAS on one
    thread until the last overlapping run ends.  Yields what the run uses,
    read in its own thread (the results file records it)."""
    n = apply_in_this_thread()
    info = {"threads": n, "source": thread_source(), "fft": fft_threads(),
            "kernels": kernel_threads(), "torch": _torch_threads()}
    with _blas_one_thread() as blas:
        info["blas"] = blas
        yield info
    # kernel modules the run imported take the setting as they load
    info["kernels"] = kernel_threads()


def in_run_scope(fn):
    """Decorator: ``fn`` runs inside :func:`run_scope`; a result object that
    takes attributes gets ``threads_info``."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with run_scope() as info:
            result = fn(*args, **kwargs)
        try:
            if getattr(result, "threads_info", None) is None:
                result.threads_info = info
        except (AttributeError, TypeError):
            pass
        return result
    return wrapper


_blas_lock = threading.Lock()
_blas_depth = 0
_blas_limiter = None
_blas_controller = None             # threadpoolctl's view of the libraries
_blas_controller_time = 0.0
#: Seconds a threadpoolctl library scan is reused (a scan costs about a
#: millisecond; a BLAS library loaded later is picked up within this time).
BLAS_RESCAN_S = 5.0


@contextlib.contextmanager
def _blas_one_thread():
    """BLAS on one thread while any run is inside; yields 1, or ``None``
    when it could not be limited.  Overlapping runs (two threads) share one
    limit, restored when the last of them ends."""
    global _blas_depth, _blas_limiter
    if not LIMIT_BLAS_IN_RUNS:
        yield None
        return
    with _blas_lock:
        if _blas_depth == 0:
            _blas_limiter = _limit_blas()
        _blas_depth += 1
        limited = _blas_limiter is not None
    try:
        yield 1 if limited else None
    finally:
        with _blas_lock:
            _blas_depth -= 1
            if _blas_depth == 0 and _blas_limiter is not None:
                try:
                    _blas_limiter.restore_original_limits()
                except Exception as exc:           # noqa: BLE001
                    _warn_once("blas-restore",
                               "could not restore BLAS threads: %s", exc)
                _blas_limiter = None


def _limit_blas():
    global _blas_controller, _blas_controller_time
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        _warn_once("threadpoolctl", "threadpoolctl is not installed: BLAS "
                   "keeps its own thread count during runs")
        return None
    try:
        now = time.monotonic()
        if (_blas_controller is None
                or now - _blas_controller_time > BLAS_RESCAN_S):
            _blas_controller = ThreadpoolController()
            _blas_controller_time = now
        return _blas_controller.limit(limits=1, user_api="blas")
    except Exception as exc:                       # noqa: BLE001
        _warn_once("blas", "could not limit BLAS threads: %s", exc)
        return None


def worker_threads(n_workers: int) -> int:
    """Threads for each of ``n_workers`` processes sharing the setting:
    ``max(1, threads // n_workers)``."""
    return max(1, get_threads() // max(1, int(n_workers)))


def init_worker(threads: int) -> None:
    """Pool-worker initializer: this process uses ``threads`` threads."""
    set_threads(threads)


def _warn_once(key: str, msg: str, *args) -> None:
    if key in _warned:
        return
    _warned.add(key)
    _log.warning(msg, *args)
