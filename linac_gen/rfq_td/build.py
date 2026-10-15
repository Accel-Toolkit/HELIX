"""Build (or fetch from the cache) the field map of an RFQ.

The windows are solved in separate processes (spawn context, one BLAS
thread each, the scan-pool pattern of ``linac_gen.parallel``); a map
that exists on disk under :func:`~linac_gen.rfq_td.fieldmap.cache_dir`
with the same content key is loaded instead, and maps used in this
process stay in memory.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import socket
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from linac_gen.rfq_td.fieldmap import RfqFieldMap, cache_dir, map_key
from linac_gen.rfq_td.laplace import plan_windows, solve_block

_log = logging.getLogger(__name__)

#: field-map settings (all part of the cache key)
DEFAULT_SETTINGS = dict(h=0.1, hz_per_cell=32, min_core_mm=40.0, end_hz=0.2,
                        upstream_mm=20.0, downstream_mm=20.0, x_max=16.0,
                        x_store_min=8.0, x_store_max=12.0, pad_mm=20.0)

_MEMORY: dict = {}            # key -> RfqFieldMap (this process)
_MEMORY_MAX = 2
#: maps on disk beyond this many bytes are pruned, least recently used first
#: (``HELIX_RFQTD_CACHE_GB`` overrides)
CACHE_BUDGET_BYTES = 4 * 1024 ** 3
#: a window whose Laplace solve ends above this relative residual is an error
MAX_REL_RESIDUAL = 1e-6


def _solve_task(args):
    electrodes, win, h, pad = args
    info = {}
    # BLAS on one thread: N solver processes x a threaded BLAS each
    # oversubscribes the machine many times over
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        z_first, hz, phi = solve_block(electrodes, win, h=h, pad_mm=pad,
                                       info=info)
    return z_first, hz, phi, info


def _pid_alive(pid: int) -> bool:
    """Whether process ``pid`` exists on this machine (never signals it)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL,
                                    wintypes.DWORD)
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE,
                                           ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        h = k32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5    # access denied: it exists
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True                          # unknown: assume alive
            return code.value == 259                # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_owner(lock: Path):
    """(pid, host) written in a lock file, or None."""
    try:
        text = lock.read_text().strip()
    except OSError:
        return None
    pid, _, host = text.partition("@")
    try:
        return int(pid), (host or socket.gethostname())
    except ValueError:
        return None


def _lock(path: Path, timeout_s: float = 3600.0, should_abort=None,
          ready=None):
    """Cross-process lock: an exclusive lock file next to the map holding
    ``pid@host``.  A lock whose owner has died (on this host), or older
    than ``timeout_s``, is stale and taken over.  Waiting is logged, ends
    when ``should_abort()`` turns true, and returns None (no lock) as soon
    as ``ready()`` is true -- the map another process was building can be
    loaded."""
    from linac_gen.rfq_td.tracker import Aborted
    lock = path.with_suffix(path.suffix + ".lock")
    me = f"{os.getpid()}@{socket.gethostname()}"
    t0 = time.time()
    told = False
    polls = 0
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return lock
        except FileExistsError:
            pass
        owner = _lock_owner(lock)
        try:
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            continue
        dead = (owner is not None and owner[1] == socket.gethostname()
                and not _pid_alive(owner[0]))
        if dead or age > timeout_s:
            _log.warning("time-domain RFQ: removing a stale field-map lock "
                         "%s (owner %s, %.0f s old)", lock,
                         "%d@%s" % owner if owner else "?", age)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:                 # e.g. Windows: in use
                _log.warning("time-domain RFQ: cannot remove the stale "
                             "lock %s (%s)", lock, exc)
                time.sleep(0.5)
            continue
        polls += 1
        if ready is not None and polls % 4 == 0 and ready():
            return None
        if not told:
            _log.info("time-domain RFQ: waiting for the field map that "
                      "process %s is building", "%d@%s" % owner
                      if owner else "?")
            told = True
        if should_abort is not None and should_abort():
            raise Aborted("field-map wait aborted")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"waited {timeout_s:.0f} s for {lock}")
        time.sleep(0.5)


def _unlock(lock: Path) -> None:
    """Remove ``lock`` if this process still owns it (a waiter that took
    it over as stale owns it now)."""
    owner = _lock_owner(lock)
    if owner is not None and (owner[0], owner[1]) != (os.getpid(),
                                                      socket.gethostname()):
        return
    try:
        lock.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:                         # e.g. Windows: in use
        _log.warning("time-domain RFQ: cannot remove the lock %s (%s); "
                     "waiting processes load the map when it is complete",
                     lock, exc)


def _map_path(key: str) -> Path:
    return cache_dir() / f"map-{key[:32]}.npz"


def _try_load(path: Path, key: str, quiet: bool = False):
    """The cached map at ``path`` if it holds ``key``; None if missing,
    unreadable (logged unless ``quiet``; it is rebuilt) or another key's."""
    if not path.exists():
        return None
    try:
        fmap = RfqFieldMap.load(path)
    except Exception as exc:                          # corrupt: rebuild
        if not quiet:
            _log.warning("time-domain RFQ: cached field map %s unreadable "
                         "(%s) -- rebuilding", path, exc)
        return None
    if fmap.key != key:
        return None
    try:
        os.utime(path)                                # recently used
    except OSError:
        pass
    return fmap


def map_cached(electrodes, edges_mm, settings: dict | None = None) -> bool:
    """Whether the map of ``electrodes`` is in memory or on disk."""
    st = dict(DEFAULT_SETTINGS)
    st.update(settings or {})
    key = map_key(electrodes, edges_mm, st)
    return key in _MEMORY or _map_path(key).exists()


def cache_writable() -> str | None:
    """None if field maps can be written to the cache directory, else
    why not."""
    import tempfile
    d = cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".probe")
        os.close(fd)
        os.unlink(tmp)
    except OSError as exc:
        return f"{d}: {exc}"
    return None


def get_field_map(electrodes, edges_mm, *, settings: dict | None = None,
                  workers: int | None = None,
                  should_abort=None) -> RfqFieldMap:
    """The field map of ``electrodes`` for a chain with cell edges
    ``edges_mm`` (chain coordinates): from memory, from disk, or built.
    A build reports its progress on this module's logger (INFO, one line
    per window; the GUI shows them in the status bar)."""
    st = dict(DEFAULT_SETTINGS)
    st.update(settings or {})
    key = map_key(electrodes, edges_mm, st)
    if key in _MEMORY:
        return _MEMORY[key]
    path = _map_path(key)
    fmap = _try_load(path, key)
    if fmap is None:
        if getattr(multiprocessing.current_process(), "_inheriting", False):
            # a spawned worker importing its parent's main script, which
            # ran a simulation at module level: building here would start
            # processes during bootstrap (and wait on its parent's lock)
            raise RuntimeError(
                "a time-domain RFQ run was started while a worker process "
                "imported the main script: protect the script's top level "
                "with  if __name__ == '__main__':")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = _lock(path, should_abort=should_abort,
                     ready=lambda: _try_load(path, key, quiet=True)
                     is not None)
        try:
            fmap = _try_load(path, key)     # built while we waited?
            if fmap is None:
                fmap = _build(electrodes, edges_mm, st, key, workers,
                              should_abort)
                fmap.save(path)
                _log.info("time-domain RFQ: field map cached at %s", path)
                _prune(path)
        finally:
            if lock is not None:
                _unlock(lock)
    if len(_MEMORY) >= _MEMORY_MAX:
        _MEMORY.pop(next(iter(_MEMORY)))
    _MEMORY[key] = fmap
    return fmap


def _prune(keep: Path) -> None:
    """Delete cached maps, least recently used first, until the cache is
    within its budget; never ``keep``, never a map in use elsewhere (a
    failed delete is skipped)."""
    env = os.environ.get("HELIX_RFQTD_CACHE_GB", "").strip()
    try:
        budget = float(env) * 1024 ** 3 if env else CACHE_BUDGET_BYTES
    except ValueError:
        budget = CACHE_BUDGET_BYTES
    maps = []
    for f in keep.parent.glob("map-*.npz"):
        try:
            st = f.stat()
        except OSError:
            continue
        maps.append((st.st_mtime, st.st_size, f))
    total = sum(m[1] for m in maps)
    for _mtime, size, f in sorted(maps):
        if total <= budget:
            break
        if f == keep:
            continue
        try:
            f.unlink()
            total -= size
            _log.info("time-domain RFQ: pruned cached field map %s", f)
        except OSError:
            pass


def _build(electrodes, edges_mm, st, key, workers, should_abort):
    from linac_gen.parallel.threads import init_worker, worker_threads
    from linac_gen.rfq_td.tracker import Aborted     # the tracker's stop
    wins = plan_windows(edges_mm, electrodes, hz_per_cell=st["hz_per_cell"],
                        min_core_mm=st["min_core_mm"], end_hz=st["end_hz"],
                        upstream_mm=st["upstream_mm"],
                        downstream_mm=st["downstream_mm"],
                        x_max=st["x_max"], x_store_min=st["x_store_min"],
                        x_store_max=st["x_store_max"])
    n = len(wins)
    env = os.environ.get("HELIX_RFQTD_WORKERS", "").strip()
    nw = workers or (int(env) if env.isdigit() and int(env) > 0 else 0) \
        or max(1, min(8, (os.cpu_count() or 2) - 2, n))
    if nw > 1 and multiprocessing.current_process().daemon:
        # a multiprocessing.Pool worker (e.g. parallel CMA-ES matching)
        # may not start processes of its own
        nw = 1
    _log.info("time-domain RFQ: building the field map (%d windows, %d "
              "processes); cached afterwards", n, nw)
    t0 = time.time()
    tasks = [(electrodes, w, st["h"], st["pad_mm"]) for w in wins]
    results = [None] * n
    if nw == 1:
        for i, t in enumerate(tasks):
            if should_abort is not None and should_abort():
                raise Aborted("field-map build aborted")
            results[i] = _solve_task(t)
            _log.info("time-domain RFQ field map: window %d/%d", i + 1, n)
    else:
        ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=nw, mp_context=ctx,
                                   initializer=init_worker,
                                   initargs=(worker_threads(nw),))
        try:
            futs = {pool.submit(_solve_task, t): i for i, t in enumerate(tasks)}
            pending = set(futs)
            done_n = 0
            while pending:
                if should_abort is not None and should_abort():
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise Aborted("field-map build aborted")
                done, pending = wait(pending, timeout=0.5,
                                     return_when=FIRST_COMPLETED)
                for f in done:
                    results[futs[f]] = f.result()
                    done_n += 1
                    _log.info("time-domain RFQ field map: window %d/%d", done_n, n)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    bad = [(i, r[3].get("rel_residual")) for i, r in enumerate(results)
           if (r[3].get("rel_residual") or 0.0) > MAX_REL_RESIDUAL]
    if bad:
        raise RuntimeError(
            "time-domain RFQ field map: the Laplace solve of window(s) "
            + ", ".join(f"{i + 1} (residual {rr:.2g})" for i, rr in bad)
            + " did not converge; nothing was cached")
    cores = [(w.z0, w.z1) for w in wins]
    blocks = [(r[0], r[1], r[2]) for r in results]
    fmap = RfqFieldMap.from_blocks(st["h"], cores, blocks, key=key)
    _log.info("time-domain RFQ: field map built in %.0f s (%.0f MB)",
              time.time() - t0, fmap.nbytes() / 1e6)
    return fmap
