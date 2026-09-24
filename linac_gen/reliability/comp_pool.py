"""A spawn-context process pool for the campaign's non-point jobs
(compensations, seed draws): the bounded-wait loop and kill-on-stop of
``linac_gen.parallel.scan_pool`` applied to any picklable
``fn(job) -> dict``.  ``max_workers <= 1`` runs in process."""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

__all__ = ["run_jobs"]


def run_jobs(fn, jobs, *, max_workers: int | None = None, on_done=None,
             should_stop=None) -> list:
    jobs = list(jobs)
    results: list = [None] * len(jobs)
    if not jobs:
        return results
    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    if workers <= 1:
        for i, job in enumerate(jobs):
            if should_stop is not None and should_stop():
                break
            row = fn(job)
            results[i] = row
            if on_done is not None:
                on_done(i, row)
        return results
    os.environ.setdefault("LINAC_GEN_FFT_WORKERS", "1")
    pool = ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"))

    def _kill_pool() -> None:
        pool.shutdown(wait=False, cancel_futures=True)
        procs = getattr(pool, "_processes", None) or {}
        for proc in list(procs.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in list(procs.values()):
            try:
                proc.join(timeout=2.0)
            except Exception:
                pass

    try:
        futures = {pool.submit(fn, j): i for i, j in enumerate(jobs)}
        pending = set(futures)
        while pending:
            if should_stop is not None and should_stop():
                _kill_pool()
                return results
            done_set, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for fut in done_set:
                i = futures[fut]
                row = fut.result()
                results[i] = row
                if on_done is not None:
                    on_done(i, row)
    except BaseException:
        _kill_pool()
        raise
    pool.shutdown(wait=True)
    return results
