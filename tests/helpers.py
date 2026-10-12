"""Shared plain-Python test helpers (no Qt, no fixtures).

Used by the run-current provenance tests (tests/tracking/
test_run_current_stamp.py) and the orbit-correction dead-beam tests —
both probe getattr-based readers that must distinguish "field absent"
from "field is None" from "field is 0.0".
"""
from types import SimpleNamespace


def stub_results(**fields):
    """A SimpleNamespace standing in for a results object
    (DiagnosticRecorder / EnvelopeResults / _LoadedResults).

    Only the passed fields exist, so ``getattr(obj, name, default)``
    readers see exactly the attribute surface the test declares.
    """
    return SimpleNamespace(**fields)


def kept_thread_setting():
    """Context manager: whatever a test does to the HELIX thread setting
    (linac_gen.parallel.threads) -- set_threads, HELIX_THREADS, torch's
    count, the warn-once register -- is undone afterwards, kernels
    included."""
    import contextlib
    import os

    from linac_gen.parallel import threads

    @contextlib.contextmanager
    def _kept():
        try:                    # torch first, so its count can be restored
            import torch
        except ImportError:     # pragma: no cover
            torch = None
        saved = (threads._setting, threads._requested,
                 os.environ.get(threads.ENV), threads._env_before,
                 set(threads._warned))
        torch_n = torch.get_num_threads() if torch is not None else None
        try:
            yield threads
        finally:
            (threads._setting, threads._requested, env,
             threads._env_before, warned) = saved
            threads._warned.clear()
            threads._warned.update(warned)
            if env is None:
                os.environ.pop(threads.ENV, None)
            else:
                os.environ[threads.ENV] = env
            threads.sync_kernels()
            if torch_n is not None:
                torch.set_num_threads(torch_n)
    return _kept()
