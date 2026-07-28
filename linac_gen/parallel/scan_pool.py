"""Convergence / parameter-scan point-pool runner.

Each scan point is an **independent** multi-particle simulation — so the
work is embarrassingly parallel.  This module exposes a lightweight API
that the GUI scan workers (classic + Interphase) can call to execute N
points concurrently in worker processes.

Design:

* Each worker runs one :class:`ScanPoint` in a fresh Python process
  (``concurrent.futures.ProcessPoolExecutor``).  Processes are used
  (not threads) because NumPy releases the GIL for FFTs but the tracker
  itself holds it during the Python-level per-element loop, so threads
  would barely speed anything up.
* Results are delivered to a user-supplied callback as soon as each
  point finishes (in arbitrary order).  The callback also receives the
  ORIGINAL input index so callers can slot the result into the right
  table row.
* Every worker uses ``LINAC_GEN_FFT_WORKERS=1`` internally to avoid
  thread oversubscription: the pool gives us O(N_cores) parallelism
  across points, and we do not want each point to also try to use
  O(N_cores) threads inside.
* A serial fallback (:func:`run_scan_points_serial`) is shipped with
  the identical signature so regression tests can exercise the same
  control flow without spawning processes.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class ScanPoint:
    """One independent scan row.

    Attributes capture everything the worker needs to rebuild the beam,
    space-charge config, and step config for this specific run — the
    top-level worker shouldn't need the caller's state after spawn.
    """
    lattice_path: str
    beam_config: dict               # serialisable BeamConfig.__dict__
    nx: int
    grid_extent: float
    step1: float
    step2: float
    seed: int = 42
    use_gpu: str = "auto"           # "auto" | "cpu" | "gpu" | "cuda" | "mps"
    # --- batch-mode CLI extensions (additive; GUI callers leave defaults) ---
    mode: str = "mp"                # "mp" | "envelope"
    env_solver: str = "matrix"      # "matrix" | "sacherer" (envelope mode)
    element_overrides: tuple = ()   # ((selector, value), …) applied post-parse
    sc_overrides: tuple = ()        # (("kernel", "cic"), …) extra SC kwargs


def _parse_lattice_for_scan(path: str):
    """Parse a ``.dat`` / ``.madx`` / ``.lat`` lattice file → ``Lattice``
    (the parser is chosen from the file extension)."""
    from pathlib import Path
    suf = Path(path).suffix.lower()
    if suf in (".madx", ".seq"):
        from linac_gen.io.madx_parser import parse_madx
        return parse_madx(path)[0]
    if suf in (".lat", ".flat"):
        from linac_gen.io.mad8_parser import parse_mad8
        return parse_mad8(path)[0]
    if suf == ".lte":
        from linac_gen.io.elegant_parser import parse_elegant
        return parse_elegant(path)[0]
    from linac_gen.io.tracewin_parser import parse_tracewin
    return parse_tracewin(path)[0]


def _scan_metrics(res, elapsed: float) -> dict:
    """End-of-lattice scalars from a results object (``DiagnosticRecorder``
    or ``EnvelopeResults``).

    The historical five keys keep their ``0.0`` fallback so existing GUI
    callers are byte-unaffected; the richer metrics fall back to ``None``
    when the field is absent (e.g. the envelope solver records no
    ``transmission``)."""
    def _last0(name):
        arr = getattr(res, name, None)
        return float(arr[-1]) if arr else 0.0

    def _lastn(name):
        arr = getattr(res, name, None)
        try:
            return float(arr[-1]) if arr is not None and len(arr) else None
        except (TypeError, IndexError, ValueError):
            return None

    out = {
        "elapsed":      elapsed,
        "sigma_x":      _last0("sigma_x"),
        "sigma_y":      _last0("sigma_y"),
        "sigma_phi":    _last0("sigma_phi"),
        "emit_x":       _last0("emit_x"),
        "emit_y":       _last0("emit_y"),
        "sigma_w":      _lastn("sigma_w"),
        "emit_z":       _lastn("emit_z"),
        "transmission": _lastn("transmission"),
        "ref_w_kin":    _lastn("ref_w_kin"),
        "x_max":        _lastn("x_max"),
        "y_max":        _lastn("y_max"),
        "ref_beta":     _lastn("ref_beta"),
        "ref_gamma":    _lastn("ref_gamma"),
    }
    # Tail quantiles (HALO-PIC): present only when the recorder was
    # configured with configure_tail(); additive keys, None otherwise.
    tail = getattr(res, "tail", None)
    if isinstance(tail, dict):
        for key, arr in tail.items():
            try:
                out[f"tail_{key}"] = (float(arr[-1])
                                      if arr is not None and len(arr)
                                      else None)
            except (TypeError, IndexError, ValueError):
                out[f"tail_{key}"] = None
    # Normalised RMS emittances: read from the recorder when present (MP);
    # transverse falls back to εn = βγ·ε_geo so envelope reports them too.
    # Mirrors linac_gen.failures.study.recorder_metrics — keep the two in sync.
    b, g = out["ref_beta"], out["ref_gamma"]
    bg = (b * g) if (b is not None and g is not None) else None
    enx, eny, enz = _lastn("emit_nx"), _lastn("emit_ny"), _lastn("emit_nz")
    if bg is not None:
        if enx is None and out["emit_x"] is not None:
            enx = out["emit_x"] * bg
        if eny is None and out["emit_y"] is not None:
            eny = out["emit_y"] * bg
        if enz is None:
            ez_mmmrad = _lastn("emit_z_mmmrad")
            if ez_mmmrad is not None:
                enz = ez_mmmrad * bg
    out["emit_nx"], out["emit_ny"], out["emit_nz"] = enx, eny, enz
    return out


def _run_one_point_worker(point: ScanPoint) -> dict:
    """Sub-process entry point.  Rebuilds everything from serialisable inputs.

    Returns a dict with end-of-lattice scalars + timing.  We serialise
    only primitives through the process boundary so we never have to
    teach pickle about Qt/QThread/QApplication internals.
    """
    # Cap the per-point FFT threading so the pool doesn't oversubscribe.
    os.environ.setdefault("LINAC_GEN_FFT_WORKERS", "1")

    import time
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig
    from linac_gen.core.step_config import StepConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam

    lattice = _parse_lattice_for_scan(point.lattice_path)
    lattice.step_config = StepConfig(
        integration_steps_per_metre=float(point.step1),
        sc_steps_per_metre=float(point.step2),
    )
    # Element-parameter overrides (a batch-mode scan over e.g. a quad
    # gradient).  Empty for the GUI convergence-tab caller — no import then.
    if point.element_overrides:
        from linac_gen.cli.common import apply_element_override
        for selector, value in point.element_overrides:
            apply_element_override(lattice, selector, value)

    cfg = BeamConfig(**point.beam_config)
    beam = create_beam(cfg, seed=point.seed)

    t0 = time.time()
    if point.mode == "envelope":
        from linac_gen.cli.common import run_envelope_sim
        res = run_envelope_sim(lattice, cfg, env_solver=point.env_solver)
    else:
        sc = None
        if cfg.current > 0:
            sc = SpaceChargeConfig(
                nx=int(point.nx), ny=int(point.nx), nz=int(point.nx),
                grid_extent=float(point.grid_extent),
                use_gpu=point.use_gpu,
                **dict(point.sc_overrides),
            )
        res = Simulation(lattice, beam, space_charge=sc).run()
    elapsed = time.time() - t0
    return _scan_metrics(res, elapsed)


def run_scan_points(
    points: Iterable[ScanPoint],
    on_done: Optional[Callable[[int, dict], None]] = None,
    max_workers: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """Run every point in a process pool and return results in input order.

    Parameters
    ----------
    points
        Iterable of :class:`ScanPoint`.  Consumed into a list internally.
    on_done
        Optional callback invoked once per completed point with
        ``(original_index, result_dict)``.  Useful for GUIs that want
        to update their table live instead of waiting for the whole
        scan to finish.
    max_workers
        Pool size.  ``None`` uses all available cores.
    should_stop
        Optional zero-arg callable polled ~4×/s while waiting on the
        pool.  When it returns True, queued points are cancelled, the
        worker processes are terminated (a running point cannot be
        interrupted from the outside — without the kill it would run to
        completion and block shutdown for minutes), and the rows
        completed so far are returned.
    """
    import logging
    from dataclasses import replace

    _log = logging.getLogger("linac_gen.parallel")

    pts = list(points)
    if not pts:
        return []

    # Parallel + GPU is a foot-gun: every worker process opens its own GPU
    # context (CUDA / Metal), competes for the single device, and ends up
    # serialised while still paying the per-process VRAM/init cost.  When
    # the caller asked for >1 worker AND any point requested a GPU backend
    # (auto / gpu / cuda / mps), force all points to CPU so the processes
    # actually run in parallel.
    actual_workers = max_workers if max_workers is not None else os.cpu_count() or 1
    if actual_workers > 1:
        _GPU_LIKE = ("auto", "gpu", "cuda", "mps")
        gpu_pts = [p for p in pts if p.use_gpu in _GPU_LIKE]
        if gpu_pts:
            _log.info(
                "scan_pool: %d worker(s) requested → forcing CPU backend "
                "per worker (GPU doesn't parallelise across processes).",
                actual_workers,
            )
            pts = [replace(p, use_gpu="cpu") if p.use_gpu != "cpu" else p
                   for p in pts]
    results: list[dict | None] = [None] * len(pts)
    # Use the "spawn" start method explicitly.  The default on Linux is
    # "fork", which copies the parent's thread table; if the parent has
    # any background threads at fork time (pytest plugins, Qt, OpenMP,
    # numpy/scipy thread pools, …) the child inherits a broken state and
    # deadlocks waiting on IPC from a thread that doesn't exist.  "spawn"
    # starts a fresh interpreter, avoiding this entirely at the cost of
    # re-importing modules in each worker.
    mp_ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx)

    def _kill_pool() -> None:
        """Cancel queued points and SIGTERM the running children.

        ``shutdown(cancel_futures=True)`` only drops QUEUED work; a
        point already executing runs to completion regardless, and both
        the executor exit and the interpreter's atexit hook would then
        block on it.  The points are pure spawn-context compute with no
        external state, so terminating the processes is safe.
        """
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
        futures = {pool.submit(_run_one_point_worker, p): i
                   for i, p in enumerate(pts)}
        pending = set(futures)
        while pending:
            if should_stop is not None and should_stop():
                _kill_pool()
                return [r for r in results if r is not None]
            # Bounded wait instead of as_completed(): with minutes-long
            # points, a blocking iterator would sit on the stop request
            # until the NEXT point happened to finish.
            done_set, pending = wait(pending, timeout=0.25,
                                     return_when=FIRST_COMPLETED)
            for fut in done_set:
                i = futures[fut]
                row = fut.result()
                results[i] = row
                if on_done is not None:
                    on_done(i, row)
    except BaseException:
        # A failed point (or KeyboardInterrupt): don't leave siblings
        # running for minutes behind the raised error.
        _kill_pool()
        raise
    pool.shutdown(wait=True)
    # Narrow Optional → dict now that every slot is filled
    return [r for r in results if r is not None]


def run_scan_points_serial(
    points: Iterable[ScanPoint],
    on_done: Optional[Callable[[int, dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """In-process serial fallback with the same signature as :func:`run_scan_points`.

    Used for regression testing and for environments where
    ``ProcessPoolExecutor`` is unavailable or undesirable (e.g. certain
    WSL configurations).  Produces identical numerical results.
    """
    pts = list(points)
    results: list[dict] = []
    for i, p in enumerate(pts):
        if should_stop is not None and should_stop():
            break
        row = _run_one_point_worker(p)
        results.append(row)
        if on_done is not None:
            on_done(i, row)
    return results
