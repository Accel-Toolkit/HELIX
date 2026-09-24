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
    # --- study-manager extensions (additive; scan/batch/GUI callers
    #     leave the defaults and the worker is bit-identical to before) ---
    out_path: str | None = None     # when set: write full results here
    out_format: str = "hdf5"        # hdf5 | openpmd | partran
    drift_single_push: bool = True  # StepConfig.drift_single_push for the worker
    capture_errors: bool = False    # True: a raised point returns an
    #                                 {"error", "traceback"} row instead
    #                                 of killing the whole pool
    # --- reliability-campaign extensions (additive; every existing caller
    #     leaves the defaults and the worker is bit-identical to before) ---
    snapshot_elements: tuple = ()   # element NAMES whose exit phase space
    #                                 is snapshotted (mp mode; lands in the
    #                                 HDF5 ``particles/`` group)
    lattice_beam: dict | None = None  # the beam a MAD8 deck with no rigidity
    #                                 of its own is converted with: None =
    #                                 beam_config (GUI/assistant/replay points,
    #                                 whose beam never varies across a scan);
    #                                 {} = none (a bare CLI lattice refuses, as
    #                                 `run` does); build_scan_point sets the
    #                                 project beam BEFORE --beam overrides so
    #                                 an energy scan never rescales magnets
    loss_metrics: bool = False      # True: the row also carries the loss-
    #                                 power scalars of _loss_metrics()


def _parse_lattice_for_scan(path: str, beam_config=None):
    """Parse a lattice file → ``Lattice`` in a worker process, through the same
    suffix dispatcher as the CLI and the GUI (``linac_gen.io.formats``): a
    private copy here once sent ``.jl`` / ``.bmad`` decks to the TraceWin
    parser, silently.  Warnings are the parent's business (it parsed the
    same file first).  ``beam_config`` is the point's beam: a MAD8 file with
    no rigidity of its own takes it from there, as the parent did."""
    from linac_gen.io.formats import parse_lattice_file
    return parse_lattice_file(path, warn_unknown=False,
                              fallback_beam=beam_config)[0]


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


LOSS_METRIC_KEYS = ("loss_w_total", "loss_w_per_m_peak",
                    "loss_w_per_m_peak_s_m", "n_lost")


def _loss_metrics(res, cfg) -> dict:
    """Loss-power scalars for a results object (opt-in through
    ``ScanPoint.loss_metrics``): the total lost beam power, the peak
    lineal loss density over 1 m bins and the bin centre it sits at, and
    the lost-macroparticle count — all at the beam's current × duty
    cycle (``BeamConfig.duty_cycle``), so a project saved at 1.1 % duty
    reports watts, not the CW figure.  Every key is ``None`` when the
    results carry no loss record (the envelope solver, or a run whose
    ``n_macro`` is unknown) so a row's key-set stays fixed per point."""
    lt = getattr(res, "loss_table", None)
    n_macro = getattr(res, "n_macro", None)
    if lt is None or not n_macro:
        return {k: None for k in LOSS_METRIC_KEYS}
    import numpy as np
    from linac_gen.analysis.loss_power import (loss_power_profile,
                                               loss_power_summary)
    s_arr = getattr(res, "s", None)
    s_end = float(s_arr[-1]) if s_arr is not None and len(s_arr) else 0.0
    summ = loss_power_summary(lt, current_mA=float(cfg.current),
                              duty_pct=float(cfg.duty_cycle),
                              n_macro=int(n_macro))
    centers, wpm = loss_power_profile(lt, current_mA=float(cfg.current),
                                      duty_pct=float(cfg.duty_cycle),
                                      n_macro=int(n_macro),
                                      s_end_mm=max(s_end, 1.0))
    j = int(np.argmax(wpm))
    return {
        "loss_w_total": float(summ["lost_w"]),
        "loss_w_per_m_peak": float(wpm[j]),
        "loss_w_per_m_peak_s_m": float(centers[j]) * 1e-3,
        "n_lost": int(summ["n_lost"]),
    }


def _run_one_point_worker(point: ScanPoint) -> dict:
    """Sub-process entry point.  Rebuilds everything from serialisable inputs.

    Returns a dict with end-of-lattice scalars + timing.  We serialise
    only primitives through the process boundary so we never have to
    teach pickle about Qt/QThread/QApplication internals.
    """
    # Cap the per-point FFT threading so the pool doesn't oversubscribe.
    os.environ.setdefault("LINAC_GEN_FFT_WORKERS", "1")

    import time

    t0 = time.time()
    try:
        from linac_gen.core.config import BeamConfig, SpaceChargeConfig
        from linac_gen.core.step_config import StepConfig
        from linac_gen.core.simulation import Simulation
        from linac_gen.distributions.factory import create_beam

        lattice = _parse_lattice_for_scan(
            point.lattice_path,
            point.beam_config if point.lattice_beam is None
            else (point.lattice_beam or None))
        lattice.step_config = StepConfig(
            integration_steps_per_metre=float(point.step1),
            sc_steps_per_metre=float(point.step2),
            drift_single_push=bool(point.drift_single_push),
        )
        # Element-parameter overrides (a batch-mode scan over e.g. a quad
        # gradient).  Empty for the GUI convergence-tab caller — no import
        # then.
        if point.element_overrides:
            from linac_gen.cli.common import apply_element_override
            for selector, value in point.element_overrides:
                apply_element_override(lattice, selector, value)

        cfg = BeamConfig(**point.beam_config)
        beam = create_beam(cfg, seed=point.seed)

        t0 = time.time()
        sc = None
        if point.mode == "envelope":
            from linac_gen.cli.common import run_envelope_sim
            res = run_envelope_sim(lattice, cfg,
                                   env_solver=point.env_solver)
        else:
            if cfg.current > 0:
                sc = SpaceChargeConfig(
                    nx=int(point.nx), ny=int(point.nx), nz=int(point.nx),
                    grid_extent=float(point.grid_extent),
                    use_gpu=point.use_gpu,
                    **dict(point.sc_overrides),
                )
            res = Simulation(
                lattice, beam, space_charge=sc,
                snapshot_elements=(list(point.snapshot_elements)
                                   if point.snapshot_elements else None),
            ).run()
        elapsed = time.time() - t0

        if point.out_path:
            # atomic finalize: "out_path exists" is a sufficient
            # completeness test for study resume — a crash mid-write can
            # only ever leave a .part orphan, never a torn results file
            from linac_gen.cli.common import write_results
            tmp = str(point.out_path) + ".part"
            write_results(res, tmp, point.out_format, cfg, lattice,
                          lattice_path=point.lattice_path,
                          seed=point.seed, sc_config=sc)
            os.replace(tmp, point.out_path)

        metrics = _scan_metrics(res, elapsed)
        if point.loss_metrics:
            metrics.update(_loss_metrics(res, cfg))
        metrics["error"] = None
        if point.out_path:
            metrics["results_path"] = str(point.out_path)
        return metrics
    except Exception as exc:
        if not point.capture_errors:
            raise
        import traceback
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed": time.time() - t0,
        }


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
