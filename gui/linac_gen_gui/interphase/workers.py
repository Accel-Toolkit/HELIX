"""QThread workers wrapping the envelope and multi-particle solvers.

Both workers expose:

* ``progress_s(float, float)`` — current (s_mm, total_length_mm) so a GUI
  progress bar can show real longitudinal progress instead of a two-step
  "10 %, 100 %" placeholder.
* ``progress(int)`` — percent complete (back-compat for clients that only
  cared about a single integer).
* ``aborted()`` — emitted if the user pressed Stop mid-run.
* ``request_stop()`` — thread-safe cancel; the solver exits cleanly at
  the next element boundary.

The multi-particle worker also emits ``status(str)``: the time-domain RFQ
solver's progress lines (its field-map build on a first run takes minutes
with nothing else moving).
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal

#: Every background lattice walk holds this while it walks: the results-tab
#: σ₀ and coupled per-cell workers, the companion probe, the footprint, and
#: the envelope / multi-particle runs.  compute_transfer_matrix and the
#: envelope solver mutate the lattice's own elements (reset_run_state /
#: fitted_matrix / advance_ref; a FieldMap3D's _step_idx is rewritten by
#: every Jacobian), and until the coupled walks left the GUI thread the
#: frozen window was the only thing keeping a run and a walk apart.  The GUI
#: thread never takes it; acquire through acquire_walk_lock so a worker told
#: to stop while queued can still honour the stop.
WALK_LOCK = threading.Lock()


class _SignalLogHandler(logging.Handler):
    """Forwards log records to a Qt signal (emit is thread-safe)."""

    def __init__(self, signal):
        super().__init__(logging.INFO)
        self._signal = signal

    def emit(self, record) -> None:
        try:
            self._signal.emit(record.getMessage())
        except Exception:                               # noqa: BLE001
            pass                    # worker torn down mid-run


# INFO must be enabled on the logger for a handler to see INFO records; a
# retired run can overlap the next one, so the level is reference-counted
_STATUS_LOGGER = "linac_gen.rfq_td"
_status_lock = threading.Lock()
_status_users = {"n": 0, "level": logging.NOTSET}


def _attach_status(handler) -> None:
    log = logging.getLogger(_STATUS_LOGGER)
    with _status_lock:
        if _status_users["n"] == 0:
            _status_users["level"] = log.level
            if log.getEffectiveLevel() > logging.INFO:
                log.setLevel(logging.INFO)
        _status_users["n"] += 1
        log.addHandler(handler)


def _detach_status(handler) -> None:
    log = logging.getLogger(_STATUS_LOGGER)
    with _status_lock:
        log.removeHandler(handler)
        _status_users["n"] -= 1
        if _status_users["n"] == 0:
            log.setLevel(_status_users["level"])


def acquire_walk_lock(should_stop, poll_s: float = 0.2) -> bool:
    """Block until WALK_LOCK is held; return False, not holding it, if
    ``should_stop()`` turns true while waiting."""
    while not WALK_LOCK.acquire(timeout=poll_s):
        if should_stop is not None and should_stop():
            return False
    return True


class EnvelopeWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    progress_s = pyqtSignal(float, float)   # current s_mm, total_length_mm
    aborted = pyqtSignal()

    def __init__(self, lattice, ref, initial_twiss: dict, current_mA: float,
                 solver_kind: str = "matrix",
                 record_substeps: bool = False,
                 phase_probe: bool = True):
        super().__init__()
        self.lattice = lattice
        self.ref = ref
        self.initial = initial_twiss
        self.current_mA = current_mA
        self.solver_kind = solver_kind  # "matrix" (default) or "sacherer"
        self.record_substeps = bool(record_substeps)
        # Phase probe defaults ON in the GUI: negligible cost (a few 6×6
        # products per slice) and it feeds the channel-tune KPIs and the
        # tune-depression popup with exact numbers.  The Sacherer branch
        # has no probe.
        self.phase_probe = bool(phase_probe)
        self._stop_event = threading.Event()
        self._total_length_mm = sum(getattr(e, "length", 0.0)
                                    for e in lattice.elements) if lattice else 0.0

    def request_stop(self) -> None:
        """Thread-safe cancel — solver returns partial results at next boundary."""
        self._stop_event.set()

    def _on_progress(self, s_mm: float, i: int, n: int) -> None:
        # Called from the solver thread (== this QThread) per element.
        # Percent = POSITION along the lattice, not element count —
        # elements are unevenly distributed in s (a lattice can have
        # hundreds of short drifts/markers in one region), so i/n reads
        # "50%" nowhere near the halfway point of the machine.
        if self._total_length_mm > 0:
            pct = int(round(100.0 * s_mm / self._total_length_mm))
        else:
            pct = int(round(100.0 * i / max(n, 1)))
        self.progress.emit(min(pct, 100))
        self.progress_s.emit(float(s_mm), float(self._total_length_mm))

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        # Serialise with the results-tab walks (WALK_LOCK): a run and a walk
        # mutate the same element objects.  The app stops popup walks before
        # starting a run, so this wait is at most one walk cell.
        if not acquire_walk_lock(self._should_abort):
            self.aborted.emit()
            return
        try:
            self._run_locked()
        finally:
            WALK_LOCK.release()

    def _run_locked(self) -> None:
        try:
            if self.solver_kind == "sacherer":
                # SachererSolver doesn't yet support progress/abort callbacks;
                # the GUI shows an indeterminate progress bar for this branch.
                from linac_gen.tracking.sacherer import SachererSolver
                solver = SachererSolver(
                    self.lattice, self.ref, self.initial,
                    current=self.current_mA,
                )
            else:
                from linac_gen.tracking.envelope import EnvelopeSolver
                solver = EnvelopeSolver(
                    self.lattice, self.ref, self.initial,
                    current=self.current_mA,
                    progress_callback=self._on_progress,
                    should_abort=self._should_abort,
                    record_substeps=self.record_substeps,
                    phase_probe=self.phase_probe,
                )
            results = solver.run()
            # NOTE: the EnvelopeSolver populates ``results.ref_frequency``
            # as a per-step list inside ``_record``.  Do NOT overwrite it
            # with a scalar — downstream analyzers (IBS, σ_z) iterate over
            # the per-step values to handle FREQ-card boundaries.
            if getattr(solver, "aborted", False):
                self.aborted.emit()
                return
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class MultiparticleWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    progress_s = pyqtSignal(float, float)
    aborted = pyqtSignal()
    status = pyqtSignal(str)        # time-domain RFQ progress lines

    def __init__(self, lattice, beam, space_charge,
                 record_substeps: bool = False,
                 density_axes: tuple = (),
                 snapshot_every_n: int | None = None,
                 snapshot_elements=None,
                 density_n_bins: int = 200,
                 density_extent: dict | None = None,
                 record_action_scan: bool = False):
        super().__init__()
        # macOS default QThread stack (~544 KB) SIGBUSes numpy/OpenBLAS
        # workspace allocations — same failure mode documented on
        # BacktrackWorker/_MatchWorker.  16 MB matches the main thread.
        self.setStackSize(16 * 1024 * 1024)
        self.lattice = lattice
        self.beam = beam
        self.sc = space_charge
        self.record_substeps = record_substeps
        self.record_action_scan = bool(record_action_scan)
        self.density_axes = tuple(density_axes or ())
        self.snapshot_every_n = snapshot_every_n
        self.snapshot_elements = set(snapshot_elements or ())
        self.density_n_bins = int(density_n_bins)
        self.density_extent = dict(density_extent or {})
        self._stop_event = threading.Event()
        self._total_length_mm = sum(getattr(e, "length", 0.0)
                                    for e in lattice.elements) if lattice else 0.0

    def request_stop(self) -> None:
        self._stop_event.set()

    def _on_progress(self, s_mm: float, i: int, n: int) -> None:
        # Position-based percent (see the note in the MP worker).
        if self._total_length_mm > 0:
            pct = int(round(100.0 * s_mm / self._total_length_mm))
        else:
            pct = int(round(100.0 * i / max(n, 1)))
        self.progress.emit(min(pct, 100))
        self.progress_s.emit(float(s_mm), float(self._total_length_mm))

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        # Serialise with the results-tab walks (WALK_LOCK): a run and a walk
        # mutate the same element objects.  The app stops popup walks before
        # starting a run, so this wait is at most one walk cell.
        if not acquire_walk_lock(self._should_abort):
            self.aborted.emit()
            return
        try:
            self._run_locked()
        finally:
            WALK_LOCK.release()

    def _run_locked(self) -> None:
        handler = _SignalLogHandler(self.status)
        _attach_status(handler)
        try:
            self._run_sim()
        finally:
            _detach_status(handler)

    def _run_sim(self) -> None:
        try:
            from linac_gen.core.simulation import Simulation
            sim = Simulation(
                self.lattice, self.beam,
                space_charge=self.sc,
                record_substeps=self.record_substeps,
                density_axes=self.density_axes,
                snapshot_every_n=self.snapshot_every_n,
                snapshot_elements=self.snapshot_elements,
                density_n_bins=self.density_n_bins,
                density_extent=self.density_extent,
                record_action_scan=self.record_action_scan,
                progress_callback=self._on_progress,
                should_abort=self._should_abort,
            )
            self.progress.emit(0)
            results = sim.run()
            # (``results.ref_frequency`` is the recorder's PER-STEP list —
            # it used to be overwritten here with one scalar, which threw
            # away the FREQ-jump history the IBS / σ_z conversions need and
            # broke every consumer that indexes it per step.)
            # If the tracker saw the abort flag it returns partial results;
            # surface that to the GUI as an abort rather than a "finished" run.
            tracker_aborted = False
            try:
                # Tracker stores ``aborted`` on itself; Simulation returns the
                # DiagnosticRecorder, so we peek via the final simulation
                # state instead (Simulation doesn't expose the tracker).  If
                # the stop event was set, treat as aborted.
                tracker_aborted = self._stop_event.is_set()
            except Exception:
                pass
            if tracker_aborted:
                self.aborted.emit()
                return
            self.progress.emit(100)
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainWorker(QThread):
    """Multibunch / pulse-study worker (Simulate → Multibunch / Pulse
    Study…).

    Runs :func:`linac_gen.train.run_train` (mode-dispatched: tracked /
    envelope / fast / hybrid) off the GUI thread.  Same signal contract
    as the other run workers plus:

    * ``progress_bunch(done, total)`` — per-bunch progress (the train
      axis, not s along the lattice).
    * ``aborted(object)`` — carries the PARTIAL :class:`TrainResults`:
      a mid-train Stop leaves a loadable result (the runners stop
      cooperatively between bunches and mark ``truncated``), and the
      app still auto-dumps it.
    """

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    progress_bunch = pyqtSignal(int, int)
    aborted = pyqtSignal(object)                  # partial TrainResults

    def __init__(self, lattice, beam_config, train_config, space_charge,
                 lattice_path: Optional[str] = None):
        super().__init__()
        # macOS default QThread stack (~544 KB) SIGBUSes numpy/OpenBLAS
        # workspace allocations — same failure mode documented on
        # BacktrackWorker/_MatchWorker.  16 MB matches the main thread.
        self.setStackSize(16 * 1024 * 1024)
        self.lattice = lattice
        self.beam_config = beam_config
        self.train_config = train_config
        self.sc = space_charge
        self.lattice_path = lattice_path
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Thread-safe cancel — the train drivers check between bunches
        and return partial results (``truncated=True``)."""
        self._stop_event.set()

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def _on_progress(self, done: int, total: int) -> None:
        # Called from this QThread by the runners: tracked/hybrid per
        # bunch, fast mode every few thousand slots.
        self.progress_bunch.emit(int(done), int(total))
        self.progress.emit(min(int(round(100.0 * done / max(total, 1))),
                               100))

    def run(self) -> None:
        try:
            from linac_gen.train import run_train
            results = run_train(
                self.lattice, self.beam_config, self.train_config,
                sc_config=self.sc, lattice_path=self.lattice_path,
                progress_callback=self._on_progress,
                should_abort=self._should_abort,
            )
            fast = getattr(results, "fast", None)
            truncated = bool(
                getattr(results, "truncated", False)
                or (fast is not None and getattr(fast, "truncated",
                                                 False)))
            if self._stop_event.is_set() or truncated:
                self.aborted.emit(results)
                return
            self.progress.emit(100)
            self.finished_ok.emit(results)
        except Exception as exc:                            # noqa: BLE001
            self.failed.emit(str(exc))


class BacktrackWorker(QThread):
    """Backward-tracking worker (Simulate → Backtrack Distribution…).

    Clone of :class:`MultiparticleWorker`'s contract — same five
    signals + ``request_stop`` — but drives
    :meth:`Simulation.run_backtrack`: the supplied ``beam`` is the EXIT
    distribution and the result recorder is reversed to increasing s
    (index 0 = reconstructed entrance, ``direction == "backward"``), so
    the Results tab consumes it unchanged.
    """

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    progress_s = pyqtSignal(float, float)
    aborted = pyqtSignal()

    def __init__(self, lattice, beam, space_charge, entrance_ref, *,
                 start: int = 0, end: int | None = None,
                 field_map_mode: str = "rk4",
                 allow_dc_crossing: bool = False):
        super().__init__()
        # Qt's default QThread stack is ~544 KB on macOS — not enough for
        # OpenBLAS workspace under the backtracker's per-element
        # np.linalg.inv/cond calls (SIGBUS "Bus error", same failure mode
        # documented on _MatchWorker).  16 MB matches the main thread.
        self.setStackSize(16 * 1024 * 1024)
        self.lattice = lattice
        self.beam = beam
        self.sc = space_charge
        self.entrance_ref = entrance_ref
        # NOT ``self.start`` — that would SHADOW QThread.start() with an
        # int and ``worker.start()`` would raise "'int' object is not
        # callable" (bug found live in the GUI, 2026-07-11).
        self._range_start = int(start)
        self._range_end = end
        self.field_map_mode = field_map_mode
        self.allow_dc_crossing = bool(allow_dc_crossing)
        self._stop_event = threading.Event()
        self._total_length_mm = sum(getattr(e, "length", 0.0)
                                    for e in lattice.elements) if lattice else 0.0

    def request_stop(self) -> None:
        self._stop_event.set()

    def _on_progress(self, s_mm: float, i: int, n: int) -> None:
        # Position-based percent (see the note in the MP worker).
        if self._total_length_mm > 0:
            pct = int(round(100.0 * s_mm / self._total_length_mm))
        else:
            pct = int(round(100.0 * i / max(n, 1)))
        self.progress.emit(min(pct, 100))
        self.progress_s.emit(float(s_mm), float(self._total_length_mm))

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            import warnings as _warnings

            from linac_gen.core.simulation import Simulation
            from linac_gen.tracking.backtrack import BacktrackWarning
            sim = Simulation(
                self.lattice, self.beam,
                space_charge=self.sc,
                progress_callback=self._on_progress,
                should_abort=self._should_abort,
            )
            self.progress.emit(0)
            # Physics caveats (surrogate → linear fallback, survivors-
            # only reconstruction, adaptive-grid SC fallback, …) are
            # emitted as BacktrackWarning — invisible to a GUI user on
            # stderr, so capture them and hand them to the done-handler.
            with _warnings.catch_warnings(record=True) as wrec:
                _warnings.simplefilter("always")
                results = sim.run_backtrack(
                    entrance_ref=self.entrance_ref,
                    start=self._range_start, end=self._range_end,
                    field_map_mode=self.field_map_mode,
                    allow_dc_crossing=self.allow_dc_crossing,
                    # CSR has no backward model; the API refuses unless
                    # the caller accepts an approximate undo.  The GUI
                    # opts in exactly when the forward run had CSR on —
                    # the loud "SKIPPED … approximate" BacktrackWarning
                    # is captured below and SHOWN in the results dialog
                    # caveats, so the degradation is visible, not a
                    # dead-end error the user cannot act on.
                    approximate_backtracking=bool(
                        getattr(self.sc, "csr_enabled", False)),
                )
            caveats = [str(w.message) for w in wrec
                       if issubclass(w.category, BacktrackWarning)]
            # de-duplicate, preserve order
            results.backtrack_warnings = list(dict.fromkeys(caveats))
            # ref_frequency stays the per-step list (see the MP worker).
            if self._stop_event.is_set():
                self.aborted.emit()
                return
            self.progress.emit(100)
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class RingWorker(QThread):
    """Ring-mode run (Simulate → Ring Mode, Ring tab ▶ Run).

    The worker copies the lattice first — holding WALK_LOCK only while it
    copies — and runs on the copy: the window's lattice is never touched.
    Stop takes effect at the end of the turns being tracked (a turn of a
    large bunch can take seconds); the turns already done come back with
    ``aborted`` (None when no turn was done).  Progress is emitted at most
    once per percent; the statistics (``live``) at most five times a second,
    each time only the rows added since the last (a failure there never
    stops the run).  With ``dump_path`` the results — partial ones too — are
    written there (HDF5, ring schema, with ``lattice_path`` and the beam
    settings) before they are handed over; ``dumped`` carries the path, a
    failed write ``results.meta["dump_error"]``.
    """

    finished_ok = pyqtSignal(object)        # RingResults
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)              # percent
    progress_turn = pyqtSignal(int, int)    # (turn, n_turns)
    aborted = pyqtSignal(object)            # RingResults of the turns done | None
    # {"start": first row's index, "turns", "n_alive", "cov" (rows from
    #  start on), "n_turns", "beta0gamma0", "p0c_mev" (each row's reference),
    #  "ramp" (a run through a ramp)}
    live = pyqtSignal(object)
    dumped = pyqtSignal(str)                # the results file written

    LIVE_INTERVAL_S = 0.2

    def __init__(self, lattice, beam_config, ring_config, dump_path=None, lattice_path=None):
        super().__init__()
        # 16 MB like every numpy worker (macOS default ~544 KB → SIGBUS)
        self.setStackSize(16 * 1024 * 1024)
        self.lattice = lattice
        self.beam_config = beam_config
        self.ring_config = ring_config
        self.dump_path = dump_path
        self.lattice_path = lattice_path
        self._stop_event = threading.Event()
        self._last_pct = -1
        self._last_live = 0.0
        self._live_sent = 0

    def request_stop(self) -> None:
        self._stop_event.set()

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def _on_turn(self, turn: int, n: int) -> None:
        pct = int(100 * turn / max(n, 1))
        if pct != self._last_pct:
            self._last_pct = pct
            self.progress.emit(pct)
            self.progress_turn.emit(int(turn), int(n))

    def _on_live(self, stats: dict, n_turns: int, ref: dict) -> None:
        import time as _time
        if _time.monotonic() - self._last_live < self.LIVE_INTERVAL_S:
            return
        try:
            import numpy as _np
            k0, k1 = self._live_sent, len(stats["turns"])
            if k1 <= k0:
                return
            # (each row's reference: it changes in a ramp)
            bg = stats.get("ref_bg") or []
            pc = stats.get("ref_p0c") or []
            d = {"start": k0,
                 "turns": _np.asarray(stats["turns"][k0:k1]),
                 "n_alive": _np.asarray(stats["n_alive"][k0:k1]),
                 "cov": _np.asarray(stats["cov"][k0:k1]).reshape(-1, 6, 6),
                 "n_turns": int(n_turns),
                 "beta0gamma0": (_np.asarray(bg[k0:k1], dtype=float) if len(bg) >= k1
                                 else _np.full(k1 - k0, float(ref["beta0gamma0"]))),
                 "p0c_mev": (_np.asarray(pc[k0:k1], dtype=float) if len(pc) >= k1
                             else _np.full(k1 - k0, float(ref["p0c_mev"]))),
                 "ramp": getattr(self.ring_config, "ramp", None) is not None}
            self._live_sent = k1
            self.live.emit(d)
        except Exception:                                   # noqa: BLE001 — never the run
            pass
        finally:
            # (measured from the end of this update: the interval holds however
            # long the copy took)
            self._last_live = _time.monotonic()

    def _dump(self, results) -> None:
        if results is None:
            return
        if not self.dump_path:
            return
        try:
            from linac_gen.ring.results import save_ring_hdf5
            path = save_ring_hdf5(results, self.dump_path, lattice_path=self.lattice_path,
                                  beam_config=self.beam_config)
            results.meta["dump_path"] = str(path)
            self.dumped.emit(str(path))
        except Exception as exc:                            # noqa: BLE001
            results.meta["dump_error"] = f"{type(exc).__name__}: {exc}"

    def run(self) -> None:
        try:
            import copy

            from linac_gen.ring.engine import RingSimulation
            if not acquire_walk_lock(self._should_abort):
                self.aborted.emit(None)
                return
            try:
                lattice = copy.deepcopy(self.lattice)
            finally:
                WALK_LOCK.release()
            self.progress.emit(0)
            results = RingSimulation(lattice, self.beam_config, self.ring_config).run(
                progress_callback=self._on_turn, should_abort=self._should_abort,
                live_callback=self._on_live)
            if self._stop_event.is_set() and results.truncated:
                # (stopped before a turn was done: nothing to show — the
                # results of the previous run stay)
                partial = results if results.turns_done > 0 else None
                self._dump(partial)
                self.aborted.emit(partial)
                return
            self._dump(results)
            self.progress.emit(100)
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class RingOpticsWorker(QThread):
    """Ring optics (Ring tab ▶ Ring & Optics ▶ Compute optics).

    Compiles a copy of the lattice for ring mode (WALK_LOCK held only while
    it copies) and computes its periodic optics from the ring kernels; with
    ``six_d`` also the synchrotron tune and the RF bucket.  The first run in
    a session also compiles the ring kernels (a few seconds).  ``finished_ok``
    carries a dict: ``optics`` (RingOptics), ``bucket`` (dict or None),
    ``harmonic``, ``rf_buckets`` (the circumference over the RF period of the
    cavities with a voltage; None without RF), ``n_nodes``, ``warnings``,
    ``kinetic_mev``, ``p0c_mev``,
    ``constants``, ``compiled`` (False: numba is not installed and the kernels
    ran as plain Python).  A stop request ends the computation before its
    next pass around the ring, without a result.
    """

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, lattice, ring_config, beam_config, six_d: bool = False):
        super().__init__()
        self.setStackSize(16 * 1024 * 1024)
        self.lattice = lattice
        self.ring_config = ring_config
        self.beam_config = beam_config
        self.six_d = bool(six_d)
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            import copy

            from linac_gen.core.cancelled import OperationCancelled
            from linac_gen.ring.compile import compile_from_config
            from linac_gen.ring.kernels.cpu import numba_available
            from linac_gen.ring.optics import rf_bucket, twiss
            if not acquire_walk_lock(self._should_abort):
                return
            try:
                lattice = copy.deepcopy(self.lattice)
            finally:
                WALK_LOCK.release()
            rl = compile_from_config(lattice, self.ring_config, beam=self.beam_config)
            try:
                optics = twiss(rl, six_d=self.six_d, should_stop=self._should_abort)
            except OperationCancelled:
                return
            bucket = rf_bucket(rl, optics) if self.six_d else None
            if self._stop_event.is_set():
                return
            from linac_gen.ring.injection.injector import bucket_length
            from linac_gen.ring.sc.placement import element_starts
            period = bucket_length(rl, bool(rl.rf_on)) if rl.rf_on else None
            self.finished_ok.emit({"optics": optics, "bucket": bucket, "harmonic": rl.harmonic,
                                   "rf_on": bool(rl.rf_on),
                                   "rf_buckets": (int(round(float(rl.length) / period))
                                                  if period else None),
                                   "element_starts_s": element_starts(rl)[1],
                                   "n_nodes": rl.n_nodes, "warnings": list(rl.warnings),
                                   "kinetic_mev": rl.reference.kinetic0,
                                   "p0c_mev": rl.reference.p0c,
                                   "constants": self.ring_config.constants,
                                   "compiled": numba_available()})
        except Exception as exc:
            self.failed.emit(str(exc))


class RingMatchedBeamWorker(QThread):
    """The beam matched to the ring at the start of the run (Ring tab ▶ Ring &
    Optics ▶ Use the matched beam in the Beam tab).

    Compiles a copy of the lattice (WALK_LOCK held only while it copies) with
    the run's settings — the element model, the ramp, the injection magnets —
    and computes the optics of the run's first turn
    (:meth:`~linac_gen.ring.engine.RingSimulation.matched_beam`).
    ``finished_ok`` carries its dict: the Twiss functions x, y and, with the RF
    on, z in the Beam tab's units, the dispersion per MeV, the closed orbit as
    a centroid, the reference and notes.  A stop request ends the computation
    before its next pass around the ring, without a result.
    """

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, lattice, ring_config, beam_config):
        super().__init__()
        self.setStackSize(16 * 1024 * 1024)
        self.lattice = lattice
        self.ring_config = ring_config
        self.beam_config = beam_config
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            import copy

            from linac_gen.core.cancelled import OperationCancelled
            from linac_gen.ring.engine import RingSimulation
            if not acquire_walk_lock(self._should_abort):
                return
            try:
                lattice = copy.deepcopy(self.lattice)
            finally:
                WALK_LOCK.release()
            try:
                out = RingSimulation(lattice, self.beam_config, self.ring_config).matched_beam(
                    should_stop=self._should_abort)
            except OperationCancelled:
                return
            if not self._stop_event.is_set():
                self.finished_ok.emit(out)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.failed.emit(str(exc))


class RingRampWorker(QThread):
    """The reference particle through a ramp (Ring tab ▶ Programs ▶ Preview).

    Compiles a copy of the lattice for ring mode (WALK_LOCK held only while
    it copies) at the ramp's start, takes gamma_t from the ring's optics
    unless the ramp gives it, and runs the reference alone: to the end of the
    energy program, or ``n_turns`` turns for a flat one.  ``finished_ok``
    carries a dict: ``orbit`` (Ramp.orbit), ``turns``, ``turns_to_end``,
    ``mass_mev``, ``gamma_t``, ``harmonic``, ``length_m``, ``voltage_eff_factor``,
    ``phase_offset_deg``.  More than ``MAX_TURNS`` turns is refused; a stop
    request ends it without a result.
    """

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    MAX_TURNS = 1_000_000

    def __init__(self, lattice, ring_config, beam_config, n_turns: int):
        super().__init__()
        self.setStackSize(16 * 1024 * 1024)
        self.lattice = lattice
        self.ring_config = ring_config
        self.beam_config = beam_config
        self.n_turns = int(n_turns)
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _should_abort(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            import copy

            from linac_gen.core.cancelled import OperationCancelled
            from linac_gen.ring.engine import RingSimulation
            if not acquire_walk_lock(self._should_abort):
                return
            try:
                lattice = copy.deepcopy(self.lattice)
            finally:
                WALK_LOCK.release()
            try:
                ramp = RingSimulation(lattice, self.beam_config, self.ring_config).ramp()
                n_end = ramp.turns_to_end(max_turns=self.MAX_TURNS,
                                          should_stop=self._should_abort)
                n = n_end if n_end is not None else min(self.n_turns, self.MAX_TURNS)
                orbit = ramp.orbit(n, should_stop=self._should_abort)
            except OperationCancelled:
                return
            if self._stop_event.is_set():
                return
            self.finished_ok.emit({"orbit": orbit, "turns": n, "turns_to_end": n_end,
                                   "mass_mev": ramp.mass, "gamma_t": ramp.gamma_t,
                                   "harmonic": int(ramp.h), "length_m": ramp.length,
                                   "voltage_eff_factor": float(ramp.r_unit),
                                   "phase_offset_deg": math.degrees(ramp.arg)})
        except Exception as exc:
            self.failed.emit(str(exc))
