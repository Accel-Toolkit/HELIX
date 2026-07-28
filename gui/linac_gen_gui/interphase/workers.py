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
"""
from __future__ import annotations

import math
import threading
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal


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

    def __init__(self, lattice, beam, space_charge,
                 record_substeps: bool = False,
                 density_axes: tuple = (),
                 snapshot_every_n: int | None = None,
                 snapshot_elements=None,
                 density_n_bins: int = 200,
                 density_extent: dict | None = None):
        super().__init__()
        self.lattice = lattice
        self.beam = beam
        self.sc = space_charge
        self.record_substeps = record_substeps
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
                progress_callback=self._on_progress,
                should_abort=self._should_abort,
            )
            self.progress.emit(0)
            results = sim.run()
            try:
                results.ref_frequency = self.beam.ref.frequency
            except Exception:
                pass
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
            try:
                results.ref_frequency = self.beam.ref.frequency
            except Exception:
                pass
            if self._stop_event.is_set():
                self.aborted.emit()
                return
            self.progress.emit(100)
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
