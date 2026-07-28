"""Application-wide state container with Qt signal wiring.

Every workspace subscribes to the signals it needs; chrome subscribes to
all of them.  Keeping state in one place avoids the PyQt "who owns what"
hairball and gives us a single place to serialise session state later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class Tweaks:
    accent: str = "#22d3ee"
    density: float = 1.0
    palette: str = "turbo"
    live: bool = True
    inspector: str = "right"  # "left" | "right" | "off"
    dock: bool = True


class AppState(QObject):
    """Signals out every mutation the UI cares about."""

    tab_changed             = pyqtSignal(int)
    lattice_changed         = pyqtSignal(object)         # Lattice | None
    selected_element_changed = pyqtSignal(object)        # Element | None
    beam_config_changed     = pyqtSignal(object)         # BeamConfig
    s_cursor_changed        = pyqtSignal(float)          # mm along lattice
    running_changed         = pyqtSignal(bool)
    results_changed         = pyqtSignal(object)         # DiagnosticRecorder | EnvelopeResults
    particles_changed       = pyqtSignal(object)         # np.ndarray (N, 6) | None
    tweaks_changed          = pyqtSignal(object)         # Tweaks
    status_message          = pyqtSignal(str)
    log_event               = pyqtSignal(str, str)       # (level, message)
    matrix_cache_warmed     = pyqtSignal(int)            # n_entries after warm-up pass
    # Project-level dirty flag: True when in-memory Beam/Convergence/
    # correction-settings or session metadata diverge from the on-disk
    # .lgproj file.  Distinct from `bus.dirty` (which tracks only the
    # lattice .dat file).  Set by tab Apply handlers; cleared by Save
    # Project / Load Project.
    project_dirty_changed   = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._tab: int = 0
        self._lattice: Any = None
        self._lattice_path: Optional[str] = None
        # True when the in-memory lattice carries FITTED values (matcher
        # Apply / orbit-correction apply) that the on-disk source file
        # does not — plain Save then reroutes to Save-As so a curated
        # deck can't be silently overwritten by optimizer output (the
        # footgun that once clobbered the HWR matching fixture).
        self.lattice_fitted: bool = False
        self._selected: Any = None
        self._beam_config: Any = None
        self._s_cursor: float = 0.0
        self._running: bool = False
        self._results: Any = None
        # ErrorStudyResults bag (Monte Carlo run output).  Set by the
        # Error Study tab worker; consumed by the Results-tab ensemble plots.
        self.error_study_results: Any = None
        # FailureStudyResults bag (element-failure sweep output).  Set by
        # the Failure Study tab worker.
        self.failure_study_results: Any = None
        # Optional TraceWin partran.out overlay for side-by-side
        # comparison.  ``None`` until the user opens a file via the
        # Results-tab "Compare with TraceWin" action.  Stored as a
        # plain dict (output of ``read_partran_out``) — no special class.
        self.partran_overlay: Any = None
        # Orbit-correction defaults.  Mirrored in the Error Study tab UI;
        # picked up by the standalone Lattice-tab correction button and
        # by the auto-correction-on-load hook (when
        # ``auto_correction_mode == "always"``).
        self.correction_settings: dict = {
            "enabled": False,
            "method": "auto",
            "n_iter": 5,
            "tol_mm": 0.05,
            "bpm_noise": 0.0,
            "bpm_pattern": "BPM_*",
            "steerer_pattern": "STEER_*",
        }
        # When and whether to auto-fire orbit correction.  ``"never"`` is
        # explicit — never auto-correct; ``"on_errors_only"`` honours
        # ``ADJUST_STEERER`` cards only inside the Monte-Carlo loop;
        # ``"always"`` runs the standalone correction the moment a
        # lattice with ADJUST_STEERER cards is loaded.
        self.auto_correction_mode: str = "on_errors_only"
        # Shared per-element transfer-matrix cache.  Consumers (phase-advance
        # popups, matching tab) pass this dict to ``compute_transfer_matrix``
        # / ``structure_phase_advance``; a background warmer pre-populates
        # it on lattice load so the first user click is instant.  Replaced
        # wholesale on lattice or beam-config change.
        self.matrix_cache: dict = {}
        # Project-level dirty flag.  See project_dirty_changed signal
        # docstring above for semantics.  Toggled via mark_project_dirty
        # / mark_project_clean; queried via the `project_dirty` property.
        self._project_dirty: bool = False
        self._tweaks: Tweaks = Tweaks()
        # Lattice command bus — every list/param mutation flows through it.
        # Built lazily so importing AppState doesn't pull Qt symbols early.
        from linac_gen_gui.interphase.commands import CommandBus
        self.bus: CommandBus = CommandBus(lambda: self._lattice)
        # Re-broadcast bus-level changes through lattice_changed so views
        # using the existing signal pick up undo/redo / param edits.
        self.bus.changed.connect(lambda: self.lattice_changed.emit(self._lattice))

    # --- tab -----------------------------------------------------------
    @property
    def tab(self) -> int: return self._tab
    def set_tab(self, i: int) -> None:
        if i != self._tab:
            self._tab = i
            self.tab_changed.emit(i)

    # --- lattice -------------------------------------------------------
    @property
    def lattice(self): return self._lattice
    @property
    def lattice_path(self) -> Optional[str]: return self._lattice_path
    def set_lattice(self, lattice: Any, path: Optional[str] = None) -> None:
        self._lattice = lattice
        self._lattice_path = path
        self._selected = None
        self._s_cursor = 0.0
        # A freshly-installed lattice is presumed to match its source
        # file; Apply handlers re-set this AFTER calling the setter.
        self.lattice_fitted = False
        # Results belong to the lattice they were computed on — drop them
        # BEFORE any signal fires so lattice_changed consumers that read
        # state.results never see new-lattice + old-results.
        self._results = None
        self.error_study_results = None
        self.failure_study_results = None
        # Stale matrices from the previous lattice must not leak: replace
        # wholesale (not .clear()) so any consumer still holding a reference
        # to the old dict sees a stable snapshot.
        self.matrix_cache = {}
        # Loading a fresh lattice resets the undo stack and dirty flag.
        if hasattr(self, "bus"):
            self.bus.reset()
        self.lattice_changed.emit(lattice)
        self.selected_element_changed.emit(None)
        self.s_cursor_changed.emit(0.0)
        self.results_changed.emit(None)

    @property
    def dirty(self) -> bool:
        """True if there are un-saved lattice edits."""
        return bool(getattr(self, "bus", None) and self.bus.dirty)

    def mark_saved(self) -> None:
        """Clear the dirty flag (call after a successful Save)."""
        if hasattr(self, "bus"):
            self.bus.mark_clean()

    # --- project-dirty -------------------------------------------------
    # Independent of the lattice bus.dirty -- Beam/Convergence/correction
    # edits live in the .lgproj JSON, not the .dat.  Both flags are
    # checked by app.py:_confirm_discard before destructive operations.
    @property
    def project_dirty(self) -> bool:
        return bool(getattr(self, "_project_dirty", False))

    def mark_project_dirty(self) -> None:
        """Flag the in-memory project state as diverging from on-disk."""
        if not self._project_dirty:
            self._project_dirty = True
            self.project_dirty_changed.emit(True)

    def mark_project_clean(self) -> None:
        """Clear the project dirty flag (call after Save Project / load)."""
        if self._project_dirty:
            self._project_dirty = False
            self.project_dirty_changed.emit(False)

    # --- selection -----------------------------------------------------
    @property
    def selected(self): return self._selected
    def set_selected(self, element: Any) -> None:
        self._selected = element
        self.selected_element_changed.emit(element)

    # --- s-cursor ------------------------------------------------------
    @property
    def s_cursor(self) -> float: return self._s_cursor
    def set_s_cursor(self, s: float) -> None:
        s = max(0.0, float(s))
        if abs(s - self._s_cursor) > 1e-9:
            self._s_cursor = s
            self.s_cursor_changed.emit(s)

    # --- running flag --------------------------------------------------
    @property
    def running(self) -> bool: return self._running
    def set_running(self, r: bool) -> None:
        if r != self._running:
            self._running = r
            self.running_changed.emit(r)

    # --- results -------------------------------------------------------
    @property
    def results(self): return self._results
    def set_results(self, results: Any) -> None:
        self._results = results
        self.results_changed.emit(results)

    def set_particles(self, particles) -> None:
        """Broadcast a (N, 6) particle array to interested views (phase space)."""
        self.particles_changed.emit(particles)

    # --- beam ----------------------------------------------------------
    @property
    def beam_config(self): return self._beam_config
    def set_beam_config(self, cfg: Any) -> None:
        self._beam_config = cfg
        # Cached matrices are keyed by ref state at element entry — a beam
        # config change means a new entry energy, so old entries are
        # useless.  Replace the cache rather than .clear() so consumers
        # holding the old dict see a stable snapshot.
        self.matrix_cache = {}
        self.beam_config_changed.emit(cfg)

    # --- tweaks --------------------------------------------------------
    @property
    def tweaks(self) -> Tweaks: return self._tweaks
    def update_tweaks(self, **kwargs) -> None:
        changed = False
        for k, v in kwargs.items():
            if getattr(self._tweaks, k, None) != v:
                setattr(self._tweaks, k, v)
                changed = True
        if changed:
            self.tweaks_changed.emit(self._tweaks)


# Top-level tab registry (TraceWin-style, left-to-right).
TABS: list[tuple[str, str]] = [
    # (id, label).  IDs are stable API (code keys off them — e.g. the
    # scroll-wrap set in app.py); labels are display-only.  "Numerics"
    # (né "Convergence") holds the simulation settings every run uses
    # plus the convergence scans that validate them; the two study tabs
    # are named for what they run, not for "Errors" (which read like a
    # software error log).
    ("beam",        "Beam"),
    ("lattice",     "Lattice"),
    ("matching",    "Matching"),
    ("convergence", "Numerics"),
    ("surrogates",  "Surrogates"),
    ("errors",      "Error Study"),
    ("failures",    "Failure Study"),
    ("results",     "Results"),
]
