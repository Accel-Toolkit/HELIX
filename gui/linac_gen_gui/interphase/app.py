"""Interphase v2 — TraceWin-style 5-tab dark-theme GUI for Linac_Gen.

Launch with:  python -m linac_gen_gui.interphase
"""
from __future__ import annotations

# Cap BLAS thread counts BEFORE numpy/scipy load.  OpenBLAS's parallel
# LU (`dgetrf_parallel`) allocates a large workspace on the calling
# thread's stack and SIGBUSes any QThread (default ~544 KB stack on
# macOS).  These env vars are only consulted at BLAS library init, so
# they must be set before the first numpy import.  Both the `python -m
# linac_gen_gui.interphase` and `python -m linac_gen_gui.interphase.app`
# launch paths hit this module, so this is the single source of truth.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget, QFileDialog,
    QFormLayout, QMessageBox, QDialog,
)


# ---------------------------------------------------------------------------
# Apply HELIX's preferred QFormLayout defaults to every form, everywhere.
#
# Qt's macOS native style defaults QFormLayout to FieldsStayAtSizeHint +
# FormAlignment=AlignHCenter, which makes forms look shrunk-and-centred.
# We override at __init__ time so all forms — the tabs, the popups, the
# matching/scan/add-element dialogs, anything created in the future —
# uniformly use the cross-platform stretched flush-left look.
# Idempotent: explicit per-site .setFieldGrowthPolicy()/.setFormAlignment()
# calls in convergence_tab.py / error_study_tab.py still win because they
# run after __init__.
# ---------------------------------------------------------------------------
_ORIG_QFORM_INIT = QFormLayout.__init__


def _helix_qform_init(self, *args, **kwargs):
    _ORIG_QFORM_INIT(self, *args, **kwargs)
    self.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    self.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    self.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


QFormLayout.__init__ = _helix_qform_init


_SETTINGS_LAST_LATTICE   = "lastLatticePath"
_SETTINGS_LAST_DIR       = "lastLatticeDir"
_SETTINGS_LAST_PROJECT   = "lastProjectPath"
_SETTINGS_SESSION_BEAM   = "sessionBeamConfig"
_SETTINGS_FONT_SIZE      = "fontSize"
_SETTINGS_CALC_DIR       = "calcDir"
_SETTINGS_RECENT_PROJECTS = "recentProjects"
_SETTINGS_WINDOW_GEOMETRY = "windowGeometry"
_RECENT_PROJECTS_MAX     = 8

# Suffixes handled by the non-TraceWin importers.  Files with these
# suffixes must never be written in place by "Save" — write_tracewin
# would silently overwrite the MAD source with TraceWin text.
_MADX_SUFFIXES = (".madx", ".seq")
_MAD8_SUFFIXES = (".lat", ".flat")
_ELEGANT_SUFFIXES = (".lte",)


def _parse_lattice_file(fp: str):
    """Extension-dispatched lattice parse → ``(lattice, metadata)``.

    One shared implementation for the three GUI load paths (open dialog,
    startup restore, project restore) so a new format only has to be
    wired once.
    """
    suf = Path(fp).suffix.lower()
    if suf in _MADX_SUFFIXES:
        from linac_gen.io.madx_parser import parse_madx
        return parse_madx(fp)
    if suf in _MAD8_SUFFIXES:
        from linac_gen.io.mad8_parser import parse_mad8
        return parse_mad8(fp)
    if suf in _ELEGANT_SUFFIXES:
        from linac_gen.io.elegant_parser import parse_elegant
        return parse_elegant(fp)
    from linac_gen.io.tracewin_parser import parse_tracewin
    return parse_tracewin(fp)
_FONT_MIN, _FONT_MAX     = 9, 22


def _project_start_dir() -> str:
    """Best starting directory for the project open/save dialogs:
    alongside the last project, else the last lattice dir, else cwd
    (dialogs used to always open in the launch directory)."""
    s = _settings()
    lp = s.value(_SETTINGS_LAST_PROJECT)
    if lp and isinstance(lp, str):
        d = os.path.dirname(lp)
        if os.path.isdir(d):
            return d
    ld = s.value(_SETTINGS_LAST_DIR)
    if ld and isinstance(ld, str) and os.path.isdir(ld):
        return ld
    return str(Path.cwd())

# Flipped by the window teardown so late signal deliveries / the
# excepthook know the app is going down and must not pop UI.
_SHUTTING_DOWN = False

# Strong references to worker threads that outlive their owner (retire
# timeout, shutdown deadline): a live QThread whose last Python
# reference is dropped gets destroyed while running, which
# qFatal-aborts the process.  Pruned when each thread finishes.
_PARKED_WORKERS: list = []


def _park_worker(worker) -> None:
    _PARKED_WORKERS.append(worker)

    def _prune() -> None:
        try:
            _PARKED_WORKERS.remove(worker)
        except ValueError:
            pass
    try:
        worker.finished.connect(_prune)
        if not worker.isRunning():   # finished between wait() and here
            _prune()
    except Exception:
        pass


def _install_excepthook() -> None:
    """Route unhandled exceptions to stderr + one non-modal dialog.

    PyQt6 calls ``qFatal`` (→ abort, silently from the user's point of
    view) when a Python exception escapes a slot while ``sys.excepthook``
    is still the interpreter default.  Installing any custom hook makes
    Qt deliver the exception here instead, so a GUI bug degrades into a
    visible traceback rather than a vanished application.
    """
    import traceback

    hook_state: dict = {"dialog": None}

    def _hook(etype, value, tb) -> None:
        # stderr first — must survive even if the Qt side below fails.
        try:
            traceback.print_exception(etype, value, tb)
        except Exception:
            pass
        if issubclass(etype, KeyboardInterrupt):
            return
        try:
            app = QApplication.instance()
            if app is None or app.closingDown() or _SHUTTING_DOWN:
                return
            if hook_state["dialog"] is not None:
                return  # one visible dialog at a time — no popup storms
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Unexpected error")
            short = "".join(
                traceback.format_exception_only(etype, value)).strip()
            box.setText(
                "An unexpected error occurred — the last action may be "
                f"incomplete.\n\n{short}")
            box.setDetailedText(
                "".join(traceback.format_exception(etype, value, tb)))
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            hook_state["dialog"] = box
            box.finished.connect(
                lambda _r: hook_state.update(dialog=None))
            # show(), never exec(): re-entering the event loop from an
            # arbitrary interrupted stack (mid-paint, mid-signal) is how
            # crash dialogs cause second crashes.
            box.show()
        except Exception:
            pass

    sys.excepthook = _hook


def _project_dict_diff(current: dict, saved: dict, *, limit: int = 40,
                       _prefix: str = "") -> list[str]:
    """Recursive field-by-field diff of two project dicts.

    Returns ``"beam.current: 5 → 4.8"``-style lines (saved → current),
    skipping ``__dunder__`` bookkeeping keys.  Floats compare with a
    1e-12 relative tolerance so JSON round-trip noise never reports a
    phantom change.  At most ``limit`` lines + an ellipsis line.
    """
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        return repr(v)

    def eq(a, b):
        if isinstance(a, float) and isinstance(b, (int, float)):
            return abs(a - float(b)) <= 1e-12 * max(1.0, abs(a))
        if isinstance(b, float) and isinstance(a, (int, float)):
            return abs(b - float(a)) <= 1e-12 * max(1.0, abs(b))
        return a == b

    _MISSING = object()
    changed: list[str] = []
    added: list[str] = []

    def walk(cur, old, prefix):
        for k in sorted(set(cur) | set(old)):
            if k.startswith("__"):
                continue
            label = f"{prefix}{k}"
            a, b = old.get(k, _MISSING), cur.get(k, _MISSING)
            if isinstance(a, dict) or isinstance(b, dict):
                # Recurse into sections, treating an absent side as
                # empty so a newly-added section itemizes per field.
                walk(b if isinstance(b, dict) else {},
                     a if isinstance(a, dict) else {}, label + ".")
            elif a is _MISSING:
                # Field the saved file predates (schema evolution) —
                # saving adds it with the value shown.  Collected apart
                # and collapsed to ONE summary line so genuine edits
                # are never buried under schema-default noise.
                added.append(label)
            elif b is _MISSING:
                changed.append(f"{label}: {fmt(a)} → removed")
            elif not eq(a, b) and fmt(a) != fmt(b):
                # fmt(a) == fmt(b) means the difference is below display
                # precision (widget round-trip noise) — a "X → X" line
                # cannot inform a Save/Discard choice, so drop it.
                changed.append(f"{label}: {fmt(a)} → {fmt(b)}")

    walk(current, saved, _prefix)
    lines = list(changed)
    if len(lines) > limit:
        lines = lines[:limit] + [f"… and {len(lines) - limit} more"]
    if added:
        lines.append(
            f"({len(added)} newer-schema field"
            f"{'s' if len(added) != 1 else ''} not in the saved file — "
            "added with the current values on save)")
    return lines


def _recent_projects_load() -> list[str]:
    """Read the recent-projects list from QSettings (JSON-encoded)."""
    import json as _json
    raw = _settings().value(_SETTINGS_RECENT_PROJECTS, "")
    if not raw:
        return []
    try:
        paths = _json.loads(str(raw))
    except (ValueError, TypeError):
        return []
    if not isinstance(paths, list):
        return []
    # Filter to existing files so stale entries (renamed / deleted projects)
    # don't pollute the menu.
    return [p for p in paths
            if isinstance(p, str) and os.path.isfile(p)]


def _recent_projects_save(paths: list[str]) -> None:
    import json as _json
    _settings().setValue(_SETTINGS_RECENT_PROJECTS, _json.dumps(paths))


def _recent_projects_add(path: str) -> list[str]:
    """Add ``path`` to the front of the recent list, dedupe, trim.

    Returns the updated list so the caller can hand it to the toolbar
    without re-reading QSettings.
    """
    path = os.path.abspath(path)
    paths = [p for p in _recent_projects_load() if p != path]
    paths.insert(0, path)
    paths = paths[:_RECENT_PROJECTS_MAX]
    _recent_projects_save(paths)
    return paths


def _default_calc_dir() -> Path:
    """Where to dump run artefacts if the user has not picked a directory.

    Defaults to ``<cwd>/runs``.  Auto-created when first written to.
    """
    return Path.cwd() / "runs"


def _resolve_calc_dir() -> Path:
    """Return the user-configured calc dir, falling back to the default."""
    raw = _settings().value(_SETTINGS_CALC_DIR, "")
    return Path(str(raw)) if raw else _default_calc_dir()


def _settings() -> QSettings:
    from linac_gen_gui.interphase.app_settings import make_settings
    return make_settings("Linac_Gen", "Interphase")

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.scrollwrap import scroll_wrap
from linac_gen_gui.interphase.state import AppState, TABS
from linac_gen_gui.interphase.chrome import TitleBar, Toolbar, StatusBar
from linac_gen_gui.interphase.tabs import (
    BeamTab, LatticeTab, MatchingTab, ConvergenceTab, SurrogatesTab,
    ErrorStudyTab, FailureStudyTab, ResultsTab,
)
from linac_gen_gui.interphase.workers import (
    BacktrackWorker, EnvelopeWorker, MultiparticleWorker,
)


def clamp_rect(frame, avail):
    """Return ``frame`` shrunk to fit inside ``avail`` and translated
    fully on-screen.  Pure geometry (unit-testable); returns an equal
    rect when the frame already fits, so callers can no-op.

    Used when the window starts on — or is dragged to — a screen
    smaller than the geometry it had (laptop ↔ monitor switching):
    without the clamp, part of the window (typically the bottom status
    bar and tab content) sits off-screen and looks like hidden UI.
    """
    from PyQt6.QtCore import QRect
    if avail.contains(frame):
        return QRect(frame)
    out = QRect(frame)
    if out.width() > avail.width():
        out.setWidth(avail.width())
    if out.height() > avail.height():
        out.setHeight(avail.height())
    if out.right() > avail.right():
        out.moveRight(avail.right())
    if out.bottom() > avail.bottom():
        out.moveBottom(avail.bottom())
    if out.left() < avail.left():
        out.moveLeft(avail.left())
    if out.top() < avail.top():
        out.moveTop(avail.top())
    return out


# Tabs → keep in order matching state.TABS
def _tab_qss(base: int | None = None) -> str:
    bs = int(base) if base else theme.FONT_SIZE
    return (
        f"QTabWidget::pane {{ background:{theme.BG_0}; border-top:1px solid {theme.BORDER_0}; "
        f" margin:0; top:-1px; }}"
        f"QTabBar {{ qproperty-drawBase:0; }}"
        f"QTabBar::tab {{ background:{theme.BG_1}; color:{theme.TEXT_2};"
        f" padding:8px 22px; border:1px solid {theme.BORDER_0};"
        f" border-bottom:0; font-size:{bs}px;"
        f" border-top-left-radius:4px; border-top-right-radius:4px; }}"
        f"QTabBar::tab:selected {{ background:{theme.BG_0}; color:{theme.TEXT_0};"
        f" border-top:1px solid {theme.ACCENT}; }}"
        f"QTabBar::tab:!selected:hover {{ color:{theme.TEXT_0}; }}"
    )


class InterphaseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # A fresh window is a fresh session — clear the module-level
        # shutdown latch (matters for test processes that construct and
        # close several windows).
        global _SHUTTING_DOWN
        _SHUTTING_DOWN = False
        self.setWindowTitle("HELIX — Hybrid Envelope-multiparticle LInac eXplorer")
        self.setWindowIcon(icon("atom", 32, theme.ACCENT))
        # Keep the floor BELOW small laptop profiles (a Retina Mac scaled
        # to 'larger text' offers as little as 1168x755 logical): the old
        # 1200x780 minimum exceeded the usable screen, defeated Qt's own
        # restoreGeometry clamping, and forced layouts below their
        # minimums.  Dense tabs now scroll instead of compressing.
        self.setMinimumSize(1000, 640)
        self.resize(1440, 900)
        # Restore the previous session's window geometry (the manual has
        # promised this for a while; saveGeometry lands in closeEvent).
        try:
            geo = _settings().value(_SETTINGS_WINDOW_GEOMETRY)
            if geo is not None:
                self.restoreGeometry(geo)
        except Exception:
            pass
        # Clamp to the CURRENT screen — always, not only when a saved
        # geometry existed: the first-run 1440x900 default also exceeds
        # small laptop profiles, and a geometry saved on the external
        # monitor restores oversized on the laptop (Qt's own
        # restoreGeometry clamp helps but misses the first-run path and
        # every later screen hop).  Deferred so Qt finishes its own
        # geometry adjustments first.
        self._screen_watch_connected = False
        QTimer.singleShot(0, self._clamp_to_screen)

        self.state = AppState()

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self._titlebar = TitleBar(self.state)
        v.addWidget(self._titlebar)

        self._toolbar = Toolbar(self.state)
        v.addWidget(self._toolbar)

        # --- Tabs -------------------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(_tab_qss())
        self._tabs.setDocumentMode(True)

        self.beam_tab        = BeamTab(self.state)
        self.lattice_tab     = LatticeTab(self.state)
        self.matching_tab    = MatchingTab(self.state, self.beam_tab)
        self.convergence_tab = ConvergenceTab(self.state)
        self.surrogates_tab  = SurrogatesTab(self.state)
        self.errors_tab      = ErrorStudyTab(self.state)
        self.failures_tab    = FailureStudyTab(self.state)
        self.results_tab     = ResultsTab(
            self.state,
            open_sigma_cb=self._open_sigma_matrix,
            open_tmatrix_cb=self._open_transfer_matrix,
            open_convergence_cb=lambda: self._tabs.setCurrentIndex(3),
        )
        # Live match preview: stream the matcher's current iterate into
        # opted-in Results popups; restore them on match end.  Bound
        # methods on tab-level relay signals (both tabs live as long as
        # the app — no lifecycle hazard).
        self.matching_tab.preview_results.connect(
            self.results_tab.preview_refresh)
        self.matching_tab.match_ended.connect(self.results_tab.end_preview)

        # Dense form-style pages get a scroll fallback so a small or
        # scaled-down screen degrades to scrolling instead of Qt
        # compressing widgets below their minimums (overlap/clipping).
        # NOT wrapped: LatticeTab (fits the window minimum; its
        # inspector already scrolls internally), FailureStudyTab (only
        # its left control column scrolls — the plot splitter must stay
        # visible), ResultsTab (scrolls its card grid internally).
        _WRAPPED = {"beam", "matching", "convergence", "surrogates",
                    "errors"}
        for (tid, label), widget in zip(
            TABS,
            [self.beam_tab, self.lattice_tab, self.matching_tab,
             self.convergence_tab, self.surrogates_tab, self.errors_tab,
             self.failures_tab, self.results_tab],
        ):
            self._tabs.addTab(
                scroll_wrap(widget) if tid in _WRAPPED else widget, label)

        self._tabs.currentChanged.connect(self.state.set_tab)
        v.addWidget(self._tabs, stretch=1)

        # Error studies must run with the SAME space-charge setup as the
        # toolbar MP run — hand the Error Study tab the canonical builder.
        self.errors_tab.sc_config_provider = (
            lambda current, continuous=False:
                self.convergence_tab.current_sc_config(
                    current, continuous=continuous))

        self._statusbar = StatusBar(self.state)
        v.addWidget(self._statusbar)

        # --- Toolbar wiring -----------------------------------------------
        self._toolbar.open_lattice_requested.connect(self._open_lattice)
        self._toolbar.save_lattice_requested.connect(self._save_lattice)
        self._toolbar.save_lattice_as_requested.connect(self._save_lattice_as)
        self._toolbar.open_project_requested.connect(self._open_project)
        self._toolbar.save_project_requested.connect(self._save_project)
        self._toolbar.run_envelope_requested.connect(self._run_envelope)
        self._toolbar.run_mp_requested.connect(self._run_mp)
        self._toolbar.run_backtrack_requested.connect(self._run_backtrack)
        self._toolbar.open_assistant_requested.connect(self._open_assistant)
        self._toolbar.open_console_requested.connect(self._open_console)
        self._toolbar.open_transfer_matrix_requested.connect(self._open_transfer_matrix)
        self._toolbar.open_sigma_matrix_requested.connect(self._open_sigma_matrix)
        self._toolbar.open_docs_requested.connect(self._open_docs)
        self._toolbar.open_about_requested.connect(self._open_about)
        self._toolbar.open_parameter_scan_requested.connect(self._open_parameter_scan)
        self._toolbar.stop_requested.connect(self._stop_active_worker)
        self._toolbar.font_size_changed.connect(self._apply_font_size)
        self._toolbar.export_tracewin_requested.connect(self._export_tracewin)
        self._toolbar.export_openpmd_requested.connect(self._export_openpmd)
        self._toolbar.set_calc_dir_requested.connect(self._set_calc_dir)
        self._toolbar.open_recent_requested.connect(self._open_recent_project)
        self._toolbar.clear_recent_requested.connect(self._clear_recent_projects)
        # Populate the Recent Projects submenu at startup so the user sees
        # their history immediately after launching, before any open/save.
        self._toolbar.set_recent_projects(_recent_projects_load())

        # Lattice tab wiring (its own Open/Save buttons)
        self.lattice_tab.open_requested.connect(self._open_lattice)
        self.lattice_tab.save_requested.connect(self._save_lattice)
        self.lattice_tab.save_as_requested.connect(self._save_lattice_as)

        # Status messages are shown by StatusBar itself, which connects
        # state.status_message → a transient, auto-clearing message segment.
        # (The old hook here dropped the text and force-repainted the whole
        # bar on every emit, so no message was ever visible.)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._open_lattice)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_lattice)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._run_envelope)
        QShortcut(QKeySequence("Ctrl+Shift+R"), self, activated=self._run_mp)
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self._bump_font(+1))
        QShortcut(QKeySequence("Ctrl++"), self, activated=lambda: self._bump_font(+1))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self._bump_font(-1))
        QShortcut(QKeySequence("Ctrl+0"), self, activated=lambda: self._apply_font_size(theme.FONT_SIZE))
        QShortcut(QKeySequence("F1"), self, activated=self._open_manual_for_selected)

        # Restore persisted font size, if any.
        s = _settings()
        try:
            saved_pt = int(s.value(_SETTINGS_FONT_SIZE, theme.FONT_SIZE))
        except (TypeError, ValueError):
            saved_pt = theme.FONT_SIZE
        saved_pt = max(_FONT_MIN, min(_FONT_MAX, saved_pt))
        self._toolbar.set_font_size(saved_pt)
        self._apply_font_size(saved_pt, persist=False)

        self._envelope_worker: EnvelopeWorker | None = None
        self._mp_worker: MultiparticleWorker | None = None
        self._backtrack_worker: BacktrackWorker | None = None
        # Background cache warmer disabled — was competing for CPU on
        # project / lattice load.  The on-demand cache path (Phase
        # Advance / Tune Depression popups passing ``cache=``) still
        # works; the first call just isn't pre-warmed.

        # Persist the working beam across sessions (see _save_session_beam).
        # Connected BEFORE the restore so a project-restore's beam is also
        # captured as the session beam.
        self.state.beam_config_changed.connect(self._save_session_beam)
        QTimer.singleShot(0, self._restore_last_session)
        # Catch macOS Cmd+Q / system-menu Quit which doesn't fire the
        # window's closeEvent.  Has to run after QApplication exists,
        # which it does by the time __init__ runs (the launcher always
        # creates the app first), but defer one tick to be safe.
        QTimer.singleShot(0, self._install_quit_event_filter)

    # ------------------------------------------------------------------
    # Itemized unsaved-changes details for the discard prompt ("Show
    # Details…").  Best-effort: any failure returns what it has — the
    # prompt itself must never break over its details.
    def _unsaved_change_details(self, lattice_dirty: bool,
                                project_dirty: bool) -> str:
        sections: list[str] = []
        if lattice_dirty:
            try:
                bus = getattr(self.state, "bus", None)
                lines = (bus.describe_changes_since_clean()
                         if bus is not None else [])
            except Exception:                                 # noqa: BLE001
                lines = ["(itemization failed)"]
            sections.append("Lattice edits since the last save:\n"
                            + "\n".join(f"  • {ln}" for ln in lines))
        if project_dirty:
            try:
                lines = self._project_diff_lines()
            except Exception:                                 # noqa: BLE001
                lines = ["(comparison with the saved .lgproj failed)"]
            sections.append("Project settings vs the saved .lgproj:\n"
                            + "\n".join(f"  • {ln}" for ln in lines))
        return "\n\n".join(sections)

    def _project_diff_lines(self, limit: int = 40) -> list[str]:
        """Field-by-field diff of the live project vs the on-disk file."""
        import json
        path = _settings().value(_SETTINGS_LAST_PROJECT)
        if not path or not os.path.isfile(str(path)):
            return ["project was never saved — all current settings "
                    "are unsaved"]
        path = str(path)
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        current = self._collect_project_dict(None, project_path=path)
        lines = _project_dict_diff(current, saved, limit=limit)
        if not lines:
            return [f"no field differences vs {os.path.basename(path)} "
                    "(edited then restored?)"]
        return lines

    # ------------------------------------------------------------------
    # Dirty / unsaved-changes prompt.  Returns True if the caller should
    # proceed with the destructive action (Open, Reload, Quit), False if
    # the user picked Cancel.  "Discard" returns True without saving.
    def _confirm_discard(self, title: str = "Unsaved changes") -> bool:
        bus = getattr(self.state, "bus", None)
        lattice_dirty = bool(bus is not None and bus.dirty)
        project_dirty = bool(getattr(self.state, "project_dirty", False))
        if not lattice_dirty and not project_dirty:
            return True
        # Build the prompt text from whichever flags are set so the user
        # knows exactly what's at risk (lattice file, project .lgproj,
        # or both).  Save offers the matching action; if both are dirty,
        # Save runs both writers in sequence.  (For MAD-X/MAD8 sources
        # "Save" routes to Save-As — HELIX never overwrites them.)
        parts = []
        if lattice_dirty:
            parts.append("the lattice file")
        if project_dirty:
            parts.append("the project .lgproj")
        what = " and ".join(parts)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(f"You have unsaved changes to {what}.")
        box.setInformativeText("Save your changes before continuing?")
        try:
            details = self._unsaved_change_details(lattice_dirty,
                                                   project_dirty)
            if details:
                # Native "Show Details…" section — itemizes exactly what
                # is at risk so Save/Discard is an informed choice.
                box.setDetailedText(details)
        except Exception:                                     # noqa: BLE001
            pass
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            if lattice_dirty:
                self._save_lattice()
            if project_dirty:
                self._save_project()
            # Proceed only if BOTH writers succeeded (their handlers
            # clear the corresponding dirty flag on success).
            still_lattice_dirty = bool(bus is not None and bus.dirty)
            still_project_dirty = bool(getattr(self.state,
                                               "project_dirty", False))
            return not (still_lattice_dirty or still_project_dirty)
        if choice == QMessageBox.StandardButton.Discard:
            return True
        return False

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        # windowHandle() only exists once the platform window does —
        # here it is guaranteed.  Watch for the window being moved to a
        # different screen (drag across monitors, monitor unplugged) and
        # re-clamp; a bool guard because Qt.UniqueConnection is
        # unreliable for Python slots.
        if not self._screen_watch_connected and self.windowHandle() is not None:
            self.windowHandle().screenChanged.connect(self._on_screen_changed)
            self._screen_watch_connected = True
        QTimer.singleShot(0, self._clamp_to_screen)

    def _on_screen_changed(self, _screen) -> None:
        # Defer: let Qt finish its own DPR/geometry adjustment for the
        # new screen before measuring.  The clamp is idempotent, so a
        # clamp-triggered move cannot loop.
        QTimer.singleShot(0, self._clamp_to_screen)

    def _clamp_to_screen(self) -> None:
        """Shrink/move the window so it fits the screen it is on."""
        if _SHUTTING_DOWN:
            return
        try:
            scr = self.screen()
        except RuntimeError:
            return   # deferred call outlived the window (test teardown)
        if scr is None and self.windowHandle() is not None:
            scr = self.windowHandle().screen()
        if scr is None:
            return
        avail = scr.availableGeometry()
        frame = self.frameGeometry()
        target = clamp_rect(frame, avail)
        if target == frame:
            return
        # frameGeometry includes the native title bar — convert the
        # clamped frame back to a client-area size.
        delta_w = frame.width() - self.width()
        delta_h = frame.height() - self.height()
        self.resize(max(self.minimumWidth(), target.width() - delta_w),
                    max(self.minimumHeight(), target.height() - delta_h))
        self.move(target.topLeft())

    def closeEvent(self, ev) -> None:
        global _SHUTTING_DOWN
        if _SHUTTING_DOWN:
            ev.accept()
            return
        if not self._confirm_discard("Quit"):
            ev.ignore()
            return
        _SHUTTING_DOWN = True
        try:
            _settings().setValue(_SETTINGS_WINDOW_GEOMETRY,
                                 self.saveGeometry())
        except Exception:
            pass
        self._shutdown_workers()
        ev.accept()

    def _shutdown_workers(self) -> None:
        """Stop every background thread before the window goes away.

        Quitting used to accept the close with workers still running —
        Qt then destroys live QThreads on the way out, which aborts the
        process ("QThread: Destroyed while thread is still running").

        Sequence (one shared 5 s deadline):
        1. Signal everything, wait nothing — cooperative stop flags on
           the envelope/MP workers and every tab worker via
           ``shutdown_begin()`` (which also closes progress dialogs
           first: a worker blocked in a BlockingQueuedConnection emit is
           only released when its receiver dies).
        2. Pumped bounded wait — ``processEvents`` + short ``wait()``
           round-robin, so queued worker signals keep draining while we
           join (a plain wait() on the GUI thread would deadlock any
           blocking-queued emitter).
        3. Survivors (e.g. a match mid-cost-evaluation, ~30 s latency)
           are parked in a module list — never GC'd alive — settings are
           flushed, and the process exits via os._exit(0): it skips the
           Qt destructors (no qFatal) and the concurrent.futures atexit
           join (no multi-minute hang).  Never QThread.terminate().
        """
        from PyQt6.QtCore import QElapsedTimer, QEventLoop

        workers: list = []
        for w in (self._envelope_worker, self._mp_worker,
                  self._backtrack_worker):
            if w is not None and w.isRunning():
                try:
                    w.request_stop()
                except Exception:
                    pass
                workers.append(w)
        for tab in (self.matching_tab, self.convergence_tab,
                    self.errors_tab, self.failures_tab,
                    self.surrogates_tab, self.results_tab,
                    self.lattice_tab):
            begin = getattr(tab, "shutdown_begin", None)
            if callable(begin):
                try:
                    workers.extend(begin() or [])
                except Exception:
                    pass
        # Dialog-owned workers the tab sweep doesn't see.
        # Assistant panel: close it FIRST (its closeEvent cancels the SDK
        # turn + shuts down its job pool); then sweep its agent worker so a
        # long compute can't leave a running QThread at interpreter exit.
        asst = getattr(self, "_assistant_panel", None)
        if asst is not None:
            try:
                asst.close()
            except Exception:
                pass
            aw = getattr(asst, "_worker", None)
            if aw is not None and hasattr(aw, "isRunning") \
                    and aw.isRunning():
                workers.append(aw)
        ps_dlg = getattr(self, "_param_scan_dlg", None)
        psw = getattr(ps_dlg, "_worker", None) if ps_dlg is not None else None
        if psw is not None and hasattr(psw, "isRunning") and psw.isRunning():
            try:
                psw.request_stop()
            except Exception:
                pass
            workers.append(psw)
        # Long-retired workers parked to avoid GC of a live QThread — hand
        # them to the same bounded wait / os._exit path, or interpreter
        # teardown destroys a running thread and qFatal-aborts on quit.
        workers.extend(_PARKED_WORKERS)
        workers = [w for w in workers
                   if w is not None and hasattr(w, "isRunning")
                   and w.isRunning()]
        if not workers:
            return

        # Sever every GUI-facing connection before pumping: a worker
        # finishing during the wait would otherwise invoke its normal
        # completion handler — modal result dialogs and Save-PNG pickers
        # opening mid-quit stall the teardown.
        for w in workers:
            try:
                w.disconnect()
            except (TypeError, RuntimeError):
                pass

        app = QApplication.instance()
        timer = QElapsedTimer()
        timer.start()
        while workers and timer.elapsed() < 5000:
            if app is not None:
                app.processEvents(
                    QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            for w in list(workers):
                try:
                    if not w.isRunning() or w.wait(25):
                        workers.remove(w)
                except Exception:
                    workers.remove(w)

        if workers:
            _PARKED_WORKERS.extend(workers)
            try:
                _settings().sync()
            except Exception:
                pass
            print(f"[shutdown] {len(workers)} worker thread(s) still "
                  "running after 5 s — forcing process exit.",
                  file=sys.stderr)
            os._exit(0)

    def _install_quit_event_filter(self) -> None:
        """Hook the application-level Quit event so macOS Cmd+Q (and
        other paths that bypass the window's closeEvent) also fire
        ``_confirm_discard``."""
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Catch macOS Cmd+Q / App menu → Quit which bypasses the
        window's closeEvent.  Qt only fires closeEvent when the red
        close button is clicked; the system Quit menu sends a
        ``QEvent.Quit`` directly to the application instance instead.
        The Quit is swallowed and re-routed through ``self.close()`` so
        there is exactly ONE teardown path (confirm prompt + worker
        shutdown live in closeEvent); if the user cancels, close() is a
        no-op and the app stays open.
        """
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.Quit and not _SHUTTING_DOWN:
            self.close()
            return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # File / lattice
    def _open_lattice(self) -> None:
        if not self._confirm_discard("Open Lattice"):
            return
        s = _settings()
        start_dir = str(s.value(_SETTINGS_LAST_DIR, str(Path.cwd())))
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Lattice", start_dir,
            "Lattice files (*.dat *.madx *.seq *.lat *.flat *.lte);;"
            "TraceWin (*.dat);;MAD-X (*.madx *.seq);;"
            "MAD8 (*.lat *.flat);;Elegant (*.lte);;All Files (*)"
        )
        if not fp:
            return
        self._detach_workers_for_new_lattice()
        try:
            # Route by file extension: MAD-X (.madx/.seq) and MAD8
            # (.lat/.flat) → the in-house subset parsers; everything
            # else → the TraceWin parser.
            lattice, meta = _parse_lattice_file(fp)
            self.state.set_lattice(lattice, fp)
            s.setValue(_SETTINGS_LAST_LATTICE, fp)
            s.setValue(_SETTINGS_LAST_DIR, str(Path(fp).parent))
            # A bare-lattice open ends any project session — drop the
            # remembered project so the next launch restores this lattice.
            s.remove(_SETTINGS_LAST_PROJECT)
            msg = f"Loaded {len(lattice.elements)} elements"
            warnings = meta.get("warnings", []) if isinstance(meta, dict) else []
            if warnings:
                msg += f"  ·  {len(warnings)} warning(s) — see console"
                for w in warnings:
                    print(f"[lattice import] {w}")
            self.state.status_message.emit(msg)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def _apply_font_size(self, pt: int, *, persist: bool = True) -> None:
        """Re-render the QSS at the requested base size and apply it
        application-wide.  Optionally persist the new value."""
        pt = max(_FONT_MIN, min(_FONT_MAX, int(pt)))
        app = QApplication.instance()
        if app is not None:
            # Setting the app font means any widget *without* an explicit
            # inline font-size (e.g. inline `setStyleSheet` calls scattered
            # across panels) inherits the new size from Qt's font system.
            font = app.font()
            font.setPointSize(pt)
            app.setFont(font)
            app.setStyleSheet(theme.dark_qss(base=pt))
        # The tab-bar uses its own scoped QSS.
        self._tabs.setStyleSheet(_tab_qss(base=pt))
        # Forward to chrome widgets AND every tab page so inline-styled
        # labels (titlebar path, statusbar segments, inspector rows, …)
        # also scale — the app-wide font only reaches widgets without an
        # explicit inline font-size.  Direct page references, NOT
        # self._tabs.widget(i): pages may be wrapped in QScrollAreas,
        # and widget(i) would return the wrapper (forwarding would then
        # die silently).
        for w in (self._titlebar, self._toolbar, self._statusbar,
                  self.beam_tab, self.lattice_tab, self.matching_tab,
                  self.convergence_tab, self.surrogates_tab,
                  self.errors_tab, self.failures_tab, self.results_tab):
            apply = getattr(w, "apply_font_size", None)
            if callable(apply):
                apply(pt)
        if persist:
            _settings().setValue(_SETTINGS_FONT_SIZE, pt)
        self._toolbar.set_font_size(pt)
        self.state.status_message.emit(f"Font size {pt} pt")

    def _bump_font(self, delta: int) -> None:
        cur = int(_settings().value(_SETTINGS_FONT_SIZE, theme.FONT_SIZE) or theme.FONT_SIZE)
        self._apply_font_size(cur + delta)

    def _open_manual_for_selected(self) -> None:
        """F1: open the manual chapter for the currently-selected element."""
        sel = self.state.selected
        if sel is None:
            self.state.status_message.emit("F1: select a lattice element first")
            return
        from linac_gen_gui.interphase.manual_help import open_for_element
        ok, msg = open_for_element(sel)
        self.state.status_message.emit(msg)

    def _restore_last_lattice(self) -> None:
        """On startup, reload the last successfully-opened lattice if its
        path is still valid.  Silent on missing file or parse error — we
        don't want a popup before the user has done anything."""
        s = _settings()
        fp = s.value(_SETTINGS_LAST_LATTICE)
        if not fp or not isinstance(fp, str) or not os.path.exists(fp):
            return
        try:
            # Route by extension, exactly as _open_lattice does — restoring a
            # remembered .madx/.seq/.lat lattice through the TraceWin parser
            # mis-parsed it and then wiped the saved path in the except below.
            lattice, _meta = _parse_lattice_file(fp)
            self.state.set_lattice(lattice, fp)
            self.state.status_message.emit(
                f"Restored {os.path.basename(fp)} — {len(lattice.elements)} elements")
        except Exception:
            # Stale entry: clear it so we don't keep retrying on every launch.
            s.remove(_SETTINGS_LAST_LATTICE)

    def _restore_last_project(self, path: str) -> bool:
        """Reload a .lgproj silently on startup.  Returns True on success.

        Mirrors ``_restore_last_lattice`` — no popups before the user has
        acted; ``_apply_project_dict(silent=True)`` downgrades its warnings
        to console messages."""
        try:
            import json
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return False
            warns = self._apply_project_dict(data, project_path=path,
                                             silent=True)
            suffix = (f" — {len(warns)} warning(s), see console"
                      if warns else "")
            self.state.status_message.emit(
                f"Restored project {os.path.basename(path)}{suffix}")
            return True
        except Exception:
            return False

    def _restore_last_session(self) -> None:
        """On startup, reload the last project if there is one, else fall
        back to the last bare lattice.  A project already carries its own
        lattice, so the two are mutually exclusive — never load both.

        The bare-lattice fallback additionally restores the LAST SESSION'S
        BEAM: previously the beam silently reset to widget defaults
        (~2 MeV H-), which paired a correct-looking lattice with a wrong
        beam after any mid-work GUI restart (user-reported: BTL at
        800 MeV ran "completely off" until the beam was re-entered)."""
        fp = _settings().value(_SETTINGS_LAST_PROJECT)
        if fp and isinstance(fp, str) and os.path.exists(fp):
            if self._restore_last_project(fp):
                return
            # Stale / unparseable project — clear it so we stop retrying.
            _settings().remove(_SETTINGS_LAST_PROJECT)
        self._restore_last_lattice()
        self._restore_session_beam()

    def _save_session_beam(self, cfg) -> None:
        """Persist the working beam config across sessions (QSettings).

        Connected to ``state.beam_config_changed`` — fires at run starts,
        matching, and project loads, i.e. whenever a beam is actually
        used, not per keystroke."""
        try:
            import dataclasses, json
            _settings().setValue(_SETTINGS_SESSION_BEAM,
                                 json.dumps(dataclasses.asdict(cfg)))
        except Exception:
            pass                        # persistence must never break a run

    def _restore_session_beam(self) -> None:
        """Bare-lattice startup path: restore the last session's beam so a
        restart reproduces what the user was actually running."""
        raw = _settings().value(_SETTINGS_SESSION_BEAM)
        if not raw or not isinstance(raw, str):
            return
        try:
            import json
            from linac_gen.core.config import BeamConfig
            d = json.loads(raw)
            d = {k: v for k, v in d.items()
                 if k in BeamConfig.__dataclass_fields__}
            cfg = BeamConfig(**d)
            self.beam_tab.set_beam_config(cfg)
            self.state.set_beam_config(cfg)
            self.state.status_message.emit(
                f"Restored last session's beam ({cfg.species}, "
                f"{cfg.energy:g} MeV, {cfg.current:g} mA)")
        except Exception:
            # Malformed/old entry: drop it so we stop retrying.
            _settings().remove(_SETTINGS_SESSION_BEAM)

    def _save_lattice(self) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Open a lattice first."); return
        lp = self.state.lattice_path
        if getattr(self.state, "lattice_fitted", False):
            # The in-memory lattice carries FITTED values (matcher Apply /
            # orbit-correction apply) that the opened deck does not.
            # Plain Save used to overwrite the source in place — exactly
            # how a curated matching fixture once lost its documentation
            # to optimizer output.  Reroute to Save-As with a suggested
            # "<stem>.matched.dat"; picking the original path in the
            # dialog remains possible but is now an explicit choice.
            self.state.status_message.emit(
                "lattice carries fitted values — choose a file "
                "(suggested: a new .matched.dat) or overwrite explicitly")
            if lp:
                stem = str(Path(lp).with_suffix(""))
                if not stem.endswith(".matched"):
                    stem += ".matched"   # never .matched.matched.dat
                self._save_lattice_as(suggest=stem + ".dat")
            else:
                self._save_lattice_as()
            return
        if lp and Path(lp).suffix.lower() not in (_MADX_SUFFIXES
                                                  + _MAD8_SUFFIXES
                                                  + _ELEGANT_SUFFIXES):
            self._write_lattice(lp)
        else:
            # No path yet, OR the lattice came from a MAD-X/MAD8 file.
            # HELIX only writes TraceWin .dat — Ctrl+S used to silently
            # OVERWRITE the user's .madx/.seq (and would the .lat) source
            # with TraceWin millimeter text (and mark it clean).  Route to
            # Save-As, whose filter is *.dat.
            self._save_lattice_as()

    def _save_lattice_as(self, suggest: str | None = None) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Open a lattice first."); return
        s = _settings()
        start_dir = str(s.value(_SETTINGS_LAST_DIR, str(Path.cwd())))
        start = suggest if suggest else start_dir
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Lattice As", start, "TraceWin Files (*.dat)")
        if not fp:
            return
        if not self._write_lattice(fp):
            # Failed write: leave path/flag untouched — state must not
            # point at a file that was never written.
            return
        self.state._lattice_path = fp  # quietly update
        # The chosen file now holds the fitted values — Save is honest
        # against it again.
        self.state.lattice_fitted = False
        s.setValue(_SETTINGS_LAST_LATTICE, fp)
        s.setValue(_SETTINGS_LAST_DIR, str(Path(fp).parent))

    def _write_lattice(self, path: str) -> bool:
        try:
            import warnings as _warnings

            from linac_gen.io.tracewin_writer import write_tracewin
            # Surface writer warnings — they are load-bearing: "deck not
            # relocatable (absolute field-map paths)" and "ERROR_* study
            # cards dropped from the .dat" must never vanish to stderr
            # while the GUI shows a clean "Saved".
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                write_tracewin(self.state.lattice, path)
            # Clear the dirty flag — the on-disk file now matches state.
            bus = getattr(self.state, "bus", None)
            if bus is not None:
                bus.mark_clean()
            msg = f"Saved → {os.path.basename(path)}"
            if caught:
                msg += (f"  ·  {len(caught)} save warning(s) — see console")
                for w in caught:
                    print(f"[lattice save] {w.message}")
            self.state.status_message.emit(msg)
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False

    # ------------------------------------------------------------------
    # Project file (beam + convergence + SC + backend settings as JSON).
    #
    # A "project" is a self-describing JSON snapshot of everything the
    # user has configured in the GUI — *not* the lattice file itself,
    # which stays in its .dat form and is referenced by path.  Loading a
    # project restores all widgets (Beam tab form values, Convergence
    # tab step1/step2/grid/extent/backend, record-substeps checkbox) and
    # optionally re-opens the lattice file if the path is still valid.
    def _collect_project_dict(self, warnings: list[str] | None = None,
                              project_path: str | None = None) -> dict:
        """Snapshot every user-settable GUI value into a plain dict ready
        for JSON serialisation.  Does not include the lattice — only a
        reference to its path on disk.

        When ``project_path`` is given (the .lgproj destination), the
        lattice path and the beam's ``distribution_file`` are stored
        RELATIVE to it so the saved project is relocatable — move the
        directory, the project still opens.  Paths with no usable common
        ancestor (other drive / root-only) stay absolute with a status
        note.  Without ``project_path`` (dirty-check snapshots) paths are
        stored as-is.

        Sections that cannot be captured (e.g. an invalid Beam form) are
        skipped and described in ``warnings`` so the caller can tell the
        user instead of silently writing a project with missing data.
        """
        from dataclasses import asdict
        from linac_gen.io.portable_paths import best_relpath

        anchor = (os.path.dirname(os.path.abspath(project_path))
                  if project_path else None)

        def _portable(p):
            """Relativize against the project dir; absolute fallback."""
            if not p or anchor is None:
                return p
            rel, ok = best_relpath(p, anchor)
            if not ok:
                # NOT added to `warnings` — that list drives the blocking
                # "save without them?" dialog; a non-relocatable path is
                # worth a status note, not a modal.
                self.state.status_message.emit(
                    f"Note: {os.path.basename(str(p))} kept as an absolute "
                    "path (no common directory with the project) — the "
                    "project will not be relocatable.")
                return p
            return rel

        out: dict = {
            "__kind__": "linac_gen_project",
            "__version__": 1,
            "lattice_path": _portable(self.state.lattice_path),
        }
        # Beam config — pull latest from the widgets (covers unapplied edits).
        try:
            out["beam"] = asdict(self.beam_tab.get_beam_config())
            if out["beam"].get("distribution_file"):
                out["beam"]["distribution_file"] = _portable(
                    out["beam"]["distribution_file"])
        except Exception as exc:
            if warnings is not None:
                warnings.append(f"Beam settings not saved — the form is "
                                f"invalid: {exc}")
        # Orbit-correction defaults (used by both the Error Study tab
        # checkbox group and the standalone Lattice-tab button).
        out["correction"] = dict(self.state.correction_settings)
        out["auto_correction_mode"] = str(self.state.auto_correction_mode)
        # Calculation directory — where auto-saved HDF5 dumps land after
        # each run.  Only included if the user has explicitly picked one
        # (we don't want to bake the default ``<cwd>/runs`` into the
        # project file, since cwd is launch-time-dependent and would
        # mislead anyone opening this project from a different working
        # directory).
        _raw_calc = _settings().value(_SETTINGS_CALC_DIR, "")
        if _raw_calc:
            out["calc_dir"] = str(_raw_calc)
        # Numerics tab: integration / SC cadence, PIC grid, backend,
        # record-substeps flag.
        ct = self.convergence_tab
        out["convergence"] = {
            "step1_per_m":      float(ct._fixed_step1.value()),
            "step2_per_m":      float(ct._fixed_step2.value()),
            "grid_nx":          int(ct._fixed_nx.value()),
            "grid_extent_sigma":float(ct._fixed_ext.value()),
            "backend":          str(ct._fixed_backend.currentText()),
            "sc_backend":       str(ct._fixed_sc_backend.currentText()),
            "green_kind":       str(ct._fixed_green.currentText()),
            "kernel":           str(ct._fixed_kernel.currentText()),
            "grid_mode":        str(ct._fixed_grid_mode.currentText()),
            "dc_kernel":        str(ct._fixed_dc_kernel.currentText()),
            "integrator_kind":  str(ct._fixed_integ.currentText()),
            "interp_kind":      str(ct._fixed_interp.currentText()),
            "fieldmap_sampling": str(ct._fieldmap_sampling.currentText()),
            "env_solver":       str(ct._fixed_env_solver.currentText()),
            "record_substeps":  bool(ct._record_substeps.isChecked()),
            "record_density":   bool(ct._record_density.isChecked()),
            "snapshot_every_n": int(ct._snapshot_every_n.value()),
            "snapshot_elements": ct._snapshot_elements.text(),
            "density_bins":     int(ct._density_bins.value()),
            "density_extent":   float(ct._density_extent.value()),
            "csr_enabled":      bool(ct._fixed_csr.isChecked()),
        }
        return out

    def _apply_project_dict(self, data: dict, project_path: str | None = None,
                            *, silent: bool = False) -> list[str]:
        """Inverse of ``_collect_project_dict`` — push values back into
        the widgets.  Missing keys are tolerated so older / partial
        project files still load.

        ``project_path`` is the on-disk path of the .lgproj being
        loaded; used to resolve a ``lattice_path`` that doesn't exist
        as-stored (typical when a project authored on WSL is opened
        on Windows or vice-versa) by trying ``<project_dir>/<basename>``
        as a fallback.

        ``silent`` (used by the startup restore) downgrades the warning
        popups to console messages so nothing pops up before the user
        has interacted with the window.

        Returns the list of section warnings so callers can qualify
        their "Project loaded" message instead of claiming success
        unconditionally.  Non-silent loads get ONE consolidated dialog
        at the end rather than a popup per failed section.
        """
        warnings: list[str] = []

        def _warn(title: str, msg: str) -> None:
            warnings.append(f"{title}: {msg}")
            if silent:
                print(f"[restore] {title}: {msg}")
        # Lattice first, since convergence config can depend on the
        # lattice being present.
        lp = data.get("lattice_path")
        if lp:
            # Resolution order (first existing wins):
            #   relative lp → <project_dir>/<lp>  (the portable form
            #                 written by _collect_project_dict — must be
            #                 tried FIRST: a bare os.path.exists on a
            #                 relative path only works if cwd happens to
            #                 match), then cwd (legacy repo-root-relative
            #                 projects like csr_chicane.lgproj);
            #   absolute lp → as-is (legacy projects);
            #   last resort → <project_dir>/<basename(lp)> — handles
            #                 WSL ↔ Windows path mismatches when the .dat
            #                 sits next to the .lgproj (the convention in
            #                 examples/).
            candidates = []
            if not os.path.isabs(lp) and project_path:
                candidates.append(
                    os.path.join(os.path.dirname(project_path), lp))
            candidates.append(lp)
            if project_path:
                candidates.append(
                    os.path.join(os.path.dirname(project_path),
                                 os.path.basename(lp)))
            resolved = next((os.path.abspath(c) for c in candidates
                             if os.path.exists(c)), None)
            if resolved:
                try:
                    # Route by extension, as _open_lattice does — a project
                    # referencing a .madx/.seq/.lat lattice otherwise
                    # mis-loaded through the TraceWin parser.
                    lattice, _ = _parse_lattice_file(resolved)
                    self._detach_workers_for_new_lattice()
                    self.state.set_lattice(lattice, resolved)
                except Exception as exc:
                    _warn("Lattice", f"Could not reload {resolved}:\n{exc}")
            else:
                _warn("Lattice", f"Lattice file not found:\n{lp}")
        # Beam form
        beam = data.get("beam")
        if isinstance(beam, dict):
            try:
                from linac_gen.core.config import BeamConfig
                # Resolve a relative distribution_file against the project
                # dir (then cwd) — runtime (factory.load_dst) consumes the
                # path verbatim, so it must be absolute in memory.
                df = beam.get("distribution_file")
                if df and not os.path.isabs(str(df)):
                    from linac_gen.io.portable_paths import resolve_candidates
                    bases = ([os.path.dirname(project_path)]
                             if project_path else []) + [os.getcwd()]
                    hit = resolve_candidates(str(df), bases)
                    if hit:
                        beam = dict(beam)
                        beam["distribution_file"] = hit
                    else:
                        _warn("Beam", f"Distribution file not found:\n{df}")
                # Tolerate unknown keys by filtering to the dataclass fields.
                fields = {f.name for f in BeamConfig.__dataclass_fields__.values()}
                cfg = BeamConfig(**{k: v for k, v in beam.items() if k in fields})
                self.beam_tab.set_beam_config(cfg)
                # Push into app state too.  Updating only the widgets left
                # state.beam_config holding the PREVIOUS project's applied
                # config — every run then silently used the old beam while
                # the form showed the new one.
                self.state.set_beam_config(cfg)
            except Exception as exc:
                _warn("Beam", f"Beam config in project is invalid:\n{exc}")
        # Numerics tab.  Guarded like the beam section: one malformed
        # value used to abort the whole load mid-way with a generic
        # error, leaving the widgets half-restored.
        conv = data.get("convergence", {})
        ct = self.convergence_tab
        try:
            if "step1_per_m" in conv:     ct._fixed_step1.setValue(float(conv["step1_per_m"]))
            if "step2_per_m" in conv:     ct._fixed_step2.setValue(float(conv["step2_per_m"]))
            if "grid_nx" in conv:         ct._fixed_nx.setValue(int(conv["grid_nx"]))
            if "grid_extent_sigma" in conv: ct._fixed_ext.setValue(float(conv["grid_extent_sigma"]))
            if "backend" in conv:
                idx = ct._fixed_backend.findText(str(conv["backend"]))
                if idx >= 0:
                    ct._fixed_backend.setCurrentIndex(idx)
            for key, widget in (("sc_backend",      ct._fixed_sc_backend),
                                ("green_kind",      ct._fixed_green),
                                ("kernel",          ct._fixed_kernel),
                                ("grid_mode",       ct._fixed_grid_mode),
                                ("dc_kernel",       ct._fixed_dc_kernel),
                                ("integrator_kind", ct._fixed_integ),
                                ("interp_kind",     ct._fixed_interp),
                                ("fieldmap_sampling", ct._fieldmap_sampling),
                                ("env_solver",      ct._fixed_env_solver)):
                if key in conv:
                    if not widget.isEnabled():
                        # e.g. Sampling on a machine without the compiled
                        # kernel: keep the honest machine state ("scipy")
                        # instead of displaying a value that cannot apply.
                        continue
                    idx = widget.findText(str(conv[key]))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
            if "record_substeps" in conv:
                ct._record_substeps.setChecked(bool(conv["record_substeps"]))
            if "record_density" in conv:
                ct._record_density.setChecked(bool(conv["record_density"]))
            if "csr_enabled" in conv:
                ct._fixed_csr.setChecked(bool(conv["csr_enabled"]))
            if "snapshot_every_n" in conv:
                ct._snapshot_every_n.setValue(int(conv["snapshot_every_n"]))
            if "snapshot_elements" in conv:
                ct._snapshot_elements.setText(str(conv["snapshot_elements"]))
            if "density_bins" in conv:
                ct._density_bins.setValue(int(conv["density_bins"]))
            if "density_extent" in conv:
                ct._density_extent.setValue(float(conv["density_extent"]))
        except Exception as exc:
            _warn("Numerics",
                  f"Numerics settings in project are invalid:\n{exc}")

        # Calculation directory — apply before the run-time tabs so any
        # status message about "saving to <dir>" sees the project's value
        # rather than the previous user setting.  Relative paths are
        # resolved against the project file's directory so a project +
        # its runs folder can travel together.
        cd_raw = data.get("calc_dir")
        if isinstance(cd_raw, str) and cd_raw:
            cd_path = Path(cd_raw)
            if not cd_path.is_absolute() and project_path:
                cd_path = Path(project_path).parent / cd_path
            _settings().setValue(_SETTINGS_CALC_DIR, str(cd_path))
            self.state.status_message.emit(
                f"Calculation directory: {cd_path}"
            )

        # Orbit-correction settings (best-effort; old projects lack them).
        corr = data.get("correction")
        if isinstance(corr, dict):
            self.state.correction_settings.update(
                {k: v for k, v in corr.items()
                 if k in self.state.correction_settings}
            )
        mode = data.get("auto_correction_mode")
        if isinstance(mode, str) and mode in (
                "never", "on_errors_only", "always"):
            self.state.auto_correction_mode = mode

        # Auto-correction-on-load: if the user opted into "always" and
        # the loaded lattice has any ADJUST_STEERER cards, fire the
        # standalone Lattice-tab correction handler now.
        try:
            from linac_gen.elements.lattice_commands import (
                AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
            )
            lat = self.state.lattice
            if (self.state.auto_correction_mode == "always"
                    and lat is not None
                    and any(isinstance(e, (AdjustSteerer,
                                           AdjustSteererBx,
                                           AdjustSteererBy))
                            for e in lat.elements)):
                self.lattice_tab._on_correct_orbit()
        except Exception:
            pass

        # Remember this project so the next launch can restore it.
        if project_path:
            _settings().setValue(_SETTINGS_LAST_PROJECT,
                                 os.path.abspath(project_path))

        # Loading a project means the in-memory state now matches the
        # on-disk .lgproj -- clear the dirty flag.  Note: applying the
        # values above triggered every widget's change signal, which
        # the Convergence/Beam tabs wired to mark_project_dirty(); so
        # we MUST clear here, after all those handlers have run, or
        # the user sees "● unsaved" the instant they open a project.
        self.state.mark_project_clean()

        # One consolidated dialog instead of a popup per failed section.
        if warnings and not silent:
            QMessageBox.warning(
                self, "Project loaded with warnings",
                "Some sections could not be restored:\n\n"
                + "\n".join(f"• {w}" for w in warnings))
        return warnings

    def _save_project(self) -> None:
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Project",
            str(Path(_project_start_dir()) / "project.lgproj"),
            "HELIX Project (*.lgproj *.lgproj.json *.json);;All Files (*)"
        )
        if not fp:
            return
        try:
            import json
            warnings: list[str] = []
            data = self._collect_project_dict(warnings, project_path=fp)
            if warnings:
                # Never silently write a project with missing sections —
                # let the user fix the form or knowingly save without it.
                listing = "\n".join(f"• {w}" for w in warnings)
                choice = QMessageBox.warning(
                    self, "Incomplete project",
                    f"Some settings could not be captured:\n\n{listing}\n\n"
                    "Save the project without them?",
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if choice != QMessageBox.StandardButton.Save:
                    self.state.status_message.emit("Project save cancelled.")
                    return
            with open(fp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            self._toolbar.set_recent_projects(_recent_projects_add(fp))
            _settings().setValue(_SETTINGS_LAST_PROJECT, os.path.abspath(fp))
            # On-disk file now matches in-memory state -- clear dirty
            # so the close prompt doesn't fire spuriously after Save.
            self.state.mark_project_clean()
            self.state.status_message.emit(f"Project saved → {os.path.basename(fp)}")
        except Exception as exc:
            QMessageBox.critical(self, "Save project failed", str(exc))

    # ------------------------------------------------------------------
    # TraceWin output export — emits partran1.out + envelope .txt for the
    # most recent envelope/multiparticle run.  Both formats follow the
    # documented schema (PDF p. 43-44 for partran1.out, the 26-column
    # save-data tab format for the .txt).
    def _export_tracewin(self) -> None:
        if self.state.results is None:
            QMessageBox.warning(self, "No results",
                                "Run a simulation before exporting.")
            return
        if self.state.lattice is None or self.state.beam_config is None:
            QMessageBox.warning(self, "No lattice / beam",
                                "Load a lattice and configure the beam first.")
            return
        s = _settings()
        start_dir = str(s.value(_SETTINGS_LAST_DIR, str(Path.cwd())))
        dirpath = QFileDialog.getExistingDirectory(
            self, "Export TraceWin output to directory…", start_dir)
        if not dirpath:
            return
        base = Path(dirpath)
        # Fixed output names — never overwrite silently.
        existing = [p.name for p in (base / "partran1.out",
                                     base / "tracewin1.txt") if p.exists()]
        if existing:
            choice = QMessageBox.question(
                self, "Overwrite export files?",
                f"{' and '.join(existing)} already exist in\n{dirpath}\n\n"
                "Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
        try:
            from linac_gen.io.tracewin_outputs import (
                write_envelope_txt, write_partran_out,
            )
            par_path = write_partran_out(
                self.state.results, self.state.lattice,
                self.state.beam_config, base / "partran1.out",
            )
            env_path = write_envelope_txt(
                self.state.results, self.state.beam_config,
                base / "tracewin1.txt",
            )
            self.state.status_message.emit(
                f"Exported → {par_path.name}, {env_path.name}"
            )
            QMessageBox.information(
                self, "Export complete",
                f"Wrote:\n  {par_path}\n  {env_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _export_openpmd(self) -> None:
        """Write the most recent run's results to an openPMD-1.1 HDF5 file."""
        if self.state.results is None:
            QMessageBox.warning(self, "No results",
                                "Run a simulation before exporting.")
            return
        s = _settings()
        start_dir = str(s.value(_SETTINGS_LAST_DIR, str(Path.cwd())))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results as openPMD…",
            str(Path(start_dir) / "results.opmd.h5"),
            "openPMD HDF5 (*.opmd.h5);;All files (*)",
        )
        if not path:
            return
        # A bare name typed into the dialog gets the canonical extension
        # (same guard the Results-tab exporters use).
        if not path.lower().endswith((".h5", ".hdf5")):
            path += ".h5" if path.lower().endswith(".opmd") else ".opmd.h5"
            # The dialog's own overwrite prompt covered only the name as
            # typed — appending an extension can now target a DIFFERENT
            # existing file, so re-check.
            if os.path.exists(path):
                choice = QMessageBox.question(
                    self, "Overwrite?",
                    f"{os.path.basename(path)} already exists.\n\nOverwrite?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if choice != QMessageBox.StandardButton.Yes:
                    return
        try:
            from linac_gen.io.openpmd_output import save_results_openpmd
            save_results_openpmd(
                self.state.results, path,
                beam_config=self.state.beam_config,
                lattice=self.state.lattice,
            )
            s.setValue(_SETTINGS_LAST_DIR, str(Path(path).parent))
            self.state.status_message.emit(
                f"Exported openPMD → {Path(path).name}"
            )
            QMessageBox.information(
                self, "Export complete",
                f"Wrote:\n  {path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _open_project(self) -> None:
        if not self._confirm_discard("Open Project"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Open Project", _project_start_dir(),
            "HELIX Project (*.lgproj *.lgproj.json *.json);;All Files (*)"
        )
        if not fp:
            return
        try:
            import json
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or data.get("__kind__") != "linac_gen_project":
                # Still try to load — just warn about the missing tag.
                self.state.status_message.emit(
                    "File is not tagged as a HELIX project — loading anyway.")
            warns = self._apply_project_dict(data, project_path=fp)
            self._toolbar.set_recent_projects(_recent_projects_add(fp))
            suffix = (f" — {len(warns)} section(s) skipped" if warns else "")
            self.state.status_message.emit(
                f"Project loaded{suffix} ← {os.path.basename(fp)}")
        except Exception as exc:
            QMessageBox.critical(self, "Open project failed", str(exc))

    # ------------------------------------------------------------------
    # Run actions
    def _ensure_beam_config(self) -> bool:
        if self.state.beam_config is None:
            # Pull from the BeamTab
            try:
                self.state.set_beam_config(self.beam_tab.get_beam_config())
            except Exception as exc:
                QMessageBox.critical(self, "Beam config", str(exc)); return False
        return True

    def _run_envelope(self) -> None:
        if self.state.running:
            # Ctrl+R / menu paths bypass the disabled toolbar buttons; a
            # second start would spawn a parallel worker racing the first.
            self.state.status_message.emit(
                "A run is already in progress — Stop it first.")
            return
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first."); return
        if not self._ensure_beam_config(): return
        import math
        from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
        from linac_gen.core.reference import ReferenceParticle
        cfg = self.state.beam_config
        species_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
        sp = species_map.get(cfg.species, PROTON)
        bg = math.sqrt(max((1 + cfg.energy / sp.mass) ** 2 - 1, 0.0))
        ref = ReferenceParticle(species=sp, w_kin=cfg.energy, frequency=cfg.frequency)
        # Mismatch-scaled geometric emittances — the shared helper keeps
        # the envelope run modelling the same beam MP generates.
        from linac_gen.distributions.factory import geometric_emittances
        emit_x, emit_y, emit_z = geometric_emittances(cfg, bg)
        initial = dict(
            alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=emit_x,
            alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=emit_y,
            alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=emit_z,
            # DC / continuous-beam metadata — EnvelopeSolver checks these
            # to skip the longitudinal σ block and switch to 2-D SC.
            continuous=bool(getattr(cfg, "continuous", False)),
            dc_energy_spread_keV=float(getattr(cfg, "dc_energy_spread_keV", 0.0)),
            # First-moment seed — same offsets create_beam applies to MP
            # particles, so envelope and MP launch the same orbit.
            centroid=[float(getattr(cfg, "centroid_x", 0.0) or 0.0),
                      float(getattr(cfg, "centroid_xp", 0.0) or 0.0),
                      float(getattr(cfg, "centroid_y", 0.0) or 0.0),
                      float(getattr(cfg, "centroid_yp", 0.0) or 0.0),
                      float(getattr(cfg, "centroid_dphi", 0.0) or 0.0),
                      float(getattr(cfg, "centroid_dw", 0.0) or 0.0)],
        )
        # Apply FieldMap3D class-level toggles so envelope's matrix-fit
        # (which calls track_rk4 internally) uses the user-selected
        # integrator / interp kind too.
        from linac_gen.elements.field_map_3d import FieldMap3D
        from linac_gen.elements import field_map_3d as _fm3
        _fm3.set_fieldmap_numerics(
            integrator=self.convergence_tab._fixed_integ.currentText(),
            interp=self.convergence_tab._fixed_interp.currentText())
        self.state.set_running(True)
        self._toolbar.set_progress(0)
        env_solver = self.convergence_tab._fixed_env_solver.currentText()
        record_substeps = self.convergence_tab._record_substeps.isChecked()
        # Detach any prior envelope worker (e.g. one we just told to stop)
        # before replacing the reference, so its trailing signals don't
        # pollute the new run's progress / status bar.  Use a SHORT wait — the
        # worker is parked (not destroyed) if it outruns the deadline — so
        # clicking Run right after Stop can't freeze the GUI thread on
        # worker.wait() for up to 8 s while the old solver winds down.
        self._retire_worker(self._envelope_worker, timeout_ms=300)
        self._envelope_worker = EnvelopeWorker(
            self.state.lattice, ref, initial, cfg.current,
            solver_kind=env_solver,
            record_substeps=record_substeps,
        )
        self._envelope_worker.finished_ok.connect(self._env_done)
        self._envelope_worker.failed.connect(self._env_fail)
        self._envelope_worker.progress.connect(self._toolbar.set_progress)
        self._envelope_worker.progress_s.connect(self._toolbar.set_live_s)
        self._envelope_worker.aborted.connect(self._env_aborted)
        self._envelope_worker.start()

    def _env_done(self, results) -> None:
        # A queued emission from a retired worker can still land here
        # after disconnect() (the event was posted before the disconnect)
        # — never let a stale run overwrite the current lattice's results.
        if self.sender() is not self._envelope_worker:
            return
        self.state.set_running(False)
        self._toolbar.set_progress(100)
        self.state.set_results(results)
        saved_path = self._auto_dump_results(results, "env")
        msg = (
            f"Envelope done · σ_x={results.sigma_x[-1]:.3f} mm · "
            f"σ_y={results.sigma_y[-1]:.3f} mm"
        )
        if saved_path is not None:
            msg += f"  ·  saved {saved_path.name}"
        self.state.status_message.emit(msg)

    def _env_fail(self, msg: str) -> None:
        # Shared by the envelope AND the MP worker (see _run_mp wiring).
        if self.sender() not in (self._envelope_worker, self._mp_worker,
                                 self._backtrack_worker):
            return
        self.state.set_running(False)
        self._toolbar.set_progress(0)
        QMessageBox.critical(self, "Envelope failed", msg)

    def _env_aborted(self) -> None:
        if self.sender() is not self._envelope_worker:
            return
        self.state.set_running(False)
        self._toolbar.set_progress(0)
        self.state.status_message.emit("Envelope run stopped by user.")

    def _mp_aborted(self) -> None:
        if self.sender() is not self._mp_worker:
            return
        self.state.set_running(False)
        self._toolbar.set_progress(0)
        self.state.status_message.emit("Multi-particle run stopped by user.")

    def _stop_active_worker(self) -> None:
        """Route the toolbar Stop button to whichever worker is running.

        The forward-simulation workers (envelope, MP) carry a
        ``request_stop`` method that sets a thread-safe flag; their
        solver notices it at the next element boundary and returns
        partial results.  The matcher's ``_MatchWorker`` is a plain
        ``QThread`` -- we use Qt's native ``requestInterruption`` /
        ``isInterruptionRequested`` pair (see ``_MatchWorker.run``'s
        progress callback for the consumer side).  Latency is one
        cost-function evaluation (~28 s with SC on PIP-II HWR).  If
        all workers are idle this is a no-op.
        """
        for w in (self._envelope_worker, self._mp_worker,
                  self._backtrack_worker):
            if w is not None and w.isRunning():
                w.request_stop()
        mw = getattr(self.matching_tab, "_aa_worker", None)
        if mw is not None and mw.isRunning():
            mw.requestInterruption()

    def _retire_worker(self, worker, *, timeout_ms: int = 8000) -> None:
        """Detach a worker before its reference is replaced.

        Without this, clicking Run after Stop while the previous worker is
        still inside its ``run()`` (between ``request_stop`` setting the
        flag and the tracker hitting the next element boundary) causes
        signal-routing pollution: the OLD worker emits its final
        ``progress``/``aborted`` events to the same toolbar slots the NEW
        worker just connected to, so the progress bar bounces between 0
        (from the old abort handler) and the new run's rising values.

        Steps, in order:
        1. Tell the worker to stop (idempotent — safe if already stopping).
        2. Disconnect every signal that fans into the GUI so any residual
           emissions from the old thread are silently dropped.
        3. Wait up to ``timeout_ms`` for the QThread to actually exit, so
           the new worker isn't competing with the old one for solver
           resources (and so we don't trigger
           "QThread: Destroyed while running" on garbage collection).
        """
        if worker is None:
            return
        try:
            if worker.isRunning():
                worker.request_stop()
        except Exception:
            pass
        for signal_name in (
            "progress", "progress_s", "finished_ok", "failed", "aborted",
        ):
            sig = getattr(worker, signal_name, None)
            if sig is None:
                continue
            try:
                sig.disconnect()
            except (TypeError, RuntimeError):
                pass
        try:
            if worker.isRunning():
                if not worker.wait(timeout_ms):
                    # Still running past the deadline and the caller is
                    # about to rebind its reference — park it so the
                    # last reference to a live QThread is never dropped
                    # (that destroys the thread and qFatal-aborts).
                    _park_worker(worker)
        except Exception:
            pass

    def _detach_workers_for_new_lattice(self) -> None:
        """A new lattice invalidates any in-flight envelope/MP run.

        Retire both workers (the disconnect matters even for a worker
        that already FINISHED: its queued finished_ok may still be
        undelivered and would otherwise pass the sender-identity guard)
        and null the attributes so any late delivery is rejected.
        A still-running thread is parked by _retire_worker's timeout
        path, so nulling the reference here is safe.
        """
        for attr in ("_envelope_worker", "_mp_worker",
                     "_backtrack_worker"):
            w = getattr(self, attr)
            if w is not None:
                self._retire_worker(w, timeout_ms=250)
                setattr(self, attr, None)
        if self.state.running:
            self.state.set_running(False)
            self._toolbar.set_progress(0)

    def _run_mp(self) -> None:
        if self.state.running:
            self.state.status_message.emit(
                "A run is already in progress — Stop it first.")
            return
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first."); return
        if not self._ensure_beam_config(): return
        try:
            from linac_gen.distributions.factory import create_beam
            from linac_gen.core.step_config import StepConfig
            cfg = self.state.beam_config
            beam = create_beam(cfg, seed=42)
            step1 = self.convergence_tab._fixed_step1.value()
            step2 = self.convergence_tab._fixed_step2.value()
            # Push step1/step2 into the live lattice so the tracker picks them up
            self.state.lattice.step_config = StepConfig(
                integration_steps_per_metre=float(step1),
                sc_steps_per_metre=float(step2),
            )
            continuous = bool(getattr(beam, "continuous", False))
            # The torch PIC backend is a bunched-beam 3-D solver with no
            # continuous/DC kernels; the builder falls back to numpy for DC
            # beams — tell the user so the switch is never silent.
            if (continuous and cfg.current > 0 and
                    self.convergence_tab._fixed_sc_backend.currentText()
                    == "torch"):
                QMessageBox.information(
                    self, "Space-charge backend",
                    "The torch SC backend supports bunched beams only.\n"
                    "This is a continuous (DC) beam — using the numpy "
                    "backend for this run.")
            # Canonical SC config — shared with the Error Study tab studies.
            sc = self.convergence_tab.current_sc_config(
                cfg.current, continuous=continuous)
            # FieldMap3D integrator / interp are class-level toggles that
            # apply to the whole next run.  Cubic interp builds its
            # coefficient table at element __init__, so a change here only
            # takes effect for lattices loaded *after* the toggle flips.
            from linac_gen.elements import field_map_3d
            # helper mirrors into env vars -> spawned error-study workers
            # inherit; bare class-attr sets do not cross the spawn boundary
            field_map_3d.set_fieldmap_numerics(
                integrator=self.convergence_tab._fixed_integ.currentText(),
                interp=self.convergence_tab._fixed_interp.currentText())
            # Sampling implementation (fused C++ kernel vs legacy scipy) —
            # bitwise-identical results either way; runtime-switchable.
            _fused = (self.convergence_tab._fieldmap_sampling.currentText()
                      == "kernel")
            field_map_3d.use_fused_kernel(_fused)
            # Mirror into the env var so SPAWNED workers (error studies /
            # scans re-import in fresh processes) inherit the same choice.
            os.environ["LINAC_GEN_FIELDMAP_KERNEL"] = "1" if _fused else "0"
        except Exception as exc:
            QMessageBox.critical(self, "Setup error", str(exc)); return
        self.state.set_running(True)
        self._toolbar.set_progress(0)
        record_substeps = self.convergence_tab._record_substeps.isChecked()
        # Record every coord the popup's axis selector offers, so switching
        # axis in the GUI doesn't show "no data" for everything except x / y.
        density_axes = (("x", "xp", "y", "yp", "phi", "w")
                        if self.convergence_tab._record_density.isChecked()
                        else ())
        snapshot_every_n_val = self.convergence_tab._snapshot_every_n.value()
        snapshot_every_n = snapshot_every_n_val if snapshot_every_n_val > 0 else None
        density_bins = self.convergence_tab._density_bins.value()
        ext_half = self.convergence_tab._density_extent.value()
        density_extent: dict = {}
        if ext_half > 0.0:
            for axis in ("x", "xp", "y", "yp", "phi", "w"):
                density_extent[axis] = (-ext_half, ext_half)
        # Detach any prior MP worker (e.g. one we just told to stop) before
        # replacing the reference, so its trailing ``progress`` and
        # ``aborted`` emissions don't fight with the new run's signals.  Use a
        # SHORT wait — the worker is parked (not destroyed) if it outruns the
        # deadline — so clicking Run right after Stop can't freeze the GUI
        # thread on worker.wait() for up to 8 s.
        self._retire_worker(self._mp_worker, timeout_ms=300)
        self._mp_worker = MultiparticleWorker(
            self.state.lattice, beam, sc,
            record_substeps=record_substeps,
            density_axes=density_axes,
            snapshot_every_n=snapshot_every_n,
            snapshot_elements=self.convergence_tab.snapshot_element_names(),
            density_n_bins=density_bins,
            density_extent=density_extent,
        )
        self._mp_worker.progress.connect(self._toolbar.set_progress)
        self._mp_worker.progress_s.connect(self._toolbar.set_live_s)
        self._mp_worker.finished_ok.connect(self._mp_done)
        self._mp_worker.failed.connect(self._env_fail)
        self._mp_worker.aborted.connect(self._mp_aborted)
        self._mp_worker.start()

    def _mp_done(self, results) -> None:
        if self.sender() is not self._mp_worker:
            return
        self.state.set_running(False)
        self._toolbar.set_progress(100)
        # Attach the beam so the Phase Space popup can fetch alive particles
        try:
            if self._mp_worker is not None:
                results.beam = self._mp_worker.beam
        except Exception:
            pass
        self.state.set_results(results)
        saved_path = self._auto_dump_results(results, "mp")
        if saved_path is not None:
            self.state.status_message.emit(
                f"Multi-particle run complete  ·  saved {saved_path.name}"
            )
        else:
            self.state.status_message.emit("Multi-particle run complete")

    # ------------------------------------------------------------------
    # Backtracking (Simulate → Backtrack Distribution…)
    def _run_backtrack(self) -> None:
        if self.state.running:
            self.state.status_message.emit(
                "A run is already in progress — Stop it first.")
            return
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first.")
            return
        if not self._ensure_beam_config():
            return

        from linac_gen_gui.interphase.dialogs.backtrack_dialog import (
            BacktrackDialog,
        )
        results_beam = getattr(self.state.results, "beam", None)
        dlg = BacktrackDialog(
            n_elements=len(self.state.lattice.elements),
            has_results_beam=results_beam is not None,
            parent=self,
        )
        if not dlg.exec():
            return
        cfg_ui = dlg.get_settings()

        try:
            import copy

            from linac_gen.cli.common import build_ref
            from linac_gen.core.step_config import StepConfig
            from linac_gen.distributions.factory import create_beam

            beam_cfg = self.state.beam_config
            entrance_ref = build_ref(beam_cfg)

            if cfg_ui["source"] == "results":
                # Deep-copy: the backward walk mutates the particle
                # array, and results.beam still feeds the phase-space
                # popup of the forward run.
                beam = copy.deepcopy(results_beam)
            else:
                file_cfg = copy.copy(beam_cfg)
                file_cfg.source = "file"
                file_cfg.distribution_file = cfg_ui["dst_path"]
                beam = create_beam(file_cfg, seed=42)

            step1 = self.convergence_tab._fixed_step1.value()
            step2 = self.convergence_tab._fixed_step2.value()
            self.state.lattice.step_config = StepConfig(
                integration_steps_per_metre=float(step1),
                sc_steps_per_metre=float(step2),
            )
            sc = None
            if cfg_ui["space_charge"]:
                sc = self.convergence_tab.current_sc_config(
                    beam_cfg.current,
                    continuous=bool(getattr(beam, "continuous", False)))
        except Exception as exc:                                # noqa: BLE001
            QMessageBox.critical(self, "Setup error", str(exc))
            return

        self.state.set_running(True)
        self._toolbar.set_progress(0)
        # Immediate feedback — the first progress tick can take a moment
        # (replay-table build + SET_SYNC_PHASE calibration come first).
        self.state.status_message.emit(
            f"Backtracking: exit of element {cfg_ui['end']} → entrance of "
            f"element {cfg_ui['start']} "
            f"({cfg_ui['field_map_mode']} inverse"
            + (", SC on" if cfg_ui["space_charge"] else "")
            + ") — watch the s-cursor walk right-to-left…")
        self._retire_worker(self._backtrack_worker, timeout_ms=300)
        self._backtrack_worker = BacktrackWorker(
            self.state.lattice, beam, sc, entrance_ref,
            start=cfg_ui["start"], end=cfg_ui["end"],
            field_map_mode=cfg_ui["field_map_mode"],
        )
        self._backtrack_write_dst = cfg_ui["write_dst"]
        self._backtrack_worker.progress.connect(self._toolbar.set_progress)
        self._backtrack_worker.progress_s.connect(self._toolbar.set_live_s)
        self._backtrack_worker.finished_ok.connect(self._backtrack_done)
        self._backtrack_worker.failed.connect(self._env_fail)
        self._backtrack_worker.aborted.connect(self._backtrack_aborted)
        self._backtrack_worker.start()

    def _backtrack_done(self, results) -> None:
        if self.sender() is not self._backtrack_worker:
            return
        self.state.set_running(False)
        self._toolbar.set_progress(100)
        worker = self._backtrack_worker
        try:
            if worker is not None:
                results.beam = worker.beam    # reconstructed entrance dist
        except Exception:                                       # noqa: BLE001
            pass
        # Reversed-to-increasing-s recorder → Results tab needs no changes.
        self.state.set_results(results)
        msg = "Backtrack complete — index 0 is the reconstructed entrance"
        # Physics caveats collected by the worker (surrogate fallback,
        # survivors-only, SC grid fallback, …) — a GUI user would never
        # see the stderr warnings, so show them once, post-run.
        caveats = list(getattr(results, "backtrack_warnings", ()) or ())
        if caveats:
            msg += f"  ·  {len(caveats)} physics caveat(s)"
            QMessageBox.warning(
                self, "Backtrack caveats",
                "The reconstruction completed with caveats:\n\n"
                + "\n\n".join(f"• {c}" for c in caveats[:6])
                + ("\n\n(+ more — see terminal log)"
                   if len(caveats) > 6 else ""))
        out = getattr(self, "_backtrack_write_dst", None)
        if out and worker is not None:
            try:
                from linac_gen.io.tracewin_dst import write_dst
                b = worker.beam
                write_dst(out, b.particles[b.alive_mask],
                          current_mA=b.current,
                          frequency_MHz=b.ref.frequency,
                          mass_MeV=b.ref.species.mass,
                          w_kin_ref=b.ref.w_kin)
                msg += f"  ·  wrote {os.path.basename(out)}"
            except Exception as exc:                            # noqa: BLE001
                QMessageBox.warning(self, "Write failed",
                                    f"Could not write {out}:\n{exc}")
        self.state.status_message.emit(msg)

    def _backtrack_aborted(self) -> None:
        if self.sender() is not self._backtrack_worker:
            return
        self.state.set_running(False)
        self._toolbar.set_progress(0)
        self.state.status_message.emit("Backtrack stopped by user.")

    # ------------------------------------------------------------------
    # Tools (popups)
    # -- assistant navigation (called on the GUI thread only) -----------
    def _assistant_pages(self):
        # page widgets in TABS / tab-index order (the add loop zips these)
        return [self.beam_tab, self.lattice_tab, self.matching_tab,
                self.convergence_tab, self.surrogates_tab, self.errors_tab,
                self.failures_tab, self.results_tab]

    def _page_subtab_widget(self, page):
        from PyQt6.QtWidgets import QTabWidget
        kids = page.findChildren(QTabWidget) if page is not None else []
        return kids[0] if kids else None

    def assistant_tab_labels(self) -> list:
        """Tab titles the assistant may switch to (left-to-right order)."""
        return [self._tabs.tabText(i) for i in range(self._tabs.count())]

    def assistant_subtabs(self) -> dict:
        """{tab_title: [subtab_titles]} for tabs holding a nested QTabWidget
        (Lattice, Numerics, Error Study)."""
        out, pages = {}, self._assistant_pages()
        for i in range(self._tabs.count()):
            sub = self._page_subtab_widget(pages[i]) if i < len(pages) else None
            if sub is not None and sub.count():
                out[self._tabs.tabText(i)] = [
                    sub.tabText(j) for j in range(sub.count())]
        return out

    def show_tab(self, tab, subtab=None):
        """Switch to a top tab (integer index, stable id, or title —
        case-insensitive, exact then substring) and, optionally, a nested
        subtab by title.  Returns the resolved label ('Tab' or
        'Tab › Subtab'), or None if the tab didn't match.  GUI thread
        only — the assistant reaches this through a queued signal.
        """
        n = self._tabs.count()
        idx = None
        try:
            k = int(tab)
            if 0 <= k < n:
                idx = k
        except (TypeError, ValueError):
            pass
        if idx is None:
            want = str(tab).strip().casefold()
            try:
                from linac_gen_gui.interphase.state import TABS
                id2 = {tid.casefold(): lab for tid, lab in TABS}
                if want in id2:
                    want = id2[want].casefold()
            except Exception:                    # noqa: BLE001
                pass
            titles = [self._tabs.tabText(i) for i in range(n)]
            idx = next((i for i, t in enumerate(titles)
                        if t.casefold() == want), None)
            if idx is None:
                idx = next((i for i, t in enumerate(titles)
                            if want and want in t.casefold()), None)
        if idx is None:
            return None
        self._tabs.setCurrentIndex(idx)
        label = self._tabs.tabText(idx)
        if subtab:
            pages = self._assistant_pages()
            sub = self._page_subtab_widget(pages[idx]) if idx < len(pages) \
                else None
            if sub is not None:
                sw = str(subtab).strip().casefold()
                for j in range(sub.count()):
                    tj = sub.tabText(j)
                    if tj.casefold() == sw or sw in tj.casefold():
                        sub.setCurrentIndex(j)
                        return f"{label} › {tj}"
        return label

    def result_plot_catalog(self):
        """[(key, label)] of every Results-tab plot."""
        return self.results_tab.plot_catalog()

    def assistant_run_simulation(self, kind: str) -> str:
        """Press the real Run button for the assistant.  GUI thread only
        (reached via the panel's run_on_gui round-trip).  Returns
        'started' | 'busy' | an error string — never raises, never pops a
        modal (guards are checked here so the assistant gets a value)."""
        if self.state.running:
            return "busy"
        if self.state.lattice is None:
            return "no lattice is loaded in the GUI"
        if self.state.beam_config is None:
            return "no beam configuration is set in the GUI"
        try:
            if kind == "mp":
                self._run_mp()          # identical to clicking Run (MP)
            else:
                self._run_envelope()    # identical to clicking Run (envelope)
        except Exception as exc:        # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        # the slot may early-return without starting (its own guards); the
        # running flag is the truth
        return "started" if self.state.running else \
            "the GUI declined to start the run (check its status bar)"

    def assistant_open_plots(self) -> list:
        """Labels of result plot windows currently open/visible.  GUI
        thread only (reached via the panel's run_on_gui round-trip)."""
        labels = dict(self.results_tab.plot_catalog())
        out = []
        for key, dlg in getattr(self.results_tab, "_popups", {}).items():
            try:
                if dlg.isVisible():
                    out.append(labels.get(key, key))
            except RuntimeError:                 # deleted Qt object
                continue
        return out

    def show_result_plot(self, key) -> "str | None":
        """Switch to Results and open the plot window for `key`.  Returns the
        plot's label, or None if the key is unknown.  GUI thread only."""
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i).casefold() == "results":
                self._tabs.setCurrentIndex(i)
                break
        if self.results_tab.open_plot(key):
            for k, lab in self.results_tab.plot_catalog():
                if k == key:
                    return lab
            return key
        return None

    def _open_assistant(self) -> None:
        # OPTIONAL feature — lazy-imported here so nothing about the
        # assistant loads at GUI startup; non-modal (show(), not exec())
        # so the main window stays live while a run is in flight.
        try:
            from linac_gen_gui.interphase.dialogs.assistant_panel import (
                AssistantPanel)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.information(
                self, "Assistant",
                f"The optional AI assistant could not load:\n{exc}")
            return
        existing = getattr(self, "_assistant_panel", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        self._assistant_panel = AssistantPanel(self, self.state)
        self._assistant_panel.show()
        self._assistant_panel.raise_()

    def _open_console(self) -> None:
        try:
            from PyQt6.QtGui import QTextCursor
            from PyQt6.QtWidgets import QLineEdit, QPlainTextEdit, QDialog, QVBoxLayout, QHBoxLayout, QLabel
            import io, sys as _sys, traceback
            dlg = QDialog(self); dlg.setWindowTitle("Python Console"); dlg.resize(800, 520)
            v = QVBoxLayout(dlg); v.setContentsMargins(10, 10, 10, 10); v.setSpacing(6)
            out = QPlainTextEdit(readOnly=True)
            out.setStyleSheet(
                f"background:{theme.BG_INSET}; color:{theme.TEXT_1};"
                f"font-family:{theme.FONT_MONO}; font-size:11px;"
                f"border:1px solid {theme.BORDER_0}; border-radius:3px;"
            )
            banner = ("Interphase Python console · 'lg' = linac_gen, 'state' = AppState.\n"
                      "Access: lattice, beam_config, results.\n")
            out.setPlainText(banner)
            v.addWidget(out, stretch=1)
            row = QHBoxLayout()
            prompt = QLabel(">>>"); prompt.setStyleSheet(f"color:{theme.ACCENT}; font-family:{theme.FONT_MONO};")
            row.addWidget(prompt)
            edit = QLineEdit()
            edit.setStyleSheet(
                f"background:{theme.BG_INSET}; color:{theme.TEXT_0};"
                f"font-family:{theme.FONT_MONO}; font-size:11px;"
                f"border:1px solid {theme.BORDER_1}; border-radius:3px; padding:5px 8px;"
            )
            row.addWidget(edit, stretch=1)
            v.addLayout(row)

            import linac_gen as lg
            globs = {"lg": lg, "state": self.state}

            def run():
                code = edit.text().strip()
                if not code: return
                edit.clear()
                globs["lattice"]     = self.state.lattice
                globs["beam_config"] = self.state.beam_config
                globs["results"]     = self.state.results
                buf_o, buf_e = io.StringIO(), io.StringIO()
                old_o, old_e = _sys.stdout, _sys.stderr
                _sys.stdout, _sys.stderr = buf_o, buf_e
                out.moveCursor(QTextCursor.MoveOperation.End)
                out.insertPlainText(f"\n>>> {code}\n")
                try:
                    try:
                        val = eval(code, globs)
                        if val is not None:
                            out.insertPlainText(repr(val) + "\n")
                    except SyntaxError:
                        exec(code, globs)
                except Exception:
                    out.insertPlainText(traceback.format_exc())
                finally:
                    _sys.stdout, _sys.stderr = old_o, old_e
                    o = buf_o.getvalue(); e = buf_e.getvalue()
                    if o: out.insertPlainText(o)
                    if e: out.insertPlainText(e)

            edit.returnPressed.connect(run)
            dlg.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Console", str(exc))

    def _open_transfer_matrix(self) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first."); return
        if not self._ensure_beam_config(): return
        try:
            from linac_gen_gui.dialogs.transfer_matrix_dialog import TransferMatrixDialog
            TransferMatrixDialog(self, self.state.lattice, self.state.beam_config).exec()
        except Exception as exc:
            QMessageBox.critical(self, "Transfer matrix", str(exc))

    def _open_sigma_matrix(self) -> None:
        if self.state.results is None:
            QMessageBox.warning(self, "No results",
                                "Run a simulation first (Envelope or MP)."); return
        try:
            from linac_gen_gui.dialogs.sigma_matrix_dialog import SigmaMatrixDialog
            SigmaMatrixDialog(self, self.state.results,
                              beam_config=self.state.beam_config).exec()
        except Exception as exc:
            QMessageBox.critical(self, "Sigma matrix", str(exc))

    def _open_parameter_scan(self) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "Parameter Scan",
                                "Load a lattice first."); return
        # Keep a single instance so repeated menu clicks don't spawn
        # multiple windows.  Non-modal so the main GUI stays usable.
        dlg = getattr(self, "_param_scan_dlg", None)
        if dlg is not None and not dlg.isVisible():
            w = getattr(dlg, "_worker", None)
            if w is None or not hasattr(w, "isRunning") or not w.isRunning():
                # Release the closed instance — it keeps a lattice_changed
                # connection alive and would silently rebuild its combos on
                # every lattice edit for the rest of the session.  Proactively
                # disconnect before deleteLater so the slot never fires on
                # the half-dead dialog (the guard in _on_lattice_changed is
                # the safety net; this avoids relying on it).
                try:
                    self.state.lattice_changed.disconnect(
                        dlg._on_lattice_changed)
                except (TypeError, RuntimeError):
                    pass
                dlg.deleteLater()
                dlg = None
            # else: a scan is still winding down inside the closed dialog —
            # reuse it rather than orphaning a live QThread.
        if dlg is None:
            from linac_gen_gui.interphase.dialogs import ParameterScanDialog
            dlg = ParameterScanDialog(self, self.state)
            self._param_scan_dlg = dlg
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def _open_about(self) -> None:
        from PyQt6.QtGui import QPixmap
        from linac_gen_gui.interphase.splash import LOGO_PATH, _last_modified_date
        box = QMessageBox(self)
        box.setWindowTitle("About HELIX")
        if LOGO_PATH.is_file():
            box.setIconPixmap(QPixmap(str(LOGO_PATH)).scaled(
                96, 96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        box.setText(
            "HELIX 1.0\n"
            "Hybrid Envelope-multiparticle LInac eXplorer\n\n"
            "Beam dynamics studio for linear accelerators.\n"
            "Tabs: Beam · Lattice · Matching · Numerics · Results\n\n"
            "Engines: matrix · envelope · multiparticle\n"
            "Space charge: analytical · PIC fixed · PIC adaptive\n"
            "Backends: CPU (C++/OpenMP) · CUDA (cupy) · MPS (torch, Apple Silicon)\n"
            "I/O: TraceWin .dat / .dst / partran1.out · HDF5\n\n"
            f"Developer  ·  Abhishek Pathak\n"
            f"Last modified  ·  {_last_modified_date()}"
        )
        box.exec()

    # ------------------------------------------------------------------
    # Recent projects (File → Open Recent submenu)
    def _open_recent_project(self, path: str) -> None:
        """Re-enter the existing open-project flow for a path picked from the
        Recent submenu.  Reuses ``_apply_project_dict`` so behaviour matches
        File → Open Project exactly (lattice resolution, calc_dir restore,
        beam-config push, etc.)."""
        if not self._confirm_discard("Open Recent Project"):
            return
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "Recent project missing",
                f"The file no longer exists:\n{path}\n\n"
                "Removing it from the recent list."
            )
            # Drop the stale entry and refresh the submenu.
            remaining = [p for p in _recent_projects_load() if p != path]
            _recent_projects_save(remaining)
            self._toolbar.set_recent_projects(remaining)
            return
        try:
            import json
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            warns = self._apply_project_dict(data, project_path=path)
            self._toolbar.set_recent_projects(_recent_projects_add(path))
            suffix = (f" — {len(warns)} section(s) skipped" if warns else "")
            self.state.status_message.emit(
                f"Project loaded{suffix} ← {os.path.basename(path)}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Open recent failed", str(exc))

    def _clear_recent_projects(self) -> None:
        _recent_projects_save([])
        self._toolbar.set_recent_projects([])
        self.state.status_message.emit("Cleared recent projects list.")

    # ------------------------------------------------------------------
    # Calculation directory + auto-save
    def _set_calc_dir(self) -> None:
        """Prompt for the directory where auto-saved HDF5 dumps land."""
        current = _resolve_calc_dir()
        chosen = QFileDialog.getExistingDirectory(
            self, "Select calculation directory…", str(current)
        )
        if not chosen:
            return
        _settings().setValue(_SETTINGS_CALC_DIR, chosen)
        self.state.status_message.emit(f"Calculation directory: {chosen}")

    def _auto_dump_results(self, results, run_type: str):
        """Dump ``results`` to two parallel files under the calc dir:

        * ``<timestamp>_<run_type>.h5``        — HELIX-native HDF5 schema.
        * ``<timestamp>_<run_type>.opmd.h5``   — openPMD-1.1 (interop).

        Returns the HELIX-native :class:`pathlib.Path` so existing callers
        keep working (or ``None`` if the HDF5 save itself failed).
        The openPMD failure is silently logged to the status bar — it is
        never fatal because the HELIX-native file is the source of truth.
        """
        import datetime as _dt

        from linac_gen.io.hdf5_output import save_results_hdf5
        from linac_gen.io.openpmd_output import save_results_openpmd

        calc_dir = _resolve_calc_dir()
        try:
            calc_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.state.status_message.emit(
                f"Could not create calc dir {calc_dir}: {exc}"
            )
            return None
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hdf5_path = calc_dir / f"{ts}_{run_type}.h5"
        try:
            _bc = self.state.beam_config
            save_results_hdf5(
                results, str(hdf5_path),
                beam_config=_bc,
                lattice=self.state.lattice,
                # Provenance: pin WHICH deck produced this file.
                lattice_path=self.state.lattice_path,
                # ...and WHICH imported beam, when the run consumed one.
                input_beam_path=(
                    getattr(_bc, "distribution_file", None)
                    if getattr(_bc, "source", "generate") == "file"
                    else None),
            )
        except Exception as exc:
            self.state.status_message.emit(f"Auto-save (HDF5) failed: {exc}")
            return None

        # openPMD companion file — non-fatal on failure.
        opmd_path = calc_dir / f"{ts}_{run_type}.opmd.h5"
        try:
            save_results_openpmd(
                results, str(opmd_path),
                beam_config=self.state.beam_config,
                lattice=self.state.lattice,
            )
        except Exception as exc:
            self.state.status_message.emit(
                f"Auto-save (openPMD) failed: {exc}"
            )
        return hdf5_path

    def _open_docs(self) -> None:
        """Open the HELIX manual in the system's default browser.

        Probe order:
        1. Built mkdocs site at ``<repo>/site/index.html`` (preferred —
           styled HTML, search, navigation).
        2. Markdown source at ``<repo>/docs/manual/index.md`` (renders
           in any markdown-aware editor; raw text in a plain browser).
        3. Repo URL from ``docs/mkdocs.yml`` if neither local copy is
           usable (best-effort fallback).
        """
        from pathlib import Path

        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        # Locate repo root from this module's path: .../HELIX/gui/linac_gen_gui/...
        here = Path(__file__).resolve()
        repo_root = next(
            (p for p in here.parents
             if (p / "docs" / "manual" / "index.md").is_file()),
            None,
        )
        candidates = []
        if repo_root is not None:
            candidates.extend([
                repo_root / "site" / "index.html",
                repo_root / "docs" / "manual" / "_build" / "site" / "index.html",
                repo_root / "docs" / "manual" / "index.md",
            ])

        target = next((p for p in candidates if p.is_file()), None)
        if target is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
            self.state.status_message.emit(f"Opened documentation: {target.name}")
            return

        # Last resort: offer the user a hint instead of opening a dead link.
        QMessageBox.information(
            self, "Documentation",
            "Could not locate the HELIX manual on disk.\n\n"
            "Build the manual once with:\n"
            "    cd docs && mkdocs build\n\n"
            "Or open `docs/manual/index.md` directly in any markdown viewer."
        )


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    app = QApplication(argv)
    app.setApplicationName("HELIX")
    app.setOrganizationName("Helix")
    app.setStyleSheet(theme.dark_qss())
    _install_excepthook()

    # ---- App icon (dock / title-bar / Alt-Tab) ----
    from PyQt6.QtGui import QIcon
    from linac_gen_gui.interphase.splash import LOGO_PATH, HelixSplash
    if LOGO_PATH.is_file():
        app.setWindowIcon(QIcon(str(LOGO_PATH)))

    # ---- Splash, shown for 5 s before the main window ----
    splash = HelixSplash()
    splash.show()
    splash.raise_()
    app.processEvents()

    win = InterphaseWindow()
    if LOGO_PATH.is_file():
        win.setWindowIcon(QIcon(str(LOGO_PATH)))

    # Auto-load the FODO example if present
    try:
        example = Path(__file__).resolve().parents[3] / "examples" / "fodo_cell.dat"
        if example.exists():
            from linac_gen.io.tracewin_parser import parse_tracewin
            lat, _ = parse_tracewin(str(example))
            win.state.set_lattice(lat, str(example))
    except Exception:
        pass

    # Show main window after splash duration; splash closes itself.
    from PyQt6.QtCore import QTimer
    def _finish_splash() -> None:
        if _SHUTTING_DOWN:
            # User quit within the splash window — don't re-show a
            # window that is already torn down.
            splash.close()
            return
        win.show()
        win.raise_()
        win.activateWindow()
        splash.close()
    QTimer.singleShot(5000, _finish_splash)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
