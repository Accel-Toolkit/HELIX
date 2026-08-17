"""Lattice tab — file ops, outline tree, layout strip, element inspector."""
from __future__ import annotations

from collections import Counter
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QFrame, QMenu, QTabWidget,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.panels import (
    OutlineTree, ElementInspector, ElementPalette, TypeChipStrip,
    LatticeListing,
)
from linac_gen_gui.interphase.plots.lattice_track import LatticeTimeline
from linac_gen_gui.interphase.commands import (
    DeleteCommand, InsertCommand, MacroCommand, MoveCommand,
    ParamChangeCommand,
)
from linac_gen_gui.interphase.element_factory import (
    clone as _clone, make_default, make_field_map_from_file,
)
from linac_gen_gui.interphase.lattice_validation import validate as _validate_lattice


class _CorrectionWorker(QThread):
    """Run orbit correction off the GUI thread, on a lattice SNAPSHOT.

    ``run_correction_from_lattice`` mutates steerers in place, so it
    must never touch the live lattice from a background thread — the
    computed kicks are applied on the GUI thread by the done handler,
    through the command bus (undo + dirty flag for free).
    """

    finished_ok = pyqtSignal(object)   # result dict (kicks/history/…)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, lattice_snapshot, beam_cfg, parent=None):
        super().__init__(parent)
        self._lattice = lattice_snapshot
        self._beam_cfg = beam_cfg
        import threading
        self._stop_event = threading.Event()
        # OpenBLAS LU workspace can blow the default macOS QThread
        # stack — same headroom as the surrogate workers.
        self.setStackSize(16 * 1024 * 1024)

    def request_stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            from linac_gen.core.cancelled import OperationCancelled
            from linac_gen.distributions.factory import create_beam
            from linac_gen.errors.correction import run_correction_from_lattice
            cfg = self._beam_cfg
            try:
                res = run_correction_from_lattice(
                    self._lattice,
                    lambda: create_beam(cfg, seed=12345),
                    n_iter=5, tol_mm=0.05, history=True,
                    should_stop=self._stopping,
                    # Steer onto DIAG_POSITION targets when the deck (or
                    # a loaded BPM-targets file) carries them; target-
                    # less decks resolve to zeros = the old flatten.
                    targets="deck",
                )
            except OperationCancelled:
                self.cancelled.emit()
                return
            self.finished_ok.emit(res)
        except Exception as exc:                              # noqa: BLE001
            self.failed.emit(str(exc))


class LatticeTab(QWidget):
    open_requested = pyqtSignal()
    save_requested = pyqtSignal()
    save_as_requested = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._corr_worker: _CorrectionWorker | None = None
        self._corr_lattice_at_launch = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ---- Toolbar strip -------------------------------------------
        bar = QFrame()
        bar.setObjectName("stageToolbar")
        bar.setFixedHeight(36)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 4, 10, 4); bl.setSpacing(6)
        for label, cb in (("Open…", self.open_requested.emit),
                          ("Save",  self.save_requested.emit),
                          ("Save As…", self.save_as_requested.emit),
                          ("Reload", lambda: self._reload())):
            b = QPushButton(label)
            b.clicked.connect(cb)
            bl.addWidget(b)

        bl.addSpacing(6)
        # ---- Edit toolbar (Add / Duplicate / Delete / Undo / Redo) ---
        self._btn_add  = QPushButton("+ Add…")
        self._btn_add.setToolTip("Add element  (Insert)")
        self._btn_add.clicked.connect(self._add_element)
        bl.addWidget(self._btn_add)

        self._btn_dup  = QPushButton("Duplicate")
        self._btn_dup.setToolTip("Duplicate selected element  (Ctrl+D)")
        self._btn_dup.clicked.connect(self._duplicate_selected)
        bl.addWidget(self._btn_dup)

        self._btn_del  = QPushButton("Delete")
        self._btn_del.setToolTip("Delete selected element  (Delete)")
        self._btn_del.clicked.connect(self._delete_selected)
        bl.addWidget(self._btn_del)

        bl.addSpacing(6)
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setToolTip("Undo  (Ctrl+Z)")
        self._btn_undo.clicked.connect(self.state.bus.undo)
        bl.addWidget(self._btn_undo)

        self._btn_redo = QPushButton("Redo")
        self._btn_redo.setToolTip("Redo  (Ctrl+Y / Ctrl+Shift+Z)")
        self._btn_redo.clicked.connect(self.state.bus.redo)
        bl.addWidget(self._btn_redo)

        bl.addSpacing(6)
        self._btn_correct = QPushButton("Correct orbit")
        self._btn_correct.setToolTip(
            "Run a one-shot orbit-correction pass on the current lattice "
            "using ADJUST_STEERER + DIAG_POSITION cards (or BPM_*/STEER_* "
            "name patterns when no cards are present).  DIAG_POSITION "
            "targets (deck operands or a loaded BPM-targets file) are "
            "steered TO; target-less BPMs are flattened to zero.  BPM "
            "readings come from multi-particle tracking."
        )
        self._btn_correct.clicked.connect(self._on_correct_orbit)
        bl.addWidget(self._btn_correct)

        self._btn_bpm_targets = QPushButton("BPM targets…")
        self._btn_bpm_targets.setToolTip(
            "Load an external BPM-target file (one 'x_mm y_mm [weight]' "
            "row per is_bpm marker, lattice order; nan leaves a plane "
            "free).  Targets override DIAG_POSITION operands for orbit "
            "correction and diagnostic matching; they are runtime-only "
            "and never saved into the .dat."
        )
        self._btn_bpm_targets.clicked.connect(self._on_load_bpm_targets)
        bl.addWidget(self._btn_bpm_targets)

        self._btn_to_vane = QPushButton("RfqCell → VaneRFQ")
        self._btn_to_vane.setToolTip(
            "Pick a .vane file and replace this lattice's RfqCell chain "
            "with one VaneRFQ element (Toutatis-equivalent path).  Use "
            "the inspector field_model dropdown to choose 2term / 8term "
            "/ laplace2d after the swap."
        )
        self._btn_to_vane.clicked.connect(self._on_replace_with_vane_rfq)
        bl.addWidget(self._btn_to_vane)

        bl.addSpacing(12)
        self._summary = QLabel("no lattice loaded")
        self._summary.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:11px;"
        )
        bl.addWidget(self._summary)
        bl.addStretch(1)
        v.addWidget(bar)

        # ---- Four-column layout: Palette | Outline | Timeline | Inspector
        body = QHBoxLayout()
        body.setContentsMargins(10, 10, 10, 10); body.setSpacing(10)
        self._palette = ElementPalette(on_activate=self._add_default_of_type)
        body.addWidget(self._palette)
        self._outline = OutlineTree(state)
        self._outline.setFixedWidth(240)
        body.addWidget(self._outline)

        # Center: type-chip strip + timeline + element-type breakdown
        center = QVBoxLayout(); center.setSpacing(8)
        self._chips = TypeChipStrip()
        center.addWidget(self._chips)
        self._timeline = LatticeTimeline()
        center.addWidget(self._timeline)

        # Bottom-of-center pane has two views the user can flip between:
        # the per-type breakdown summary and the sequential card listing.
        # The listing is what tells the user *which* element they are
        # editing — selecting on the timeline highlights the matching row.
        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.setStyleSheet(
            f"QTabWidget::pane {{ background:{theme.BG_INSET}; "
            f"border:1px solid {theme.BORDER_0}; border-radius:4px; }}"
            f"QTabBar::tab {{ background:{theme.BG_2}; color:{theme.TEXT_2}; "
            f"padding:4px 14px; border:1px solid {theme.BORDER_0}; "
            f"border-bottom:0; font-size:11px; }}"
            f"QTabBar::tab:selected {{ background:{theme.BG_INSET}; "
            f"color:{theme.TEXT_0}; border-top:1px solid {theme.ACCENT}; }}"
        )
        # Sequential card listing — primary view; opens by default.
        self._listing = LatticeListing(state)
        self._bottom_tabs.addTab(self._listing, "Sequence")
        # Per-type breakdown summary.
        self._breakdown = QLabel("")
        self._breakdown.setStyleSheet(
            f"background:{theme.BG_INSET}; padding:10px; color:{theme.TEXT_1};"
            f"font-family:{theme.FONT_MONO}; font-size:11px;"
        )
        self._breakdown.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._breakdown.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._bottom_tabs.addTab(self._breakdown, "Breakdown")
        center.addWidget(self._bottom_tabs, stretch=1)
        body.addLayout(center, stretch=1)

        self._inspector = ElementInspector(state)
        self._inspector.setFixedWidth(300)
        body.addWidget(self._inspector)
        v.addLayout(body, stretch=1)

        # ---- wiring ----------------------------------------------------
        state.lattice_changed.connect(self._on_lattice)
        state.selected_element_changed.connect(self._timeline.set_selected)
        state.s_cursor_changed.connect(self._timeline.set_s_cursor)
        self._timeline.element_clicked.connect(state.set_selected)
        # Palette → timeline drop integration.
        self._timeline.element_dropped.connect(self._on_palette_drop)
        # Outline tree drag-reorder → MoveCommand.
        self._outline.move_requested.connect(self._on_tree_move)
        # Type-chip filter → timeline dimming.
        self._chips.filter_changed.connect(self._timeline.set_type_filter)
        # Re-validate after every bus change.
        state.bus.changed.connect(self._refresh_validation)

        # In-process clipboard for Cut / Copy / Paste (deep-copy).
        self._clipboard = None
        self._install_shortcuts()
        self._install_context_menus()
        # Sync undo/redo button enabled state with the bus.
        state.bus.can_undo_changed.connect(self._btn_undo.setEnabled)
        state.bus.can_redo_changed.connect(self._btn_redo.setEnabled)
        self._btn_undo.setEnabled(state.bus.can_undo)
        self._btn_redo.setEnabled(state.bus.can_redo)

        if state.lattice is not None:
            self._on_lattice(state.lattice)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------
    def _install_shortcuts(self) -> None:
        s = self.state
        # Insert → add dialog.  (Ctrl+N belongs to File → New Project at
        # the window level; a second window-context binding here would
        # make Qt treat BOTH as ambiguous and fire neither.)
        QShortcut(QKeySequence("Ins"),    self, activated=self._add_element)
        # Delete → remove selected
        QShortcut(QKeySequence("Del"),    self, activated=self._delete_selected)
        # Ctrl+D → duplicate
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self._duplicate_selected)
        # Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z → undo / redo
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=s.bus.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=s.bus.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=s.bus.redo)
        # Ctrl+C / X / V → clipboard
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._copy_selected)
        QShortcut(QKeySequence("Ctrl+X"), self, activated=self._cut_selected)
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._paste_clipboard)
        # Ctrl+0 → reset timeline zoom
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._timeline.reset_zoom)

    # ------------------------------------------------------------------
    def apply_font_size(self, base: int) -> None:
        """Forward the global font-size change to inline-styled children."""
        apply = getattr(self._inspector, "apply_font_size", None)
        if callable(apply):
            apply(base)
        # The toolbar strip's height was frozen at construction; grow it
        # with the font so its buttons don't clip at large sizes.
        bar = self.findChild(QFrame, "stageToolbar")
        if bar is not None:
            bar.setFixedHeight(max(36, int(base) + 24))

    def shutdown_begin(self) -> list:
        """App teardown: cancel a running orbit correction."""
        w = self._corr_worker
        if w is not None and w.isRunning():
            w.request_stop()
            w.requestInterruption()
            return [w]
        return []

    # ------------------------------------------------------------------
    # Element-CRUD actions
    # ------------------------------------------------------------------
    def _selected_index(self) -> int | None:
        lat = self.state.lattice; sel = self.state.selected
        if lat is None or sel is None:
            return None
        for i, e in enumerate(lat.elements):
            if e is sel:
                return i
        return None

    def _add_element(self) -> None:
        if self.state.lattice is None:
            QMessageBox.information(self, "No lattice", "Load a lattice first.")
            return
        from linac_gen_gui.interphase.dialogs.add_element import AddElementDialog
        dlg = AddElementDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        new = dlg.element()
        if new is None:
            return
        idx = self._selected_index()
        insert_at = (idx + 1) if idx is not None else len(self.state.lattice.elements)
        self.state.bus.do(InsertCommand(insert_at, new))
        self.state.set_selected(new)
        self.state.status_message.emit(f"added {type(new).__name__} '{new.name}'")

    def _delete_selected(self) -> None:
        if self.state.lattice is None:
            return
        # If the timeline has a Shift+Click multi-select, delete all of
        # them as a single undoable MacroCommand.
        multi = list(self._timeline.selected_elements()) if hasattr(self._timeline, "selected_elements") else []
        if len(multi) > 1:
            cmds = [DeleteCommand(e) for e in multi]
            self.state.bus.do(MacroCommand(cmds, label=f"Delete {len(multi)} elements"))
            self.state.set_selected(None)
            self.state.status_message.emit(f"deleted {len(multi)} elements")
            return
        sel = self.state.selected
        if sel is None:
            return
        self.state.bus.do(DeleteCommand(sel))
        self.state.set_selected(None)
        self.state.status_message.emit(f"deleted '{getattr(sel, 'name', '?')}'")

    def _duplicate_selected(self) -> None:
        sel = self.state.selected
        idx = self._selected_index()
        if sel is None or idx is None or self.state.lattice is None:
            return
        new = _clone(sel)
        self.state.bus.do(InsertCommand(idx + 1, new))
        self.state.set_selected(new)
        self.state.status_message.emit(f"duplicated '{sel.name}' → '{new.name}'")

    def _copy_selected(self) -> None:
        sel = self.state.selected
        if sel is None: return
        self._clipboard = _clone(sel)
        self.state.status_message.emit(f"copied '{sel.name}'")

    def _cut_selected(self) -> None:
        sel = self.state.selected
        if sel is None: return
        self._clipboard = _clone(sel)
        self.state.bus.do(DeleteCommand(sel))
        self.state.set_selected(None)
        self.state.status_message.emit(f"cut '{sel.name}'")

    def _paste_clipboard(self) -> None:
        if self._clipboard is None or self.state.lattice is None: return
        new = _clone(self._clipboard)
        idx = self._selected_index()
        insert_at = (idx + 1) if idx is not None else len(self.state.lattice.elements)
        self.state.bus.do(InsertCommand(insert_at, new))
        self.state.set_selected(new)
        self.state.status_message.emit(f"pasted '{new.name}'")

    # ------------------------------------------------------------------
    # Right-click context menu (timeline + outline tree)
    # ------------------------------------------------------------------
    def _install_context_menus(self) -> None:
        self._timeline.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._timeline.customContextMenuRequested.connect(self._show_ctx_menu)
        self._outline.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._outline.customContextMenuRequested.connect(self._show_ctx_menu)

    def _show_ctx_menu(self, pos) -> None:
        sender = self.sender()
        m = QMenu(self)
        has_sel = self.state.selected is not None
        m.addAction("Insert before",
                    lambda: self._add_element_at(self._selected_index()))
        m.addAction("Insert after", self._add_element)
        m.addSeparator()
        a_dup = m.addAction("Duplicate", self._duplicate_selected); a_dup.setEnabled(has_sel)
        a_del = m.addAction("Delete",    self._delete_selected);    a_del.setEnabled(has_sel)
        m.addSeparator()
        a_cut = m.addAction("Cut",   self._cut_selected);   a_cut.setEnabled(has_sel)
        a_cpy = m.addAction("Copy",  self._copy_selected);  a_cpy.setEnabled(has_sel)
        a_pst = m.addAction("Paste", self._paste_clipboard)
        a_pst.setEnabled(self._clipboard is not None)
        m.exec(sender.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Palette → drop / double-click integration
    # ------------------------------------------------------------------
    def _new_element_of_type(self, type_name: str):
        """Construct a fresh element of ``type_name`` for insertion.

        For FieldMap / FieldMap3D this opens a file picker so the user
        can supply the external field data; everything else falls
        through to ``make_default``.  Returns ``None`` if the user
        cancels or the read fails (caller should silently skip).
        """
        if type_name in ("FieldMap", "FieldMap3D"):
            from PyQt6.QtWidgets import QFileDialog
            if type_name == "FieldMap":
                pat = ("Field maps (*.edz *.csv);;"
                       "TraceWin .edz (*.edz);;CSV (*.csv);;All files (*.*)")
            else:
                pat = ("3-D field maps (*.bdz *.edz *.bdx *.bdy *.edx *.edy);;"
                       "All files (*.*)")
            fname, _ = QFileDialog.getOpenFileName(
                self, f"Select {type_name} data file", "", pat)
            if not fname:
                return None
            try:
                return make_field_map_from_file(type_name, fname)
            except Exception as exc:
                QMessageBox.critical(
                    self, f"Load {type_name}",
                    f"Failed to read field-map file:\n{exc}")
                return None
        return make_default(type_name)

    def _add_default_of_type(self, type_name: str) -> None:
        """Append a placeholder element of ``type_name`` after the
        current selection (or at the end).  Used by palette
        double-click for accessibility."""
        if self.state.lattice is None:
            QMessageBox.information(self, "No lattice", "Load a lattice first.")
            return
        try:
            new = self._new_element_of_type(type_name)
        except Exception as exc:
            QMessageBox.critical(self, "Add element", str(exc)); return
        if new is None:
            return  # user cancelled the file picker
        idx = self._selected_index()
        insert_at = (idx + 1) if idx is not None else len(self.state.lattice.elements)
        self.state.bus.do(InsertCommand(insert_at, new))
        self.state.set_selected(new)
        self.state.status_message.emit(f"added {type_name} '{new.name}' at #{insert_at}")

    def _on_palette_drop(self, type_name: str, idx: int) -> None:
        """Handle a drop from the ElementPalette onto the timeline."""
        if self.state.lattice is None:
            return
        try:
            new = self._new_element_of_type(type_name)
        except Exception as exc:
            QMessageBox.critical(self, "Add element", str(exc)); return
        if new is None:
            return  # user cancelled the file picker
        self.state.bus.do(InsertCommand(idx, new))
        self.state.set_selected(new)
        self.state.status_message.emit(f"dropped {type_name} '{new.name}' at #{idx}")

    def _on_tree_move(self, from_idx: int, to_idx: int) -> None:
        """Outline tree → MoveCommand.  The bus rebuilds via
        lattice_changed → tree.rebuild, so we don't trust Qt's
        internal-move state."""
        if self.state.lattice is None:
            return
        if from_idx == to_idx or from_idx < 0:
            return
        n = len(self.state.lattice.elements)
        # Clamp targets so a drop at "end" works regardless of tree shape.
        to = max(0, min(int(to_idx), n - 1))
        self.state.bus.do(MoveCommand(int(from_idx), to))

    def _add_element_at(self, idx: int | None) -> None:
        from linac_gen_gui.interphase.dialogs.add_element import AddElementDialog
        if self.state.lattice is None: return
        dlg = AddElementDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        new = dlg.element()
        if new is None: return
        insert_at = idx if idx is not None else len(self.state.lattice.elements)
        self.state.bus.do(InsertCommand(insert_at, new))
        self.state.set_selected(new)

    # ------------------------------------------------------------------
    def _reload(self) -> None:
        if not self.state.lattice_path:
            return
        if self.state.bus.dirty:
            r = QMessageBox.question(
                self, "Reload",
                "The lattice has unsaved changes. Reload from disk?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        try:
            # Route by extension, exactly as app._open_lattice does — a
            # loaded .madx/.seq/.lat lattice must reload through its own
            # parser, not TraceWin (MAD8 ``NAME: TYPE`` lines resemble
            # TraceWin label syntax and would mis-parse silently).
            from linac_gen_gui.interphase.app import _parse_lattice_file
            lp = self.state.lattice_path
            lat, _ = _parse_lattice_file(lp)
            self.state.set_lattice(lat, lp)
        except Exception as exc:
            QMessageBox.critical(self, "Reload failed", str(exc))

    def _refresh_validation(self) -> None:
        warnings = _validate_lattice(self.state.lattice)
        self._timeline.set_validation(warnings)

    # ------------------------------------------------------------------
    def _on_correct_orbit(self) -> None:
        """Run a one-shot orbit correction on the loaded lattice.

        The correction (n_iter tracking passes) runs in a worker on a
        deepcopy — it used to run INLINE on the GUI thread, freezing the
        app for its whole duration.  The resulting ``bx_l``/``by_l``
        kicks are applied to the live lattice via an undoable
        MacroCommand when the worker finishes.  While running, the
        button turns into a Cancel.  Use the Error Study tab when you want
        correction to run per-seed inside an error study instead.
        """
        prev = self._corr_worker
        if prev is not None and prev.isRunning():
            prev.request_stop()
            prev.requestInterruption()
            if self.state.lattice is getattr(prev, "_live_lattice_at_launch",
                                             None):
                # Same lattice → this is the user's second click: cancel.
                self.state.status_message.emit("cancelling orbit correction…")
                return
            # The lattice changed under the running worker (e.g. the
            # auto-correct-on-load hook firing after a project load):
            # cancel the stale run and fall through to correct the NEW
            # lattice.  The old worker is parented to this tab (no GC
            # hazard) and the sender guards below reject its late
            # emissions.
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice",
                                "Load a lattice first.")
            return
        if self.state.beam_config is None:
            QMessageBox.warning(self, "No beam",
                                "Configure a beam first (Beam tab).")
            return

        import copy
        worker = _CorrectionWorker(
            copy.deepcopy(self.state.lattice),
            copy.deepcopy(self.state.beam_config),
            parent=self,
        )
        # Staleness reference lives ON the worker: a tab-level attribute
        # would be overwritten by a relaunch and let the OLD worker's
        # result pass the guard.
        worker._live_lattice_at_launch = self.state.lattice
        worker.finished_ok.connect(self._on_correction_done)
        worker.failed.connect(self._on_correction_failed)
        worker.cancelled.connect(self._on_correction_cancelled)
        self._corr_worker = worker
        self._btn_correct.setText("Cancel correction")
        self.state.status_message.emit("running orbit correction…")
        worker.start()

    def _restore_correct_btn(self) -> None:
        self._btn_correct.setText("Correct orbit")

    def _on_load_bpm_targets(self) -> None:
        """Load an external BPM-target file onto the LIVE lattice state.

        Overrides are runtime-only attributes on the is_bpm markers —
        both the Correct-orbit snapshot (deepcopy) and the Matching
        tab's lattice copy inherit them; the writer never serializes
        them, so save never bakes them into the deck.
        """
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first.")
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load BPM targets", "",
            "Target files (*.txt *.dat *.csv);;All files (*)")
        if not path:
            return
        from linac_gen.io.diag_targets import (apply_diag_targets,
                                               load_diag_targets)
        try:
            n = apply_diag_targets(self.state.lattice,
                                   load_diag_targets(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "BPM targets", str(exc))
            return
        self.state.status_message.emit(
            f"BPM targets: {n} row(s) loaded from {path} "
            "(runtime override — not saved into the .dat)")

    def _on_correction_done(self, res) -> None:
        w = self.sender()
        if w is not None and w is not self._corr_worker:
            return   # a cancelled predecessor finishing late
        self._restore_correct_btn()
        if self.state.lattice is not getattr(w, "_live_lattice_at_launch",
                                             self._corr_lattice_at_launch):
            self.state.status_message.emit(
                "orbit correction discarded — the lattice changed "
                "while it was computing")
            return
        if not res["kicks"]:
            QMessageBox.information(
                self, "No correction",
                "No ADJUST_STEERER cards found and no BPM_*/STEER_* "
                "name pattern matches.  Add steerers + BPMs (or DIAG_POSITION "
                "cards) to enable orbit correction.",
            )
            return
        # The worker corrected a SNAPSHOT — apply the kicks to the live
        # lattice here on the GUI thread, through the command bus, so
        # the correction is a single undoable step and the dirty flag /
        # lattice_changed broadcast come for free.
        from linac_gen.elements.steerer import Steerer
        by_name = {e.name: e for e in self.state.lattice.elements
                   if isinstance(e, Steerer)}
        cmds = []
        for name, k in res["kicks"].items():
            el = by_name.get(name)
            if el is None:
                continue
            cmds.append(ParamChangeCommand(el, "bx_l",
                                           float(el.bx_l), float(k["bx_l"])))
            cmds.append(ParamChangeCommand(el, "by_l",
                                           float(el.by_l), float(k["by_l"])))
        if cmds:
            self.state.bus.do(MacroCommand(cmds, label="Orbit correction"))
            # Fitted steerer values — reroute plain Save to Save-As so
            # the source deck can't be silently overwritten.
            self.state.lattice_fitted = True

        hist = res["history"]
        rms_first = hist[0]["rms_orbit_mm"] if hist else 0.0
        rms_final = hist[-1]["rms_orbit_mm"] if hist else 0.0
        n_pairs = res["n_pairs"]
        method = res["method"]
        msg = (
            f"Method: {method}\n"
            f"Steerer/BPM pairs: {n_pairs}\n"
            f"RMS orbit error vs targets, iter 1: {rms_first:.4f} mm\n"
            f"RMS orbit error vs targets, final:  {rms_final:.4f} mm\n"
            f"(target-less BPMs steer to zero)\n"
            f"\nApplied kicks:\n"
        )
        for name, k in res["kicks"].items():
            msg += f"  {name}: bx_l={k['bx_l']:+.3e}  by_l={k['by_l']:+.3e}\n"
        QMessageBox.information(self, "Orbit correction", msg)
        self.state.status_message.emit(
            f"correction: RMS {rms_first:.3f} → {rms_final:.4f} mm "
            f"({method}, {n_pairs} pairs)"
        )

    def _on_correction_failed(self, msg: str) -> None:
        w = self.sender()
        if w is not None and w is not self._corr_worker:
            return
        self._restore_correct_btn()
        QMessageBox.critical(self, "Correction failed", msg)
        self.state.status_message.emit("correction failed")

    def _on_correction_cancelled(self) -> None:
        w = self.sender()
        if w is not None and w is not self._corr_worker:
            return   # the predecessor a relaunch cancelled — not ours
        self._restore_correct_btn()
        self.state.status_message.emit(
            "orbit correction cancelled — no kicks applied")

    def _on_replace_with_vane_rfq(self) -> None:
        """Replace the lattice's RfqCell chain with one VaneRFQ element."""
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first.")
            return
        from linac_gen.elements.rfq_cell import RfqCell
        n_cells = sum(1 for e in self.state.lattice.elements
                      if isinstance(e, RfqCell))
        if n_cells == 0:
            QMessageBox.information(
                self, "No RfqCells",
                "This lattice has no RfqCell elements to consolidate. "
                "Open a TraceWin lattice with RFQ_CELL cards first.")
            return
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select .vane file", "",
            "Vane geometry (*.vane);;All files (*.*)")
        if not fname:
            return
        models = ["2term", "8term", "8term_full", "laplace2d", "laplace3d"]
        model, ok = QInputDialog.getItem(
            self, "Field model", "field_model:", models, 0, editable=False)
        if not ok:
            return
        try:
            from linac_gen.io.vane_rfq_helper import replace_rfq_cells_with_vane
            replace_rfq_cells_with_vane(self.state.lattice, fname,
                                        field_model=model)
        except Exception as exc:
            QMessageBox.critical(self, "VaneRFQ swap failed", str(exc))
            return
        # Wholesale element replacement — flag divergence from the .dat.
        if hasattr(self.state, "bus"):
            self.state.bus.mark_dirty()
        self.state.lattice_changed.emit(self.state.lattice)
        self.state.status_message.emit(
            f"replaced {n_cells} RfqCells with VaneRFQ ({model})"
        )

    def _on_lattice(self, lattice) -> None:
        self._timeline.set_lattice(lattice)
        self._chips.set_lattice(lattice)
        self._refresh_validation()
        if lattice is None:
            self._summary.setText("no lattice loaded")
            self._breakdown.setText("")
            return
        counts = Counter(type(e).__name__ for e in lattice.elements)
        total = sum(e.length for e in lattice.elements)
        self._summary.setText(
            f"{len(lattice.elements)} elements · total length {total:.1f} mm"
        )
        pct_rows = []
        total_length = sum(e.length for e in lattice.elements) or 1.0
        length_by_type = Counter()
        for e in lattice.elements:
            length_by_type[type(e).__name__] += e.length
        head = f"{'type':<14}{'count':>8}{'length':>12}{'pct':>8}"
        lines = [head, "-" * len(head)]
        for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            L = length_by_type[t]
            lines.append(f"{t:<14}{n:>8}{L:>11.1f}{L/total_length*100:>7.1f}%")
        lines.append("-" * len(head))
        lines.append(f"{'TOTAL':<14}{len(lattice.elements):>8}{total:>11.1f}{100.0:>7.1f}%")
        self._breakdown.setText("\n".join(lines))
