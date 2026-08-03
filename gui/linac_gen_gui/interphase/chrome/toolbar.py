"""Global toolbar — menus + Run controls + s-cursor (doubles as run progress).

Replaces the menubar + subbar from the 17-workspace layout.  All
simulation actions live here so they work from any tab.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QPushButton, QMenu, QLabel, QSlider,
    QSpinBox,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.state import AppState


class Toolbar(QFrame):
    # Menu signals
    open_lattice_requested   = pyqtSignal()
    save_lattice_requested   = pyqtSignal()
    save_lattice_as_requested = pyqtSignal()
    # Project = beam config + step / SC grid / backend + record-substeps
    # + a reference to the currently-loaded lattice file.  Stored as a
    # self-describing JSON.
    open_project_requested   = pyqtSignal()
    save_project_requested   = pyqtSignal()

    # Run signals
    run_envelope_requested   = pyqtSignal()
    run_mp_requested         = pyqtSignal()
    run_backtrack_requested  = pyqtSignal()

    # Tools
    open_assistant_requested = pyqtSignal()
    open_console_requested   = pyqtSignal()
    open_transfer_matrix_requested = pyqtSignal()
    open_sigma_matrix_requested = pyqtSignal()
    open_parameter_scan_requested = pyqtSignal()
    open_docs_requested      = pyqtSignal()
    open_about_requested     = pyqtSignal()
    check_updates_requested  = pyqtSignal()
    update_check_toggled     = pyqtSignal(bool)

    # Stop a running simulation (app wires this to the active worker's
    # request_stop so the solver cancels at the next element boundary).
    stop_requested           = pyqtSignal()

    # Export TraceWin-compatible result files (partran1.out + envelope .txt)
    # from the most recent envelope/multi-particle run.
    export_tracewin_requested = pyqtSignal()

    # Export results in openPMD-beamphysics format (interop with
    # openPMD-viewer / Astra / Genesis / LUME wrappers).
    export_openpmd_requested = pyqtSignal()

    # Pick the directory where auto-saved HDF5 dumps land after each run.
    set_calc_dir_requested = pyqtSignal()

    # Recent Projects submenu — emitted with the full path when a user
    # picks an entry.  ``clear_recent_requested`` empties the list.
    open_recent_requested  = pyqtSignal(str)
    clear_recent_requested = pyqtSignal()

    # Global font-size scaling (spinbox in the toolbar).  Emits the new
    # base point size; the app re-applies the stylesheet.
    font_size_changed        = pyqtSignal(int)

    def __init__(self, state: AppState):
        super().__init__()
        self.setObjectName("menubar")
        self.setFixedHeight(44)
        self._state = state

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(4)

        # Recent Projects — submenu re-populated by ``set_recent_projects``
        # whenever the app opens or saves a .lgproj.  Stored as a QMenu so
        # ``_mk_menu`` adds it via QMenu.addMenu (submenu carat appears
        # automatically).
        self._recent_menu = QMenu(self)
        self._recent_menu.setTitle("Open Recent")
        self._recent_menu.setStyleSheet(self._menu_qss())
        # Placeholder so the submenu isn't empty before any project is opened.
        self._render_recent_projects([])

        # ---- Menus ----------------------------------------------------
        lay.addWidget(self._mk_menu("File", [
            ("Open Lattice…",   self.open_lattice_requested.emit),
            ("Save Lattice",    self.save_lattice_requested.emit),
            ("Save Lattice As…",self.save_lattice_as_requested.emit),
            ("---", None),
            ("Open Project…",   self.open_project_requested.emit),
            ("Open Recent",     self._recent_menu),
            ("Save Project…",   self.save_project_requested.emit),
            ("---", None),
            ("Set Calculation Directory…", self.set_calc_dir_requested.emit),
            ("Export TraceWin output…", self.export_tracewin_requested.emit),
            ("Export openPMD output…",  self.export_openpmd_requested.emit),
            ("---", None),
            ("Exit",            lambda: self.window().close()),
        ]))
        lay.addWidget(self._mk_menu("Simulate", [
            ("Run Envelope",       self.run_envelope_requested.emit),
            ("Run Multi-particle", self.run_mp_requested.emit),
            ("---", None),
            ("Backtrack Distribution…", self.run_backtrack_requested.emit),
        ]))
        lay.addWidget(self._mk_menu("Tools", [
            ("Assistant…  (AI, optional)", self.open_assistant_requested.emit),
            ("Python Console…",        self.open_console_requested.emit),
            ("Show Transfer Matrix…",  self.open_transfer_matrix_requested.emit),
            ("Show Sigma Matrix…",     self.open_sigma_matrix_requested.emit),
            ("Parameter Scan…",        self.open_parameter_scan_requested.emit),
        ]))
        # Retained QActions so the app can retitle / sync them later
        self._update_action = QAction("Check for Updates…", self)
        self._update_action.triggered.connect(
            self.check_updates_requested.emit)
        self._update_toggle = QAction("Check for Updates at Startup",
                                      self)
        self._update_toggle.setCheckable(True)
        self._update_toggle.setChecked(True)
        self._update_toggle.toggled.connect(
            self.update_check_toggled.emit)
        lay.addWidget(self._mk_menu("Help", [
            ("Documentation", self.open_docs_requested.emit),
            ("---", None),
            ("Check for Updates…", self._update_action),
            ("Check for Updates at Startup", self._update_toggle),
            ("---", None),
            ("About HELIX", self.open_about_requested.emit),
        ]))

        self._add_separator(lay)

        # ---- Run controls --------------------------------------------
        self._run_env_btn = QPushButton("  Run Envelope")
        self._run_env_btn.setIcon(icon("play", 12, "#00161c"))
        self._run_env_btn.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; border-radius:3px;"
            f"padding:6px 12px; font-weight:600;"
        )
        self._run_env_btn.clicked.connect(self.run_envelope_requested)
        lay.addWidget(self._run_env_btn)

        self._run_mp_btn = QPushButton("  Run Multi-particle")
        self._run_mp_btn.setIcon(icon("zap", 12))
        self._run_mp_btn.setStyleSheet(
            f"background:{theme.BG_2}; color:{theme.TEXT_0}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:3px; padding:6px 12px;"
        )
        self._run_mp_btn.clicked.connect(self.run_mp_requested)
        lay.addWidget(self._run_mp_btn)

        # Labelled red "Stop" button — previously a 30-px icon-only square
        # that was easy to miss.  Now shows text so the user can see it
        # and styled like a primary button when enabled.
        self._stop_btn = QPushButton("  Stop")
        self._stop_btn.setIcon(icon("stop", 12, "#ffffff"))
        self._stop_btn.setStyleSheet(
            f"QPushButton {{ background:{theme.ERR}; color:#ffffff; "
            f"border:0; border-radius:3px; padding:6px 14px; font-weight:600; }}"
            f"QPushButton:disabled {{ background:{theme.BG_2}; "
            f"color:{theme.TEXT_2}; border:1px solid {theme.BORDER_1}; "
            f"font-weight:400; }}"
        )
        # Clicking Stop now both flips the UI running flag (so buttons
        # re-enable without waiting for the worker to finish) and fires
        # ``stop_requested`` so the app can cancel the active worker.
        def _on_stop_clicked():
            self.stop_requested.emit()
            state.set_running(False)
        self._stop_btn.clicked.connect(_on_stop_clicked)
        self._stop_btn.setToolTip("Cancel the running simulation at the next element")
        lay.addWidget(self._stop_btn)

        self._add_separator(lay)

        # ---- s-cursor ------------------------------------------------
        # During a run the slider doubles as the progress indicator (it
        # sweeps the lattice via set_live_s) and the label carries a live
        # "· N%" suffix.  The toolbar used to also have a QProgressBar
        # here, but the two filled in lockstep — one was removed.
        self._run_pct: int | None = None
        self._live_s_text: str | None = None
        self._s_label = QLabel("s = 0 / — mm")
        lay.addWidget(self._s_label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setFixedWidth(160)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self._slider)

        lay.addStretch(1)

        # ---- Font-size scaler ----------------------------------------
        self._add_separator(lay)
        self._font_lbl = QLabel("Font")
        lay.addWidget(self._font_lbl)
        self._font_spin = QSpinBox()
        self._font_spin.setRange(9, 22)
        self._font_spin.setValue(theme.FONT_SIZE)
        self._font_spin.setSuffix(" pt")
        # 70 px is too narrow on the macOS native spinbox style — it clips
        # the up/down arrows off the right edge, leaving the user with a
        # bare number field and no way to step.  96 px gives the value
        # area + both arrows enough room on every platform.
        self._font_spin.setFixedWidth(96)
        self._font_spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self._font_spin.setToolTip("Application font size (Ctrl++ / Ctrl+-)")
        self._font_spin.valueChanged.connect(self.font_size_changed.emit)
        lay.addWidget(self._font_spin)

        state.s_cursor_changed.connect(self._refresh_s)
        state.lattice_changed.connect(self._refresh_lattice)
        state.running_changed.connect(self._refresh_run_buttons)
        # Stop is disabled until a run starts; Run until a lattice loads.
        self._stop_btn.setEnabled(False)
        self._update_run_enabled()

    # ------------------------------------------------------------------
    def set_font_size(self, pt: int) -> None:
        """Sync the spin-box value silently (no signal emit)."""
        self._font_spin.blockSignals(True)
        self._font_spin.setValue(int(pt))
        self._font_spin.blockSignals(False)

    def apply_font_size(self, base: int) -> None:
        """Re-style inline-styled toolbar elements to match the new base."""
        sz = max(8, int(base) - 2)
        self._s_label.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO};"
            f" font-size:{sz}px; padding:0 6px;"
        )
        self._font_lbl.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:{sz}px; padding:0 4px 0 2px;"
        )
        # Toolbar height grows with base so the controls remain centred.
        self.setFixedHeight(max(44, int(base) + 30))

    # ------------------------------------------------------------------
    def set_progress(self, pct: int) -> None:
        """Slot for the run workers' ``progress`` signal.

        The percentage renders as a suffix on the s-label
        ("s = 3400 / 18957 mm · 18%").  Values <= 0 or >= 100 clear the
        suffix (run start / done / fail / abort) — completion itself is
        announced by the status-bar state pill, not here.
        """
        p = int(pct)
        self._run_pct = p if 0 < p < 100 else None
        if self._live_s_text is not None:
            self._render_live_s()

    def set_live_s(self, s_mm: float, total_mm: float) -> None:
        """Display the solver's current longitudinal position while a
        simulation is running.  Updates the s-label text (with the live
        percent suffix) and the slider position so the user can see
        progress in real s along the lattice, not just a 0–100 %
        abstraction.
        """
        self._live_s_text = f"s = {s_mm:.0f} / {total_mm:.0f} mm"
        self._render_live_s()
        if total_mm > 0 and self._slider.maximum() > 0:
            self._slider.blockSignals(True)
            self._slider.setValue(int(min(s_mm, self._slider.maximum())))
            self._slider.blockSignals(False)

    def _render_live_s(self) -> None:
        txt = self._live_s_text or ""
        if self._run_pct is not None:
            txt += f" · {self._run_pct}%"
        self._s_label.setText(txt)

    # ------------------------------------------------------------------
    def _add_separator(self, lay: QHBoxLayout) -> None:
        sep = QFrame()
        sep.setFixedSize(1, 20)
        sep.setStyleSheet(f"background:{theme.BORDER_1};")
        lay.addSpacing(6); lay.addWidget(sep); lay.addSpacing(6)

    def _menu_qss(self) -> str:
        """Stylesheet shared between the top-level menus and the Recent submenu."""
        return f"""
            QMenu {{ background:{theme.BG_2}; border:1px solid {theme.BORDER_2};
                    color:{theme.TEXT_0}; padding:4px; }}
            QMenu::item {{ padding:5px 18px; }}
            QMenu::item:selected {{
                background:rgba(34,211,238,0.18); color:{theme.TEXT_0};
            }}
            QMenu::separator {{ height:1px; background:{theme.BORDER_1}; margin:4px 0; }}
        """

    def set_update_available(self, tag: str) -> None:
        """Emphasise the Help entry when a newer release exists."""
        self._update_action.setText(
            f"Check for Updates…  ({tag} available)")
        f = self._update_action.font()
        f.setBold(True)
        self._update_action.setFont(f)

    def set_update_check_enabled(self, on: bool) -> None:
        self._update_toggle.blockSignals(True)
        self._update_toggle.setChecked(bool(on))
        self._update_toggle.blockSignals(False)

    def _mk_menu(self, label: str, actions: list) -> QPushButton:
        btn = QPushButton(label)
        btn.setFlat(True)
        # No explicit font-size — inherit the QApplication / global QSS
        # rule so the menu buttons scale with the user-controlled base.
        btn.setStyleSheet(
            f"background:transparent; border:0; color:{theme.TEXT_1};"
            f"padding:6px 10px; border-radius:3px;"
        )
        menu = QMenu(btn)
        menu.setStyleSheet(self._menu_qss())
        for name, cb in actions:
            if name == "---":
                menu.addSeparator(); continue
            # Allow nesting: if ``cb`` is a QMenu, attach it as a submenu so
            # File → Open Recent shows the right carat and expands on hover.
            if isinstance(cb, QMenu):
                cb.setTitle(name)
                menu.addMenu(cb)
                continue
            # A retained QAction (owned by the caller, e.g. so the app
            # can retitle "Check for Updates…" when one is available)
            if isinstance(cb, QAction):
                menu.addAction(cb)
                continue
            act = QAction(name, self)
            if cb is not None:
                act.triggered.connect(cb)
            menu.addAction(act)
        btn.setMenu(menu)
        return btn

    # ------------------------------------------------------------------
    # Recent Projects submenu — repopulated on demand by the app.
    def set_recent_projects(self, paths: list[str]) -> None:
        """Repopulate the Open Recent submenu from ``paths``.

        ``paths`` is a list of absolute .lgproj file paths in most-recent-
        first order.  Each entry becomes one action labeled with the
        basename; the full path is shown as the tooltip.  Clicking an
        entry emits :pyattr:`open_recent_requested` with the path so the
        app can route to its existing ``_open_project`` flow.
        """
        self._render_recent_projects(paths)

    def _render_recent_projects(self, paths: list[str]) -> None:
        import os as _os
        self._recent_menu.clear()
        if not paths:
            empty = QAction("(no recent projects)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for p in paths:
            label = _os.path.basename(p) or p
            act = QAction(label, self)
            act.setToolTip(p)
            # Bind ``p`` into the lambda's defaults so each entry emits its
            # own path, not the loop's final value.
            act.triggered.connect(
                lambda _checked=False, path=p: self.open_recent_requested.emit(path)
            )
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clr = QAction("Clear Recent Projects", self)
        clr.triggered.connect(self.clear_recent_requested.emit)
        self._recent_menu.addAction(clr)

    def _refresh_s(self, s: float) -> None:
        lat = self._state.lattice
        total = sum(e.length for e in lat.elements) if lat else 0
        self._s_label.setText(f"s = {s:.0f} / {total:.0f} mm")
        if self._slider.maximum() > 0:
            self._slider.blockSignals(True)
            self._slider.setValue(int(s))
            self._slider.blockSignals(False)

    def _refresh_lattice(self, lattice) -> None:
        total = int(sum(e.length for e in lattice.elements)) if lattice else 0
        self._slider.setRange(0, max(total, 1))
        self._refresh_s(self._state.s_cursor)
        # Run buttons key off lattice presence too.
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        """Run buttons need BOTH an idle app AND a loaded lattice —
        they used to be enabled at startup with nothing to run, and the
        click just bounced off a warning dialog."""
        can_run = (not self._state.running
                   and self._state.lattice is not None)
        self._run_env_btn.setEnabled(can_run)
        self._run_mp_btn.setEnabled(can_run)
        tip = "Load a lattice first" if self._state.lattice is None else ""
        self._run_env_btn.setToolTip(tip or "Run the envelope solver (Ctrl+R)")
        self._run_mp_btn.setToolTip(
            tip or "Run the multi-particle tracker (Ctrl+Shift+R)")

    def _refresh_run_buttons(self, running: bool) -> None:
        self._update_run_enabled()
        # Stop button is only meaningful while a simulation is in flight.
        self._stop_btn.setEnabled(running)
        if not running and self._live_s_text is not None:
            # Run over — re-render the label without the "· N%" suffix
            # (the last live tick would otherwise leave e.g. "· 99%"
            # frozen on screen) and drop the live state so user scrubs
            # via _refresh_s start clean.
            self._run_pct = None
            self._s_label.setText(self._live_s_text)
            self._live_s_text = None

    def _on_slider(self, v: int) -> None:
        self._state.set_s_cursor(float(v))
