"""Error Study tab — register element / beam errors and run a Monte Carlo study.

Top:    form to add an ErrorDef (pattern, parameter, distribution, σ / ½-w)
Middle: live list of registered errors (delete via context menu)
Bottom: Run button + n_seeds spinbox + status label

The run kicks off ``ErrorStudy(...).run()`` in a worker thread; results
land on ``state.error_study_results`` so the Results tab can render
ensemble plots.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget, QLineEdit, QTabWidget,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.state import AppState


# Per-target parameter menus.  TraceWin coverage + the new field-error knobs.
_QUAD_PARAMS = ["dx", "dy", "tilt_deg", "gradient_rel",
                "g3_rel", "g4_rel"]
_CAV_PARAMS = ["dx", "dy", "tilt_deg",
               "voltage_rel", "phase_offset", "frequency_offset"]
_BEND_PARAMS = ["dx", "dy", "tilt_deg", "field_rel"]
_SOL_PARAMS = ["dx", "dy", "tilt_deg", "field_rel"]
_BEAM_PARAMS = [
    "centroid_x", "centroid_y", "centroid_xp", "centroid_yp",
    "centroid_dphi", "centroid_dw",
    "emit_nx_rel", "emit_ny_rel", "emit_z_rel",
    "mismatch_x", "mismatch_y", "mismatch_z",
    "current_rel",
]


def _group_qss() -> str:
    return (
        f"QGroupBox {{ color:{theme.TEXT_2}; border:1px solid {theme.BORDER_0};"
        f" border-radius:4px; margin-top:12px; padding-top:6px; }} "
        f"QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 6px;"
        f" color:{theme.TEXT_2}; font-size:10px; letter-spacing:1px;"
        f" text-transform:uppercase; background:{theme.BG_0}; }}"
    )


# ---------------------------------------------------------------------------
class _StudyWorker(QThread):
    """Run an ErrorStudy.run() off the GUI thread."""
    progress = pyqtSignal(int, int)        # (seeds_done, total)
    done = pyqtSignal(object, bool)        # (ErrorStudyResults, stopped)
    failed = pyqtSignal(str)

    def __init__(self, study, n_seeds: int, base_seed: int = 0):
        super().__init__()
        # OpenBLAS LU/SVD workspace can blow the default macOS QThread
        # stack — same 16 MB headroom as the other numpy-heavy workers
        # (the correction solvers run np.linalg.svd on this thread).
        self.setStackSize(16 * 1024 * 1024)
        self._study = study
        self._n = n_seeds
        self._base_seed = base_seed
        # Cooperative stop — same threading.Event pattern as the
        # envelope/MP workers; also usable when run() is driven
        # synchronously in tests (requestInterruption() is a no-op on a
        # thread that was never started).
        import threading
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            self._study.n_seeds = self._n
            self._study.base_seed = self._base_seed
            results = self._study.run(
                should_stop=self._stopping,
                progress_cb=lambda i, n: self.progress.emit(i, n),
            )
            self.done.emit(results, self._stopping())
        except Exception as exc:                                # pragma: no cover
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
class ErrorStudyTab(QWidget):
    """Top-level GUI for error / misalignment Monte Carlo studies."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._element_errors: list[dict] = []
        self._beam_errors: list[dict] = []
        self._worker: Optional[_StudyWorker] = None

        self._build_ui()
        self._refresh_lists()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12); v.setSpacing(10)

        # ---- two-column form: element errors / beam errors ----------
        form_tabs = QTabWidget()
        form_tabs.setStyleSheet(self._tab_qss())
        form_tabs.addTab(self._element_form(), "Element errors")
        form_tabs.addTab(self._beam_form(), "Beam errors")
        v.addWidget(form_tabs)

        # ---- registered list ----------------------------------------
        list_box = QGroupBox("Registered errors")
        list_box.setStyleSheet(_group_qss())
        lv = QVBoxLayout(list_box); lv.setContentsMargins(8, 8, 8, 8)
        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background:{theme.BG_0}; border:1px solid"
            f" {theme.BORDER_0}; border-radius:3px; color:{theme.TEXT_1};"
            f" font-family:{theme.FONT_MONO}; font-size:12px; }}"
            f"QListWidget::item:selected {{ background:{theme.ACCENT}40;"
            f" color:{theme.TEXT_0}; }}"
        )
        lv.addWidget(self._list)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_selected)
        lv.addWidget(del_btn, alignment=Qt.AlignmentFlag.AlignRight)
        v.addWidget(list_box, stretch=1)

        # ---- orbit correction --------------------------------------
        # Optional post-error correction pass.  When enabled, the
        # per-seed lattice copy is fed through
        # ``run_correction_from_lattice`` between error injection and
        # tracking so the recorded centroid is the *corrected* orbit.
        corr_box = QGroupBox("Orbit correction")
        corr_box.setStyleSheet(_group_qss())
        cv = QFormLayout(corr_box)
        cv.setContentsMargins(8, 8, 8, 8); cv.setSpacing(6)
        # Override the macOS QFormLayout defaults (FieldsStayAtSizeHint +
        # FormAlignment=AlignHCenter) so fields stretch and labels stay
        # flush-left — matches Linux/Windows behaviour.
        cv.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        cv.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        cv.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        from PyQt6.QtWidgets import QCheckBox
        self._corr_enable = QCheckBox("Apply orbit correction after errors")
        self._corr_enable.toggled.connect(self._toggle_corr_widgets)
        cv.addRow("", self._corr_enable)
        self._corr_method = QComboBox()
        self._corr_method.addItems(["auto", "one_to_one", "svd"])
        self._corr_method.setToolTip(
            "auto: pick one-to-one when #steerers == #BPMs (clean 1:1), "
            "else SVD"
        )
        cv.addRow("Method", self._corr_method)
        self._corr_n_iter = QSpinBox(); self._corr_n_iter.setRange(1, 50)
        self._corr_n_iter.setValue(5)
        cv.addRow("n_iter", self._corr_n_iter)
        self._corr_tol = QDoubleSpinBox()
        self._corr_tol.setRange(0.001, 100.0); self._corr_tol.setDecimals(4)
        self._corr_tol.setValue(0.05)
        self._corr_tol.setSuffix(" mm")
        cv.addRow("Tolerance", self._corr_tol)
        self._corr_noise = QDoubleSpinBox()
        self._corr_noise.setRange(0.0, 10.0); self._corr_noise.setDecimals(4)
        self._corr_noise.setValue(0.0)
        self._corr_noise.setSuffix(" mm")
        cv.addRow("BPM noise", self._corr_noise)
        self._corr_targets = QCheckBox("Steer onto DIAG_POSITION targets")
        self._corr_targets.setToolTip(
            "Correct every Monte-Carlo sample onto the deck's recorded "
            "DIAG_POSITION targets (or a loaded BPM-targets file) instead "
            "of flattening to zero — TraceWin's 'matched with "
            "diagnostics' per error sample.  Target-less BPMs still "
            "steer to zero."
        )
        cv.addRow("", self._corr_targets)
        self._corr_backend = QComboBox()
        self._corr_backend.addItems(["mp", "envelope (fast)"])
        self._corr_backend.setToolTip(
            "How each per-seed correction reads the BPMs:\n"
            "  mp — track a fresh particle beam per reading (legacy; "
            "sampling noise, slower).\n"
            "  envelope (fast) — envelope-solver centroid readings: "
            "deterministic, no sampling noise, ~an order of magnitude "
            "faster per seed."
        )
        cv.addRow("Readings", self._corr_backend)
        v.addWidget(corr_box)
        # Children disabled until the box is checked.
        for w_ in (self._corr_method, self._corr_n_iter,
                   self._corr_tol, self._corr_noise,
                   self._corr_targets, self._corr_backend):
            w_.setEnabled(False)

        # ---- run controls -------------------------------------------
        run_box = QGroupBox("Run")
        run_box.setStyleSheet(_group_qss())
        rv = QHBoxLayout(run_box); rv.setContentsMargins(8, 8, 8, 8); rv.setSpacing(10)
        self._n_seeds = QSpinBox(); self._n_seeds.setRange(2, 10_000)
        self._n_seeds.setValue(50)
        rv.addWidget(QLabel("n_seeds")); rv.addWidget(self._n_seeds)
        self._base_seed = QSpinBox()
        self._base_seed.setRange(0, 1_000_000_000)
        self._base_seed.setValue(0)
        self._base_seed.setToolTip(
            "Offset added to every per-seed RNG draw so chained studies "
            "produce statistically independent ensembles. Leave at 0 for "
            "the canonical run; bump it (e.g. 1000) to extend an "
            "existing 50-seed study to seeds [1000-1049]."
        )
        rv.addWidget(QLabel("base seed")); rv.addWidget(self._base_seed)
        self._run_btn = QPushButton("Run study")
        self._run_btn.clicked.connect(self._on_run)
        rv.addWidget(self._run_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip(
            "Stop after the seed currently being tracked; completed "
            "seeds are kept as a partial ensemble."
        )
        self._stop_btn.clicked.connect(self._on_stop)
        rv.addWidget(self._stop_btn)
        self._progress = QProgressBar(); self._progress.setMaximum(100)
        self._progress.setValue(0)
        rv.addWidget(self._progress, stretch=1)
        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        rv.addWidget(self._status)
        v.addWidget(run_box)

    # ------------------------------------------------------------------
    def _toggle_corr_widgets(self, on: bool) -> None:
        for w_ in (self._corr_method, self._corr_n_iter,
                   self._corr_tol, self._corr_noise,
                   self._corr_targets, self._corr_backend):
            w_.setEnabled(on)

    # ------------------------------------------------------------------
    def _element_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w); f.setContentsMargins(8, 8, 8, 8); f.setSpacing(6)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        f.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._target = QComboBox()
        self._target.addItems(["Quadrupole", "Cavity / RFGap", "Bend / Dipole",
                                "Solenoid"])
        self._target.currentTextChanged.connect(self._refresh_param_menu)

        self._pattern = QLineEdit("QUAD_*")
        self._pattern.setToolTip(
            "fnmatch wildcard pattern matched against element names. "
            "Examples: QUAD_*, QF_*, GAP_001, *Cav*"
        )

        self._param = QComboBox()
        self._refresh_param_menu("Quadrupole")

        self._dist = QComboBox(); self._dist.addItems(["gaussian", "uniform"])

        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(0.0, 1e6); self._sigma.setDecimals(6)
        self._sigma.setValue(0.1)
        self._sigma.setToolTip("σ for gaussian, half-width for uniform")

        self._cutoff = QDoubleSpinBox()
        self._cutoff.setRange(0.0, 100.0); self._cutoff.setDecimals(2)
        self._cutoff.setValue(3.0)
        self._cutoff.setToolTip("Gaussian truncation in σ units (uniform ignores it)")

        f.addRow("Target type", self._target)
        f.addRow("Name pattern", self._pattern)
        f.addRow("Parameter", self._param)
        f.addRow("Distribution", self._dist)
        f.addRow("σ / half-width", self._sigma)
        f.addRow("Cutoff (σ)", self._cutoff)

        add_btn = QPushButton("Add element error")
        add_btn.clicked.connect(self._on_add_element_error)
        f.addRow("", add_btn)
        return w

    def _beam_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w); f.setContentsMargins(8, 8, 8, 8); f.setSpacing(6)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        f.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._b_param = QComboBox(); self._b_param.addItems(_BEAM_PARAMS)
        self._b_dist = QComboBox(); self._b_dist.addItems(["gaussian", "uniform"])
        self._b_sigma = QDoubleSpinBox()
        self._b_sigma.setRange(0.0, 1e6); self._b_sigma.setDecimals(6)
        self._b_sigma.setValue(0.1)
        self._b_cutoff = QDoubleSpinBox()
        self._b_cutoff.setRange(0.0, 100.0); self._b_cutoff.setDecimals(2)
        self._b_cutoff.setValue(3.0)

        f.addRow("Parameter", self._b_param)
        f.addRow("Distribution", self._b_dist)
        f.addRow("σ / half-width", self._b_sigma)
        f.addRow("Cutoff (σ)", self._b_cutoff)
        add_btn = QPushButton("Add beam error")
        add_btn.clicked.connect(self._on_add_beam_error)
        f.addRow("", add_btn)
        return w

    def _refresh_param_menu(self, target: str) -> None:
        params = {
            "Quadrupole": _QUAD_PARAMS,
            "Cavity / RFGap": _CAV_PARAMS,
            "Bend / Dipole": _BEND_PARAMS,
            "Solenoid": _SOL_PARAMS,
        }.get(target, _QUAD_PARAMS)
        self._param.clear(); self._param.addItems(params)
        # Default pattern hint
        hint = {"Quadrupole": "QUAD_*", "Cavity / RFGap": "GAP_*",
                "Bend / Dipole": "BEND_*", "Solenoid": "SOL_*"}
        self._pattern.setText(hint.get(target, "*"))

    def _refresh_lists(self) -> None:
        self._list.clear()
        for spec in self._element_errors:
            txt = (f"[ELEM]  pattern={spec['pattern']:<12} "
                   f"param={spec['parameter']:<18} "
                   f"dist={spec['distribution']:<8} "
                   f"σ/h={spec['sigma'] or spec['half_width']:.4g}")
            self._list.addItem(QListWidgetItem(txt))
        for spec in self._beam_errors:
            txt = (f"[BEAM]  param={spec['parameter']:<18} "
                   f"dist={spec['distribution']:<8} "
                   f"σ/h={spec['sigma'] or spec['half_width']:.4g}")
            self._list.addItem(QListWidgetItem(txt))

    # ------------------------------------------------------------------
    def _on_add_element_error(self) -> None:
        spec = {
            "pattern": self._pattern.text().strip() or "*",
            "parameter": self._param.currentText(),
            "distribution": self._dist.currentText(),
            "sigma": self._sigma.value() if self._dist.currentText() == "gaussian" else 0.0,
            "half_width": self._sigma.value() if self._dist.currentText() == "uniform" else 0.0,
            "cutoff": self._cutoff.value(),
        }
        self._element_errors.append(spec)
        self._refresh_lists()

    def _on_add_beam_error(self) -> None:
        spec = {
            "parameter": self._b_param.currentText(),
            "distribution": self._b_dist.currentText(),
            "sigma": self._b_sigma.value() if self._b_dist.currentText() == "gaussian" else 0.0,
            "half_width": self._b_sigma.value() if self._b_dist.currentText() == "uniform" else 0.0,
            "cutoff": self._b_cutoff.value(),
        }
        self._beam_errors.append(spec)
        self._refresh_lists()

    def _delete_selected(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        n_elem = len(self._element_errors)
        if row < n_elem:
            del self._element_errors[row]
        else:
            del self._beam_errors[row - n_elem]
        self._refresh_lists()

    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice",
                                "Load a lattice first (Lattice tab).")
            return
        if self.state.beam_config is None:
            QMessageBox.warning(self, "No beam",
                                "Configure a beam first (Beam tab).")
            return
        # Errors can come from three places: the form's element-error list,
        # the form's beam-error list, OR ERROR_* directives parsed from the
        # TraceWin .dat into ``lattice.errors`` / ``lattice.beam_errors``.
        # If all three are empty, refuse — there is nothing to randomise.
        lat_err = getattr(self.state.lattice, "errors", []) or []
        lat_beam_err = getattr(self.state.lattice, "beam_errors", []) or []
        if (not self._element_errors and not self._beam_errors
                and not lat_err and not lat_beam_err):
            QMessageBox.warning(self, "No errors",
                                "Add at least one error spec — either via the "
                                "form above or as TraceWin ERROR_* directives "
                                "in the .dat file.")
            return

        # Space charge — pull the canonical config from the Convergence
        # tab (wired by the app as ``sc_config_provider``).  The old code
        # read ``state.sc_config``, an attribute that was never set, so
        # every study silently ran with space charge OFF.
        cfg = self.state.beam_config
        sc_cfg = None
        provider = getattr(self, "sc_config_provider", None)
        if provider is not None:
            try:
                sc_cfg = provider(
                    float(getattr(cfg, "current", 0.0)),
                    continuous=bool(getattr(cfg, "continuous", False)),
                )
            except Exception as exc:
                QMessageBox.critical(self, "Space-charge config", str(exc))
                return

        # Build the ErrorStudy and register everything we collected.
        # Snapshot lattice + beam on the GUI thread: the study deepcopies
        # per seed on the WORKER thread, and deepcopying the live objects
        # there races any concurrent GUI edit or lattice replacement.
        import copy
        from linac_gen.errors.error_model import ErrorStudy
        study = ErrorStudy(
            lattice=copy.deepcopy(self.state.lattice),
            beam_config=copy.deepcopy(self.state.beam_config),
            n_seeds=self._n_seeds.value(),
            sc_config=sc_cfg,
            base_seed=self._base_seed.value(),
        )
        for s in self._element_errors:
            study.add_error(
                pattern=s["pattern"], parameter=s["parameter"],
                distribution=s["distribution"],
                sigma=s["sigma"], half_width=s["half_width"],
                cutoff=s["cutoff"],
            )
        for s in self._beam_errors:
            study.add_beam_error(
                parameter=s["parameter"], distribution=s["distribution"],
                sigma=s["sigma"], half_width=s["half_width"],
                cutoff=s["cutoff"],
            )

        if self._corr_enable.isChecked():
            method_text = self._corr_method.currentText()
            method = None if method_text == "auto" else method_text
            study.enable_correction(
                method=method,
                n_iter=self._corr_n_iter.value(),
                tol_mm=self._corr_tol.value(),
                bpm_noise=self._corr_noise.value(),
                targets=("deck" if self._corr_targets.isChecked()
                         else None),
                reading_backend=("envelope"
                                 if self._corr_backend.currentText()
                                 .startswith("envelope")
                                 else "mp"),
            )

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status.setText(
            f"running {self._n_seeds.value()} seeds… "
            f"(space charge {'ON' if sc_cfg is not None else 'off'})")
        self._progress.setValue(0)

        self._worker = _StudyWorker(study, self._n_seeds.value(),
                                    base_seed=self._base_seed.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def shutdown_begin(self) -> list:
        """App teardown: signal the study worker; the seed loop and the
        tracker poll the stop flag per element."""
        w = self._worker
        if w is not None and w.isRunning():
            w.request_stop()
            w.requestInterruption()
            return [w]
        return []

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.requestInterruption()
            self._stop_btn.setEnabled(False)
            self._status.setText("stopping — finishing the current seed…")

    def _on_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / max(total, 1))
        self._progress.setValue(pct)
        self._status.setText(f"seed {current}/{total} done")

    def _on_done(self, results, stopped: bool = False) -> None:
        n = getattr(results, "n_seeds", 0)
        n_req = getattr(results, "n_requested", self._n_seeds.value())
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if n > 0:
            # Partial ensembles are statistically valid (only whole seeds
            # are aggregated) — store them so the Results tab can plot.
            self.state.error_study_results = results
        if stopped:
            if n == 0:
                self._status.setText(
                    "stopped before the first seed completed — nothing stored")
            else:
                self._status.setText(
                    f"stopped — {n}/{n_req} seeds kept as a partial ensemble")
                self.state.status_message.emit(
                    f"error study stopped: {n}/{n_req} seeds — "
                    "open the Results tab.")
            return
        self._status.setText(f"done — {n} seeds, results stored on app state")
        self._progress.setValue(100)
        self.state.status_message.emit(
            f"error study finished: {n} seeds — open the Results tab."
        )

    def _on_failed(self, msg: str) -> None:
        self._status.setText(f"failed: {msg}")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Error study failed", msg)

    # ------------------------------------------------------------------
    @staticmethod
    def _tab_qss() -> str:
        return (
            f"QTabWidget::pane {{ background:{theme.BG_0};"
            f" border:1px solid {theme.BORDER_0}; border-radius:4px; top:-1px; }}"
            f"QTabBar::tab {{ background:{theme.BG_1}; color:{theme.TEXT_2};"
            f" padding:6px 14px; border:1px solid {theme.BORDER_0}; border-bottom:0;"
            f" border-top-left-radius:3px; border-top-right-radius:3px; }}"
            f"QTabBar::tab:selected {{ background:{theme.BG_0}; color:{theme.TEXT_0}; }}"
        )
