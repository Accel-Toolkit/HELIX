"""Multi-objective design dialog — explore the Pareto trade-off surface
between competing objectives over the lattice's ADJUST knobs.

Pick >=2 objectives, an algorithm (NSGA-II genetic, or qNEHVI Bayesian
multi-objective), run it in a background thread, and inspect the Pareto
front as a scatter + table.  "Apply knee point" writes the balanced
design back into the lattice (and flags the project dirty, like the
matcher's Apply).
"""
from __future__ import annotations

import copy

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QGridLayout, QMessageBox,
)

from linac_gen_gui.interphase import theme


class _MoWorker(QThread):
    finished_ok = pyqtSignal(object)     # ParetoResult
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)           # evals done

    def __init__(self, lattice, beam_cfg, objectives, *, algorithm,
                 cost_solver, mp_n_particles, space_charge,
                 pop_size, n_gen, parent=None):
        super().__init__(parent)
        self.setStackSize(16 * 1024 * 1024)
        self._lattice = lattice
        self._beam_cfg = beam_cfg
        self._objectives = objectives
        self._algorithm = algorithm
        self._cost_solver = cost_solver
        self._mp_n_particles = mp_n_particles
        self._space_charge = space_charge
        self._pop_size = pop_size
        self._n_gen = n_gen

    def run(self):  # noqa: D401
        try:
            from linac_gen.matching.multiobjective import pareto_optimize

            def cb(done, _total):
                if self.isInterruptionRequested():
                    raise StopIteration("cancelled")
                self.progress.emit(int(done))

            try:
                res = pareto_optimize(
                    self._lattice, self._beam_cfg, self._objectives,
                    algorithm=self._algorithm,
                    space_charge=self._space_charge,
                    cost_solver=self._cost_solver,
                    mp_n_particles=self._mp_n_particles,
                    pop_size=self._pop_size, n_gen=self._n_gen, seed=0,
                    callback=cb,
                )
            except StopIteration:
                self.failed.emit("cancelled by user")
                return
            self.finished_ok.emit(res)
        except Exception as exc:  # noqa: BLE001
            import traceback
            import sys
            traceback.print_exc(file=sys.stderr)
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MultiObjectiveDialog(QDialog):
    def __init__(self, state, beam_tab, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-objective design — Pareto front")
        self.setModal(False)
        self.setStyleSheet(f"QDialog {{ background:{theme.BG_0}; }}")
        self.resize(960, 640)
        self._state = state
        self._beam_tab = beam_tab
        self._worker = None
        self._result = None

        from linac_gen.matching.multiobjective import OBJECTIVES
        self._obj_names = sorted(OBJECTIVES)
        self._obj_labels = {n: OBJECTIVES[n][1] for n in self._obj_names}

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        hint = QLabel(
            "Pick 2+ competing objectives (all minimised; smaller = "
            "better) and explore the Pareto trade-off over the lattice's "
            "ADJUST knobs.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        v.addWidget(hint)

        # ---- objective checklist --------------------------------------
        objbox = QGroupBox("Objectives")
        og = QGridLayout(objbox)
        self._obj_checks = {}
        for i, n in enumerate(self._obj_names):
            cb = QCheckBox(f"{n}")
            cb.setToolTip(self._obj_labels[n])
            self._obj_checks[n] = cb
            og.addWidget(cb, i // 2, i % 2)
        # sensible defaults: a classic conflicting pair
        if "emit_nz_growth" in self._obj_checks:
            self._obj_checks["emit_nz_growth"].setChecked(True)
        if "neg_exit_energy" in self._obj_checks:
            self._obj_checks["neg_exit_energy"].setChecked(True)
        v.addWidget(objbox)

        # ---- settings row ---------------------------------------------
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Algorithm:"))
        self._algo = QComboBox(); self._algo.addItems(["nsga2", "qnehvi"])
        self._algo.setToolTip(
            "nsga2 — genetic, robust, cheap forward pass.\n"
            "qnehvi — Bayesian MO, sample-efficient for expensive (mp) "
            "objectives.")
        srow.addWidget(self._algo)
        srow.addWidget(QLabel("Cost solver:"))
        self._cost = QComboBox(); self._cost.addItems(["envelope", "mp"])
        srow.addWidget(self._cost)
        srow.addWidget(QLabel("pop/init:"))
        self._pop = QSpinBox(); self._pop.setRange(4, 200); self._pop.setValue(24)
        srow.addWidget(self._pop)
        srow.addWidget(QLabel("gen/iter:"))
        self._gen = QSpinBox(); self._gen.setRange(1, 200); self._gen.setValue(15)
        srow.addWidget(self._gen)
        srow.addStretch(1)
        v.addLayout(srow)

        # ---- run / stop ------------------------------------------------
        brow = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._on_run)
        brow.addWidget(self._run_btn)
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        brow.addWidget(self._stop_btn)
        self._apply_btn = QPushButton("Apply knee point")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply_knee)
        brow.addWidget(self._apply_btn)
        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        brow.addWidget(self._status)
        brow.addStretch(1)
        v.addLayout(brow)

        # ---- plot + table ---------------------------------------------
        body = QHBoxLayout()
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._all_scatter = pg.ScatterPlotItem(
            size=6, brush=pg.mkBrush("#888888"), pen=None)
        self._front_scatter = pg.ScatterPlotItem(
            size=11, brush=pg.mkBrush("#f97316"), pen=pg.mkPen("#f97316"))
        self._knee_scatter = pg.ScatterPlotItem(
            size=16, symbol="star", brush=pg.mkBrush(theme.ACCENT),
            pen=pg.mkPen(theme.ACCENT))
        self._plot.addItem(self._all_scatter)
        self._plot.addItem(self._front_scatter)
        self._plot.addItem(self._knee_scatter)
        body.addWidget(self._plot, stretch=3)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        body.addWidget(self._table, stretch=2)
        v.addLayout(body, stretch=1)

    # ------------------------------------------------------------------
    def _selected_objectives(self):
        return [n for n in self._obj_names if self._obj_checks[n].isChecked()]

    def _on_run(self):
        if self._state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first.")
            return
        objs = self._selected_objectives()
        if len(objs) < 2:
            QMessageBox.warning(self, "Pick objectives",
                                "Select at least 2 objectives.")
            return
        try:
            cfg = self._beam_tab.get_beam_config()
            self._state.set_beam_config(cfg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Beam config error", str(exc))
            return

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._status.setText("running…")
        self._worker = _MoWorker(
            copy.deepcopy(self._state.lattice), copy.deepcopy(cfg), objs,
            algorithm=self._algo.currentText(),
            cost_solver=self._cost.currentText(),
            mp_n_particles=1000,
            space_charge=(self._cost.currentText() == "mp"),
            pop_size=int(self._pop.value()), n_gen=int(self._gen.value()),
        )
        # Bound method, not a lambda — a lambda has no receiver QObject, so
        # Qt can't drop the connection when this dialog's C++ side dies
        # (same failure class as the statusbar timer segfault).
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, n) -> None:
        try:
            self._status.setText(f"running… {n} evals")
        except RuntimeError:
            pass          # C++ label destroyed during teardown

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._status.setText("stopping…")

    def closeEvent(self, ev) -> None:                         # noqa: N802
        # Closing the window must not orphan a running Pareto search with no
        # reachable Stop button — request interruption; the worker exits at
        # the next generation boundary.  (App shutdown also collects this
        # worker via matching_tab.shutdown_begin.)
        w = getattr(self, "_worker", None)
        if w is not None and w.isRunning():
            w.requestInterruption()
        super().closeEvent(ev)

    def _on_failed(self, msg):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.setText(msg)

    def _on_done(self, res):
        self._result = res
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._apply_btn.setEnabled(len(res.pareto_F) > 0)
        self._status.setText(
            f"{res.n_eval} evals → {len(res.pareto_F)} Pareto designs")

        names = res.objective_names
        F = res.pareto_F
        allF = res.all_F
        # Plot first two objectives.
        self._plot.setLabel("bottom", names[0])
        self._plot.setLabel("left", names[1] if len(names) > 1 else names[0])
        # With >2 objectives the scatter can only show the first two; the
        # table lists every objective and the knee point is computed over
        # all of them, so flag the projection to avoid misreading the plot.
        self._plot.setTitle(
            f"axes: {names[0]} vs {names[1]} — of {len(names)} objectives; "
            f"table & knee use all" if len(names) > 2 else "")
        if allF.shape[1] >= 2:
            self._all_scatter.setData(allF[:, 0], allF[:, 1])
            self._front_scatter.setData(F[:, 0], F[:, 1])
            knee = self._knee_index(F)
            if knee is not None:
                self._knee_scatter.setData([F[knee, 0]], [F[knee, 1]])
        # Table: objective columns + variable columns.  Use the
        # link-group-deduplicated, column-aligned labels so the header
        # width matches pareto_x (followers don't get a phantom column).
        var_labels = res.column_variable_labels()
        cols = list(names) + var_labels
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(F))
        for r, (fr, xr) in enumerate(zip(F, res.pareto_x)):
            for c, val in enumerate(list(fr) + list(xr)):
                self._table.setItem(r, c,
                                    QTableWidgetItem(f"{val:.5g}"))
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

    @staticmethod
    def _knee_index(F):
        if F.shape[0] == 0:
            return None
        rng = F.max(axis=0) - F.min(axis=0)
        rng[rng == 0] = 1.0
        norm = (F - F.min(axis=0)) / rng
        return int(((norm ** 2).sum(axis=1) ** 0.5).argmin())

    def _on_apply_knee(self):
        if self._result is None or len(self._result.pareto_x) == 0:
            return
        # Prefer the table's selected row; else the knee point.
        sel = self._table.currentRow()
        if sel is None or sel < 0:
            sel = self._knee_index(self._result.pareto_F)
        x = self._result.pareto_x[sel]
        try:
            from linac_gen.matching.engine import _apply_x
            from linac_gen.matching.variables import collect_variables
            from linac_gen.matching.engine import _link_group_index
            variables = collect_variables(self._state.lattice,
                                          self._state.beam_config)
            col_for_var, _ = _link_group_index(variables)
            _apply_x(np.asarray(x, dtype=float), variables, col_for_var)
            # Optimizer output in the live lattice: route Ctrl+S through
            # Save-As, same as the matcher's Apply.
            self._state.lattice_fitted = True
            if hasattr(self._state, "bus"):
                self._state.bus.mark_dirty()
            self._state.mark_project_dirty()
            self._state.lattice_changed.emit(self._state.lattice)
            self._status.setText(
                f"Applied design row {sel} to lattice (project dirty).")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Apply error", str(exc))
