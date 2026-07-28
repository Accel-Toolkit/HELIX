"""Failure Study tab — element failure impact analysis + fault recovery.

Targets:  pick element types + the elements to fail.
Mode:     OFF / cavity DETUNE / magnet PARTIAL.
Combo:    single / all pairs (N×N heatmap) / custom sets.
Recover:  optionally re-tune neighbouring elements to recover the beam.

The sweep runs in a worker thread (reusing the parallel scan pool); results
(criticality ranking, pair heatmap, recovery verdicts) land on
``state.failure_study_results``.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.state import AppState

_TYPE_LABELS = [("cavity", "Cavity"), ("quad", "Quad"),
                ("solenoid", "Solenoid"), ("dipole", "Dipole")]


class _FailureWorker(QThread):
    prepared = pyqtSignal(list, int)       # (element_names, total scenarios)
    progress = pyqtSignal(int, int)        # (done, total)
    scenario_done = pyqtSignal(int, object)  # (index, ScenarioImpact)
    done = pyqtSignal(object)              # FailureStudyResults
    failed = pyqtSignal(str)

    def __init__(self, lattice, beam_config, *, types, kind, amp, phase,
                 combination, custom_sets, only_names, mode, workers,
                 compensate, comp_cfg, n_compensate):
        super().__init__()
        self.setStackSize(16 * 1024 * 1024)
        self._lattice = lattice
        self._beam = beam_config
        self._types = types
        self._kind = kind
        self._amp = amp
        self._phase = phase
        self._combination = combination
        self._custom_sets = custom_sets
        self._only_names = only_names
        self._mode = mode
        self._workers = workers
        self._compensate = compensate
        self._comp_cfg = comp_cfg
        self._n_compensate = n_compensate

    def run(self):  # noqa: D401
        try:
            from linac_gen.failures import (FailureStudy, compensate,
                                            enumerate_scenarios)

            scenarios, n2c, names = enumerate_scenarios(
                self._lattice, types=self._types, kind=self._kind,
                combination=self._combination, amp_scale=self._amp,
                phase_deg=self._phase, custom_sets=self._custom_sets,
                only_names=self._only_names)
            if not scenarios:
                self.failed.emit("no failable elements match the selection")
                return

            total = len(scenarios)
            self.prepared.emit(list(names), total)
            # In-memory serial sweep on the (possibly edited) in-memory
            # lattice — element names are honoured exactly, no temp-file
            # write→reparse rename.
            study = FailureStudy(lattice=self._lattice, beam_config=self._beam,
                                 mode=self._mode)
            counter = {"n": 0}

            def _on_done(i, im):
                counter["n"] += 1
                self.progress.emit(counter["n"], total)
                self.scenario_done.emit(i, im)
                if self.isInterruptionRequested():
                    raise StopIteration("cancelled")

            results = study.run(
                scenarios, names, n2c, combination=self._combination,
                on_scenario_done=_on_done)

            # optional compensation of the worst scenarios
            comp = {}
            if self._compensate:
                for rank_i in results.ranking[:max(0, self._n_compensate)]:
                    if self.isInterruptionRequested():
                        break
                    im = results.impacts[rank_i]
                    comp[rank_i] = compensate(
                        self._lattice, self._beam, im.scenario, n2c,
                        results.baseline, self._comp_cfg)
            results.compensation = comp     # attach (dynamic attr)
            self.done.emit(results)
        except StopIteration:
            self.failed.emit("cancelled by user")
        except Exception as exc:  # noqa: BLE001
            import traceback
            import sys
            traceback.print_exc(file=sys.stderr)
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class FailureStudyTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._worker: Optional[_FailureWorker] = None
        self._custom_sets: list[list[str]] = []
        self._results = None
        self._run_combo = "single"
        self._pair_mat = None
        self._pair_names: list[str] = []
        self._pair_idx: dict[str, int] = {}
        self._seen_impacts: list = []
        self._short_map: dict[str, str] = {}
        self._build_ui()
        # Repopulate elements AND drop any custom failure sets on a lattice
        # swap — their element NAMES came from the previous lattice and are
        # stale (a "custom" run would otherwise pass names that no longer
        # exist in the new lattice).  NOTE: lattice_changed also re-fires on
        # every bus command (param edit / undo) with the SAME lattice object
        # (state re-broadcasts bus.changed), so clear only when the lattice
        # object identity actually changed — not on mere edits.
        self._known_lattice = self.state.lattice
        self.state.lattice_changed.connect(self._on_lattice_changed)
        self._populate_elements()

    def _on_lattice_changed(self, lat) -> None:
        if lat is not self._known_lattice:
            self._known_lattice = lat
            # PRUNE rather than clear: a new lattice object often carries the
            # same element names (matching Apply installs a deep copy with
            # only parameter values changed; re-opening the same .dat too) —
            # those custom sets are still valid.  Drop only sets naming
            # elements that no longer exist.
            self._prune_custom_sets(lat)
        self._populate_elements()

    def _prune_custom_sets(self, lat) -> None:
        if not self._custom_sets:
            return
        names = {getattr(e, "name", None)
                 for e in (lat.elements if lat is not None else [])}
        kept = [s for s in self._custom_sets if all(n in names for n in s)]
        if len(kept) != len(self._custom_sets):
            self._custom_sets = kept
            self._set_list.clear()
            for s in kept:
                self._set_list.addItem(" + ".join(s))

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(8)

        hint = QLabel(
            "Fail elements (off / cavity detune / magnet partial) one at a "
            "time, in pairs, or in custom sets; rank by beam impact; "
            "optionally re-tune neighbours to recover the beam.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        left.addWidget(hint)

        # ---- Targets --------------------------------------------------
        tg = QGroupBox("Targets")
        tv = QVBoxLayout(tg)
        trow = QHBoxLayout()
        self._type_checks = {}
        for tid, lbl in _TYPE_LABELS:
            cb = QCheckBox(lbl); cb.setChecked(True)
            cb.toggled.connect(self._populate_elements)
            self._type_checks[tid] = cb
            trow.addWidget(cb)
        trow.addStretch(1)
        tv.addLayout(trow)
        self._elem_list = QListWidget()
        self._elem_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self._elem_list.setMaximumHeight(150)
        tv.addWidget(QLabel("Elements (select a subset, or none = all):"))
        tv.addWidget(self._elem_list)
        left.addWidget(tg)

        # ---- Mode -----------------------------------------------------
        mg = QGroupBox("Failure mode")
        mf = QFormLayout(mg)
        self._mode = QComboBox(); self._mode.addItems(["off", "detune", "partial"])
        self._mode.currentTextChanged.connect(self._on_mode_changed)
        mf.addRow("Mode:", self._mode)
        self._amp = QDoubleSpinBox(); self._amp.setRange(0.0, 2.0)
        self._amp.setSingleStep(0.05); self._amp.setValue(0.90)
        self._amp.setToolTip("Amplitude fraction of nominal (detune/partial)")
        mf.addRow("Amplitude scale:", self._amp)
        self._phase = QDoubleSpinBox(); self._phase.setRange(-180.0, 180.0)
        self._phase.setValue(0.0)
        self._phase.setToolTip("Phase offset [deg] for cavity detune")
        mf.addRow("Phase offset [deg]:", self._phase)
        left.addWidget(mg)
        self._on_mode_changed("off")

        # ---- Combination ---------------------------------------------
        cg = QGroupBox("Combination")
        cv = QVBoxLayout(cg)
        crow = QHBoxLayout()
        self._combo = QComboBox()
        self._combo.addItems(["single", "pairs", "custom"])
        crow.addWidget(QLabel("Mode:")); crow.addWidget(self._combo)
        crow.addStretch(1)
        cv.addLayout(crow)
        cbtns = QHBoxLayout()
        add_btn = QPushButton("Add selected as set")
        add_btn.clicked.connect(self._add_custom_set)
        clr_btn = QPushButton("Clear sets")
        clr_btn.clicked.connect(self._clear_custom_sets)
        cbtns.addWidget(add_btn); cbtns.addWidget(clr_btn); cbtns.addStretch(1)
        cv.addLayout(cbtns)
        self._set_list = QListWidget(); self._set_list.setMaximumHeight(80)
        cv.addWidget(self._set_list)
        left.addWidget(cg)

        # ---- Compensation --------------------------------------------
        pg_ = QGroupBox("Fault recovery (compensation)")
        pf = QFormLayout(pg_)
        self._comp_on = QCheckBox("Re-tune neighbours to recover the beam")
        pf.addRow(self._comp_on)
        self._strategy = QComboBox()
        self._strategy.addItems(["k_out_of_n", "l_neighboring_lattices", "manual"])
        pf.addRow("Strategy:", self._strategy)
        self._k = QSpinBox(); self._k.setRange(1, 20); self._k.setValue(2)
        pf.addRow("k / l:", self._k)
        self._comp_algo = QComboBox()
        self._comp_algo.addItems(["cmaes", "least_squares", "bayesopt"])
        pf.addRow("Algorithm:", self._comp_algo)
        self._comp_cost = QComboBox(); self._comp_cost.addItems(["envelope", "mp"])
        pf.addRow("Cost solver:", self._comp_cost)
        self._n_comp = QSpinBox(); self._n_comp.setRange(1, 200); self._n_comp.setValue(5)
        pf.addRow("Compensate top-N:", self._n_comp)
        left.addWidget(pg_)

        # ---- Forward model + run -------------------------------------
        fg = QGroupBox("Run")
        fv = QFormLayout(fg)
        self._forward = QComboBox(); self._forward.addItems(["envelope", "mp"])
        self._forward.setToolTip(
            "mp = multi-particle tracking → transmission, beam loss and full "
            "emittances. envelope = fast RMS model with NO particle loss, so "
            "the T [%] and loss [%] columns stay '—'. (This is separate from "
            "the compensation Cost solver.)")
        fv.addRow("Forward model:", self._forward)
        self._workers = QSpinBox(); self._workers.setRange(0, 64)
        self._workers.setValue(max(1, (os.cpu_count() or 2) - 2))
        self._workers.setEnabled(False)
        self._workers.setToolTip(
            "The GUI sweeps in-process (serial) on the in-memory lattice so "
            "edited/unsaved element names are honoured exactly. For a parallel "
            "multi-core sweep use the CLI: python -m linac_gen failures …")
        fv.addRow("Workers (CLI only):", self._workers)
        brow = QHBoxLayout()
        self._run_btn = QPushButton("Run failure study")
        self._run_btn.clicked.connect(self._on_run)
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        brow.addWidget(self._run_btn); brow.addWidget(self._stop_btn)
        fv.addRow(brow)
        self._progress = QProgressBar()
        fv.addRow(self._progress)
        self._status = QLabel("idle")
        self._status.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        fv.addRow(self._status)
        left.addWidget(fg)
        left.addStretch(1)

        # The six stacked control groups sum to ~950px of minimum height
        # — far past a small screen.  Scroll the column instead of
        # letting Qt compress the groups into each other (the plot
        # splitter on the right must stay visible, so only this column
        # scrolls, not the whole tab).
        lw = QWidget(); lw.setLayout(left)
        from linac_gen_gui.interphase.scrollwrap import scroll_wrap
        lscroll = scroll_wrap(lw)
        lscroll.setMaximumWidth(380)
        root.addWidget(lscroll)

        # ---- Results (right) -----------------------------------------
        rsplit = QSplitter(Qt.Orientation.Vertical)
        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        rsplit.addWidget(self._table)
        self._bar = pg.PlotWidget()
        self._bar.setLabel("left", "criticality")
        self._bar.setLabel("bottom", "failed element(s) — worst first")
        self._bar.showGrid(x=False, y=True, alpha=0.25)
        _bar_font = QFont(); _bar_font.setPointSize(7)
        self._bar.getAxis("bottom").setStyle(tickFont=_bar_font,
                                             autoExpandTextSpace=True)
        rsplit.addWidget(self._bar)
        self._heat = pg.PlotWidget()
        self._heat.setLabel("bottom", "failed element j")
        self._heat.setLabel("left", "failed element i")
        _tick_font = QFont(); _tick_font.setPointSize(7)
        for _ax in ("bottom", "left"):
            self._heat.getAxis(_ax).setStyle(
                tickFont=_tick_font, autoExpandTextSpace=True)
        self._heat_img = pg.ImageItem(axisOrder="row-major")
        self._heat.addItem(self._heat_img)
        self._heat_cmap = pg.colormap.get("inferno")
        self._heat_img.setLookupTable(self._heat_cmap.getLookupTable(0.0, 1.0, 256))
        try:                                   # colorbar (pyqtgraph ≥ 0.13)
            self._heat_cbar = pg.ColorBarItem(
                colorMap=self._heat_cmap, values=(0.0, 1.0),
                interactive=False, orientation="v", width=12)
            self._heat_cbar.setImageItem(self._heat_img, insert_in=self._heat.plotItem)
        except Exception:                      # noqa: BLE001 — degrade gracefully
            self._heat_cbar = None
        self._heat_ranged = False
        rsplit.addWidget(self._heat)
        rsplit.setSizes([260, 200, 200])
        root.addWidget(rsplit, stretch=1)

    # ------------------------------------------------------------------
    def _checked_types(self):
        return {tid for tid, cb in self._type_checks.items() if cb.isChecked()}

    def _populate_elements(self):
        self._elem_list.clear()
        if self.state.lattice is None:
            return
        from linac_gen.failures import failable_elements
        for n, lbl, _c in failable_elements(self.state.lattice,
                                            self._checked_types()):
            self._elem_list.addItem(f"{n}  [{lbl}]")

    def _on_mode_changed(self, text):
        self._amp.setEnabled(text in ("detune", "partial"))
        self._phase.setEnabled(text == "detune")
        if text == "partial":
            self._amp.setValue(0.90)
        elif text == "detune" and self._amp.value() == 0.90:
            self._amp.setValue(1.0)

    def _selected_names(self):
        return [it.text().split("  [")[0]
                for it in self._elem_list.selectedItems()]

    def _add_custom_set(self):
        names = self._selected_names()
        if len(names) >= 1:
            self._custom_sets.append(names)
            self._set_list.addItem(" + ".join(names))
            self._combo.setCurrentText("custom")

    def _clear_custom_sets(self):
        self._custom_sets = []
        self._set_list.clear()

    # ------------------------------------------------------------------
    def _on_run(self):
        if self.state.lattice is None or self.state.beam_config is None:
            QMessageBox.warning(self, "Not ready",
                                "Load a lattice and set the beam first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        # Build the worker COMPLETELY (including the potentially slow
        # deepcopy of a big lattice) before touching any button state —
        # a constructor failure used to leave Run disabled / Stop enabled
        # with no worker to stop, wedging the tab.
        try:
            from linac_gen.failures import CompensationConfig, FailureKind
            kind = FailureKind(self._mode.currentText())
            combo = self._combo.currentText()
            self._run_combo = combo
            self._pair_mat = None
            self._heat_ranged = False
            # empty selection ("none = all") must mean None, not an empty filter
            only = (self._selected_names() or None) if combo != "custom" else None
            comp_cfg = CompensationConfig(
                strategy=self._strategy.currentText(), k=self._k.value(),
                l=self._k.value(), algorithm=self._comp_algo.currentText(),
                cost_solver=self._comp_cost.currentText())
            import copy
            worker = _FailureWorker(
                copy.deepcopy(self.state.lattice), copy.deepcopy(self.state.beam_config),
                types=self._checked_types(), kind=kind, amp=self._amp.value(),
                phase=self._phase.value(), combination=combo,
                custom_sets=list(self._custom_sets), only_names=only,
                mode=self._forward.currentText(), workers=self._workers.value(),
                compensate=self._comp_on.isChecked(), comp_cfg=comp_cfg,
                n_compensate=self._n_comp.value())
        except Exception as exc:
            QMessageBox.critical(self, "Failure study", str(exc))
            return
        self._worker = worker
        self._worker.prepared.connect(self._on_prepared)
        self._worker.progress.connect(self._on_progress)
        self._worker.scenario_done.connect(self._on_scenario_done)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._run_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        self._status.setText("running…"); self._progress.setValue(0)
        self._table.setRowCount(0)
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            ["scenario", "criticality", "T [%]", "εnx", "εny", "εnz",
             "ΔE [MeV]", "loss [%]"])
        self._worker.start()

    @staticmethod
    def _fmt(v, p: int = 4) -> str:
        return "" if v is None else f"{float(v):.{p}g}"

    @staticmethod
    def _loss(metrics) -> str:
        """Beam loss [%] = 100 - transmission; '—' when no loss is tracked
        (envelope mode records no transmission)."""
        t = metrics.get("transmission")
        return "—" if t is None else f"{max(0.0, 100.0 - float(t)):.2f}"

    @staticmethod
    def _dash(v, p: int = 4) -> str:
        """Like _fmt but shows '—' for None — used for transmission, which is
        not modelled in envelope mode (so the cell reads 'n/a', not blank)."""
        return "—" if v is None else f"{float(v):.{p}g}"

    # ---- heatmap ------------------------------------------------------
    @staticmethod
    def _short_labels(names) -> list:
        """Strip a common prefix up to its last separator (``FMAP_001`` →
        ``001``) so axis ticks stay short; keep full names if that collides or
        doesn't shorten."""
        import os
        import re
        names = list(names)
        if len(names) < 2:
            return names
        cp = os.path.commonprefix(names)
        m = re.search(r"^.*[_\-.]", cp)          # cut at the last separator
        prefix = m.group(0) if m else ""
        if len(prefix) >= 2:
            short = [n[len(prefix):] or n for n in names]
            if len(set(short)) == len(short):    # still unique → safe
                return short
        return names

    @staticmethod
    def _thinned_ticks(labels, step: int):
        """pyqtgraph tick spec showing every ``step``-th label (others blank)."""
        n = len(labels)
        return [[(i + 0.5, labels[i]) for i in range(0, n, step)]]

    def _clear_heatmap(self):
        self._heat_img.clear()
        self._heat.getAxis("bottom").setTicks(None)
        self._heat.getAxis("left").setTicks(None)
        self._heat.setTitle("")
        self._heat_ranged = False

    def _render_heatmap(self, M, names):
        """Draw the N×N pair-criticality matrix with element-name ticks, a
        colour scale, and (once) an auto-ranged, aspect-locked view. Shared by
        the live incremental fill and the final authoritative render."""
        if M is None or getattr(M, "size", 0) == 0 or not names:
            self._clear_heatmap()
            return
        N = M.shape[0]
        finite = M[np.isfinite(M)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        self._heat_img.setImage(np.nan_to_num(M, nan=lo), levels=(lo, hi))
        self._heat_img.setRect(pg.QtCore.QRectF(0, 0, N, N))
        if self._heat_cbar is not None:
            self._heat_cbar.setLevels((lo, hi))
        # short labels (strip a common prefix like "FMAP_") + thin them when
        # crowded so the tick text doesn't overlap.
        labels = self._short_labels(names)
        self._heat.getAxis("bottom").setTicks(
            self._thinned_ticks(labels, max(1, -(-N // 20))))   # ≤ ~20 across
        self._heat.getAxis("left").setTicks(
            self._thinned_ticks(labels, max(1, -(-N // 40))))   # ≤ ~40 down
        self._heat.setTitle("pair-failure criticality")
        if not self._heat_ranged:              # fit the view once
            vb = self._heat.getViewBox()
            # do NOT lock aspect: a square N×N image in the wide/short pane
            # would scale to the short side and sit as a band in the centre.
            vb.setAspectLocked(False)
            vb.setRange(xRange=(0, N), yRange=(0, N), padding=0.02)
            self._heat_ranged = True

    def _bar_label(self, im) -> str:
        """Short element/scenario label for a bar tick (FMAP_016 -> 016, pair
        -> 016+012) so a bar can be read back to the ranking table by name."""
        return "+".join(self._short_map.get(n, n)
                        for n in im.scenario.element_names)

    def _render_bar(self, impacts):
        """Criticality bar: top-15 worst impacts so far (worst on the left),
        each tick labelled with its element(s). Shared by the live incremental
        fill and the final render; sets the Y-range explicitly (a BarGraphItem
        does not auto-range the view)."""
        self._bar.clear()
        top = sorted(impacts, key=lambda im: im.criticality, reverse=True)[:15]
        ys = [float(im.criticality) for im in top]
        if not ys:
            self._bar.getAxis("bottom").setTicks(None)
            return
        xs = list(range(len(ys)))
        self._bar.addItem(pg.BarGraphItem(x=xs, height=ys, width=0.7,
                                          brush=theme.ACCENT))
        self._bar.getAxis("bottom").setTicks(
            [[(i, self._bar_label(top[i])) for i in range(len(top))]])
        ymax = max(ys) or 1.0
        self._bar.setXRange(-0.6, len(ys) - 0.4, padding=0)
        self._bar.setYRange(0.0, ymax * 1.08, padding=0)

    def _on_progress(self, done, total):
        self._progress.setValue(int(100 * done / max(1, total)))
        self._status.setText(f"running… {done}/{total}")

    def _on_prepared(self, names, total):
        """Worker has enumerated: set up the live pair matrix + count/hint."""
        self._status.setText(f"running… 0/{total}")
        self._seen_impacts = []
        self._bar.clear()
        # short element labels (FMAP_001 -> 001) for the bar x-axis ticks
        self._short_map = (dict(zip(names, self._short_labels(names)))
                           if names else {})
        if self._run_combo == "pairs" and names:
            self._pair_names = list(names)
            self._pair_idx = {n: i for i, n in enumerate(names)}
            n = len(names)
            self._pair_mat = np.full((n, n), np.nan)
            self._heat_ranged = False
            self._render_heatmap(self._pair_mat, self._pair_names)
            if total > 60:
                self._status.setText(
                    f"running… 0/{total} (pairs of {n}; slow — consider "
                    f"envelope, a subset, or the CLI --workers)")
        else:
            self._pair_mat = None
            self._clear_heatmap()

    def shutdown_begin(self) -> list:
        """App teardown: signal the failure-sweep worker (it polls
        isInterruptionRequested per scenario)."""
        w = self._worker
        if w is not None and w.isRunning():
            w.requestInterruption()
            return [w]
        return []

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._status.setText("stopping…")

    def _on_scenario_done(self, idx, im):
        r = self._table.rowCount()
        self._table.insertRow(r)
        m = im.metrics
        vals = [im.label, f"{im.criticality:.4g}",
                self._dash(m.get("transmission")),
                self._fmt(m.get("emit_nx")), self._fmt(m.get("emit_ny")),
                self._fmt(m.get("emit_nz")),
                self._fmt(im.d_energy_mev), self._loss(m)]
        for c, v in enumerate(vals):
            self._table.setItem(r, c, QTableWidgetItem(str(v)))

        # live criticality bar (top-15 so far) — updates as the sweep runs
        self._seen_impacts.append(im)
        self._render_bar(self._seen_impacts)

        # live pair-heatmap fill so long sweeps show structure as they run
        if self._run_combo == "pairs" and self._pair_mat is not None:
            fnames = im.scenario.element_names
            idxs = [self._pair_idx[f] for f in fnames if f in self._pair_idx]
            if len(idxs) == 1:
                self._pair_mat[idxs[0], idxs[0]] = im.criticality
            elif len(idxs) == 2:
                a, b = idxs
                self._pair_mat[a, b] = self._pair_mat[b, a] = im.criticality
            self._render_heatmap(self._pair_mat, self._pair_names)

    def _on_failed(self, msg):
        self._run_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        self._status.setText(msg)

    def _on_done(self, results):
        self._results = results
        self.state.failure_study_results = results
        self._run_btn.setEnabled(True); self._stop_btn.setEnabled(False)
        comp = getattr(results, "compensation", {}) or {}
        self._status.setText(
            f"done — {len(results.impacts)} scenarios"
            + (f", {sum(1 for c in comp.values() if c.recovered)}/"
               f"{len(comp)} recovered" if comp else ""))

        # rebuild the table in ranking order: failed-case metrics, recovered
        # flag, and (when compensation ran) the recovered-case metrics.
        base = ["scenario", "criticality", "T [%]", "εnx", "εny", "εnz",
                "ΔE [MeV]", "loss [%]", "recovered"]
        rec_cols = (["T(rec)", "εnx(rec)", "εny(rec)", "εnz(rec)"]
                    if comp else [])
        self._table.setColumnCount(len(base) + len(rec_cols))
        self._table.setHorizontalHeaderLabels(base + rec_cols)
        self._table.setRowCount(0)
        for rank_i in results.ranking:
            im = results.impacts[rank_i]
            m = im.metrics
            r = self._table.rowCount(); self._table.insertRow(r)
            cr = comp.get(rank_i)
            rec = ("✓" if (cr and cr.recovered) else ("✗" if cr else ""))
            vals = [im.label, f"{im.criticality:.4g}",
                    self._dash(m.get("transmission")),
                    self._fmt(m.get("emit_nx")), self._fmt(m.get("emit_ny")),
                    self._fmt(m.get("emit_nz")),
                    self._fmt(im.d_energy_mev), self._loss(m), rec]
            if comp:
                ma = (cr.metrics_after if cr else {}) or {}
                vals += [self._dash(ma.get("transmission")),
                         self._fmt(ma.get("emit_nx")),
                         self._fmt(ma.get("emit_ny")),
                         self._fmt(ma.get("emit_nz"))]
            for c, v in enumerate(vals):
                self._table.setItem(r, c, QTableWidgetItem(str(v)))
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

        # criticality bar — final authoritative render (top 15, worst on left)
        self._render_bar(results.impacts)

        # pair heatmap (authoritative final render from the full matrix)
        if results.pair_matrix is not None:
            self._render_heatmap(results.pair_matrix, results.element_names)
        else:
            self._clear_heatmap()
