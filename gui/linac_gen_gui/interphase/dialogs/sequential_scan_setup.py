"""Setup dialog for the ``sequential_scan`` matcher algorithm.

Opened from the Matching tab when the user picks ``Algorithm =
sequential_scan`` and clicks Match.  Shows every tunable element (those
with at least one ADJUST card pointing at them) with its category,
varied attributes, and ADJUST bounds; lets the user tick which to
include and configure scan settings (passes, steps, step size, reversal
criterion).

Returns ``(selected_element_names, settings_dict)`` via ``get_settings()``
after ``exec()`` returns ``QDialog.Accepted``.
"""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QRadioButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme


class SequentialScanSetupDialog(QDialog):
    """Multi-element selector + scan-settings dialog.

    Parameters
    ----------
    lattice :
        Loaded :class:`linac_gen.core.lattice.Lattice` from the GUI state.
        Used to enumerate ADJUST'd elements and their bounds.
    """

    def __init__(self, lattice, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Sequential scan setup")
        self.setMinimumWidth(720)

        # 1. Enumerate tunable elements + their ADJUST bounds.  Importing
        # here keeps the dialog importable even when the matching package
        # has heavy dependencies that the GUI shouldn't pull on import.
        from linac_gen.core.config import BeamConfig
        from linac_gen.matching.variables import (
            categorize_fieldmap, collect_variables, group_variables_by_element,
        )
        from linac_gen.elements.field_map import FieldMap
        from linac_gen.elements.field_map_3d import FieldMap3D

        try:
            variables = collect_variables(lattice, BeamConfig())
        except Exception:                                  # noqa: BLE001
            variables = []
        groups = group_variables_by_element(variables)

        # rows is a list of (element_name, category, attrs, bounds_text)
        self._rows: list[tuple[str, str, list[str], str]] = []
        for elem, elem_vars in groups.items():
            # Only show lattice elements (skip beam-config variables).
            if not hasattr(elem, "name"):
                continue
            name = str(elem.name)
            if isinstance(elem, (FieldMap, FieldMap3D)):
                cat = categorize_fieldmap(elem).capitalize()
            else:
                cat = type(elem).__name__
            attrs = [v.attr for v in elem_vars]
            bounds_parts = [f"{v.attr}:[{v.vmin:.3g}, {v.vmax:.3g}]"
                            for v in elem_vars]
            self._rows.append((name, cat, attrs, "  ".join(bounds_parts)))

        # 2. Layout
        v = QVBoxLayout(self)

        # Description strip
        hint = QLabel(
            "Sequential scan walks each selected element in lattice order, "
            "brackets each ADJUST'd parameter in a one-at-a-time sweep, and "
            "reverses direction when the user-set emittance growth criterion "
            "fires.  Tracks the best end-of-line emittance across all "
            "passes; the running-best lattice is saved on Stop or completion."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        v.addWidget(hint)

        # ---- Element table -----------------------------------------------
        pick = QGroupBox(f"Elements with ADJUST cards ({len(self._rows)} found)")
        pv = QVBoxLayout(pick)
        self._table = QTableWidget(len(self._rows), 4)
        self._table.setHorizontalHeaderLabels(
            ["Use", "Name", "Category", "Tunable attrs & bounds"])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)

        self._checkboxes: list[QCheckBox] = []
        for row_idx, (name, cat, attrs, bounds_text) in enumerate(self._rows):
            cb = QCheckBox()
            cb.setChecked(True)
            self._checkboxes.append(cb)
            # Center the checkbox in its cell via a thin wrapper.
            wrap = QWidget()
            wlay = QHBoxLayout(wrap)
            wlay.setContentsMargins(0, 0, 0, 0)
            wlay.addWidget(cb, alignment=Qt.AlignmentFlag.AlignCenter)
            self._table.setCellWidget(row_idx, 0, wrap)
            self._table.setItem(row_idx, 1, QTableWidgetItem(name))
            self._table.setItem(row_idx, 2, QTableWidgetItem(cat))
            self._table.setItem(row_idx, 3, QTableWidgetItem(bounds_text))
        self._table.setMinimumHeight(min(360, 28 * (len(self._rows) + 1)))
        pv.addWidget(self._table)

        # Selection toolbar
        sel_row = QHBoxLayout()
        for label, picker in [
            ("Select all",       lambda r: True),
            ("Select none",      lambda r: False),
            ("Solenoids only",   lambda r: r[1] == "Solenoid"),
            ("Cavities only",    lambda r: r[1] == "Cavity"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _, p=picker: self._apply_selection(p))
            sel_row.addWidget(btn)
        sel_row.addStretch(1)
        pv.addLayout(sel_row)
        v.addWidget(pick)

        # ---- Scan settings -----------------------------------------------
        settings = QGroupBox("Scan settings")
        sf = QFormLayout(settings)
        self._passes = QSpinBox()
        self._passes.setRange(1, 10)
        self._passes.setValue(2)
        self._passes.setToolTip(
            "Number of passes through the lattice.  Each pass refines "
            "the previous result.  2-3 is typical.")
        sf.addRow("Passes through lattice:", self._passes)

        self._steps = QSpinBox()
        self._steps.setRange(3, 51)
        self._steps.setSingleStep(2)
        self._steps.setValue(7)
        self._steps.setToolTip(
            "Bracket scan length per parameter.  Higher = finer scan, "
            "longer wall time.")
        sf.addRow("Steps per parameter:", self._steps)

        self._step_frac = QDoubleSpinBox()
        self._step_frac.setRange(0.01, 0.50)
        self._step_frac.setSingleStep(0.05)
        self._step_frac.setDecimals(3)
        self._step_frac.setValue(0.10)
        self._step_frac.setToolTip(
            "Per-step displacement as a fraction of the ADJUST bound width.  "
            "Default 0.10 = 10% of (vmax - vmin) per step.")
        sf.addRow("Step size (fraction of bound width):",
                  self._step_frac)

        # Reversal criterion
        rev_label = QLabel("Direction reversal:")
        rev_box = QGroupBox()
        rev_box.setFlat(True)
        rev_box.setStyleSheet("QGroupBox { border: 0; }")
        rev_v = QVBoxLayout(rev_box)
        rev_v.setContentsMargins(0, 0, 0, 0)
        self._rev_both = QRadioButton(
            "Reverse when BOTH εnx_out AND εnz_out grow above reference")
        self._rev_both.setChecked(True)
        self._rev_any = QRadioButton(
            "Reverse on ANY emit growth above reference (stricter)")
        self._rev_group = QButtonGroup(self)
        self._rev_group.addButton(self._rev_both)
        self._rev_group.addButton(self._rev_any)
        rev_v.addWidget(self._rev_both)
        rev_v.addWidget(self._rev_any)
        sf.addRow(rev_label, rev_box)

        # Reversal threshold reference -- two physical interpretations
        # of "growth above reference".  Default = "input" preserves the
        # historical behaviour for any saved projects / muscle memory.
        thresh_label = QLabel("Reversal reference ε:")
        thresh_box = QGroupBox()
        thresh_box.setFlat(True)
        thresh_box.setStyleSheet("QGroupBox { border: 0; }")
        thresh_v = QVBoxLayout(thresh_box)
        thresh_v.setContentsMargins(0, 0, 0, 0)
        self._thresh_input = QRadioButton(
            "vs INPUT beam ε  (tight; reversal fires when trial exit "
            "ε > beam input ε)")
        self._thresh_input.setChecked(True)
        self._thresh_input.setToolTip(
            "Compare trial exit emittance to the BEAM INPUT emittance "
            "(envelope's first step).  Tight criterion: for lattices "
            "with intrinsic ε growth, reversal fires on every step.")
        self._thresh_seed = QRadioButton(
            "vs SEED EXIT ε  (natural for ε minimization with intrinsic "
            "growth)")
        self._thresh_seed.setToolTip(
            "Compare trial exit emittance to the NOMINAL / unmatched "
            "lattice's exit emittance (seed run's last step).  Reversal "
            "fires only when the trial is WORSE than the unmatched "
            "baseline -- typically what you want when minimising ε.")
        self._thresh_group = QButtonGroup(self)
        self._thresh_group.addButton(self._thresh_input)
        self._thresh_group.addButton(self._thresh_seed)
        thresh_v.addWidget(self._thresh_input)
        thresh_v.addWidget(self._thresh_seed)
        sf.addRow(thresh_label, thresh_box)

        # Hard-rejection of loss-inducing steps.  Closes the ε-gaming
        # gap where killing halo lowers measured ε in alive particles.
        # Independent of MIN_TRANSMISSION (which is a soft residual);
        # this is a HARD rule -- a loss-inducing step is rolled back
        # entirely.
        self._reject_loss = QCheckBox(
            "Reject any step that causes beam loss (HARD rule)")
        self._reject_loss.setToolTip(
            "When ticked, a step whose trial transmission falls below "
            "the threshold is rolled back -- best_x is NOT updated to "
            "it -- and direction is flipped.  Requires Cost solver = "
            "Multi-particle (envelope mode doesn't track losses).")
        self._reject_loss.setChecked(False)
        sf.addRow("Beam loss:", self._reject_loss)

        self._loss_thresh = QDoubleSpinBox()
        self._loss_thresh.setRange(0.0, 100.0)
        self._loss_thresh.setDecimals(2)
        self._loss_thresh.setSingleStep(0.1)
        self._loss_thresh.setValue(100.0)
        self._loss_thresh.setSuffix(" %")
        self._loss_thresh.setToolTip(
            "Transmission floor for the rejection check.  100.0 = any "
            "drop below 100% rejects the step; 99.9 = tolerate "
            "sub-permille losses; 99.0 = tolerate up to 1% loss.")
        self._loss_thresh.setEnabled(False)
        self._reject_loss.toggled.connect(self._loss_thresh.setEnabled)
        sf.addRow("Loss-rejection threshold:", self._loss_thresh)

        v.addWidget(settings)

        # ---- Constraints summary (read-only) -----------------------------
        from linac_gen.elements.lattice_commands import (
            MinEmit4DGrowth, MinEmitGrowth, SetKeOutMin,
        )
        constraint_lines: list[str] = []
        for el in lattice.elements:
            if isinstance(el, MinEmitGrowth):
                constraint_lines.append(
                    f"• MIN_EMIT_GROWTH  {el.plane}  weight={el.weight}")
            elif isinstance(el, MinEmit4DGrowth):
                constraint_lines.append(
                    f"• MIN_EMIT_4D_GROWTH  weight={el.weight}  "
                    f"tol_4d={el.tol_4d}  tol_z={el.tol_z}")
            elif isinstance(el, SetKeOutMin):
                constraint_lines.append(
                    f"• SET_KE_OUT_MIN  E={el.energy_mev} MeV  "
                    f"weight={el.weight}")
        if constraint_lines:
            cgroup = QGroupBox("Active constraints in lattice")
            cv = QVBoxLayout(cgroup)
            for line in constraint_lines:
                lbl = QLabel(line)
                lbl.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
                cv.addWidget(lbl)
            v.addWidget(cgroup)

        # ---- OK / Cancel -------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start Scan")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def _apply_selection(self, predicate) -> None:
        for row_idx, row in enumerate(self._rows):
            self._checkboxes[row_idx].setChecked(bool(predicate(row)))

    def _on_accept(self) -> None:
        selected_names = [r[0] for r, cb
                          in zip(self._rows, self._checkboxes)
                          if cb.isChecked()]
        if not selected_names:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "No elements selected",
                "Select at least one element to scan, or click Cancel."
            )
            return
        self.accept()

    def get_settings(self) -> tuple[list[str], dict[str, Any]]:
        """Return ``(selected_element_names, settings_dict)``.

        ``settings_dict`` keys match the ``seqscan_*`` kwargs of
        ``linac_gen.matching.engine.match``.
        """
        selected = [r[0] for r, cb
                    in zip(self._rows, self._checkboxes)
                    if cb.isChecked()]
        settings = dict(
            seqscan_element_names=selected,
            seqscan_passes=int(self._passes.value()),
            seqscan_steps=int(self._steps.value()),
            seqscan_step_frac=float(self._step_frac.value()),
            seqscan_reversal=("any_grew" if self._rev_any.isChecked()
                              else "both_grew"),
            seqscan_threshold=("seed_exit" if self._thresh_seed.isChecked()
                               else "input"),
            seqscan_reject_loss=bool(self._reject_loss.isChecked()),
            seqscan_loss_threshold_pct=float(self._loss_thresh.value()),
        )
        return selected, settings
