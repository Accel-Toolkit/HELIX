"""Backtrack-distribution setup dialog (Simulate → Backtrack Distribution…).

Collects everything :class:`~linac_gen_gui.interphase.workers.BacktrackWorker`
needs: the exit-beam source (the final beam of the last MP run, or an
exit-plane ``.dst`` file), the element range to walk backwards, the
field-map inverse mode, and whether space charge is undone (using the
Numerics tab's SC settings — no hidden defaults).
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QDialogButtonBox, QRadioButton, QSpinBox, QComboBox,
    QCheckBox, QLineEdit, QFileDialog,
)


class BacktrackDialog(QDialog):
    """Modal setup for a backward-tracking run.

    ``get_settings()`` returns a dict:
    ``{"source": "results"|"dst", "dst_path": str|None,
    "start": int, "end": int, "field_map_mode": "rk4"|"linear",
    "space_charge": bool, "write_dst": str|None}``.
    """

    def __init__(self, n_elements: int, has_results_beam: bool,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Backtrack Distribution")
        self.setMinimumWidth(460)

        lay = QVBoxLayout(self)

        # --- exit-beam source -----------------------------------------
        src_group = QGroupBox("Exit distribution (walked backwards)")
        src_lay = QVBoxLayout(src_group)
        self._src_results = QRadioButton("Final beam of the last "
                                         "multi-particle run")
        self._src_results.setEnabled(has_results_beam)
        self._src_results.setToolTip(
            "Undo the forward run you just made — the exact-inverse walk "
            "reconstructs its input distribution."
            if has_results_beam else
            "Run a multi-particle simulation first to enable this source.")
        self._src_dst = QRadioButton("Exit-plane .dst file (measured or "
                                     "exported)")
        (self._src_results if has_results_beam else self._src_dst
         ).setChecked(True)
        src_lay.addWidget(self._src_results)
        row = QHBoxLayout()
        row.addWidget(self._src_dst)
        self._dst_edit = QLineEdit()
        self._dst_edit.setPlaceholderText("path/to/exit.dst")
        browse = QPushButton("Browse…")

        def _pick() -> None:
            fp, _ = QFileDialog.getOpenFileName(
                self, "Exit distribution", "",
                "TraceWin distribution (*.dst);;All Files (*)")
            if fp:
                self._dst_edit.setText(fp)
                self._src_dst.setChecked(True)

        browse.clicked.connect(_pick)
        row.addWidget(self._dst_edit, stretch=1)
        row.addWidget(browse)
        src_lay.addLayout(row)
        lay.addWidget(src_group)

        # --- range + physics options -----------------------------------
        opt_group = QGroupBox("Range · Physics")
        form = QFormLayout(opt_group)
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, max(n_elements - 1, 0))
        self._start_spin.setValue(0)
        self._start_spin.setToolTip(
            "The walk reconstructs the beam at the ENTRANCE of this "
            "element (0 = lattice entrance).")
        self._end_spin = QSpinBox()
        self._end_spin.setRange(0, max(n_elements - 1, 0))
        self._end_spin.setValue(max(n_elements - 1, 0))
        self._end_spin.setToolTip(
            "The supplied distribution sits at the EXIT of this element "
            "(default: last element).")
        form.addRow("Reconstruct at element", self._start_spin)
        form.addRow("Exit beam at element", self._end_spin)

        self._inverse_combo = QComboBox()
        self._inverse_combo.addItems(["rk4", "linear"])
        self._inverse_combo.setToolTip(
            "Field-map inverse:\n"
            "  rk4    — exact closed-form undo of every integration step\n"
            "           (round trips close at ~1e-13; default).\n"
            "  linear — v1 inverted fitted matrices (~2% closure through\n"
            "           strong bunchers); kept for comparison.")
        form.addRow("Field-map inverse", self._inverse_combo)

        self._sc_check = QCheckBox("Undo space charge (Numerics-tab "
                                   "SC settings)")
        self._sc_check.setChecked(False)
        form.addRow(self._sc_check)
        lay.addWidget(opt_group)

        # --- output ------------------------------------------------------
        out_group = QGroupBox("Output (optional)")
        out_lay = QHBoxLayout(out_group)
        out_lay.addWidget(QLabel("Write reconstructed .dst"))
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("(leave empty to skip)")
        out_browse = QPushButton("Browse…")

        def _pick_out() -> None:
            fp, _ = QFileDialog.getSaveFileName(
                self, "Reconstructed distribution", "entrance.dst",
                "TraceWin distribution (*.dst)")
            if fp:
                self._out_edit.setText(fp)

        out_browse.clicked.connect(_pick_out)
        out_lay.addWidget(self._out_edit, stretch=1)
        out_lay.addWidget(out_browse)
        lay.addWidget(out_group)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    # ------------------------------------------------------------------
    def _validate_and_accept(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        if self._src_dst.isChecked():
            p = self._dst_edit.text().strip()
            if not p or not os.path.isfile(p):
                QMessageBox.warning(self, "Missing .dst",
                                    "Pick an existing exit-plane .dst "
                                    "file (or use the last MP run).")
                return
        if self._end_spin.value() < self._start_spin.value():
            QMessageBox.warning(self, "Bad range",
                                "The exit element must be at or after "
                                "the reconstruction element.")
            return
        self.accept()

    def get_settings(self) -> dict:
        return {
            "source": "dst" if self._src_dst.isChecked() else "results",
            "dst_path": (self._dst_edit.text().strip() or None),
            "start": int(self._start_spin.value()),
            "end": int(self._end_spin.value()),
            "field_map_mode": self._inverse_combo.currentText(),
            "space_charge": bool(self._sc_check.isChecked()),
            "write_dst": (self._out_edit.text().strip() or None),
        }
