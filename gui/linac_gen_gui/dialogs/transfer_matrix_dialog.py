"""Transfer-matrix viewer dialog.

User picks a start element index and an end element index (both inclusive)
from the currently-loaded lattice; the dialog multiplies the per-element
6x6 transfer matrices -- *pure linear transport, no space-charge* -- and
shows the resulting matrix, its determinant, and simple Twiss extractions
when the result is decoupled.
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QComboBox,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QGroupBox, QTextEdit, QMessageBox,
)

from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)
from linac_gen.tracking.longitudinal_coords import matrix_to_tracewin

SPECIES_MAP = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}

COORD_LABELS_OURS = ["x (mm)", "x' (mrad)", "y (mm)", "y' (mrad)",
                     "phi (deg)", "W (MeV)"]
COORD_LABELS_TW   = ["x (mm)", "x' (mrad)", "y (mm)", "y' (mrad)",
                     "z (m)",   "δ = Δp/p"]
COORD_LABELS = COORD_LABELS_OURS   # default; updated at runtime


class TransferMatrixDialog(QDialog):
    """Dialog: pick [i, j] range over the lattice, display M_[i..j]."""

    def __init__(self, parent, lattice, beam_config):
        super().__init__(parent)
        self.setWindowTitle("Transfer Matrix Between Elements")
        from linac_gen_gui.interphase.scrollwrap import screen_capped
        self.setMinimumSize(*screen_capped(self, 900, 720))

        self._lattice = lattice
        self._beam_config = beam_config
        self._n_elements = len(lattice.elements)

        root = QVBoxLayout(self)

        header = QLabel(
            "<b>Transfer Matrix Viewer</b> — choose a start and end element; "
            "the dialog multiplies the per-element 6 × 6 matrices in order.  "
            "<b>Pure linear transport</b> — no space-charge, no aperture, "
            "no non-linear contributions."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        # --- Element list (for reference) ---------------------------------
        list_group = QGroupBox(f"Elements (0 … {self._n_elements - 1})")
        list_layout = QVBoxLayout(list_group)
        self._element_list = QListWidget()
        self._populate_element_list()
        self._element_list.setMaximumHeight(140)
        list_layout.addWidget(self._element_list)
        root.addWidget(list_group)

        # --- Range selectors ----------------------------------------------
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("From element #"))
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, max(self._n_elements - 1, 0))
        self._start_spin.setValue(0)
        self._start_spin.valueChanged.connect(self._on_range_changed)
        sel_row.addWidget(self._start_spin)

        sel_row.addWidget(QLabel("to element #"))
        self._end_spin = QSpinBox()
        self._end_spin.setRange(0, max(self._n_elements - 1, 0))
        self._end_spin.setValue(max(self._n_elements - 1, 0))
        self._end_spin.valueChanged.connect(self._on_range_changed)
        sel_row.addWidget(self._end_spin)

        self._range_label = QLabel("")
        sel_row.addWidget(self._range_label, stretch=1)

        sel_row.addWidget(QLabel("Basis:"))
        self._basis_combo = QComboBox()
        self._basis_combo.addItem("(Δφ, ΔW) — our code", userData="ours")
        self._basis_combo.addItem("(z, δ) — TraceWin",   userData="tracewin")
        self._basis_combo.currentIndexChanged.connect(self._compute)
        sel_row.addWidget(self._basis_combo)

        self._compute_btn = QPushButton("Compute")
        self._compute_btn.clicked.connect(self._compute)
        sel_row.addWidget(self._compute_btn)

        root.addLayout(sel_row)

        # --- Auto-detected period picker ---------------------------------
        # Convenience: snap (start, end) to a detected period so the
        # Twiss readout in the summary becomes "phase advance per cell"
        # without manual index-counting.
        period_row = QHBoxLayout()
        period_row.addWidget(QLabel("Auto-period:"))
        self._period_combo = QComboBox()
        self._period_combo.setMinimumWidth(280)
        self._populate_periods()
        period_row.addWidget(self._period_combo, stretch=1)
        snap_btn = QPushButton("Snap range to period")
        snap_btn.clicked.connect(self._snap_to_period)
        period_row.addWidget(snap_btn)
        root.addLayout(period_row)

        # --- Matrix display ----------------------------------------------
        matrix_group = QGroupBox("Transfer matrix M")
        m_layout = QVBoxLayout(matrix_group)
        self._matrix_table = QTableWidget(6, 6)
        mono = QFont("Courier")
        self._matrix_table.setFont(mono)
        self._matrix_table.setHorizontalHeaderLabels(COORD_LABELS)
        self._matrix_table.setVerticalHeaderLabels(COORD_LABELS)
        self._matrix_table.horizontalHeader().setDefaultSectionSize(110)
        self._matrix_table.verticalHeader().setDefaultSectionSize(26)
        self._matrix_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        m_layout.addWidget(self._matrix_table)
        root.addWidget(matrix_group, stretch=1)

        # --- Summary stats -----------------------------------------------
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setFont(mono)
        self._summary.setMaximumHeight(140)
        root.addWidget(self._summary)

        # --- Close button -------------------------------------------------
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        self._on_range_changed()
        if self._n_elements:
            self._compute()

    # ------------------------------------------------------------------
    def _populate_periods(self) -> None:
        """Run period detection once at dialog open."""
        from linac_gen.analysis.period_detect import detect_periods
        try:
            self._periods = detect_periods(self._lattice)
        except Exception:                                # noqa: BLE001
            self._periods = []
        self._period_combo.clear()
        if not self._periods:
            self._period_combo.addItem("(no periods detected)")
            return
        for p in self._periods:
            tag = {
                "lattice_card":            "[LATTICE]",
                "lattice_card_recovered":  "[LATTICE+]",
                "type_sequence":           "[auto]",
                "fallback":                "[full]",
            }.get(p.source, "[?]")
            self._period_combo.addItem(f"{tag}  {p.label}")

    def _snap_to_period(self) -> None:
        if not getattr(self, "_periods", None):
            return
        idx = self._period_combo.currentIndex()
        if idx < 0 or idx >= len(self._periods):
            return
        p = self._periods[idx]
        # SpinBoxes use inclusive end indices; the period struct uses
        # an exclusive ``inner_slice_end``.
        self._start_spin.setValue(p.start)
        self._end_spin.setValue(max(p.start, p.inner_slice_end - 1))
        self._compute()

    # ------------------------------------------------------------------
    def _populate_element_list(self) -> None:
        """Fill the element listbox with [idx type name length] entries."""
        for i, e in enumerate(self._lattice.elements):
            self._element_list.addItem(QListWidgetItem(
                f"[{i:3d}]  {type(e).__name__:12s}  "
                f"{getattr(e, 'name', '?'):14s}  L = {e.length:g} mm"
            ))

    def _on_range_changed(self) -> None:
        s = self._start_spin.value()
        e = self._end_spin.value()
        # Keep end >= start
        if e < s:
            self._end_spin.blockSignals(True)
            self._end_spin.setValue(s)
            self._end_spin.blockSignals(False)
            e = s
        n = e - s + 1
        L = sum(el.length for el in self._lattice.elements[s:e + 1])
        self._range_label.setText(
            f"  {n} element{'s' if n != 1 else ''},  total L = {L:g} mm"
        )

    # ------------------------------------------------------------------
    def _build_ref(self) -> ReferenceParticle:
        """Construct a reference particle from the beam-config panel state."""
        cfg = self._beam_config
        species = SPECIES_MAP.get(cfg.species, PROTON)
        return ReferenceParticle(
            species=species, w_kin=cfg.energy, frequency=cfg.frequency,
        )

    def _compute(self) -> None:
        if self._n_elements == 0:
            QMessageBox.warning(self, "No lattice", "No elements loaded.")
            return
        s = self._start_spin.value()
        e = self._end_spin.value()
        try:
            ref = self._build_ref()
            M = compute_transfer_matrix(self._lattice, ref, start=s, end=e)
        except Exception as exc:              # noqa: BLE001
            QMessageBox.critical(self, "Computation failed", str(exc))
            return

        # Apply basis transform if the user asked for TraceWin (z, δ).
        basis = self._basis_combo.currentData()
        if basis == "tracewin":
            labels = COORD_LABELS_TW
            M_disp = matrix_to_tracewin(M, ref)
        else:
            labels = COORD_LABELS_OURS
            M_disp = M

        self._matrix_table.setHorizontalHeaderLabels(labels)
        self._matrix_table.setVerticalHeaderLabels(labels)
        self._display_matrix(M_disp)
        self._display_summary(M_disp, s, e, basis=basis)

    # ------------------------------------------------------------------
    def _display_matrix(self, M: np.ndarray) -> None:
        for i in range(6):
            for j in range(6):
                v = M[i, j]
                # Compact numeric formatting: fixed for small, exponent for big
                if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-4):
                    text = f"{v:+.4e}"
                else:
                    text = f"{v:+.6f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                # Colour-code: diagonal = blue, off-diagonal = grey, near-zero = faint
                if i == j:
                    item.setForeground(Qt.GlobalColor.cyan)
                elif abs(v) < 1e-12:
                    item.setForeground(Qt.GlobalColor.darkGray)
                self._matrix_table.setItem(i, j, item)

    def _display_summary(self, M: np.ndarray, s: int, e: int,
                         basis: str = "ours") -> None:
        det_full = float(np.linalg.det(M))
        det_xy = float(np.linalg.det(M[:4, :4]))

        elements = self._lattice.elements[s:e + 1]
        L = sum(el.length for el in elements)

        # Energy change across the range — handy for spotting accelerating
        # sections where "phase advance" is per-cell-at-entry-energy and
        # not strictly periodic Twiss.
        try:
            from linac_gen.analysis.phase_advance import _energy_change
            ref = self._build_ref()
            w_in, w_out = _energy_change(self._lattice, ref, s, e)
            dw = w_out - w_in
        except Exception:                                # noqa: BLE001
            w_in = w_out = dw = float("nan")

        basis_label = ("(z, δ) [TraceWin basis]" if basis == "tracewin"
                       else "(Δφ, ΔW) [our code's basis]")
        lines = [
            f"Range:            elements {s} .. {e}  ({len(elements)} of "
            f"{self._n_elements})",
            f"Total arc length: {L:g} mm",
            f"Energy:           W_in = {w_in:.4f} MeV → W_out = {w_out:.4f} MeV "
            f"(ΔW = {dw:+.4f} MeV)",
            f"Basis:            {basis_label}",
            f"det(M_6x6)       = {det_full:+.6e}    "
            f"(1.0 for symplectic transport)",
            f"det(M_4x4 xy)    = {det_xy:+.6e}      "
            f"(1.0 when transverse is symplectic)",
        ]

        # Try Twiss extraction -- only valid for decoupled transverse lattices
        # and if the range is traversed as a full period.
        for plane in ("x", "y"):
            try:
                tw = compute_twiss(M, plane=plane)
                lines.append(
                    f"Twiss ({plane}): beta = {tw['beta']:+.4f} mm/mrad   "
                    f"alpha = {tw['alpha']:+.4f}   "
                    f"mu = {tw['mu']:+.3f} deg"
                )
            except ValueError as exc:
                lines.append(f"Twiss ({plane}): not extractable ({exc}).")

        self._summary.setPlainText("\n".join(lines))
