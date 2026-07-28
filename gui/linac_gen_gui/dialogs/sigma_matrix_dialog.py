"""Sigma matrix viewer dialog.

Shows the full 6 x 6 beam sigma matrix at every recorded step of the
most recent simulation run (both multi-particle and envelope modes
expose this through ``results.sigma_matrix``).  The user picks a step
from the list, the 6 x 6 table updates with that step's sigma matrix,
and a short summary reports the per-plane RMS sizes and emittance.
"""
from __future__ import annotations

import math

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QGroupBox, QTextEdit, QMessageBox,
)

from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
from linac_gen.tracking.longitudinal_coords import sigma_to_tracewin_custom

SPECIES_MAP = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}

COORD_LABELS_OURS = ["x (mm)", "x' (mrad)", "y (mm)", "y' (mrad)",
                     "phi (deg)", "W (MeV)"]
COORD_LABELS_TW   = ["x (mm)", "x' (mrad)", "y (mm)", "y' (mrad)",
                     "z (m)",   "δ = Δp/p"]
COORD_LABELS = COORD_LABELS_OURS   # default; updated at runtime

_PLANE_SLICES = (("x",  0, 1),
                 ("y",  2, 3),
                 ("z",  4, 5))


def _twiss_from_block(s11: float, s12: float, s22: float) -> tuple[float, float, float]:
    """Return (emit, alpha, beta) from 2x2 sigma block entries; 0s if degenerate."""
    det = s11 * s22 - s12 * s12
    if det <= 0.0:
        return 0.0, 0.0, 0.0
    emit = math.sqrt(det)
    beta = s11 / emit
    alpha = -s12 / emit
    return emit, alpha, beta


class SigmaMatrixDialog(QDialog):
    """Browse the 6 x 6 sigma matrix at each recorded simulation step."""

    def __init__(self, parent, results, beam_config=None):
        """
        Parameters
        ----------
        results : DiagnosticRecorder | EnvelopeResults
            Must expose ``.s``, ``.sigma_matrix``, ``.element_names``, and
            ideally ``.ref_beta`` / ``.ref_gamma`` for per-step basis
            conversion.  Both recorders satisfy this post-refactor.
        beam_config : BeamConfig, optional
            Used to obtain species mass and the reference RF wavelength
            for the (z, δ) TraceWin-basis conversion.
        """
        super().__init__(parent)
        self.setWindowTitle("Sigma Matrix Viewer")
        from linac_gen_gui.interphase.scrollwrap import screen_capped
        self.setMinimumSize(*screen_capped(self, 880, 680))
        self._results = results
        self._beam_config = beam_config

        if not getattr(results, "sigma_matrix", None):
            QMessageBox.warning(parent, "No results",
                                "Run a simulation first; sigma matrix is empty.")
            # The dialog is still constructed so the user sees the empty state
            # and can close it.

        root = QVBoxLayout(self)

        header = QLabel(
            "<b>Beam Sigma Matrix (6 &times; 6)</b> at every recorded step of "
            "the last simulation run.  Pick a step from the list on the left; "
            "the table shows the corresponding sigma matrix "
            f"({'multi-particle' if hasattr(results, 'transmission') else 'envelope'} "
            "results)."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        # Basis selector
        basis_row = QHBoxLayout()
        basis_row.addWidget(QLabel("Basis:"))
        self._basis_combo = QComboBox()
        self._basis_combo.addItem("(Δφ, ΔW) — our code", userData="ours")
        self._basis_combo.addItem("(z, δ) — TraceWin",   userData="tracewin")
        self._basis_combo.currentIndexChanged.connect(
            lambda _: self._on_step_changed(self._step_list.currentRow())
        )
        basis_row.addWidget(self._basis_combo)
        basis_row.addStretch(1)
        root.addLayout(basis_row)

        # ------- Body: list on left, matrix + summary on right -----------
        body = QHBoxLayout()
        root.addLayout(body, stretch=1)

        # List of steps
        steps_group = QGroupBox("Step")
        sg_layout = QVBoxLayout(steps_group)
        self._step_list = QListWidget()
        self._populate_step_list()
        self._step_list.currentRowChanged.connect(self._on_step_changed)
        sg_layout.addWidget(self._step_list)
        steps_group.setMaximumWidth(340)
        body.addWidget(steps_group)

        # Right: matrix + summary
        right = QVBoxLayout()
        body.addLayout(right, stretch=1)

        matrix_group = QGroupBox("Sigma matrix")
        m_layout = QVBoxLayout(matrix_group)
        self._matrix_table = QTableWidget(6, 6)
        mono = QFont("Courier")
        self._matrix_table.setFont(mono)
        self._matrix_table.setHorizontalHeaderLabels(COORD_LABELS)
        self._matrix_table.setVerticalHeaderLabels(COORD_LABELS)
        self._matrix_table.horizontalHeader().setDefaultSectionSize(108)
        self._matrix_table.verticalHeader().setDefaultSectionSize(26)
        self._matrix_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        m_layout.addWidget(self._matrix_table)
        right.addWidget(matrix_group, stretch=1)

        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setFont(mono)
        self._summary.setMaximumHeight(130)
        right.addWidget(self._summary)

        # Close button
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        if getattr(results, "sigma_matrix", None):
            self._step_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    def _populate_step_list(self) -> None:
        s_arr = getattr(self._results, "s", [])
        names = getattr(self._results, "element_names", None) or []
        for i in range(len(s_arr)):
            name = names[i] if i < len(names) else f"step_{i}"
            s = s_arr[i]
            self._step_list.addItem(
                QListWidgetItem(f"[{i:4d}]   s = {s:9.2f} mm   —   {name}")
            )

    def _on_step_changed(self, row: int) -> None:
        sigmas = getattr(self._results, "sigma_matrix", None)
        if not sigmas or row < 0 or row >= len(sigmas):
            return
        S = np.asarray(sigmas[row])
        basis = self._basis_combo.currentData()
        if basis == "tracewin":
            S_disp = self._convert_to_tracewin(S, row)
            labels = COORD_LABELS_TW
        else:
            S_disp = S
            labels = COORD_LABELS_OURS
        self._matrix_table.setHorizontalHeaderLabels(labels)
        self._matrix_table.setVerticalHeaderLabels(labels)
        self._display_matrix(S_disp)
        self._display_summary(S_disp, row, basis=basis)

    def _convert_to_tracewin(self, sigma: np.ndarray, row: int) -> np.ndarray:
        """Convert Σ at ``row`` to TraceWin's (z, δ) basis.

        Uses the recorded ``ref_beta`` / ``ref_gamma`` for this step and
        the species / wavelength inferred from ``beam_config``.  Falls
        back to the full Σ unchanged if the conversion inputs are missing.
        """
        if self._beam_config is None:
            return sigma
        beta  = self._safe_get("ref_beta",  row, self._beam_config_beta_default())
        gamma = self._safe_get("ref_gamma", row, self._beam_config_gamma_default())
        mass  = SPECIES_MAP.get(self._beam_config.species, PROTON).mass
        wavelength_mm = self._beam_config_wavelength_mm()
        try:
            return sigma_to_tracewin_custom(
                sigma, beta=beta, gamma=gamma,
                mass_MeV=mass, wavelength_mm=wavelength_mm,
            )
        except ValueError:
            return sigma

    def _safe_get(self, attr: str, row: int, fallback: float) -> float:
        arr = getattr(self._results, attr, None)
        if arr is None or row >= len(arr):
            return fallback
        return float(arr[row])

    def _beam_config_beta_default(self) -> float:
        cfg = self._beam_config
        mass = SPECIES_MAP.get(cfg.species, PROTON).mass
        gamma = 1.0 + cfg.energy / mass
        return math.sqrt(max(1.0 - 1.0 / (gamma * gamma), 0.0))

    def _beam_config_gamma_default(self) -> float:
        cfg = self._beam_config
        mass = SPECIES_MAP.get(cfg.species, PROTON).mass
        return 1.0 + cfg.energy / mass

    def _beam_config_wavelength_mm(self) -> float:
        from linac_gen.core.constants import C_LIGHT
        # frequency [MHz] -> wavelength [mm]
        return C_LIGHT / (self._beam_config.frequency * 1e6) * 1000.0

    # ------------------------------------------------------------------
    def _display_matrix(self, S: np.ndarray) -> None:
        for i in range(6):
            for j in range(6):
                v = float(S[i, j])
                if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-4):
                    text = f"{v:+.4e}"
                else:
                    text = f"{v:+.6f}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                if i == j:
                    item.setForeground(Qt.GlobalColor.cyan)
                elif abs(v) < 1e-14:
                    item.setForeground(Qt.GlobalColor.darkGray)
                self._matrix_table.setItem(i, j, item)

    def _display_summary(self, S: np.ndarray, row: int,
                         basis: str = "ours") -> None:
        s_arr = getattr(self._results, "s", [])
        names = getattr(self._results, "element_names", None) or []
        s_pos = s_arr[row] if row < len(s_arr) else float("nan")
        name = names[row] if row < len(names) else "?"

        basis_label = ("(z, δ) [TraceWin basis]" if basis == "tracewin"
                       else "(Δφ, ΔW) [our code's basis]")
        if basis == "tracewin":
            long_labels = ("z", "z'", "<z z'>", "m", "(dimensionless)", "m.")
        else:
            long_labels = ("phi", "W", "<phi W>", "deg", "MeV", "deg.")

        lines = [
            f"Step {row}   s = {s_pos:.3f} mm   element = {name}",
            f"Basis: {basis_label}",
            "",
            f"sigma_x = {math.sqrt(max(S[0, 0], 0.0)):.4f} mm   "
            f"sigma_x' = {math.sqrt(max(S[1, 1], 0.0)):.4f} mrad   "
            f"<xx'> = {S[0, 1]:+.5f}",
            f"sigma_y = {math.sqrt(max(S[2, 2], 0.0)):.4f} mm   "
            f"sigma_y' = {math.sqrt(max(S[3, 3], 0.0)):.4f} mrad   "
            f"<yy'> = {S[2, 3]:+.5f}",
            f"sigma_{long_labels[0]} = {math.sqrt(max(S[4, 4], 0.0)):.4f} "
            f"{long_labels[3]}   "
            f"sigma_{long_labels[1]} = {math.sqrt(max(S[5, 5], 0.0)):.5e} "
            f"{long_labels[4]}   "
            f"{long_labels[2]} = {S[4, 5]:+.5e}",
        ]

        # Per-plane emittance / Twiss extracted from the 2x2 diagonal block
        lines.append("")
        for plane, i, j in _PLANE_SLICES:
            emit, alpha, beta = _twiss_from_block(S[i, i], S[i, j], S[j, j])
            if plane == "z":
                if basis == "tracewin":
                    unit_emit = "m"
                    unit_beta = "m"
                else:
                    unit_emit = "deg.MeV"
                    unit_beta = "deg/MeV"
            else:
                unit_emit = "mm.mrad"
                unit_beta = "mm/mrad"
            lines.append(
                f"  {plane}: emit = {emit:.4f} {unit_emit}   "
                f"alpha = {alpha:+.4f}   beta = {beta:.4f} {unit_beta}"
            )

        self._summary.setPlainText("\n".join(lines))
