"""Phase space scatter/density plot."""
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                             QLabel, QCheckBox)
import pyqtgraph as pg
import numpy as np

from gui.linac_gen_gui.widgets.plot_style import (
    style_plot, COLORS, SCATTER_SIZE,
)


class PhaseSpacePlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self._plane_combo = QComboBox()
        self._plane_combo.addItems(["x-x'", "y-y'", "phi-dW", "x-y"])
        self._plane_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._plane_combo)

        ctrl.addWidget(QLabel("Basis:"))
        self._basis_combo = QComboBox()
        self._basis_combo.addItem("(Δφ, ΔW) — our code", userData="ours")
        self._basis_combo.addItem("(z, δ) — TraceWin",   userData="tracewin")
        self._basis_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._basis_combo)

        # Bunch-train view.  An RFQ turns a DC beam into a train of
        # bunches one RF period apart; particles that slip into a
        # neighbouring bucket are drawn 360° away and the plot shows a
        # row of stripes instead of one bunch.  Folding restores the
        # single-bucket view TraceWin/Toutatis display.  Only meaningful
        # on the φ-ΔW plane, so it is disabled elsewhere.
        self._wrap_check = QCheckBox("fold φ to ±180°")
        self._wrap_check.setChecked(True)
        self._wrap_check.setToolTip(
            "Fold the phase into ONE RF period about the bunch centroid "
            "(TraceWin/Toutatis convention).\n"
            "Particles one period away are in the neighbouring bucket of "
            "the same bunch train — the same physical bunch.\n"
            "Uncheck to see the raw unwrapped phase."
        )
        self._wrap_check.stateChanged.connect(self._replot)
        ctrl.addWidget(self._wrap_check)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        self._plot = pg.PlotWidget()
        style_plot(self._plot, title="Phase Space")
        layout.addWidget(self._plot)
        self._scatter = pg.ScatterPlotItem(
            size=SCATTER_SIZE, pen=None, brush=COLORS["scatter"],
        )
        self._plot.addItem(self._scatter)

        self._particles = None
        self._ref = None  # (beta, gamma, mass_MeV, wavelength_m)

    def set_particles(self, particles):
        """Set particle data (N,6) array."""
        self._particles = particles
        self._replot()

    def set_reference(self, beta: float, gamma: float,
                      mass_MeV: float, wavelength_m: float) -> None:
        """Reference state used for (Δφ, ΔW) <-> (z, δ) conversion."""
        self._ref = (float(beta), float(gamma),
                     float(mass_MeV), float(wavelength_m))

    def _replot(self):
        if self._particles is None or len(self._particles) == 0:
            self._scatter.setData([], [])
            return
        plane = self._plane_combo.currentText()
        basis = self._basis_combo.currentData()
        col_map = {"x-x'": (0, 1), "y-y'": (2, 3), "phi-dW": (4, 5), "x-y": (0, 2)}
        i, j = col_map[plane]
        p = self._particles
        if len(p) > 50000:
            idx = np.random.default_rng(0).choice(len(p), 50000, replace=False)
            p = p[idx]

        # Fold the bunch train into one RF period (φ-ΔW plane only).
        # Uses the SAME helper as the physics moments so plot and
        # reported σ_φ / ε_z always agree.
        self._wrap_check.setEnabled(plane == "phi-dW")
        n_folded = 0
        if plane == "phi-dW" and self._wrap_check.isChecked():
            from linac_gen.diagnostics.moments import wrap_phase_column
            p, n_folded = wrap_phase_column(p)

        xi, yi = p[:, i], p[:, j]
        if plane == "phi-dW" and basis == "tracewin" and self._ref is not None:
            beta, gamma, mass, wl_m = self._ref
            if beta > 0 and wl_m > 0 and mass > 0 and gamma > 0:
                # Δφ[deg] -> z[m]: z = -Δφ · β · λ / 360
                # ΔW[MeV]  -> δ:    δ = ΔW / (β² · γ · m)
                xi = -xi * beta * wl_m / 360.0
                yi = yi / (beta * beta * gamma * mass)
                xlabel, ylabel = ("z", "m"), ("δ = Δp/p", "")
            else:
                xlabel, ylabel = ("dphi", "deg"), ("dW", "MeV")
        else:
            labels = {
                "x-x'":   (("x", "mm"),    ("x'", "mrad")),
                "y-y'":   (("y", "mm"),    ("y'", "mrad")),
                "phi-dW": (("dphi", "deg"), ("dW", "MeV")),
                "x-y":    (("x", "mm"),    ("y", "mm")),
            }
            xlabel, ylabel = labels[plane]

        self._scatter.setData(xi, yi)
        title = "Phase Space"
        if n_folded:
            title += (f"  —  {n_folded} particle"
                      f"{'s' if n_folded != 1 else ''} folded from "
                      "neighbouring RF buckets")
        style_plot(self._plot, title=title, xlabel=xlabel, ylabel=ylabel)

    def clear_data(self):
        self._scatter.setData([], [])
        self._particles = None
