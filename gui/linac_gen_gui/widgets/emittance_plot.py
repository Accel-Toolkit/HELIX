"""Emittance evolution: emit_x(s), emit_y(s), emit_z(s), emit_t(s) — mm.mrad.

emit_t is the 4-D transverse coupling-invariant emittance √det(σ_4D),
equivalent to TraceWin's εt.  In an uncoupled lattice it equals
emit_x · emit_y; under solenoid coupling it stays smooth while the 2-D
projections wobble.

The longitudinal emittance is natively in deg.MeV; the recorder /
envelope results also store it converted to mm.mrad (via the local
reference-particle Jacobian) so all curves share a single y-axis.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

from gui.linac_gen_gui.widgets.plot_style import (
    style_plot, mkpen, styled_legend,
)


class EmittancePlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self._plot = pg.PlotWidget()
        style_plot(self._plot,
                   title="Emittance Evolution",
                   xlabel=("s", "mm"), ylabel=("emittance", "mm.mrad"))
        styled_legend(self._plot)
        layout.addWidget(self._plot)
        self._curve_x = self._plot.plot(pen=mkpen("x"), name="emit_x")
        self._curve_y = self._plot.plot(pen=mkpen("y"), name="emit_y")
        self._curve_z = self._plot.plot(pen=mkpen("z"),
                                        name="emit_z (mm.mrad)")
        # 4-D transverse coupling-invariant emittance (= TraceWin εt).
        self._curve_t = self._plot.plot(pen=mkpen("t"), name="emit_t (4D)")

    def update_data(self, results):
        s = np.array(results.s)
        self._curve_x.setData(s, np.array(results.emit_x))
        self._curve_y.setData(s, np.array(results.emit_y))
        # Prefer the mm.mrad-converted longitudinal emittance so all curves
        # share units.  Fall back to raw deg.MeV for older recorders.
        emit_z_mm = getattr(results, 'emit_z_mmmrad', None) or getattr(results, 'emit_z', None)
        if emit_z_mm:
            self._curve_z.setData(s, np.array(emit_z_mm))
        # 4-D transverse coupling-invariant emittance.  Older recorders
        # may not populate emit_4d — skip the curve in that case.
        emit_4d = getattr(results, 'emit_4d', None)
        if emit_4d:
            self._curve_t.setData(s, np.array(emit_4d))
        else:
            self._curve_t.setData([], [])

    def clear_data(self):
        self._curve_x.setData([], [])
        self._curve_y.setData([], [])
        self._curve_z.setData([], [])
        self._curve_t.setData([], [])
