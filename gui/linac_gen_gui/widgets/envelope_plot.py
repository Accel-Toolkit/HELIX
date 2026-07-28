"""RMS envelope plot: sigma_x(s), sigma_y(s) vs position."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

from gui.linac_gen_gui.widgets.plot_style import (
    style_plot, mkpen, styled_legend,
)


class EnvelopePlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self._plot = pg.PlotWidget()
        style_plot(self._plot,
                   title="RMS Envelope",
                   xlabel=("s", "mm"), ylabel=("sigma", "mm"))
        styled_legend(self._plot)
        layout.addWidget(self._plot)
        self._curve_x = self._plot.plot(pen=mkpen("x"), name="sigma_x")
        self._curve_y = self._plot.plot(pen=mkpen("y"), name="sigma_y")

    def update_data(self, results):
        """Update plot from DiagnosticRecorder or EnvelopeResults."""
        s = np.array(results.s) if hasattr(results, 's') else np.array(getattr(results, 's', []))
        sx = np.array(results.sigma_x) if hasattr(results, 'sigma_x') else np.array([])
        sy = np.array(results.sigma_y) if hasattr(results, 'sigma_y') else np.array([])
        if len(s) > 0 and len(sx) > 0:
            self._curve_x.setData(s, sx)
            self._curve_y.setData(s, sy)

    def clear_data(self):
        self._curve_x.setData([], [])
        self._curve_y.setData([], [])
