"""Beam loss map: loss histogram (left axis) + transmission curve (right axis).

Using dual y-axes so a small transmission dip (say 100 -> 99.3 %) is
plainly visible while the loss histogram still uses its natural counts
range.  Both axes share the lattice s-coordinate on x.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

from gui.linac_gen_gui.widgets.plot_style import (
    style_plot, mkpen, COLORS,
)

_LABEL_STYLE = {"color": "#ffffff", "font-size": "12pt"}


class LossMapPlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        self._plot = pg.PlotWidget()
        style_plot(self._plot,
                   title="Loss Map",
                   xlabel=("s", "mm"),
                   ylabel=("losses per bin (%)",))
        layout.addWidget(self._plot)

        plot_item = self._plot.getPlotItem()

        # -- Secondary ViewBox for the transmission curve on the right axis
        plot_item.showAxis("right")
        right_ax = plot_item.getAxis("right")
        right_ax.setLabel("transmission (%)", **_LABEL_STYLE)
        right_ax.setPen(pg.mkPen("#cccccc", width=1.2))
        right_ax.setTextPen("#ffffff")

        self._trans_vb = pg.ViewBox()
        plot_item.scene().addItem(self._trans_vb)
        right_ax.linkToView(self._trans_vb)
        self._trans_vb.setXLink(plot_item)

        self._trans_curve = pg.PlotCurveItem(pen=mkpen("x", width=2.5))
        self._trans_vb.addItem(self._trans_curve)

        # Keep the two ViewBoxes aligned on resize.
        plot_item.vb.sigResized.connect(self._sync_view)
        self._sync_view()

        # Mini text legend so the two series are identifiable without
        # an auto-legend (which doesn't play nicely with a second VB).
        self._legend_text = pg.TextItem(
            html=("<span style='color:#ff5050; font-size:11pt'>bars: losses/bin % of N (left)</span>"
                  "&nbsp;&nbsp;&nbsp;"
                  "<span style='color:#4da6ff; font-size:11pt'>line: transmission % (right)</span>"),
            anchor=(0, 0),
        )
        self._legend_text.setParentItem(plot_item)
        self._legend_text.setPos(60, 8)

        self._bars = None
        self._empty_text = None
        self._plot.setXRange(0, 1, padding=0.02)
        self._plot.setYRange(0, 1)
        self._trans_vb.setYRange(99.0, 100.1, padding=0)

    def _sync_view(self):
        """Keep the secondary ViewBox matched to the primary plot's geometry."""
        vb = self._plot.getPlotItem().vb
        self._trans_vb.setGeometry(vb.sceneBoundingRect())
        self._trans_vb.linkedViewChanged(vb, self._trans_vb.XAxis)

    # ------------------------------------------------------------------
    def update_data(self, loss_table, s_max, n_particles, recorder=None):
        """Redraw.

        loss_table   : Beam.loss_table (structured) -- may be empty.
        s_max        : lattice total length (mm) for the x-axis.
        n_particles  : total particle count (used to convert bin counts
                       to percent-of-beam).
        recorder     : DiagnosticRecorder, optional.  If given, its
                       ``s`` / ``transmission`` arrays drive the right-axis
                       transmission curve.
        """
        n_particles = max(int(n_particles), 1)
        # Discard prior bars / annotations
        if self._bars is not None:
            self._plot.removeItem(self._bars)
            self._bars = None
        if self._empty_text is not None:
            self._plot.removeItem(self._empty_text)
            self._empty_text = None

        s_max = max(float(s_max), 1.0)
        self._plot.setXRange(0, s_max, padding=0.02)

        # ----- Transmission curve on the right axis ---------------------
        if recorder is not None and getattr(recorder, "s", None):
            trans = np.asarray(recorder.transmission, dtype=float)
            s_arr = np.asarray(recorder.s, dtype=float)
            self._trans_curve.setData(s_arr, trans)
            t_min = float(trans.min()) if trans.size else 100.0
            # Zoom the right axis so even a 0.x% dip is obvious.
            if t_min >= 99.99:
                # all ~100% - show a slim band so the line is clearly at the top
                self._trans_vb.setYRange(99.0, 100.05, padding=0)
            else:
                span = max(1.0, 100.0 - t_min)
                lo = max(0.0, t_min - 0.1 * span)
                self._trans_vb.setYRange(lo, 100.3, padding=0)
        else:
            self._trans_curve.setData([], [])
            self._trans_vb.setYRange(99.0, 100.1, padding=0)

        # ----- Loss histogram on the left axis (as % of total beam) ----
        if len(loss_table) > 0:
            s_vals = loss_table["s"]
            n_bins = max(50, int(s_max / 10))
            hist, edges = np.histogram(s_vals, bins=n_bins, range=(0, s_max))
            hist_pct = hist.astype(float) * (100.0 / n_particles)
            self._bars = pg.BarGraphItem(
                x=edges[:-1],
                height=hist_pct,
                width=edges[1] - edges[0],
                brush=COLORS["loss"],
            )
            self._plot.addItem(self._bars)
            peak_pct = float(hist_pct.max()) if hist_pct.size else 0.01
            self._plot.setYRange(0, 1.10 * max(peak_pct, 0.01), padding=0)
        else:
            t_final = (float(recorder.transmission[-1]) if recorder
                       and getattr(recorder, "transmission", None)
                       else 100.0)
            msg = f"No losses recorded  (transmission {t_final:.2f} %)"
            self._empty_text = pg.TextItem(msg, color="#ffd24d",
                                           anchor=(0.5, 0.5))
            self._empty_text.setPos(s_max / 2, 0.5)
            self._plot.addItem(self._empty_text)
            self._plot.setYRange(0, 1.0, padding=0)  # in %

    def clear_data(self):
        if self._bars is not None:
            self._plot.removeItem(self._bars)
            self._bars = None
        if self._empty_text is not None:
            self._plot.removeItem(self._empty_text)
            self._empty_text = None
        self._trans_curve.setData([], [])
