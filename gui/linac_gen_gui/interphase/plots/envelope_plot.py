"""Envelope plot widgets.

Two variants:
  * :class:`EnvelopePlot` — single plot, σ_x/σ_y/σ_φ overlaid (compact, for
    inline previews like the Lattice view and Tracking view).
  * :class:`EnvelopeTriple` — three stacked sub-plots with a shared x-axis,
    one plane each (x, y, z).  Used in the dedicated Envelope / Twiss
    workspace where the user wants separate envelopes per plane.

Both can overlay the lattice aperture as dashed boundary curves on the σ_x
and σ_y panes when ``set_lattice(lat)`` is provided.

Envelopes are drawn **symmetrically** about the beam axis: ±σ_x and ±σ_y
appear as a single filled band centred on zero (via
:class:`pyqtgraph.FillBetweenItem`).  σ_φ / σ_z stay one-sided.

The aperture overlay can be toggled via ``set_aperture_visible(bool)``;
toggling triggers an auto-range refresh so the y-axis fits whichever
curves are currently shown.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from linac_gen.analysis.aperture_profile import aperture_profile
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.plots.plot_style import (
    add_legend, curve_pen, cursor_pen, filled_curve, style_plot, CURVE_WIDTH,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _aperture_pen(rgb: str = "#94a3b8") -> pg.QtGui.QPen:
    """Dashed mid-grey pen for the aperture boundary — readable but
    visually subordinate to the σ envelope."""
    return pg.mkPen(color=rgb, width=1.2, style=Qt.PenStyle.DashLine)


def _style_pg(p: pg.PlotWidget, ylabel: str, ylabel_units: str = "") -> None:
    """Backwards-compatible wrapper around the shared style helper."""
    style_plot(p, ylabel, ylabel_units)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _make_symmetric_band(plot: pg.PlotWidget, color: str,
                         fill_alpha: int = 64,
                         name: str | None = None
                         ) -> tuple[pg.PlotDataItem, pg.PlotDataItem,
                                    pg.FillBetweenItem]:
    """Create ``±curve`` lines plus a ``FillBetweenItem`` that fills the
    full band between them.  Returns ``(curve_pos, curve_neg, fill)``.

    Caller feeds data via ``set_band_data`` below.  Setting both curves
    to the same data produces a degenerate band — use ``y_pos`` for the
    upper curve and ``-y_pos`` for the lower.
    """
    pen = pg.mkPen(color=color, width=CURVE_WIDTH)
    curve_pos = plot.plot(pen=pen, name=name)
    curve_neg = plot.plot(pen=pen)
    r, g, b = _hex_to_rgb(color)
    fill = pg.FillBetweenItem(curve_pos, curve_neg,
                              brush=pg.mkBrush(r, g, b, fill_alpha))
    plot.addItem(fill)
    return curve_pos, curve_neg, fill


def set_band_data(curve_pos: pg.PlotDataItem,
                  curve_neg: pg.PlotDataItem,
                  s: np.ndarray, y_pos: np.ndarray) -> None:
    """Set a symmetric band: ``+y_pos`` to the upper curve, ``-y_pos``
    to the lower.  ``FillBetweenItem`` handles the polygon between
    them."""
    curve_pos.setData(s,  y_pos)
    curve_neg.setData(s, -y_pos)


def _make_aperture_curves(plot: pg.PlotWidget, color: str = "#94a3b8"
                          ) -> tuple[pg.PlotDataItem, pg.PlotDataItem]:
    """Create the (positive, negative) aperture boundary curves on a
    pyqtgraph plot widget."""
    pos = plot.plot(pen=_aperture_pen(color))
    neg = plot.plot(pen=_aperture_pen(color))
    return pos, neg


def set_aperture_data(curve_pos: pg.PlotDataItem,
                      curve_neg: pg.PlotDataItem,
                      lattice,
                      *, axis: str = "x") -> None:
    """Resolve the lattice aperture profile and feed it to the two
    curves.  Pass ``lattice=None`` to clear."""
    if lattice is None:
        curve_pos.setData([], [])
        curve_neg.setData([], [])
        return
    s_mm, rx_mm, ry_mm = aperture_profile(lattice)
    if s_mm.size == 0:
        curve_pos.setData([], [])
        curve_neg.setData([], [])
        return
    r = rx_mm if axis == "x" else ry_mm
    curve_pos.setData(s_mm,  r)
    curve_neg.setData(s_mm, -r)


def _set_visible_pair(curves, visible: bool) -> None:
    for c in curves:
        c.setVisible(bool(visible))


def _set_aperture_visibility(curves: tuple[pg.PlotDataItem, pg.PlotDataItem],
                             cached_data: tuple[np.ndarray, np.ndarray] | None,
                             visible: bool) -> None:
    """Show/hide aperture curves while ensuring auto-range tightens.

    pyqtgraph's ViewBox auto-range still polls hidden items' dataBounds,
    so toggling visibility alone keeps the y-axis stretched to fit the
    aperture even when it's not shown.  We work around that by *clearing*
    the data when hidden and restoring it from ``cached_data`` when
    visible.
    """
    pos, neg = curves
    if visible:
        if cached_data is not None:
            s_mm, r_mm = cached_data
            pos.setData(s_mm,  r_mm)
            neg.setData(s_mm, -r_mm)
        pos.setVisible(True); neg.setVisible(True)
    else:
        pos.setData([], [])
        neg.setData([], [])
        # Keep visible=True on the items themselves; data clearing alone
        # is enough to remove them from the auto-range computation.


def _autorange(plot: pg.PlotWidget) -> None:
    """Re-fit the y-axis to whichever curves are currently visible.

    Called after toggling the aperture overlay so the plot tightens onto
    the σ envelope when the aperture is hidden, and expands to include
    the pipe boundary when it's shown.
    """
    try:
        plot.enableAutoRange(axis=pg.ViewBox.YAxis)
    except Exception:
        try:
            plot.autoRange()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
class EnvelopePlot(pg.PlotWidget):
    """Compact overlaid envelope (σ_x, σ_y, σ_φ).

    σ_x and σ_y are drawn as filled ±σ bands centred on zero; σ_φ is
    one-sided (it's a phase spread, not a transverse extent).
    """

    def __init__(self):
        super().__init__()
        _style_pg(self, "σ", "mm")
        add_legend(self)
        self._cx_pos, self._cx_neg, self._fx = _make_symmetric_band(
            self, theme.ACCENT, name="σ_x")
        self._cy_pos, self._cy_neg, self._fy = _make_symmetric_band(
            self, "#a3e635", name="σ_y")
        self._curve_z = self.plot(pen=curve_pen("#fbbf24"), name="σ_φ (deg)")
        self._ap_x = _make_aperture_curves(self)
        self._ap_y = _make_aperture_curves(self, "#7286a0")
        self._cursor = pg.InfiniteLine(pos=0, angle=90, pen=cursor_pen())
        self.addItem(self._cursor)
        self._lattice = None
        self._aperture_visible = True
        self._ap_x_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._ap_y_cache: tuple[np.ndarray, np.ndarray] | None = None

    def set_data(self, results) -> None:
        if results is None:
            for c in (self._cx_pos, self._cx_neg,
                      self._cy_pos, self._cy_neg, self._curve_z):
                c.setData([], [])
            return
        s  = np.asarray(getattr(results, "s", []), dtype=float)
        sx = np.asarray(getattr(results, "sigma_x", []), dtype=float)
        sy = np.asarray(getattr(results, "sigma_y", []), dtype=float)
        sp = np.asarray(getattr(results, "sigma_phi", []), dtype=float)
        if s.size == sx.size and s.size > 0:
            set_band_data(self._cx_pos, self._cx_neg, s, sx)
        if s.size == sy.size and s.size > 0:
            set_band_data(self._cy_pos, self._cy_neg, s, sy)
        if s.size == sp.size and s.size > 0:
            self._curve_z.setData(s, sp)

    def set_lattice(self, lattice) -> None:
        self._lattice = lattice
        # Cache (s, r) so toggling visibility doesn't require re-walking
        # the lattice and still triggers a tight auto-range when hidden.
        if lattice is None:
            self._ap_x_cache = None
            self._ap_y_cache = None
        else:
            s_mm, rx_mm, ry_mm = aperture_profile(lattice)
            if s_mm.size == 0:
                self._ap_x_cache = None
                self._ap_y_cache = None
            else:
                self._ap_x_cache = (s_mm, rx_mm)
                self._ap_y_cache = (s_mm, ry_mm)
        self.set_aperture_visible(self._aperture_visible)

    def set_aperture_visible(self, visible: bool) -> None:
        self._aperture_visible = bool(visible)
        _set_aperture_visibility(self._ap_x, self._ap_x_cache, visible)
        _set_aperture_visibility(self._ap_y, self._ap_y_cache, visible)
        _autorange(self)

    def set_s_cursor(self, s: float) -> None:
        self._cursor.setPos(s)


# --------------------------------------------------------------------------- #
class EnvelopeTriple(QWidget):
    """Three stacked plots (x / y / z) with a shared x-axis.

    Shows ±σ_x and ±σ_y as filled bands centred on zero so the σ
    envelope reads naturally against the beam pipe drawn on the same
    axes.  σ_z is shown one-sided (it's a longitudinal extent).
    """

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        self._p_x = pg.PlotWidget()
        self._p_y = pg.PlotWidget()
        self._p_z = pg.PlotWidget()
        _style_pg(self._p_x, "σ_x", "mm")
        _style_pg(self._p_y, "σ_y", "mm")
        _style_pg(self._p_z, "σ_z", "mm")

        self._p_y.setXLink(self._p_x)
        self._p_z.setXLink(self._p_x)

        # Symmetric ±σ filled bands on the transverse panes.
        self._cx_pos, self._cx_neg, self._fx = _make_symmetric_band(
            self._p_x, theme.ACCENT)
        self._cy_pos, self._cy_neg, self._fy = _make_symmetric_band(
            self._p_y, "#a3e635")
        # σ_z stays one-sided (length, not transverse position).
        self._c_z = filled_curve(self._p_z, "#fbbf24")

        self._ap_x = _make_aperture_curves(self._p_x)
        self._ap_y = _make_aperture_curves(self._p_y)

        self._cursors = []
        for p in (self._p_x, self._p_y, self._p_z):
            c = pg.InfiniteLine(pos=0, angle=90, pen=cursor_pen())
            p.addItem(c)
            self._cursors.append(c)

        for p in (self._p_x, self._p_y, self._p_z):
            v.addWidget(p, stretch=1)
        self._lattice = None
        self._aperture_visible = True
        self._ap_x_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._ap_y_cache: tuple[np.ndarray, np.ndarray] | None = None

    def set_data(self, results) -> None:
        if results is None:
            for c in (self._cx_pos, self._cx_neg,
                      self._cy_pos, self._cy_neg, self._c_z):
                c.setData([], [])
            return
        s   = np.asarray(getattr(results, "s", []), dtype=float)
        sx  = np.asarray(getattr(results, "sigma_x", []), dtype=float)
        sy  = np.asarray(getattr(results, "sigma_y", []), dtype=float)
        sphi = np.asarray(getattr(results, "sigma_phi", []), dtype=float)
        if s.size == sx.size and s.size > 0:
            set_band_data(self._cx_pos, self._cx_neg, s, sx)
        if s.size == sy.size and s.size > 0:
            set_band_data(self._cy_pos, self._cy_neg, s, sy)
        # σ_z from σ_φ using ref_beta + wavelength if recorded
        ref_beta = getattr(results, "ref_beta", None)
        try:
            from linac_gen.core.constants import C_LIGHT
            if ref_beta and len(ref_beta) == s.size and s.size > 0:
                freq_mhz = getattr(results, "ref_frequency", None)
                if freq_mhz is None:
                    raise AttributeError
                wavelength_mm = C_LIGHT / (freq_mhz * 1e6) * 1000.0
                beta_arr = np.asarray(ref_beta, dtype=float)
                sigma_z = np.asarray(sphi) * beta_arr * wavelength_mm / 360.0
                self._c_z.setData(s, sigma_z)
                self._p_z.setLabel("left", "σ_z", units="mm")
                return
        except Exception:
            pass
        if s.size == sphi.size and s.size > 0:
            self._c_z.setData(s, sphi)
            self._p_z.setLabel("left", "σ_φ", units="deg")

    def set_lattice(self, lattice) -> None:
        self._lattice = lattice
        if lattice is None:
            self._ap_x_cache = None
            self._ap_y_cache = None
        else:
            s_mm, rx_mm, ry_mm = aperture_profile(lattice)
            if s_mm.size == 0:
                self._ap_x_cache = None
                self._ap_y_cache = None
            else:
                self._ap_x_cache = (s_mm, rx_mm)
                self._ap_y_cache = (s_mm, ry_mm)
        self.set_aperture_visible(self._aperture_visible)

    def set_aperture_visible(self, visible: bool) -> None:
        self._aperture_visible = bool(visible)
        _set_aperture_visibility(self._ap_x, self._ap_x_cache, visible)
        _set_aperture_visibility(self._ap_y, self._ap_y_cache, visible)
        for p in (self._p_x, self._p_y):
            _autorange(p)

    def set_s_cursor(self, s: float) -> None:
        for c in self._cursors:
            c.setPos(s)
