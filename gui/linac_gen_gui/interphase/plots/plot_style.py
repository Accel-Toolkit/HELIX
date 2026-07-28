"""Shared pyqtgraph styling for the Interphase plots.

Centralises colours, font sizes, curve widths, and pen configuration so
every plot widget (envelope, σ popups, emittance popup, phase-space, …)
shares the same look.
"""
from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtGui import QFont

from linac_gen_gui.interphase import theme


# Apply once per process, idempotent.  Turns on high-quality drawing and
# sets a readable dark default background.
_PG_GLOBAL_CONFIGURED = False


def configure_pyqtgraph_defaults() -> None:
    global _PG_GLOBAL_CONFIGURED
    if _PG_GLOBAL_CONFIGURED:
        return
    pg.setConfigOptions(
        antialias=True,
        useOpenGL=False,          # WSL2 Mesa can be flaky with GL; keep CPU path
        background=theme.BG_INSET,
        foreground=theme.TEXT_0,  # default text colour (ticks, legends)
    )
    _PG_GLOBAL_CONFIGURED = True


# Font sizes in pt
_AXIS_LABEL_PT = 11
_AXIS_TICK_PT  = 10
_TITLE_PT      = 12

# Default curve pen width — chunkier lines read better on dark backgrounds
CURVE_WIDTH       = 2.4
CURVE_WIDTH_BOLD  = 2.8
CURSOR_WIDTH      = 1.2

# Ordered palette used by plots that cycle colours per series
PALETTE = [
    theme.ACCENT,       # cyan
    "#a3e635",          # lime
    "#fbbf24",          # amber
    "#f472b6",          # pink
    "#a78bfa",          # violet
    "#fb923c",          # orange
]


def _html_label(text: str, units: str = "") -> str:
    """Return an HTML-formatted axis label using the brighter TEXT_0 colour.

    NB: we deliberately do NOT embed ``[units]`` here — pyqtgraph handles
    the unit string separately (via ``setLabel(..., units=...)``) so its
    auto-SI-prefix code can rescale "mm" → "µm" etc. directly instead of
    tacking on "(×0.001)" next to a frozen ``[mm]`` literal.
    """
    return (
        f"<span style='color:{theme.TEXT_0}; "
        f"font-size:{_AXIS_LABEL_PT}pt; font-weight:500;'>"
        f"{text}</span>"
    )


def style_plot(p: pg.PlotWidget, ylabel: str, yunits: str = "",
               xlabel: str = "s", xunits: str = "mm",
               title: str | None = None) -> None:
    """Apply the canonical Interphase dark style to a PlotWidget.

    Bright axis labels/ticks, chunkier grid, legend with a subtle
    background, and a consistent accent-coloured title.
    """
    configure_pyqtgraph_defaults()
    p.setBackground(theme.BG_INSET)
    p.showGrid(x=True, y=True, alpha=0.35)

    # Tick labels, axis line — set FIRST because pyqtgraph's setPen()
    # mutates the axis label's style dict, which would otherwise clobber
    # our label colour below.
    tick_font = QFont()
    tick_font.setPointSize(_AXIS_TICK_PT)
    for axis_name in ("left", "bottom"):
        ax = p.getAxis(axis_name)
        ax.setTextPen(theme.TEXT_0)
        ax.setPen(theme.BORDER_2)
        ax.setTickFont(tick_font)
        # Auto-SI-prefix misfires on composite / non-base units ("mm·mrad",
        # "deg/MeV", "mm³·mrad³") because pyqtgraph parses the first char
        # as a milli prefix and multiplies ticks by 1000.  Keep it on only
        # for clean base units (m, s, V, A, T …); switch it off otherwise.
        # Test: if the unit string contains anything other than a single
        # non-prefixed base token, disable.
        ax.enableAutoSIPrefix(False)

    # Now set the label with our bright colour — after setPen so it wins.
    label_style = dict(color=theme.TEXT_0,
                       **{"font-size": f"{_AXIS_LABEL_PT}pt"})
    p.setLabel("left", _html_label(ylabel), units=yunits or None,
               **label_style)
    p.setLabel("bottom", _html_label(xlabel), units=xunits or None,
               **label_style)

    if title:
        p.setTitle(
            f"<span style='color:{theme.ACCENT}; "
            f"font-size:{_TITLE_PT}pt; font-weight:600; letter-spacing:1px;'>"
            f"{title.upper()}</span>"
        )


def add_legend(p: pg.PlotWidget, offset: tuple[int, int] = (-10, 10)):
    """Attach a legend with a translucent dark background and bright labels."""
    legend = p.addLegend(
        offset=offset,
        labelTextColor=theme.TEXT_0,
        brush=pg.mkBrush(6, 10, 15, 220),   # BG_INSET-ish with alpha
        pen=pg.mkPen(theme.BORDER_1),
        labelTextSize=f"{_AXIS_TICK_PT}pt",
    )
    return legend


def curve_pen(color: str, width: float = CURVE_WIDTH) -> pg.mkPen:
    """Return a solid pen of the given colour + width."""
    return pg.mkPen(color, width=width)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def filled_curve(p: pg.PlotWidget, color: str,
                 width: float = CURVE_WIDTH,
                 fill_alpha: int = 64,
                 name: str | None = None):
    """Plot curve + translucent fill beneath it that does NOT hijack autoscale.

    The default ``PlotDataItem(fillLevel=0)`` includes the fill rectangle in
    the item's bounding box, so the plot's auto-range expands to y=0 even
    when the curve lives at e.g. 0.5–3.5 mm and visible variation gets
    squashed.  We work around it by recomputing ``fillLevel`` on every
    ``setData`` call to sit just below the curve's own minimum — the fill
    stays cosmetically visible but the curve's data range drives the axis.

    Returns the PlotDataItem so the caller can keep calling ``.setData(s, y)``
    exactly like a regular curve.
    """
    import numpy as np

    r, g, b = _hex_to_rgb(color)
    fill_brush = pg.mkBrush(r, g, b, fill_alpha)
    kw = {"pen": pg.mkPen(color, width=width),
          "fillLevel": 0.0,
          "brush": fill_brush}
    if name:
        kw["name"] = name
    item = p.plot(**kw)

    _orig_setData = item.setData

    def _setData(*args, **kwargs):
        _orig_setData(*args, **kwargs)
        y = kwargs.get("y")
        if y is None and len(args) >= 2:
            y = args[1]
        elif y is None and len(args) == 1:
            y = args[0]
        if y is None:
            return
        y = np.asarray(y, dtype=float)
        if y.size == 0 or not np.isfinite(y).any():
            return
        y_finite = y[np.isfinite(y)]
        y_min = float(y_finite.min())
        span  = float(y_finite.max()) - y_min
        # Fill level sits 25 % of the data span below the curve's own
        # minimum so the gradient fill is clearly visible.  The
        # ``dataBounds`` override below hides this from the ViewBox's
        # auto-range, so the y-axis still snaps to the curve's real
        # variation instead of being dragged down to include the fill.
        pad = max(span * 0.25, 1e-6)
        item.setFillLevel(y_min - pad)

    item.setData = _setData       # type: ignore[method-assign]

    # --- Hide the fill rectangle from auto-range -----------------------
    # ``PlotDataItem.dataBounds`` includes ``fillLevel`` in its y-bounds,
    # so the ViewBox would otherwise pan to include the fill and squash
    # the curve's actual variation.  Override it to return only the
    # curve's own data bounds.
    _orig_dataBounds = item.dataBounds

    def _dataBounds(ax, frac=1.0, orthoRange=None):
        bounds = _orig_dataBounds(ax, frac, orthoRange)
        if ax == 1 and bounds is not None and item.yData is not None:
            y = item.yData
            if y is not None and len(y) > 0:
                y = np.asarray(y, dtype=float)
                y = y[np.isfinite(y)]
                if y.size:
                    return (float(y.min()), float(y.max()))
        return bounds

    item.dataBounds = _dataBounds    # type: ignore[method-assign]
    return item


def cursor_pen(color: str | None = None):
    """Dashed cursor line."""
    from PyQt6.QtCore import Qt
    return pg.mkPen(color or theme.ACCENT, width=CURSOR_WIDTH,
                    style=Qt.PenStyle.DashLine)


# ---------------------------------------------------------------------------
# 2-D density / heatmap support
# ---------------------------------------------------------------------------

# Perceptually-uniform matplotlib colormap used for density plots.
# "inferno" reads as black → deep purple → red → orange → yellow → white,
# like a plasma temperature map — professional and dramatic on the
# near-black BG_INSET.  Alternatives are "magma", "viridis", "plasma",
# and "turbo"; set via the LINAC_GEN_PLOT_CMAP env var (default inferno).
import os as _os
_DENSITY_CMAP_NAME = _os.environ.get("LINAC_GEN_PLOT_CMAP", "magma")


def _density_colormap():
    """Return the active density colormap with an alpha ramp at the bottom
    end so zero-density regions stay transparent (letting BG_INSET show
    through) instead of painting the whole plot the cmap's darkest shade.
    """
    import numpy as np

    # Try the matplotlib-backed colormap first (pyqtgraph >= 0.13 ships
    # with the standard set under the "matplotlib" source).  Fall back to
    # pg.colormap.get(name) which also works for a subset.
    cmap = None
    try:
        cmap = pg.colormap.get(_DENSITY_CMAP_NAME, source="matplotlib")
    except Exception:
        try:
            cmap = pg.colormap.get(_DENSITY_CMAP_NAME)
        except Exception:
            cmap = None
    if cmap is None:
        # Ultra-safe fallback: reinstate the neon gradient.
        stops = [
            (0.00, (  0,   0,   0,   0)),
            (0.08, (  8,  32,  56, 180)),
            (0.25, ( 34, 211, 238, 230)),
            (0.55, (167, 139, 250, 255)),
            (0.80, (244, 114, 182, 255)),
            (1.00, (255, 240, 220, 255)),
        ]
        pos = np.array([s[0] for s in stops], dtype=float)
        col = np.array([s[1] for s in stops], dtype=np.ubyte)
        return pg.ColorMap(pos, col)

    # Sample the colormap at 256 stops and overlay an alpha ramp that
    # makes the first ~8 % fully transparent so the near-black BG_INSET
    # shows through in empty regions — otherwise the cmap's darkest
    # shade paints over the whole plot, which looks muddy.
    pos = np.linspace(0.0, 1.0, 256)
    rgba = cmap.getLookupTable(0.0, 1.0, 256, alpha=True).astype(np.int32)
    # rgba may come back as (256, 3) without alpha — normalise shape.
    if rgba.shape[1] == 3:
        alpha = np.full((256, 1), 255, dtype=np.int32)
        rgba = np.concatenate([rgba, alpha], axis=1)
    ramp = np.clip((pos - 0.02) / 0.06, 0.0, 1.0)    # 0 at bin 0, 1 by ~8 %
    rgba[:, 3] = (rgba[:, 3] * ramp).astype(np.int32)
    rgba = np.clip(rgba, 0, 255).astype(np.ubyte)
    return pg.ColorMap(pos, rgba)


def density_heatmap(p: pg.PlotWidget,
                    x, y,
                    bins: int = 140,
                    sigma: float = 0.9,
                    x_range: tuple | None = None,
                    y_range: tuple | None = None,
                    log_scale: bool = True):
    """Render an (x, y) 2-D density heatmap on ``p``.

    Uses log-intensity so diffuse halo is visible simultaneously with
    the dense core.  Returns the ``pg.ImageItem`` so the caller can
    update it later without recreating it.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        img = pg.ImageItem(np.zeros((2, 2)))
        p.addItem(img)
        return img

    # Ranges: 4σ by default (captures 99.99 % of a Gaussian; clipping
    # the last-sigma halo lets the colour scale stretch over the core).
    if x_range is None:
        xm = float(np.mean(x)); xs = float(np.std(x)) or 1.0
        x_range = (xm - 4.5 * xs, xm + 4.5 * xs)
    if y_range is None:
        ym = float(np.mean(y)); ys = float(np.std(y)) or 1.0
        y_range = (ym - 4.5 * ys, ym + 4.5 * ys)

    H, xe, ye = np.histogram2d(
        x, y, bins=bins,
        range=(x_range, y_range),
    )
    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=sigma, mode="nearest")
        except ImportError:
            pass

    if log_scale:
        H = np.log1p(H)

    # ImageItem takes (rows, cols) — numpy histogram2d returns (nx, ny)
    # already in row-major (x along rows), which matches what ImageItem
    # expects when axisOrder='row-major'.
    img = pg.ImageItem(axisOrder="row-major")
    img.setImage(H.T)                          # transpose so x→horizontal
    cm = _density_colormap()
    img.setLookupTable(cm.getLookupTable(0.0, 1.0, 256))

    # Map pixel coords to data coords
    dx = (x_range[1] - x_range[0]) / bins
    dy = (y_range[1] - y_range[0]) / bins
    img.setRect(pg.QtCore.QRectF(
        x_range[0], y_range[0],
        x_range[1] - x_range[0], y_range[1] - y_range[0],
    ))

    # Draw under curves/cursors
    img.setZValue(-10)
    p.addItem(img)
    return img


class DensityPanel(pg.GraphicsLayoutWidget):
    """Self-contained phase-space density panel: title, axes, heatmap, colorbar.

    Internally a pyqtgraph GraphicsLayoutWidget with two columns: the
    PlotItem (col 0) holds axes + the ImageItem, the ColorBarItem (col 1)
    is a narrow vertical gradient labelled "log(1+N)" that reflects the
    current image levels.  The panel exposes ``set_data(x, y)`` which
    re-bins the density and updates the colorbar range.
    """

    def __init__(self, title: str = "",
                 xlabel: str = "", xunits: str = "",
                 ylabel: str = "", yunits: str = "",
                 bins: int = 140, sigma: float = 0.9):
        import numpy as np
        from PyQt6.QtGui import QFont
        configure_pyqtgraph_defaults()
        super().__init__()
        self.setBackground(theme.BG_INSET)
        self._bins = bins
        self._sigma = sigma

        # Plot + axes — pens FIRST so they don't clobber label style.
        self._plot = self.addPlot(row=0, col=0)
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=True, alpha=0.35)
        tick_font = QFont()
        tick_font.setPointSize(_AXIS_TICK_PT)
        for axname in ("left", "bottom"):
            ax = self._plot.getAxis(axname)
            ax.setTextPen(theme.TEXT_0)
            ax.setPen(theme.BORDER_2)
            ax.setTickFont(tick_font)
            ax.enableAutoSIPrefix(False)
        label_style = dict(color=theme.TEXT_0,
                           **{"font-size": f"{_AXIS_LABEL_PT}pt"})
        self._plot.setLabel("left",   _html_label(ylabel),
                            units=yunits or None, **label_style)
        self._plot.setLabel("bottom", _html_label(xlabel),
                            units=xunits or None, **label_style)

        if title:
            self._plot.setTitle(
                f"<span style='color:{theme.ACCENT}; "
                f"font-size:{_TITLE_PT}pt; font-weight:600; letter-spacing:1px;'>"
                f"{title.upper()}</span>"
            )

        # Image item (starts empty; ``set_data`` fills it in)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._image.setImage(np.zeros((2, 2)))
        self._plot.addItem(self._image)

        # Shared colormap + colorbar
        self._cmap = _density_colormap()
        self._image.setLookupTable(self._cmap.getLookupTable(0.0, 1.0, 256))

        # ColorBarItem — pyqtgraph ≥ 0.13 provides a dedicated item that
        # paints a gradient strip and shares levels with an ImageItem.
        self._cbar = pg.ColorBarItem(
            colorMap=self._cmap,
            values=(0.0, 1.0),
            label=f"<span style='color:{theme.TEXT_2}; font-size:{_AXIS_TICK_PT}pt;'>"
                  f"log₁₀(1+N)</span>",
            interactive=False,
            orientation="v",
            width=14,
        )
        self._cbar.setFixedWidth(48)
        self._cbar.getAxis("right").setTextPen(theme.TEXT_0)
        self._cbar.getAxis("right").setPen(theme.BORDER_2)
        self._cbar.getAxis("right").setTickFont(tick_font)
        # Attach the bar to the image so levels/cmap stay in sync.
        self._cbar.setImageItem(self._image, insert_in=self._plot)

    # ------------------------------------------------------------------
    @property
    def plot(self):
        """Exposed so callers can tweak labels (e.g. basis switch)."""
        return self._plot

    def set_scatter(self, x, y, c=None, *,
                    cmap_name: str = "viridis", point_size: int = 3,
                    label: str = "") -> None:
        """Switch the panel into scatter mode (per-particle dots).

        ``c`` is a 1-D array of the same length as ``x`` / ``y`` used to
        colour each point.  If ``None`` every point gets the same accent
        colour (useful for a "no color-by" toggle).  The histogram image
        is hidden and a :class:`pyqtgraph.ScatterPlotItem` takes its
        place; the colorbar is re-labelled with ``label``.
        """
        import numpy as np
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        if x.size == 0:
            # Empty data — clear and bail.
            self._image.setImage(np.zeros((2, 2)))
            if getattr(self, "_scatter", None) is not None:
                self._scatter.setData([], [])
            return
        # Lazily build the ScatterPlotItem the first time we need one.
        if not hasattr(self, "_scatter") or self._scatter is None:
            self._scatter = pg.ScatterPlotItem(pxMode=True, size=point_size,
                                                 useCache=True)
            self._plot.addItem(self._scatter)
        self._image.setImage(np.zeros((2, 2)))   # hide histogram
        self._image.setVisible(False)
        self._scatter.setVisible(True)
        # Build per-point brushes.
        if c is None:
            from PyQt6.QtGui import QColor, QBrush
            brush = QBrush(QColor(theme.ACCENT))
            self._scatter.setData(x=x, y=y, brush=brush, size=point_size)
            self._cbar.setLevels((0.0, 1.0))
            return
        c = np.asarray(c, dtype=float)
        cmin = float(np.nanmin(c)); cmax = float(np.nanmax(c))
        if cmax <= cmin:
            cmax = cmin + 1.0
        norm = (c - cmin) / (cmax - cmin)
        try:
            cmap = pg.colormap.get(cmap_name)
        except Exception:
            cmap = self._cmap
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        idx = np.clip((norm * 255).astype(int), 0, 255)
        from PyQt6.QtGui import QColor
        brushes = [pg.mkBrush(QColor(*lut[i])) for i in idx]
        self._scatter.setData(x=x, y=y, brush=brushes, size=point_size)
        # Sync the colorbar to the data range so the legend is meaningful.
        try:
            self._cbar.setColorMap(cmap)
        except Exception:
            pass
        self._cbar.setLevels((cmin, cmax))
        if label:
            try:
                self._cbar.getAxis("right").setLabel(
                    f"<span style='color:{theme.TEXT_2}; "
                    f"font-size:{_AXIS_TICK_PT}pt;'>{label}</span>"
                )
            except Exception:
                pass

    def set_data(self, x, y,
                 x_range: tuple | None = None,
                 y_range: tuple | None = None,
                 log_scale: bool = True) -> None:
        import numpy as np
        # If a previous call to ``set_scatter`` swapped to scatter mode,
        # restore the histogram view here.
        if hasattr(self, "_scatter") and self._scatter is not None:
            self._scatter.setVisible(False)
        self._image.setVisible(True)
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            self._image.setImage(np.zeros((2, 2)))
            self._cbar.setLevels((0.0, 1.0))
            return
        if x_range is None:
            xm = float(np.mean(x)); xs = float(np.std(x)) or 1.0
            x_range = (xm - 4.5 * xs, xm + 4.5 * xs)
        if y_range is None:
            ym = float(np.mean(y)); ys = float(np.std(y)) or 1.0
            y_range = (ym - 4.5 * ys, ym + 4.5 * ys)
        H, _, _ = np.histogram2d(x, y, bins=self._bins,
                                  range=(x_range, y_range))
        if self._sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=self._sigma, mode="nearest")
            except ImportError:
                pass
        if log_scale:
            H = np.log1p(H)
        H_t = H.T
        self._image.setImage(H_t, autoLevels=False)
        lo = float(H_t.min()); hi = float(H_t.max())
        if hi <= lo:
            hi = lo + 1.0
        self._image.setLevels((lo, hi))
        self._cbar.setLevels((lo, hi))
        self._image.setRect(pg.QtCore.QRectF(
            x_range[0], y_range[0],
            x_range[1] - x_range[0], y_range[1] - y_range[0],
        ))


def update_density(img: pg.ImageItem, x, y,
                   bins: int = 140, sigma: float = 0.9,
                   x_range: tuple | None = None,
                   y_range: tuple | None = None,
                   log_scale: bool = True) -> None:
    """Re-bin (x, y) and update ``img`` in place."""
    import numpy as np
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        img.setImage(np.zeros((2, 2)))
        return
    if x_range is None:
        xm = float(np.mean(x)); xs = float(np.std(x)) or 1.0
        x_range = (xm - 4.5 * xs, xm + 4.5 * xs)
    if y_range is None:
        ym = float(np.mean(y)); ys = float(np.std(y)) or 1.0
        y_range = (ym - 4.5 * ys, ym + 4.5 * ys)
    H, _, _ = np.histogram2d(x, y, bins=bins, range=(x_range, y_range))
    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=sigma, mode="nearest")
        except ImportError:
            pass
    if log_scale:
        H = np.log1p(H)
    img.setImage(H.T)
    img.setRect(pg.QtCore.QRectF(
        x_range[0], y_range[0],
        x_range[1] - x_range[0], y_range[1] - y_range[0],
    ))
