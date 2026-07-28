"""Shared styling for all pyqtgraph plots in the GUI.

Centralised so the look & feel stays consistent across widgets:
- Dark background, pure-white foreground (high contrast).
- Thicker curve pens so thin signals remain visible against the black.
- Larger tick / label / title / legend fonts so the GUI reads well on
  normal-resolution monitors.

Import at module scope from every plot widget so the global pyqtgraph
config options are applied before any ``PlotWidget`` is constructed.
"""
import pyqtgraph as pg
from PyQt6.QtGui import QFont

# ---------------------------------------------------------------------------
# Global pyqtgraph config -- applied once, at first import.
# ---------------------------------------------------------------------------
pg.setConfigOption("background", "k")           # black background
pg.setConfigOption("foreground", "w")           # pure-white axes / text
pg.setConfigOption("antialias", True)

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
LINE_WIDTH = 2.5                    # default curve pen width (pixels)
SCATTER_SIZE = 4                    # default scatter-point size (pixels)

_TITLE_SIZE_PT = 14
_LABEL_SIZE_PT = 12
_TICK_SIZE_PT = 11
_LEGEND_SIZE_PT = 11

#: High-contrast curve palette on a black background.
#: Keys match the physical plane/quantity they usually represent.
COLORS = {
    "x":        "#4da6ff",          # bright blue
    "y":        "#ff6b6b",          # bright red
    "z":        "#7cff7c",          # bright green
    "w":        "#ffd24d",          # amber
    "phi":      "#ff9ad6",          # pink
    "loss":     "#ff5050",          # alarm red
    "scatter":  "#4dd6ff",          # cyan
    "highlight": "#ffffff",
    "t":        "#c08cff",          # purple — 4-D transverse / TraceWin εt
}


def _qfont(size_pt: int, bold: bool = False) -> QFont:
    f = QFont()
    f.setPointSize(size_pt)
    if bold:
        f.setBold(True)
    return f


def style_plot(plot_widget, title: str | None = None,
               xlabel: tuple | None = None,
               ylabel: tuple | None = None) -> None:
    """Apply the shared look to *plot_widget*.

    Parameters
    ----------
    plot_widget : pg.PlotWidget
        The plot to restyle.
    title : str, optional
        Title text; set only if given (so this helper can be called on an
        existing plot without clobbering an earlier title).
    xlabel, ylabel : (text, units) or (text,), optional
        If given, replace the axis labels with the styled version.
    """
    plot_item = plot_widget.getPlotItem()

    # Title
    if title is not None:
        plot_item.setTitle(title, color="w", size=f"{_TITLE_SIZE_PT}pt")

    # NOTE: we deliberately bypass pyqtgraph's ``units=`` kwarg and splice
    # the units straight into the label text.  Passing ``units="mm"`` makes
    # pyqtgraph auto-apply SI prefixes (e.g. axis "0..1600 mm" renders as
    # "0..1.6 m", and "0.4..1 mm" renders as "400..1000 um"), which reads
    # as broken on a fixed-mm codebase.  With plain text no rescaling.
    label_style = {"color": "#ffffff", "font-size": f"{_LABEL_SIZE_PT}pt"}

    def _format(label_tuple):
        if len(label_tuple) > 1 and label_tuple[1]:
            return f"{label_tuple[0]} ({label_tuple[1]})"
        return label_tuple[0]

    if xlabel is not None:
        plot_widget.setLabel("bottom", _format(xlabel), **label_style)
    if ylabel is not None:
        plot_widget.setLabel("left", _format(ylabel), **label_style)

    # Axis tick fonts & pen
    tick_font = _qfont(_TICK_SIZE_PT)
    axis_pen = pg.mkPen("#cccccc", width=1.2)
    for side in ("left", "bottom", "right", "top"):
        ax = plot_item.getAxis(side)
        ax.setTickFont(tick_font)
        ax.setPen(axis_pen)
        ax.setTextPen("#ffffff")

    # Grid (subtle)
    plot_item.showGrid(x=True, y=True, alpha=0.25)


def mkpen(color_key_or_hex: str, width: float = LINE_WIDTH) -> pg.mkPen:
    """Return a thick pen in one of the palette colours."""
    color = COLORS.get(color_key_or_hex, color_key_or_hex)
    return pg.mkPen(color, width=width)


def styled_legend(plot_widget, offset=(10, 10)):
    """Add a legend with larger, high-contrast text."""
    legend = plot_widget.addLegend(
        offset=offset,
        labelTextSize=f"{_LEGEND_SIZE_PT}pt",
        labelTextColor="w",
    )
    return legend
