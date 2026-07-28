"""Thin lattice-impression strip for the Results-tab plot popups.

A pyqtgraph PlotWidget that lives **above** a data plot and shows
coloured rectangles per finite-length lattice element along the s-axis
(mm).  Its x-axis is **linked** to the data plot via
``pg.ViewBox.setXLink``, so panning / zooming the data plot moves the
strip with it.  Mouse interaction on the strip itself is disabled — it
is read-only.

Zero-length elements that carry physical meaning (Foil, RFGap with
L=0, Aperture) render as thin vertical lines.  Pure passive markers
(LatticeCommand, Marker, zero-length SET_*/ADJUST_*) are skipped to
avoid saturating the strip on PIP-II-scale lattices.

Reuses the existing colour map ``theme.EL_COLORS`` so the strip
matches the Lattice-tab timeline.
"""
from __future__ import annotations

from typing import Iterable

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QBrush, QColor, QPen

from linac_gen_gui.interphase import theme


# Element-type names that carry physical meaning even with zero length.
# These render as vertical-line markers in the strip.
_ZERO_LENGTH_KEEP: frozenset[str] = frozenset({
    "Foil",          # stripping / scattering foil — Month-1 addition
    "Aperture",
    "BPM",
    "Marker",        # only if the user flagged it as a snapshot point
    "RFGap",         # zero-length RF gap
    "Steerer",       # zero-length thin corrector
    "ThinLens",
})

# Element types that we deliberately skip (too many of them on real
# lattices, no useful visual signal).  All TraceWin SET_*/ADJUST_*
# commands match this via the LatticeCommand base.
_SKIP_CLASS_NAMES: frozenset[str] = frozenset()


def _color_for(elem) -> str:
    """Theme colour for an element type, with a fallback."""
    return theme.EL_COLORS.get(type(elem).__name__, theme.TEXT_1)


def _is_passive_command(elem) -> bool:
    """True for the TraceWin SET_*/ADJUST_* family — they have no visual
    on the strip and would otherwise saturate it on real PIP-II lattices.
    """
    try:
        from linac_gen.elements.lattice_commands import LatticeCommand
    except ImportError:
        return False
    return isinstance(elem, LatticeCommand)


class LatticeStripWidget(pg.PlotWidget):
    """Thin lattice-strip plot — see module docstring.

    The widget exposes one public method, :meth:`set_lattice`, which
    rebuilds the rectangles + lines for a new lattice.  Construction
    initialises an empty strip; call ``set_lattice(lattice)`` to fill it.
    """

    # Fixed pixel height — the strip is purely cosmetic, no need for
    # vertical extent.  Matches the typical lattice-bar height in
    # TraceWin and MAD-X plotting tools.
    STRIP_HEIGHT_PX: int = 36

    def __init__(self, parent=None) -> None:
        # Disable the default Auto-range/Mouse menu — read-only widget.
        super().__init__(parent=parent, background=theme.BG_1)
        self._items: list = []    # references so Python doesn't GC them
        self._total_length_mm: float = 0.0

        # No axes, no grid, no mouse.
        plot_item = self.getPlotItem()
        plot_item.hideAxis("left")
        plot_item.hideAxis("bottom")
        plot_item.showGrid(False, False)
        plot_item.setMenuEnabled(False)
        plot_item.setMouseEnabled(x=False, y=False)
        plot_item.hideButtons()
        plot_item.getViewBox().setBackgroundColor(theme.BG_1)
        # No padding around items — the rectangles touch the strip edges.
        plot_item.getViewBox().setDefaultPadding(0.0)

        # Fixed height; flexible width.
        self.setFixedHeight(self.STRIP_HEIGHT_PX)
        self.setMinimumWidth(100)

        # Vertical extent is purely cosmetic.  Use a (0, 1) y-range so
        # rectangle heights are normalised — easy to reason about.
        self.setYRange(0.0, 1.0, padding=0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lattice(self, lattice) -> None:
        """Rebuild the strip from ``lattice.elements``.

        Safe to call repeatedly.  ``lattice=None`` clears the strip.
        """
        self._clear_items()
        if lattice is None or not getattr(lattice, "elements", None):
            return
        s_cursor_mm = 0.0
        for elem in lattice.elements:
            length = float(getattr(elem, "length", 0.0) or 0.0)
            type_name = type(elem).__name__
            color = _color_for(elem)
            if length > 0.0:
                self._add_rect(s_cursor_mm, length, color, elem)
            else:
                # Skip passive lattice commands and unflagged markers —
                # they would clutter the strip with no useful signal.
                if _is_passive_command(elem):
                    pass  # SET_*/ADJUST_*: silent
                elif type_name in _ZERO_LENGTH_KEEP:
                    # Render as a thin vertical accent line so the
                    # element is still locatable on the s-axis.
                    self._add_marker_line(s_cursor_mm, color, elem)
                # else: silently skip unknown zero-length elements.
            s_cursor_mm += length
        self._total_length_mm = s_cursor_mm
        # Initial x-range covers the whole lattice; setXLink (below)
        # will then keep it in sync with the data plot.
        if s_cursor_mm > 0.0:
            self.setXRange(0.0, s_cursor_mm, padding=0.0)

    def link_x_to(self, other_plot) -> None:
        """Link x-axis to another pyqtgraph plot.

        ``other_plot`` can be a :class:`pg.PlotWidget` directly, OR a
        container widget that holds one or more ``pg.PlotWidget`` children
        (e.g. ``EnvelopeTriple``).  In the latter case the link target is
        the first ``pg.PlotWidget`` child found via ``findChildren``.

        After linking, pan/zoom on ``other_plot`` automatically moves
        this strip.  The link is one-way (strip follows, not drives) —
        the strip has mouse-interaction disabled so it cannot drive
        anyway.
        """
        if other_plot is None:
            return
        target_plot: pg.PlotWidget | None = None
        if isinstance(other_plot, pg.PlotWidget):
            target_plot = other_plot
        else:
            # Container widget — find the first pg.PlotWidget child.
            try:
                children = other_plot.findChildren(pg.PlotWidget)
            except Exception:
                children = []
            if children:
                target_plot = children[0]
        if target_plot is None:
            return
        try:
            self.getPlotItem().getViewBox().setXLink(target_plot.getPlotItem())
        except Exception:
            # pyqtgraph occasionally surfaces RuntimeError when widgets
            # are mid-teardown — tolerate it.
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clear_items(self) -> None:
        """Remove all rectangles and marker lines from the plot."""
        plot_item = self.getPlotItem()
        for item in self._items:
            try:
                plot_item.removeItem(item)
            except Exception:
                pass
        self._items.clear()

    def _add_rect(self, s_start_mm: float, length_mm: float,
                   color: str, elem) -> None:
        """Add a filled rectangle for a finite-length element."""
        # pyqtgraph's QGraphicsRectItem in data coordinates.
        rect = QRectF(s_start_mm, 0.05, length_mm, 0.90)
        item = pg.QtWidgets.QGraphicsRectItem(rect)
        brush = QBrush(QColor(color))
        pen = QPen(QColor(color))
        pen.setWidthF(0.0)             # cosmetic 0-width pen = no outline drift on zoom
        item.setBrush(brush)
        item.setPen(pen)
        # Hover tooltip — single per-rect setToolTip is cheap enough at
        # < 2000 elements; the rect uses standard Qt tooltip mechanics.
        item.setAcceptHoverEvents(True)
        item.setToolTip(
            f"{type(elem).__name__}  {getattr(elem, 'name', '?')}\n"
            f"s = {s_start_mm:.1f}–{s_start_mm + length_mm:.1f} mm  "
            f"(L = {length_mm:.2f} mm)"
        )
        self.getPlotItem().addItem(item)
        self._items.append(item)

    def _add_marker_line(self, s_mm: float, color: str, elem) -> None:
        """Add a thin vertical line for a physically-meaningful
        zero-length element."""
        pen = pg.mkPen(QColor(color), width=1.5, style=Qt.PenStyle.SolidLine)
        line = pg.InfiniteLine(pos=s_mm, angle=90, pen=pen, movable=False)
        # InfiniteLine doesn't support hover tooltips natively — bake the
        # name into a custom text via the label= keyword (small, sparse).
        # Skip the label if the element is unnamed.
        name = getattr(elem, "name", None)
        if name:
            line.label = pg.InfLineLabel(
                line, text=f"{type(elem).__name__}", color=color,
                position=0.95, anchor=(1, 0),
            )
        self.getPlotItem().addItem(line)
        self._items.append(line)


def make_lattice_strip(parent, main_plot: pg.PlotWidget,
                        lattice) -> LatticeStripWidget:
    """Build a :class:`LatticeStripWidget`, populate it from ``lattice``,
    and link its x-axis to ``main_plot``.

    Parameters
    ----------
    parent : QWidget or None
        Qt parent for the returned widget.
    main_plot : pg.PlotWidget
        The data plot whose x-axis the strip will follow.  May be None;
        the strip will still build but won't auto-pan.
    lattice : Lattice or None
        HELIX lattice object.  ``None`` returns an empty strip.

    Returns
    -------
    LatticeStripWidget
        Ready to be inserted into a vertical layout above ``main_plot``.
    """
    strip = LatticeStripWidget(parent=parent)
    if lattice is not None:
        strip.set_lattice(lattice)
    if main_plot is not None:
        strip.link_x_to(main_plot)
    return strip


__all__ = ["LatticeStripWidget", "make_lattice_strip"]
