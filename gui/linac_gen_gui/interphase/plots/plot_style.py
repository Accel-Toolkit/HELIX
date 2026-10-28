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

# DensityPanel profile strips (px): the strip's plot area, the axes with the
# tick values and title (left strip: width, lower strip: height), and the
# map's tick-only axes while the strips show
_PROFILE_STRIP_PX = 62
_PROFILE_AXIS_W   = 56
_PROFILE_AXIS_H   = 44
_PROFILE_TICK_PX  = 6
_PROFILE_GAP_PX   = 4        # between a strip and the map

# What a DensityPanel's colour bar measures: the image is log10(1 + counts)
_DENSITY_LABEL = "log₁₀(1+N)"

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


class HelixItemSample(pg.ItemSample):
    """Legend sample that, after pyqtgraph's own click-to-hide, makes the
    view refit the curves that remain.  pyqtgraph informs the ViewBox only
    of position/transform changes, so hiding a curve never re-runs the
    auto-range by itself (and after a manual zoom auto-range is off)."""

    def mouseClickEvent(self, event):
        from PyQt6.QtCore import Qt
        is_left = event.button() == Qt.MouseButton.LeftButton
        super().mouseClickEvent(event)
        if not is_left:
            return
        item = self.item
        for other in getattr(item, "_helix_linked", ()) or ():
            try:
                other.setVisible(item.isVisible())
            except RuntimeError:
                pass
        refit_view_to_visible(item)


def refit_view_to_visible(item) -> None:
    """Re-enable auto-range on the item's ViewBox and refit now, by the
    rule of :func:`_rearm_view`: y always, x only when the view does not
    follow another view's x (a follower that auto-ranged x would drive its
    leader).  Skipped for items that take no part in the bounds
    (``ignoreBounds``) and for plots that opted out with
    :func:`fix_plot_range` (deliberate fixed scales)."""
    try:
        vb = item.getViewBox()
    except Exception:                                        # noqa: BLE001
        return
    if vb is None or _is_fixed(vb):
        return
    if item not in getattr(vb, "addedItems", ()):
        return
    vb.itemBoundsChanged(item)
    for other in getattr(item, "_helix_linked", ()) or ():
        vb.itemBoundsChanged(other)
    _rearm_view(vb)
    vb.updateAutoRange()


def _view_box(p):
    """The ViewBox of a PlotWidget, or of a PlotItem (a plot inside a
    GraphicsLayoutWidget)."""
    get_item = getattr(p, "getPlotItem", None)
    return (get_item() if get_item is not None else p).getViewBox()


def fix_plot_range(p, xrange=None, yrange=None, padding=None) -> None:
    """Mark a plot's view as deliberately scaled: legend clicks still hide
    curves but never refit it, and View All fits it once instead of
    switching auto-range on.  ``xrange`` / ``yrange`` (``(lo, hi)``, with
    pyqtgraph's ``padding``) are its designed frame: set now, and restored
    by :func:`rearm_autorange` whenever the plot is drawn anew.  Plots that
    set their frame at every redraw just call ``fix_plot_range(p)``.
    ``p`` is a PlotWidget or a PlotItem."""
    try:
        vb = _view_box(p)
    except Exception:                                        # noqa: BLE001
        return
    vb._helix_fixed_range = True
    if xrange is not None or yrange is not None:
        vb._helix_frame = (
            None if xrange is None else (float(xrange[0]), float(xrange[1])),
            None if yrange is None else (float(yrange[0]), float(yrange[1])),
            padding,
        )
        _restore_frame(vb)


def release_plot_range(p) -> None:
    """Undo :func:`fix_plot_range`: the plot auto-ranges like any other
    (a panel that switches from a designed frame to a quantity of its own
    scale).  ``p`` is a PlotWidget or a PlotItem."""
    try:
        vb = _view_box(p)
    except Exception:                                        # noqa: BLE001
        return
    vb._helix_fixed_range = False
    vb._helix_frame = None
    vb.enableAutoRange(pg.ViewBox.YAxis, True)
    if _x_leader(vb) is None:
        vb.enableAutoRange(pg.ViewBox.XAxis, True)


# ---------------------------------------------------------------------------
# Auto-range: every plot fits all its data whenever it is drawn
# ---------------------------------------------------------------------------
# pyqtgraph turns a view's auto-range off whenever its range is set: by
# code, by a mouse zoom or pan, and by the menu's "View All" itself, which
# fits once.  The plots here live for the whole session and are drawn again
# for every run, so a view once fitted by hand stayed frozen.  The rules:
#   * a newly drawn plot is re-armed (rearm_autorange): y auto-range on
#     every view, x on every view that does not follow another view's x.
#     A follower (a lower panel, a lattice strip) keeps x auto-range off:
#     pyqtgraph forwards a follower's range change to its leader and
#     switches the leader's x auto-range off, so a follower must never
#     fit x itself.
#   * "View All", the "A" button and the x "Auto" option switch auto-range
#     on and leave it on, the x of a follower on its leader.
#   * plots with a designed frame (fix_plot_range) keep it, axis by axis:
#     re-arming restores the recorded frame on the axes it covers and arms
#     the others; View All, the "A" button and the x "Auto" option fit a
#     framed axis once and keep fitting an unframed one.  A fixed follower
#     never touches its leader's x.
#   * a view that copies an axis from another view one way (``vb.
#     _helix_follows = {axis: leader}``: the profile strips of a
#     DensityPanel) takes the leader's range on that axis instead of
#     auto-ranging it; the copy is kept up to date by the owner's
#     range-changed slot, never by a pyqtgraph link.

def _is_fixed(vb) -> bool:
    return bool(getattr(vb, "_helix_fixed_range", False))


def _follows(vb) -> dict:
    """``{axis: view}``: the axes ``vb`` copies one way from another view
    (0 = x, 1 = y), or an empty dict."""
    return getattr(vb, "_helix_follows", None) or {}


def _copy_followed(vb) -> None:
    """Give each followed axis of ``vb`` its leader's current range, with
    auto-range off on it."""
    for axis, leader in _follows(vb).items():
        try:
            lo, hi = leader.viewRange()[axis]
            vb.enableAutoRange(axis, False)
            if axis == 0:
                vb.setXRange(lo, hi, padding=0)
            else:
                vb.setYRange(lo, hi, padding=0)
        except RuntimeError:                   # C++ side already deleted
            pass


def _x_leader(vb):
    """The view whose x range ``vb`` follows, or None."""
    try:
        return vb.linkedView(pg.ViewBox.XAxis)
    except (RuntimeError, AttributeError):
        return None


def _frame(vb):
    """``(xrange, yrange, padding)`` recorded by fix_plot_range, or Nones."""
    return getattr(vb, "_helix_frame", None) or (None, None, None)


def _restore_frame(vb) -> None:
    xr, yr, padding = _frame(vb)
    if xr is not None and _x_leader(vb) is None:
        vb.setXRange(xr[0], xr[1], padding=padding)
    if yr is not None:
        vb.setYRange(yr[0], yr[1], padding=padding)


def _rearm_view(vb) -> None:
    """Give one view the auto-range of a newly drawn plot (rules above):
    an axis with a recorded frame gets the frame back, every other axis
    auto-ranges -- x only on a view that does not follow another's x."""
    if vb is None:
        return
    try:
        xr, yr, padding = _frame(vb)
        follows = _follows(vb)
        if 0 in follows:
            pass                               # copied below
        elif _x_leader(vb) is not None:
            vb.enableAutoRange(pg.ViewBox.XAxis, False)
        elif xr is not None:
            vb.setXRange(xr[0], xr[1], padding=padding)
        else:
            vb.enableAutoRange(pg.ViewBox.XAxis, True)
        if 1 in follows:
            pass
        elif yr is not None:
            vb.setYRange(yr[0], yr[1], padding=padding)
        else:
            vb.enableAutoRange(pg.ViewBox.YAxis, True)
        if follows:
            _copy_followed(vb)
    except RuntimeError:                       # C++ side already deleted
        pass


def _views_under(root):
    """The ViewBoxes of the plots in ``root`` (a popup, tab, dialog or
    plot widget): every PlotWidget, and every PlotItem of every
    GraphicsLayoutWidget (DensityPanel, heat plots); colour bars excluded."""
    widgets = []
    if isinstance(root, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
        widgets.append(root)
    try:
        widgets.extend(root.findChildren(pg.PlotWidget))
        widgets.extend(root.findChildren(pg.GraphicsLayoutWidget))
    except (RuntimeError, AttributeError):
        pass
    views, seen = [], set()
    for w in widgets:
        try:
            if isinstance(w, pg.PlotWidget):
                items = [w.getPlotItem()]
            else:
                items = list(w.ci.items.keys())
        except (RuntimeError, AttributeError):
            continue
        for item in items:
            if not isinstance(item, pg.PlotItem) or isinstance(item, pg.ColorBarItem):
                continue
            try:
                vb = item.getViewBox()
            except RuntimeError:
                continue
            if vb is not None and id(vb) not in seen:
                seen.add(id(vb))
                views.append(vb)
    return views


def rearm_autorange(root) -> None:
    """Re-arm every plot in ``root`` as a newly drawn plot: auto-range on
    (x on the views that lead, y on all), designed frames restored.  Call
    it before drawing a NEW result into long-lived plots (a popup opened,
    a new run, a new study or scan) -- not on live updates, so that a zoom
    lasts until the next new plot."""
    if root is None:
        return
    views = _views_under(root)
    # Followers' x off first: switching a follower's pending x auto-range
    # off runs it once more, which would reach a leader already re-armed.
    for vb in views:
        try:
            if _x_leader(vb) is not None:
                vb.enableAutoRange(pg.ViewBox.XAxis, False)
        except RuntimeError:
            pass
    for vb in views:
        _rearm_view(vb)


def _fit_and_keep(vb) -> None:
    """What View All and the "A" button do: switch auto-range on and leave
    it on -- y on this view, x on the view it follows (or on itself) unless
    that one is fixed.  On a fixed view an axis with a designed frame (or
    every axis, when the view records none) is fitted once instead, and a
    fixed follower leaves its leader's x alone."""
    if vb is None:
        return
    try:
        if _follows(vb):
            _rearm_view(vb)                    # followed axes copied
            return
        if _is_fixed(vb):
            _fit_fixed(vb, x=True, y=True)
            return
        vb.enableAutoRange(pg.ViewBox.YAxis, True)
        _fit_and_keep_x(vb)
    except RuntimeError:                       # C++ side already deleted
        pass


def _fit_fixed(vb, x: bool, y: bool) -> None:
    xr, yr, _padding = _frame(vb)
    framed = getattr(vb, "_helix_frame", None) is not None
    if y:
        if yr is not None or not framed:
            _fit_once(vb, 1)
        else:
            vb.enableAutoRange(pg.ViewBox.YAxis, True)
    if x and _x_leader(vb) is None:
        if xr is not None or not framed:
            _fit_once(vb, 0)
        else:
            vb.enableAutoRange(pg.ViewBox.XAxis, True)


def _fit_once(vb, axis: int) -> None:
    """Fit one axis to the items' bounds once (auto-range left off)."""
    import numpy as np
    try:
        r = vb.childrenBounds()[axis]
    except (RuntimeError, AttributeError, IndexError, TypeError):
        return
    if r is None or not np.all(np.isfinite(r)):
        return
    if axis == 0:
        vb.setXRange(float(r[0]), float(r[1]))
    else:
        vb.setYRange(float(r[0]), float(r[1]))


def _menu_view(menu):
    try:
        return menu.view()
    except (RuntimeError, TypeError, AttributeError):
        return None


def _view_all(self, *args) -> None:
    """ViewBoxMenu "View All": fit and keep fitting (pyqtgraph fits once)."""
    _fit_and_keep(_menu_view(self))


def _x_auto(menu, stock_name: str, *args) -> None:
    """ViewBoxMenu x "Auto" (and its percentage): pyqtgraph's own on a view
    that leads and is not fixed; a follower auto-ranges its leader's x; a
    fixed view fits a framed x once and leaves a leader alone."""
    vb = _menu_view(menu)
    if vb is None:
        return
    try:
        if _is_fixed(vb):
            _fit_fixed(vb, x=True, y=False)
        elif _x_leader(vb) is not None:
            _fit_and_keep_x(vb)
        else:
            stock = _PG_STOCK.get(stock_name)
            if stock is not None:
                stock(menu, *args)
            else:
                vb.enableAutoRange(pg.ViewBox.XAxis, True)
    except RuntimeError:
        pass


def _x_auto_clicked(self, *args) -> None:
    _x_auto(self, "xAutoClicked")


def _x_auto_spin_changed(self, val=None, *args) -> None:
    if val is None:
        _x_auto(self, "xAutoClicked")
    else:
        _x_auto(self, "xAutoSpinChanged", val)


def _fit_and_keep_x(vb) -> None:
    """x auto-range on, and left on: on the view itself when it leads, else
    on its leader (unless that one is fixed), the follower's own x off."""
    leader = _x_leader(vb)
    try:
        if leader is None:
            vb.enableAutoRange(pg.ViewBox.XAxis, True)
            return
        vb.enableAutoRange(pg.ViewBox.XAxis, False)
        if not _is_fixed(leader):
            leader.enableAutoRange(pg.ViewBox.XAxis, True)
    except RuntimeError:
        pass


def _auto_button(self, *args) -> None:
    """PlotItem "A" button: as View All."""
    try:
        if getattr(self.autoBtn, "mode", "auto") != "auto":
            self.disableAutoRange()
            return
        _fit_and_keep(self.getViewBox())
        self.autoBtn.hide()
    except RuntimeError:
        pass


# pyqtgraph's own slots, kept on the patched class itself so that a reload
# of this module still finds them.
_PG_STOCK: dict = {}


def _install_view_patches() -> None:
    """Route View All, the x "Auto" option and the "A" button through
    :func:`_fit_and_keep`.  Done at import: PyQt binds these slots when each
    menu or plot is built, so plots built earlier keep pyqtgraph's own.
    Plain functions taking ``*args`` (a ``@Slot`` would be connected to the
    original method or refused).  A pyqtgraph without one of the methods
    keeps its own behaviour for it."""
    global _PG_STOCK
    try:
        from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu
        from pyqtgraph.graphicsItems.PlotItem.PlotItem import PlotItem
    except ImportError:
        return
    stock = getattr(ViewBoxMenu, "_helix_stock", None)
    if stock is None:
        stock = {}
        for cls, name, fn in (
                (ViewBoxMenu, "autoRange", _view_all),
                (ViewBoxMenu, "xAutoClicked", _x_auto_clicked),
                (ViewBoxMenu, "xAutoSpinChanged", _x_auto_spin_changed),
                (PlotItem, "autoBtnClicked", _auto_button)):
            if callable(getattr(cls, name, None)):
                stock[name] = getattr(cls, name)
                setattr(cls, name, fn)
        ViewBoxMenu._helix_stock = stock
    _PG_STOCK = stock


_install_view_patches()


def add_legend(p: pg.PlotWidget, offset: tuple[int, int] = (-10, 10)):
    """Attach a legend with a translucent dark background and bright labels.
    Clicking an entry hides its curve and refits the view (HelixItemSample)."""
    legend = p.addLegend(
        offset=offset,
        labelTextColor=theme.TEXT_0,
        brush=pg.mkBrush(6, 10, 15, 220),   # BG_INSET-ish with alpha
        pen=pg.mkPen(theme.BORDER_1),
        labelTextSize=f"{_AXIS_TICK_PT}pt",
        sampleType=HelixItemSample,
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
        H = np.log10(1.0 + H)

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


def fit_colorbar_width(cbar) -> int:
    """Fix a vertical ``pg.ColorBarItem`` to the width of its parts: the
    titled left axis, the gradient and the right axis with the tick values
    (pyqtgraph sets that one to 45 px).  A narrower fixed width cuts the
    tick values off ("2.0" drawn as "2.C").  Returns the width (px)."""
    import math
    lay = cbar.layout
    left, _top, right, _bottom = lay.getContentsMargins()
    width = math.ceil(left + right + lay.columnMinimumWidth(1)
                      + cbar.getAxis("left").minimumWidth()
                      + cbar.getAxis("right").minimumWidth())
    cbar.setFixedWidth(width)
    return width


class DensityPanel(pg.GraphicsLayoutWidget):
    """Self-contained phase-space density panel: title, axes, heatmap, colorbar.

    Internally a pyqtgraph GraphicsLayoutWidget.  The PlotItem holds axes +
    the ImageItem, with a narrow vertical gradient labelled "log₁₀(1+N)"
    that reflects the current image levels.  The panel exposes
    ``set_data(x, y)`` which re-bins the density and updates the colorbar
    range.

    With ``profiles=True`` the map gets the two projected distributions:
    the horizontal quantity's histogram in a strip below it and the
    vertical quantity's in a strip to its left (counts grow away from the
    map), each scaled to its own peak, with the rms-equivalent Gaussian
    dashed over it.  The strips carry the tick values and axis titles; they
    copy the map's range one way, so zooming the map moves them.
    """

    def __init__(self, title: str = "",
                 xlabel: str = "", xunits: str = "",
                 ylabel: str = "", yunits: str = "",
                 bins: int = 140, sigma: float = 0.9,
                 profiles: bool = False):
        import numpy as np
        from PyQt6.QtGui import QFont
        configure_pyqtgraph_defaults()
        super().__init__()
        self.setBackground(theme.BG_INSET)
        self._bins = bins
        self._sigma = sigma
        self._labels = (xlabel, xunits, ylabel, yunits)
        self._prof_x = self._prof_y = None
        self._profiles_on = False
        self._gauss_on = True

        # Plot + axes — pens FIRST so they don't clobber label style.
        self._plot = self.addPlot(row=0, col=1 if profiles else 0)
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

        self._title = title
        if title:
            self._plot.setTitle(self._title_html(title))

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
            label=self._cbar_label_html(_DENSITY_LABEL),
            interactive=False,
            orientation="v",
            width=14,
        )
        self._cbar_label = _DENSITY_LABEL
        self._cbar.getAxis("right").setTextPen(theme.TEXT_0)
        self._cbar.getAxis("right").setPen(theme.BORDER_2)
        self._cbar.getAxis("right").setTickFont(tick_font)
        fit_colorbar_width(self._cbar)
        # Attach the bar to the image so levels/cmap stay in sync.
        self._cbar.setImageItem(self._image, insert_in=self._plot)
        if profiles:
            self._build_profiles(tick_font)
            self.set_profiles_visible(True)

    # ------------------------------------------------------------- profiles
    def _build_profiles(self, tick_font) -> None:
        """The two profile strips (not yet in the layout) and their curves."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor, QTransform
        main_vb = self._plot.getViewBox()
        line = pg.mkPen(theme.ACCENT, width=1.6)
        fill = QColor(theme.ACCENT)
        fill.setAlpha(70)
        dash = pg.mkPen(theme.TEXT_1, width=1.2, style=Qt.PenStyle.DashLine)
        swap = QTransform(0, 1, 1, 0, 0, 0)          # (q, n) -> (n, q)
        strips = []
        for axis in (0, 1):                          # 0: below (x), 1: left (y)
            p = pg.PlotItem()
            p.setMenuEnabled(False)
            p.hideButtons()
            p.setMouseEnabled(x=False, y=False)
            p.showGrid(x=True, y=True, alpha=0.2)
            for axname in ("left", "bottom", "right"):
                ax = p.getAxis(axname)
                ax.setTextPen(theme.TEXT_0)
                ax.setPen(theme.BORDER_2)
                ax.setTickFont(tick_font)
                ax.enableAutoSIPrefix(False)
            vb = p.getViewBox()
            vb._helix_follows = {axis: main_vb}
            # Curves are drawn as (quantity, counts); the left strip shows
            # them through an axis swap, so a curve's data stays
            # (quantity, counts) in exports either way.
            curve = pg.PlotDataItem(pen=line, fillLevel=0.0, brush=fill)
            gauss = pg.PlotDataItem(pen=dash, name="rms Gaussian")
            for item in (curve, gauss):
                if axis == 1:
                    item.setTransform(swap)
                p.addItem(item)
            strips.append((p, curve, gauss))
        (self._prof_x, self._px_curve, self._px_gauss), \
            (self._prof_y, self._py_curve, self._py_gauss) = strips
        self._prof_y.getViewBox().invertX(True)      # counts grow leftwards
        # Alignment: the strips' blank axes match the map's tick-only axes
        # and its colour bar, so each strip spans exactly the map's view.
        for p, axname, size in ((self._prof_x, "left", _PROFILE_TICK_PX),
                                (self._prof_x, "right", self._cbar_width()),
                                (self._prof_y, "bottom", _PROFILE_TICK_PX)):
            ax = p.getAxis(axname)
            p.showAxis(axname)
            ax.setStyle(showValues=False, tickLength=0)
            ax.setPen(pg.mkPen(None))
            (ax.setWidth if axname in ("left", "right") else ax.setHeight)(size)
        self._prof_x.getAxis("bottom").setHeight(_PROFILE_AXIS_H)
        self._prof_y.getAxis("left").setWidth(_PROFILE_AXIS_W)
        self._set_count_range(0, 1.0)
        self._set_count_range(1, 1.0)
        main_vb.sigXRangeChanged.connect(self._sync_profile_x)
        main_vb.sigYRangeChanged.connect(self._sync_profile_y)
        self._apply_profile_labels()

    def _cbar_width(self) -> int:
        """Width (px) the colour bar takes right of the map's view: the bar
        and the gap pyqtgraph keeps before it (column 4 of the plot's
        grid, the bar being in column 5)."""
        try:
            bar = float(self._cbar.maximumWidth())
        except Exception:                                    # noqa: BLE001
            bar = 48.0
        try:
            gap = float(self._plot.layout.columnMinimumWidth(4))
        except Exception:                                    # noqa: BLE001
            gap = 5.0
        return int(round(bar + gap))

    @staticmethod
    def _cbar_label_html(text: str) -> str:
        return (f"<span style='color:{theme.TEXT_2}; "
                f"font-size:{_AXIS_TICK_PT}pt;'>{text}</span>")

    def _set_cbar_label(self, text: str) -> None:
        """The colour bar's title.  pyqtgraph puts a vertical bar's title on
        its LEFT axis (the right one carries the tick values); an empty
        title keeps its space, so the bar does not move."""
        text = text or " "
        if text == getattr(self, "_cbar_label", None):
            return
        self._cbar.getAxis("left").setLabel(self._cbar_label_html(text))
        self._cbar_label = text

    def _set_count_range(self, axis: int, top: float) -> None:
        """The count axis of the strip for ``axis``: 0 .. a little above
        ``top``, recorded as its frame so re-arming keeps it."""
        p = self._prof_x if axis == 0 else self._prof_y
        if p is None:
            return
        rng = (0.0, 1.05 * max(1.0, float(top)))
        vb = p.getViewBox()
        if axis == 0:
            vb._helix_frame = (None, rng, 0.0)
            vb.enableAutoRange(pg.ViewBox.YAxis, False)
            vb.setYRange(*rng, padding=0.0)
        else:
            vb._helix_frame = (rng, None, 0.0)
            vb.enableAutoRange(pg.ViewBox.XAxis, False)
            vb.setXRange(*rng, padding=0.0)

    def _sync_profile_x(self, *_args) -> None:
        if self._prof_x is None:
            return
        try:
            lo, hi = self._plot.getViewBox().viewRange()[0]
            vb = self._prof_x.getViewBox()
            vb.enableAutoRange(pg.ViewBox.XAxis, False)
            vb.setXRange(lo, hi, padding=0.0)
        except RuntimeError:                   # C++ side already deleted
            pass

    def _sync_profile_y(self, *_args) -> None:
        if self._prof_y is None:
            return
        try:
            lo, hi = self._plot.getViewBox().viewRange()[1]
            vb = self._prof_y.getViewBox()
            vb.enableAutoRange(pg.ViewBox.YAxis, False)
            vb.setYRange(lo, hi, padding=0.0)
        except RuntimeError:
            pass

    def set_profiles_visible(self, on: bool) -> None:
        """Show or hide the two profile strips.  Hidden, the panel is laid
        out exactly as one built without them."""
        if self._prof_x is None:
            return
        on = bool(on)
        layout = self.ci.layout
        if on and not self._profiles_on:
            self.ci.addItem(self._prof_y, row=0, col=0)
            self.ci.addItem(self._prof_x, row=1, col=1)
        elif not on and self._profiles_on:
            self.ci.removeItem(self._prof_y)
            self.ci.removeItem(self._prof_x)
        self._profiles_on = on
        w = _PROFILE_AXIS_W + _PROFILE_STRIP_PX if on else 0
        h = _PROFILE_AXIS_H + _PROFILE_STRIP_PX if on else 0
        layout.setColumnMinimumWidth(0, w)
        layout.setColumnMaximumWidth(0, w)
        layout.setRowMinimumHeight(1, h)
        layout.setRowMaximumHeight(1, h)
        # The empty column / row would still get the spacing.
        layout.setHorizontalSpacing(_PROFILE_GAP_PX if on else 0)
        layout.setVerticalSpacing(_PROFILE_GAP_PX if on else 0)
        for axname in ("left", "bottom"):
            ax = self._plot.getAxis(axname)
            ax.setStyle(showValues=not on)
            ax.showLabel(not on)
            if axname == "left":
                ax.setWidth(_PROFILE_TICK_PX if on else None)
            else:
                ax.setHeight(_PROFILE_TICK_PX if on else None)
        self._apply_profile_labels()
        if on:
            self._sync_profile_x()
            self._sync_profile_y()

    def profiles_visible(self) -> bool:
        return self._profiles_on

    def set_gaussian_visible(self, on: bool) -> None:
        """Show or hide the dashed rms-equivalent Gaussians."""
        self._gauss_on = bool(on)
        if self._prof_x is not None:
            self._px_gauss.setVisible(self._gauss_on)
            self._py_gauss.setVisible(self._gauss_on)

    def set_axis_labels(self, xlabel: str, xunits: str = "",
                        ylabel: str = "", yunits: str = "") -> None:
        """Axis titles: on the map's own axes (exports read them there) and,
        with the profile strips showing, on the strips' outer axes."""
        self._labels = (xlabel, xunits, ylabel, yunits)
        style = dict(color=theme.TEXT_0, **{"font-size": f"{_AXIS_LABEL_PT}pt"})
        self._plot.setLabel("left", _html_label(ylabel), units=yunits or None, **style)
        self._plot.setLabel("bottom", _html_label(xlabel), units=xunits or None, **style)
        if self._profiles_on:
            for axname in ("left", "bottom"):
                self._plot.getAxis(axname).showLabel(False)
        self._apply_profile_labels()

    def _apply_profile_labels(self) -> None:
        if self._prof_x is None:
            return
        xlabel, xunits, ylabel, yunits = self._labels
        style = dict(color=theme.TEXT_0, **{"font-size": f"{_AXIS_LABEL_PT}pt"})
        self._prof_x.setLabel("bottom", _html_label(xlabel), units=xunits or None, **style)
        self._prof_y.setLabel("left", _html_label(ylabel), units=yunits or None, **style)
        self._px_curve.opts["name"] = f"{xlabel} profile".strip()
        self._py_curve.opts["name"] = f"{ylabel} profile".strip()
        # The left strip carries a blank title as tall as the map's, so the
        # two views start at the same height.
        self._prof_y.setTitle(self._title_html(" ") if self._title else None)

    def _update_profiles(self, x, y, x_range, y_range) -> None:
        """Histogram each coordinate over the map's range (same bins), scale
        to its peak, and lay the rms-equivalent Gaussian over it."""
        if self._prof_x is None:
            return
        import numpy as np
        for axis, q, rng, curve, gauss in (
                (0, x, x_range, self._px_curve, self._px_gauss),
                (1, y, y_range, self._py_curve, self._py_gauss)):
            q = np.asarray(q, dtype=float)
            q = q[np.isfinite(q)]
            if q.size == 0 or rng is None or not rng[1] > rng[0]:
                curve.setData([], [])
                gauss.setData([], [])
                self._set_count_range(axis, 1.0)
                continue
            h, edges = np.histogram(q, bins=self._bins, range=rng)
            peak = float(h.max())
            if peak <= 0:
                curve.setData([], [])
                gauss.setData([], [])
                self._set_count_range(axis, 1.0)
                continue
            curve.setData(edges, h / peak, stepMode="center",
                          fillLevel=0.0)
            top = 1.0
            sd = float(q.std())
            if sd > 0:
                g_x = np.linspace(rng[0], rng[1], 400)
                width = (rng[1] - rng[0]) / self._bins
                g = (q.size * width / (sd * np.sqrt(2.0 * np.pi))
                     * np.exp(-0.5 * ((g_x - q.mean()) / sd) ** 2)) / peak
                gauss.setData(g_x, g)
                top = max(top, float(g.max()))
            else:
                gauss.setData([], [])
            gauss.setVisible(self._gauss_on)
            self._set_count_range(axis, top)

    # ------------------------------------------------------------------
    @property
    def plot(self):
        """The map's PlotItem: overlays (ellipses) go here.  Set axis titles
        with :meth:`set_axis_labels`, which also labels the profile strips."""
        return self._plot

    @staticmethod
    def _title_html(title: str) -> str:
        return (f"<span style='color:{theme.ACCENT}; "
                f"font-size:{_TITLE_PT}pt; font-weight:600; letter-spacing:1px;'>"
                f"{title.upper()}</span>")

    def set_title(self, title: str) -> None:
        """Replace the panel title (e.g. to append a BMAG value).  The
        plain text is kept in ``_title`` for callers and tests."""
        self._title = title
        self._plot.setTitle(self._title_html(title) if title else None)
        if self._prof_y is not None:
            self._prof_y.setTitle(self._title_html(" ") if title else None)

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
        # Profiles over the points' own extent, which the view fits.
        self._update_profiles(x, y, _finite_extent(x), _finite_extent(y))
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
            self._set_cbar_label(label)
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
            self._cmap_swapped = True          # set_data puts the density map back
        except Exception:
            pass
        self._cbar.setLevels((cmin, cmax))
        self._set_cbar_label(label)

    def set_data(self, x, y,
                 x_range: tuple | None = None,
                 y_range: tuple | None = None,
                 log_scale: bool = True) -> None:
        import numpy as np
        # If a previous call to ``set_scatter`` swapped to scatter mode,
        # restore the histogram view here -- with the density colour map,
        # which the colour bar (attached to the image) had swapped too.
        if hasattr(self, "_scatter") and self._scatter is not None:
            self._scatter.setVisible(False)
        if getattr(self, "_cmap_swapped", False):
            self._cbar.setColorMap(self._cmap)
            self._cmap_swapped = False
        self._set_cbar_label(_DENSITY_LABEL if log_scale else "N")
        self._image.setVisible(True)
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            self._image.setImage(np.zeros((2, 2)))
            self._cbar.setLevels((0.0, 1.0))
            self._update_profiles([], [], None, None)
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
            H = np.log10(1.0 + H)          # what the colour bar's title says
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
        self._update_profiles(x, y, x_range, y_range)


def _finite_extent(a):
    """``(min, max)`` of the finite values of ``a``, or None."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(a.min()), float(a.max())


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
        H = np.log10(1.0 + H)
    img.setImage(H.T)
    img.setRect(pg.QtCore.QRectF(
        x_range[0], y_range[0],
        x_range[1] - x_range[0], y_range[1] - y_range[0],
    ))


# ---------------------------------------------------------------------------
# Sequential colormap without the transparent low end + matplotlib dark style
# ---------------------------------------------------------------------------
def opaque_sequential_colormap(name: str = "magma", start: float = 0.12):
    """``name`` resampled from ``start`` to 1 with NO alpha ramp.

    For maps whose lowest level carries information (e.g. one particle
    outside an ellipse): unlike :func:`_density_colormap` the bottom colour
    is opaque, and skipping the near-black end keeps it visible on the
    dark theme.  Missing data should be NaN (drawn transparent).
    """
    import numpy as np
    try:
        base = pg.colormap.get(name, source="matplotlib")
    except Exception:
        base = pg.colormap.get(name)
    pos = np.linspace(0.0, 1.0, 256)
    rgba = base.map(start + (1.0 - start) * pos, mode="byte")
    return pg.ColorMap(pos, rgba)


def mpl_dark_rc() -> dict:
    """matplotlib rcParams for the Interphase dark theme — use inside
    ``matplotlib.rc_context`` (never set globally)."""
    return {
        "figure.facecolor": theme.BG_1,
        "axes.facecolor": theme.BG_INSET,
        "axes.edgecolor": theme.BORDER_0,
        "axes.labelcolor": theme.TEXT_1,
        "text.color": theme.TEXT_0,
        "xtick.color": theme.TEXT_2,
        "ytick.color": theme.TEXT_2,
        "grid.color": theme.BORDER_0,
    }


def style_mpl_3d(fig, ax) -> None:
    """Dark panes, labels and ticks for an mplot3d ``Axes3D``."""
    from matplotlib.colors import to_rgba
    fig.patch.set_facecolor(theme.BG_1)
    ax.set_facecolor(theme.BG_1)
    pane = to_rgba(theme.BG_INSET, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis.label.set_color(theme.TEXT_1)
        axis.line.set_color(theme.BORDER_1)
    ax.tick_params(colors=theme.TEXT_2, which="both")
    ax.title.set_color(theme.TEXT_0)


# ---------------------------------------------------------------------------
# Per-step data on an even s grid (heatmaps along the lattice)
# ---------------------------------------------------------------------------
def resample_on_even_s(values, s, *, pool: str = "max",
                       min_cols: int = 1000, max_cols: int = 3000):
    """Columns on an EVEN s grid from rows recorded at uneven s.

    Recorder steps are unevenly spaced (many short elements in one
    section, long drifts in another), so drawing one image column per
    step would stretch and squeeze the s axis.  Here every display column
    covers the same length of beamline:

    * a column holding one or more recorded steps shows them combined —
      ``pool="max"`` (NaN-ignoring maximum) or ``"mean"`` (NaN-ignoring
      average);
    * an empty column (between two records) holds the last step recorded
      before it.

    ``values`` is (S, K), ``s`` (S,) in the record order; repeated or
    slightly backward s values (negative drifts) are handled through the
    running maximum.  Returns ``(image, s0, s1)`` with ``image`` of shape
    (n_cols, K), float32, n_cols = clip(S, min_cols, max_cols), to be drawn
    over [s0, s1].
    """
    import numpy as np
    vals = np.asarray(values, dtype=np.float32)
    s_mono = np.maximum.accumulate(np.asarray(s, dtype=float))
    n_steps = s_mono.size
    s0, s1 = float(s_mono[0]), float(s_mono[-1])
    if s1 <= s0:
        s1 = s0 + 1.0
    n_cols = int(np.clip(n_steps, min_cols, max_cols))
    col = np.clip(((s_mono - s0) / (s1 - s0) * n_cols).astype(int), 0,
                  n_cols - 1)
    starts = np.flatnonzero(np.r_[True, np.diff(col) > 0])
    ends = np.r_[starts[1:] - 1, n_steps - 1]      # last step per column
    occupied = col[starts]
    img = np.full((n_cols, vals.shape[1]), np.nan, dtype=np.float32)
    if pool == "max":
        img[occupied] = np.fmax.reduceat(vals, starts, axis=0)
    elif pool == "mean":
        valid = ~np.isnan(vals)
        tot = np.add.reduceat(np.where(valid, vals, 0.0), starts, axis=0)
        cnt = np.add.reduceat(valid.astype(np.int64), starts, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            img[occupied] = np.where(cnt > 0, tot / cnt, np.nan)
    else:
        raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
    slot = np.full(n_cols, -1)
    slot[occupied] = np.arange(starts.size)
    slot = np.maximum.accumulate(slot)
    hold = slot >= 0
    hold[occupied] = False
    img[hold] = vals[ends[slot[hold]]]
    return img, s0, s1
