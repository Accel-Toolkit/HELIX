"""Results-tab window for the halo action scan.

Shows, for one phase-space plane, the part of the beam outside the
ellipses of the beam's own rms shape and emittance n·ε_rms at every
recorded position (see :mod:`linac_gen.diagnostics.action_scan`):

* **Map** — s horizontally, the ellipse size n = ε_ellipse / ε_rms
  vertically, colour = % of the particles present at that s lying outside
  (log).  Lines mark where 1 %, 0.1 % and 0.01 % remain (dashed: the same
  levels for a Gaussian beam, n = 9.21 / 13.82 / 18.42) and the outermost
  particle.  Below it the local normalized ε_rms, then a position slider,
  the % outside vs n at the chosen position (with the Gaussian 100·e^(−n/2))
  and a readout.  Hovering the map reads the exact counts; a click moves
  the position.
* **3D** — the same data as a rotatable surface (matplotlib mplot3d:
  pyqtgraph's OpenGL module is not a HELIX dependency).

The data come from a multi-particle run made with "Record halo action
scan" (Numerics tab) — live, or re-imported from its HDF5 file.
"""
from __future__ import annotations

import math
import traceback

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSizePolicy,
    QSlider, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from linac_gen.diagnostics.action_scan import (
    LEVELS_PCT, action_scan_from_results, gaussian_level_n,
    gaussian_percent_outside,
)
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.plots.plot_style import (
    add_legend, curve_pen, mpl_dark_rc, opaque_sequential_colormap,
    resample_on_even_s, style_mpl_3d, style_plot,
)
from linac_gen_gui.interphase.tabs.results_tab import (
    _Curve, _Image, _Panel, _PopupPlot, _mk_mode_combo,
)

_PLANES = (("x", "x – x′"), ("y", "y – y′"), ("z", "φ – W"))
_PLANE_LABEL = {"x": "x – x′ (dispersion-corrected)", "y":
                "y – y′ (dispersion-corrected)", "z": "φ – W",
                "x_raw": "x – x′ (raw)", "y_raw": "y – y′ (raw)"}
# Level-line colours: the dark-mode categorical set blue / aqua / orange
# (all-pairs colour-vision-deficiency safe on the dark plot surface), each
# placed where it contrasts with the magma image under it — the 1 % line
# runs over red-orange, 0.1 % over magenta, 0.01 % over purple.  The
# outermost particle is a neutral dotted annotation line.
_LEVEL_COLORS = {1.0: "#3987e5", 0.1: "#199e70", 0.01: "#d95926"}
_OUTER_COLOR = "#f8fafc"
_CMAP_START = 0.12
# Display-grid limits (the stored data are never resampled).
_MIN_COLS, _MAX_COLS = 1000, 3000
_MAX_ROWS, _MIN_ROWS, _MAX_CELLS = 800, 200, 12_000_000
_S3D, _N3D = 100, 120
# The map and the ε_n strip share the s axis pixel for pixel: same left
# axis width, and the strip reserves the colour bar's column on its right.
_AXIS_W, _CBAR_W = 76, 72
_N_ELL = "n = ε_ellipse / ε_rms"
_ENABLE = ("it is recorded by multi-particle runs made with “Record halo "
           "action scan” ticked on the Numerics tab (or with "
           "record_action_scan=True / --action-scan)")


def _fmt_pct(p: float) -> str:
    return f"{p:g} %"


def _nlen(x) -> int:
    try:
        return 0 if x is None else len(x)
    except TypeError:
        return 0


def _colorbar(cmap):
    kw = dict(values=(-3.0, 2.0), colorMap=cmap, label="beam outside",
              interactive=False, width=14)
    try:                              # colorMapMenu: pyqtgraph >= 0.13.5
        return pg.ColorBarItem(colorMapMenu=False, **kw)
    except TypeError:
        return pg.ColorBarItem(**kw)


def _elastic(label: QLabel) -> QLabel:
    """A one-line readout that never forces its window wider."""
    label.setSizePolicy(QSizePolicy.Policy.Ignored,
                        QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(1)
    return label


class HaloActionScanPopup(_PopupPlot):
    """Map, slice and 3D views of the recorded halo action scan."""

    def __init__(self, parent, state=None):
        super().__init__(parent, "Halo action scan  —  beam outside "
                         "n·ε_rms ellipses", size=(1200, 860))
        self._state = state if state is not None else getattr(
            parent, "state", None)
        self._results = None
        self._scan = None
        self._scan_src = None
        self._scan_key = None
        self._views: dict = {}          # plane -> view (one n cap each)
        self._s_mono = None
        self._step = 0
        self._keep_s = None             # position to restore after a reload
        self._fig = self._canvas = self._ax3d = None
        self._mpl_cmap = None
        self._mpl_error = None
        self._dirty_3d = True

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)

        # ── controls ────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("plane:"))
        self._plane_cb = QComboBox()
        for key, label in _PLANES:
            self._plane_cb.addItem(label, userData=key)
        self._plane_cb.currentIndexChanged.connect(self._on_view_changed)
        ctrl.addWidget(self._plane_cb)
        ctrl.addSpacing(8)
        ctrl.addWidget(QLabel("Display:"))
        self._mode_cb = _mk_mode_combo(self._on_view_changed)
        self._mode_cb.blockSignals(True)
        self._mode_cb.setCurrentIndex(1)            # betatron by default
        self._mode_cb.blockSignals(False)
        ctrl.addWidget(self._mode_cb)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("n max:"))
        self._nmax = QDoubleSpinBox()
        self._nmax.setRange(0.0, 400.0)
        self._nmax.setDecimals(1)
        self._nmax.setSingleStep(5.0)
        self._nmax.setSpecialValueText("auto")
        self._nmax.setValue(0.0)
        self._nmax.setKeyboardTracking(False)       # one redraw per entry
        self._nmax.setToolTip(
            "Top of the n axis (ellipse size in units of the local ε_rms).  "
            "auto = 1.1 × the 95th percentile of the outermost particle's n "
            "along the line (at least 20), so one stray particle does not "
            "squash the map.")
        self._nmax.valueChanged.connect(self._on_view_changed)
        ctrl.addWidget(self._nmax)
        ctrl.addSpacing(12)
        self._chk_lat = QCheckBox("lattice")
        self._chk_lat.setChecked(True)
        self._chk_lat.toggled.connect(self.set_lattice_strip_visible)
        ctrl.addWidget(self._chk_lat)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color:{theme.TEXT_2}; padding:2px 0;")
        self._hint.setVisible(False)
        v.addWidget(self._hint)

        self._tabs = QTabWidget()
        v.addWidget(self._tabs, stretch=1)

        # ── Map page: map block over slice block, in a splitter ─────────
        page = QWidget()
        pv = QVBoxLayout(page)
        pv.setContentsMargins(0, 4, 0, 0)
        pv.setSpacing(4)
        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(4)

        # No in-plot legend: the image fills the plot, so a legend would
        # hide data — the key row above the map names the lines instead
        # (the curves keep their names for Ctrl+S export).
        self._map = pg.PlotWidget()
        style_plot(self._map, _N_ELL, "")
        self._map.setMinimumHeight(220)             # the main display
        self._image = pg.ImageItem(axisOrder="row-major")
        self._image.setZValue(-10)
        self._map.addItem(self._image)
        self._cmap = opaque_sequential_colormap("magma", _CMAP_START)
        self._image.setLookupTable(self._cmap.getLookupTable(0.0, 1.0, 256))
        self._cbar = _colorbar(self._cmap)
        self._cbar.setFixedWidth(_CBAR_W)
        self._cbar.setImageItem(self._image,
                                insert_in=self._map.getPlotItem())
        self._map.getAxis("left").setWidth(_AXIS_W)
        # room above the plot for the colour bar's top ("100 %") label
        self._map.getPlotItem().layout.setContentsMargins(1, 9, 1, 1)
        self._level_curves = {}
        self._gauss_lines = {}
        for i, lvl in enumerate(LEVELS_PCT):
            col = _LEVEL_COLORS[lvl]
            self._level_curves[lvl] = self._map.plot(
                [], [], pen=curve_pen(col, 2.0), connect="finite",
                name=f"{_fmt_pct(lvl)} outside")
            self._gauss_lines[lvl] = self._map.plot(
                [], [], pen=pg.mkPen(col, width=1.2,
                                     style=Qt.PenStyle.DashLine),
                name="Gaussian levels (dashed)" if i == 0 else None)
        self._outer_curve = self._map.plot(
            [], [], pen=pg.mkPen(_OUTER_COLOR, width=1.4,
                                 style=Qt.PenStyle.DotLine),
            connect="finite", name="outermost particle")
        self._cursor = pg.InfiniteLine(
            0.0, angle=90, movable=False,
            pen=pg.mkPen(_OUTER_COLOR, width=1, style=Qt.PenStyle.DashDotLine))
        self._map.addItem(self._cursor)

        from linac_gen_gui.interphase.plots.lattice_strip import (
            make_lattice_strip)
        self._lattice_strip = make_lattice_strip(
            page, self._map, getattr(self._state, "lattice", None))
        if self._state is not None and hasattr(self._state, "lattice_changed"):
            try:
                self._state.lattice_changed.connect(self._on_lattice_changed)
            except Exception:                                # noqa: BLE001
                pass
        key = QLabel(
            "  ".join(f"<span style='color:{_LEVEL_COLORS[lvl]}'>━━</span>"
                      f" {_fmt_pct(lvl)} outside" for lvl in LEVELS_PCT)
            + f"  <span style='color:{_OUTER_COLOR}'>┈┈</span> outermost "
            "particle  ·  dashed: the same levels for a Gaussian beam "
            + "(n = " + " / ".join(f"{gaussian_level_n(lvl):.2f}"
                                   for lvl in LEVELS_PCT) + ")")
        key.setTextFormat(Qt.TextFormat.RichText)
        key.setStyleSheet(f"color:{theme.TEXT_1};")
        tv.addWidget(_elastic(key))
        tv.addWidget(self._lattice_strip)
        tv.addWidget(self._map, stretch=3)

        # normalized rms emittance of the plane on display; x-linked to the
        # map, so its s axis needs no title of its own.  Two-line label:
        # it fits the short plot.
        self._eps_plot = pg.PlotWidget()
        style_plot(self._eps_plot, "", "", xlabel="", xunits="")
        self._eps_plot.setLabel(
            "left", f"<span style='color:{theme.TEXT_0};'>ε_n<br>"
                    f"mm·mrad</span>")
        self._eps_plot.setMinimumHeight(105)
        self._eps_plot.getAxis("left").setWidth(_AXIS_W)
        _lay = self._eps_plot.getPlotItem().layout
        _pad = pg.GraphicsWidget()
        _pad.setFixedWidth(_CBAR_W)
        _lay.addItem(_pad, 2, 5)            # where the map's colour bar sits
        _lay.setColumnFixedWidth(4, 5)
        self._eps_plot.setXLink(self._map)
        self._eps_curve = self._eps_plot.plot(
            [], [], pen=curve_pen(theme.ACCENT, 2.0), connect="finite",
            name="ε_rms, normalized")
        self._eps_cursor = pg.InfiniteLine(
            0.0, angle=90, movable=False,
            pen=pg.mkPen(_OUTER_COLOR, width=1, style=Qt.PenStyle.DashDotLine))
        self._eps_plot.addItem(self._eps_cursor)
        tv.addWidget(self._eps_plot, stretch=1)

        self._hover = QLabel(" ")
        self._hover.setStyleSheet(
            f"color:{theme.TEXT_1}; font-family:{theme.FONT_MONO};")
        self._hover.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        tv.addWidget(_elastic(self._hover))

        row = QHBoxLayout()
        row.addWidget(QLabel("position:"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_step_changed)
        row.addWidget(self._slider, stretch=3)
        self._step_label = QLabel("")
        self._step_label.setStyleSheet(
            f"color:{theme.TEXT_1}; font-family:{theme.FONT_MONO};")
        row.addWidget(_elastic(self._step_label), stretch=2)
        tv.addLayout(row)

        bottom_w = QWidget()
        bottom_w.setMinimumHeight(150)
        bottom = QHBoxLayout(bottom_w)
        bottom.setContentsMargins(0, 0, 0, 0)
        self._slice = pg.PlotWidget()
        style_plot(self._slice, "beam outside", "%", xlabel=_N_ELL, xunits="")
        add_legend(self._slice)
        self._slice.setLogMode(x=False, y=True)
        self._slice_curve = self._slice.plot(
            [], [], pen=curve_pen(theme.ACCENT, 2.0), connect="finite",
            name="this position")
        self._slice_gauss = self._slice.plot(
            [], [], pen=pg.mkPen(theme.TEXT_2, width=1.4,
                                 style=Qt.PenStyle.DashLine),
            connect="finite", name="Gaussian 100·e^(−n/2)")
        self._slice_nmax = pg.InfiniteLine(
            0.0, angle=90, movable=False,
            pen=pg.mkPen(_OUTER_COLOR, width=1.2, style=Qt.PenStyle.DotLine))
        self._slice.addItem(self._slice_nmax)
        bottom.addWidget(self._slice, stretch=3)
        self._info = QLabel("")
        self._info.setTextFormat(Qt.TextFormat.RichText)
        self._info.setWordWrap(True)
        self._info.setAlignment(Qt.AlignmentFlag.AlignTop
                                | Qt.AlignmentFlag.AlignLeft)
        self._info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._info.setStyleSheet(
            f"color:{theme.TEXT_1}; font-family:{theme.FONT_MONO};"
            f" background:{theme.BG_INSET}; padding:8px;")
        bottom.addWidget(self._info, stretch=2)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(bottom_w)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([560, 250])
        pv.addWidget(split, stretch=1)
        self._map_page = page
        self._tabs.addTab(page, "Map")

        # ── 3D page (matplotlib, built on first view) ───────────────────
        self._page3d = QWidget()
        self._page3d_layout = QVBoxLayout(self._page3d)
        self._page3d_layout.setContentsMargins(0, 4, 0, 0)
        self._tabs.addTab(self._page3d, "3D")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Hover readout + click-to-select on the map.  Bound methods; the
        # proxy and the scene both belong to this window.
        self._hover_proxy = pg.SignalProxy(
            self._map.scene().sigMouseMoved, rateLimit=30,
            slot=self._on_map_hover)
        self._map.scene().sigMouseClicked.connect(self._on_map_click)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def refresh(self, results) -> None:
        self._results = results
        try:
            self._load(results)
            self._redraw()
        except Exception as exc:                             # noqa: BLE001
            # A malformed file or an unforeseen state must not break the
            # window; say what happened instead of showing a stale map.
            traceback.print_exc()
            self._fail(exc)

    def _fail(self, exc) -> None:
        """Forget the scan (every slot then no-ops) and say why."""
        self._scan = self._s_mono = None
        self._scan_src = self._scan_key = None      # retry on next refresh
        self._views = {}
        self._show_empty(f"Could not display the action scan: "
                         f"{type(exc).__name__}: {exc}")

    def _load(self, results) -> None:
        key = _nlen(getattr(results, "s", None)) if results is not None else -1
        if results is self._scan_src and key == self._scan_key:
            return
        prev_s = self._keep_s
        if self._scan is not None and self._scan.s.size:
            prev_s = float(self._scan.s[min(self._step,
                                             self._scan.s.size - 1)])
        self._keep_s = None
        self._scan_src, self._scan_key = results, key
        self._views = {}
        self._dirty_3d = True
        self._scan = (action_scan_from_results(results)
                      if results is not None else None)
        sc = self._scan
        if sc is None or sc.s.size == 0:
            self._s_mono = None
            self._keep_s = prev_s        # e.g. a live preview passes through
            return
        self._s_mono = np.maximum.accumulate(sc.s)
        step = 0 if prev_s is None else self._nearest_record(prev_s)
        self._slider.blockSignals(True)
        self._slider.setRange(0, sc.s.size - 1)
        self._slider.setValue(step)
        self._slider.blockSignals(False)
        self._step = step

    def closeEvent(self, ev) -> None:                        # noqa: N802
        """Release the data of a closed window: its display arrays can be
        hundreds of MB, and it would pin the previous results in memory
        (closed popups are not refreshed).  ``_open_popup`` refreshes it
        on the next open; the position is kept.  (Not on hide: a window
        minimised with its parent must stay live.)"""
        if self._scan is not None and self._scan.s.size:
            self._keep_s = float(self._scan.s[min(self._step,
                                                  self._scan.s.size - 1)])
        self._results = self._scan = self._s_mono = None
        self._scan_src = self._scan_key = None
        self._views = {}
        super().closeEvent(ev)

    def _empty_reason(self, results) -> str:
        if results is None:
            return f"No results yet — {_ENABLE}."
        src = str(getattr(results, "source_path", "") or "")
        error = getattr(results, "action_scan_error", None)
        if error:
            return f"This file's action scan could not be read: {error}."
        if (src.lower().endswith(".opmd.h5")
                and getattr(results, "sigma_matrix", None) is None):
            # openPMD files written before 2026-09-27 (no per-step beam
            # matrix either) never carried the scan
            return ("This openPMD file predates the action scan in openPMD "
                    "files — import the HELIX .h5 file written next to it.")
        if getattr(results, "direction", None) == "backward":
            return "Backtracked runs do not record the action scan."
        if getattr(results, "n_macro", None) is None:
            return ("The action scan is recorded by single-bunch "
                    f"multi-particle runs only — {_ENABLE}.")
        if src:
            return ("This results file has no action scan (its run was made "
                    f"without it) — {_ENABLE}.")
        return f"This run did not record the action scan — {_ENABLE}."

    def _record_at(self, s_mm: float) -> int:
        """Last recorded step at or before ``s_mm`` (what the map shows)."""
        sm = self._s_mono
        i = int(np.searchsorted(sm, s_mm, side="right")) - 1
        return int(min(max(i, 0), sm.size - 1))

    def _nearest_record(self, s_mm: float) -> int:
        """Recorded step nearest to ``s_mm`` (the last one at that s)."""
        sm = self._s_mono
        j = int(np.searchsorted(sm, s_mm, side="left"))
        cand = [c for c in (j - 1, j) if 0 <= c < sm.size]
        best = min(cand, key=lambda c: abs(sm[c] - s_mm))
        return int(np.searchsorted(sm, sm[best], side="right")) - 1

    def _plane(self) -> str:
        base = self._plane_cb.currentData() or "x"
        if base in ("x", "y") and self._mode_cb.currentIndex() == 0:
            return f"{base}_raw"
        return base

    def _undefined(self, plane: str) -> np.ndarray:
        # no rms ellipse; for z also the DC records (ActionScan decides)
        return self._scan._undefined(plane)

    def _plane_view(self, plane: str) -> dict:
        """Display arrays for ``plane`` at the current n cap (one cached
        view per plane)."""
        user_cap = float(self._nmax.value())
        hit = self._views.get(plane)
        if hit is not None and hit[0] == user_cap:
            return hit[1]
        sc = self._scan
        grid = sc.n
        bad = self._undefined(plane)
        n_max = np.where(bad, np.nan, sc.n_max[plane].astype(float))
        if user_cap > 0.0:
            cap = min(user_cap, float(grid[-1]))
        else:
            fin = n_max[np.isfinite(n_max)]
            cap = (float(np.clip(1.1 * np.percentile(fin, 95.0), 20.0,
                                 grid[-1])) if fin.size
                   else float(min(50.0, grid[-1])))
        n_steps = sc.s.size
        n_rows = int(np.clip(_MAX_CELLS // max(n_steps, 1),
                             _MIN_ROWS, _MAX_ROWS))
        n_lo = np.linspace(0.0, cap, n_rows, endpoint=False)
        j = np.clip(np.searchsorted(grid, n_lo, side="right") - 1, 0,
                    grid.size - 1)
        na = sc.n_alive.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = (100.0 * sc.counts[plane][:, j].astype(np.float32)
                   / na[:, None])
            logp = np.where(pct > 0.0, np.log10(pct), np.nan).astype(
                np.float32)
        logp[bad, :] = np.nan
        # Columns: an even s grid.  A column holding recorded steps shows
        # the largest value among them; an empty column (between two
        # records) holds the last step recorded before it.
        img, s0, s1 = resample_on_even_s(logp, sc.s, pool="max",
                                         min_cols=_MIN_COLS,
                                         max_cols=_MAX_COLS)
        n0 = float(max(int(sc.n_alive.max()), 1))
        view = {
            "cap": cap, "img": img.T, "rect": (s0, s1),
            "lo": math.log10(100.0 / n0), "n_max": n_max,
            "levels": {lvl: sc.level_crossing(plane, lvl)
                       for lvl in LEVELS_PCT},
            "eps_n": np.where(bad, np.nan, sc.eps_n[plane]),
        }
        self._views[plane] = (user_cap, view)
        return view

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _show_empty(self, text: str) -> None:
        self._hint.setText(text)
        self._hint.setVisible(True)
        self._image.clear()
        for c in (*self._level_curves.values(), *self._gauss_lines.values(),
                  self._outer_curve, self._eps_curve, self._slice_curve,
                  self._slice_gauss):
            c.setData([], [])
        for item in (self._cursor, self._eps_cursor, self._slice_nmax,
                     self._cbar):
            item.setVisible(False)
        self._info.setText("")
        self._hover.setText(" ")
        self._step_label.setText("")
        self._slider.blockSignals(True)
        self._slider.setRange(0, 0)
        self._slider.blockSignals(False)
        self._dirty_3d = True
        if self._ax3d is not None:
            self._ax3d.clear()
            self._canvas.draw()

    def _redraw(self) -> None:
        sc = self._scan
        if sc is None or sc.s.size == 0:
            self._show_empty(self._empty_reason(self._results))
            return
        plane = self._plane()
        self._mode_cb.setEnabled(plane != "z")
        if plane not in sc.counts:
            self._show_empty(f"This action scan has no {_PLANE_LABEL[plane]} "
                             f"data (planes: {', '.join(sc.planes)}).")
            return
        self._hint.setVisible(False)
        if self._slider.maximum() != sc.s.size - 1:     # after an empty state
            self._slider.blockSignals(True)
            self._slider.setRange(0, sc.s.size - 1)
            self._slider.setValue(min(self._step, sc.s.size - 1))
            self._slider.blockSignals(False)
        view = self._plane_view(plane)
        s0, s1 = view["rect"]
        cap, lo = view["cap"], view["lo"]
        self._image.setImage(view["img"], autoLevels=False)
        self._image.setRect(pg.QtCore.QRectF(s0, 0.0, s1 - s0, cap))
        self._cbar.setLevels((lo, 2.0))
        ticks = [(float(t), _fmt_pct(10.0 ** t))
                 for t in range(int(math.ceil(lo)), 3)]
        self._cbar.axis.setTicks([ticks])
        s = sc.s
        for lvl in LEVELS_PCT:
            self._level_curves[lvl].setData(s, view["levels"][lvl])
            g = gaussian_level_n(lvl)
            self._gauss_lines[lvl].setData([s0, s1], [g, g])
        self._outer_curve.setData(s, view["n_max"])
        self._map.setXRange(s0, s1, padding=0.01)
        self._map.setYRange(0.0, cap, padding=0.0)
        self._eps_curve.setData(s, view["eps_n"])
        self._eps_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        for item in (self._cursor, self._eps_cursor, self._cbar):
            item.setVisible(True)
        self._update_step()
        if self._tabs.currentWidget() is self._page3d:
            self._draw_3d()
        else:
            self._dirty_3d = True

    def _update_step(self) -> None:
        sc = self._scan
        if sc is None or sc.s.size == 0 or self._plane() not in sc.counts:
            return
        i = int(min(max(self._step, 0), sc.s.size - 1))
        plane = self._plane()
        view = self._plane_view(plane)
        s_i = float(sc.s[i])
        self._cursor.setValue(s_i)
        self._eps_cursor.setValue(s_i)
        name = sc.element[i] if sc.element else ""
        self._step_label.setText(
            f"s = {s_i / 1000.0:.3f} m  {name}  ({i + 1} / {sc.s.size})")

        undefined = bool(self._undefined(plane)[i])
        na = int(sc.n_alive[i])
        cap, lo = view["cap"], view["lo"]
        if undefined or na <= 0:
            self._slice_curve.setData([], [])
        else:
            pct = 100.0 * sc.counts[plane][i].astype(float) / na
            self._slice_curve.setData(sc.n, np.where(pct > 0.0, pct, np.nan))
        n_g = np.linspace(0.0, cap, 400)
        g = gaussian_percent_outside(n_g)
        self._slice_gauss.setData(n_g, np.where(g >= 10.0 ** (lo - 1.0), g,
                                                np.nan))
        nmax_i = view["n_max"][i]
        self._slice_nmax.setVisible(bool(np.isfinite(nmax_i)))
        if np.isfinite(nmax_i):
            self._slice_nmax.setValue(float(nmax_i))
        self._slice.setXRange(0.0, cap, padding=0.02)
        self._slice.setYRange(lo - 0.3, 2.1, padding=0.0)
        self._info.setText(self._info_html(i, plane, view, undefined))

    def _info_html(self, i: int, plane: str, view: dict,
                   undefined: bool) -> str:
        sc = self._scan
        na = int(sc.n_alive[i])
        n_launch = getattr(self._results, "n_macro", None) or int(
            sc.n_alive.max())
        lines = [f"<b>{_PLANE_LABEL[plane]}</b>",
                 f"particles here: {na:,} ({100.0 * na / max(n_launch, 1):.2f} %"
                 f" of {int(n_launch):,})"]
        if undefined:
            if plane == "z" and bool(sc.continuous[i]):
                lines.append("DC (unbunched) beam at this position: φ–W has "
                             "no rms ellipse to measure against.")
            else:
                lines.append("No rms ellipse at this position (fewer than 3 "
                             "particles, a degenerate plane or non-finite "
                             "moments).")
            return "<br>".join(lines)
        eps, eps_n = float(sc.eps[plane][i]), float(sc.eps_n[plane][i])
        geo = "deg·MeV" if plane == "z" else "mm·mrad"
        lines.append(f"ε_rms: {eps_n:.5g} mm·mrad normalized · "
                     f"{eps:.5g} {geo} geometric")
        nm = float(view["n_max"][i])
        lines.append(f"outermost particle: n = {nm:.4g} "
                     f"({math.sqrt(max(nm, 0.0)):.3g} rms sizes)")
        for lvl in LEVELS_PCT:
            gauss = f"(Gaussian {gaussian_level_n(lvl):.2f})"
            n_l = view["levels"][lvl][i]
            if lvl / 100.0 * na < 1.0:
                lines.append(f"{_fmt_pct(lvl)} level: not resolved with "
                             f"{na:,} particles {gauss}")
            elif not np.isfinite(n_l):
                lines.append(f"more than {_fmt_pct(lvl)} outside even at "
                             f"n = {sc.n[-1]:g} {gauss}")
            else:
                lines.append(f"fewer than {_fmt_pct(lvl)} outside beyond "
                             f"n = {n_l:g} {gauss}")
        if (plane == "z" and sc.periodic_phase is False
                and bool(np.any(sc.continuous))):
            lines.append("<i>This beam was bunched from a DC beam without "
                         "periodic phase coordinates: φ spans the bunch "
                         "train, so φ–W describes the train, not one "
                         "bunch.</i>")
        return "<br>".join(lines)

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------
    def _ensure_3d(self) -> bool:
        if self._ax3d is not None:
            return True
        if self._mpl_error is not None:
            return False
        try:
            import matplotlib
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.colors import ListedColormap
            from matplotlib.figure import Figure
        except Exception as exc:                             # noqa: BLE001
            self._mpl_error = str(exc)
            note = QLabel(f"The 3D view needs matplotlib ({exc}).")
            note.setStyleSheet(f"color:{theme.TEXT_2};")
            self._page3d_layout.addWidget(note)
            return False
        self._fig = Figure(figsize=(9.0, 6.5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        with matplotlib.rc_context(mpl_dark_rc()):
            self._ax3d = self._fig.add_subplot(111, projection="3d")
        self._ax3d.view_init(elev=24, azim=-58)
        cmap = ListedColormap(matplotlib.colormaps["magma"](
            np.linspace(_CMAP_START, 1.0, 256)))
        cmap.set_under(theme.BG_2)          # the floor: nothing outside
        self._mpl_cmap = cmap
        note = QLabel("drag to rotate · right-drag to zoom · same data as the "
                      "map (columns: largest value in each s block)")
        note.setStyleSheet(f"color:{theme.TEXT_2};")
        self._page3d_layout.addWidget(note)
        self._page3d_layout.addWidget(self._canvas, stretch=1)
        return True

    def _surface(self, plane: str):
        """Coarse (n, s, log10 %) surface from the map's display grid."""
        view = self._plane_view(plane)
        img = view["img"]                              # (rows = n, cols = s)
        n_rows, n_cols = img.shape
        s0, s1 = view["rect"]
        c_edges = np.unique(np.linspace(0, n_cols, _S3D + 1).astype(int)[:-1])
        z = np.fmax.reduceat(img, c_edges, axis=1)     # max over s blocks
        r_idx = np.unique(np.linspace(0, n_rows - 1, _N3D).astype(int))
        z = z[r_idx, :]
        n_vals = r_idx * (view["cap"] / n_rows)
        c_mid = c_edges + 0.5 * np.diff(np.r_[c_edges, n_cols])
        s_vals = (s0 + (s1 - s0) * c_mid / n_cols) / 1000.0
        floor = view["lo"] - 0.5
        return n_vals, s_vals, np.where(np.isfinite(z), z, floor), floor

    def _draw_3d(self) -> None:
        if not self._ensure_3d():
            return
        import matplotlib
        ax = self._ax3d
        elev, azim = ax.elev, ax.azim
        with matplotlib.rc_context(mpl_dark_rc()):
            ax.clear()
            sc = self._scan
            plane = self._plane()
            if sc is not None and sc.s.size and plane in sc.counts:
                view = self._plane_view(plane)
                n_vals, s_vals, z, floor = self._surface(plane)
                N, S = np.meshgrid(n_vals, s_vals)
                ax.plot_surface(N, S, z.T, cmap=self._mpl_cmap,
                                vmin=view["lo"], vmax=2.0, rstride=1,
                                cstride=1, linewidth=0, antialiased=False,
                                shade=False)
                ax.set_xlim(0.0, view["cap"])
                ax.set_zlim(floor, 2.0)
                ticks = list(range(int(math.ceil(view["lo"])), 3))
                ax.set_zticks(ticks)
                ax.set_zticklabels([_fmt_pct(10.0 ** t) for t in ticks])
                ax.set_xlabel(_N_ELL)
                ax.set_ylabel("s (m)")
                ax.set_zlabel("beam outside")
                ax.set_title(_PLANE_LABEL[plane])
            ax.view_init(elev=elev, azim=azim)
            style_mpl_3d(self._fig, ax)
        self._canvas.draw()
        self._dirty_3d = False

    # ------------------------------------------------------------------
    # slots — every one survives a bad scan (the error path forgets it)
    # ------------------------------------------------------------------
    def _on_view_changed(self, *_args) -> None:
        if self._scan is None:
            return
        try:
            self._redraw()
        except Exception as exc:                             # noqa: BLE001
            traceback.print_exc()
            self._fail(exc)

    def _on_step_changed(self, value: int) -> None:
        self._step = int(value)
        if self._scan is None:
            return
        try:
            self._update_step()
        except Exception as exc:                             # noqa: BLE001
            traceback.print_exc()
            self._fail(exc)

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._page3d and self._dirty_3d:
            try:
                self._draw_3d()
            except Exception as exc:                         # noqa: BLE001
                traceback.print_exc()
                self._fail(exc)

    def _on_map_hover(self, evt) -> None:
        sc = self._scan
        if sc is None or sc.s.size == 0:
            return
        try:
            pos = evt[0]
            vb = self._map.getPlotItem().vb
            if not vb.sceneBoundingRect().contains(pos):
                return
            pt = vb.mapSceneToView(pos)
            self._hover.setText(self._hover_text(float(pt.x()),
                                                 float(pt.y())))
        except Exception:                                    # noqa: BLE001
            # 30 events/s: never flood the excepthook from a readout
            self._hover.setText(" ")

    def _hover_text(self, s_mm: float, n: float) -> str:
        sc = self._scan
        plane = self._plane()
        i = self._record_at(s_mm)
        grid = sc.n
        j = int(np.searchsorted(grid, n, side="right")) - 1
        where = f"s = {sc.s[i] / 1000.0:.3f} m"
        if sc.element:
            where += f" {sc.element[i]}"
        if plane not in sc.counts:
            return f"{where} · no {_PLANE_LABEL[plane]} data"
        if j < 0 or n > grid[-1]:
            return f"{where} · n outside the recorded range 0–{grid[-1]:g}"
        if bool(self._undefined(plane)[i]) or sc.n_alive[i] <= 0:
            return f"{where} · no rms ellipse at this position"
        cnt = int(sc.counts[plane][i, j])
        pct = 100.0 * cnt / int(sc.n_alive[i])
        eps_n = float(sc.eps_n[plane][i])
        return (f"{where} · outside n = {grid[j]:g}: {pct:.4g} % "
                f"({cnt:,} particles) · ε_rms,n = {eps_n:.4g} mm·mrad · "
                f"ellipse n·ε_rms,n = {grid[j] * eps_n:.4g} mm·mrad "
                f"(√n = {math.sqrt(grid[j]):.3g} rms sizes)")

    def _on_map_click(self, ev) -> None:
        if self._scan is None or self._s_mono is None:
            return
        try:
            if ev.button() != Qt.MouseButton.LeftButton:
                return
            vb = self._map.getPlotItem().vb
            pos = ev.scenePos()
            if not vb.sceneBoundingRect().contains(pos):
                return
            self._slider.setValue(self._nearest_record(
                float(vb.mapSceneToView(pos).x())))
        except RuntimeError:                  # widget torn down mid-event
            pass

    # ------------------------------------------------------------------
    # Ctrl+S: the exact recorded data of the plane on display
    # ------------------------------------------------------------------
    def _extra_panels(self) -> list[_Panel]:
        sc = self._scan
        if sc is None or sc.s.size == 0:
            return []
        plane = self._plane()
        if plane not in sc.counts:
            return []
        s_m = sc.s / 1000.0
        extent = (float(sc.n[0]), float(sc.n[-1]), float(s_m[0]),
                  float(s_m[-1]))
        return [_Panel(
            label=f"halo_action_scan_{plane}",
            xlabel="s (m)", ylabel="",
            curves=[
                _Curve("eps_rms_normalized_mm_mrad", s_m, sc.eps_n[plane]),
                _Curve("eps_rms_geometric", s_m, sc.eps[plane]),
                _Curve("n_max_outermost", s_m, sc.n_max[plane]),
                _Curve("n_alive", s_m, sc.n_alive.astype(float)),
                _Curve("continuous_dc", s_m, sc.continuous.astype(float)),
                _Curve("no_rms_ellipse", s_m,
                       self._undefined(plane).astype(float)),
                _Curve("n_grid", np.arange(sc.n.size, dtype=float), sc.n),
            ],
            images=[
                _Image("count_outside__rows_s__cols_n", sc.counts[plane],
                       extent),
                _Image("percent_outside__rows_s__cols_n",
                       sc.percent_outside(plane), extent),
            ])]
