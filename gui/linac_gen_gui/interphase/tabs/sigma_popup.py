"""The Results tab's Σ-matrix panel: the 21 independent terms of the beam's
6×6 second-moment matrix along s, and the whole matrix at one position.

The numbers come from :mod:`linac_gen.analysis.sigma_terms`.  The panels
are laid out as the lower triangle of the matrix: rms values on the
diagonal, correlations r_ij (or covariances) below it.  Undefined values
(continuous-beam phase terms, zero variances, rows without particles) are
gaps.  Click a panel to move the position; double-click it to see that
term full width with the lattice strip.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSlider,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from linac_gen.analysis import sigma_terms as st
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.app_settings import make_settings
from linac_gen_gui.interphase.plots.plot_style import (
    configure_pyqtgraph_defaults, curve_pen, fix_plot_range, rearm_autorange,
    release_plot_range, style_plot,
)
from linac_gen_gui.interphase.tabs.results_tab import _PopupPlot

#: x, y, z: the Results tab's plane colours; terms between planes stand out
PLANE_COLORS = (theme.ACCENT, "#a3e635", theme.WARN)
COUPLING_COLOR = theme.MAG
CURSOR_COLOR = "#facc15"
NAN_RGBA = (58, 74, 94, 255)                        # theme.BORDER_0
_R_FRAME = (-1.05, 1.05)
_QUANTITY_KEY = "linac_gen/sigma_terms/quantity"
_BASIS_KEY = "linac_gen/sigma_terms/basis"
NO_RESULTS = "No results yet: run a simulation."
NO_SIGMA = ("These results have no Σ matrix: results files written before "
            "2026-09-27, Sacherer envelope runs and bunch-train files do not "
            "record one.")


def term_color(i: int, j: int) -> str:
    if st.PLANE[i] != st.PLANE[j]:
        return COUPLING_COLOR
    return PLANE_COLORS[st.PLANE[i]]


def term_title(basis: str, kind: str, i: int, j: int) -> str:
    """``σ(x) [mm]``, ``r(x, x′)`` or ``⟨x x′⟩ [mm·mrad]``."""
    c, u = st.COORDS[basis], st.UNITS[basis]
    if i == j:
        return f"σ({c[i]}) [{u[i]}]"
    if kind == "corr":
        return f"r({c[j]}, {c[i]})"
    return f"⟨{c[j]} {c[i]}⟩ [{u[j]}·{u[i]}]"


def _num(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}g}" if np.isfinite(v) else "—"


def _where(t, row: int) -> str:
    """The position of a row: "s = … m" along a line, "turn N" for a ring
    run turn by turn."""
    if getattr(t, "domain", "s") == "turn":
        return f"turn {int(round(float(t.s[row])))}"
    return f"s = {float(t.s[row]):.4f} m"


class SigmaTermsPopup(_PopupPlot):
    """All 21 terms along s, the matrix at the chosen position, and a
    full-width view of one term."""

    def __init__(self, parent, state):
        super().__init__(parent, "Σ matrix along s — the 21 independent terms",
                         size=(1480, 900))
        configure_pyqtgraph_defaults()
        self._state = state
        self._results = None
        self._cache: dict = {}
        self._row = None
        self._pos_s = None                  # s (m) of the chosen position
        self._focus_term = None
        self._settings = make_settings("linac_gen", "linac_gen_gui")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Quantity:"))
        self._quantity = QComboBox()
        self._quantity.addItem("correlation r", userData="corr")
        self._quantity.addItem("covariance Σ (with units)", userData="cov")
        top.addWidget(self._quantity)
        top.addSpacing(16)
        top.addWidget(QLabel("Basis:"))
        self._basis = QComboBox()
        self._basis.addItem("(x, x′, y, y′, Δφ, ΔW) — as recorded",
                            userData="ours")
        self._basis.addItem("(x, x′, y, y′, z, δ) — TraceWin",
                            userData="tracewin")
        top.addWidget(self._basis)
        top.addStretch(1)
        root.addLayout(top)
        # Remembered choices: read before connecting, so restoring them
        # does not count as a change.
        self._restore_choice(self._quantity, _QUANTITY_KEY)
        self._restore_choice(self._basis, _BASIS_KEY)
        self._quantity.currentIndexChanged.connect(self._on_quantity)
        self._basis.currentIndexChanged.connect(self._on_basis)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color:{theme.TEXT_2}; font-size:12px;")
        root.addWidget(self._note)

        split = QSplitter(Qt.Orientation.Horizontal)
        self._stack = QStackedWidget()
        split.addWidget(self._stack)
        self._build_grid()
        self._build_focus()
        split.addWidget(self._build_matrix_side())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 480])
        root.addWidget(split, stretch=1)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("position:"))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider)
        bottom.addWidget(self._slider, stretch=3)
        self._pos_label = QLabel("")
        self._pos_label.setStyleSheet(
            f"color:{theme.TEXT_1}; font-family:{theme.FONT_MONO};")
        self._pos_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        bottom.addWidget(self._pos_label, stretch=2)
        root.addLayout(bottom)
        self._hover = QLabel(" ")
        self._hover.setStyleSheet(
            f"color:{theme.TEXT_1}; font-family:{theme.FONT_MONO};")
        self._hover.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._hover)
        self._draw()

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def _build_grid(self) -> None:
        self._grid = pg.GraphicsLayoutWidget()
        self._grid.setBackground(theme.BG_INSET)
        self._plots: dict = {}
        self._curves: dict = {}
        self._cursors: list = []
        tick_font = QFont()
        tick_font.setPointSize(8)
        anchor = None
        for i, j in st.TERMS:
            p = self._grid.addPlot(row=i, col=j)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.hideButtons()
            for side in ("left", "bottom"):
                ax = p.getAxis(side)
                ax.setTextPen(theme.TEXT_1)
                ax.setPen(theme.BORDER_0)
                ax.setTickFont(tick_font)
                ax.enableAutoSIPrefix(False)
            p.getAxis("left").setWidth(46)
            if i < 5:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "s (m)", color=theme.TEXT_1)
            curve = p.plot([], [], pen=curve_pen(term_color(i, j), width=1.3),
                           connect="finite")
            # 21 curves of a long run (a run with sub-step rows has tens of
            # thousands): draw only what the panel can show.  The export
            # reads the full rows (_collect_panels), not the drawn ones.
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            if i != j:
                p.addItem(pg.InfiniteLine(
                    0.0, angle=0, movable=False,
                    pen=pg.mkPen(theme.TEXT_3, width=1,
                                 style=Qt.PenStyle.DotLine)), ignoreBounds=True)
            cursor = pg.InfiniteLine(0.0, angle=90, movable=False,
                                     pen=pg.mkPen(CURSOR_COLOR, width=1.6))
            cursor.setVisible(False)
            p.addItem(cursor, ignoreBounds=True)
            if anchor is None:
                anchor = p
            else:
                p.setXLink(anchor)
            self._plots[(i, j)] = p
            self._curves[(i, j)] = curve
            self._cursors.append(cursor)
        self._key = self._grid.addLabel("", row=0, col=2, rowspan=2, colspan=4)
        self._stack.addWidget(self._grid)
        # hover readout and click-to-select; bound methods, own scene
        self._grid_proxy = pg.SignalProxy(self._grid.scene().sigMouseMoved,
                                          rateLimit=30, slot=self._on_grid_hover)
        self._grid.scene().sigMouseClicked.connect(self._on_grid_click)

    def _build_focus(self) -> None:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        back = QPushButton("◀  all 21 terms")
        back.clicked.connect(self._show_grid)
        head.addWidget(back)
        self._focus_title = QLabel("")
        self._focus_title.setStyleSheet(f"color:{theme.TEXT_0}; font-size:13px;")
        head.addWidget(self._focus_title, stretch=1)
        lay.addLayout(head)
        self._focus = pg.PlotWidget()
        style_plot(self._focus, "", "", xlabel="s", xunits="m")
        self._focus_curve = self._focus.plot([], [], pen=curve_pen(theme.ACCENT),
                                             connect="finite")
        self._focus_curve.setClipToView(True)
        self._focus_curve.setDownsampling(auto=True, method="peak")
        self._focus_zero = pg.InfiniteLine(
            0.0, angle=0, movable=False,
            pen=pg.mkPen(theme.TEXT_3, width=1, style=Qt.PenStyle.DotLine))
        self._focus.addItem(self._focus_zero, ignoreBounds=True)
        self._focus_cursor = pg.InfiniteLine(0.0, angle=90, movable=False,
                                             pen=pg.mkPen(CURSOR_COLOR, width=1.6))
        self._focus.addItem(self._focus_cursor, ignoreBounds=True)
        # The plot is nested in the stack, out of reach of
        # attach_lattice_strip: build the strip here (s in m).
        from linac_gen_gui.interphase.plots.lattice_strip import (
            make_lattice_strip)
        self._lattice_strip = make_lattice_strip(
            page, self._focus, getattr(self._state, "lattice", None), scale=1e-3)
        if self._state is not None and hasattr(self._state, "lattice_changed"):
            try:
                self._state.lattice_changed.connect(self._on_lattice_changed)
            except Exception:                                # noqa: BLE001
                pass
        lay.addWidget(self._lattice_strip)
        lay.addWidget(self._focus, stretch=1)
        self._stack.addWidget(page)
        self._focus_proxy = pg.SignalProxy(self._focus.scene().sigMouseMoved,
                                           rateLimit=30, slot=self._on_focus_hover)
        self._focus.scene().sigMouseClicked.connect(self._on_focus_click)

    def _build_matrix_side(self) -> QWidget:
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(6)
        title = QLabel("At the position: correlations r (colour) and Σ")
        title.setStyleSheet(f"color:{theme.TEXT_0}; font-size:13px;")
        lay.addWidget(title)

        self._mat_gl = pg.GraphicsLayoutWidget()
        self._mat_gl.setBackground(theme.BG_INSET)
        mp = self._mat_gl.addPlot(row=0, col=0)
        mp.setMenuEnabled(False)
        mp.hideButtons()
        mp.setMouseEnabled(False, False)
        mp.invertY(True)
        mp.setAspectLocked(True)
        mp.showAxis("top")
        mp.hideAxis("bottom")
        for side_name in ("left", "top"):
            ax = mp.getAxis(side_name)
            ax.setTextPen(theme.TEXT_0)
            ax.setPen(theme.BORDER_0)
        fix_plot_range(mp, xrange=(0.0, 6.0), yrange=(0.0, 6.0), padding=0.02)
        self._mat_plot = mp
        self._mat_img = pg.ImageItem(axisOrder="row-major")
        mp.addItem(self._mat_img)
        try:
            self._cmap = pg.colormap.get("CET-D1A")
        except Exception:                                    # noqa: BLE001
            self._cmap = pg.colormap.get("magma")
        cbar = pg.ColorBarItem(colorMap=self._cmap, values=(-1.0, 1.0),
                               label="r", interactive=False,
                               orientation="v", width=12)
        self._mat_gl.addItem(cbar, row=0, col=1)
        self._cbar = cbar
        self._mat_text = {}
        for i in range(6):
            for j in range(6):
                t = pg.TextItem("", anchor=(0.5, 0.5))
                t.setPos(j + 0.5, i + 0.5)
                mp.addItem(t)
                self._mat_text[(i, j)] = t
        lay.addWidget(self._mat_gl, stretch=3)

        self._table = QTableWidget(6, 6)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self._table.setFont(mono)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self._table, stretch=2)
        self._table_note = QLabel("Σ in the units of its row × column; the "
                                  "diagonal holds the variances.")
        self._table_note.setWordWrap(True)
        self._table_note.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        lay.addWidget(self._table_note)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._summary.setStyleSheet(
            f"color:{theme.TEXT_1}; font-family:{theme.FONT_MONO}; font-size:11px;")
        lay.addWidget(self._summary)
        return side

    # ------------------------------------------------------------------
    # export (Ctrl+S)
    # ------------------------------------------------------------------
    def _collect_panels(self):
        """Every term at full resolution (the panels draw downsampled),
        the Σ table and the correlation matrix at the position."""
        from linac_gen_gui.interphase.tabs.results_tab import (
            _Curve, _Image, _Panel)
        t = self._terms()
        if t is None or len(t) == 0:
            return []
        basis, kind = self._basis_key(), self._kind()
        panels = []
        for i, j in st.TERMS:
            title = term_title(basis, kind, i, j)
            panels.append(_Panel(label=title,
                                 xlabel="turn" if t.domain == "turn" else "s (m)", ylabel=title,
                                 curves=[_Curve(name=title, x=np.asarray(t.s),
                                                y=np.asarray(self._values(
                                                    t, kind, i, j)))]))
        row = len(t) - 1 if self._row is None else self._row
        where = _where(t, row)
        heads = [f"{name} ({u})" for name, u in zip(st.COORDS[basis],
                                                     st.UNITS[basis])]
        idx = np.arange(6, dtype=float)
        panels.append(_Panel(label=f"Σ at {where}", xlabel="row", ylabel="Σ",
                             curves=[_Curve(name=h, x=idx, y=t.sigma[row][:, k])
                                     for k, h in enumerate(heads)]))
        panels.append(_Panel(label=f"correlations r at {where}",
                             images=[_Image(name="r", data=np.array(t.corr[row]),
                                            extent=(0.0, 6.0, 0.0, 6.0))]))
        return panels

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def refresh(self, results) -> None:
        domain = getattr(results, "x_domain", "s")
        old = getattr(self, "_terms_domain", "s")
        kept = self.__dict__.setdefault("_pos_kept", {})
        if self._results is not None and self._pos_s is not None:
            # (the position chosen on this run, for when it is shown again)
            kept[old] = (self._results, self._pos_s)
        if results is not None:
            if domain != old or self._results is None:
                # another axis, or results back after a clear: the position
                # chosen on these same results, else the end
                back = kept.get(domain)
                self._pos_s = back[1] if back is not None and back[0] is results else None
            self._terms_domain = domain
        self._results = results
        self._cache = {}
        self._draw()

    def _redraw(self) -> None:
        self._draw()

    def _basis_key(self) -> str:
        return self._basis.currentData() or "ours"

    def _kind(self) -> str:
        return self._quantity.currentData() or "corr"

    def _terms(self):
        basis = self._basis_key()
        if basis not in self._cache:
            self._cache[basis] = (None if self._results is None
                                  else st.sigma_terms(self._results, basis))
        return self._cache[basis]

    @staticmethod
    def _values(t, kind: str, i: int, j: int) -> np.ndarray:
        if i == j:
            return t.rms[:, i]
        return t.corr[:, i, j] if kind == "corr" else t.sigma[:, i, j]

    def _draw(self) -> None:
        basis, kind = self._basis_key(), self._kind()
        t = self._terms()
        self._set_titles(basis, kind)
        if t is None or len(t) == 0:
            for c in self._curves.values():
                c.setData([], [])
            for cur in self._cursors:
                cur.setVisible(False)
            self._focus_curve.setData([], [])
            self._focus_cursor.setVisible(False)
            self._slider.blockSignals(True)
            self._slider.setRange(0, 0)
            self._slider.blockSignals(False)
            self._row = None
            if self._results is None:       # a new lattice: start at its end
                self._pos_s = None
            self._pos_label.setText("")
            self._hover.setText(" ")
            self._clear_matrix()
            self._note.setText(NO_RESULTS if self._results is None else NO_SIGMA)
            return
        for (i, j), curve in self._curves.items():
            curve.setData(t.s, self._values(t, kind, i, j),
                          name=term_title(basis, kind, i, j))
            p = self._plots[(i, j)]
            if i != j and kind == "corr":
                fix_plot_range(p, yrange=_R_FRAME, padding=0.0)
            else:
                release_plot_range(p)
        if self._focus_term is not None:
            self._draw_focus()
        n = len(t)
        self._slider.blockSignals(True)
        self._slider.setRange(0, n - 1)
        self._slider.blockSignals(False)
        # the same s as before (another basis or quantity, a new run of the
        # same lattice); the end of the lattice the first time
        row = n - 1 if self._pos_s is None else self._nearest_row(self._pos_s)
        self._note.setText(self._notes(t))
        self._set_row(row)

    def _set_titles(self, basis: str, kind: str) -> None:
        for (i, j), p in self._plots.items():
            p.setTitle(term_title(basis, kind, i, j), size="9pt",
                       color=term_color(i, j))
        c = st.COORDS[basis]
        below = ("correlations r(column, row), from −1 to 1" if kind == "corr"
                 else "covariances ⟨column row⟩ with units")
        swatch = "".join(
            f"<span style='color:{col}'>━</span> {name}&nbsp;&nbsp;"
            for col, name in zip(PLANE_COLORS, ("x", "y", "z")))
        self._key.setText(
            f"<span style='color:{theme.TEXT_0}'>diagonal: rms values; "
            f"below it: {below}</span><br>"
            f"<span style='color:{theme.TEXT_2}'>in one plane: {swatch}"
            f"<span style='color:{COUPLING_COLOR}'>━</span> between planes"
            f"</span><br><span style='color:{theme.TEXT_2}'>click a panel: "
            f"position · double-click: full width</span>", size="9pt")
        ticks = [[(k + 0.5, name) for k, name in enumerate(c)]]
        for side_name in ("left", "top"):
            self._mat_plot.getAxis(side_name).setTicks(ticks)
        units = st.UNITS[basis]
        # two lines across (narrow columns), one line down
        self._table.setHorizontalHeaderLabels(
            [f"{name}\n{u}" for name, u in zip(c, units)])
        self._table.setVerticalHeaderLabels(
            [f"{name} ({u})" for name, u in zip(c, units)])

    def _notes(self, t) -> str:
        res = self._results
        if t.domain == "turn":
            parts = ["Ring run: the covariance of the circulating particles, turn after "
                     "turn at the start of the ring (s = 0)."]
            if t.continuous.all():
                parts.append("A coasting beam (no RF): no Δφ terms.")
            return "  ".join(parts)
        parts = [("Multiparticle run: covariance of the particles the "
                  "statistics describe." if hasattr(res, "transmission")
                  else "Envelope run: the propagated Σ.")]
        exits = getattr(res, "element_exit_idx", None)
        if exits is not None and len(exits) and len(t) <= len(exits) + 2:
            parts.append("Rows at element exits only: the terms inside "
                         "solenoids and field maps (where a solenoid couples "
                         "x and y′) need Numerics → Record per-sub-step.")
        if t.continuous.any():
            rows = np.flatnonzero(t.continuous)
            if rows[-1] - rows[0] + 1 == rows.size and rows[0] == 0:
                parts.append(f"The beam is continuous up to s = "
                             f"{t.s[rows[-1]]:.3f} m: no Δφ terms there.")
            else:
                parts.append(f"The beam is continuous on {rows.size} rows: "
                             f"no Δφ terms there.")
        if t.basis == "tracewin":
            if not t.converted:
                parts.append("No reference-particle data in these results: "
                             "the z and δ terms are not available.")
            else:
                parts.append("z = −βλ·Δφ/360, so correlations with z have the "
                             "opposite sign of those with Δφ.")
        else:
            freq = getattr(res, "ref_frequency", None)
            try:
                f = np.asarray(freq, dtype=float).ravel()
            except (TypeError, ValueError):
                f = np.empty(0)
            if f.size == len(t) and f.size and np.nanmax(f) != np.nanmin(f):
                parts.append("Δφ is in degrees of the local RF frequency, so "
                             "its scale changes at a frequency jump; z in the "
                             "TraceWin basis does not.")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # position and the matrix at it
    # ------------------------------------------------------------------
    def _nearest_row(self, s_m: float) -> int:
        t = self._terms()
        s = t.s
        k = int(np.searchsorted(s, s_m))
        if k <= 0:
            return 0
        if k >= s.size:
            return s.size - 1
        return k if abs(s[k] - s_m) < abs(s_m - s[k - 1]) else k - 1

    def _on_slider(self, value: int) -> None:
        try:
            self._set_row(int(value))
        except RuntimeError:                                 # torn down
            pass

    def _set_row(self, row: int) -> None:
        t = self._terms()
        if t is None or len(t) == 0:
            return
        row = int(np.clip(row, 0, len(t) - 1))
        self._row = row
        s = float(t.s[row])
        self._pos_s = s
        for cur in self._cursors:
            cur.setValue(s)
            cur.setVisible(True)
        self._focus_cursor.setValue(s)
        self._focus_cursor.setVisible(True)
        if self._slider.value() != row:
            self._slider.blockSignals(True)
            self._slider.setValue(row)
            self._slider.blockSignals(False)
        name = t.element[row]
        self._pos_label.setText(f"{_where(t, row)}   {name}   row {row} of "
                                f"{len(t) - 1}")
        self._show_matrix(t, row)

    def _set_matrix_image(self, rgba: np.ndarray) -> None:
        """Colours as given (no levels).  The rect goes AFTER the image: set
        before, it is computed for an empty image and the first cell's
        colour fills the whole matrix."""
        self._mat_img.setImage(rgba, autoLevels=False, levels=None)
        self._mat_img.setRect(QRectF(0.0, 0.0, 6.0, 6.0))

    def _clear_matrix(self) -> None:
        self._set_matrix_image(np.zeros((6, 6, 4), dtype=np.uint8))
        for item in self._mat_text.values():
            item.setText("")
        for i in range(6):
            for j in range(6):
                self._table.setItem(i, j, QTableWidgetItem(""))
        self._summary.setText("")

    def _show_matrix(self, t, row: int) -> None:
        r = t.corr[row]
        finite = np.isfinite(r)
        rgba = self._cmap.map(np.clip((np.where(finite, r, 0.0) + 1.0) / 2.0,
                                      0.0, 1.0), mode="byte")
        rgba = np.array(rgba, dtype=np.uint8)
        rgba[~finite] = NAN_RGBA
        self._set_matrix_image(rgba)
        for (i, j), item in self._mat_text.items():
            if i == j:
                text = _num(float(t.rms[row, i]), 3)
            else:
                text = f"{r[i, j]:+.2f}" if finite[i, j] else "—"
            dark = finite[i, j] and abs(r[i, j]) < 0.55
            item.setText(text, color=theme.BG_0 if dark else "#ffffff")
        S = t.sigma[row]
        for i in range(6):
            for j in range(6):
                cell = QTableWidgetItem(_num(float(S[i, j])))
                cell.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                cell.setForeground(QBrush(QColor(
                    term_color(i, j) if i != j else theme.TEXT_0)))
                self._table.setItem(i, j, cell)
        lines = []
        for p in st.plane_summary(t, row):
            (a, b), (ua, ub) = p["coords"], p["units"]
            lines.append(
                f"{p['plane']}: σ({a}) {_num(p['rms'][0])} {ua} · "
                f"σ({b}) {_num(p['rms'][1])} {ub} · ε {_num(p['eps'])} "
                f"{p['eps_units']} · α {_num(p['alpha'])} · "
                f"β {_num(p['beta'])} {p['beta_units']}")
        self._summary.setText("<br>".join(lines))

    # ------------------------------------------------------------------
    # mouse
    # ------------------------------------------------------------------
    def _plot_at(self, pos):
        for key, p in self._plots.items():
            if p.vb.sceneBoundingRect().contains(pos):
                return key, p
        return None, None

    def _readout(self, key, s_m: float) -> str:
        t = self._terms()
        i, j = key
        k = self._nearest_row(s_m)
        v = float(self._values(t, self._kind(), i, j)[k])
        return (f"{_where(t, k)}  {t.element[k]}  ·  "
                f"{term_title(t.basis, self._kind(), i, j)} = {_num(v)}")

    def _on_grid_hover(self, evt) -> None:
        t = self._terms()
        if t is None or len(t) == 0:
            return
        try:
            key, p = self._plot_at(evt[0])
            if key is not None:
                s_m = float(p.vb.mapSceneToView(evt[0]).x())
                self._hover.setText(self._readout(key, s_m))
        except Exception:                                    # noqa: BLE001
            # 30 events/s: never flood the excepthook from a readout
            self._hover.setText(" ")

    def _on_grid_click(self, ev) -> None:
        t = self._terms()
        if t is None or len(t) == 0:
            return
        try:
            if ev.button() != Qt.MouseButton.LeftButton:
                return
            key, p = self._plot_at(ev.scenePos())
            if key is None:
                return
            if ev.double():
                self._show_focus(*key)
            else:
                self._set_row(self._nearest_row(
                    float(p.vb.mapSceneToView(ev.scenePos()).x())))
        except RuntimeError:                     # widget torn down mid-event
            pass

    def _on_focus_hover(self, evt) -> None:
        t = self._terms()
        if t is None or len(t) == 0 or self._focus_term is None:
            return
        try:
            vb = self._focus.getPlotItem().vb
            if vb.sceneBoundingRect().contains(evt[0]):
                self._hover.setText(self._readout(
                    self._focus_term, float(vb.mapSceneToView(evt[0]).x())))
        except Exception:                                    # noqa: BLE001
            self._hover.setText(" ")

    def _on_focus_click(self, ev) -> None:
        t = self._terms()
        if t is None or len(t) == 0:
            return
        try:
            if ev.button() != Qt.MouseButton.LeftButton:
                return
            vb = self._focus.getPlotItem().vb
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                self._set_row(self._nearest_row(
                    float(vb.mapSceneToView(ev.scenePos()).x())))
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # one term full width
    # ------------------------------------------------------------------
    def _show_focus(self, i: int, j: int) -> None:
        self._focus_term = (i, j)
        self._stack.setCurrentIndex(1)
        rearm_autorange(self._focus)
        self._draw_focus()

    def _draw_focus(self) -> None:
        t = self._terms()
        if t is None or self._focus_term is None:
            self._focus_curve.setData([], [])
            return
        i, j = self._focus_term
        kind = self._kind()
        title = term_title(t.basis, kind, i, j)
        self._focus_title.setText(title)
        self._focus_curve.setPen(curve_pen(term_color(i, j)))
        self._focus_curve.setData(t.s, self._values(t, kind, i, j), name=title)
        self._focus_zero.setVisible(i != j)
        if i != j and kind == "corr":
            fix_plot_range(self._focus, yrange=_R_FRAME, padding=0.0)
        else:
            release_plot_range(self._focus)
        if self._row is not None:
            self._focus_cursor.setValue(float(t.s[self._row]))

    def _show_grid(self) -> None:
        # the hidden full-width curve would otherwise be exported twice
        self._focus_term = None
        self._focus_curve.setData([], [])
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # choices
    # ------------------------------------------------------------------
    def _restore_choice(self, combo: QComboBox, key: str) -> None:
        try:
            value = self._settings.value(key, None)
        except Exception:                                    # noqa: BLE001
            value = None
        if value is not None:
            idx = combo.findData(str(value))
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _save_choice(self, combo: QComboBox, key: str) -> None:
        try:
            self._settings.setValue(key, combo.currentData())
        except Exception:                                    # noqa: BLE001
            pass

    def _on_quantity(self, _index: int) -> None:
        self._save_choice(self._quantity, _QUANTITY_KEY)
        self._redraw_new()

    def _on_basis(self, _index: int) -> None:
        self._save_choice(self._basis, _BASIS_KEY)
        self._redraw_new()
