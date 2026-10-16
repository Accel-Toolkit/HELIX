"""The Results tab's RFQ panels: one popup per tile of the RFQ section.

Geometry and design panels need only the loaded lattice; the run panels add
the run's values at every cell exit when the results belong to that lattice
(matched by cell name).  The numbers come from :mod:`linac_gen.analysis.
rfq_design`; nothing here builds a field map or solves a vane profile.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QSlider,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from linac_gen.analysis import rfq_design as rd
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.plots.plot_style import (
    _html_label, add_legend, curve_pen, rearm_autorange, style_plot,
)
from linac_gen_gui.interphase.tabs.results_tab import _PopupPlot

C_X = "#22d3ee"          # x plane / first quantity
C_Y = "#f472b6"          # y plane / second quantity
C_3 = "#a3e635"          # third quantity
C_REF = theme.TEXT_3     # reference lines (r0, limits)


def _dash(color, width=1.4):
    return pg.mkPen(color, width=width, style=Qt.PenStyle.DashLine)


def _dot(color, width=1.2):
    return pg.mkPen(color, width=width, style=Qt.PenStyle.DotLine)


def _vanes_of(src):
    return ("the vane file's vanes" if src == "vane file"
            else "the vanes built from the RFQ_CELL cards")


def _x_label(p, text, units):
    """Relabel the bottom axis in the style ``style_plot`` gave it.  Not
    ``style_plot`` again: on a plot that already has a range, its
    ``enableAutoSIPrefix(False)`` makes pyqtgraph recompute the SI scale
    from that range (×0.001 for an RFQ in mm) and the ticks go wrong."""
    ax = p.getAxis("bottom")
    ax.setLabel(_html_label(text), units=units or None, **ax.labelStyle)


# ---------------------------------------------------------------------------
# context: everything the panels read, built from the state and the results
# ---------------------------------------------------------------------------
class RfqContext:
    """The loaded lattice's RFQ with its per-cell table and electrodes."""

    def __init__(self, lattice, chain, table, species, f_MHz, current_mA,
                 duty_pct):
        self.lattice = lattice
        self.chain = chain
        self.table = table
        self.species = species
        self.f_MHz = f_MHz
        self.current_mA = current_mA
        self.duty_pct = duty_pct
        self._solver = None
        self._cards = "unset"

    @property
    def has_run(self) -> bool:
        return not self.table.get("run_note")

    def solver(self):
        """(electrodes, scale, source, note) of the time-domain solver."""
        if self._solver is None:
            self._solver = rd.solver_electrodes(self.lattice, self.chain)
        return self._solver

    def cards(self):
        if self._cards == "unset":
            self._cards = rd.cards_electrodes(self.lattice, self.chain)
        return self._cards

    def n_chains_note(self) -> str:
        n = int(getattr(self.chain, "n_chains", 1) or 1)
        return (f"The lattice has {n} RFQ cell chains; these panels show the "
                "first." if n > 1 else "")

    def electrodes(self):
        """The best electrodes to draw: (electrodes, label, note)."""
        el, _scale, src, note = self.solver()
        if el is not None:
            from pathlib import Path
            label = (f"vane file {Path(str(self.lattice.rfq_geom_file)).name}"
                     if src == "vane file" else "the RFQ_CELL cards")
            return el, label, ""
        cards = self.cards()
        why = f"the time-domain solver cannot use the vanes ({note})"
        return cards, "the RFQ_CELL cards", why


def rfq_context(state, results) -> RfqContext | None:
    """The RFQ context of the state's lattice, or None without an RFQ."""
    lat = getattr(state, "lattice", None)
    if lat is None:
        return None
    chain = rd.find_rfq(lat)
    if chain is None:
        return None
    bc = getattr(state, "beam_config", None)
    species = str(getattr(bc, "species", "proton") or "proton")
    f_MHz = rd.rfq_frequency(lat, chain, results,
                             default_MHz=getattr(bc, "frequency", None))
    table = rd.cell_table(lat, species=species, frequency_MHz=f_MHz,
                          results=results)
    current = getattr(bc, "current", None)
    duty = float(getattr(bc, "duty_cycle", 100.0) or 100.0)
    return RfqContext(lat, chain, table, species, f_MHz,
                      None if current is None else float(current), duty)


# ---------------------------------------------------------------------------
# base popup
# ---------------------------------------------------------------------------
class _RfqPopup(_PopupPlot):
    """Note line, empty state and stacked plots with linked s axes; the
    first plot carries the lattice strip.  The s plots can be drawn against
    the cell number instead (cell k centred on k)."""

    def __init__(self, parent, state, title, size=(1050, 760)):
        super().__init__(parent, title, size=size)
        self._state = state
        self._results = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet(f"color:{theme.TEXT_2}; font-size:12px;")
        top.addWidget(self._note, stretch=1)
        self._cell_axis = QCheckBox("x: cell number")
        self._cell_axis.setToolTip("Plot against the cell number (cell k "
                                   "centred on k) instead of s along the "
                                   "lattice.")
        self._cell_axis.toggled.connect(self._on_cell_axis)
        self._cell_axis.hide()                  # shown with a first s plot
        top.addWidget(self._cell_axis, alignment=Qt.AlignmentFlag.AlignTop)
        lay.addLayout(top)
        self._empty = QLabel("")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(f"color:{theme.TEXT_3}; font-size:14px;"
                                  "padding:40px;")
        self._empty.hide()
        lay.addWidget(self._empty)
        self._plots: list[pg.PlotWidget] = []
        self._extra_widgets: list = []
        # The panels draw the lattice: redraw an open panel after the
        # lattice is edited (coalesced; a new lattice also refreshes the
        # results, which redraws at once and cancels this).
        self._lattice_timer = QTimer(self)
        self._lattice_timer.setSingleShot(True)
        self._lattice_timer.setInterval(150)
        self._lattice_timer.timeout.connect(self._redraw_for_lattice)
        if hasattr(state, "lattice_changed"):
            state.lattice_changed.connect(self._lattice_edited)

    def _lattice_edited(self, _lattice) -> None:
        try:
            self._lattice_timer.start()
        except RuntimeError:                          # popup destroyed
            pass

    def _redraw_for_lattice(self) -> None:
        try:
            if self.isVisible():
                self.refresh(getattr(self._state, "results", None))
        except RuntimeError:
            pass

    def _add_plot(self, ylabel, yunits="", *, xlabel="s", xunits="mm",
                  link=True, stretch=1) -> pg.PlotWidget:
        p = pg.PlotWidget()
        style_plot(p, ylabel, yunits, xlabel=xlabel, xunits=xunits)
        p._rfq_s_axis = xlabel == "s"
        if p._rfq_s_axis:
            self._cell_axis.show()
        add_legend(p)
        self.layout().addWidget(p, stretch)
        if self._plots and link:
            p.setXLink(self._plots[0])
        if not self._plots:
            self._plots.append(p)
            self.attach_lattice_strip(p)
        else:
            self._plots.append(p)
        return p

    def _s_plots(self):
        return [p for p in self._plots if getattr(p, "_rfq_s_axis", False)]

    def _show_empty(self, msg: str) -> None:
        for w in self._plots + self._extra_widgets:
            w.hide()
        if self._lattice_strip is not None:
            self._lattice_strip.hide()
        self._cell_axis.hide()
        self._note.setText("")
        self._empty.setText(msg)
        self._empty.show()

    def _show_body(self) -> None:
        self._empty.hide()
        for w in self._plots + self._extra_widgets:
            w.show()
        cells = self._cell_axis.isChecked()
        if self._lattice_strip is not None:
            self._lattice_strip.setVisible(not cells)  # the strip is in s
        self._cell_axis.setVisible(bool(self._s_plots()))

    def refresh(self, results):
        self._lattice_timer.stop()
        self._results = results
        for p in self._plots:
            p.clear()
        for p in self._s_plots():
            _x_label(p, "s", "mm")
        ctx = rfq_context(self._state, results)
        if ctx is None:
            self._show_empty("No RFQ cells in the current lattice")
            return
        self._show_body()
        self._draw(ctx, results)
        if self._cell_axis.isChecked() and self._empty.isHidden():
            self._to_cell_axis(ctx)
        self._tidy_legends()

    def _on_cell_axis(self, _on) -> None:
        """A new x axis: a new plot, fitted to its data."""
        rearm_autorange(self)
        self.refresh(self._results)

    def _to_cell_axis(self, ctx) -> None:
        """Redraw the s plots against the cell number: s maps linearly
        through each cell onto [k − ½, k + ½]."""
        edges = np.concatenate([ctx.chain.s0[:1], ctx.chain.s1])
        number = np.arange(edges.size) + 0.5

        def to_cell(x):
            return np.interp(np.asarray(x, float), edges, number)
        for p in self._s_plots():
            for it in list(p.getPlotItem().items):
                if isinstance(it, pg.BarGraphItem):
                    x = np.asarray(it.opts["x"], float)
                    w = np.asarray(it.opts["width"], float)
                    a, b = to_cell(x - 0.5 * w), to_cell(x + 0.5 * w)
                    it.setOpts(x=0.5 * (a + b), width=0.9 * (b - a))
                elif isinstance(it, pg.PlotDataItem) and \
                        it.xData is not None and it.xData.size:
                    it.setData(x=to_cell(it.xData), y=it.yData)
            _x_label(p, "cell", "")

    def _tidy_legends(self) -> None:
        """Hide the legend of a plot that has nothing named in it."""
        for p in self._plots:
            leg = getattr(p.getPlotItem(), "legend", None)
            if leg is not None:
                leg.setVisible(bool(getattr(leg, "items", None)))

    def _draw(self, ctx: RfqContext, results) -> None:      # pragma: no cover
        raise NotImplementedError

    # helpers ----------------------------------------------------------------
    @staticmethod
    def _cell_steps(p, ctx, values, color, name, pen=None):
        """A per-cell value as a step curve over the cells' lattice s."""
        edges = np.concatenate([ctx.chain.s0[:1], ctx.chain.s1])
        v = np.asarray(values, float)
        if not np.any(np.isfinite(v)):
            return None
        return p.plot(edges, v, stepMode="center", connect="finite",
                      pen=pen or curve_pen(color), name=name)

    @staticmethod
    def _at_exits(p, ctx, values, color, name, *, symbol=None, pen=None):
        v = np.asarray(values, float)
        if not np.any(np.isfinite(v)):
            return None
        kw = dict(pen=pen if pen is not None else curve_pen(color), name=name,
                  connect="finite")
        if symbol:
            kw.update(symbol=symbol, symbolSize=5, symbolBrush=color,
                      symbolPen=None)
        return p.plot(ctx.chain.s1, v, **kw)

    def _run_note(self, ctx) -> str:
        if ctx.has_run:
            return "Run values at each cell exit."
        note = ctx.table.get("run_note") or "no run"
        return ("No run values yet." if note == "no run"
                else f"No run values: {note}.")


# ---------------------------------------------------------------------------
# 1  vane geometry
# ---------------------------------------------------------------------------
class RfqGeometryPopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ vane geometry — vane tips "
                         "and the beam along the RFQ", size=(1100, 780))
        self._top = self._add_plot("x", "mm")
        self._side = self._add_plot("y", "mm")

    def _draw(self, ctx, results):
        el, label, why = ctx.electrodes()
        t = ctx.table
        notes = [f"Vanes: {label}."]
        if why:
            notes.append(f"({why}.)")
        for p, kp, km, color, plane in ((self._top, "R_x", "R_mx", C_X, "x"),
                                        (self._side, "R_y", "R_my", C_Y, "y")):
            if el is not None:
                tips = rd.tips_along(el)
                s = ctx.chain.to_s(tips["z"])
                p.plot(s, tips[kp], pen=curve_pen(color),
                       name=f"+{plane} vane tip")
                p.plot(s, -tips[km], pen=curve_pen(color),
                       name=f"−{plane} vane tip")
            cards = ctx.cards()
            if cards is not None and el is not None and cards is not el and \
                    getattr(ctx.lattice, "rfq_geom_file", None):
                ct = rd.tips_along(cards)
                s = ctx.chain.to_s(ct["z"])
                p.plot(s, ct[kp], pen=_dash(theme.TEXT_2, 1.0),
                       name="cards' tips")
                p.plot(s, -ct[km], pen=_dash(theme.TEXT_2, 1.0))
            self._cell_steps(p, ctx, t["r0"], C_REF, "r0",
                             pen=_dot(C_REF))
            self._cell_steps(p, ctx, -t["r0"], C_REF, None, pen=_dot(C_REF))
            mx, sg = t.get(f"{plane}_max"), t.get(f"sigma_{plane}")
            if mx is not None:
                self._at_exits(p, ctx, mx, C_3, f"beam: largest |{plane}|",
                               pen=curve_pen(C_3, 1.2))
                self._at_exits(p, ctx, -np.asarray(mx), C_3, None,
                               pen=curve_pen(C_3, 1.2))
            if sg is not None:
                self._at_exits(p, ctx, 3 * np.asarray(sg), C_3, f"beam: 3σ_{plane}",
                               pen=_dash(C_3, 1.2))
                self._at_exits(p, ctx, -3 * np.asarray(sg), C_3, None,
                               pen=_dash(C_3, 1.2))
        if getattr(ctx.lattice, "rfq_geom_file", None):
            notes.append("Dashed grey: the tips the RFQ_CELL cards describe.")
        notes.append(self._run_note(ctx))
        self._note.setText("  ".join(notes))


# ---------------------------------------------------------------------------
# 2  cross-section
# ---------------------------------------------------------------------------
class RfqSectionPopup(_RfqPopup):
    _STEP_MM = 0.1

    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ vane cross-section",
                         size=(820, 860))
        row = QHBoxLayout()
        self._cell = QSpinBox()
        self._cell.setPrefix("cell ")
        self._cell.setMinimum(1)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._pos = QLabel("")
        self._pos.setStyleSheet(f"color:{theme.TEXT_1}; font-size:12px;")
        row.addWidget(self._cell)
        row.addWidget(self._slider, stretch=1)
        self.layout().addLayout(row)
        self.layout().addWidget(self._pos)
        self._extra_widgets += [self._cell, self._slider, self._pos]
        self._plot = pg.PlotWidget()
        style_plot(self._plot, "y", "mm", xlabel="x", xunits="mm")
        add_legend(self._plot)
        self._plot.setAspectLocked(True)
        self.layout().addWidget(self._plot, stretch=1)
        self._extra_widgets.append(self._plot)
        self._img = pg.ImageItem(axisOrder="row-major")
        lut = np.zeros((256, 4), np.uint8)
        lut[:, :3] = 150
        lut[:, 3] = np.linspace(0, 235, 256).astype(np.uint8)
        self._img.setLookupTable(lut)
        self._ctx = None
        self._el = None
        self._slider.valueChanged.connect(self._on_slider)
        self._cell.valueChanged.connect(self._on_cell)
        self._syncing = False

    def _draw(self, ctx, results):
        el, label, why = ctx.electrodes()
        self._ctx, self._el = ctx, el
        if el is None:
            self._show_empty("The vanes cannot be drawn: " + (why or "no "
                             "electrodes"))
            return
        z0, z1 = el.z_span
        self._syncing = True
        self._slider.setRange(int(np.ceil(z0 / self._STEP_MM)),
                              int(np.floor(z1 / self._STEP_MM)))
        self._cell.setMaximum(len(ctx.chain.cells))
        self._syncing = False
        self._note.setText(f"Vanes: {label}." + (f" ({why}.)" if why else "")
                           + "  Metal in grey; dashed circle: the smallest "
                           "tip distance at this z; dotted: r0.")
        if self._slider.value() <= self._slider.minimum():
            mid = 0.5 * (ctx.chain.edges[0] + ctx.chain.edges[-1])
            self._syncing = True
            self._slider.setValue(int(round(mid / self._STEP_MM)))
            self._syncing = False
        self._render()

    def _on_slider(self, _v):
        if not self._syncing:
            self._render()

    def _on_cell(self, k):
        if self._syncing or self._ctx is None:
            return
        e = self._ctx.chain.edges
        k = int(np.clip(k, 1, len(e) - 1))
        z = 0.5 * (e[k - 1] + e[k])
        self._slider.setValue(int(round(z / self._STEP_MM)))

    def _render(self):
        ctx, el = self._ctx, self._el
        if ctx is None or el is None:
            return
        z = self._slider.value() * self._STEP_MM
        k = int(np.searchsorted(ctx.chain.edges, z, side="right"))
        k = int(np.clip(k, 1, len(ctx.chain.cells)))
        self._syncing = True
        self._cell.setValue(k)
        self._syncing = False
        tips_here = rd.tips_along(el, [z])
        rmax = max(float(tips_here[n][0]) for n in ("R_x", "R_y", "R_mx", "R_my"))
        half = float(np.clip(1.8 * rmax + 2.0 * float(tips_here["rho_x"][0]),
                             6.0, 40.0))
        xs, ys, metal, tips = rd.cross_section(el, z, half, n=301)
        self._plot.clear()
        self._img.setImage(metal.astype(np.uint8) * 255, levels=(0, 255))
        self._img.setRect(QRectF(float(xs[0]), float(ys[0]),
                                 float(xs[-1] - xs[0]), float(ys[-1] - ys[0])))
        self._plot.addItem(self._img)
        a = min(tips[n] for n in ("R_x", "R_y", "R_mx", "R_my"))
        th = np.linspace(0, 2 * np.pi, 361)
        self._plot.plot(a * np.cos(th), a * np.sin(th), pen=_dash(C_3),
                        name=f"smallest tip distance {a:.3f} mm")
        r0 = float(ctx.table["r0"][k - 1])
        self._plot.plot(r0 * np.cos(th), r0 * np.sin(th), pen=_dot(C_REF),
                        name=f"r0 {r0:.3f} mm")
        self._plot.setXRange(-half, half, padding=0.0)
        self._plot.setYRange(-half, half, padding=0.0)
        name = ctx.chain.names[k - 1]
        self._pos.setText(
            f"z = {z:.1f} mm in the RFQ (s = {ctx.chain.to_s(z):.1f} mm), "
            f"{name} (cell {k}, type {int(ctx.table['type'][k - 1])}):  "
            f"+x tip {tips['R_x']:.3f}, −x {tips['R_mx']:.3f}, "
            f"+y {tips['R_y']:.3f}, −y {tips['R_my']:.3f} mm;  "
            f"tip radius {tips['rho_x']:.3f} mm")

    def refresh(self, results):
        self._plot.clear()
        super().refresh(results)


# ---------------------------------------------------------------------------
# 3  modulation and aperture
# ---------------------------------------------------------------------------
class RfqModulationPopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ modulation and aperture per cell",
                         size=(1050, 820))
        self._m = self._add_plot("modulation m")
        self._ap = self._add_plot("aperture", "mm")
        self._len = self._add_plot("cell length", "mm")

    def _draw(self, ctx, results):
        t = ctx.table
        self._cell_steps(self._m, ctx, t["m"], C_X, "m")
        self._cell_steps(self._ap, ctx, t["r0"], C_REF, "r0 (mean aperture)")
        self._cell_steps(self._ap, ctx, t["a_min"], C_Y,
                         "a (smallest tip distance)")
        self._cell_steps(self._ap, ctx, t["m"] * t["a_min"], C_3,
                         "m·a (widest tip distance)", pen=_dash(C_3))
        self._cell_steps(self._len, ctx, t["L"], C_X, "L")
        bl2 = t.get("beta_lambda_half")
        run_beta = bool(t.get("beta_from_run"))
        if run_beta and bl2 is not None and np.any(np.isfinite(bl2)):
            self._cell_steps(self._len, ctx, bl2, C_3,
                             "βλ/2 of the reference particle", pen=_dash(C_3))
        self._note.setText(
            "From the RFQ_CELL cards: a is the smallest tip distance of each "
            "cell (the two-term tip condition, the flare law in ±3 cells).  "
            + ("Dashed: βλ/2 of the run's reference particle at each cell exit "
               "(synchronism: L = βλ/2)." if run_beta else
               "After a run, βλ/2 of the reference particle is drawn against "
               "L."))


# ---------------------------------------------------------------------------
# 4  synchronous phase and energy
# ---------------------------------------------------------------------------
class RfqPhasePopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ synchronous phase and energy",
                         size=(1050, 860))
        self._phi = self._add_plot("φ_s", "deg")
        self._w = self._add_plot("energy", "MeV")
        self._dphi = self._add_plot("⟨φ⟩ − φ_ref", "deg")
        self._dw = self._add_plot("⟨W⟩ − W_ref", "keV")

    def _draw(self, ctx, results):
        t = ctx.table
        self._cell_steps(self._phi, ctx, t["phi_s"], C_X, "φ_s (cards)")
        mid = 0.5 * (ctx.chain.s0 + ctx.chain.s1)
        if np.any(np.isfinite(t["phi_s_from_L"])):
            self._phi.plot(mid, t["phi_s_from_L"], pen=None, symbol="o",
                           symbolSize=5, symbolBrush=C_Y, symbolPen=None,
                           name="φ_s from the cell lengths")
        mc2 = rd.species_constants(ctx.species)[0] * 1e-6
        if "W" in t:
            self._at_exits(self._w, ctx, t["W"], C_X, "reference particle")
            if "W_beam" in t:
                self._at_exits(self._w, ctx, t["W_beam"], C_3, "beam mean")
        else:
            b = np.asarray(t["beta"], float)
            with np.errstate(invalid="ignore"):
                w = (1.0 / np.sqrt(1.0 - b * b) - 1.0) * mc2
            self._at_exits(self._w, ctx, w, C_X, "design (β = 2L/λ)",
                           symbol="o", pen=None)
        if "phi_offset" in t:
            self._at_exits(self._dphi, ctx, t["phi_offset"], C_X, "⟨φ⟩ − φ_ref")
        if "w_offset" in t:
            self._at_exits(self._dw, ctx, np.asarray(t["w_offset"]) * 1e3, C_3,
                           "⟨W⟩ − W_ref")
        self._note.setText(
            "φ_s from the cell lengths: the phase that delivers each cell's "
            "design energy gain (L = βλ/2) with its A10 and V, below the crest.  "
            + self._run_note(ctx))


# ---------------------------------------------------------------------------
# 5  focusing and acceleration
# ---------------------------------------------------------------------------
class RfqFocusingPopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ focusing and acceleration "
                         "(smooth approximation)", size=(1050, 900))
        self._a = self._add_plot("A10")
        self._d = self._add_plot("RF defocusing Δ_rf")
        self._s = self._add_plot("σ0 per βλ", "deg")
        self._i = self._add_plot("current limit", "mA")

    def _draw(self, ctx, results):
        t = ctx.table
        self._cell_steps(self._a, ctx, t["A10"], C_X, "A10 (cards)")
        if np.any(t["A10_theory"] > 0):
            th = np.where(t["A10_theory"] > 0, t["A10_theory"], np.nan)
            self._cell_steps(self._a, ctx, th, C_Y, "A10 from (r0, m, L)",
                             pen=_dash(C_Y))
        self._cell_steps(self._d, ctx, t["delta_rf"], C_X, "Δ_rf")
        self._cell_steps(self._s, ctx, t["sigma0t_deg"], C_X, "σ0 transverse")
        self._cell_steps(self._s, ctx, t["sigma0l_deg"], C_Y, "σ0 longitudinal")
        self._cell_steps(self._i, ctx, t["I_t_mA"], C_X, "transverse")
        self._cell_steps(self._i, ctx, t["I_l_mA"], C_Y, "longitudinal")
        if ctx.current_mA:
            self._i.plot(ctx.chain.s0[[0]].tolist() + [ctx.chain.s1[-1]],
                         [float(ctx.current_mA)] * 2, pen=_dash(C_3),
                         name=f"beam current {ctx.current_mA:g} mA")
        B = np.asarray(t["B"], float)
        if np.any(np.isfinite(B)):
            b_lo, b_hi = np.nanmin(B), np.nanmax(B)
            b_txt = (f"{b_lo:.2f}" if abs(b_hi - b_lo) < 5e-3
                     else f"{b_lo:.2f}–{b_hi:.2f}")
        else:
            b_txt = "unknown (no RF frequency)"
        self._note.setText(
            f"Focusing strength B = qVλ²/(mc²r0²) = {b_txt}.  Δ_rf, σ0 and "
            "the current limits use "
            f"{t['beta_source']}; the current limits assume a uniform "
            "ellipsoidal bunch filling the smallest aperture and the "
            "separatrix (Results-tab manual page).")


# ---------------------------------------------------------------------------
# 6  surface field
# ---------------------------------------------------------------------------
class RfqSurfacePopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ peak surface field and the "
                         "Kilpatrick limit", size=(1050, 760))
        self._e = self._add_plot("peak surface field", "MV/m")
        self._k = self._add_plot("surface field / Kilpatrick")

    def _draw(self, ctx, results):
        el, scale, src, why = ctx.solver()
        if el is None:
            self._show_empty("The surface field needs the time-domain solver's "
                             f"vanes, which cannot be used here:\n{why}")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            fmap = rd.cached_field_map(el, ctx.chain)
        finally:
            QApplication.restoreOverrideCursor()
        if fmap is None:
            self._show_empty(
                "The surface field comes from this RFQ's 3-D field map, which "
                "is built on its first time-domain run\n(Numerics tab → RFQ "
                "solver: Auto or Time domain).  No map is cached yet; none is "
                "built from here.")
            return
        h = float(fmap.h)
        z0, z1 = el.z_span
        z = np.linspace(z0 + 2.0, z1 - 2.0, 1500)
        x_store = (int(np.max(fmap.nx)) - 1) * h - 2.0 * h
        ex, ey = rd.peak_surface_field(rd._field_fn(fmap), el, z,
                                       offsets_mm=(3 * h, 5 * h),
                                       x_store_mm=x_store, scale=scale)
        s = ctx.chain.to_s(z)
        ex_mv, ey_mv = ex * 1e-3, ey * 1e-3            # V/mm -> MV/m
        self._e.plot(s, ex_mv, pen=curve_pen(C_X), name="±x vane tips",
                     connect="finite")
        self._e.plot(s, ey_mv, pen=curve_pen(C_Y), name="±y vane tips",
                     connect="finite")
        ek = rd.kilpatrick_field(ctx.f_MHz) if ctx.f_MHz else float("nan")
        if not np.any(np.isfinite(ex_mv) | np.isfinite(ey_mv)):
            peak = "No point where the estimate applies.  "
        elif np.isfinite(ek):
            self._e.plot([float(s[0]), float(s[-1])], [ek, ek],
                         pen=_dash(C_REF),
                         name=f"Kilpatrick {ek:.2f} MV/m at {ctx.f_MHz:g} MHz")
            ratio = np.fmax(ex_mv, ey_mv) / ek
            self._k.plot(s, ratio, pen=curve_pen(C_3), name="peak / Kilpatrick",
                         connect="finite")
            self._k.addItem(pg.InfiniteLine(pos=1.0, angle=0, pen=_dot(C_REF)))
            k = int(np.nanargmax(ratio))
            peak = (f"Peak {np.nanmax(np.fmax(ex_mv, ey_mv)):.2f} MV/m = "
                    f"{ratio[k]:.2f} Kilpatrick at s = {s[k]:.0f} mm.  ")
        else:
            peak = "RF frequency unknown: no Kilpatrick ratio.  "
        self._note.setText(
            peak + "At the RF crest, from the 3-D field map of "
            f"{_vanes_of(src)}: |E| sampled around each tip at {3 * h:.1f} and "
            f"{5 * h:.1f} mm from the metal and extrapolated to the surface "
            "(within 1 % of a converged solve on a plain quadrupole).  Left "
            "out where a tip is steep or stepped along z (flares, vane ends, "
            "steps between cards).")


# ---------------------------------------------------------------------------
# 7  fields
# ---------------------------------------------------------------------------
class RfqFieldsPopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ fields along the axis",
                         size=(1050, 860))
        self._ez = self._add_plot("E_z on axis", "MV/m")
        self._g = self._add_plot("∂E_x/∂x", "MV/m²")
        self._prof = self._add_plot("g_x, g_y")

    def _draw(self, ctx, results):
        z = np.linspace(float(ctx.chain.edges[0]), float(ctx.chain.edges[-1]),
                        6000)
        s = ctx.chain.to_s(z)
        notes = []
        el, scale, src, why = ctx.solver()
        fmap = None
        if el is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                fmap = rd.cached_field_map(el, ctx.chain)
            finally:
                QApplication.restoreOverrideCursor()
        ez_c, g_c = rd.cards_axis_fields(ctx.chain, z)
        if fmap is not None:
            ez, g = rd.axis_fields(fmap, z, scale=scale)
            self._ez.plot(s, ez * 1e-3, pen=curve_pen(C_X),
                          name="time-domain field map")
            self._g.plot(s, g, pen=curve_pen(C_X),
                         name="time-domain field map")
            pen = _dash(theme.TEXT_2, 1.0)
            notes.append(f"Solid: the time-domain solver's field map of "
                         f"{_vanes_of(src)}; dashed: the cards' two-term "
                         "field.")
        else:
            pen = curve_pen(C_X)
            notes.append("The cards' two-term field (the time-domain field map "
                         "is not cached; it is built on the first time-domain "
                         "run).")
        self._ez.plot(s, ez_c * 1e-3, pen=pen, name="cards' two-term field")
        self._g.plot(s, g_c, pen=pen, name="cards' two-term field")
        if fmap is not None:
            # the cards' field is piecewise per cell and jumps where
            # neighbouring cells do not match: frame on the vanes' field
            from linac_gen_gui.interphase.plots.plot_style import fix_plot_range
            for p, y in ((self._ez, ez * 1e-3), (self._g, g)):
                lo, hi = np.nanmin(y), np.nanmax(y)
                pad = 0.08 * max(hi - lo, 1e-9)
                fix_plot_range(p, yrange=(lo - pad, hi + pad))
            notes.append("Framed on the vanes' field; the cards' field jumps "
                         "at cell boundaries where neighbouring cards do not "
                         "match.")
        prof = rd.cached_vane_profile(getattr(ctx.lattice, "rfq_geom_file",
                                              None))
        if prof is not None:
            zp, gx, gy = prof
            sp = ctx.chain.to_s(zp)
            self._prof.plot(sp, gx, pen=curve_pen(C_X), name="g_x")
            self._prof.plot(sp, gy, pen=curve_pen(C_Y), name="g_y")
            notes.append("Bottom: the per-cell model's vane-field profile g_x, "
                         "g_y (quadrupole gradient of the vane file per plane, "
                         "normalised to the unmodulated plateau).")
        else:
            self._prof.hide()
        self._note.setText("  ".join(notes))


# ---------------------------------------------------------------------------
# 8  losses
# ---------------------------------------------------------------------------
class RfqLossesPopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ losses", size=(1050, 940))
        self._n = self._add_plot("lost particles per cell")
        self._en = self._add_plot("energy at loss", "MeV")
        self._w = self._add_plot("lost power per cell", "W")
        self._xy = self._add_plot("y at loss", "mm", xlabel="x at loss",
                                  link=False)
        self._xy.setAspectLocked(True)

    def _draw(self, ctx, results):
        lo = rd.cell_losses(results, ctx.chain, current_mA=ctx.current_mA,
                            duty_pct=ctx.duty_pct)
        if lo is None or not ctx.has_run:
            self._show_empty("No losses to show: " + (
                "run a multiparticle simulation first" if lo is None
                else ctx.table.get("run_note", "")))
            return
        mid = 0.5 * (ctx.chain.s0 + ctx.chain.s1)
        width = np.maximum(ctx.chain.s1 - ctx.chain.s0, 1.0)
        r, g, b = pg.mkColor(C_X).getRgb()[:3]
        self._n.addItem(pg.BarGraphItem(x=mid, height=lo["count"], width=width,
                                        brush=pg.mkBrush(r, g, b, 170),
                                        pen=pg.mkPen(C_X, width=0.8)))
        if lo["s"].size:
            self._en.plot(lo["s"], lo["energy"], pen=None, symbol="o",
                          symbolSize=4, symbolBrush=C_Y, symbolPen=None,
                          name="lost particle")
            self._xy.plot(lo["x"], lo["y"], pen=None, symbol="o", symbolSize=4,
                          symbolBrush=C_3, symbolPen=None, name="loss point")
        if np.any(np.isfinite(lo["watts"])):
            r, g, b = pg.mkColor(C_3).getRgb()[:3]
            self._w.addItem(pg.BarGraphItem(x=mid, height=np.nan_to_num(lo["watts"]),
                                            width=width,
                                            brush=pg.mkBrush(r, g, b, 170),
                                            pen=pg.mkPen(C_3, width=0.8)))
        n = int(lo["count"].sum())
        tot = (f"; {np.nansum(lo['watts']):.3g} W in all at "
               f"{ctx.current_mA:g} mA, {ctx.duty_pct:g} % duty"
               if np.any(np.isfinite(lo["watts"])) else "")
        used = getattr(results, "rfq_solver_used", None)
        where = {"time_domain": "the time-domain solver: the point where it "
                                "touched the metal or left the stored field "
                                "region between the vanes",
                 "cells": "the per-cell model: where it left the cell's "
                          "vane-tip boundary"}.get(used, "")
        self._note.setText(
            f"{n} macroparticles lost in the RFQ{tot}.  Energy and position "
            "where each was lost" + (f" ({where})." if where else "."))


# ---------------------------------------------------------------------------
# 9  bunching
# ---------------------------------------------------------------------------
class RfqBunchingPopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ bunching and transmission",
                         size=(1050, 940))
        self._tr = self._add_plot("transmission", "%")
        self._sp = self._add_plot("phase", "deg")
        self._sw = self._add_plot("σ_W (rms)", "keV")
        self._ps = self._add_plot("W − W_ref", "MeV",
                                  xlabel="φ − φ_ref", xunits="deg", link=False)

    def _draw(self, ctx, results):
        t = ctx.table
        if not ctx.has_run:
            self._show_empty(self._run_note(ctx).rstrip("."))
            return
        if "trans" in t:
            self._at_exits(self._tr, ctx, t["trans"], C_X, "transmission")
        if "sigma_phi" in t:
            self._at_exits(self._sp, ctx, t["sigma_phi"], C_X, "σ_φ (rms)")
        self._cell_steps(self._sp, ctx, np.abs(t["phi_s"]), C_REF, "|φ_s|",
                         pen=_dash(C_REF))
        if "sigma_w" in t:
            self._at_exits(self._sw, ctx, np.asarray(t["sigma_w"]) * 1e3, C_Y,
                           "σ_W")
        beam = getattr(results, "beam", None)
        names = list(getattr(results, "element_names", None) or [])
        after = self._drift_after_rfq_mm(ctx)
        at_end = (bool(names) and str(names[-1]) == str(
            getattr(ctx.lattice.elements[-1], "name", "")) and after is not None
            and after <= 100.0)
        end_note = ""
        if beam is not None and at_end:
            q = np.asarray(beam.particles)[np.asarray(beam.alive_mask, bool)]
            n_fold = 0
            if not getattr(results, "periodic_phase", False):
                # one bunch, as the Phase Space plot draws it (view only)
                from linac_gen.diagnostics.moments import wrap_phase_column
                q, n_fold = wrap_phase_column(q)
            self._ps.plot(q[:, 4], q[:, 5], pen=None, symbol="o",
                          symbolSize=2, symbolBrush=pg.mkBrush(34, 211, 238, 90),
                          symbolPen=None, name=f"{len(q)} particles")
            end_note = ("Bottom: the beam at the end of the run (the RFQ exit"
                        + (f" and {after:.1f} mm of drift)" if after > 0
                           else ")"))
            end_note += (f", φ folded into one RF period ({n_fold} particles "
                         "from the neighbouring buckets)." if n_fold else ".")
        else:
            self._ps.hide()
            end_note = ("The exit phase space is drawn when the run ends at "
                        "the RFQ exit.")
        self._note.setText("At each cell exit.  " + end_note)

    @staticmethod
    def _drift_after_rfq_mm(ctx):
        """Length (mm) of what follows the RFQ when it is only drift and
        zero-length elements, else None."""
        total = 0.0
        for e in list(ctx.lattice.elements)[ctx.chain.index[-1] + 1:]:
            L = float(getattr(e, "length", 0.0) or 0.0)
            if L > 0 and type(e).__name__ != "Drift":
                return None
            total += L
        return total


# ---------------------------------------------------------------------------
# 10  cell table
# ---------------------------------------------------------------------------
_TABLE_COLUMNS = (
    ("cell", "cell", "{:d}"), ("name", "name", "{}"), ("type", "type", "{:d}"),
    ("s1", "s exit [mm]", "{:.1f}"), ("L", "L [mm]", "{:.3f}"),
    ("V", "V [kV]", "{:.2f}"), ("r0", "r0 [mm]", "{:.3f}"),
    ("m", "m", "{:.3f}"), ("a_min", "a [mm]", "{:.3f}"),
    ("A10", "A10", "{:.4f}"), ("phi_s", "φs [deg]", "{:.1f}"),
    ("phi_s_from_L", "φs from L [deg]", "{:.1f}"), ("beta", "beta", "{:.5f}"),
    ("W", "W_ref [MeV]", "{:.4f}"), ("E0T_MV_m", "E0T [MV/m]", "{:.3f}"),
    ("B", "B", "{:.2f}"),
    ("delta_rf", "Δ_rf", "{:.4f}"), ("sigma0t_deg", "σ0t [deg]", "{:.1f}"),
    ("sigma0l_deg", "σ0l [deg]", "{:.1f}"), ("I_t_mA", "I_t [mA]", "{:.0f}"),
    ("I_l_mA", "I_l [mA]", "{:.0f}"), ("trans", "transmission [%]", "{:.2f}"),
    ("lost", "lost", "{:d}"), ("sigma_x", "σx [mm]", "{:.3f}"),
    ("sigma_y", "σy [mm]", "{:.3f}"), ("sigma_phi", "σφ [deg]", "{:.2f}"),
)


class RfqTablePopup(_RfqPopup):
    def __init__(self, parent, state):
        super().__init__(parent, state, "RFQ cell table", size=(1250, 760))
        self._table = QTableWidget(0, 0)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.layout().addWidget(self._table, stretch=1)
        self._extra_widgets.append(self._table)

    def _draw(self, ctx, results):
        t = dict(ctx.table)
        t["V"] = np.asarray(t["V"]) * 1e-3
        lo = rd.cell_losses(results, ctx.chain) if ctx.has_run else None
        if lo is not None:
            t["lost"] = lo["count"]
        cols = [(k, h, f) for k, h, f in _TABLE_COLUMNS
                if t.get(k) is not None]
        C = len(ctx.chain.cells)
        # filled with updates off and sized once at the end (sizing the
        # columns on every insert into a shown table took minutes)
        self._table.setUpdatesEnabled(False)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels([h for _k, h, _f in cols])
        self._table.setRowCount(C)
        for j, (k, _h, f) in enumerate(cols):
            col = t[k]
            for i in range(C):
                v = col[i]
                try:
                    if f.endswith("d}"):
                        txt = f.format(int(v))
                    elif f == "{}":
                        txt = str(v)
                    else:
                        txt = "" if not np.isfinite(float(v)) else f.format(float(v))
                except (TypeError, ValueError):
                    txt = str(v)
                it = QTableWidgetItem(txt)
                if f != "{}":
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(i, j, it)
        self._table.setUpdatesEnabled(True)
        self._table.resizeColumnsToContents()
        self._note.setText(self._summary(ctx, results))

    def _summary(self, ctx, results) -> str:
        parts = [f"{len(ctx.chain.cells)} cells, s = {ctx.chain.s_start:.1f}–"
                 f"{ctx.chain.s1[-1]:.1f} mm."]
        used = getattr(results, "rfq_solver_used", None) if results is not None \
            else None
        if used:
            label = {"time_domain": "the time-domain solver",
                     "cells": "the per-cell model"}.get(used, used)
            parts.append(f"The RFQ was tracked by {label}.")
            note = getattr(results, "rfq_solver_note", None)
            if note:
                parts.append(f"(Why not the time domain: {note}.)")
        info = getattr(results, "rfq_td", None) if results is not None else None
        if isinstance(info, dict) and info:
            parts.append(
                f"Time domain: {info.get('n_tracked', '?')} particles, "
                f"{info.get('n_lost', '?')} lost ({info.get('n_stalled', 0)} "
                f"stalled, {info.get('n_left_map', 0)} left the stored field), "
                f"{info.get('steps_per_period', '?')} steps per RF "
                f"period, {float(info.get('track_s', 0) or 0):.0f} s tracking; "
                f"bunch {float(info.get('exit_w_offset_kev', 0) or 0):+.1f} keV "
                "from the reference at the exit.")
            n_out = int(info.get("n_left_out", 0) or 0)
            n_un = int(info.get("n_unaccelerated", 0) or 0)
            if n_out:
                parts.append(
                    f"The statistics leave out {n_out} particle(s) from where "
                    f"they fall behind the bunch: {n_un} leave the RFQ "
                    "unaccelerated (they stay in the beam and the "
                    f"transmission), {n_out - n_un} are lost on the vanes "
                    "later (Numerics: RFQ statistics).")
        if ctx.n_chains_note():
            parts.append(ctx.n_chains_note())
        parts.append(self._run_note(ctx))
        return "  ".join(parts)


#: tile key -> popup class
POPUPS = {
    "rfq_geometry": RfqGeometryPopup,
    "rfq_section": RfqSectionPopup,
    "rfq_modulation": RfqModulationPopup,
    "rfq_phase": RfqPhasePopup,
    "rfq_focusing": RfqFocusingPopup,
    "rfq_surface": RfqSurfacePopup,
    "rfq_fields": RfqFieldsPopup,
    "rfq_losses": RfqLossesPopup,
    "rfq_bunching": RfqBunchingPopup,
    "rfq_table": RfqTablePopup,
}
