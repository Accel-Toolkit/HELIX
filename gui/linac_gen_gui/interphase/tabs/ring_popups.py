"""The Results tab's RING RUN popups: what a ring run shows beyond the beam
turn by turn (which the other tiles show) — the ramp, the bunch in eV s, the
losses along the ring and per turn, the tunes, phase space at the snapshot
turns, the periodic optics and the injection.

Imported only when a RING RUN tile is opened (a linac session loads no ring
module).  Each popup's ``refresh(results)`` takes the Results tab's results:
a :class:`~linac_gen.ring.view.RingResultsView` (the run itself in its
``ring``), or None / a linac run's results — then it shows that there is no
ring run.  Units: turns counted from 0 (the start; a statistics row at turn
t is the beam after t turns), the losses and the injection's records against
the turn in which they happened (1 = the first), s along the ring in m,
positions in mm, momenta in mrad (P/P0 x 1e3), zeta in m, delta x 1e3,
emittances in eV s.
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.panels import ring_plots as RP
from linac_gen_gui.interphase.plots.plot_style import (
    DensityPanel,
    add_legend,
    curve_pen,
    style_plot,
)
from linac_gen_gui.interphase.tabs.results_tab import _PopupPlot

C_LIGHT = 299792458.0
NO_RING = "No ring run is shown (Results tab: Run shown — Ring run)."


def _ring(results):
    """The ring run behind the Results tab's results, or None."""
    return getattr(results, "ring", None) if getattr(results, "is_ring_view", False) else None


def _bins_used(sc: dict) -> list:
    """The line-density bins a run used: [(turn, bins), ...] (none without
    space charge)."""
    used = sc.get("z_bins_used")
    return [] if used is None else [(int(t), int(b)) for t, b in used]


def _mk(ylabel: str, xlabel: str = "turn", xunits: str = "", yunits: str = "",
        legend: bool = False) -> pg.PlotWidget:
    p = pg.PlotWidget()
    style_plot(p, ylabel, yunits, xlabel=xlabel, xunits=xunits)
    if legend:
        add_legend(p)
    return p


def _note() -> QLabel:
    lb = QLabel("")
    lb.setWordWrap(True)
    lb.setStyleSheet(f"color:{theme.TEXT_2}; font-size:12px;")
    lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lb


# ---------------------------------------------------------------------------
class RingRampPopup(_PopupPlot):
    """The reference's kinetic energy, the synchronous phase, the RF voltage
    and frequency turn by turn, transition marked."""

    def __init__(self, parent, state=None):
        super().__init__(parent, "Ring ramp  —  energy and RF, turn by turn", size=(1100, 720))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        self.ramp_plots = RP.RampPlots()
        v.addWidget(self.ramp_plots, stretch=1)

    def refresh(self, results) -> None:
        res = _ring(results)
        r = (getattr(res, "ramp", None) or {}) if res is not None else {}
        if res is None:
            self.ramp_plots.clear(NO_RING)
            return
        if not r or not len(r.get("t_s", ())):
            self.ramp_plots.clear("No ramp: the run is at the ring's fixed energy (a ramp is "
                                  "set on the Programs page).")
            return
        n = len(r["t_s"])
        # (the phase rule's phase plus the paraphasing groups' offset: the
        # phase of the net voltage)
        self.ramp_plots.show_ramp(np.arange(n), r["kinetic_mev"],
                                  np.asarray(r["phi_deg"], dtype=float)
                                  + float(r.get("phase_offset_deg", 0.0) or 0.0),
                                  r["voltage_mv"], r["frequency_hz"], r.get("transition_turn"),
                                  RP.ramp_note(r))


# ---------------------------------------------------------------------------
class RingLongitudinalPopup(_PopupPlot):
    """The bunch turn by turn in the ring's units: the rms longitudinal
    emittance (eV s), the bunch length (m and ns), the energy spread, and with
    space charge the line-density bins the run used."""

    def __init__(self, parent, state=None):
        super().__init__(parent, "Ring bunch  —  eV s, bunch length, energy spread",
                         size=(1100, 720))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        self.note = _note()
        v.addWidget(self.note)
        g = QGridLayout()
        self.p_el = _mk("ε_L rms", yunits="eV·s")
        self.c_el = self.p_el.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[2]))
        self.p_sz = _mk("σ_ζ rms", yunits="m")
        self.c_sz = self.p_sz.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[0]), name="σ_ζ (m)")
        self.p_st = _mk("σ_t rms", yunits="ns")
        self.c_st = self.p_st.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[4]))
        self.p_sd = _mk("σ_δ rms (1e-3)")
        self.c_sd = self.p_sd.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[1]))
        self.p_bins = _mk("line-density bins")
        # (steps: each number of bins from the turn it was set to the next
        # change, the last to the end — x the edges, one more than y)
        self.c_bins = self.p_bins.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[3]),
                                       stepMode="center")
        for k, p in enumerate((self.p_el, self.p_sz, self.p_st, self.p_sd)):
            if p is not self.p_el:
                p.setXLink(self.p_el)
            g.addWidget(p, k // 2, k % 2)
        # (not x-linked: twice as wide as the plots above, a link would line it
        # up with them on the screen and show twice their turns)
        g.addWidget(self.p_bins, 2, 0, 1, 2)
        for c in (self.c_el, self.c_sz, self.c_st, self.c_sd):
            RP.thin_long(c)
        v.addLayout(g, stretch=1)

    def refresh(self, results) -> None:
        res = _ring(results)
        for c in (self.c_el, self.c_sz, self.c_st, self.c_sd, self.c_bins):
            c.setData([], [])
        if res is None or not getattr(results, "turns", None):
            self.note.setText(NO_RING if res is None else "The ring run has no statistics row.")
            self.p_bins.setVisible(False)
            return
        t = np.asarray(results.turns, dtype=float)
        sig = np.asarray(results.sigma_ring, dtype=float).reshape(-1, 6)
        beta = np.asarray(results.ref_beta, dtype=float)
        self.c_el.setData(t, np.asarray(results.emit_long_evs, dtype=float))
        self.c_sz.setData(t, sig[:, 4])
        self.c_st.setData(t, sig[:, 4] / (beta * C_LIGHT) * 1e9)
        self.c_sd.setData(t, sig[:, 5] * 1e3)
        sc = (res.meta or {}).get("space_charge") or {}
        used = _bins_used(sc)
        if used and sc.get("bunched", True):
            xs, ys = [], []
            for turn, bins in used:
                if xs and float(turn) == xs[-1]:
                    ys[-1] = float(bins)        # (at a turn, the last bins set)
                else:
                    xs.append(float(turn))
                    ys.append(float(bins))
            end = float(res.turns_done)
            xs.append(max(end, xs[-1]) if xs else end)
            self.c_bins.setData(np.asarray(xs), np.asarray(ys))
            self.p_bins.setXRange(0.0, max(end, 1.0), padding=0.02)
            self.p_bins.setVisible(True)
            bins_note = (f"Line-density bins: {'following the bunch (auto)' if sc.get('z_bins') == 'auto' else sc.get('z_bins')}.")
        else:
            self.p_bins.setVisible(False)
            bins_note = ""
        self.note.setText(
            f"{len(t)} statistics rows over {int(res.turns_done)} turns, at the start of the "
            "ring (s = 0).  σ_t = σ_ζ / (β0 c).  " + bins_note)


# ---------------------------------------------------------------------------
class RingLossesPopup(_PopupPlot):
    """The run's losses along the ring (all turns) and against the turn."""

    def __init__(self, parent, state=None):
        super().__init__(parent, "Ring losses  —  along the ring and turn by turn",
                         size=(1100, 720))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        self.p_loss_s = _mk("particles lost", xlabel="s", xunits="m")
        self.b_loss_s = pg.BarGraphItem(x=[], height=[], width=1.0,
                                        brush=pg.mkBrush(theme.PLOT_PALETTE[3]))
        self.p_loss_s.addItem(self.b_loss_s)
        self.p_loss_t = _mk("particles lost", xlabel="turn (1 = the first)")
        self.b_loss_t = pg.BarGraphItem(x=[], height=[], width=1.0,
                                        brush=pg.mkBrush(theme.PLOT_PALETTE[4]))
        self.p_loss_t.addItem(self.b_loss_t)
        self.loss_note = _note()
        v.addWidget(self.loss_note)
        v.addWidget(self.p_loss_s, stretch=1)
        v.addWidget(self.p_loss_t, stretch=1)
        self.attach_lattice_strip(self.p_loss_s, scale=1e-3)

    def refresh(self, results) -> None:
        res = _ring(results)
        out = None if res is None else RP.loss_histograms(res, (res.meta or {}).get("length_m"))
        if out is None:
            self.b_loss_s.setOpts(x=[], height=[])
            self.b_loss_t.setOpts(x=[], height=[])
            self.loss_note.setText(NO_RING if res is None else "No particle was lost.")
            return
        (xs, hs, w), (xt, ht), note = out
        self.b_loss_s.setOpts(x=xs, height=hs, width=w)
        self.b_loss_t.setOpts(x=xt, height=ht, width=1.0)
        self.loss_note.setText(note)


# ---------------------------------------------------------------------------
def resonance_lines(x0: float, x1: float, y0: float, y1: float, order: int = 4):
    """The lines m Qx + n Qy = p (p an integer, |m| + |n| <= order) crossing
    the frame [x0, x1] x [y0, y1] of fractional tunes, each once and with its
    order (the smallest |m| + |n| that gives it: 2 Qx + 2 Qy = 2 is the
    second-order Qx + Qy = 1): [(xa, ya, xb, yb, order)]."""
    out, seen = [], set()
    for mm in range(-order, order + 1):
        for nn in range(-order, order + 1):
            k = abs(mm) + abs(nn)
            if k == 0 or k > order:
                continue
            corners = [mm * x + nn * y for x in (x0, x1) for y in (y0, y1)]
            for p in range(math.floor(min(corners)), math.ceil(max(corners)) + 1):
                if math.gcd(math.gcd(abs(mm), abs(nn)), abs(p)) > 1:
                    continue                    # (a lower order's line)
                key = (mm, nn, p)
                neg = (-mm, -nn, -p)
                if key in seen or neg in seen:
                    continue
                seen.add(key)
                pts = []
                if nn != 0:
                    for x in (x0, x1):
                        y = (p - mm * x) / nn
                        if y0 - 1e-12 <= y <= y1 + 1e-12:
                            pts.append((x, y))
                if mm != 0:
                    for y in (y0, y1):
                        x = (p - nn * y) / mm
                        if x0 - 1e-12 <= x <= x1 + 1e-12:
                            pts.append((x, y))
                pts = sorted(set((round(a, 12), round(b, 12)) for a, b in pts))
                if len(pts) >= 2:
                    (xa, ya), (xb, yb) = pts[0], pts[-1]
                    if abs(xa - xb) + abs(ya - yb) > 1e-9:
                        out.append((xa, ya, xb, yb, k))
    return out


class RingTunesPopup(_PopupPlot):
    """The tune diagram: the probes' NAFF tunes, the bare working point, the
    analytic tune shift at the bunch centre and the resonance lines up to
    the fourth order, framed on the points; the tunes as text."""

    MARGIN = 0.03

    def __init__(self, parent, state=None):
        super().__init__(parent, "Ring tunes  —  diagram", size=(1000, 860))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        self.tunes_text = QLabel("")
        self.tunes_text.setWordWrap(True)
        self.tunes_text.setStyleSheet(f"font-family:{theme.FONT_MONO}; color:{theme.TEXT_0};")
        self.tunes_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.tunes_text)
        # (the key below the plot, not a legend in it: the points can be anywhere)
        self.p_fp = _mk("Qy (fractional)", xlabel="Qx (fractional)")
        self._lines: list = []
        self.s_fp = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush(theme.PLOT_PALETTE[0]),
                                       name="probes (NAFF)")
        self.s_bare = pg.ScatterPlotItem(size=18, symbol="+", brush=None,
                                         pen=pg.mkPen(theme.PLOT_PALETTE[1], width=3),
                                         name="bare working point")
        self.s_shift = pg.ScatterPlotItem(size=16, symbol="x", brush=None,
                                          pen=pg.mkPen(theme.PLOT_PALETTE[3], width=3),
                                          name="bare + analytic shift")
        for it in (self.s_fp, self.s_bare, self.s_shift):
            self.p_fp.addItem(it)
        v.addWidget(self.p_fp, stretch=1)
        self.note = _note()
        self.note.setTextFormat(Qt.TextFormat.RichText)
        key = (f'<span style="color:{theme.PLOT_PALETTE[0]}">&#9679;</span> the probes\' '
               f'tunes (NAFF) &nbsp;&nbsp; <span style="color:{theme.PLOT_PALETTE[1]}">'
               f'<b>+</b></span> the bare working point &nbsp;&nbsp; '
               f'<span style="color:{theme.PLOT_PALETTE[3]}"><b>&#215;</b></span> the bare '
               "working point plus the analytic tune shift at the bunch centre.  Dashed: "
               "resonance lines m Qx + n Qy = p up to the fourth order (brighter and "
               "thicker: lower order).")
        self.note.setText(key)
        v.addWidget(self.note)

    def refresh(self, results) -> None:
        res = _ring(results)
        for ln in self._lines:
            self.p_fp.removeItem(ln)
        self._lines = []
        for s in (self.s_fp, self.s_bare, self.s_shift):
            s.setData([], [])
        if res is None:
            self.tunes_text.setText(NO_RING)
            return
        lines, bare, shift = RP.tunes_text(res)
        self.tunes_text.setText("\n".join(lines))
        q = np.asarray((res.probes or {}).get("tunes", np.zeros((0, 3))))
        xs, ys = [], []
        if q.size:
            ok = np.isfinite(q[:, 0]) & np.isfinite(q[:, 1])
            self.s_fp.setData(q[ok, 0], q[ok, 1])
            xs += list(q[ok, 0])
            ys += list(q[ok, 1])
        o = res.optics or {}
        if bare is None:
            q6 = o.get("eigen_tunes_6d")
            if q6:
                bare = (q6[0], q6[1])
            elif o.get("qx") is not None:
                bare = (float(o["qx"]) % 1.0, float(o["qy"]) % 1.0)
        if bare is not None:
            self.s_bare.setData([bare[0]], [bare[1]])
            xs.append(bare[0])
            ys.append(bare[1])
            if shift is not None and max(abs(shift[0]), abs(shift[1])) <= 0.5:
                sx, sy = (bare[0] + shift[0]) % 1.0, (bare[1] + shift[1]) % 1.0
                self.s_shift.setData([sx], [sy])
                xs.append(sx)
                ys.append(sy)
        if not xs:
            return
        m = self.MARGIN
        x0, x1 = max(0.0, min(xs) - m), min(1.0, max(xs) + m)
        y0, y1 = max(0.0, min(ys) - m), min(1.0, max(ys) + m)
        # (a frame at least 0.1 wide, so a lone point sits among its lines)
        for lo, hi, axis in ((x0, x1, "x"), (y0, y1, "y")):
            if hi - lo < 0.1:
                c = 0.5 * (lo + hi)
                lo, hi = max(0.0, c - 0.05), min(1.0, c + 0.05)
            if axis == "x":
                x0, x1 = lo, hi
            else:
                y0, y1 = lo, hi
        alpha = {1: 230, 2: 190, 3: 140, 4: 95}
        for xa, ya, xb, yb, k in resonance_lines(x0, x1, y0, y1, 4):
            c = pg.mkColor(theme.TEXT_2)
            c.setAlpha(alpha.get(k, 80))
            ln = pg.PlotDataItem([xa, xb], [ya, yb],
                                 pen=pg.mkPen(c, width=max(0.8, 2.2 - 0.4 * k),
                                              style=Qt.PenStyle.DashLine))
            self.p_fp.addItem(ln, ignoreBounds=True)
            self._lines.append(ln)
        vb = self.p_fp.getPlotItem().getViewBox()
        vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.0)


# ---------------------------------------------------------------------------
class RingPhaseSpacePopup(_PopupPlot):
    """Phase space at the snapshot turns: density maps of x–px, y–py and
    zeta–delta (all the particles, log colour, framed at 4.5 rms, with their
    profiles; the separatrix of the turn's RF bucket on zeta–delta) and the
    line density along the bunch, with the run's line-density bins and the
    Gaussian of the same rms."""

    def __init__(self, parent, state=None):
        super().__init__(parent, "Ring phase space  —  snapshot turns", size=(1300, 980))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        row = QHBoxLayout()
        row.addWidget(QLabel("Snapshot at turn"))
        self.snap_combo = QComboBox()
        self.snap_combo.currentIndexChanged.connect(self._on_snapshot)
        row.addWidget(self.snap_combo)
        row.addSpacing(16)
        row.addWidget(QLabel("Show as"))
        self.mode = QComboBox()
        self.mode.addItem("density", "density")
        self.mode.addItem("dots", "dots")
        self.mode.currentIndexChanged.connect(self._on_snapshot_again)
        row.addWidget(self.mode)
        row.addStretch(1)
        v.addLayout(row)
        g = QGridLayout()
        self.d_x = DensityPanel("x – px", "x", "mm", "px", "mrad", bins=160, profiles=True)
        self.d_y = DensityPanel("y – py", "y", "mm", "py", "mrad", bins=160, profiles=True)
        # (the panel's title is set in capitals: words, not Greek capitals)
        self.d_z = DensityPanel("zeta – delta", "ζ", "m", "δ", "1e-3", bins=160, profiles=True)
        self.c_sep_up = self.d_z.plot.plot([], [], pen=pg.mkPen(theme.TEXT_1, width=1.5))
        self.c_sep_dn = self.d_z.plot.plot([], [], pen=pg.mkPen(theme.TEXT_1, width=1.5))
        self.p_line = _mk("line density", xlabel="ζ", xunits="m", yunits="1/m", legend=True)
        self.c_line = self.p_line.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[2], 1.6),
                                       stepMode="center", name="particles")
        self.c_gauss = self.p_line.plot([], [], pen=pg.mkPen(theme.TEXT_1, width=1.5,
                                                               style=Qt.PenStyle.DashLine),
                                        name="Gaussian of the same rms")
        g.addWidget(self.d_x, 0, 0)
        g.addWidget(self.d_y, 0, 1)
        g.addWidget(self.d_z, 1, 0)
        g.addWidget(self.p_line, 1, 1)
        v.addLayout(g, stretch=1)
        self.phase_note = _note()
        v.addWidget(self.phase_note)
        self._res = None

    def refresh(self, results) -> None:
        res = _ring(results)
        self._res = res
        self.snap_combo.blockSignals(True)
        self.snap_combo.clear()
        if res is not None:
            for t in sorted(res.snapshots):
                self.snap_combo.addItem(str(t), int(t))
            if self.snap_combo.count():
                # (the last snapshot drawn once: the combo's signal stays blocked)
                self.snap_combo.setCurrentIndex(self.snap_combo.count() - 1)
        self.snap_combo.blockSignals(False)
        if self.snap_combo.count():
            self._on_snapshot(self.snap_combo.count() - 1)
            return
        for d in (self.d_x, self.d_y, self.d_z):
            d.set_data([], [])
        for c in (self.c_sep_up, self.c_sep_dn, self.c_line, self.c_gauss):
            c.setData([], [])
        self.phase_note.setText(NO_RING if res is None else
                                "No snapshot was asked for (Run page: Snapshot turns).")

    def _on_snapshot_again(self, _i: int) -> None:
        if self.snap_combo.count():
            self._on_snapshot(self.snap_combo.currentIndex())

    def _on_snapshot(self, i: int) -> None:
        res = self._res
        if res is None or i < 0:
            return
        t = self.snap_combo.itemData(i)
        snap = res.snapshots.get(t)
        if snap is None:
            return
        X = np.asarray(snap["coords"], dtype=float)
        dots = self.mode.currentData() == "dots"
        for d, (a, b, sa, sb) in ((self.d_x, (0, 1, 1e3, 1e3)), (self.d_y, (2, 3, 1e3, 1e3)),
                                   (self.d_z, (4, 5, 1.0, 1e3))):
            if dots:
                d.set_scatter(X[a] * sa, X[b] * sb, point_size=3)
            else:
                d.set_data(X[a] * sa, X[b] * sb)
        self._separatrix(res, snap, t)
        self._line_density(res, X[4], t)

    def _separatrix(self, res, snap, t) -> None:
        b = (res.optics or {}).get("bucket")
        sr = snap.get("reference")
        acc = RP.accelerating_bucket(sr) if sr else None
        n = snap["coords"].shape[1]
        if acc is not None:
            z = np.linspace(acc["zeta_min"], acc["zeta_max"], 801)
            d = acc["separatrix"](z) * 1e3
            self.c_sep_up.setData(z, d, connect="finite")
            self.c_sep_dn.setData(z, -d, connect="finite")
            self.phase_note.setText(
                f"Turn {t}: {n} particles; the white line is the separatrix of the turn's "
                f"bucket ({float(sr['kinetic_mev']):.6g} MeV, synchronous phase "
                f"{float(sr['phi_eff_deg']):.4g} deg).")
        elif sr is None and b and b.get("stable"):
            L = float(b["bucket_length_m"])
            z = np.linspace(-L / 2, L / 2, 401)
            k = 2.0 * math.pi / L
            d = float(b["delta_max"]) * np.abs(np.cos(k * z / 2.0)) * 1e3
            self.c_sep_up.setData(z, d)
            self.c_sep_dn.setData(z, -d)
            self.phase_note.setText(f"Turn {t}: {n} particles; the white line is the "
                                    "separatrix of the stationary RF bucket.")
        else:
            self.c_sep_up.setData([], [])
            self.c_sep_dn.setData([], [])
            self.phase_note.setText(f"Turn {t}: {n} particles.")

    def _line_density(self, res, z, t) -> None:
        """The snapshot's line density over its extent, binned as the run's
        line density was (its bins over the RF period), and the Gaussian of
        the same rms."""
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        if z.size < 2:
            self.c_line.setData([], [])
            self.c_gauss.setData([], [])
            return
        meta = res.meta or {}
        period = float(meta.get("zeta_wrap_m") or 0.0)
        sc = meta.get("space_charge") or {}
        nz = None
        for turn, bins in _bins_used(sc):
            if turn <= int(t):
                nz = bins
        if nz is None and isinstance(sc.get("z_bins"), int):
            nz = int(sc["z_bins"])
        lo, hi = float(z.min()), float(z.max())
        if period > 0.0 and nz:
            # (the run's cells: -period/2 + k dz)
            dz = period / nz
            k0 = math.floor((lo + 0.5 * period) / dz)
            k1 = math.ceil((hi + 0.5 * period) / dz)
            edges = -0.5 * period + np.arange(k0, k1 + 1) * dz
        else:
            edges = np.linspace(lo, hi, 121)
        if edges.size < 2:
            edges = np.array([lo, hi + 1e-12])
        h, edges = np.histogram(z, bins=edges)
        dzs = np.diff(edges)
        lam = h / (z.size * dzs)
        self.c_line.setData(edges, lam)
        mu, sd = float(z.mean()), float(z.std())
        if sd > 0.0:
            zz = np.linspace(edges[0], edges[-1], 400)
            self.c_gauss.setData(zz, np.exp(-0.5 * ((zz - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)))
        else:
            self.c_gauss.setData([], [])


# ---------------------------------------------------------------------------
class RingOpticsPopup(_PopupPlot):
    """The run's periodic optics along the ring — beta_x, beta_y and the
    horizontal dispersion at every element's exit — with the tunes,
    chromaticities, momentum compaction, transition and synchrotron tune."""

    def __init__(self, parent, state=None):
        super().__init__(parent, "Ring optics  —  periodic β and dispersion", size=(1200, 760))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setStyleSheet(f"font-family:{theme.FONT_MONO}; color:{theme.TEXT_0};")
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.text)
        self.p_beta = _mk("β", xlabel="s", xunits="m", yunits="m", legend=True)
        self.c_bx = self.p_beta.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[0]), name="β_x")
        self.c_by = self.p_beta.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[1]), name="β_y")
        self.p_disp = _mk("D_x", xlabel="s", xunits="m", yunits="m")
        self.c_dx = self.p_disp.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[2]))
        self.p_disp.setXLink(self.p_beta)
        v.addWidget(self.p_beta, stretch=2)
        v.addWidget(self.p_disp, stretch=1)
        self.attach_lattice_strip(self.p_beta, scale=1e-3)

    def refresh(self, results) -> None:
        res = _ring(results)
        t = (getattr(res, "optics_table", None) or {}) if res is not None else {}
        for c in (self.c_bx, self.c_by, self.c_dx):
            c.setData([], [])
        if res is None:
            self.text.setText(NO_RING)
            return
        o = res.optics or {}
        when = "  (at the start of the ramp)" if getattr(res, "ramp", None) else ""
        lines = []
        if o.get("qx") is not None:
            lines.append(f"Q (4-D, RF off)   x {float(o['qx']):.6f}   y {float(o['qy']):.6f}"
                         + when)
        q6 = o.get("eigen_tunes_6d")
        if q6:
            lines.append(f"Q (6-D)           x {q6[0]:.6f}   y {q6[1]:.6f}   s {q6[2]:.6f}")
        if o.get("dqx") is not None:
            lines.append(f"Q' (per delta)    x {float(o['dqx']):.4f}   y {float(o['dqy']):.4f}")
        extra = []
        for key, name, fmt in (("alpha_c", "alpha_c", "{:.6g}"), ("gamma_t", "gamma_t", "{:.6f}"),
                               ("eta", "eta", "{:.6g}"), ("qs", "Qs", "{:.6g}")):
            if o.get(key) is not None:
                extra.append(f"{name} {fmt.format(float(o[key]))}")
        if extra:
            lines.append("   ".join(extra))
        if not t:
            lines.append("This ring run carries no optics table (a run made before "
                         "2026-10-25).")
        self.text.setText("\n".join(lines))
        if t:
            s = np.asarray(t["s"], dtype=float)
            self.c_bx.setData(s, np.asarray(t["betx"], dtype=float))
            self.c_by.setData(s, np.asarray(t["bety"], dtype=float))
            self.c_dx.setData(s, np.asarray(t["dx"], dtype=float))


# ---------------------------------------------------------------------------
class RingInjectionPopup(_PopupPlot):
    """The injection turn by turn — the macroparticles each turn injects and
    leaves as waste, the foil's crossings, the particles circulating — the
    closed orbit at the foil and the foil's hit map."""

    def __init__(self, parent, state=None):
        super().__init__(parent, "Ring injection  —  foil and painting", size=(1200, 820))
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        self.inj_note = _note()
        v.addWidget(self.inj_note)
        h = QHBoxLayout()
        col = QVBoxLayout()
        # (each turn's records at that turn counted from 1 — the circulating
        # count at a turn is that after it — as the losses)
        turn1 = "turn (1 = the first)"
        self.p_inj = _mk("macroparticles", xlabel=turn1, legend=True)
        self.c_inj = self.p_inj.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[0]),
                                     name="injected (each turn)")
        self.c_waste = self.p_inj.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[3]),
                                       name="waste (each turn)")
        self.c_hits = self.p_inj.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[1]),
                                      name="foil crossings of the circulating beam (each turn)")
        self.p_circ = _mk("particles circulating", xlabel=turn1)
        self.c_circ = self.p_circ.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[4]))
        self.p_co = _mk("closed orbit at the foil", xlabel=turn1, yunits="mm", legend=True)
        self.c_cox = self.p_co.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[0]), name="x")
        self.c_coy = self.p_co.plot([], [], pen=curve_pen(theme.PLOT_PALETTE[1]), name="y")
        for p in (self.p_circ, self.p_co):
            p.setXLink(self.p_inj)
        for c in (self.c_inj, self.c_waste, self.c_hits, self.c_cox, self.c_coy, self.c_circ):
            RP.thin_long(c)
        for p in (self.p_inj, self.p_circ, self.p_co):
            col.addWidget(p, stretch=1)
        h.addLayout(col, stretch=2)
        self.p_map = _mk("y", xlabel="x", xunits="mm", yunits="mm")
        self.i_map = pg.ImageItem()
        self.p_map.addItem(self.i_map)
        h.addWidget(self.p_map, stretch=1)
        v.addLayout(h, stretch=1)

    def _clear(self, note: str) -> None:
        for c in (self.c_inj, self.c_waste, self.c_hits, self.c_cox, self.c_coy, self.c_circ):
            c.setData([], [])
        self.i_map.clear()
        self.inj_note.setText(note)

    def refresh(self, results) -> None:
        res = _ring(results)
        inj = (getattr(res, "injection", None) or {}) if res is not None else {}
        if res is None or not inj:
            self._clear(NO_RING if res is None else
                        "No foil in the ring (injection is set on the Injection page).")
            return
        n = len(inj.get("foil_hits", ()))
        t = np.arange(1, n + 1, dtype=float)
        waste = (np.asarray(inj["h0"]) + np.asarray(inj["hminus"]) + np.asarray(inj["missed"]))
        self.c_inj.setData(t, np.asarray(inj["injected"], dtype=float))
        self.c_waste.setData(t, waste.astype(float))
        self.c_hits.setData(t, np.asarray(inj["foil_hits"], dtype=float))
        self.c_circ.setData(np.asarray(res.turns, dtype=float), np.asarray(res.n_alive, dtype=float))
        co = np.asarray(inj.get("closed_orbit_foil", np.zeros((0, 4))), dtype=float)
        # (row k: the magnets of turn k counted from 0 — drawn at k + 1; the
        # last row is the turn after the run)
        co = co.reshape(-1, 4)[:n] if n else co.reshape(-1, 4)
        if co.size and np.isfinite(co).any():
            tc = np.arange(1, co.shape[0] + 1, dtype=float)
            self.c_cox.setData(tc, co[:, 0] * 1e3, connect="finite")
            self.c_coy.setData(tc, co[:, 2] * 1e3, connect="finite")
        else:
            self.c_cox.setData([], [])
            self.c_coy.setData([], [])
        hm, ext = inj.get("hit_map"), inj.get("map_extent_m")
        if hm is not None and ext is not None:
            from PyQt6.QtCore import QRectF
            self.i_map.setImage(np.asarray(hm, dtype=float), autoLevels=True)
            x0, x1, y0, y1 = (float(v) * 1e3 for v in ext)
            self.i_map.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        else:
            self.i_map.clear()
        f = inj.get("foil") or {}
        mh = inj.get("mean_hits_circulating")
        hits = (f"; {float(mh):.3g} crossings per circulating particle" if mh is not None
                and np.isfinite(float(mh)) else "")
        co_note = ("" if co.size and np.isfinite(co).any() else
                   " (No closed orbit at the foil: no bump or painting magnets, or a ramp.)")
        mev_j = 1.602176634e-13
        dep = float(np.sum(inj.get("deposit_mev", 0.0))) * mev_j
        el = float(np.sum(inj.get("electron_mev", 0.0))) * mev_j
        energy = (f" Ionization energy left in the foil {dep:.4g} J"
                  + (f"; the stripped electrons carry {el:.4g} J (not tracked)" if el else "")
                  + ".")
        self.inj_note.setText(
            f"Foil {f.get('name')}: {f.get('material')} {float(f.get('thickness_ug_cm2', 0)):g} "
            f"ug/cm2, scattering theta0 {float(f.get('theta0_rad', 0)) * 1e6:.4g} urad for one "
            "crossing (a particle's crossings add up to Highland at their total thickness), mean "
            f"loss {float(f.get('mean_loss_mev', 0)) * 1e3:.4g} keV a crossing{hits}.{energy} The "
            "hit map: the ring's real particles per bin over the foil's rectangle." + co_note)


POPUPS = {
    "ring_ramp": RingRampPopup,
    "ring_long": RingLongitudinalPopup,
    "ring_losses": RingLossesPopup,
    "ring_tunes": RingTunesPopup,
    "ring_phase": RingPhaseSpacePopup,
    "ring_optics": RingOpticsPopup,
    "ring_injection": RingInjectionPopup,
}
