"""Ring-mode plots outside the Results tab, and their helpers.

* :class:`RingLiveStrip` — the Ring tab's Run page: survival, normalized
  emittances x / y and the longitudinal emittance turn by turn, fed with the
  worker's statistics rows during a run (``show_live``: only the new rows
  each time, appended) and with the results when it ends (``show_results``).
  Its plots never take the mouse wheel (it scrolls the page).
* :class:`RampPlots` — a ramp turn by turn (the reference's kinetic energy,
  the synchronous phase, the RF voltage and frequency, the transition turn
  marked): the Programs page's preview and the Results tab's "Ring ramp".
* Pure helpers shared with the Results tab's RING RUN popups
  (``tabs/ring_popups.py``): the survival in its three modes, the loss
  histograms, the tunes' text, the accelerating bucket of a snapshot.

Units: survival in % (of the particles received with injection), or the
particles circulating for a ring that started empty; normalized rms
emittances in pi mm mrad (each row with its own reference: it changes in a
ramp); the longitudinal rms emittance in eV s.  The results of a ring run are
in the Results tab ("Run shown: Ring run").
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import theme

C_LIGHT = 299792458.0
_DASH = Qt.PenStyle.DashLine
_AXIS_PT = "11pt"


def _title(w, side: str, text: str) -> None:
    """A plain-text axis title in the theme's text colour (pyqtgraph's own is
    dark grey on the dark background)."""
    w.setLabel(side, text, **{"color": theme.TEXT_1, "font-size": _AXIS_PT})


def _plot(left: str, bottom: str = "turn", cls=pg.PlotWidget) -> pg.PlotWidget:
    """A plot with plain-text axis titles (their size in the style) and the
    numbers in the units of the title (pyqtgraph's own scaling would put a
    factor such as x0.001 into the title, which a narrow plot cuts off)."""
    from PyQt6.QtGui import QFont
    w = cls()
    w.setBackground(theme.BG_1)
    tick = QFont()
    tick.setPointSize(10)
    for side in ("left", "bottom"):
        ax = w.getAxis(side)
        ax.enableAutoSIPrefix(False)
        ax.setTickFont(tick)
        ax.setTextPen(theme.TEXT_1)
    _title(w, "left", left)
    _title(w, "bottom", bottom)
    w.showGrid(x=True, y=True, alpha=0.15)
    return w


def thin_long(curve) -> None:
    """A curve of a run's turns (up to some 10^5 points) drawn at the screen's
    resolution — each pixel column's lowest and highest point — the visible
    part only, and as separate segments (a wide pen on one long zig-zag path
    is slow to stroke); its data keeps every point."""
    curve.setDownsampling(auto=True, method="peak")
    curve.setClipToView(True)
    curve.curve.setSegmentedLineMode("on")


class _NoWheelPlot(pg.PlotWidget):
    """A plot that never takes the mouse wheel — it scrolls the page it sits
    on (pyqtgraph's axes accept wheel events even with the mouse off)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        pi = self.getPlotItem()
        pi.setMouseEnabled(x=False, y=False)
        pi.setMenuEnabled(False)
        pi.hideButtons()

    def wheelEvent(self, ev):                    # noqa: N802 — Qt override
        ev.ignore()


def _emit2(cov: np.ndarray, i: int) -> np.ndarray:
    det = cov[:, i, i] * cov[:, i + 1, i + 1] - cov[:, i, i + 1] ** 2
    return np.sqrt(np.maximum(det, 0.0))


def _emittances(cov, bg, p0c):
    """Normalized rms emittances x, y [pi mm mrad] and the longitudinal one
    [eV s] of statistics rows, each with its reference (arrays or one value)."""
    return (_emit2(cov, 0) * bg * 1e6, _emit2(cov, 2) * bg * 1e6,
            _emit2(cov, 4) * p0c * 1e6 / C_LIGHT)


def accelerating_bucket(sr: dict):
    """The bucket of a ramp's snapshot (its reference and RF), or None when
    it has none (no voltage, or the phase at 90 deg)."""
    try:
        from linac_gen.ring.optics import accelerating_bucket as _bucket
        b = _bucket(float(sr["voltage_eff_mv"]), float(sr["phi_eff_deg"]),
                    float(sr["harmonic"]), float(sr["eta"]), float(sr["beta0"]),
                    float(sr["energy_mev"]), float(sr["length_m"]))
    except (KeyError, TypeError, ValueError):
        return None
    return b if b.get("stable") and b.get("bucket_length_m", 0.0) > 0.0 else None


def received(res, turns):
    """With injection, the particles the ring received by each row's turn (None
    without injection) — :func:`linac_gen.ring.view.received_by_turn`."""
    from linac_gen.ring.view import received_by_turn
    return received_by_turn(res, turns)


def survival_series(turns, n_alive, rec=None):
    """(axis title, values): the survival in %, of the particles received with
    injection (``rec``), or the particles circulating for a ring that started
    empty."""
    turns = np.asarray(turns, dtype=float)
    n_alive = np.asarray(n_alive, dtype=float)
    if n_alive.size and n_alive[0] == 0.0:
        return "particles circulating", n_alive
    if rec is not None:
        return "survival (% received)", 100.0 * n_alive / np.asarray(rec, dtype=float)
    n0 = float(n_alive[0]) if n_alive.size else float("nan")
    return "survival (%)", 100.0 * n_alive / (n0 or float("nan"))


def loss_histograms(res, length_m: float):
    """The run's losses along the ring and per turn: (centres_m, counts,
    width_m), (turns, counts) — a loss in turn t, counted from 1, drawn at t,
    where the survival drops (``losses["turn"]`` counts the turns completed
    before it) — and a note naming where most were lost; None without
    losses."""
    L = res.losses or {}
    s = np.asarray(L.get("s", []), dtype=float)
    if not s.size:
        return None
    length = float(length_m) if length_m else float(s.max() + 1.0)
    bins = max(10, min(400, int(length)))
    h, edges = np.histogram(s, bins=bins, range=(0.0, length))
    turns = np.asarray(L.get("turn", []), dtype=int)
    nt = int(res.turns_done) + 1
    ht = np.bincount(np.clip(turns, 0, None) + 1, minlength=nt)
    names = (res.meta or {}).get("element_names", [])
    counts: dict = {}
    for i in np.asarray(L.get("element", []), dtype=int):
        nm = names[i] if 0 <= i < len(names) else "?"
        counts[nm] = counts.get(nm, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    note = (f"{s.size} particles lost; most at "
            + ", ".join(f"{k} ({v})" for k, v in top) + ".")
    return ((0.5 * (edges[1:] + edges[:-1]), h, float(edges[1] - edges[0])),
            (np.arange(ht.size), ht), note)


def tunes_text(res):
    """The tunes of a run as text lines, the bare working point (fractional
    Qx, Qy; None without space charge or optics) and the analytic tune shift
    at the bunch centre (or None)."""
    t, o = res.tunes or {}, res.optics or {}
    lines = []
    if t:
        lines.append("NAFF (smallest probe): " + "  ".join(
            f"{k.capitalize()} {t[k]:.8f}" for k in ("qx", "qy", "qs") if k in t))
    q6 = o.get("eigen_tunes_6d")
    # (a ramp's optics are those of its first turn)
    when = " (at the start of the ramp)" if getattr(res, "ramp", None) else ""
    if q6:
        lines.append(f"6-D one-turn map:      Qx {q6[0]:.8f}  Qy {q6[1]:.8f}  Qs {q6[2]:.8f}"
                     + when)
    if o.get("qx") is not None:
        lines.append(f"4-D optics (RF off):   Qx {float(o['qx']):.8f}  Qy {float(o['qy']):.8f}"
                     + when)
    if not t:
        cfg = (getattr(res, "meta", None) or {}).get("config") or {}
        if not int(cfg.get("tune_probes", 0) or 0):
            lines.append("No tune probe (Tune probes 0).")
        else:
            lines.append("NAFF found no tune in the probes' motion: too few turns, or a "
                         "motion too far from regular (with space charge, a tune shift "
                         "beyond half an integer, for example).")
    sc = (getattr(res, "meta", None) or {}).get("space_charge")
    bare, shift = None, None
    if sc:
        bare = (q6[0], q6[1]) if q6 else (
            (float(o["qx"]) % 1.0, float(o["qy"]) % 1.0) if o.get("qx") is not None else None)
        q = sc.get("laslett_gaussian")
        if bare is not None and "qx" in t and "qy" in t:
            # (fractional tunes: the difference taken to within half an integer)
            dqx = (t["qx"] - bare[0] + 0.5) % 1.0 - 0.5
            dqy = (t["qy"] - bare[1] + 0.5) % 1.0 - 0.5
            lines.append(f"Space charge: the probe's fractional tunes less the bare ones "
                         f"dQx {dqx:+.5f}  dQy {dqy:+.5f} "
                         "(+ on the plot: the bare working point)")
        if q:
            where = ("at the centre of a Gaussian bunch" if sc.get("bunched")
                     else "for a coasting Gaussian beam")
            lines.append(f"Analytic, {where}: dQx {q[0]:+.5f}  dQy {q[1]:+.5f}"
                         + ("  — beyond half an integer: the fractional tunes cannot show "
                            "it" if max(abs(q[0]), abs(q[1])) > 0.5 else ""))
            shift = (float(q[0]), float(q[1]))
    return lines, bare, shift


class _Rows:
    """Columns that grow by appending (capacity doubled when full)."""

    def __init__(self, names):
        self.names = tuple(names)
        self.n = 0
        self.cols = {k: np.empty(1024) for k in self.names}

    def clear(self) -> None:
        self.n = 0

    def append(self, **cols) -> None:
        m = len(next(iter(cols.values())))
        need = self.n + m
        cap = len(self.cols[self.names[0]])
        if need > cap:
            while cap < need:
                cap *= 2
            for k in self.names:
                grown = np.empty(cap)
                grown[:self.n] = self.cols[k][:self.n]
                self.cols[k] = grown
        for k in self.names:
            self.cols[k][self.n:need] = cols[k]
        self.n = need

    def __getitem__(self, k):
        return self.cols[k][:self.n]


class RampPlots(QWidget):
    """A ramp turn by turn in a 2 x 2 grid: the reference's kinetic energy W,
    the synchronous phase phi_s (of the net voltage, sine convention), the
    total RF voltage V and the RF frequency, x-linked, the transition turn
    marked by a dashed line; a note above (it names the quantities)."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(self.note)
        self.legend = QLabel("W: the reference's kinetic energy; phi_s: the synchronous phase "
                             "of the net voltage (sine convention); V: the total RF voltage; "
                             "f_RF: the RF frequency — each at the start of its turn. Dashed "
                             "line: the turn transition is crossed.")
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(self.legend)
        self.p_energy = _plot("W (MeV)")
        self.c_energy = self.p_energy.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[0], width=2))
        self.p_phase = _plot("phi_s (deg)")
        self.c_phase = self.p_phase.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[1], width=2))
        self.p_volt = _plot("V (MV)")
        self.c_volt = self.p_volt.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[2], width=2))
        self.p_freq = _plot("f_RF (MHz)")
        self.c_freq = self.p_freq.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[3], width=2))
        self.tr_lines = []
        grid = QGridLayout()
        for k, p in enumerate((self.p_energy, self.p_phase, self.p_volt, self.p_freq)):
            if p is not self.p_energy:
                p.setXLink(self.p_energy)
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen(theme.TEXT_2, width=1, style=_DASH))
            line.setVisible(False)
            p.addItem(line)
            self.tr_lines.append(line)
            p.setMinimumHeight(100)
            grid.addWidget(p, k // 2, k % 2)
        v.addLayout(grid, stretch=1)
        for c in (self.c_energy, self.c_phase, self.c_volt, self.c_freq):
            thin_long(c)

    def clear(self, note: str = "") -> None:
        for c in (self.c_energy, self.c_phase, self.c_volt, self.c_freq):
            c.setData([], [])
        for line in self.tr_lines:
            line.setVisible(False)
        self.note.setText(note)

    def show_ramp(self, turns, kinetic_mev, phi_deg, voltage_mv, frequency_hz,
                  transition_turn, note: str = "") -> None:
        turns = np.asarray(turns, dtype=float)
        self.c_energy.setData(turns, np.asarray(kinetic_mev, dtype=float))
        self.c_phase.setData(turns, np.asarray(phi_deg, dtype=float))
        self.c_volt.setData(turns, np.asarray(voltage_mv, dtype=float))
        self.c_freq.setData(turns, np.asarray(frequency_hz, dtype=float) * 1e-6)
        for line in self.tr_lines:
            line.setVisible(transition_turn is not None)
            if transition_turn is not None:
                line.setValue(float(transition_turn))
        self.note.setText(note)


def ramp_note(r: dict) -> str:
    """One line on a ramp's record: energies, turns, times, transition, the
    turns capped at 90 deg."""
    n = len(r["t_s"])
    k = r.get("transition_turn")
    gt = r.get("gamma_t")
    gts = "none" if gt is None else f"{float(gt):.6f}"
    tr = (f"transition (gamma_t {gts}) at turn {k}, the dashed line" if k is not None
          else f"above transition (gamma_t {gts}) from the start"
          if r.get("above_transition_at_start") else f"transition (gamma_t {gts}) not crossed")
    capped = int(r.get("n_capped", 0) or 0)
    return (f"{float(r['kinetic_mev'][0]):.6g} -> {float(r['kinetic_end_mev']):.6g} MeV over "
            f"{n} turns, t {float(r['t_s'][0]) * 1e3:.6g} -> {float(r['t_end_s']) * 1e3:.6g} "
            f"ms; {tr}" + (f"; {capped} turn(s) capped at 90 deg (the gain beyond the "
                           "voltage)" if capped else "") + ".")


class RingLiveStrip(QWidget):
    """The Run page's live view: survival | normalized emittances x, y |
    longitudinal emittance, turn by turn, side by side."""

    HEIGHT_PX = 240

    def __init__(self):
        super().__init__()
        self._results = None
        self._meta: dict = {}
        self._live = _Rows(("turn", "n_alive", "ex", "ey", "el"))
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.p_surv = _plot("survival (%)", cls=_NoWheelPlot)
        self.c_surv = self.p_surv.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[0], width=2))
        self.p_emit = _plot("normalized rms emittance (pi mm mrad)", cls=_NoWheelPlot)
        from ..plots.plot_style import add_legend
        add_legend(self.p_emit)
        self.c_ex = self.p_emit.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[0], width=2),
                                     name="x")
        self.c_ey = self.p_emit.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[1], width=2),
                                     name="y")
        self.p_long = _plot("longitudinal rms emittance (eV s)", cls=_NoWheelPlot)
        self.c_el = self.p_long.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[2], width=2))
        for p in (self.p_emit, self.p_long):
            p.setXLink(self.p_surv)
        for c in (self.c_surv, self.c_ex, self.c_ey, self.c_el):
            thin_long(c)
        for p in (self.p_surv, self.p_emit, self.p_long):
            h.addWidget(p, stretch=1)
        # (at least this tall; it takes the room the page leaves)
        self.setMinimumHeight(self.HEIGHT_PX)

    def clear(self) -> None:
        self._results = None
        self._meta = {}
        self._live.clear()
        for c in (self.c_surv, self.c_ex, self.c_ey, self.c_el):
            c.setData([], [])

    def show_live(self, d: dict) -> None:
        """Statistics rows during a run — rows ``start`` on: turns, n_alive,
        cov (k, 6, 6), with the reference of each row (beta0gamma0, p0c_mev:
        arrays, or one value for all) — appended to those shown (a new run
        starts at row 0)."""
        start = int(d.get("start", 0))
        if start == 0 or start != self._live.n:
            self._live.clear()
        turns, n_alive = np.asarray(d["turns"]), np.asarray(d["n_alive"])
        if not len(turns):
            return
        bg = np.broadcast_to(np.asarray(d.get("beta0gamma0", np.nan), dtype=float),
                             turns.shape)
        pc = np.broadcast_to(np.asarray(d.get("p0c_mev", np.nan), dtype=float), turns.shape)
        self._meta = {"beta0gamma0": float(bg[0]), "p0c_mev": float(pc[0])}
        ex, ey, el = _emittances(np.asarray(d["cov"]), bg, pc)
        self._live.append(turn=turns, n_alive=n_alive, ex=ex, ey=ey, el=el)
        r = self._live
        self._survival(r["turn"], r["n_alive"])
        self.c_ex.setData(r["turn"], r["ex"])
        self.c_ey.setData(r["turn"], r["ey"])
        self.c_el.setData(r["turn"], r["el"])

    def show_results(self, res) -> None:
        """A finished (or stopped) run's rows; None clears."""
        self._results = res
        if res is None:
            self.clear()
            return
        self._live.clear()
        self._meta = dict(res.meta or {})
        turns, n_alive = np.asarray(res.turns), np.asarray(res.n_alive)
        if not len(turns):
            return
        self._survival(turns, n_alive, received(res, turns))
        # (normalized with each row's reference)
        en = np.asarray(res.emit_normalized) * 1e6
        self.c_ex.setData(turns, en[:, 0])
        self.c_ey.setData(turns, en[:, 1])
        self.c_el.setData(turns, np.asarray(res.emit_long_evs))

    def _survival(self, turns, n_alive, rec=None) -> None:
        label, y = survival_series(turns, n_alive, rec)
        fin = y[np.isfinite(y)]
        if label == "survival (%)" and fin.size and float(fin.max()) > 100.0 + 1e-9:
            # (more than at the start: a foil injects during the run — the
            # particles received come with the results)
            label = "circulating (% of the start)"
        _title(self.p_surv, "left", label)
        self.c_surv.setData(np.asarray(turns, dtype=float), y)
        vb = self.p_surv.getPlotItem().getViewBox()
        if label.startswith("survival"):
            # (nothing lost: 100 % at the top of a 95-100.5 % frame, not a flat
            # line on a +-0.4 % axis)
            lo = min(95.0, float(fin.min()) - 2.0) if fin.size else 95.0
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            vb.setYRange(max(0.0, lo), 100.5, padding=0.0)
        else:
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
