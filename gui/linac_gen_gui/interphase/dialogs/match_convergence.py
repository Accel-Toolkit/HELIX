"""Live convergence-plot popup for the matcher worker.

Subscribes to the same ``progress(iter, cost)`` pyqtSignal that drives
the matching tab's status line, and renders cost vs evaluation index
in a dark-themed pyqtgraph plot.  Stays open after the match finishes
so the user can review the trace before closing.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from linac_gen_gui.interphase import theme


class MatchConvergenceDialog(QDialog):
    """Non-modal live plot showing the matcher's cost trace.

    The matching tab opens this *before* starting the worker and wires
    the worker's existing ``progress(iter, cost)`` signal into
    :meth:`append_point`.  Updates are cheap (a list append + a line
    redraw); rendering is throttled to 10 Hz so a CMA-ES burst doesn't
    flood Qt's event queue.

    Closing the dialog mid-run does **not** stop the match; that's the
    Stop button's job.  The dialog can be reopened (via a future
    "Show convergence plot" button) without affecting the worker.
    """

    def __init__(self, *, algo: str, popsize: int = 0, max_iter: int = 0,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Match convergence")
        self.setModal(False)
        # Stay on top so the user can keep watching while interacting
        # with the main window; allow normal close ("X") to hide it.
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(f"QDialog {{ background:{theme.BG_0}; }}")
        from linac_gen_gui.interphase.scrollwrap import screen_capped
        self.resize(*screen_capped(self, 820, 760))

        self._algo = algo
        self._popsize = max(int(popsize), 0)
        self._max_iter = max(int(max_iter), 0)
        self._start_time = time.time()
        # Buffers -- one (eval, cost) point per worker `progress` emit;
        # best_so_far is a parallel monotone-decreasing trace.
        self._xs: list[int] = []
        self._ys: list[float] = []
        self._best: list[float] = []
        self._best_cost = float("inf")
        # Emittance buffers -- parallel arrays populated by
        # progress_detail (one per eval that carries info).  Stored
        # separately because progress_detail can arrive without a
        # corresponding progress signal (e.g. fixture tests) and vice
        # versa.  εnx/εny/εnz indexed by info["iter"].
        self._emit_xs: list[int] = []
        self._emit_nx: list[float] = []
        self._emit_ny: list[float] = []
        self._emit_nz: list[float] = []
        # Throttling: collect points and redraw at 10 Hz max.
        self._dirty = False
        self._tick = QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._maybe_redraw)
        self._tick.start()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ---- Title strip ----------------------------------------------
        self._title = QLabel("Waiting for first evaluation...")
        self._title.setStyleSheet(
            f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; "
            f"font-size:12px; font-weight:600; padding:2px 0;")
        outer.addWidget(self._title)

        # ---- Plot widget ----------------------------------------------
        pg.setConfigOption("background", theme.BG_INSET)
        pg.setConfigOption("foreground", theme.TEXT_2)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "cost")
        self._plot.setLabel("bottom", "evaluation")
        self._plot.setLogMode(x=False, y=True)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.getAxis("left").setTextPen(theme.TEXT_2)
        self._plot.getAxis("bottom").setTextPen(theme.TEXT_2)
        # Per-eval curve (bright accent, line + markers) and best-so-far
        # trace (dimmer, dashed, monotone-decreasing).
        self._line_evals = self._plot.plot(
            [], [],
            pen=pg.mkPen(theme.ACCENT, width=1.2),
            symbol="o", symbolSize=3,
            symbolBrush=theme.ACCENT, symbolPen=theme.ACCENT,
            name="cost",
        )
        self._line_best = self._plot.plot(
            [], [],
            pen=pg.mkPen(theme.TEXT_2, width=1.6,
                         style=Qt.PenStyle.DashLine),
            name="best so far",
        )
        # Vertical line marking the boundary where CMA-ES hands off to
        # the LS-refinement polish.  Set when we know it.
        self._ls_boundary: pg.InfiniteLine | None = None
        outer.addWidget(self._plot, stretch=1)

        # ---- Emittance plot ------------------------------------------
        # Three traces (εnx/εny/εnz) on a shared X axis with the cost
        # plot.  Linear Y because emittance values differ by only ~10x
        # between transverse and longitudinal and the user wants to see
        # absolute changes, not orders of magnitude.  A pyqtgraph
        # LegendItem is added inside the plot so users can tell the
        # traces apart at a glance.
        self._emit_plot = pg.PlotWidget()
        self._emit_plot.setLabel("left", "εn (mm·mrad)")
        self._emit_plot.setLabel("bottom", "evaluation")
        self._emit_plot.showGrid(x=True, y=True, alpha=0.25)
        self._emit_plot.getAxis("left").setTextPen(theme.TEXT_2)
        self._emit_plot.getAxis("bottom").setTextPen(theme.TEXT_2)
        self._emit_plot.addLegend(offset=(10, 10),
                                  labelTextColor=theme.TEXT_2)
        # Distinct colors: εnx (accent / blueish), εny (warning / orange),
        # εnz (text-0 / lighter).  Same line width / marker style as
        # the cost plot for visual continuity.
        self._line_enx = self._emit_plot.plot(
            [], [],
            pen=pg.mkPen(theme.ACCENT, width=1.4),
            symbol="o", symbolSize=3,
            symbolBrush=theme.ACCENT, symbolPen=theme.ACCENT,
            name="εnx",
        )
        self._line_eny = self._emit_plot.plot(
            [], [],
            pen=pg.mkPen(theme.WARN, width=1.4),
            symbol="t", symbolSize=3,
            symbolBrush=theme.WARN, symbolPen=theme.WARN,
            name="εny",
        )
        self._line_enz = self._emit_plot.plot(
            [], [],
            pen=pg.mkPen(theme.TEXT_0, width=1.4),
            symbol="s", symbolSize=3,
            symbolBrush=theme.TEXT_0, symbolPen=theme.TEXT_0,
            name="εnz",
        )
        outer.addWidget(self._emit_plot, stretch=1)

        # ---- Live values panel ---------------------------------------
        # Two label rows below the plot showing the most recent
        # per-eval payload (emit_nx/ny/nz, W_kin, current element +
        # ADJUST attr being scanned).  Empty until the worker emits its
        # first progress_detail.
        self._live_beam = QLabel("εnx —  εny —  εnz —  W —")
        self._live_beam.setStyleSheet(
            f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; "
            f"font-size:11px; padding:2px 4px;")
        outer.addWidget(self._live_beam)
        self._live_scan = QLabel("(no scan info)")
        self._live_scan.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; "
            f"font-size:11px; padding:2px 4px;")
        outer.addWidget(self._live_scan)

        # ---- Footer row -----------------------------------------------
        footer = QHBoxLayout()
        footer.addStretch(1)
        close = QPushButton("Close")
        close.setStyleSheet(
            f"background:{theme.BG_2}; color:{theme.TEXT_0}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:3px; "
            f"padding:6px 14px;")
        close.clicked.connect(self.accept)
        footer.addWidget(close)
        outer.addLayout(footer)

    # ------------------------------------------------------------------
    # Public slots (called by the matching tab)
    # ------------------------------------------------------------------
    def set_popsize(self, popsize: int) -> None:
        """Late-binding popsize update from the worker.

        Lets the title strip render the `gen` count once CMA-ES has
        reported its actual popsize (or once we've estimated it from
        the variable count).
        """
        self._popsize = max(int(popsize), 0)
        self._dirty = True

    def append_point(self, eval_idx: int, cost: float) -> None:
        """Append one ``(eval_idx, cost)`` sample from the worker."""
        # CMA-ES can briefly emit `cost = 0.0` when a population member
        # hits the feasible plateau on the one-sided constraints.
        # Substitute a small positive sentinel so the log-Y axis is
        # well-defined; we still record the raw value in best_so_far.
        c = max(float(cost), 1e-12)
        self._xs.append(int(eval_idx))
        self._ys.append(c)
        if c < self._best_cost:
            self._best_cost = c
        self._best.append(self._best_cost)
        self._dirty = True

    def update_detail(self, info: dict) -> None:
        """Slot: refresh the live-values panel with the latest per-eval
        payload (emit_nx/ny/nz at end of line, W_kin out, current
        element + ADJUST attr being scanned + pass / step counters).

        Called from the worker's ``progress_detail`` signal.  Cheap:
        ~6 string formats + 2 setText calls per eval; not throttled
        because the rate is whatever the matcher's per-eval rate is
        (~3-30 s/eval for envelope, ~30-300 s/eval for MP).
        """
        # Beam-state row: always show whatever was reported.
        def _fmt(v, unit=""):
            if v is None:
                return "—"
            return f"{float(v):.4f}{unit}"
        beam_line = (
            f"εnx {_fmt(info.get('emit_nx_out'))}  "
            f"εny {_fmt(info.get('emit_ny_out'))}  "
            f"εnz {_fmt(info.get('emit_nz_out'))}  "
            f"W_out {_fmt(info.get('w_kin_out'), ' MeV')}"
        )
        self._live_beam.setText(beam_line)

        # Push the three emittances into their plot buffers.  Use the
        # eval index from the info dict so the X axis matches the cost
        # plot exactly.  None values become NaN so pyqtgraph leaves a
        # gap rather than drawing through them.
        it = info.get("iter")
        if it is not None:
            def _val(key):
                v = info.get(key)
                if v is None:
                    return float("nan")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")
            self._emit_xs.append(int(it))
            self._emit_nx.append(_val("emit_nx_out"))
            self._emit_ny.append(_val("emit_ny_out"))
            self._emit_nz.append(_val("emit_nz_out"))
            self._dirty = True

        # Scan-state row: only meaningful for sequential_scan.
        elem = info.get("element_name")
        if elem is not None:
            pass_n = info.get("pass", "?")
            pass_tot = info.get("total_passes", "?")
            step_n = info.get("step", "?")
            step_tot = info.get("total_steps", "?")
            attrs = info.get("attrs", [])
            xvals = info.get("x_values", {}) or {}
            direction = info.get("direction", 0)
            dir_sym = "+" if direction > 0 else ("−" if direction < 0 else "·")
            attrs_str = ",".join(attrs) if attrs else "?"
            xvals_str = (
                "  ".join(
                    f"{a}={xvals.get(a, float('nan')):.4f}"
                    for a in attrs)
                if xvals else "(no current values)"
            )
            self._live_scan.setText(
                f"pass {pass_n}/{pass_tot}  step {step_n}/{step_tot} ({dir_sym})  "
                f"·  {elem}.[{attrs_str}]  ·  {xvals_str}"
            )
        # else: leave the previous scan text in place (other algorithms
        # don't populate scan_info -- no element-by-element info to show).

    def mark_ls_boundary(self, eval_idx: int) -> None:
        """Vertical line at the CMA-ES → LS polish handoff."""
        if self._ls_boundary is not None:
            return
        line = pg.InfiniteLine(
            pos=eval_idx, angle=90,
            pen=pg.mkPen(theme.WARN, width=1.5,
                         style=Qt.PenStyle.DotLine),
            label="LS polish",
            labelOpts={"position": 0.95, "color": theme.WARN,
                       "fill": (0, 0, 0, 0)})
        self._plot.addItem(line)
        self._ls_boundary = line

    # ------------------------------------------------------------------
    # Internal: throttled paint
    # ------------------------------------------------------------------
    def _maybe_redraw(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        xs = np.asarray(self._xs, dtype=float)
        ys = np.asarray(self._ys, dtype=float)
        best = np.asarray(self._best, dtype=float)
        # Emittance arrays (may be shorter than xs if progress_detail
        # hasn't fired yet; pyqtgraph happily renders shorter traces).
        exs = np.asarray(self._emit_xs, dtype=float)
        enx = np.asarray(self._emit_nx, dtype=float)
        eny = np.asarray(self._emit_ny, dtype=float)
        enz = np.asarray(self._emit_nz, dtype=float)
        if xs.size == 0 and exs.size == 0:
            return
        if xs.size > 0:
            self._line_evals.setData(xs, ys)
            self._line_best.setData(xs, best)
        if exs.size > 0:
            self._line_enx.setData(exs, enx)
            self._line_eny.setData(exs, eny)
            self._line_enz.setData(exs, enz)
        # Title strip mirrors the matching-tab status label so the
        # plot is self-describing if undocked from the tab.  Only
        # renders the cost-summary suffix when the cost buffer has at
        # least one sample; otherwise (emittance arrived first) just
        # show the algo + elapsed.
        elapsed = time.time() - self._start_time
        if xs.size > 0:
            if self._popsize > 0:
                gen = int(xs[-1]) // self._popsize
                counter = f"gen {gen} · eval {int(xs[-1])}"
                if self._max_iter > 0:
                    counter += f"  (gen cap {self._max_iter})"
            else:
                counter = f"eval {int(xs[-1])}"
            cost_now = float(ys[-1])
            cost_best = float(best[-1])
            self._title.setText(
                f"{self._algo} · {counter}  ·  "
                f"cost={cost_now:.3e}  best={cost_best:.3e}  "
                f"elapsed {elapsed:.1f} s"
            )
        else:
            self._title.setText(
                f"{self._algo} · elapsed {elapsed:.1f} s"
            )
