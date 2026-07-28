"""Result card: a clickable tile with icon, title, sparkline, and value.

Each card exposes ``set_results(results)`` to refresh its sparkline from
the live diagnostic recorder and its footer with the end value + trend.
Clicking the card emits ``clicked`` — the Results tab wires that to the
popup that owns the full-size plot.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon


# ---------------------------------------------------------------------------
class _Sparkline(pg.PlotWidget):
    """Tiny axis-less plot embedded in a ResultCard."""

    def __init__(self, accent: str):
        super().__init__()
        self.setBackground(None)
        self.setFixedHeight(58)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        for axis in ("left", "bottom", "top", "right"):
            self.getAxis(axis).hide()
        self.setContentsMargins(0, 0, 0, 0)
        pi = self.getPlotItem()
        pi.hideAxis("left"); pi.hideAxis("bottom")
        pi.setContentsMargins(0, 0, 0, 0)
        pi.getViewBox().setDefaultPadding(0.04)
        r, g, b = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)
        self._curve = self.plot(
            pen=pg.mkPen(accent, width=1.8),
            fillLevel=0,
            brush=pg.mkBrush(r, g, b, 70),
        )

    def set_series(self, xs, ys):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
            self._curve.setData([], []); return
        finite = np.isfinite(ys)
        if not finite.any():
            self._curve.setData([], []); return
        # Re-anchor fillLevel just below the min so auto-range ignores it
        y_min = float(ys[finite].min())
        pad = max((float(ys[finite].max()) - y_min) * 0.25, 1e-6)
        self._curve.setFillLevel(y_min - pad)
        self._curve.setData(xs, ys)
        # Simulate our dataBounds-override so auto-range only sees the curve
        self.getPlotItem().getViewBox().setYRange(
            y_min, float(ys[finite].max()), padding=0.10,
        )


# ---------------------------------------------------------------------------
class ResultCard(QFrame):
    """Clickable tile: icon + title + sparkline + (value, trend).

    Parameters
    ----------
    key
        Identifier; the Results tab looks this up to open the right popup.
    title, icon_name, accent
        Visual fields.
    series_attr
        Name of the attribute on ``results`` to pull the sparkline y-values
        from (e.g. ``"sigma_x"``).  When ``None`` (and no ``series_fn``),
        the card is a plain tile (used for matrix viewers, phase-space, etc.).
    series_fn
        Optional callable ``fn(results) -> ys | (xs, ys) | None`` for
        DERIVED sparklines that aren't a plain results attribute (per-cell
        tune depression, dispersion from Σ cross terms, per-element
        lattice parameters…).  Takes precedence over ``series_attr``; it
        is also called with ``results=None`` so lattice-derived series can
        render before any run.  Exceptions and ``None``/empty returns fall
        back to the "—" placeholder.
    unit
        Short unit string shown next to the end value.
    value_fmt
        ``str.format`` pattern for the end value (default ``"{:.3g}"``).
    """

    clicked = pyqtSignal(str)

    def __init__(self, key: str, title: str, icon_name: str,
                 accent: str = theme.ACCENT,
                 series_attr: str | None = None,
                 unit: str = "",
                 value_fmt: str = "{:.3g}",
                 series_fn=None):
        super().__init__()
        self._key = key
        self._accent = accent
        self._series_attr = series_attr
        self._series_fn = series_fn
        self._unit = unit
        self._value_fmt = value_fmt

        self.setObjectName("resultCard")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(self._qss())
        self.setMinimumSize(QSize(260, 180))

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 12)
        v.setSpacing(6)

        # ---- header: icon + title ------------------------------------
        header = QHBoxLayout(); header.setSpacing(10)
        self._icon_lbl = QLabel()
        self._icon_lbl.setPixmap(icon(icon_name, 22, accent).pixmap(22, 22))
        self._icon_lbl.setFixedSize(26, 26)
        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(
            f"color:{theme.TEXT_0}; font-size:14px; font-weight:600;"
            f" letter-spacing:0.3px;"
        )
        self._title_lbl.setWordWrap(True)
        header.addWidget(self._icon_lbl)
        header.addWidget(self._title_lbl, stretch=1)
        v.addLayout(header)

        # ---- sparkline ----------------------------------------------
        if series_attr is not None or series_fn is not None:
            self._spark = _Sparkline(accent)
            v.addWidget(self._spark)
        else:
            self._spark = None
            # Replace sparkline area with a subtle divider for alignment
            divider = QFrame(); divider.setFixedHeight(58)
            divider.setStyleSheet(
                f"background:{theme.BG_INSET}; border-radius:3px;"
            )
            v.addWidget(divider)

        # ---- footer: end-value + trend ------------------------------
        footer = QHBoxLayout(); footer.setSpacing(6)
        self._value_lbl = QLabel("—")
        self._value_lbl.setStyleSheet(
            f"color:{accent}; font-family:{theme.FONT_MONO};"
            f" font-size:20px; font-weight:700;"
        )
        self._unit_lbl = QLabel(unit)
        self._unit_lbl.setStyleSheet(
            f"color:{theme.TEXT_2}; font-size:12px; padding-bottom:2px;"
        )
        self._trend_lbl = QLabel("")
        self._trend_lbl.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO};"
            f" font-size:13px; font-weight:600;"
        )
        footer.addWidget(self._value_lbl)
        footer.addWidget(self._unit_lbl, alignment=Qt.AlignmentFlag.AlignBottom)
        footer.addStretch(1)
        footer.addWidget(self._trend_lbl)
        v.addLayout(footer)

    # ------------------------------------------------------------------
    def _qss(self) -> str:
        accent = self._accent
        # Build an RGBA glow from the accent hex (used in the hover state)
        r = int(accent[1:3], 16); g = int(accent[3:5], 16); b = int(accent[5:7], 16)
        glow = f"rgba({r},{g},{b},0.18)"
        stripe = accent
        return (
            "QFrame#resultCard {"
            "  background: qlineargradient(x1:0 y1:0 x2:0 y2:1, "
            f"    stop:0 {theme.BG_3}, stop:1 {theme.BG_2});"
            f"  border:1px solid {theme.BORDER_2};"
            f"  border-left: 4px solid {stripe};"
            "  border-radius:10px;"
            "}"
            "QFrame#resultCard:hover {"
            "  background: qlineargradient(x1:0 y1:0 x2:0 y2:1, "
            f"    stop:0 {theme.BG_4}, stop:1 {theme.BG_3});"
            f"  border:1px solid {accent};"
            f"  border-left: 4px solid {accent};"
            "}"
            "QFrame#resultCard QLabel {"
            "  background: transparent;"
            "}"
        )

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._key)
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    def _set_placeholder(self) -> None:
        if self._spark is not None:
            self._spark.set_series([], [])
        self._value_lbl.setText("—")
        self._trend_lbl.setText("")

    def _set_footer(self, y: np.ndarray) -> None:
        """Footer value (end-of-lattice) + trend vs start."""
        y_end = float(y[-1])
        y_start = float(y[0])
        try:
            self._value_lbl.setText(self._value_fmt.format(y_end))
        except Exception:
            self._value_lbl.setText(f"{y_end:.3g}")
        if abs(y_start) > 1e-30:
            pct = (y_end - y_start) / abs(y_start) * 100.0
            arrow = "↑" if pct > 0.5 else ("↓" if pct < -0.5 else "→")
            color = (theme.OK if abs(pct) < 5
                      else theme.WARN if abs(pct) < 25
                      else theme.ERR)
            self._trend_lbl.setStyleSheet(
                f"color:{color}; font-family:{theme.FONT_MONO};"
                f" font-size:11px; font-weight:600;"
            )
            self._trend_lbl.setText(f"{arrow} {pct:+.1f}%")
        else:
            self._trend_lbl.setText("")

    def set_results(self, results) -> None:
        """Update sparkline + footer.  Gracefully handles None."""
        if self._series_fn is not None:
            # Derived series — fn may consume results, the lattice (via a
            # closure), or both; called even with results=None so lattice
            # tiles render before any run.  Any failure → placeholder.
            try:
                out = self._series_fn(results)
            except Exception:                                # noqa: BLE001
                out = None
            if out is None:
                self._set_placeholder()
                return
            if isinstance(out, tuple):
                xs, ys = out
            else:
                ys = out
                s = (np.asarray(getattr(results, "s", []), dtype=float)
                     if results is not None else np.empty(0))
                xs = (s if (s.size and s.size == len(np.atleast_1d(ys)))
                      else np.arange(len(np.atleast_1d(ys)), dtype=float))
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            fin = np.isfinite(ys)
            if ys.size == 0 or xs.size != ys.size or not fin.any():
                self._set_placeholder()
                return
            if self._spark is not None:
                self._spark.set_series(xs, ys)
            self._set_footer(ys[fin])
            return

        if results is None or self._series_attr is None:
            self._set_placeholder()
            return
        s = np.asarray(getattr(results, "s", []), dtype=float)
        y = np.asarray(getattr(results, self._series_attr, []), dtype=float)
        # Envelope mode doesn't record some per-particle / per-loss fields;
        # synthesise them from σ-matrix moments so the sparkline still works.
        # Only kicks in when the recorded array is missing or wrong-length —
        # multiparticle results are passed through unchanged.
        if s.size and y.size != s.size:
            y = _derive_envelope_fallback(self._series_attr, results, s)
        if s.size == 0 or y.size != s.size:
            self._set_placeholder()
            return
        if self._spark is not None:
            self._spark.set_series(s, y)
        self._set_footer(y)


# ---------------------------------------------------------------------------
def _derive_envelope_fallback(attr: str, results, s: np.ndarray) -> np.ndarray:
    """Synthesise an envelope-mode value for one of the per-particle /
    per-loss fields the multi-particle recorder normally provides.

    Returns an array of length ``s.size`` if a sensible derivation exists,
    else an empty array (the caller will then render the "—" placeholder).
    """
    if attr in ("emit_nx", "emit_ny", "emit_nz"):
        beta = np.asarray(getattr(results, "ref_beta", []), dtype=float)
        gamma = np.asarray(getattr(results, "ref_gamma", []), dtype=float)
        if beta.size != s.size or gamma.size != s.size:
            return np.empty(0)
        bg = beta * gamma
        geom_attr = {"emit_nx": "emit_x", "emit_ny": "emit_y",
                     "emit_nz": "emit_z_mmmrad"}[attr]
        e = np.asarray(getattr(results, geom_attr, []), dtype=float)
        return e * bg if e.size == s.size else np.empty(0)
    if attr == "transmission":
        # Envelope solver is lossless by construction → 100% throughout.
        return np.full(s.size, 100.0, dtype=float)
    if attr in ("x_max", "y_max"):
        sigma_attr = "sigma_x" if attr == "x_max" else "sigma_y"
        sg = np.asarray(getattr(results, sigma_attr, []), dtype=float)
        # 5σ envelope (matches TraceWin's aperture convention).
        return 5.0 * sg if sg.size == s.size else np.empty(0)
    return np.empty(0)


# ---------------------------------------------------------------------------
def section_header(text: str) -> QWidget:
    """Accent-styled ribbon header for a tile group."""
    from PyQt6.QtWidgets import QWidget, QHBoxLayout
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    h = QHBoxLayout(w); h.setContentsMargins(0, 14, 0, 8); h.setSpacing(10)
    # Small accent square (a "chip") to the left of the title
    chip = QLabel()
    chip.setFixedSize(10, 10)
    chip.setStyleSheet(
        f"background:{theme.ACCENT}; border-radius:2px;"
    )
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{theme.TEXT_0}; font-size:13px; font-weight:700;"
        f" letter-spacing:3px; background:transparent;"
    )
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.HLine)
    rule.setStyleSheet(
        f"background: qlineargradient(x1:0 y1:0 x2:1 y2:0,"
        f"    stop:0 {theme.BORDER_1}, stop:1 transparent); "
        f"border:0; max-height:1px;"
    )
    h.addWidget(chip)
    h.addWidget(lbl)
    h.addWidget(rule, stretch=1)
    return w
