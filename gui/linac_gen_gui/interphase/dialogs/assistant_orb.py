"""Animated state orb for the assistant panel.

A pure-local QPainter animation (30 fps QTimer) that visualises the
assistant's STATE (and, when listening, the microphone level) — nothing
more.  No audio leaves the machine; the "responding" voiceprint is a
synthesised equalizer, not real TTS amplitude.

Ported from the MIRAGE assistant's orb and re-themed to HELIX's cyan
accent so it reads as native to the app.  States:

    idle · thinking · responding · listening · awaiting-confirm · error
"""
from __future__ import annotations

import math
import random

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

# HELIX palette (cyan accent on dark) with state hues layered on top
_HUE = {
    "idle":            "#22d3ee",   # ACCENT cyan
    "responding":      "#67e8f9",   # ACCENT_2 bright cyan
    "listening":       "#42a5f5",   # blue
    "thinking":        "#fbbf24",   # amber
    "awaiting-confirm": "#f472b6",  # pink (matches HELIX confirm accents)
    "error":           "#ef4444",   # red
    "starting":        "#64748b",   # slate
}

_N_BARS = 44


class AssistantOrb(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.state = "starting"
        self._level = 0.0
        self._pulse = 0.22
        self._phase = 0.0
        self._noise = 0.0
        self._bars = [0.1] * _N_BARS
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)                     # ~30 fps

    # -- inputs ----------------------------------------------------------
    def set_state(self, state: str) -> None:
        self.state = state or "idle"

    def set_level(self, level: float) -> None:
        self._level = min(1.0, max(0.0, float(level) * 8.0))

    def stop(self) -> None:
        self._timer.stop()

    # -- animation -------------------------------------------------------
    def _tick(self) -> None:
        self._phase = (self._phase + 0.03) % (2 * math.pi)
        s = self.state
        if s == "responding":
            self._noise = 0.7 * self._noise + 0.3 * random.random()
            target = 0.35 + 0.6 * self._noise
            kick = int(random.random() * _N_BARS)
            for i in range(_N_BARS):
                nb = (self._bars[i - 1]
                      + self._bars[(i + 1) % _N_BARS]) / 2
                v = 0.55 * self._bars[i] + 0.35 * nb
                if i == kick:
                    v += 0.5 * self._noise + 0.2
                self._bars[i] = min(1.0, max(0.06, v))
        elif s == "listening":
            target = 0.25 + 0.75 * self._level
            self._bars = [b * 0.8 for b in self._bars]
        elif s == "thinking":
            target = 0.45 + 0.15 * math.sin(self._phase * 3)
        elif s == "awaiting-confirm":
            target = 0.5 + 0.45 * math.sin(self._phase * 4)
        elif s == "error":
            target = 0.15
        else:                                     # idle / starting
            target = 0.22 + 0.12 * math.sin(self._phase)
        self._pulse = 0.8 * self._pulse + 0.2 * target
        self.update()

    # -- painting --------------------------------------------------------
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        base = min(w, h) * 0.34
        r = base * (0.75 + 0.30 * self._pulse)
        col = QColor(_HUE.get(self.state, _HUE["idle"]))

        # soft radial glow
        glow = QRadialGradient(cx, cy, r * 2.3)
        g = QColor(col)
        g.setAlpha(int(55 + 90 * self._pulse))
        glow.setColorAt(0.0, g)
        g2 = QColor(col)
        g2.setAlpha(0)
        glow.setColorAt(1.0, g2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(int(cx - r * 2.3), int(cy - r * 2.3),
                      int(r * 4.6), int(r * 4.6))

        # solid core
        core = QColor(col)
        core.setAlpha(235)
        p.setBrush(core)
        p.drawEllipse(int(cx - r * 0.26), int(cy - r * 0.26),
                      int(r * 0.52), int(r * 0.52))

        if self.state == "responding":
            self._paint_voiceprint(p, cx, cy, r, col)
        elif self.state == "thinking":
            self._paint_processing(p, cx, cy, r, col)
        else:
            self._paint_rings(p, cx, cy, r, col)
        p.end()

    def _paint_rings(self, p, cx, cy, r, col):
        p.setBrush(Qt.BrushStyle.NoBrush)
        for factor, width, alpha, speed, span in (
                (0.62, 2.4, 220, 1.0, 250),
                (0.85, 1.8, 160, -0.6, 190),
                (1.06, 1.3, 110, 0.35, 300)):
            c = QColor(col)
            c.setAlpha(int(alpha * (0.5 + 0.5 * self._pulse)))
            pen = QPen(c, width * (0.7 + 0.6 * self._pulse))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            rr = r * factor
            start = int(math.degrees(self._phase) * speed) * 16
            p.drawArc(int(cx - rr), int(cy - rr), int(rr * 2), int(rr * 2),
                      start, span * 16)

    def _paint_voiceprint(self, p, cx, cy, r, col):
        inner = r * 0.55
        for i in range(_N_BARS):
            v = self._bars[i]
            a = 2 * math.pi * i / _N_BARS + self._phase * 0.25
            length = inner * 0.15 + r * 0.75 * v
            c = QColor(col)
            c.setAlpha(int(90 + 165 * v))
            p.setPen(QPen(c, 2.6, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(int(cx + inner * math.cos(a)),
                       int(cy + inner * math.sin(a)),
                       int(cx + (inner + length) * math.cos(a)),
                       int(cy + (inner + length) * math.sin(a)))

    def _paint_processing(self, p, cx, cy, r, col):
        p.setBrush(Qt.BrushStyle.NoBrush)
        for factor, speed, segs in ((0.66, 2.2, 3), (0.92, -1.4, 4),
                                    (1.14, 0.8, 6)):
            c = QColor(col)
            c.setAlpha(int(180 * (0.5 + 0.5 * self._pulse)))
            p.setPen(QPen(c, 2.0, cap=Qt.PenCapStyle.RoundCap))
            rr = r * factor
            for k in range(segs):
                start = int(math.degrees(self._phase) * speed
                            + k * 360 / segs) * 16
                p.drawArc(int(cx - rr), int(cy - rr),
                          int(rr * 2), int(rr * 2),
                          start, int(360 / segs * 0.55) * 16)
        # orbiting satellites
        for k in range(3):
            a = self._phase * (2.0 + 0.5 * k) + k * 2.1
            orad = r * (0.66 + 0.24 * k)
            d = 3.5 + 1.5 * math.sin(self._phase * 5 + k)
            c = QColor(col)
            c.setAlpha(220)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(int(cx + orad * math.cos(a) - d),
                          int(cy + orad * math.sin(a) - d),
                          int(d * 2), int(d * 2))
