"""The HELIX Nebula — the assistant's animated presence.

A pure-local QPainter animation (30 fps QTimer) that visualises the
assistant's STATE and the microphone level — nothing more.  No audio
leaves the machine; the "responding" voiceprint is a synthesised
equalizer, not real TTS amplitude.

2026-07-29 full redesign — an improvisation on the MIRAGE orb's
additive-light language (no painted sphere; light ADDS like light via
``CompositionMode_Plus``), pushed into real 3-D:

  double helix     the app's own name drawn in light: two intertwined
                   particle strands wound around a ring (a DNA helix
                   bent into a torus), living in true xyz — rotated by
                   a precessing rotation matrix and perspective-
                   projected, so nearer particles are larger and
                   brighter and the whole form turns with genuine
                   parallax
  star shell       ~120 dim motes on a 3-D sphere around it, counter-
                   rotating — volumetric depth behind the strands
  glow stack       halo → bloom → hot core with a white heart
  voice physics    a fast-attack envelope makes your voice BOIL the
                   strands, flare the bloom, and launch expanding
                   shockwave rings in the tilted galaxy plane
  state layers     thinking: strands accelerate and grow comet trails;
                   responding: two-pass glowing voice-print equalizer;
                   confirm: attention pulses; off: dormant ember
  color            the hue MORPHS smoothly between states (exponential
                   lerp per tick) — it never snaps

States: idle · thinking · responding · listening · awaiting-confirm
· error (+ starting/off dormant).
"""
from __future__ import annotations

import math
import random

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QSizePolicy, QWidget

# HELIX palette (cyan accent on dark) with state hues layered on top
_HUE = {
    "idle":            "#22d3ee",   # ACCENT cyan
    "responding":      "#67e8f9",   # ACCENT_2 bright cyan
    "listening":       "#42a5f5",   # blue
    "thinking":        "#fbbf24",   # amber
    "awaiting-confirm": "#f472b6",  # pink (matches HELIX confirm accents)
    "error":           "#ef4444",   # red
    "starting":        "#64748b",   # slate
    "off":             "#64748b",   # hands-free disabled — gray, static
}

_N_BARS = 44
_N_STRAND = 88                       # particles per helix strand
_WINDINGS = 9                        # helix turns around the ring
_N_STARS = 120
_TRAIL = 7
_ALIVE = ("idle", "listening", "responding", "thinking",
          "awaiting-confirm")


class AssistantOrb(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)               # the orb IS the face now
        # Expanding: in the voice-only view the panel gives the orb
        # stretch and it fills the WHOLE window, scaling with min(w, h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.state = "starting"
        self._level = 0.0
        #: fast-attack / slow-decay voice envelope — a single spoken word
        #: must visibly kick the orb even between 30 fps ticks
        self._env = 0.0
        self._pulse = 0.22
        self._phase = 0.0
        self._yaw = 0.0                          # 3-D rotation angle
        self._noise = 0.0
        self._bars = [0.1] * _N_BARS
        self._col = QColor(_HUE["starting"])     # morphing hue
        # helix strand parameterisation (stable per-particle jitter)
        rng = random.Random(20260729)
        self._us = [i / _N_STRAND * 2 * math.pi + rng.uniform(-0.01, 0.01)
                    for i in range(_N_STRAND)]
        self._jit = [rng.uniform(0.6, 1.4) for _ in range(_N_STRAND)]
        self._trails: list[list[list[tuple[float, float]]]] = [
            [[] for _ in range(_N_STRAND)] for _ in range(2)]
        # ambient star shell: random points on a sphere, r in 1.5..2.1
        self._stars = []
        for _ in range(_N_STARS):
            z = rng.uniform(-1.0, 1.0)
            a = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(1.5, 2.1)
            s = math.sqrt(max(0.0, 1 - z * z))
            self._stars.append((rr * s * math.cos(a), rr * z,
                               rr * s * math.sin(a),
                               rng.uniform(0.5, 1.0)))
        self._waves: list[list[float]] = []      # [scale, alpha]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)                     # ~30 fps

    # -- inputs ----------------------------------------------------------
    def set_state(self, state: str) -> None:
        self.state = state or "idle"

    def set_level(self, level: float) -> None:
        self._level = min(1.0, max(0.0, float(level) * 8.0))
        if self._level > self._env:               # instant attack
            self._env = self._level

    def stop(self) -> None:
        self._timer.stop()

    # -- animation -------------------------------------------------------
    def _tick(self) -> None:
        self._phase = (self._phase + 0.03) % (2 * math.pi)
        self._env *= 0.90                         # slow decay (~0.3 s tail)
        s = self.state
        if s == "responding":
            self._noise = 0.7 * self._noise + 0.3 * random.random()
            target = 0.35 + 0.6 * self._noise
            self._dance_bars(self._noise, 0.5, 0.2)
        elif s == "listening":
            # the equalizer + strand shimmer are LIVE mic meters
            target = 0.30 + 0.70 * self._env
            self._dance_bars(self._env, 0.9, 0.05)
            if self._env > 0.14 and random.random() < 0.5:
                self._waves.append([0.55, 170.0])    # voice shockwave
        elif s == "thinking":
            target = 0.45 + 0.15 * math.sin(self._phase * 3)
        elif s == "awaiting-confirm":
            target = 0.5 + 0.45 * math.sin(self._phase * 4)
            if random.random() < 0.12:
                self._waves.append([0.5, 140.0])     # attention pulse
        elif s == "error":
            target = 0.15
        elif s == "off":                          # listening disabled
            target = 0.14
        else:                                     # idle / starting
            # MIRAGE parity: while hands-free idles, the orb BREATHES
            # and visibly reacts to the room — proof it is hearing you
            # before you ever say the wake word.
            target = (0.22 + 0.12 * math.sin(self._phase)
                      + 0.55 * self._level)
            if self._env > 0.04:                  # words ripple the ring
                self._dance_bars(self._env, 0.6, 0.0)
                if self._env > 0.25 and random.random() < 0.35:
                    self._waves.append([0.55, 150.0])
            else:
                self._bars = [b * 0.85 for b in self._bars]
        self._pulse = 0.8 * self._pulse + 0.2 * target

        # hue morph — never snap (MIRAGE heritage)
        want = QColor(_HUE.get(s, _HUE["idle"]))
        k = 0.12
        self._col = QColor(
            int(self._col.red() + k * (want.red() - self._col.red())),
            int(self._col.green() + k * (want.green() - self._col.green())),
            int(self._col.blue() + k * (want.blue() - self._col.blue())))

        # 3-D rotation: voice and thought spin the galaxy faster
        if s in _ALIVE:
            spin = 0.010 + 0.012 * self._pulse + 0.030 * self._env
            if s == "thinking":
                spin += 0.022
            self._yaw = (self._yaw + spin) % (2 * math.pi)

        # shockwaves expand + fade
        for wv in self._waves:
            wv[0] += 0.055 + 0.05 * self._env
            wv[1] *= 0.90
        self._waves = [wv for wv in self._waves if wv[1] > 6]
        self.update()

    def _dance_bars(self, drive: float, kick_gain: float,
                    kick_floor: float) -> None:
        """Equalizer dynamics: neighbour diffusion + a random kick
        proportional to ``drive`` — the ring shimmers like a voiceprint."""
        kick = int(random.random() * _N_BARS)
        for i in range(_N_BARS):
            nb = (self._bars[i - 1] + self._bars[(i + 1) % _N_BARS]) / 2
            v = 0.55 * self._bars[i] + 0.35 * nb
            if i == kick:
                v += kick_gain * drive + kick_floor
            self._bars[i] = min(1.0, max(0.05, v))

    # -- 3-D projection --------------------------------------------------
    def _project(self, x, y, z, cx, cy, scale, cy_, sy_, ct, st):
        """Yaw about Y, tilt about X, weak perspective.  Returns
        (sx, sy, depth 0..1 — 1 is nearest)."""
        x2 = x * cy_ + z * sy_
        z2 = -x * sy_ + z * cy_
        y2 = y * ct - z2 * st
        z3 = y * st + z2 * ct
        pf = 1.0 / (1.0 + 0.30 * (z3 / 2.2))     # weak perspective
        d = 0.5 - 0.5 * (z3 / 2.2)               # depth cue
        return (cx + scale * x2 * pf, cy + scale * y2 * pf,
                min(1.0, max(0.0, d)))

    # -- painting --------------------------------------------------------
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        base = min(w, h) * 0.30
        r = base * (0.72 + 0.30 * self._pulse + 0.08 * self._env)
        col = self._col
        alive = self.state in _ALIVE

        # LIGHT ADDS LIKE LIGHT — the MIRAGE rule, kept everywhere
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        # glow stack: halo → bloom (flare with voice)
        for rf, a0 in ((2.7, 22 + 42 * self._pulse + 40 * self._env),
                       (1.55, 40 + 66 * self._pulse + 70 * self._env)):
            g = QRadialGradient(cx, cy, r * rf)
            gc = QColor(col)
            gc.setAlpha(int(min(190.0, a0)))
            g.setColorAt(0.0, gc)
            g.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(g)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRectF(cx - r * rf, cy - r * rf,
                                 r * rf * 2, r * rf * 2))

        # precompute the rotation for this frame
        tilt = 0.52 + 0.10 * math.sin(self._phase * 0.5)   # precession
        cy_, sy_ = math.cos(self._yaw), math.sin(self._yaw)
        ct, st = math.cos(tilt), math.sin(tilt)
        scale = r * 0.72

        # ambient star shell (counter-rotating, dim, depth-shaded)
        cyb, syb = math.cos(-self._yaw * 0.35), math.sin(-self._yaw * 0.35)
        p.setPen(Qt.PenStyle.NoPen)
        star_amp = 1.0 if alive else 0.35
        for x, y, z, b in self._stars:
            sx, sy, d = self._project(x, y, z, cx, cy, scale,
                                      cyb, syb, ct, st)
            a = int((14 + 60 * d * b) * (0.5 + 0.5 * self._pulse)
                    * star_amp)
            c = QColor(col)
            c.setAlpha(a)
            p.setBrush(c)
            ds = (0.6 + 1.5 * d) * max(1.0, r / 90.0)
            p.drawEllipse(QRectF(sx - ds, sy - ds, ds * 2, ds * 2))

        if alive:
            # voice shockwaves: rings living IN the tilted galaxy plane
            p.setBrush(Qt.BrushStyle.NoBrush)
            for scale_w, alpha in self._waves:
                rr = r * scale_w
                c = QColor(col)
                c.setAlpha(int(alpha))
                pen = QPen(c, max(1.4, r * 0.02))
                p.setPen(pen)
                p.drawEllipse(QRectF(cx - rr, cy - rr * ct,
                                     rr * 2, rr * ct * 2))

            # THE DOUBLE HELIX — two particle strands in true 3-D
            thinking = self.state == "thinking"
            boil = 1.0 + 0.45 * self._env
            R, a_t = 1.55, 0.42 * boil
            listen_pull = 0.85 if self.state == "listening" else 1.0
            for strand in range(2):
                psi = strand * math.pi
                trails = self._trails[strand]
                for i, u in enumerate(self._us):
                    ph = _WINDINGS * u + psi + self._phase * 1.7
                    rad = (R + a_t * self._jit[i] * math.cos(ph)) \
                        * listen_pull
                    x = rad * math.cos(u)
                    z = rad * math.sin(u)
                    y = a_t * 1.25 * math.sin(ph)
                    sx, sy, d = self._project(x, y, z, cx, cy, scale,
                                              cy_, sy_, ct, st)
                    hist = trails[i]
                    if thinking:
                        hist.append((sx, sy))
                        if len(hist) > _TRAIL:
                            hist.pop(0)
                        if len(hist) > 1:
                            pen = QPen(col)
                            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                            for j in range(1, len(hist)):
                                c = QColor(col)
                                c.setAlpha(int(8 + 50 * j / len(hist) * d))
                                pen.setColor(c)
                                pen.setWidthF(
                                    max(0.8, r * 0.012) * j / len(hist))
                                p.setPen(pen)
                                p.drawLine(int(hist[j - 1][0]),
                                           int(hist[j - 1][1]),
                                           int(hist[j][0]),
                                           int(hist[j][1]))
                    elif hist:
                        hist.pop(0)
                    # per-particle shimmer: mic level licks the strands
                    bar = self._bars[i % _N_BARS]
                    glow = 0.35 + 0.65 * self._pulse + 0.6 * self._env * bar
                    c = QColor(col.lighter(120 + int(60 * d)))
                    c.setAlpha(int(min(235.0, (30 + 190 * d) * glow)))
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(c)
                    ds = (0.8 + 2.4 * d) * max(1.0, r / 80.0) \
                        * (1.0 + 0.5 * self._env * bar)
                    p.drawEllipse(QRectF(sx - ds, sy - ds, ds * 2, ds * 2))

            # state layer OUTSIDE the galaxy
            if self.state == "responding":
                self._paint_voiceprint(p, cx, cy, r, col)
            elif self.state == "thinking":
                self._paint_processing(p, cx, cy, r, col)
            elif self.state == "listening" and self._env > 0.03:
                self._paint_voiceprint(p, cx, cy, r, col)

        # hot core with a white heart (dim ember when dormant)
        heart_a = (150 + 90 * self._pulse) if alive else 70
        core = QRadialGradient(cx, cy, r * 0.36)
        core.setColorAt(0.0, QColor(255, 255, 255, int(heart_a)))
        mid = QColor(col)
        mid.setAlpha(200 if alive else 90)
        core.setColorAt(0.45, mid)
        core.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(core)
        p.drawEllipse(QRectF(cx - r * 0.36, cy - r * 0.36,
                             r * 0.72, r * 0.72))
        p.end()

    # -- overlays --------------------------------------------------------
    def _paint_voiceprint(self, p, cx, cy, r, col):
        """Radial equalizer, two passes: wide faint glow + thin bright
        (MIRAGE's double-pass trick — reads as neon)."""
        inner = r * 1.18
        for width, amul in ((6.0, 0.35), (2.4, 1.0)):
            pen = QPen(col)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            for i, v in enumerate(self._bars):
                a = 2 * math.pi * i / _N_BARS + self._phase * 0.25
                length = r * 0.06 + r * 0.5 * v
                c = QColor(col)
                c.setAlpha(int((70 + 180 * v) * amul))
                pen.setColor(c)
                pen.setWidthF(width)
                p.setPen(pen)
                p.drawLine(int(cx + inner * math.cos(a)),
                           int(cy + inner * math.sin(a)),
                           int(cx + (inner + length) * math.cos(a)),
                           int(cy + (inner + length) * math.sin(a)))

    def _paint_processing(self, p, cx, cy, r, col):
        """Counter-rotating segmented arcs while the strands grow
        comet tails."""
        p.setBrush(Qt.BrushStyle.NoBrush)
        for rf, speed, nseg in ((1.18, 2.2, 3), (1.32, -1.4, 4),
                                (1.48, 0.8, 6)):
            rr = r * rf
            rect = QRectF(cx - rr, cy - rr, rr * 2, rr * 2)
            pen = QPen(QColor(col.red(), col.green(), col.blue(), 170))
            pen.setWidthF(2.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            start = math.degrees(self._phase) * speed * 2
            seg = 360.0 / nseg
            for k in range(nseg):
                p.drawArc(rect, int((start + k * seg) * 16),
                          int(seg * 0.55 * 16))
