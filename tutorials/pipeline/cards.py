"""Title / section / terminal cards rendered with Qt (offscreen).

QPainter gives pixel-identical typography to the GUI with zero new
dependencies.  All cards are 1920x1080 PNG on the HELIX dark palette.
"""
from __future__ import annotations

from pathlib import Path

W, H = 1920, 1080
BG = "#0b1220"
PANEL = "#111a2e"
FG = "#e5eaf5"
DIM = "#8b97ad"
ACCENT = "#4ade80"
PROMPT = "#4ade80"
CMD = "#e5eaf5"
OUT = "#9aa7bd"


def _painter(img):
    from PyQt6.QtGui import QPainter
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    return p


def _img():
    from PyQt6.QtGui import QColor, QImage
    img = QImage(W, H, QImage.Format.Format_RGB32)
    img.fill(QColor(BG))
    return img


def _font(size: int, bold: bool = False, mono: bool = False):
    from PyQt6.QtGui import QFont
    f = QFont("Menlo" if mono else "Helvetica Neue", size)
    f.setBold(bold)
    return f


def title_card(path: str | Path, title: str, subtitle: str = "",
               kicker: str = "HELIX TUTORIALS") -> str:
    """Full-screen episode/section title."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QPen
    img = _img()
    p = _painter(img)
    p.setPen(QColor(ACCENT))
    p.setFont(_font(26, bold=True))
    p.drawText(QRectF(0, H * 0.30, W, 60),
               Qt.AlignmentFlag.AlignHCenter, kicker)
    p.setPen(QColor(FG))
    p.setFont(_font(64, bold=True))
    p.drawText(QRectF(80, H * 0.38, W - 160, 200),
               Qt.AlignmentFlag.AlignHCenter
               | Qt.AlignmentFlag.AlignVCenter, title)
    if subtitle:
        p.setPen(QColor(DIM))
        p.setFont(_font(30))
        p.drawText(QRectF(80, H * 0.58, W - 160, 120),
                   Qt.AlignmentFlag.AlignHCenter, subtitle)
    p.setPen(QPen(QColor(ACCENT), 4))
    p.drawLine(int(W * 0.42), int(H * 0.56), int(W * 0.58), int(H * 0.56))
    p.end()
    img.save(str(path))
    return str(path)


def terminal_card(path: str | Path, lines, title: str = "Terminal") -> str:
    """Fake terminal window: ``lines`` = [("cmd" | "out", text), ...]."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor
    img = _img()
    p = _painter(img)
    # window chrome
    x0, y0, w, h = 160, 120, W - 320, H - 260
    p.setBrush(QColor(PANEL))
    p.setPen(QColor("#233047"))
    p.drawRoundedRect(x0, y0, w, h, 14, 14)
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        p.setBrush(QColor(c))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(x0 + 26 + i * 34, y0 + 22, 16, 16)
    p.setPen(QColor(DIM))
    p.setFont(_font(20, mono=True))
    p.drawText(QRectF(x0, y0 + 14, w, 32),
               Qt.AlignmentFlag.AlignHCenter, title)
    # content
    y = y0 + 92
    p.setFont(_font(26, mono=True))
    for kind, text in lines:
        if kind == "cmd":
            p.setPen(QColor(PROMPT))
            p.drawText(x0 + 44, y, "$")
            p.setPen(QColor(CMD))
            p.drawText(x0 + 78, y, text)
        elif kind == "gap":
            y += 18
            continue
        else:
            p.setPen(QColor(OUT))
            p.drawText(x0 + 78, y, text)
        y += 46
    p.end()
    img.save(str(path))
    return str(path)


def outro_card(path: str | Path, lines) -> str:
    """Closing card with next-steps bullet lines."""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor
    img = _img()
    p = _painter(img)
    p.setPen(QColor(FG))
    p.setFont(_font(54, bold=True))
    p.drawText(QRectF(0, 180, W, 100),
               Qt.AlignmentFlag.AlignHCenter, "You just ran your first beam")
    p.setFont(_font(32))
    y = 400
    for ln in lines:
        p.setPen(QColor(ACCENT))
        p.drawText(560, y, "•")
        p.setPen(QColor(FG))
        p.drawText(610, y, ln)
        y += 74
    p.setPen(QColor(DIM))
    p.setFont(_font(24))
    p.drawText(QRectF(0, H - 160, W, 60),
               Qt.AlignmentFlag.AlignHCenter,
               "github.com/Accel-Toolkit/HELIX")
    p.end()
    img.save(str(path))
    return str(path)
