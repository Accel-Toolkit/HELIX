"""Launch-time splash screen for HELIX.

Shown for ~5 seconds before the main InterphaseWindow appears.  Displays
the HELIX logo, a one-line description, the codebase's last-modified
date (auto-detected from source-tree mtime), and the developer name.

Used from :func:`linac_gen_gui.interphase.app.main`.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme


ASSETS = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS / "helix_logo.png"


def _last_modified_date() -> str:
    """Newest mtime across the package source tree, formatted as YYYY-MM-DD.

    Falls back to today if the source tree can't be located.
    """
    try:
        root = Path(__file__).resolve().parents[2]   # .../HELIX_v3/gui
        newest = max(
            (p.stat().st_mtime for p in root.rglob("*.py") if p.is_file()),
            default=0.0,
        )
        if newest > 0:
            return _dt.datetime.fromtimestamp(newest).strftime("%Y-%m-%d")
    except Exception:
        pass
    return _dt.date.today().isoformat()


class HelixSplash(QDialog):
    """Frameless splash dialog with logo + intro + dev credit.

    Sizes itself to ~520 x 600 px regardless of screen.  Auto-closes
    when the caller's QTimer fires.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setModal(False)
        self.setFixedSize(520, 600)
        self._build_ui()
        self.setStyleSheet(self._qss())

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        card = QFrame(self)
        card.setObjectName("splash_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(14)
        outer.addWidget(card)

        # ---------- logo ----------
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.is_file():
            pm = QPixmap(str(LOGO_PATH)).scaled(
                280, 280,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo.setPixmap(pm)
        else:
            logo.setText("HELIX")
        layout.addWidget(logo)

        # ---------- title ----------
        title = QLabel("HELIX")
        title.setObjectName("splash_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont(); f.setPointSize(28); f.setBold(True); f.setLetterSpacing(
            QFont.SpacingType.PercentageSpacing, 110)
        title.setFont(f)
        layout.addWidget(title)

        # ---------- tagline ----------
        tagline = QLabel("Hybrid Envelope-multiparticle LInac eXplorer")
        tagline.setObjectName("splash_tagline")
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tagline)

        # ---------- intro blurb ----------
        intro = QLabel(
            "Beam-dynamics simulator for linear accelerators —\n"
            "envelope · multiparticle · PIC space charge · ML surrogates."
        )
        intro.setObjectName("splash_intro")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addSpacing(4)

        # ---------- meta row ----------
        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)

        dev = QLabel("Developer  ·  Abhishek Pathak")
        dev.setObjectName("splash_meta")
        meta_row.addWidget(dev, 0, Qt.AlignmentFlag.AlignLeft)

        meta_row.addStretch(1)

        mod = QLabel(f"Last modified  ·  {_last_modified_date()}")
        mod.setObjectName("splash_meta")
        meta_row.addWidget(mod, 0, Qt.AlignmentFlag.AlignRight)

        layout.addLayout(meta_row)

        # ---------- loading bar ----------
        bar = QProgressBar()
        bar.setObjectName("splash_bar")
        bar.setRange(0, 0)               # indeterminate
        bar.setTextVisible(False)
        bar.setFixedHeight(4)
        layout.addWidget(bar)

    def _qss(self) -> str:
        return (
            f"QFrame#splash_card {{"
            f"  background: {theme.BG_1};"
            f"  border: 1px solid {theme.ACCENT};"
            f"  border-radius: 12px;"
            f"}}"
            f"QLabel#splash_title {{"
            f"  color: {theme.ACCENT_2};"
            f"}}"
            f"QLabel#splash_tagline {{"
            f"  color: {theme.TEXT_0};"
            f"  font-size: 13px;"
            f"  font-style: italic;"
            f"}}"
            f"QLabel#splash_intro {{"
            f"  color: {theme.TEXT_1};"
            f"  font-size: 12px;"
            f"}}"
            f"QLabel#splash_meta {{"
            f"  color: {theme.TEXT_2};"
            f"  font-size: 11px;"
            f"}}"
            f"QProgressBar#splash_bar {{"
            f"  background: {theme.BG_INSET};"
            f"  border: 0;"
            f"  border-radius: 2px;"
            f"}}"
            f"QProgressBar#splash_bar::chunk {{"
            f"  background: {theme.ACCENT};"
            f"  border-radius: 2px;"
            f"}}"
        )
