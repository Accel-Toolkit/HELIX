"""Scroll-fallback wrapper for dense pages.

With ``widgetResizable=True`` the wrapped page always receives at least
the viewport size (plots keep expanding on big screens); scrollbars
appear only when the viewport drops below the page's minimum size —
which previously made Qt compress widgets below their size hints and
produced overlapping boxes / clipped text on small or scaled screens.
Styling matches the Results tab's in-production scroll area; NoFocus
keeps the wrapper out of Tab-key navigation.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QWidget


def screen_capped(widget: QWidget, w: int, h: int,
                  frac: float = 0.92) -> tuple[int, int]:
    """Cap a requested dialog size to the screen it will appear on.

    Several dialogs hardcode minimum/initial sizes up to 900x720 — on a
    scaled-down laptop profile (looks-like 1168x755) that is taller than
    the usable screen, hiding the bottom button row.  ``frac`` leaves
    headroom for the window frame and dock.
    """
    scr = widget.screen()
    if scr is None:
        from PyQt6.QtGui import QGuiApplication
        scr = QGuiApplication.primaryScreen()
    if scr is None:                      # headless corner case
        return w, h
    avail = scr.availableGeometry()
    return (min(w, int(avail.width() * frac)),
            min(h, int(avail.height() * frac)))


def scroll_wrap(page: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    scroll.setStyleSheet(
        "QScrollArea { background: transparent; border: 0; }"
        "QScrollArea > QWidget > QWidget { background: transparent; }"
    )
    scroll.setWidget(page)
    return scroll
