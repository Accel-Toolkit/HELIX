"""Inline manual viewer for the element inspector's ? button.

Reads the chapter's *markdown source* (``docs/manual/...``) and renders
it in a QTextBrowser via ``setMarkdown``.  This bypasses the mkdocs-
built HTML in ``site/`` -- those pages need an HTTP server to look
right (mkdocs-material's sidebar/search/JS assume XHR), and opening
them via ``file://`` produces a broken half-styled view.

Falls back to the file:// browser path (``manual_help.open_for_element``)
only if the markdown source isn't found on disk -- e.g. when running
from a PyInstaller bundle that ships ``site/`` but not ``docs/``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout,
)

from linac_gen_gui.interphase import theme


# Element class → markdown chapter (relative to ``docs/manual/`` root).
# Parallel to ``manual_help._CHAPTER_BY_TYPE`` but pointed at the .md
# source rather than the rendered .html so we don't fight mkdocs-material.
#
# Covers every class the parser can produce + every LatticeCommand
# subclass (all of which appear in the inspector's element list).
# LatticeCommands all route to the single 07_matching/02_set_adjust.md
# chapter because their content is one unified reference.
_SET_ADJUST = "07_matching/02_set_adjust.md"
_CHAPTER_MD_BY_TYPE: dict[str, str] = {
    # ---- Physical elements ----------------------------------------
    "Drift":           "03_elements/01_drift.md",
    "Quadrupole":      "03_elements/02_quadrupole.md",
    "Sextupole":       "03_elements/11_multipole.md",
    "Octupole":        "03_elements/11_multipole.md",
    "Multipole":       "03_elements/11_multipole.md",
    "Solenoid":        "03_elements/03_solenoid.md",
    "Dipole":          "03_elements/04_dipole.md",
    "Edge":            "03_elements/05_edge.md",
    "RFGap":           "03_elements/06_rfgap.md",
    "FieldMap":        "03_elements/07_fieldmap.md",
    "FieldMap3D":      "03_elements/08_fieldmap3d.md",
    "RfqCell":         "03_elements/09_rfqcell.md",
    "VaneRFQ":         "03_elements/10_vanerfq.md",
    "Aperture":        "03_elements/12_aperture.md",
    "Marker":          "03_elements/13_marker.md",
    "Steerer":         "03_elements/14_steerer.md",
    "Foil":            "03_elements/15_foil.md",
    "ThinLens":        "03_elements/00_overview.md",
    "SpaceChargeComp": "03_elements/00_overview.md",
    # ---- LatticeCommand subclasses (SET_* / MIN_* / ADJUST_*) -----
    # All point at the unified SET/ADJUST reference chapter.
    "SetSyncPhase":         _SET_ADJUST,
    "SetBeamPhaseError":    _SET_ADJUST,
    "SetBeamE0P0":          _SET_ADJUST,
    "SetBeamEnergy":        _SET_ADJUST,
    "SetGaussianCutOff":    _SET_ADJUST,
    "SetTwiss":             _SET_ADJUST,
    "SetPosition":          _SET_ADJUST,
    "SetAchromat":          _SET_ADJUST,
    "SetSize":              _SET_ADJUST,
    "SetSizeMax":           _SET_ADJUST,
    "SetSizeMin":           _SET_ADJUST,
    "SetBeamPhaseAdv":      _SET_ADJUST,
    "SetSeparation":        _SET_ADJUST,
    "SetAdv":               _SET_ADJUST,
    "MinEmitGrowth":        _SET_ADJUST,
    "MinEmit4DGrowth":      _SET_ADJUST,
    "MinTransmission":      _SET_ADJUST,
    "SetKeOutMin":          _SET_ADJUST,
    "Adjust":               _SET_ADJUST,
    "AdjustSteerer":        _SET_ADJUST,
    "AdjustSteererBx":      _SET_ADJUST,
    "AdjustSteererBy":      _SET_ADJUST,
    "AdjustBeamTwiss":      _SET_ADJUST,
    "AdjustBeamCentroid":   _SET_ADJUST,
    "AdjustBeamEmit":       _SET_ADJUST,
    "AdjustBeamCurrent":    _SET_ADJUST,
}


def _candidate_md_roots() -> list[Path]:
    """Search the same places ``manual_help`` does, but for the markdown
    source dir (``docs/manual/``) rather than the built ``site/``."""
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass) / "docs" / "manual")
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "docs" / "manual"
        if cand.is_dir():
            roots.append(cand)
            break
    roots.append(Path.cwd() / "docs" / "manual")
    return roots


def _resolve_md(rel: str) -> Optional[Path]:
    for root in _candidate_md_roots():
        p = root / rel
        if p.is_file():
            return p
    return None


class ManualPopup(QDialog):
    """Inline manual viewer.

    Builds a non-modal dialog showing a single chapter's markdown,
    rendered via Qt's built-in markdown engine.  Stays open after the
    user navigates away from the element so they can keep referring
    to it while editing parameters.
    """

    def __init__(self, *, title: str, markdown_text: str,
                 source_path: Optional[Path] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        # Stay-on-top by default so the help stays visible while the
        # user clicks around the main window.  Re-orderable via the
        # OS window manager.
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(f"QDialog {{ background:{theme.BG_0}; }}")
        from linac_gen_gui.interphase.scrollwrap import screen_capped
        self.resize(*screen_capped(self, 880, 720))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # ---- Body: QTextBrowser with markdown rendering --------------
        self._view = QTextBrowser()
        self._view.setOpenExternalLinks(True)
        self._view.setReadOnly(True)
        # Anchor base for relative links inside the markdown (images
        # in the chapter, links to sibling chapters).  Without this,
        # ``[next](14_steerer.md)`` clicks open nothing.
        if source_path is not None:
            self._view.setSearchPaths([str(source_path.parent)])
        body_font = QFont(theme.FONT_BODY if hasattr(theme, "FONT_BODY")
                          else "")
        body_font.setPointSize(11)
        self._view.setFont(body_font)
        self._view.setStyleSheet(
            f"QTextBrowser {{ background:{theme.BG_INSET}; "
            f"color:{theme.TEXT_0}; "
            f"border:1px solid {theme.BORDER_1}; "
            f"border-radius:3px; padding:8px; }}"
        )
        self._view.setMarkdown(markdown_text)
        layout.addWidget(self._view, stretch=1)

        # ---- Footer: source path hint + Close ------------------------
        footer = QHBoxLayout()
        if source_path is not None:
            from PyQt6.QtWidgets import QLabel
            src_lbl = QLabel(f"source: {source_path}")
            src_lbl.setStyleSheet(
                f"color:{theme.TEXT_2}; font-size:10px; "
                f"font-family:{theme.FONT_MONO};"
            )
            src_lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            footer.addWidget(src_lbl)
        footer.addStretch(1)
        close = QPushButton("Close")
        close.setStyleSheet(
            f"background:{theme.BG_2}; color:{theme.TEXT_0}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:3px; "
            f"padding:6px 14px;"
        )
        close.clicked.connect(self.accept)
        footer.addWidget(close)
        layout.addLayout(footer)


def open_inline(element, parent=None) -> tuple[bool, str]:
    """Open the inline manual popup for *element*.

    Returns ``(ok, message)`` so the caller can route the result into
    the status bar.  ``ok=False`` if no markdown chapter is registered
    or the source file isn't on disk (e.g. PyInstaller bundle without
    docs/).  In that case the caller should fall back to
    ``manual_help.open_for_element`` which opens the rendered HTML.
    """
    type_name = type(element).__name__
    rel = _CHAPTER_MD_BY_TYPE.get(type_name)
    if rel is None:
        return False, f"no manual chapter registered for '{type_name}'"
    src = _resolve_md(rel)
    if src is None:
        return False, f"manual source not found: docs/manual/{rel}"
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"failed to read {src}: {exc}"
    dlg = ManualPopup(
        title=f"HELIX manual — {type_name}",
        markdown_text=text,
        source_path=src,
        parent=parent,
    )
    dlg.show()
    return True, f"opened {src.name}"
