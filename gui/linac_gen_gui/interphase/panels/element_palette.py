"""Element Palette — vertical strip of draggable type cards.

Cards carry an MIME payload of ``application/x-linacgen-element-type``
holding the type-name as utf-8 bytes.  Drop targets (LatticeTimeline,
OutlineTree) decode that name, build a placeholder element via
``element_factory.make_default``, and issue an ``InsertCommand``.

Double-clicking a card adds the element at the current selection (or
appends to the end), so the palette is usable without drag-drop on
touch / accessibility setups.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QMimeData, QSize, QPoint
from PyQt6.QtGui import (
    QDrag, QPainter, QColor, QPen, QBrush, QFont, QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QSizePolicy, QToolButton, QScrollArea,
    QWidget,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.element_factory import supported_types


PALETTE_MIME = "application/x-linacgen-element-type"

# Compact display-name overrides — keep cards readable at 64 px wide.
_LABELS: dict[str, str] = {
    "Quadrupole":      "Quad",
    "SpaceChargeComp": "SCC",
    "FieldMap3D":      "FMap3D",
    "FieldMap":        "FMap",
    "RfqCell":         "RFQ Cell",
    "Solenoid":        "Sol",
    "Dipole":          "Bend",
    "Steerer":         "Steerer",
    "Foil":            "Foil",
    "Aperture":        "Apert",
    "Marker":          "Mark",
    "ThinLens":        "Lens",
    "Drift":           "Drift",
    "RFGap":           "RFGap",
    "Sextupole":       "Sext",
    "Octupole":        "Oct",
    "Multipole":       "Mpole",
    # SET_*/ADJUST_* commands — strip the noisy prefix and keep the
    # essential keyword.  The full name shows up on hover.
    "SET_SYNC_PHASE":        "SyncPhi",
    "SET_BEAM_PHASE_ERROR":  "ϕ-err",
    "SET_BEAM_ENERGY":       "E-set",
    "SET_BEAM_E0P0":         "E0P0",
    "SET_GAUSS_CUTOFF":      "GCutoff",
    "SET_TWISS":             "Twiss",
    "SET_POSITION":          "Pos",
    "SET_ACHROMAT":          "Achrom",
    "SET_SIZE":              "Size",
    "SET_SIZE_MAX":          "SizeMax",
    "SET_SIZE_MIN":          "SizeMin",
    "SET_BEAM_PHASE_ADV":    "PhAdv",
    "SET_SEPARATION":        "Sep",
    "SET_ADV":               "Adv",
    "ADJUST":                "Adjust",
    "ADJUST_STEERER":        "AdjStr",
    "ADJUST_STEERER_BX":     "AdjStrX",
    "ADJUST_STEERER_BY":     "AdjStrY",
    "ADJUST_BEAM_TWISS":     "AdjTw",
    "ADJUST_BEAM_CENTROID":  "AdjCen",
    "ADJUST_BEAM_EMIT":      "AdjEm",
    "ADJUST_BEAM_CURRENT":   "AdjCur",
}


class _PaletteCard(QToolButton):
    """A single draggable type card.

    Drag starts on `mouseMoveEvent` once the user has moved past Qt's
    drag-distance threshold; until then the card behaves as a normal
    button (single click selects, double click triggers ``onActivate``).
    """

    def __init__(self, type_name: str, *, on_activate=None):
        super().__init__()
        self._type_name = type_name
        self._on_activate = on_activate
        self._press_pos: QPoint | None = None

        color = QColor(theme.EL_COLORS.get(type_name, theme.TEXT_2))
        label = _LABELS.get(type_name, type_name)
        self.setText(label)
        self.setToolTip(
            f"<b>{type_name}</b><br><span style='color:{theme.TEXT_2}'>"
            f"Drag onto the lattice — or double-click to append.</span>"
        )
        self.setFixedSize(QSize(56, 40))
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setStyleSheet(self._qss(color))

    def _qss(self, color: QColor) -> str:
        # Border tinted with the element's own colour, fill is dark.
        bg, hover = theme.BG_3, theme.BG_4
        return (
            f"QToolButton {{ background:{bg}; color:{theme.TEXT_0}; "
            f"border:1px solid {color.name()}; border-left:3px solid {color.name()}; "
            f"border-radius:3px; padding:0 4px; "
            f"font-family:{theme.FONT_MONO}; font-size:10px; font-weight:600;}}"
            f"QToolButton:hover {{ background:{hover}; }}"
            f"QToolButton:pressed {{ background:{theme.BG_INSET}; }}"
        )

    # ------------------------------------------------------------------
    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.pos()
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        if self._on_activate is not None:
            self._on_activate(self._type_name)
        super().mouseDoubleClickEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._press_pos is None or not (ev.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(ev); return
        if (ev.pos() - self._press_pos).manhattanLength() < 6:
            return  # too small a move; treat as click
        # --- start drag ------------------------------------------------
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(PALETTE_MIME, self._type_name.encode("utf-8"))
        # Plain-text fallback so drops onto debug widgets are visible.
        mime.setText(f"linacgen:{self._type_name}")
        drag.setMimeData(mime)
        drag.setPixmap(self._make_drag_pixmap())
        drag.setHotSpot(QPoint(28, 20))
        drag.exec(Qt.DropAction.CopyAction)
        self._press_pos = None

    def _make_drag_pixmap(self) -> QPixmap:
        color = QColor(theme.EL_COLORS.get(self._type_name, theme.TEXT_2))
        pm = QPixmap(56, 40); pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(QBrush(QColor(theme.BG_3)))
        p.setPen(QPen(color, 2))
        p.drawRoundedRect(1, 1, 54, 38, 4, 4)
        p.setPen(QPen(QColor(theme.TEXT_0)))
        f = QFont(theme.FONT_MONO, 8); f.setBold(True); p.setFont(f)
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter,
                   _LABELS.get(self._type_name, self._type_name))
        p.end()
        return pm


class ElementPalette(QFrame):
    """Vertical strip of palette cards.

    Sits to the LEFT of the lattice tab's body, between the outline
    tree and the timeline.  Width is fixed at 78 px so cards have
    breathing room without crowding the timeline.
    """

    def __init__(self, on_activate=None) -> None:
        super().__init__()
        self.setObjectName("element_palette")
        self.setFixedWidth(78)
        self.setStyleSheet(
            f"#element_palette {{ background:{theme.BG_1}; "
            f"border-right:1px solid {theme.BORDER_0}; }}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        title = QLabel("PALETTE")
        title.setStyleSheet(
            f"color:{theme.TEXT_3}; font-size:9px; letter-spacing:1px;"
            f"font-weight:700; padding:8px 8px 4px 8px;"
        )
        outer.addWidget(title)

        # Wrap the cards in a scroll area so the SET_*/ADJUST_* group
        # doesn't overflow the available height (palette holds ~38
        # types now).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(8, 4, 8, 8); v.setSpacing(6)

        # Group cards into logical sections so 38 types are still
        # navigable.  Each section header is a small caps label with a
        # tinted background.
        sections: list[tuple[str, list[str]]] = [
            ("Magnets",       ["Quadrupole", "Dipole", "Solenoid",
                               "Steerer", "Sextupole", "Octupole",
                               "Multipole"]),
            ("Cavities",      ["RFGap", "FieldMap", "FieldMap3D",
                               "RfqCell"]),
            ("Drift / passive", ["Drift", "ThinLens", "Aperture",
                                 "Marker", "SpaceChargeComp"]),
            ("SET_* commands", [t for t in supported_types()
                                if t.startswith("SET_")]),
            ("ADJUST_* commands", [t for t in supported_types()
                                   if t.startswith("ADJUST")]),
        ]
        seen: set[str] = set()
        for header, names in sections:
            present = [n for n in names if n in supported_types()]
            if not present:
                continue
            lab = QLabel(header.upper())
            lab.setStyleSheet(
                f"color:{theme.TEXT_3}; font-size:8px; letter-spacing:1px;"
                f"padding:6px 0 2px 0;"
            )
            v.addWidget(lab)
            for tname in present:
                v.addWidget(_PaletteCard(tname, on_activate=on_activate))
                seen.add(tname)
        # Anything not categorised above (defensive — keeps the palette
        # truthful if someone adds a new factory entry without updating
        # the section list).
        leftover = [t for t in supported_types() if t not in seen]
        if leftover:
            lab = QLabel("OTHER")
            lab.setStyleSheet(
                f"color:{theme.TEXT_3}; font-size:8px; letter-spacing:1px;"
                f"padding:6px 0 2px 0;"
            )
            v.addWidget(lab)
            for tname in leftover:
                v.addWidget(_PaletteCard(tname, on_activate=on_activate))
        v.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
