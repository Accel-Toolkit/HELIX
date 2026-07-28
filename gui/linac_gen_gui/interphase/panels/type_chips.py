"""Element-type LED chip strip — a clickable filter row above the timeline.

Each chip shows ``Type · count`` painted in the type's brand colour.
Clicking toggles whether bars of that type are shown at full opacity or
dimmed in the LatticeTimeline.  An empty filter set means "show all".
"""
from __future__ import annotations

from collections import Counter

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QPushButton, QSizePolicy,
)

from linac_gen_gui.interphase import theme


class _Chip(QPushButton):
    def __init__(self, type_name: str, count: int, *, on_toggle):
        super().__init__()
        self._type_name = type_name
        self._on_toggle = on_toggle
        self._active = False
        color = QColor(theme.EL_COLORS.get(type_name, theme.TEXT_2))
        self._color = color
        self.setText(f"{type_name} · {count}")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(20)
        self._refresh_qss()
        self.toggled.connect(self._on_toggled)

    def _refresh_qss(self) -> None:
        c = self._color.name()
        if self._active:
            self.setStyleSheet(
                f"QPushButton {{ background:{c}; color:#06121a; "
                f"border:1px solid {c}; border-radius:10px; padding:0 10px; "
                f"font-family:{theme.FONT_MONO}; font-size:10px; font-weight:700; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ background:transparent; color:{theme.TEXT_1}; "
                f"border:1px solid {c}; border-radius:10px; padding:0 10px; "
                f"font-family:{theme.FONT_MONO}; font-size:10px; }}"
                f"QPushButton:hover {{ background:{theme.BG_4}; }}"
            )

    def _on_toggled(self, checked: bool) -> None:
        self._active = checked
        self._refresh_qss()
        self._on_toggle(self._type_name, checked)


class TypeChipStrip(QFrame):
    """Row of clickable type-filter chips."""

    filter_changed = pyqtSignal(set)

    def __init__(self):
        super().__init__()
        self.setObjectName("type_chips")
        self.setStyleSheet(
            f"#type_chips {{ background:transparent; }}"
        )
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0); self._lay.setSpacing(6)
        self._lay.addStretch(1)
        self._chips: dict[str, _Chip] = {}
        self._active: set[str] = set()

    def set_lattice(self, lattice) -> None:
        # Clear existing chips.
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._chips.clear()
        # Reset filter so a freshly-loaded lattice shows everything.
        self._active.clear()
        if lattice is None or not lattice.elements:
            self._lay.addStretch(1)
            self.filter_changed.emit(set())
            return
        counts = Counter(type(e).__name__ for e in lattice.elements)
        # Sort by descending count so the heaviest types come first.
        for type_name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            chip = _Chip(type_name, n, on_toggle=self._on_chip_toggled)
            self._lay.addWidget(chip)
            self._chips[type_name] = chip
        self._lay.addStretch(1)
        self.filter_changed.emit(set())

    def _on_chip_toggled(self, type_name: str, active: bool) -> None:
        if active:
            self._active.add(type_name)
        else:
            self._active.discard(type_name)
        self.filter_changed.emit(set(self._active))
