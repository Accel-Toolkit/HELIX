"""KPI card widget and helpers."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel

from linac_gen_gui.interphase import theme


def kpi_card(label: str, value: str = "—", unit: str = "") -> QFrame:
    card = QFrame()
    card.setProperty("class", "kpi")
    card.setStyleSheet(
        f"QFrame {{ background:{theme.BG_2}; border:1px solid {theme.BORDER_0};"
        f" border-radius:4px; }}"
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(2)

    l = QLabel(label.upper())
    l.setObjectName("label")
    l.setStyleSheet(f"color:{theme.TEXT_3}; font-size:9px; letter-spacing:1px; background:transparent;")
    lay.addWidget(l)

    v_row = QHBoxLayout()
    v_row.setSpacing(4)
    v = QLabel(value)
    v.setObjectName("value")
    v.setStyleSheet(
        f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; font-size:18px; font-weight:500;"
        f"background:transparent;"
    )
    v_row.addWidget(v)
    if unit:
        u = QLabel(unit)
        u.setObjectName("unit")
        u.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px; background:transparent;")
        u.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        v_row.addWidget(u)
    v_row.addStretch(1)
    lay.addLayout(v_row)
    return card


def kpi_set(card: QFrame, value: str) -> None:
    """Mutate the value label of a KPI card."""
    for child in card.findChildren(QLabel):
        if child.objectName() == "value":
            child.setText(value)
            return


def make_kpi_row(labels: list[tuple[str, str]]) -> tuple[QHBoxLayout, dict[str, QFrame]]:
    """Build an HBox with one KPI per ``labels`` entry.  Returns (layout, map)."""
    lay = QHBoxLayout()
    lay.setSpacing(8)
    cards: dict[str, QFrame] = {}
    for name, unit in labels:
        c = kpi_card(name, "—", unit)
        cards[name] = c
        lay.addWidget(c, stretch=1)
    return lay, cards
