"""Sequential textual representation of the lattice.

Read-only one-line-per-element listing in TraceWin-card style.  Clicking
a row selects that element in :class:`AppState`; conversely, when the
selection changes elsewhere (timeline, outline tree, inspector) the
matching row is highlighted and scrolled into view.

The point is to give the user a *positional* index — counts running
``#001`` upward — that mirrors what a TraceWin .dat would look like, so
they always know where in the chain they are.  ``bus.changed`` triggers
a per-row refresh (no full rebuild) so live spinbox edits update the
visible card immediately.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QBrush, QFont
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.manual_help import open_for_element
from linac_gen_gui.interphase.state import AppState


# ---------------------------------------------------------------------------
# Per-type compact card formatter.  Each formatter is allowed to read
# whatever attributes it needs off the element; missing attrs fall back to
# ``getattr(...,  default)`` so a partially-constructed element still
# renders something sensible.
# ---------------------------------------------------------------------------
def _f(v, default="-"):
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _card_for(elem) -> tuple[str, str]:
    """Return ``(keyword, args)`` for ``elem`` in TraceWin-style.

    Returned as two strings so the listing can pad the keyword column."""
    type_name = type(elem).__name__
    g = lambda a, d=None: getattr(elem, a, d)

    if type_name == "Freq":
        return "FREQ", f"f={_f(g('frequency_mhz'))}MHz"
    if type_name == "Drift":
        return "DRIFT", f"L={_f(g('length'))}mm  R={_f(g('aperture'))}mm"
    if type_name == "Quadrupole":
        return "QUAD", (
            f"L={_f(g('length'))}mm  G={_f(g('gradient'))}T/m  "
            f"R={_f(g('aperture'))}mm"
        )
    if type_name == "Solenoid":
        return "SOLENOID", (
            f"L={_f(g('length'))}mm  Bz={_f(g('field'))}T  "
            f"R={_f(g('aperture'))}mm"
        )
    if type_name == "Dipole":
        return "BEND", (
            f"angle={_f(g('angle'))}rad  ρ={_f(g('rho'))}mm  "
            f"R={_f(g('aperture'))}mm"
        )
    if type_name == "Edge":
        return "EDGE", f"β={_f(g('pole_rotation'))}rad  ρ={_f(g('rho'))}mm"
    if type_name == "RFGap":
        return "GAP", (
            f"V={_f(g('voltage'))}MV  φ={_f(g('phase'))}°  "
            f"f={_f(g('frequency'))}MHz"
        )
    if type_name == "Steerer":
        return "THIN_STEERING", f"BLx={_f(g('bx_l'))}T·m  BLy={_f(g('by_l'))}T·m"
    if type_name == "Aperture":
        return "APERTURE", f"dx={_f(g('dx'))}mm  dy={_f(g('dy'))}mm  type={_f(g('aperture_type'))}"
    if type_name == "Marker":
        return ("DIAG_PHASE" if g("snapshot", False) else "MARKER"), ""
    if type_name == "SpaceChargeComp":
        return "SPACE_CHARGE_COMP", f"k={_f(g('factor'))}"
    if type_name == "ThinLens":
        return "THIN_LENS", f"fx={_f(g('fx'))}mm  fy={_f(g('fy'))}mm"
    if type_name == "RfqCell":
        return "RFQ_CELL", (
            f"V={_f(g('voltage_V'))}V  r0={_f(g('r0_mm'))}mm  "
            f"A10={_f(g('A10'))}  m={_f(g('modulation'))}  "
            f"L={_f(g('length'))}mm  φs={_f(g('phi_s_deg'))}°  "
            f"type={_f(g('cell_type'))}"
        )
    if type_name == "VaneRFQ":
        n_cells = len(g("cells", []) or [])
        return "VANE_RFQ", f"L={_f(g('length'))}mm  cells={n_cells}"
    if type_name == "NCells":
        return "NCELLS", (
            f"mode={g('mode')}  Nc={g('n_cells')}  βg={_f(g('beta_g'))}  "
            f"EoT={_f(g('eot_v_per_m'))}V/m  θs={_f(g('theta_s_deg'))}°  "
            f"P={g('p_flag')}"
        )
    if type_name == "FieldMap":
        return "FIELD_MAP", (
            f"L={_f(g('length'))}mm  ke={_f(g('ke'))}  kb={_f(g('kb'))}  "
            f"φ={_f(g('phase'))}°"
        )
    if type_name == "FieldMap3D":
        return "FIELD_MAP3D", (
            f"L={_f(g('length'))}mm  ke={_f(g('ke'))}  kb={_f(g('kb'))}  "
            f"φ={_f(g('phase'))}°  f={_f(g('frequency'))}MHz"
        )
    if type_name == "Foil":
        return "FOIL", (
            f"material={g('material')}  "
            f"x={_f(g('thickness_ug_cm2'))}μg/cm²"
        )
    # Fallback — show type name and any length attribute.
    return type_name.upper(), f"L={_f(g('length'))}mm"


def _row_text(idx: int, elem, s_mm: float) -> str:
    keyword, args = _card_for(elem)
    name = getattr(elem, "name", "?")
    L = getattr(elem, "length", 0.0) or 0.0
    head = f"#{idx + 1:03d}  s={s_mm:>7.1f}  "
    body = f"{keyword:<14} {args}"
    tail = f"   ; {name}"
    return head + body + tail


# ---------------------------------------------------------------------------
class LatticeListing(QFrame):
    """Click-to-select sequential card listing of the lattice."""

    def __init__(self, state: AppState):
        super().__init__()
        self._state = state
        self.setObjectName("lattice_listing")
        self.setStyleSheet(
            f"#lattice_listing {{ background:{theme.BG_INSET}; "
            f"border:1px solid {theme.BORDER_0}; border-radius:4px; }}"
        )

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8); v.setSpacing(4)

        header = QLabel("LATTICE  (sequential)")
        header.setStyleSheet(
            f"color:{theme.TEXT_3}; font-size:10px; letter-spacing:1px;"
            f"font-weight:700;"
        )
        v.addWidget(header)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setUniformItemSizes(True)
        font = QFont(theme.FONT_MONO, 10)
        self._list.setFont(font)
        self._list.setStyleSheet(
            f"QListWidget {{ background:{theme.BG_INSET}; "
            f"color:{theme.TEXT_1}; border:none; }}"
            f"QListWidget::item:selected {{ background:{theme.ACCENT_DIM}; "
            f"color:{theme.TEXT_0}; }}"
            f"QListWidget::item:hover {{ background:{theme.BG_4}; }}"
        )
        v.addWidget(self._list, stretch=1)

        # Two-way mapping between list rows and element ids so we can
        # round-trip select-by-id and rebuild quickly on undo/redo.
        self._row_by_id: dict[int, int] = {}
        self._id_by_row: dict[int, int] = {}

        self._list.itemClicked.connect(self._on_row_clicked)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        state.lattice_changed.connect(self._rebuild)
        state.selected_element_changed.connect(self._highlight)
        # Param edits / inserts / deletes — refresh visible text in place.
        state.bus.changed.connect(self._refresh_visible_rows)

        if state.lattice is not None:
            self._rebuild(state.lattice)

    # ------------------------------------------------------------------
    def _rebuild(self, lattice) -> None:
        self._list.clear()
        self._row_by_id.clear()
        self._id_by_row.clear()
        if lattice is None:
            return
        s_mm = 0.0
        for idx, elem in enumerate(lattice.elements):
            text = _row_text(idx, elem, s_mm)
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, id(elem))
            it.setForeground(QBrush(QColor(self._color_for(elem))))
            self._list.addItem(it)
            self._row_by_id[id(elem)] = idx
            self._id_by_row[idx] = id(elem)
            L = getattr(elem, "length", 0.0) or 0.0
            s_mm += L
        sel = self._state.selected
        if sel is not None:
            self._highlight(sel)

    def _refresh_visible_rows(self) -> None:
        """Re-render rows to pick up post-edit attribute values, but
        avoid a full clear — the list might be wholly intact (param
        edit) or just have one row added/removed (insert/delete)."""
        lattice = self._state.lattice
        if lattice is None:
            self._list.clear()
            self._row_by_id.clear(); self._id_by_row.clear()
            return
        # If the row count drifted, bail to a full rebuild.
        if self._list.count() != len(lattice.elements):
            self._rebuild(lattice); return
        # Same row count — refresh text in place.
        s_mm = 0.0
        for idx, elem in enumerate(lattice.elements):
            it = self._list.item(idx)
            if it is None:
                self._rebuild(lattice); return
            # Element identity might have shifted (Move) — reset id mapping.
            it.setData(Qt.ItemDataRole.UserRole, id(elem))
            it.setText(_row_text(idx, elem, s_mm))
            it.setForeground(QBrush(QColor(self._color_for(elem))))
            self._row_by_id[id(elem)] = idx
            self._id_by_row[idx] = id(elem)
            L = getattr(elem, "length", 0.0) or 0.0
            s_mm += L

    def _color_for(self, elem) -> str:
        return theme.EL_COLORS.get(type(elem).__name__, theme.TEXT_1)

    # ------------------------------------------------------------------
    def _on_row_clicked(self, item: QListWidgetItem) -> None:
        eid = item.data(Qt.ItemDataRole.UserRole)
        lattice = self._state.lattice
        if eid is None or lattice is None:
            return
        for el in lattice.elements:
            if id(el) == eid:
                self._state.set_selected(el)
                return

    def _highlight(self, element) -> None:
        if element is None:
            self._list.clearSelection()
            return
        row = self._row_by_id.get(id(element))
        if row is None:
            return
        # Block recursive itemClicked firing while we set the selection.
        self._list.blockSignals(True)
        try:
            self._list.setCurrentRow(row)
            self._list.scrollToItem(
                self._list.item(row),
                QListWidget.ScrollHint.PositionAtCenter,
            )
        finally:
            self._list.blockSignals(False)

    # ------------------------------------------------------------------
    def _on_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        eid = item.data(Qt.ItemDataRole.UserRole)
        lattice = self._state.lattice
        if eid is None or lattice is None:
            return
        elem = next((el for el in lattice.elements if id(el) == eid), None)
        if elem is None:
            return
        type_name = type(elem).__name__
        menu = QMenu(self._list)
        act = QAction(f"Open manual for {type_name}…", menu)
        act.setShortcut("F1")
        act.triggered.connect(lambda _checked=False, e=elem: self._open_manual(e))
        menu.addAction(act)
        menu.exec(self._list.viewport().mapToGlobal(pos))

    def _open_manual(self, element) -> None:
        ok, msg = open_for_element(element)
        try:
            self._state.status_message.emit(msg)
        except Exception:
            pass
