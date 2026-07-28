"""Horizontal lattice timeline — colored element bars on a QGraphicsScene.

Click to select an element; emits a signal with the element object.
An ``s_cursor`` vertical line tracks the app-state s-cursor.

Accepts drops from the :class:`ElementPalette` — the drop position
maps to an insertion index in the lattice and is published via the
``element_dropped`` signal so the parent tab can issue an
``InsertCommand``.  A 2-px cyan vertical "gap indicator" tracks the
nearest insertion point during drag-over.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPen, QPainter
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem,
    QGraphicsItem, QGraphicsTextItem,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.panels.element_palette import PALETTE_MIME


class LatticeTimeline(QGraphicsView):
    element_clicked = pyqtSignal(object)
    # Emitted on a successful drop; arg is (type_name, insert_index).
    element_dropped = pyqtSignal(str, int)
    # Emitted whenever the multi-selection set changes; arg is list[Element].
    selection_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(80)
        self.setStyleSheet(
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_0};"
            f"border-radius:4px;"
        )
        self._bars: dict[int, QGraphicsRectItem] = {}
        self._elements: dict[int, object] = {}
        # Selection-glow halos rendered under each selected bar.
        self._halos: dict[int, QGraphicsRectItem] = {}
        # Validation badges (small triangles) overlaid on flagged bars.
        self._badges: dict[int, QGraphicsRectItem] = {}
        # Type-filter set; if non-empty, bars whose type is NOT in here are dimmed.
        self._type_filter: set[str] = set()
        # Preserve insertion order so we can map drop x-coords → element index.
        self._element_order: list[int] = []
        self._cursor_line: QGraphicsLineItem | None = None
        self._selected_id: int | None = None
        # Multi-select set; always includes _selected_id when non-None.
        self._selected_ids: list[int] = []
        self._total_length = 0.0
        # Drag-and-drop state
        self.setAcceptDrops(True)
        self._gap_indicator: QGraphicsLineItem | None = None
        # Wheel-zoom (Ctrl + wheel; clamped 0.25× – 10×).
        self._zoom_x = 1.0

    # ------------------------------------------------------------------
    def set_lattice(self, lattice) -> None:
        self._scene.clear()
        self._bars.clear()
        self._elements.clear()
        self._halos.clear()
        self._badges.clear()
        self._element_order.clear()
        self._cursor_line = None
        self._gap_indicator = None
        self._selected_id = None
        self._selected_ids.clear()

        if lattice is None or not lattice.elements:
            self._total_length = 0.0
            return

        self._total_length = sum(e.length for e in lattice.elements)
        if self._total_length <= 0:
            # All elements are zero-length (diagnostics-only section, or only
            # markers/steerers/BPMs/SET_* cards).  There is nothing to lay out
            # positionally; skip drawing rather than divide by zero below.  The
            # scene was already cleared above, so the strip simply shows empty.
            return
        w = max(self.viewport().width() - 20, 800)
        h = 56
        self._scene.setSceneRect(0, 0, w, h)
        self.setSceneRect(0, 0, w, h)

        s = 0.0
        for el in lattice.elements:
            length = getattr(el, "length", 0.0)
            type_name = type(el).__name__
            is_drift = type_name == "Drift"
            x = (s / self._total_length) * w
            bw = max((length / self._total_length) * w, 1.0)
            if type_name == "Marker":
                bw = max(bw, 1.5)

            color = QColor(theme.EL_COLORS.get(type_name, theme.TEXT_2))
            top    = 24 if is_drift else 12
            bottom = 32 if is_drift else 44
            bar = QGraphicsRectItem(x, top, bw, bottom - top)
            bar.setBrush(QBrush(color))
            bar.setPen(QPen(Qt.PenStyle.NoPen))
            if is_drift:
                bar.setOpacity(0.55)
            bar.setData(0, id(el))
            bar.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
            bar.setToolTip(self._tooltip_for(el, s, length))
            self._scene.addItem(bar)
            self._bars[id(el)] = bar
            self._elements[id(el)] = el
            self._element_order.append(id(el))
            s += length

        # Baseline
        baseline = QGraphicsLineItem(0, 28, w, 28)
        baseline.setPen(QPen(QColor(theme.BORDER_1), 0.5))
        self._scene.addItem(baseline)

    def set_selected(self, element) -> None:
        # Clear all multi-selection pens + halos first.
        for eid in list(self._selected_ids):
            self._clear_halo(eid)
            b = self._bars.get(eid)
            if b is not None:
                b.setPen(QPen(Qt.PenStyle.NoPen))
                b.setZValue(0)
        self._selected_ids = []
        self._selected_id = id(element) if element is not None else None
        if self._selected_id is not None:
            self._selected_ids = [self._selected_id]
        nb = self._bars.get(self._selected_id) if self._selected_id else None
        if nb is not None:
            nb.setPen(QPen(QColor("#ffffff"), 1.5))
            nb.setZValue(3)
            self._add_halo(self._selected_id, color=theme.ACCENT)
        self.selection_changed.emit(self.selected_elements())

    def _add_halo(self, eid: int, color: str = None) -> None:
        """Render a soft cyan halo just outside the selected bar."""
        bar = self._bars.get(eid)
        if bar is None:
            return
        r = bar.rect()
        pad = 3.0
        halo = QGraphicsRectItem(r.x() - pad, r.y() - pad,
                                 r.width() + 2 * pad, r.height() + 2 * pad)
        halo.setBrush(QBrush(QColor(color or theme.ACCENT)))
        halo.setOpacity(0.22)
        halo.setPen(QPen(Qt.PenStyle.NoPen))
        halo.setZValue(1.5)
        self._scene.addItem(halo)
        self._halos[eid] = halo

    def _clear_halo(self, eid: int) -> None:
        h = self._halos.pop(eid, None)
        if h is not None:
            self._scene.removeItem(h)

    def selected_elements(self) -> list:
        """Return the multi-select set as a list of element instances
        in lattice order."""
        out = []
        for eid in self._element_order:
            if eid in self._selected_ids:
                el = self._elements.get(eid)
                if el is not None:
                    out.append(el)
        return out

    def set_s_cursor(self, s: float) -> None:
        if self._total_length <= 0:
            return
        w = self._scene.width()
        x = (s / self._total_length) * w
        if self._cursor_line is None:
            self._cursor_line = QGraphicsLineItem(x, 0, x, 56)
            self._cursor_line.setPen(QPen(QColor(theme.ACCENT), 1.2))
            self._cursor_line.setZValue(10)
            self._scene.addItem(self._cursor_line)
        else:
            self._cursor_line.setLine(x, 0, x, 56)

    # ------------------------------------------------------------------
    def mousePressEvent(self, ev):
        pos = self.mapToScene(ev.pos())
        item = self.scene().itemAt(pos, self.transform())
        while item is not None and not isinstance(item, QGraphicsRectItem):
            item = item.parentItem()
        if isinstance(item, QGraphicsRectItem):
            eid = item.data(0)
            el = self._elements.get(eid)
            if el is not None:
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._toggle_in_selection(eid)
                else:
                    # Single-select: AppState owns the canonical "selected"
                    # field — let it broadcast back via set_selected().
                    self.element_clicked.emit(el)
        super().mousePressEvent(ev)

    def _toggle_in_selection(self, eid: int) -> None:
        """Shift+click: add to or remove from the multi-selection set."""
        bar = self._bars.get(eid)
        if bar is None:
            return
        if eid in self._selected_ids:
            self._selected_ids.remove(eid)
            # If the toggled bar was the "primary", drop primary too.
            if self._selected_id == eid:
                self._selected_id = self._selected_ids[-1] if self._selected_ids else None
            self._clear_halo(eid)
            bar.setPen(QPen(Qt.PenStyle.NoPen))
            bar.setZValue(0)
        else:
            self._selected_ids.append(eid)
            bar.setPen(QPen(QColor(theme.ACCENT_2), 1.5))
            bar.setZValue(3)
            self._add_halo(eid, color=theme.ACCENT_2)
        self.selection_changed.emit(self.selected_elements())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Re-layout at new width.  set_lattice resets the scene so we
        # need to re-apply the active zoom + type filter manually.
        if self._total_length > 0 and self._elements:
            current = [self._elements[i] for i in self._element_order]
            tfilter = set(self._type_filter)
            zoom = self._zoom_x

            class _DummyLattice:
                def __init__(self, els):
                    self.elements = els
            self.set_lattice(_DummyLattice(current))

            # set_lattice() rebuilds the scene but does NOT reset the view
            # transform, so re-applying self.scale(zoom) compounded the zoom on
            # every resize (2×→4×→8×…) and reset_zoom could no longer recover
            # 1×.  Reset first so the transform ends up as exactly `zoom`.
            self.resetTransform()
            if zoom != 1.0:
                self.scale(zoom, 1.0)
                self._zoom_x = zoom
            if tfilter:
                self.set_type_filter(tfilter)

    # ------------------------------------------------------------------
    # Drag-and-drop  (accept palette cards → insert at gap index)
    # ------------------------------------------------------------------
    def _has_palette_payload(self, ev) -> bool:
        return ev.mimeData().hasFormat(PALETTE_MIME)

    def _gap_index_at(self, view_x: float) -> int:
        """Return the lattice index that a drop at ``view_x`` should
        insert AT.  ``0`` for "before everything", ``len`` for "after
        everything", otherwise the index of the bar that the drop is
        on the LEFT half of."""
        if not self._element_order:
            return 0
        scene_x = self.mapToScene(int(view_x), 0).x()
        for i, eid in enumerate(self._element_order):
            bar = self._bars.get(eid)
            if bar is None:
                continue
            r = bar.rect()
            mid = r.x() + r.width() / 2.0
            if scene_x < mid:
                return i
        return len(self._element_order)

    def _gap_x(self, idx: int) -> float:
        """Return the scene-x where the gap indicator should be drawn
        for an insertion at ``idx``."""
        if not self._element_order:
            return 0.0
        if idx <= 0:
            bar = self._bars.get(self._element_order[0])
            return bar.rect().x() if bar else 0.0
        if idx >= len(self._element_order):
            bar = self._bars.get(self._element_order[-1])
            return bar.rect().x() + bar.rect().width() if bar else 0.0
        bar = self._bars.get(self._element_order[idx])
        return bar.rect().x() if bar else 0.0

    def _show_gap(self, idx: int) -> None:
        x = self._gap_x(idx)
        if self._gap_indicator is None:
            line = QGraphicsLineItem(x, 4, x, 52)
            line.setPen(QPen(QColor(theme.ACCENT), 2))
            line.setZValue(20)
            self._scene.addItem(line)
            self._gap_indicator = line
        else:
            self._gap_indicator.setLine(x, 4, x, 52)
            self._gap_indicator.setVisible(True)

    def _hide_gap(self) -> None:
        if self._gap_indicator is not None:
            self._gap_indicator.setVisible(False)

    def dragEnterEvent(self, ev) -> None:
        if self._has_palette_payload(ev):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev) -> None:
        if self._has_palette_payload(ev):
            idx = self._gap_index_at(ev.position().x())
            self._show_gap(idx)
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dragLeaveEvent(self, ev) -> None:
        self._hide_gap()
        super().dragLeaveEvent(ev)

    def dropEvent(self, ev) -> None:
        if not self._has_palette_payload(ev):
            super().dropEvent(ev); return
        type_name = bytes(ev.mimeData().data(PALETTE_MIME)).decode("utf-8")
        idx = self._gap_index_at(ev.position().x())
        self._hide_gap()
        ev.acceptProposedAction()
        self.element_dropped.emit(type_name, idx)

    # ------------------------------------------------------------------
    # Wheel zoom (Ctrl + wheel scales x-only).  Zoom is applied via
    # ``self.scale()`` so the scene coordinates stay stable; only the
    # view transform changes.
    # ------------------------------------------------------------------
    def wheelEvent(self, ev) -> None:
        if not (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            super().wheelEvent(ev); return
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_zoom = max(0.25, min(10.0, self._zoom_x * factor))
        rel = new_zoom / self._zoom_x
        self.scale(rel, 1.0)
        self._zoom_x = new_zoom
        ev.accept()

    def reset_zoom(self) -> None:
        if self._zoom_x != 1.0:
            self.scale(1.0 / self._zoom_x, 1.0)
            self._zoom_x = 1.0

    # ------------------------------------------------------------------
    # Type filter (called by the LED-chip strip).  Empty = show all.
    # Bars whose type is NOT in the filter set are dimmed.
    # ------------------------------------------------------------------
    def set_type_filter(self, types: set[str]) -> None:
        self._type_filter = set(types or [])
        for eid in self._element_order:
            el = self._elements.get(eid)
            bar = self._bars.get(eid)
            if el is None or bar is None:
                continue
            if not self._type_filter or type(el).__name__ in self._type_filter:
                # Drifts keep their original 0.55 opacity, others go to 1.0.
                is_drift = type(el).__name__ == "Drift"
                bar.setOpacity(0.55 if is_drift else 1.0)
            else:
                bar.setOpacity(0.15)

    # ------------------------------------------------------------------
    # Validation badges — small triangle markers on warned bars.
    # ``warnings`` maps id(element) → list of warning strings.
    # ------------------------------------------------------------------
    def set_validation(self, warnings: dict[int, list[str]]) -> None:
        # Clear previous badges first.
        for badge in self._badges.values():
            self._scene.removeItem(badge)
        self._badges.clear()
        if not warnings:
            return
        for eid, msgs in warnings.items():
            bar = self._bars.get(eid)
            if bar is None or not msgs:
                continue
            r = bar.rect()
            sz = 6
            badge = QGraphicsRectItem(
                r.x() + r.width() - sz - 1, r.y() + 1, sz, sz
            )
            badge.setBrush(QBrush(QColor(theme.WARN)))
            badge.setPen(QPen(Qt.PenStyle.NoPen))
            badge.setZValue(5)
            badge.setToolTip("; ".join(msgs))
            self._scene.addItem(badge)
            self._badges[eid] = badge

    # ------------------------------------------------------------------
    # Rich per-bar tooltip — extends the basic name/type/length with
    # the element's own type-specific key params (read straight from
    # attributes; no schema dependency).
    # ------------------------------------------------------------------
    def _tooltip_for(self, el, s: float, length: float) -> str:
        type_name = type(el).__name__
        # Per-type "headline" attributes (kept short for hover use).
        head_attrs: dict[str, list[str]] = {
            "Quadrupole": ["gradient"],
            "Solenoid":   ["Bz", "field"],
            "RFGap":      ["amplitude", "phase", "frequency"],
            "Dipole":     ["angle", "rho"],
            "FieldMap":   ["ke", "kb", "phase"],
            "FieldMap3D": ["ke", "kb", "phase"],
            "RfqCell":    ["voltage_V", "modulation", "phi_s_deg"],
            "Aperture":   ["dx", "dy"],
            "Steerer":    ["bx_l", "by_l"],
            "ThinLens":   ["fx", "fy"],
            "SpaceChargeComp": ["factor"],
        }
        lines = [
            f"<b>{getattr(el, 'name', '?')}</b>  "
            f"<span style='color:{theme.TEXT_2}'>· {type_name}</span>",
            f"L = {length:g} mm   "
            f"s ∈ [{s:.1f}, {s + length:.1f}] mm",
        ]
        for a in head_attrs.get(type_name, []):
            v = getattr(el, a, None)
            if v is None:
                continue
            try:
                lines.append(f"{a} = {float(v):.4g}")
            except (TypeError, ValueError):
                lines.append(f"{a} = {v}")
        return "<br>".join(lines)
