"""Lattice outline tree widget (groups elements by name prefix + search).

Drag-reorder is wired via Qt's ``InternalMove`` mode.  Because the tree
groups elements by name-prefix while the lattice itself is a flat list,
naive drop semantics would scramble the ordering — so the tree
overrides ``dropEvent`` to translate the drop into a *flat-list*
``MoveCommand`` and emits ``move_requested(from_idx, to_idx)``.  The
parent tab is responsible for issuing the command on the bus.  After
the bus mutates the lattice, ``state.lattice_changed`` triggers a full
``_rebuild`` — Qt's transient drag-state is never trusted.
"""
from __future__ import annotations

from collections import defaultdict
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QLineEdit, QMenu, QTreeWidget, QTreeWidgetItem,
    QAbstractItemView,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.manual_help import open_for_element
from linac_gen_gui.interphase.state import AppState


class _DraggableTree(QTreeWidget):
    """QTreeWidget that emits ``move_requested(from_idx, to_idx)`` on
    a successful internal move and refuses the drop itself, so the
    parent can route the change through the command bus."""

    move_requested = pyqtSignal(int, int)

    def __init__(self, owner: "OutlineTree"):
        super().__init__()
        self._owner = owner
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def dropEvent(self, ev) -> None:
        # Resolve source element first (must be a leaf with element-id).
        src_item = self.currentItem()
        if src_item is None or src_item.parent() is None:
            ev.ignore(); return
        src_eid = src_item.data(0, Qt.ItemDataRole.UserRole)
        # Find drop target leaf and its drop position relative to that leaf.
        dst_item = self.itemAt(ev.position().toPoint())
        pos = self.dropIndicatorPosition()
        # Rebuild: forbid drops on group headers (no UserRole).
        dst_eid = dst_item.data(0, Qt.ItemDataRole.UserRole) if dst_item else None
        ev.ignore()  # Qt must NOT mutate the tree — we'll rebuild from state.
        if src_eid is None:
            return
        from_idx = self._owner._flat_index_of(src_eid)
        if from_idx < 0:
            return
        if dst_eid is None:
            # Dropped outside any leaf → append to the end.
            to_idx = self._owner._flat_count() - 1
        else:
            to_idx = self._owner._flat_index_of(dst_eid)
            if pos == QAbstractItemView.DropIndicatorPosition.BelowItem:
                to_idx += 1
            elif pos == QAbstractItemView.DropIndicatorPosition.OnItem:
                # Treat as drop AFTER the target.
                to_idx += 1
        # Adjust target for the source removal that MoveCommand performs.
        if to_idx > from_idx:
            to_idx -= 1
        self.move_requested.emit(from_idx, to_idx)


class OutlineTree(QFrame):
    move_requested = pyqtSignal(int, int)

    def __init__(self, state: AppState):
        super().__init__()
        self.setObjectName("sidebar")
        self._state = state
        self._elements_by_id: dict[int, object] = {}
        # Flat (lattice-order) list of element ids, kept in sync with `_rebuild`.
        self._flat_ids: list[int] = []

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        header = QLabel("OUTLINE")
        header.setStyleSheet(
            f"color:{theme.TEXT_3}; font-size:10px; letter-spacing:1px;"
            f"padding:8px 10px 4px; font-weight:600;"
        )
        v.addWidget(header)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter elements…")
        self._search.textChanged.connect(self._apply_filter)
        v.addWidget(self._search)

        self._tree = _DraggableTree(self)
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setIndentation(12)
        self._tree.itemClicked.connect(self._on_click)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        # Bubble the inner tree's reorder signal up to consumers.
        self._tree.move_requested.connect(self.move_requested.emit)
        v.addWidget(self._tree, stretch=1)

        state.lattice_changed.connect(self._rebuild)
        state.selected_element_changed.connect(self._highlight)

    # ------------------------------------------------------------------
    # Lookups for the inner tree's drop-handler.
    def _flat_index_of(self, eid: int) -> int:
        try:
            return self._flat_ids.index(eid)
        except ValueError:
            return -1

    def _flat_count(self) -> int:
        return len(self._flat_ids)

    # ------------------------------------------------------------------
    def _rebuild(self, lattice) -> None:
        self._tree.clear()
        self._elements_by_id.clear()
        self._flat_ids.clear()
        if lattice is None:
            return
        # Capture flat lattice order BEFORE grouping — drop-handler maps
        # tree leaves back to this order via element id.
        for el in lattice.elements:
            self._flat_ids.append(id(el))
            self._elements_by_id[id(el)] = el
        groups: dict[str, list] = defaultdict(list)
        for el in lattice.elements:
            name = getattr(el, "name", "?") or "?"
            if "." in name:
                prefix = name.split(".", 1)[0]
            elif "_" in name:
                prefix = name.split("_", 1)[0]
            else:
                prefix = "OTHER"
            groups[prefix].append(el)
        for sec in sorted(groups):
            sec_item = QTreeWidgetItem([f"{sec}  ({len(groups[sec])})"])
            sec_item.setForeground(0, QBrush(QColor(theme.TEXT_2)))
            # Group headers must NOT be drop targets nor drag sources.
            sec_item.setFlags(
                sec_item.flags()
                & ~Qt.ItemFlag.ItemIsDragEnabled
                & ~Qt.ItemFlag.ItemIsDropEnabled
            )
            self._tree.addTopLevelItem(sec_item)
            for el in groups[sec]:
                name = getattr(el, "name", "?")
                type_name = type(el).__name__
                # ``LatticeCommand`` instances get a ⚙ glyph so they're
                # visually distinct from optics elements at a glance.
                try:
                    from linac_gen.elements.lattice_commands import LatticeCommand
                    is_cmd = isinstance(el, LatticeCommand)
                except Exception:
                    is_cmd = False
                badge = "⚙ " if is_cmd else ""
                item = QTreeWidgetItem([f"{badge}{name}  ·  {type_name}"])
                item.setForeground(0, QBrush(QColor(theme.TEXT_1)))
                item.setData(0, Qt.ItemDataRole.UserRole, id(el))
                # Leaves are draggable but not drop targets — drops always
                # happen on the gap between leaves.
                item.setFlags(
                    (item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
                    & ~Qt.ItemFlag.ItemIsDropEnabled
                )
                sec_item.addChild(item)
            sec_item.setExpanded(len(groups[sec]) <= 40)

    def _on_click(self, item: QTreeWidgetItem, _col: int) -> None:
        eid = item.data(0, Qt.ItemDataRole.UserRole)
        if eid is None: return
        el = self._elements_by_id.get(eid)
        if el is not None:
            self._state.set_selected(el)

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        eid = item.data(0, Qt.ItemDataRole.UserRole)
        el = self._elements_by_id.get(eid) if eid is not None else None
        if el is None:
            return
        type_name = type(el).__name__
        menu = QMenu(self._tree)
        act = QAction(f"Open manual for {type_name}…", menu)
        act.setShortcut("F1")
        act.triggered.connect(lambda _checked=False, e=el: self._open_manual(e))
        menu.addAction(act)
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _open_manual(self, element) -> None:
        ok, msg = open_for_element(element)
        try:
            self._state.status_message.emit(msg)
        except Exception:
            pass

    def _highlight(self, element) -> None:
        target_id = id(element) if element is not None else None
        for i in range(self._tree.topLevelItemCount()):
            sec = self._tree.topLevelItem(i)
            for j in range(sec.childCount()):
                child = sec.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == target_id:
                    self._tree.setCurrentItem(child)
                    return

    def _apply_filter(self, text: str) -> None:
        q = text.lower().strip()
        for i in range(self._tree.topLevelItemCount()):
            sec = self._tree.topLevelItem(i)
            any_visible = False
            for j in range(sec.childCount()):
                child = sec.child(j)
                visible = (q == "") or (q in child.text(0).lower())
                child.setHidden(not visible)
                any_visible = any_visible or visible
            sec.setHidden(not any_visible)
