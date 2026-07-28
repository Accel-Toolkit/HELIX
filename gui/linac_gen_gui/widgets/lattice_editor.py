"""Lattice editor widget: tree view of elements with properties panel."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QFormLayout, QLabel,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal

ELEMENT_COLORS = {
    "Drift": "#888888",
    "Quadrupole": "#4488ff",
    "Solenoid": "#ff8844",
    "RFGap": "#ff4444",
    "FieldMap": "#ff4444",
    "Dipole": "#44ff44",
    "Steerer": "#ffff44",
    "Aperture": "#ffffff",
    "Marker": "#aaaaaa",
    "Multipole": "#88ffff",
    "ThinLens": "#ff88ff",
    "SpaceChargeComp": "#cccccc",
}


class LatticeEditorWidget(QWidget):
    element_selected = pyqtSignal(object)  # emits the selected element

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Element tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Type", "Length (mm)"])
        self._tree.setColumnWidth(0, 120)
        self._tree.setColumnWidth(1, 100)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)

        # Properties panel
        self._props_group = QGroupBox("Element Properties")
        self._props_layout = QFormLayout(self._props_group)
        layout.addWidget(self._props_group)

        self._lattice = None

    def set_lattice(self, lattice):
        self._lattice = lattice
        self._tree.clear()
        for elem in lattice.elements:
            type_name = type(elem).__name__
            item = QTreeWidgetItem([elem.name, type_name, f"{elem.length:.1f}"])
            color = ELEMENT_COLORS.get(type_name, "#ffffff")
            item.setForeground(1, QColor(color))
            self._tree.addTopLevelItem(item)

    def _on_selection_changed(self, current, previous):
        if not current or not self._lattice:
            return
        idx = self._tree.indexOfTopLevelItem(current)
        if 0 <= idx < len(self._lattice.elements):
            elem = self._lattice.elements[idx]
            self._show_properties(elem)
            self.element_selected.emit(elem)

    def _show_properties(self, elem):
        # Clear existing properties
        while self._props_layout.count():
            item = self._props_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Show key properties
        for attr in ["name", "length", "aperture", "n_steps"]:
            if hasattr(elem, attr):
                val = getattr(elem, attr)
                self._props_layout.addRow(f"{attr}:", QLabel(str(val)))

        # Element-specific properties
        specific_attrs = {
            "gradient": "T/m",
            "field": "T",
            "voltage": "MV",
            "phase": "deg",
            "frequency": "MHz",
            "angle": "deg",
            "rho": "mm",
            "bx_l": "T.m",
            "by_l": "T.m",
            "factor": "",
            "fx": "m",
            "fy": "m",
        }
        for attr, unit in specific_attrs.items():
            if hasattr(elem, attr):
                val = getattr(elem, attr)
                label = f"{attr} ({unit}):" if unit else f"{attr}:"
                self._props_layout.addRow(label, QLabel(f"{val}"))
