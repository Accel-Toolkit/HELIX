"""Thin lattice layout strip showing colored element rectangles."""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import Qt, QRectF

ELEMENT_COLORS = {
    "Drift": QColor(136, 136, 136),
    "Quadrupole": QColor(68, 136, 255),
    "Solenoid": QColor(255, 136, 68),
    "RFGap": QColor(255, 68, 68),
    "FieldMap": QColor(255, 68, 68),
    "Dipole": QColor(68, 255, 68),
    "Steerer": QColor(255, 255, 68),
    "Aperture": QColor(200, 200, 200),
    "Marker": QColor(170, 170, 170),
}


class LatticeLayoutWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(40)
        self.setMaximumHeight(60)
        self._lattice = None

    def set_lattice(self, lattice):
        self._lattice = lattice
        self.update()

    def paintEvent(self, event):
        if not self._lattice or not self._lattice.elements:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        total_len = self._lattice.total_length
        if total_len <= 0:
            return
        s = 0.0
        for elem in self._lattice.elements:
            if elem.length <= 0:
                continue
            x0 = s / total_len * w
            x1 = (s + elem.length) / total_len * w
            color = ELEMENT_COLORS.get(type(elem).__name__, QColor(128, 128, 128))
            painter.fillRect(QRectF(x0, 2, x1 - x0, h - 4), color)
            s += elem.length
        painter.end()
