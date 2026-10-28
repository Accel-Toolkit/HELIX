"""The Lattice tab's ring banner: a lattice that closes on itself is offered
ring mode — it is never switched on by itself.

The banner shows only when the lattice looks like a ring (its horizontal
bends add up to 360 degrees, or the deck declares ``;@LG ring``), ring mode is
off, and the user has not chosen to keep this lattice as a line.  A cheap
check of the bend angles comes first, so a linac lattice never loads ring
mode; only a lattice that passes it gets the full verdict (closure of the
survey).  "Keep as line" is remembered for the lattice's file — a reload, a
matching Apply and the saved project keep it — and changes nothing else: a
project gains a "ring" section holding only that choice.
"""
from __future__ import annotations

import math
import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from . import theme


def looks_like_a_ring(lattice) -> bool:
    """Bends adding up to +-360 degrees (within 1e-3 deg), or a ``;@LG ring``
    declaration — without loading ring mode."""
    if lattice is None:
        return False
    lg = getattr(lattice, "lg_options", None) or {}
    if str(lg.get("ring", "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    total = 0.0
    any_bend = False
    for e in lattice.elements:
        if type(e).__name__ == "Dipole" and int(getattr(e, "hv", 0) or 0) == 0:
            a = float(getattr(e, "angle", 0.0) or 0.0)
            if a:
                total += a
                any_bend = True
        elif getattr(e, "is_curved", False):
            # curved field-map cluster: its horizontal bend (survey sign)
            from linac_gen.elements.curved_field_map import survey_geometry
            A = survey_geometry(e)[2]
            a = math.degrees(math.atan2(A[0, 2], A[2, 2]))
            if a:
                total += a
                any_bend = True
    return any_bend and abs(abs(total) - 360.0) < 1e-3 and math.isfinite(total)


def line_choice_key(lattice, path):
    """What "Keep as line" remembers: the lattice's file, or the lattice
    itself when it has none."""
    if path:
        return os.path.realpath(str(path))
    return lattice


def kept_as_line(choice, lattice, path) -> bool:
    """Whether the remembered choice (:func:`line_choice_key`) is this
    lattice."""
    if choice is None or lattice is None:
        return False
    if isinstance(choice, str):
        return bool(path) and os.path.realpath(str(path)) == choice
    return choice is lattice


class RingBanner(QFrame):
    use_ring_requested = pyqtSignal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._lattice = None
        self.setObjectName("ringBanner")
        self.setStyleSheet(f"#ringBanner {{ background:{theme.BG_2}; border:1px solid "
                           f"{theme.ACCENT_DIM}; border-radius:4px; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 6, 10, 6)
        row = QHBoxLayout()
        self._text = QLabel("")
        self._text.setWordWrap(True)
        self._text.setStyleSheet(f"color:{theme.TEXT_0};")
        row.addWidget(self._text, stretch=1)
        self._use = QPushButton("Use ring mode")
        self._use.clicked.connect(self._on_use)
        self._keep = QPushButton("Keep as line")
        self._keep.clicked.connect(self._on_keep)
        self._why = QPushButton("Why?")
        self._why.setCheckable(True)
        self._why.toggled.connect(self._on_why)
        for b in (self._use, self._keep, self._why):
            row.addWidget(b)
        v.addLayout(row)
        self._details = QLabel("")
        self._details.setWordWrap(True)
        self._details.setStyleSheet(f"color:{theme.TEXT_2};")
        self._details.hide()
        v.addWidget(self._details)
        self.hide()
        state.ring_mode_changed.connect(self.refresh)

    def on_lattice(self, lattice) -> None:
        self._lattice = lattice
        self.refresh()

    def refresh(self, *_args) -> None:
        lat = self._lattice
        show = (lat is not None and not self.state.ring_mode
                and not kept_as_line(self.state.ring_line_choice, lat, self.state.lattice_path)
                and looks_like_a_ring(lat))
        if show:
            try:
                from linac_gen.ring.detect import detect_ring
                v = detect_ring(lat)
            except Exception:             # noqa: BLE001 — a banner must never break the tab
                v = None
            show = v is not None and v.is_ring
        if not show:
            self.hide()
            return
        h = f", RF harmonic {v.harmonic}" if v.harmonic else ""
        if v.declared:
            head = f"This lattice is declared a ring (;@LG ring{h})"
        else:
            head = (f"This lattice closes on itself (bends {v.total_angle_deg:.6f} deg, survey "
                    f"closes within {v.closure_mm:.3g} mm{h})")
        self._text.setText(head + ": ring mode tracks it turn after turn.")
        self._details.setText(
            f"Why: {v.reason}.  Length {v.length_m:.6f} m.  Ring mode is never switched on "
            "by itself; it is Simulate → Ring Mode, or this button.  Keeping the lattice as a "
            "line hides this note for it (saved with the project).")
        self.show()

    def _on_use(self) -> None:
        self.use_ring_requested.emit()
        self.state.set_ring_mode(True)

    def _on_keep(self) -> None:
        # (ring mode stays unused: the project keeps only this choice)
        self.state.ring_line_choice = line_choice_key(self._lattice, self.state.lattice_path)
        self.state.mark_project_dirty()
        self.hide()

    def _on_why(self, on: bool) -> None:
        self._details.setVisible(bool(on))
