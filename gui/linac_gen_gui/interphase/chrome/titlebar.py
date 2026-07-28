"""Top-most 28 px strip — app brand, current lattice path, live clock."""
from __future__ import annotations

import os
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.state import AppState


class TitleBar(QFrame):
    def __init__(self, state: AppState):
        super().__init__()
        self.setObjectName("titlebar")
        self.setFixedHeight(28)
        self._state = state

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(8)

        # Brand
        a = QLabel("HELIX")
        a.setObjectName("appname")
        a.setStyleSheet(f"color:{theme.ACCENT}; font-weight:600; letter-spacing:2px;")
        lay.addWidget(a)

        sep = QFrame()
        sep.setFixedSize(1, 14)
        sep.setStyleSheet(f"background:{theme.BORDER_1};")
        lay.addWidget(sep)

        # Lattice path / dirty flag
        self._path_label = QLabel()
        lay.addWidget(self._path_label)
        lay.addStretch(1)

        # Meta pills (placeholder — PyQt / numpy / platform, keeps visual density)
        import sys, platform
        self._meta_label = QLabel(
            f"HELIX · Python {sys.version_info.major}.{sys.version_info.minor} · "
            f"{platform.system()}"
        )
        lay.addWidget(self._meta_label)

        self._clock = QLabel()
        lay.addWidget(self._clock)

        # Initial styling sized to the default base.
        self.apply_font_size(theme.FONT_SIZE)

        self._refresh_path(state.lattice)
        self._tick()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        state.lattice_changed.connect(self._refresh_path)

    def apply_font_size(self, base: int) -> None:
        """Re-render inline-styled labels with `base-2`-sized mono text so
        the titlebar scales together with the rest of the UI."""
        sz = max(8, int(base) - 2)
        self.setFixedHeight(max(28, int(base) + 16))
        css = f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:{sz}px;"
        for w in (self._path_label, self._meta_label, self._clock):
            w.setStyleSheet(css)

    def _tick(self) -> None:
        self._clock.setText(datetime.now().strftime("%H:%M:%S"))

    def _refresh_path(self, lattice) -> None:
        if lattice is None or self._state.lattice_path is None:
            self._path_label.setText("(no lattice loaded)")
        else:
            basename = os.path.basename(self._state.lattice_path)
            n = len(lattice.elements) if hasattr(lattice, "elements") else 0
            self._path_label.setText(f"{basename}  ·  {n} elements")
