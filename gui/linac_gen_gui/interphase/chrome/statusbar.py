"""Bottom 22-px status bar — live run state and reference-particle stats."""
from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.state import AppState


class StatusBar(QFrame):
    def __init__(self, state: AppState):
        super().__init__()
        self.setObjectName("statusbar")
        self.setFixedHeight(22)
        self._state = state

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 10, 0)
        lay.setSpacing(0)

        self._state_seg = QLabel("READY")
        self._state_seg.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; padding:0 10px; font-weight:600;"
            f"font-family:{theme.FONT_MONO};"
        )
        lay.addWidget(self._state_seg)

        self._lattice_seg = self._seg("no lattice")
        self._s_seg       = self._seg("s = 0 / — mm")
        self._energy_seg  = self._seg("W = — MeV")
        self._sigma_seg   = self._seg("σ_x = — mm")
        self._loss_seg    = self._seg("loss 0.0 %")
        for w in (self._lattice_seg, self._s_seg, self._energy_seg,
                  self._sigma_seg, self._loss_seg):
            lay.addWidget(w)

        # Unsaved-changes pill — hidden by default; shown when bus is dirty.
        self._unsaved_seg = QLabel("● unsaved")
        self._unsaved_seg.setStyleSheet(
            f"color:{theme.WARN}; padding:0 10px; "
            f"border-right:1px solid {theme.BORDER_0}; font-family:{theme.FONT_MONO}; "
            f"font-weight:600;"
        )
        self._unsaved_seg.setVisible(False)
        lay.addWidget(self._unsaved_seg)

        # Transient status message (Saved →, Exported →, element edits, font
        # size, …).  Before this segment every status_message.emit had no
        # visible sink — the only listener repainted the bar and dropped the
        # text — so the user got no confirmation for saves / exports / edits.
        self._msg_seg = QLabel("")
        self._msg_seg.setStyleSheet(
            f"color:{theme.TEXT_2}; padding:0 10px; font-family:{theme.FONT_MONO};"
        )
        # A long message must not widen the WINDOW's minimum width (QLabel's
        # minimumSizeHint tracks its text).  Ignored → the label's width is
        # layout-driven, clipping long text; the stretch factor lets it share
        # the row's free space with the trailing addStretch.
        self._msg_seg.setSizePolicy(QSizePolicy.Policy.Ignored,
                                    QSizePolicy.Policy.Preferred)
        lay.addWidget(self._msg_seg, 1)
        self._msg_timer = QTimer(self)
        self._msg_timer.setSingleShot(True)
        # Bound method, NOT a lambda: a lambda has no receiver QObject, so Qt
        # cannot auto-disconnect it when this widget's C++ side is destroyed —
        # a pending timeout then segfaults on the dead QLabel (reproduced in
        # the offscreen test suite).  A bound method carries the receiver
        # context and is dropped with the widget.
        self._msg_timer.timeout.connect(self._clear_message)

        lay.addStretch(1)
        version = QLabel("HELIX 0.1")
        version.setStyleSheet(
            f"color:{theme.TEXT_2}; padding:0 10px; font-family:{theme.FONT_MONO};"
        )
        lay.addWidget(version)

        state.running_changed.connect(self._refresh_state)
        state.lattice_changed.connect(self._refresh_lattice)
        state.s_cursor_changed.connect(self._refresh_s)
        state.results_changed.connect(self._refresh_results)
        # Bus may not exist on very old AppState builds — guard the connect.
        bus = getattr(state, "bus", None)
        if bus is not None:
            bus.dirty_changed.connect(self._refresh_dirty)
        state.status_message.connect(self._on_message)

        # Track the version label and current loss colour so apply_font_size
        # can re-render with the new size.
        self._version_lbl = version
        self._loss_color = theme.TEXT_1
        self.apply_font_size(theme.FONT_SIZE)

    def _seg(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{theme.TEXT_1}; padding:0 10px; "
            f"border-right:1px solid {theme.BORDER_0}; font-family:{theme.FONT_MONO};"
        )
        return lbl

    # ------------------------------------------------------------------
    def _refresh_state(self, running: bool) -> None:
        self._state_seg.setText("RUNNING" if running else ("DONE" if self._state.results else "READY"))

    def _refresh_lattice(self, lattice) -> None:
        if lattice is None:
            self._lattice_seg.setText("no lattice")
        else:
            n = len(lattice.elements)
            total = sum(e.length for e in lattice.elements)
            self._lattice_seg.setText(f"{n} elements · {total:.0f} mm")
            self._s_seg.setText(f"s = 0 / {total:.0f} mm")

    def _refresh_s(self, s: float) -> None:
        lat = self._state.lattice
        if lat:
            total = sum(e.length for e in lat.elements)
            self._s_seg.setText(f"s = {s:.0f} / {total:.0f} mm")
        else:
            self._s_seg.setText(f"s = {s:.0f} mm")

    def _refresh_dirty(self, dirty: bool) -> None:
        self._unsaved_seg.setVisible(bool(dirty))

    def _refresh_results(self, results) -> None:
        if results is None:
            self._sigma_seg.setText("σ_x = — mm")
            self._loss_seg.setText("loss 0.0 %")
            return
        # End-of-lattice σ_x
        sigma_x = getattr(results, "sigma_x", None)
        if sigma_x:
            self._sigma_seg.setText(f"σ_x = {sigma_x[-1]:.3f} mm")
        # Transmission → loss
        trans = getattr(results, "transmission", None)
        if trans:
            loss = 100.0 - trans[-1]
            self._loss_color = theme.WARN if loss > 1.0 else theme.TEXT_1
            self._loss_seg.setText(f"loss {loss:.2f} %")
            self.apply_font_size(self.font().pointSize() or theme.FONT_SIZE)

    def _clear_message(self) -> None:
        try:
            self._msg_seg.setText("")
        except RuntimeError:
            pass          # C++ label already destroyed (teardown race)

    def _on_message(self, msg: str) -> None:
        """Show a transient status message; auto-clear after a few seconds."""
        try:
            self._msg_seg.setText(str(msg))
            self._msg_timer.start(6000)
        except RuntimeError:
            pass          # sender (AppState) outlived this bar — ignore

    def apply_font_size(self, base: int) -> None:
        """Restyle every inline-styled segment so the statusbar scales."""
        bs = max(8, int(base) - 2)
        self.setFixedHeight(max(22, int(base) + 10))
        # Active state pill (READY / RUNNING / DONE).
        self._state_seg.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; padding:0 10px;"
            f" font-weight:600; font-family:{theme.FONT_MONO};"
            f" font-size:{bs}px;"
        )
        # Generic segments.
        seg_css = (
            f"color:{theme.TEXT_1}; padding:0 10px;"
            f" border-right:1px solid {theme.BORDER_0};"
            f" font-family:{theme.FONT_MONO}; font-size:{bs}px;"
        )
        for w in (self._lattice_seg, self._s_seg, self._energy_seg,
                  self._sigma_seg):
            w.setStyleSheet(seg_css)
        # Loss segment uses a colour that depends on threshold.
        self._loss_seg.setStyleSheet(
            f"color:{self._loss_color}; padding:0 10px;"
            f" border-right:1px solid {theme.BORDER_0};"
            f" font-family:{theme.FONT_MONO}; font-size:{bs}px;"
        )
        self._unsaved_seg.setStyleSheet(
            f"color:{theme.WARN}; padding:0 10px;"
            f" border-right:1px solid {theme.BORDER_0};"
            f" font-family:{theme.FONT_MONO}; font-weight:600;"
            f" font-size:{bs}px;"
        )
        self._version_lbl.setStyleSheet(
            f"color:{theme.TEXT_2}; padding:0 10px;"
            f" font-family:{theme.FONT_MONO}; font-size:{bs}px;"
        )
        self._msg_seg.setStyleSheet(
            f"color:{theme.TEXT_2}; padding:0 10px;"
            f" font-family:{theme.FONT_MONO}; font-size:{bs}px;"
        )
