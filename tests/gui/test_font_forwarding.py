"""Guard: the global font-size control reaches every tab PAGE directly.

The forwarding loop must iterate the stored page references — going
through self._tabs.widget(i) breaks silently once pages are wrapped in
QScrollAreas (widget(i) returns the wrapper, which has no
apply_font_size, and e.g. the element inspector stops scaling).
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def test_apply_font_size_reaches_wrapped_pages(qapp, monkeypatch):
    from linac_gen_gui.interphase.app import InterphaseWindow

    win = InterphaseWindow()
    try:
        seen = []
        monkeypatch.setattr(win.lattice_tab, "apply_font_size",
                            lambda pt: seen.append(("lattice", pt)),
                            raising=True)
        # A page that doesn't implement apply_font_size must simply be
        # skipped — give one a recorder too to prove the loop visits it.
        monkeypatch.setattr(win.matching_tab, "apply_font_size",
                            lambda pt: seen.append(("matching", pt)),
                            raising=False)

        win._apply_font_size(14, persist=False)

        assert ("lattice", 14) in seen
        assert ("matching", 14) in seen
    finally:
        win._apply_font_size(12, persist=False)
        win.close()
        win.deleteLater()
