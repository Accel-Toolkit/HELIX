"""Guard: chrome bar heights are owned by Python, never by QSS.

theme.py used to pin #titlebar/#menubar/#statusbar/#stageToolbar with
QSS min/max-height while the widgets grew themselves with font-derived
setFixedHeight.  Qt re-applies QSS geometry on every repolish, so the
stylesheet silently overwrote the Python values and clipped the text —
the #menubar pin (38px) contradicted the toolbar's own
setFixedHeight(max(44, base+30)) even at the default font, and the
lattice tab's 36px toolbar strip was clamped to 32px on every launch.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("PyQt6")

from linac_gen_gui.interphase import theme  # noqa: E402

_PYTHON_OWNED = ("#titlebar", "#menubar", "#statusbar", "#stageToolbar")


def _rule_bodies(qss: str, selector_fragment: str) -> list[str]:
    """Bodies of top-level rules whose selector mentions the fragment.
    CSS comments are stripped first (the explanatory notes in theme.py
    mention 'max-height' in prose)."""
    qss = re.sub(r"/\*.*?\*/", "", qss, flags=re.DOTALL)
    out = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", qss):
        selector, body = m.group(1), m.group(2)
        if selector_fragment in selector:
            out.append((selector.strip(), body))
    return out


@pytest.mark.parametrize("base", [9, 12, 22])
def test_no_qss_height_pins_on_python_owned_bars(base):
    qss = theme.dark_qss(base=base)
    for frag in _PYTHON_OWNED:
        rules = _rule_bodies(qss, frag)
        assert rules, f"no rule found for {frag} (selector renamed?)"
        for selector, body in rules:
            # Only the FRAME rule itself must stay geometry-free; child
            # rules (buttons etc.) may size their own content.
            if selector.startswith(f"QFrame{frag}"):
                assert "min-height" not in body, (selector, body)
                assert "max-height" not in body, (selector, body)


def test_toolbar_height_survives_repolish(qapp):
    """End-to-end: after a font change (which re-applies the app
    stylesheet), the toolbar keeps its font-derived fixed height
    instead of being clamped back by QSS."""
    from linac_gen_gui.interphase.app import InterphaseWindow

    win = InterphaseWindow()
    try:
        win._apply_font_size(20, persist=False)
        expected = max(44, 20 + 30)
        assert win._toolbar.minimumHeight() == expected
        assert win._toolbar.maximumHeight() == expected
        assert win._statusbar.maximumHeight() == max(22, 20 + 10)
        assert win._titlebar.maximumHeight() == max(28, 20 + 16)
    finally:
        win._apply_font_size(12, persist=False)
        win.close()
        win.deleteLater()
