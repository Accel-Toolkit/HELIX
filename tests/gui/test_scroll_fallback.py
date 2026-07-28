"""Guard: dense tab pages sit inside scroll areas; small windows
degrade to scrolling instead of Qt compressing widgets below their
minimums (the overlapping-boxes / hidden-text symptom on small or
scaled screens)."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QScrollArea  # noqa: E402

WRAPPED = ["beam", "matching", "convergence", "surrogates", "errors"]
UNWRAPPED = ["lattice", "failures", "results"]


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.close()
    w.deleteLater()


def _tab_index(tab_id: str) -> int:
    from linac_gen_gui.interphase.state import TABS
    return [tid for tid, _ in TABS].index(tab_id)


def test_dense_tabs_are_scroll_wrapped(win):
    pages = {
        "beam": win.beam_tab, "matching": win.matching_tab,
        "convergence": win.convergence_tab, "surrogates": win.surrogates_tab,
        "errors": win.errors_tab,
    }
    for tid, page in pages.items():
        holder = win._tabs.widget(_tab_index(tid))
        assert isinstance(holder, QScrollArea), tid
        assert holder.widget() is page, tid
        assert holder.widgetResizable(), tid
        # The wrapper must not add a Tab-key stop.
        assert holder.focusPolicy() == Qt.FocusPolicy.NoFocus, tid


def test_special_tabs_stay_unwrapped(win):
    assert win._tabs.widget(_tab_index("lattice")) is win.lattice_tab
    assert win._tabs.widget(_tab_index("failures")) is win.failures_tab
    assert win._tabs.widget(_tab_index("results")) is win.results_tab


def test_failure_tab_left_column_scrolls(win):
    scrolls = win.failures_tab.findChildren(QScrollArea)
    capped = [s for s in scrolls if s.maximumWidth() == 380]
    assert capped, "left control column is no longer scroll-wrapped"
    assert capped[0].widgetResizable()


def test_small_window_resize_smoke(win):
    """At the new 1000x640 floor every tab must lay out without the
    page being squeezed below the viewport (scrollbars absorb the
    pressure instead)."""
    win.resize(1000, 640)
    win.show()
    for i in range(win._tabs.count()):
        win._tabs.setCurrentIndex(i)
        QApplication.processEvents()
        holder = win._tabs.widget(i)
        if isinstance(holder, QScrollArea):
            page = holder.widget()
            assert page.width() >= holder.viewport().width() - 1, i
    assert win.minimumWidth() == 1000
    assert win.minimumHeight() == 640
