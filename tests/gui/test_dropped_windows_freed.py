"""A window a test drops is destroyed before the next test runs.

Parentless tabs and popups sit in reference cycles, so refcounting never
frees them.  Left to the cyclic collector, a collection could fire inside a
later test's processEvents() while Qt walked one of their plot scenes and
free that scene's items mid-walk (SIGBUS in QGraphicsItem::sceneBoundingRect,
full suite 2026-10-29).  The autouse ``_flush_deferred_deletes`` fixture
collects them at teardown.  The two tests run in file order.
"""
from __future__ import annotations

import weakref

import pytest

pytest.importorskip("PyQt6")

_noop = lambda *a, **k: None                                  # noqa: E731
_dropped: dict = {}


def test_a_window_with_plot_scenes_is_dropped(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    from PyQt6.QtWidgets import QGraphicsView
    tab = ResultsTab(AppState(), _noop, _noop, _noop)
    views = tab.findChildren(QGraphicsView)
    assert views                                   # it carries plot scenes
    _dropped["tab"] = weakref.ref(tab)
    _dropped["views"] = [weakref.ref(v) for v in views]
    del tab, views
    # still alive here: only the cyclic collector frees it
    assert _dropped["tab"]() is not None, (
        "ResultsTab is no longer in a reference cycle: this pair (and the "
        "conftest's gc.collect) no longer guards anything")


def test_the_dropped_window_is_gone_before_the_next_test(qapp):
    assert _dropped, "runs after test_a_window_with_plot_scenes_is_dropped"
    assert _dropped["tab"]() is None
    assert not [r for r in _dropped["views"] if r() is not None]
