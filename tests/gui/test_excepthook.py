"""Tests for the global exception hook installed by app.main().

Without a custom sys.excepthook, PyQt6 aborts the whole process when a
Python exception escapes a slot — the hook converts that into a stderr
traceback plus one (rate-limited, non-modal) dialog.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from linac_gen_gui.interphase import app as app_mod  # noqa: E402


def _visible_boxes() -> list[QMessageBox]:
    return [w for w in QApplication.topLevelWidgets()
            if isinstance(w, QMessageBox) and w.isVisible()]


@pytest.fixture()
def hook(qapp):
    old = sys.excepthook
    old_flag = app_mod._SHUTTING_DOWN
    # An earlier test may have closed a window, latching the module
    # shutdown flag — this fixture tests the hook in a LIVE app.
    app_mod._SHUTTING_DOWN = False
    app_mod._install_excepthook()
    yield sys.excepthook
    sys.excepthook = old
    app_mod._SHUTTING_DOWN = old_flag
    for box in _visible_boxes():
        box.done(0)
        box.deleteLater()


def _boom():
    try:
        raise RuntimeError("synthetic failure for the hook test")
    except RuntimeError:
        return sys.exc_info()


def test_hook_prints_traceback_and_shows_one_dialog(hook, capsys):
    etype, value, tb = _boom()
    hook(etype, value, tb)
    err = capsys.readouterr().err
    assert "synthetic failure for the hook test" in err
    assert "RuntimeError" in err
    assert len(_visible_boxes()) == 1

    # Second exception while the first dialog is still up → no new dialog.
    hook(etype, value, tb)
    assert len(_visible_boxes()) == 1

    # Dismissing the dialog re-arms the hook.
    _visible_boxes()[0].done(0)
    hook(etype, value, tb)
    assert len(_visible_boxes()) == 1


def test_hook_ignores_keyboard_interrupt(hook):
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        etype, value, tb = sys.exc_info()
    hook(etype, value, tb)
    assert len(_visible_boxes()) == 0


def test_hook_silent_while_shutting_down(hook, capsys):
    etype, value, tb = _boom()
    app_mod._SHUTTING_DOWN = True
    try:
        hook(etype, value, tb)
    finally:
        app_mod._SHUTTING_DOWN = False
    # Traceback still logged, but no UI while tearing down.
    assert "RuntimeError" in capsys.readouterr().err
    assert len(_visible_boxes()) == 0
