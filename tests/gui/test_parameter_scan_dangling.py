"""Regression: the Parameter Scan dialog must not leave a stale signal
connection that raises on its dead C++ widgets.

Bug (user-reported 2026-07-16): opening a project after the scan dialog
had been opened+reopened surfaced an error dialog with
    RuntimeError: wrapped C/C++ object of type QComboBox has been deleted
because __init__ connected a LAMBDA to the app-wide ``lattice_changed``
signal.  PyQt keeps the lambda connected after the dialog is destroyed
(a lambda has no identifiable receiver to auto-disconnect, and — as
measured — even a bound method stays registered here), so the slot fires
on the deleted dialog at the next lattice load.

Note the failure does NOT propagate through ``emit()`` — PyQt routes the
slot exception to ``sys.excepthook`` (the GUI's hook shows the error
box), so a test that merely calls ``set_lattice`` and checks for a raised
exception passes even when broken.  These tests capture ``sys.excepthook``
instead, and directly exercise the guard.

Fix: bound method + RuntimeError guard that self-disconnects on a dead
object.  See feedback_pyqt_lambda_connections.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip
from PyQt6.QtWidgets import QApplication

from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.dialogs.parameter_scan import ParameterScanDialog
from linac_gen.io.tracewin_parser import parse_tracewin


def _lattice():
    lat, _ = parse_tracewin("examples/halo_fodo.dat")
    return lat


@pytest.fixture()
def capture_slot_exceptions():
    """Record exceptions PyQt routes to sys.excepthook from slot calls."""
    caught: list = []
    saved = sys.excepthook
    sys.excepthook = lambda _t, v, _tb: caught.append(v)
    try:
        yield caught
    finally:
        sys.excepthook = saved


def test_destroyed_dialog_does_not_surface_error_on_lattice_load(
        qapp, capture_slot_exceptions):
    """Destroy a scan dialog, then load a lattice: no exception may reach
    sys.excepthook (with the pre-fix lambda this recorded RuntimeError)."""
    state = AppState()
    state.set_lattice(_lattice(), "examples/halo_fodo.dat")

    d1 = ParameterScanDialog(None, state)     # connects to lattice_changed
    d2 = ParameterScanDialog(None, state)     # a live second instance
    sip.delete(d1)                            # reopen path's deleteLater

    state.set_lattice(_lattice(), "examples/halo_fodo.dat")
    QApplication.processEvents()

    assert capture_slot_exceptions == [], (
        f"stale slot surfaced {capture_slot_exceptions!r}")
    assert d2._element_combo.count() > 0      # the live dialog refreshed
    sip.delete(d2)


def test_guard_swallows_dead_widget_access(qapp):
    """Direct call: a dialog whose combo is force-deleted handles a
    lattice_changed callback without raising (the guard)."""
    state = AppState()
    state.set_lattice(_lattice(), "examples/halo_fodo.dat")
    dlg = ParameterScanDialog(None, state)
    sip.delete(dlg._element_combo)            # combo dead, dialog alive
    dlg._on_lattice_changed(None)             # must NOT raise
    sip.delete(dlg)
