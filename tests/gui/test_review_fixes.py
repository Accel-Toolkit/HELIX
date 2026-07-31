"""Regression tests for the 2026-07 GUI review fixes.

Each test pins one verified defect from the full-GUI audit so it cannot
silently return:

* SC-matching ignored ``emit_ny`` (matching_dialog).
* Surrogate "Use" checkboxes desynced from the runtime registry, and
  "Deselect all" could not unregister a stale registration.
* CommandBus kept the redo stack across a coalesced ParamChange.
* Add-element wrote list params (Multipole.knl/ksl) back as strings.
* LatticeTimeline divided by zero on an all-zero-length lattice.
* status_message emissions had no visible sink in the status bar.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")


# ---------------------------------------------------------------------------
# emit_ny in SC matching
# ---------------------------------------------------------------------------
def test_sc_matching_uses_emit_ny(qapp, mini_lattice, monkeypatch):
    """Both SC-matched paths fed emit_nx-derived emittance to BOTH planes;
    the y-plane must use emit_ny."""
    from dataclasses import replace

    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.dialogs.matching_dialog import MatchingDialog

    cfg = replace(BeamConfig(), current=5.0, emit_nx=0.2, emit_ny=0.4)

    class _BeamStub:
        def get_beam_config(self):
            return cfg

    captured: list[dict] = []

    class _FakeSolver:
        def __init__(self, lattice, ref, initial, *args, **kwargs):
            captured.append(dict(initial))
            self._i = initial

        def run(self):
            i = self._i
            return SimpleNamespace(
                alpha_x=[i["alpha_x"]], beta_x=[i["beta_x"]],
                alpha_y=[i["alpha_y"]], beta_y=[i["beta_y"]],
            )

    import linac_gen.matching.periodic as periodic_mod
    import linac_gen.tracking.envelope as envelope_mod
    monkeypatch.setattr(
        periodic_mod, "find_periodic_twiss",
        lambda lattice, ref: {"alpha_x": 0.0, "beta_x": 1.0,
                              "alpha_y": 0.0, "beta_y": 1.0})
    monkeypatch.setattr(envelope_mod, "EnvelopeSolver", _FakeSolver)

    dlg = MatchingDialog(mini_lattice, _BeamStub())
    try:
        assert dlg._mode_combo.currentIndex() == 0   # whole-lattice mode
        dlg._compute_sc_matched()
        assert captured, "envelope solver never invoked"
        initial = captured[0]
        # emit_ny is 2× emit_nx, so the geometric emittances must differ 2×.
        assert initial["emit_y"] == pytest.approx(2.0 * initial["emit_x"])
    finally:
        dlg.deleteLater()


# ---------------------------------------------------------------------------
# Surrogate Use-checkbox ↔ registry sync
# ---------------------------------------------------------------------------
def test_surrogate_table_reflects_registry_and_deselect_unregisters(qapp):
    """A rebuilt table row must show the surrogate's TRUE registry state,
    and 'Deselect all' must actually unregister it (previously the boxes
    were hard-reset to unchecked, so setChecked(False) fired no signal
    and the registry silently kept the surrogate engaged)."""
    from linac_gen.surrogates import registry as _reg
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    tab = SurrogatesTab(AppState())
    meta = SimpleNamespace(
        val_mape=0.01, lattice_hash="test-hash-review",
        element_key="FMAP_RV",
        scope=SimpleNamespace(input_names=["w"], input_lo=[2.0],
                              input_hi=[2.5]),
    )
    surr = object()
    _reg.clear()
    try:
        tab._trained["FMAP_RV"] = (surr, Path("."), meta)
        _reg.register(surr, lattice_hash="test-hash-review",
                      element_key="FMAP_RV")

        tab._refresh_table()
        cb = tab._table.cellWidget(0, 3)
        assert cb is not None and cb.isChecked(), \
            "registered surrogate must show a ticked Use box after rebuild"

        tab._set_all_use(False)
        assert _reg.get("test-hash-review", "FMAP_RV") is None, \
            "Deselect all must unregister the surrogate"
        assert _reg.get_by_element_name("FMAP_RV") is None
    finally:
        _reg.clear()
        tab.deleteLater()


# ---------------------------------------------------------------------------
# Redo cleared on coalesced ParamChange
# ---------------------------------------------------------------------------
def test_redo_cleared_on_coalesced_param_change(qapp, mini_lattice):
    """The coalesce fast-path in CommandBus.do returned without clearing
    the redo stack, so Redo could re-apply an edit the user had undone."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.state import AppState

    state = AppState()
    state.set_lattice(mini_lattice, None)
    bus = state.bus
    d1, q1 = mini_lattice.elements[0], mini_lattice.elements[1]

    bus.do(ParamChangeCommand(q1, "gradient", 10.0, 11.0))
    bus.do(ParamChangeCommand(d1, "length", 100.0, 120.0))
    assert bus.undo()
    assert bus.can_redo            # the undone length change sits in redo

    # Same (element, attr) within the coalesce window → fast path.
    bus.do(ParamChangeCommand(q1, "gradient", 11.0, 12.0))
    assert q1.gradient == 12.0
    assert not bus.can_redo, \
        "a coalesced edit is a fresh action — redo history must be cleared"


# ---------------------------------------------------------------------------
# Add-element list parameters
# ---------------------------------------------------------------------------
def test_add_element_list_params_stay_lists(qapp):
    """Multipole.knl/ksl fell through to a plain QLineEdit and were written
    back as the literal string "[0.0, 0.0, 0.0]" — a user-created element
    that crashed tracking."""
    from linac_gen_gui.interphase.dialogs.add_element import AddElementDialog

    dlg = AddElementDialog()
    try:
        assert dlg._type_combo.findText("Multipole") >= 0
        dlg._type_combo.setCurrentText("Multipole")
        # Edit the knl text the way a user would, then accept.
        ed = dlg._field_widgets["knl"]
        ed.setText("0.1, 0.2")
        dlg._accept()
        el = dlg._element
        assert el is not None
        assert isinstance(el.knl, list) and el.knl == [0.1, 0.2]
        assert isinstance(el.ksl, list) and \
            all(isinstance(v, float) for v in el.ksl)
    finally:
        dlg.deleteLater()


# ---------------------------------------------------------------------------
# Timeline zero-total-length guard
# ---------------------------------------------------------------------------
def test_timeline_survives_all_zero_length_lattice(qapp):
    """sum(length)==0 with a non-empty element list divided by zero in
    set_lattice (the None/empty guard didn't cover it)."""
    from linac_gen_gui.interphase.plots.lattice_track import LatticeTimeline

    t = LatticeTimeline()
    try:
        fake = SimpleNamespace(elements=[
            SimpleNamespace(length=0.0, name="M1"),
            SimpleNamespace(length=0.0, name="M2"),
        ])
        t.set_lattice(fake)            # must not raise ZeroDivisionError
        assert t._total_length == 0.0
        t.set_lattice(None)            # existing guard still fine
    finally:
        t.deleteLater()


# ---------------------------------------------------------------------------
# MAD-X source files are never overwritten by the TraceWin writer
# ---------------------------------------------------------------------------
def test_save_lattice_routes_madx_to_save_as(qapp, mini_lattice, monkeypatch,
                                             tmp_path):
    """Ctrl+S on a lattice loaded from .madx/.seq used to silently overwrite
    the MAD-X source with TraceWin .dat text (and mark the bus clean)."""
    from linac_gen_gui.interphase.app import InterphaseWindow

    win = InterphaseWindow()
    try:
        src = tmp_path / "line.madx"
        src.write_text("! madx source — must survive Ctrl+S")
        win.state.set_lattice(mini_lattice, str(src))

        routed = []
        monkeypatch.setattr(win, "_save_lattice_as",
                            lambda: routed.append(True))
        win._save_lattice()
        assert routed, ".madx path must route to Save-As"
        assert src.read_text() == "! madx source — must survive Ctrl+S"

        # A .dat path still writes in place.
        written = []
        monkeypatch.setattr(win, "_write_lattice",
                            lambda p: written.append(p))
        win.state.set_lattice(mini_lattice, str(tmp_path / "line.dat"))
        win._save_lattice()
        assert written and written[0].endswith("line.dat")
    finally:
        win.close()
        win.deleteLater()


# ---------------------------------------------------------------------------
# Retraining an ENGAGED surrogate re-registers the fresh weights
# ---------------------------------------------------------------------------
def test_retrain_replaces_engaged_registration(qapp, mini_lattice,
                                               monkeypatch):
    """With registry-reflecting checkboxes, a retrain of a registered element
    must swap the registry entry to the NEW surrogate — otherwise the table
    shows the fresh row ticked while runs silently use the old weights."""
    from linac_gen.surrogates import registry as _reg
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab

    state = AppState()
    state.set_lattice(mini_lattice, None)
    tab = SurrogatesTab(state)

    meta = SimpleNamespace(
        val_mape=0.02, lattice_hash="retrain-hash", element_key="Q1",
        scope=SimpleNamespace(input_names=["w"], input_lo=[2.0],
                              input_hi=[2.5]),
    )
    import linac_gen.surrogates.base as base_mod
    import linac_gen.surrogates.training as training_mod
    monkeypatch.setattr(training_mod, "load_surrogate",
                        lambda out_dir: (object(), meta))
    monkeypatch.setattr(
        base_mod, "SurrogateFieldMap",
        lambda element, mlp, m: SimpleNamespace(metadata=m))

    _reg.clear()
    try:
        old = object()
        _reg.register(old, lattice_hash="retrain-hash", element_key="Q1")

        tab._worker = SimpleNamespace(out_dir=Path("."))
        tab._on_finished(meta)

        now = _reg.get("retrain-hash", "Q1")
        assert now is not None and now is not old, \
            "retrain must replace the engaged registration with fresh weights"
        # And the rebuilt row must show it ticked.
        assert tab._table.cellWidget(0, 3).isChecked()
    finally:
        _reg.clear()
        tab.deleteLater()


# ---------------------------------------------------------------------------
# Failure-study custom sets: prune stale names, keep valid ones
# ---------------------------------------------------------------------------
def test_failure_custom_sets_pruned_not_cleared(qapp, mini_lattice):
    """A lattice swap must drop only sets naming missing elements.  Matching
    Apply installs a deep copy with IDENTICAL names — those sets stay; and
    a mere param edit (same object re-broadcast) must not touch them."""
    import copy as _copy

    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.failure_study_tab import (
        FailureStudyTab,
    )

    state = AppState()
    state.set_lattice(mini_lattice, None)
    tab = FailureStudyTab(state)
    try:
        tab._custom_sets = [["Q1"], ["Q1", "GHOST"]]
        tab._set_list.addItem("Q1")
        tab._set_list.addItem("Q1 + GHOST")

        # Param edit: same lattice object re-broadcast → untouched.
        state.lattice_changed.emit(state.lattice)
        assert tab._custom_sets == [["Q1"], ["Q1", "GHOST"]]

        # Apply-style swap: new object, same names → stale set pruned only.
        state.set_lattice(_copy.deepcopy(mini_lattice), None)
        assert tab._custom_sets == [["Q1"]]
        assert tab._set_list.count() == 1
    finally:
        tab.deleteLater()


# ---------------------------------------------------------------------------
# Status messages are visible
# ---------------------------------------------------------------------------
def test_statusbar_displays_status_message(qapp):
    """Every status_message.emit used to vanish — the only listener
    repainted the bar and dropped the text."""
    from linac_gen_gui.interphase.chrome.statusbar import StatusBar
    from linac_gen_gui.interphase.state import AppState

    state = AppState()
    bar = StatusBar(state)
    try:
        state.status_message.emit("Saved → test.dat")
        assert bar._msg_seg.text() == "Saved → test.dat"
        assert bar._msg_timer.isActive()    # auto-clear armed
        bar._clear_message()
        assert bar._msg_seg.text() == ""
    finally:
        bar.deleteLater()


def test_statusbar_accepts_ndarray_results(qapp):
    """2026-07-28 live crash: ``if sigma_x:`` on an ndarray raises
    'truth value of an array is ambiguous' — results loaded from h5
    carry numpy arrays, not lists."""
    import numpy as np
    from linac_gen_gui.interphase.chrome.statusbar import StatusBar
    from linac_gen_gui.interphase.state import AppState

    class _R:
        sigma_x = np.array([1.0, 2.0, 3.25])
        transmission = np.array([100.0, 99.0, 97.4])

    sb = StatusBar(AppState())
    sb._refresh_results(_R())                  # must not raise
    assert "3.250" in sb._sigma_seg.text()
    assert "2.60" in sb._loss_seg.text()
    sb._refresh_results(None)                  # clear path intact
    assert "—" in sb._sigma_seg.text()


def test_excepthook_dialog_survives_window_close(tmp_path):
    """2026-07-28 native SIGSEGV: closing the excepthook's QMessageBox via
    the window-manager close button destroyed the C++ dialog from INSIDE
    its own closeEvent (the ``finished`` handler dropped the last Python
    reference mid-emission) — use-after-free in QDialogButtonBox.  The
    killer sequence must run in a subprocess so a regression cannot take
    pytest down with it."""
    import subprocess
    import sys as _sys

    code = """
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
app = QApplication(sys.argv)
from linac_gen_gui.interphase.app import _install_excepthook
_install_excepthook()
try:
    raise RuntimeError("boom")            # summon the dialog
except RuntimeError:
    sys.excepthook(*sys.exc_info())
dlg = app.activeModalWidget() or next(
    (w for w in app.topLevelWidgets() if w.isVisible()
     and w.metaObject().className() == "QMessageBox"), None)
assert dlg is not None, "excepthook dialog never appeared"
dlg.close()                               # the killer: WM close path
app.processEvents()                       # unwind closeEvent frames
app.processEvents()                       # deferred deleteLater runs
# a SECOND dialog must be possible (state actually released)
try:
    raise RuntimeError("boom2")
except RuntimeError:
    sys.excepthook(*sys.exc_info())
QTimer.singleShot(0, app.processEvents)
app.processEvents()
n = sum(1 for w in app.topLevelWidgets()
        if w.isVisible() and w.metaObject().className() == "QMessageBox")
assert n == 1, f"second dialog blocked (n={n}) — state leaked"
print("EXCEPTHOOK_CLOSE_OK")
"""
    import os as _os
    from pathlib import Path as _Path
    env = dict(_os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    # the subprocess must find linac_gen_gui regardless of how the
    # PARENT pytest got it (conftest sys.path vs shell PYTHONPATH)
    repo = _Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = _os.pathsep.join(
        [str(repo), str(repo / "gui"),
         env.get("PYTHONPATH", "")]).rstrip(_os.pathsep)
    res = subprocess.run([_sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120,
                         env=env)
    assert res.returncode == 0, (
        f"crashed rc={res.returncode}\n{res.stderr[-2000:]}")
    assert "EXCEPTHOOK_CLOSE_OK" in res.stdout
