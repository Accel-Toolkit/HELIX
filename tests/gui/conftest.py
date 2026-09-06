"""Test fixtures for the GUI test suite.

Provides:
    qapp           — process-wide QApplication (offscreen platform)
    mini_lattice   — Drift, Quad, Drift, RFGap, Drift fixture
    rfq_lattice    — same plus an RFQ_CELL chain so writer-roundtrip
                     tests have something to bite on.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Force offscreen Qt before any Qt imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Offscreen alone is not enough on hosts where Qt can't locate PyQt6's
# bundled platform plugins (observed on macOS + anaconda: a bare
# ``pytest`` run SIGABRTed the whole suite at the first GUI test with
# "no Qt platform plugin could be initialized").  Point QT_PLUGIN_PATH
# at the PyQt6 wheel's plugin dir when the caller hasn't set it —
# harmless where Qt already finds them.
if "QT_PLUGIN_PATH" not in os.environ:
    try:
        import PyQt6  # noqa: F401  (path probe only — no Qt classes yet)
        _plugins = os.path.join(
            os.path.dirname(PyQt6.__file__), "Qt6", "plugins")
        if os.path.isdir(_plugins):
            os.environ["QT_PLUGIN_PATH"] = _plugins
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Sandbox QSettings for the WHOLE test process — before any GUI import.
#
# The app persists real user state (recent projects, last project /
# lattice paths, window geometry, panel state) through the
# make_settings factory in linac_gen_gui.interphase.app_settings.
# Tests that exercise the real save/load paths would otherwise write
# into the developer's actual settings store — one test literally
# filled the user's File → Recent Projects menu with pytest-tmp
# "broken.lgproj" entries.  Setting HELIX_QSETTINGS_DIR makes the
# factory return throwaway INI files instead.  (QSettings.setDefaultFormat
# is NOT a workable alternative: on macOS the two-argument constructor
# resolves to NativeFormat/CFPreferences regardless — verified.)
# ---------------------------------------------------------------------------
os.environ.setdefault("HELIX_QSETTINGS_DIR",
                      tempfile.mkdtemp(prefix="helix-test-qsettings-"))

import pytest

from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.rfq_cell import RfqCell


@pytest.fixture(autouse=True)
def _no_stt_prewarm(monkeypatch):
    # panel construction must not load the real Whisper model
    monkeypatch.setenv("HELIX_ASSIST_NO_PREWARM", "1")
    yield


@pytest.fixture(autouse=True)
def gui_message_boxes(monkeypatch):
    """Modal tripwire for EVERY gui test: no QMessageBox can ever block.

    An offscreen modal box has no one to click it — a single unstubbed
    ``QMessageBox.critical``/``exec`` hangs the whole suite forever
    (2026-09-02 full-suite hang: a settings-leak-induced backtrack
    failure surfaced through an unstubbed critical box in a module
    whose local ``win`` fixture shadowed the stubbing one).  Stub the
    four statics plus instance ``exec`` process-wide, record
    ``(kind, title, text)`` and answer Yes.

    Record-only, NEVER fail-at-teardown: many green tests intentionally
    provoke boxes (backtrack caveats, train sidecar refusals, all of
    test_project_io).  Per-test monkeypatches layer on the same
    function-scoped MonkeyPatch — applied later, undone first, their
    return values win — so existing per-test stubs keep working
    unchanged.  Request the fixture by name to read the recorded list
    (the shared ``win`` fixture exposes it as ``win.message_boxes``).
    """
    from PyQt6.QtWidgets import QMessageBox

    boxes: list[tuple[str, str, str]] = []

    def _static(kind):
        def _record(*a, **k):
            boxes.append((kind,
                          str(a[1]) if len(a) > 1 else "",
                          str(a[2]) if len(a) > 2 else ""))
            return QMessageBox.StandardButton.Yes
        return _record

    for _name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, _name, staticmethod(_static(_name)))

    def _exec(self, *a, **k):
        boxes.append(("exec", self.windowTitle(), self.text()))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "exec", _exec)
    yield boxes


@pytest.fixture(autouse=True)
def _flush_deferred_deletes():
    """Actually destroy widgets scheduled with deleteLater() after EVERY test.

    deleteLater() only posts a DeferredDelete event, which needs a running
    event loop to be processed — pytest never runs one, so every `win`
    fixture's teardown (`w.close(); w.deleteLater()`) leaked the whole
    window.  Windows accumulated for the entire session (+~464 top-level
    widgets each), and because InterphaseWindow.__init__ applies an
    app-wide stylesheet that re-polishes every live widget in the process,
    construction cost grew superlinearly — the external Windows report
    measured 3.65 s -> 34.4 s by the 6th window (7.6x), and on macOS we
    measured 1.36 s -> 5.20 s (3.8x).  Flushing the posted DeferredDelete
    events after each test frees them and flattens the cost.

    Autouse fixtures are instantiated before a test's own fixtures, so this
    finalizer runs AFTER their teardowns have posted the deletions.

    Residual (measured, accepted): ~365 pyqtgraph context menus per window
    (ViewBoxMenu + submenus) are created parentless by pyqtgraph and are
    not covered by the window's deleteLater; they cost ~0.07 s/window of
    re-polish versus 0.77 s/window when the whole window leaked.  The
    external report's own "flushed" column shows the same residual slope.
    """
    yield
    # sendPostedEvents processes the deletions synchronously; loop until no
    # new deferred deletions were scheduled by the destructors themselves.
    app = QCoreApplication.instance()
    if app is not None:
        # ONE pass only, deliberately: a second pass would also deliver
        # the DeferredDeletes scheduled BY the first pass's destructors
        # (pyqtgraph ViewBox -> menu chains), and forcing that deep
        # cascade segfaulted the full suite deterministically
        # (assistant_chat._content_bottom, 3/3 at the same site; clean
        # without it).  One pass frees the window trees, which is the
        # entire measured cost.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def _settings_hygiene():
    """Scrub session-restore keys from the process-wide (sandboxed)
    QSettings store before AND after EVERY gui test.

    The store is process-wide: a key left behind by one module leaks
    into every later one.  Two incident classes so far — a leftover
    ``lastProjectPath`` flips ``_save_project`` into silent-to-current-
    path mode, and a leftover ``lastLatticePath`` (2026-09-02 full-suite
    hang) let a later window's restore timer silently swap
    ``state.lattice`` mid-test.  CLEAR-only, deliberately not
    snapshot/restore: no gui test needs one of these keys to survive a
    test boundary (tests that need one set it themselves, inside the
    test body), and clearing self-heals instead of re-planting captured
    pollution.

    The finalizer constructs a FRESH ``_settings()`` handle: a handle
    held across the yield dies with any QApplication a test destroys
    (the ``_flush_deferred_deletes`` immune shape — a wrapped-C/C++-
    object-deleted RuntimeError otherwise)."""
    from linac_gen_gui.interphase import app as app_mod
    keys = (app_mod._SETTINGS_SESSION_BEAM, app_mod._SETTINGS_LAST_LATTICE,
            app_mod._SETTINGS_LAST_PROJECT, app_mod._SETTINGS_LAST_DIR)
    s = app_mod._settings()
    for k in keys:
        s.remove(k)
    del s                               # never hold a handle across the yield
    yield
    s = app_mod._settings()             # fresh handle (see docstring)
    for k in keys:
        s.remove(k)
    s.sync()


@pytest.fixture(autouse=True)
def _sandbox_calc_dir(tmp_path):
    """Point the auto-dump calc dir at a per-test tmp for EVERY GUI
    test.  The QSettings sandbox (HELIX_QSETTINGS_DIR) leaves calcDir
    unset, and the fallback is cwd-relative (``cwd/runs``) — so any
    e2e test that completes a run auto-dumped into the developer's
    REAL runs/ directory (observed 2026-08-11: a backtrack e2e mp file
    landed next to the user's own results).  Whole-family hardening,
    not per-test opt-in; tests that need to *inspect* the dump keep
    using the explicit ``calc_dir`` fixture, which overrides this one.

    ``old`` is a plain str, safe to hold across the yield; the QSettings
    handles are NOT — construct a fresh one on each side (same
    QSettings-lifetime hazard as ``_settings_hygiene``)."""
    from linac_gen_gui.interphase.app import _SETTINGS_CALC_DIR, _settings
    old = _settings().value(_SETTINGS_CALC_DIR, "")
    _settings().setValue(_SETTINGS_CALC_DIR, str(tmp_path))
    yield
    _settings().setValue(_SETTINGS_CALC_DIR, old)


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole test session."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def mini_lattice() -> Lattice:
    lat = Lattice()
    lat.add(Drift(name="D1",  length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="Q1",  length=200.0, gradient=10.0, aperture=10.0))
    lat.add(Drift(name="D2",  length=100.0, aperture=10.0))
    lat.add(RFGap(name="G1",  voltage=0.5, phase=-30.0, frequency=162.5))
    lat.add(Drift(name="D3",  length=100.0, aperture=10.0))
    return lat


@pytest.fixture
def rfq_lattice() -> Lattice:
    lat = Lattice()
    lat.add(Drift(name="LEAD", length=50.0, aperture=10.0))
    # A small RFQ-cell chain so the writer has something to chew on.
    # A10 below the 0.02 consistency-check gate keeps the fixture
    # silent (the triplet here is a writer exercise, not physics).
    for i in range(3):
        lat.add(RfqCell(
            name=f"RFQ_{i}",
            voltage_V=70_000.0, r0_mm=3.5, A10=0.01,
            modulation=1.5 + 0.1 * i, length_mm=10.0,
            phi_s_deg=-30.0, cell_type=2,
        ))
    lat.add(Drift(name="TAIL", length=50.0, aperture=10.0))
    return lat


# ---------------------------------------------------------------------------
# Shared e2e window scaffold (fix-plan shared infrastructure).
#
# The same scaffold was re-implemented ad hoc in
# tests/gui/test_backtrack_end_to_end.py and in the defect repro scripts
# (tune-stale, assist-mutate); this fixture is the single copy.  A test
# module that defines its own ``win`` fixture shadows this one.
# ---------------------------------------------------------------------------
@pytest.fixture()
def win(qapp, monkeypatch, gui_message_boxes):
    """A real InterphaseWindow with only the human seams stubbed.

    * ``_launch_update_check`` is a no-op (belt and braces on top of the
      PYTEST_CURRENT_TEST guard — never a network probe from a test).
    * ``QMessageBox`` is already stubbed process-wide by the autouse
      ``gui_message_boxes`` tripwire (record-and-return-Yes); its
      recorded ``(kind, title, text)`` tuples are exposed as
      ``win.message_boxes``.
    * Auto-dump calc dir and QSettings are already sandboxed by the
      autouse fixtures above.

    Helpers attached to the window:
        win.open_lattice(path)            File > Open Lattice via the real slot
        win.pump(sec=0.3)                 process events for ``sec`` seconds
        win.pump_until(pred, timeout_ms)  pump until pred() is True
        win.wait_worker(worker)           join a run worker + deliver signals
        win.wait_popup_idle(dlg)          wait for a results popup's workers
    """
    import time as _time

    from linac_gen_gui.interphase import app as appmod
    from linac_gen_gui.interphase.app import InterphaseWindow

    monkeypatch.setattr(InterphaseWindow, "_launch_update_check",
                        lambda self: None)

    w = InterphaseWindow()
    w.message_boxes = gui_message_boxes

    def _pump(sec: float = 0.3) -> None:
        t0 = _time.time()
        while _time.time() - t0 < sec:
            qapp.processEvents()
            _time.sleep(0.02)

    def _pump_until(pred, timeout_ms: int = 60_000) -> bool:
        t0 = _time.time()
        while (_time.time() - t0) * 1000.0 < timeout_ms:
            qapp.processEvents()
            if pred():
                return True
            _time.sleep(0.02)
        return False

    def _wait_worker(worker, timeout_ms: int = 300_000) -> None:
        assert worker is not None, "worker was never constructed"
        assert worker.wait(timeout_ms), "worker thread did not finish"
        _pump(0.5)

    def _wait_popup_idle(dlg, timeout_s: float = 300.0) -> None:
        t0 = _time.time()
        while _time.time() - t0 < timeout_s:
            qapp.processEvents()
            wk = getattr(dlg, "_worker", None)
            pw = getattr(dlg, "_probe_worker", None)
            busy = ((wk is not None and wk.isRunning())
                    or (pw is not None and pw.isRunning()))
            if not busy and getattr(dlg, "_pending_key", None) is None:
                break
            _time.sleep(0.05)
        _pump(0.5)

    def _open_lattice(path) -> None:
        monkeypatch.setattr(
            appmod.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (str(path), "")))
        w._open_lattice()
        qapp.processEvents()

    w.pump = _pump
    w.pump_until = _pump_until
    w.wait_worker = _wait_worker
    w.wait_popup_idle = _wait_popup_idle
    w.open_lattice = _open_lattice

    yield w

    # Teardown: stop popup-owned workers, then any run workers, then the
    # window itself (the autouse DeferredDelete flusher frees it).
    for wk in w.results_tab.shutdown_begin():
        wk.wait(5000)
    for wk in (getattr(w, "_envelope_worker", None),
               getattr(w, "_mp_worker", None)):
        if wk is not None and wk.isRunning():
            if hasattr(wk, "request_stop"):
                wk.request_stop()
            wk.wait(5000)
    w.close()
    w.deleteLater()
