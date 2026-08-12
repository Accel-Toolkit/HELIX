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
def _sandbox_calc_dir(tmp_path):
    """Point the auto-dump calc dir at a per-test tmp for EVERY GUI
    test.  The QSettings sandbox (HELIX_QSETTINGS_DIR) leaves calcDir
    unset, and the fallback is cwd-relative (``cwd/runs``) — so any
    e2e test that completes a run auto-dumped into the developer's
    REAL runs/ directory (observed 2026-08-11: a backtrack e2e mp file
    landed next to the user's own results).  Whole-family hardening,
    not per-test opt-in; tests that need to *inspect* the dump keep
    using the explicit ``calc_dir`` fixture, which overrides this one."""
    from linac_gen_gui.interphase.app import _SETTINGS_CALC_DIR, _settings
    s = _settings()
    old = s.value(_SETTINGS_CALC_DIR, "")
    s.setValue(_SETTINGS_CALC_DIR, str(tmp_path))
    yield
    s.setValue(_SETTINGS_CALC_DIR, old)


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
