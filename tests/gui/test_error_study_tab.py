"""Smoke tests for the new Error Study tab."""
from __future__ import annotations

import os
import pytest
import sys

# QT_QPA_PLATFORM is set by tests/gui/conftest.py — but in case this test runs
# in isolation, ensure we have an offscreen platform.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure the GUI package is importable (repo-relative; pyproject's
# pythonpath already covers pytest runs — this keeps standalone
# execution working too).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "gui"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_error_study_tab_loads(qapp):
    from linac_gen_gui.interphase.tabs.error_study_tab import ErrorStudyTab
    from linac_gen_gui.interphase.state import AppState

    state = AppState()
    tab = ErrorStudyTab(state)
    assert tab is not None
    assert tab._target.count() == 4   # Quadrupole / Cavity / Bend / Solenoid


def test_add_element_error_updates_list(qapp):
    from linac_gen_gui.interphase.tabs.error_study_tab import ErrorStudyTab
    from linac_gen_gui.interphase.state import AppState
    state = AppState()
    tab = ErrorStudyTab(state)

    tab._pattern.setText("QUAD_*")
    tab._param.setCurrentText("dx")
    tab._sigma.setValue(0.1)
    tab._on_add_element_error()
    assert len(tab._element_errors) == 1
    assert tab._list.count() == 1


def test_add_beam_error_updates_list(qapp):
    from linac_gen_gui.interphase.tabs.error_study_tab import ErrorStudyTab
    from linac_gen_gui.interphase.state import AppState
    state = AppState()
    tab = ErrorStudyTab(state)
    tab._b_param.setCurrentText("centroid_x")
    tab._b_sigma.setValue(0.05)
    tab._on_add_beam_error()
    assert len(tab._beam_errors) == 1


def test_delete_selected_removes(qapp):
    from linac_gen_gui.interphase.tabs.error_study_tab import ErrorStudyTab
    from linac_gen_gui.interphase.state import AppState
    state = AppState()
    tab = ErrorStudyTab(state)
    tab._on_add_element_error()
    tab._on_add_beam_error()
    tab._list.setCurrentRow(0)
    tab._delete_selected()
    assert len(tab._element_errors) == 0
    assert len(tab._beam_errors) == 1   # the beam error remains


def test_run_button_disabled_when_no_lattice(qapp, monkeypatch):
    """Running with no lattice in state surfaces a popup, doesn't crash."""
    from linac_gen_gui.interphase.tabs import error_study_tab as estab
    from linac_gen_gui.interphase.state import AppState
    # QMessageBox.warning would be modal in interactive mode and block
    # the test; stub it to a no-op call counter.
    calls = []
    monkeypatch.setattr(
        estab.QMessageBox, "warning",
        staticmethod(lambda *a, **kw: calls.append(("warning", a))),
    )
    state = AppState()
    tab = estab.ErrorStudyTab(state)
    tab._on_add_element_error()
    tab._on_run()
    assert any(c[0] == "warning" for c in calls)


def test_app_loads_with_errors_tab(qapp):
    """The full app loads with the new tab in the right slot."""
    from linac_gen_gui.interphase.app import InterphaseWindow
    main = InterphaseWindow()
    labels = [main._tabs.tabText(i) for i in range(main._tabs.count())]
    assert "Error Study" in labels
    # Error Study sits between Surrogates (M6) and Results.
    from linac_gen_gui.interphase.state import TABS
    assert labels[[t for t, _ in TABS].index("errors")] == "Error Study"
    main.close()


def test_ensemble_popup_handles_empty_state(qapp):
    from linac_gen_gui.interphase.tabs.results_tab import _EnsemblePopup
    from linac_gen_gui.interphase.state import AppState
    state = AppState()
    dlg = _EnsemblePopup(parent=None, state=state)
    dlg.refresh(None)
    assert "No error study" in dlg._summary.text()


def test_stop_button_lifecycle_and_partial_results(qapp, monkeypatch,
                                                   mini_lattice):
    """Stop is enabled only while a study runs; a stopped study stores a
    partial ensemble (whole seeds only) — unless zero seeds completed,
    in which case nothing may overwrite the previous results."""
    from dataclasses import replace

    from linac_gen_gui.interphase.tabs import error_study_tab as estab
    from linac_gen_gui.interphase.state import AppState
    from linac_gen.core.config import BeamConfig

    monkeypatch.setattr(estab._StudyWorker, "start", lambda self: None)

    state = AppState()
    state.set_lattice(mini_lattice, None)
    state.set_beam_config(replace(BeamConfig(), current=0.0))
    tab = estab.ErrorStudyTab(state)
    tab._on_add_element_error()

    assert not tab._stop_btn.isEnabled()
    tab._on_run()
    assert tab._stop_btn.isEnabled()
    assert not tab._run_btn.isEnabled()

    class _FakeResults:
        n_seeds = 2
        n_requested = 50

    prior = object()
    state.error_study_results = prior
    tab._on_done(_FakeResults(), stopped=True)
    assert state.error_study_results is not prior      # partial stored
    assert "2/50" in tab._status.text()
    assert tab._run_btn.isEnabled()
    assert not tab._stop_btn.isEnabled()

    class _EmptyResults:
        n_seeds = 0
        n_requested = 50

    state.error_study_results = prior
    tab._on_done(_EmptyResults(), stopped=True)
    assert state.error_study_results is prior          # nothing stored
    assert "nothing stored" in tab._status.text()


def test_study_uses_sc_config_provider(qapp, monkeypatch, mini_lattice):
    """Regression: studies used to read state.sc_config — an attribute
    that is never set — so every Monte Carlo run silently had space
    charge OFF.  The tab must now pull the canonical config through the
    provider the app wires in."""
    from dataclasses import replace

    from linac_gen_gui.interphase.tabs import error_study_tab as estab
    from linac_gen_gui.interphase.state import AppState
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig

    # Never actually spawn the QThread.
    monkeypatch.setattr(estab._StudyWorker, "start", lambda self: None)

    state = AppState()
    state.set_lattice(mini_lattice, None)
    state.set_beam_config(replace(BeamConfig(), current=5.0))

    tab = estab.ErrorStudyTab(state)
    sentinel = SpaceChargeConfig(nx=24, ny=24, nz=24)
    seen_calls = []

    def provider(current, continuous=False):
        seen_calls.append((current, continuous))
        return sentinel

    tab.sc_config_provider = provider
    tab._on_add_element_error()
    tab._on_run()

    assert seen_calls == [(5.0, False)]
    assert tab._worker is not None
    assert tab._worker._study.sc_config is sentinel
    assert "space charge ON" in tab._status.text()


def test_correction_targets_and_backend_wiring(qapp, monkeypatch,
                                               mini_lattice):
    """Review gap closure: the run slot must map the 'Steer onto
    DIAG_POSITION targets' checkbox to targets='deck' and the
    'envelope (fast)' combo item to reading_backend='envelope' — and
    the defaults must preserve legacy behavior (no targets, mp)."""
    from dataclasses import replace

    from linac_gen_gui.interphase.tabs import error_study_tab as estab
    from linac_gen_gui.interphase.state import AppState
    from linac_gen.core.config import BeamConfig

    monkeypatch.setattr(estab._StudyWorker, "start", lambda self: None)
    state = AppState()
    state.set_lattice(mini_lattice, None)
    state.set_beam_config(replace(BeamConfig(), current=0.0))

    tab = estab.ErrorStudyTab(state)
    tab._on_add_element_error()
    tab._corr_enable.setChecked(True)
    tab._corr_targets.setChecked(True)
    idx = tab._corr_backend.findText("envelope (fast)")
    assert idx >= 0, "combo item renamed — update the startswith mapping"
    tab._corr_backend.setCurrentIndex(idx)
    tab._on_run()
    kw = tab._worker._study._correction_kwargs
    assert kw["targets"] == "deck"
    assert kw["reading_backend"] == "envelope"

    tab2 = estab.ErrorStudyTab(state)
    tab2._on_add_element_error()
    tab2._corr_enable.setChecked(True)
    tab2._on_run()
    kw2 = tab2._worker._study._correction_kwargs
    assert kw2["targets"] is None
    assert kw2["reading_backend"] == "mp"
