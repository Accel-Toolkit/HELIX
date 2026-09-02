"""GUI-side docs-drift regressions (2026-09 round).

Three seams where the GUI's own text or plots drifted from what the
docs promise:

* the About dialog's tab list is derived from ``state.TABS`` so it can
  never drift again (it had drifted from 5 to 9 tabs);
* the Matching-tab AUTO-ADJUST hint no longer names a single algorithm
  (the tab exposes all seven ``MATCH_ALGORITHMS``);
* the Results-tab error-study ensemble popup draws the "max over seeds"
  overlay the manual's three-plot TL;DR promises
  (``08_errors/06_interpreting.md``), in the no-study, 1-seed and
  n-seed regimes.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def test_about_dialog_lists_every_tab(win, monkeypatch):
    """Help → About lists every ``state.TABS`` label (derived, not
    hand-maintained).  Exec is already stubbed by the ``win`` fixture —
    no modal ever runs offscreen."""
    from PyQt6.QtWidgets import QMessageBox
    from linac_gen_gui.interphase.state import TABS

    captured: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "setText",
        lambda self, text: captured.append(str(text)))

    # Real entry path: the toolbar's About action signal.
    win._toolbar.open_about_requested.emit()

    assert captured, "About dialog never called setText"
    text = captured[-1]
    for _tab_id, label in TABS:
        assert label in text, f"About dialog omits the {label!r} tab"
    tabs_line = next((ln for ln in text.splitlines()
                      if ln.startswith("Tabs:")), "")
    assert tabs_line, "About dialog has no 'Tabs:' line"
    assert tabs_line.count(" · ") >= len(TABS) - 1, (
        f"'Tabs:' line does not list all {len(TABS)} tabs: {tabs_line!r}")


def test_matching_hint_names_no_single_algorithm(qapp, mini_lattice):
    """The AUTO-ADJUST hint must not claim a single fixed optimiser —
    the tab exposes all seven MATCH_ALGORITHMS via its Algorithm combo."""
    from PyQt6.QtWidgets import QLabel
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    from linac_gen_gui.interphase.tabs.matching_tab import MatchingTab

    st = AppState()
    st.set_lattice(mini_lattice, "/tmp/docs_drift_matching.dat")
    beam_tab = BeamTab(st)
    tab = MatchingTab(st, beam_tab)
    try:
        labels = [lb.text() for lb in tab.findChildren(QLabel)]
        assert not any("Levenberg-Marquardt matcher" in t for t in labels), (
            "Matching-tab hint still names the Levenberg-Marquardt "
            "matcher as THE matcher")
        assert any(t == "Algorithm" for t in labels), (
            "Matching tab lost its Algorithm selector label")
    finally:
        tab.deleteLater()
        beam_tab.deleteLater()


def _stub_recorder(s, sigma_x, sigma_y, transmission):
    return SimpleNamespace(
        s=np.asarray(s, dtype=float),
        sigma_x=np.asarray(sigma_x, dtype=float),
        sigma_y=np.asarray(sigma_y, dtype=float),
        transmission=np.asarray(transmission, dtype=float),
    )


def _curve_xy(item):
    x, y = item.getData()
    x = np.asarray(x if x is not None else [], dtype=float)
    y = np.asarray(y if y is not None else [], dtype=float)
    return x, y


@pytest.mark.parametrize("n_seeds", [1, 3])
def test_ensemble_popup_max_over_seeds_overlay(qapp, n_seeds):
    """Both regimes: no study → max curves cleared + 'No error study'
    summary; a study → the dotted max curve equals np.max over the
    recorders at every s (1-seed and n-seed ensembles)."""
    from linac_gen.errors.error_model import ErrorStudyResults
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import _EnsemblePopup

    s = [0.0, 100.0, 200.0, 300.0]
    recs = [
        _stub_recorder(
            s,
            sigma_x=1.0 + 0.1 * i + np.linspace(0.0, 0.5, 4),
            sigma_y=2.0 - 0.2 * i + np.linspace(0.0, 0.3, 4),
            transmission=[100.0, 100.0, 99.5 - i, 99.0 - i],
        )
        for i in range(n_seeds)
    ]
    study = ErrorStudyResults(recs)

    state = AppState()
    dlg = _EnsemblePopup(parent=None, state=state)
    try:
        # Regime A: no study — everything cleared, honest summary.
        dlg.refresh(None)
        assert "No error study" in dlg._summary.text()
        for item in (dlg._sx_max, dlg._sy_max):
            x, y = _curve_xy(item)
            assert x.size == 0 and y.size == 0

        # Regime B: a study — max curve == np.max over recorders per s.
        state.error_study_results = study
        dlg.refresh(None)
        sx_expect = np.max([r.sigma_x for r in recs], axis=0)
        sy_expect = np.max([r.sigma_y for r in recs], axis=0)
        x, y = _curve_xy(dlg._sx_max)
        np.testing.assert_array_equal(x, np.asarray(s, dtype=float))
        np.testing.assert_array_equal(y, sx_expect)
        x, y = _curve_xy(dlg._sy_max)
        np.testing.assert_array_equal(x, np.asarray(s, dtype=float))
        np.testing.assert_array_equal(y, sy_expect)

        # Clearing the study clears the overlay again (no stale curve).
        state.error_study_results = None
        dlg.refresh(None)
        for item in (dlg._sx_max, dlg._sy_max):
            x, y = _curve_xy(item)
            assert x.size == 0 and y.size == 0
    finally:
        dlg.close()
        dlg.deleteLater()
