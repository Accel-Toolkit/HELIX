"""Live match preview: per-popup opt-in streaming of matcher iterates.

End-of-run popup sync is always-on and untouched; the new channel is the
opt-in 'live match preview' checkbox + ResultsTab.preview_refresh /
end_preview + the worker's 1 Hz throttle.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.results_tab import (ResultsTab,
                                                       _CentroidPopup)

COMMITTED = SimpleNamespace(s=np.array([0.0, 1.0]),
                            centroid=[np.zeros(6), np.zeros(6)],
                            element_exit_idx=[1])
PREVIEW = SimpleNamespace(s=np.array([0.0, 1.0]),
                          centroid=[np.zeros(6), np.ones(6)],
                          element_exit_idx=[1])


def _dummy_tab(popups):
    """Minimal stand-in for ResultsTab: the preview methods only touch
    self._popups and self.state.results."""
    return SimpleNamespace(_popups=popups,
                           state=SimpleNamespace(results=COMMITTED))


def _popup_with_toggle(qapp, checked, visible):
    dlg = _CentroidPopup(None, SimpleNamespace(lattice=None))
    ResultsTab._install_live_preview_toggle(SimpleNamespace(), dlg)
    assert dlg._live_cb is not None
    dlg._live_cb.setChecked(checked)
    if visible:
        dlg.show()
    calls = []
    dlg.refresh = lambda r: calls.append(r)      # record refreshes
    return dlg, calls


def test_checkbox_injected_default_off(qapp):
    dlg = _CentroidPopup(None, SimpleNamespace(lattice=None))
    ResultsTab._install_live_preview_toggle(SimpleNamespace(), dlg)
    assert dlg._live_cb is not None
    assert not dlg._live_cb.isChecked()          # opt-in per user decision
    assert "live match preview" in dlg._live_cb.text()
    dlg.close()


def test_preview_reaches_only_checked_visible_popups(qapp):
    on_vis, c1 = _popup_with_toggle(qapp, checked=True, visible=True)
    off_vis, c2 = _popup_with_toggle(qapp, checked=False, visible=True)
    on_hid, c3 = _popup_with_toggle(qapp, checked=True, visible=False)
    tab = _dummy_tab({"a": on_vis, "b": off_vis, "c": on_hid})
    ResultsTab.preview_refresh(tab, PREVIEW, 7)
    assert c1 == [PREVIEW]
    assert c2 == [] and c3 == []
    assert "LIVE match iter 7" in on_vis.windowTitle()
    assert on_vis._previewing
    for d in (on_vis, off_vis, on_hid):
        d.close()


def test_end_preview_restores_title_and_committed_results(qapp):
    dlg, calls = _popup_with_toggle(qapp, checked=True, visible=True)
    tab = _dummy_tab({"a": dlg})
    ResultsTab.preview_refresh(tab, PREVIEW, 3)
    ResultsTab.end_preview(tab)
    assert calls == [PREVIEW, COMMITTED]
    assert "LIVE" not in dlg.windowTitle()
    assert "Ctrl+S to save" in dlg.windowTitle()
    assert not dlg._previewing
    dlg.close()


def test_preview_none_results_is_noop(qapp):
    dlg, calls = _popup_with_toggle(qapp, checked=True, visible=True)
    ResultsTab.preview_refresh(_dummy_tab({"a": dlg}), None, 1)
    assert calls == []
    dlg.close()


def test_worker_throttle_one_per_second(qapp):
    from linac_gen_gui.interphase.tabs.matching_tab import _MatchWorker
    w = _MatchWorker(None, None, space_charge=False, max_iter=1)
    got = []
    w.preview_results.connect(lambda r, i: got.append((r, i)))
    assert w._maybe_emit_preview("R1", 1, now=10.0) is True
    assert w._maybe_emit_preview("R2", 2, now=10.5) is False   # throttled
    assert w._maybe_emit_preview("R3", 3, now=11.1) is True
    qapp.processEvents()
    assert [i for _r, i in got] == [1, 3]


def test_sigma0_popups_skip_refresh_while_hidden(qapp):
    from linac_gen_gui.interphase.tabs.results_tab import _PhaseAdvancePopup
    state = AppState()
    dlg = _PhaseAdvancePopup(None, state)
    calls = []
    dlg.refresh = lambda r: calls.append(r)
    dlg._on_results_changed(COMMITTED)           # hidden → skipped
    assert calls == []
    dlg.show()                                    # showEvent refreshes (None)
    calls.clear()
    dlg._on_results_changed(COMMITTED)            # visible → passes through
    assert calls == [COMMITTED]
    dlg.close()
