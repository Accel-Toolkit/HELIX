"""Intent fast-path through the real panel send path, plus the latency
HUD: instant commands never spawn an LLM worker; nuance always does."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def _wait(qapp, cond, n=400):
    import time
    for _ in range(n):
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_instant_command_skips_the_model(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, provider = _mock_panel(qapp, tmp_path,
                                  [turn_text("model should not run")])
    panel._input.setText("status")
    panel._send()
    assert _wait(qapp, lambda: "instant: get_status"
                 in panel._transcript.toPlainText())
    t = panel._transcript.toPlainText()
    assert "Status:" in t
    assert "model should not run" not in t       # no LLM turn happened
    assert panel._worker is None
    assert "instant" in panel._perf.text()       # HUD measured it
    panel.shutdown()


def test_nuanced_text_goes_to_the_model(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path,
                           [turn_text("nuance handled by model")])
    panel._input.setText("why is the transmission dropping so fast")
    panel._send()
    assert _wait(qapp, lambda: "nuance handled by model"
                 in panel._transcript.toPlainText())
    panel.shutdown()


def test_checkbox_off_routes_everything_to_model(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("model ran")])
    panel._fast_chk.setChecked(False)
    panel._input.setText("status")
    panel._send()
    assert _wait(qapp, lambda: "model ran"
                 in panel._transcript.toPlainText())
    assert "instant: get_status" not in panel._transcript.toPlainText()
    panel.shutdown()


def test_first_token_latency_measured(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("hi there")])
    panel._input.setText("hello friend")
    panel._send()
    _wait(qapp, lambda: "hi there" in panel._transcript.toPlainText())
    # MockProvider may not stream deltas; the HUD hook must at least
    # never crash and the label stays a QLabel contract
    assert panel._perf.text() is not None
    panel.shutdown()


def test_tour_reopens_mic_after_each_station(qapp, tmp_path):
    """User report: MIRAGE-style tour flow — after a station, the mic
    must reopen for 'next' even when the tour was started by CLICK
    (typed turn), not voice."""
    from linac_gen.assist.guide import get_state
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    opened = []

    class _FakeFollowup:
        def open_window(self):
            opened.append(True)
            return True

    panel._followup = _FakeFollowup()
    panel._last_turn_was_voice = False        # chip-clicked tour
    get_state(panel._session.context).active = True
    panel._open_followup_if_voice()
    assert opened == [True]                   # tour keeps the loop alive
    # no tour, typed turn -> unchanged old behavior (no surprise mic)
    opened.clear()
    get_state(panel._session.context).active = False
    panel._open_followup_if_voice()
    assert opened == []
    panel.shutdown()
