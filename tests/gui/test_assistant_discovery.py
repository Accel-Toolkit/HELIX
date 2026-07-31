"""Discoverability layer: quick-action chips, welcome capabilities card,
warm-up messaging, mic-permission hint, TTS-backend tooltip.  The
features these surface (tour, drills, run_python) all existed — the
user could not FIND them, which is the bug."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def test_chips_exist_and_send_through_real_path(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("Welcome aboard.")])
    labels = [b.text() for b in panel._chip_btns]
    assert any("Tour" in x for x in labels)
    assert any("Drill" in x for x in labels)
    assert any("Python" in x for x in labels)
    tour_btn = next(b for b in panel._chip_btns if "Tour" in b.text())
    tour_btn.click()
    t = panel._transcript.toPlainText()
    assert "▶ start the guided tour" in t          # real _send path
    for _ in range(400):
        qapp.processEvents()
        w = panel._worker
        if w is not None and not w.isRunning():
            w.wait(2000)
            break
    panel.shutdown()


def test_warmup_line_present_at_open(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    t = panel._transcript.toPlainText()
    assert "queued, not lost" in t
    panel.shutdown()


def test_welcome_card_shown_once(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    first = panel._transcript.toPlainText()
    welcomed = "guided tour" in first and "[hello]" in first
    panel.shutdown()
    panel2, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    second = panel2._transcript.toPlainText()
    panel2.shutdown()
    if welcomed:                       # isolated settings: shown once
        assert "[hello]" not in second
    else:                              # settings pre-marked: never shown
        assert "[hello]" not in first


def test_tts_backend_tooltip(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._set_tts_tooltip("kokoro")
    assert "TTS: kokoro" in panel._mic_btn.toolTip()
    panel.shutdown()


def test_mic_failure_prints_permission_hint(qapp, tmp_path, monkeypatch):
    from linac_gen.assist.testing import turn_text
    from tests.gui.test_assistant_stop_events import _mock_panel
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])

    class _Boom:
        def __init__(self, *a, **k): ...
        def start(self):
            raise RuntimeError("PaErrorCode -9986")

    monkeypatch.setattr(
        "linac_gen.assist.voice.PushToTalkRecorder", _Boom)
    panel._mic_pressed()
    # recorder start now runs OFF the GUI thread — the failure arrives
    # via the queued gui_call bridge
    import time as _time
    t0 = _time.time()
    while ("Privacy & Security" not in panel._transcript.toPlainText()
           and _time.time() - t0 < 8):
        qapp.processEvents()
        _time.sleep(0.01)
    t = panel._transcript.toPlainText()
    assert "microphone unavailable" in t
    assert "Privacy & Security" in t
    panel.shutdown()
