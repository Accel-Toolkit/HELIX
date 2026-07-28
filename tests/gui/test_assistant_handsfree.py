"""Phase-2 hands-free panel wiring: wake toggle, tap-based PTT while the
wake stream owns the mic, local voice confirmation, SPACE push-to-talk,
follow-up after a confirm echo.  No audio hardware — fakes throughout."""
from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from tests.gui.test_assistant_stop_events import _mock_panel  # noqa: E402


class _FakeMicStream:
    """MicStream-shaped: tap registry the test can push into."""

    running = True                # set False to play a dead wake stream

    def __init__(self):
        self.taps = []

    def subscribe(self, fn):
        self.taps.append(fn)

    def unsubscribe(self, fn):
        if fn in self.taps:
            self.taps.remove(fn)

    def push(self, level, n=1):
        for _ in range(n):
            for fn in list(self.taps):
                fn(np.full(160, level, np.float32))

    def close(self):
        pass


def test_voice_confirm_denies_pending_request(qapp, tmp_path):
    from linac_gen.assist.agent import Decision
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    ap = panel._approver
    ap._gen = 1
    ap._pending = object()                 # a request is mid-flight
    panel._confirm_gen = 1
    panel._on_voice_text("no thanks, cancel that")
    assert ap._event.is_set()
    assert ap._decision is Decision.DENY
    assert "denied" in panel._transcript.toPlainText()
    panel.close()


def test_voice_confirm_approves_pending_request(qapp, tmp_path):
    from linac_gen.assist.agent import Decision
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    ap = panel._approver
    ap._gen = 2
    ap._pending = object()
    panel._confirm_gen = 2
    panel._on_voice_text("yes, go ahead")
    assert ap._event.is_set()
    assert ap._decision is Decision.APPROVE
    panel.close()


def test_voice_text_without_pending_autosends_as_voice_turn(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("done")])
    panel._on_voice_text("what is the beam size")
    for _ in range(400):
        qapp.processEvents()
        w = panel._worker
        if w is not None and not w.isRunning():
            w.wait(2000)
            break
        time.sleep(0.01)
    qapp.processEvents()
    assert "▶ what is the beam size" in panel._transcript.toPlainText()
    assert panel._last_turn_was_voice      # earns a follow-up window
    panel.close()


def test_typed_send_is_not_a_voice_turn(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("done")])
    panel._last_turn_was_voice = True      # stale from a previous turn
    panel._input.setText("typed question")
    panel._send()
    assert not panel._last_turn_was_voice
    for _ in range(400):
        qapp.processEvents()
        w = panel._worker
        if w is not None and not w.isRunning():
            w.wait(2000)
            break
        time.sleep(0.01)
    panel.close()


def test_ptt_uses_capture_tap_while_wake_stream_owns_mic(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    mic = _FakeMicStream()
    panel._mic_stream = mic
    grabbed = {}
    panel._transcribe = lambda audio: grabbed.__setitem__("audio", audio)
    panel._mic_pressed()
    assert panel._capture_tap is not None  # tap, not a second stream
    assert not panel._wake_btn.isEnabled()  # ownership frozen mid-hold
    mic.push(0.4, n=3)
    panel._mic_released()
    assert panel._capture_tap is None
    assert panel._wake_btn.isEnabled()
    assert grabbed["audio"].shape == (3 * 160,)
    panel._mic_stream = None
    panel.close()


def test_space_key_drives_ptt(qapp, tmp_path):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    calls = []
    panel._mic_pressed = lambda: calls.append("press")
    panel._mic_released = lambda: calls.append("release")
    # This test pins the space-bar ROUTING; audio never runs (the mic
    # slots are stubbed above).  Without the voice extras installed the
    # button is correctly disabled and the guard would mask the routing.
    panel._mic_btn.setEnabled(True)
    panel._send_btn.setFocus()             # input line NOT focused
    down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space,
                     Qt.KeyboardModifier.NoModifier)
    panel.keyPressEvent(down)
    assert calls == ["press"]
    panel._recorder = object()             # simulate active recording
    up = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space,
                   Qt.KeyboardModifier.NoModifier)
    panel.keyReleaseEvent(up)
    assert calls == ["press", "release"]
    panel._recorder = None
    panel.close()


def test_space_in_input_line_types_normally(qapp, tmp_path):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel.show()
    qapp.processEvents()
    calls = []
    panel._mic_pressed = lambda: calls.append("press")
    panel._input.setFocus()
    qapp.processEvents()
    down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space,
                     Qt.KeyboardModifier.NoModifier)
    panel.keyPressEvent(down)
    assert calls == []                     # typing a space, not PTT
    panel.close()


def test_confirm_echo_opens_followup_with_mic_armed(qapp, tmp_path):
    """A pending confirmation with the wake mic armed opens a follow-up
    window so a bare spoken 'yes' works hands-free."""
    from linac_gen.assist.agent import ConfirmationRequest
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._mic_stream = _FakeMicStream()

    class _FakeFollowup:
        def __init__(self):
            self.opened = 0
            self.active = False

        def open_window(self):
            self.opened += 1
            return True

        def cancel(self):
            pass

    fl = _FakeFollowup()
    panel._followup = fl
    req = ConfirmationRequest(tool="run_mp", tier="compute", params={},
                              pretty="run_mp()", allow_session_auto=True)
    panel._show_confirmation(req)
    assert fl.opened == 1                  # speech off → opens immediately
    assert panel._last_turn_was_voice
    panel._mic_stream = None
    panel._followup = None
    panel.close()


def test_wake_button_disabled_without_stt(qapp, tmp_path, monkeypatch):
    import linac_gen.assist.voice as V
    from linac_gen.assist.testing import turn_text
    monkeypatch.setattr(V, "stt_available", lambda: False)
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._init_voice()
    assert not panel._wake_btn.isEnabled()
    panel.close()


def test_stt_prewarms_at_panel_open(qapp, tmp_path, monkeypatch):
    """MIRAGE-parity latency fix: the Whisper model warms when the
    panel opens, not on first use (a cold load made the first
    push-to-talk take ~15 s)."""
    import linac_gen.assist.voice as V
    from linac_gen.assist.testing import turn_text

    built = []

    class _FakeStt:
        ready = False

        def __init__(self, *a, **k):
            built.append(self)

        def transcribe(self, audio, samplerate=16000):
            return ""

    monkeypatch.setattr(V, "WhisperSTT", _FakeStt)
    monkeypatch.setattr(V, "stt_available", lambda: True)
    monkeypatch.setenv("HELIX_ASSIST_NO_PREWARM", "0")
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._init_voice()
    assert built                              # STT created at open
    assert panel._wake_stt is built[0]
    panel.close()


# ---- dead-stream resilience (user report: "● rec but nothing heard") --
def test_ptt_falls_back_to_recorder_when_wake_stream_dead(
        qapp, tmp_path, monkeypatch):
    """A wake MicStream whose reader died must NOT be tapped (the tap
    would capture nothing, forever) — PTT falls back to a fresh
    PushToTalkRecorder."""
    from linac_gen.assist.testing import turn_text
    from linac_gen.assist import voice as V
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    mic = _FakeMicStream()
    mic.running = False                       # reader died silently
    panel._mic_stream = mic

    made = {}

    class _FakeRec:
        def __init__(self, on_level=None):
            made["rec"] = self

        def start(self):
            made["started"] = True

        def stop(self):
            return np.zeros(0, np.float32)

    monkeypatch.setattr(V, "PushToTalkRecorder", _FakeRec)
    panel._transcribe = lambda audio, recorder=None: None
    panel._mic_pressed()
    assert made.get("started")                # recorder path, not the tap
    assert panel._capture_tap is None
    panel._mic_released()
    panel.close()


def test_mic_died_recovery_cycles_wake_toggle(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._mic_stream = _FakeMicStream()
    calls = []
    panel._wake_btn.setChecked = calls.append      # record the cycle
    panel._on_mic_died()
    assert calls == [False, True]
    assert "microphone stream lost" in panel._transcript.toPlainText()
    # no stream (already shut down) → recovery is a no-op
    calls.clear()
    panel._mic_stream = None
    panel._on_mic_died()
    assert calls == []
    panel.close()


def test_voice_failure_resets_state_to_idle(qapp, tmp_path):
    """Regression: a failed transcription left the orb stuck on
    'thinking' forever — the record cue that never went away."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    states = []
    panel._set_state = states.append
    panel._on_voice_failed("transcription failed: boom")
    assert states == ["idle"]
    assert "boom" in panel._transcript.toPlainText()
    panel.close()
