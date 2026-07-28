"""Offscreen test of the assistant panel's push-to-talk voice wiring —
audio engines mocked (no mic, no model download, no sound)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _state(qapp):
    from linac_gen_gui.interphase.state import AppState
    return AppState()


def test_mic_press_release_transcribes_and_fills_input(
        qapp, monkeypatch):
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap

    # fake recorder: start/stop yields a fixed audio buffer
    class _Rec:
        def __init__(self, *a, **k): ...
        def start(self): self._started = True
        def stop(self): return np.ones(1600, dtype=np.float32)

    monkeypatch.setattr(ap, "_TranscribeWorker", _ImmediateTranscribe)
    monkeypatch.setattr(
        "linac_gen.assist.voice.PushToTalkRecorder", _Rec)

    panel = ap.AssistantPanel(None, _state(qapp))
    try:
        panel._session = None            # no LLM needed for this test
        panel._mic_pressed()
        assert panel._mic_btn.text() == "● rec"
        panel._mic_released()
        for _ in range(50):
            qapp.processEvents()
            if panel._input.text():
                break
        # transcription landed in the input box for user review
        assert panel._input.text() == "run the envelope"
        assert panel._mic_btn.text() == "🎤 Hold"
    finally:
        panel.close()


def test_speak_replies_summarises_not_verbatim(qapp, monkeypatch):
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap

    spoken = {}

    class _Speaker:
        backend_name = "test"
        def say(self, text): spoken["said"] = text
        def stop(self): ...

    panel = ap.AssistantPanel(None, _state(qapp))
    try:
        panel._speaker = _Speaker()          # inject: bypass backend pick
        panel._speak.setChecked(True)
        panel._maybe_speak(
            "Ran the envelope. sigma_x at exit is 0.6234 mm.")
        assert "said" in spoken
        # the exact figure is coarsened, not read verbatim
        assert "0.6234" not in spoken["said"]
        assert "about 0.62" in spoken["said"]
    finally:
        panel.close()


def test_voice_absent_disables_controls(qapp, monkeypatch):
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap

    monkeypatch.setattr("linac_gen.assist.voice.stt_available",
                        lambda: False)
    monkeypatch.setattr("linac_gen.assist.voice.tts_available",
                        lambda: False)
    panel = ap.AssistantPanel(None, _state(qapp))
    try:
        assert not panel._mic_btn.isEnabled()
        assert not panel._speak.isEnabled()
    finally:
        panel.close()


class _ImmediateTranscribe:
    """Stand-in QThread that emits a fixed transcript synchronously."""
    def __init__(self, audio, parent=None, recorder=None):
        from PyQt6.QtCore import QObject, pyqtSignal

        class _S(QObject):
            transcribed = pyqtSignal(str)
            failed = pyqtSignal(str)
            too_short = pyqtSignal(float)
        self._s = _S()
        self.transcribed = self._s.transcribed
        self.failed = self._s.failed
        self.too_short = self._s.too_short
        self.recorder = recorder

    def start(self):
        if self.recorder is not None:            # release path: stop here
            self.recorder.stop()
        self.transcribed.emit("run the envelope")

    def isRunning(self):
        return False

    def wait(self, ms):
        return True


def test_speak_as_it_goes_sentences_and_no_double_speak(
        qapp, monkeypatch, tmp_path):
    """Streaming replies are spoken sentence-by-sentence (coarsened), the
    remainder flushes at stream end, and end-of-turn does NOT re-speak."""
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)

    panel = ap.AssistantPanel(None, AppState())
    try:
        said = []

        class _Speaker:
            backend_name = "test"
            def say(self, t): said.append(t)
            def stop(self): ...

        panel._speaker = _Speaker()
        panel._speak.setChecked(True)
        panel._stream_buf = ""; panel._streaming = False
        panel._speak_pend = ""; panel._spoke_streaming = False
        panel._last_reply = ""
        panel._stream_delta("Sigma x is 0.6234 mm. ")
        panel._stream_delta("Energy is 800 MeV. Still going")
        assert len(said) == 2                      # two complete sentences
        assert "about 0.62" in said[0]             # per-sentence coarsening
        panel._stream_done()                       # flush the remainder
        assert said[-1].startswith("Still going")
        assert panel._spoke_streaming is True
        n = len(said)
        panel._on_turn_done()                      # must NOT re-speak
        assert len(said) == n
    finally:
        panel.close()


def test_streamed_code_fences_are_never_spoken(qapp, monkeypatch, tmp_path):
    """Adversarial H2: a code fence spanning streamed sentences must not be
    read aloud — fence state persists across deltas (incl. a split ```)."""
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)
    panel = ap.AssistantPanel(None, AppState())
    try:
        said = []

        class _Speaker:
            backend_name = "test"
            def say(self, t): said.append(t)
            def stop(self): ...

        panel._speaker = _Speaker()
        panel._speak.setChecked(True)
        for attr, v in (("_stream_buf", ""), ("_streaming", False),
                        ("_speak_pend", ""), ("_speak_carry", ""),
                        ("_speak_in_fence", False),
                        ("_spoke_streaming", False), ("_last_reply", "")):
            setattr(panel, attr, v)
        panel._stream_delta("Set it like this: ``")
        panel._stream_delta("`\nsecret_token = 'abc'. gradient = 5.2.\n``")
        panel._stream_delta("` That is all. Done now.")
        panel._stream_done()
        spoken = " ".join(said)
        assert "secret_token" not in spoken          # fenced content silent
        assert "abc" not in spoken
        assert "That is all." in spoken              # prose still spoken
    finally:
        panel.close()


def test_release_never_joins_recorder_on_gui_thread(qapp, monkeypatch):
    """Regression: rec.stop() joins a reader that can sit in a blocking
    read >1 s on a cold device — on the GUI thread that froze the ● rec
    cue.  The stop must run inside the transcribe worker."""
    import threading
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap

    stopped_on = {}

    class _Rec:
        def __init__(self, *a, **k): ...
        def start(self): ...

        def stop(self):
            stopped_on["thread"] = threading.current_thread()
            return np.ones(16000, dtype=np.float32)

    monkeypatch.setattr(
        "linac_gen.assist.voice.PushToTalkRecorder", _Rec)

    class _CaptureStt:
        def __init__(self, *a, **k): ...
        def transcribe(self, audio): return "ok"
    monkeypatch.setattr("linac_gen.assist.voice.WhisperSTT", _CaptureStt)

    panel = ap.AssistantPanel(None, _state(qapp))
    try:
        panel._session = None
        panel._mic_pressed()
        panel._mic_released()                    # must NOT stop inline
        assert "thread" not in stopped_on        # GUI thread untouched
        w = panel._voice_worker
        for _ in range(300):
            qapp.processEvents()
            if w is not None and not w.isRunning():
                w.wait(2000)
                break
        assert stopped_on["thread"] is not threading.main_thread()
    finally:
        panel.close()


def test_quick_click_reports_too_short_not_nothing_heard(qapp, monkeypatch):
    """A sub-0.3 s click must say so honestly (the STT gate would return
    '' and the old message blamed silence) and return to idle."""
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap

    class _Rec:
        def __init__(self, *a, **k): ...
        def start(self): ...
        def stop(self):
            return np.ones(1600, dtype=np.float32)   # 0.1 s click

    monkeypatch.setattr(
        "linac_gen.assist.voice.PushToTalkRecorder", _Rec)

    def _no_stt(*a, **k):
        raise AssertionError("Whisper must not run for a 0.1 s blip")
    monkeypatch.setattr("linac_gen.assist.voice.WhisperSTT", _no_stt)

    panel = ap.AssistantPanel(None, _state(qapp))
    try:
        panel._session = None
        panel._mic_pressed()
        panel._mic_released()
        w = panel._voice_worker
        for _ in range(300):
            qapp.processEvents()
            if w is not None and not w.isRunning():
                w.wait(2000)
                break
        qapp.processEvents()
        assert "too short" in panel._transcript.toPlainText()
        assert "nothing heard" not in panel._transcript.toPlainText()
    finally:
        panel.close()
