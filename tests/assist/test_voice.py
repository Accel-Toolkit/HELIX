# tests/assist/test_voice.py
"""Offline voice layer.  The precision-safety summariser and engine
selection are fully tested; macOS ``say`` TTS is exercised for real
(it's built in); mic capture and Whisper are mocked (no hardware / no
model download in the suite)."""
from __future__ import annotations

import shutil
import wave

import numpy as np
import pytest

from linac_gen.assist import voice


# ---- precision safety ------------------------------------------------
def test_summarize_rounds_numbers_and_drops_data_lines():
    text = ("Ran the envelope on BTL.\n"
            "sigma_x at exit is 0.6234 mm and output energy 800.000 MeV.\n"
            "| s | sigma_x | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |\n"
            "```\nraw = [1,2,3,4,5]\n```")
    spoken = voice.summarize_for_speech(text)
    # the exact 0.6234 is NOT spoken verbatim — coarsened
    assert "0.6234" not in spoken
    assert "about 0.62" in spoken
    assert "800" in spoken                      # integer stays
    # the table and code rows are dropped
    assert "raw =" not in spoken and "| s |" not in spoken
    assert "envelope" in spoken.lower()


def test_summarize_truncates_and_marks_screen():
    spoken = voice.summarize_for_speech("word " * 400, max_chars=100)
    assert len(spoken) <= 140
    assert "on screen" in spoken


def test_summarize_empty():
    assert voice.summarize_for_speech("") == ""
    assert voice.summarize_for_speech(None) == ""


# ---- engine selection + availability --------------------------------
def test_tts_engine_selection_prefers_piper_then_say(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_piper(name, *a, **k):
        if name == "piper":
            raise ImportError("no piper")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_piper)
    # with piper absent, macOS 'say' is chosen when present
    if shutil.which("say"):
        assert voice._pick_tts_engine() == "say"
    # status string is descriptive either way
    assert "speech-out" in voice.voice_status()


@pytest.mark.skipif(not shutil.which("say"),
                    reason="macOS 'say' not present")
def test_macos_say_produces_audio(tmp_path):
    """The zero-dependency offline TTS actually renders speech."""
    import subprocess
    out = tmp_path / "hello.aiff"
    subprocess.run(["say", "-o", str(out),
                    "sigma x is about zero point six millimetres"],
                   check=True, timeout=30)
    assert out.exists() and out.stat().st_size > 1000


# ---- STT orchestration (Whisper mocked — no model download) ---------
def test_whisper_transcribe_joins_segments(monkeypatch):
    class _Seg:
        def __init__(self, t): self.text = t

    class _FakeModel:
        def transcribe(self, audio, **k):
            assert audio.dtype == np.float32
            return ([_Seg(" run the "), _Seg("envelope ")], object())

    stt = voice.WhisperSTT()
    monkeypatch.setattr(stt, "_model", lambda: _FakeModel())
    text = stt.transcribe(np.ones(16000, dtype=np.float32))
    assert text == "run the envelope"


def test_whisper_empty_audio_no_model_call():
    stt = voice.WhisperSTT()
    # empty audio short-circuits before any model load
    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""


# ---- recorder (sounddevice mocked — no live mic) --------------------
def test_push_to_talk_recorder_uses_blocking_read_not_callback(monkeypatch):
    """Regression: the recorder must use blocking stream.read() and must
    NOT install a Python callback (the RT-thread callback UAF crashed the
    whole GUI — macOS SIGSEGV in general_invoke_callback)."""
    import sys
    import types

    captured = {}
    chunks = [np.ones((100, 1), np.float32), np.ones((50, 1), np.float32)]

    class _Stream:
        def __init__(self, **kw):
            captured["kwargs"] = kw
            captured["sr"] = kw["samplerate"]
            self._i = 0

        def start(self): ...

        def read(self, n):
            if self._i >= len(chunks):
                raise RuntimeError("stream drained")   # ends the loop
            d = chunks[self._i]
            self._i += 1
            return d, False

        def stop(self): ...
        def close(self): ...

    fake_sd = types.SimpleNamespace(InputStream=_Stream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    rec = voice.PushToTalkRecorder(samplerate=16000)
    rec.start()
    rec._thread.join(timeout=2.0)          # reader drains both chunks
    audio = rec.stop()
    assert audio.shape == (150,)
    assert captured["sr"] == 16000
    # the crash-causing kwarg must never be passed
    assert "callback" not in captured["kwargs"]


def test_recorder_stop_without_start_is_safe():
    rec = voice.PushToTalkRecorder()
    assert rec.stop().shape == (0,)


# ---- streaming Speaker (Phase 6) -------------------------------------
def test_speaker_queues_sentences_and_barges_in():
    import time
    from linac_gen.assist import voice as V

    spoken, events = [], []

    class _SlowBackend:
        def speak(self, text):
            spoken.append(text)
            time.sleep(0.05)
        def cut(self):
            events.append("cut")

    sp = V.Speaker.__new__(V.Speaker)          # skip backend autodetect
    sp.on_speaking = events.append
    sp.backend, sp.backend_name = _SlowBackend(), "test"
    import queue as q, threading
    sp._q = q.Queue(); sp._muted = False; sp._gen = 0
    sp._state_lock = threading.Lock()
    sp._turn_open = False; sp._reported = False; sp._playing = False
    threading.Thread(target=sp._player, daemon=True).start()

    sp.say("First point. Second point! Third?")
    time.sleep(0.3)
    assert spoken == ["First point.", "Second point!", "Third?"]
    assert events[0] is True and events[-1] is False     # speaking toggled
    sp.say("Never spoken. Because muted.")
    sp.set_muted(True)                                    # mute -> stop
    assert "cut" in events
    sp.set_muted(False)
    sp.say("A. B. C.")
    sp.stop()                                             # barge-in drains
    assert sp._q.empty()
    # generation invalidation: a sentence dequeued around stop() is NOT
    # spoken, and post-stop speech still works (regression: barge-in race)
    n_before = len(spoken)
    sp._q.put((sp._gen - 1, "Stale sentence."))           # pre-stop gen
    sp.say("Fresh sentence.")
    time.sleep(0.3)
    assert "Stale sentence." not in spoken
    assert spoken[-1] == "Fresh sentence." and len(spoken) == n_before + 1


def test_speaker_backend_falls_back_without_kokoro(monkeypatch):
    from linac_gen.assist import voice as V
    monkeypatch.setattr(V, "_kokoro_files", lambda: (None, None))
    be, name = V.pick_speaker_backend()
    assert name in ("say", "piper", "pyttsx3", "none")
    assert hasattr(be, "speak") and hasattr(be, "cut")


def test_recorder_on_level_reports_rms(monkeypatch):
    """The blocking-read loop feeds per-block RMS to on_level."""
    import sys, types
    import numpy as np
    from linac_gen.assist import voice as V

    chunks = [np.full((100, 1), 0.5, np.float32)]

    class _Stream:
        def __init__(self, **kw): self._i = 0
        def start(self): ...
        def read(self, n):
            if self._i >= len(chunks):
                raise RuntimeError("drained")
            d = chunks[self._i]; self._i += 1
            return d, False
        def stop(self): ...
        def close(self): ...

    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=_Stream))
    levels = []
    rec = V.PushToTalkRecorder(on_level=levels.append)
    rec.start(); rec._thread.join(timeout=2); rec.stop()
    assert levels and abs(levels[0] - 0.5) < 1e-6


def test_kokoro_playback_uses_persistent_chunked_stream(monkeypatch):
    """Regression (crackling): playback must use ONE persistent
    OutputStream with chunked blocking writes — not sd.play() per
    sentence — and cut() must abort mid-sentence."""
    import sys, types
    import numpy as np
    from linac_gen.assist import voice as V

    streams = []

    class _Out:
        def __init__(self, samplerate, channels, dtype, latency):
            self.samplerate = samplerate
            self.latency = latency
            self.writes = []
            self.aborted = False
            streams.append(self)
        def start(self): ...
        def write(self, data): self.writes.append(len(data))
        def abort(self): self.aborted = True
        def close(self): ...

    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(OutputStream=_Out))

    be = V._KokoroBackend.__new__(V._KokoroBackend)   # skip model load
    import threading
    be._active = threading.Event(); be._cancel = threading.Event()
    be._out = None

    class _K:                       # 1.0 s of audio at 24 kHz
        def create(self, text, voice, speed):
            return np.zeros(24000, dtype="float32"), 24000
    be._k = _K()

    be.speak("one")
    be.speak("two")
    assert len(streams) == 1                       # persistent stream
    assert streams[0].latency == "high"
    assert len(streams[0].writes) == 10            # 2 x 5 chunks of 0.2 s
    assert max(streams[0].writes) <= 24000 * 0.2 + 1

    # cut arriving DURING synthesis -> nothing written for that sentence
    class _KCut:
        def create(self, text, voice, speed):
            be._cancel.set()            # barge-in mid-synthesis
            return np.zeros(24000, dtype="float32"), 24000
    be._k = _KCut()
    n = len(streams[0].writes)
    be.speak("three")
    assert len(streams[0].writes) == n             # synthesis-window cut


# ---- TTS-safe numbers (decimal point audibility) ---------------------
def test_speakable_numbers_makes_decimals_and_notation_audible():
    """Regression: kokoro's G2P drops a bare decimal point, so '21.24'
    was heard as 'twenty one … two four' with no 'point'.  Numbers must
    reach the TTS with the point (and sci-notation / unicode minus)
    spelled out."""
    from linac_gen.assist import voice as V
    assert V.speakable_numbers("21.24") == "21 point 24"
    assert V.speakable_numbers("about 0.62 mm") == "about 0 point 62 mm"
    assert V.speakable_numbers("3.14159") == "3 point 14159"
    assert V.speakable_numbers("v1.2.3") == "v1 point 2 point 3"
    assert V.speakable_numbers("2.1e+04") == "2 point 1 times ten to the 4"
    assert V.speakable_numbers("1e-12") == "1 times ten to the minus 12"
    assert V.speakable_numbers("kb = −1.8") == "kb = minus 1 point 8"
    # prose with the word 'point' and sentence-ending dots is untouched
    assert V.speakable_numbers("First point. Second.") == \
        "First point. Second."
    # idempotent — safe even if applied twice
    once = V.speakable_numbers("0.5")
    assert V.speakable_numbers(once) == once == "0 point 5"


def test_speaker_normalizes_numbers_before_backend():
    """Speaker.say is the choke point: every backend must receive the
    normalized text."""
    import time
    from linac_gen.assist import voice as V

    spoken = []

    class _Backend:
        def speak(self, text):
            spoken.append(text)
        def cut(self):
            pass

    sp = V.Speaker.__new__(V.Speaker)
    sp.on_speaking = lambda active: None
    sp.backend, sp.backend_name = _Backend(), "test"
    import queue as q, threading
    sp._q = q.Queue(); sp._muted = False; sp._gen = 0
    sp._state_lock = threading.Lock()
    sp._turn_open = False; sp._reported = False; sp._playing = False
    threading.Thread(target=sp._player, daemon=True).start()

    sp.say("Sigma x at exit is about 0.62 mm. Energy 800 MeV.")
    time.sleep(0.3)
    # since the MIRAGE-parity port, units are ALSO expanded to words
    # (normalize_units_for_speech runs before speakable_numbers)
    assert spoken == ["Sigma x at exit is about 0 point 62 millimeters.",
                      "Energy 800 M e V."]


# ---- speaking-state machine: turn bracketing (smoothness parity) ----
def _bare_speaker(backend, on_speaking):
    """A Speaker with an injected backend and a live player thread."""
    import queue as q
    import threading
    from linac_gen.assist import voice as V
    sp = V.Speaker.__new__(V.Speaker)
    sp.on_speaking = on_speaking
    sp.backend, sp.backend_name = backend, "test"
    sp._q = q.Queue(); sp._muted = False; sp._gen = 0
    sp._state_lock = threading.Lock()
    sp._turn_open = False; sp._reported = False; sp._playing = False
    threading.Thread(target=sp._player, daemon=True).start()
    return sp


def _wait(cond, timeout=2.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_speaker_turn_bracket_suppresses_sentence_gap_flap():
    """THE smoothness fix: during a streamed reply (begin_turn open) a
    drained queue is a sentence gap — speaking must NOT report False
    (that reset the barge-in bleed calibration, dropped the wake pause
    so the assistant heard itself, and flickered the orb)."""
    import time
    spoken, events = [], []

    class _B:
        def speak(self, text):
            spoken.append(text)
        def cut(self): ...

    sp = _bare_speaker(_B(), events.append)
    sp.begin_turn()
    sp.say("First streamed sentence.")
    assert _wait(lambda: len(spoken) == 1)
    time.sleep(0.1)                       # a real inter-sentence gap
    assert events == [True]               # no False during the gap
    sp.say("Second streamed sentence.")
    assert _wait(lambda: len(spoken) == 2)
    sp.end_turn()
    assert _wait(lambda: events and events[-1] is False)
    assert events == [True, False]        # ONE transition each way


def test_speaker_end_turn_without_speech_is_silent():
    events = []

    class _B:
        def speak(self, text): ...
        def cut(self): ...

    sp = _bare_speaker(_B(), events.append)
    sp.begin_turn()
    sp.end_turn()
    assert events == []                   # never spoke — nothing to report


def test_speaker_stop_between_sentences_reports_stopped():
    """A barge-in stop with sentences still queued must not leave the
    speaking state stuck True (wake would stay paused forever)."""
    import time
    events = []

    class _Slow:
        def speak(self, text):
            time.sleep(0.15)
        def cut(self): ...

    sp = _bare_speaker(_Slow(), events.append)
    sp.say("A. B. C.")
    assert _wait(lambda: events[:1] == [True])
    sp.stop()                             # flushes B/C mid-"A"
    assert _wait(lambda: events and events[-1] is False)
    assert events.count(True) == 1


def test_whisper_default_is_small_en():
    """Parity with MIRAGE: the English-only small model hears wake words
    and commands noticeably better than multilingual base."""
    assert voice.WhisperSTT().model_size == "small.en"


def test_whisper_short_blip_gated_before_model_load(monkeypatch):
    stt = voice.WhisperSTT()

    def _boom():
        raise AssertionError("model must not load for a 0.19 s blip")
    monkeypatch.setattr(stt, "_model", _boom)
    assert stt.transcribe(np.ones(3000, dtype=np.float32)) == ""
