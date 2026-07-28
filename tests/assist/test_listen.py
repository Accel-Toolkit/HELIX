"""Hands-free listening (`assist.listen`): wake word, follow-up window,
barge-in, voice confirmation, mic-stream ownership.  No audio hardware:
sources are injected, sounddevice is faked where a stream is exercised.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from linac_gen.assist import listen as L


# ---------------------------------------------------------------------------
# wake regex + confirm words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "helix", "Helix, what's the beam size?", "HELICS", "heelix",
    "he licks", "felix", "helux", "healix", "hey helix", "heliks",
])
def test_wake_regex_accepts_mishearings(text):
    assert L.WAKE_RE.search(text)


@pytest.mark.parametrize("text", [
    "helipad", "helical lattice", "helicopter", "heliport", "hello",
    "the helicity of the beam",
])
def test_wake_regex_rejects_lookalikes(text):
    assert not L.WAKE_RE.search(text)


@pytest.mark.parametrize("text,expect", [
    ("yes", "yes"), ("Yeah, go ahead", "yes"), ("confirm", "yes"),
    ("do it", "yes"), ("approved", "yes"),
    ("no", "no"), ("nope", "no"), ("cancel that", "no"),
    ("don't", "no"), ("abort", "no"),
    ("no, yes I mean no", "no"),           # NO outranks YES
    ("yes... no wait", "no"),
    ("maybe later", None), ("", None), ("what does it do?", None),
])
def test_interpret_confirm(text, expect):
    assert L.interpret_confirm(text) == expect


def test_followup_window_env(monkeypatch):
    monkeypatch.setenv("HELIX_FOLLOWUP_S", "0")
    assert L.followup_window_s() == 0.0
    monkeypatch.setenv("HELIX_FOLLOWUP_S", "4.5")
    assert L.followup_window_s() == 4.5
    monkeypatch.delenv("HELIX_FOLLOWUP_S")
    assert L.followup_window_s() == 10.0
    monkeypatch.setenv("HELIX_BARGE_IN", "0")
    assert not L.barge_in_enabled()


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _FakeSource:
    """subscribe/unsubscribe + start/stop; the test pushes blocks."""

    def __init__(self):
        self.fn = None
        self.started = 0
        self.stopped = 0

    def subscribe(self, fn):
        self.fn = fn

    def unsubscribe(self, fn):
        if self.fn is fn:
            self.fn = None

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def push(self, level, n=1):
        for _ in range(n):
            if self.fn is not None:
                self.fn(np.full(160, level, np.float32))


class _FakeStt:
    ready = True

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(np.asarray(audio).size)
        return self._texts.pop(0) if self._texts else ""


# ---------------------------------------------------------------------------
# FollowUpListener
# ---------------------------------------------------------------------------
def _followup(src, stt, **kw):
    out = {"text": None, "timeout": 0}
    fl = L.FollowUpListener(
        stt, on_text=lambda t: out.__setitem__("text", t),
        source_factory=lambda: src,
        on_timeout=lambda: out.__setitem__("timeout", out["timeout"] + 1),
        window_s=kw.pop("window_s", 0.6), endpoint_s=0.1, max_s=3.0,
        poll_s=0.01, **kw)
    return fl, out


def test_followup_times_out_silently():
    src = _FakeSource()
    fl, out = _followup(src, _FakeStt(["never"]))
    assert fl.open_window()
    time.sleep(0.35)
    src.push(0.001, n=3)                   # room noise only
    time.sleep(1.0)
    assert out["timeout"] == 1
    assert out["text"] is None
    assert not fl.active
    assert src.stopped == 1                # tap released


def test_followup_captures_speech_to_endpoint():
    """Wait-until, never fixed sleeps: shared CI runners (first seen on
    the macOS leg) can stall the listener thread past any fixed timing
    window.  We speak in full utterance cycles — a burst of speech then
    a stretch of quiet — reopening the window if it times out unheard,
    until the endpoint fires or a generous deadline expires."""
    src = _FakeSource()
    fl, out = _followup(src, _FakeStt(["set the quad to five"]))
    assert fl.open_window()
    t_end = time.time() + 15.0
    while out["text"] is None and time.time() < t_end:
        if not fl.active and out["text"] is None:
            fl.open_window()               # window expired unheard
        for _ in range(30):                # ~0.4 s of speech
            src.push(0.2, n=1)
            time.sleep(0.012)
        for _ in range(30):                # ~0.4 s of quiet → endpoint
            if out["text"] is not None:
                break
            src.push(0.001, n=1)
            time.sleep(0.012)
    assert out["text"] == "set the quad to five"
    for _ in range(200):                   # listener winds down
        if not fl.active:
            break
        time.sleep(0.01)
    assert not fl.active


def test_followup_cancel_and_reopen_guard():
    src = _FakeSource()
    fl, out = _followup(src, _FakeStt([]), window_s=5.0)
    assert fl.open_window()
    assert not fl.open_window()            # already listening
    fl.cancel()
    for _ in range(100):
        if not fl.active:
            break
        time.sleep(0.01)
    assert out["timeout"] == 0 and out["text"] is None


def test_followup_disabled_by_zero_window():
    fl, _ = _followup(_FakeSource(), _FakeStt([]), window_s=0.0)
    assert not fl.open_window()


# ---------------------------------------------------------------------------
# BargeListener
# ---------------------------------------------------------------------------
def test_barge_ignores_bleed_fires_on_sustained_speech():
    fired = threading.Event()
    b = L.BargeListener(on_barge=fired.set, calib_blocks=3,
                        sustain_blocks=2, bleed_ratio=2.5, abs_floor=0.02)
    assert b.arm()
    assert not b.arm()                     # already armed
    blk = lambda v: np.full(160, v, np.float32)   # noqa: E731
    for _ in range(3):
        b.feed(blk(0.05))                  # calibration: bleed floor 0.05
    b.feed(blk(0.06))                      # bleed-level — no fire
    b.feed(blk(0.30))                      # one hot block — not sustained
    b.feed(blk(0.04))                      # resets the hot counter
    assert not fired.is_set()
    b.feed(blk(0.30))
    b.feed(blk(0.30))                      # two consecutive → fire
    assert fired.wait(1.0)
    assert not b.active                    # one shot per arm
    fired.clear()
    b.feed(blk(0.5))                       # disarmed: ignored
    time.sleep(0.05)
    assert not fired.is_set()


# ---------------------------------------------------------------------------
# WakeListener (fake mic + fake stt; small blocks for speed)
# ---------------------------------------------------------------------------
class _FakeMic:
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


def test_wake_detects_then_captures_command(monkeypatch):
    monkeypatch.setattr(L, "BLOCK", 160)   # 0.01 s blocks → fast test
    mic = _FakeMic()
    stt = _FakeStt(["hey helix", "run the envelope"])
    got = {"cmd": None, "woke": 0, "chimed": 0}
    wl = L.WakeListener(
        mic, stt, on_command=lambda t: got.__setitem__("cmd", t),
        is_paused=lambda: False,
        on_wake=lambda: got.__setitem__("woke", got["woke"] + 1),
        chime_fn=lambda: got.__setitem__("chimed", got["chimed"] + 1))
    wl.POLL_S = 0.05
    wl.start()
    try:
        # feed loud audio so the rolling window passes the silence gate
        feeder_stop = threading.Event()

        def _feed():
            phase = {"n": 0}
            while not feeder_stop.is_set():
                # loud while "speaking", quiet afterwards so the capture
                # endpoint (1 s of quiet at 0.01 s blocks) is reached
                level = 0.2 if phase["n"] < 260 else 0.0005
                mic.push(level)
                phase["n"] += 1
                time.sleep(0.005)

        th = threading.Thread(target=_feed, daemon=True)
        th.start()
        for _ in range(600):
            if got["cmd"] is not None:
                break
            time.sleep(0.02)
        feeder_stop.set()
        th.join(timeout=2)
    finally:
        wl.shutdown()
    assert got["woke"] == 1
    assert got["chimed"] == 1
    assert got["cmd"] == "run the envelope"


def test_wake_paused_never_transcribes(monkeypatch):
    monkeypatch.setattr(L, "BLOCK", 160)
    mic = _FakeMic()
    stt = _FakeStt(["helix"])
    wl = L.WakeListener(mic, stt, on_command=lambda t: None,
                        is_paused=lambda: True)
    wl.POLL_S = 0.02
    wl.start()
    try:
        mic.push(0.5, n=200)
        time.sleep(0.3)
    finally:
        wl.shutdown()
    assert stt.calls == []                 # paused: no STT, no self-wake


# ---------------------------------------------------------------------------
# MicStream ownership (fake sounddevice)
# ---------------------------------------------------------------------------
def test_micstream_blocking_read_and_tap_fanout(monkeypatch):
    reads = {"n": 0}

    class _Stream:
        def __init__(self, **kw):
            assert "callback" not in kw    # NEVER a PortAudio callback

        def start(self):
            pass

        def read(self, n):
            reads["n"] += 1
            if reads["n"] > 3:
                raise RuntimeError("drained")
            return np.full((n, 1), 0.25, np.float32), False

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=_Stream))
    ms = L.MicStream(blocksize=64)
    seen = []
    ms.subscribe(seen.append)
    ms.open()
    ms._thread.join(timeout=2.0)           # reader drains then closes
    ms.close()
    assert len(seen) == 3
    assert seen[0].ndim == 1 and abs(float(seen[0][0]) - 0.25) < 1e-6


def test_capture_tap_accumulates_between_start_stop():
    mic = _FakeMic()
    tap = L.CaptureTap(mic)
    mic.push(0.9, n=2)                     # before start: ignored
    tap.start()
    mic.push(0.5, n=3)
    audio = tap.stop()
    mic.push(0.5, n=2)                     # after stop: ignored
    assert audio.shape == (3 * 160,)
    assert abs(float(audio[0]) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# speech unit expansion (voice.py, ordering pinned)
# ---------------------------------------------------------------------------
def test_normalize_units_for_speech():
    from linac_gen.assist.voice import normalize_units_for_speech as f
    assert f("sigma is 2.5 mm") == "sigma is 2.5 millimeters"
    assert f("energy 800.6 MeV") == "energy 800.6 M e V"
    assert f("0.21 pi.mm.mrad") == "0.21 pi millimeter milliradian"
    assert f("emit_z 0.06 deg.MeV") == "emit_z 0.06 degree M e V"
    assert f("loss 2.7 %") == "loss 2.7 percent"
    assert f("QF-3 gradient") == "QF 3 gradient"
    assert f("HWR:CAV1 phase") == "HWR CAV1 phase"
    # bare single-letter units are never mangled
    assert f("5 m from the start") == "5 m from the start"


def test_units_expand_before_number_rewrite():
    """Order matters: unit matching must see raw digits.  The full
    Speaker path applies units FIRST, then speakable_numbers."""
    from linac_gen.assist.voice import (
        normalize_units_for_speech, speakable_numbers,
    )
    t = speakable_numbers(normalize_units_for_speech("21.24 mm"))
    assert t == "21 point 24 millimeters"
    # the wrong order would leave 'mm' unexpanded after '21 point 24'
    wrong = normalize_units_for_speech(speakable_numbers("21.24 mm"))
    assert "millimeters" in t
    assert t != wrong or "millimeters" in wrong


# ---------------------------------------------------------------------------
# markdown must never be spoken (the MIRAGE correction, ported)
# ---------------------------------------------------------------------------
def test_strip_markdown_for_speech():
    from linac_gen.assist.voice import strip_markdown_for_speech as f
    assert f("**bold** words") == "bold words"
    assert f("*italic* words") == "italic words"
    assert f("## Header line") == "Header line"
    assert f("- a bullet") == "a bullet"
    assert f("see [the manual](http://x/y) here") == "see the manual here"
    assert f("use `run_mp` now") == "use run_mp now"
    assert "fence" not in f("before\n```\nfence body\n```\nafter")
    # identifiers with underscores are NEVER mangled
    assert f("emit_x and beta_y stay") == "emit_x and beta_y stay"
    for ch in "*#`":
        assert ch not in f("**x** ## y `z` * stray")


def test_speaker_never_says_asterisk(qapp_none=None):
    """End-to-end through Speaker.say: the exact user report — bold
    markdown in a reply must reach the TTS backend clean."""
    import queue as q
    import threading
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
    sp._q = q.Queue()
    sp._muted = False
    sp._gen = 0
    sp._state_lock = threading.Lock()
    sp._turn_open = False
    sp._reported = False
    sp._playing = False
    threading.Thread(target=sp._player, daemon=True).start()
    sp.say("The exit is **0.62 mm** — see `sigma_x` and *note* the "
           "[plot](http://x).")
    time.sleep(0.3)
    joined = " ".join(spoken)
    assert "*" not in joined and "`" not in joined and "[" not in joined
    assert "0 point 62 millimeters" in joined
    assert "sigma_x" in joined


def test_whisper_ready_property(monkeypatch):
    from linac_gen.assist import voice as V
    stt = V.WhisperSTT(model_size="unit-test-model")
    assert not stt.ready
    monkeypatch.setitem(V._MODEL_CACHE,
                        ("unit-test-model", "cpu", "int8"), object())
    assert stt.ready


# ---- dead-stream notification ----------------------------------------
def test_micstream_on_died_fires_on_error_exit_not_on_close(monkeypatch):
    """A device error (sleep/wake, headset change) kills the reader —
    on_died must fire so the GUI can recover; a clean close() must NOT
    fire it."""
    import sys
    import time
    import types

    class _ErrStream:
        def __init__(self, **kw):
            self._n = 0

        def start(self): ...

        def read(self, n):
            self._n += 1
            if self._n > 2:
                raise RuntimeError("device gone")
            return np.zeros((n, 1), np.float32), False

        def stop(self): ...
        def close(self): ...

    class _OkStream(_ErrStream):
        def read(self, n):
            time.sleep(0.005)
            return np.zeros((n, 1), np.float32), False

    died = []
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=_ErrStream))
    ms = L.MicStream(blocksize=64, on_died=lambda: died.append("died"))
    ms.open()
    ms._thread.join(timeout=2.0)
    assert died == ["died"]

    died2 = []
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=_OkStream))
    ms2 = L.MicStream(blocksize=64, on_died=lambda: died2.append("died"))
    ms2.open()
    time.sleep(0.05)
    t = ms2._thread
    ms2.close()
    t.join(timeout=2.0)
    assert died2 == []
