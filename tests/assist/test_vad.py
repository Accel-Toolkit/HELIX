"""Silero-VAD gate: mechanical model tests (skipped without the model
file) and fake-VAD behavior tests for every listener that consumes it."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from linac_gen.assist import listen as L
from linac_gen.assist import vad as V


class _FakeVad:
    """Scripted speech probabilities; repeats the last one."""

    def __init__(self, probs):
        self._p = list(probs)

    def prob(self, block):
        return self._p.pop(0) if len(self._p) > 1 else self._p[0]


# ---- mechanical (real ONNX model when present) ------------------------
needs_model = pytest.mark.skipif(not V.available(),
                                 reason="silero model not installed")


@needs_model
def test_model_loads_and_scores_silence_low():
    vad = V.SileroVAD()
    p = vad.prob(np.zeros(3200, np.float32))
    assert 0.0 <= p < 0.2
    noise = (0.03 * np.random.default_rng(1)
             .standard_normal(3200)).astype(np.float32)
    assert 0.0 <= vad.prob(noise) <= 1.0
    vad.reset()
    assert 0.0 <= vad.prob(np.zeros(400, np.float32)) <= 1.0  # short pad


def test_make_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("HELIX_VAD", "0")
    assert V.make() is None
    assert not V.available()


def test_prob_never_raises():
    vad = V.SileroVAD.__new__(V.SileroVAD)      # broken instance
    assert vad.prob(np.ones(512, np.float32)) == 0.0


# ---- barge-in: loud AND speech ---------------------------------------
def test_barge_requires_speech_not_just_loudness():
    fired = threading.Event()
    b = L.BargeListener(on_barge=fired.set, calib_blocks=2,
                        sustain_blocks=2, bleed_ratio=2.5,
                        abs_floor=0.02, vad=_FakeVad([0.1]))
    b.arm()
    blk = lambda v: np.full(160, v, np.float32)   # noqa: E731
    for _ in range(2):
        b.feed(blk(0.05))                        # calibration
    b.feed(blk(0.5))                             # LOUD but not speech
    b.feed(blk(0.5))
    time.sleep(0.05)
    assert not fired.is_set()                    # a dropped mug: ignored


def test_barge_fires_on_loud_speech():
    fired = threading.Event()
    b = L.BargeListener(on_barge=fired.set, calib_blocks=2,
                        sustain_blocks=2, bleed_ratio=2.5,
                        abs_floor=0.02, vad=_FakeVad([0.9]))
    b.arm()
    blk = lambda v: np.full(160, v, np.float32)   # noqa: E731
    for _ in range(2):
        b.feed(blk(0.05))
    b.feed(blk(0.5))
    b.feed(blk(0.5))
    assert fired.wait(1.0)


# ---- follow-up: VAD decides onset/endpoint ---------------------------
def test_followup_vad_onset_and_endpoint():
    from tests.assist.test_listen import _FakeSource, _FakeStt
    src = _FakeSource()
    out = {"text": None}
    fl = L.FollowUpListener(
        _FakeStt(["adjust the last quad"]),
        on_text=lambda t: out.__setitem__("text", t),
        source_factory=lambda: src,
        window_s=2.0, endpoint_s=0.1, max_s=3.0, poll_s=0.01,
        vad=_FakeVad([0.9, 0.9, 0.9, 0.05]))
    fl.TAIL_DISCARD_S = 0.05     # keep the timing script fast (the
    fl.MIN_SPEECH_S = 0.0        # scripted blocks are milliseconds long
    assert fl.open_window()      # production 0.5 s echo guard is not
    time.sleep(0.3)              # under test here)
    src.push(0.01, n=3)          # QUIET blocks — but VAD says speech
    time.sleep(0.05)
    src.push(0.01, n=1)          # VAD now says silence → endpoint
    for _ in range(200):
        if out["text"] is not None:
            break
        time.sleep(0.01)
    assert out["text"] == "adjust the last quad"


# ---- wake: VAD gates STT and capture endpointing ---------------------
def test_wake_vad_gates_transcription(monkeypatch):
    monkeypatch.setattr(L, "BLOCK", 160)
    from tests.assist.test_listen import _FakeMic, _FakeStt

    class _Ctl:
        value = 0.0
        def prob(self, block):
            return self.value

    mic = _FakeMic()
    # extra script entries keep the test self-healing if a starved runner
    # lets a capture window expire empty and re-poll the wake window
    stt = _FakeStt(["helix", "open the results tab"] * 10)
    got = {"cmd": None}
    vad = _Ctl()
    wl = L.WakeListener(
        mic, stt, on_command=lambda t: got.__setitem__("cmd", t),
        is_paused=lambda: False, chime_fn=lambda: None, vad=vad)
    wl.BACKSTOP_S = 99.0     # this test pins the VAD gate, not the backstop
    wl.POLL_S = 0.03
    wl.start()
    try:
        for _ in range(130):                    # LOUD blocks, VAD low —
            mic.push(0.3, n=1)                  # window fills (>1.2 s)
            time.sleep(0.005)
        assert stt.calls == []                  # loudness alone: no STT
        # --- wake: utterance cycles (speech then a real offset) until the
        #     wake window transcribes.  Long monotonic phases — short
        #     alternation can alias against a starved listener thread.
        t0 = time.time()
        while not stt.calls and time.time() - t0 < 20.0:
            vad.value = 0.9
            for _ in range(120):                # ~1.4 s of speech
                if stt.calls:
                    break
                mic.push(0.3, n=1)
                time.sleep(0.01)
            vad.value = 0.02
            tq = time.time()
            while not stt.calls and time.time() - tq < 1.5:
                mic.push(0.01, n=1)             # offset fires the window
                time.sleep(0.01)
        assert stt.calls, "wake window never transcribed"
        # --- capture: latch 'heard' with sustained speech, then sustained
        #     quiet until the endpoint (capture's 10 s cap transcribes even
        #     on timeout once speech was heard, so this cannot hang).
        vad.value = 0.9
        t1 = time.time()
        while time.time() - t1 < 2.0:
            mic.push(0.3, n=1)
            time.sleep(0.01)
        vad.value = 0.02
        t2 = time.time()
        while got["cmd"] is None and time.time() - t2 < 15.0:
            mic.push(0.01, n=1)                 # endpoint on VAD silence
            time.sleep(0.01)
        assert got["cmd"] == "open the results tab"
    finally:
        wl.shutdown()


def test_wake_event_driven_fires_on_utterance_end(monkeypatch):
    """With VAD, Whisper runs when the utterance ENDS — never on a
    fixed cadence mid-word."""
    monkeypatch.setattr(L, "BLOCK", 160)
    from tests.assist.test_listen import _FakeMic, _FakeStt

    class _Ctl:
        value = 0.0
        def prob(self, block):
            return self.value

    mic = _FakeMic()
    stt = _FakeStt(["nothing relevant"])
    vad = _Ctl()
    wl = L.WakeListener(mic, stt, on_command=lambda t: None,
                        is_paused=lambda: False, chime_fn=lambda: None,
                        vad=vad)
    wl.BACKSTOP_S = 99.0     # event-trigger semantics, not the backstop
    wl.start()
    try:
        # STATE-driven phases (not wall-clock): starved CI runners can
        # stall the listener thread arbitrarily, so we push speech until
        # the listener has ACTUALLY accumulated >=0.4 s of utterance
        # (well under the 2.5 s monologue cap), then flip to silence.
        vad.value = 0.9                          # speaking …
        t0 = time.time()
        while getattr(wl, "_vad_speech_s", 0.0) < 0.4 \
                and time.time() - t0 < 30.0:
            mic.push(0.3, n=1)
            time.sleep(0.01)
        assert getattr(wl, "_vad_speech_s", 0.0) >= 0.4, \
            "listener never registered the utterance"
        assert stt.calls == []                   # no STT mid-speech
        vad.value = 0.02                         # … utterance ends
        t0 = time.time()
        while not stt.calls and time.time() - t0 < 30.0:
            mic.push(0.01, n=1)
            time.sleep(0.01)
        assert len(stt.calls) >= 1               # fired on the offset
    finally:
        wl.shutdown()


def test_wake_event_driven_long_monologue_still_checked(monkeypatch):
    """Continuous speech >2.5 s transcribes anyway — 'HELIX' dropped
    mid-sentence must not be missed."""
    monkeypatch.setattr(L, "BLOCK", 160)
    from tests.assist.test_listen import _FakeMic, _FakeStt

    class _Ctl:
        value = 0.9                              # never stops talking
        def prob(self, block):
            return self.value

    mic = _FakeMic()
    stt = _FakeStt(["still going"])
    wl = L.WakeListener(mic, stt, on_command=lambda t: None,
                        is_paused=lambda: False, chime_fn=lambda: None,
                        vad=_Ctl())
    wl.start()
    try:
        # STATE-driven: keep feeding speech until the listener has
        # ACTUALLY accumulated the 2.5 s monologue cap (a starved CI
        # runner may credit far less than wall-clock time), then give
        # the trigger a moment to fire.
        t0 = time.time()
        while not stt.calls \
                and getattr(wl, "_vad_speech_s", 0.0) < 2.6 \
                and time.time() - t0 < 60.0:
            mic.push(0.3, n=1)
            time.sleep(0.01)
        t0 = time.time()
        while not stt.calls and time.time() - t0 < 30.0:
            mic.push(0.3, n=1)
            time.sleep(0.01)
        assert len(stt.calls) >= 1               # the 2.5 s cap fired
    finally:
        wl.shutdown()


def test_followup_fresh_stream_keeps_first_words():
    """User report: 'it doesn't wait for me to speak'.  A TRANSIENT
    follow-up source has no TTS bleed — the old 0.25 s discard was
    eating the start of the sentence.  owns_stream sources keep frame
    one."""
    from tests.assist.test_listen import _FakeSource, _FakeStt

    class _Owning(_FakeSource):
        owns_stream = True

    src = _Owning()
    out = {"text": None}
    fl = L.FollowUpListener(
        _FakeStt(["yes do it"]),
        on_text=lambda t: out.__setitem__("text", t),
        source_factory=lambda: src,
        window_s=2.0, endpoint_s=0.1, max_s=3.0, poll_s=0.01)
    fl.MIN_SPEECH_S = 0.0     # scripted blocks are milliseconds long
    assert fl.open_window()
    time.sleep(0.05)                  # IMMEDIATELY — inside the old
    src.push(0.2, n=4)                # 0.25 s discard window
    time.sleep(0.05)
    src.push(0.001, n=2)
    for _ in range(200):
        if out["text"] is not None:
            break
        time.sleep(0.01)
    assert out["text"] == "yes do it"     # first words survived


def test_wake_hysteresis_keeps_flappy_speech_in_one_utterance(monkeypatch):
    monkeypatch.setattr(L, "BLOCK", 160)
    from tests.assist.test_listen import _FakeMic, _FakeStt

    class _Ctl:
        value = 0.0
        def prob(self, block):
            return self.value

    mic = _FakeMic()
    stt = _FakeStt(["nothing"])
    vad = _Ctl()
    wl = L.WakeListener(mic, stt, on_command=lambda t: None,
                        is_paused=lambda: False, chime_fn=lambda: None,
                        vad=vad)
    wl.start()
    try:
        # STATE-driven phases — see the event-driven test for why.
        vad.value = 0.6                       # onset (>0.45)
        t0 = time.time()
        while getattr(wl, "_vad_speech_s", 0.0) < 0.35 \
                and time.time() - t0 < 30.0:
            mic.push(0.3, n=1)
            time.sleep(0.01)
        mark = getattr(wl, "_vad_speech_s", 0.0)
        assert mark >= 0.35, "listener never registered the onset"
        vad.value = 0.35                      # flappy but >0.30: STAY
        t0 = time.time()
        while getattr(wl, "_vad_speech_s", 0.0) < mark + 0.3 \
                and time.time() - t0 < 30.0:
            mic.push(0.3, n=1)
            time.sleep(0.01)
        assert getattr(wl, "_vad_in_speech", False), \
            "hysteresis dropped the utterance at p=0.35"
        assert stt.calls == []                # still one utterance
        vad.value = 0.05                      # real offset
        t0 = time.time()
        while not stt.calls and time.time() - t0 < 30.0:
            mic.push(0.01, n=1)
            time.sleep(0.01)
        assert len(stt.calls) >= 1
    finally:
        wl.shutdown()


def test_wake_loudness_backstop_when_vad_never_triggers(monkeypatch):
    """Event mode must never hear WORSE than the legacy cadence: a loud
    room where VAD stays low still gets transcription checks."""
    monkeypatch.setattr(L, "BLOCK", 160)
    from tests.assist.test_listen import _FakeMic, _FakeStt

    class _Deaf:
        def prob(self, block):
            return 0.0                        # VAD never fires

    mic = _FakeMic()
    stt = _FakeStt(["helix", "status please"])
    got = {"cmd": None}
    wl = L.WakeListener(mic, stt, on_command=lambda t: got.__setitem__("cmd", t),
                        is_paused=lambda: False, chime_fn=lambda: None,
                        vad=_Deaf())
    wl.start()
    try:
        t0 = time.time()
        while len(stt.calls) < 1 and time.time() - t0 < 20.0:
            mic.push(0.3, n=2)                # LOUD blocks (starved-CI safe)
            time.sleep(0.01)
        assert len(stt.calls) >= 1            # backstop transcribed
    finally:
        wl.shutdown()


@needs_model
@pytest.mark.skipif(not __import__("shutil").which("say"),
                    reason="macOS say not present")
def test_real_speech_scores_high(tmp_path):
    """THE anchor that was missing: the deaf-VAD bug (silero context
    carry omitted) passed silence-scores-low tests while real speech
    scored ~0.  Generate REAL speech and require the model to hear it."""
    import subprocess
    import wave
    wav = tmp_path / "helix.wav"
    subprocess.run(["say", "-o", str(wav), "--data-format=LEI16@16000",
                    "hey helix what is the beam status"],
                   check=True, timeout=30)
    with wave.open(str(wav), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    audio = (np.frombuffer(raw, dtype=np.int16)
             .astype(np.float32) / 32768.0)
    assert sr == 16000
    vad = V.SileroVAD()
    probs = [vad.prob(audio[i:i + 3200])
             for i in range(0, audio.size - 3200, 3200)]
    assert max(probs) > 0.6, f"deaf VAD: max prob {max(probs):.3f}"
    # and silence after a reset stays low
    vad.reset()
    assert vad.prob(np.zeros(3200, np.float32)) < 0.2


def test_make_refuses_until_selftest_passes(monkeypatch):
    """A deaf VAD must never gate anything again: make() returns None
    (pure MIRAGE RMS mode) unless the model PROVED it hears speech.
    Since 2026-07-28 'pending' is RESOLVED synchronously (marker or a
    real run) instead of refused — refusing it made listener startup a
    race against the async self-test."""
    monkeypatch.setattr(V, "available", lambda: True)
    monkeypatch.setattr(V, "model_path", lambda: "/nonexistent/m.onnx")
    monkeypatch.setattr(V, "_SELFTEST", "pending")

    def fake_selftest_fails():
        V._SELFTEST = "fail"
        return False
    monkeypatch.setattr(V, "self_test", fake_selftest_fails)
    assert V.make() is None                 # pending -> decided -> fail
    monkeypatch.setattr(V, "_SELFTEST", "fail")
    assert V.make() is None
    made = {}
    monkeypatch.setattr(V, "SileroVAD", lambda: made.setdefault("v", object()))
    monkeypatch.setattr(V, "_SELFTEST", "pass")
    assert V.make() is made["v"]


@needs_model
@pytest.mark.skipif(not __import__("shutil").which("say"),
                    reason="macOS say not present")
def test_selftest_passes_with_fixed_wrapper():
    assert V.self_test()
    assert V._SELFTEST == "pass"


@pytest.mark.parametrize("text,expect", [
    ("hillix run the envelope", True),      # fuzzy accent renderings
    ("healex", True),
    ("halix status", True),
    ("headaches", True),                    # empirical STT rendering
    ("Helix's status", True),
    ("check the helipad", False),           # calibrated exclusions
    ("helical lattice", False),
    ("just relax", False),
    ("this helps a lot", False),
    ("", False),
])
def test_fuzzy_wake_matching(text, expect):
    assert L.wake_matches(text) is expect


def test_wake_miss_shows_what_was_heard(monkeypatch):
    monkeypatch.setattr(L, "BLOCK", 160)
    from tests.assist.test_listen import _FakeMic, _FakeStt
    mic = _FakeMic()
    stt = _FakeStt(["hello there computer"])
    status = []
    wl = L.WakeListener(mic, stt, on_command=lambda t: None,
                        is_paused=lambda: False, chime_fn=lambda: None,
                        on_status=status.append)
    wl.POLL_S = 0.05
    wl.start()
    try:
        t0 = time.time()
        while (not any("heard" in x for x in status)
               and time.time() - t0 < 15):     # generous: starved runners
            mic.push(0.3, n=2)
            time.sleep(0.01)
        assert any("hello there computer" in x for x in status)
    finally:
        wl.shutdown()


# --- deterministic make() (2026-07-28 wake regression) -----------------
def test_make_resolves_pending_selftest_synchronously(monkeypatch, tmp_path):
    """A cold make() must never lose a race against the async self-test:
    it resolves 'pending' itself (via _self_test_locked — marker file
    first, else a real run) when the lock is free."""
    from linac_gen.assist import vad as V

    calls = []
    monkeypatch.setattr(V, "available", lambda: True)
    monkeypatch.setattr(V, "model_path", lambda: str(tmp_path / "m.onnx"))
    monkeypatch.setattr(V, "SileroVAD", lambda *a, **k: "vad-instance")

    def fake_locked():
        calls.append(1)
        V._SELFTEST = "pass"
        return True
    monkeypatch.setattr(V, "_self_test_locked", fake_locked)

    monkeypatch.setattr(V, "_SELFTEST", "pending")
    assert V.make() == "vad-instance"       # ran the test itself
    assert calls == [1]


def test_make_never_blocks_on_a_held_lock(monkeypatch, tmp_path):
    """AUDIT F1: a proof already running elsewhere must NOT convoy a
    make() on the main thread — it falls back to None (RMS gates)
    within the 50 ms grace instead of freezing the GUI."""
    import time as _t

    from linac_gen.assist import vad as V

    monkeypatch.setattr(V, "available", lambda: True)
    monkeypatch.setattr(V, "model_path", lambda: str(tmp_path / "m.onnx"))
    monkeypatch.setattr(V, "SileroVAD", lambda *a, **k: "vad-instance")
    monkeypatch.setattr(V, "_SELFTEST", "pending")
    assert V._SELFTEST_LOCK.acquire(timeout=1.0)   # simulate async proof
    try:
        t0 = _t.monotonic()
        assert V.make() is None                    # RMS fallback, no wait
        assert _t.monotonic() - t0 < 0.5
    finally:
        V._SELFTEST_LOCK.release()


def test_selftest_locked_marker_short_circuit(monkeypatch, tmp_path):
    """AUDIT F1 root cause: _self_test_locked used to IGNORE the marker
    and re-run the 1.2 s `say` proof at every launch.  With the marker
    present it must pass instantly and spawn NOTHING."""
    import subprocess as _sp

    from linac_gen.assist import vad as V

    model = tmp_path / "m.onnx"
    monkeypatch.setattr(V, "model_path", lambda: str(model))
    monkeypatch.setattr(V, "available", lambda: True)
    open(str(model) + ".selftest_ok", "w").close()

    def boom(*a, **k):
        raise AssertionError("self-test spawned a subprocess despite "
                             "the marker")
    monkeypatch.setattr(_sp, "run", boom)
    monkeypatch.setattr(V, "_SELFTEST", "pending")
    assert V.self_test() is True
    assert V._SELFTEST == "pass"


def test_selftest_pass_writes_marker(monkeypatch, tmp_path):
    """The marker is only written on the PROVEN pass (the ``say``
    branch) — on say-less machines self_test passes trustingly WITHOUT
    a marker, so this test fakes ``say`` and its wav output to walk the
    proof path deterministically on every platform (the old version
    passed on macOS only by accident of ``say`` existing)."""
    import shutil
    import subprocess
    import wave

    import numpy as np

    from linac_gen.assist import vad as V

    model = tmp_path / "m.onnx"
    monkeypatch.setattr(V, "model_path", lambda: str(model))
    monkeypatch.setattr(V, "available", lambda: True)

    class _FakeVad:
        def prob(self, block):
            return 0.95                     # demonstrably hears
    monkeypatch.setattr(V, "SileroVAD", lambda *a, **k: _FakeVad())
    monkeypatch.setattr(V, "_SELFTEST", "pending")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/say")

    def _fake_say(args, check=False, timeout=None, **_kw):
        out = args[args.index("-o") + 1]
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            rng = np.random.default_rng(0)
            wf.writeframes((rng.normal(0, 0.1, 16000) * 32767)
                           .astype(np.int16).tobytes())
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", _fake_say)

    assert V.self_test() is True
    assert (tmp_path / "m.onnx.selftest_ok").exists()
