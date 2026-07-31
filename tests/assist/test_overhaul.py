"""2026-07-28 assistant-overhaul regressions (module layer).

Each test pins one fix from the three-agent architecture audit:
wake head-clip, afplay watchdog, TTS prefetch pipeline, derived
Speaker.busy, follow-up stale-deliver guard, mic data watchdog,
atomic open_window, SDK ask() heartbeat, PTT stop-before-start."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from linac_gen.assist import listen as L
from linac_gen.assist import voice as V


# ---------------------------------------------------------------------------
# wake head-clip: post-wake words become the command head
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expect", [
    ("helix open the results tab", "open the results tab"),
    ("Helix, open the tab.", "open the tab"),    # edge punct stripped
    ("please helix", ""),
    ("healix show sigma x", "show sigma x"),          # fuzzy token
    ("no wake word here", ""),
    ("helix ... helix show twiss", "show twiss"),     # last occurrence
    ("", ""),
])
def test_split_after_wake(text, expect):
    assert L.split_after_wake(text) == expect


class _HeadMic:
    def __init__(self):
        self.taps = []
        self.on_died = None

    def subscribe(self, fn):
        self.taps.append(fn)

    def unsubscribe(self, fn):
        if fn in self.taps:
            self.taps.remove(fn)


def test_wake_window_words_are_not_lost(monkeypatch):
    """"helix open the results tab" in ONE breath: the words after the
    token used to vanish (capture started only after transcription).
    They must now arrive as the command — and capture must already be
    ARMED while the wake window transcribes."""
    monkeypatch.setattr(L, "BLOCK", 160)
    wl_box = {}

    class _Stt:
        def transcribe(self, audio):
            # the head-clip contract: by the time Whisper runs on the
            # wake window, the capture buffer is already recording
            assert wl_box["wl"]._capturing is True
            return "helix open the results tab"

    cmds = []
    wl = L.WakeListener(_HeadMic(), _Stt(), on_command=cmds.append,
                        is_paused=lambda: False, chime_fn=lambda: None)
    wl_box["wl"] = wl
    wl._handle_window(np.zeros(1600, np.float32))
    # capture heard nothing (silence): the head alone IS the command,
    # dispatched after the shortened 1.2 s grace
    assert cmds == ["open the results tab"]


def test_capture_armed_keeps_pretranscription_audio():
    """Blocks that landed in the capture buffer while Whisper was busy
    on the wake window must be transcribed, not cleared."""
    heard = {}

    class _Stt:
        def transcribe(self, audio):
            heard["n"] = int(np.asarray(audio).size)
            return "tail words"

    wl = L.WakeListener(_HeadMic(), _Stt(), on_command=lambda t: None,
                        is_paused=lambda: False, chime_fn=lambda: None)
    with wl._lock:
        wl._capturing = True
        for _ in range(4):                       # 4 loud blocks pre-seeded
            wl._cap_buf.append((np.full(160, 0.3, np.float32), None))
    out = wl._capture(armed=True, grace_s=1.0)
    assert out == "tail words"
    assert heard["n"] == 4 * 160                 # nothing was dropped


# ---------------------------------------------------------------------------
# afplay watchdog: a wedged playback can never hold the player forever
# ---------------------------------------------------------------------------
def test_afplay_watchdog_kills_wedged_playback(monkeypatch):
    import subprocess as sp

    class _WedgedProc:
        def __init__(self):
            self.terminated = threading.Event()

        def wait(self, timeout=None):
            if self.terminated.is_set():
                return 0
            raise sp.TimeoutExpired("afplay", timeout)

        def terminate(self):
            self.terminated.set()

        def kill(self):
            self.terminated.set()

    proc = _WedgedProc()
    monkeypatch.setattr(V, "_afplay_available", lambda: True)
    monkeypatch.setattr(V, "_write_wav", lambda *a, **k: None)
    monkeypatch.setattr(V.subprocess, "Popen", lambda *a, **k: proc)

    backend = object.__new__(V._KokoroBackend)
    backend._active = threading.Event()
    backend._cancel = threading.Event()
    flat = np.zeros(1600, np.float32)            # 0.1 s clip → ~2.1 s cap
    t0 = time.monotonic()
    backend.play((flat, 16000))
    took = time.monotonic() - t0
    assert proc.terminated.is_set()              # escalated, not stuck
    assert took < 6.0
    assert backend._proc is None


def test_afplay_cut_mid_playback_is_honoured(monkeypatch):
    import subprocess as sp

    class _Proc:
        def __init__(self):
            self.terminated = threading.Event()

        def wait(self, timeout=None):
            if self.terminated.is_set():
                return 0
            raise sp.TimeoutExpired("afplay", timeout)

        def terminate(self):
            self.terminated.set()

        def kill(self):
            self.terminated.set()

    proc = _Proc()
    monkeypatch.setattr(V, "_afplay_available", lambda: True)
    monkeypatch.setattr(V, "_write_wav", lambda *a, **k: None)
    monkeypatch.setattr(V.subprocess, "Popen", lambda *a, **k: proc)

    backend = object.__new__(V._KokoroBackend)
    backend._active = threading.Event()
    backend._cancel = threading.Event()
    threading.Timer(0.3, backend.cut).start()    # cut() during playback
    flat = np.zeros(16000 * 30, np.float32)      # "30 s clip"
    t0 = time.monotonic()
    backend.play((flat, 16000))
    assert time.monotonic() - t0 < 3.0           # cut, not played out


# ---------------------------------------------------------------------------
# Speaker: prefetch pipeline + derived busy()
# ---------------------------------------------------------------------------
class _PipedBackend:
    """prepare/play backend that records overlap timing."""

    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def prepare(self, text):
        with self.lock:
            self.events.append(("prep0", text, time.monotonic()))
        time.sleep(0.05)
        with self.lock:
            self.events.append(("prep1", text, time.monotonic()))
        return text

    def play(self, prepared):
        with self.lock:
            self.events.append(("play0", prepared, time.monotonic()))
        time.sleep(0.12)
        with self.lock:
            self.events.append(("play1", prepared, time.monotonic()))

    def cut(self):
        pass


def _speaker_with(monkeypatch, backend):
    monkeypatch.setattr(V, "pick_speaker_backend",
                        lambda: (backend, "fake"))
    return V.Speaker()


def test_tts_prefetch_synthesizes_next_while_current_plays(monkeypatch):
    be = _PipedBackend()
    sp = _speaker_with(monkeypatch, be)
    sp.begin_turn()
    sp.say("First one. Second one.")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5:
        with be.lock:
            if sum(1 for e in be.events if e[0] == "play1") >= 2:
                break
        time.sleep(0.01)
    sp.end_turn()
    with be.lock:
        events = list(be.events)
    starts = {(k, t): ts for k, t, ts in events}
    # sentence 2's synthesis STARTED before sentence 1 finished playing
    assert starts[("prep0", "Second one.")] < starts[("play1", "First one.")]
    # and its playback started with no synthesis gap: within ~60 ms of
    # sentence 1 ending (was 50 ms synth + scheduling before the fix)
    gap = starts[("play0", "Second one.")] - starts[("play1", "First one.")]
    assert gap < 0.06


def test_stop_with_prefetched_sentence_releases_busy(monkeypatch):
    """Diff-reread catch: stop() during playback while sentence N+1 sat
    PREFETCHED (outside the queue) left _reported wedged True — busy()
    then paused the mic forever.  The gen-mismatch skip must recompute
    the drain."""
    be = _PipedBackend()
    sp = _speaker_with(monkeypatch, be)
    sp.say("First one. Second one.")              # no turn bracket
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3:              # wait for play of #1
        with be.lock:
            if any(e[0] == "play0" for e in be.events):
                break
        time.sleep(0.005)
    sp.stop()                                     # barge mid-sentence
    t0 = time.monotonic()
    while sp.busy() and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    assert sp.busy() is False                     # never wedged


def test_end_turn_never_drains_while_sentence_prefetched(monkeypatch):
    """SHE-HEARS-HERSELF bug (user report, live): a prefetched sentence
    sits OUTSIDE the queue, so end_turn() between sentences saw
    queue-empty + not-playing and reported speech done — the follow-up
    mic then opened and captured the assistant's own next sentence.
    The drain must wait for the prefetched item too."""
    reports = []

    class _SlowPrep(_PipedBackend):
        def prepare(self, text):
            time.sleep(0.25)                     # wide between-sentence gap
            return super().prepare(text)

    be = _SlowPrep()
    monkeypatch.setattr(V, "pick_speaker_backend", lambda: (be, "fake"))
    sp = V.Speaker(on_speaking=lambda a: reports.append(
        (a, time.monotonic())))
    sp.begin_turn()
    sp.say("First one. Second one.")
    # wait until sentence 1 finished playing (the dangerous gap begins)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5:
        with be.lock:
            if any(e[0] == "play1" and e[1] == "First one."
                   for e in be.events):
                break
        time.sleep(0.005)
    sp.end_turn()                                # turn closes IN the gap
    assert sp.busy() is True                     # prefetched #2 still owed
    assert not any(r[0] is False for r in reports)   # NOT reported done
    # after #2 actually plays, the drain reports exactly once
    t0 = time.monotonic()
    while sp.busy() and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert sp.busy() is False
    with be.lock:
        t_play1_2 = next(ts for k, t, ts in be.events
                         if k == "play1" and t == "Second one.")
    offs = [t for a, t in reports if a is False]
    assert len(offs) == 1 and offs[0] >= t_play1_2   # done AFTER audio


def test_speaker_busy_is_derived_from_player(monkeypatch):
    be = _PipedBackend()
    sp = _speaker_with(monkeypatch, be)
    assert sp.busy() is False
    sp.say("Hello there.")
    t0 = time.monotonic()
    while not sp.busy() and time.monotonic() - t0 < 3:
        time.sleep(0.005)
    assert sp.busy() is True                     # while audio plays
    while sp.busy() and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert sp.busy() is False                    # drained → derived False


# ---------------------------------------------------------------------------
# follow-up: cancel during the FINAL transcription drops the stale turn
# ---------------------------------------------------------------------------
class _NullSource:
    owns_stream = True           # fresh-stream path: no tail discard

    def subscribe(self, fn):
        pass

    def unsubscribe(self, fn):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def test_followup_cancel_during_final_transcribe_drops_result():
    gate = threading.Event()
    entered = threading.Event()

    class _SlowStt:
        ready = True

        def transcribe(self, audio):
            entered.set()
            gate.wait(5)
            return "stale text"

    texts = []
    fl = L.FollowUpListener(_SlowStt(), on_text=texts.append,
                            source_factory=_NullSource,
                            window_s=5.0, endpoint_s=0.15, poll_s=0.01)
    fl.MIN_SPEECH_S = 0.0     # scripted blocks are milliseconds long
    assert fl.open_window() is True
    for _ in range(6):                           # speech, then silence
        fl._on_block(np.full(160, 0.4, np.float32))
    assert entered.wait(3)                       # endpoint hit → STT running
    fl.cancel()                                  # ...user pressed PTT/stop
    gate.set()
    t0 = time.monotonic()
    while fl.active and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    time.sleep(0.05)
    assert texts == []                           # stale turn NOT delivered


# ---------------------------------------------------------------------------
# mic data watchdog: a starved (not dead) stream fires on_died
# ---------------------------------------------------------------------------
def test_short_answer_survives_min_speech_gate():
    """Refuter C2 (units bug): speech_s counted poll wall-time instead
    of block time (~4× undercount) — 'yes', 'next', every answer under
    ~1.4 s was silently discarded at the PRODUCTION gate.  This test
    runs the real MIN_SPEECH_S with realtime-paced 0.2 s blocks."""
    out = {"text": None}
    fl = L.FollowUpListener(
        _make_stt(["yes go ahead"]),
        on_text=lambda t: out.__setitem__("text", t),
        source_factory=_NullSource,
        window_s=4.0, endpoint_s=0.3, poll_s=0.02)
    fl.TAIL_DISCARD_S = 0.05
    assert fl.open_window()
    time.sleep(0.15)
    for _ in range(3):                           # 0.6 s of real speech,
        fl._on_block(np.full(L.BLOCK, 0.4, np.float32))
        time.sleep(0.2)                          # paced like a real mic
    t0 = time.monotonic()
    while out["text"] is None and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    assert out["text"] == "yes go ahead"         # NOT discarded


def _make_stt(script):
    class _S:
        ready = True

        def __init__(self):
            self._script = list(script)

        def transcribe(self, audio):
            return self._script.pop(0) if len(self._script) > 1 \
                else self._script[0]
    return _S()


def test_per_block_prob_pairing_counts_speech_blocks():
    """Refuter P3: a speech block followed by a silence block drained
    in ONE poll was counted entirely at the silence probability —
    systematically losing the end of every utterance.  Each block must
    advance the trigger state with ITS OWN probability."""
    probs = iter([0.9, 0.9, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

    class _V:
        def prob(self, block):
            return next(probs, 0.02)

    fired = []

    class _Stt:
        def transcribe(self, audio):
            fired.append(len(np.asarray(audio).reshape(-1)))
            return "helix"

    wl = L.WakeListener(_HeadMic(), _Stt(), on_command=lambda t: None,
                        is_paused=lambda: False, chime_fn=lambda: None,
                        vad=_V())
    wl.BACKSTOP_S = 99.0
    wl.MINISH = None
    # two speech blocks + quiet blocks all pushed BETWEEN polls: the
    # old pairing credited everything to the last (silent) prob and
    # never triggered
    wl.start()
    try:
        for _ in range(2):                       # 0.4 s of speech
            wl._tap(np.full(L.BLOCK, 0.3, np.float32))
        time.sleep(0.25)                         # one poll sees both
        for _ in range(6):                       # then 1.2 s of quiet
            wl._tap(np.full(L.BLOCK, 0.01, np.float32))
            time.sleep(0.12)
        t0 = time.monotonic()
        while not fired and time.monotonic() - t0 < 4:
            wl._tap(np.full(L.BLOCK, 0.01, np.float32))
            time.sleep(0.1)
        assert fired, "offset trigger lost the speech blocks"
    finally:
        wl.shutdown()


def test_wake_ring_cleared_across_pause():
    """Refuter C3: the ring kept the user's own just-dispatched command
    through the turn's pause — the first post-pause backstop re-matched
    'helix' in it and ran the command a SECOND time.  The paused branch
    must clear the ring."""
    paused = [True]
    wl = L.WakeListener(_HeadMic(), object(), on_command=lambda t: None,
                        is_paused=lambda: paused[0],
                        chime_fn=lambda: None,
                        vad=type("V", (), {"prob": lambda s, b: 0.0})())
    for _ in range(8):                           # "the user's command"
        wl._tap(np.full(L.BLOCK, 0.3, np.float32))
    assert len(wl._buf) == 8
    wl.start()                                   # VAD-event poll loop
    try:
        time.sleep(0.35)                         # a few paused polls
        with wl._lock:
            n = len(wl._buf)
        assert n == 0                            # ring cleared on pause
    finally:
        paused[0] = False
        wl.shutdown()


def test_mic_stall_watchdog_fires_on_died():
    died = threading.Event()

    class _StalledMic:
        last_block_t = time.monotonic() - 10.0   # nothing for 10 s
        on_died = staticmethod(died.set)

        def subscribe(self, fn):
            pass

        def unsubscribe(self, fn):
            pass

    wl = L.WakeListener(_StalledMic(), object(), on_command=lambda t: None,
                        is_paused=lambda: False, chime_fn=lambda: None)
    wl._check_mic_stall()
    assert died.wait(2)
    assert wl._stall_reported is True
    # flow resumes → watchdog re-arms
    wl.mic.last_block_t = time.monotonic()
    wl._check_mic_stall()
    assert wl._stall_reported is False


def test_micstream_records_overflow_and_reports_once():
    ms = L.MicStream()
    hits = []
    ms.on_overflow = lambda: hits.append(1)

    class _St:
        def __init__(self):
            self.n = 0

        def read(self, blocksize):
            self.n += 1
            if self.n > 3:
                raise RuntimeError("done")
            return np.zeros((blocksize, 1), np.float32), True  # overflow

        def stop(self):
            pass

        def close(self):
            pass

    ms._stream = _St()
    ms._reader()
    assert ms.overflows == 3
    assert hits == [1]                           # first overflow only


# ---------------------------------------------------------------------------
# open_window is atomic under thread races
# ---------------------------------------------------------------------------
def test_open_window_atomic_under_race(monkeypatch):
    fl = L.FollowUpListener(object(), on_text=lambda t: None,
                            source_factory=_NullSource, window_s=10.0)
    monkeypatch.setattr(L.FollowUpListener, "_run", lambda self: None)
    wins = []
    barrier = threading.Barrier(8)

    def _try():
        barrier.wait()
        wins.append(fl.open_window())

    threads = [threading.Thread(target=_try) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for w in wins if w) == 1        # exactly one window


# ---------------------------------------------------------------------------
# SDK ask(): heartbeat detects a dead backend loop (never blocks forever)
# ---------------------------------------------------------------------------
def test_sdk_ask_heartbeat_detects_dead_loop():
    import asyncio

    from linac_gen.assist.sdk_backend import ClaudeSdkBackend

    class _Sess:
        def __init__(self):
            self._abort = threading.Event()
            self.errors = []

        def _emit(self, **ev):
            self.errors.append(ev)

    be = object.__new__(ClaudeSdkBackend)
    be.session = _Sess()
    be._loop = asyncio.new_event_loop()          # never run → fut never done
    be._client = None
    be._start_error = None
    be._turn_lock = threading.Lock()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    be._thread = dead                            # loop thread is DEAD
    t0 = time.monotonic()
    out = be.ask("hello")
    assert time.monotonic() - t0 < 5.0           # returned, not wedged
    assert "died" in out
    assert any(e.get("type") == "error" for e in be.session.errors)
    be._loop.close()


# ---------------------------------------------------------------------------
# PTT: stop() before an async start() must never open a stream
# ---------------------------------------------------------------------------
def test_ptt_stop_before_start_never_opens_stream(monkeypatch):
    import sys
    import types

    opened = []

    class _NeverStream:
        def __init__(self, *a, **k):
            opened.append(1)

        def start(self):
            pass

    fake_sd = types.SimpleNamespace(InputStream=_NeverStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    rec = V.PushToTalkRecorder()
    rec.stop()                                   # release beat the start
    rec.start()                                  # ...now start arrives late
    assert opened == []                          # no orphaned hot mic
