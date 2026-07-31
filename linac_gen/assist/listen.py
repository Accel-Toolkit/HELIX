"""Hands-free listening: wake word, follow-up windows, barge-in.

Ported from the MIRAGE assistant and re-architected for HELIX's audio
rules.  THE mic-ownership rule (CoreAudio constraints, both observed in
production):

1. **No Python PortAudio callbacks, ever.**  A Python callback on the
   RT audio thread is a use-after-free SIGSEGV when the stream is torn
   down (see ``voice.PushToTalkRecorder``).  Everything here uses
   blocking ``stream.read()`` on a dedicated reader thread.
2. **Exactly one InputStream at a time, and only its owning reader
   thread touches start/stop.**  Concurrent stream start/stop across
   threads deadlocks CoreAudio's HAL mutex (observed: GUI thread frozen
   in AudioDeviceStop).  While the wake toggle is ON, :class:`MicStream`
   is the single always-open stream; everything else (wake detection,
   barge-in, follow-up capture, even push-to-talk) listens through taps
   on it.  With the toggle OFF, transient single-owner streams
   (:class:`TransientMic`, ``voice.PushToTalkRecorder``) are used one
   at a time.

Nothing in this module imports Qt; sounddevice is imported lazily; every
component takes injectable sources/clocks so the logic is testable with
no audio hardware.

Env knobs: ``HELIX_FOLLOWUP_S`` (follow-up window length, 0 disables),
``HELIX_BARGE_IN=0`` (disable talk-over interruption).
"""
from __future__ import annotations

import collections
import os
import re
import threading
import time

import numpy as np

from .voice import pa_lifecycle

SR = 16000
BLOCK = SR // 5                       # 0.2 s blocks


# ---------------------------------------------------------------------------
# wake word
# ---------------------------------------------------------------------------
# Whisper renders "HELIX" many ways — accept the common mishearings so
# detection is reliable (the chime + visible orb cue make a false trigger
# obvious and harmless: a capture of silence times out and nothing is
# sent).  NOT accepted: helipad/helical/helicopter-class words.
WAKE_RE = re.compile(
    r"\b(helix|helixx?|helics|heelix|helux|healix|helis|heliks|"
    r"he\s?licks|he\s?lix|felix|helic|helix'?s|heylix|"
    r"headaches?)\b", re.I)
# ("headaches" is empirically what some STT engines make of a spoken
# 'HELIX' — harmless to accept: the mic is armed, the chime is audible,
# and a false capture of silence times out.)


def wake_matches(text: str) -> bool:
    """Regex OR per-token fuzzy match (ratio >= 0.72 vs 'helix') —
    accents produce renderings no fixed list anticipates (hillix,
    healex, halix ...), while helipad/helical/relax stay excluded
    (calibrated: they score <= 0.67)."""
    if not text:
        return False
    if WAKE_RE.search(text):
        return True
    from difflib import SequenceMatcher
    for tok in re.findall(r"[a-zA-Z']+", text.lower()):
        if 4 <= len(tok) <= 8 and SequenceMatcher(
                None, tok, "helix").ratio() >= 0.72:
            return True
    return False


def split_after_wake(text: str) -> str:
    """Words spoken AFTER the wake token in the same utterance — the
    head of the command, not noise.  "helix open the results tab" in
    one breath used to lose everything after the token (the capture
    only started later); the head is now forwarded to the command."""
    if not text:
        return ""
    m = None
    for m in WAKE_RE.finditer(text):
        pass                                  # last occurrence wins
    if m is not None:
        return text[m.end():].strip(" \t,.;:!?-")
    from difflib import SequenceMatcher
    toks = list(re.finditer(r"[a-zA-Z']+", text))
    for i in range(len(toks) - 1, -1, -1):
        tok = toks[i].group(0).lower()
        if 4 <= len(tok) <= 8 and SequenceMatcher(
                None, tok, "helix").ratio() >= 0.72:
            return text[toks[i].end():].strip(" \t,.;:!?-")
    return ""

# local yes/no for confirmation-by-voice — the model is NEVER in this
# loop.  "no" outranks "yes" ("no, don't do it" must deny).
YES_RE = re.compile(r"\b(yes|yeah|yep|confirm|confirmed|go ahead|do it|"
                    r"proceed|affirmative|sure|approve|approved)\b", re.I)
NO_RE = re.compile(r"\b(no|nope|cancel|stop|abort|negative|deny|denied|"
                   r"don'?t)\b", re.I)


def interpret_confirm(text: str) -> str | None:
    """``"yes"`` / ``"no"`` / ``None`` (unrecognised).  NO outranks YES."""
    t = (text or "").strip()
    if not t:
        return None
    if NO_RE.search(t):
        return "no"
    if YES_RE.search(t):
        return "yes"
    return None


def followup_window_s(default: float = 10.0) -> float:
    """Follow-up window length; ``HELIX_FOLLOWUP_S=0`` disables."""
    try:
        return float(os.environ.get("HELIX_FOLLOWUP_S", default))
    except (TypeError, ValueError):
        return default


def barge_in_enabled() -> bool:
    return os.environ.get("HELIX_BARGE_IN", "1") != "0"


_CHIME_WAV: list = [None]            # lazily rendered once, then reused
_CHIME_PROCS: list = []              # live afplay children (reaped on use)


def _render_chime() -> str:
    """Render the two-tone cue to a wav ONCE (16-bit PCM, stdlib)."""
    import tempfile

    def tone(f, dur):
        t = np.linspace(0, dur, int(SR * dur), False)
        return (np.sin(2 * np.pi * f * t)
                * np.hanning(t.size)).astype(np.float32)

    from linac_gen.assist.voice import _write_wav
    path = os.path.join(tempfile.gettempdir(), "helix_chime.wav")
    _write_wav(0.3 * np.concatenate([tone(660, 0.08), tone(990, 0.10)]),
               SR, path)
    return path


def chime() -> None:
    """Short rising two-tone 'ready' cue (output side — not part of the
    input-stream ownership rules).  Best-effort.

    Playback goes through ``afplay`` (its own OS process): the chime
    fires at the MOST CPU-loaded instant of the whole pipeline — Whisper
    is transcribing the wake word on every performance core — and an
    in-process ``sd.play()`` there opens a throwaway stream with a tiny
    buffer that pops on open/close and underruns (crackle) under load.
    """
    try:
        from linac_gen.assist.voice import _afplay_available
        if _afplay_available():
            import subprocess
            if _CHIME_WAV[0] is None or not os.path.exists(_CHIME_WAV[0]):
                _CHIME_WAV[0] = _render_chime()
            # keep handles + poll: reaps finished afplay children so
            # they never linger as zombies (defensive poll — the whole
            # chime is best-effort and must never raise)
            def _live(p):
                try:
                    return p.poll() is None
                except Exception:                           # noqa: BLE001
                    return False
            _CHIME_PROCS[:] = [p for p in _CHIME_PROCS if _live(p)]
            _CHIME_PROCS.append(subprocess.Popen(
                ["afplay", _CHIME_WAV[0]]))
            return
        import sounddevice as sd

        def tone(f, dur):
            t = np.linspace(0, dur, int(SR * dur), False)
            return (np.sin(2 * np.pi * f * t)
                    * np.hanning(t.size)).astype(np.float32)
        sd.play(0.3 * np.concatenate([tone(660, 0.08), tone(990, 0.10)]),
                SR)
    except Exception:                                       # noqa: BLE001
        pass


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(
        np.asarray(block, dtype=np.float32)))))


# ---------------------------------------------------------------------------
# the single always-open stream (wake mode)
# ---------------------------------------------------------------------------
#: streams whose reader thread wedged (CoreAudio can hang a blocking
#: read after system sleep/wake).  Parked = strong ref so GC can never
#: finalize (close) the stream under the stuck reader — that frees the
#: PortAudio ring buffer mid-memmove and SIGSEGVs the process.
_ABANDONED_STREAMS: list = []


class MicStream:
    """THE microphone while wake listening is on.

    Blocking ``stream.read()`` on one reader thread; every 1-D float32
    block is fanned out to registered taps (tap = tiny callable, runs on
    the reader thread, must never raise — same contract as MIRAGE's
    ``wake.add_tap``).  ``close()`` stops the reader FIRST, then the
    stream, on the reader's owner — nobody else ever touches start/stop.
    """

    def __init__(self, samplerate: int = SR, blocksize: int = BLOCK,
                 on_died=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        #: called (from the READER thread) when the stream dies without
        #: close() — device change, sleep/wake, unplugged headset.  A
        #: dead wake stream is otherwise SILENT: no wake word, PTT taps
        #: capture nothing, and the user just sees "(nothing heard)".
        self.on_died = on_died
        self._taps: list = []
        self._lock = threading.Lock()
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # 1.0 s: the reader wakes every 0.2 s block — if it hasn't
        # exited in a second it is WEDGED and gets parked; longer joins
        # only freeze the caller (the GUI at toggle-off) for nothing
        self._join_timeout_s = 1.0
        #: wall-clock of the last delivered block (real time.monotonic,
        #: never an injected clock) — the data watchdog's heartbeat.  A
        #: CoreAudio stream can stall silently (sleep/wake, device
        #: switch) with the reader alive but starved; on_died never
        #: fires and wake is just... dead.  Listeners compare against
        #: this to detect the starvation.
        self.last_block_t: float | None = None
        self.overflows = 0
        #: called once (reader thread) on the FIRST input overflow —
        #: dropped mic audio must be visible, not silent
        self.on_overflow = None

    # tap registry (thread-safe)
    def subscribe(self, fn) -> None:
        with self._lock:
            if fn not in self._taps:
                self._taps.append(fn)

    def unsubscribe(self, fn) -> None:
        with self._lock:
            if fn in self._taps:
                self._taps.remove(fn)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def open(self) -> None:
        """Open on the caller so device errors surface synchronously."""
        import sounddevice as sd
        if self.running:
            return
        if self._stream is not None:
            # a previous (wedged) reader still owns a stream — opening
            # again would swap self._stream under it; never reuse
            return
        self._stop.clear()
        with pa_lifecycle():
            self._stream = sd.InputStream(
                samplerate=self.samplerate, channels=1, dtype="float32",
                blocksize=self.blocksize)
            self._stream.start()
        # armed from open time so a stream that NEVER delivers a block
        # (phantom device) also trips the data watchdog
        self.last_block_t = time.monotonic()
        self._thread = threading.Thread(target=self._reader, daemon=True,
                                        name="assist-micstream")
        self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                data, _overflow = self._stream.read(self.blocksize)
            except Exception:                               # noqa: BLE001
                break
            self.last_block_t = time.monotonic()
            if _overflow:
                self.overflows += 1
                if self.overflows == 1 and self.on_overflow is not None:
                    try:
                        self.on_overflow()
                    except Exception:                       # noqa: BLE001
                        pass
            block = np.asarray(data, dtype=np.float32).reshape(-1)
            with self._lock:
                taps = list(self._taps)
            for fn in taps:
                try:
                    fn(block)
                except Exception:                           # noqa: BLE001
                    pass
        died = not self._stop.is_set()      # error exit, not close()
        # the READER closes the stream — single-owner teardown
        with pa_lifecycle():
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:                               # noqa: BLE001
                pass
        self._stream = None
        if died and self.on_died is not None:
            try:
                self.on_died()
            except Exception:                               # noqa: BLE001
                pass

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_s)
            if self._thread.is_alive():
                # wedged reader (post-sleep CoreAudio hang): the reader
                # still owns the stream — park us forever so nothing
                # (incl. GC) can close it under the in-flight read.
                _ABANDONED_STREAMS.append(self)
            self._thread = None


class CaptureTap:
    """Accumulate blocks from a source between start()/stop() — the
    push-to-talk path while the wake MicStream owns the microphone."""

    def __init__(self, source, on_level=None):
        self._source = source
        self._on_level = on_level
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()

    def _tap(self, block) -> None:
        with self._lock:
            self._chunks.append(np.asarray(block, dtype=np.float32))
        if self._on_level is not None:
            try:
                self._on_level(_rms(block))
            except Exception:                               # noqa: BLE001
                pass

    def start(self) -> None:
        with self._lock:
            self._chunks = []
        self._source.subscribe(self._tap)

    def stop(self) -> np.ndarray:
        self._source.unsubscribe(self._tap)
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks).reshape(-1)


class TransientMic:
    """A short-lived single-owner stream with the same subscribe API as
    :class:`MicStream` — the follow-up source when wake mode is OFF.
    Blocking reads on its own thread; one instance = one open/close."""

    #: fresh stream: no TTS bleed to discard, but a cold-device warm-up
    #: to absorb (the follow-up loop keys off this)
    owns_stream = True

    def __init__(self, samplerate: int = SR, blocksize: int = BLOCK):
        self._inner = MicStream(samplerate, blocksize)

    def subscribe(self, fn) -> None:
        self._inner.subscribe(fn)

    def unsubscribe(self, fn) -> None:
        self._inner.unsubscribe(fn)

    def start(self) -> None:
        self._inner.open()

    def stop(self) -> None:
        self._inner.close()


class _SharedSource:
    """Adapter giving a MicStream tap the start/stop shape the follow-up
    session expects — start/stop only touch the TAP, never the stream."""

    #: always-open stream: warm, but carries the speaker's tail
    owns_stream = False

    def __init__(self, mic: MicStream):
        self._mic = mic
        self._fn = None

    def subscribe(self, fn) -> None:
        self._fn = fn

    def unsubscribe(self, fn) -> None:
        if self._fn is fn:
            self._fn = None

    def start(self) -> None:
        if self._fn is not None:
            self._mic.subscribe(self._fn)

    def stop(self) -> None:
        if self._fn is not None:
            self._mic.unsubscribe(self._fn)


# ---------------------------------------------------------------------------
# speculative STT (MIRAGE port)
# ---------------------------------------------------------------------------
class _SpecSTT:
    """Launch Whisper at the FIRST plausible pause, generation-keyed;
    join at the real endpoint.  If no new speech arrived after the
    launch, the speculative text IS the answer — the transcript is
    ready the moment the user stops talking instead of endpoint +
    full-STT-time (the single biggest perceived-latency win MIRAGE
    had over us).  Generation counting (not sample counts): silence
    blocks keep arriving, so only speech events advance the key."""

    def __init__(self, stt):
        self.stt = stt
        self.gen = None
        self.text = None
        self._th: threading.Thread | None = None

    def launch(self, gen: int, audio) -> None:
        if self.gen == gen and self._th is not None:
            return                        # this pause is already
        if self._th is not None and self._th.is_alive():
            return                        # speculated — never re-burn
        self.gen, self.text = gen, None   # Whisper on the same audio

        def _run(a=audio, g=gen):
            try:
                t = self.stt.transcribe(a)
            except Exception:                               # noqa: BLE001
                t = ""
            if self.gen == g:
                self.text = t
        self._th = threading.Thread(target=_run, daemon=True,
                                    name="assist-spec-stt")
        self._th.start()

    def take(self, gen: int, timeout: float = 8.0):
        """The speculative transcript if it matches ``gen``; None means
        speech arrived after the launch (or the speculation heard
        nothing) — transcribe for real."""
        if self.gen != gen or self._th is None:
            return None
        self._th.join(timeout=timeout)
        t = self.text if self.gen == gen else None
        return t or None                  # "" → let the caller re-run


# ---------------------------------------------------------------------------
# wake listener
# ---------------------------------------------------------------------------
class WakeListener:
    """Say "HELIX", hear the chime, speak a command.

    Consumes blocks from a :class:`MicStream` tap into a rolling ~2.5 s
    window; a poll thread transcribes the window every ~1 s (adaptive
    silence gate skips STT when the room is quiet).  On a wake-word
    match: ``on_wake()`` + chime, then a silence-endpointed capture
    (1 s trailing quiet / 10 s max / 3.5 s grace to start talking) is
    transcribed and handed to ``on_command``.

    ``is_paused()`` (TTS speaking, PTT recording, follow-up open) gates
    both detection and the mic-level cue — the assistant can never wake
    itself, and saying "HELIX" in normal conversation only matters while
    the toggle is armed.
    """

    #: 6 s, NOT 2.5: a natural one-breath command ("helix run the
    #: envelope and report sigma x" ≈ 3.5 s) scrolled the wake token
    #: out of a 2.5 s ring before the end-of-utterance trigger fired —
    #: a near-certain miss for every command longer than the ring
    WINDOW_S = 6.0
    POLL_S = 1.0
    #: loudness-backstop cadence in VAD event mode (tests raise it to
    #: keep timing-sensitive phases deterministic under CI load).
    #: 1.0 s mirrors MIRAGE's overlap guarantee: any loud audio is
    #: checked at least once a second no matter what VAD thinks.
    BACKSTOP_S = 1.0
    #: mic data watchdog — reader alive but no blocks flowing for this
    #: long (CoreAudio stall after sleep/device switch) fires on_died
    STALL_S = 2.5

    def __init__(self, mic: MicStream, transcriber, on_command, is_paused,
                 on_status=None, on_wake=None, on_level=None,
                 chime_fn=chime, clock=time.monotonic, vad=None):
        #: optional speech-probability gate (assist.vad.SileroVAD-shaped:
        #: .prob(block)->float).  None -> the RMS gates below, unchanged.
        self.vad = vad
        self.mic = mic
        self.stt = transcriber
        self.on_command = on_command
        self.is_paused = is_paused
        self.on_status = on_status or (lambda s: None)
        self.on_wake = on_wake or (lambda: None)
        self.on_level = on_level or (lambda lvl: None)
        self._chime = chime_fn
        self._clock = clock
        self._buf: collections.deque = collections.deque(
            maxlen=int(self.WINDOW_S / (BLOCK / SR)))
        self._cap_buf: collections.deque = collections.deque()
        self._capturing = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._noise = 0.006
        self._last_prob = 0.0
        self._blocks_seen = 0            # tap-side counter: the poll
        self._pending_probs: collections.deque = collections.deque()
        self._stall_reported = False     # accumulates time PER BLOCK

    # -- tap (mic reader thread — tiny, never raises) -------------------
    def _tap(self, block) -> None:
        # VAD scoring lives HERE: every block exactly once, in order —
        # silero carries recurrent state, so re-feeding overlapping
        # windows from the poll loop smears it.
        p = None
        if self.vad is not None:
            p = self.vad.prob(block)
            self._last_prob = p
        with self._lock:
            self._buf.append(block)
            self._blocks_seen += 1
            if self.vad is not None:
                self._pending_probs.append(p)
                if len(self._pending_probs) > 64:
                    self._pending_probs.popleft()
            if self._capturing:
                self._cap_buf.append((block, p))
        if not self.is_paused():
            self.on_level(_rms(block))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.mic.subscribe(self._tap)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="assist-wake")
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self.mic.unsubscribe(self._tap)
        if self._thread is not None:
            # 1.0 s: a poll thread blocked in Whisper won't exit sooner
            # anyway — the _stop guards make its late results harmless,
            # so don't freeze the caller waiting on it
            self._thread.join(timeout=1.0)
            self._thread = None

    # -- poll loop -------------------------------------------------------
    def _run(self) -> None:
        self.on_status("say 'HELIX'")
        if self.vad is not None:
            self._run_vad_events()                # utterance-triggered
        else:
            self._run_cadence()                   # legacy 1 s polling

    def _handle_window(self, audio, quiet: bool = False) -> None:
        """Transcribe a candidate window; on a wake-word match run the
        full wake sequence (chime → endpointed capture → command).

        Capture is armed BEFORE the (0.3–1 s) transcription: words the
        user keeps speaking while Whisper decides land in ``_cap_buf``
        instead of vanishing — the head-clip fix.  Words after the wake
        token inside the window itself become the command HEAD, so
        "helix open the results tab" in one breath works.

        ``quiet=True`` means the trigger was a clean end-of-utterance:
        when the head already carries a substantive command, the
        capture (a pure 1.2 s dead wait in that case) is skipped and
        the command dispatches IMMEDIATELY."""
        with self._lock:
            self._cap_buf.clear()
            self._capturing = True
        try:
            text = self.stt.transcribe(audio)
        except Exception:                                   # noqa: BLE001
            with self._lock:
                self._capturing = False
                self._cap_buf.clear()
            return
        if self._stop.is_set():          # torn down mid-transcription —
            with self._lock:             # never call back into a dead
                self._capturing = False  # panel (SIGSEGV class)
            return
        if not wake_matches(text):
            with self._lock:
                self._capturing = False
                self._cap_buf.clear()
            if text:
                # SHOW the user what was heard — a wake miss must never
                # be a mystery (accent renderings taught us the hard way)
                self.on_status(f"heard “{text.strip()[:48]}” — "
                               "say 'HELIX'")
            return
        head = split_after_wake(text)
        with self._lock:
            self._buf.clear()
        self.on_wake()
        try:
            self._chime()
        except Exception:                                   # noqa: BLE001
            pass
        if quiet and len(head.split()) >= 3:
            with self._lock:
                self._capturing = False      # capture never runs here
                self._cap_buf.clear()
            cmd = ""        # complete one-breath command: dispatch NOW
        else:               # (the grace was a pure 1.2 s dead wait)
            cmd = self._capture(armed=True,
                                grace_s=1.2 if head else 3.5)
        full = f"{head} {cmd}".strip() if head else cmd
        if self._stop.is_set():
            return
        if full:
            self.on_command(full)
        self.on_status("say 'HELIX'")

    def _check_mic_stall(self) -> None:
        """Data watchdog: the reader can be alive while CoreAudio
        starves it (sleep/wake, device switch) — on_died would never
        fire and wake would be silently dead.  Real wall-clock on both
        sides (mic timestamps are real monotonic, never a test clock)."""
        lb = getattr(self.mic, "last_block_t", None)
        if lb is None:
            return
        if time.monotonic() - lb > self.STALL_S:
            if not self._stall_reported:
                self._stall_reported = True
                died = getattr(self.mic, "on_died", None)
                if died is not None:
                    threading.Thread(target=died, daemon=True,
                                     name="assist-mic-stall").start()
        else:
            self._stall_reported = False

    def _run_cadence(self) -> None:
        need = int(1.2 / (BLOCK / SR))
        while not self._stop.is_set():
            if self._stop.wait(self.POLL_S):
                break
            self._check_mic_stall()
            if self.is_paused():
                continue
            with self._lock:
                if len(self._buf) < need:
                    continue
                audio = np.concatenate(list(self._buf))
            level = _rms(audio)
            if level < max(self._noise * 2.2, 0.006):
                self._noise = 0.9 * self._noise + 0.1 * level
                continue                          # silence — skip STT
            self._handle_window(audio)

    def _run_vad_events(self) -> None:
        """Event-driven wake: Whisper fires the moment an utterance
        ENDS (VAD offset) instead of on a fixed cadence — no polling
        delay, no mid-speech transcriptions of half a word.  Hysteresis
        (enter 0.45 / stay 0.30) keeps flappy far-field speech in one
        utterance; a long monologue is still checked every ~2.5 s; and
        a 1.0 s LOUDNESS backstop transcribes anyway when the room is
        loud but VAD never triggered — event mode can never hear WORSE
        than the legacy cadence.

        Offset minimums (speech >= 0.35 s, quiet >= 0.45 s — MIRAGE's
        calibration): sub-word blips and within-word dips no longer
        fire Whisper on half a syllable; real utterances trigger on a
        clean seam, and the backstop covers everything else."""
        poll = 0.1
        need = int(1.2 / (BLOCK / SR))
        in_speech = False
        speech_s = quiet_s = 0.0
        last_stt = self._clock()
        last_seen = 0
        while not self._stop.is_set():
            if self._stop.wait(poll):
                break
            self._check_mic_stall()
            if self.is_paused():
                in_speech = False
                speech_s = quiet_s = 0.0
                # do NOT reset last_stt: resetting it added a fixed
                # +1.0 s deaf window after EVERY pause (each TTS reply)
                with self._lock:
                    last_seen = self._blocks_seen
                    self._pending_probs.clear()
                    # CLEAR the ring: pause-period audio is the user's
                    # own just-dispatched command (or our speech tail) —
                    # the first post-pause backstop used to re-match the
                    # "helix" in it and run the command a SECOND time
                    self._buf.clear()
                continue
            with self._lock:
                if not self._buf:
                    continue
                n_seen = self._blocks_seen
                probs = list(self._pending_probs)
                self._pending_probs.clear()
            # TRUE per-block accounting: each block advances the state
            # with ITS OWN probability (crediting a whole drained burst
            # to the latest prob systematically under-counted speech
            # whenever the final block of an utterance was silent)
            new = n_seen - last_seen
            last_seen = n_seen
            if probs:
                budget = 0.6              # TIME cap per poll (bursts
                for p in probs:           # after a long STT call) —
                    dt = min(BLOCK / SR, max(0.0, budget))
                    budget -= dt          # every block still advances
                    if p is not None and p > (0.30 if in_speech
                                              else 0.45):  # hysteresis
                        in_speech = True
                        speech_s += dt
                        quiet_s = 0.0
                    elif in_speech:
                        quiet_s += dt
            elif new:                     # RMS-only mode: no probs queued
                dt = min(new * (BLOCK / SR), 0.6)
                p = self._last_prob
                if p > (0.30 if in_speech else 0.45):
                    in_speech = True
                    speech_s += dt
                    quiet_s = 0.0
                elif in_speech:
                    quiet_s += dt
            trigger = in_speech and (
                (quiet_s >= 0.30 and speech_s >= 0.35) or speech_s >= 2.5)
            if trigger and self._clock() - last_stt < 0.5:
                trigger = False           # STT rate floor: state is kept,
                #                           the trigger re-fires next poll
            if not trigger and self._clock() - last_stt >= self.BACKSTOP_S:
                # backstop: loud window with no VAD event — check it.
                # Gate on the PEAK block, not the whole-window RMS: a
                # 0.5 s word in a 6 s window dilutes to ~0.3× its RMS
                # and the old gate missed it entirely in room noise.
                keep = max(need, int(3.0 / (BLOCK / SR)))
                with self._lock:
                    blocks = list(self._buf)[-keep:]  # last ~3 s: a wake
                #                                       token older than
                #                                       that is stale
                if len(blocks) >= need:
                    peak = max(_rms(b) for b in blocks[-10:])
                    if peak >= max(self._noise * 2.5, 0.008):
                        last_stt = self._clock()
                        self._handle_window(np.concatenate(blocks))
                    else:
                        # clamped: the floor used to RATCHET up to the
                        # ambient level and gate out real speech in any
                        # loud room (total deafness in RMS-only mode)
                        self._noise = min(
                            0.02, 0.9 * self._noise + 0.1 * peak)
                continue
            if not trigger:
                continue
            offset_quiet = quiet_s >= 0.30        # clean end-of-utterance
            span = int((speech_s + quiet_s + 1.2) / (BLOCK / SR))
            span = max(6, min(span, int(self.WINDOW_S / (BLOCK / SR))))
            in_speech = False
            speech_s = quiet_s = 0.0
            last_stt = self._clock()
            with self._lock:
                audio = np.concatenate(list(self._buf)[-span:])
            self._handle_window(audio, quiet=offset_quiet)

    def _capture(self, armed: bool = False, grace_s: float = 3.5) -> str:
        """Record after the wake word until trailing silence (10 s max,
        ``grace_s`` to start talking).  ``armed=True``: capture was
        already running during wake transcription — KEEP those blocks
        (the head-clip fix); the wake sequence shortens the grace when
        the window already carried command words.  The silence endpoint
        is 0.7 s with VAD (clean offsets), 1.0 s on RMS."""
        self.on_status("listening — go ahead …")
        if not armed:
            with self._lock:
                self._cap_buf.clear()
                self._capturing = True
        chunks: list[np.ndarray] = []
        heard = False
        quiet = 0.0
        sgen = 0                         # speech generation (speculation)
        spec = _SpecSTT(self.stt)
        t0 = self._clock()
        thr = max(self._noise * 2.5, 0.008)
        endpoint = 0.7 if self.vad is not None else 1.0
        try:
            while self._clock() - t0 < 10.0 and not self._stop.is_set():
                time.sleep(BLOCK / SR)
                fresh: list[np.ndarray] = []
                probs: list[float] = []
                with self._lock:
                    while self._cap_buf:
                        blk, p = self._cap_buf.popleft()
                        fresh.append(blk)
                        if p is not None:
                            probs.append(p)
                if fresh:
                    chunks.extend(fresh)
                if not chunks:
                    continue
                if fresh:
                    # MAX over the drained burst, not the last block: a
                    # speech-then-silence burst used to read as silence
                    # and endpoint mid-utterance
                    level = max(_rms(b) for b in fresh)
                    if self.vad is not None and probs:
                        speech = max(probs) > 0.45
                    else:
                        speech = level > thr
                        if not armed and self._clock() - t0 < 0.35:
                            speech = False   # chime bleed (RMS mode,
                            #                  fresh capture only): the
                            #                  cue itself set heard;
                            #                  ARMED captures carry real
                            #                  pre-wake user speech
                            #                  that must latch
                    if speech:
                        heard = True
                        quiet = 0.0
                        sgen += 1
                    elif heard:
                        quiet += len(fresh) * (BLOCK / SR)
                elif heard:
                    quiet += BLOCK / SR
                if heard and 0.30 <= quiet < endpoint:
                    # first plausible pause: speculate — the transcript
                    # is ready AT the endpoint instead of after it
                    spec.launch(sgen, np.concatenate(chunks))
                if heard and quiet >= endpoint:
                    break
                if not heard and self._clock() - t0 > grace_s:
                    break
        finally:
            with self._lock:
                self._capturing = False
        if self._stop.is_set():          # teardown — no final STT, no
            return ""                    # callbacks into a dead panel
        if not heard or not chunks:
            return ""
        text = spec.take(sgen)
        if text is not None:
            return text
        try:
            return self.stt.transcribe(np.concatenate(chunks))
        except Exception:                                   # noqa: BLE001
            return ""


# ---------------------------------------------------------------------------
# follow-up listener (conversational window)
# ---------------------------------------------------------------------------
class FollowUpListener:
    """After the assistant finishes SPEAKING a reply to a voice turn,
    keep the mic open ~10 s — talk and it's the next turn, stay quiet
    and the window closes.  One capture session per ``open_window()``.

    ``source_factory()`` returns an object with ``subscribe/unsubscribe``
    + ``start()/stop()`` (a :class:`_SharedSource` over the wake
    MicStream, a :class:`TransientMic`, or a test fake).
    """

    #: shared-stream capture start delay: the speaker's audible tail
    #: (CoreAudio buffer drain + room reverb) must die before we listen,
    #: or the assistant hears her own last words and answers herself
    TAIL_DISCARD_S = 0.5
    #: minimum DETECTED speech before a capture is worth transcribing —
    #: a click/noise blip fed 30 s of room audio to Whisper, whose
    #: hallucination became a phantom user turn (tests lower this)
    MIN_SPEECH_S = 0.35

    def __init__(self, transcriber, on_text, source_factory,
                 on_timeout=None, on_status=None, on_level=None,
                 window_s=None, endpoint_s=0.9, max_s=30.0, poll_s=0.05,
                 clock=time.monotonic, vad=None, busy_probe=None):
        #: live "is the assistant audibly speaking" probe — gates the
        #: tail discard so it drops ONLY actual bleed, not 0.5 s of the
        #: user's first words
        self._busy_probe = busy_probe or (lambda: False)
        #: optional speech gate (assist.vad) — None keeps the RMS gates
        self.vad = vad
        self.stt = transcriber
        self.on_text = on_text
        self.on_timeout = on_timeout or (lambda: None)
        self.on_status = on_status or (lambda s: None)
        self.on_level = on_level or (lambda lvl: None)
        self.window_s = (followup_window_s() if window_s is None
                         else float(window_s))
        self.endpoint_s = float(endpoint_s)
        self.max_s = float(max_s)
        self.poll_s = float(poll_s)
        self._source_factory = source_factory
        self._clock = clock
        self._buf: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._active = threading.Event()
        self._cancel = threading.Event()
        self._open_lock = threading.Lock()
        self._noise = 0.006

    @property
    def active(self) -> bool:
        return self._active.is_set()

    def open_window(self) -> bool:
        """Arm one listening window.  False if disabled or already
        listening — ATOMIC test-and-set (the GUI thread and the TTS
        player thread both open windows; a race here once started two
        capture sessions over two streams = CoreAudio HAL deadlock)."""
        with self._open_lock:
            if self.window_s <= 0 or self._active.is_set():
                return False
            self._active.set()
        self._cancel.clear()
        threading.Thread(target=self._run, daemon=True,
                         name="assist-followup").start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------
    def _on_block(self, block) -> None:
        b = np.asarray(block, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buf.append(b)
        self.on_level(_rms(b))

    def _drain(self):
        with self._lock:
            out, self._buf = self._buf, []
        return out

    def _run(self) -> None:
        try:
            source = self._source_factory()
            source.subscribe(self._on_block)
            source.start()
        except Exception as e:                              # noqa: BLE001
            self.on_status(f"follow-up mic unavailable: {e}")
            self._active.clear()
            return
        self.on_status(f"listening — just talk "
                       f"({self.window_s:.0f} s window) …")
        chunks: list[np.ndarray] = []
        heard = False
        quiet = 0.0
        try:
            if not getattr(source, "owns_stream", False):
                # GATED tail discard: drop blocks only while the
                # assistant is actually still audible (busy probe), max
                # TAIL_DISCARD_S — the old blind 0.5 s sleep destroyed
                # the user's first half-second on EVERY turn
                t0d = self._clock()
                while (not self._cancel.is_set()
                       and self._clock() - t0d < self.TAIL_DISCARD_S
                       and (self._busy_probe()
                            or self._clock() - t0d < 0.25)):
                    # 0.25 s FLOOR even when the probe reads idle: on a
                    # barge the player reports drained instantly while
                    # the terminated afplay's buffer is still sounding
                    time.sleep(self.poll_s)
                time.sleep(self.poll_s)   # one beat for stragglers
                self._drain()             # discard only the actual bleed
            else:
                # FRESH stream: nothing to discard — the user's first
                # words must be kept.  Cold CoreAudio delivers no frames
                # for up to ~0.5 s; hold the window clock until audio
                # actually flows so that warm-up never eats the window.
                t_warm = self._clock()
                while (not self._cancel.is_set()
                       and self._clock() - t_warm < 1.5):
                    with self._lock:
                        if self._buf:
                            break
                    time.sleep(self.poll_s)
            t0 = last = self._clock()
            speech_s = 0.0
            sgen = 0
            spec = _SpecSTT(self.stt)
            while not self._cancel.is_set():
                time.sleep(self.poll_s)
                now = self._clock()
                dt, last = now - last, now
                fresh = self._drain()
                if fresh:
                    chunks.extend(fresh)
                    joined = np.concatenate(fresh)
                    level = _rms(joined)
                    if self.vad is not None:
                        speech = self.vad.prob(joined) > 0.4
                    else:
                        thr = max(self._noise * 2.5, 0.008)
                        speech = level > thr
                    if speech:
                        heard = True
                        # BLOCK time, not poll wall-time: blocks arrive
                        # every 0.2 s but only ~1 poll in 4 sees fresh
                        # audio, so += dt undercounted speech 4× and the
                        # MIN_SPEECH gate silently discarded every
                        # answer shorter than ~1.4 s ("yes", "next"…)
                        speech_s += len(fresh) * (BLOCK / SR)
                        quiet = 0.0
                        sgen += 1
                    elif heard:
                        quiet += dt
                    else:              # adapt to the room while waiting
                        self._noise = min(
                            0.02, 0.9 * self._noise + 0.1 * level)
                elif heard:
                    quiet += dt
                if heard and 0.30 <= quiet < self.endpoint_s:
                    # speculative STT at the first plausible pause —
                    # the transcript is ready AT the endpoint
                    spec.launch(sgen, np.concatenate(chunks))
                if heard and quiet >= self.endpoint_s:
                    break
                if not heard and now - t0 >= self.window_s:
                    break              # nobody spoke — the window expires
                if now - t0 >= self.max_s:
                    break
        finally:
            try:
                source.stop()
                source.unsubscribe(self._on_block)
            except Exception:                               # noqa: BLE001
                pass
        if self._cancel.is_set():
            self._active.clear()
            self.on_status("")
            return
        if not heard or not chunks or speech_s < self.MIN_SPEECH_S:
            # <0.35 s of detected speech = a click / room noise, not an
            # utterance — 30 s of it fed to Whisper used to hallucinate
            # a phantom user turn
            self._active.clear()
            self.on_status("")           # never leave "just talk" stale
            self.on_timeout()
            return
        audio = np.concatenate(chunks).reshape(-1)
        if not getattr(self.stt, "ready", True):
            self.on_status("heard you — voice model warming up "
                           "(one-time), no need to repeat")
        text = spec.take(sgen) or ""
        if not text:
            try:
                text = self.stt.transcribe(audio)
            except Exception:                               # noqa: BLE001
                pass
        if self._cancel.is_set():        # cancelled DURING the final
            self._active.clear()         # transcription — a stale turn
            return                       # must not fire into the panel
        self.on_status("")
        self._active.clear()
        self.on_text(text)


# ---------------------------------------------------------------------------
# barge-in (stream-free passive analyzer)
# ---------------------------------------------------------------------------
class BargeListener:
    """Cut the assistant's speech by talking over it — STREAM-FREE.

    A passive analyzer owning NO microphone: it is fed blocks through a
    :class:`MicStream` tap, so arming/disarming never starts or stops an
    audio stream (the CoreAudio HAL-mutex rule).  Echo mitigation for
    open-speaker rigs: the first ``calib_blocks`` after arming learn the
    speaker-bleed floor (MAX, so the player's silent spin-up can't drag
    it under the real bleed); a barge fires only on ``sustain_blocks``
    consecutive blocks above ``max(bleed_ratio·floor, abs_floor)``.
    One shot per arm; ``on_barge`` runs on a fresh thread."""

    def __init__(self, on_barge, calib_blocks=5, sustain_blocks=2,
                 bleed_ratio=2.5, abs_floor=0.02, vad=None):
        #: optional speech gate: a barge must be loud AND speech — a
        #: dropped mug no longer cuts the voice.  The level calibration
        #: still does the speaker-bleed separation (TTS bleed IS speech,
        #: so VAD alone could never arbitrate talk-over).
        self.vad = vad
        self.on_barge = on_barge
        self.calib_blocks = int(calib_blocks)
        self.sustain_blocks = int(sustain_blocks)
        self.bleed_ratio = float(bleed_ratio)
        self.abs_floor = float(abs_floor)
        self._lock = threading.Lock()
        self._armed = False
        self._floor = 0.0
        self._n_calib = 0
        self._hot = 0

    @property
    def active(self) -> bool:
        return self._armed

    def arm(self) -> bool:
        with self._lock:
            if self._armed:
                return False
            self._armed = True
            self._floor = 0.0
            self._n_calib = 0
            self._hot = 0
        return True

    def disarm(self) -> None:
        with self._lock:
            self._armed = False

    def feed(self, block) -> None:
        """Called from the mic reader thread — keep tiny."""
        with self._lock:
            if not self._armed:
                return
            rms = _rms(block)
            if self._n_calib < self.calib_blocks:
                self._floor = max(self._floor, rms)
                self._n_calib += 1
                return
            thr = max(self.bleed_ratio * self._floor, self.abs_floor)
            hot = rms > thr
            if hot and self.vad is not None:
                try:
                    hot = self.vad.prob(block) > 0.6
                except Exception:                           # noqa: BLE001
                    pass
            if hot:
                self._hot += 1
                if self._hot < self.sustain_blocks:
                    return
                self._armed = False          # one shot per arm
            else:
                self._hot = 0
                self._floor = max(self._floor * 0.98, rms)
                return
        # fired — leave the reader thread immediately
        threading.Thread(target=self.on_barge, daemon=True,
                         name="assist-barge-fire").start()
