"""Voice layer for the assistant — OFFLINE, push-to-talk.

Speech in, speech out, wrapped around the *unchanged* agent loop:

    mic → record → transcribe (faster-whisper) → agent.ask(text)
        → reply → summarize_for_speech → speak (piper / macOS say)

Everything is optional and lazy-imported: with no audio stack the
assistant is text-only and nothing here is touched.  Engines degrade
gracefully — Piper if installed (best offline voice), else macOS
``say`` (offline, zero-dependency), else pyttsx3, else silent.

PRECISION SAFETY: beam physics is full of exact numbers/signs/units
that voice would mangle, so the assistant SPEAKS a concise summary
while the exact digits and the echoed tool call stay on screen —
:func:`summarize_for_speech` enforces that.  You talk to it and hear
the gist; you never rely on *hearing* a critical value.
"""
from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
import threading

# ---------------------------------------------------------------------------
# PortAudio lifecycle serialization
# ---------------------------------------------------------------------------
#: Process-wide serializer for input-stream lifecycle transitions
#: (create/start and stop/close).  Concurrent transitions on different
#: threads can deadlock CoreAudio's HAL mutex (observed: GUI thread
#: frozen inside AudioDeviceStop).  Timed acquire — a wedged transition
#: must degrade to *unserialized*, never freeze other threads.
_PA_LIFECYCLE_LOCK = threading.Lock()


@contextlib.contextmanager
def pa_lifecycle(timeout: float = 2.0):
    got = _PA_LIFECYCLE_LOCK.acquire(timeout=timeout)
    try:
        yield
    finally:
        if got:
            _PA_LIFECYCLE_LOCK.release()


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------
def stt_available() -> bool:
    try:
        import faster_whisper           # noqa: F401
        import sounddevice              # noqa: F401
        return True
    except Exception:                                       # noqa: BLE001
        return False


def tts_available() -> bool:
    return bool(_pick_tts_engine())


def _pick_tts_engine() -> str:
    """Best available offline TTS: piper > macOS say > pyttsx3."""
    try:
        import piper                    # noqa: F401
        return "piper"
    except Exception:                                       # noqa: BLE001
        pass
    if shutil.which("say"):             # macOS, offline, built-in
        return "say"
    try:
        import pyttsx3                   # noqa: F401
        return "pyttsx3"
    except Exception:                                       # noqa: BLE001
        return ""


def voice_status() -> str:
    stt = "faster-whisper" if stt_available() else "unavailable"
    tts = _pick_tts_engine() or "unavailable"
    return f"speech-in: {stt} · speech-out: {tts}"


# ---------------------------------------------------------------------------
# speech → text (faster-whisper)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


class WhisperSTT:
    """Local Whisper transcription via faster-whisper (CTranslate2).

    The model is downloaded once on first use and cached; ``model_size``
    trades speed for accuracy.  Default ``small.en`` (what MIRAGE runs):
    the English-only variant hears wake words and commands noticeably
    better than multilingual ``base``, and short utterances still
    transcribe in well under a second on CPU int8."""

    def __init__(self, model_size: str = "small.en", device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    @property
    def ready(self) -> bool:
        """True once the model is loaded — surfaces use this for the
        'heard you — voice model warming up, no need to repeat' cue."""
        return ((self.model_size, self.device, self.compute_type)
                in _MODEL_CACHE)

    def _model(self):
        key = (self.model_size, self.device, self.compute_type)
        if key not in _MODEL_CACHE:
            from faster_whisper import WhisperModel
            _MODEL_CACHE[key] = WhisperModel(
                self.model_size, device=self.device,
                compute_type=self.compute_type)
        return _MODEL_CACHE[key]

    def transcribe(self, audio, samplerate: int = 16000) -> str:
        """``audio``: float32 mono numpy array at ``samplerate`` Hz."""
        import numpy as np
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        if a.size < int(0.3 * samplerate):      # too short to be speech
            return ""
        segments, _info = self._model().transcribe(a, language="en",
                                                   beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------------------
# text → speech (piper / say / pyttsx3)
# ---------------------------------------------------------------------------
class TextToSpeech:
    def __init__(self, engine: str | None = None,
                 piper_voice: str | None = None, rate_wpm: int = 190):
        self.engine = engine or _pick_tts_engine()
        self.piper_voice = piper_voice
        self.rate_wpm = rate_wpm
        self._proc = None

    def speak(self, text: str, blocking: bool = False) -> None:
        text = (text or "").strip()
        if not text or not self.engine:
            return
        if self.engine == "say":
            args = ["say", "-r", str(self.rate_wpm), text]
            if blocking:
                subprocess.run(args, check=False)
            else:
                self._proc = subprocess.Popen(args)
        elif self.engine == "piper":
            self._speak_piper(text, blocking)
        elif self.engine == "pyttsx3":
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", self.rate_wpm)
            eng.say(text)
            eng.runAndWait()

    def _speak_piper(self, text, blocking):
        import io
        import wave

        from piper import PiperVoice
        if not self.piper_voice:
            return
        voice = PiperVoice.load(self.piper_voice)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize(text, wf)
        buf.seek(0)
        import numpy as np
        import sounddevice as sd
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        data = np.frombuffer(frames, dtype=np.int16)
        sd.play(data, sr)
        if blocking:
            sd.wait()

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:                                   # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# streaming Speaker — sentence queue with a natural-voice backend chain
# ---------------------------------------------------------------------------
import os as _os
import queue as _queue

#: drop `kokoro*.onnx` + `voices*.bin` here for the natural voice
MODELS_DIR = _os.path.join(_os.path.expanduser("~"), ".helix",
                           "assistant_models")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# TTS-safe numbers.  kokoro's grapheme-to-phoneme step drops a bare
# decimal point — "0.62" is spoken "zero … six two" with no "point" —
# so every number is rewritten with the point made explicit before it
# reaches any backend (macOS ``say`` reads both forms correctly).
_DEC_POINT = re.compile(r"(\d)\.(?=\d)")
_SCI_NOTE = re.compile(r"(\d+(?:\.\d+)?)[eE]([+-]?)0*(\d+)")
_UNI_MINUS = re.compile(r"[−](?=\s*\d)")


def speakable_numbers(text: str) -> str:
    """Rewrite numeric notation into words a TTS engine speaks reliably:
    ``21.24`` → ``21 point 24``; ``2.1e+04`` → ``2 point 1 times ten to
    the 4``; a unicode minus before a digit → ``minus``."""
    t = _SCI_NOTE.sub(
        lambda m: (f"{m.group(1)} times ten to the "
                   f"{'minus ' if m.group(2) == '-' else ''}{m.group(3)}"),
        str(text))
    t = _UNI_MINUS.sub("minus ", t)
    return _DEC_POINT.sub(r"\1 point ", t)


# Markdown must never be SPOKEN (ported from the MIRAGE assistant's
# clean_for_speech — "asterisk asterisk two point one four" was its
# bug too).  Word-boundary guards keep identifiers like emit_x intact.
_MD_FENCE = re.compile(r"```.*?(```|$)", re.S)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_CODE = re.compile(r"`([^`\n]*)`")
_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.M)
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_MD_ITALIC = re.compile(r"(?<![\w*])[*_]([^*_\n]+)[*_](?![\w*])")
_MD_STRAY = re.compile(r"(?<!\w)[*#`]+|[*#`]+(?!\w)")


def strip_markdown_for_speech(text: str) -> str:
    """Remove markdown SYNTAX, keep its content: **bold** → bold,
    [label](url) → label, `code` → code, headers/bullets unwrapped,
    fenced blocks dropped entirely.  ``emit_x``-style identifiers are
    never mangled (word-boundary guards)."""
    t = str(text)
    t = _MD_FENCE.sub(" ", t)
    t = _MD_IMAGE.sub(" ", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _MD_CODE.sub(r"\1", t)
    t = _MD_HEADER.sub("", t)
    t = _MD_BULLET.sub("", t)
    t = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2) or "", t)
    t = _MD_ITALIC.sub(r"\1", t)
    t = _MD_STRAY.sub(" ", t)
    return re.sub(r"[ \t]+", " ", t)


# Units → how a physicist's voice should say them (multi-char symbols
# only, so bare A/V/T/m/s are never mangled); longest-first order.
# HELIX's dialect: emittances in pi.mm.mrad, longitudinal in deg.MeV.
_UNIT_WORDS = [
    ("pi.mm.mrad", "pi millimeter milliradian"),
    ("π.mm.mrad", "pi millimeter milliradian"),
    ("pi mm mrad", "pi millimeter milliradian"),
    ("deg.MeV", "degree M e V"),
    ("deg·MeV", "degree M e V"),
    ("MeV", "M e V"), ("keV", "k e V"), ("GeV", "G e V"),
    ("mrad", "milliradians"), ("mA", "milliamps"), ("uA", "microamps"),
    ("µA", "microamps"),
    ("MHz", "megahertz"), ("kHz", "kilohertz"),
    ("mm", "millimeters"), ("cm", "centimeters"),
    ("ns", "nanoseconds"), ("µs", "microseconds"), ("ms", "milliseconds"),
    ("deg", "degrees"),
]
_IDENT_PUNCT = re.compile(r"(?<=\w)[:/](?=\w)")
_IDENT_DASH = re.compile(r"(?<=[A-Za-z])-(?=\d)")


def normalize_units_for_speech(text: str) -> str:
    """Expand number-adjacent unit symbols into words and turn
    identifier punctuation into pauses (``QF-3`` → ``QF 3``,
    ``HWR:CAV1`` → ``HWR CAV1``) so the voice never says "colon" or
    runs symbols into digits.  Applied BEFORE ``speakable_numbers`` —
    unit matching must see the raw digits, not "21 point 24"."""
    t = str(text)
    t = t.replace("%", " percent").replace("°", " degrees")
    for sym, word in _UNIT_WORDS:
        t = re.sub(rf"(?<=[\d\s]){re.escape(sym)}\b", word, t)
    t = _IDENT_PUNCT.sub(" ", t)
    t = _IDENT_DASH.sub(" ", t)
    t = t.replace("—", ", ")
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _kokoro_files():
    """Any (model.onnx, voices*.bin) pair dropped into MODELS_DIR."""
    import glob
    models = sorted(glob.glob(_os.path.join(MODELS_DIR, "kokoro*.onnx")))
    voices = sorted(glob.glob(_os.path.join(MODELS_DIR, "voices*.bin")))
    return (models[-1], voices[-1]) if models and voices else (None, None)


class _KokoroBackend:
    """Natural neural voice via kokoro-onnx (fully local).

    Playback uses ONE persistent OutputStream with chunked BLOCKING
    writes and a high-latency buffer: ``sd.play()`` per sentence opened
    and closed a fresh stream each time (audible pops at every sentence
    boundary) and its small buffer underruns — crackles — whenever the
    CPU is busy (next-sentence ONNX synthesis, GUI streaming).  Blocking
    writes also give barge-in a ~0.2 s cut granularity."""

    _CHUNK_S = 0.2                       # write granularity (cut latency)

    def __init__(self):
        from kokoro_onnx import Kokoro
        model, voices = _kokoro_files()
        self._k = Kokoro(model, voices)
        self._active = threading.Event()
        self._cancel = threading.Event()
        self._out = None                 # persistent OutputStream

    def _stream(self, sr: int):
        import sounddevice as sd
        if self._out is not None and getattr(self._out, "samplerate",
                                             None) == sr:
            return self._out
        if self._out is not None:
            try:
                self._out.close()
            except Exception:                               # noqa: BLE001
                pass
        self._out = sd.OutputStream(samplerate=sr, channels=1,
                                    dtype="float32", latency="high")
        self._out.start()
        return self._out

    def speak(self, text: str) -> None:
        import numpy as np
        self._cancel.clear()
        samples, sr = self._k.create(text, voice="af_sarah", speed=1.05)
        if self._cancel.is_set():        # cut arrived during synthesis
            return
        data = np.ascontiguousarray(
            np.asarray(samples, dtype="float32").reshape(-1, 1))
        self._active.set()
        try:
            out = self._stream(int(sr))
            block = max(1, int(sr * self._CHUNK_S))
            for i in range(0, len(data), block):
                if self._cancel.is_set():
                    try:                 # drop what's still buffered
                        out.abort()
                        out.start()
                    except Exception:                       # noqa: BLE001
                        pass
                    break
                out.write(data[i:i + block])
        except Exception:                                   # noqa: BLE001
            # a dead device/stream must not kill the player thread; drop
            # the stream so the next sentence reopens it fresh
            try:
                if self._out is not None:
                    self._out.close()
            except Exception:                               # noqa: BLE001
                pass
            self._out = None
        finally:
            self._active.clear()

    def cut(self) -> None:
        self._cancel.set()               # covers synthesis + write loop


class _SayBackend:
    """macOS `say` via Popen so cut() can terminate mid-sentence."""

    def __init__(self):
        self._proc = None

    def speak(self, text: str) -> None:
        self._proc = subprocess.Popen(["say", "-r", "195", text])
        self._proc.wait()
        self._proc = None

    def cut(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            p.terminate()


class _FallbackBackend:
    """Last resort: the basic TextToSpeech chain (pyttsx3 etc.).
    Not interruptible mid-sentence — cut() only prevents queued ones."""

    def __init__(self):
        self._tts = TextToSpeech()

    def speak(self, text: str) -> None:
        self._tts.speak(text, blocking=True)

    def cut(self) -> None:
        self._tts.stop()


def pick_speaker_backend():
    """(backend, name) — kokoro when its model files + package exist,
    then macOS `say` (cuttable), then the basic engine chain."""
    if _kokoro_files()[0]:
        try:
            return _KokoroBackend(), "kokoro"
        except Exception:                                   # noqa: BLE001
            pass
    if shutil.which("say"):
        return _SayBackend(), "say"
    return _FallbackBackend(), _pick_tts_engine() or "none"


class Speaker:
    """Queue sentences; a daemon thread plays them in order.

    ``say(text)`` splits into sentences and queues them (so streaming
    replies are spoken as they complete); ``stop()`` flushes the queue and
    cuts the current audio — the push-to-talk barge-in.  ``on_speaking``
    is called (from the player thread!) with True/False around playback."""

    def __init__(self, on_speaking=None):
        self.on_speaking = on_speaking or (lambda active: None)
        self.backend, self.backend_name = pick_speaker_backend()
        self._q: _queue.Queue = _queue.Queue()
        self._muted = False
        # stop() bumps the generation: sentences queued before the stop are
        # skipped even if already dequeued — race-free barge-in.
        self._gen = 0
        # speaking-state machine: ``on_speaking`` fires only on actual
        # transitions.  Between begin_turn()/end_turn() an EMPTY QUEUE is
        # a sentence gap in a streamed reply, not the end of speech — the
        # MIRAGE lesson: flapping the state per sentence resets the
        # barge-in bleed calibration, drops the wake pause (the assistant
        # hears itself say "HELIX"), and flickers the orb.
        self._state_lock = threading.Lock()
        self._turn_open = False
        self._reported = False
        self._playing = False
        threading.Thread(target=self._player, daemon=True,
                         name="assist-tts").start()

    def say(self, text: str) -> None:
        if self._muted:
            return
        # THE speech choke point: markdown syntax out first (never say
        # "asterisk"), then units (they must see raw digits), then
        # number rewriting
        text = strip_markdown_for_speech(str(text).strip())
        text = normalize_units_for_speech(text)
        text = speakable_numbers(text)
        for sent in _SENT_SPLIT.split(text):
            sent = sent.strip()
            if sent:
                self._q.put((self._gen, sent))

    def stop(self) -> None:
        self._gen += 1                   # invalidate everything queued so far
        try:
            while True:
                self._q.get_nowait()
        except _queue.Empty:
            pass
        try:
            self.backend.cut()
        except Exception:                                   # noqa: BLE001
            pass
        # a stop between sentences leaves no player pass to report the
        # drain — close the speaking state here (skip if a sentence is
        # mid-play: its own teardown or end_turn() will report)
        with self._state_lock:
            done = (self._reported and not self._playing
                    and self._q.empty() and not self._turn_open)
            if done:
                self._reported = False
        if done:
            try:
                self.on_speaking(False)
            except Exception:                               # noqa: BLE001
                pass

    def begin_turn(self) -> None:
        """Bracket a streamed reply: until :meth:`end_turn`, queue
        drains are sentence gaps and must NOT report speaking-stopped."""
        with self._state_lock:
            self._turn_open = True

    def end_turn(self) -> None:
        """Close the bracket; if speech already drained, report the
        pending speaking-stopped exactly once."""
        with self._state_lock:
            self._turn_open = False
            done = (self._reported and self._q.empty()
                    and not self._playing)
            if done:
                self._reported = False
        if done:
            try:
                self.on_speaking(False)
            except Exception:                               # noqa: BLE001
                pass

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if muted:
            self.stop()

    def _player(self) -> None:
        while True:
            gen, sent = self._q.get()
            if gen != self._gen:         # barge-in invalidated this sentence
                continue
            with self._state_lock:
                self._playing = True
                announce = not self._reported
                self._reported = True
            if announce:                 # transition only, not per sentence
                try:
                    self.on_speaking(True)
                except Exception:                           # noqa: BLE001
                    pass
            try:
                self.backend.speak(sent)
            except Exception:                               # noqa: BLE001
                pass
            finally:
                with self._state_lock:
                    self._playing = False
                    done = (self._q.empty() and not self._turn_open
                            and self._reported)
                    if done:
                        self._reported = False
                if done:
                    try:
                        self.on_speaking(False)
                    except Exception:                       # noqa: BLE001
                        pass


# ---------------------------------------------------------------------------
# push-to-talk recorder (sounddevice)
# ---------------------------------------------------------------------------
class PushToTalkRecorder:
    """Hold-to-record via sounddevice's **blocking** read API.

    Recording runs on a dedicated worker thread that copies audio with
    ``stream.read()``.  Crucially, **no Python callback is ever invoked
    on PortAudio's real-time audio thread** — a Python callback on the RT
    thread is the classic use-after-free that hard-crashes the host
    process (SIGSEGV in ``general_invoke_callback``) when the stream is
    torn down.  Blocking reads keep all Python on our own thread, so the
    mic path can never take down the app.  The mic is live only between
    :meth:`start` and :meth:`stop`."""

    #: recorders whose reader thread wedged (CoreAudio can hang a
    #: blocking read after system sleep/wake).  Parked = strong ref so
    #: GC can never finalize (close) the stream under the stuck reader
    #: — closing a stream while a reader is inside ReadStream frees the
    #: ring buffer mid-memmove and SIGSEGVs the process (observed).
    _ABANDONED: list = []

    def __init__(self, samplerate: int = 16000, blocksize: int = 512,
                 on_level=None):
        # 512 frames = 32 ms at 16 kHz: the reader's blocking read is
        # never far from returning, so stop() joins almost instantly
        # (with 2048-frame blocks a cold device could hold the first
        # read >1 s) and the level meter is smooth.
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._on_level = on_level        # rms per block (recorder thread!)
        self._stream = None
        self._chunks: list = []
        self._stop = threading.Event()
        self._thread = None
        self._join_timeout_s = 3.0

    def start(self) -> None:
        import sounddevice as sd
        if self._thread is not None and self._thread.is_alive():
            return                                  # already recording
        if self._stream is not None:
            # a previous (wedged) reader still owns a stream — starting
            # again would swap self._stream under it; never reuse
            return
        self._chunks = []
        self._stop.clear()
        # open on the caller so device errors surface synchronously
        with pa_lifecycle():
            self._stream = sd.InputStream(
                samplerate=self.samplerate, channels=1, dtype="float32",
                blocksize=self.blocksize)
            self._stream.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # blocking reads on THIS worker thread — never the RT audio thread
        import numpy as np
        while not self._stop.is_set():
            try:
                data, _overflow = self._stream.read(self.blocksize)
            except Exception:                               # noqa: BLE001
                break
            self._chunks.append(data.copy())
            if self._on_level is not None:
                try:                     # live mic level (drives the orb)
                    self._on_level(float(np.sqrt(np.mean(data ** 2))))
                except Exception:                           # noqa: BLE001
                    pass
        # THE READER closes the stream — nobody else, ever.  Closing
        # from another thread while this one is inside stream.read()
        # frees PortAudio's ring buffer mid-memmove (SIGSEGV, observed
        # after a system sleep/wake wedged the read).
        with pa_lifecycle():
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:                               # noqa: BLE001
                pass
        self._stream = None

    def stop(self):
        import numpy as np
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_s)
            if self._thread.is_alive():
                # Reader wedged (CoreAudio post-sleep hang).  Do NOT
                # close the stream from here and do NOT let GC finalize
                # it: park a strong reference forever.  A leaked wedged
                # stream is harmless; a freed one is a crash.
                PushToTalkRecorder._ABANDONED.append(self)
            self._thread = None
        chunks = list(self._chunks)     # snapshot — a wedged reader may
        if not chunks:                  # still append after we return
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks).reshape(-1)


# ---------------------------------------------------------------------------
# precision safety: what the assistant SPEAKS (summary, not exact data)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def summarize_for_speech(text: str, max_chars: int = 320) -> str:
    """Concise spoken form of an assistant reply.

    Drops code/JSON/table blocks and long numeric runs (those stay on
    screen), keeps the prose gist, and rounds any spoken number to a
    coarse value so a mis-heard digit can never masquerade as an exact
    result."""
    if not text:
        return ""
    lines = []
    in_fence = False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("```"):          # code-fence toggle — drop fences
            in_fence = not in_fence
            continue
        if in_fence or not s:
            continue
        # drop table rows, bracketed data, and JSON-ish lines
        if s.startswith(("|", "{", "}", "[")):
            continue
        if re.search(r"[=:]\s*[\[\{]", s):    # foo = [...] / "k": {...}
            continue
        lines.append(s)
    spoken = " ".join(lines)

    def _round(m):
        v = m.group(0).replace(",", "")
        try:
            f = float(v)
        except ValueError:
            return m.group(0)
        if f == int(f):
            return str(int(f))
        return f"about {f:.2g}"

    spoken = _NUM.sub(_round, spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    if len(spoken) > max_chars:
        cut = spoken[:max_chars].rsplit(" ", 1)[0]
        spoken = cut + " … (details on screen)"
    return spoken
