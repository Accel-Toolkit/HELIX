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
import tempfile
import subprocess
import threading
import time

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
_MODEL_CACHE_LOCK = threading.Lock()


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
        # locked: prewarm + wake + follow-up can race first use — two
        # concurrent WhisperModel loads double memory and one is wasted
        with _MODEL_CACHE_LOCK:
            if key not in _MODEL_CACHE:
                from faster_whisper import WhisperModel
                _MODEL_CACHE[key] = WhisperModel(
                    self.model_size, device=self.device,
                    compute_type=self.compute_type)
            return _MODEL_CACHE[key]

    _INFER_LOCK = threading.Lock()   # spec-STT + final transcribe share
    #                                  ONE WhisperModel — serialize inference

    def transcribe(self, audio, samplerate: int = 16000) -> str:
        """``audio``: float32 mono numpy array at ``samplerate`` Hz."""
        import numpy as np
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        if a.size < int(0.3 * samplerate):      # too short to be speech
            return ""
        with WhisperSTT._INFER_LOCK:
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


# --- numbers spoken as WORDS -------------------------------------------
# kokoro's grapheme-to-phoneme step is unreliable on multi-digit
# numerals (verified live: "297.6" was voiced "ninety seven" — the
# hundreds digit AND the fraction dropped).  Digits must never reach
# the engine: integers become English words, fractional digits are
# spelled one by one ("297.65" → "two hundred ninety seven point six
# five").

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
         "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"),
           (10 ** 6, "million"), (10 ** 3, "thousand"), (100, "hundred")]


def _int_words(n: int) -> str:
    """0 ≤ n < 10**15 → English words (no 'and', TTS-friendly)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + (f" {_ONES[r]}" if r else "")
    for scale, name in _SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            out = f"{_int_words(head)} {name}"
            return f"{out} {_int_words(rest)}" if rest else out
    return _ONES[0]                                     # unreachable


_DEC_TRIM = re.compile(r"(\d+)\.(\d{3,})")


def _trim_spoken_decimal(m: re.Match) -> str:
    """MIRAGE parity: the VOICE says at most 2 fraction digits
    ("800.638" -> "800.63"); tiny values keep digits through the first
    non-zero so 0.0021 never collapses to "zero point zero zero"."""
    frac = m.group(2)
    keep = 2
    if set(frac[:2]) == {"0"}:
        nz = next((i for i, d in enumerate(frac) if d != "0"), len(frac) - 1)
        keep = min(nz + 2, len(frac))
    return f"{m.group(1)}.{frac[:keep]}"


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)*")


def _number_words(m: re.Match) -> str:
    tok = m.group(0)
    parts = tok.replace(",", "").split(".")
    try:
        head = int(parts[0])
    except ValueError:                                  # noqa: PERF203
        return tok
    if head >= 10 ** 15:                # astronomical: spell digit-wise
        spoken = " ".join(_ONES[int(d)] for d in parts[0])
    else:
        spoken = _int_words(head)
    for frac in parts[1:]:              # each .xxx group digit by digit
        spoken += " point " + " ".join(_ONES[int(d)] for d in frac)
    return spoken


# --- symbols spoken as WORDS -------------------------------------------
# The phonemizer silently DROPS glyphs it has no phonemes for — σ_x was
# voiced as just "x".  Every Greek letter physicists write must reach
# the engine as its name.
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "Γ": "gamma",
    "δ": "delta", "Δ": "delta", "∆": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "Θ": "theta",
    "λ": "lambda", "Λ": "lambda", "μ": "mu", "µ": "mu", "ν": "nu",
    "π": "pi", "ρ": "rho", "σ": "sigma", "Σ": "sigma", "τ": "tau",
    "φ": "phi", "Φ": "phi", "χ": "chi", "ψ": "psi", "Ψ": "psi",
    "ω": "omega", "Ω": "omega", "±": " plus or minus ", "≈": " about ",
    "→": " to ", "≥": " at least ", "≤": " at most ",
}
# subscript tokens a physicist reads letter-by-letter (or as "prime")
_SUBSCRIPT = re.compile(r"\b(xx|yy|zz|xy|yx|nx|ny|xp|yp)\b")
_SUB_WORDS = {"xx": "x x", "yy": "y y", "zz": "z z", "xy": "x y",
              "yx": "y x", "nx": "n x", "ny": "n y",
              "xp": "x prime", "yp": "y prime"}
_ASCII_MINUS = re.compile(r"(?<![\w.])-(?=\d)")


def speechify(text: str) -> str:
    """THE speech normalizer — everything ``say()`` voices goes through
    here, in an order where each stage still sees what it needs:

    1. markdown syntax out (never say "asterisk");
    2. scientific notation → words (BEFORE the identifier-dash rule,
       which used to eat the exponent's minus: 2.1e-3 → "2.1e 3");
    3. units (they must see raw digits);
    4. Greek letters / comparison symbols → their names;
    5. underscores → spaces, subscripts letter-by-letter
       (sigma_x → "sigma x", σ_xx → "sigma x x");
    6. every numeral → English words (kokoro misreads digits);
    7. a leading ASCII minus → "minus".
    """
    t = strip_markdown_for_speech(str(text).strip())
    t = _SCI_NOTE.sub(
        lambda m: (f"{m.group(1)} times ten to the "
                   f"{'minus ' if m.group(2) == '-' else ''}{m.group(3)}"),
        t)
    t = normalize_units_for_speech(t)
    for glyph, word in _GREEK.items():
        if glyph in t:
            t = t.replace(glyph, f" {word} ")
    t = t.replace("_", " ")
    t = t.replace(" - ", ", ")          # MIRAGE parity: dash = pause
    t = _SUBSCRIPT.sub(lambda m: _SUB_WORDS[m.group(1)], t)
    t = _DEC_TRIM.sub(_trim_spoken_decimal, t)
    t = _UNI_MINUS.sub("minus ", t)
    t = _ASCII_MINUS.sub("minus ", t)
    t = _NUMBER.sub(_number_words, t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return re.sub(r"[ \t]+", " ", t).strip()


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
    ("deg.MeV", "degree mega-electron-volts"),
    ("deg·MeV", "degree mega-electron-volts"),
    ("MeV", "mega-electron-volts"), ("keV", "kilo-electron-volts"),
    ("GeV", "giga-electron-volts"),
    ("kV", "kilovolts"), ("MV", "megavolts"),
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


def _edge_fade(samples, sr: int, fade_s: float = 0.005):
    """5 ms linear fade at both ends — kills the boundary click every
    raw clip start/stop produces (MIRAGE parity)."""
    import numpy as np
    n = min(int(fade_s * sr), samples.size // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        samples[:n] *= ramp
        samples[-n:] *= ramp[::-1]
    return samples


def _write_wav(samples, sr: int, path: str) -> None:
    """float32 [-1,1] mono → 16-bit PCM wav via stdlib (no soundfile dep)."""
    import wave

    import numpy as np
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def _afplay_available() -> bool:
    import sys as _sys
    return _sys.platform == "darwin" and bool(shutil.which("afplay"))


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

    def prepare(self, text: str):
        """Synthesis only, no audio out — runs on the player's prefetch
        thread while the PREVIOUS sentence is still playing, so the
        next sentence starts with zero synthesis gap (the single
        biggest 'MIRAGE feels smoother' factor)."""
        import numpy as np
        samples, sr = self._k.create(text, voice="af_sarah", speed=1.05)
        return (_edge_fade(np.asarray(samples, dtype="float32")
                           .reshape(-1).copy(), int(sr)), int(sr))

    def speak(self, text: str) -> None:
        self._cancel.clear()
        prepared = self.prepare(text)
        if self._cancel.is_set():        # cut arrived during synthesis
            return
        self.play(prepared)

    def play(self, prepared) -> None:
        import numpy as np
        flat, sr = prepared
        # lifecycle start for this sentence: re-arm after any earlier cut
        # (stale queued audio is already dropped by Speaker's generation
        # check before play() is ever called)
        self._cancel.clear()
        # macOS: hand playback to afplay — a separate OS process the
        # Python side cannot starve.  In-process sounddevice writes
        # underrun (crackle) whenever ONNX synthesis / Whisper / the GUI
        # saturate the cores and the GIL; MIRAGE hit exactly this and
        # the afplay hand-off is her proven cure.
        if _afplay_available():
            self._active.set()
            try:
                self._n = getattr(self, "_n", 0) + 1
                path = _os.path.join(tempfile.gettempdir(),
                                     f"helix_tts_{self._n % 4}.wav")
                _write_wav(flat, int(sr), path)
                if self._cancel.is_set():
                    return
                self._proc = proc = subprocess.Popen(["afplay", path])
                # Watchdog wait: a wedged afplay must never hold the
                # player (and with it the mic-pause gate) forever — cap
                # at clip length + 2 s, then terminate → kill.  The
                # 0.25 s poll also closes the cut()-vs-Popen race: a
                # cancel that misses _proc is honoured within a beat.
                deadline = (time.monotonic()
                            + len(flat) / max(int(sr), 1) + 2.0)
                while True:
                    try:
                        proc.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if (self._cancel.is_set()
                                or time.monotonic() > deadline):
                            proc.terminate()
                            try:
                                proc.wait(timeout=0.5)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            break
                # (device-buffer drain grace moved to the PLAYER, end of
                # run only — per-sentence it was ~0.25 s dead air at
                # every sentence boundary)
                self._proc = None
            except Exception:                               # noqa: BLE001
                pass
            finally:
                self._active.clear()
            return
        data = np.ascontiguousarray(flat.reshape(-1, 1))
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
        p = getattr(self, "_proc", None)
        if p is not None:                # afplay path: instant cut
            try:
                p.terminate()
            except Exception:                               # noqa: BLE001
                pass


class _SayBackend:
    """macOS `say` via Popen so cut() can terminate mid-sentence."""

    def __init__(self):
        self._proc = None

    def speak(self, text: str) -> None:
        self._proc = proc = subprocess.Popen(["say", "-r", "195", text])
        # bounded wait (same wedge class as the afplay watchdog): a
        # hung `say` must never hold the player/mic-pause gate forever
        # ~16 chars/s at 195 wpm; generous headroom, never unbounded
        deadline = time.monotonic() + max(10.0, len(text) / 10.0 + 5.0)
        while True:
            try:
                proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() > deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
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
        # a sentence pulled from the queue for prefetch but not yet
        # played — MUST count as pending speech everywhere (the drain
        # checks once missed it: the follow-up mic opened between
        # sentences and the assistant heard its own next sentence)
        self._ahead_pending = False
        # sentences DEQUEUED by the player but not yet through play() —
        # covers the first-sentence synthesis window (0.3–3 s) where
        # every other flag is false: busy() read False there, the
        # follow-up mic opened ON TOP of the reply, recorded the
        # assistant, and wake went deaf for the whole cascade (audit R1
        # — the root cause behind all three live complaints)
        self._pending_play = 0
        # a stop() flushes the pipeline: the drain may then report even
        # inside an open turn (barge/Stop used to wedge the mic-pause
        # gate until turn end)
        self._flushed = False
        threading.Thread(target=self._player, daemon=True,
                         name="assist-tts").start()

    def say(self, text: str) -> None:
        if self._muted:
            return
        with self._state_lock:
            self._flushed = False        # fresh speech: turn rules apply
        # THE speech choke point — full normalization chain (order
        # documented on speechify)
        text = speechify(text)
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
        # mid-play: its own teardown will report).  _flushed lets the
        # drain report even inside an OPEN turn: a barge/Stop used to
        # leave _reported wedged True until turn end, pausing the mic
        # for the rest of a possibly long turn.
        with self._state_lock:
            self._flushed = True
            done = (self._reported and not self._playing
                    and not self._ahead_pending
                    and self._pending_play == 0
                    and self._q.empty())
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
        pending speaking-stopped exactly once.  A PREFETCHED sentence
        (outside the queue, not yet playing) blocks the drain — this
        was the she-hears-herself bug: the follow-up mic opened in the
        gap and captured the assistant's own next sentence."""
        with self._state_lock:
            self._turn_open = False
            done = (self._reported and self._q.empty()
                    and not self._ahead_pending
                    and self._pending_play == 0
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

    def busy(self) -> bool:
        """Live speech state, DERIVED from the player — never mirror
        this into a stored flag (a missed transition once wedged the
        mic-pause gate forever).  A PREFETCHED sentence counts, and so
        does one the player has DEQUEUED but not yet played (the
        first-sentence synthesis window — audit R1)."""
        with self._state_lock:
            return (self._playing or self._reported
                    or self._ahead_pending or self._pending_play > 0
                    or not self._q.empty())

    def _prefetch(self):
        """One-deep lookahead: pull the next queued sentence and start
        synthesizing it on a side thread NOW, while the current one is
        about to play.  Returns (gen, box) or None."""
        if not (hasattr(self.backend, "prepare")
                and hasattr(self.backend, "play")):
            return None
        try:
            ngen, nsent = self._q.get_nowait()
        except _queue.Empty:
            return None
        with self._state_lock:
            self._ahead_pending = True   # out of the queue, still unspoken
        box = [None, threading.Event()]

        def _synth(b=box, s=nsent):
            try:
                b[0] = self.backend.prepare(s)
            except Exception:                               # noqa: BLE001
                b[0] = None
            finally:
                b[1].set()
        th = threading.Thread(target=_synth, daemon=True,
                              name="assist-tts-prefetch")
        try:
            th.start()
        except RuntimeError:
            # thread exhaustion / interpreter finalization: never leak
            # _ahead_pending=True (busy() would wedge forever) — hand
            # the sentence back with an empty ready box so the player's
            # payload-None path synthesizes it inline
            with self._state_lock:
                self._ahead_pending = False
            box[1].set()
            return (ngen, nsent, box)
        return (ngen, nsent, box)

    def _player(self) -> None:
        split = (hasattr(self.backend, "prepare")
                 and hasattr(self.backend, "play"))
        ahead = None                     # (gen, sent, box) synthesized ahead
        while True:
            if ahead is not None:
                gen, sent, box = ahead
                ahead = None
                with self._state_lock:
                    self._ahead_pending = False   # ...now _pending_play's
                    self._pending_play += 1
                box[1].wait(timeout=30.0)   # bounded: a wedged synth
                payload = box[0]            # must not wedge the player
            else:
                gen, sent = self._q.get()
                # dequeued-but-unplayed MUST count as speech: this is
                # the first-sentence synthesis window (0.3–3 s) where
                # every other flag was false — busy() read False, the
                # follow-up mic opened ON TOP of the reply and recorded
                # the assistant (audit R1, the root of all three live
                # complaints)
                with self._state_lock:
                    self._pending_play += 1
                payload = None
            if gen != self._gen:         # barge-in invalidated this sentence
                # a skipped item is outside the queue, so a stop() that
                # raced it saw _playing/queue clear but not this —
                # recompute the drain here or _reported wedges True and
                # busy() pauses the mic forever
                with self._state_lock:
                    self._pending_play = max(0, self._pending_play - 1)
                    done = (self._q.empty() and ahead is None
                            and not self._ahead_pending
                            and self._pending_play == 0
                            and (not self._turn_open or self._flushed)
                            and self._reported
                            and not self._playing)
                    if done:
                        self._reported = False
                if done:
                    try:
                        self.on_speaking(False)
                    except Exception:                       # noqa: BLE001
                        pass
                continue
            try:
                if split and payload is None:
                    try:
                        payload = self.backend.prepare(sent)
                    except Exception:                       # noqa: BLE001
                        payload = None
                with self._state_lock:
                    self._playing = True
                    announce = not self._reported
                    self._reported = True
                if split:
                    # prefetch AFTER _playing is up: the next item's
                    # pending flag must never be cleared by this one
                    ahead = self._prefetch()   # synth next WHILE this plays
                if announce:             # transition only, not per sentence
                    try:
                        self.on_speaking(True)
                    except Exception:                       # noqa: BLE001
                        pass
                if split:
                    if payload is not None:
                        self.backend.play(payload)
                else:
                    self.backend.speak(sent)
                # device-buffer drain grace ONLY at the end of the run
                # (afplay exits before the room is silent) — doing it
                # per sentence inside play() added ~0.25 s of dead air
                # at EVERY sentence boundary
                with self._state_lock:
                    tail = (self._q.empty() and ahead is None
                            and not self._ahead_pending)
                cancelled = False
                c = getattr(self.backend, "_cancel", None)
                if c is not None:
                    try:
                        cancelled = c.is_set()
                    except Exception:                       # noqa: BLE001
                        pass
                if tail and not cancelled:
                    time.sleep(0.25)
            except Exception:                               # noqa: BLE001
                pass                     # a dead backend must never kill
            finally:                     # the player thread
                with self._state_lock:
                    self._playing = False
                    self._pending_play = max(0, self._pending_play - 1)
                    done = (self._q.empty() and ahead is None
                            and not self._ahead_pending
                            and self._pending_play == 0
                            and (not self._turn_open or self._flushed)
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
        if getattr(self, "_dead", False):
            return              # stop() already ran (async-start race)
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
        if getattr(self, "_dead", False):
            # stop() raced our open (start now runs off the GUI thread):
            # the reader sees the flag and closes the stream itself
            self._stop.set()

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
        self._dead = True        # a start() that hasn't run yet must not
        self._stop.set()         # open a stream nobody will ever stop
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
