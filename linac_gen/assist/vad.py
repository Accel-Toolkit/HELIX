"""Silero-VAD speech gate — optional, offline, tiny.

The hands-free stack's RMS thresholds (wake silence gate, capture
endpointing, follow-up onset, barge-in) are fragile against fans, HVAC
and typing.  When the ~2 MB silero ONNX model is present
(``~/.helix/assistant_models/silero_vad.onnx``, same drop-in dir as the
kokoro voice) every gate upgrades to a real speech probability; when it
is absent — or ``HELIX_VAD=0`` — everything falls back to the RMS gates
unchanged.

Barge-in caveat: the assistant's own TTS bleed IS speech, so VAD alone
cannot arbitrate talk-over.  Barge keeps its level-based bleed
calibration and ANDs it with VAD — a dropped mug no longer cuts the
voice mid-sentence, but the bleed floor still does the speaker/user
separation.

Threading: one :class:`SileroVAD` instance per consumer thread (the
model carries recurrent state; instances are cheap, ~1 ms/32 ms frame).
"""
from __future__ import annotations

import os
import threading

import numpy as np

#: the model's native hop at 16 kHz — 32 ms
FRAME = 512
SR = 16000


def model_path() -> str:
    from linac_gen.assist.voice import MODELS_DIR
    p = os.path.join(MODELS_DIR, "silero_vad.onnx")
    return p if os.path.exists(p) else ""


def available() -> bool:
    if os.environ.get("HELIX_VAD", "1") == "0":
        return False
    if not model_path():
        return False
    try:
        import onnxruntime                        # noqa: F401
        return True
    except Exception:                             # noqa: BLE001
        return False


#: outcome of the one-time real-speech self-test on THIS machine:
#: "pending" | "pass" | "fail".  A deaf VAD once shipped (context-carry
#: bug) and silently killed wake/follow-up/barge — so until the model
#: PROVES it hears generated speech, make() returns None and every
#: gate runs the plain loudness algorithm (exactly MIRAGE's, which is
#: known-good).  The upgrade may only ADD, never subtract.
_SELFTEST = "pending"
_SELFTEST_LOCK = threading.Lock()


def _marker_path() -> str:
    """Per-machine cache of a PASSED self-test, next to the model file.
    The 1.2 s real-speech proof runs once per machine, ever — later
    launches must not race it (a pending self-test at listener build
    time silently locked whole sessions into RMS-only wake)."""
    mp = model_path()
    return (mp + ".selftest_ok") if mp else ""


def self_test() -> bool:
    """Score real generated speech (macOS ``say``); True = the model
    demonstrably hears.  Without ``say`` the CI real-speech test is the
    guard and this passes trustingly."""
    global _SELFTEST
    with _SELFTEST_LOCK:
        if _SELFTEST != "pending":            # already decided
            return _SELFTEST == "pass"
        return _self_test_locked()


def _self_test_locked() -> bool:
    global _SELFTEST
    try:
        import shutil
        import subprocess
        import tempfile
        import wave
        if not available():
            _SELFTEST = "fail"
            return False
        m = _marker_path()
        if m and os.path.exists(m):           # proved once on this machine
            _SELFTEST = "pass"                # — never re-run the 1.2 s
            return True                       # ``say`` proof again
        if not shutil.which("say"):
            _SELFTEST = "pass"
            return True
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "vad_selftest.wav")
            subprocess.run(["say", "-o", wav,
                            "--data-format=LEI16@16000",
                            "helix check one two"],
                           check=True, timeout=30)
            with wave.open(wav, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
        audio = (np.frombuffer(raw, dtype=np.int16)
                 .astype(np.float32) / 32768.0)
        vad = SileroVAD()
        best = max(vad.prob(audio[i:i + 3200])
                   for i in range(0, max(audio.size - 3200, 1), 3200))
        _SELFTEST = "pass" if best > 0.6 else "fail"
        if _SELFTEST == "pass":
            try:                              # remember across launches
                m = _marker_path()
                if m:
                    open(m, "w").close()
            except Exception:                 # noqa: BLE001
                pass
        return _SELFTEST == "pass"
    except Exception:                             # noqa: BLE001
        _SELFTEST = "fail"
        return False


def self_test_async() -> None:
    """Run the self-test off-thread (panel prewarm calls this)."""
    threading.Thread(target=self_test, daemon=True,
                     name="assist-vad-selftest").start()


def make() -> "SileroVAD | None":
    """A ready VAD instance — or None, in which case callers keep
    their RMS gates (MIRAGE's known-good algorithm).  None until the
    self-test has PASSED on this machine.

    DETERMINISTIC (2026-07-28 wake regression): a still-pending
    self-test is resolved HERE — from the on-disk marker (instant on
    every launch after the first), else by running the 1.2 s
    real-speech proof once.  Before this, listeners built within ~1 s
    of panel startup lost the race against the async self-test and the
    whole session silently ran RMS-only wake.

    NEVER-BLOCK-THE-GUI (2026-07-28 audit F1): the main thread must
    not convoy behind a proof already running elsewhere — it gets a
    50 ms lock grace and otherwise falls back to None (RMS gates).
    Worker threads may wait, keeping the deterministic guarantee."""
    global _SELFTEST
    if not available():
        return None
    if _SELFTEST == "pending":
        wait_s = (0.05 if threading.current_thread()
                  is threading.main_thread() else 6.0)
        if not _SELFTEST_LOCK.acquire(timeout=wait_s):
            return None                       # proof in flight — RMS for now
        try:
            if _SELFTEST == "pending":
                _self_test_locked()
        except Exception:                     # noqa: BLE001
            return None
        finally:
            _SELFTEST_LOCK.release()
    if _SELFTEST != "pass":
        return None
    try:
        return SileroVAD()
    except Exception:                             # noqa: BLE001
        return None


class SileroVAD:
    """Streaming speech probability for 16 kHz float32 blocks."""

    #: silero v5 expects each 512-sample frame PREFIXED with the last
    #: 64 samples of the previous frame (576 in total).  Feeding bare
    #: frames breaks the recurrent state and collapses speech
    #: probabilities to ~0 — a DEAF vad (observed: real speech at rms
    #: 0.013 scored 0.00x while Whisper transcribed it perfectly).
    CONTEXT = 64

    def __init__(self, path: str | None = None):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(
            path or model_path(), providers=["CPUExecutionProvider"])
        self._state = np.zeros((2, 1, 128), np.float32)
        self._ctx = np.zeros(self.CONTEXT, np.float32)
        self._sr = np.array(SR, dtype=np.int64)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._state = np.zeros((2, 1, 128), np.float32)
            self._ctx = np.zeros(self.CONTEXT, np.float32)

    def prob(self, block) -> float:
        """Max speech probability over the block's 32 ms frames.
        Never raises — a model hiccup reads as silence (0.0)."""
        try:
            a = np.asarray(block, dtype=np.float32).reshape(-1)
            if a.size == 0:
                return 0.0
            if a.size < FRAME:                    # pad a short tail
                pad = np.zeros(FRAME, np.float32)
                pad[:a.size] = a
                a = pad
            best = 0.0
            with self._lock:
                for i in range(0, a.size - FRAME + 1, FRAME):
                    frame = a[i:i + FRAME]
                    inp = np.concatenate([self._ctx, frame])[None, :]
                    out = self._sess.run(
                        None, {"input": inp,
                               "state": self._state, "sr": self._sr})
                    best = max(best, float(out[0].reshape(-1)[0]))
                    self._state = out[1]
                    self._ctx = frame[-self.CONTEXT:]
            return best
        except Exception:                         # noqa: BLE001
            return 0.0
