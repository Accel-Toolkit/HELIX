"""Narration synthesis for tutorial videos (kokoro, fully local).

Reuses the assistant's own voice stack: the kokoro model files under
``~/.helix/assistant_models`` and :func:`linac_gen.assist.voice.
speakable_numbers` (kokoro misreads bare digits — "2.1 MeV" must reach
the synthesizer as words).  Default narrator ``af_sarah`` — the HELIX
assistant's in-app voice, so tutorials sound like the assistant.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_KOKORO = None
DEFAULT_VOICE = "af_sarah"
#: measured pace that reads clearly for tutorial narration
SPEED = 1.0


def _kokoro():
    global _KOKORO
    if _KOKORO is None:
        from kokoro_onnx import Kokoro

        from linac_gen.assist.voice import _kokoro_files
        model, voices = _kokoro_files()
        if not model:
            raise RuntimeError(
                "kokoro model files not found under ~/.helix/"
                "assistant_models — the assistant voice setup provides "
                "them (kokoro-v*.onnx + voices-v*.bin)")
        _KOKORO = Kokoro(model, voices)
    return _KOKORO


def synthesize(text: str, wav_path: str | Path,
               voice: str = DEFAULT_VOICE) -> float:
    """Text → 16-bit mono WAV at ``wav_path``; returns duration [s].

    Sentences are synthesized one at a time with a natural 0.35 s
    inter-sentence pause — one giant create() call flattens the
    prosody and occasionally truncates long paragraphs.
    """
    from linac_gen.assist.voice import speakable_numbers
    import re

    k = _kokoro()
    sentences = [s.strip() for s in
                 re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: list[np.ndarray] = []
    sr = 24000
    pause = None
    for sent in sentences:
        samples, sr = k.create(speakable_numbers(sent), voice=voice,
                               speed=SPEED)
        a = np.asarray(samples, dtype=np.float32).reshape(-1)
        if pause is None:
            pause = np.zeros(int(0.35 * sr), dtype=np.float32)
        chunks.append(a)
        chunks.append(pause)
    if not chunks:
        raise ValueError("empty narration text")
    audio = np.concatenate(chunks[:-1])          # no trailing pause
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return len(pcm) / float(sr)


def _fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_srt(entries, srt_path: str | Path) -> None:
    """``entries`` = [(start_s, end_s, text)] → SubRip file (sidecar
    captions for accessibility / YouTube upload)."""
    lines = []
    for i, (t0, t1, text) in enumerate(entries, start=1):
        lines += [str(i), f"{_fmt_ts(t0)} --> {_fmt_ts(t1)}",
                  text.strip(), ""]
    Path(srt_path).write_text("\n".join(lines), encoding="utf-8")
