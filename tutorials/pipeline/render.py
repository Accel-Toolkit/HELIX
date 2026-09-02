"""Assemble tutorial scenes into an H.264/AAC MP4 with ffmpeg.

A scene = one still PNG (``image``) OR a recorded frame sequence
(``frames_dir`` of ``%05d.png`` at ``fps``) + one narration WAV.
Scene duration = narration + ``tail_s`` of quiet, floored by ``min_s``
— and for motion scenes also by the clip's own length, so a recorded
action is never cut short (the last frame holds if narration runs
longer).  Segments are encoded identically and joined with the concat
demuxer; the first/last scene get a fade from/to black.  Captions ship
as a sidecar ``.srt``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

FPS = 30
TAIL_S = 0.7          # breathing room after each narration
FADE_S = 0.6
MAX_UPSCALE = 3      # cap on enlargement of small captures (3x keeps crops sharp)


@dataclass
class Scene:
    name: str                 # segment stem, e.g. "010_title"
    narration: str            # spoken + captioned text
    image: str = ""           # PNG path (filled by the storyboard)
    frames_dir: str = ""      # OR: dir of %05d.png frames (motion clip)
    fps: float = 12.0         # capture rate of frames_dir
    min_s: float = 0.0        # floor on scene duration
    duration: float = field(default=0.0, init=False)   # set at build

    def clip_len(self) -> float:
        if not self.frames_dir:
            return 0.0
        n = len(list(Path(self.frames_dir).glob("*.png")))
        return n / self.fps if n else 0.0


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({' '.join(cmd[:6])} …):\n{r.stderr[-1500:]}")


def _probe_duration(path: str | Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    return float(r.stdout.strip())


def build_video(scenes: list[Scene], workdir: str | Path,
                out_mp4: str | Path, voice: str | None = None) -> dict:
    """Synthesize narration, encode each scene, concat, write .srt.

    Returns ``{"mp4": path, "srt": path, "duration": s}``.
    """
    from . import narrate
    workdir = Path(workdir)
    (workdir / "audio").mkdir(parents=True, exist_ok=True)
    (workdir / "seg").mkdir(parents=True, exist_ok=True)
    voice = voice or narrate.DEFAULT_VOICE

    # 1) narration (cached: skip synthesis when the wav already matches)
    for sc in scenes:
        wav = workdir / "audio" / f"{sc.name}.wav"
        txt = workdir / "audio" / f"{sc.name}.txt"
        if not (wav.exists() and txt.exists()
                and txt.read_text() == sc.narration):
            spoken = narrate.synthesize(sc.narration, wav, voice=voice)
            txt.write_text(sc.narration)
        else:
            spoken = _probe_duration(wav)
        sc.duration = max(spoken + TAIL_S, sc.min_s, sc.clip_len() + 0.3)

    # 2) per-scene segments (identical encoding params for concat)
    seg_files = []
    for i, sc in enumerate(scenes):
        if sc.frames_dir:
            assert sc.clip_len() > 0, \
                f"scene {sc.name}: no frames in {sc.frames_dir}"
            src = ["-framerate", f"{sc.fps}",
                   "-i", str(Path(sc.frames_dir) / "%05d.png")]
            tune: list[str] = []
            # hold the last frame if narration outlasts the clip
            hold = "tpad=stop=-1:stop_mode=clone,"
        else:
            assert sc.image and Path(sc.image).exists(), \
                f"scene {sc.name}: image missing ({sc.image})"
            src = ["-loop", "1", "-i", sc.image]
            tune = ["-tune", "stillimage"]
            hold = ""
        seg = workdir / "seg" / f"{sc.name}.mp4"
        # Fit to 1080p but never upscale beyond MAX_UPSCALE: a 330-px
        # widget crop blown up ~6x reads as mush.  Small captures are
        # scaled at most 2x (Lanczos) and letterboxed on the canvas.
        vf = (f"scale='min(1920,iw*{MAX_UPSCALE})':"
              f"'min(1080,ih*{MAX_UPSCALE})':"
              "force_original_aspect_ratio=decrease:flags=lanczos,"
              "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#0b1220,"
              f"{hold}fps={FPS},format=yuv420p")
        if i == 0:
            vf += f",fade=t=in:st=0:d={FADE_S}"
        if i == len(scenes) - 1:
            vf += f",fade=t=out:st={max(sc.duration - FADE_S, 0):.3f}:d={FADE_S}"
        _run(["ffmpeg", "-y", *src,
              "-i", str(workdir / "audio" / f"{sc.name}.wav"),
              "-t", f"{sc.duration:.3f}",
              "-vf", vf,
              "-c:v", "libx264", "-preset", "medium", "-crf", "20",
              *tune,
              "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "1",
              "-af", "apad", "-shortest",
              str(seg)])
        seg_files.append(seg)

    # 3) concat
    lst = workdir / "concat.txt"
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in seg_files))
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
          "-c", "copy", "-movflags", "+faststart", str(out_mp4)])

    # 4) captions from the ACTUAL segment timings
    entries, t = [], 0.0
    for sc, seg in zip(scenes, seg_files):
        d = _probe_duration(seg)
        entries.append((t + 0.05, t + d - 0.05, sc.narration))
        t += d
    srt = out_mp4.with_suffix(".srt")
    narrate.write_srt(entries, srt)
    return {"mp4": str(out_mp4), "srt": str(srt),
            "duration": _probe_duration(out_mp4)}
