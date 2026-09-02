"""Re-encode a finished episode from its cached captures — no GUI, no
recapture, no narration resynthesis.  Use after a pipeline/render.py
change (e.g. the upscale cap) to refresh every episode cheaply.

    PYTHONPATH=.:gui:tutorials python3 tutorials/reencode.py ep10_failure_study

Clip playback rate comes from ``frames/<scene>/fps.txt`` (written by
Recorder.finish) or, for clips captured outside the Recorder, from the
previous segment encode (frames / segment duration).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "tutorials")]


def _seg_fps(seg: Path, n_frames: int) -> float | None:
    if not seg.exists():
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=nw=1:nk=1",
                        str(seg)], capture_output=True, text=True)
    try:
        dur = float(r.stdout.strip())
    except ValueError:
        return None
    # segment = clip + narration tail; clip_len <= dur.  Only trust it
    # when the clip filled most of the segment (no long held frame).
    return None if dur <= 0 else n_frames / dur


def main(ep: str) -> None:
    sb = ROOT / "tutorials" / "storyboards" / f"{ep}.py"
    ns = {"__name__": "__reencode__", "__file__": str(sb)}
    src = sb.read_text()
    exec(compile(src, str(sb), "exec"), ns)          # defines SCENES/WORK/OUT
    SCENES, WORK, OUT = ns["SCENES"], ns["WORK"], ns["OUT"]
    from pipeline.render import build_video
    meta = {}
    if (WORK / "clips.json").exists():
        try:
            meta = json.loads((WORK / "clips.json").read_text())
        except Exception:                                   # noqa: BLE001
            meta = {}
    missing, unknown_fps = [], []
    for sc in SCENES:
        fd = WORK / "frames" / sc.name
        pngs = sorted(fd.glob("*.png")) if fd.is_dir() else []
        if pngs:
            sc.frames_dir = str(fd)
            fps_file = fd / "fps.txt"
            clips = meta.get("clips", meta)          # ep13 nests under "clips"
            if sc.name in clips and isinstance(clips[sc.name], (int, float)):
                sc.fps = float(clips[sc.name])       # recorded truth wins
            elif sc.name in clips and isinstance(clips[sc.name], dict) and "fps" in clips[sc.name]:
                sc.fps = float(clips[sc.name]["fps"])
            elif fps_file.exists():
                sc.fps = float(fps_file.read_text().strip())
            else:
                est = _seg_fps(WORK / "seg" / f"{sc.name}.mp4", len(pngs))
                if est:
                    sc.fps = est
                    unknown_fps.append(f"{sc.name}~{est:.2f}")
                else:
                    unknown_fps.append(f"{sc.name}=default{sc.fps}")
            continue
        img = WORK / "shots" / f"{sc.name}.png"
        if img.exists():
            sc.image = str(img)
        else:
            missing.append(sc.name)
    if missing:
        sys.exit(f"MISSING captures for {ep}: {missing}")
    if unknown_fps:
        print(f"[fps from previous segment] {' '.join(unknown_fps)}")
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")


if __name__ == "__main__":
    main(sys.argv[1])
