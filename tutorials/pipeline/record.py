"""Motion-clip recorder for tutorial storyboards.

Generalises the inline machinery that ep13 introduced: capture real
widget frames at a paced rate while the GUI does real work, then attach
the clip to a Scene at the ACHIEVED capture rate so playback speed is
honest even when grabs run slower than the nominal fps.

Usage in a storyboard::

    from pipeline.record import Recorder
    rec = Recorder(WORK / "frames", settle)
    rec.start("040_run")
    rec.hold(grab, 1.0)
    btn.click()
    rec.wait_until(grab, lambda: btn.isEnabled(), cap_s=300)
    rec.hold(grab, 2.0)
    rec.finish(scene("040_run"))

``grab`` is any callable returning a QImage (a widget grab or a
composed canvas).  All waits keep the Qt event loop pumping, so real
timers, workers, and repaints run while frames are captured.
"""
from __future__ import annotations

import time as _time
from pathlib import Path

MAX_FRAMES = 2400        # per clip — bounds disk/encode; logged if hit


class Recorder:
    def __init__(self, frames_root: str | Path, settle) -> None:
        self.root = Path(frames_root)
        self._settle = settle          # callable(n): pump the event loop
        self._st: dict | None = None

    # -- lifecycle ----------------------------------------------------
    def start(self, key: str, fps: float = 10.0) -> None:
        d = self.root / key
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.png"):
            old.unlink()
        self._st = {"dir": d, "n": 0, "fps": fps, "capped": False,
                    "next": _time.monotonic(), "t0": _time.monotonic()}

    def finish(self, sc) -> None:
        """Attach the clip to Scene ``sc`` at the achieved rate."""
        st = self._st
        assert st is not None, "finish() before start()"
        wall = max(_time.monotonic() - st["t0"], 1e-6)
        sc.frames_dir = str(st["dir"])
        sc.fps = max(st["n"] / wall, 1.0)
        # persist the achieved rate so a capture-free re-encode replays
        # the clip at true speed (see tutorials/reencode.py)
        (st["dir"] / "fps.txt").write_text(f"{sc.fps:.4f}\n")
        print(f"[clip] {sc.name}: {st['n']} frames in {wall:.1f}s "
              f"-> {sc.fps:.2f} fps")
        self._st = None

    # -- capture ------------------------------------------------------
    def tick(self, grab, frames: int = 1) -> None:
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication
        st = self._st
        assert st is not None, "tick() before start()"
        for _ in range(frames):
            self._settle(2)
            if st["n"] < MAX_FRAMES:
                grab().save(str(st["dir"] / f"{st['n']:05d}.png"))
                st["n"] += 1
            elif not st["capped"]:
                st["capped"] = True
                print(f"[warn] clip {st['dir'].name} hit MAX_FRAMES "
                      f"({MAX_FRAMES}) — recording paused, waits continue")
            st["next"] += 1.0 / st["fps"]
            while _time.monotonic() < st["next"]:
                QApplication.processEvents(
                    QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
                _time.sleep(0.004)
            st["next"] = max(st["next"],
                             _time.monotonic() - 0.5 / st["fps"])

    def hold(self, grab, seconds: float) -> None:
        st = self._st
        self.tick(grab, frames=max(1, int(seconds * st["fps"])))

    def type_into(self, grab, line_edit, text: str,
                  chars_per_frame: int = 2) -> None:
        """Type ``text`` into a QLineEdit character by character on
        camera (real widget, real setText — the slot wiring fires)."""
        line_edit.clear()
        self.tick(grab, 2)
        for i in range(1, len(text) + 1):
            line_edit.setText(text[:i])
            if i % chars_per_frame == 0 or i == len(text):
                self.tick(grab)
        self.tick(grab, 3)

    def wait_until(self, grab, cond, cap_s: float,
                   stable_s: float = 0.0) -> bool:
        """Record until ``cond()`` holds (for ``stable_s`` continuous
        seconds if given), or ``cap_s`` elapses.  Returns success."""
        t0 = _time.monotonic()
        ok_since = None
        while _time.monotonic() - t0 < cap_s:
            self.tick(grab)
            try:
                ok = bool(cond())
            except Exception:                           # noqa: BLE001
                ok = False
            if ok:
                if stable_s <= 0:
                    return True
                if ok_since is None:
                    ok_since = _time.monotonic()
                elif _time.monotonic() - ok_since >= stable_s:
                    return True
            else:
                ok_since = None
        print(f"[warn] wait_until hit cap {cap_s}s "
              f"({self._st['dir'].name})")
        return False
