"""Capture a single GUI tab as a PNG.

Run from the repo root, once per tab::

    PYTHONPATH=$(pwd):$(pwd)/gui python3 docs/manual/_build/capture_gui.py lattice
    PYTHONPATH=$(pwd):$(pwd)/gui python3 docs/manual/_build/capture_gui.py beam
    PYTHONPATH=$(pwd):$(pwd)/gui python3 docs/manual/_build/capture_gui.py errors
    PYTHONPATH=$(pwd):$(pwd)/gui python3 docs/manual/_build/capture_gui.py results
    PYTHONPATH=$(pwd):$(pwd)/gui python3 docs/manual/_build/capture_gui.py lattice_correct

Each run launches the GUI fresh, switches to the named tab, lets the
event loop settle, captures the window via ``QScreen.grabWindow`` (the
live composited pixels — ``QWidget.grab`` returns the back-buffer
which doesn't update reliably across tab swaps on Wayland/WSLg), and
saves to ``docs/manual/_build/figures/gui/<name>_tab.png``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Sandbox QSettings BEFORE any GUI import: a capture run must never save
# its offscreen window geometry / recents into the developer's real
# settings store (same trap the test suite hit).
os.environ.setdefault("HELIX_QSETTINGS_DIR",
                      tempfile.mkdtemp(prefix="helix-capture-qsettings-"))

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "docs" / "manual" / "_build" / "figures" / "gui"
DEMO_DAT = REPO_ROOT / "examples" / "correction_demo" / "correction_demo.dat"


MEBT_DAT = REPO_ROOT / "examples" / "pipii" / "mebt" / "mebt.dat"

# Tab order in the shipping app (state.py TABS):
#   0 Beam · 1 Lattice · 2 Matching · 3 Numerics · 4 Surrogates
#   5 Error Study · 6 Failure Study · 7 Results
# tab name → (tab index, output filename, post-switch hook, lattice file)
_TARGETS: dict[str, tuple[int, str, str | None, Path]] = {
    "beam":            (0, "beam_tab.png",            None,            DEMO_DAT),
    "lattice":         (1, "lattice_tab.png",         None,            DEMO_DAT),
    "matching":        (2, "matching_tab.png",        None,            DEMO_DAT),
    "numerics":        (3, "numerics_tab.png",        None,            DEMO_DAT),
    "surrogates":      (4, "surrogates_tab.png",      None,            MEBT_DAT),
    "errors":          (5, "errors_tab.png",          "tick_correction",
                        DEMO_DAT),
    "failures":        (6, "failures_tab.png",        None,            MEBT_DAT),
    "results":         (7, "results_tab.png",         "run_envelope",  DEMO_DAT),
    "lattice_correct": (1, "lattice_correct_orbit.png",
                        "focus_correct_button",       DEMO_DAT),
}


def _capture(target: str) -> int:
    if target not in _TARGETS:
        print(f"unknown target '{target}'; valid: {sorted(_TARGETS)}",
              file=sys.stderr)
        return 2
    tab_index, fname, hook, dat_file = _TARGETS[target]

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gui"))
    from linac_gen_gui.interphase.app import InterphaseWindow
    from linac_gen.io.tracewin_parser import parse_tracewin

    app = QApplication(sys.argv)
    win = InterphaseWindow()
    win.show()
    win.resize(1400, 880)

    # Step 1: load the per-target lattice so the screenshot has content.
    def step_load() -> None:
        try:
            lat, _ = parse_tracewin(str(dat_file))
            win.state.set_lattice(lat, str(dat_file))
        except Exception as exc:
            print(f"[load] failed: {exc}", file=sys.stderr)

    # Step 2: switch to the requested tab + run any per-target hook.
    def step_select() -> None:
        win._tabs.setCurrentIndex(tab_index)
        if hook == "tick_correction":
            try:
                win.errors_tab._corr_enable.setChecked(True)
            except Exception:
                pass
        elif hook == "focus_correct_button":
            try:
                win.lattice_tab._btn_correct.setFocus()
            except Exception:
                pass
        elif hook == "run_envelope":
            # Populate the Results tab with a real run (the correction
            # demo lattice solves in well under a second).
            try:
                win._run_envelope()
            except Exception as exc:
                print(f"[run_envelope] failed: {exc}", file=sys.stderr)

    # Step 3: capture and quit.
    def step_capture() -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        win.repaint()
        QApplication.processEvents()
        screen = QGuiApplication.primaryScreen()
        pix = screen.grabWindow(int(win.winId()))
        if pix.isNull() or pix.width() <= 0:
            pix = win.grab()
        out = OUT_DIR / fname
        pix.save(str(out), "PNG")
        sz = out.stat().st_size if out.exists() else 0
        print(f"[capture] {fname}  ({pix.width()}x{pix.height()}, "
              f"{sz/1024:.1f} KiB)", flush=True)

    def step_quit() -> None:
        app.quit()

    # The results target needs extra settle time for its envelope run.
    settle = 8000 if hook == "run_envelope" else 4000
    QTimer.singleShot(1500, step_load)
    QTimer.singleShot(2500, step_select)
    QTimer.singleShot(settle, step_capture)
    QTimer.singleShot(settle + 500, step_quit)

    return app.exec()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <target>\n"
              f"targets: {sorted(_TARGETS)}", file=sys.stderr)
        sys.exit(2)
    sys.exit(_capture(sys.argv[1]))
