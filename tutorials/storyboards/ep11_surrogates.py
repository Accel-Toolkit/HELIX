"""Episode 11 — The Surrogates tab in depth (ML stand-ins for physics).

Build:  PYTHONPATH=.:gui python3 tutorials/storyboards/ep11_surrogates.py
Output: tutorials/rendered/ep11_surrogates.mp4 (+ .srt)

DEV-ONLY DEMO MACHINE (per user 2026-08-27: fine for the video, never
cut to public): examples/pipii/mebt/mebt.dat — the PIP-II MEBT with
four real buncher field maps, whose surrogates are ALREADY trained and
cached under linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_00x,
so the table auto-populates with genuine models.

Maximum-detail cut.  Everything on screen is the real widget doing
real work: the cache-aware Train modal, the training dialog, a live
smoke training in the 2x2 progress dialog, cancel-by-closing, a real
two-element batch, the row Compare (with its measured verdict and the
source-verified reason for it), the lattice-swap registry clear, the
MP-hybrid toggles, two Compare-MP runs (safe vs fast path), the
weights store, the CLI, and a scrolling tour of the manual chapter.

Every physics number that is spoken is read from the report objects
produced ON CAMERA in this build (narration is composed after capture),
so speech can never drift from the frame.

The demo cache directory is backed up before the training scenes and
restored right after them (retraining overwrites it).

Offscreen traps honoured (see ep07/ep02): no win.close(), no modal
exec() — QMessageBox.question/information/warning, QFileDialog and the
two QDialog.exec()s are patched to non-modal replicas built from the
tab's own strings, and the QtWebEngine manual tour runs in a helper
subprocess (this same file, ``--webtour``) so it never shares a process
with the HELIX window.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "tutorials")]
_qcfg = Path(tempfile.mkdtemp()) / "offscreen.json"
_qcfg.write_text('{"screens": [{"name": "tut", "x": 0, "y": 0, '
                 '"width": 1920, "height": 1080, "logicalDpi": 96, '
                 '"physicalDpi": 96}]}')
os.environ.setdefault("QT_QPA_PLATFORM", f"offscreen:configfile={_qcfg}")
os.environ.setdefault("HELIX_QSETTINGS_DIR", tempfile.mkdtemp())
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep11_work"
SHOTS = WORK / "shots"
FRAMES = WORK / "frames"
OUT = ROOT / "tutorials" / "rendered" / "ep11_surrogates.mp4"
DECK = ROOT / "examples/pipii/mebt/mebt.dat"
FODO = ROOT / "examples/fodo_cell.dat"
CACHE_DIR = ROOT / "linac_gen/surrogates/weights/bb088243c9fb33ce"

SCENES = [
    Scene("010_title",
          "Deep dive number nine. The Surrogates tab. Some elements "
          "are expensive to track. A three dimensional field map "
          "means a Runge Kutta integration through measured field "
          "data for every particle, every step. A surrogate is a "
          "small neural network that learns that element's transfer "
          "behaviour once, and then answers in microseconds. This "
          "tab trains them, scores them, and swaps them in, and this "
          "deep dive walks every control on it. Our demo machine is "
          "a medium energy beam transport with four real buncher "
          "cavities driven by field maps.",
          min_s=7.0),
    Scene("020_idea",
          "The idea, honestly stated. A surrogate does not replace "
          "physics. It memorises one element's response over a "
          "declared range of energies and settings, and it is only "
          "trusted inside that range. HELIX trains on data it "
          "generates from its own full tracking, validates on held "
          "out samples, and reports the validation error to you "
          "before you decide to use it. The tab's own hint line "
          "states the contract: tick Use to engage a surrogate in "
          "envelope mode runs, and Compare runs the envelope twice, "
          "baseline against surrogate, and shows the difference.",
          min_s=8.0),
    Scene("025_when_to_use",
          "Before training anything, the economics, and the manual's "
          "overview chapter has a table for it. A single envelope "
          "pass on a novel lattice is not worth surrogating: training "
          "costs more than it saves. A parameter scan of the same "
          "elements breaks even at five to ten runs, a matching loop "
          "with hundreds of forward passes pays off handsomely, and a "
          "tolerance study across many seeds is where the payoff "
          "dominates. Everything in this episode has a written "
          "reference in this chapter.",
          min_s=7.0),
    Scene("030_table",
          "The heart of the tab is the trained surrogates table, and "
          "it fills automatically from cached weights when you load "
          "a lattice. We go to the empty tab first, then load the "
          "M E B T deck on camera. Watch the table. It found all four "
          "bunchers, trained earlier, one row each with the element "
          "name, its validation error, its scope, a Use checkbox and "
          "a Compare button.",
          min_s=7.0),
    Scene("031_autoloaded",
          "The status line at the foot of the tab is the evidence. "
          "Auto loaded four cached surrogates from the weights "
          "directory named by the lattice hash. Nothing was "
          "retrained; the rows came from disk.",
          min_s=4.0),
    Scene("032_dropdown",
          "The element dropdown lists what can be surrogated. Each "
          "entry shows the lattice index, the element name, its "
          "concrete class and its length: four FieldMap3D bunchers "
          "of two hundred forty millimetres. Only field map classes "
          "appear. R F Q elements carry internal state a surrogate "
          "cannot replicate yet, so they are excluded by "
          "construction.",
          min_s=6.0),
    Scene("034_scope",
          "Two columns deserve a closer look. Val M A P E is the mean "
          "absolute percentage error over all thirty six transfer "
          "matrix entries, measured on held out samples, never on "
          "the training set. It is a strict metric: entries that are "
          "nearly zero inflate it, which is why these smoke trained "
          "values sit between zero point four and zero point nine "
          "rather than near zero. Scope is the surrogate's contract. "
          "It answers only inside these windows: kinetic energy from "
          "two to two point five M e V, cavity amplitude k e plus or "
          "minus twenty percent around the card value, and phase "
          "from minus one hundred ten to minus seventy degrees. "
          "Outside scope it raises an out of scope error and the "
          "solver falls back to full physics. Slower, never wrong.",
          min_s=8.0),
    Scene("036_use_registry",
          "Use is the engagement switch. Ticking it registers the "
          "surrogate in the runtime registry, and the status line "
          "confirms it: registered surrogate F MAP one. The next "
          "zero current envelope run routes that element's full "
          "matrix through the network automatically; runs with "
          "space charge stay on full physics. You can mix and "
          "match, two cavities surrogated, two on full physics. "
          "Select all and Deselect all fire the same per row toggle "
          "for every row, and the status line reports registered "
          "all four surrogates, then unregistered all four.",
          min_s=7.0),
    Scene("040_cache_modal",
          "Now training. Clicking Train surrogate on an element that "
          "already has cached weights asks first. The dialog quotes "
          "the cache location, the cached validation error, and the "
          "sample and epoch budget it was trained with. Open loads "
          "the cached weights instantly, Retry retrains from scratch, "
          "Cancel aborts. Retraining is a deliberate act, never the "
          "default. We choose Retry.",
          min_s=7.0),
    Scene("042_train_dialog",
          "The training dialog, every field. Samples: two hundred "
          "Latin hypercube points, the smoke default. Epochs: forty. "
          "Hidden dims: two layers of sixty four. Workers: C P U "
          "processes for data generation, defaulting to the core "
          "count minus two, twelve on this machine. The energy "
          "window is auto detected, and the label says so: an "
          "envelope forward pass brackets this element's entry and "
          "exit energy with a five percent margin. Then the sweep "
          "rows, which are dynamic per element. This cavity sweeps "
          "amplitude, k e, and phase, because it has an active "
          "electric channel; a pure magnetic solenoid would show "
          "only k b. This deck declares no ADJUST cards, so the "
          "defaults are plus or minus twenty percent and twenty "
          "degrees around the current values; with ADJUST bounds the "
          "row label would quote them instead. Setting a sweep to "
          "zero drops that knob from training entirely. Watch the "
          "phase row. We restore it and accept.",
          min_s=10.0),
    Scene("044_train_live",
          "Training runs on a background thread and this live dialog "
          "tracks it. Stage one, data generation: the bar reports "
          "samples done, the rate and the E T A, while the training "
          "panels say waiting for training to start. Each sample is "
          "a full Runge Kutta integration of the field map at a "
          "different energy, amplitude and phase. Stage two fits the "
          "network: train loss and held out validation M A P E "
          "animate per epoch, with the best so far tracked, and the "
          "six by six heatmap shows which transfer matrix entries "
          "the network finds hardest. When both stages complete the "
          "weights and metadata land on disk and the window title "
          "reads done.",
          min_s=8.0),
    Scene("045_train_done",
          "Back on the tab the row for F MAP one has updated in "
          "place with the fresh validation error, and the status "
          "line names the saved weights path under the lattice hash "
          "directory.",
          min_s=4.0),
    Scene("046_cancel_train",
          "Closing the live dialog is the cancel gesture. We start a "
          "second run, on the second buncher, and close the window "
          "during data generation. The trainer polls a stop flag per "
          "sample and per epoch, stops, and reports cancelled, never "
          "failed. The status reads training cancelled, nothing was "
          "saved, and the title gains a cancelled suffix. Weights are "
          "only written after both stages complete, so nothing "
          "partial ever reaches the disk.",
          min_s=7.0),
    Scene("047_batch_dialog",
          "Train all FieldMap batches the whole machine. Shared "
          "settings on top, and note the production grade defaults: "
          "five thousand samples, two hundred epochs, two layers of "
          "one hundred twenty eight. Below, a checkbox per candidate "
          "element with Select all and Select none. For the demo we "
          "select none, tick the third and fourth bunchers, and drop "
          "to the smoke budget of two hundred samples and forty "
          "epochs.",
          min_s=7.0),
    Scene("048_batch_train",
          "Elements train sequentially on purpose: the data "
          "generation inside each is already multi process, so "
          "parallelising across elements would oversubscribe the "
          "C P U. The window title counts one of two, then two of "
          "two, and each element still gets its own auto detected "
          "energy window and sweep bounds. A failed element would be "
          "skipped, not fatal, and closing the dialog stops the "
          "current element and skips the rest. At the end the status "
          "reads batch training, two of two elements done.",
          min_s=8.0),
    Scene("050_compare_live",
          "Trust, then verify. The Compare button on a row runs the "
          "same beam twice in a background worker, pure Runge Kutta "
          "first, then with the surrogate registered, and the status "
          "line reports comparing F MAP one while the training "
          "buttons grey out. Compare swaps the process wide registry, "
          "so it is guarded: it refuses to overlap another solve.",
          min_s=6.0),
    Scene("051_compare_summary",
          "(composed after capture)",
          min_s=6.0),
    Scene("052_compare_png",
          "The saved figure overlays the two envelopes along the full "
          "line, sigma x, sigma y, sigma phi and sigma W, with the "
          "speedup and the worst relative difference in the title. "
          "Disagreement is shown, not claimed. This plot is the "
          "honest gate before any production use, and here the two "
          "curves lie exactly on top of each other.",
          min_s=6.0),
    Scene("054_m3_limit",
          "Why identical? Because of the current. We ran the same "
          "compare at zero and at five milliamps at capture time. "
          "At zero current the envelope solver takes each element's "
          "full end to end matrix, and the registry serves exactly "
          "that from the network: one query per surrogated cavity, "
          "a real speedup, and a small nonzero difference at the "
          "network's own accuracy, all printed in the summary. With "
          "current, the cavity is sliced into substeps for space "
          "charge kicks, and the surrogate delegates every partial "
          "slice back to Runge Kutta, because a linear end to end "
          "matrix cannot be cut into honest sub slices of a time "
          "varying R F field. So with space charge the network is "
          "never asked, the N N query line reads zero, and the "
          "compare shows identical envelopes. The speedup case with "
          "current is multi particle tracking, coming up next. "
          "Either way the answer stays physically defensible: "
          "unsupported or out of scope queries fall back to full "
          "physics.",
          min_s=9.0),
    Scene("056_lattice_swap",
          "Now the safety rule. All four surrogates are engaged. We "
          "load a different lattice on camera, episode one's F O D O "
          "cell, which has no field maps at all. The table and the "
          "dropdown empty, and the runtime registry is cleared with "
          "them, because the registry is keyed by element name, and "
          "a surrogate trained for one machine's F MAP one must "
          "never answer for another's.",
          min_s=7.0),
    Scene("057_no_fieldmaps",
          "Press Train on this lattice and the tab refuses with a "
          "plain warning: the loaded lattice contains no FieldMap or "
          "FieldMap3D elements. Pure magnet lines do not benefit "
          "from surrogates; the speedup case is field maps, where "
          "Runge Kutta is expensive.",
          min_s=5.0),
    Scene("058_swap_back",
          "Switch back to the M E B T and the cached weights are "
          "rediscovered from the lattice hash directory: four rows "
          "return, and the status line says auto loaded four cached "
          "surrogates. But the Use ticks stay off until you re engage "
          "them deliberately.",
          min_s=5.0),
    Scene("059_weights_disk",
          "Where surrogates live on disk. The weights root holds one "
          "directory per lattice, named by the first sixteen hex "
          "digits of the S H A two fifty six hash of the dot dat "
          "file. Edit a single number in the lattice and you get a "
          "fresh directory; old weights are never silently reused. "
          "Inside, one folder per element with weights dot p t and "
          "metadata dot json. The metadata records the element, its "
          "class, the training seed, sample and epoch counts, the "
          "validation error, the HELIX commit and the creation time: "
          "full provenance. A weights directory is self contained; "
          "copy it to another machine with the same dot dat and it "
          "just works. Rows persist across restarts; the Use ticks "
          "deliberately do not.",
          min_s=8.0),
    Scene("060_hybrid",
          "Below the table, the multi particle hybrid section, and "
          "notice that it is honest about its own maturity. The hint "
          "says it in plain text: safe mode, the default, delegates "
          "to the native integrator, bit identical, no speedup; the "
          "experimental fast path transports particles through the "
          "surrogate's linear matrix; and the planned residual mode "
          "is not implemented, so the substeps control is reserved. "
          "Four controls: the master toggle, the fast path opt in, "
          "the reserved substeps spinner, and Compare M P.",
          min_s=7.0),
    Scene("062_mp_toggles",
          "Operate them. Engage in M P runs alone changes nothing "
          "physical, and the status says so: registered surrogates "
          "delegate to native Runge Kutta, bit identical to baseline, "
          "no neural inference. The fast path is a second, separate "
          "opt in, because its accuracy depends on training quality; "
          "its status says fast path engaged, run Compare M P to "
          "validate. The substeps spinner reports its value "
          "propagated to four registered surrogates, though no "
          "tracking path reads it yet. We put it back to fifteen and "
          "switch the fast path off again.",
          min_s=8.0),
    Scene("063_settings_ini",
          "These three settings persist across sessions. They are "
          "written to the application settings file the moment you "
          "touch them, here the surrogates group with m p enabled, "
          "m p fast path and the residual substeps. The Use ticks "
          "are not stored anywhere; engagement is always explicit.",
          min_s=5.0),
    Scene("064_mp_safe_compare",
          "Compare M P with the fast path off. The beam is the "
          "project's own: H minus at two point one two M e V, five "
          "milliamps, one thousand particles for the demo. Two "
          "complete multi particle simulations run back to back, "
          "baseline against hybrid. The button reads running "
          "comparison, a thin busy bar appears, and the accent "
          "status names the particle count, the four surrogates and "
          "the substep setting.",
          min_s=6.0),
    Scene("065_mp_safe_summary",
          "(composed after capture)",
          min_s=6.0),
    Scene("066_mp_fast_compare",
          "Tick the fast path and run the comparison again. Now the "
          "per substep Runge Kutta inside each buncher is replaced "
          "by a cached matrix apply.",
          min_s=4.0),
    Scene("067_mp_fast_summary",
          "(composed after capture)",
          min_s=8.0),
    Scene("068_mp_fast_png",
          "The saved multi particle figure: sigma x, y and phi, both "
          "normalised emittances and transmission, baseline solid "
          "against hybrid dashed, with substeps, speedup and worst "
          "difference in the title.",
          min_s=5.0),
    Scene("069_cli",
          "Everything this tab does is scriptable. The surrogates "
          "C L I has four subcommands: train, compare and run "
          "envelope mirror the buttons, and register multi shares "
          "one trained surrogate across several identical cavities, "
          "a cryomodule of eight driven by the same field map, for "
          "instance. The Python A P I underneath is a third entry "
          "point, documented in the manual.",
          min_s=6.0),
    Scene("072_manual_tour",
          "The manual's surrogates section has five chapters: "
          "overview, G U I walkthrough, C L I, Python A P I, and the "
          "training guide. The overview spells out which envelope "
          "runs engage the network and which fall back to slice "
          "delegation, and its measured table sets the fast path's "
          "accuracy against training quality. The "
          "training guide's cycle table tells you what a budget "
          "buys: two hundred samples for pipeline checks, fifty "
          "thousand for science. Its worker benchmark shows parallel "
          "data generation is bit identical to serial at any worker "
          "count. And the troubleshooting section covers the out of "
          "scope flood, high validation error, and the no field "
          "maps case.",
          min_s=8.0),
    Scene("099_outro",
          "That is the Surrogates tab. Learn the expensive parts "
          "once, verify against full physics, engage deliberately, "
          "and let the compare tools tell you the truth about the "
          "shortcut. Next, space charge and real beams, the physics "
          "that makes intense linacs hard, and how HELIX computes "
          "it. See you there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


# ---------------------------------------------------------------------------
# spoken-number helpers (the pipeline spells digits; keep forms simple)
# ---------------------------------------------------------------------------
def pct(rel: float) -> str:
    """Relative difference -> spoken percentage ('0.26 percent')."""
    if rel == 0.0:
        return "exactly zero"
    v = rel * 100.0
    if v >= 1.0:
        return f"{v:.1f} percent"
    if v >= 0.01:
        return f"{v:.2f} percent"
    return f"{v:.3f} percent"


def times(x: float) -> str:
    return f"{x:.2f} times"


# ---------------------------------------------------------------------------
# QtWebEngine helper mode (separate process): manual stills + scroll clip
# ---------------------------------------------------------------------------
def _webtour_main(out_root: str) -> None:
    import time
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
    from PyQt6 import QtWebEngineWidgets                     # noqa: F401
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWidgets import QApplication
    app = QApplication(["mantour"])
    v = QWebEngineView()
    v.resize(1920, 1080)
    v.show()
    out_root = Path(out_root)
    shots = out_root / "shots"
    frames = out_root / "frames" / "072_manual_tour"
    shots.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)
    for old in frames.glob("*.png"):
        old.unlink()

    def pump(sec: float) -> None:
        t0 = time.time()
        while time.time() - t0 < sec:
            app.processEvents()
            time.sleep(0.005)

    def load(page: str) -> None:
        done = {}
        v.loadFinished.connect(lambda ok: done.setdefault("ok", ok))
        v.load(QUrl.fromLocalFile(str(ROOT / "site/13_surrogates" / page)))
        t0 = time.time()
        while "ok" not in done and time.time() - t0 < 40:
            app.processEvents()
            time.sleep(0.01)
        assert done.get("ok"), f"page {page} failed to load"
        v.setZoomFactor(1.35)
        pump(2.5)
        try:
            v.loadFinished.disconnect()
        except Exception:                                   # noqa: BLE001
            pass

    def js(code: str):
        out = {}
        v.page().runJavaScript(code, lambda r: out.setdefault("r", r))
        t0 = time.time()
        while "r" not in out and time.time() - t0 < 5:
            app.processEvents()
            time.sleep(0.005)
        return out.get("r")

    def y_of(anchor: str, offset: int = -160) -> float:
        if anchor.startswith("table:"):
            key = anchor[6:]
            y = js("(function(){var ts=document.querySelectorAll('table');"
                   "for(var i=0;i<ts.length;i++){if(ts[i].textContent."
                   "indexOf('%s')>=0){return ts[i].getBoundingClientRect()"
                   ".top+window.scrollY;}}return -1})()" % key)
        else:
            y = js("(function(){var e=document.getElementById('%s');"
                   "return e?e.getBoundingClientRect().top+window.scrollY:-1})()"
                   % anchor)
        assert y is not None and y >= 0, f"anchor {anchor} not found"
        return max(0.0, float(y) + offset)

    def scroll_to(y: float, settle_s: float = 0.35) -> None:
        js(f"window.scrollTo(0,{y:.0f})")
        pump(settle_s)

    # ---- still: "When to use a surrogate" table --------------------
    load("01_overview.html")
    scroll_to(y_of("when-to-use-a-surrogate"))
    v.grab().save(str(shots / "025_when_to_use.png"))

    # ---- scrolling clip at a fixed 10 fps -------------------------
    n = [0]

    def frame() -> None:
        v.grab().save(str(frames / f"{n[0]:05d}.png"))
        n[0] += 1

    def hold(sec: float) -> None:
        for _ in range(int(sec * 10)):
            pump(0.03)
            frame()

    def glide(y_from: float, y_to: float, sec: float = 1.4) -> None:
        steps = max(1, int(sec * 10))
        for i in range(1, steps + 1):
            t = i / steps
            e = t * t * (3 - 2 * t)                      # ease in/out
            scroll_to(y_from + (y_to - y_from) * e, settle_s=0.06)
            frame()

    def tour(page: str, anchors: list[tuple[str, int, float]]) -> None:
        load(page)
        scroll_to(0)
        y = 0.0
        hold(2.0)
        for anchor, off, dwell in anchors:
            y2 = y_of(anchor, off)
            glide(y, y2)
            hold(dwell)
            y = y2

    tour("01_overview.html", [
        ("which-envelope-runs-engage-the-nn", -160, 3.2),
        ("table:Configuration", -190, 3.6),        # the measured MP table
    ])
    tour("05_training_guide.html", [
        ("table:Cycle", -190, 3.6),                # smoke vs production
        ("table:Bit-identical", -190, 3.2),        # worker benchmark
        ("out-of-scope-ood-handling", -160, 2.6),
        ("troubleshooting", -160, 3.2),
    ])
    print(f"[webtour] wrote {n[0]} frames to {frames}")
    sys.stdout.flush()
    os._exit(0)


if "--webtour" in sys.argv:
    _webtour_main(sys.argv[sys.argv.index("--webtour") + 1])


# ---------------------------------------------------------------------------
def capture_visuals() -> None:
    import shutil
    import subprocess
    import time as _time

    from PyQt6.QtCore import QEventLoop, QPoint, QRect, QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter
    from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog,
                                 QFileDialog, QLabel, QMessageBox,
                                 QPushButton)

    from pipeline import cards
    from pipeline.record import Recorder

    os.chdir(ROOT)            # SurrogatesTab._weights_root is CWD-relative
    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}
    scratch = Path(tempfile.mkdtemp(prefix="helix_ep11_"))

    # ---- manual tour in its own process (QtWebEngine + HELIX window in
    # one process is untested; the helper is this same file) ----------
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--webtour", str(WORK)],
                       capture_output=True, text=True, timeout=900,
                       env=os.environ.copy())
    print(r.stdout[-400:])
    assert (SHOTS / "025_when_to_use.png").exists(), \
        f"webtour failed: {r.stderr[-800:]}"
    tour_frames = sorted((FRAMES / "072_manual_tour").glob("*.png"))
    assert len(tour_frames) > 50, "manual tour produced no frames"
    scene_by_name("072_manual_tour").frames_dir = str(FRAMES / "072_manual_tour")
    scene_by_name("072_manual_tour").fps = 10.0

    app = QApplication.instance() or QApplication([])
    cards.title_card(s["010_title"], "The Surrogates Tab",
                     "Deep dive — neural stand-ins, verified against physics")
    cards.outro_card(s["099_outro"], [
        "Train once · verify against physics · engage deliberately",
        "Next — space charge & real beams",
        "github.com/Accel-Toolkit/HELIX",
    ])

    # ---- cards from REAL captured command output --------------------
    def run_out(cmd: list[str]) -> list[str]:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                           env=dict(os.environ, PYTHONPATH="."))
        return (p.stdout or "").rstrip("\n").splitlines()

    def wrap(text: str, width: int = 92) -> list[str]:
        out, line = [], ""
        for w in text.split():
            if len(line) + len(w) + 1 > width and line:
                out.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            out.append(line)
        return out

    hashes = run_out(["ls", "linac_gen/surrogates/weights/"])
    elems = run_out(["ls", "linac_gen/surrogates/weights/bb088243c9fb33ce/"])
    meta_lines = run_out(
        ["python3", "-m", "json.tool",
         "linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_001/metadata.json"])
    keep = ("element_key", "element_class", "training_seed", "n_samples",
            '"epochs"', "val_mape", "helix_commit_sha", "created_iso")
    meta_sel = [ln.strip() for ln in meta_lines if any(k in ln for k in keep)]
    assert len(meta_sel) >= 7, meta_sel
    lines = [("cmd", "ls linac_gen/surrogates/weights/")]
    lines += [("out", ln) for ln in wrap("  ".join(hashes))]
    lines += [("cmd", "ls linac_gen/surrogates/weights/bb088243c9fb33ce/"),
              ("out", "  ".join(elems)),
              ("cmd", "python -m json.tool .../bb088243c9fb33ce/FMAP_001/"
                      "metadata.json | grep -E 'key|class|seed|samples|"
                      "epochs|mape|sha|created'")]
    lines += [("out", ln) for ln in meta_sel]
    cards.terminal_card(s["059_weights_disk"], lines,
                        title="linac_gen/surrogates/weights — the cache")

    help_lines = run_out(["python3", "-m", "linac_gen.surrogates.cli", "--help"])
    assert any("register-multi" in ln for ln in help_lines), help_lines
    cli_lines = [("cmd", "python -m linac_gen.surrogates.cli --help")]
    cli_lines += [("out", ln[:96]) for ln in help_lines if ln.strip()][:14]
    cards.terminal_card(s["069_cli"], cli_lines,
                        title="python -m linac_gen.surrogates.cli")

    # ---- the window -------------------------------------------------
    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    win = InterphaseWindow()
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def rect_in_win(widget, pad: int = 8) -> QRect:
        tl = widget.mapTo(win, QPoint(0, 0))
        return QRect(max(0, tl.x() - pad), max(0, tl.y() - pad),
                     widget.width() + 2 * pad, widget.height() + 2 * pad)

    def crop_rect(rc: QRect) -> QImage:
        img = win.grab().toImage()
        x, y = max(0, rc.x()), max(0, rc.y())
        w = min(img.width() - x, rc.width())
        h = min(img.height() - y, rc.height())
        return img.copy(x, y, w, h)

    def crop_widget(widget, pad: int = 8) -> QImage:
        settle()
        return crop_rect(rect_in_win(widget, pad))

    def crop_span(top_widget, bottom_widget, pad: int = 8) -> QImage:
        settle()
        a = rect_in_win(top_widget, pad)
        b = rect_in_win(bottom_widget, pad)
        return crop_rect(a.united(b))

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    BG = QColor("#0b1220")

    def canvas() -> QImage:
        c = QImage(1920, 1080, QImage.Format.Format_RGB32)
        c.fill(BG)
        return c

    def fit(img: QImage, max_w: int, max_h: int, max_scale: float = 2.2) -> QImage:
        sc = min(max_w / img.width(), max_h / img.height(), max_scale)
        return img.scaled(int(img.width() * sc), int(img.height() * sc),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    def compose_over(base: QImage, top: QImage, dim: int = 150,
                     max_h: int = 900) -> QImage:
        """Window grab dimmed, with a top-level widget grab centred."""
        c = canvas()
        p = QPainter(c)
        p.drawImage(0, 0, base.scaled(1920, 1080,
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, dim))
        t = fit(top, 1700, max_h)
        x, y = (1920 - t.width()) // 2, (1080 - t.height()) // 2
        p.fillRect(x - 6, y - 6, t.width() + 12, t.height() + 12,
                   QColor("#2b3b55"))
        p.drawImage(x, y, t)
        p.end()
        return c

    def zoom_still(img: QImage, factor: float = 2.0) -> QImage:
        c = canvas()
        p = QPainter(c)
        t = fit(img, 1900, 1040, max_scale=factor)
        p.drawImage((1920 - t.width()) // 2, (1080 - t.height()) // 2, t)
        p.end()
        return c

    FONT_BUMP = ("QLabel, QCheckBox, QPushButton, QSpinBox, QDoubleSpinBox, "
                 "QLineEdit, QGroupBox { font-size: 15px; } "
                 "QMessageBox QLabel { font-size: 17px; }")

    # ---- offscreen-safe replicas of the tab's modal calls -----------
    shown_boxes: list = []           # non-modal boxes the storyboard grabs
    save_target = {"path": str(scratch / "compare.png")}
    orig_question = QMessageBox.question
    orig_info = QMessageBox.information
    orig_warning = QMessageBox.warning
    orig_save = QFileDialog.getSaveFileName

    def make_box(parent, icon, title, text, buttons=None, default=None):
        box = QMessageBox(parent)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box._helix_title = title          # offscreen QMessageBox may blank windowTitle()
        box.setText(text)
        if buttons is not None:
            box.setStandardButtons(buttons)
        if default is not None:
            box.setDefaultButton(default)
        box.setStyleSheet(FONT_BUMP)
        box.setModal(False)
        box.show()
        settle(8)
        shown_boxes.append(box)
        return box

    question_ctx = {"shot": None, "answer": QMessageBox.StandardButton.Retry}

    def fake_question(parent, title, text, buttons=None, default=None,
                      *a, **k):
        box = make_box(parent, QMessageBox.Icon.Question, title, text,
                       buttons, default)
        if question_ctx["shot"]:
            compose_over(win.grab().toImage(), box.grab().toImage(),
                         max_h=700).save(s[question_ctx["shot"]])
            question_ctx["shot"] = None
        box.close()
        shown_boxes.remove(box)
        return question_ctx["answer"]

    def fake_info(parent, title, text, *a, **k):
        make_box(parent, QMessageBox.Icon.Information, title, text)
        return QMessageBox.StandardButton.Ok

    def fake_warning(parent, title, text, *a, **k):
        make_box(parent, QMessageBox.Icon.Warning, title, text)
        return QMessageBox.StandardButton.Ok

    QMessageBox.question = staticmethod(fake_question)
    QMessageBox.information = staticmethod(fake_info)
    QMessageBox.warning = staticmethod(fake_warning)
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (save_target["path"], "PNG (*.png)"))

    def close_boxes() -> None:
        for b in shown_boxes:
            try:
                b.close()
            except Exception:                               # noqa: BLE001
                pass
        shown_boxes.clear()
        settle(4)

    # ---- go to the (empty) Surrogates tab FIRST, then load the lattice
    # on camera — the trained-surrogates table fills from cached weights
    go_tab("surrogates")
    settle(8)
    st = win.surrogates_tab
    table = st._table
    rec = Recorder(FRAMES, settle)
    win_frame = lambda: win.grab().toImage()                 # noqa: E731

    def st_grab() -> QImage:
        return crop_widget(st, 8)

    def load_deck(path: Path) -> None:
        lattice, _meta = _parse_lattice_file(str(path))
        win.state.set_lattice(lattice, str(path))

    from linac_gen.core.config import BeamConfig

    def mebt_beam(current: float, n: int = 1000) -> BeamConfig:
        # the project's own beam (examples/pipii/mebt/mebt.lgproj) at a
        # demo particle count
        return BeamConfig(
            species="H-", energy=2.1226695, frequency=162.5,
            current=current, n_particles=n, distribution="gaussian",
            cutoff=4.0, emit_nx=0.21, alpha_x=1.228, beta_x=0.316,
            emit_ny=0.21, alpha_y=-0.095394, beta_y=0.113,
            emit_z=0.06231832, alpha_z=0.0, beta_z=819.05492)

    rec.start("030_table")
    rec.hold(win_frame, 1.5)
    load_deck(DECK)
    win.state.set_beam_config(mebt_beam(0.0))
    rec.wait_until(win_frame, lambda: table.rowCount() > 0,
                   cap_s=60, stable_s=0.8)
    rec.hold(win_frame, 3.5)
    rec.finish(scene_by_name("030_table"))
    settle(10)
    assert table.rowCount() == 4, "surrogates table did not populate (4)"
    assert st._status.text().startswith("auto-loaded 4 cached"), st._status.text()

    # 020: the hint line + the train group (the narrated contract on screen)
    hint = next(l for l in st.findChildren(QLabel)
                if l.text().startswith("Train ML surrogates"))
    zoom_still(crop_span(hint, st._train_btn.parentWidget(), 10), 1.6) \
        .save(s["020_idea"])
    # 031: the auto-loaded status line, zoomed
    src = rect_in_win(st._mp_section, 10).united(rect_in_win(st._status, 10))
    src.setWidth(min(src.width(), 1060))
    zoom_still(crop_rect(src), 1.8).save(s["031_autoloaded"])

    # 032: element dropdown — real popup grabbed as a top-level window;
    # honest fallback: cycle the current index with a tight crop
    combo = st._elem_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    assert len(items) == 4 and all("FieldMap3D, 240 mm" in t for t in items), items
    popup_ok = False
    try:
        combo.showPopup()
        settle(12)
        popw = combo.view().window()
        if popw is not None and popw is not win and popw.isVisible():
            pimg = popw.grab().toImage()
            if pimg.width() > 50 and pimg.height() > 50:
                base = win.grab().toImage()
                off = popw.mapToGlobal(QPoint(0, 0)) - win.mapToGlobal(QPoint(0, 0))
                p = QPainter(base)
                p.drawImage(off.x(), off.y(), pimg)
                p.end()
                rc = rect_in_win(combo, 12).united(
                    QRect(off.x(), off.y(), pimg.width(), pimg.height()))
                rc.adjust(-12, -12, 12, 12)
                zoom_still(base.copy(rc), 1.8).save(s["032_dropdown"])
                popup_ok = True
        combo.hidePopup()
        settle(6)
    except Exception as exc:                                # noqa: BLE001
        print(f"[warn] combo popup grab failed: {exc!r}")
    if not popup_ok:
        rec.start("032_dropdown")
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            rec.hold(lambda: zoom_still(crop_widget(combo.parentWidget(), 10), 1.8), 1.6)
        combo.setCurrentIndex(0)
        rec.finish(scene_by_name("032_dropdown"))
    combo.setCurrentIndex(0)

    # 034: Val MAPE + Scope columns of the four rows, zoomed
    hh = table.horizontalHeader()
    rows_h = hh.height() + sum(table.rowHeight(r) for r in range(table.rowCount()))
    trc = rect_in_win(table, 6)
    scope_w = sum(table.columnWidth(c) for c in range(3)) + 20
    zoom_still(crop_rect(QRect(trc.x(), trc.y(), min(scope_w, trc.width()),
                               rows_h + 14)), 1.7).save(s["034_scope"])
    cells = [(table.item(r, 0).text(), table.item(r, 1).text(),
              table.item(r, 2).text()) for r in range(4)]
    print("[table]", cells)
    assert cells[0][0] == "FMAP_001" and "ke∈[0.0544,0.0816]" in cells[0][2], cells

    # 036: Use toggles + Select all / Deselect all, live status line
    rec.start("036_use_registry")
    rec.hold(st_grab, 1.5)
    table.cellWidget(0, 3).setChecked(True)
    rec.hold(st_grab, 2.5)
    assert st._status.text().startswith("registered surrogate 'FMAP_001'"), st._status.text()
    st._select_all_btn.click()
    rec.hold(st_grab, 2.5)
    assert st._status.text() == "registered all 4 surrogate(s)", st._status.text()
    st._deselect_all_btn.click()
    rec.hold(st_grab, 2.5)
    assert st._status.text() == "unregistered all 4 surrogate(s)", st._status.text()
    rec.finish(scene_by_name("036_use_registry"))

    # ---- TRAINING SCENES: back the demo cache up first ---------------
    backup = scratch / "cache_backup"
    shutil.copytree(CACHE_DIR, backup)
    fmap2_meta_before = (CACHE_DIR / "FMAP_002/metadata.json").read_bytes()

    def restore_cache() -> None:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        shutil.copytree(backup, CACHE_DIR)
        assert (CACHE_DIR / "FMAP_001/metadata.json").read_bytes() == \
            (backup / "FMAP_001/metadata.json").read_bytes()
        print("[cache] restored", CACHE_DIR)


    try:
        from linac_gen_gui.interphase.tabs import surrogates_tab as stmod
        orig_train_exec = stmod._TrainDialog.exec
        orig_batch_exec = stmod._BatchTrainDialog.exec
        train_ctx = {"record": False, "workers": None}

        def dlg_frame_for(dlg):
            def f() -> QImage:
                return compose_over(win.grab().toImage(), dlg.grab().toImage(),
                                    max_h=960)
            return f

        def fake_train_exec(self):
            self.setStyleSheet(FONT_BUMP)
            if train_ctx["workers"]:
                self._workers.setValue(int(train_ctx["workers"]))
            if train_ctx["record"]:
                self.show()
                settle(10)
                labels = [w.text() for w in self.findChildren(QLabel)]
                print("[train-dialog labels]", labels)
                assert any("auto from envelope" in t for t in labels), labels
                assert self._samples.value() == 200 and self._epochs.value() == 40
                f = dlg_frame_for(self)
                rec.start("042_train_dialog")
                rec.hold(f, 4.0)
                for v in (250, 300, 350, 300, 250, 200):
                    self._samples.setValue(v)
                    rec.hold(f, 0.4)
                rec.hold(f, 2.0)
                ph = self._param_widgets["phase"][2]
                ph.setValue(0.0)
                rec.hold(f, 2.5)
                ph.setValue(20.0)
                rec.hold(f, 2.0)
                rec.finish(scene_by_name("042_train_dialog"))
                self.hide()
                settle(4)
            return QDialog.DialogCode.Accepted

        stmod._TrainDialog.exec = fake_train_exec

        def title_strip(p: QPainter, text: str) -> None:
            p.fillRect(0, 0, 1920, 56, QColor("#111a2e"))
            p.setPen(QColor("#e5eaf5"))
            f = QFont("Helvetica Neue", 22)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(0, 0, 1920, 56), Qt.AlignmentFlag.AlignCenter, text)

        def train_frame_for(get_dlg, bottom_widgets):
            def f() -> QImage:
                dlg = get_dlg()
                c = canvas()
                p = QPainter(c)
                if dlg is not None:
                    if dlg.width() != 1440 or dlg.height() != 870:
                        dlg.resize(1440, 870)
                        settle(3)
                    title_strip(p, dlg.windowTitle())
                    d = dlg.grab().toImage()
                    p.drawImage((1920 - d.width()) // 2, 66, d)
                settle(1)
                ws = [w for w in bottom_widgets if w.isVisible()]
                if ws:
                    rc = rect_in_win(ws[0], 6)
                    for w in ws[1:]:
                        rc = rc.united(rect_in_win(w, 6))
                    strip = crop_rect(rc)
                    t = fit(strip, 1440, 110, max_scale=1.0)
                    p.drawImage((1920 - t.width()) // 2, 1080 - t.height() - 10, t)
                p.end()
                return c
            return f

        # ---- 040 + 042 + 044: one real click on Train surrogate ----------
        combo.setCurrentIndex(0)
        question_ctx["shot"] = "040_cache_modal"
        train_ctx.update(record=True, workers=None)
        st._train_btn.click()
        settle(6)
        assert st._worker is not None and st._progress_dlg is not None
        assert question_ctx["shot"] is None, "cache modal was not shown"
        t_train0 = _time.monotonic()
        rec.start("044_train_live")
        frame044 = train_frame_for(lambda: st._progress_dlg,
                                   [st._progress_label, st._progress_bar])
        ok = rec.wait_until(frame044,
                            lambda: (not st._worker.isRunning())
                            and st._train_btn.isEnabled()
                            and st._status.text().startswith("trained FMAP_001"),
                            cap_s=230, stable_s=0.4)
        rec.hold(frame044, 3.0)
        rec.finish(scene_by_name("044_train_live"))
        assert ok, f"training did not finish: {st._status.text()!r}"
        print(f"[train] FMAP_001 wall {_time.monotonic() - t_train0:.1f} s; "
              f"status {st._status.text()!r}; title {st._progress_dlg.windowTitle()!r}")
        assert st._progress_dlg.windowTitle().endswith("[done]")
        zoom_still(crop_span(table, st._status, 10), 1.5).save(s["045_train_done"])

        # ---- 046: cancel by closing the live dialog (second buncher) ------
        combo.setCurrentIndex(1)
        assert combo.currentText().split(":")[1].strip().startswith("FMAP_002")
        train_ctx.update(record=False, workers=4)          # longer data-gen window
        st._train_btn.click()
        settle(6)
        dlg2 = st._progress_dlg
        assert dlg2 is not None and st._worker.isRunning()
        rec.start("046_cancel_train")
        frame046 = train_frame_for(lambda: dlg2, [st._status])
        ok = rec.wait_until(frame046, lambda: dlg2._data_done >= 30,
                            cap_s=120, stable_s=0.0)
        assert ok, "data generation never reached 30 samples"
        rec.hold(frame046, 1.0)
        dlg2.close()                                    # THE cancel gesture
        ok = rec.wait_until(frame046,
                            lambda: st._status.text().startswith("training cancelled"),
                            cap_s=90, stable_s=0.4)
        rec.hold(frame046, 3.0)
        rec.finish(scene_by_name("046_cancel_train"))
        assert ok, f"cancel not reported: {st._status.text()!r}"
        assert dlg2.windowTitle().endswith("[cancelled]"), dlg2.windowTitle()
        assert (CACHE_DIR / "FMAP_002/metadata.json").read_bytes() == fmap2_meta_before, \
            "cancelled run touched FMAP_002 on disk"
        print(f"[cancel] status {st._status.text()!r}; title {dlg2.windowTitle()!r}")

        # ---- 047 + 048: real batch of two ---------------------------------
        def fake_batch_exec(self):
            self.setStyleSheet(FONT_BUMP)
            self.show()
            settle(10)
            assert self._samples.value() == 5000 and self._epochs.value() == 200
            assert self._hidden.text() == "128,128" and len(self._boxes) == 4
            f = dlg_frame_for(self)
            rec.start("047_batch_dialog")
            rec.hold(f, 4.0)
            none_btn = next(b for b in self.findChildren(QPushButton)
                            if b.text() == "Select none")
            none_btn.click()
            rec.hold(f, 1.5)
            self._boxes[2].setChecked(True)
            rec.hold(f, 0.8)
            self._boxes[3].setChecked(True)
            rec.hold(f, 1.2)
            for v in (4000, 3000, 2000, 1000, 500, 200):
                self._samples.setValue(v)
                rec.hold(f, 0.3)
            for v in (150, 100, 60, 40):
                self._epochs.setValue(v)
                rec.hold(f, 0.3)
            rec.hold(f, 2.5)
            rec.finish(scene_by_name("047_batch_dialog"))
            self.hide()
            settle(4)
            return QDialog.DialogCode.Accepted

        stmod._BatchTrainDialog.exec = fake_batch_exec
        combo.setCurrentIndex(0)
        st._batch_train_btn.click()
        settle(6)
        ctl = st._batch_controller
        assert ctl is not None and len(ctl._elements) == 2
        assert [e.name for e in ctl._elements] == ["FMAP_003", "FMAP_004"]
        rec.start("048_batch_train")
        frame048 = train_frame_for(lambda: ctl._dlg, [st._status])
        ok = rec.wait_until(frame048,
                            lambda: st._batch_train_btn.isEnabled()
                            and st._status.text() == "Batch training: 2 / 2 elements done",
                            cap_s=235, stable_s=0.4)
        rec.hold(frame048, 3.0)
        rec.finish(scene_by_name("048_batch_train"))
        assert ok, f"batch did not finish: {st._status.text()!r}"
        print(f"[batch] status {st._status.text()!r}")

        # ---- restore the demo cache NOW (retraining overwrote 001/003/004)
        stmod._TrainDialog.exec = orig_train_exec
        stmod._BatchTrainDialog.exec = orig_batch_exec
        for d in (st._progress_dlg, ctl._dlg):
            if d is not None:
                d.cancel_cb = None
                d.close()
        settle(6)
    finally:
        restore_cache()

    # ---- 056 / 057 / 058: lattice swap clears table + registry --------
    st._select_all_btn.click()
    settle(6)
    from linac_gen.surrogates import registry as _reg
    assert len(_reg.list_registered()) == 4
    rec.start("056_lattice_swap")
    rec.hold(st_grab, 2.0)
    load_deck(FODO)
    rec.hold(st_grab, 4.0)
    rec.finish(scene_by_name("056_lattice_swap"))
    assert table.rowCount() == 0 and combo.count() == 0
    assert len(_reg.list_registered()) == 0, "registry not cleared on swap"
    st._train_btn.click()                # -> real 'No element' warning
    settle(8)
    assert shown_boxes and shown_boxes[-1]._helix_title == "No element" \
        and "no FieldMap" in shown_boxes[-1].text(), \
        [(getattr(b, "_helix_title", ""), b.text()[:40]) for b in shown_boxes]
    print("[warn-box]", shown_boxes[-1].text())
    compose_over(win.grab().toImage(), shown_boxes[-1].grab().toImage(),
                 max_h=600).save(s["057_no_fieldmaps"])
    close_boxes()
    rec.start("058_swap_back")
    rec.hold(st_grab, 1.5)
    load_deck(DECK)
    rec.wait_until(st_grab, lambda: table.rowCount() == 4, cap_s=60,
                   stable_s=0.6)
    rec.hold(st_grab, 3.5)
    rec.finish(scene_by_name("058_swap_back"))
    assert st._status.text().startswith("auto-loaded 4 cached"), st._status.text()
    assert not any(table.cellWidget(r, 3).isChecked() for r in range(4))
    assert table.item(0, 1).text() == "6.75e-01", table.item(0, 1).text()

    # ---- 050 / 051 / 052: row Compare, live -----------------------------
    # CAPTURE CONFIG: 5 mA.  The scene 051 narration (composed below) and
    # the static 054_m3_limit narration are WORDED for this current —
    # "configured five milliamps", NN queries == 0, identical envelopes.
    # If this current ever changes, re-word both narrations (and the
    # branch of the 051/054 assertions that runs) before rebuilding.
    win.state.set_beam_config(mebt_beam(5.0))
    settle(6)
    report = {}
    save_target["path"] = str(scratch / "FMAP_001_compare.png")
    cmp_btn = table.cellWidget(0, 4)
    assert isinstance(cmp_btn, QPushButton) and cmp_btn.text() == "Compare"
    rec.start("050_compare_live")
    rec.hold(st_grab, 1.5)
    cmp_btn.click()
    settle(2)
    assert st._env_compare_worker is not None
    st._env_compare_worker.finished_ok.connect(
        lambda name, rep: report.__setitem__("env", rep))
    assert st._status.text().startswith("Comparing 'FMAP_001'"), st._status.text()
    assert not st._train_btn.isEnabled()
    ok = rec.wait_until(st_grab,
                        lambda: st._train_btn.isEnabled() and "env" in report
                        and len(shown_boxes) == 1,
                        cap_s=180, stable_s=0.3)
    rec.hold(st_grab, 2.0)
    rec.finish(scene_by_name("050_compare_live"))
    assert ok, "envelope compare did not finish"
    box = shown_boxes[-1]
    assert box._helix_title == "Compare — FMAP_001", box._helix_title
    print("[env-compare]\n" + box.text())
    compose_over(win.grab().toImage(), box.grab().toImage(), max_h=900) \
        .save(s["051_compare_summary"])
    close_boxes()
    assert Path(save_target["path"]).exists()
    shutil.copy(save_target["path"], s["052_compare_png"])
    env = report["env"]
    env_sp = env.speedup()
    # The GUI compare runs at the beam-config current (5 mA was set
    # above): SC bundles slice-walk RK4, so the NN is never queried and
    # the two envelopes are identical.  At 0 mA the same compare would
    # query the NN once per surrogated cavity and show non-zero diffs
    # (asserted in the 054 loop below).
    if float(win.state.beam_config.current) > 0.0:
        assert env.nn_calls == 0, env.summary_text()
        assert env.worst_rel_diff() == 0.0, env.summary_text()
    else:
        assert env.nn_calls == 1, env.summary_text()
        assert env.worst_rel_diff() > 0.0, env.summary_text()
    scene_by_name("051_compare_summary").narration = (
        "The summary reports the wall clock of both runs, the speedup, "
        "the end of line sigma differences, and an honesty line: the "
        "number of N N full element queries. This compare ran at the "
        "configured five milliamps, so the N N query line reads zero, "
        "every relative difference is exactly zero and the speedup is "
        f"{times(env_sp)}: identical envelopes, no gain, and the note "
        "under the table says why. We will unpack that in a moment. "
        "First, the plot you were prompted to save.")

    # ---- 054: the measured pair (0 mA vs 5 mA) as a real-text card ------
    from linac_gen.distributions.factory import create_beam, geometric_emittances
    from linac_gen.surrogates.compare import compare_envelope
    surr1 = st._trained["FMAP_001"][0]
    blocks = []
    for cur in (0.0, 5.0):
        cfg = mebt_beam(cur)
        beam = create_beam(cfg, seed=42)
        ex, ey, ez = geometric_emittances(cfg, max(float(beam.ref.bg), 1e-9))
        tw = dict(alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=ex,
                  alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=ey,
                  alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=ez)
        rep = compare_envelope(win.state.lattice, beam.ref, tw, current=cur,
                               surrogates=[surr1])
        if cur == 0.0:
            # Pure-linear path: the one registered cavity's full matrix
            # is served by the NN -> one query, non-zero diff at NN
            # accuracy.
            assert rep.nn_calls == 1, rep.summary_text()
            assert rep.worst_rel_diff() > 0.0, rep.summary_text()
        else:
            # SC bundles slice-walk RK4 -> zero queries, identical runs.
            assert rep.nn_calls == 0, rep.summary_text()
            assert rep.worst_rel_diff() == 0.0, rep.summary_text()
        lines = [ln for ln in rep.summary_text().splitlines()
                 if ln.startswith(("Wall-clock", "Worst", "NN full-element"))
                 or ln.strip().startswith("sigma")]
        assert len(lines) == 7, lines
        blocks.append((cur, lines))
    card = []
    for cur, lines in blocks:
        card.append(("cmd", f"compare FMAP_001 at current = {cur:.0f} mA"
                            f"   (surrogate registered for the 2nd run)"))
        for ln in lines:
            card.append(("out", ln[:96]))
        card.append(("gap", ""))
    cards.terminal_card(s["054_m3_limit"], card[:-1],
                        title="Envelope compare, measured at capture — "
                              "0 mA and 5 mA")

    # ---- 060: MP-hybrid section expanded on camera --------------------
    rec.start("060_hybrid")
    rec.hold(st_grab, 1.5)
    st._mp_section._toggle.setChecked(True)
    rec.hold(st_grab, 5.0)
    rec.finish(scene_by_name("060_hybrid"))
    settle(6)

    def mp_grab() -> QImage:
        return zoom_still(crop_widget(st._mp_section, 10), 1.35)

    # ---- 062: the three toggles, live ----------------------------------
    assert not st._mp_engage.isChecked() and not st._mp_fast_path.isChecked()
    assert st._mp_substeps.value() == 15
    rec.start("062_mp_toggles")
    rec.hold(mp_grab, 1.5)
    st._mp_engage.setChecked(True)
    rec.hold(mp_grab, 4.0)
    assert st._mp_status.text().startswith("MP-mode engaged (safe mode)")
    st._mp_fast_path.setChecked(True)
    rec.hold(mp_grab, 3.5)
    assert st._mp_status.text().startswith("Fast path ENGAGED")
    st._mp_substeps.setValue(8)
    rec.hold(mp_grab, 3.5)
    assert st._mp_status.text().startswith("Residual RK4 substeps set to 8 "
                                           "(propagated to 4 registered")
    st._mp_substeps.setValue(15)
    rec.hold(mp_grab, 1.5)
    st._mp_fast_path.setChecked(False)
    rec.hold(mp_grab, 2.5)
    rec.finish(scene_by_name("062_mp_toggles"))
    print("[mp-status]", st._mp_status.text())

    # ---- 063: the persisted settings, real file content ----------------
    from linac_gen_gui.interphase.app_settings import make_settings
    make_settings("HELIX", "linac_gen_gui").sync()
    ini = Path(os.environ["HELIX_QSETTINGS_DIR"]) / "HELIX_linac_gen_gui.ini"
    ini_lines = ini.read_text().splitlines()
    sect = []
    grab_on = False
    for ln in ini_lines:
        if ln.startswith("["):
            grab_on = ln.strip() == "[surrogates]"
            if grab_on:
                sect.append(ln)
            continue
        if grab_on and ln.strip():
            sect.append(ln)
    assert any("mp_enabled=true" in ln for ln in sect), sect
    cards.terminal_card(s["063_settings_ini"],
                        [("cmd", "grep -A6 '\\[surrogates\\]' "
                                 "$HELIX_QSETTINGS_DIR/HELIX_linac_gen_gui.ini")]
                        + [("out", ln[:96]) for ln in sect],
                        title="QSettings — persisted MP-section state")

    # ---- 064 / 065: Compare MP, safe mode --------------------------------
    win.state.set_beam_config(mebt_beam(5.0, n=1000))
    settle(4)
    st._mp_fast_path.setChecked(False)
    save_target["path"] = str(scratch / "mp_compare_safe.png")
    rec.start("064_mp_safe_compare")
    rec.hold(mp_grab, 1.5)
    st._mp_compare_btn.click()
    settle(2)
    st._mp_compare_worker.finished_ok.connect(
        lambda rep: report.__setitem__("mp_safe", rep))
    assert st._mp_compare_btn.text().startswith("Running comparison")
    assert "1000-particle beam with 4 surrogate(s)" in st._mp_status.text()
    ok = rec.wait_until(mp_grab,
                        lambda: st._mp_compare_btn.isEnabled()
                        and "mp_safe" in report and len(shown_boxes) == 1,
                        cap_s=240, stable_s=0.3)
    rec.hold(mp_grab, 2.5)
    rec.finish(scene_by_name("064_mp_safe_compare"))
    assert ok, "MP safe compare did not finish"
    box = shown_boxes[-1]
    print("[mp-safe]\n" + box.text())
    compose_over(win.grab().toImage(), box.grab().toImage(), max_h=900) \
        .save(s["065_mp_safe_summary"])
    close_boxes()
    mps = report["mp_safe"]
    assert mps.worst_rel_diff() == 0.0 and not mps.fast_path_enabled
    scene_by_name("065_mp_safe_summary").narration = (
        "The summary. In safe mode every relative difference is exactly "
        "zero: sigma x, sigma y, sigma z, both emittances and "
        f"transmission, at a speedup of {times(mps.speedup())}. The "
        "delegate really is bit identical, and now you have seen it "
        "measured rather than promised.")

    # ---- 066 / 067 / 068: Compare MP with the fast path ------------------
    save_target["path"] = str(scratch / "mp_compare_fast.png")
    rec.start("066_mp_fast_compare")
    rec.hold(mp_grab, 1.5)
    st._mp_fast_path.setChecked(True)
    rec.hold(mp_grab, 2.0)
    st._mp_compare_btn.click()
    settle(2)
    st._mp_compare_worker.finished_ok.connect(
        lambda rep: report.__setitem__("mp_fast", rep))
    ok = rec.wait_until(mp_grab,
                        lambda: st._mp_compare_btn.isEnabled()
                        and "mp_fast" in report and len(shown_boxes) == 1,
                        cap_s=240, stable_s=0.3)
    rec.hold(mp_grab, 2.5)
    rec.finish(scene_by_name("066_mp_fast_compare"))
    assert ok, "MP fast compare did not finish"
    box = shown_boxes[-1]
    print("[mp-fast]\n" + box.text())
    compose_over(win.grab().toImage(), box.grab().toImage(), max_h=900) \
        .save(s["067_mp_fast_summary"])
    close_boxes()
    assert Path(save_target["path"]).exists()
    shutil.copy(save_target["path"], s["068_mp_fast_png"])
    mpf = report["mp_fast"]
    assert mpf.fast_path_enabled and mpf.worst_rel_diff() > 0.0
    e = mpf.end_of_line
    worst_key = max(("sigma_x", "sigma_y", "sigma_z", "emit_nx", "emit_ny"),
                    key=lambda k: e[f"rel_diff_{k}"])
    worst_name = {"sigma_x": "sigma x", "sigma_y": "sigma y",
                  "sigma_z": "sigma z", "emit_nx": "normalised emittance in x",
                  "emit_ny": "normalised emittance in y"}[worst_key]
    scene_by_name("067_mp_fast_summary").narration = (
        "The price tag arrives with the speedup. Speedup "
        f"{times(mpf.speedup())} on this short line with four cavities. "
        f"Sigma x now differs by {pct(e['rel_diff_sigma_x'])}, sigma z by "
        f"{pct(e['rel_diff_sigma_z'])}, and the worst entry, "
        f"{worst_name}, by {pct(mpf.worst_rel_diff())}; the note records "
        "that the fast path was enabled for the run. Smoke trained "
        "surrogates are not accurate enough for science; the manual's "
        "measured table shows the longitudinal error dropping six fold "
        "with better training. Run Compare M P after every toggle change.")
    st._mp_fast_path.setChecked(False)
    st._mp_engage.setChecked(False)
    settle(4)

    # ---- restore the patched statics -----------------------------------
    QMessageBox.question = orig_question
    QMessageBox.information = orig_info
    QMessageBox.warning = orig_warning
    QFileDialog.getSaveFileName = orig_save

    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    assert all(not sc.narration.startswith("(composed") for sc in SCENES)
    # NO win.close(): see ep07 — offscreen modal-confirm hang.


def main() -> None:
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)          # training pools / worker threads must not hang exit


if __name__ == "__main__":
    main()
