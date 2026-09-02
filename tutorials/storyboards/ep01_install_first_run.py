"""Episode 1 — Install HELIX and run your first beam (maximum-detail cut).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep01_install_first_run.py
Output: tutorials/rendered/ep01_install_first_run.mp4 (+ .srt captions)

Visuals are captured from the REAL GUI, driven offscreen with a
sandboxed HELIX_QSETTINGS_DIR (never the user's settings), so the video
re-renders faithfully after any GUI change.  Manual pages are rendered
by QtWebEngine in a SEPARATE subprocess (``--webshots``) so the web
stack never shares a process with the main window.  Every number spoken
in the narration is measured in this build and injected via «TOKENS».
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "tutorials")]
# Offscreen platform with a REAL 1920x1080 screen: the default offscreen
# screen is ~1000x1000 and silently clamps the main window, so captures
# came out 1000x796 and upscaled soft.
_qcfg = Path(tempfile.mkdtemp()) / "offscreen.json"
_qcfg.write_text('{"screens": [{"name": "tut", "x": 0, "y": 0, '
                 '"width": 1920, "height": 1080, "logicalDpi": 96, '
                 '"physicalDpi": 96}]}')
os.environ.setdefault("QT_QPA_PLATFORM", f"offscreen:configfile={_qcfg}")
os.environ.setdefault("HELIX_QSETTINGS_DIR", tempfile.mkdtemp())
# Two InterphaseWindows are built in this storyboard (session-restore
# scene) — never warm the assistant mic/VAD stack for either.
os.environ.setdefault("HELIX_ASSIST_NO_PREWARM", "1")
if "QT_PLUGIN_PATH" not in os.environ:          # anaconda PyQt6 needs it
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep01_work"
SHOTS = WORK / "shots"
FRAMES = WORK / "frames"
OUT = ROOT / "tutorials" / "rendered" / "ep01_install_first_run.mp4"
SITE = ROOT / "site"

SCENES = [
    Scene("010_title",
          "Welcome to HELIX. In this opening episode you will install "
          "the code, verify it, learn the layout of the interface, "
          "create a project, and send a proton beam through a simple "
          "focusing lattice. No experience with accelerator codes is "
          "needed.",
          min_s=6.0),
    Scene("020_what",
          "HELIX is a beam dynamics simulator for linear accelerators, "
          "developed at Fermi National Accelerator Laboratory for the "
          "P I P two superconducting linac. You describe the machine "
          "as a sequence of elements in the TraceWin lattice format, "
          "and HELIX tracks a beam through it three ways: a fast "
          "envelope solver, a multi-particle tracker with three "
          "dimensional space charge, and linear matrix tracking — on "
          "Windows, mac O S and Linux."),
    Scene("022_prereqs",
          "The manual's prerequisites table. "
          "Python three point ten or newer; NumPy, SciPy, H five P Y "
          "and matplotlib arrive automatically. PyTorch is required — "
          "surrogates and the Apple Silicon G P U path build on it. "
          "The rest — a C plus plus compiler "
          "for the fast kernels, CUDA with cupy, Py Q T six for the "
          "interface — is optional."),
    Scene("030_get",
          "First, get the code. Clone it from the Accel Toolkit "
          "organisation on GitHub, or use the download zip button — "
          "the result is the same folder. Everything runs locally; "
          "nothing ever leaves your computer."),
    Scene("040_setup",
          "One command sets everything up — this is the real "
          "transcript of setup dot S H. It finds a Python of at least "
          "three point ten, builds a private virtual environment "
          "inside the repository — your system Python is never "
          "modified — installs everything, via U V when available, "
          "then smoke-tests the install and reports "
          "whether the fast C plus plus kernels were built. A missing "
          "compiler is not fatal — HELIX falls back to pure Python "
          "space charge, about twenty times slower — or use the "
          "switch on the last line for a hard error. Re-run it after "
          "any git pull — it updates in place."),
    Scene("042_setup_windows_mac",
          "On Windows, double click setup dot bat — it also "
          "pre-checks the two hundred sixty character path limit that "
          "can break the PyTorch install. On a Mac, setup dot command "
          "runs the same script and keeps the Terminal open, so any "
          "error stays readable. And with no Python at all, Windows "
          "users can run the bundled H E L I X dot E X E — verified "
          "with the dash dash smoke check shown here, from the manual."),
    Scene("044_manual_install",
          "Prefer full control? Clone, make a virtual environment, "
          "and install with the G U I and dev extras — about four "
          "hundred megabytes, all inside a dot venv. On a Mac, "
          "install the Xcode command line tools "
          "once so the kernels compile. The G P U extra is N V I D I "
          "A only — on Apple Silicon it is a deliberate no-op, since "
          "Metal ships through the bundled torch."),
    Scene("046_smoke_test",
          "Verify in thirty seconds: ask the physics core for a three "
          "M E V proton, and the reference particle answers — beta "
          "«BETA», gamma «GAMMA» — captured live while building this "
          "video. Setup already ran a check like this for you; for "
          "certainty, the full test suite takes about three minutes."),
    Scene("050_launch",
          "Launch with run G U I dot S H. Its first line names the "
          "interpreter it picked — the first thing to check if a "
          "launch fails. It prefers the repository's dot venv, then U "
          "V, then plain python three. H E L I X dot command is the "
          "double-clickable Mac launcher, Windows has run G U I dot "
          "bat, and any platform can run the Python module directly."),
    Scene("052_splash",
          "The splash screen holds for about five seconds while Python "
          "imports torch and matplotlib. H E L I X stands for Hybrid "
          "Envelope multiparticle LInac eXplorer, and the card carries "
          "the developer credit and the date the code last changed. "
          "First launch is slowest; later ones reuse compiled files.",
          min_s=7.0),
    Scene("054_env_overrides",
          "Four environment knobs. H E L I X "
          "PYTHON points the double-click launcher at a specific "
          "interpreter. Q T Q P A PLATFORM forces a Qt backend — "
          "cocoa on a stubborn Mac. LINAC GEN USE G P U set to c p u "
          "skips G P U initialisation. And offscreen runs the whole "
          "interface with no window — exactly how this series is "
          "filmed."),
    Scene("060_tour",
          "The main window — nine tabs, walked left to right. Beam "
          "defines the particles you inject. Lattice is "
          "the machine editor. Matching tunes elements to hit optics "
          "targets. Numerics holds the solver settings "
          "every run uses. Surrogates trains fast machine-learned "
          "models. Param Study scans parameters across runs. Error "
          "Study adds alignment and field errors. Failure Study asks "
          "what breaks when elements drop out. And Results holds "
          "every plot. Around them: title bar, toolbar, status bar. "
          "Every tab gets its own deep dive later in the series."),
    Scene("062_menus",
          "Four menus. File is the project life "
          "cycle: new, open and save for lattices and projects, the "
          "recent list, the calculation directory, and export to "
          "TraceWin or open P M D. Simulate mirrors the two Run "
          "buttons, plus backtracking and the multibunch pulse study. "
          "Tools opens the optional A I assistant, a Python console, "
          "the matrix viewers, and parameter scans. Help: "
          "documentation, update checks — with an optional check at "
          "startup — and the About box."),
    Scene("064_statusbar",
          "The status bar, magnified. The "
          "left pill is the run state — READY now, RUNNING while a "
          "simulation is in flight. Then the loaded lattice — none yet "
          "— the s position, the reference energy, the beam size, and "
          "the loss fraction. Transient confirmations appear beside "
          "them and fade. Worth knowing in advance: without working C "
          "plus plus kernels, HELIX prints exactly this warning at "
          "startup — everything still runs, on the slower pure Python "
          "path."),
    Scene("066_shortcuts",
          "The shortcuts: control N, new project. Control O, open "
          "lattice. Control S, save. Control R, run envelope. Control "
          "shift R, run multi-particle. Control plus and minus "
          "rescale every font in the interface — the toolbar spinner "
          "tracks them — and control zero resets. F one is contextual "
          "help."),
    Scene("070_wizard",
          "Control N opens the New Project wizard. We type a name — "
          "my first beam. Location points at a projects folder, and "
          "the hint spells out the contract: one folder per project, "
          "holding the lattice and a runs output directory, so the "
          "whole thing moves as one unit. Then the starting point: "
          "blank lattice, import an existing file — watch its row "
          "wake as the radio is chosen — or a bundled example. We "
          "take the F O D O cell."),
    Scene("072_wizard_validation",
          "The wizard also defends you. Names may use letters, "
          "digits, spaces, dots, hyphens and underscores. Type C O N "
          "and it refuses — a reserved device name on Windows, "
          "rejected on every platform. A trailing dot or space is "
          "refused too. "
          "The same gate blocks a missing, unwritable, or "
          "already-occupied location. Proper name back in, and "
          "accept."),
    Scene("074_blank_deck",
          "One aside — a blank project is not empty. This is the deck "
          "it writes: a valid one-drift TraceWin file at one hundred "
          "sixty two point five megahertz — a one hundred millimetre "
          "drift with a fifteen millimetre aperture — immediately "
          "loadable, runnable and editable."),
    Scene("075_import_mode",
          "And the import path: by default the chosen file is copied "
          "into the project folder, keeping the project portable; "
          "untick the box to reference it in place. Bundled examples "
          "are always copied — your edits never touch the originals."),
    Scene("076_project_created",
          "O K creates the folder, loads the lattice, and writes the "
          "project file — the status bar now counts «NELEM» elements "
          "and confirms the save. The project also joins File, Open "
          "Recent, which gains a clear entry as the list grows."),
    Scene("080_fodo",
          "The Lattice tab, in four columns: element palette, outline "
          "tree, timeline with a type chip strip, and the inspector — "
          "with the Sequence listing underneath. The status bar "
          "counts «NELEM» elements over «LTOT». A F O D O line — the "
          "hello world of accelerator physics. The "
          "first quadrupole: fifty millimetres, plus five tesla per "
          "metre, twenty millimetre aperture. The next is minus five "
          "— defocusing — with two hundred millimetre drifts between, "
          "four periods over. Alternating focusing is how every large "
          "accelerator holds its beam together.",
          min_s=8.0),
    Scene("082_f1_help",
          "Press F one with nothing selected and the status bar "
          "answers: select a lattice element first. Select a "
          "quadrupole and press it again — opened, the quadrupole "
          "chapter. This page: the element's parameters and physics, "
          "straight from the manual. Every element type has a "
          "chapter, and most input fields carry hover tooltips too."),
    Scene("090_beam",
          "The Beam tab. On the left, the particle: species — proton, "
          "deuteron or H minus — kinetic energy, R F frequency and "
          "peak current, with derived beta and gamma updating live. "
          "We set a proton at one M E V, the deck's three hundred "
          "fifty two point two one megahertz, and current zero — "
          "space charge off. The middle column is the Twiss grid; in "
          "go the matched values for this cell from HELIX's own "
          "matcher: alpha x «AX», beta x «BX» millimetres per "
          "milliradian; in y, «AY» and «BY». Apply hands the beam to "
          "the solvers and regenerates the preview — four genuine "
          "density plots of the sampled distribution. The other "
          "buttons: redraw the preview, reset the defaults, or import "
          "a TraceWin D S T particle file."),
    Scene("100_results",
          "Run. On «LTOTMM» millimetres of lattice the envelope "
          "solver is done in well under a second — watch the status "
          "bar: the endpoint arrives as sigma x «SXEND» millimetres, "
          "and the run is saved automatically. While a longer "
          "simulation is in flight, the pill reads RUNNING, the red "
          "Stop button arms — cancelling at the next element — and "
          "the toolbar slider doubles as a progress bar with a live "
          "percentage. Then Results: six K P I "
          "cards — end sigma x, y and z, emittance growth, "
          "transmission and loss. Sigma x and y and the «GROWTH» "
          "growth fill in; the rest stay dashed on an envelope run — "
          "transmission and loss are particle concepts. Below, a wall "
          "of cards, one per quantity, each with a live sparkline; "
          "cards fill only when the physics exists. Import Results "
          "reloads any saved run, and a click opens the full plot."),
    Scene("110_rms",
          "The R M S beam size plot, exactly as it opens: show "
          "aperture is ticked, so the twenty and thirty millimetre "
          "bores frame the picture and the beam is a thin ribbon "
          "through the middle — far from where a real machine would "
          "scrape. Untick it, and the axes zoom to the physics. "
          "Blue is horizontal, green is vertical, the strip above "
          "marks each quadrupole — and wherever one plane peaks, the "
          "other dips. That alternating ripple is the fingerprint of "
          "strong focusing, and you just computed it.",
          min_s=8.0),
    Scene("112_more_tiles",
          "Two more tiles a first run already fills. Transverse "
          "Twiss: alpha and beta rippling with the cell's period — "
          "the beam size again, in optics language. And energy: "
          "kinetic energy dead flat at one M E V, gamma constant — no "
          "cavities — while the transmission panel underneath stays "
          "empty for an envelope run, for the same reason as the "
          "dashed K P I cards."),
    Scene("113_project_on_disk",
          "On disk it is all one folder. Straight after the wizard: "
          "the lattice copy and the project file. After the first "
          "run, a runs directory has joined them — the H D F five "
          "archive and its open P M D companion, saved automatically. "
          "The project file records "
          "that directory as a relative path, so the folder moves as "
          "one unit."),
    Scene("114_save_and_dirty",
          "Nudge a quadrupole gradient "
          "through the undo-aware edit path, and the amber unsaved "
          "pill lights the moment memory differs from disk. Control S "
          "writes the lattice — saved, pill out. Save Project does "
          "the same for the project side: beam and settings. And "
          "closing with unsaved work always prompts first."),
    Scene("116_relaunch_restore",
          "Come back tomorrow and HELIX reopens where you left off. "
          "This is a genuinely fresh window: on startup it found the "
          "last project and restored it — the status bar says so — "
          "lattice, beam and settings together. The restore is "
          "deliberately quiet; a missing or broken file is skipped, "
          "never a popup."),
    Scene("118_docs_tour",
          "Help, then Documentation, opens the manual — the same one "
          "F one deep-links into. The installation chapter closes "
          "with the common issues list: the exact error texts you "
          "might meet, each with its fix — permission denied on the "
          "launcher, a missing Py Q T six, a splash with no main "
          "window, the mac O S lib omp clash."),
    Scene("120_outro",
          "And that is the whole loop: install, verify, launch, "
          "create a project, load a lattice, define a matched beam, "
          "run, and read the plots — with everything saved and "
          "restorable in one folder. Ahead: the full interface tour, "
          "every tab in depth, space charge, matching, and the voice "
          "assistant. The manual is one keypress away. See you "
          "there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def _inject(name: str, **tok) -> None:
    sc = scene_by_name(name)
    for k, v in tok.items():
        sc.narration = sc.narration.replace("«" + k + "»", str(v))


# ---------------------------------------------------------------------------
# QtWebEngine child process — renders the built manual (site/) offscreen.
# Kept OUT of the main process so the Chromium stack never coexists with
# the InterphaseWindow.  Writes:
#   SHOTS/022_prereqs.png            still of the Prerequisites table
#   SHOTS/webquad_page.png           still of the Quadrupole chapter
#   FRAMES/118_docs_tour/*.png       genuine scroll through common issues
#   WORK/web_meta.json               achieved frame count + fps
# ---------------------------------------------------------------------------
def _webshots_child() -> None:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
    from PyQt6 import QtWebEngineWidgets   # noqa: F401 — MUST precede QApplication
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import QApplication

    app = QApplication(["ep01web"])        # argv0 required by WebEngine
    v = QWebEngineView()
    v.resize(1920, 1080)
    v.show()
    state: dict = {}
    v.loadFinished.connect(lambda ok: state.__setitem__("done", ok))

    def pump(sec: float) -> None:
        t0 = _time.monotonic()
        while _time.monotonic() - t0 < sec:
            app.processEvents()
            _time.sleep(0.01)

    def load(path: Path) -> None:
        state.pop("done", None)
        v.load(QUrl.fromLocalFile(str(path)))
        t0 = _time.monotonic()
        while "done" not in state and _time.monotonic() - t0 < 30:
            app.processEvents()
            _time.sleep(0.01)
        assert state.get("done"), f"webshots: load failed for {path}"
        pump(2.0)                          # let Chromium paint fully

    def js(code: str, wait: float = 0.5) -> None:
        v.page().runJavaScript(code)
        pump(wait)

    SHOTS.mkdir(parents=True, exist_ok=True)
    inst = SITE / "01_getting_started" / "02_installation.html"

    # 022 — prerequisites table
    load(inst)
    js("document.getElementById('prerequisites').scrollIntoView();"
       "window.scrollBy(0,-120);", 1.0)
    v.grab().save(str(SHOTS / "022_prereqs.png"))

    # 118 — genuine scroll through 'Common installation issues'
    js("document.getElementById('common-installation-issues')"
       ".scrollIntoView(); window.scrollBy(0,-80);", 1.0)
    fdir = FRAMES / "118_docs_tour"
    fdir.mkdir(parents=True, exist_ok=True)
    for old in fdir.glob("*.png"):
        old.unlink()
    n = 0
    t0 = _time.monotonic()
    for _hold in range(10):                # settle on the heading first
        v.grab().save(str(fdir / f"{n:05d}.png")); n += 1
        pump(0.10)
    for _step in range(70):
        js("window.scrollBy(0, 46);", 0.10)
        v.grab().save(str(fdir / f"{n:05d}.png")); n += 1
    wall = max(_time.monotonic() - t0, 1e-6)

    # 082 tail — the Quadrupole chapter F1 opens
    load(SITE / "03_elements" / "02_quadrupole.html")
    js("window.scrollTo(0, 120);", 0.8)    # past the site header chrome
    v.grab().save(str(SHOTS / "webquad_page.png"))

    (WORK / "web_meta.json").write_text(json.dumps(
        {"docs_frames": n, "docs_fps": n / wall}))
    sys.stdout.flush()
    os._exit(0)                            # skip Qt/WebEngine teardown


# ---------------------------------------------------------------------------
def _measure_smoke() -> tuple[str, str, str]:
    """Run the manual's 30-second smoke test for real; return
    (output_line, beta_str, gamma_str)."""
    code = (
        "from linac_gen.core.particle import PROTON\n"
        "from linac_gen.core.reference import ReferenceParticle\n"
        "ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)\n"
        "print(f'{PROTON.name} at {ref.w_kin} MeV: "
        "β={ref.beta:.4f}, γ={ref.gamma:.4f}')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=120)
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    assert "proton at 3.0 MeV" in line, f"smoke test failed: {r.stderr[-400:]}"
    beta = line.split("β=")[1].split(",")[0]
    gamma = line.split("γ=")[1].strip()
    return line, beta, gamma


def _measure_setup_probe() -> tuple[str, str]:
    """Reproduce setup.sh's interpreter probe (lines 24-37) and its
    uv-vs-venv branch line — measured on this machine, not invented."""
    import shutil as _shutil
    pybin = ""
    for c in ("python3.13", "python3.12", "python3.11", "python3.10",
              "python3"):
        p = _shutil.which(c)
        if not p:
            continue
        try:
            ok = subprocess.run(
                [p, "-c",
                 "import sys; raise SystemExit(0 if sys.version_info >= "
                 "(3,10) else 1)"],
                capture_output=True, timeout=20).returncode == 0
        except Exception:
            ok = False
        if ok:
            pybin = p
            break
    assert pybin, "no Python >= 3.10 found for the setup transcript"
    ver = subprocess.run(
        [pybin, "-c", "import platform; print(platform.python_version())"],
        capture_output=True, text=True, timeout=20).stdout.strip()
    # Same line setup.sh prints, with $HOME shown as "~" so the card
    # matches the neutral-path style of the other terminal cards.
    home = str(Path.home())
    shown = "~" + pybin[len(home):] if pybin.startswith(home) else pybin
    using = f"Using {shown} ({ver})"
    branch = ("uv detected - using it for a faster install."
              if _shutil.which("uv")
              else "Creating virtual environment in .venv/ ...")
    return using, branch


# ---------------------------------------------------------------------------
def capture_visuals() -> None:
    """Render every card and drive the real GUI for the screenshots."""
    # 0) WebEngine captures in an isolated child process (never share a
    #    process between Chromium and the main window).
    WORK.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--webshots"], cwd=str(ROOT), timeout=600,
                       capture_output=True, text=True)
    web_meta = json.loads((WORK / "web_meta.json").read_text())
    for need in ("022_prereqs.png", "webquad_page.png"):
        assert (SHOTS / need).is_file(), \
            f"webshots child produced no {need}: {r.stderr[-500:]}"
    sc118 = scene_by_name("118_docs_tour")
    sc118.frames_dir = str(FRAMES / "118_docs_tour")
    sc118.fps = max(float(web_meta["docs_fps"]), 1.0)

    from PyQt6.QtCore import QEventLoop, QPoint, QRect, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import QApplication, QPushButton, QScrollArea

    from pipeline import cards
    from pipeline.record import Recorder

    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    # QApplication must exist before ANY font/paint work (cards included)
    app = QApplication.instance() or QApplication(["ep01"])

    # ---- measured install facts (cards + narration) -------------------
    smoke_line, beta_s, gamma_s = _measure_smoke()
    using_line, branch_line = _measure_setup_probe()
    from linac_gen import __version__ as lg_version
    from linac_gen.cli.common import cpp_kernels_built
    kernels = cpp_kernels_built()
    if kernels:
        kern_line = "C++ kernels: built (fast PIC path active)."
    else:
        kern_line = ("C++ kernels: not built - HELIX runs on the "
                     "pure-Python fallback (~20x slower PIC).")
    _inject("046_smoke_test", BETA=beta_s, GAMMA=gamma_s)

    # ---- static cards -------------------------------------------------
    cards.title_card(s["010_title"],
                     "Install & First Run",
                     "Episode 1 — from zero to your first tracked beam")
    cards.terminal_card(s["030_get"], [
        ("cmd", "git clone https://github.com/Accel-Toolkit/HELIX.git"),
        ("out", "Cloning into 'HELIX'..."),
        ("out", "Receiving objects: 100%, done."),
        ("gap", ""),
        ("out", "# no git?  Code > Download ZIP on the same page gives the same folder"),
        ("cmd", "cd HELIX"),
    ], title="Get the code")
    cards.terminal_card(s["040_setup"], [
        ("cmd", "./setup.sh                    # Linux / macOS terminal"),
        ("out", using_line),
        ("out", branch_line),
        ("gap", ""),
        ("out", kern_line),
        ("out", f"Smoke test OK: linac_gen {lg_version} with PyQt6."),
        ("gap", ""),
        ("out", "Setup complete.  Launch the GUI any time with:  ./run_gui.sh"),
        ("out", "Launch it now? [y/N]"),
        ("gap", ""),
        ("out", "# LINAC_GEN_REQUIRE_CPP=1 ./setup.sh -> missing compiler = hard error"),
    ], title="One-click setup — real transcript")
    cards.terminal_card(s["042_setup_windows_mac"], [
        ("out", "Windows :  double-click setup.bat   (pre-checks the 260-char MAX_PATH limit)"),
        ("out", "macOS   :  double-click setup.command - runs setup.sh, keeps the window open:"),
        ("out", '           "Done - press any key to close..."'),
        ("gap", ""),
        ("cmd", "HELIX.exe --smoke             # Windows bundle - no Python install"),
        ("out", "OK  C++ kernel: <module 'linac_gen._pic_kernels' ...>"),
        ("out", "OK  cupy 14.0.1, CUDA runtime 12090, devices=1"),
        ("out", "OK  cupy compute: sum=6.0"),
        ("gap", ""),
        ("out", "# expected output as documented in the manual's installation chapter"),
    ], title="Windows & macOS")
    cards.terminal_card(s["044_manual_install"], [
        ("cmd", "git clone https://github.com/Accel-Toolkit/HELIX.git"),
        ("cmd", "cd HELIX"),
        ("cmd", "uv venv --python 3.11"),
        ("cmd", 'uv pip install -e ".[gui,dev]"'),
        ("out", "~400 MB of wheels: PyQt6, torch, numpy, scipy, h5py, matplotlib ..."),
        ("gap", ""),
        ("out", "# macOS, once beforehand: xcode-select --install   (C++ compilers)"),
        ("gap", ""),
        ("cmd", 'pip install -e ".[gpu]"       # NVIDIA CUDA only'),
        ("out", "# silent no-op on macOS - Apple Silicon gets Metal via the bundled torch"),
    ], title="Manual install (pip / uv)")
    cards.terminal_card(s["046_smoke_test"], [
        ("cmd", "python - <<'EOF'              # the manual's 30-second smoke test"),
        ("out", "from linac_gen.core.particle import PROTON"),
        ("out", "from linac_gen.core.reference import ReferenceParticle"),
        ("out", "ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)"),
        ("out", "print(f'{PROTON.name} at {ref.w_kin} MeV: β={ref.beta:.4f}, γ={ref.gamma:.4f}')"),
        ("out", "EOF"),
        ("gap", ""),
        ("out", smoke_line + "        <- captured live in this build"),
        ("gap", ""),
        ("cmd", "pytest tests/                 # full suite - about 3 minutes"),
    ], title="Verify the install")
    cards.terminal_card(s["050_launch"], [
        ("cmd", "./run_gui.sh"),
        ("out", "Launching HELIX GUI from ~/HELIX (python: ~/HELIX/.venv/bin/python) ..."),
        ("gap", ""),
        ("out", "# the same app, three other ways:"),
        ("cmd", "./HELIX.command               # macOS - double-clickable in Finder"),
        ("cmd", "run_gui.bat                   # Windows"),
        ("cmd", "python -m linac_gen_gui.interphase.app    # any platform, directly"),
    ], title="Launch")
    cards.terminal_card(s["054_env_overrides"], [
        ("out", "# environment knobs the launchers honour:"),
        ("cmd", "HELIX_PYTHON=/opt/homebrew/bin/python3.12 ./HELIX.command"),
        ("cmd", "QT_QPA_PLATFORM=cocoa ./run_gui.sh"),
        ("cmd", "LINAC_GEN_USE_GPU=cpu ./run_gui.sh"),
        ("cmd", "QT_QPA_PLATFORM=offscreen ./run_gui.sh    # headless - no window"),
    ], title="Environment overrides")
    starter = "; my_linac — created by HELIX New Project"
    cards.terminal_card(s["074_blank_deck"], [
        ("out", starter),
        ("out", "TITLE my_linac"),
        ("out", "FREQ 162.5"),
        ("out", "DRIFT 100 15"),
        ("out", "END"),
    ], title="my_linac.dat — the blank-project starter deck")
    cards.outro_card(s["120_outro"], [
        "Next — the full GUI tour",
        "Then — every tab in depth, one episode each",
        "Later — space charge, matching, the voice assistant",
        "F1 — the manual is one keypress away",
    ])

    # ---- splash (top-level dialog: grab it DIRECTLY, never via win) ---
    from linac_gen_gui.interphase.splash import HelixSplash
    splash = HelixSplash()
    splash.show()
    for _ in range(12):
        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    simg = splash.grab().toImage()
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    canvas.fill(QColor("#0b1220"))
    p = QPainter(canvas)
    p.drawImage((1920 - simg.width()) // 2, (1080 - simg.height()) // 2, simg)
    p.end()
    canvas.save(s["052_splash"])
    splash.hide()

    # ---- main window --------------------------------------------------
    from linac_gen_gui.interphase import app as app_mod
    from linac_gen_gui.interphase import manual_help as mh
    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    from PyQt6.QtWidgets import QMessageBox as _RealMB

    # Hang-proofing: ANY modal QMessageBox spins forever offscreen.  The
    # app module gets a silent stand-in (nothing in this storyboard
    # should trigger one — a log line here is a finding to investigate).
    class _SilentMB(_RealMB):
        @staticmethod
        def warning(*a, **k):
            print(f"[mb.warning] {a[1:3]}"); return _RealMB.StandardButton.Ok
        @staticmethod
        def critical(*a, **k):
            print(f"[mb.critical] {a[1:3]}"); return _RealMB.StandardButton.Ok
        @staticmethod
        def information(*a, **k):
            print(f"[mb.information] {a[1:3]}"); return _RealMB.StandardButton.Ok
        @staticmethod
        def question(*a, **k):
            print(f"[mb.question] {a[1:3]}"); return _RealMB.StandardButton.No
    app_mod.QMessageBox = _SilentMB
    # F1 demo: run the REAL resolve path but do not hijack the user's
    # desktop browser from an offscreen build.
    mh._open_path = lambda p: True

    win = InterphaseWindow()
    win._release_fetcher = lambda *a, **k: None   # injectable seam — no network
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def shot(key: str) -> None:
        settle()
        win.grab().save(s[key])

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    rec = Recorder(FRAMES, settle)
    win_frame = lambda: win.grab().toImage()               # noqa: E731

    # -- composite helpers ---------------------------------------------
    def zoom_frame(w=win, region: str = "segs", crop_w: int = 780,
                   scale: float = 2.4):
        """Window grab + a magnified inset of part of the status bar.
        region='segs' magnifies from the left edge (state pill onward);
        'msg' centres on the transient-message segment.  Real pixels,
        only enlarged."""
        img = w.grab().toImage()
        sb = w._statusbar
        tl = sb.mapTo(w, QPoint(0, 0))
        if region == "msg":
            x0 = max(sb._msg_seg.mapTo(w, QPoint(0, 0)).x() - 16, 0)
        else:
            x0 = tl.x()
        crop_w2 = min(crop_w, 1920 - x0)
        crop = img.copy(QRect(x0, tl.y(), crop_w2, sb.height()))
        z = crop.scaled(int(crop_w2 * scale), int(sb.height() * scale),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        out = img.copy()
        pp = QPainter(out)
        zx = (1920 - z.width()) // 2
        zy = tl.y() - z.height() - 26
        pp.fillRect(zx - 3, zy - 3, z.width() + 6, z.height() + 6,
                    QColor("#22d3ee"))
        pp.drawImage(zx, zy, z)
        pp.end()
        return out

    def widget_zoom_frame(wdg, scale: float = 3.0):
        """Window grab + a magnified inset of one toolbar widget."""
        img = win.grab().toImage()
        tl = wdg.mapTo(win, QPoint(0, 0))
        pad = 8
        r = QRect(max(tl.x() - pad, 0), max(tl.y() - pad, 0),
                  wdg.width() + 2 * pad, wdg.height() + 2 * pad)
        crop = img.copy(r)
        z = crop.scaled(int(r.width() * scale), int(r.height() * scale),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        out = img.copy()
        pp = QPainter(out)
        zx = min(max(tl.x() + wdg.width() // 2 - z.width() // 2, 12),
                 1920 - z.width() - 12)
        zy = tl.y() + wdg.height() + 30
        pp.fillRect(zx - 3, zy - 3, z.width() + 6, z.height() + 6,
                    QColor("#22d3ee"))
        pp.drawImage(zx, zy, z)
        pp.end()
        return out

    # ---- 020 hero -----------------------------------------------------
    shot("020_what")

    # ---- 060 nine-tab tour (motion) -----------------------------------
    tab_order = ["Beam", "Lattice", "Matching", "Numerics", "Surrogates",
                 "Param Study", "Error Study", "Failure Study", "Results"]
    rec.start("060_tour")
    rec.hold(win_frame, 1.5)
    for label in tab_order:
        go_tab(label)
        rec.hold(win_frame, 3.6)
    rec.finish(scene_by_name("060_tour"))
    go_tab("Beam")

    # ---- 062 menus opened for real (motion) ---------------------------
    menu_btns = {b.text(): b for b in win._toolbar.findChildren(QPushButton)
                 if b.menu() is not None}
    for name in ("File", "Simulate", "Tools", "Help"):
        assert name in menu_btns, f"menu button {name!r} not found"

    def menu_frame_fn(btn):
        menu = btn.menu()
        def _f():
            img = win.grab().toImage()
            out = img.copy()
            pp = QPainter(out)
            mimg = menu.grab().toImage()
            pt = btn.mapTo(win, QPoint(0, btn.height()))
            pp.fillRect(pt.x() - 2, pt.y() - 2, mimg.width() + 4,
                        mimg.height() + 4, QColor("#2b3b55"))
            pp.drawImage(pt.x(), pt.y(), mimg)
            pp.end()
            return out
        return _f

    rec.start("062_menus")
    holds = {"File": 8.5, "Simulate": 5.5, "Tools": 6.0, "Help": 7.0}
    for name in ("File", "Simulate", "Tools", "Help"):
        btn = menu_btns[name]
        menu = btn.menu()
        gpos = btn.mapToGlobal(QPoint(0, btn.height()))
        menu.popup(gpos)
        settle(6)
        rec.hold(menu_frame_fn(btn), holds[name])
        menu.close()
        settle(4)
    rec.finish(scene_by_name("062_menus"))

    # ---- 064 status bar decoded (motion, magnified) -------------------
    rec.start("064_statusbar")
    rec.hold(lambda: zoom_frame(region="segs"), 8.0)
    # The EXACT degraded-install warning the app emits at startup when
    # cpp_kernels_built() is False (app.py) — rendered through the real
    # status_message sink; narration frames it as conditional.
    win.state.status_message.emit(
        "C++ PIC kernels not built — pure-Python space charge "
        "(~20x slower); rebuild with pip install -e . "
        "(needs a C++ compiler)")
    settle(4)
    rec.hold(lambda: zoom_frame(region="msg", crop_w=1100, scale=1.7), 8.0)
    rec.finish(scene_by_name("064_statusbar"))
    win._statusbar._clear_message()

    # ---- 066 font shortcuts (motion) ----------------------------------
    from linac_gen_gui.interphase import theme
    fspin = win._toolbar._font_spin
    rec.start("066_shortcuts")
    rec.hold(lambda: widget_zoom_frame(fspin), 2.5)
    for _ in range(3):
        win._bump_font(+1)                 # the Ctrl+= slot
        settle(6)
        rec.hold(lambda: widget_zoom_frame(fspin), 1.6)
    rec.hold(lambda: widget_zoom_frame(fspin), 1.5)
    win._apply_font_size(theme.FONT_SIZE)  # the Ctrl+0 slot
    settle(6)
    rec.hold(lambda: widget_zoom_frame(fspin), 2.5)
    rec.finish(scene_by_name("066_shortcuts"))
    win._statusbar._clear_message()

    # ---- 070/072/075/076 — the New Project wizard ---------------------
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    from linac_gen_gui.interphase.dialogs.new_project import NewProjectDialog

    # Validation popups must be NON-modal offscreen: swap the module's
    # QMessageBox for a shim that shows a real, grabbable QMessageBox.
    class _PopupMB:
        last = None
        @staticmethod
        def warning(parent, title, text, *a, **k):
            mb = _RealMB()
            mb.setIcon(_RealMB.Icon.Warning)
            mb.setWindowTitle(title)
            mb.setText(text)
            mb.show()
            _PopupMB.last = mb
            return _RealMB.StandardButton.Ok
        critical = warning
    np_mod.QMessageBox = _PopupMB

    ex_dir = ROOT / "examples"
    examples = [("FODO cell", str(ex_dir / "fodo_cell.dat")),
                ("Solenoid channel", str(ex_dir / "solenoid_channel.dat")),
                ("DTL section", str(ex_dir / "dtl_section.dat"))]
    for _lab, _p in examples:
        assert Path(_p).is_file(), f"example missing: {_p}"

    dlg = NewProjectDialog(win,
                           start_dir=str(Path.home() / "HELIX-projects"),
                           examples=examples)
    dlg.resize(900, 640)
    dlg.show()
    settle(8)
    dlg_frame = lambda: dlg.grab().toImage()               # noqa: E731

    rec.start("070_wizard")
    rec.hold(dlg_frame, 1.5)
    rec.type_into(dlg_frame, dlg._name, "my_first_beam")
    rec.hold(dlg_frame, 2.0)
    dlg._rb_import.setChecked(True)        # real radio: import row wakes
    rec.hold(dlg_frame, 2.5)
    dlg._rb_example.setChecked(True)       # real radio: example combo wakes
    rec.hold(dlg_frame, 1.5)
    dlg._example_combo.setCurrentIndex(0)  # FODO cell
    rec.hold(dlg_frame, 3.0)
    rec.finish(scene_by_name("070_wizard"))

    # -- 072 validation (motion, dialog + popup composite) --------------
    def dlg_popup_frame():
        out = QImage(1920, 1080, QImage.Format.Format_RGB32)
        out.fill(QColor("#0b1220"))
        pp = QPainter(out)
        dimg = dlg.grab().toImage()
        pp.drawImage(70, (1080 - dimg.height()) // 2, dimg)
        mb = _PopupMB.last
        if mb is not None and mb.isVisible():
            pimg = mb.grab().toImage()
            px = min(1000, 1920 - pimg.width() - 40)
            py = (1080 - pimg.height()) // 2
            pp.fillRect(px - 4, py - 4, pimg.width() + 8,
                        pimg.height() + 8, QColor("#f59e0b"))
            pp.drawImage(px, py, pimg)
        pp.end()
        return out

    rec.start("072_wizard_validation")
    rec.hold(dlg_popup_frame, 1.0)
    rec.type_into(dlg_popup_frame, dlg._name, "CON")
    dlg._accept()                          # real validation path -> popup
    settle(6)
    rec.hold(dlg_popup_frame, 4.5)
    if _PopupMB.last: _PopupMB.last.hide()
    rec.type_into(dlg_popup_frame, dlg._name, "beam.")
    dlg._accept()
    settle(6)
    rec.hold(dlg_popup_frame, 4.0)
    if _PopupMB.last: _PopupMB.last.hide()
    rec.type_into(dlg_popup_frame, dlg._name, "my_first_beam")
    rec.hold(dlg_popup_frame, 1.5)
    rec.finish(scene_by_name("072_wizard_validation"))

    # -- 075 import-mode still ------------------------------------------
    dlg._rb_import.setChecked(True)
    settle(6)
    dlg.grab().save(s["075_import_mode"])
    dlg._rb_example.setChecked(True)
    settle(4)

    # -- accept for REAL into a sandboxed location ----------------------
    sandbox_projects = Path(tempfile.mkdtemp()) / "HELIX-projects"
    sandbox_projects.mkdir(parents=True)
    dlg._location.setText(str(sandbox_projects))
    dlg._accept()
    res = dlg.project_result()
    assert res and res["mode"] == "example", "wizard accept failed"
    proj_dir = Path(res["project_dir"])

    # Post-dialog replica of app._new_project (that slot needs dlg.exec,
    # which is modal and hangs offscreen): parse + load + settings +
    # .lgproj with relative calc_dir, exactly as app.py does.
    lattice, _meta = _parse_lattice_file(res["lattice_path"])
    win.state.set_lattice(lattice, res["lattice_path"])
    st = app_mod._settings()
    st.setValue(app_mod._SETTINGS_LAST_LATTICE, res["lattice_path"])
    st.setValue(app_mod._SETTINGS_LAST_DIR, res["project_dir"])
    st.remove(app_mod._SETTINGS_LAST_PROJECT)
    st.setValue(app_mod._SETTINGS_CALC_DIR, str(proj_dir / "runs"))
    fp = os.path.join(res["project_dir"], res["name"] + ".lgproj")
    assert win._write_project_file(fp, extra={"calc_dir": "runs"})
    settle(6)
    # The wizard flow ends with the .lgproj written — list the folder
    # exactly as a user would see it after clicking OK.
    ls_before = sorted(p.name for p in proj_dir.iterdir())

    n_elem = len(lattice.elements)
    total_mm = sum(e.length for e in lattice.elements)
    _inject("076_project_created", NELEM=n_elem)
    _inject("080_fodo", NELEM=n_elem,
            LTOT=f"{total_mm / 1000:.1f} metres")
    _inject("100_results", LTOTMM=f"{total_mm:.0f}")

    # -- 076 clip: save confirmation + Open Recent entry ----------------
    rec.start("076_project_created")
    rec.hold(lambda: zoom_frame(region="msg", crop_w=900, scale=2.0), 5.0)
    rmenu = win._toolbar._recent_menu
    fbtn = menu_btns["File"]
    rmenu.popup(fbtn.mapToGlobal(QPoint(30, fbtn.height() + 40)))
    settle(6)

    def recent_frame():
        img = win.grab().toImage()
        out = img.copy()
        pp = QPainter(out)
        mimg = rmenu.grab().toImage()
        pt = fbtn.mapTo(win, QPoint(30, fbtn.height() + 40))
        pp.fillRect(pt.x() - 2, pt.y() - 2, mimg.width() + 4,
                    mimg.height() + 4, QColor("#2b3b55"))
        pp.drawImage(pt.x(), pt.y(), mimg)
        pp.end()
        return out

    rec.hold(recent_frame, 5.0)
    rmenu.close()
    settle(4)
    rec.finish(scene_by_name("076_project_created"))

    # ---- 080 FODO in the Lattice tab (motion: QF then QD) -------------
    go_tab("Lattice")
    settle(8)
    quads = [el for el in lattice.elements
             if type(el).__name__ == "Quadrupole"]
    qf = next(q for q in quads if q.gradient > 0)
    qd = next(q for q in quads if q.gradient < 0)
    rec.start("080_fodo")
    rec.hold(win_frame, 3.0)
    win.state.set_selected(qf)             # real selection -> inspector
    settle(6)
    rec.hold(win_frame, 5.0)
    win.state.set_selected(qd)
    settle(6)
    rec.hold(win_frame, 5.0)
    rec.finish(scene_by_name("080_fodo"))

    # ---- 082 F1 contextual help (motion) ------------------------------
    win.state.set_selected(None)
    win._statusbar._clear_message()
    settle(4)
    quad_page = QImage(str(SHOTS / "webquad_page.png"))
    assert not quad_page.isNull()
    rec.start("082_f1_help")
    win._open_manual_for_selected()        # F1 slot, nothing selected
    settle(4)
    rec.hold(lambda: zoom_frame(region="msg", crop_w=900, scale=2.0), 4.0)
    win.state.set_selected(qf)
    settle(4)
    win._open_manual_for_selected()        # F1 slot -> quadrupole chapter
    settle(4)
    rec.hold(lambda: zoom_frame(region="msg", crop_w=900, scale=2.0), 4.0)
    rec.hold(lambda: quad_page, 6.0)       # the page F1 opens (webshot)
    rec.finish(scene_by_name("082_f1_help"))
    win._statusbar._clear_message()

    # ---- 090 Beam tab: matched values typed into the REAL form --------
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching import find_fodo_cells, find_matched_input_twiss
    ref = ReferenceParticle(species=PROTON, w_kin=1.0, frequency=352.21)
    cells = find_fodo_cells(lattice)
    c0, c1 = (cells[0] if cells else (0, len(lattice.elements) - 1))
    tw = find_matched_input_twiss(lattice, ref, c0, c1)

    def _spoken(v: float) -> str:
        # kokoro-safe sign: spell the minus rather than trusting "-1.44"
        return (f"minus {abs(v):.2f}" if v < 0 else f"{v:.2f}")
    _inject("090_beam",
            AX=_spoken(tw["alpha_x"]), BX=_spoken(tw["beta_x"]),
            AY=_spoken(tw["alpha_y"]), BY=_spoken(tw["beta_y"]))

    go_tab("Beam")
    settle(8)
    bt = win.beam_tab
    rec.start("090_beam")
    rec.hold(win_frame, 2.0)
    bt._species.setCurrentText("proton")
    settle(4); rec.hold(win_frame, 1.5)
    bt._energy.setValue(1.0)
    settle(4); rec.hold(win_frame, 1.5)
    bt._freq.setValue(352.21)
    settle(4); rec.hold(win_frame, 1.0)
    bt._current.setValue(0.0)
    settle(4); rec.hold(win_frame, 1.5)
    for spin, val in ((bt._alpha_x, tw["alpha_x"]),
                      (bt._beta_x, tw["beta_x"]),
                      (bt._alpha_y, tw["alpha_y"]),
                      (bt._beta_y, tw["beta_y"])):
        spin.setValue(val)
        settle(3)
        rec.hold(win_frame, 1.2)
    rec.hold(win_frame, 1.0)
    bt._apply_btn.click()                  # real Apply -> config + preview
    rec.wait_until(win_frame, lambda: win.state.beam_config is not None,
                   cap_s=30, stable_s=0.2)
    rec.hold(win_frame, 4.0)
    rec.finish(scene_by_name("090_beam"))
    cfgv = win.state.beam_config
    assert cfgv is not None and cfgv.species == "proton" \
        and abs(cfgv.energy - 1.0) < 1e-9, "beam Apply did not take"

    # ---- 100 the run + Results wall (motion) --------------------------
    rec.start("100_results", fps=12.0)
    rec.hold(win_frame, 1.0)
    assert win._toolbar._run_env_btn.isEnabled()
    win._toolbar._run_env_btn.click()      # the real Run Envelope button
    rec.wait_until(win_frame, lambda: win.state.results is not None,
                   cap_s=60, stable_s=0.3)
    assert win.state.results is not None, "envelope run produced no results"
    settle(4)
    rec.hold(lambda: zoom_frame(region="msg", crop_w=1100, scale=1.7), 3.5)
    go_tab("Results")
    settle(8)
    rec.hold(win_frame, 4.0)
    sa = win.results_tab.findChild(QScrollArea)
    vsb = sa.verticalScrollBar()
    top = vsb.value()
    span = max(vsb.maximum() - top, 1)
    for i in range(100):
        vsb.setValue(top + int(span * (i + 1) / 100))
        rec.tick(win_frame)
    rec.hold(win_frame, 2.0)
    vsb.setValue(0)
    settle(4)
    rec.finish(scene_by_name("100_results"))

    resv = win.state.results
    sx_end = float(resv.sigma_x[-1])
    growth = float(resv.emit_x[-1] / max(resv.emit_x[0], 1e-12))
    # EnvelopeResults carries NO transmission/particle counts — the
    # transmission + loss KPI cards stay dashed and the energy popup's
    # transmission panel stays empty; the narration teaches exactly that.
    assert not hasattr(resv, "transmission")
    _inject("100_results", SXEND=f"{sx_end:.3f}", GROWTH=f"{growth:.2f}")

    # ---- 110 RMS popup: the aperture toggle taught, not hidden --------
    win.show_result_plot("rms")
    settle(8)
    pop = win.results_tab._popups.get("rms")
    assert pop is not None
    pop.resize(1600, 900)
    settle(8)
    assert pop._chk_ap.isChecked(), "aperture overlay no longer defaults on"
    pop_frame = lambda: pop.grab().toImage()               # noqa: E731
    rec.start("110_rms")
    rec.hold(pop_frame, 7.0)
    pop._chk_ap.setChecked(False)          # real toggle -> axes re-zoom
    settle(6)
    rec.hold(pop_frame, 7.0)
    rec.finish(scene_by_name("110_rms"))
    pop.hide()

    # ---- 112 Twiss + Energy popups (motion) ---------------------------
    win.show_result_plot("twiss")
    settle(8)
    pop_t = win.results_tab._popups.get("twiss")
    assert pop_t is not None
    pop_t.resize(1600, 900)
    settle(8)
    rec.start("112_more_tiles")
    rec.hold(lambda: pop_t.grab().toImage(), 7.0)
    pop_t.hide()
    win.show_result_plot("energy")
    settle(8)
    pop_e = win.results_tab._popups.get("energy")
    assert pop_e is not None
    pop_e.resize(1600, 900)
    settle(8)
    rec.hold(lambda: pop_e.grab().toImage(), 7.0)
    rec.finish(scene_by_name("112_more_tiles"))
    pop_e.hide()

    # ---- 113 what landed on disk (measured listings) ------------------
    ls_after = sorted(p.name + ("/" if p.is_dir() else "")
                      for p in proj_dir.iterdir())
    runs_dir = proj_dir / "runs"
    ls_runs = (sorted(p.name for p in runs_dir.iterdir())
               if runs_dir.is_dir() else [])
    assert ls_runs, "no auto-saved run artefacts in the project runs/"
    cards.terminal_card(s["113_project_on_disk"], [
        ("out", "# straight after the wizard:"),
        ("cmd", "ls my_first_beam/"),
        ("out", "    ".join(ls_before)),
        ("gap", ""),
        ("out", "# after the first run:"),
        ("cmd", "ls my_first_beam/"),
        ("out", "    ".join(ls_after)),
        ("cmd", "ls my_first_beam/runs/"),
        *[("out", nm) for nm in ls_runs[:4]],
        ("gap", ""),
        ("out", '# my_first_beam.lgproj stores  "calc_dir": "runs"  (relative)'),
    ], title="The project on disk — real listing")

    # ---- 114 unsaved pill -> Save (motion, magnified) -----------------
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    go_tab("Lattice")                      # show the inspector while editing
    win.state.set_selected(qf)
    settle(8)
    rec.start("114_save_and_dirty")
    rec.hold(lambda: zoom_frame(region="segs"), 2.5)
    old_g = qf.gradient
    win.state.bus.do(ParamChangeCommand(qf, "gradient", old_g,
                                        round(old_g * 1.04, 3)))
    settle(6)
    rec.hold(lambda: zoom_frame(region="segs"), 5.0)   # pill ON
    win._save_lattice()                    # the Ctrl+S slot
    settle(6)
    rec.hold(lambda: zoom_frame(region="segs", crop_w=1100, scale=1.7),
             4.5)                                       # pill OUT + Saved
    win._save_project()                    # silent: project path known
    settle(6)
    rec.hold(lambda: zoom_frame(region="msg", crop_w=900, scale=2.0), 4.5)
    rec.finish(scene_by_name("114_save_and_dirty"))

    # ---- 116 a second window restores the session ---------------------
    win2 = InterphaseWindow()
    win2._release_fetcher = lambda *a, **k: None
    win2.resize(1920, 1080)
    win2.show()
    settle(20)                             # QTimer(0) restore fires here
    msg = win2._statusbar._msg_seg.text()
    assert "Restored project" in msg, f"unexpected restore message: {msg!r}"
    rec.start("116_relaunch_restore")
    rec.hold(lambda: zoom_frame(w=win2, region="msg", crop_w=1000,
                                scale=1.8), 9.0)
    rec.finish(scene_by_name("116_relaunch_restore"))
    win2.hide()                            # NEVER win.close() (modal trap)

    # ---- attach stills ------------------------------------------------
    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    for sc in SCENES:
        assert "«" not in sc.narration, \
            f"unfilled narration token in {sc.name}"
    # NO win.close(): dirty state -> modal unsaved-changes confirm ->
    # offscreen spin-hang (see ep07). Process exit tears Qt down.


def main() -> None:
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)     # worker QThreads ran in this process — skip teardown


if __name__ == "__main__":
    if "--webshots" in sys.argv:
        _webshots_child()
    else:
        main()
