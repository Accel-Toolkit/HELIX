"""Episode 12 — Space charge & real beams (the physics special).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep12_space_charge.py
Output: tutorials/rendered/ep12_space_charge.mp4 (+ .srt)

Every physics claim is a filmed, measured demonstration (numbers spoken
in the narration were read from the runs this script performs — see the
``[measure]`` lines in the build log):

  * the matched 1 MeV FODO from ep01 with a REALISTIC bunch (10 deg /
    3 keV — the BeamConfig default longitudinal phase space is a 0.06 mm
    pancake with a 55 % energy spread and made every space-charge number
    an artefact);
  * a REAL current scan 0→15 mA through the Param Study engine (Overlay
    view, select-all + Draw on camera);
  * the energy law as a flip-book of three current-pair studies, each
    matched at its own energy (1 / 3 / 10 MeV);
  * tune depression on an SC-matched 5 mA beam; the tune footprint and
    the Hofmann chart on the shipped demo linac (refusals filmed first);
  * the emittance-exchange validation pair tracked with the full PIC;
  * MP vs envelope at 5 mA (RMS popup controls on camera, then the pair);
  * the mismatched-FODO halo benchmark (phase-space colour modes, H(s));
  * DC workflow: Beam-tab continuous toggle live, DC kernels, Sacherer,
    the LEBT compensation (SCC) dashboard computing on camera;
  * CSR in the shipped chicane (checkbox on camera, measured pair);
  * the documented TraceWin parity story.

The PIC-cycle manual tour is rendered by QtWebEngine in a SUBPROCESS
(``--manual-frames DIR``) so the web engine never shares a process with
the offscreen GUI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
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

WORK = ROOT / "tutorials" / "rendered" / "ep12_work"
SHOTS = WORK / "shots"
FRAMES = WORK / "frames"
OUT = ROOT / "tutorials" / "rendered" / "ep12_space_charge.mp4"

# Realistic bunch for the 1 MeV / 352.21 MHz FODO: sigma_phi = 10 deg,
# sigma_W = 3 keV  (sigma_phi = sqrt(eps_z*beta_z), sigma_W = sqrt(eps_z/beta_z)).
EZ, BZ = 0.03, 3333.333

SCENES = [
    Scene("010_title",
          "Episode twelve, and a change of pace. Almost no new tabs today. "
          "This one is about the physics that makes intense linacs hard: "
          "space charge. Every particle in the bunch repels every other, "
          "and most of what is difficult about machines like PIP two traces "
          "back to this force. Today we measure it, on camera.",
          min_s=6.0),
    Scene("020_what",
          "The essentials. Space charge is the collective Coulomb repulsion "
          "of the bunch, a defocusing lens in every plane. Its strength is "
          "the generalised perveance: current over beta cubed gamma cubed. "
          "At low energy the beta cubed bites, slow particles are dense "
          "along the line and easy to deflect. Near the speed of light "
          "magnetic attraction cancels electric repulsion. So the source, "
          "the R F Q and the M E B T are where space charge rules. We "
          "measure that law shortly.",
          min_s=7.0),
    Scene("032_pic_manual_tour",
          "The particle in cell cycle from the manual, one F one press from "
          "the G U I. Boost into the bunch rest frame, set up the grid, "
          "deposit the charge, solve Poisson with a doubled grid F F T and "
          "the integrated Green function, gather, kick, boost back. The "
          "reference particle is never touched. Each macroparticle carries "
          "a fixed share of the current, so the field scales with the "
          "surviving current. It is the Qiang two thousand and six "
          "algorithm, the kernel OPAL and Cheetah use. Below the figure, "
          "the three F F T backends: C P U and CUDA in double precision, "
          "Apple's Metal G P U in single.",
          min_s=8.0),
    Scene("035_sc_knobs",
          "The knobs. Base step two is the kick cadence, fifty space charge "
          "kicks per metre. Grid forty eight cubed and extent seven sigma "
          "set the Poisson box. P I C backend: auto means CUDA when "
          "present, otherwise the C P U, never the single precision Apple G "
          "P U unless you ask for M P S. S C engine numpy is production; "
          "torch is the differentiable P I C behind gradient matching, "
          "bunched beams only. Green's function stays on I G F; point only "
          "reproduces legacy runs. Cloud in cell is the kernel; T S C costs "
          "three times more. Adaptive grid mode is for transport lines "
          "where the bunch outgrows its box. The D C kernel row comes "
          "later.",
          min_s=8.0),
    Scene("040_experiment",
          "The experiment. Episode one's matched F O D O at one M e V with "
          "a realistic bunch: ten degrees long, three k e V of energy "
          "spread. The Param Study manager scanned the current from zero to "
          "fifteen milliamps. This is the Overlay view: it pre selects "
          "three runs and redraws only on Draw, so we select all four and "
          "press it. At zero current the matched beam breathes gently. At "
          "five milliamps it peaks twenty four percent bigger. At ten, "
          "forty five. At fifteen, sixty seven percent. Matched at zero, "
          "badly mismatched at fifteen.",
          min_s=8.0),
    Scene("045_energy",
          "The energy law, measured. Same lattice, fifteen milliamps "
          "against zero, the beam matched at zero current for each energy, "
          "so within each pair the only difference is space charge. One M e "
          "V: the loaded beam peaks sixty seven percent above the unloaded "
          "one. Three M e V: thirty percent. Ten M e V: seven percent. Beta "
          "gamma squared grows tenfold from one to ten M e V, and the "
          "excess swell falls almost exactly tenfold. That is why linacs "
          "spend their space charge budget at the front end.",
          min_s=8.0),
    Scene("050_depression",
          "The standard measure is the tune depression, eta: the phase "
          "advance the beam accumulates over what it would with no current, "
          "per cell. This beam was matched at five milliamps with the space "
          "charge aware matcher, so no mismatch warning. The Period combo "
          "locked onto the auto detected four cell period; Recompute "
          "redraws. Green squares are the model, from the depressed channel "
          "maps; amber circles the beam, TraceWin's integrated one over "
          "beta ratio, valid only when matched. The model sits at 0.71 in "
          "the first cell and climbs to 0.83 by the fourth: no R F, so the "
          "bunch lengthens and the force fades. Eta z falls from 0.95 to "
          "0.58.",
          min_s=8.0),
    Scene("055_footprint",
          "Every particle sits at a different amplitude, so every particle "
          "has its own tune. The footprint popup tracks an amplitude ladder "
          "repeatedly through one cell with the space charge field frozen "
          "from the first pass and takes each tune with an F F T. Turns "
          "sets the resolution, one hundred twenty eight is about two point "
          "eight degrees; Particles sets the ladder. Our F O D O is too "
          "weakly focused for this tool above two milliamps, so here is the "
          "shipped Hofmann demo linac, five milliamps, a bare cell of forty "
          "point two degrees. Compute footprint. Core tunes near eleven "
          "degrees, the deepest depression at the smallest amplitude, and "
          "the cloud stretches up towards thirty five degrees as the "
          "amplitude grows.",
          min_s=8.0),
    Scene("062_refusal_fodo",
          "One level deeper. When both planes are loaded, energy can flow "
          "between them through parametric resonances, and the Hofmann "
          "stability chart maps where. It is strict about its assumptions, "
          "so ask it. Compute chart on the F O D O: no finite longitudinal "
          "channel tune in any cell, the chart coordinate R is undefined. "
          "No R F, no chart.",
          min_s=7.0),
    Scene("063_refusal_coupled",
          "The demo linac's fourth section, six solenoid cells. The caption "
          "already says transverse equals mode two: coupled normal modes. "
          "Compute chart, and it refuses again: an x y coupled lattice's "
          "tunes are normal modes one and two, not Hofmann's x and z "
          "planes, refusing rather than mislabelling a mode as x. A tool "
          "that knows its limits is worth more than one that always draws "
          "something.",
          min_s=7.0),
    Scene("064_hofmann_chart",
          "Back to section A, sixteen quadrupole cells, uncoupled and "
          "periodic. Chart steps trades resolution for solver time; one "
          "hundred took about three seconds here. Compute chart. The "
          "background is the computed growth rate from the corrected "
          "anisotropic K V dispersion relations, ported bitwise from the "
          "published solver. One dot per cell. Green: valid and stable. "
          "Red: flagged, growth above one percent inside the perturbative "
          "gate. Grey hollow: outside the gate. Amber ring: fold risk. One "
          "of thirteen valid cells is flagged. Legacy bands overlays the "
          "old heuristic, qualitative only. P unstable adds a Monte Carlo "
          "probability per cell under an engineering jitter budget. "
          "Recompute: maximum probability 0.96. That flag is not a fluke.",
          min_s=9.0),
    Scene("066_exchange_proof",
          "Does the chart predict reality? HELIX ships a validation pair: "
          "two one hundred and fifty cell channels with the same twenty "
          "milliamp beam. The resonant one, left, sits on the l equals two "
          "coupling band, every cell flagged. The control, right, has "
          "quadrupoles ten percent stiffer, off the band, no cell flagged. "
          "Both tracked with the full particle in cell solver, twenty "
          "thousand particles. On the resonant channel the longitudinal "
          "emittance, bottom panel, dips eighteen percent within the first twenty cells "
          "while the transverse emittances jump fifty percent, energy "
          "exchanged, not created; it then self detunes and settles at plus "
          "thirty eight and minus nine percent. The control shows one short "
          "transient and its longitudinal emittance ends within one and a "
          "half percent of where it started.",
          min_s=8.0),
    Scene("070_check",
          "The honesty check. The same matched F O D O, five milliamps, ten "
          "thousand particles, tracked both ways. Control R runs the "
          "envelope, control shift R the particles. This is the particle "
          "run's R M S popup with Show aperture on: the bore squashes the "
          "beam into a ribbon. Untick it and the plot rescales. Show "
          "lattice adds the element strip; Display can switch to dispersion "
          "corrected sizes.",
          min_s=7.0),
    Scene("071_check_pair",
          "Particles left, envelope right, same transverse axes. Along the "
          "whole line the sigmas agree to better than one percent, the "
          "particle run's own shot noise. That licenses the fast model for "
          "everyday work and matching. When they disagree, as in the "
          "overloaded D T L of the Results episode, believe the particles.",
          min_s=7.0),
    Scene("072_halo",
          "What the envelope cannot see. The shipped halo benchmark: twenty "
          "four F O D O cells, H minus at two point one M e V, five "
          "milliamps, matched with space charge and then deliberately "
          "mismatched forty percent in both planes. A mismatched beam under "
          "space charge pumps particles outward through the resonance "
          "between core oscillation and single particle motion. Colour the "
          "phase space by transverse radius and the halo skirt separates "
          "from the core in every panel.",
          min_s=7.0),
    Scene("073_halo_pair",
          "The halo popup tracks H, the kurtosis measure: zero is a uniform "
          "core, the dashed line at two a Gaussian. Matched, left: H x "
          "stays near 1.4, never above 1.6, emittance growth two percent. "
          "Mismatched, right: H x climbs past 2.2, peaking at 2.6, H y past "
          "three, and the R M S emittance grows thirty eight percent. Same "
          "lattice, same current, twenty thousand particles each. "
          "Transmission claims need particles, not sigmas.",
          min_s=7.0),
    Scene("075_kernels",
          "Cloud in cell versus T S C. Cloud in cell spreads each particle "
          "to eight grid corners, T S C to twenty seven cells at three "
          "times the cost. With R F cavities the restoring force damps grid "
          "noise, so cloud in cell is the default. On long R F free "
          "transport its self force aliasing accumulates: on the PIP two "
          "transfer line the manual measured longitudinal emittance times "
          "fourteen with cloud in cell, times one point six with T S C, "
          "against TraceWin's one point seven. On the halo benchmark we "
          "just ran the two kernels agree within four percent, so that "
          "growth is not the kernel. Green's function: I G F always; point "
          "only reproduces legacy runs.",
          min_s=8.0),
    Scene("080_dc",
          "Before the R F Q there are no bunches. On the Beam tab, tick "
          "Continuous beam: the longitudinal Twiss greys out, D C delta W "
          "enables, and Periodic phase un greys. Periodic phase is for a D "
          "C beam an R F Q then bunches: the simulation seeds one R F "
          "period, space charge pushes particles across the bucket "
          "boundary, and folding the phase into one bunch spacing, the "
          "Toutatis convention, keeps the solver looking at one compact "
          "bunch instead of a three bucket clump. Not invertible, so "
          "backtracking refuses it, and it cannot combine with C S R.",
          min_s=7.0),
    Scene("082_dc_numerics",
          "Two continuous beam choices live in Numerics. The D C S C "
          "kernel: uniform is the analytic linear kick that matches the "
          "envelope, gaussian the Bassetti Erskine field, pic two d a two "
          "dimensional P I C over the actual particles, TraceWin's PICNIC "
          "two D analogue. And the envelope solver can switch from matrix "
          "to sacherer, the coupled K V envelope equation, for accelerator "
          "free low energy lines.",
          min_s=7.0),
    Scene("085_scc",
          "A continuous beam ionises the residual gas and traps the "
          "compensating charge, partly neutralising its own space charge. "
          "The shipped S C C demo: thirty k e V H minus, two solenoids, "
          "fifteen milliamps, D C envelope with substeps recorded. Gas set "
          "to nitrogen, eight times ten to the minus six millibar on the "
          "log slider, three hundred kelvin. Computed mode solves the "
          "Poisson Boltzmann balance; Assumed takes your eta. Compute: "
          "neutralisation along the line, complete in the middle and "
          "tapered at the ends, mean 0.76; the on axis potential, three "
          "hundred volts deep; the build up time constant, eleven point six "
          "microseconds; gas survival, ninety four percent. Set t to ten "
          "microseconds and Compute: still building, mean 0.44. Raise the "
          "pressure tenfold: the time constant drops to one point two "
          "microseconds, but survival falls to fifty five percent. Back at "
          "base pressure, Iterate re runs the envelope with damped "
          "compensation cards until they stop moving: self consistent in "
          "three iterations. Export cards and Apply to lattice write real "
          "space charge comp cards; the cleared region row models a chopper "
          "sweeping the ions out. The cross section calibration is "
          "inherited from a measured machine, not literature, as the manual "
          "discloses.",
          min_s=10.0),
    Scene("088_csr",
          "The relative: coherent synchrotron radiation. In a bend, "
          "radiation from the tail overtakes the head along the chord and "
          "acts back on the bunch. HELIX applies the one dimensional steady "
          "state wake per sub step inside every dipole, multi particle "
          "only, because the wake needs the actual line density. Under "
          "Collective effects, off by default. Tick it.",
          min_s=6.0),
    Scene("089_csr_pair",
          "The shipped six bend chicane, eight hundred M e V H minus, "
          "twenty thousand particles, run twice: C S R off left, on right, "
          "momentum spread along s rising from four point one to five point "
          "seven times ten to the minus five in both. Indistinguishable. "
          "The exit energy spread is seventy one point two k e V without "
          "and seventy one point three with, a two tenths of a percent "
          "effect, the right answer for a heavy ion bunch at this energy. "
          "Steady state only: entrance and exit transients are not "
          "modelled.",
          min_s=7.0),
    Scene("091_parity",
          "Against the industry reference. Every space charge path is "
          "benchmarked against TraceWin partran on the PIP two sections: "
          "per section sigmas within about three percent at five milliamps, "
          "and over the full two hundred fifty six metre linac 8.6 percent "
          "R M S in x and 13.3 in y, within the noise of five thousand "
          "particle runs. A three percent P I C calibration offset is known "
          "and stable: expect it, do not chase it. One trap: HELIX reports "
          "sigma phi at the local cavity frequency, TraceWin at the fixed "
          "bunch frequency, up to four times apart at six hundred fifty "
          "megahertz for the same bunch length. The Compare with TraceWin "
          "partran card overlays your run on a genuine export.",
          min_s=8.0),
    Scene("090_outro",
          "That is space charge: measured on the current scan and the "
          "energy law, read in the tune depression and the footprint, "
          "refused honestly and then charted by Hofmann, proven by an "
          "emittance exchange the chart predicted, checked particles "
          "against envelope, caught in a halo, compensated in a L E B T, "
          "and joined by its relative in a bend. Still to come, the special "
          "one: the voice assistant inside HELIX. See you there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


# ---------------------------------------------------------------------
# Manual tour (QtWebEngine) — runs in its own process
# ---------------------------------------------------------------------
def capture_manual_frames(out_dir: Path) -> None:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
    from PyQt6.QtWebEngineWidgets import QWebEngineView   # before QApplication
    from PyQt6.QtCore import QEventLoop, QUrl
    from PyQt6.QtWidgets import QApplication
    from pipeline.record import Recorder

    app = QApplication(["ep12manual"])
    v = QWebEngineView()
    v.resize(1920, 1080)
    v.show()
    done: dict = {}
    v.loadFinished.connect(lambda ok: done.update(ok=bool(ok)))
    v.load(QUrl.fromLocalFile(
        str(ROOT / "site" / "05_space_charge" / "02_pic_solver.html")))

    def pump(n: int = 4) -> None:
        for _ in range(n):
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

    t0 = time.time()
    while "ok" not in done and time.time() - t0 < 90:
        pump(); time.sleep(0.01)
    assert done.get("ok"), "manual page did not load"
    t0 = time.time()
    while time.time() - t0 < 2.5:
        pump(); time.sleep(0.01)

    def scroll_to(y: int) -> None:
        v.page().runJavaScript(f"window.scrollTo(0,{int(y)})")
        pump(6)

    def grab():
        return v.grab().toImage()

    rec = Recorder(out_dir.parent, pump)
    rec.start(out_dir.name)

    def glide(y0: int, y1: int, step: int) -> None:
        for y in range(y0, y1, step):
            scroll_to(y)
            rec.tick(grab)
        scroll_to(y1)
        rec.tick(grab)

    scroll_to(0)
    rec.hold(grab, 3.0)                    # TL;DR block
    glide(0, 1650, 60); rec.hold(grab, 5.0)      # the 8-step list
    glide(1650, 2150, 45); rec.hold(grab, 7.0)   # Figure 5.2 (8 boxes)
    glide(2150, 2900, 55); rec.hold(grab, 6.0)   # backend table
    from types import SimpleNamespace
    sc = SimpleNamespace(name=out_dir.name, frames_dir="", fps=0.0)
    rec.finish(sc)
    (out_dir / "clip.json").write_text(json.dumps({"fps": sc.fps}))
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------------
def capture_visuals() -> None:
    from PyQt6.QtCore import QEventLoop, QPoint, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import (QApplication, QListWidget, QPushButton,
                                 QScrollArea)
    import numpy as np

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["ep12"])
    cards.title_card(s["010_title"], "Space Charge & Real Beams",
                     "The physics special — the force measured on camera")
    cards.title_card(s["020_what"], "The Force",
                     "Collective repulsion · defocusing everywhere · "
                     "K ∝ I / (βγ)³",
                     kicker="SPACE CHARGE IN ONE MINUTE")
    cards.terminal_card(s["075_kernels"], [
        ("out", "Particle-mesh kernel   cells   cost    role"),
        ("out", "  cic (default)          8     1x     RF-damped linac stages"),
        ("out", "  tsc                   27    ~3.4x   long RF-free transport"),
        ("gap", ""),
        ("out", "BTL (MEBT exit -> BTL end), eps_z growth:"),
        ("out", "  CIC          x14"),
        ("out", "  TSC          x1.6"),
        ("out", "  TraceWin     x1.7   (reference)"),
        ("gap", ""),
        ("out", "Green's function: igf (Qiang 2006) always;"),
        ("out", "  point = sampled 1/(4 pi eps0 r), legacy only"),
        ("gap", ""),
        ("out", "halo_fodo, this build (N=20000, 48^3, 5 mA):"),
        ("out", "  eps_z growth  cic x8.38   tsc x8.05   (SC off: x1.00)"),
    ], title="docs/manual/05_space_charge/03_kernels.md")
    cards.outro_card(s["090_outro"], [
        "Space charge measured, not asserted",
        "Envelope for speed, particles for truth",
        "Next — the voice assistant inside HELIX",
    ])

    # ---- the PIC-cycle manual tour: separate process ------------------
    man_dir = FRAMES / "032_pic_manual_tour"
    r = subprocess.run([sys.executable, __file__, "--manual-frames",
                        str(man_dir)], env=dict(os.environ),
                       capture_output=True, text=True, timeout=600)
    print(r.stdout[-800:])
    if r.returncode not in (0, 139, 138) and not (man_dir / "clip.json").exists():
        print(r.stderr[-2000:])
        raise RuntimeError("manual-tour subprocess failed")
    man_sc = scene_by_name("032_pic_manual_tour")
    man_sc.frames_dir = str(man_dir)
    man_sc.fps = float(json.loads((man_dir / "clip.json").read_text())["fps"])
    print(f"[clip] 032_pic_manual_tour attached at {man_sc.fps:.2f} fps")

    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    win = InterphaseWindow()
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def region(widget, pad: int = 8):
        tl = widget.mapTo(win, QPoint(0, 0))
        x = max(0, tl.x() - pad)
        y = max(0, tl.y() - pad)
        return x, y, widget.width() + 2 * pad, widget.height() + 2 * pad

    def crop_img(widget, pad: int = 8) -> QImage:
        settle()
        img = win.grab().toImage()
        x, y, w, h = region(widget, pad)
        return img.copy(x, y, min(img.width() - x, w),
                        min(img.height() - y, h))

    def span_img(widgets, pad: int = 8) -> QImage:
        settle()
        img = win.grab().toImage()
        rs = [region(w, pad) for w in widgets]
        x0 = min(r[0] for r in rs); y0 = min(r[1] for r in rs)
        x1 = max(r[0] + r[2] for r in rs); y1 = max(r[1] + r[3] for r in rs)
        return img.copy(x0, y0, min(img.width() - x0, x1 - x0),
                        min(img.height() - y0, y1 - y0))

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    def open_popup(key: str, size=(1600, 900)):
        assert win.show_result_plot(key), f"no plot {key}"
        settle(10)
        pop = win.results_tab._popups.get(key)
        pop.resize(*size)
        settle(8)
        return pop

    def wait_results(cap: float = 1800.0) -> float:
        t0 = time.time()
        while win.state.results is None and time.time() - t0 < cap:
            settle(10)
            time.sleep(0.05)
        assert win.state.results is not None, "run did not finish"
        return time.time() - t0

    def run_env() -> float:
        win.state.set_results(None)
        win._run_envelope()
        return wait_results()

    def run_mp() -> float:
        win.state.set_results(None)
        win._run_mp()
        return wait_results()

    def worker_idle(pop):
        return lambda: (getattr(pop, "_worker", None) is None
                        or not pop._worker.isRunning())

    def load(path: Path):
        lattice, _meta = _parse_lattice_file(str(path))
        win.state.set_lattice(lattice, str(path))
        return lattice

    BG = QColor("#0b1220")

    def montage(images, pad: int = 14) -> QImage:
        """Side-by-side on a 1920x1080 canvas (ep07 recipe)."""
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(BG)
        p = QPainter(canvas)
        n = len(images)
        avail_w = 1920 - pad * (n + 1)
        scaled = []
        for img in images:
            sc_img = img.scaledToWidth(
                min(int(avail_w / n), img.width()),
                Qt.TransformationMode.SmoothTransformation)
            if sc_img.height() > 1080 - 2 * pad:
                sc_img = img.scaledToHeight(
                    1080 - 2 * pad, Qt.TransformationMode.SmoothTransformation)
            scaled.append(sc_img)
        total_w = sum(i.width() for i in scaled) + pad * (n - 1)
        x = (1920 - total_w) // 2
        for img in scaled:
            p.drawImage(x, (1080 - img.height()) // 2, img)
            x += img.width() + pad
        p.end()
        return canvas

    rec = Recorder(FRAMES, settle)
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.particle import PROTON, H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching import (find_fodo_cells, find_matched_input_twiss,
                                    find_sc_matched_input_twiss)
    from linac_gen.study.engine import StudyManager
    from linac_gen.study.spec import ParamSpec, StudySpec
    from linac_gen_gui.interphase.panels.study_plots import read_envelope
    fields = {f.name for f in BeamConfig.__dataclass_fields__.values()}
    ct = win.convergence_tab

    # ==================================================================
    # 035 — Numerics: Step density + Space charge & PIC (defaults)
    # ==================================================================
    go_tab("numerics")
    settle(10)
    secs = ct._sections
    for t, sec in secs.items():
        sec.setExpanded(t in ("Step density", "Space charge & PIC"))
    settle(10)
    span_img([secs["Step density"], secs["Space charge & PIC"]]).save(
        s["035_sc_knobs"])
    print(f"[measure] 035 defaults: step2={ct._fixed_step2.value()} "
          f"nx={ct._fixed_nx.value()} ext={ct._fixed_ext.value()} "
          f"backend={ct._fixed_backend.currentText()} "
          f"engine={ct._fixed_sc_backend.currentText()} "
          f"green={ct._fixed_green.currentText()} "
          f"kernel={ct._fixed_kernel.currentText()} "
          f"grid={ct._fixed_grid_mode.currentText()} "
          f"dc={ct._fixed_dc_kernel.currentText()}")

    # ==================================================================
    # 066 — emittance-exchange validation pair (full PIC, project beams)
    # ==================================================================
    # Runs FIRST — before any Results popup exists.  A hidden tune-
    # depression popup (still subscribed to lattice_changed) starts a
    # sigma_0 struct worker on this 1350-element lattice and starved a
    # 13 s multi-particle run into a 15-minute stall in the probe.
    hof_dir = ROOT / "examples/hofmann_stability"
    ct._fixed_nx.setValue(32)
    ct._fixed_ext.setValue(5.0)
    ex_imgs = []
    for name in ("exchange_resonant", "exchange_control"):
        pj = json.loads((hof_dir / f"{name}.lgproj").read_text())
        load(hof_dir / pj["lattice_path"])
        win.state.set_beam_config(
            BeamConfig(**{k: v for k, v in pj["beam"].items() if k in fields}))
        dt = run_mp()
        res = win.state.results
        ex_ = np.asarray(res.emit_x); ez_ = np.asarray(res.emit_z)
        ey_ = np.asarray(res.emit_y)
        print(f"[measure] 066 {name} ({dt:.1f}s, N={win.state.beam_config.n_particles}): "
              f"eps_x {ex_[0]:.4f}->{ex_[-1]:.4f} ({100 * (ex_[-1] / ex_[0] - 1):+.1f}%) "
              f"eps_y ({100 * (ey_[-1] / ey_[0] - 1):+.1f}%) "
              f"eps_z {ez_[0]:.5f}->{ez_[-1]:.5f} ({100 * (ez_[-1] / ez_[0] - 1):+.1f}%) "
              f"eps_x max {ex_.max():.4f} ({100 * (ex_.max() / ex_[0] - 1):+.1f}%) "
              f"eps_z min {ez_.min():.5f} ({100 * (ez_.min() / ez_[0] - 1):+.1f}%) "
              f"at s={float(np.asarray(res.s)[np.argmin(ez_)]):.0f} mm")
        go_tab("results"); settle(8)
        pop = open_popup("emit", (1600, 900))
        ex_imgs.append(pop.grab().toImage())
        pop.close(); settle(2)
    montage(ex_imgs).save(s["066_exchange_proof"])
    ct._fixed_nx.setValue(48)
    ct._fixed_ext.setValue(7.0)

    # ==================================================================
    # the FODO lab rat: zero-current matched Twiss + realistic bunch
    # ==================================================================
    deck = ROOT / "examples/fodo_cell.dat"
    lattice = load(deck)
    ref = ReferenceParticle(species=PROTON, w_kin=1.0, frequency=352.21)
    c0, c1 = find_fodo_cells(lattice)[0]
    tw = find_matched_input_twiss(lattice, ref, c0, c1)

    def fodo_beam(energy: float):
        r = ReferenceParticle(species=PROTON, w_kin=energy, frequency=352.21)
        t = find_matched_input_twiss(lattice, r, c0, c1)
        return dict(species="proton", energy=energy, frequency=352.21,
                    alpha_x=float(t["alpha_x"]), beta_x=float(t["beta_x"]),
                    alpha_y=float(t["alpha_y"]), beta_y=float(t["beta_y"]),
                    emit_z=EZ, alpha_z=0.0, beta_z=BZ), t

    def cfg(cur: float, n: int = 2000, **kw) -> BeamConfig:
        d = dict(species="proton", energy=1.0, frequency=352.21,
                 current=cur, n_particles=n,
                 alpha_x=tw["alpha_x"], beta_x=tw["beta_x"],
                 alpha_y=tw["alpha_y"], beta_y=tw["beta_y"],
                 emit_z=EZ, alpha_z=0.0, beta_z=BZ)
        d.update(kw)
        return BeamConfig(**d)

    win.state.set_beam_config(cfg(15.0))

    # ---- 040: current scan, engine-driven (explicit values), Overlay
    # loads the finished study; select-all + Draw ON CAMERA.
    study_root = Path(tempfile.mkdtemp(prefix="helix_ep12_"))
    beam1, _ = fodo_beam(1.0)
    spec = StudySpec(
        name="current_scan", input=str(deck), mode="envelope",
        parameters=[ParamSpec(selector="current",
                              values=[0.0, 5.0, 10.0, 15.0])],
        beam=beam1)
    StudyManager.create(study_root / "current_scan", spec).run(serial=True)
    go_tab("param study")
    settle(10)
    st = win.study_tab
    an = st._analysis

    def show_overlay_tab():
        for i in range(an.count()):
            if an.tabText(i).startswith("Overlay"):
                an.setCurrentIndex(i)

    def an_grab():
        img = win.grab().toImage()
        x, y, w, h = region(an, 8)
        return img.copy(x, y, min(img.width() - x, w),
                        min(img.height() - y, h))

    def overlay_select_all_and_draw():
        ov = an.currentWidget()
        lst = ov.findChildren(QListWidget)
        if lst:
            lst[0].selectAll()
        for b in ov.findChildren(QPushButton):
            if b.text().strip().lower() == "draw":
                b.click()
                break

    def overlay_maxima(tag: str):
        ov = an.currentWidget()
        lst = ov.findChildren(QListWidget)[0]
        out = []
        for it in lst.selectedItems():
            data = read_envelope(it.data(Qt.ItemDataRole.UserRole), "sigma_x")
            if data is not None:
                out.append((it.text(), float(np.max(data[1]))))
        print(f"[measure] {tag} sigma_x maxima: {out}")
        return out

    st._analysis.load_study(str(study_root / "current_scan"))
    settle(10)
    show_overlay_tab()
    settle(8)
    rec.start("040_experiment")
    rec.hold(an_grab, 1.5)
    ov = an.currentWidget()
    ov.findChildren(QListWidget)[0].selectAll()
    rec.hold(an_grab, 1.5)
    for b in ov.findChildren(QPushButton):
        if b.text().strip().lower() == "draw":
            b.click()
            break
    rec.hold(an_grab, 5.0)
    rec.finish(scene_by_name("040_experiment"))
    mx = dict(overlay_maxima("040"))
    base = [v for k, v in mx.items() if "0000" in k]
    if base:
        for k, v in mx.items():
            print(f"[measure] 040 swell {k}: x{v / base[0]:.4f}")
    settle(6)

    # ---- 045: energy flip-book — three 2-run studies (0 vs 15 mA),
    # each matched at its own energy; load + select-all + Draw on camera.
    e_dirs = {}
    for E in (1.0, 3.0, 10.0):
        bE, tE = fodo_beam(E)
        print(f"[measure] 045 E={E}: bare mu={float(tE['mu_x']):.2f} deg, "
              f"beta_x={float(tE['beta_x']):.3f}")
        specE = StudySpec(
            name=f"energy_{E:g}", input=str(deck), mode="envelope",
            parameters=[ParamSpec(selector="current", values=[0.0, 15.0])],
            beam=bE)
        d = study_root / f"energy_{E:g}"
        StudyManager.create(d, specE).run(serial=True)
        e_dirs[E] = d
    rec.start("045_energy")
    for E in (1.0, 3.0, 10.0):
        st._analysis.load_study(str(e_dirs[E]))
        settle(8)
        show_overlay_tab()
        rec.hold(an_grab, 0.8)
        overlay_select_all_and_draw()
        rec.hold(an_grab, 6.5)
        mxE = overlay_maxima(f"045 E={E}")
        if len(mxE) == 2:
            print(f"[measure] 045 E={E}: swell x{mxE[1][1] / mxE[0][1]:.4f}")
    rec.finish(scene_by_name("045_energy"))
    settle(6)

    # ==================================================================
    # 050 — tune depression on an SC-MATCHED 5 mA beam (no mismatch flag)
    # ==================================================================
    egeo = 0.25 / ref.bg
    tw5 = find_sc_matched_input_twiss(
        lattice, ref, c0, c1, 5.0,
        dict(emit_x=egeo, emit_y=egeo, emit_z=EZ, alpha_z=0.0, beta_z=BZ))
    print(f"[measure] SC-matched 5 mA twiss: "
          f"{ {k: round(float(tw5[k]), 4) for k in ('alpha_x', 'beta_x', 'alpha_y', 'beta_y')} }")
    cfg5 = cfg(5.0, n=10000,
               alpha_x=float(tw5["alpha_x"]), beta_x=float(tw5["beta_x"]),
               alpha_y=float(tw5["alpha_y"]), beta_y=float(tw5["beta_y"]))
    win.state.set_beam_config(cfg5)
    run_env()
    go_tab("results")
    settle(8)
    pop = open_popup("tune_depr", (1600, 900))
    t0 = time.time()
    while time.time() - t0 < 60 and not worker_idle(pop)():
        settle(10); time.sleep(0.05)
    settle(15)
    print("[measure] 050 tune_depr:", pop._combo.currentText(), "|",
          pop._info.text())
    for pl in ("x", "y", "z"):
        m = pop._rows[pl]["model"]; c = pop._rows[pl]["curve"]
        print(f"[measure] 050 eta_{pl} model={None if m.yData is None else np.round(m.yData, 3)} "
              f"beam={None if c.yData is None else np.round(c.yData, 3)}")
    pop.grab().save(s["050_depression"])
    pop.close(); settle(2)

    # ---- 062: the FODO refusal, on camera
    pop = open_popup("hofmann", (1600, 900))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("062_refusal_fodo")
    rec.hold(pop_grab, 3.0)
    pop._btn.click()
    rec.wait_until(pop_grab, worker_idle(pop), cap_s=120, stable_s=0.5)
    settle(15)
    print("[measure] 062 hofmann FODO:", pop._info.text())
    rec.hold(pop_grab, 6.0)
    rec.finish(scene_by_name("062_refusal_fodo"))
    pop.close(); settle(2)

    # ==================================================================
    # the demo linac (project beam) — footprint, coupled refusal, chart
    # ==================================================================
    hof_dir = ROOT / "examples/hofmann_stability"
    proj = json.loads((hof_dir / "hofmann_demo.lgproj").read_text())
    hcfg = BeamConfig(**{k: v for k, v in proj["beam"].items() if k in fields})
    load(hof_dir / "hofmann_demo_linac.dat")
    win.state.set_beam_config(hcfg)
    run_env()
    go_tab("results")
    settle(8)

    # ---- 055: footprint on Section A (period 0)
    pop = open_popup("footprint", (1600, 900))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    print("[measure] 055 footprint period:", pop._combo.currentText(),
          "turns", pop._turns.value(), "particles", pop._nparts.value())
    rec.start("055_footprint")
    rec.hold(pop_grab, 3.0)
    pop._btn.click()
    rec.wait_until(pop_grab, worker_idle(pop), cap_s=600, stable_s=0.5)
    settle(15)
    print("[measure] 055 footprint:", pop._info.text())
    sp = pop._scatter.getData()
    if sp is not None and sp[0] is not None and len(sp[0]):
        print(f"[measure] 055 mu_x {np.nanmin(sp[0]):.2f}..{np.nanmax(sp[0]):.2f} "
              f"mu_y {np.nanmin(sp[1]):.2f}..{np.nanmax(sp[1]):.2f} n={len(sp[0])}")
    rec.hold(pop_grab, 8.0)
    rec.finish(scene_by_name("055_footprint"))
    pop.close(); settle(2)

    # ---- 063: coupled refusal on Section D (solenoids)
    pop = open_popup("hofmann", (1600, 900))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    labels = [pop._combo.itemText(i) for i in range(pop._combo.count())]
    print("[measure] hofmann periods:", labels)
    # Section D = the 6-cell solenoid bracket ("6 × 5-element cell");
    # "6 ×" alone would also match "16 × 7-element cell" (Section A).
    idx_d = next(i for i, lab in enumerate(labels) if "6 × 5-element" in lab)
    pop._combo.setCurrentIndex(idx_d)
    settle(12)
    print("[measure] 063 before compute:", pop._info.text())
    rec.start("063_refusal_coupled")
    rec.hold(pop_grab, 3.5)
    pop._btn.click()
    rec.wait_until(pop_grab, worker_idle(pop), cap_s=120, stable_s=0.5)
    settle(15)
    print("[measure] 063 hofmann D:", pop._info.text())
    rec.hold(pop_grab, 6.0)
    rec.finish(scene_by_name("063_refusal_coupled"))

    # ---- 064: chart computed on Section A, bands + P(unstable) on camera
    pop._combo.setCurrentIndex(0)
    settle(12)
    rec.start("064_hofmann_chart")
    rec.hold(pop_grab, 3.5)
    t0 = time.time()
    pop._btn.click()
    rec.wait_until(pop_grab, worker_idle(pop), cap_s=300, stable_s=0.5)
    settle(15)
    print(f"[measure] 064 chart {time.time() - t0:.1f}s:", pop._info.text())
    rec.hold(pop_grab, 8.0)
    pop._chk_bands.setChecked(True)
    rec.hold(pop_grab, 4.0)
    pop._chk_bands.setChecked(False)
    rec.hold(pop_grab, 1.0)
    pop._chk_prob.setChecked(True)
    rec.hold(pop_grab, 1.0)
    pop._btn.click()
    rec.wait_until(pop_grab, worker_idle(pop), cap_s=300, stable_s=0.5)
    settle(15)
    print("[measure] 064 chart+prob:", pop._info.text())
    rec.hold(pop_grab, 6.0)
    rec.finish(scene_by_name("064_hofmann_chart"))
    pop._chk_prob.setChecked(False)
    pop.close(); settle(2)

    # ==================================================================
    # 070/071 — MP vs envelope at 5 mA on the SC-matched FODO
    # ==================================================================
    lattice = load(deck)
    win.state.set_beam_config(cfg5)
    dt = run_mp()
    mp = win.state.results
    print(f"[measure] 070 MP N=10000 in {dt:.1f}s")
    go_tab("results"); settle(8)
    pop = open_popup("rms", (1600, 900))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    print("[measure] 070 rms controls: ap", pop._chk_ap.isChecked(),
          "lat", pop._chk_lat.isChecked(), "display",
          pop._mode_combo.currentText())
    rec.start("070_check")
    rec.hold(pop_grab, 5.0)
    pop._chk_ap.setChecked(False)
    rec.hold(pop_grab, 6.0)
    rec.finish(scene_by_name("070_check"))
    img_mp = pop.grab().toImage()
    pop.close(); settle(2)
    run_env()
    env = win.state.results
    s_mp = np.asarray(mp.s); s_env = np.asarray(env.s)
    for pl in ("sigma_x", "sigma_y"):
        a = np.asarray(getattr(mp, pl))
        b = np.interp(s_mp, s_env, np.asarray(getattr(env, pl)))
        rel = np.abs(a - b) / b
        print(f"[measure] 071 {pl}: max rel {100 * rel.max():.2f}% "
              f"mean {100 * rel.mean():.2f}% end mp {a[-1]:.4f} env {b[-1]:.4f}")
    go_tab("results"); settle(8)
    pop = open_popup("rms", (1600, 900))
    pop._chk_ap.setChecked(False)
    settle(8)
    img_env = pop.grab().toImage()
    pop.close(); settle(2)
    montage([img_mp, img_env]).save(s["071_check_pair"])

    # ==================================================================
    # 072/073 — halo benchmark (matched vs breathing-mode mismatched)
    # ==================================================================
    halo = ROOT / "examples/halo_fodo.dat"
    lat_h = load(halo)
    W, F, I_H, EN = 2.1226695, 162.5, 5.0, 0.21
    rh = ReferenceParticle(species=H_MINUS, w_kin=W, frequency=F)
    from linac_gen.analysis.period_detect import detect_periods
    per = next(p for p in detect_periods(lat_h) if p.source == "lattice_card")
    h0, h1 = per.spans()[0]
    mh = find_sc_matched_input_twiss(
        lat_h, rh, h0, h1, I_H,
        dict(emit_x=EN / rh.bg, emit_y=EN / rh.bg, emit_z=0.06231832,
             alpha_z=0.0, beta_z=819.05492))
    print(f"[measure] halo SC-matched twiss: "
          f"{ {k: round(float(mh[k]), 4) for k in ('alpha_x', 'beta_x', 'alpha_y', 'beta_y')} }")

    def halo_cfg(mm: float) -> BeamConfig:
        return BeamConfig(
            species="H-", energy=W, frequency=F, current=I_H,
            n_particles=20000, emit_nx=EN, emit_ny=EN,
            alpha_x=float(mh["alpha_x"]), beta_x=float(mh["beta_x"]) * mm ** 2,
            alpha_y=float(mh["alpha_y"]), beta_y=float(mh["beta_y"]) * mm ** 2,
            emit_z=0.06231832, alpha_z=0.0, beta_z=819.05492)

    halo_imgs = {}
    for tag, mm in (("matched", 1.0), ("mismatched", 1.4)):
        win.state.set_beam_config(halo_cfg(mm))
        dt = run_mp()
        res = win.state.results
        hx = np.asarray(res.halo_x); hy = np.asarray(res.halo_y)
        ex_ = np.asarray(res.emit_x)
        print(f"[measure] 073 {tag} ({dt:.1f}s): H_x {hx[0]:.3f}->{hx[-1]:.3f} "
              f"max {hx.max():.3f}; H_y {hy[0]:.3f}->{hy[-1]:.3f} max {hy.max():.3f}; "
              f"eps_x {ex_[0]:.4f}->{ex_[-1]:.4f} ({100 * (ex_[-1] / ex_[0] - 1):+.1f}%); "
              f"sigma_x {np.min(res.sigma_x):.3f}..{np.max(res.sigma_x):.3f}")
        go_tab("results"); settle(8)
        if tag == "mismatched":
            pop = open_popup("phase", (1600, 900))
            pop_grab = lambda: pop.grab().toImage()      # noqa: E731
            rec.start("072_halo")
            rec.hold(pop_grab, 4.0)
            i_r = next(i for i in range(pop._colour.count())
                       if pop._colour.itemText(i).startswith("|r|"))
            pop._colour.setCurrentIndex(i_r)
            rec.hold(pop_grab, 7.0)
            rec.finish(scene_by_name("072_halo"))
            pop.close(); settle(2)
        pop = open_popup("halo", (1600, 760))
        halo_imgs[tag] = pop.grab().toImage()
        pop.close(); settle(2)
    montage([halo_imgs["matched"], halo_imgs["mismatched"]]).save(
        s["073_halo_pair"])

    # ==================================================================
    # 080/082 — DC workflow: Beam-tab toggle live, Numerics DC rows
    # ==================================================================
    scc_deck = ROOT / "examples/lebt_scc_demo.dat"
    load(scc_deck)
    scc_cfg = dict(
        species="H-", energy=0.03, frequency=162.5, current=15.0,
        n_particles=1000, distribution="gaussian", cutoff=4.0,
        emit_nx=0.023977, alpha_x=0.0, beta_x=0.6,
        emit_ny=0.023977, alpha_y=0.0, beta_y=0.6,
        emit_z=0.0, continuous=False)
    bt = win.beam_tab
    bt.set_beam_config(BeamConfig(**scc_cfg))
    go_tab("beam")
    settle(10)
    grp_beam = bt._continuous.parentWidget()
    grp_twiss = bt._emit_z.parentWidget()

    def beam_grab():
        return span_img([grp_beam, grp_twiss])

    rec.start("080_dc")
    rec.hold(beam_grab, 3.0)
    bt._continuous.setChecked(True)
    rec.hold(beam_grab, 6.0)
    rec.finish(scene_by_name("080_dc"))
    print(f"[measure] 080 after tick: emit_z enabled={bt._emit_z.isEnabled()} "
          f"dc_dw enabled={bt._dc_dw.isEnabled()} "
          f"periodic enabled={bt._periodic_phase.isEnabled()}")
    bt._apply(quiet=True)          # push the DC config into AppState

    go_tab("numerics")
    settle(8)
    for t, sec in secs.items():
        sec.setExpanded(t in ("Space charge & PIC", "Envelope solver",
                              "Collective effects"))
    settle(10)
    span_img([secs["Space charge & PIC"], secs["Collective effects"]]).save(
        s["082_dc_numerics"])
    print(f"[measure] 082 env solver items: "
          f"{[ct._fixed_env_solver.itemText(i) for i in range(ct._fixed_env_solver.count())]} "
          f"dc items: {[ct._fixed_dc_kernel.itemText(i) for i in range(ct._fixed_dc_kernel.count())]}")

    # ==================================================================
    # 085 — SCC dashboard computing on camera
    # ==================================================================
    ct._record_substeps.setChecked(True)
    run_env()
    ct._record_substeps.setChecked(False)
    print(f"[measure] 085 DC env: sigma_x max {np.max(win.state.results.sigma_x):.2f} mm, "
          f"continuous={getattr(win.state.results, 'continuous', None)}")
    go_tab("results"); settle(8)
    pop = open_popup("scc", (1500, 940))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    ngas = pop._gas.findText("N2")
    rec.start("085_scc")
    rec.hold(pop_grab, 4.0)
    pop._gas.setCurrentIndex(ngas)
    rec.hold(pop_grab, 1.5)
    print("[measure] 085 pressure label:", pop._p_label.text(),
          "T", pop._temp.value(), "mode", pop._mode.currentText(),
          "ttrap", pop._ttrap.value(), "taper", pop._chk_taper.isChecked())

    def scc_compute(tag: str):
        pop._btn.click()
        rec.wait_until(pop_grab, worker_idle(pop), cap_s=300, stable_s=0.4)
        settle(12)
        a = pop._payload
        print(f"[measure] 085 {tag}: {pop._info.text()}")
        if a and a.get("reason") is None:
            print(f"[measure] 085 {tag} mean_fc {a['mean_fc']:.3f} "
                  f"tau {a['tau_scc_global_us']:.1f} us "
                  f"T {a['transmission_gas_pct']:.1f}% phi_min {a['phi_min_V']:.0f} V "
                  f"fc mid {np.asarray(a['fc'])[len(a['fc']) // 2]:.3f}")

    scc_compute("steady 8e-6")
    rec.hold(pop_grab, 9.0)
    pop._tbuild.setValue(10.0)
    rec.hold(pop_grab, 1.0)
    scc_compute("t=10us 8e-6")
    rec.hold(pop_grab, 5.0)
    pop._p_slider.setValue(pop._pressure_to_slider(8.0e-5))
    rec.hold(pop_grab, 1.0)
    scc_compute("t=10us 8e-5")
    rec.hold(pop_grab, 5.0)
    pop._p_slider.setValue(pop._pressure_to_slider(8.0e-6))
    pop._tbuild.setValue(0.0)
    rec.hold(pop_grab, 1.0)
    pop._btn_iter.click()
    rec.wait_until(pop_grab, worker_idle(pop), cap_s=600, stable_s=0.4)
    settle(12)
    print("[measure] 085 iterate:", pop._info.text())
    print("[measure] 085 export enabled", pop._btn_export.isEnabled(),
          "apply", pop._btn_apply.isEnabled())
    rec.hold(pop_grab, 9.0)
    rec.finish(scene_by_name("085_scc"))
    pop.close(); settle(2)

    # ==================================================================
    # 088/089 — CSR chicane: checkbox on camera, measured pair
    # ==================================================================
    load(ROOT / "examples/csr_chicane.dat")
    pj = json.loads((ROOT / "examples/csr_chicane.lgproj").read_text())
    d = {k: v for k, v in pj["beam"].items() if k in fields}
    d["n_particles"] = 20000
    win.state.set_beam_config(BeamConfig(**d))
    ct._fixed_nx.setValue(32)
    ct._fixed_csr.setChecked(False)
    dt = run_mp()
    res_off = win.state.results
    go_tab("results"); settle(8)
    pop = open_popup("dpp", (1600, 760))
    img_off = pop.grab().toImage()
    pop.close(); settle(2)
    go_tab("numerics"); settle(8)
    for t, sec in secs.items():
        sec.setExpanded(t in ("Envelope solver", "Collective effects"))
    settle(10)

    def coll_grab():
        return span_img([secs["Envelope solver"], secs["Collective effects"]])

    rec.start("088_csr")
    rec.hold(coll_grab, 3.5)
    ct._fixed_csr.setChecked(True)
    rec.hold(coll_grab, 4.0)
    rec.finish(scene_by_name("088_csr"))
    dt2 = run_mp()
    res_on = win.state.results
    sw0 = np.asarray(res_off.sigma_w); sw1 = np.asarray(res_on.sigma_w)
    print(f"[measure] 089 CSR off ({dt:.1f}s) sigma_w end {sw0[-1] * 1e3:.3f} keV; "
          f"on ({dt2:.1f}s) {sw1[-1] * 1e3:.3f} keV; ratio {sw1[-1] / sw0[-1]:.5f}; "
          f"emit_z ratio {res_on.emit_z[-1] / res_off.emit_z[-1]:.5f}")
    go_tab("results"); settle(8)
    pop = open_popup("dpp", (1600, 760))
    img_on = pop.grab().toImage()
    pop.close(); settle(2)
    montage([img_off, img_on]).save(s["089_csr_pair"])
    ct._fixed_csr.setChecked(False)
    ct._fixed_nx.setValue(48)

    # ==================================================================
    # 091 — TraceWin parity card + the partran tile on the Results tab
    # ==================================================================
    card_png = str(SHOTS / "091_parity_card.png")
    cards.terminal_card(card_png, [
        ("out", "Validation residuals (5 mA, SC on)"),
        ("out", "  MEBT                        sigma_x/y  < 3 %"),
        ("out", "  MEBT + HWR                  sigma_x/y  ~ 3 %"),
        ("out", "  MEBT + HWR + SSR1 + SSR2    sigma_x/y  ~ 3 %"),
        ("out", "  Full PIP-II 256 m           8.6 % / 13.3 % rms"),
        ("gap", ""),
        ("out", "End of machine   HELIX     TW       diff"),
        ("out", "  sigma_x        3.69 mm   3.55 mm  +3.9 %"),
        ("out", "  sigma_y        4.71 mm   4.94 mm  -4.6 %"),
        ("gap", ""),
        ("out", "Known to differ:"),
        ("out", "  PIC kernel calibration  ~3 % (stable)"),
        ("out", "  sigma_phi: HELIX local f, TW 162.5 MHz"),
        ("out", "    -> up to 4x apart at 650 MHz"),
    ], title="docs/manual/12_validation/01_tracewin_parity.md")
    go_tab("results"); settle(8)
    card = next(c for c in win.results_tab._cards
                if getattr(c, "_key", getattr(c, "key", None)) == "partran")
    scroll = win.results_tab.findChildren(QScrollArea)[0]
    scroll.ensureWidgetVisible(card, 50, 90)
    settle(10)
    img = win.grab().toImage()
    x, y, w, h = region(card, 16)
    tile = img.copy(x, max(0, y - 64), min(img.width() - x, w + 24),
                    min(img.height() - max(0, y - 64), h + 80))
    base_img = QImage(card_png)
    p = QPainter(base_img)
    t_img = tile.scaledToWidth(540, Qt.TransformationMode.SmoothTransformation)
    tx, ty = 1920 - 160 - 36 - t_img.width(), 560
    p.fillRect(tx - 8, ty - 8, t_img.width() + 16, t_img.height() + 16,
               QColor("#2b3b55"))
    p.drawImage(tx, ty, t_img)
    p.end()
    base_img.save(s["091_parity"])

    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    # NO win.close(): offscreen modal-confirm hang (see ep07).


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--manual-frames":
        capture_manual_frames(Path(sys.argv[2]))
        return
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
