"""Episode 6 — The Matching tab in depth: every tool, every algorithm.

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep06_matching_tab.py
Output: tutorials/rendered/ep06_matching_tab.mp4 (+ .srt)

Maximum-detail cut.  Real demonstrations throughout:
  * the SET/ADJUST card language scrolled in the rendered manual;
  * REAL periodic / cell-mode / SC-matched Twiss computations in the
    Matching Dialog on the four-period FODO, applied back to the Beam
    tab through the tab's own reconciliation path;
  * the Phase Advance panel at 0 and 15 mA after real envelope runs;
  * real matches through the real Match button: least_squares,
    gradient, differential_evolution, dual_annealing, a live 6-knob
    CMA-ES with the convergence dialog, a graceful mid-run Stop, an
    MP-cost match, sequential scan (setup dialog + live run), and a
    short Bayesian-optimisation run;
  * real Apply / Save-matched-.dat / stale-result invalidation;
  * the real finite-bounds and inert-constraint refusals (CLI output
    captured at build time — the GUI failure path pops a modal box);
  * the Multi-objective Pareto designer end to end (NSGA-II run,
    knee-point apply);
  * the live-match-preview bridge into a Results-tab popup;
  * the CLI twins (matcher module + `mo --list-objectives`).

Sourced from docs/manual/10_gui/05_matching_tab.md and 07_matching/*.
All spoken numbers were measured with the engine (fixed seeds) and are
re-verified on-frame per the review protocol.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gui"), str(ROOT / "tutorials")]
_qcfg = Path(tempfile.mkdtemp()) / "offscreen.json"
# Screen is declared TALLER than 1080: the Matching tab's full stack
# (KPIs + phase panel + auto-adjust incl. tables and status line) needs
# ~1400 px — at 1080 the constraint table and the status line fall off
# the bottom of the window (verified on the first build's frames).
_qcfg.write_text('{"screens": [{"name": "tut", "x": 0, "y": 0, '
                 '"width": 1920, "height": 1600, "logicalDpi": 96, '
                 '"physicalDpi": 96}]}')
os.environ.setdefault("QT_QPA_PLATFORM", f"offscreen:configfile={_qcfg}")
os.environ.setdefault("HELIX_QSETTINGS_DIR", tempfile.mkdtemp())
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep06_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep06_matching_tab.mp4"

SCENES = [
    Scene("010_title",
          "Deep dive number four, the Matching tab. Matching means "
          "letting the computer turn the knobs, and three tools and "
          "a designer live here. The periodic matching dialog, a "
          "phase advance analyser, the auto adjust matcher with "
          "seven algorithms, and a Pareto designer.",
          min_s=6.0),
    Scene("020_concept",
          "The concept first. Knobs and targets live in the lattice "
          "file itself. An ADJUST card frees a parameter. Family "
          "QUAD, index two, the gradient, bounds minus thirty to "
          "plus thirty tesla per metre. Two cards, both link group "
          "one, so they move in lockstep, a ganged pair. The "
          "target, SET SIZE, weight one, sigma x four millimetres "
          "at the exit, y and longitudinal free.",
          min_s=8.0),
    Scene("025_card_language",
          "The deck is the whole problem statement, and the manual "
          "documents every card. SET TWISS targets exit alpha and "
          "beta. Its six k flags are the only selection mechanism, "
          "all zeros means silently inert, and the z flags follow "
          "HELIX's alpha z sign and degrees per M e V beta z "
          "conventions. SET SIZE has a sign rule, positive fourth "
          "field, sigma phi in degrees, negative, sigma z in "
          "millimetres. MAX and MIN variants bound the worst sigma "
          "over a look back window. SET BEAM PHASE ADV is the one "
          "card whose file position matters, its span runs forward. "
          "The MIN family are one sided, growth costs, shrinkage is "
          "free. Nine SET cards are parsed but inert stubs. One "
          "generic ADJUST card with a parameter index covers eight "
          "element classes, unsupported pairs fail loudly, and the "
          "beam variants free input Twiss, centroid, emittance and "
          "current.",
          min_s=10.0),
    Scene("030_twiss_kpis",
          "The tab's top row echoes the Beam tab's input Twiss as K "
          "P I cards, refreshed live on every edit. Every matching "
          "tool starts from here.",
          min_s=5.0),
    Scene("040_dialog",
          "Open Matching Dialog is the classic periodic matcher. "
          "Four zones, mode, zero current Twiss, space charge "
          "matched Twiss, and a log. Whole lattice mode for "
          "genuinely periodic machines, FODO cell for transfer "
          "lines. And never whole lattice on an accelerating "
          "section, the eigenvalues leave the unit circle.",
          min_s=7.0),
    Scene("045_dialog_periodic_run",
          "The four period FODO, three M e V, zero current. Compute "
          "Periodic Twiss. Alpha x minus one point four three, beta "
          "two point two nine metres, the y plane mirrored, forty "
          "point two degrees of phase advance, ten per period. "
          "Dispersion rows, exact zeros, nothing bends. Episode "
          "one's matched beam came from here. Apply to Beam Setup "
          "is armed.",
          min_s=8.0),
    Scene("046_dialog_cell_mode",
          "Cell mode, the transfer line question. The cell selector "
          "wakes, filled by quad to quad detection, cell zero, "
          "elements three to seven. Compute, and the log records "
          "the matched input, ten point one degrees per cell. The "
          "rule, PIP two class lattices use cell mode, and "
          "dispersive arcs use the space charge path, which matches "
          "dispersion in an eight state formulation.",
          min_s=8.0),
    Scene("047_dialog_sc_matched",
          "Fifteen milliamps now, iteration cap two hundred. Compute "
          "S C matched Twiss iterates the envelope solver from the "
          "zero current seed, damping as it goes, oscillation "
          "suspected, damping reduced. Converged in sixty three "
          "iterations. Beta grows from two point two nine to about "
          "two point five one metres, ten per cent, space charge "
          "defocuses. A verification pass re checks the fixed point.",
          min_s=8.0),
    Scene("048_dialog_apply_back",
          "Apply pushes it into the Beam tab, and the closing "
          "dialog makes the tab reconcile. Status, closed, beam "
          "config updated from dialog, project flagged dirty. The "
          "K P I cards read minus one point five three and two "
          "point five one. Unchanged gets reported honestly too.",
          min_s=7.0),
    Scene("050_autoadjust",
          "The Auto Adjust panel drives those cards. Row one, solver "
          "setup. Space charge in the cost or not. Max iterations, "
          "evaluations for least squares, generations for "
          "populations. Cost solver, envelope, fast and linear, or M "
          "P, full particle in cell, fifty to one hundred times "
          "slower, M P particles greyed until chosen. Allow inert "
          "constraints, an escape hatch. Row two, the C M A E S and "
          "Bayesian knobs, then Match, Stop, Apply, Save. Algorithm "
          "and cap are session settings, not saved in the project.",
          min_s=8.0),
    Scene("052_phase_panel_sigma0",
          "The Phase Advance panel, the stability instrument. Period "
          "candidates are auto detected, here a type sequence repeat "
          "tagged auto. The L E D is green, half trace below one, "
          "stable. Sigma nought, ten point zero five degrees per "
          "plane. Sigma model, the depressed channel tune, identical "
          "at zero current, eta one point zero zero zero. Sigma "
          "beam, ten point zero six. The status line is strict to a "
          "fault, flagging a mismatch of nought point nought per "
          "cent.",
          min_s=8.0),
    Scene("054_phase_panel_depressed",
          "Fifteen milliamps, space charge matched beam. Sigma "
          "model drops to nine point six two, eta nought point "
          "nine five seven, the tune depression. The beam row, "
          "nine point one five, ratio nought point nine one. And "
          "the status explains, mismatch nought point four per "
          "cent, eta spreading nought point seven to nought point "
          "nine eight across cells, trust the model number. "
          "Glance here before any match.",
          min_s=8.0),
    Scene("056_options_choreography",
          "The Algorithm dropdown, seven entries in shipped order. "
          "Least squares, differential evolution, dual annealing, "
          "gradient, C M A E S, sequential scan, Bayesian "
          "optimisation. Sigma nought and popsize wake for C M A E "
          "S. No L S polish serves C M A E S and Bayesian "
          "optimisation, the prior is Bayesian only. Cost solver M P "
          "un greys M P particles.",
          min_s=8.0),
    Scene("060_run",
          "Run one. Least squares, envelope cost, Match. The "
          "variables table fills with the linked pair's shared "
          "column, quad one's gradient, minus five to minus three "
          "point four three, written to both quadrupoles. The "
          "constraints table, SET SIZE, R M S residual zero. And the "
          "status, O K, eleven iterations, cost from eight point "
          "five times ten to the minus two to zero.",
          min_s=8.0),
    Scene("065_after",
          "Two buttons arm after a match. Apply installs the "
          "matched values in the live lattice and beam. Save "
          "matched dot dat exports a TraceWin file. Until Apply, "
          "nothing is touched, experimenting is free.",
          min_s=5.0),
    Scene("070_apply_save_real",
          "Apply. Applied, lattice and beam updated, save to "
          "persist. Both flagged dirty, plain save reroutes to "
          "save as, the title bar keeps your file. Then Save "
          "matched dot dat, and the status names the written "
          "path.",
          min_s=6.0),
    Scene("073_matched_deck",
          "The written deck. ADJUST and SET SIZE round trip "
          "untouched, and both quadrupoles carry the matched "
          "gradient, minus three point four three, where plus and "
          "minus five stood. Machine and problem, one portable "
          "file.",
          min_s=6.0),
    Scene("075_invalidation",
          "One honest rule. A match is valid only for the lattice "
          "it was computed on. Edit the lattice, the tables "
          "clear, Apply and Save grey out. Lattice changed since "
          "last match, click Match again.",
          min_s=6.0),
    Scene("080_local",
          "Fresh deck, algorithm gradient, exact Jacobians from "
          "differentiable tracking. Same answer, minus three "
          "point four three, in six iterations against eleven. "
          "Its scope is narrow and fail loud, linear lattices "
          "only, no R F, field maps, centroid or longitudinal "
          "targets, it points you back to least squares. Both are "
          "local, they find the nearest valley.",
          min_s=7.0),
    Scene("090_global",
          "Global now. Differential evolution breeds a population "
          "and mixes the best. The price, seven hundred and sixty "
          "eight evaluations against eleven, same answer. Global "
          "searches explore a box, so every ADJUST needs finite "
          "bounds.",
          min_s=6.0),
    Scene("092_dual_annealing",
          "Dual annealing, simulated annealing with restarts, "
          "sometimes accepts a worse point to escape a trap. "
          "Fifty two evaluations here. Use these two when the "
          "nearest valley may not be the best.",
          min_s=6.0),
    Scene("094_bounds_error",
          "Enforced for real. This deck's ADJUST leaves min and "
          "max at zero, the unbounded convention. Differential "
          "evolution stops with a value error naming quad one's "
          "gradient, and the fix, add bounds or use least "
          "squares. The same check guards dual annealing, C M A E "
          "S, sequential scan and Bayesian optimisation.",
          min_s=7.0),
    Scene("100_cmaes",
          "C M A E S, the recommended pick for five to thirty "
          "coupled knobs, learns which directions improve the "
          "beam and walks diagonally. Sigma nought, step size as "
          "a fraction of the bound box, default nought point two. "
          "Popsize zero renders auto, four plus three log N. A "
          "least squares polish finishes it unless unticked, and "
          "Max iter counts generations.",
          min_s=7.0),
    Scene("105_cmaes_live",
          "A real six knob match on the shipped demo, five "
          "milliamps, space charge on. Every match opens this "
          "window. Cost per evaluation on a log axis, the noisy "
          "cloud is the population, the dashed line the best so far, "
          "the one you keep. The title counts generations and "
          "evaluations, ticking on through the least squares polish, "
          "the flat tail. Two hundred and twenty two evaluations, "
          "best just above two times ten to the minus three, an "
          "order of magnitude down. Below, the physics, emittances "
          "per plane, and the live row of the evaluation being "
          "scored. Closing this window does not stop the match.",
          min_s=9.0),
    Scene("107_stop_graceful",
          "The Stop button does. Same deck, M P cost, three "
          "hundred macroparticles, big budget. Stop, and the "
          "cancel lands at the next evaluation boundary. The "
          "status comes back, cancelled, best of run, with the "
          "iteration count and cost from baseline to best. "
          "Tables fill, Apply and Save arm. An interrupted match "
          "is never wasted, on any algorithm.",
          min_s=8.0),
    Scene("110_seq",
          "Sequential scan tunes like an operator, in beam order, "
          "bracketing each parameter, reversing on emittance "
          "growth. It asks questions first, Match opens a setup "
          "dialog.",
          min_s=5.0),
    Scene("112_seqscan_dialog",
          "The scan demo, six knobs. Every ADJUSTed element with "
          "category, attributes, bounds. Toolbar filters, all, none, "
          "solenoids only, cavities only, and solenoids only leaves "
          "the two focusing knobs. Passes, steps per parameter, step "
          "size as a bound fraction. Reversal, flip when both "
          "transverse and longitudinal emittance grew, or on any "
          "growth, measured against the input beam or the unmatched "
          "seed's exit. The hard loss rule, tick it, the threshold "
          "wakes, any step losing beam is rolled back, and it needs "
          "the M P solver, the envelope tracks no losses. The deck's "
          "active constraints are listed too.",
          min_s=9.0),
    Scene("114_seqscan_run",
          "Start, and the scan row names the element and "
          "attribute under the wrench, pass by pass, step by "
          "step, direction in brackets, emittances reacting per "
          "bracket. You learn which knob did what. Best settings "
          "kept.",
          min_s=6.0),
    Scene("116_mp_cost_run",
          "Honest physics. Cost solver M P, three hundred "
          "macroparticles, least squares, the Numerics tab's "
          "settings, no hidden defaults. Feel the cadence, a few "
          "dozen evaluations in twenty seconds, where envelope "
          "just did seven hundred and sixty eight in two and a "
          "half. We stop at best of run. Envelope for the shape, "
          "M P for the final polish, particles against wall "
          "time.",
          min_s=7.0),
    Scene("118_allow_inert",
          "This deck carries a transmission floor, but envelope "
          "tracks no particle loss, so the matcher refuses to "
          "pretend. Match would silently ignore active "
          "constraints, MIN TRANSMISSION, inert. Three ways out, "
          "fix the deck, switch cost solver, or allow inert "
          "constraints.",
          min_s=7.0),
    Scene("119_inert_escape",
          "Allow inert constraints, ticked, the match proceeds. "
          "In the constraints table, real residuals on the "
          "emittance rows, the energy floor at zero, and the "
          "transmission row exactly zero, the tell, it "
          "contributed nothing by declaration. The better fix is "
          "still M P.",
          min_s=6.0),
    Scene("120_bayes",
          "Bayesian optimisation builds a Gaussian process of the "
          "cost and spends each evaluation where expected "
          "improvement is highest, so it needs few, made for M P "
          "cost matches where evaluations cost minutes. No L S "
          "polish it shares with C M A E S. B O physics prior is its "
          "own, and only acts with cost solver M P, scouting on the "
          "cheap envelope first. For cheap matches, least squares "
          "wins on wall clock.",
          min_s=7.0),
    Scene("122_bayes_run",
          "Proof, same six knobs, envelope cost. Sixty five "
          "evaluations against two hundred and twenty two for C M "
          "A E S, essentially the same cost. After the space "
          "filling samples every point is model chosen, and a "
          "least squares polish finishes unless unticked.",
          min_s=6.0),
    Scene("125_multiobjective",
          "The multi objective button opens the Pareto designer, "
          "competing goals over the same knobs. Ten objectives, all "
          "minimised, emittance growths, beam loss, negative exit "
          "energy, exit and peak sizes. Tick two or more, here the "
          "default pair, longitudinal growth against exit energy. N "
          "S G A two, the genetic default, q N E H V I the Bayesian "
          "one. Population sixteen, eight generations, run. One "
          "hundred and twenty eight evaluations, ten Pareto designs. "
          "Grey dominated, orange front, the star is the knee. And "
          "honestly, the growth axis spans ten to the minus ten, "
          "energy comes nearly free here. Each row is a full design, "
          "objectives plus knobs. Take a row or the knee, Apply "
          "writes it to the lattice, project dirty.",
          min_s=9.0),
    Scene("127_cli_bridge",
          "Everything has a headless twin. The matcher module "
          "writes the matched deck next to the input, report "
          "prints the same tables. The mo subcommand lists the "
          "same ten objectives. And only the command line runs C "
          "M A E S in parallel, the G U I is sequential.",
          min_s=6.0),
    Scene("129_live_preview",
          "Any results popup gains a live match preview tick box. "
          "On, the curve becomes the matcher's current iterate, "
          "refreshed about once a second, throttled, read only. The "
          "fastest way to catch a cost win that is physically ugly "
          "along the line. At the end it snaps back to the committed "
          "results.",
          min_s=6.0),
    Scene("130_rule",
          "The rule of thumb. Close to the answer, least squares. "
          "Five to thirty coupled knobs, C M A E S. Treacherous "
          "landscape, differential evolution or dual annealing. "
          "Expensive M P cost, Bayesian optimisation. Operator "
          "style, sequential scan with the loss rule. Competing "
          "goals, the Pareto designer. And glance at the phase "
          "advance panel before you tune. Next, the Results tab, "
          "tile by tile.",
          min_s=7.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def _wrap(text: str, width: int = 70) -> list:
    """Wrap one long line into ('out', …) card lines, verbatim."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(("out", cur))
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(("out", cur))
    return lines


def _hardwrap(line: str, width: int = 74) -> list:
    """Wrap a pre-formatted line WITHOUT collapsing its inner spacing
    (keeps column alignment of the untouched head of each line)."""
    out = []
    while len(line) > width:
        cut = line.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        out.append(("out", line[:cut]))
        line = "        " + line[cut:].strip()
    out.append(("out", line))
    return out


def _run_cli(args: list, cwd: str | None = None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, "-m", *args],
                          cwd=cwd or str(ROOT), env=env,
                          capture_output=True, text=True, timeout=900)


def capture_visuals() -> None:
    # WebEngine must be imported BEFORE the QApplication is created.
    import PyQt6.QtWebEngineWidgets  # noqa: F401
    from PyQt6.QtCore import QEventLoop, QPoint, QRect, Qt, QUrl
    from PyQt6.QtGui import QPainter
    from PyQt6.QtWidgets import QApplication, QFrame, QPushButton

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["ep06"])

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    # ------------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------------
    cards.title_card(s["010_title"], "The Matching Tab",
                     "Seven algorithms, three tools, one tab")
    cards.outro_card(s["130_rule"], [
        "Next — the Results tab, tile by tile",
        "Also in the series — Param Study, Error Study, Surrogates",
        "Docs: the matching chapter pairs every recipe with its CLI",
    ])

    # 020 — the real demo deck, verbatim
    deck_path = ROOT / "examples/matching_demo.dat"
    deck_lines = deck_path.read_text().strip().splitlines()
    cards.terminal_card(
        s["020_concept"],
        [("cmd", "cat examples/matching_demo.dat")]
        + [("out", ln) for ln in deck_lines],
        title="matching_demo.dat — knobs and targets live in the deck")

    # ------------------------------------------------------------------
    # 025 — the SET/ADJUST card language in the rendered manual
    # ------------------------------------------------------------------
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    view = QWebEngineView()
    view.resize(1920, 1080)
    view.show()
    loaded = {"ok": False}
    view.loadFinished.connect(lambda ok: loaded.update(ok=True))
    view.load(QUrl.fromLocalFile(
        str(ROOT / "site/07_matching/02_set_adjust.html")))
    t0 = time.time()
    while not loaded["ok"] and time.time() - t0 < 60:
        settle(5)
        time.sleep(0.02)
    t0 = time.time()
    while time.time() - t0 < 2.5:
        settle(5)
        time.sleep(0.02)

    rec = Recorder(WORK / "frames", settle)

    def web_grab():
        return view.grab().toImage()

    def web_js(code: str) -> None:
        view.page().runJavaScript(code)

    def web_pause(sec: float) -> None:
        t1 = time.time()
        while time.time() - t1 < sec:
            settle(3)
            time.sleep(0.02)

    rec.start("025_card_language")
    rec.hold(web_grab, 2.0)                       # page top / SET_TWISS head
    stations = [
        ("set_twiss-family-x-x-y-y-z-z-kax-kbx-kay-kby-kaz-kbz", 3.0, 380),
        ("set_size-k-x_mm-y_mm-phi_or_z-k2", 3.0, 0),
        ("set_size_max-k-n_elems-x_mm-y_mm-phi_or_z-k2", 2.2, 0),
        ("set_beam_phase_adv-k-n_elems-mu_x_deg-mu_y_deg-mu_z_deg", 2.2, 0),
        ("min_emit_growth-plane-weight", 2.6, 0),
        ("parsed-but-inert-set_-cards", 2.6, 0),
        ("adjust-target-param_idx-link_group-vmin-vmax-start_step-kn",
         3.0, 620),
        ("adjust_beam_twiss-diag_n-alpx_flag-betx_flag-alpy_flag-"
         "bety_flag-alpz_flag-betz_flag", 2.6, 0),
    ]
    for anchor, hold_s, nudge in stations:
        web_js(f"document.getElementById('{anchor}').scrollIntoView();"
               f"window.scrollBy(0,-72);")
        web_pause(0.4)
        rec.hold(web_grab, hold_s)
        if nudge:
            web_js(f"window.scrollBy(0,{nudge});")
            web_pause(0.3)
            rec.hold(web_grab, 1.6)
    rec.finish(scene_by_name("025_card_language"))
    view.hide()
    view.deleteLater()
    settle(6)

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------
    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    win = InterphaseWindow()
    # Taller than the video frame on purpose: the Matching tab's whole
    # stack must be on screen (crops are letterboxed at render time).
    win.resize(1920, 1400)
    win.show()
    settle(10)

    def crop_widget(key: str, widget, pad: int = 8) -> None:
        settle()
        img = win.grab().toImage()
        tl = widget.mapTo(win, QPoint(0, 0))
        x = max(0, tl.x() - pad)
        y = max(0, tl.y() - pad)
        w = min(img.width() - x, widget.width() + 2 * pad)
        h = min(img.height() - y, widget.height() + 2 * pad)
        img.copy(x, y, w, h).save(key if key.endswith(".png") else s[key])

    def union_crop(key: str, widgets, pad: int = 10) -> None:
        settle()
        img = win.grab().toImage()
        rect = None
        for wdg in widgets:
            tl = wdg.mapTo(win, QPoint(0, 0))
            r = QRect(tl.x(), tl.y(), wdg.width(), wdg.height())
            rect = r if rect is None else rect.united(r)
        x = max(0, rect.x() - pad)
        y = max(0, rect.y() - pad)
        w = min(img.width() - x, rect.width() + 2 * pad)
        h = min(img.height() - y, rect.height() + 2 * pad)
        img.copy(x, y, w, h).save(s[key])

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    from linac_gen.core.config import BeamConfig

    def set_beam(cfg) -> None:
        win.beam_tab.set_beam_config(cfg)
        win.state.set_beam_config(cfg)
        settle(4)

    def load_deck(path: str, cfg) -> None:
        lat, _meta = _parse_lattice_file(path)
        win.state.set_lattice(lat, path)
        set_beam(cfg)
        go_tab("matching")
        settle(8)

    def wait_results(cap_s: float = 300) -> None:
        t1 = time.time()
        while win.state.results is None and time.time() - t1 < cap_s:
            settle(10)
            time.sleep(0.05)
        assert win.state.results is not None, "run did not finish"

    mt = win.matching_tab

    # ==================================================================
    # Stage A — matching_demo.dat: the auto-adjust story
    # ==================================================================
    demo_deck = str(ROOT / "examples/matching_demo.dat")
    cfg_demo = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                          current=0.0)
    load_deck(demo_deck, cfg_demo)

    # 030 — the six CURRENT TWISS KPI cards, cropped precisely
    union_crop("030_twiss_kpis", [mt._kax, mt._kbx, mt._kex,
                                  mt._kay, mt._kby, mt._key])

    # Auto-adjust panel (QFrame that hosts the Match button)
    panel = mt._aa_run.parentWidget()
    while panel is not None and not isinstance(panel, QFrame):
        panel = panel.parentWidget()

    def panel_grab():
        img = win.grab().toImage()
        tl = panel.mapTo(win, QPoint(0, 0))
        x = max(0, tl.x() - 8)
        y = max(0, tl.y() - 8)
        w = min(img.width() - x, panel.width() + 16)
        h = min(img.height() - y, panel.height() + 16)
        return img.copy(x, y, w, h)

    # 050 — the panel at rest (least_squares / envelope defaults)
    crop_widget("050_autoadjust", panel)

    # 056 — options choreography with the dropdown OPEN (zoom inset:
    # the popup is a top-level window that win.grab() never contains,
    # so grab it directly and composite it, scaled up, beside the
    # combo — a zoom of a real render).
    mt._aa_algorithm.showPopup()
    settle(8)
    pop_img = mt._aa_algorithm.view().window().grab().toImage()
    popup_scaled = pop_img.scaledToWidth(
        int(pop_img.width() * 2.4),
        Qt.TransformationMode.SmoothTransformation)
    combo_tl = mt._aa_algorithm.mapTo(win, QPoint(0, 0))
    panel_tl = panel.mapTo(win, QPoint(0, 0))
    inset_x = combo_tl.x() - panel_tl.x() - popup_scaled.width() - 30
    inset_y = combo_tl.y() - panel_tl.y() + 8

    def panel_grab_with_popup():
        base = panel_grab()
        p = QPainter(base)
        p.drawImage(max(0, inset_x + 8), max(0, inset_y + 8), popup_scaled)
        p.end()
        return base

    rec.start("056_options_choreography")
    rec.hold(panel_grab_with_popup, 6.0)
    mt._aa_algorithm.hidePopup()
    settle(4)
    rec.hold(panel_grab, 1.0)
    for algo, hold_s in (("gradient", 1.2), ("cmaes", 2.2),
                         ("bayesopt", 2.2)):
        mt._aa_algorithm.setCurrentText(algo)
        settle(4)
        rec.hold(panel_grab, hold_s)
    mt._aa_cost_solver.setCurrentText("mp")
    settle(4)
    rec.hold(panel_grab, 2.5)
    mt._aa_cost_solver.setCurrentText("envelope")
    mt._aa_algorithm.setCurrentText("least_squares")
    settle(4)
    rec.hold(panel_grab, 1.2)
    rec.finish(scene_by_name("056_options_choreography"))
    settle(6)

    # 060 — REAL least-squares match through the real button
    rec.start("060_run")
    rec.hold(panel_grab, 1.5)
    mt._aa_run.click()
    rec.wait_until(panel_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=120, stable_s=0.4)
    rec.hold(panel_grab, 6.0)
    rec.finish(scene_by_name("060_run"))
    settle(10)

    # 065 — the finished panel (Apply/Save armed)
    crop_widget("065_after", panel)

    # 070 — REAL Apply, then REAL Save via a patched file dialog
    from linac_gen_gui.interphase.tabs import matching_tab as mt_mod
    out_dat = WORK / "matched_demo.dat"
    out_dat.parent.mkdir(parents=True, exist_ok=True)
    orig_gsf = mt_mod.QFileDialog.getSaveFileName
    mt_mod.QFileDialog.getSaveFileName = (
        lambda *a, **k: (str(out_dat), "TraceWin .dat (*.dat)"))
    try:
        rec.start("070_apply_save_real")
        rec.hold(panel_grab, 1.5)
        mt._aa_apply.click()
        settle(6)
        rec.hold(panel_grab, 4.0)
        mt._aa_save.click()
        settle(6)
        rec.hold(panel_grab, 4.5)
        rec.finish(scene_by_name("070_apply_save_real"))
    finally:
        mt_mod.QFileDialog.getSaveFileName = orig_gsf
    settle(6)
    assert out_dat.exists(), "Save matched .dat wrote nothing"

    # 073 — the written deck, verbatim
    matched_lines = out_dat.read_text().strip().splitlines()
    cards.terminal_card(
        s["073_matched_deck"],
        [("cmd", "cat matched_demo.dat")]
        + [("out", ln) for ln in matched_lines],
        title="the matched machine, written by Save matched .dat")

    # 075 — stale-result invalidation on a lattice change
    win.state.set_lattice(win.state.lattice, win.state.lattice_path)
    settle(8)
    crop_widget("075_invalidation", panel)

    # 080 — REAL gradient match on a freshly reloaded deck
    load_deck(demo_deck, cfg_demo)
    mt._aa_algorithm.setCurrentText("gradient")
    settle(4)
    rec.start("080_local")
    rec.hold(panel_grab, 1.5)
    mt._aa_run.click()
    rec.wait_until(panel_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=120, stable_s=0.4)
    rec.hold(panel_grab, 5.0)
    rec.finish(scene_by_name("080_local"))
    settle(8)

    # 090 — REAL differential-evolution match (50 generations)
    load_deck(demo_deck, cfg_demo)
    mt._aa_algorithm.setCurrentText("differential_evolution")
    mt._aa_iter.setValue(50)
    settle(4)
    rec.start("090_global")
    rec.hold(panel_grab, 1.5)
    mt._aa_run.click()
    rec.wait_until(panel_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=180, stable_s=0.4)
    rec.hold(panel_grab, 5.0)
    rec.finish(scene_by_name("090_global"))
    settle(8)

    # 092 — REAL dual-annealing match (20 iterations)
    load_deck(demo_deck, cfg_demo)
    mt._aa_algorithm.setCurrentText("dual_annealing")
    mt._aa_iter.setValue(20)
    settle(4)
    rec.start("092_dual_annealing")
    rec.hold(panel_grab, 1.5)
    mt._aa_run.click()
    rec.wait_until(panel_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=180, stable_s=0.4)
    rec.hold(panel_grab, 4.5)
    rec.finish(scene_by_name("092_dual_annealing"))
    settle(8)
    mt._aa_iter.setValue(200)

    # 094 — the REAL finite-bounds error, captured from a real CLI run
    # (the GUI failure path pops a modal QMessageBox — offscreen trap).
    scratch = Path(tempfile.mkdtemp(prefix="helix_ep06_"))
    unbounded = scratch / "unbounded.dat"
    unbounded.write_text(deck_path.read_text().replace(
        "ADJUST QUAD 2 1 -30 30 0.5 0", "ADJUST QUAD 2 1 0 0 0 0"))
    r = _run_cli(["linac_gen.matching", "unbounded.dat",
                  "--algorithm", "differential_evolution"],
                 cwd=str(scratch))
    err_line = next(ln for ln in r.stderr.splitlines()
                    if ln.startswith("ValueError:"))
    cards.terminal_card(
        s["094_bounds_error"],
        [("cmd", "grep ADJUST unbounded.dat"),
         ("out", "ADJUST QUAD 2 1 0 0 0 0"),
         ("out", "ADJUST QUAD 2 1 0 0 0 0"),
         ("gap", ""),
         ("cmd", "python -m linac_gen.matching unbounded.dat \\"),
         ("out", "        --algorithm differential_evolution"),
         *_wrap(err_line)],
        title="global algorithms refuse open bounds — real output")

    # 100 — CMA-ES options row (σ₀ + popsize 'auto' awake)
    mt._aa_algorithm.setCurrentText("cmaes")
    settle(6)
    img = win.grab().toImage()
    tl = panel.mapTo(win, QPoint(0, 0))
    vt_tl = mt._aa_var_table.mapTo(win, QPoint(0, 0))
    img.copy(max(0, tl.x() - 8), max(0, tl.y() - 8),
             min(img.width() - tl.x() + 8, panel.width() + 16),
             vt_tl.y() - tl.y() + 4).save(s["100_cmaes"])

    # ==================================================================
    # Stage B — bo_demo.dat: CMA-ES live, Stop, MP cost, bayesopt,
    #           live preview, and the multi-objective designer
    # ==================================================================
    bo_deck = str(ROOT / "examples/ml_bayesopt/bo_demo.dat")
    cfg_bo = BeamConfig(species="proton", energy=2.5, frequency=162.5,
                        current=5.0,
                        emit_nx=0.30, emit_ny=0.30, emit_z=0.40,
                        alpha_x=-1.2, beta_x=0.32,
                        alpha_y=2.0, beta_y=0.05)
    load_deck(bo_deck, cfg_bo)

    # one envelope run so the Results popup (129) has committed results
    win.state.set_results(None)
    win._run_envelope()
    wait_results()
    go_tab("matching")
    settle(8)

    # 105 — REAL 6-knob CMA-ES with the convergence dialog on camera
    mt._aa_sc.setChecked(True)
    mt._aa_algorithm.setCurrentText("cmaes")
    mt._aa_iter.setValue(15)
    settle(4)
    mt._aa_run.click()
    settle(1)                       # minimal — catch the run while live
    conv = mt._aa_convergence_dlg
    assert conv is not None

    def conv_grab():
        return conv.grab().toImage()

    rec.start("105_cmaes_live")
    rec.wait_until(conv_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=300, stable_s=0.5)
    rec.hold(conv_grab, 8.0)
    rec.finish(scene_by_name("105_cmaes_live"))
    settle(8)

    # 107 — REAL graceful Stop on an expensive MP-cost CMA-ES run
    mt._aa_cost_solver.setCurrentText("mp")
    mt._aa_mp_n.setValue(300)
    mt._aa_iter.setValue(200)
    settle(4)
    rec.start("107_stop_graceful")
    rec.hold(panel_grab, 1.5)
    mt._aa_run.click()
    rec.hold(panel_grab, 14.0)          # a few slow MP evaluations
    mt._aa_stop.click()
    rec.hold(panel_grab, 2.0)           # "Stopping … waiting" visible
    rec.wait_until(panel_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=180, stable_s=0.4)
    rec.hold(panel_grab, 6.0)
    rec.finish(scene_by_name("107_stop_graceful"))
    settle(8)

    # 116 — REAL MP-cost least-squares crawl (stopped at best-of-run)
    mt._aa_algorithm.setCurrentText("least_squares")
    settle(4)
    rec.start("116_mp_cost_run")
    rec.hold(panel_grab, 1.5)
    mt._aa_run.click()
    rec.hold(panel_grab, 20.0)          # feel the per-eval cadence
    mt._aa_stop.click()
    rec.wait_until(panel_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=180, stable_s=0.4)
    rec.hold(panel_grab, 4.0)
    rec.finish(scene_by_name("116_mp_cost_run"))
    settle(8)

    # 122 — REAL short Bayesian-optimisation run (envelope cost)
    mt._aa_cost_solver.setCurrentText("envelope")
    mt._aa_algorithm.setCurrentText("bayesopt")
    mt._aa_iter.setValue(15)
    settle(4)
    rec.start("122_bayes_run")
    rec.hold(panel_grab, 1.5)
    mt._aa_run.click()
    rec.wait_until(panel_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=300, stable_s=0.4)
    rec.hold(panel_grab, 5.0)
    rec.finish(scene_by_name("122_bayes_run"))
    settle(8)

    # 129 — live match preview streaming into a Results-tab popup
    assert win.show_result_plot("rms"), "no rms popup"
    settle(10)
    pop = win.results_tab._popups.get("rms")
    pop.resize(1500, 950)
    if hasattr(pop, "_chk_ap"):
        pop._chk_ap.setChecked(False)   # mm-scale physics, not aperture
    assert getattr(pop, "_live_cb", None) is not None
    pop._live_cb.setChecked(True)
    settle(8)

    def pop_grab():
        return pop.grab().toImage()

    mt._aa_algorithm.setCurrentText("cmaes")
    mt._aa_iter.setValue(200)
    settle(4)
    rec.start("129_live_preview")
    rec.hold(pop_grab, 1.5)
    mt._aa_run.click()
    rec.wait_until(pop_grab, lambda: mt._aa_apply.isEnabled(),
                   cap_s=300, stable_s=0.4)
    rec.hold(pop_grab, 4.0)
    rec.finish(scene_by_name("129_live_preview"))
    pop._live_cb.setChecked(False)
    pop.close()
    settle(6)
    mt._aa_iter.setValue(200)
    mt._aa_sc.setChecked(False)

    # 125 — the Multi-objective Pareto designer, end to end
    mo_deck = str(ROOT / "examples/ml_multiobjective/mo_demo.dat")
    load_deck(mo_deck, cfg_bo)
    mt._mo_btn.click()
    settle(10)
    mo = mt._mo_dlg
    assert mo is not None
    mo.resize(1600, 900)
    mo._pop.setValue(16)
    mo._gen.setValue(8)
    settle(6)

    def mo_grab():
        return mo.grab().toImage()

    rec.start("125_multiobjective")
    rec.hold(mo_grab, 2.5)
    mo._run_btn.click()
    rec.wait_until(mo_grab, lambda: mo._apply_btn.isEnabled(),
                   cap_s=300, stable_s=0.5)
    rec.hold(mo_grab, 6.0)
    mo._apply_btn.click()
    settle(6)
    rec.hold(mo_grab, 4.0)
    rec.finish(scene_by_name("125_multiobjective"))
    mo.close()
    settle(6)

    # ==================================================================
    # Stage C — sequential scan: the setup dialog + a live run
    # ==================================================================
    seq_deck = str(ROOT / "examples/sequential_scan/seqscan_demo.dat")
    load_deck(seq_deck, cfg_bo)

    from linac_gen_gui.interphase.dialogs import (
        sequential_scan_setup as sss_mod,
    )
    sdlg = sss_mod.SequentialScanSetupDialog(win.state.lattice, parent=mt)
    sdlg.resize(1150, 980)
    sdlg.show()
    settle(10)

    def sdlg_grab():
        return sdlg.grab().toImage()

    def _click_button(dlg, label: str) -> None:
        for b in dlg.findChildren(QPushButton):
            if b.text().strip() == label:
                b.click()
                return
        raise LookupError(label)

    rec.start("112_seqscan_dialog")
    rec.hold(sdlg_grab, 5.0)
    _click_button(sdlg, "Solenoids only")
    settle(4)
    rec.hold(sdlg_grab, 3.0)
    sdlg._reject_loss.setChecked(True)
    settle(4)
    rec.hold(sdlg_grab, 2.0)
    sdlg._loss_thresh.setValue(99.9)
    settle(4)
    rec.hold(sdlg_grab, 3.0)
    _click_button(sdlg, "Select all")
    settle(4)
    rec.hold(sdlg_grab, 1.5)
    rec.finish(scene_by_name("112_seqscan_dialog"))
    sdlg.close()
    settle(4)

    # 114 — REAL sequential scan through the real Match path; the
    # tab's exec() call would block offscreen, so exec is patched to
    # show + trim (1 pass, 7 steps) + auto-accept.
    orig_exec = sss_mod.SequentialScanSetupDialog.exec

    def _auto_exec(self2):
        self2.show()
        settle(6)
        # 3 passes x 51 steps (~918 envelope evals): enough wall time
        # that the scan row visibly advances on camera — at the default
        # 1x7 the whole scan finished before the recorder's first frame.
        self2._passes.setValue(3)
        self2._steps.setValue(51)
        self2._reject_loss.setChecked(False)
        settle(2)
        self2.accept()
        settle(2)
        return self2.DialogCode.Accepted

    sss_mod.SequentialScanSetupDialog.exec = _auto_exec
    try:
        mt._aa_sc.setChecked(True)
        mt._aa_algorithm.setCurrentText("sequential_scan")
        settle(4)
        mt._aa_run.click()
        settle(1)                   # minimal — catch the scan while live
        conv = mt._aa_convergence_dlg
        assert conv is not None

        def conv2_grab():
            return conv.grab().toImage()

        rec.start("114_seqscan_run")
        rec.wait_until(conv2_grab, lambda: mt._aa_apply.isEnabled(),
                       cap_s=300, stable_s=0.5)
        rec.hold(conv2_grab, 7.0)
        rec.finish(scene_by_name("114_seqscan_run"))
    finally:
        sss_mod.SequentialScanSetupDialog.exec = orig_exec
    settle(8)
    mt._aa_sc.setChecked(False)
    mt._aa_algorithm.setCurrentText("least_squares")

    # ==================================================================
    # Stage D — MIN_TRANSMISSION: the refusal and the escape hatch
    # ==================================================================
    mtd_rel = "examples/min_transmission/min_transmission_demo.dat"
    r = _run_cli(["linac_gen.matching", "min_transmission_demo.dat"],
                 cwd=str(ROOT / "examples/min_transmission"))
    err_line = next(ln for ln in r.stderr.splitlines()
                    if ln.startswith("ValueError:"))
    mtd_text = (ROOT / mtd_rel).read_text()
    mtcard_line = next(ln for ln in mtd_text.splitlines()
                       if ln.startswith("MIN_TRANSMISSION"))
    cards.terminal_card(
        s["118_allow_inert"],
        [("cmd", "grep MIN_TRANSMISSION min_transmission_demo.dat"),
         *_hardwrap(mtcard_line),
         ("gap", ""),
         ("cmd", "python -m linac_gen.matching min_transmission_demo.dat"),
         *_wrap(err_line)],
        title="the pre-run constraint audit refuses — real output")

    # 119 — the escape hatch: allow-inert ticked, match proceeds
    load_deck(str(ROOT / mtd_rel), cfg_bo)
    mt._aa_sc.setChecked(True)
    mt._aa_allow_inert.setChecked(True)
    mt._aa_algorithm.setCurrentText("least_squares")
    settle(4)
    mt._aa_run.click()
    t1 = time.time()
    while not mt._aa_apply.isEnabled() and time.time() - t1 < 180:
        settle(10)
        time.sleep(0.02)
    settle(10)
    crop_widget("119_inert_escape", panel)
    mt._aa_allow_inert.setChecked(False)
    mt._aa_sc.setChecked(False)

    # 110 / 120 — algorithm-identity stills (options rows only)
    def options_crop(key: str) -> None:
        settle(6)
        img2 = win.grab().toImage()
        tl2 = panel.mapTo(win, QPoint(0, 0))
        vt2 = mt._aa_var_table.mapTo(win, QPoint(0, 0))
        img2.copy(max(0, tl2.x() - 8), max(0, tl2.y() - 8),
                  min(img2.width() - tl2.x() + 8, panel.width() + 16),
                  vt2.y() - tl2.y() + 4).save(s[key])

    mt._aa_algorithm.setCurrentText("sequential_scan")
    options_crop("110_seq")
    mt._aa_algorithm.setCurrentText("bayesopt")
    options_crop("120_bayes")
    mt._aa_algorithm.setCurrentText("least_squares")
    settle(4)

    # ==================================================================
    # Stage E — the Matching Dialog + Phase Advance panel on the FODO
    # ==================================================================
    fodo_deck = str(ROOT / "examples/fodo_cell.dat")
    cfg_f0 = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                        current=0.0)
    load_deck(fodo_deck, cfg_f0)

    from linac_gen_gui.dialogs import matching_dialog as md_mod
    dlg = md_mod.MatchingDialog(win.state.lattice, mt._beam_tab, parent=mt)
    dlg.resize(980, 780)
    dlg.show()
    settle(10)

    def dlg_grab():
        return dlg.grab().toImage()

    # 040 — the untouched dialog (four zones visible)
    dlg_grab().save(s["040_dialog"])

    # 045 — REAL whole-lattice periodic Twiss
    rec.start("045_dialog_periodic_run")
    rec.hold(dlg_grab, 2.0)
    dlg._compute_periodic()
    settle(6)
    rec.hold(dlg_grab, 7.0)
    rec.finish(scene_by_name("045_dialog_periodic_run"))
    p_twiss = dict(dlg._matched_twiss)

    # 046 — cell mode: combo enables, cells listed, recompute
    rec.start("046_dialog_cell_mode")
    rec.hold(dlg_grab, 1.5)
    dlg._mode_combo.setCurrentIndex(1)
    settle(6)
    rec.hold(dlg_grab, 3.0)
    dlg._compute_periodic()
    settle(6)
    rec.hold(dlg_grab, 6.0)
    rec.finish(scene_by_name("046_dialog_cell_mode"))

    # 047 — REAL SC-matched Twiss at 15 mA (whole-lattice mode,
    # iteration cap raised so the damped loop actually converges)
    dlg._mode_combo.setCurrentIndex(0)
    cfg_f15 = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                         current=15.0)
    set_beam(cfg_f15)
    dlg._max_iter_spin.setValue(200)
    settle(4)
    rec.start("047_dialog_sc_matched")
    rec.hold(dlg_grab, 2.5)
    dlg._compute_sc_matched()
    settle(8)
    rec.hold(dlg_grab, 8.0)
    rec.finish(scene_by_name("047_dialog_sc_matched"))
    sc_twiss = dict(dlg._matched_twiss)
    dlg.close()
    settle(4)

    # 048 — Apply-back through the tab's REAL reconciliation path:
    # _open_matching calls dlg.exec() (modal trap offscreen), so exec
    # is patched to show + apply the real SC-matched Twiss + close.
    orig_mexec = md_mod.MatchingDialog.exec

    def _fake_exec(self2):
        self2.show()
        settle(8)
        self2._matched_twiss = dict(sc_twiss)
        self2._apply_matched()
        settle(4)
        self2.close()
        settle(2)
        return 0

    md_mod.MatchingDialog.exec = _fake_exec
    try:
        mt._open_matching()
    finally:
        md_mod.MatchingDialog.exec = orig_mexec
    settle(10)
    union_crop("048_dialog_apply_back",
               [mt._kax, mt._kbx, mt._kex, mt._kay, mt._kby, mt._key,
                mt._open_btn, mt._status])

    # 052 — Phase Advance panel at 0 mA on the periodic-matched beam
    cfg_m0 = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                        current=0.0,
                        alpha_x=p_twiss["alpha_x"],
                        beta_x=p_twiss["beta_x"],
                        alpha_y=p_twiss["alpha_y"],
                        beta_y=p_twiss["beta_y"])
    set_beam(cfg_m0)
    win.state.set_results(None)
    win._run_envelope()
    wait_results()
    go_tab("matching")
    settle(12)
    pa_panel = mt._pa_combo.parentWidget()
    crop_widget("052_phase_panel_sigma0", pa_panel)

    # 054 — the same panel at 15 mA with the SC-matched beam
    cfg_m15 = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                         current=15.0,
                         alpha_x=sc_twiss["alpha_x"],
                         beta_x=sc_twiss["beta_x"],
                         alpha_y=sc_twiss["alpha_y"],
                         beta_y=sc_twiss["beta_y"])
    set_beam(cfg_m15)
    win.state.set_results(None)
    win._run_envelope()
    wait_results()
    go_tab("matching")
    settle(12)
    crop_widget("054_phase_panel_depressed", pa_panel)

    # ------------------------------------------------------------------
    # 127 — the CLI twins, captured from real runs
    # ------------------------------------------------------------------
    cli_dir = scratch / "cli127"
    cli_dir.mkdir(exist_ok=True)
    (cli_dir / "matching_demo.dat").write_text(deck_path.read_text())
    r1 = _run_cli(["linac_gen.matching", "matching_demo.dat"],
                  cwd=str(cli_dir))
    ok_line = next(ln for ln in r1.stdout.splitlines()
                   if ln.startswith("[match] OK")
                   or ln.startswith("[match] FAILED"))
    wrote_line = next(ln for ln in r1.stdout.splitlines()
                      if ln.startswith("[match] wrote"))
    r2 = _run_cli(["linac_gen", "mo", "--list-objectives"])
    mo_lines = [ln for ln in r2.stdout.splitlines() if ln.strip()]
    cards.terminal_card(
        s["127_cli_bridge"],
        [("cmd", "python -m linac_gen.matching matching_demo.dat"),
         ("out", ok_line),
         ("out", wrote_line),
         ("gap", ""),
         ("cmd", "python -m linac_gen mo --list-objectives"),
         *[part for ln in mo_lines for part in _hardwrap(ln)]],
        title="the headless twins — real output")

    # ------------------------------------------------------------------
    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    # NO win.close(): dirty-state modal confirm hangs offscreen (ep07).


def main() -> None:
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")


if __name__ == "__main__":
    main()
