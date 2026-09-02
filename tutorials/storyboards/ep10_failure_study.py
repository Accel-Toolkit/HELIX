"""Episode 10 — The Failure Study tab in depth (maximum-detail cut).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep10_failure_study.py
Output: tutorials/rendered/ep10_failure_study.mp4 (+ .srt)

Everything on camera is real:
  * live Targets / mode / custom-set interaction clips on the DTL;
  * a REAL single-OFF envelope sweep (18 elements) with the live
    table/bar fill and the ranking-order rebuild;
  * a REAL all-elements MP pairs launch (171 scenarios) with the
    built-in slow-sweep warning, honestly cancelled with Stop;
  * a worst-6 pairs run filmed to heatmap completion;
  * an MP forward run on the halo channel that fills T/loss with
    measured numbers (matched beam, real apertures);
  * a REAL recovery on examples/failure_analysis/demo.dat with
    3/3 recovered verdicts;
  * genuine CLI transcripts and a scrolling manual-page tour.
"""
from __future__ import annotations

import os
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
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --no-sandbox")
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep10_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep10_failure_study.mp4"

SCENES = [
    Scene("010_title",
          "Deep dive number eight. The Failure Study tab. The Error "
          "Study asked what many small imperfections do together. "
          "This tab asks the opposite. What happens when one thing "
          "breaks completely. A cavity trips, a magnet supply dies. "
          "For a superconducting linac like PIP two, which must "
          "keep running through cavity failures, this is the "
          "resilience question. Everything here is produced live. "
          "Ranking, heatmap, and at the end, a genuine rescue.",
          min_s=6.0),
    Scene("015_layout",
          "The whole tab, D T L section loaded, protons at three "
          "M e V. Down the left, the questions. Which "
          "elements may fail. How they fail. Alone, in pairs, or "
          "in named sets. Whether to attempt a rescue. Plus a Run "
          "group. The column scrolls on its own, so the plots "
          "never get squeezed. On the right, the "
          "answers. Ranking table, criticality bar, and a pane "
          "reserved for the pair failure heatmap. Each will earn "
          "its keep.",
          min_s=6.0),
    Scene("020_targets",
          "Targets first. Four type boxes, Cavity, Quad, Solenoid, "
          "Dipole, and the list of failable elements, each with "
          "its classification in brackets. Only named, active, "
          "uniquely named elements qualify. A drift cannot break, "
          "and a duplicated name is excluded, because failures are "
          "injected by name and a name matching two elements would "
          "be ambiguous. And field maps classify by content, "
          "accelerating means cavity, otherwise solenoid.",
          min_s=6.0),
    Scene("022_targets_toggle",
          "The boxes filter the list live. Untick Quad and the "
          "eight quadrupoles leave, eighteen entries become ten. "
          "Untick Cavity and only the two solenoids remain. Tick "
          "them back and the roster returns. No dipoles in this "
          "lattice, so that box changes nothing here.",
          min_s=6.0),
    Scene("025_subset_select",
          "Selection narrows the sweep. Click four elements and "
          "only those four fail. The rule sits right above the "
          "list. Select a subset, or none means all. An empty "
          "selection is the full sweep, not an empty one. "
          "Remember this trick for the pairs run later.",
          min_s=6.0),
    Scene("030_modes",
          "The failure mode. Off is total. It rides the element's "
          "additive error slot, relative strength to minus one "
          "hundred percent, nothing transfers. Partial scales a "
          "magnet to a fraction of design. Detune keeps a cavity "
          "running but off set point, amplitude scale times "
          "nominal plus an additive phase offset in degrees. "
          "Amplitude accepts zero to two, phase spans plus or "
          "minus one hundred eighty.",
          min_s=6.0),
    Scene("032_mode_wakeup",
          "Fields wake only for modes that use them. On off, both "
          "are greyed. Detune enables both, and the amplitude "
          "flips from zero point nine to one, so a pure phase "
          "detune cannot silently scale the voltage. Partial greys "
          "the phase and resets amplitude to zero point nine, a "
          "ten percent droop. Back on detune we dial zero point "
          "nine and plus ten degrees. A run would label each "
          "scenario, detune, amp zero point nine zero, phi plus "
          "ten. Now back to off.",
          min_s=7.0),
    Scene("040_combination",
          "Combination sets the ambition. Single fails each "
          "element alone, giving the criticality ranking. Pairs "
          "adds every unordered pair, N plus N choose two. Our "
          "eighteen elements mean one hundred seventy one "
          "scenarios, which is why the manual says run single "
          "first, then pairs on the few worst. The singles run "
          "first and fill the heatmap diagonal. Custom builds "
          "explicit failure sets, exactly the scenario your review "
          "board asked about. Let us build one.",
          min_s=7.0),
    Scene("042_custom_sets",
          "Select a quadrupole and its neighbouring gap, then Add "
          "selected as set. The pair lands in the set list joined "
          "by a plus, and the combination flips to custom by "
          "itself. Each set fails as one unit. A second set, the "
          "same way. Clear sets empties the list. Sets survive a "
          "reload with the same names and are pruned if their "
          "names disappear. We reset to single.",
          min_s=7.0),
    Scene("050_recovery",
          "The most interesting group. Fault recovery. Tick re "
          "tune neighbours and HELIX tries to rescue the worst "
          "cases after the sweep, MYRRHA style. Strategy picks the "
          "rescuers. K out of n, the k nearest same category "
          "elements, upstream winning ties. L neighbouring "
          "lattices, whole focusing periods. Manual, names you "
          "supply in the C L I or A P I. One spinner feeds both k "
          "and l. Algorithms, C M A E S, least squares, or "
          "Bayesian optimisation. The cost solver scores each "
          "attempt, envelope for speed, M P when transmission "
          "must be recovered, separate from the forward model. "
          "Top N caps the attempts.",
          min_s=8.0),
    Scene("062_run_group",
          "The Run group. Forward model is the sweep's physics. "
          "Envelope, the fast R M S model, tracks no particle "
          "loss, so transmission and loss will read a dash. M P "
          "tracks real particles through real apertures. Workers "
          "is deliberately greyed out. The G U I sweeps serially, "
          "in process, on the in memory lattice, so unsaved "
          "edits are honoured exactly. Parallel workers belong to "
          "the C L I. Run starts a background worker, Stop "
          "cancels between scenarios, and the status reads idle.",
          min_s=7.0),
    Scene("064_run",
          "Time to break things. Eighteen elements, mode off, "
          "envelope model. Run. Nineteen simulations, the healthy "
          "baseline plus eighteen failures, finish inside a "
          "second, envelope speed is the point of this model. The "
          "table lands already sorted in ranking order, worst "
          "first, a recovered column header appears for later, "
          "and the status reads done, eighteen scenarios. On a "
          "heavier sweep the table, bar and heatmap fill live, "
          "and the pairs run will show exactly that.",
          min_s=7.0),
    Scene("070_ranking",
          "The ranking tells a physical story. Two quadrupoles top "
          "the chart. Losing QUAD zero zero seven blows the "
          "normalised horizontal emittance from one point seven "
          "seven up to three point six two, more than double, "
          "outweighing any single cavity. The eight gaps rank "
          "together below, each costing the same half M e V of "
          "exit energy. And notice the honesty at the bottom. Four "
          "quadrupoles and both solenoids score at or near zero. "
          "The score counts damage, never improvement. The dashes "
          "mean not modelled, not zero, the envelope model "
          "declaring its limits. And epsilon n z is beta gamma "
          "times epsilon z, the usual convention.",
          min_s=8.0),
    Scene("080_bar",
          "The criticality bar, worst on the left, each bar "
          "labelled with its failed element, so a bar reads "
          "straight back to a table row. It caps at the top "
          "fifteen. With eighteen singles, the three mildest never "
          "chart. On a pairs run the label joins both names with a "
          "plus. This is the review slide figure. One glance says "
          "which failure hurts most.",
          min_s=6.0),
    Scene("090_criticality_score",
          "Where the number comes from. A weighted sum of damage "
          "against the healthy baseline. Fractional transmission "
          "loss times ten. Fractional exit energy deviation times "
          "five. Growth of each plane's normalised emittance, "
          "times one apiece. Normalised, so a decelerated beam is "
          "not punished twice through lost adiabatic damping. "
          "Unrecorded terms contribute zero. And a scenario is "
          "flagged beam lost when transmission falls below one "
          "percent or the exit energy goes non finite.",
          min_s=7.0),
    Scene("120_pairs_stop",
          "Now the expensive question. Pairs, on every element, "
          "forward model switched to M P, fifty thousand macro "
          "particles per scenario. One hundred seventy one "
          "scenarios, and the status line says so with a warning. "
          "Pairs of eighteen, slow. Consider envelope, a subset, "
          "or the C L I workers flag. The eighteen singles run "
          "first, so the heatmap diagonal fills before any off "
          "diagonal cell. We let it grind for a while, then "
          "Stop. The worker is interrupted "
          "between scenarios, the status reports cancelled by "
          "user, and Run comes back.",
          min_s=8.0),
    Scene("130_pairs_subset",
          "The affordable version. Back on envelope, we select the "
          "six worst offenders from the single ranking, two "
          "quadrupoles and four gaps. Pairs on six is twenty one "
          "scenarios, and the envelope model finishes the whole "
          "matrix in about a second, same fill order as before, "
          "diagonal then pairs. The pane takes its title, pair "
          "failure criticality. The bright row and column belong "
          "to QUAD zero zero seven. And the brightest off "
          "diagonal cell is QUAD zero zero five plus QUAD zero "
          "zero seven, two point eight seven, worse than either "
          "alone. The interaction is the finding.",
          min_s=8.0),
    Scene("140_heatmap_read",
          "Reading it. The matrix is symmetric, failing A with B "
          "is failing B with A, both cells written together. Axes, "
          "failed element i and j. The scale is inferno with its "
          "colour bar, dark benign, bright dangerous, and the "
          "diagonal is the single ranking you already know. Tick "
          "labels adapt. A shared family prefix like F MAP "
          "underscore gets stripped, and a huge matrix thins its "
          "ticks. Our six names simply fit.",
          min_s=7.0),
    Scene("145_mp_forward",
          "Every transmission cell so far was a dash. Time to "
          "fill them. A new venue, the halo benchmark channel, "
          "twenty four identical FODO cells, and a matched one M e "
          "V proton beam the healthy machine transports without "
          "loss. Five mid channel quadrupoles, mode off, forward "
          "model M P. Every scenario is now a real multi particle "
          "run through real apertures, and the columns fill with "
          "measured numbers. The worst, QUAD zero one two, drops "
          "transmission to eighty nine point six percent. Ten "
          "point four percent of the beam, actually lost, not "
          "just emittance grown.",
          min_s=8.0),
    Scene("150_recovery_run",
          "The finale. Can the machine save itself. The failure "
          "analysis demo, four cavities and three solenoids, "
          "protons at two and a half M e V. Re tune neighbours "
          "on, k out of n with k two, least squares on the "
          "envelope cost, compensate top three. Run. The sweep "
          "ranks the cavities, worst is GAP zero zero four, over "
          "one point one M e V of missing energy. Then, for each "
          "of the three worst, HELIX plants temporary ADJUST "
          "cards on the neighbours' voltage and phase, half to "
          "one and a half times amplitude, plus or minus thirty "
          "degrees, adds a set K E out min objective, and reruns "
          "the matcher. The table gains recovered columns, and "
          "the status settles at done, seven scenarios, three of "
          "three recovered.",
          min_s=9.0),
    Scene("160_recovery_verdict",
          "The verdicts up close. A check mark is a rescue the "
          "matcher achieved, design exit energy back within five "
          "hundredths of an M e V. All three succeed, each using "
          "the three surviving cavities. Rows below the top three "
          "were never attempted, so their verdict stays blank, "
          "and a cross, when you see one, is an honest no. T rec "
          "stays a dash because the cost solver was envelope. "
          "Switch it to M P when transmission itself must be "
          "recovered. The recovered case emittances sit beside "
          "the broken ones.",
          min_s=7.0),
    Scene("170_cli_card",
          "The same engine drives a headless command line. List "
          "elements prints the failable roster, here filtered to "
          "the four cavities. The full run wires every option you "
          "have seen, and here Workers is real, a parallel process "
          "pool for full pairs sweeps. This transcript is genuine. "
          "Seven scenarios, the worst compensated, recovered true "
          "with the three neighbours named, and a C S V carrying "
          "every metric plus the recovered flag, compensators, and "
          "matched settings, ready for a notebook.",
          min_s=8.0),
    Scene("180_api_card",
          "Prefer scripting it. Three calls. Enumerate scenarios "
          "builds the failure list and the name to class map the "
          "injector needs. Failure study dot run sweeps them, in "
          "memory, exactly like the G U I. Compensate takes the "
          "worst scenario and a config, and returns the verdict, "
          "compensator names, and matched settings. Every name is "
          "a real export, and a runnable demo ships in examples "
          "slash failure analysis, ending on the same recovered "
          "true.",
          min_s=7.0),
    Scene("190_manual_tour",
          "Everything here is written down. The failure study "
          "chapter opens with the layout, then every control "
          "group. The mode table. The order N squared pairs "
          "warning. The strategies. Further down, the exact "
          "criticality weights, the C L I invocation, the A P I "
          "snippet, and three notes worth reading twice. Envelope "
          "tracks no loss, a dash is a statement. A blank heatmap "
          "usually means the sweep is still running. And "
          "recovered false is an honest answer. It lives right "
          "after the error study tab.",
          min_s=7.0),
    Scene("200_outro",
          "That is the Failure Study tab. Break everything on "
          "purpose, alone and in pairs. Rank the damage with a "
          "score that cannot be flattered. Stop a sweep that "
          "outgrows its budget. Measure real loss with M P. And "
          "let the matcher practise the rescue before the real "
          "machine needs one. The manual carries every number we "
          "quoted. Elsewhere in the series, the Surrogates tab "
          "teaches neural networks to stand in for expensive "
          "physics. See you there.",
          min_s=7.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def capture_visuals() -> None:
    from PyQt6 import QtWebEngineWidgets  # noqa: F401  (import BEFORE QApplication)
    from PyQt6.QtCore import QEventLoop, QPoint, QRect, QUrl
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import QApplication, QGroupBox, QPushButton

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["ep10"])

    # ---- cards -------------------------------------------------------
    cards.title_card(s["010_title"], "The Failure Study Tab",
                     "Deep dive — break it on purpose, rank it, recover it")
    cards.outro_card(s["200_outro"], [
        "Next — Surrogates: learned stand-ins for physics",
        "Try it — examples/failure_analysis: ranking + rescue",
        "Manual — GUI section, Failure Study chapter",
    ])
    # Terminal cards: transcripts captured from REAL runs of these exact
    # commands (see build notes) — do not edit the output lines.
    cards.terminal_card(s["170_cli_card"], [
        ("cmd", "python -m linac_gen failures examples/failure_analysis/demo.dat \\"),
        ("out", "      --types cavity --list-elements"),
        ("out", "  GAP_001              cavity     RFGap"),
        ("out", "  GAP_002              cavity     RFGap"),
        ("out", "  GAP_003              cavity     RFGap"),
        ("out", "  GAP_004              cavity     RFGap"),
        ("cmd", "python -m linac_gen failures examples/failure_analysis/demo.dat \\"),
        ("out", "      --mode off --forward envelope --energy 2.5 --freq 162.5 --workers 8 \\"),
        ("out", "      --compensate --strategy k_out_of_n --k 2 --top 3 \\"),
        ("out", "      --comp-algorithm least_squares --out failures.csv"),
        ("out", "[failures] 7 scenario(s) over 7 element(s); mode=off; forward=envelope"),
        ("out", "…"),
        ("out", "[failures] compensating: GAP_004:off …"),
        ("out", "           recovered=True compensators=['GAP_003', 'GAP_002', 'GAP_001']"),
        ("out", "…"),
        ("out", "[failures] wrote failures.csv"),
    ], title="Failure sweep + recovery — CLI")
    cards.terminal_card(s["180_api_card"], [
        ("out", "from linac_gen.failures import (FailureKind, enumerate_scenarios,"),
        ("out", "                                FailureStudy, compensate,"),
        ("out", "                                CompensationConfig)"),
        ("gap", ""),
        ("out", "scenarios, n2c, names = enumerate_scenarios(lattice,"),
        ("out", "    kind=FailureKind.OFF, combination='single')"),
        ("out", "res = FailureStudy(lattice=lattice, beam_config=cfg).run("),
        ("out", "    scenarios, names, n2c, combination='single')"),
        ("out", "worst = res.impacts[res.ranking[0]]"),
        ("out", "cr = compensate(lattice, cfg, worst.scenario, n2c, res.baseline,"),
        ("out", "                CompensationConfig(strategy='k_out_of_n', k=2))"),
        ("out", "print(cr.recovered, cr.compensator_names, cr.settings)"),
        ("gap", ""),
        ("out", "# runnable demo: examples/failure_analysis/  (demo.dat + README)"),
    ], title="Python API")

    # ---- window ------------------------------------------------------
    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    win = InterphaseWindow()
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def widget_rect(widget, pad: int = 8) -> QRect:
        tl = widget.mapTo(win, QPoint(0, 0))
        x = max(0, tl.x() - pad)
        y = max(0, tl.y() - pad)
        return QRect(x, y, min(1920 - x, widget.width() + 2 * pad),
                     min(1080 - y, widget.height() + 2 * pad))

    def crop_widget(key: str, widget, pad: int = 8) -> None:
        settle()
        img = win.grab().toImage()
        r = widget_rect(widget, pad)
        img.copy(r).save(s[key])

    def crop_grabber(rect: QRect):
        return lambda: win.grab().toImage().copy(rect)

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    from linac_gen.core.config import BeamConfig

    deck = str(ROOT / "examples/dtl_section.dat")
    lattice, _meta = _parse_lattice_file(deck)
    win.state.set_lattice(lattice, deck)
    # 50k macroparticles: envelope runs ignore it; the MP pairs launch in
    # scene 120 uses it (measured ~0.36 s/scenario — slow enough to watch
    # and to cancel honestly).
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=0.0,
        n_particles=50000))
    go_tab("failure study")
    settle(10)

    ft = win.failures_tab
    groups = {g.title(): g for g in ft.findChildren(QGroupBox)}
    add_btn = next(b for b in ft.findChildren(QPushButton)
                   if b.text() == "Add selected as set")
    clr_btn = next(b for b in ft.findChildren(QPushButton)
                   if b.text() == "Clear sets")

    def list_names() -> list:
        return [ft._elem_list.item(i).text().split("  [")[0]
                for i in range(ft._elem_list.count())]

    def select_names(names) -> None:
        ft._elem_list.clearSelection()
        want = set(names)
        for i in range(ft._elem_list.count()):
            it = ft._elem_list.item(i)
            if it.text().split("  [")[0] in want:
                it.setSelected(True)

    rec = Recorder(WORK / "frames", settle)
    win_frame = lambda: win.grab().toImage()   # noqa: E731

    # ---- 015: whole-tab layout (idle) --------------------------------
    settle(6)
    win.grab().save(s["015_layout"])

    # ---- 020: Targets group still ------------------------------------
    crop_widget("020_targets", groups["Targets"])
    print(f"[measure] element list: {ft._elem_list.count()} items; "
          f"first: {list_names()[:4]}")

    # ---- 022: type checkboxes filter the list live -------------------
    tg_rect = widget_rect(groups["Targets"])
    g_t = crop_grabber(tg_rect)
    rec.start("022_targets_toggle", fps=6.0)
    rec.hold(g_t, 1.5)
    ft._type_checks["quad"].setChecked(False)
    settle(4)
    print(f"[measure] quad off -> {ft._elem_list.count()} items")
    rec.hold(g_t, 2.0)
    ft._type_checks["cavity"].setChecked(False)
    settle(4)
    print(f"[measure] cavity off too -> {ft._elem_list.count()} items")
    rec.hold(g_t, 2.0)
    ft._type_checks["quad"].setChecked(True)
    settle(4)
    rec.hold(g_t, 1.2)
    ft._type_checks["cavity"].setChecked(True)
    settle(4)
    print(f"[measure] restored -> {ft._elem_list.count()} items")
    rec.hold(g_t, 1.5)
    ft._type_checks["dipole"].setChecked(False)
    settle(4)
    rec.hold(g_t, 1.2)
    ft._type_checks["dipole"].setChecked(True)
    settle(4)
    rec.hold(g_t, 1.5)
    rec.finish(scene_by_name("022_targets_toggle"))

    # ---- 025: subset selection ---------------------------------------
    rec.start("025_subset_select", fps=6.0)
    rec.hold(g_t, 1.0)
    for nm in ("QUAD_001", "GAP_001", "QUAD_002", "GAP_002"):
        for i in range(ft._elem_list.count()):
            it = ft._elem_list.item(i)
            if it.text().split("  [")[0] == nm:
                it.setSelected(True)
        settle(2)
        rec.hold(g_t, 0.8)
    rec.hold(g_t, 2.0)
    rec.finish(scene_by_name("025_subset_select"))

    # ---- 030: mode group still ---------------------------------------
    crop_widget("030_modes", groups["Failure mode"])

    # ---- 032: mode-dependent wake-up ---------------------------------
    mg_rect = widget_rect(groups["Failure mode"])
    g_m = crop_grabber(mg_rect)
    rec.start("032_mode_wakeup", fps=6.0)
    rec.hold(g_m, 1.5)
    ft._mode.setCurrentText("detune")          # amp 0.90 -> 1.0, both enable
    settle(3)
    rec.hold(g_m, 2.0)
    ft._mode.setCurrentText("partial")         # amp resets 0.90, phase greys
    settle(3)
    rec.hold(g_m, 2.0)
    ft._mode.setCurrentText("detune")          # back: amp flips to 1.0 again
    settle(3)
    rec.hold(g_m, 1.2)
    for v in (0.95, 0.90):
        ft._amp.setValue(v)
        rec.hold(g_m, 0.5)
    for v in (5.0, 10.0):
        ft._phase.setValue(v)
        rec.hold(g_m, 0.5)
    rec.hold(g_m, 1.5)
    ft._mode.setCurrentText("off")
    ft._phase.setValue(0.0)
    settle(3)
    rec.hold(g_m, 1.2)
    rec.finish(scene_by_name("032_mode_wakeup"))

    # ---- 040: combination group still --------------------------------
    crop_widget("040_combination", groups["Combination"])

    # ---- 042: custom-set builder -------------------------------------
    tg_r = widget_rect(groups["Targets"])
    cg_r = widget_rect(groups["Combination"])
    both = tg_r.united(cg_r)
    g_tc = crop_grabber(both)
    rec.start("042_custom_sets", fps=6.0)
    select_names([])
    rec.hold(g_tc, 1.0)
    select_names(["QUAD_007", "GAP_007"])
    ft._elem_list.scrollToItem(
        next(ft._elem_list.item(i) for i in range(ft._elem_list.count())
             if ft._elem_list.item(i).text().startswith("QUAD_007")))
    settle(3)
    rec.hold(g_tc, 1.2)
    add_btn.click()
    settle(3)
    print(f"[measure] set list rows: "
          f"{[ft._set_list.item(i).text() for i in range(ft._set_list.count())]}; "
          f"combo now {ft._combo.currentText()!r}")
    rec.hold(g_tc, 2.0)
    select_names(["QUAD_005", "GAP_005"])
    settle(2)
    rec.hold(g_tc, 1.0)
    add_btn.click()
    settle(3)
    rec.hold(g_tc, 2.0)
    clr_btn.click()
    settle(3)
    rec.hold(g_tc, 1.5)
    ft._combo.setCurrentText("single")
    select_names([])
    settle(3)
    rec.hold(g_tc, 1.5)
    rec.finish(scene_by_name("042_custom_sets"))

    # ---- 050 / 062: recovery + run group stills ----------------------
    crop_widget("050_recovery", groups["Fault recovery (compensation)"])
    # Scroll the left control column to its bottom so the Run group —
    # buttons, progress bar and STATUS LINE — is fully on camera for the
    # Run-group still and for every recorded run that follows.
    from PyQt6.QtWidgets import QScrollArea
    _w = groups["Run"]
    while _w is not None and not isinstance(_w, QScrollArea):
        _w = _w.parentWidget()
    lcol = _w
    assert lcol is not None, "left scroll column not found"
    lcol.verticalScrollBar().setValue(lcol.verticalScrollBar().maximum())
    settle(6)
    print(f"[measure] column scrolled to "
          f"{lcol.verticalScrollBar().value()}/"
          f"{lcol.verticalScrollBar().maximum()}")
    crop_widget("062_run_group", groups["Run"])
    print(f"[measure] status text before any run: {ft._status.text()!r}")

    # ---- 064: THE SPINE — real single-OFF envelope sweep -------------
    rec.start("064_run", fps=6.0)
    rec.hold(win_frame, 1.5)
    t0 = time.monotonic()
    ft._run_btn.click()
    rec.wait_until(win_frame, lambda: ft._run_btn.isEnabled(),
                   cap_s=300, stable_s=0.4)
    print(f"[measure] 064 single sweep wall: {time.monotonic()-t0:.1f} s; "
          f"status: {ft._status.text()!r}; rows: {ft._table.rowCount()}; "
          f"cols: {ft._table.columnCount()}")
    rec.hold(win_frame, 3.0)
    rec.finish(scene_by_name("064_run"))
    settle(10)

    # ---- 070 / 080 / 090: ranking, bar, score-anatomy stills ---------
    win.grab().save(s["070_ranking"])
    img = win.grab().toImage()
    tbl_r = widget_rect(ft._table)
    bar_r = widget_rect(ft._bar)
    img.copy(bar_r).save(s["080_bar"])
    # table close-up: top rows only
    top_r = QRect(tbl_r.x(), tbl_r.y(), tbl_r.width(),
                  min(tbl_r.height(), 340))
    img.copy(top_r).save(s["090_criticality_score"])
    hdr = [ft._table.horizontalHeaderItem(c).text()
           for c in range(ft._table.columnCount())]
    print(f"[measure] final headers: {hdr}")
    r0 = [ft._table.item(0, c).text() if ft._table.item(0, c) else ""
          for c in range(ft._table.columnCount())]
    print(f"[measure] worst row: {r0}")

    # ---- 120: all-elements MP pairs launch + honest Stop -------------
    ft._combo.setCurrentText("pairs")
    ft._forward.setCurrentText("mp")
    select_names([])
    settle(4)
    rec.start("120_pairs_stop", fps=10.0)
    rec.hold(win_frame, 1.0)
    ft._run_btn.click()
    rec.hold(win_frame, 1.0)      # catch the >60-scenario slow-sweep hint
    print(f"[measure] 120 early status: {ft._status.text()!r}")
    rec.hold(win_frame, 14.0)     # diagonal (18 singles) + pair fill
    print(f"[measure] 120 status at stop click: {ft._status.text()!r}; "
          f"progress {ft._progress.value()}%")
    ft._stop_btn.click()
    rec.wait_until(win_frame, lambda: ft._run_btn.isEnabled(),
                   cap_s=120, stable_s=0.3)
    print(f"[measure] 120 status after stop: {ft._status.text()!r}; "
          f"rows: {ft._table.rowCount()}")
    rec.hold(win_frame, 2.5)
    rec.finish(scene_by_name("120_pairs_stop"))
    settle(10)

    # ---- 130: worst-6 pairs to completion (envelope) -----------------
    ft._forward.setCurrentText("envelope")
    worst6 = ["QUAD_007", "QUAD_005", "GAP_001", "GAP_005", "GAP_007",
              "GAP_004"]
    select_names(worst6)
    ft._elem_list.scrollToItem(
        next(ft._elem_list.item(i) for i in range(ft._elem_list.count())
             if ft._elem_list.item(i).text().startswith("QUAD_005")))
    settle(4)
    rec.start("130_pairs_subset", fps=6.0)
    rec.hold(win_frame, 1.5)
    t0 = time.monotonic()
    ft._run_btn.click()
    rec.wait_until(win_frame, lambda: ft._run_btn.isEnabled(),
                   cap_s=300, stable_s=0.4)
    print(f"[measure] 130 subset pairs wall: {time.monotonic()-t0:.1f} s; "
          f"status: {ft._status.text()!r}")
    rec.hold(win_frame, 3.5)
    rec.finish(scene_by_name("130_pairs_subset"))
    settle(10)

    # ---- 140: heatmap close-up ---------------------------------------
    crop_widget("140_heatmap_read", ft._heat)

    # ---- 145: MP forward model fills T/loss (halo channel) -----------
    halo = str(ROOT / "examples/halo_fodo.dat")
    lat2, _m2 = _parse_lattice_file(halo)
    win.state.set_lattice(lat2, halo)
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching import find_fodo_cells, find_matched_input_twiss
    ref = ReferenceParticle(species=PROTON, w_kin=1.0, frequency=162.5)
    tw = find_matched_input_twiss(lat2, ref, *find_fodo_cells(lat2)[0])
    print(f"[measure] halo matched twiss: ax={tw['alpha_x']:.4f} "
          f"bx={tw['beta_x']:.4f}")
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=1.0, frequency=162.5, current=0.0,
        n_particles=2000,
        alpha_x=tw["alpha_x"], beta_x=tw["beta_x"],
        alpha_y=tw["alpha_y"], beta_y=tw["beta_y"]))
    settle(8)
    ft._combo.setCurrentText("single")     # 130 left it on "pairs"
    mid5 = ["QUAD_010", "QUAD_011", "QUAD_012", "QUAD_013", "QUAD_014"]
    select_names(mid5)
    ft._elem_list.scrollToItem(
        next(ft._elem_list.item(i) for i in range(ft._elem_list.count())
             if ft._elem_list.item(i).text().startswith("QUAD_012")))
    ft._forward.setCurrentText("mp")
    settle(4)
    rec.start("145_mp_forward", fps=6.0)
    rec.hold(win_frame, 1.5)
    t0 = time.monotonic()
    ft._run_btn.click()
    rec.wait_until(win_frame, lambda: ft._run_btn.isEnabled(),
                   cap_s=600, stable_s=0.4)
    print(f"[measure] 145 MP subset wall: {time.monotonic()-t0:.1f} s; "
          f"status: {ft._status.text()!r}")
    rec.hold(win_frame, 3.5)
    rec.finish(scene_by_name("145_mp_forward"))
    settle(10)
    r0 = [ft._table.item(0, c).text() if ft._table.item(0, c) else ""
          for c in range(ft._table.columnCount())]
    print(f"[measure] 145 worst row: {r0}")

    # ---- 150: REAL recovery on the failure-analysis demo -------------
    demo = str(ROOT / "examples/failure_analysis/demo.dat")
    lat3, _m3 = _parse_lattice_file(demo)
    win.state.set_lattice(lat3, demo)
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=2.5, frequency=162.5, current=0.0,
        emit_nx=0.25, emit_ny=0.25, emit_z=0.30,
        beta_x=1.0, beta_y=1.0, beta_z=10.0))
    settle(8)
    ft._forward.setCurrentText("envelope")
    ft._combo.setCurrentText("single")
    select_names([])
    ft._comp_on.setChecked(True)
    ft._strategy.setCurrentText("k_out_of_n")
    ft._k.setValue(2)
    ft._comp_algo.setCurrentText("least_squares")
    ft._comp_cost.setCurrentText("envelope")
    ft._n_comp.setValue(3)
    settle(4)
    rec.start("150_recovery_run", fps=6.0)
    rec.hold(win_frame, 1.5)
    t0 = time.monotonic()
    ft._run_btn.click()
    rec.wait_until(win_frame, lambda: ft._run_btn.isEnabled(),
                   cap_s=600, stable_s=0.4)
    print(f"[measure] 150 recovery wall: {time.monotonic()-t0:.1f} s; "
          f"status: {ft._status.text()!r}")
    rec.hold(win_frame, 3.5)
    rec.finish(scene_by_name("150_recovery_run"))
    settle(10)
    hdr = [ft._table.horizontalHeaderItem(c).text()
           for c in range(ft._table.columnCount())]
    print(f"[measure] 150 headers: {hdr}")
    for r in range(min(4, ft._table.rowCount())):
        row = [ft._table.item(r, c).text() if ft._table.item(r, c) else ""
               for c in range(ft._table.columnCount())]
        print(f"[measure] 150 row {r}: {row}")

    # ---- 160: verdict close-up (table + status composite) ------------
    settle(4)
    img = win.grab().toImage()
    tbl_img = img.copy(widget_rect(ft._table))
    run_img = img.copy(widget_rect(groups["Run"]))
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    canvas.fill(QColor("#0b1220"))
    p = QPainter(canvas)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    tw_ = min(1840, tbl_img.width() * 2)
    tbl_scaled = tbl_img.scaledToWidth(tw_)
    if tbl_scaled.height() > 640:
        tbl_scaled = tbl_img.scaledToWidth(1500)
    p.drawImage((1920 - tbl_scaled.width()) // 2, 60, tbl_scaled)
    run_scaled = run_img.scaledToWidth(760)
    p.drawImage((1920 - run_scaled.width()) // 2,
                min(1080 - run_scaled.height() - 30,
                    tbl_scaled.height() + 120), run_scaled)
    p.end()
    canvas.save(s["160_recovery_verdict"])

    # ---- 190: scrolling manual-page tour (QtWebEngine) ---------------
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    page_fp = ROOT / "site" / "10_gui" / "06c_failures_tab.html"
    v = QWebEngineView()
    v.resize(1920, 1080)
    v.show()
    loaded = {"ok": False}
    v.loadFinished.connect(lambda ok: loaded.__setitem__("ok", True))
    v.load(QUrl.fromLocalFile(str(page_fp)))
    t0 = time.monotonic()
    while not loaded["ok"] and time.monotonic() - t0 < 30:
        settle(2)
        time.sleep(0.02)
    for _ in range(40):                      # ~2 s render settle
        settle(2)
        time.sleep(0.05)
    height = {"v": 0}
    v.page().runJavaScript(
        "document.body.scrollHeight",
        lambda h: height.__setitem__("v", int(h or 0)))
    t0 = time.monotonic()
    while height["v"] == 0 and time.monotonic() - t0 < 5:
        settle(2)
        time.sleep(0.02)
    maxy = max(0, height["v"] - 1080)
    print(f"[measure] manual page scrollHeight={height['v']} maxy={maxy}")
    g_web = lambda: v.grab().toImage()       # noqa: E731
    rec.start("190_manual_tour", fps=6.0)
    rec.hold(g_web, 3.0)
    y = 0
    while y < maxy:
        y = min(y + 150, maxy)
        v.page().runJavaScript(f"window.scrollTo(0,{y})")
        rec.tick(g_web, 3)
    rec.hold(g_web, 3.0)
    rec.finish(scene_by_name("190_manual_tour"))

    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    # NO win.close(): dirty-state modal confirm hangs offscreen (ep07).


def main() -> None:
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
