"""Episode 9 — The Error Study tab in depth (tolerances).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep09_error_study.py
Output: tutorials/rendered/ep09_error_study.mp4 (+ .srt)

Maximum-detail cut.  Real demonstrations, all captured live:
  * every form control exercised on camera (target types, patterns,
    uniform half-widths, beam errors, row deletion, run guardrails);
  * a REAL 20-seed FODO ensemble with the progress bar counting;
  * a tolerance A/B on the 6-cell 5 MeV FODO (0.2 mm vs 1.5 mm)
    with the two ensemble popups side by side;
  * an on-camera Stop of a 200-seed run with the partial ensemble;
  * the quad_alignment example project running from deck ERROR_*
    directives alone, space charge ON;
  * a forensics terminal card computed in-process from the study;
  * manual-tour captures of the directive reference and honesty pages.
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
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep09_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep09_error_study.mp4"

SCENES = [
    Scene("010_title",
          "Deep dive number seven. The Error Study tab. Until now "
          "every machine we tracked was perfect. Real magnets are "
          "misaligned by fractions of a millimetre, real fields are "
          "off by fractions of a percent, and the real question is "
          "not what does my design do, but what does the ensemble "
          "of imperfect machines that will actually get built do. "
          "That is a tolerance study, and this tab runs them.",
          min_s=6.0),
    Scene("020_idea",
          "The idea in one breath. You describe errors as "
          "statistical distributions. Quadrupole offsets, gaussian, "
          "zero point two millimetres R M S. The tab builds many "
          "copies of your machine, each with its own random draw of "
          "every error, tracks the same beam through all of them, "
          "and aggregates the results into bands. The copies are "
          "called seeds, and the four zones on this screen take you "
          "from declaration to results.",
          min_s=6.0),
    Scene("025_anatomy",
          "The tab is a map of the workflow. On top, two form sub "
          "tabs that describe errors, element side and beam side. "
          "Below them, the registered errors list, the reviewable "
          "budget. Then the orbit correction group, which decides "
          "how each seed is operated. At the bottom, the run group. "
          "Declare, review, correct, run. No keyboard shortcuts "
          "here, everything is form driven.",
          min_s=5.0),
    Scene("030_element_form",
          "Registering an element error takes one form. Target "
          "type. A name pattern, an fnmatch wildcard against "
          "element names, auto filled to the whole family. The "
          "parameter, whose menu follows the type. The "
          "distribution. And the six decimal size box, sigma for a "
          "gaussian, half width for a uniform, in the parameter's "
          "own units. Cutoff truncates gaussian draws at three "
          "sigma by default. We register zero point two millimetre "
          "offsets in d x, add, then d y, add.",
          min_s=6.0),
    Scene("032_target_types",
          "Switch the target type and two things rewrite at once, "
          "the parameter menu and the pattern hint, which follows "
          "the family, QUAD, GAP, BEND, or SOL, each with a "
          "wildcard. Quadrupoles offer offsets, tilt, a fractional "
          "gradient error, and g three and g four higher pole "
          "content. Cavities offer relative voltage, phase offset "
          "in degrees, and frequency offset. Bends and solenoids "
          "share a single fractional field error, and offsets and "
          "tilt are common to every type.",
          min_s=6.0),
    Scene("034_uniform_single",
          "Patterns narrow as far as you like. Type one exact name, "
          "QUAD zero zero three, pick the fractional gradient "
          "error, switch to uniform, and the size box now reads as "
          "a half width, zero point zero five. Add, and a third "
          "row lands, targeting one element. Uniform draws ignore "
          "the cutoff. Gaussian truncation is by redrawing until "
          "the value falls inside it, the TraceWin convention, so "
          "no probability piles up at the limits.",
          min_s=6.0),
    Scene("040_beam_form",
          "The second sub tab holds beam errors, the machine's "
          "input side. Thirteen parameters in four groups. Six "
          "centroid channels, three fractional emittance growths, "
          "three mismatch channels, and shot to shot current "
          "variation. Units follow the channel, millimetres, "
          "milliradians, degrees, M e V, or fractions.",
          min_s=5.0),
    Scene("042_beam_register",
          "Two get registered for real. Centroid x, gaussian, zero "
          "point one millimetres, add. Then mismatch x, uniform, "
          "zero point zero five, add again. Each lands as a BEAM "
          "row. Per seed, the study perturbs a copy of the beam "
          "configuration before the beam is generated.",
          min_s=5.0),
    Scene("050_registered",
          "Every registered error is one row. A source tag, element "
          "or beam, then pattern, parameter, distribution, and "
          "size. Element rows list first, beam rows after. This "
          "list, plus any directives the deck itself carries, is "
          "the tolerance budget under test.",
          min_s=5.0),
    Scene("052_delete_row",
          "List hygiene is one button. Select the uniform gradient "
          "row, delete selected, and exactly that row disappears. "
          "The beam rows go the same way. We prune back to the two "
          "quadrupole offsets, one clean question for the headline "
          "ensemble.",
          min_s=5.0),
    Scene("056_guardrails",
          "Empty the list entirely and Run protects you. The "
          "warning fires only when there is truly nothing to "
          "randomise, and it looks in three places. The element "
          "list, the beam list, and ERROR directives carried by "
          "the parsed lattice itself. So an empty list does not "
          "mean an empty study, the deck can bring its own budget. "
          "Run also refuses without a lattice or a beam.",
          min_s=5.0),
    Scene("060_correction",
          "Now the subtle part. A real machine with misaligned "
          "quadrupoles is never operated raw, operators correct "
          "the orbit first. Tick apply orbit correction and the "
          "greyed controls wake, and every seed gets its own "
          "correction pass before tracking. Method auto picks one "
          "to one steering when steerer and monitor counts match, "
          "S V D otherwise, or you can force either. n iter caps "
          "the passes. Tolerance is the target R M S orbit. B P M "
          "noise adds measurement error to the readings. The "
          "targets box steers onto recorded set points instead of "
          "zero, and readings can come from envelope centroids "
          "instead of tracked particles. Without this box, a "
          "tolerance study of a correctable machine is unfairly "
          "pessimistic.",
          min_s=7.0),
    Scene("062_standalone",
          "The same corrector exists outside studies. On the "
          "lattice tab, correct orbit runs a one shot pass driven "
          "by the deck's adjust steerer cards, and applies the "
          "fitted kicks through the undo bus as a single step. A "
          "fitted lattice reroutes plain save to save as, so the "
          "source deck is never silently overwritten. Next to it, "
          "B P M targets loads runtime set points, and a project "
          "setting can fire correction automatically when a "
          "project with steerer cards loads.",
          min_s=6.0),
    Scene("070_run",
          "Run study. n seeds accepts two to ten thousand and "
          "defaults to fifty. Twenty here, for speed. The status "
          "line opens with an important statement, whether space "
          "charge is on or off, taken from the numerics tab "
          "exactly as a toolbar run would take it. Zero current, "
          "so off. Then the progress counts seeds, one full multi "
          "particle tracking pass each. An error study has no "
          "envelope mode. Every seed is the real thing.",
          min_s=6.0),
    Scene("071_statusbar",
          "When the last seed lands, the app wide status bar, "
          "magnified here, answers the obvious question. Error "
          "study finished, twenty seeds, open the results tab. Off "
          "we go.",
          min_s=5.0),
    Scene("080_ensemble",
          "The ensemble view. Sigma x and sigma y along the "
          "machine, a solid mean over all twenty seeds with dashed "
          "plus and minus one sigma bands. Narrow bands mean a "
          "robust machine, wide bands mean sensitivity. Below, the "
          "histogram of final transmission, and the summary line "
          "condenses it, twenty seeds, mean one hundred percent, "
          "minimum one hundred, sigma zero. Every imperfect "
          "machine transmits everything. With this budget, this "
          "matched line loses nothing.",
          min_s=6.0),
    Scene("082_tolerance_ab",
          "The same view earning its keep. A six cell F O D O line "
          "at five M e V with twenty millimetre apertures, and the "
          "same study twice, twelve seeds each. Left, a zero point "
          "two millimetre alignment budget, every seed at one "
          "hundred percent. Right, the budget relaxed to one point "
          "five millimetres. The mean drops to about fifty eight "
          "percent, seven seeds transmit everything, five lose "
          "essentially the whole beam. Two lumps and nothing "
          "between is a brittle machine, and the mean alone would "
          "never tell you.",
          min_s=6.0),
    Scene("084_forensics",
          "When a histogram looks like that, interrogate the "
          "results object. transmission stats condenses the "
          "ensemble. mean, standard deviation, and percentile "
          "return any recorded quantity across seeds, and "
          "percentile one hundred is the worst case at every "
          "location. The per seed finals name the guilty seed, "
          "index four, zero percent. Draws are deterministic per "
          "seed, so that machine can be regenerated and diffed "
          "against the nominal one. And with correction enabled, "
          "corrected kicks and correction history expose each "
          "seed's steerer solution.",
          min_s=6.0),
    Scene("086_base_seed",
          "Two spinboxes govern the statistics. n seeds, fifty by "
          "default, one hundred to two hundred for production. And the "
          "base seed, which offsets every per seed random draw. "
          "Finished a study and want more statistics? Set the base "
          "seed to one thousand and run again. The new batch is "
          "statistically independent, and the two ensembles pool "
          "cleanly.",
          min_s=5.0),
    Scene("088_stop_partial",
          "Stop deserves a live demonstration. Two hundred seeds "
          "requested, on a fifty thousand particle beam so there "
          "is time to think. Run, the counter climbs, and mid "
          "flight we press stop. The status flashes stopping, "
          "finishing the current seed, because the stop is "
          "cooperative. Then the verdict, stopped, with exactly "
          "how many of the two hundred were kept. Only whole seeds "
          "enter the ensemble, a truncated recording would corrupt "
          "every mean and band.",
          min_s=6.0),
    Scene("089_partial_view",
          "And the partial ensemble is a first class citizen. The "
          "view plots the kept seeds like any finished study, and "
          "the statistics are valid because every seed in there "
          "is complete. Partial statistics are still honest "
          "statistics.",
          min_s=5.0),
    Scene("090_cards",
          "One more source of errors exists, the lattice file "
          "itself. ERROR directives in a TraceWin deck declare "
          "the same errors inline, and HELIX reads them on "
          "import. This is the shipped combined realistic budget, "
          "verbatim. The cutoff card sets three sigma truncation "
          "for the whole file, applied at end of parse, last card "
          "wins. Each card covers the next N elements of its "
          "family, six quads here, two cavities, one bend. The r "
          "slot picks the distribution, two is gaussian, one is "
          "uniform. Alignments read in millimetres, rotations in "
          "degrees, field errors in percent, and the beam card's "
          "final slot is percent current jitter, not milliamps. "
          "The comment in this deck says one milliamp, but the "
          "manual and the parser both read that slot as percent.",
          min_s=7.0),
    Scene("092_cards_run_sc",
          "Deck directives run with an empty list. This is the "
          "quad alignment example project, ten M e V protons at "
          "ten milliamps, nothing registered in the form. Run "
          "proceeds anyway, because the directives live on the "
          "parsed lattice and merge in at run time. And read the "
          "status line, space charge ON, with exactly the P I C "
          "configuration the numerics tab holds. Ten seeds, each "
          "a full multi particle pass with space charge. Five "
          "such ready made projects ship in the examples folder.",
          min_s=6.0),
    Scene("094_directive_reference",
          "Every directive is documented on one manual page. The "
          "quick map names the families, then the distribution "
          "codes, with honest rows. TraceWin's constant mode maps "
          "to a gaussian with a parse warning. ERROR SET RATIO is "
          "parsed but never consumed, so sweep amplitudes by "
          "scaling sigmas. And the variant table is equally "
          "plain. Dynamic and coupled cards are absorbed as "
          "static uncoupled errors, and the stat file and per "
          "cell R F Q cards are deferred no op markers.",
          min_s=6.0),
    Scene("096_honest_edges",
          "The manual is just as direct about what is not "
          "modelled. In the alignment table, d z draws are stored "
          "on the element but no longitudinal shift is performed, "
          "and pitch and yaw are reserved slots, not yet "
          "honoured. The tracker honours d x, d y, and tilt.",
          min_s=5.0),
    Scene("097_draw_conventions",
          "Every draw is static per seed, held constant through "
          "the simulation, so time varying jitter is not "
          "modelled. Seeds run serially, the n workers argument "
          "is accepted and ignored, parallelise across studies "
          "with different base seeds instead. And since the "
          "draw convention change of twenty twenty six, results "
          "are deliberately not bit identical to older runs. The "
          "new statistics are the correct ones.",
          min_s=5.0),
    Scene("098_performance",
          "Planning wall time is a table lookup. A small lattice "
          "without space charge costs seconds for fifty seeds. A "
          "full machine ensemble with the P I C solver on belongs "
          "overnight. The working rule sits underneath. Iterate "
          "the budget without space charge, verify the final "
          "budget once with the solver on.",
          min_s=5.0),
    Scene("100_outro",
          "That is the Error Study tab in full. Declare "
          "imperfections through the form or ship them in the "
          "deck, correct like an operator, run with real space "
          "charge, stop honestly, read bands and histograms, and "
          "interrogate the results object when a second lump "
          "appears. Next, the Failure Study tab, where instead of "
          "many small errors we ask about single large ones. What "
          "happens when a cavity or a magnet simply dies. See you "
          "there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def capture_visuals() -> None:
    # QtWebEngine must be imported before the QApplication exists.
    from PyQt6 import QtWebEngineWidgets  # noqa: F401
    from PyQt6.QtCore import QEventLoop, QPoint, QRect, QTimer, QUrl, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen
    from PyQt6.QtWidgets import (QApplication, QGroupBox, QPushButton,
                                 QTabWidget)

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["ep09"])
    cards.title_card(s["010_title"], "The Error Study Tab",
                     "Deep dive — tolerances, seeds, ERROR decks")
    cards.outro_card(s["100_outro"], [
        "Next — the Failure Study tab",
        "Then — Surrogates",
        "Later — space charge physics, the voice assistant",
    ])

    # 090: the combined_realistic ERROR_* block, verbatim from disk.
    deck_cr = ROOT / "examples/error_studies/combined_realistic/combined_realistic.dat"
    cr_lines = deck_cr.read_text().splitlines()
    i0 = next(i for i, ln in enumerate(cr_lines)
              if ln.startswith("; ---- error directives"))
    i1 = next(i for i, ln in enumerate(cr_lines)
              if ln.startswith("ERROR_BEAM_STAT"))
    block = [("cmd", f"sed -n '{i0 + 1},{i1 + 1}p' combined_realistic.dat")]
    block += [("out", ln) for ln in cr_lines[i0:i1 + 1]]
    cards.terminal_card(s["090_cards"], block,
                        title="examples/error_studies/combined_realistic/"
                              "combined_realistic.dat")

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
        return QRect(max(0, tl.x() - pad), max(0, tl.y() - pad),
                     widget.width() + 2 * pad, widget.height() + 2 * pad)

    def crop_img(rect: QRect) -> QImage:
        img = win.grab().toImage()
        w = min(img.width() - rect.x(), rect.width())
        h = min(img.height() - rect.y(), rect.height())
        return img.copy(rect.x(), rect.y(), w, h)

    def crop_widget(key: str, widget, pad: int = 8) -> None:
        settle()
        crop_img(widget_rect(widget, pad)).save(s[key])

    def union_rect(widgets, pad: int = 8) -> QRect:
        r = widget_rect(widgets[0], pad)
        for w_ in widgets[1:]:
            r = r.united(widget_rect(w_, pad))
        return r

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    def find_button(parent, text: str) -> QPushButton:
        for b in parent.findChildren(QPushButton):
            if b.text().strip() == text:
                return b
        raise LookupError(f"button {text!r} not found")

    def inset_composite(path: str, base: QImage, strip: QImage,
                        scale: float, y_frac: float = 0.80) -> None:
        """Full-window base + a magnified strip inset with accent border."""
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        p = QPainter(canvas)
        p.drawImage(0, 0, base)
        mag = strip.scaled(int(strip.width() * scale),
                           int(strip.height() * scale),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        x = (1920 - mag.width()) // 2
        y = int(1080 * y_frac) - mag.height() // 2
        p.fillRect(x - 6, y - 6, mag.width() + 12, mag.height() + 12,
                   QColor("#0b1220"))
        p.drawImage(x, y, mag)
        p.setPen(QPen(QColor("#4ade80"), 3))
        p.drawRect(x - 4, y - 4, mag.width() + 8, mag.height() + 8)
        p.end()
        canvas.save(path)

    # ---- lattice + matched beam (the healthy baseline from ep01) ----
    deck = str(ROOT / "examples/fodo_cell.dat")
    lattice, _meta = _parse_lattice_file(deck)
    win.state.set_lattice(lattice, deck)
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching import find_fodo_cells, find_matched_input_twiss
    ref = ReferenceParticle(species=PROTON, w_kin=1.0, frequency=352.21)
    tw = find_matched_input_twiss(lattice, ref, *find_fodo_cells(lattice)[0])

    def fodo_cfg(n_particles: int) -> BeamConfig:
        return BeamConfig(
            species="proton", energy=1.0, frequency=352.21, current=0.0,
            n_particles=n_particles,
            alpha_x=tw["alpha_x"], beta_x=tw["beta_x"],
            alpha_y=tw["alpha_y"], beta_y=tw["beta_y"])

    win.state.set_beam_config(fodo_cfg(1500))
    go_tab("error study")
    settle(10)

    et = win.errors_tab
    groups = {g.title(): g for g in et.findChildren(QGroupBox)}
    form_tabs = next(t for t in et.findChildren(QTabWidget)
                     if any(t.tabText(i).startswith("Element")
                            for i in range(t.count())))
    list_box = groups["Registered errors"]
    run_grp = groups["Run"]
    rec = Recorder(WORK / "frames", settle)

    # ---- 056: the "No errors" guardrail, modal grabbed by a timer ----
    # The QMessageBox blocks the event loop; a repeating timer armed
    # BEFORE the click grabs the top-level modal and closes it (the
    # documented offscreen-modal pattern — never leave one open).
    modal_shot: list = []
    tries = {"n": 0}

    def _grab_close_modal() -> None:
        m = QApplication.activeModalWidget()
        if m is not None:
            try:
                modal_shot.append(m.grab().toImage())
            finally:
                m.close()
            mtimer.stop()
            return
        tries["n"] += 1
        if tries["n"] > 60:
            mtimer.stop()

    mtimer = QTimer(win)
    mtimer.setInterval(250)
    mtimer.timeout.connect(_grab_close_modal)
    mtimer.start()
    et._run_btn.click()          # empty list + directive-free deck -> refusal
    settle(6)
    base = win.grab().toImage()
    if modal_shot:
        inset_composite(s["056_guardrails"], base, modal_shot[0],
                        scale=2.2, y_frac=0.50)
    else:
        crop_img(union_rect([list_box, run_grp])).save(s["056_guardrails"])

    # Safety net for the REST of the storyboard: no scene below wants a
    # modal, so close any stray one (e.g. an unexpected worker-failure
    # box) instead of hanging the offscreen event loop forever.
    def _close_stray_modal() -> None:
        m = QApplication.activeModalWidget()
        if m is not None:
            print(f"[warn] closed stray modal: {m.windowTitle()!r}")
            m.close()

    stray_timer = QTimer(win)
    stray_timer.setInterval(1500)
    stray_timer.timeout.connect(_close_stray_modal)
    stray_timer.start()

    # ---- 030: register dx + dy on camera (form + list crop) ---------
    fl_rect = union_rect([form_tabs, list_box])

    def fl_grab():
        return crop_img(fl_rect)

    def set_combo(combo, text: str) -> None:
        for j in range(combo.count()):
            if combo.itemText(j) == text:
                combo.setCurrentIndex(j)
                return
        raise LookupError(f"combo item {text!r} not found")

    add_elem_btn = find_button(et, "Add element error")
    rec.start("030_element_form", fps=7.0)
    rec.hold(fl_grab, 1.2)
    rec.type_into(fl_grab, et._pattern, "QUAD_*")
    set_combo(et._param, "dx")
    rec.hold(fl_grab, 0.8)
    et._sigma.setValue(0.2)
    rec.hold(fl_grab, 1.0)
    add_elem_btn.click()
    rec.hold(fl_grab, 1.4)
    set_combo(et._param, "dy")
    rec.hold(fl_grab, 0.8)
    et._sigma.setValue(0.2)
    rec.hold(fl_grab, 0.8)
    add_elem_btn.click()
    rec.hold(fl_grab, 1.8)
    rec.finish(scene_by_name("030_element_form"))
    settle(4)

    # ---- 025: anatomy still (whole tab, two rows registered) --------
    win.grab().save(s["025_anatomy"])

    # ---- 032: cycle the four target types --------------------------
    form_rect = widget_rect(form_tabs)

    def form_grab():
        return crop_img(form_rect)

    showcase = {"Quadrupole": "gradient_rel", "Cavity / RFGap": "voltage_rel",
                "Bend / Dipole": "field_rel", "Solenoid": "field_rel"}
    rec.start("032_target_types", fps=7.0)
    rec.hold(form_grab, 1.2)
    for i in range(et._target.count()):
        et._target.setCurrentIndex(i)
        rec.hold(form_grab, 0.9)
        set_combo(et._param, showcase[et._target.currentText()])
        rec.hold(form_grab, 1.4)
    et._target.setCurrentIndex(0)          # back to Quadrupole (QUAD_*)
    rec.hold(form_grab, 1.0)
    rec.finish(scene_by_name("032_target_types"))
    settle(4)

    # ---- 034: exact-name pattern + uniform half-width --------------
    assert any(e.name == "QUAD_003" for e in win.state.lattice.elements)
    rec.start("034_uniform_single", fps=7.0)
    rec.hold(fl_grab, 1.0)
    rec.type_into(fl_grab, et._pattern, "QUAD_003")
    set_combo(et._param, "gradient_rel")
    rec.hold(fl_grab, 0.8)
    set_combo(et._dist, "uniform")
    rec.hold(fl_grab, 0.8)
    et._sigma.setValue(0.05)
    rec.hold(fl_grab, 1.0)
    add_elem_btn.click()
    rec.hold(fl_grab, 2.0)
    rec.finish(scene_by_name("034_uniform_single"))
    set_combo(et._dist, "gaussian")
    et._sigma.setValue(0.1)
    settle(4)

    # ---- 040/042: beam form + two live registrations ---------------
    for i in range(form_tabs.count()):
        if form_tabs.tabText(i).startswith("Beam"):
            form_tabs.setCurrentIndex(i)
    settle(6)
    crop_widget("040_beam_form", form_tabs)

    add_beam_btn = find_button(et, "Add beam error")
    rec.start("042_beam_register", fps=7.0)
    rec.hold(fl_grab, 1.2)
    set_combo(et._b_param, "centroid_x")
    et._b_sigma.setValue(0.1)
    rec.hold(fl_grab, 1.0)
    add_beam_btn.click()
    rec.hold(fl_grab, 1.6)
    set_combo(et._b_param, "mismatch_x")
    rec.hold(fl_grab, 0.8)
    set_combo(et._b_dist, "uniform")
    et._b_sigma.setValue(0.05)
    rec.hold(fl_grab, 1.0)
    add_beam_btn.click()
    rec.hold(fl_grab, 2.0)
    rec.finish(scene_by_name("042_beam_register"))
    set_combo(et._b_dist, "gaussian")
    form_tabs.setCurrentIndex(0)
    settle(4)

    # ---- 050: the five-row list, then the full-window idea shot ----
    crop_widget("050_registered", list_box)
    win.grab().save(s["020_idea"])

    # ---- 052: delete three rows on camera --------------------------
    list_rect = widget_rect(list_box)

    def list_grab():
        return crop_img(list_rect)

    del_btn = find_button(list_box, "Delete selected")
    rec.start("052_delete_row", fps=7.0)
    rec.hold(list_grab, 1.2)
    for _ in range(3):                      # row 2 = uniform, then 2x beam
        et._list.setCurrentRow(2)
        rec.hold(list_grab, 1.0)
        del_btn.click()
        rec.hold(list_grab, 1.2)
    rec.hold(list_grab, 1.5)
    rec.finish(scene_by_name("052_delete_row"))
    settle(4)
    assert len(et._element_errors) == 2 and not et._beam_errors

    # ---- 060: correction children wake on the tick -----------------
    corr_rect = widget_rect(groups["Orbit correction"])

    def corr_grab():
        return crop_img(corr_rect)

    rec.start("060_correction", fps=6.0)
    rec.hold(corr_grab, 2.0)
    et._corr_enable.setChecked(True)
    rec.hold(corr_grab, 4.0)
    rec.finish(scene_by_name("060_correction"))
    et._corr_enable.setChecked(False)
    settle(2)

    # ---- 062: the standalone corrector on the Lattice tab ----------
    go_tab("lattice")
    settle(8)
    lt = win.lattice_tab
    base = win.grab().toImage()
    strip_rect = (widget_rect(lt._btn_correct, 6)
                  .united(widget_rect(lt._btn_bpm_targets, 6)))
    strip = base.copy(strip_rect)
    inset_composite(s["062_standalone"], base, strip, scale=2.4, y_frac=0.82)
    go_tab("error study")
    settle(6)

    # ---- 070: the REAL 20-seed run, recorded -----------------------
    et._n_seeds.setValue(20)
    et._base_seed.setValue(0)
    run_rect = widget_rect(run_grp)

    def run_grab():
        return crop_img(run_rect)

    rec.start("070_run", fps=8.0)
    rec.hold(run_grab, 1.2)
    et._run_btn.click()
    rec.wait_until(run_grab, lambda: et._run_btn.isEnabled(),
                   cap_s=900, stable_s=0.4)
    rec.hold(run_grab, 3.0)
    rec.finish(scene_by_name("070_run"))
    settle(6)

    # ---- 071: the app-wide completion message, magnified -----------
    base = win.grab().toImage()
    sb = win._statusbar
    msg_rect = widget_rect(sb._msg_seg, 4)
    inset_composite(s["071_statusbar"], base, base.copy(msg_rect),
                    scale=2.0, y_frac=0.78)

    # ---- 080: ensemble popup for the 20-seed study -----------------
    def ensemble_shot(key: str, size=(1600, 900)) -> None:
        assert win.show_result_plot("ensemble"), "no ensemble plot"
        settle(10)
        pop = win.results_tab._popups.get("ensemble")
        pop.resize(*size)
        settle(8)
        pop.grab().save(s[key])
        pop.close()
        settle(2)

    ensemble_shot("080_ensemble")
    go_tab("error study")
    settle(4)

    # ---- 082/084: tolerance A/B + forensics (engine-driven; the
    # popup renders whatever study sits on app state, exactly as the
    # tab stores it at error_study_tab.py:510) ------------------------
    import numpy as np
    from linac_gen.errors.error_model import ErrorStudy
    lat_ab, _m = _parse_lattice_file(
        str(ROOT / "examples/correction_demo/correction_demo.dat"))
    cfg_ab = BeamConfig(
        species="proton", energy=5.0, frequency=352.21,
        current=0.0, n_particles=1500,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.05, emit_ny=0.05, emit_z=0.10,
        alpha_x=0.0, beta_x=2.0, alpha_y=0.0, beta_y=2.0,
        alpha_z=0.0, beta_z=1.0)

    def ab_study(sigma_mm: float):
        st = ErrorStudy(lat_ab, cfg_ab, n_seeds=12)
        st.add_error(pattern="QUAD_*", parameter="dx", sigma=sigma_mm)
        st.add_error(pattern="QUAD_*", parameter="dy", sigma=sigma_mm)
        return st.run()

    res_tight = ab_study(0.2)
    res_loose = ab_study(1.5)

    def popup_img(results, size=(950, 1000)) -> QImage:
        win.state.error_study_results = results
        assert win.show_result_plot("ensemble")
        settle(10)
        pop = win.results_tab._popups.get("ensemble")
        pop.resize(*size)
        settle(8)
        img = pop.grab().toImage()
        pop.close()
        settle(2)
        return img

    img_a = popup_img(res_tight)
    img_b = popup_img(res_loose)
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    canvas.fill(QColor("#0b1220"))
    p = QPainter(canvas)
    p.drawImage(7, 40, img_a)
    p.drawImage(963, 40, img_b)
    p.end()
    canvas.save(s["082_tolerance_ab"])
    go_tab("error study")
    settle(4)

    # forensics card: every value computed from the live study object
    ts = res_loose.transmission_stats()
    finals = [float(np.asarray(r.transmission)[-1])
              for r in res_loose._recorders]
    worst = int(np.argmin(finals))
    p100 = float(np.nanmax(res_loose.percentile("sigma_x", 100)))
    cards.terminal_card(s["084_forensics"], [
        ("cmd", "results = state.error_study_results   # the 1.5 mm study"),
        ("cmd", "results.transmission_stats()"),
        ("out", "{'mean': %.3f, 'min': %.3f, 'max': %.3f, 'std': %.3f}"
                % (ts["mean"], ts["min"], ts["max"], ts["std"])),
        ("gap", ""),
        ("cmd", "finals = [r.transmission[-1] for r in results._recorders]"),
        ("out", "[" + ", ".join(f"{f:.4g}" for f in finals) + "]"),
        ("cmd", "int(np.argmin(finals))                # the guilty seed"),
        ("out", str(worst)),
        ("gap", ""),
        ("cmd", "results.percentile('sigma_x', 100).max()   # worst case"),
        ("out", "%.3f   # mm, anywhere along the machine" % p100),
        ("cmd", "results.n_seeds, results.n_requested"),
        ("out", "(%d, %d)" % (res_loose.n_seeds, res_loose.n_requested)),
    ], title="python — interrogating ErrorStudyResults")

    # ---- 086: base-seed spinbox stepped on camera ------------------
    rec.start("086_base_seed", fps=6.0)
    rec.hold(run_grab, 1.5)
    for v in (250, 500, 750, 1000):
        et._base_seed.setValue(v)
        rec.hold(run_grab, 0.7)
    rec.hold(run_grab, 2.5)
    rec.finish(scene_by_name("086_base_seed"))
    et._base_seed.setValue(0)
    settle(2)

    # ---- 088: honest Stop of a 200-seed run ------------------------
    win.state.set_beam_config(fodo_cfg(50_000))
    settle(4)
    et._n_seeds.setValue(200)
    rec.start("088_stop_partial", fps=8.0)
    rec.hold(run_grab, 1.2)
    et._run_btn.click()
    rec.wait_until(run_grab, lambda: et._progress.value() >= 2, cap_s=180)
    rec.hold(run_grab, 1.0)
    et._stop_btn.click()                    # -> "stopping — finishing…"
    rec.wait_until(run_grab, lambda: et._run_btn.isEnabled(),
                   cap_s=300, stable_s=0.3)
    rec.hold(run_grab, 3.5)
    rec.finish(scene_by_name("088_stop_partial"))
    settle(6)
    print(f"[088] final status: {et._status.text()!r}")

    # ---- 089: the partial ensemble is a first-class citizen --------
    ensemble_shot("089_partial_view")
    go_tab("error study")
    settle(4)

    # ---- 092: example project, empty list, space charge ON ---------
    import json
    pj = ROOT / ("examples/error_studies/quad_alignment_tolerance/"
                 "quad_alignment.lgproj")
    win._apply_project_dict(json.loads(pj.read_text()),
                            project_path=str(pj), silent=True)
    settle(8)
    go_tab("error study")
    settle(6)
    while et._list.count():                 # empty the form lists for real
        et._list.setCurrentRow(0)
        et._delete_selected()
    settle(4)
    assert not et._element_errors and not et._beam_errors
    et._n_seeds.setValue(10)
    et._base_seed.setValue(0)
    lr_rect = union_rect([list_box, run_grp])

    def lr_grab():
        return crop_img(lr_rect)

    rec.start("092_cards_run_sc", fps=6.0)
    rec.hold(lr_grab, 2.0)
    et._run_btn.click()
    rec.wait_until(lr_grab, lambda: et._run_btn.isEnabled(),
                   cap_s=600, stable_s=0.4)
    rec.hold(lr_grab, 3.0)
    rec.finish(scene_by_name("092_cards_run_sc"))
    settle(6)

    # ---- 094/096/097/098: manual-tour captures (QtWebEngine) -------
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    import time as _time
    view = QWebEngineView()
    view.resize(1920, 1080)
    view.show()
    load_state = {}
    view.loadFinished.connect(lambda ok: load_state.update(done=True, ok=ok))

    def web_load(page: str) -> None:
        load_state.clear()
        view.load(QUrl.fromLocalFile(str(ROOT / "site" / page)))
        t0 = _time.time()
        while not load_state.get("done") and _time.time() - t0 < 30:
            settle(5)
            _time.sleep(0.02)
        t0 = _time.time()
        while _time.time() - t0 < 2.0:       # let fonts/layout settle
            settle(5)
            _time.sleep(0.02)

    def web_scroll(anchor_js: str, wait_s: float = 1.2) -> None:
        view.page().runJavaScript(anchor_js)
        t0 = _time.time()
        while _time.time() - t0 < wait_s:
            settle(5)
            _time.sleep(0.02)

    def web_grab():
        return view.grab().toImage()

    web_load("08_errors/02_error_directives.html")
    rec.start("094_directive_reference", fps=6.0)
    web_scroll("document.getElementById('quick-map')"
               ".scrollIntoView({behavior:'auto'})", 0.4)
    rec.hold(web_grab, 3.0)
    web_scroll("document.getElementById('distribution-code-r')"
               ".scrollIntoView({behavior:'smooth'})", 0.1)
    rec.hold(web_grab, 3.2)
    web_scroll("document.getElementById('error_set_ratio')"
               ".scrollIntoView({behavior:'smooth'})", 0.1)
    rec.hold(web_grab, 2.8)
    web_scroll("document.getElementById('variant-and-deferred-directives')"
               ".scrollIntoView({behavior:'smooth'})", 0.1)
    rec.hold(web_grab, 3.2)
    rec.finish(scene_by_name("094_directive_reference"))

    web_load("08_errors/01_overview.html")
    web_scroll("document.getElementById('element-alignment-errors')"
               ".scrollIntoView({behavior:'auto'})", 0.5)
    web_grab().save(s["096_honest_edges"])

    web_load("08_errors/05_running_studies.html")
    web_scroll("[...document.querySelectorAll('.admonition')]"
               ".find(a => a.textContent.includes('Draw conventions'))"
               ".scrollIntoView({behavior:'auto'}); window.scrollBy(0, -430)", 0.6)
    web_grab().save(s["097_draw_conventions"])
    web_scroll("document.getElementById('performance-tips')"
               ".scrollIntoView({behavior:'auto'})", 0.5)
    web_grab().save(s["098_performance"])

    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    # NO win.close(): dirty state -> modal unsaved-changes confirm ->
    # offscreen hang (see ep07). Process exit tears Qt down instead.


def main() -> None:
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)          # worker/WebEngine threads: skip Qt teardown


if __name__ == "__main__":
    main()
