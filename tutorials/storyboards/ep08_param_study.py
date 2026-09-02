"""Episode 8 — The Param Study manager in depth (maximum-detail cut).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep08_param_study.py
Output: tutorials/rendered/ep08_param_study.mp4 (+ .srt)

Runs SIX real studies through the tab's own widgets and the CLI twin:
a 5-point line scan, a 25-run 5x5 two-quad grid, a deliberately
failing beam-energy scan, a 32-sample LHS survey, a 12-run
multi-particle seed-repeats study (with an on-camera Stop -> resume),
and a hand-written study.json with an observable, run headless.
Every number narrated is measured in this build or verified at source.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
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

WORK = ROOT / "tutorials" / "rendered" / "ep08_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep08_param_study.mp4"

SCENES = [
    Scene("010_title",
          "Deep dive number six. The Param Study manager. One run "
          "answers one question. A study answers a family of them. "
          "What happens as I scan this gradient, that voltage, the "
          "beam current, or a solver setting. This tab defines the "
          "family, runs it headless with resume and provenance "
          "built in, and analyses it live. Six real studies this "
          "episode. A line scan, a grid, a failure, a survey, seed "
          "repeats, and a hand written spec.",
          min_s=8.0),
    Scene("020_layout",
          "The left half defines the study in three groups. "
          "Parameters, strategy, and execution. The right half "
          "analyses it, four views named on their tabs. Runs, one D "
          "sweep, Map, and Overlay sigma of z. All four update live "
          "while a study runs, and sit empty until one loads. The "
          "machine is the D T L section, loaded from a file saved "
          "on disk. That matters in a moment.",
          min_s=7.0),
    Scene("025_file_on_disk",
          "Because a study never runs what is on the screen. It "
          "runs the saved file, headless, one subprocess per point. "
          "So the tab guards that seam. The lattice here carries an "
          "unsaved edit, and Start refuses with this dialog. The "
          "study executes the file on disk, and refusing beats "
          "silently studying different physics. No lattice at all, "
          "or an empty parameter table, are refused the same way. "
          "We undo the edit and carry on clean.",
          min_s=7.0),
    Scene("030_params",
          "Parameters. Pick an element, pick a numeric attribute, "
          "press Add. The row pre fills around the live value. "
          "Gradient eight, so start six point four, stop nine point "
          "six. Twenty percent either side, five points, linear. "
          "Selector and label are fixed. Baseline, start, stop, n, "
          "and spacing edit in place, and we widen to four through "
          "twelve tesla per metre. Spacing accepts log, geometric "
          "steps, positive endpoints only. Back to linear. The "
          "picker also offers Beam, the input distribution, as a "
          "pseudo element. Beam current becomes a row the same way, "
          "pre filled minus one to plus one around zero. Remove "
          "selected takes it out again.",
          min_s=10.0),
    Scene("035_selector_grammar",
          "Everything the tab writes is a plain study dot json, and "
          "you can write one by hand. This one runs later in the "
          "episode. The selector grammar is one language "
          "everywhere. At six dot gradient is element six, one "
          "based, attribute gradient. The element name works too. A "
          "bare word like current is a beam field. And n x is a "
          "numerics knob, one of four structural selectors that "
          "live in the spec and the C L I. A parameter can carry an "
          "explicit values list instead of start, stop, n. The "
          "observables block we meet near the end.",
          min_s=8.0),
    Scene("040_strategy",
          "Strategy decides how rows combine. The strategy list, "
          "seed repeats, base seed, forty two by default, and "
          "Samples, read only by random and latin hypercube. So the "
          "counter has something to count, a second quadrupole row "
          "is staged in the table above, minus twelve to minus "
          "four, five points.",
          min_s=6.0),
    Scene("045_run_counter",
          "Watch the total runs counter. Grid is the cross product, "
          "five times five, twenty five. Zip walks the rows in "
          "lockstep, five runs, and unequal lengths are rejected "
          "loudly. One at a time gives eleven. Five plus five, plus "
          "one all nominal reference run, which is why the baseline "
          "column exists. Random ignores n and takes the sample "
          "budget, thirty two, seeded so the same spec always draws "
          "the same points. Latin hypercube, same budget, spread "
          "evenly. And seed repeats multiply everything. Times "
          "three is ninety six, seeds forty two, forty three, forty "
          "four.",
          min_s=9.0),
    Scene("050_execution",
          "Execution. The name pre fills with a time stamp. We type "
          "our own. The folder is the study root, defaulting to "
          "runs slash studies under the working directory, browse "
          "button beside it. Mode, envelope or multi particle. "
          "Workers, one subprocess per run. One means serial, in "
          "process, and the range tops out one below your core "
          "count. Start, Stop, and Open study share the bottom row.",
          min_s=7.0),
    Scene("060_running",
          "One row, grid, five gradients, envelope, serial. Start "
          "study creates the folder and begins. Start greys out "
          "while the worker lives, the bar counts five runs, and "
          "the status line reports done and failed. On longer "
          "studies it grows a live E T A, coming up soon. Envelope "
          "points cost milliseconds, so this is over immediately. "
          "Complete, with the summary path in the status line.",
          min_s=7.0),
    Scene("070_folder",
          "On disk, a study is a folder you can archive, share, or "
          "resume anywhere. Study dot json is the full "
          "specification, including a S H A two five six pin of "
          "the input deck. Lattice holds a provenance snapshot. "
          "Runs has one folder per run, its results file plus a "
          "status dot json recording ok or failed, parameters, and "
          "metrics. Results are written to a dot part file and "
          "renamed only when complete, so a crash never leaves a "
          "half written file looking valid. Summary holds two "
          "derived views. Summary dot C S V, and runs manifest dot "
          "json, the recorded plan.",
          min_s=8.0),
    Scene("075_integrity_guards",
          "Those files have teeth. The deck is pinned by S H A two "
          "five six at creation. We edited it afterwards and asked "
          "the command line to continue. It refuses with exactly "
          "this message, results would mix two different machines. "
          "Restore the bytes and the same study answers again. The "
          "manifest is the second guard. If the spec no longer "
          "expands to the recorded plan, the load refuses just as "
          "loudly.",
          min_s=8.0),
    Scene("078_grid25",
          "The second quadrupole row goes back in, and the same "
          "Start button launches the twenty five run grid, two "
          "workers this time. The analysis views refresh after "
          "every finished run, and the grid completes in seconds. "
          "Every study in this episode goes through this identical "
          "path.",
          min_s=6.0),
    Scene("080_runs_view",
          "The Runs view is a live, sortable table. One row per "
          "run. Index, the tag encoding the parameter values, seed, "
          "status, the parameters as columns, then transmission, "
          "sizes, emittances, energy, elapsed, and an error column, "
          "empty on healthy rows. Transmission just reads none, "
          "envelope mode counts no particles. Click the sigma x "
          "header and the family sorts by final beam size, "
          "ascending, then descending. Failed rows go dark red, in "
          "a moment, for real.",
          min_s=8.0),
    Scene("082_open_in_results",
          "Every run wrote a complete results file, so the table is "
          "a door. Double click a healthy row and the app opens "
          "that run in the Results tab, tiles and all. A study is a "
          "browsable library of full runs.",
          min_s=6.0),
    Scene("084_failed_runs",
          "Failure handling, on purpose. We scanned beam energy "
          "from minus one to plus three M e V through the beam "
          "pseudo element. Two points are unphysical. The failed "
          "rows go dark red, the error column carries the "
          "exception, value error, energy must be greater than zero "
          "M e V, and the status still reads complete. The healthy "
          "three ran, the summary was written. Failed runs are "
          "rows, not crashes.",
          min_s=8.0),
    Scene("090_1d",
          "The one D sweep, on the grid. X is a parameter, Y is any "
          "of seventeen built in quantities plus your observables. "
          "Transmission is the default, and in envelope mode it is "
          "honestly empty. Sigma x shows the response, five "
          "gradients, each point the mean over the five second quad "
          "settings, bars showing that spread. Normalised "
          "emittance. Final energy, flat, these knobs never touch "
          "the R F. Elapsed, the cost per point. And log x, log y, "
          "one checkbox each.",
          min_s=9.0),
    Scene("094_group_by",
          "Group by unfolds the mean. Grouped by the second "
          "quadrupole, twenty five runs become five coloured "
          "series, and the legend names each value. Strongest first "
          "gradient, weakest second, smallest final size, about ten "
          "point six millimetres. The opposite corner, thirteen "
          "point two. In log y, the error bars step aside where "
          "they would lie.",
          min_s=7.0),
    Scene("100_map_heatmap",
          "Two parameters against a quantity is the Map view. On a "
          "full grid it renders a true heatmap, one cell per run, "
          "colour bar labelled with the quantity. Final horizontal "
          "size, ten point six to thirteen point two millimetres, "
          "darkest where the gradients balance best. Repeats "
          "collapse by mean per cell. A missing cell, and it falls "
          "back to a scatter rather than interpolate a lie.",
          min_s=7.0),
    Scene("101_overlay",
          "Overlay reads each selected run's results file and "
          "draws full curves along the machine. The list multi "
          "selects, three come pre selected, we take six, and Draw "
          "redraws. The legend is the run tag, and the curves run "
          "the full length of the machine. Fourteen quantities are "
          "offered, each run contributes what its file actually "
          "holds. Sigma x. Beta x, the Twiss function. Longitudinal "
          "emittance. This is how a family of machines differs "
          "along the line, not just at the exit.",
          min_s=8.0),
    Scene("102_map_scatter",
          "The same two gradients, surveyed instead. A latin "
          "hypercube study, thirty two samples, run through the "
          "same Start button in seconds. No grid, so the Map view "
          "draws its honest fallback, colour mapped points, same "
          "axes, same scale. Sample broadly, find the interesting "
          "corner, then grid it finely.",
          min_s=7.0),
    Scene("104_mp_stop_resume",
          "Multi particle mode, where runs cost real time. Four "
          "gradient points, three seed repeats, twelve runs at a "
          "quarter million macro particles, two workers. Start. "
          "Once the first runs land, the status line grows a live "
          "E T A, and finished rows fill in above grey pending "
          "ones. Now the killer feature. Stop is graceful. In "
          "flight runs finish cleanly, then, stopped. Resume with "
          "Start. Completed runs are kept. Press Start again, and "
          "the bar comes back pre filled with the runs already "
          "done. The engine re expands the plan, skips the "
          "finished, and completes the rest. A crash or a power "
          "cycle behaves exactly the same. You only lose what was "
          "in flight.",
          min_s=12.0),
    Scene("106_mp_error_bars",
          "Twelve runs, four gradient points. The one D view "
          "aggregates the three seeds at each point into mean, plus "
          "and minus one standard deviation. The bars are real seed "
          "scatter, about a hundredth of a millimetre here, on a "
          "beam of eight millimetres, far below the trend. When the "
          "bars rival your effect, the study is telling you to "
          "raise the particle count, or accept the uncertainty.",
          min_s=7.0),
    Scene("107_observables",
          "The exit is not the only place worth measuring. An "
          "observable extracts any envelope quantity at any s "
          "position or named element, per run, resolved to a "
          "position at creation, so evaluation never needs the "
          "lattice again. The command line twin runs our hand "
          "written spec. Same engine, same folders, live progress, "
          "and an observable called s x mid, sigma x at the fifth "
          "quadrupole.",
          min_s=7.0),
    Scene("108_observables_gui",
          "Open the finished folder in the tab, the status line "
          "confirms it, and the observable is a first class "
          "column. There is s x mid in the Runs table, per run. And "
          "it joins the one D quantity menu, sigma x at the mid "
          "line against gradient, no results file ever reopened.",
          min_s=6.0),
    Scene("110_open",
          "The tab also remembers. The study root and the last "
          "study are stored per user. This is a brand new copy of "
          "the tab, fresh from the constructor, and it has already "
          "reopened the study, loaded last study, on its own. "
          "Define here, run overnight elsewhere with the C L I, "
          "and tomorrow it is open before you ask.",
          min_s=6.0),
    Scene("112_cli_plan",
          "The command line twin has four verbs. Plan, run, "
          "resume, summarize. Plan prints the expanded run table "
          "and executes nothing. Five runs, every selector, value, "
          "and seed. The sanity check before a thousand run "
          "commitment.",
          min_s=6.0),
    Scene("114_cli_verbs",
          "Run takes a spec or a folder, parallel sets the pool, "
          "resume is the same verb again. Retry failed re queues "
          "only failed rows. Here it re ran the two unphysical "
          "energy points, they failed again, honestly, and the "
          "closing line says so. Force wipes and starts over. "
          "Summarize rebuilds the C S V from the per run status "
          "files. The C S V is a view, the folders are the truth. "
          "And the assistant drives this same engine through its "
          "run study tool.",
          min_s=8.0),
    Scene("116_manual_tour",
          "Everything here is written down. The G U I chapter "
          "covers the tab, defining a study, the study folder, "
          "analysing results, and the headless twins. The running "
          "chapter covers the C L I, the spec, and the folder "
          "layout with its resume semantics. Search reaches both "
          "pages from anywhere in the manual.",
          min_s=7.0),
    Scene("120_outro",
          "That is the Param Study manager. Six real studies in one "
          "sitting, every folder resumable, every claim inspectable "
          "on disk. Next, the Error Study tab, where the question "
          "changes from what if I turn this knob, to what happens "
          "when every magnet is slightly wrong at once. See you "
          "there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def _wrap(text: str, width: int = 70) -> list[str]:
    return textwrap.wrap(text, width) or [""]


def capture_visuals() -> None:
    from PyQt6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (QApplication, QGroupBox, QMessageBox,
                                 QPushButton)

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication([])
    cards.title_card(s["010_title"], "The Param Study Manager",
                     "Deep dive — scan, organise, analyse")
    cards.outro_card(s["120_outro"], [
        "Ran: line scan · 5×5 grid · LHS survey · seeded repeats · CLI",
        "Next — the Error Study tab (tolerances)",
        "Then — Failure Study and Surrogates",
        "Later — space charge physics, the voice assistant",
    ])

    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    win = InterphaseWindow()
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def region_img(widget, pad: int = 8) -> QImage:
        img = win.grab().toImage()
        tl = widget.mapTo(win, QPoint(0, 0))
        x = max(0, tl.x() - pad)
        y = max(0, tl.y() - pad)
        w = min(img.width() - x, widget.width() + 2 * pad)
        h = min(img.height() - y, widget.height() + 2 * pad)
        return img.copy(x, y, w, h)

    def crop_widget(key: str, widget, pad: int = 8) -> None:
        settle()
        region_img(widget, pad).save(s[key])

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    # ------------------------------------------------------------------
    # sandboxed working copy of the deck (the SHA scene mutates it; the
    # repo tree is never touched)
    # ------------------------------------------------------------------
    stroot = Path(tempfile.mkdtemp(prefix="helix_studies_"))
    deck = stroot / "dtl_section.dat"
    shutil.copy2(ROOT / "examples/dtl_section.dat", deck)
    deck_bytes = deck.read_bytes()

    lattice, _meta = _parse_lattice_file(str(deck))
    win.state.set_lattice(lattice, str(deck))
    from linac_gen.core.config import BeamConfig
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=0.0,
        n_particles=20000))
    go_tab("param study")
    settle(10)

    st = win.study_tab
    st._folder.setText(str(stroot))
    from PyQt6.QtWidgets import QSplitter
    split = st.findChild(QSplitter)
    split.setSizes([820, 1080])   # left groups are ~763 px wide
    settle(4)
    groups = {g.title(): g for g in st.findChildren(QGroupBox)}
    add_btn = next(b for b in st.findChildren(QPushButton)
                   if b.text() == "Add")
    rm_btn = next(b for b in st.findChildren(QPushButton)
                  if b.text() == "Remove selected")
    an = st._analysis
    rec = Recorder(WORK / "frames", settle)

    def an_tab(prefix: str) -> None:
        for i in range(an.count()):
            if an.tabText(i).casefold().startswith(prefix.casefold()):
                an.setCurrentIndex(i)
                return
        raise LookupError(f"analysis tab {prefix!r} not found")

    def pick_element(text: str) -> None:
        idx = st._elem_combo.findText(text, Qt.MatchFlag.MatchContains)
        assert idx >= 0, f"element {text!r} not in picker"
        st._elem_combo.setCurrentIndex(idx)
        settle(3)

    def pick_attr(data: str) -> None:
        for j in range(st._attr_combo.count()):
            if st._attr_combo.itemData(j) == data:
                st._attr_combo.setCurrentIndex(j)
                settle(2)
                return
        raise LookupError(f"attr {data!r} not in picker")

    def set_cells(row: int, start: str, stop: str, n: str,
                  spacing: str | None = None) -> None:
        t = st._ptable
        t.item(row, 3).setText(start)
        t.item(row, 4).setText(stop)
        t.item(row, 5).setText(n)
        if spacing is not None:
            t.item(row, 6).setText(spacing)
        settle(3)

    def clear_rows() -> None:
        while st._ptable.rowCount():
            st._ptable.removeRow(0)
        st._refresh_run_count()
        settle(2)

    def start_and_wait(cap_s: float = 300) -> None:
        st._start.click()
        settle(4)
        import time as _t
        t0 = _t.monotonic()
        while not st._start.isEnabled():
            settle(4)
            if _t.monotonic() - t0 > cap_s:
                raise RuntimeError("study did not finish in time")
        settle(6)

    def run_cli(*args: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        return subprocess.run(
            [sys.executable, "-m", "linac_gen", *args],
            capture_output=True, text=True, cwd=str(stroot), env=env,
            timeout=600)

    # ------------------------------------------------------------------
    # 020 layout — empty table, defaults visible
    # ------------------------------------------------------------------
    settle(8)
    win.grab().toImage().save(s["020_layout"])

    # ------------------------------------------------------------------
    # 025 unsaved-edits refusal (real modal, grabbed + closed by timer)
    # ------------------------------------------------------------------
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    quad1 = win.state.lattice.elements[5]           # @6 QUAD_001
    assert getattr(quad1, "name", "") == "QUAD_001"
    win.state.bus.do(ParamChangeCommand(quad1, "gradient", 8.0, 8.5))
    settle(3)
    assert win.state.bus.dirty, "bus should be dirty for the guard demo"
    bg_img = win.grab().toImage()
    box: dict = {"img": None, "text": ""}
    poke_t = QTimer()
    poke_t.setInterval(250)

    def _poke() -> None:
        w = QApplication.activeModalWidget()
        if w is not None and isinstance(w, QMessageBox):
            box["img"] = w.grab().toImage()
            box["text"] = w.text()
            w.close()
            poke_t.stop()
    poke_t.timeout.connect(_poke)
    poke_t.start()
    st._start.click()                       # blocks in the modal loop
    poke_t.stop()
    settle(4)
    assert box["img"] is not None, "unsaved-edits dialog was not captured"
    assert "unsaved edits" in box["text"], box["text"]
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    p = QPainter(canvas)
    p.drawImage(0, 0, bg_img)
    p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, 130))
    dlg = box["img"]
    p.drawImage((1920 - dlg.width()) // 2, (1080 - dlg.height()) // 2, dlg)
    p.end()
    canvas.save(s["025_file_on_disk"])
    win.state.bus.undo()
    settle(3)
    assert not win.state.bus.dirty, "undo should restore the clean state"

    # ------------------------------------------------------------------
    # 030 parameters MOTION — add, prefill, edit, log, beam row, remove
    # ------------------------------------------------------------------
    par_grp = groups["Parameters"]

    def par_grab() -> QImage:
        return region_img(par_grp)

    rec.start("030_params")
    rec.hold(par_grab, 1.5)
    pick_element("QUAD_001")
    pick_attr("gradient")
    rec.hold(par_grab, 1.5)
    add_btn.click()
    settle(3)
    t = st._ptable
    t.resizeColumnsToContents()
    settle(2)
    rec.hold(par_grab, 2.5)                 # prefilled 6.4 .. 9.6, 5, lin
    assert t.item(0, 3).text() == "6.4" and t.item(0, 4).text() == "9.6"
    set_cells(0, "4", "12", "5")
    rec.hold(par_grab, 2.0)
    t.item(0, 6).setText("log")
    settle(2)
    rec.hold(par_grab, 2.0)
    t.item(0, 6).setText("lin")
    settle(2)
    rec.hold(par_grab, 1.0)
    pick_element("Beam (input distribution)")
    pick_attr("current")
    rec.hold(par_grab, 1.0)
    add_btn.click()
    settle(3)
    assert t.item(1, 3).text() == "-1" and t.item(1, 4).text() == "1"
    rec.hold(par_grab, 2.0)
    t.selectRow(1)
    settle(2)
    rm_btn.click()
    settle(2)
    rec.hold(par_grab, 2.0)
    rec.finish(scene_by_name("030_params"))
    assert t.rowCount() == 1

    # ------------------------------------------------------------------
    # 035 hand-written study.json (also drives the observables study)
    # ------------------------------------------------------------------
    obs_json_text = (
        '{ "__kind__": "linac_gen_study", "__version__": 1,\n'
        '  "name": "obs_demo",\n'
        '  "input": "dtl_section.dat",\n'
        '  "mode": "envelope",\n'
        '  "strategy": "grid",\n'
        '  "parameters": [\n'
        '    { "selector": "@6.gradient", "values": [4, 6, 8, 10, 12] },\n'
        '    { "selector": "current",     "values": [0.0] },\n'
        '    { "selector": "nx",          "values": [32] }\n'
        '  ],\n'
        '  "observables": [\n'
        '    { "name": "sx_mid", "quantity": "sigma_x",\n'
        '      "at": { "element": "QUAD_005" } }\n'
        '  ] }\n')
    (stroot / "obs_demo.json").write_text(obs_json_text)
    from linac_gen.study.spec import load_spec
    load_spec(stroot / "obs_demo.json")          # must parse + validate
    lines = [("cmd", "cat obs_demo.json")]
    lines += [("out", ln) for ln in obs_json_text.rstrip("\n").split("\n")]
    cards.terminal_card(s["035_selector_grammar"], lines,
                        title="study.json — the selector grammar")

    # ------------------------------------------------------------------
    # 040/045 strategy group — stage the second quad row, cycle counter
    # ------------------------------------------------------------------
    pick_element("QUAD_002")
    pick_attr("gradient")
    add_btn.click()
    settle(3)
    set_cells(1, "-12", "-4", "5")
    st._strategy.setCurrentText("grid")
    settle(4)
    assert st._run_count.text() == "25", st._run_count.text()
    crop_widget("040_strategy", groups["Strategy"])

    strat_grp = groups["Strategy"]

    def strat_grab() -> QImage:
        return region_img(strat_grp)

    expected = {"grid": "25", "zip": "5", "oat": "11",
                "random": "32", "lhs": "32"}
    rec.start("045_run_counter")
    rec.hold(strat_grab, 1.5)
    for name in ("grid", "zip", "oat", "random", "lhs"):
        st._strategy.setCurrentText(name)
        settle(4)
        got = st._run_count.text()
        assert got == expected[name], f"{name}: counter {got}"
        rec.hold(strat_grab, 2.6)
    st._repeats.setValue(3)
    settle(4)
    assert st._run_count.text() == "96", st._run_count.text()
    rec.hold(strat_grab, 3.0)
    rec.finish(scene_by_name("045_run_counter"))
    print("[measure] run counter: grid 25, zip 5, oat 11, random 32, "
          "lhs 32, lhs x3 repeats 96 — all read from the widget")
    st._repeats.setValue(1)
    st._strategy.setCurrentText("grid")
    st._ptable.removeRow(1)
    st._refresh_run_count()
    settle(3)
    assert st._run_count.text() == "5"

    # ------------------------------------------------------------------
    # 050 execution MOTION — type the study name, workers to 1
    # ------------------------------------------------------------------
    exe_grp = groups["Execution"]

    def exe_grab() -> QImage:
        return region_img(exe_grp)

    rec.start("050_execution")
    rec.hold(exe_grab, 2.0)                 # timestamped default name
    rec.type_into(exe_grab, st._name, "quad_line_scan")
    st._workers.setValue(1)
    settle(2)
    rec.hold(exe_grab, 2.5)
    rec.finish(scene_by_name("050_execution"))

    # ------------------------------------------------------------------
    # 060 the 5-run line study, recorded
    # ------------------------------------------------------------------
    st._mode.setCurrentText("envelope")
    settle(2)
    rec.start("060_running")
    rec.hold(exe_grab, 1.2)
    st._start.click()
    rec.wait_until(exe_grab, lambda: st._start.isEnabled(),
                   cap_s=300, stable_s=0.4)
    rec.hold(exe_grab, 3.0)
    rec.finish(scene_by_name("060_running"))
    settle(10)
    line_dir = stroot / "quad_line_scan"
    assert (line_dir / "study.json").exists()

    # ------------------------------------------------------------------
    # 070 folder card from the REAL study folder
    # ------------------------------------------------------------------
    runs = sorted(p.name for p in (line_dir / "runs").iterdir())
    r0 = sorted((line_dir / "runs" / runs[0]).iterdir())
    r0names = " · ".join(p.name for p in r0)
    lines = [("cmd", "tree quad_line_scan/"),
             ("out", "quad_line_scan/"),
             ("out", "├─ study.json            # full spec + lattice_sha256 pin"),
             ("out", "├─ lattice/              # provenance snapshot of the deck"),
             ("out", f"├─ runs/                 # {len(runs)} runs, one folder each"),
             ("out", f"│   ├─ {runs[0]}/"),
             ("out", f"│   │    └─ {r0names}   (.h5 written .part → renamed)"),
             ("out", f"│   └─ … {len(runs) - 1} more"),
             ("out", "└─ summary/"),
             ("out", "    ├─ summary.csv       # one row per run — params + metrics"),
             ("out", "    └─ runs_manifest.json  # the recorded run plan")]
    cards.terminal_card(s["070_folder"], lines, title="A study on disk")

    # ------------------------------------------------------------------
    # 075 SHA-256 pin refusal — real CLI, real error, byte-exact restore
    # ------------------------------------------------------------------
    deck.write_bytes(deck_bytes + b"\n; tweaked after the fact\n")
    r = run_cli("study", "run", "quad_line_scan")
    assert r.returncode == 2, (r.returncode, r.stderr)
    sha_err = r.stderr.strip()
    assert "results would mix two different machines" in sha_err
    deck.write_bytes(deck_bytes)
    r2 = run_cli("study", "summarize", "quad_line_scan")
    assert r2.returncode == 0, r2.stderr
    lines = [("cmd", "python -m linac_gen study run quad_line_scan")]
    lines += [("out", ln) for ln in _wrap(sha_err, 70)]
    lines += [("gap", ""),
              ("out", "# restore the deck byte-for-byte — the pin matches again"),
              ("cmd", "python -m linac_gen study summarize quad_line_scan"),
              ("out", r2.stdout.strip())]
    cards.terminal_card(s["075_integrity_guards"], lines,
                        title="The lattice SHA-256 pin")
    print("[measure] SHA guard stderr:", sha_err[:120], "…")

    # ------------------------------------------------------------------
    # 078 the 25-run grid, 2 workers, recorded
    # ------------------------------------------------------------------
    pick_element("QUAD_002")
    pick_attr("gradient")
    add_btn.click()
    settle(3)
    set_cells(1, "-12", "-4", "5")
    assert st._run_count.text() == "25"
    st._name.setText("quad_grid_5x5")
    st._workers.setValue(2)
    settle(3)
    rec.start("078_grid25")
    rec.hold(exe_grab, 1.2)
    st._start.click()
    rec.wait_until(exe_grab, lambda: st._start.isEnabled(),
                   cap_s=300, stable_s=0.4)
    rec.hold(exe_grab, 3.0)
    rec.finish(scene_by_name("078_grid25"))
    settle(10)
    grid_dir = stroot / "quad_grid_5x5"

    import csv as _csv
    with open(grid_dir / "summary" / "summary.csv") as f:
        grows = list(_csv.DictReader(f))
    assert len(grows) == 25
    assert all(g["status"] == "ok" for g in grows)
    gsx = [(float(g["sigma_x"]), float(g["@6.gradient"]),
            float(g["@10.gradient"])) for g in grows]
    print(f"[measure] grid sigma_x min {min(gsx)[0]:.2f} mm at "
          f"({min(gsx)[1]:g},{min(gsx)[2]:g}); max {max(gsx)[0]:.2f} mm "
          f"at ({max(gsx)[1]:g},{max(gsx)[2]:g})")
    assert abs(min(gsx)[0] - 10.63) < 0.05 and abs(max(gsx)[0] - 13.17) < 0.05

    # ------------------------------------------------------------------
    # 080 Runs view MOTION — full columns + real header-click sorting
    # ------------------------------------------------------------------
    split.setSizes([440, 1460])
    settle(4)
    an_tab("runs")
    settle(8)
    table = st._analysis.runs.table
    table.resizeColumnsToContents()
    settle(4)
    hdr_labels = [table.horizontalHeaderItem(c).text()
                  for c in range(table.columnCount())]
    sx_col = hdr_labels.index("sigma_x")
    header = table.horizontalHeader()

    def an_grab() -> QImage:
        return region_img(an)

    def click_header(col: int) -> None:
        x = header.sectionViewportPosition(col) \
            + header.sectionSize(col) // 2
        QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier,
                         QPoint(x, header.height() // 2))
        settle(4)

    rec.start("080_runs_view")
    rec.hold(an_grab, 3.0)
    click_header(sx_col)                       # ascending
    rec.hold(an_grab, 3.0)
    top_asc = float(table.item(0, sx_col).text())
    click_header(sx_col)                       # descending
    rec.hold(an_grab, 3.0)
    top_desc = float(table.item(0, sx_col).text())
    assert top_asc < top_desc
    click_header(sx_col)                       # back to ascending
    rec.hold(an_grab, 2.0)
    rec.finish(scene_by_name("080_runs_view"))

    # ------------------------------------------------------------------
    # 082 double-click a row -> Results tab (the real slot chain)
    # ------------------------------------------------------------------
    def win_grab() -> QImage:
        return win.grab().toImage()

    item = table.item(0, 1)                    # tag cell, best run on top
    rect = table.visualItemRect(item)
    _on_results = (lambda: win._tabs.tabText(win._tabs.currentIndex())
                   .casefold() == "results")
    rec.start("082_open_in_results")
    rec.hold(win_grab, 1.5)
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, rect.center())
    settle(2)
    QTest.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier, rect.center())
    ok = rec.wait_until(win_grab, _on_results, cap_s=15, stable_s=0.3)
    if not ok:                     # same slot chain, signal-level
        table.itemDoubleClicked.emit(item)
        ok = rec.wait_until(win_grab, _on_results, cap_s=45,
                            stable_s=0.3)
    assert ok, "double-click did not reach the Results tab"
    rec.hold(win_grab, 3.5)
    rec.finish(scene_by_name("082_open_in_results"))
    go_tab("param study")
    settle(6)

    # ------------------------------------------------------------------
    # 084 the deliberately failing beam-energy scan
    # ------------------------------------------------------------------
    clear_rows()
    pick_element("Beam (input distribution)")
    pick_attr("energy")
    add_btn.click()
    settle(3)
    set_cells(0, "-1", "3", "5")
    st._name.setText("energy_scan")
    st._workers.setValue(1)
    settle(2)
    start_and_wait()
    edir = stroot / "energy_scan"
    stats = []
    for rd in sorted((edir / "runs").iterdir()):
        stj = json.loads((rd / "status.json").read_text())
        stats.append((stj["params"]["energy"], stj["status"],
                      stj.get("error") or ""))
    n_failed = sum(1 for _, stt, _ in stats if stt == "failed")
    print("[measure] energy scan:", [(e, stt) for e, stt, _ in stats])
    assert n_failed == 2, stats
    assert any("energy must be > 0 MeV" in err for _, _, err in stats)
    split.setSizes([700, 1220])
    an_tab("runs")
    settle(6)
    table = st._analysis.runs.table
    table.resizeColumnsToContents()
    settle(4)
    ecol = [table.horizontalHeaderItem(c).text()
            for c in range(table.columnCount())].index("error")
    table.scrollToItem(table.item(0, ecol))
    settle(4)
    win.grab().toImage().save(s["084_failed_runs"])

    # ------------------------------------------------------------------
    # 090 1D view MOTION on the grid — Y cycling + log toggles
    # ------------------------------------------------------------------
    split.setSizes([440, 1460])
    st._analysis.load_study(str(grid_dir))
    settle(6)
    an_tab("1D")
    settle(6)
    p1d = st._analysis.p1d
    rec.start("090_1d")
    p1d.y.setCurrentText("transmission")       # honestly empty (envelope)
    settle(4)
    rec.hold(an_grab, 3.0)
    for qty, hold_s in (("sigma_x", 3.5), ("emit_nx", 2.5),
                        ("ref_w_kin", 2.5), ("elapsed", 2.5)):
        p1d.y.setCurrentText(qty)
        settle(4)
        rec.hold(an_grab, hold_s)
    p1d.y.setCurrentText("sigma_x")
    settle(3)
    p1d.logx.setChecked(True)
    settle(3)
    rec.hold(an_grab, 1.8)
    p1d.logx.setChecked(False)
    p1d.logy.setChecked(True)
    settle(3)
    rec.hold(an_grab, 1.8)
    p1d.logy.setChecked(False)
    settle(3)
    rec.hold(an_grab, 1.2)
    rec.finish(scene_by_name("090_1d"))

    # ------------------------------------------------------------------
    # 094 group-by MOTION — five coloured series
    # ------------------------------------------------------------------
    rec.start("094_group_by")
    rec.hold(an_grab, 1.5)
    p1d.group.setCurrentText("@10.gradient")
    settle(5)
    rec.hold(an_grab, 4.0)
    p1d.logy.setChecked(True)
    settle(4)
    rec.hold(an_grab, 2.5)
    p1d.logy.setChecked(False)
    settle(3)
    rec.hold(an_grab, 1.5)
    rec.finish(scene_by_name("094_group_by"))

    # ------------------------------------------------------------------
    # 100 Map heatmap (full grid -> genuine viridis ImageItem + bar)
    # ------------------------------------------------------------------
    an_tab("map")
    settle(4)
    p2d = st._analysis.p2d
    p2d.z.setCurrentText("sigma_x")
    settle(8)
    assert p2d.x.currentText() == "@6.gradient"
    assert p2d.y.currentText() == "@10.gradient"
    crop_widget("100_map_heatmap", an)

    # ------------------------------------------------------------------
    # 101 Overlay MOTION — six runs, Draw, quantity cycling
    # ------------------------------------------------------------------
    an_tab("overlay")
    settle(6)
    ov = st._analysis.overlay
    draw_btn = next(b for b in ov.findChildren(QPushButton)
                    if b.text() == "Draw")
    rec.start("101_overlay")
    rec.hold(an_grab, 2.0)                     # first three pre-selected
    for i in range(min(6, ov.runs.count())):
        ov.runs.item(i).setSelected(True)
    settle(3)
    rec.hold(an_grab, 1.0)
    draw_btn.click()
    settle(5)
    rec.hold(an_grab, 3.0)
    for qty in ("beta_x", "emit_z"):
        ov.qty.setCurrentText(qty)
        settle(5)
        rec.hold(an_grab, 2.8)
    ov.qty.setCurrentText("sigma_x")
    settle(4)
    rec.hold(an_grab, 1.5)
    rec.finish(scene_by_name("101_overlay"))

    # ------------------------------------------------------------------
    # 102 LHS survey -> Map scatter fallback
    # ------------------------------------------------------------------
    clear_rows()
    pick_element("QUAD_001")
    pick_attr("gradient")
    add_btn.click()
    settle(2)
    set_cells(0, "4", "12", "5")
    pick_element("QUAD_002")
    pick_attr("gradient")
    add_btn.click()
    settle(2)
    set_cells(1, "-12", "-4", "5")
    st._strategy.setCurrentText("lhs")
    st._name.setText("lhs_survey")
    st._workers.setValue(2)
    settle(3)
    assert st._run_count.text() == "32"
    start_and_wait()
    an_tab("map")
    settle(4)
    p2d.z.setCurrentText("sigma_x")
    settle(8)
    from linac_gen_gui.interphase.panels.study_plots import detect_grid
    m = st._analysis.model
    assert detect_grid(m.ok_records(), "@6.gradient", "@10.gradient",
                       "sigma_x", m.column) is None, \
        "LHS points must NOT form a grid (scatter fallback expected)"
    assert len(m.ok_records()) == 32
    crop_widget("102_map_scatter", an)

    # ------------------------------------------------------------------
    # 104 MP mode: 5 gradients x 3 seeds, Stop -> resume, live ETA
    # ------------------------------------------------------------------
    st._ptable.removeRow(1)
    st._refresh_run_count()
    set_cells(0, "6", "12", "4")     # 6, 8, 10, 12 T/m
    st._strategy.setCurrentText("grid")
    st._repeats.setValue(3)
    st._mode.setCurrentText("mp")
    st._name.setText("mp_repeats")
    st._workers.setValue(2)
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=0.0,
        n_particles=250000))       # ~10 s per run — stoppable on camera
    settle(3)
    assert st._run_count.text() == "12", st._run_count.text()
    split.setSizes([700, 1220])
    an_tab("runs")
    settle(3)
    rec.start("104_mp_stop_resume")
    rec.hold(win_grab, 1.5)
    st._start.click()
    ok = rec.wait_until(win_grab,
                        lambda: "ETA" in st._status.text(),
                        cap_s=150)
    assert ok, "ETA never appeared in the status line"
    rec.hold(win_grab, 4.0)
    rec.wait_until(win_grab, lambda: st._bar.value() >= 4, cap_s=150)
    st._stop.click()
    settle(2)
    ok = rec.wait_until(
        win_grab,
        lambda: st._start.isEnabled()
        and st._status.text().startswith("stopped"),
        cap_s=240, stable_s=0.3)
    assert ok, f"stop did not drain cleanly: {st._status.text()!r}"
    done_at_stop = st._bar.value()
    assert 4 <= done_at_stop < 12, done_at_stop
    rec.hold(win_grab, 6.0)
    st._start.click()                          # resume: bar pre-filled
    settle(3)
    ok = rec.wait_until(
        win_grab,
        lambda: st._start.isEnabled()
        and st._status.text().startswith("complete"),
        cap_s=480, stable_s=0.3)
    assert ok, f"resume did not complete: {st._status.text()!r}"
    rec.hold(win_grab, 5.0)
    rec.finish(scene_by_name("104_mp_stop_resume"))
    print(f"[measure] mp stop/resume: stopped at {done_at_stop}/12, "
          f"resumed to completion")

    mp_dir = stroot / "mp_repeats"
    import collections
    import numpy as np
    by_g = collections.defaultdict(list)
    for rd in sorted((mp_dir / "runs").iterdir()):
        stj = json.loads((rd / "status.json").read_text())
        assert stj["status"] == "ok"
        by_g[round(stj["params"]["@6.gradient"], 3)].append(
            stj["metrics"]["sigma_x"])
    stds = {g: float(np.std(v)) for g, v in by_g.items()}
    means = {g: float(np.mean(v)) for g, v in by_g.items()}
    print("[measure] mp seed scatter (std per gradient, mm):",
          {g: round(v, 4) for g, v in sorted(stds.items())})
    print("[measure] mp means (mm):",
          {g: round(v, 3) for g, v in sorted(means.items())})
    assert 0.002 < max(stds.values()) < 0.04, \
        "narration says 'about a hundredth of a millimetre'"
    trend = max(means.values()) - min(means.values())
    assert trend > 3 * max(stds.values()), \
        f"narration says noise sits far below the trend ({trend=})"

    # ------------------------------------------------------------------
    # 106 error bars still (1D on the mp study)
    # ------------------------------------------------------------------
    split.setSizes([440, 1460])
    an_tab("1D")
    settle(4)
    p1d.y.setCurrentText("sigma_x")
    p1d.group.setCurrentText("(none)")
    settle(8)
    crop_widget("106_mp_error_bars", an)

    # ------------------------------------------------------------------
    # 107 observables study via the CLI twin (real output)
    # ------------------------------------------------------------------
    rr = run_cli("study", "run", "obs_demo.json", "--parallel", "2")
    assert rr.returncode == 0, rr.stderr
    out_lines = [ln for ln in rr.stdout.strip().split("\n") if ln]
    lines = [("cmd",
              "python -m linac_gen study run obs_demo.json --parallel 2")]
    for ln in out_lines:
        lines += [("out", w) for w in _wrap(ln, 70)]
    cards.terminal_card(s["107_observables"], lines,
                        title="The headless twin — run")
    obs_dir = stroot / "obs_demo"
    with open(obs_dir / "summary" / "summary.csv") as f:
        orows = list(_csv.DictReader(f))
    assert len(orows) == 5 and all(r["status"] == "ok" for r in orows)
    assert all(r["sx_mid"] for r in orows), "observable column must be filled"
    print("[measure] obs sx_mid:",
          [(r["@6.gradient"], round(float(r["sx_mid"]), 2))
           for r in orows])

    # 112 plan card (real output)
    rp = run_cli("study", "plan", "obs_demo.json")
    assert rp.returncode == 0, rp.stderr
    lines = [("cmd", "python -m linac_gen study plan obs_demo.json")]
    for ln in rp.stdout.strip().split("\n"):
        lines += [("out", w) for w in _wrap(ln, 70)]
    cards.terminal_card(s["112_cli_plan"], lines,
                        title="The headless twin — plan")

    # 114 retry-failed + summarize card (real output)
    rf = run_cli("study", "run", "energy_scan", "--retry-failed")
    assert rf.returncode == 0, rf.stderr
    rs = run_cli("study", "summarize", "energy_scan")
    assert rs.returncode == 0, rs.stderr
    lines = [("cmd",
              "python -m linac_gen study run energy_scan --retry-failed")]
    for ln in [x for x in rf.stdout.strip().split("\n") if x]:
        lines += [("out", w) for w in _wrap(ln, 70)]
    lines += [("gap", ""),
              ("cmd", "python -m linac_gen study summarize energy_scan")]
    for ln in [x for x in rs.stdout.strip().split("\n") if x]:
        lines += [("out", w) for w in _wrap(ln, 70)]
    cards.terminal_card(s["114_cli_verbs"], lines,
                        title="The headless twin — resume · retry · summarize")
    assert "2 run(s) failed" in rf.stdout

    # ------------------------------------------------------------------
    # 108 open the CLI-made folder in the tab (real load path body)
    # ------------------------------------------------------------------
    split.setSizes([700, 1220])
    settle(3)
    target: dict = {"w": exe_grp}

    def moving_grab() -> QImage:
        return region_img(target["w"])

    rec.start("108_observables_gui")
    st._analysis.load_study(str(obs_dir))      # _on_open_study body,
    st._study_dir = str(obs_dir)               # minus the modal picker
    st._remember()
    st._status.setText(f"loaded study: {obs_dir}")
    settle(5)
    rec.hold(moving_grab, 2.5)                 # status: loaded study …
    target["w"] = an
    an_tab("runs")
    settle(4)
    table = st._analysis.runs.table
    hdr = [table.horizontalHeaderItem(c).text()
           for c in range(table.columnCount())]
    assert "sx_mid" in hdr, hdr
    table.resizeColumnsToContents()
    table.scrollToItem(table.item(0, hdr.index("sx_mid")))
    settle(3)
    rec.hold(moving_grab, 3.0)
    an_tab("1D")
    settle(3)
    p1d.y.setCurrentText("sx_mid")
    settle(5)
    rec.hold(moving_grab, 3.5)
    rec.finish(scene_by_name("108_observables_gui"))

    # ------------------------------------------------------------------
    # 110 session persistence — a brand-new StudyTab restores the study
    # ------------------------------------------------------------------
    from linac_gen_gui.interphase.tabs.study_tab import StudyTab
    tab2 = StudyTab(win.state)
    tab2.resize(1500, 950)
    tab2.show()
    settle(15)
    assert tab2._status.text().startswith("loaded last study:"), \
        tab2._status.text()
    t2img = tab2.grab().toImage()
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    canvas.fill(QColor("#0b1220"))
    p = QPainter(canvas)
    p.drawImage((1920 - t2img.width()) // 2,
                (1080 - t2img.height()) // 2, t2img)
    p.end()
    canvas.save(s["110_open"])
    tab2.hide()
    tab2.deleteLater()
    settle(4)

    # ------------------------------------------------------------------
    # 116 manual tour — QtWebEngine in a SEPARATE process
    # ------------------------------------------------------------------
    tour_frames = WORK / "frames" / "116_manual_tour"
    if tour_frames.exists():
        shutil.rmtree(tour_frames)
    helper = WORK / "mantour_run.py"
    helper.write_text(r'''
import json, os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6, os.path as op
    os.environ["QT_PLUGIN_PATH"] = op.join(
        op.dirname(PyQt6.__file__), "Qt6", "plugins")
from PyQt6 import QtWebEngineWidgets                    # noqa: F401
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QUrl, QEventLoop

frames_dir, page1, page2 = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(frames_dir, exist_ok=True)
app = QApplication(["mantour"])
v = QWebEngineView()
v.resize(1920, 1080)
v.show()

def load(path):
    loop = QEventLoop()
    v.loadFinished.connect(loop.quit)
    v.load(QUrl.fromLocalFile(path))
    loop.exec()
    try:
        v.loadFinished.disconnect(loop.quit)
    except TypeError:
        pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        app.processEvents()
        time.sleep(0.01)

def page_height():
    res = {}
    loop = QEventLoop()
    v.page().runJavaScript(
        "document.body.scrollHeight",
        lambda h: (res.setdefault("h", h), loop.quit()))
    loop.exec()
    return int(res.get("h") or 4000)

n = 0
t0 = time.monotonic()

def snap(k=1):
    global n
    for _ in range(k):
        app.processEvents()
        v.grab().toImage().save(
            os.path.join(frames_dir, "%05d.png" % n))
        n += 1

def scroll_to(y):
    v.page().runJavaScript("window.scrollTo(0,%d);" % y)
    t = time.monotonic()
    while time.monotonic() - t < 0.08:
        app.processEvents()
        time.sleep(0.005)

for page in (page1, page2):
    load(page)
    scroll_to(0)
    snap(14)
    h = page_height()
    span = max(h - 1080, 0)
    step = max(60, span // 110)
    y = 0
    while y < span:
        y = min(y + step, span)
        scroll_to(y)
        snap(1)
    snap(12)

wall = max(time.monotonic() - t0, 1e-6)
print(json.dumps({"n": n, "fps": max(n / wall, 1.0)}))
''')
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)           # helper sets plain offscreen
    tour = subprocess.run(
        [sys.executable, str(helper), str(tour_frames),
         str(ROOT / "site/10_gui/06d_study_tab.html"),
         str(ROOT / "site/06_running/12_cli_study.html")],
        capture_output=True, text=True, timeout=420, env=env)
    assert tour.returncode == 0, tour.stderr[-800:]
    info = json.loads(tour.stdout.strip().split("\n")[-1])
    assert info["n"] > 40, info
    sc = scene_by_name("116_manual_tour")
    sc.frames_dir = str(tour_frames)
    sc.fps = float(info["fps"])
    print(f"[clip] 116_manual_tour: {info['n']} frames at "
          f"{info['fps']:.2f} fps (subprocess)")

    # ------------------------------------------------------------------
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
    os._exit(0)                    # QThread workers ran; skip Qt teardown


if __name__ == "__main__":
    main()
