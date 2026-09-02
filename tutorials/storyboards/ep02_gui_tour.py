"""Episode 2 — The full GUI tour (chrome, menus, dialogs, every tab).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep02_gui_tour.py
Output: tutorials/rendered/ep02_gui_tour.mp4 (+ .srt captions)

The tour machine is examples/dtl_section.dat (public): two solenoids,
then eight QUAD-GAP cells accelerating protons from three to about
seven M e V.  This cut covers the window chrome end to end: titlebar,
all four menus, the New Project wizard, projects and recents, a real
multi-particle run with Stop, the status bar anatomy, backtracking,
auto-saves and exporters, the Python console, both matrix dialogs, the
quick Parameter Scan, the update checker, F1 context help, the manual
site, font scaling and every keyboard shortcut.

All demos run on a scratch COPY of the deck under a sandboxed
QSettings dir, so nothing in examples/ or the user's settings is
touched.  Every modal is either constructed non-modally or its exec is
patched in a try/finally (offscreen modals hang forever).
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
# QtWebEngine (manual scenes) — offscreen recipe proven in this pipeline.
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep02_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep02_gui_tour.mp4"

# Measured on this machine before narration was written (see the build
# log): multi-particle at 120000 particles / 20 mA space charge paces
# the run so Stop can be demonstrated; 0 mA runs are near-instant
# (2000 particles with SC completed in 1.5 s — too fast to interrupt;
# 40000 gave only ~2 s of RUNNING on camera in the first cut).
N_MP = 120000
MP_CURRENT_MA = 20.0
STOP_AT_PCT = 40

SCENES = [
    Scene("010_title",
          "Welcome back to HELIX — the Hybrid Envelope-multiparticle "
          "LInac eXplorer. In episode one we installed the code and "
          "ran a first beam. This time we tour the whole cockpit — "
          "every menu, every dialog, the status bar, the shortcuts, "
          "and the manual behind them. Our "
          "machine is a drift tube linac section taking protons from "
          "3 to about 7 M e V.",
          min_s=7.0),
    Scene("020_overview",
          "Here is the interface with that machine loaded. A thin "
          "titlebar at the very top, the toolbar with the menus and "
          "Run buttons under it, the tabs, and a status bar that "
          "always shows the element count and total length — and, "
          "after a run, the final beam size and the losses. Let us "
          "take these strips one at a time, from the top."),
    Scene("022_titlebar",
          "The titlebar, enlarged in two halves. The brand, then the "
          "loaded lattice with its element count — dtl section dot "
          "dat, 38 elements. On the right, the Python version and "
          "platform, and a clock. Watch the seconds tick — this strip "
          "is live, and the label follows every lattice you open."),
    Scene("024_file_menu",
          "The File menu, enlarged on the right. New Project starts "
          "the guided wizard — also the toolbar's New button, or "
          "control N. Open, Save and Save As manage the lattice file "
          "itself. The project group bundles lattice, beam and solver "
          "settings into one file, with Open Recent remembering them. "
          "Set Calculation Directory chooses where run dumps land, "
          "then two exporters — TraceWin and open P M D. And Exit "
          "warns you first if anything is unsaved.",
          min_s=8.0),
    Scene("025_open_formats",
          "Open Lattice speaks more than TraceWin — the filter lists "
          "dot dat plus M A D X, M A D 8 and Elegant extensions, each "
          "routed to its own importer. Saving always writes TraceWin "
          "style dot dat, so HELIX never overwrites a M A D X source "
          "file."),
    Scene("026_project_wizard",
          "The New Project wizard. A name and a location — a folder "
          "of that name is created and runs land inside it, so the "
          "whole project moves as one unit. Three starting points: a "
          "blank lattice with one editable drift, importing an "
          "existing dot dat — copied into the folder — or a bundled "
          "example: the FODO cell, the solenoid channel, or this very "
          "D T L section.",
          min_s=7.0),
    Scene("028_projects_recents",
          "Watch the status bar as we save this session as a project "
          "— there is the confirmation. Every action gets visible "
          "feedback down there. File, Open Recent now lists the "
          "project, with Clear Recent Projects underneath. Projects "
          "also power session restore — the next launch reopens what "
          "you were working on, beam included.",
          min_s=7.0),
    Scene("030_lattice",
          "Now the tabs. The Lattice tab is home. On the left the "
          "palette, every element type HELIX knows, ready to drag in. "
          "Next to it, the outline groups the loaded elements by "
          "type. The centre shows the machine two ways — a layout "
          "strip you can scrub, and the sequence view, the lattice "
          "file itself, line by line."),
    Scene("040_inspector",
          "Click any element and the inspector opens on the right — "
          "here, one of the R F gaps. Every parameter is editable: "
          "the R F phase, the frequency, the error knobs, the "
          "alignment offsets and tilts. Change a number, press enter, "
          "and the next run uses it. Undo and redo cover every edit. "
          "And remember this for later: with an element selected, F 1 "
          "opens its manual chapter.",
          min_s=6.0),
    Scene("050_beam",
          "The Beam tab defines what you inject. Species, kinetic "
          "energy, bunch frequency, and current. Below that, the "
          "Twiss parameters and emittances that set the beam's size "
          "and divergence in each plane. If you have a particle file, "
          "load it here instead and HELIX tracks exactly those "
          "particles."),
    Scene("060_numerics",
          "The Numerics tab holds the solver settings — integration "
          "steps per metre, the space charge model and its grid, and "
          "the field map handling. The defaults are sensible. When a "
          "result looks suspicious, come here and double the steps to "
          "check convergence."),
    Scene("070_matching",
          "The Matching tab turns knobs for you. Choose which element "
          "parameters are free, set targets — beam sizes, Twiss "
          "values, or a matched periodic cell — and HELIX adjusts the "
          "knobs to hit them. Matching gets its own episode later."),
    Scene("080_studies",
          "Four tabs run campaigns rather than single runs. Param "
          "Study scans parameters over ranges — the big sibling of "
          "the quick Parameter Scan dialog we will meet in the Tools "
          "menu. Error Study adds random misalignments to estimate "
          "tolerances. Failure Study switches elements off to see "
          "what survives. And Surrogates trains fast machine-learned "
          "stand-ins."),
    Scene("090_run",
          "Time to run for real: 20 milliamps of current, 120000 "
          "macro particles, space charge on. The moment Run Multi particle "
          "is clicked, the pill turns RUNNING, Stop lights up red, "
          "and the s label becomes a live readout with a percent "
          "counter while the slider sweeps the machine. At the end, "
          "the status bar reports the run complete and names the "
          "file it auto-saved.",
          min_s=9.0),
    Scene("092_stop_and_scrub",
          "The same run again — but this time we pull the brake at "
          "about 40 percent. Stop cancels at the next element "
          "boundary, and the status bar confirms the run was stopped "
          "by you. No half-written results — the previous results "
          "stay. And between runs the slider is yours: scrub it, and "
          "the s position follows in the toolbar and status bar "
          "together.",
          min_s=7.0),
    Scene("094_statusbar",
          "The status bar up close. Left to right: the state pill, "
          "the lattice summary — 38 elements, 860 millimetres — your "
          "s position, and after our run the final beam size and the "
          "loss, with the version on the far right. Loss above 1 "
          "percent renders amber, and our rough hand-set optics lost "
          "about a third. Now watch — we edit a quadrupole length and "
          "the unsaved pill lights until the edit is undone. Then "
          "control S: the saved confirmation appears and clears "
          "itself after about six seconds. One honest caveat — the W "
          "segment is reserved and does not yet track live energy.",
          min_s=9.0),
    Scene("096_simulate_menu",
          "The Simulate menu. The two run entries mirror the toolbar "
          "buttons, with control R and control shift R as shortcuts. "
          "Backtrack Distribution reconstructs the beam upstream from "
          "an exit distribution — we run it next. And Multibunch, "
          "Pulse Study is the opt-in bunch-train mode for beam "
          "loading and long-pulse effects."),
    Scene("098_backtrack",
          "The Backtrack dialog. Pick the source — the final beam of "
          "the run we just did, or a measured exit-plane dot d s t "
          "file — then the element range and the field map inverse. "
          "R K 4 is the exact default. A tick undoes space charge "
          "with the Numerics settings, and the output box writes the "
          "reconstructed entrance distribution to a file.",
          min_s=7.0),
    Scene("099_backtrack_run",
          "And off it goes — the status bar announces the backward "
          "walk, exit of element 37 to the entrance of element 0, and "
          "the s cursor returns to zero as the survivors are tracked "
          "back. Lost particles cannot be resurrected — the caveat "
          "box spells out that the reconstruction covers surviving "
          "particles only.",
          min_s=7.0),
    Scene("100_results",
          "Every run lands in the Results tab. Here we fire the "
          "envelope solver — done the moment the button is released, "
          "with the completion message naming the sigma values and "
          "the auto-saved file. Each tile is one quantity — sizes, "
          "emittances, energy, losses, Twiss, phase advance. Tiles "
          "with data show a preview curve; clicking one opens the "
          "full plot window.",
          min_s=7.0),
    Scene("110_energy",
          "The Energy plot for our D T L. Each step of the staircase "
          "is one R F gap kicking the protons — eight gaps, 3 M e V "
          "in, about 7 M e V out. Every plot popup shares the same "
          "controls: zoom, pan, toggle the lattice strip, and control "
          "S to export the data or the image.",
          min_s=6.0),
    Scene("112_autosave_exports",
          "Runs never vanish. Every completed run wrote two files "
          "into the calculation directory without being asked — a "
          "HELIX-native H D F 5 file and an open P M D companion, "
          "timestamped. This listing is from the session you are "
          "watching, together with the files from File, Export "
          "TraceWin output — partran 1 dot out and an envelope text "
          "file for TraceWin-side tooling.",
          min_s=7.0),
    Scene("130_tools_menu",
          "The Tools menu — five power tools that work from any tab. "
          "The Assistant is the optional A I copilot with a full "
          "episode of its own later. The other four we open right "
          "now."),
    Scene("132_console",
          "The Python console, live against this session. l g is the "
          "linac gen package, state is the app state, and lattice, "
          "beam config and results are always in scope. The element "
          "count — 38. The final beam size of the last run. The "
          "injection energy — 3 M e V. Expressions print their value; "
          "statements just run.",
          min_s=8.0),
    Scene("134_matrices",
          "The two matrix viewers. Show Transfer Matrix multiplies "
          "the per-element linear maps over any range — pure linear "
          "transport — and reports the energy gain, 3 to about 7 M e "
          "V across the line. Show Sigma Matrix displays the measured "
          "second-moment matrix of the last run at every recorded "
          "step. One needs a lattice and beam, the other needs "
          "results.",
          min_s=7.0),
    Scene("136_param_scan",
          "Tools, Parameter Scan — the quick single-knob version. One "
          "element, one parameter, a range, envelope or multi "
          "particle, and Run scan plots the quantity you chose. It is "
          "non-modal, so the main window stays live. For "
          "multi-parameter campaigns with saved results, use the "
          "Param Study tab instead."),
    Scene("140_help_menu",
          "The Help menu. Documentation opens the full manual — built "
          "locally, so it works offline. Check for Updates asks "
          "GitHub for a newer release on demand, and the startup "
          "check is a toggle you can switch off. Nothing here ever "
          "updates the code without your explicit confirmation."),
    Scene("142_about",
          "About HELIX — the one-screen summary of this build. The "
          "version at the top. Three engines: matrix, envelope, multi "
          "particle. Space charge analytical or particle in cell. "
          "Backends from plain C P U through CUDA to Apple Silicon, "
          "and TraceWin plus H D F 5 for I O."),
    Scene("144_manual_tour",
          "Help, Documentation lands here — the HELIX manual, a local "
          "website with working search and a light-dark toggle in the "
          "header. The G U I section mirrors this tour, one chapter "
          "per tab plus overview, workflows and updates. Everything "
          "we cover on camera is written down here.",
          min_s=8.0),
    Scene("146_workflows_page",
          "One chapter is worth pinning: G U I workflows, the cheat "
          "sheet that starts from the task. I want to load a TraceWin "
          "deck and run it. I want to match Twiss to a target. An "
          "alignment tolerance study, the G P U, backtracking — each "
          "one a numbered click path. When you forget where a feature "
          "lives, start here, not in the menus.",
          min_s=8.0),
    Scene("148_f1_help",
          "Context help closes the loop. Our R F gap is still "
          "selected — press F 1, and the status bar confirms HELIX "
          "opened the gap's manual chapter in the browser. This is "
          "that page: the gap model, its parameters, its conventions. "
          "With nothing selected, F 1 just asks you to pick an "
          "element first.",
          min_s=7.0),
    Scene("150_updates",
          "Updates — driven by hand here so you can see the shape of "
          "it. A few seconds after launch, HELIX quietly asks GitHub "
          "whether a newer release exists; silence means you are "
          "current, or offline. When one exists, this pill appears in "
          "the status bar and the Help entry turns bold with the tag. "
          "Nothing happens until you click and confirm.",
          min_s=7.0),
    Scene("152_updates_page",
          "The updates chapter spells out the contract. A clean clone "
          "of the official repository fast-forwards in place with a "
          "progress window. A copy with local changes just gets the "
          "release page — your edits are never touched. Dependencies "
          "are never installed automatically, and developer checkouts "
          "are left alone entirely.",
          min_s=7.0),
    Scene("155_font_scaling",
          "One control for your eyes: the Font spinbox, 9 to 22 "
          "points. Watch the whole interface rescale together — "
          "titlebar, toolbar, tabs and status bar. Control plus and "
          "control minus do the same from the keyboard, control zero "
          "resets, and the size is remembered for the next launch.",
          min_s=7.0),
    Scene("158_shortcuts",
          "The full shortcut list, straight from the source. Control "
          "N, new project. Control O, open lattice. Control S, save. "
          "Control R runs the envelope, control shift R runs multi "
          "particle. Control plus, minus and zero drive the font "
          "size. And F 1 opens the manual chapter for the selected "
          "element. They work from every tab — the toolbar owns "
          "them.",
          min_s=7.0),
    Scene("120_outro",
          "That is the whole cockpit — every menu, the wizard, "
          "projects, a run stopped mid-flight, backtracking, the "
          "console, the matrix viewers, auto-saves, updates, and the "
          "manual. From here the series goes deep, one tab per "
          "episode — the Lattice tab first, then Beam, Numerics, "
          "Matching, Results and the study tabs, with space charge "
          "and the assistant further on. See you there.",
          min_s=7.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def capture_visuals() -> None:
    import shutil
    import time as _time

    from PyQt6.QtCore import QEventLoop, QPoint, Qt, QUrl
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import (QApplication, QDialog, QLineEdit,
                                 QMessageBox, QPlainTextEdit, QPushButton,
                                 QFileDialog)

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    # QtWebEngine must be imported before the QApplication exists.
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    app = QApplication.instance() or QApplication(["ep02"])

    cards.title_card(s["010_title"], "The Full GUI Tour",
                     "Episode 2 — chrome, menus, dialogs, every tab")
    cards.outro_card(s["120_outro"], [
        "Next — the Lattice tab in depth",
        "Then — Beam, Numerics, Matching, Results, the studies",
        "Later — space charge physics, the voice assistant",
    ])

    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file,
                                              _settings,
                                              _SETTINGS_CALC_DIR)
    win = InterphaseWindow()
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    def pump(seconds: float) -> None:
        t0 = _time.monotonic()
        while _time.monotonic() - t0 < seconds:
            settle(6)
            _time.sleep(0.005)

    # ---- scratch sandbox --------------------------------------------
    scratch = Path(tempfile.mkdtemp(prefix="helix_ep02_"))
    projects_dir = scratch / "HELIX_projects"
    projects_dir.mkdir()
    runs_dir = scratch / "runs"
    _settings().setValue(_SETTINGS_CALC_DIR, str(runs_dir))

    deck = scratch / "dtl_section.dat"
    shutil.copy(ROOT / "examples" / "dtl_section.dat", deck)
    lattice, _meta = _parse_lattice_file(str(deck))
    win.state.set_lattice(lattice, str(deck))

    from linac_gen.core.config import BeamConfig
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, alpha_x=2.0, beta_x=1.0,
                     alpha_y=2.0, beta_y=1.0, n_particles=N_MP)
    win.beam_tab.set_beam_config(cfg)          # tab shows the real beam
    win.state.set_beam_config(cfg)
    settle(10)

    rec = Recorder(WORK / "frames", settle)
    DARK = QColor("#0b1220")

    # ---- composite helpers ------------------------------------------
    def canvas() -> QImage:
        img = QImage(1920, 1080, QImage.Format.Format_RGB32)
        img.fill(DARK)
        return img

    SMOOTH = Qt.TransformationMode.SmoothTransformation

    def scaled_w(img: QImage, w: int) -> QImage:
        return img.scaledToWidth(w, SMOOTH)

    def strip_rows(full: QImage, y0: int, h: int) -> tuple[QImage, QImage]:
        """A full-width strip split in half, each half enlarged 2x."""
        left = full.copy(0, y0, 960, h)
        right = full.copy(960, y0, 960, h)
        return scaled_w(left, 1920), scaled_w(right, 1920)

    def titlebar_frame() -> QImage:
        full = win.grab().toImage()
        th = win._titlebar.height() + 2
        r1, r2 = strip_rows(full, 0, th)
        c = canvas()
        p = QPainter(c)
        p.drawImage(0, 40, r1)
        p.drawImage(0, 60 + r1.height(), r2)
        wimg = scaled_w(full, 1500)
        p.drawImage((1920 - 1500) // 2, 90 + 2 * r1.height(), wimg)
        p.end()
        return c

    def statusbar_frame() -> QImage:
        full = win.grab().toImage()
        sy = win._statusbar.mapTo(win, QPoint(0, 0)).y() - 1
        sh = win._statusbar.height() + 2
        r1, r2 = strip_rows(full, sy, sh)
        c = canvas()
        p = QPainter(c)
        wimg = scaled_w(full, 1500)
        p.drawImage((1920 - 1500) // 2, 4, wimg)
        p.drawImage(0, 866, r1)
        p.drawImage(0, 886 + r1.height(), r2)
        p.end()
        return c

    def chrome_frame(extra: QImage | None = None) -> QImage:
        """Window + zoomed run-controls strip + zoomed status strip."""
        full = win.grab().toImage()
        c = canvas()
        p = QPainter(c)
        wimg = scaled_w(full, 1500)
        p.drawImage((1920 - 1500) // 2, 4, wimg)
        # run controls: from the Run Envelope button to the slider end
        tb = win._toolbar
        ty = tb.mapTo(win, QPoint(0, 0)).y()
        x0 = tb._run_env_btn.mapTo(win, QPoint(0, 0)).x() - 10
        sl_end = (tb._slider.mapTo(win, QPoint(0, 0)).x()
                  + tb._slider.width() + 14)
        band = full.copy(x0, ty, sl_end - x0, tb.height())
        band = scaled_w(band, min(1900, band.width() * 2))
        p.drawImage((1920 - band.width()) // 2, 856, band)
        sy = win._statusbar.mapTo(win, QPoint(0, 0)).y() - 1
        sh = win._statusbar.height() + 2
        srow = scaled_w(full.copy(0, sy, 960, sh), 1920)
        p.drawImage(0, 866 + band.height(), srow)
        if extra is not None:
            ex = scaled_w(extra, min(760, extra.width()))
            p.fillRect(1920 - ex.width() - 46, 44, ex.width() + 12,
                       ex.height() + 12, QColor("#2b3b55"))
            p.drawImage(1920 - ex.width() - 40, 50, ex)
        p.end()
        return c

    def menu_still(key: str, btn_label: str) -> None:
        btn = next(b for b in win._toolbar.findChildren(QPushButton)
                   if b.menu() is not None and b.text() == btn_label)
        menu = btn.menu()
        menu.popup(btn.mapToGlobal(btn.rect().bottomLeft()))
        settle(10)
        m = menu.grab().toImage()
        wimg = win.grab().toImage()
        c = canvas()
        p = QPainter(c)
        p.drawImage(0, 0, wimg)
        p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, 110))
        gp = menu.mapToGlobal(QPoint(0, 0))
        wp = win.mapToGlobal(QPoint(0, 0))
        p.drawImage(gp.x() - wp.x(), gp.y() - wp.y(), m)
        big = scaled_w(m, min(m.width() * 2, 900))
        bx = 1920 - big.width() - 120
        by = max(60, (1080 - big.height()) // 2 - 60)
        p.fillRect(bx - 8, by - 8, big.width() + 16, big.height() + 16,
                   QColor("#2b3b55"))
        p.drawImage(bx, by, big)
        p.end()
        c.save(s[key])
        menu.hide()
        settle(4)

    def dialog_over_window(dlg, scale: float = 1.0) -> QImage:
        d = dlg.grab().toImage()
        if scale != 1.0:
            d = scaled_w(d, int(d.width() * scale))
        wimg = win.grab().toImage()
        c = canvas()
        p = QPainter(c)
        p.drawImage(0, 0, wimg)
        p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, 130))
        dx = (1920 - d.width()) // 2
        dy = max(24, (1080 - d.height()) // 2)
        p.fillRect(dx - 6, dy - 6, d.width() + 12, d.height() + 12,
                   QColor("#2b3b55"))
        p.drawImage(dx, dy, d)
        p.end()
        return c

    win_frame = lambda: win.grab().toImage()   # noqa: E731

    # =================================================================
    # 020 overview — Lattice tab, full window
    go_tab("lattice")
    settle(8)
    win._statusbar._clear_message()
    settle(4)
    win.grab().save(s["020_overview"])

    # 022 titlebar — clock ticks on camera (~7 s of wall time)
    rec.start("022_titlebar")
    rec.hold(titlebar_frame, 7.0)
    rec.finish(scene_by_name("022_titlebar"))

    # 024 File menu
    menu_still("024_file_menu", "File")

    # 025 Open Lattice file dialog (same filter string as app._open_lattice)
    dlg = QFileDialog(win, "Open Lattice", str(ROOT / "examples"),
                      "Lattice files (*.dat *.madx *.seq *.lat *.flat "
                      "*.lte);;TraceWin (*.dat);;MAD-X (*.madx *.seq);;"
                      "MAD8 (*.lat *.flat);;Elegant (*.lte);;All Files (*)")
    dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dlg.resize(1100, 660)
    dlg.show()
    settle(20)
    dialog_over_window(dlg, scale=1.25).save(s["025_open_formats"])
    dlg.hide()
    settle(4)

    # 026 New Project wizard — radios toggled on camera
    from linac_gen_gui.interphase.dialogs.new_project import NewProjectDialog
    ex_dir = ROOT / "examples"
    examples = [(lbl, str(ex_dir / fn)) for lbl, fn in
                [("FODO cell", "fodo_cell.dat"),
                 ("Solenoid channel", "solenoid_channel.dat"),
                 ("DTL section", "dtl_section.dat")]
                if (ex_dir / fn).is_file()]
    npd = NewProjectDialog(win, start_dir=str(projects_dir),
                           examples=examples)
    npd.show()
    settle(12)
    wiz_frame = lambda: dialog_over_window(npd, scale=1.6)   # noqa: E731
    rec.start("026_project_wizard")
    rec.hold(wiz_frame, 2.5)
    npd._rb_import.setChecked(True)
    rec.hold(wiz_frame, 2.5)
    npd._rb_example.setChecked(True)
    rec.hold(wiz_frame, 3.0)
    npd._rb_blank.setChecked(True)
    rec.hold(wiz_frame, 1.5)
    rec.finish(scene_by_name("026_project_wizard"))
    npd.hide()
    settle(4)

    # 028 project save + recents — real writer, real submenu
    def rec_frame_028() -> QImage:
        rm = win._toolbar._recent_menu
        extra = rm.grab().toImage() if rm.isVisible() else None
        full = win.grab().toImage()
        c = canvas()
        p = QPainter(c)
        wimg = scaled_w(full, 1500)
        p.drawImage((1920 - 1500) // 2, 4, wimg)
        sy = win._statusbar.mapTo(win, QPoint(0, 0)).y() - 1
        sh = win._statusbar.height() + 2
        r1, r2 = strip_rows(full, sy, sh)
        p.drawImage(0, 866, r1)
        p.drawImage(0, 886 + r1.height(), r2)
        if extra is not None:
            big = scaled_w(extra, min(extra.width() * 2, 820))
            p.fillRect(1920 - big.width() - 66, 60, big.width() + 16,
                       big.height() + 16, QColor("#2b3b55"))
            p.drawImage(1920 - big.width() - 58, 68, big)
        p.end()
        return c

    warn_seen: list[str] = []
    _orig_warn = QMessageBox.warning
    QMessageBox.warning = (lambda *a, **k:
                           (warn_seen.append(str(a[2:3])),
                            QMessageBox.StandardButton.Save)[1])
    try:
        rec.start("028_projects_recents")
        rec.hold(rec_frame_028, 1.5)
        ok = win._write_project_file(str(projects_dir / "dtl_tour.lgproj"))
        assert ok, "project write failed"
        rec.hold(rec_frame_028, 4.5)
        rm = win._toolbar._recent_menu
        rm.popup(win.mapToGlobal(QPoint(500, 320)))
        rec.hold(rec_frame_028, 4.0)
        rm.hide()
        rec.hold(rec_frame_028, 1.0)
        rec.finish(scene_by_name("028_projects_recents"))
    finally:
        QMessageBox.warning = _orig_warn
    assert not warn_seen, f"project writer warned: {warn_seen}"

    # 030 lattice tab still
    win._statusbar._clear_message()
    settle(4)
    win.grab().save(s["030_lattice"])

    # 040 inspector opens on a real RF gap (recorded)
    gap = next(e for e in lattice.elements
               if type(e).__name__ in ("RFGap", "RfGap"))
    rec.start("040_inspector")
    rec.hold(win_frame, 1.5)
    win.state.set_selected(gap)
    rec.hold(win_frame, 5.5)
    rec.finish(scene_by_name("040_inspector"))
    win.state.set_selected(None)

    go_tab("beam")
    settle(6)
    win.grab().save(s["050_beam"])
    go_tab("numerics")
    settle(6)
    win.grab().save(s["060_numerics"])
    go_tab("matching")
    settle(6)
    win.grab().save(s["070_matching"])

    # 080 the four campaign tabs, visited in narration order
    rec.start("080_studies")
    for label in ("param study", "error study", "failure study",
                  "surrogates"):
        go_tab(label)
        rec.hold(win_frame, 3.2)
    rec.finish(scene_by_name("080_studies"))

    # =================================================================
    # 090 real multi-particle run with space charge, to completion
    go_tab("lattice")
    settle(6)
    cfg_run = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                         current=MP_CURRENT_MA, alpha_x=2.0, beta_x=1.0,
                         alpha_y=2.0, beta_y=1.0, n_particles=N_MP)
    win.beam_tab.set_beam_config(cfg_run)
    win.state.set_beam_config(cfg_run)
    settle(6)
    rec.start("090_run")
    rec.hold(chrome_frame, 1.5)
    win._toolbar._run_mp_btn.click()
    ok = rec.wait_until(chrome_frame,
                        lambda: win.state.results is not None,
                        cap_s=200, stable_s=0.3)
    assert ok and win.state.results is not None, "MP run did not finish"
    rec.hold(chrome_frame, 4.0)
    rec.finish(scene_by_name("090_run"))
    mp_results = win.state.results

    # 092 the same run stopped mid-flight, then a hand scrub
    clicked = {"done": False}

    def _stopper(p: int) -> None:
        if p >= STOP_AT_PCT and not clicked["done"]:
            clicked["done"] = True
            win._toolbar._stop_btn.click()

    rec.start("092_stop_and_scrub")
    rec.hold(chrome_frame, 1.2)
    win._toolbar._run_mp_btn.click()
    if win._mp_worker is not None:
        win._mp_worker.progress.connect(_stopper)
    ok = rec.wait_until(
        chrome_frame,
        lambda: "stopped by user" in win._statusbar._msg_seg.text(),
        cap_s=200, stable_s=0.2)
    assert ok, "stop demo never reached the stopped-by-user message"
    rec.hold(chrome_frame, 2.5)
    total_mm = int(sum(e.length for e in lattice.elements))
    for v in list(range(0, total_mm + 1, max(1, total_mm // 24))) + \
            [total_mm // 2]:
        win._toolbar._slider.setValue(v)
        rec.tick(chrome_frame)
    rec.hold(chrome_frame, 1.5)
    rec.finish(scene_by_name("092_stop_and_scrub"))
    # the aborted run leaves the completed run's results in place
    assert win.state.results is mp_results, "stop demo clobbered results"
    win._toolbar._slider.setValue(0)
    settle(4)

    # 094 status bar anatomy: edit -> unsaved pill -> auto-clear -> undo
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    quad = next(e for e in lattice.elements
                if type(e).__name__ == "Quadrupole")
    rec.start("094_statusbar")
    rec.hold(statusbar_frame, 5.0)
    old_len = float(quad.length)
    win.state.bus.do(ParamChangeCommand(quad, "length", old_len, 25.0))
    rec.hold(statusbar_frame, 4.5)
    win.state.bus.undo()
    rec.hold(statusbar_frame, 1.5)
    win._save_lattice()          # the real Ctrl+S slot; scratch copy
    rec.hold(statusbar_frame, 7.5)   # 6-s auto-clear happens on camera
    rec.hold(statusbar_frame, 1.0)
    rec.finish(scene_by_name("094_statusbar"))
    assert float(quad.length) == old_len, "undo did not restore the quad"

    # 096 Simulate menu
    menu_still("096_simulate_menu", "Simulate")

    # 098 Backtrack dialog (constructed non-modally, never exec'd)
    from linac_gen_gui.interphase.dialogs.backtrack_dialog import (
        BacktrackDialog,
    )
    bdlg = BacktrackDialog(n_elements=len(lattice.elements),
                           has_results_beam=True, parent=win)
    bdlg.show()
    settle(12)
    dialog_over_window(bdlg, scale=1.5).save(s["098_backtrack"])
    bdlg.hide()
    settle(4)

    # 099 real backtrack run; caveat warning shown non-modally
    from linac_gen_gui.interphase.dialogs import backtrack_dialog as btmod
    shown_box: dict = {}

    def _show_warning(parent, title, text, *a, **k):
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.show()
        shown_box["box"] = box
        return QMessageBox.StandardButton.Ok

    def frame_099() -> QImage:
        box = shown_box.get("box")
        extra = box.grab().toImage() if box is not None and \
            box.isVisible() else None
        return chrome_frame(extra)

    _orig_exec = BacktrackDialog.exec
    _orig_warn = QMessageBox.warning
    BacktrackDialog.exec = lambda self: 1        # accept the defaults
    QMessageBox.warning = _show_warning
    try:
        rec.start("099_backtrack_run")
        rec.hold(frame_099, 1.2)
        win._run_backtrack()
        ok = rec.wait_until(
            frame_099,
            lambda: "Backtrack complete" in win._statusbar._msg_seg.text(),
            cap_s=200, stable_s=0.2)
        assert ok, "backtrack did not complete"
        rec.hold(frame_099, 6.0)
        rec.finish(scene_by_name("099_backtrack_run"))
    finally:
        BacktrackDialog.exec = _orig_exec
        QMessageBox.warning = _orig_warn
    box = shown_box.get("box")
    if box is not None:
        box.hide()
    settle(4)

    # =================================================================
    # 100 envelope run on camera -> Results tab
    win.state.set_beam_config(cfg_run)
    prev_results = win.state.results          # the backtrack results
    rec.start("100_results")
    rec.hold(win_frame, 1.2)
    win._run_envelope()
    rec.wait_until(
        win_frame,
        lambda: (win.state.results is not None
                 and win.state.results is not prev_results
                 and "Envelope done" in win._statusbar._msg_seg.text()),
        cap_s=90, stable_s=0.3)
    assert win.state.results is not None, "envelope run produced no results"
    assert win.state.results is not prev_results, "envelope never replaced results"
    go_tab("results")
    rec.hold(win_frame, 4.5)
    rec.finish(scene_by_name("100_results"))

    # 110 energy staircase popup
    win.show_result_plot("energy")
    settle(8)
    pop = win.results_tab._popups.get("energy")
    assert pop is not None, "energy popup did not open"
    pop.resize(1600, 900)
    settle(6)
    pop.grab().save(s["110_energy"])
    pop.hide()
    settle(4)

    # 112 auto-save + TraceWin export listing (real files, real writers)
    from linac_gen.io.tracewin_outputs import (write_envelope_txt,
                                               write_partran_out)
    write_partran_out(win.state.results, win.state.lattice,
                      win.state.beam_config, runs_dir / "partran1.out")
    write_envelope_txt(win.state.results, win.state.beam_config,
                       runs_dir / "tracewin1.txt")
    listing = sorted(p.name for p in runs_dir.iterdir())
    lines = [("cmd", "ls -1 runs/")]
    lines += [("out", name) for name in listing]
    lines += [("gap", ""),
              ("out", "auto-saved:  <timestamp>_<run>.h5  +  .opmd.h5"),
              ("out", "exported:    partran1.out  +  tracewin1.txt")]
    cards.terminal_card(s["112_autosave_exports"], lines,
                        title="Calculation directory")

    # 130 Tools menu
    menu_still("130_tools_menu", "Tools")

    # 132 Python console — typed live (QDialog.exec patched narrowly)
    _orig_dexec = QDialog.exec
    QDialog.exec = lambda self: (self.show(), 0)[1]
    try:
        win._open_console()
    finally:
        QDialog.exec = _orig_dexec
    con = next(w for w in QApplication.topLevelWidgets()
               if isinstance(w, QDialog)
               and w.windowTitle() == "Python Console")
    con.resize(1240, 760)
    settle(8)
    edit = con.findChild(QLineEdit)
    con_frame = lambda: dialog_over_window(con, scale=1.35)  # noqa: E731
    rec.start("132_console")
    rec.hold(con_frame, 1.5)
    for query in ("len(lattice.elements)",
                  "results.sigma_x[-1]",
                  "beam_config.energy"):
        rec.type_into(con_frame, edit, query, chars_per_frame=3)
        edit.returnPressed.emit()
        rec.hold(con_frame, 2.2)
    rec.hold(con_frame, 1.5)
    rec.finish(scene_by_name("132_console"))
    con.hide()
    settle(4)

    # 134 matrix viewers side by side (constructed directly, shown)
    from linac_gen_gui.dialogs.transfer_matrix_dialog import (
        TransferMatrixDialog,
    )
    from linac_gen_gui.dialogs.sigma_matrix_dialog import SigmaMatrixDialog
    tm = TransferMatrixDialog(win, win.state.lattice, win.state.beam_config)
    tm.show()
    settle(14)
    sm = SigmaMatrixDialog(win, win.state.results,
                           beam_config=win.state.beam_config)
    sm.show()
    settle(14)
    ti = tm.grab().toImage()
    si = sm.grab().toImage()
    c = canvas()
    p = QPainter(c)
    ti2 = scaled_w(ti, 950)
    si2 = scaled_w(si, 950)
    p.drawImage(4, max(0, (1080 - ti2.height()) // 2), ti2)
    p.drawImage(966, max(0, (1080 - si2.height()) // 2), si2)
    p.end()
    c.save(s["134_matrices"])
    tm.hide()
    sm.hide()
    settle(4)

    # 136 quick Parameter Scan dialog (non-modal by design)
    from linac_gen_gui.interphase.dialogs import ParameterScanDialog
    ps = ParameterScanDialog(win, win.state)
    ps.show()
    settle(12)
    dialog_over_window(ps, scale=1.3).save(s["136_param_scan"])
    ps.hide()
    settle(4)

    # 140 Help menu (un-emphasized)
    menu_still("140_help_menu", "Help")

    # 142 About box (QMessageBox.exec patched narrowly; grab inside)
    about_img: dict = {}
    _orig_mexec = QMessageBox.exec

    def _about_exec(self):
        self.show()
        settle(12)
        about_img["img"] = self.grab().toImage()
        self.hide()
        return 0

    QMessageBox.exec = _about_exec
    try:
        win._open_about()
    finally:
        QMessageBox.exec = _orig_mexec
    assert "img" in about_img, "about box was never shown"
    ai = scaled_w(about_img["img"], int(about_img["img"].width() * 1.9))
    c = canvas()
    p = QPainter(c)
    p.drawImage(0, 0, win.grab().toImage())
    p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, 130))
    p.fillRect((1920 - ai.width()) // 2 - 8, (1080 - ai.height()) // 2 - 8,
               ai.width() + 16, ai.height() + 16, QColor("#2b3b55"))
    p.drawImage((1920 - ai.width()) // 2, (1080 - ai.height()) // 2, ai)
    p.end()
    c.save(s["142_about"])

    # =================================================================
    # 144/146/148/152 — the real mkdocs manual in QtWebEngine
    view = QWebEngineView()
    view.resize(1920, 1080)
    view.show()
    load_state: dict = {}
    view.loadFinished.connect(
        lambda ok_: load_state.__setitem__("ok", ok_))

    def load_page(rel: str) -> None:
        load_state.clear()
        view.load(QUrl.fromLocalFile(str(ROOT / "site" / rel)))
        t0 = _time.monotonic()
        while "ok" not in load_state and _time.monotonic() - t0 < 30:
            settle(6)
            _time.sleep(0.01)
        assert load_state.get("ok"), f"manual page failed to load: {rel}"
        pump(2.0)

    web_frame = lambda: view.grab().toImage()   # noqa: E731

    def scroll_through(ys, hold_s: float = 1.6) -> None:
        for y in ys:
            view.page().runJavaScript(f"window.scrollTo(0, {int(y)})")
            pump(0.25)
            rec.hold(web_frame, hold_s)

    load_page("index.html")
    rec.start("144_manual_tour")
    rec.hold(web_frame, 3.0)
    scroll_through([500, 1100, 1700], hold_s=1.6)
    load_page("10_gui/01_overview.html")
    rec.hold(web_frame, 2.4)
    scroll_through([700, 1500], hold_s=1.8)
    rec.finish(scene_by_name("144_manual_tour"))

    load_page("10_gui/08_workflows.html")
    rec.start("146_workflows_page")
    rec.hold(web_frame, 2.6)
    scroll_through([600, 1300, 2100, 3000], hold_s=2.0)
    rec.finish(scene_by_name("146_workflows_page"))

    # 148 F1 context help: real handler, opener patched to a no-op
    from linac_gen_gui.interphase import manual_help
    _orig_open_path = manual_help._open_path
    manual_help._open_path = lambda p: True
    try:
        go_tab("lattice")
        settle(6)
        win.state.set_selected(gap)
        rec.start("148_f1_help")
        rec.hold(statusbar_frame, 1.5)
        win._open_manual_for_selected()
        rec.hold(statusbar_frame, 4.0)
        load_page("03_elements/06_rfgap.html")
        rec.hold(web_frame, 2.6)
        scroll_through([500, 1100], hold_s=1.8)
        rec.finish(scene_by_name("148_f1_help"))
    finally:
        manual_help._open_path = _orig_open_path
    win.state.set_selected(None)
    msg148 = win._statusbar._msg_seg.text()
    print(f"[check] F1 status message was: {msg148!r}")

    # 150 update notice, driven by hand with the real current tag
    from linac_gen import __version__ as _helix_version
    tag = f"v{_helix_version}"
    win._toolbar.set_update_available(tag)
    win._statusbar.show_update_available(tag)
    settle(6)
    hbtn = next(b for b in win._toolbar.findChildren(QPushButton)
                if b.menu() is not None and b.text() == "Help")
    hmenu = hbtn.menu()
    hmenu.popup(hbtn.mapToGlobal(hbtn.rect().bottomLeft()))
    settle(10)
    m = hmenu.grab().toImage()
    full = win.grab().toImage()
    c = canvas()
    p = QPainter(c)
    p.drawImage(0, 0, full)
    p.fillRect(0, 0, 1920, 1080, QColor(0, 0, 0, 110))
    gp = hmenu.mapToGlobal(QPoint(0, 0))
    wp = win.mapToGlobal(QPoint(0, 0))
    p.drawImage(gp.x() - wp.x(), gp.y() - wp.y(), m)
    big = scaled_w(m, min(m.width() * 2, 980))
    p.fillRect(1920 - big.width() - 112, 132, big.width() + 16,
               big.height() + 16, QColor("#2b3b55"))
    p.drawImage(1920 - big.width() - 104, 140, big)
    sy = win._statusbar.mapTo(win, QPoint(0, 0)).y() - 1
    sh = win._statusbar.height() + 2
    r1, r2 = strip_rows(full, sy, sh)
    p.drawImage(0, 1080 - r1.height() - r2.height() - 30, r1)
    p.drawImage(0, 1080 - r2.height() - 10, r2)
    p.end()
    c.save(s["150_updates"])
    hmenu.hide()
    win._statusbar.clear_update_notice()
    settle(4)

    load_page("10_gui/09_updates.html")
    rec.start("152_updates_page")
    rec.hold(web_frame, 2.6)
    scroll_through([500, 1100, 1700], hold_s=2.0)
    rec.finish(scene_by_name("152_updates_page"))
    view.hide()
    settle(4)

    # 155 live font scaling sweep, ending back at the default
    from linac_gen_gui.interphase import theme
    go_tab("lattice")
    settle(6)
    rec.start("155_font_scaling")
    rec.hold(win_frame, 1.5)
    for pt in (13, 14, 15, 16):
        win._toolbar._font_spin.setValue(pt)
        rec.hold(win_frame, 1.2)
    rec.hold(win_frame, 1.5)
    for pt in (14, theme.FONT_SIZE):
        win._toolbar._font_spin.setValue(pt)
        rec.hold(win_frame, 1.2)
    rec.hold(win_frame, 1.5)
    rec.finish(scene_by_name("155_font_scaling"))
    assert win._toolbar._font_spin.value() == theme.FONT_SIZE

    # 158 shortcuts card — verbatim from the app.py QShortcut wiring
    cards.terminal_card(s["158_shortcuts"], [
        ("out", "Ctrl+N            New Project wizard"),
        ("out", "Ctrl+O            Open Lattice"),
        ("out", "Ctrl+S            Save Lattice"),
        ("gap", ""),
        ("out", "Ctrl+R            Run Envelope"),
        ("out", "Ctrl+Shift+R      Run Multi-particle"),
        ("gap", ""),
        ("out", "Ctrl+=  /  Ctrl++ Font larger"),
        ("out", "Ctrl+-            Font smaller"),
        ("out", "Ctrl+0            Reset font size"),
        ("gap", ""),
        ("out", "F1                Manual chapter for selected element"),
    ], title="Keyboard shortcuts")

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
    os._exit(0)          # worker/web threads must never block exit


if __name__ == "__main__":
    main()
