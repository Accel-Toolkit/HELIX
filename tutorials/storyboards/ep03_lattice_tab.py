"""Episode 3 — The Lattice tab in depth (first per-tab deep dive).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep03_lattice_tab.py
Output: tutorials/rendered/ep03_lattice_tab.mp4 (+ .srt)

Maximum-detail cut.  Every operational claim is demonstrated live:
chip filtering, wheel zoom + s-cursor, Shift+Click multi-select with a
macro delete, live validation badges, outline drag-reorder, the Add
Element dialog, the palette-drop gap indicator, the full CRUD loop,
context menus, the inline manual popup, per-type inspector variants,
command-card inspectors, a REAL orbit correction on the correction
demo deck (measured numbers), the BPM-targets file flow, the
199-cell RfqCell→VaneRFQ swap, a live envelope parameter scan, and
the save-safety net (status-bar dirty chip + itemized discard prompt).
Two scenes render the real mkdocs manual through QtWebEngine in a
subprocess.  All dialogs/menus are driven non-modally (offscreen
exec = forever-hang) and every edit rides the real command bus.
"""
from __future__ import annotations

import os
import subprocess
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

WORK = ROOT / "tutorials" / "rendered" / "ep03_work"
SHOTS = WORK / "shots"
FRAMES = WORK / "frames"
OUT = ROOT / "tutorials" / "rendered" / "ep03_lattice_tab.mp4"

BG = "#0b1220"

SCENES = [
    Scene("010_title",
          "Welcome back. This deep dive covers the Lattice tab "
          "completely: navigation, editing, and the tools. A real "
          "orbit correction, the R F Q vane swap, a live parameter "
          "scan, and the safety net around Save. Our machine is the "
          "drift tube linac from the tour.",
          min_s=7.0),
    Scene("020_context",
          "The Lattice tab is home base. Everything HELIX simulates "
          "starts from what you see here. Four columns, left to "
          "right: the element palette, the outline tree, the centre "
          "column with the timeline and the listing, and the "
          "inspector, with the lattice toolbar above. We will walk "
          "through each in turn."),
    Scene("030_toolbar",
          "Open reads four formats: TraceWin dot dat, MAD X dot "
          "madx or dot seq, MAD eight dot lat or dot flat, and "
          "Elegant dot L T E; the extension picks the parser. Save "
          "writes the lattice back, Save As writes a copy, and "
          "Reload re parses the file from disk. And the summary, "
          "always visible: thirty eight elements, total length "
          "eight hundred sixty millimetres.",
          min_s=7.0),
    Scene("042_palette_scroll",
          "The palette lists every type HELIX knows, in five "
          "labelled families: magnets, cavities, drifts and "
          "passives, then the SET and ADJUST command cards, and an "
          "OTHER shelf for the foil. Thirty nine cards in all. "
          "Double click to append, or drag one into the machine. "
          "The field map cards ask for an external field file "
          "first."),
    Scene("046_taxonomy_manual",
          "This overview page is the element catalog: a pick by "
          "use case table, one row per type. Below it, the "
          "taxonomy. Every element is one of four kinds: closed "
          "form matrix elements, zero length kicks, substepped "
          "field maps, and passive markers. Right clicking any "
          "element in the G U I opens its chapter directly."),
    Scene("050_outline",
          "The outline groups the machine by name prefix, the text "
          "before the first dot or underscore; on an auto named deck "
          "that coincides with type. On a named machine it becomes "
          "section grouping, and groups over forty entries start "
          "collapsed. The FREQ entry wears a small gear, the badge "
          "of a command card. Clicking any entry selects that "
          "element everywhere."),
    Scene("060_filter",
          "Above the tree is a filter box. Type a few letters and the "
          "tree narrows to matching elements. Here we typed gap, and "
          "only the eight R F gaps remain. This is how you find one "
          "element in a machine with two thousand of them. Clearing "
          "the box restores the full tree."),
    Scene("070_timeline",
          "The centre column: type chips with per family counts, "
          "and the timeline, the machine to scale. Every rectangle "
          "is an element, colour coded: purple solenoids, cyan "
          "quadrupoles, and the thin bright lines are R F gaps, "
          "zero length cards drawn with a minimum width so they "
          "never vanish. Drifts render half height and dimmed. The "
          "small amber squares are validation badges. Click to "
          "select: the selected quadrupole gets a white outline.",
          min_s=7.0),
    Scene("072_chip_filter",
          "The chips are not just counters. Click one, and the "
          "timeline dims everything else. Only the quadrupoles stay "
          "lit, and the focusing pattern is suddenly obvious. Chips "
          "combine: add the gaps, and both families glow. An empty "
          "selection means show everything."),
    Scene("074_zoom_cursor",
          "Hold control and scroll to zoom the strip, up to ten "
          "times, down to a quarter; reset brings the whole machine "
          "back. The cyan vertical line is the s cursor, shared "
          "application state: when a results view moves it, this "
          "strip follows. Here we sweep it along the machine."),
    Scene("076_multiselect",
          "Shift click adds bars to a multi selection, each with a "
          "soft halo. These three drifts are now one unit: delete "
          "removes all of them as a single command, delete three "
          "elements, and the listing shortens. One control Z brings "
          "all three back. Grouped edits stay grouped on the undo "
          "stack."),
    Scene("078_validation",
          "HELIX re validates after every edit. Four advisory "
          "rules: apertures that "
          "are not positive, negative lengths, zero length on "
          "elements that should have one, and R F elements with no "
          "frequency upstream. This deck's gap cards are zero length "
          "by construction, so they already wear badges. Now watch: "
          "we zero the quadrupole's length, the bar collapses, and a "
          "badge appears instantly. Undo clears it. Badges never "
          "block a save.",
          min_s=7.0),
    Scene("080_sequence",
          "Below, the Sequence view: the machine as a list, in beam "
          "order. Each row shows index, position in millimetres, the "
          "TraceWin keyword, and arguments that adapt per type: a "
          "quad shows length, gradient, aperture; a gap shows "
          "voltage, phase, frequency. Rows carry the timeline's "
          "family colours, and "
          "the listing follows your selection from anywhere.",
          min_s=6.0),
    Scene("085_outline_reorder",
          "Rows in the outline drag and drop, and the tree "
          "translates the drop into a move of the flat lattice. A "
          "quadrupole moves down the machine: the listing renumbers, "
          "the timeline redraws. A move is a command like any other; "
          "control Z puts it back."),
    Scene("090_breakdown",
          "The second tab of the listing is Breakdown. Instead of "
          "beam order, it summarises the machine per element type. "
          "How many of each, their total length, and the share of the "
          "machine they occupy. It is the quickest sanity check that "
          "an imported file contains what you expect."),
    Scene("100_inspector",
          "The inspector is where elements are edited; its colour "
          "swatch matches the timeline. For a quadrupole: length, "
          "gradient, aperture, gradient error, skew "
          "angle, and g three to g six, the higher order gradients. "
          "Then alignment, and an honesty note: only the offsets in "
          "x and y and the tilt about z act on the beam in tracking. "
          "Offset in z, pitch, and yaw are stored and saved, but not "
          "used in tracking.",
          min_s=7.0),
    Scene("110_edit",
          "Editing is direct: click a field, type, enter. The "
          "gradient goes from eight to nine tesla per metre and the "
          "sequence view updates immediately. Every change is a "
          "command on a bus: undo reverts it, redo applies it again, "
          "and the status bar names every edit. Spin boxes adapt "
          "their decimals so tiny "
          "kicks are never truncated. Nothing touches disk until "
          "you press Save.",
          min_s=7.0),
    Scene("112_add_dialog",
          "The Add button, or the Insert key, opens this dialog. "
          "Every type is in the combo, and the form is built from "
          "the same schema the inspector uses: switch the type and "
          "the fields change with it, each with its units. We name "
          "ours Q DEMO, accept, and the new quadrupole lands after "
          "the current selection, already selected for editing. We "
          "will undo it shortly.",
          min_s=7.0),
    Scene("114_palette_drop",
          "Or drag a card straight onto the timeline. While you "
          "drag, a cyan line tracks the nearest insertion gap. "
          "Release, and the element is inserted at exactly that "
          "index, here a steerer, with the status bar confirming the "
          "position. We undo that too: nothing here is more than one "
          "control Z from safety."),
    Scene("116_crud",
          "The rest of the loop. Duplicate clones the selection "
          "after itself, control D. Delete "
          "removes it, the delete key. Undo and redo walk the "
          "stack, control Z, control Y. Copy, cut and paste, "
          "control C, X and V, use an internal clipboard of deep "
          "copies: copy this solenoid, select the end marker, "
          "paste, and a fresh copy appears there.",
          min_s=7.0),
    Scene("117_context_menus",
          "Right click the timeline or outline for the edit menu: "
          "insert before or after, duplicate, delete, and the "
          "clipboard actions. Paste needs something on the "
          "clipboard first. "
          "Right click a row in the outline or listing for one more "
          "entry: open the manual for this type."),
    Scene("119_manual_popup",
          "Here is that manual. The question mark in the inspector "
          "header opens the chapter for the selected type, rendered "
          "from the markdown source, and it stays on top while you "
          "edit underneath. Every chapter has the same shape: a T L "
          "D R table against the TraceWin card, a tutorial, and the "
          "full A P I reference.",
          min_s=6.0),
    Scene("122_inspector_variants",
          "The inspector adapts to every type. An aperture gets a "
          "shape dropdown, rectangular, circular, pepperpot and "
          "four more; we flip this one to pepperpot. A marker gets "
          "two checkboxes: trigger a beam snapshot, or serve as a "
          "beam position monitor. A steerer is two numbers, the "
          "integrated field in x and y, in tesla metres. And R F "
          "elements carry an error block: voltage error, phase "
          "offset, frequency offset.",
          min_s=7.0),
    Scene("124_command_card",
          "Command cards are elements too. We drop in a SET TWISS "
          "card: it sits in the machine like any magnet but carries "
          "beam commands instead of fields. In the outline it wears "
          "the gear. The inspector builds its editor automatically "
          "from the card's own parameters, keyword read only at the "
          "top.",
          min_s=6.0),
    Scene("126_adjust_flags",
          "ADJUST cards get the same treatment, plus one convention: "
          "the flags list. Zero skips a parameter, one adjusts it, "
          "two couples it to the previous one. The Matching episode "
          "uses these cards in anger."),
    Scene("132_orbit_correction",
          "Now the tools, on the correction demo deck. Six F O D O "
          "cells, four steerer and B P M pairs. First, sabotage: "
          "this quadrupole moves half a millimetre sideways. "
          "Correct orbit tracks the beam in a background worker; "
          "the button becomes Cancel correction and the G U I "
          "stays live. The report: method one to one, four pairs, "
          "and an R M S orbit error under zero point zero four "
          "millimetres after a single pass. The kicks arrive as one "
          "undoable command, and the lattice is flagged as fitted, "
          "so plain Save reroutes to Save As. One undo removes "
          "every kick, a second removes the sabotage.",
          min_s=10.0),
    Scene("134_bpm_targets",
          "B P M targets loads a measured orbit: plain text, one x "
          "y pair per monitor, nan leaves a plane free. Our DIAG "
          "POSITION cards carry explicit operands, shown in the "
          "inspector's diagnostic matching block. Load the file: "
          "the status bar confirms four rows, runtime only, never "
          "saved into the dot dat. A new inspector row appears, the "
          "file override, which wins over the deck values. Loading "
          "is not an undoable edit; reload the lattice to clear "
          "it.",
          min_s=7.0),
    Scene("142_vane_swap",
          "One button remains, RfqCell to VaneRFQ, for R F Q work. "
          "This deck holds one hundred ninety nine cell cards, each "
          "a thin bar. The button asks for a dot vane file, the same "
          "geometry Toutatis uses, and a field model. One click, and "
          "the cells become a single vane R F Q element spanning the "
          "machine; the status bar reports the swap. If you never "
          "touch R F Qs, you never need this, and that is fine.",
          min_s=7.0),
    Scene("143_vane_inspector",
          "The consolidated element keeps the whole cell chain, the "
          "listing reports one hundred ninety nine cells, and the "
          "inspector lets you switch field model any time: two term, "
          "eight term, full eight term, or the two Laplace solvers."),
    Scene("144_param_scan",
          "One more tool, from the Tools menu: Parameter Scan. "
          "Pick any element and numeric parameter, here quadrupole "
          "three's gradient, a range, six to ten tesla per metre in "
          "nine points, and a mode, envelope or multi particle. "
          "The output list is "
          "deep: R M S sizes, normalised and eigen emittances, the "
          "four D invariant, transmission, growth ratios, and two "
          "multi particle only halo metrics. Run. Nine envelope "
          "points land in under a second: more gradient, smaller "
          "horizontal beam, twelve point one down to eleven point "
          "six millimetres.",
          min_s=8.0),
    Scene("146_scan_nominal",
          "The guarantee that makes scanning safe: when the scan "
          "ends, the parameter snaps back to nominal. The lattice is "
          "never left modified, the inspector shows the gradient "
          "back at exactly eight, and the dialog is non modal, so "
          "leave it open and keep working."),
    Scene("152_dirty_chip",
          "Nearly done. Save writes the lattice back; Reload is "
          "the fastest way back to the file on disk. Watch the "
          "bottom edge: one gradient "
          "edit, and the amber unsaved dot lights in the status bar "
          "beside the edit's own message. Lattice edits and project "
          "settings are separate flags. Try to open another file or "
          "quit now, and HELIX asks first.",
          min_s=6.0),
    Scene("154_unsaved_prompt",
          "Show Details itemizes what is at risk, one line per "
          "edit: our quadrupole, gradient eight to nine. Two more "
          "protections. A lattice carrying fitted "
          "values reroutes plain Save to Save As, suggesting a dot "
          "matched dot dat name. And a lattice imported from MAD X, "
          "MAD eight or Elegant is never written back to its source: "
          "HELIX writes TraceWin format only, so Save becomes Save "
          "As.",
          min_s=7.0),
    Scene("156_manual_tour",
          "Everything here is written down in the manual's Lattice "
          "tab chapter: the layout, the shortcuts, the editing "
          "rules, orbit correction and B P M targets, the formats, "
          "and the exact saving rules. The element catalog goes "
          "deeper on every type we touched, one press of the "
          "question mark away.",
          min_s=6.0),
    Scene("160_outro",
          "That is the entire Lattice tab, for real this time. Next, "
          "the Beam tab in the same depth: species, energy, current, "
          "emittance, and the distributions behind every run. See "
          "you there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


# ----------------------------------------------------------------------
# Manual scenes — rendered by a QtWebEngine SUBPROCESS (mixing WebEngine
# into the GUI QApplication is the risky path; a separate process is the
# digest-sanctioned alternative).  Writes %05d.png frame sequences.
# ----------------------------------------------------------------------
_WEB_SCRIPT = r'''
import os, sys, tempfile, time
from pathlib import Path
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
qcfg = Path(tempfile.mkdtemp()) / "o.json"
qcfg.write_text('{"screens": [{"name":"t","x":0,"y":0,"width":1920,'
                '"height":1080,"logicalDpi":96,"physicalDpi":96}]}')
os.environ.setdefault("QT_QPA_PLATFORM", f"offscreen:configfile={qcfg}")
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QUrl
app = QApplication(["mantour"])
view = QWebEngineView(); view.resize(1920, 1080); view.show()

def pump(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents(); time.sleep(0.01)

def load(path):
    done = {}
    view.loadFinished.connect(lambda ok: done.update(ok=ok))
    view.load(QUrl.fromLocalFile(path))
    t0 = time.time()
    while "ok" not in done and time.time() - t0 < 45:
        app.processEvents(); time.sleep(0.02)
    assert done.get("ok"), f"load failed: {path}"
    pump(2.5)

def scroll_to(y):
    view.page().runJavaScript(f"window.scrollTo(0, {int(y)})")
    pump(0.06)

def scroll_height():
    got = {}
    view.page().runJavaScript(
        "Math.max(document.body.scrollHeight,"
        "document.documentElement.scrollHeight)",
        lambda h: got.update(h=h))
    t0 = time.time()
    while "h" not in got and time.time() - t0 < 10:
        app.processEvents(); time.sleep(0.01)
    return float(got.get("h", 1080))

def record(out_dir, waypoints, hold=12, step_px=45):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    n = 0
    def snap():
        nonlocal n
        view.grab().save(str(out / f"{n:05d}.png")); n += 1
    y = 0.0
    scroll_to(0); pump(0.4)
    for _ in range(hold):
        snap()
    for target in waypoints:
        while abs(target - y) > step_px:
            y += step_px if target > y else -step_px
            scroll_to(y); snap()
        y = target; scroll_to(y)
        for _ in range(hold):
            snap()
    print(f"web frames {out_dir}: {n}")

root, f046, f156 = sys.argv[1], sys.argv[2], sys.argv[3]
load(os.path.join(root, "site/03_elements/00_overview.html"))
h = scroll_height()
record(f046, [min(1250, h - 1080), max(h - 1080, 0)])
load(os.path.join(root, "site/10_gui/02_lattice_tab.html"))
h = scroll_height()
m = max(h - 1080, 0)
record(f156, [m * 0.3, m * 0.62, m])
sys.stdout.flush()
os._exit(0)
'''


def render_manual_frames() -> None:
    script = WORK / "web_render.py"
    WORK.mkdir(parents=True, exist_ok=True)
    script.write_text(_WEB_SCRIPT)
    f046 = FRAMES / "046_taxonomy_manual"
    f156 = FRAMES / "156_manual_tour"
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)      # subprocess sets its own
    r = subprocess.run(
        [sys.executable, str(script), str(ROOT), str(f046), str(f156)],
        capture_output=True, text=True, timeout=600, env=env)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        raise RuntimeError(f"web render failed:\n{r.stderr[-3000:]}")
    for sc_name, d in (("046_taxonomy_manual", f046),
                       ("156_manual_tour", f156)):
        n = len(list(d.glob("*.png")))
        assert n > 0, f"no frames for {sc_name}"
        sc = scene_by_name(sc_name)
        sc.frames_dir = str(d)
        sc.fps = 10.0


# ----------------------------------------------------------------------
def capture_visuals() -> None:
    from PyQt6.QtCore import QEventLoop, QPoint, QPointF, QRect, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter, QWheelEvent
    from PyQt6.QtWidgets import (QApplication, QComboBox, QMenu,
                                 QMessageBox, QPushButton, QScrollArea)

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication([])

    cards.title_card(s["010_title"], "The Lattice Tab",
                     "Deep dive — every panel, every button, live")
    cards.title_card(s["160_outro"], "The Lattice Tab",
                     "Palette · timeline · inspector · orbit tools · "
                     "safety net",
                     kicker="NEXT: THE BEAM TAB")

    from linac_gen_gui.interphase.app import (InterphaseWindow,
                                              _parse_lattice_file)
    from linac_gen_gui.interphase.commands import (InsertCommand,
                                                   ParamChangeCommand)
    from linac_gen.core.config import BeamConfig

    win = InterphaseWindow()
    win.resize(1920, 1080)
    win.show()

    def settle(n: int = 4) -> None:
        # Flush DeferredDelete each pass: rebuilt chips/inspector rows
        # otherwise linger as visible zombies in grabs (house lesson).
        from PyQt6.QtCore import QCoreApplication, QEvent
        for _ in range(n):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            QCoreApplication.sendPostedEvents(
                None, int(QEvent.Type.DeferredDelete))

    def full(key: str) -> None:
        settle()
        win.grab().save(s[key])

    def widget_rect(widget, pad: int = 8) -> QRect:
        tl = widget.mapTo(win, QPoint(0, 0))
        return QRect(max(0, tl.x() - pad), max(0, tl.y() - pad),
                     widget.width() + 2 * pad, widget.height() + 2 * pad)

    def crop_widget(key: str, widget, pad: int = 8) -> None:
        settle()
        img = win.grab().toImage()
        r = widget_rect(widget, pad)
        img.copy(r).save(s[key])

    def crop_grab(widget, pad: int = 8):
        def g():
            img = win.grab().toImage()
            return img.copy(widget_rect(widget, pad))
        return g

    def union_grab(widgets, pad: int = 8):
        def g():
            img = win.grab().toImage()
            r = widget_rect(widgets[0], pad)
            for w in widgets[1:]:
                r = r.united(widget_rect(w, pad))
            return img.copy(r)
        return g

    def compose(*items) -> QImage:
        """items = (QImage, x, y, scale) painted onto a 1920x1080 canvas."""
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(QColor(BG))
        p = QPainter(canvas)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        for img, x, y, scale in items:
            w = int(img.width() * scale)
            h = int(img.height() * scale)
            p.drawImage(QRect(int(x), int(y), w, h), img)
        p.end()
        return canvas

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    def load_deck(path: str):
        lattice, _meta = _parse_lattice_file(path)
        win.state.set_lattice(lattice, path)
        settle(8)
        return lattice

    def by_name(lattice, name: str):
        return next(e for e in lattice.elements if e.name == name)

    def report(tag: str) -> None:
        lat = win.state.lattice
        print(f"[state] {tag}: n={len(lat.elements) if lat else 0} "
              f"dirty={win.state.bus.dirty}")

    deck = str(ROOT / "examples/dtl_section.dat")
    lattice = load_deck(deck)
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=0.0))
    go_tab("lattice")
    settle(10)

    lt = win.lattice_tab
    # Badges are cleared by the timeline's resize-rebuild; re-validate
    # after layout settles so the deck's own badges are deterministic.
    lt._refresh_validation()
    settle(4)

    rec = Recorder(FRAMES, settle)

    # ---- 020 / 030 / 040 stills --------------------------------------
    full("020_context")
    crop_widget("030_toolbar", lt._btn_add.parentWidget())

    # ---- 042 palette scroll ------------------------------------------
    sa = lt._palette.findChild(QScrollArea)
    bar = sa.verticalScrollBar()
    pal_g = crop_grab(lt._palette)

    def pal_canvas():
        img = pal_g()
        sc = min(1050 / img.height(), 3.0)
        return compose((img, (1920 - img.width() * sc) / 2, 15, sc))

    rec.start("042_palette_scroll")
    rec.hold(pal_canvas, 2.0)
    vmax = bar.maximum()
    for i in range(1, 29):
        bar.setValue(int(vmax * i / 28))
        rec.tick(pal_canvas)
    rec.hold(pal_canvas, 2.5)
    bar.setValue(0)
    rec.hold(pal_canvas, 1.5)
    rec.finish(scene_by_name("042_palette_scroll"))

    # ---- 050 outline still -------------------------------------------
    crop_widget("050_outline", lt._outline)

    # ---- 060 filter (kept demo) --------------------------------------
    outline_g = crop_grab(lt._outline)
    rec.start("060_filter")
    rec.hold(outline_g, 1.5)
    rec.type_into(outline_g, lt._outline._search, "gap",
                  chars_per_frame=1)
    rec.hold(outline_g, 3.0)
    lt._outline._search.setText("")
    rec.hold(outline_g, 2.5)
    rec.finish(scene_by_name("060_filter"))

    # ---- 070 timeline still (chips + timeline, quad selected) --------
    quad3 = by_name(lattice, "QUAD_003")
    win.state.set_selected(quad3)
    settle(6)
    chips_tl_g = union_grab([lt._chips, lt._timeline])
    chips_tl_g().save(s["070_timeline"])

    # ---- 072 chip filter ---------------------------------------------
    rec.start("072_chip_filter")
    rec.hold(chips_tl_g, 1.5)
    lt._chips._chips["Quadrupole"].setChecked(True)
    rec.hold(chips_tl_g, 3.0)
    lt._chips._chips["RFGap"].setChecked(True)
    rec.hold(chips_tl_g, 2.5)
    lt._chips._chips["Quadrupole"].setChecked(False)
    lt._chips._chips["RFGap"].setChecked(False)
    rec.hold(chips_tl_g, 2.0)
    rec.finish(scene_by_name("072_chip_filter"))

    # ---- 074 zoom + s-cursor -----------------------------------------
    tl_g = crop_grab(lt._timeline)

    def wheel(delta: int) -> None:
        ev = QWheelEvent(QPointF(300, 30), QPointF(300, 30),
                         QPoint(0, 0), QPoint(0, delta),
                         Qt.MouseButton.NoButton,
                         Qt.KeyboardModifier.ControlModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        lt._timeline.wheelEvent(ev)

    rec.start("074_zoom_cursor")
    rec.hold(tl_g, 1.5)
    for _ in range(8):
        wheel(120)
        rec.tick(tl_g)
    rec.hold(tl_g, 2.0)
    lt._timeline.reset_zoom()
    rec.hold(tl_g, 1.5)
    total_mm = sum(e.length for e in lattice.elements)
    for i in range(26):
        win.state.set_s_cursor(total_mm * (i + 1) / 26.0)
        rec.tick(tl_g)
    rec.hold(tl_g, 1.5)
    win.state.set_s_cursor(0.0)
    rec.finish(scene_by_name("074_zoom_cursor"))

    # ---- 076 multiselect + macro delete ------------------------------
    drifts = [e for e in lattice.elements
              if type(e).__name__ == "Drift"][2:5]
    center_g = union_grab([lt._chips, lt._timeline, lt._bottom_tabs])
    rec.start("076_multiselect")
    win.state.set_selected(drifts[0])
    rec.hold(center_g, 1.5)
    lt._timeline._toggle_in_selection(id(drifts[1]))
    rec.hold(center_g, 1.0)
    lt._timeline._toggle_in_selection(id(drifts[2]))
    rec.hold(center_g, 2.0)
    lt._delete_selected()
    rec.hold(center_g, 3.0)
    win.state.bus.undo()
    rec.hold(center_g, 2.5)
    rec.finish(scene_by_name("076_multiselect"))
    report("after 076")

    # ---- 078 validation badges ---------------------------------------
    rec.start("078_validation")
    rec.hold(chips_tl_g, 2.0)
    win.state.bus.do(ParamChangeCommand(quad3, "length", 30.0, 0.0))
    rec.hold(chips_tl_g, 3.5)
    win.state.bus.undo()
    rec.hold(chips_tl_g, 2.5)
    rec.finish(scene_by_name("078_validation"))
    report("after 078")

    # ---- 080 sequence still ------------------------------------------
    win.state.set_selected(quad3)
    settle(6)
    crop_widget("080_sequence", lt._bottom_tabs)

    # ---- 085 outline reorder -----------------------------------------
    quad1 = by_name(lattice, "QUAD_001")
    from_idx = next(i for i, e in enumerate(lattice.elements)
                    if e is quad1)
    outline_seq_g = union_grab([lt._outline, lt._bottom_tabs])
    rec.start("085_outline_reorder")
    win.state.set_selected(quad1)
    rec.hold(outline_seq_g, 2.0)
    lt._on_tree_move(from_idx, from_idx + 12)
    rec.hold(outline_seq_g, 3.5)
    win.state.bus.undo()
    rec.hold(outline_seq_g, 2.5)
    rec.finish(scene_by_name("085_outline_reorder"))
    report("after 085")

    # ---- 090 breakdown still -----------------------------------------
    lt._bottom_tabs.setCurrentIndex(1)
    settle(6)
    img = win.grab().toImage()
    tl0 = lt._bottom_tabs.mapTo(win, QPoint(0, 0))
    img.copy(tl0.x(), tl0.y(), 620, 260).save(s["090_breakdown"])
    lt._bottom_tabs.setCurrentIndex(0)
    settle(4)

    # ---- 100 inspector still -----------------------------------------
    win.state.set_selected(quad3)
    settle(6)
    crop_widget("100_inspector", lt._inspector)

    # ---- 110 edit with undo/redo on the toolbar ----------------------
    # Drive the REAL inspector editor: the gradient QDoubleSpinBox's
    # valueChanged fires _commit -> ParamChangeCommand + status message.
    from PyQt6.QtWidgets import QDoubleSpinBox

    def gradient_spinbox():
        boxes = [sb for sb in lt._inspector.findChildren(QDoubleSpinBox)
                 if not sb.signalsBlocked()
                 and abs(sb.value() - 8.0) < 1e-9]
        assert boxes, "gradient spinbox not found"
        return boxes[-1]

    win_g = lambda: win.grab().toImage()          # noqa: E731
    rec.start("110_edit")
    rec.hold(win_g, 1.5)
    gradient_spinbox().setValue(9.0)
    rec.hold(win_g, 3.0)
    win.state.bus.undo()
    rec.hold(win_g, 2.5)
    win.state.bus.redo()
    rec.hold(win_g, 2.5)
    rec.finish(scene_by_name("110_edit"))
    win.state.bus.undo()
    settle(4)
    report("after 110")

    # ---- 112 Add Element dialog --------------------------------------
    from linac_gen_gui.interphase.dialogs.add_element import AddElementDialog
    win.state.set_selected(quad3)
    settle(4)
    dlg = AddElementDialog(win)
    dlg.resize(640, 760)
    dlg.show()
    settle(8)
    dlg_g = lambda: compose((dlg.grab().toImage(),                # noqa: E731
                             (1920 - dlg.width() * 1.3) / 2, 20, 1.3))
    seq_g = crop_grab(lt._bottom_tabs)
    listing_canvas = lambda: compose((seq_g(), 60, 150, 1.55))    # noqa: E731
    rec.start("112_add_dialog")
    rec.hold(dlg_g, 2.0)
    dlg._type_combo.setCurrentText("Quadrupole")
    rec.hold(dlg_g, 2.5)
    dlg._type_combo.setCurrentText("Dipole")
    rec.hold(dlg_g, 2.5)
    dlg._type_combo.setCurrentText("Quadrupole")
    rec.hold(dlg_g, 1.0)
    rec.type_into(dlg_g, dlg._name, "QDEMO", chars_per_frame=1)
    rec.hold(dlg_g, 1.5)
    dlg._accept()
    settle(4)
    new_el = dlg.element()
    assert new_el is not None and new_el.name == "QDEMO"
    idx = next(i for i, e in enumerate(lattice.elements) if e is quad3)
    win.state.bus.do(InsertCommand(idx + 1, new_el))
    win.state.set_selected(new_el)
    rec.hold(listing_canvas, 3.5)
    rec.finish(scene_by_name("112_add_dialog"))
    win.state.bus.undo()
    settle(4)
    report("after 112")

    # ---- 114 palette drop with gap indicator -------------------------
    win.state.set_selected(quad3)
    settle(4)
    sb_g = crop_grab(win._statusbar, pad=0)

    def drop_canvas():
        t = union_grab([lt._chips, lt._timeline])()
        b = sb_g()
        return compose(
            (t, (1920 - t.width() * 1.55) / 2, 240, 1.55),
            (b, 0, 560, 1.0))

    rec.start("114_palette_drop")
    rec.hold(drop_canvas, 1.5)
    for gi in (4, 8, 12):
        lt._timeline._show_gap(gi)
        rec.hold(drop_canvas, 1.2)
    lt._timeline._hide_gap()
    lt._on_palette_drop("Steerer", 12)
    rec.hold(drop_canvas, 3.5)
    win.state.bus.undo()
    rec.hold(drop_canvas, 2.0)
    rec.finish(scene_by_name("114_palette_drop"))
    report("after 114")

    # ---- 117 context menus (captured BEFORE 116 so clipboard empty) --
    lt._clipboard = None
    win.state.set_selected(quad3)
    settle(4)
    menus = {}
    orig_menu_exec = QMenu.exec

    def fake_menu_exec(self, *a):
        self.popup(a[0] if a else self.mapToGlobal(QPoint(0, 0)))
        settle(10)
        menus[fake_menu_exec.key] = self.grab().toImage()
        self.close()
        settle(2)
        return None

    QMenu.exec = fake_menu_exec
    fake_menu_exec.key = "timeline"
    lt._timeline.customContextMenuRequested.emit(QPoint(400, 40))
    settle(4)
    fake_menu_exec.key = "listing"
    row = lt._listing._row_by_id.get(id(quad3))
    item = lt._listing._list.item(row)
    pos = lt._listing._list.visualItemRect(item).center()
    lt._listing._on_context_menu(pos)
    settle(4)
    QMenu.exec = orig_menu_exec
    assert "timeline" in menus and "listing" in menus
    tl_img = union_grab([lt._chips, lt._timeline])()
    seq_img = seq_g()
    m1 = menus["timeline"]
    m2 = menus["listing"]
    compose(
        (tl_img, 40, 100, 0.78),
        (m1, 260, 240, 2.0),
        (seq_img, 40, 600, 0.6),
        (m2, 900, 640, 2.0),
    ).save(s["117_context_menus"])

    # ---- 116 CRUD loop ------------------------------------------------
    sol1 = by_name(lattice, "SOL_001")
    mark1 = by_name(lattice, "MARK_001")
    rec.start("116_crud")
    win.state.set_selected(quad3)
    rec.hold(win_g, 1.5)
    lt._duplicate_selected()
    rec.hold(win_g, 3.0)
    lt._delete_selected()
    rec.hold(win_g, 2.5)
    win.state.bus.undo()
    rec.hold(win_g, 2.0)
    win.state.bus.redo()
    rec.hold(win_g, 2.0)
    win.state.set_selected(sol1)
    lt._copy_selected()
    rec.hold(win_g, 2.0)
    win.state.set_selected(mark1)
    lt._paste_clipboard()
    rec.hold(win_g, 3.5)
    rec.finish(scene_by_name("116_crud"))
    for _ in range(3):
        win.state.bus.undo()
    settle(4)
    report("after 116")

    # ---- 119 inline manual popup -------------------------------------
    from linac_gen_gui.interphase.dialogs.manual_popup import (ManualPopup,
                                                               open_inline)
    win.state.set_selected(quad3)
    settle(4)
    ok, msg = open_inline(quad3, parent=lt._inspector)
    assert ok, msg
    settle(10)
    pop = next(w for w in QApplication.topLevelWidgets()
               if isinstance(w, ManualPopup))
    insp_g = crop_grab(lt._inspector)

    def popup_canvas():
        pi = pop.grab().toImage()
        ii = insp_g()
        return compose((pi, 60, 60, 1.28),
                       (ii, 1370, 40, 1.24))

    psb = pop._view.verticalScrollBar()
    rec.start("119_manual_popup")
    rec.hold(popup_canvas, 2.5)
    pmax = psb.maximum()
    for i in range(1, 25):
        psb.setValue(int(pmax * i / 24))
        rec.tick(popup_canvas)
    rec.hold(popup_canvas, 2.5)
    rec.finish(scene_by_name("119_manual_popup"))
    pop.close()
    settle(4)

    # ---- 122 inspector variants --------------------------------------
    insp_view = lambda: compose((insp_g(), 760, 10, 1.3))     # noqa: E731
    rec.start("122_inspector_variants")
    lt._add_default_of_type("Aperture")          # inserted + selected
    rec.hold(insp_view, 2.5)
    combo = next(c for c in lt._inspector.findChildren(QComboBox))
    combo.setCurrentIndex(2)                     # 2 — Pepperpot
    rec.hold(insp_view, 2.5)
    win.state.set_selected(mark1)
    rec.hold(insp_view, 3.0)
    lt._add_default_of_type("Steerer")
    rec.hold(insp_view, 3.0)
    win.state.set_selected(by_name(lattice, "GAP_001"))
    rec.hold(insp_view, 3.5)
    rec.finish(scene_by_name("122_inspector_variants"))
    for _ in range(3):                # steerer insert, shape, aperture
        win.state.bus.undo()
    settle(4)
    report("after 122")

    # ---- 124 / 126 command cards -------------------------------------
    win.state.set_selected(quad3)
    settle(2)
    lt._add_default_of_type("SET_TWISS")
    settle(8)
    cur = lt._outline._tree.currentItem()
    if cur is not None:
        lt._outline._tree.scrollToItem(cur)
        settle(4)
    out_img = crop_grab(lt._outline)()
    insp_img = insp_g()
    compose((out_img, 220, 60, 1.15),
            (insp_img, 900, 10, 1.24)).save(s["124_command_card"])
    lt._add_default_of_type("ADJUST_BEAM_TWISS")
    settle(8)
    compose((insp_g(), 760, 10, 1.3)).save(s["126_adjust_flags"])
    win.state.bus.undo()
    win.state.bus.undo()
    settle(4)
    report("after 126")

    # ---- 132 REAL orbit correction -----------------------------------
    corr_deck = str(ROOT / "examples/correction_demo/correction_demo.dat")
    corr_lat = load_deck(corr_deck)
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=5.0, frequency=352.21, current=0.0,
        n_particles=2000))
    settle(8)
    lt._refresh_validation()
    settle(4)
    corr_info = {}
    orig_info = QMessageBox.information

    def fake_info(parent, title, text, *a, **k):
        corr_info["title"] = str(title)
        corr_info["text"] = str(text)
        return QMessageBox.StandardButton.Ok

    QMessageBox.information = staticmethod(fake_info)
    cq = by_name(corr_lat, "QUAD_001")
    rec.start("132_orbit_correction")
    rec.hold(win_g, 2.5)
    win.state.bus.do(ParamChangeCommand(cq, "dx", 0.0, 0.5))
    win.state.set_selected(cq)
    rec.hold(win_g, 2.5)
    lt._btn_correct.click()
    rec.wait_until(win_g,
                   lambda: lt._corr_worker is not None
                   and not lt._corr_worker.isRunning(),
                   cap_s=180)
    rec.wait_until(win_g, lambda: "text" in corr_info, cap_s=30)
    QMessageBox.information = orig_info
    assert "text" in corr_info, "correction result box never shown"
    print("[corr] title:", corr_info.get("title"))
    print("[corr] text:", corr_info.get("text"))
    box = QMessageBox(win)
    box.setWindowTitle(corr_info["title"])
    box.setText(corr_info["text"])
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.setModal(False)
    box.show()
    settle(8)

    def corr_canvas():
        base = win_g()
        b = box.grab().toImage()
        return compose((base, 0, 0, 1.0),
                       (b, (1920 - b.width() * 1.5) / 2, 220, 1.5))

    rec.hold(corr_canvas, 6.0)
    box.close()
    settle(4)
    rec.hold(win_g, 1.5)
    win.state.bus.undo()               # kicks
    rec.hold(win_g, 2.5)
    win.state.bus.undo()               # planted dx
    rec.hold(win_g, 2.0)
    rec.finish(scene_by_name("132_orbit_correction"))
    report("after 132")

    # ---- 134 BPM targets file ----------------------------------------
    variant = WORK / "correction_demo_targets.dat"
    src = Path(corr_deck).read_text()
    ops = iter(["DIAG_POSITION 1 0.2 -0.1",
                "DIAG_POSITION 2 0.15 0.05",
                "DIAG_POSITION 3 -0.1 0.1",
                "DIAG_POSITION 4 0.0 0.0"])
    variant.write_text("\n".join(
        next(ops) if ln.strip() == "DIAG_POSITION" else ln
        for ln in src.splitlines()) + "\n")
    tgt = WORK / "bpm_targets.txt"
    tgt.write_text("0.20 -0.10\nnan 0.05\n-0.10 0.10 2.0\n0.00 0.00\n")
    var_lat = load_deck(str(variant))
    lt._refresh_validation()
    settle(4)
    bpm2 = next(e for e in var_lat.elements
                if getattr(e, "is_bpm", False)
                and getattr(e, "diag_family", None) == 2)
    from PyQt6.QtWidgets import QFileDialog
    orig_open = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: ("bpm_targets.txt", ""))

    def targets_canvas():
        return compose((insp_g(), 660, 10, 1.28),
                       (sb_g(), 0, 1050, 1.0))

    cwd = os.getcwd()
    rec.start("134_bpm_targets")
    win.state.set_selected(bpm2)
    rec.hold(targets_canvas, 3.0)
    try:
        os.chdir(WORK)
        lt._on_load_bpm_targets()
    finally:
        os.chdir(cwd)
    rec.hold(targets_canvas, 2.5)
    win.state.set_selected(None)
    win.state.set_selected(bpm2)       # rebuild → override row appears
    rec.hold(targets_canvas, 4.0)
    rec.finish(scene_by_name("134_bpm_targets"))
    QFileDialog.getOpenFileName = orig_open
    print("[bpm] override on bpm2:",
          getattr(bpm2, "diag_target_override", None))

    # ---- 142 / 143 RfqCell → VaneRFQ ---------------------------------
    rfq_deck = str(ROOT / "examples/rfq_demo/rfq_demo.dat")
    rfq_lat = load_deck(rfq_deck)
    lt._refresh_validation()
    settle(4)
    from PyQt6.QtWidgets import QInputDialog
    orig_open2 = QFileDialog.getOpenFileName
    orig_item = QInputDialog.getItem
    QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (str(ROOT / "examples/lebt_plus_rfq/pxie-rfq.vane"),
                         ""))
    QInputDialog.getItem = staticmethod(
        lambda *a, **k: ("2term", True))
    rec.start("142_vane_swap")
    rec.hold(win_g, 3.0)
    lt._btn_to_vane.click()
    settle(10)
    rec.hold(win_g, 3.0)
    vane = next(e for e in rfq_lat.elements
                if type(e).__name__ == "VaneRFQ")
    win.state.set_selected(vane)
    rec.hold(win_g, 3.5)
    rec.finish(scene_by_name("142_vane_swap"))
    QFileDialog.getOpenFileName = orig_open2
    QInputDialog.getItem = orig_item
    print("[vane] n elements:", len(rfq_lat.elements),
          "model:", vane.field_model, "L:", vane.length)
    compose((seq_g(), 30, 200, 0.95),
            (insp_g(), 1250, 10, 1.28)).save(s["143_vane_inspector"])

    # ---- 144 / 146 parameter scan ------------------------------------
    lattice = load_deck(deck)
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=0.0))
    go_tab("lattice")
    settle(8)
    lt._refresh_validation()
    quad3 = by_name(lattice, "QUAD_003")
    win._open_parameter_scan()
    settle(10)
    sdlg = win._param_scan_dlg
    sdlg.resize(1050, 800)
    settle(6)
    sdlg_g = lambda: compose((sdlg.grab().toImage(),           # noqa: E731
                              (1920 - sdlg.width() * 1.3) / 2, 10, 1.3))
    qidx = next(i for i, e in enumerate(lattice.elements)
                if e is quad3)
    rec.start("144_param_scan")
    rec.hold(sdlg_g, 2.0)
    sdlg._element_combo.setCurrentIndex(qidx)
    rec.hold(sdlg_g, 1.5)
    pidx = next(i for i in range(sdlg._param_combo.count())
                if sdlg._param_combo.itemData(i) == "gradient")
    sdlg._param_combo.setCurrentIndex(pidx)
    rec.hold(sdlg_g, 1.5)
    sdlg._min_spin.setValue(6.0)
    sdlg._max_spin.setValue(10.0)
    sdlg._npts_spin.setValue(9)
    sdlg._mode_env.setChecked(True)
    sdlg._out_combo.setCurrentIndex(0)          # σ_x at exit
    rec.hold(sdlg_g, 2.0)
    sdlg._run_btn.click()
    rec.wait_until(sdlg_g,
                   lambda: sdlg._worker is not None
                   and not sdlg._worker.isRunning(),
                   cap_s=120)
    rec.hold(sdlg_g, 4.0)
    rec.finish(scene_by_name("144_param_scan"))
    print("[scan] status:", sdlg._status.text(),
          "nominal:", quad3.gradient)
    assert abs(quad3.gradient - 8.0) < 1e-12
    sdlg.close()
    settle(4)
    win.state.set_selected(quad3)
    settle(6)
    compose((insp_g(), 760, 10, 1.3)).save(s["146_scan_nominal"])

    # ---- 150 full-window recap ---------------------------------------
    win.state.set_selected(None)
    settle(4)
    report("before 152")

    # ---- 152 dirty chip ----------------------------------------------
    def bottom_g():
        img = win.grab().toImage()
        h = 300
        return img.copy(0, img.height() - h, img.width(), h)

    win.state.set_selected(quad3)
    settle(6)
    rec.start("152_dirty_chip")
    rec.hold(bottom_g, 1.5)
    gradient_spinbox().setValue(9.0)     # real inspector commit path
    rec.hold(bottom_g, 4.5)
    rec.finish(scene_by_name("152_dirty_chip"))

    # ---- 154 unsaved-changes prompt with details ---------------------
    prompt = {}
    orig_exec = QMessageBox.exec

    def fake_exec(self):
        self.show()
        settle(10)
        for b in self.findChildren(QPushButton):
            if "Details" in b.text():
                b.click()
                break
        settle(12)
        prompt["img"] = self.grab().toImage()
        self.close()
        settle(2)
        return QMessageBox.StandardButton.Cancel

    QMessageBox.exec = fake_exec
    proceed = win._confirm_discard("Open Lattice")
    QMessageBox.exec = orig_exec
    assert not proceed and "img" in prompt, "discard prompt not captured"
    pi = prompt["img"]
    scale = min(1.6, 980 / pi.height(), 1800 / pi.width())
    compose((pi, (1920 - pi.width() * scale) / 2, 40,
             scale)).save(s["154_unsaved_prompt"])
    win.state.bus.undo()
    settle(4)
    report("after 154")

    # ---- attach stills ------------------------------------------------
    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    # NO win.close(): dirty-state modal confirm hangs offscreen (ep07).


def main() -> None:
    render_manual_frames()
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
