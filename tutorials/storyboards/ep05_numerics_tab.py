"""Episode 5 — The Numerics tab in depth (solver settings + scans).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep05_numerics_tab.py
Output: tutorials/rendered/ep05_numerics_tab.mp4 (+ .srt)

Maximum-detail cut: every collapsible section narrated from its
tooltips and defaults, five live interaction clips (preset binding,
torch greyout, axis-default swapping, Apply-converged, cooperative
Stop), real recorded scans (single-axis grid, full Run All, an
n_particles noise scan), a diagnostics run whose snapshot and density
payoffs are proven in Results-tab popups, and four manual-tour scenes
rendered with QtWebEngine.  All spoken numbers were measured on this
machine (serial scans, seed 42) or verified in
gui/linac_gen_gui/interphase/tabs/convergence_tab.py.
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
# QtWebEngine (manual-tour scenes) — flags must be set before the
# QApplication exists, and the module must be imported before it too.
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --no-sandbox"
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep05_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep05_numerics_tab.mp4"

SCENES = [
    Scene("010_title",
          "Deep dive number three. The Numerics tab. How finely to "
          "step the machine, how to compute space charge, what the "
          "run records, and a built in convergence scanner that "
          "measures whether your choices are good enough. Today we "
          "exercise all of it, for real.",
          min_s=6.0),
    Scene("020_layout",
          "Seven collapsible sections, only Step density open by "
          "default, and each remembers its state. Read the group's "
          "subtitle. Simulation settings, used by Run Multi particle "
          "and as the baseline for non scanned axes. That is a "
          "contract, the toolbar run and the error study campaigns "
          "build their space charge configuration from this one "
          "panel. Below sit the scan parameters, then the run row, "
          "Run All Scans, Run Single Axis, Stop, a progress bar, a "
          "status badge, and Apply converged value, and a tabbed "
          "output pane, Log, Plot, and Table."),
    Scene("030_steps",
          "Step density. Step one, integration steps per metre, how "
          "finely elements are sliced. Step two, space charge kicks "
          "per metre. Defaults, one hundred steps and fifty kicks, "
          "range five to "
          "five thousand. One caveat outranks everything. These two "
          "numbers govern drifts and field maps only. The preset "
          "combo swaps profiles in one click. Production, one "
          "hundred over fifty. Matching, thirty over fifteen, about "
          "three times faster, for optimiser loops.",
          min_s=8.0),
    Scene("032_step_presets_live",
          "Watch it work. Matching writes both spinboxes, thirty and "
          "fifteen. Production restores the one hundred and the "
          "fifty. Edit "
          "step one by hand and the preset flips itself to Custom, "
          "so the label never disagrees with the loaded values. And "
          "when a deck loads, its step config auto selects the "
          "preset. One click, and Production is back."),
    Scene("035_partran_semantics",
          "The manual is precise here. Step one "
          "drives integration substeps for drift and field map "
          "elements. Step two sets the space charge kick cadence "
          "inside them. Quads, bends, and solenoids always get "
          "exactly two substeps with one mid plane kick, whatever "
          "you type. Raising step one refines drifts and field maps. "
          "It does not re slice a quadrupole."),
    Scene("040_sc_pic",
          "Space charge and P I C. Base grid, forty eight cells per "
          "axis of the cubic mesh. Grid extent, seven sigma. Both "
          "came from an M E B T convergence scan, finer grids are "
          "noise dominated, tighter extents clip halo and inflate "
          "losses. The P I C backend has five choices. "
          "Auto, C P U, G P U, cuda, M P S. Auto takes CUDA when "
          "available, else the C P U, and never auto selects "
          "M P S, which is single precision, about one part in ten "
          "million field error. Mps forces it with a warning, gpu "
          "and cuda force hardware or error.",
          min_s=8.0),
    Scene("045_torch_greyout",
          "The S C engine pairs production numpy with "
          "differentiable PyTorch. Pick torch and three "
          "controls grey out, grid mode, P I C backend, D C kernel, "
          "because torch ignores them. Grid mode snaps to adaptive, "
          "torch has no fixed grid path. It runs double precision on "
          "C P U, bunched beams only, D C beams fall back to "
          "numpy. Switch back, and your grid mode returns."),
    Scene("050_sc_models",
          "The physics choices. Green's function. I G F, the "
          "integrated Green function, is the accurate default, point "
          "is legacy. Particle to mesh. C I C is first order, fine "
          "where R F cavities reset grid noise. T S C, twenty "
          "seven cells at three times the work, for long no R F "
          "transport where C I C noise accumulates. Grid mode. Fixed builds "
          "the mesh once, adaptive rebuilds every kick, two to three "
          "times slower, for beams that outgrow the box. And for "
          "continuous beams the D C kernel, uniform, Gaussian, or "
          "pic two d per slice, all scaling the field with "
          "surviving current, so lossy beams are never overdriven.",
          min_s=8.0),
    Scene("055_model_map",
          "The manual maps this section to four models. Bunched "
          "envelope beams, the uniform ellipsoid kick. Bunched multi "
          "particle, the three dimensional Hockney F F T P I C. "
          "Continuous envelope, the Sacherer O D E. Continuous multi "
          "particle, the two dimensional D C kick. The decision tree "
          "is the whole story in seven short lines, and mismatching physics "
          "and model is the manual's top source of disagreement with "
          "TraceWin."),
    Scene("060_fieldmaps",
          "Field maps. Integrator. K D, kick drift, or D K D, drift "
          "kick drift, symplectic, for long or periodic "
          "trajectories. Interpolation, linear or cubic. Cubic "
          "builds its table at load time, so a change applies on the "
          "next lattice load. Sampling is implementation, not "
          "physics. Kernel, the fused C plus plus sampler, measured "
          "one point eight times faster on multi particle with space "
          "charge, two point nine on envelope, bitwise identical to "
          "scipy, kept for cross checks.",
          min_s=6.0),
    Scene("070_env_coll",
          "The envelope solver. Matrix, the default, propagates the "
          "sigma matrix and tracks cavities and longitudinal "
          "dynamics. Sacherer integrates the K V envelope equations, "
          "D C, no acceleration, better under space charge strong "
          "enough that the split operator under resolves. And "
          "collective effects holds one checkbox, C S R in bends. "
          "Let us read what it buys.",
          min_s=7.0),
    Scene("075_csr_wake",
          "The C S R model is a one dimensional "
          "steady state wake, the Saldin Schneidmiller Yurkov "
          "formulation. Tail radiation catches the head along the "
          "chord, and the bend's dispersion turns the energy kick "
          "into emittance growth, applied per substep inside every "
          "dipole. The caveats. Multi particle only, the envelope "
          "solver has no bunch profile. Steady state only, entrance "
          "and exit transients are not modelled. And beyond the "
          "checkbox, just two code knobs, bin count and model "
          "selector."),
    Scene("080_diag",
          "Diagnostics and recording. Record per sub step keeps the "
          "inside of every element, about fifty times more rows. "
          "Record particle density histograms the live beam along "
          "the machine, a megabyte per axis per thousand steps, "
          "feeding the Results heatmap. Snapshot "
          "every N dumps the full six dimensional cloud "
          "periodically, and zero means only flagged markers fire. Snapshot at "
          "names specific elements, or Add selected appends the "
          "current selection. Bins and extent size that grid. Now "
          "arm them.",
          min_s=7.0),
    Scene("085_diag_live_run",
          "Substeps on. Density on. And into Snapshot at, the name "
          "of a mid lattice quadrupole, exactly as the deck names "
          "it. Zoom out and run, twenty thousand macroparticles at "
          "five milliamps, Run Multi particle. The status bar "
          "confirms completion and the auto saved results file. "
          "Every switch was live for that run."),
    Scene("087_snapshot_payoff",
          "The phase space popup's location dropdown now offers the "
          "quadrupole we named, beside the default exit view, and "
          "redraws every panel from the six dimensional cloud "
          "captured there. Mid machine inspection, and free when "
          "unused."),
    Scene("088_density_payoff",
          "The density recording feeds this heatmap, beam density "
          "along the whole line. Two hundred bins default, four hundred for "
          "sharp beams at four times the memory. Extent zero auto "
          "fits to one point one times the largest excursion. Set it "
          "explicitly when two runs must share one colour scale."),
    Scene("090_scan_controls",
          "Scan controls. Scan N particles applies during scans "
          "only, the beam configuration keeps its own count, and "
          "five thousand matches the classic dialog. Parallel "
          "workers fans points across processes, half the cores "
          "capped at eight, seven here, with results identical to "
          "serial. The pool covers grid, extent, and particle "
          "count, the step axes stay serial, and G P U backends "
          "downgrade to C P U in pool workers. Our recordings run "
          "serial.",
          min_s=5.0),
    Scene("095_axis_defaults",
          "Five axes, and the values line follows. Thirty two to one "
          "twenty eight cells for grid. Three to six sigma extent. "
          "Fifty to five hundred step one, twenty five to two "
          "hundred step two. And ten thousand to three hundred "
          "thousand particles for the fifth, G U I only, noise "
          "floor axis. The field is free text, integers "
          "for count axes, floats otherwise."),
    Scene("100_scan_params",
          "The contract. Pick an axis, give a ladder, press a "
          "button. Run Single Axis scans one axis. Run All walks "
          "grid, extent, step one, step two with default ladders, "
          "and note, the baseline stays frozen, recommendations are "
          "committed together only at the end, so no winner "
          "contaminates later sweeps. Stop halts, Apply writes a "
          "winner back.",
          min_s=7.0),
    Scene("110_scan_output",
          "Scanning for real. Grid, sixteen to forty eight cells, "
          "five thousand scan particles, serial. The log is a lab "
          "notebook. Three header lines, space charge config, beam, "
          "with scan N beside the real twenty thousand, and step "
          "config. Then one row per point, sizes, emittances, wall "
          "seconds. The Plot draws epsilon x bold and sigma y "
          "dashed. Mind the auto zoomed axis, this curve spans under "
          "half a percent, a flat line magnified. The Table adds "
          "sigma phi in degrees and the wall time price sheet.",
          min_s=7.0),
    Scene("112_converged_rule",
          "The verdict, and the rule. One percent tolerance on "
          "ending horizontal emittance. The finest row, forty "
          "eight, is the reference, and the smallest value within "
          "one percent wins. Here every row "
          "agrees, so the cheapest wins, sixteen, highlighted, and "
          "the badge reads converged at grid equals sixteen, finest "
          "drift zero point three three percent, the gap between the "
          "two finest rows, your live error bar. And read it "
          "correctly, at five thousand particles this deck simply "
          "cannot tell sixteen from forty eight. Statistics "
          "limited, the fifth axis's job."),
    Scene("114_apply_converged",
          "Apply converged value writes the winner into the base "
          "settings. Watch the grid spinbox, forty eight becomes "
          "sixteen, and the badge appends, applied. Step axes also "
          "write into the lattice's step config, the particle axis "
          "updates the beam configuration. Receipt kept, and we put "
          "the grid straight back to forty eight."),
    Scene("116_stop_demo",
          "A heavier ladder, to exercise Stop. Rows land, we press "
          "Stop, and the badge answers in two stages. Stopping, "
          "finishing the current step, the worker bails at the next "
          "element boundary, never killed mid integration. Then "
          "stopped, with the count completed. The in flight point is "
          "discarded, but finished rows survive, and the log records the stop. No "
          "recommendation and nothing to apply from a truncated "
          "scan. Half a scan is not evidence."),
    Scene("118_run_all_start",
          "The full protocol, Run All Scans. The axis combo walks "
          "itself through the four axes while the log writes one "
          "header, config, beam, five thousand scan particles "
          "against twenty thousand real, steps, and the sweep plan. "
          "Rows stream with the wall seconds meter running. Note "
          "what it does not do. The baseline stays frozen through "
          "all four sweeps, each recommendation is stashed, not "
          "applied, so the extent sweep still runs on the original "
          "forty eight cell baseline. Everything commits at once, at "
          "the end."),
    Scene("119_run_all_result",
          "Done, and the badge is the verdict. Grid one twenty "
          "eight, extent six, step one fifty, step two twenty five, "
          "applied. The log closes with the classic recommendation "
          "block, and the base settings hold the committed values. "
          "Now grade it. The step sweeps are flat to a "
          "tenth of a percent, solid winners. The grid sweep never "
          "flattened, emittance climbs steadily across the ladder, "
          "so the rule fell back to the finest row. That is grid heating on a five thousand "
          "particle scan, not convergence. The extent column "
          "collapses by a factor of fifteen, tight extents clip "
          "halo. Trust a recommendation exactly as far as it "
          "flattened."),
    Scene("122_npart_noise_scan",
          "Back on the forty eight cell, seven sigma baseline, the "
          "fifth axis, statistics. We type a ladder, five hundred "
          "to thirty thousand, and watch the plot. Five hundred "
          "particles reads two point four eight, nine percent below "
          "the thirty thousand reference, under sampled tails read "
          "low. Two thousand overshoots by three percent. Eight "
          "thousand lands within one percent, and the badge calls "
          "it, converged at n particles equals eight thousand, "
          "finest drift zero point seven five percent, the one over "
          "root N floor sinking below tolerance. The wall clock "
          "barely notices here, but on production lattices this is "
          "the accuracy versus cost axis."),
    Scene("126_manual_page_tour",
          "One page to bookmark, the manual's Numerics tab chapter, "
          "with two behaviours no widget shows. Tooltips, every "
          "field carries the physics and the reference, the "
          "integrated Green function tooltip cites Qiang two "
          "thousand six. And dirty tracking, every fixed setting "
          "serialises into the project file and marks it dirty on "
          "change, while scan widgets stay session only and section "
          "states persist per user."),
    Scene("130_outro",
          "That is the Numerics tab at full depth. Scan, read the "
          "flattening and the drift, "
          "apply, re run at production statistics. When a result "
          "surprises you, double the steps, double the grid, and if "
          "nothing changes, believe the physics. Next, the Matching "
          "tab and its seven algorithms. See you there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def capture_visuals() -> None:
    # WebEngine import must precede QApplication construction.
    from PyQt6 import QtWebEngineWidgets                     # noqa: F401
    from PyQt6.QtCore import QEventLoop, QPoint, QTimer, QUrl, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import QApplication

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["ep05"])
    cards.title_card(s["010_title"], "The Numerics Tab",
                     "Deep dive — steps, space charge, and convergence scans")
    cards.outro_card(s["130_outro"], [
        "Scan an axis · read the drift · apply · re-run",
        "Next — Matching: all seven algorithms",
        "Then — the Results tab, tile by tile",
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

    def region(widget, pad: int = 8):
        """(x, y, w, h) of widget inside the main window grab."""
        tl = widget.mapTo(win, QPoint(0, 0))
        return (max(0, tl.x() - pad), max(0, tl.y() - pad),
                widget.width() + 2 * pad, widget.height() + 2 * pad)

    def crop_widget(key: str, widget, pad: int = 8) -> None:
        settle()
        img = win.grab().toImage()
        x, y, w, h = region(widget, pad)
        img.copy(x, y, min(img.width() - x, w),
                 min(img.height() - y, h)).save(s[key])

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    deck = str(ROOT / "examples/dtl_section.dat")
    lattice, _meta = _parse_lattice_file(deck)
    win.state.set_lattice(lattice, deck)

    # Beam used by everything in this episode: 3 MeV proton, 5 mA,
    # 20 000 macroparticles (the scans override N with Scan-N=5000).
    from linac_gen.core.config import BeamConfig
    win.state.set_beam_config(BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=5.0,
        n_particles=20000))

    go_tab("numerics")
    settle(10)

    ct = win.convergence_tab
    sections = ct._sections

    def open_only(*titles):
        for t, sec in sections.items():
            sec._toggle.setChecked(t in titles)
        settle(8)

    rec = Recorder(WORK / "frames", settle)

    def make_crop_grab(widget, pad: int = 8):
        def _grab():
            img = win.grab().toImage()
            x, y, w, h = region(widget, pad)
            return img.copy(x, y, min(img.width() - x, w),
                            min(img.height() - y, h))
        return _grab

    def span_grab_img(top_widget, bottom_widget, pad: int = 8):
        img = win.grab().toImage()
        xt, yt, wt, _ = region(top_widget, pad)
        xb, yb, wb, hb = region(bottom_widget, pad)
        x0 = min(xt, xb)
        w = max(wt, wb)
        y1 = yb + hb
        return img.copy(x0, yt, min(img.width() - x0, w),
                        min(img.height() - yt, y1 - yt))

    # ---------------- section stills -------------------------------
    open_only("Step density")
    win.grab().save(s["020_layout"])
    crop_widget("030_steps", sections["Step density"])

    # 032 — preset binding, live
    step_grab = make_crop_grab(sections["Step density"])
    rec.start("032_step_presets_live", fps=6.0)
    rec.hold(step_grab, 1.5)
    ct._step_preset.setCurrentText("Matching (30/15)")
    rec.hold(step_grab, 2.5)
    ct._step_preset.setCurrentText("Production (100/50)")
    rec.hold(step_grab, 2.0)
    ct._fixed_step1.setValue(120)          # hand edit -> Custom
    rec.hold(step_grab, 2.5)
    ct._step_preset.setCurrentText("Production (100/50)")
    rec.hold(step_grab, 2.0)
    rec.finish(scene_by_name("032_step_presets_live"))

    open_only("Space charge & PIC")
    crop_widget("040_sc_pic", sections["Space charge & PIC"])
    crop_widget("050_sc_models", sections["Space charge & PIC"])

    # 045 — torch greyout, live
    sc_grab = make_crop_grab(sections["Space charge & PIC"])
    rec.start("045_torch_greyout", fps=6.0)
    rec.hold(sc_grab, 1.5)
    ct._fixed_sc_backend.setCurrentText("torch")
    rec.hold(sc_grab, 3.5)
    ct._fixed_sc_backend.setCurrentText("numpy")
    rec.hold(sc_grab, 2.5)
    rec.finish(scene_by_name("045_torch_greyout"))

    open_only("Field maps (FieldMap3D)")
    crop_widget("060_fieldmaps", sections["Field maps (FieldMap3D)"])

    open_only("Envelope solver", "Collective effects")
    settle()
    span_grab_img(sections["Envelope solver"],
                  sections["Collective effects"]).save(s["070_env_coll"])

    open_only("Diagnostics & recording")
    crop_widget("080_diag", sections["Diagnostics & recording"])

    # ---------------- 085: diagnostics armed + real MP run ---------
    quads = [e for e in win.state.lattice.elements
             if type(e).__name__ == "Quadrupole"]
    snap_name = quads[len(quads) // 2].name        # mid-lattice quad
    print(f"[ep05] snapshot element: {snap_name}")

    diag_grab = make_crop_grab(sections["Diagnostics & recording"])
    full_grab = lambda: win.grab().toImage()             # noqa: E731
    rec.start("085_diag_live_run", fps=6.0)
    rec.hold(diag_grab, 1.0)
    ct._record_substeps.setChecked(True)
    rec.hold(diag_grab, 1.5)
    ct._record_density.setChecked(True)
    rec.hold(diag_grab, 1.5)
    rec.type_into(diag_grab, ct._snapshot_elements, snap_name)
    rec.hold(diag_grab, 1.5)
    rec.hold(full_grab, 1.0)                             # zoom out
    win._run_mp()
    rec.wait_until(full_grab, lambda: win.state.results is not None,
                   cap_s=180, stable_s=0.5)
    rec.hold(full_grab, 2.5)
    rec.finish(scene_by_name("085_diag_live_run"))
    assert win.state.results is not None, "085 MP run gave no results"

    # 087/088 — the payoff popups (top-level windows: grab directly)
    go_tab("results")
    settle(8)
    assert win.show_result_plot("phase")
    settle(10)
    pop = win.results_tab._popups.get("phase")
    pop.resize(1600, 900)
    settle(8)
    loc = pop._location
    idx = next((i for i in range(loc.count())
                if loc.itemText(i).startswith(snap_name)), None)
    assert idx is not None, f"snapshot {snap_name} not in location combo"
    loc.setCurrentIndex(idx)
    settle(10)
    pop.grab().save(s["087_snapshot_payoff"])
    pop.close()
    settle(2)

    assert win.show_result_plot("density_s")
    settle(10)
    pop = win.results_tab._popups.get("density_s")
    pop.resize(1600, 900)
    settle(10)
    pop.grab().save(s["088_density_payoff"])
    pop.close()
    settle(2)

    go_tab("numerics")
    settle(6)

    open_only("Scan controls")
    crop_widget("090_scan_controls", sections["Scan controls"])
    ct._parallel_workers.setValue(0)     # serial: rows in ladder order

    # ---------------- scan area ------------------------------------
    from PyQt6.QtWidgets import QGroupBox, QTabWidget
    scan_grp = next(g for g in ct.findChildren(QGroupBox)
                    if g.title() == "Scan parameters")
    out_tabs = next(t for t in ct.findChildren(QTabWidget)
                    if any(t.tabText(i) == "Log"
                           for i in range(t.count())))

    def show_out(name: str) -> None:
        for i in range(out_tabs.count()):
            if out_tabs.tabText(i) == name:
                out_tabs.setCurrentIndex(i)

    # 095 — axis combo walks all five axes, Values follows
    open_only()                                     # all collapsed
    params_grab = make_crop_grab(scan_grp)
    rec.start("095_axis_defaults", fps=6.0)
    rec.hold(params_grab, 1.5)
    from linac_gen_gui.interphase.tabs.convergence_tab import (
        AXIS_GRID, AXIS_EXTENT, AXIS_STEP1, AXIS_STEP2, AXIS_NPART)
    for ax in (AXIS_EXTENT, AXIS_STEP1, AXIS_STEP2, AXIS_NPART):
        ct._axis.setCurrentText(ax)
        rec.hold(params_grab, 2.2)
    rec.finish(scene_by_name("095_axis_defaults"))
    ct._axis.setCurrentText(AXIS_GRID)              # back to default
    settle(4)

    # 100 — scan params + run row still
    settle()
    span_grab_img(scan_grp, ct._run_all_btn).save(s["100_scan_params"])

    def scan_grab():
        return span_grab_img(scan_grp, out_tabs)

    # ---------------- 110: single-axis grid scan (serial) ----------
    ct._values.setText("16, 24, 32, 48")
    show_out("Log")
    settle(4)
    rec.start("110_scan_output", fps=6.0)
    rec.hold(scan_grab, 1.0)
    ct._start_scan()
    rec.wait_until(scan_grab, lambda: not ct._stop_btn.isEnabled(),
                   cap_s=300, stable_s=0.5)
    rec.hold(scan_grab, 2.0)
    show_out("Plot")
    rec.hold(scan_grab, 5.0)
    show_out("Table")
    rec.hold(scan_grab, 3.5)
    rec.finish(scene_by_name("110_scan_output"))

    # 112 — converged rule still: table + badge in one crop
    settle(4)
    scan_grab().save(s["112_converged_rule"])

    # 114 — apply on camera; SC section + run row in frame
    open_only("Space charge & PIC")
    apply_grab = lambda: span_grab_img(                  # noqa: E731
        sections["Space charge & PIC"], ct._run_all_btn)
    rec.start("114_apply_converged", fps=6.0)
    rec.hold(apply_grab, 1.5)
    ct._apply_btn.click()
    rec.hold(apply_grab, 3.5)
    ct._fixed_nx.setValue(48)                 # restore baseline
    rec.hold(apply_grab, 2.5)
    rec.finish(scene_by_name("114_apply_converged"))

    # ---------------- 116: cooperative Stop ------------------------
    open_only()
    show_out("Log")
    ct._values.setText("48, 64, 96, 128")
    settle(4)

    def row_done(i: int) -> bool:
        it = ct._table.item(i, 1)
        return it is not None and it.text() not in ("", "—")

    rec.start("116_stop_demo", fps=8.0)
    rec.hold(scan_grab, 1.0)
    ct._start_scan()
    rec.wait_until(scan_grab, lambda: row_done(1), cap_s=120)
    rec.hold(scan_grab, 0.5)
    ct._stop_btn.click()
    rec.wait_until(scan_grab,
                   lambda: ct._badge.text().startswith("stopped"),
                   cap_s=90, stable_s=0.3)
    rec.hold(scan_grab, 3.0)
    rec.finish(scene_by_name("116_stop_demo"))

    # ---------------- 118/119: Run All Scans -----------------------
    settle(4)
    rec.start("118_run_all_start", fps=6.0)
    rec.hold(scan_grab, 1.0)
    ct._start_run_all()
    # record through the grid sweep and into the extent sweep
    rec.wait_until(scan_grab,
                   lambda: ct._log.toPlainText().count("grid_ext  ") >= 2,
                   cap_s=240)
    rec.hold(scan_grab, 2.0)
    rec.finish(scene_by_name("118_run_all_start"))
    # finish off camera
    import time as _t
    t0 = _t.time()
    while (not ct._badge.text().startswith("all scans done")
           and _t.time() - t0 < 600):
        settle(10)
        _t.sleep(0.05)
    assert ct._badge.text().startswith("all scans done"), \
        f"run-all did not finish: {ct._badge.text()!r}"
    print(f"[ep05] run-all badge: {ct._badge.text()}")

    # 119 still — composite: full log on top (native scale), the
    # committed Step-density/SC sections bottom-left, badge bottom-right
    open_only("Step density", "Space charge & PIC")
    settle(8)
    sec_full = span_grab_img(sections["Step density"],
                             sections["Space charge & PIC"])
    sec_img = sec_full.copy(0, 0, min(sec_full.width(), 980),
                            sec_full.height())
    badge_img = ct._badge.grab().toImage()
    open_only()
    show_out("Log")
    settle(8)
    ct._log.verticalScrollBar().setValue(0)
    settle(4)
    log_full = ct._log.grab().toImage()
    log_img = log_full.copy(0, 0, min(log_full.width(), 1540),
                            log_full.height())
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    canvas.fill(QColor("#0b1220"))
    p = QPainter(canvas)
    if log_img.height() > 620:
        log_img = log_img.scaledToHeight(
            620, Qt.TransformationMode.SmoothTransformation)
    p.drawImage(12, 8, log_img)
    y2 = 8 + log_img.height() + 14
    rem = 1080 - y2 - 10
    if sec_img.height() > rem:
        sec_img = sec_img.scaledToHeight(
            rem, Qt.TransformationMode.SmoothTransformation)
    p.drawImage(12, y2, sec_img)
    b = badge_img
    if b.width() > 860:
        b = b.scaledToWidth(860, Qt.TransformationMode.SmoothTransformation)
    bx = 1920 - b.width() - 40
    by = y2 + 40
    p.fillRect(bx - 10, by - 10, b.width() + 20, b.height() + 20,
               QColor("#16233b"))
    p.drawImage(bx, by, b)
    p.end()
    canvas.save(s["119_run_all_result"])

    # ---------------- 122: n_particles noise scan ------------------
    # back to the measured baseline: 48 cells, 7 sigma, 100/50 steps
    ct._fixed_nx.setValue(48)
    ct._fixed_ext.setValue(7.0)
    ct._step_preset.setCurrentText("Production (100/50)")
    settle(4)
    show_out("Log")
    rec.start("122_npart_noise_scan", fps=6.0)
    rec.hold(scan_grab, 1.0)
    ct._axis.setCurrentText(AXIS_NPART)
    rec.hold(scan_grab, 1.5)
    rec.type_into(scan_grab, ct._values, "500, 2000, 8000, 30000")
    rec.hold(scan_grab, 1.0)
    ct._start_scan()
    rec.wait_until(scan_grab, lambda: not ct._stop_btn.isEnabled(),
                   cap_s=300, stable_s=0.5)
    rec.hold(scan_grab, 2.0)
    show_out("Plot")
    rec.hold(scan_grab, 5.0)
    show_out("Table")
    rec.hold(scan_grab, 3.5)
    rec.finish(scene_by_name("122_npart_noise_scan"))
    print(f"[ep05] npart badge: {ct._badge.text()}")

    # ---------------- WebEngine manual tours -----------------------
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    view = QWebEngineView()
    view.resize(1920, 1080)
    view.show()

    def pause_ms(ms: int) -> None:
        t = QTimer(); t.setSingleShot(True); t.start(ms)
        loop = QEventLoop(); t.timeout.connect(loop.quit); loop.exec()

    def web_load(rel: str) -> None:
        loop = QEventLoop()
        view.loadFinished.connect(loop.quit)
        view.load(QUrl.fromLocalFile(str(ROOT / "site" / rel)))
        loop.exec()
        try:
            view.loadFinished.disconnect(loop.quit)
        except Exception:                                # noqa: BLE001
            pass
        pause_ms(2200)                     # fonts / layout settle

    def web_js(code: str):
        loop = QEventLoop(); box = {}

        def cb(r):
            box["r"] = r
            loop.quit()
        view.page().runJavaScript(code, cb)
        loop.exec()
        return box.get("r")

    def web_y(elem_id: str) -> int:
        y = web_js(
            f"(function(){{var e=document.getElementById('{elem_id}');"
            f"return e?Math.round(e.getBoundingClientRect().top"
            f"+window.scrollY):null}})()")
        assert y is not None, f"id {elem_id} not found"
        return int(y)

    def web_strong_y(text: str) -> int:
        y = web_js(
            "(function(){var els=document.querySelectorAll('strong');"
            "for (var i=0;i<els.length;i++){"
            f"if(els[i].textContent.indexOf('{text}')>=0)"
            "{return Math.round(els[i].getBoundingClientRect().top"
            "+window.scrollY);}}return null})()")
        assert y is not None, f"strong {text!r} not found"
        return int(y)

    def web_max_scroll() -> int:
        return int(web_js(
            "document.body.scrollHeight - window.innerHeight"))

    web_grab = lambda: view.grab().toImage()             # noqa: E731

    def web_set_y(y: int) -> None:
        web_js(f"window.scrollTo(0, {max(0, y)})")

    def web_clip(key: str, plan, fps: float = 7.0) -> None:
        """plan: list of ('jump', y) | ('hold', s) | ('scroll', y)."""
        cur = 0
        rec.start(key, fps=fps)
        for op, val in plan:
            if op == "jump":
                cur = max(0, int(val)); web_set_y(cur)
                rec.tick(web_grab, 2)
            elif op == "hold":
                rec.hold(web_grab, float(val))
            else:                                        # smooth scroll
                target = max(0, int(val))
                step = 42 if target >= cur else -42
                while abs(target - cur) > abs(step):
                    cur += step
                    web_set_y(cur)
                    rec.tick(web_grab)
                cur = target
                web_set_y(cur)
                rec.tick(web_grab, 2)
        rec.finish(scene_by_name(key))

    # 035 — PARTRAN_STEP semantics (convergence guide)
    web_load("05_space_charge/05_convergence.html")
    y_sem = web_strong_y("PARTRAN_STEP semantics")
    web_clip("035_partran_semantics", [
        ("jump", 140), ("hold", 3.0),
        ("scroll", y_sem - 200), ("hold", 5.0),
    ])

    # 055 — four models + decision tree
    web_load("05_space_charge/01_models.html")
    y_tree = min(web_y("decision-summary") - 120, web_max_scroll())
    web_clip("055_model_map", [
        ("jump", 240), ("hold", 3.5),
        ("scroll", y_tree), ("hold", 5.0),
    ])

    # 075 — CSR section (same page).  End with the caveat bullets and
    # config table BELOW the sticky chrome (~110 px): bullets-140.
    y_csr = web_y("coherent-synchrotron-radiation-csr")
    y_bullets = web_strong_y("Multi-particle only.")
    web_clip("075_csr_wake", [
        ("jump", y_csr - 160), ("hold", 3.0),
        ("scroll", y_bullets - 140), ("hold", 5.0),
    ])

    # 126 — GUI chapter: tooltips + dirty tracking
    web_load("10_gui/04_convergence_tab.html")
    y_dirty = min(web_y("tooltips-with-formula-references") - 150,
                  web_max_scroll())
    web_clip("126_manual_page_tour", [
        ("jump", 60), ("hold", 2.5),
        ("scroll", web_y("sections") - 150), ("hold", 3.0),
        ("scroll", y_dirty), ("hold", 5.5),
    ])
    view.hide()

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
    os._exit(0)


if __name__ == "__main__":
    main()
