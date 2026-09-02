"""Episode 7 — The Results tab in depth (maximum-detail cut).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep07_results_tab.py
Output: tutorials/rendered/ep07_results_tab.mp4 (+ .srt)

Every frame carries genuine data from real runs made inside this build:

* matched FODO (MP, 2500 macroparticles)      — the healthy wall
* DTL overloaded at 20 mA (MP, 2500)          — losses, watts, phase space
* Hofmann demo linac (envelope, 5 mA)         — tunes, stability, footprint
* DC H- LEBT (envelope, 15 mA, substeps)      — space-charge compensation
* dtl_section with an H- beam (MP, 5 mA)      — the species-gated stripping tiles
* 90-degree bend line (MP, 2000)              — dispersion family
* orbit-correction FODO, one steerer kicked   — centroid goals + BPM table
* synthetic 6-cell cavity map (MP, 800)       — field-map + TTF viewers
* matching_demo (envelope + a real match)     — live match preview

Modal traps (documented forever-hangs offscreen) are disarmed up front:
QMessageBox statics are replaced by logging stubs and every QFileDialog
static is monkeypatched to return a scripted path.  The sigma/tmatrix
cards are never clicked (their routes end in .exec()); the dialogs are
constructed directly and .show()n.  The SCC "Apply to lattice" button is
never clicked.  NO win.close() at the end (unsaved-changes modal).
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
if "QT_PLUGIN_PATH" not in os.environ:
    import PyQt6
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt6.__file__), "Qt6", "plugins")

from pipeline.render import Scene, build_video          # noqa: E402

WORK = ROOT / "tutorials" / "rendered" / "ep07_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep07_results_tab.mp4"
SCRATCH = Path(tempfile.mkdtemp(prefix="ep07_"))

SCENES = [
    Scene("010_title",
          "Deep dive number five. The Results tab in full. One wall, "
          "forty seven tiles, and the popup behind every one, fed by "
          "seven real machines. From the matched F O D O and the "
          "overloaded D T L to a ninety degree bend, an orbit "
          "correction line, and a D C H minus transport.",
          min_s=7.0),
    Scene("020_wall",
          "The wall after the healthy F O D O run. On top, six live "
          "numbers, sizes to loss. Import Results, top right, loads "
          "a saved run back. Then nine sections. Beam size and "
          "emittance. Twiss, divergence and halo. Energy and "
          "kinematics. Losses and transmission. Centroid and "
          "dispersion. Phase space and diagnostics. Lattice "
          "parameters. Cross checks against TraceWin. And the "
          "matrix viewers.",
          min_s=9.0),
    Scene("023_tile_anatomy",
          "Every card shares one anatomy. Icon, name, a live "
          "sparkline along the machine, and a footer with the end "
          "value. The arrow compares end against start. Up, down, "
          "or flat within half a percent. A dash is honest "
          "emptiness. The error study tile stays blank until an "
          "ensemble exists. And intra beam stripping applies to H "
          "minus only, so on this proton run its tile sits empty "
          "and disabled.",
          min_s=7.0),
    Scene("030_rms",
          "The R M S beam size plot. "
          "Horizontal, vertical, and the bunch length, with the "
          "lattice strip on top. Show aperture draws the vacuum chamber on "
          "the same axes. Show lattice toggles the strip. The "
          "Display dropdown switches between raw and dispersion "
          "corrected betatron sizes. Escape closes a popup. Control "
          "S saves it.",
          min_s=7.0),
    Scene("035_emit",
          "Emittance, two windows tiled side by side. Left, the "
          "geometric emittances, with the four D invariant overlaid "
          "on the horizontal panel. Right, the normalised "
          "emittances, which divide out acceleration and should "
          "stay flat under clean transport. The honest health "
          "indicator, and here they end where they started.",
          min_s=7.0),
    Scene("040_advanced_emittances",
          "Three more emittance views. The six dimensional "
          "emittance, the product of the three eigen emittances, a "
          "volume no linear symplectic map can change. The four D "
          "invariant, conserved even when transverse coupling makes "
          "the plane projections oscillate. And the eigen "
          "emittances themselves. Uncoupled "
          "here, they match the plane emittances. A solenoid or "
          "skew error splits them.",
          min_s=7.0),
    Scene("045_twiss_family",
          "The Twiss family. Alpha and beta per plane, with the "
          "same Display dropdown. Longitudinal Twiss, from the "
          "sigma matrix in the delta phi delta W convention. And "
          "divergence, sigma x prime and sigma y prime, straight "
          "off the sigma matrix diagonal.",
          min_s=6.0),
    Scene("050_energy",
          "Energy and kinematics, now on the D T L, overloaded at "
          "twenty milliamps. Beam power shares the energy section. Current times energy "
          "times transmission. This run enters at sixty kilowatts "
          "and leaves with about twenty three, the sparkline "
          "falling with every loss even as the energy climbs.",
          min_s=7.0),
    Scene("055_losses",
          "Losses. The cumulative profile first. Over eighty "
          "percent of the beam is gone before the exit. Then the "
          "aperture map, the autopsy. The orange line is the vacuum "
          "chamber, every yellow dot one recorded particle death, "
          "in both planes. Two thousand and eighty four deaths "
          "across twenty three elements, priced in the table, "
          "nearly fifty eight kilowatts in total.",
          min_s=8.0),
    Scene("058_loss_power",
          "The loss power popup prices the record. Lineal loss "
          "density in watts per metre, energies taken at the loss "
          "point, against the dashed one watt per metre hands on "
          "criterion. This run peaks near fifty eight thousand, "
          "four orders of magnitude above it. Below, the surviving "
          "beam's power density on the terminal plane. Over a "
          "hundred kilowatts per square centimetre at the peak.",
          min_s=7.0),
    Scene("060_halo_peak",
          "Two tail diagnostics. The halo parameter H, fourth "
          "moment over second moment squared, minus one. Zero for "
          "a uniform core, two for a Gaussian, the dashed lines. H "
          "falls here, from about one point five toward zero point "
          "nine, because the aperture is eating the tails. H says "
          "the shape changed, the loss map says why. Beside it, "
          "peak excursion, the outermost survivor against the "
          "chamber. Its Display dropdown only affects the envelope "
          "fallback.",
          min_s=7.5),
    Scene("065_phase_space",
          "Phase space, the microscope. Snapshots were recorded "
          "every four elements, so the location selector scrubs "
          "the actual particles along the machine. Watch the bunch "
          "shear and tear. Colour by energy, particle index, or "
          "transverse radius. The basis selector flips to the "
          "TraceWin z delta convention. Fold phi wraps neighbouring "
          "R F buckets back onto the bunch. And Beam parameters "
          "swaps the panels for a full table of the selected "
          "distribution.",
          min_s=8.0),
    Scene("070_density",
          "The density map. Every live particle histogrammed along "
          "the machine. The beam broadens and thins as it is "
          "scraped. The axis dropdown walks positions, angles, "
          "phase, and energy. The sigma overlay draws the R M S "
          "envelope, the aperture overlay the chamber. And log "
          "scale keeps the faint halo visible. Switch it off and "
          "the tails vanish.",
          min_s=7.0),
    Scene("075_lattice_params",
          "Lattice parameters plot the machine itself, before any "
          "run. Quadrupole gradients and their integrated "
          "strength. R F voltage per gap. Synchronous phase, minus "
          "thirty degrees everywhere here. Peak solenoid field and "
          "the focusing integral B squared d z, from the two "
          "solenoids up front. The dipole tiles and floor plan "
          "wait for the bend line.",
          min_s=7.5),
    Scene("080_channel_tunes",
          "Now the space charge diagnostics, on a forty six period "
          "linac, envelope at five milliamps, phase probe on. "
          "Sigma zero is the bare focusing. Sigma is what the beam "
          "gets under space charge. The period picker locked onto "
          "the first section's bracket, sixteen seven element "
          "cells, and the caption reads its bare cell. Forty point "
          "two degrees transverse, twenty four point nine "
          "longitudinal. Tune depression compresses that to eta "
          "per cell, sigma over sigma zero. The green channel "
          "model markers put eta near 0.76 transverse and 0.69 "
          "longitudinal, and the caption flags that the beam "
          "markers assume a matched beam.",
          min_s=10.0),
    Scene("082_hofmann",
          "The Hofmann stability chart. Depressed tune ratio k z "
          "over k x against depression k x over k zero x, one dot "
          "per cell. "
          "Compute chart solves the corrected anisotropic K V "
          "dispersion relations for modes two, three and four. "
          "Green cells, valid and stable. Red, flagged above one "
          "percent growth. Grey hollow, outside the perturbative "
          "gate. An amber ring, fold risk. Here exactly one of "
          "thirteen valid cells is flagged. The pink bands are the "
          "legacy heuristic overlay.",
          min_s=9.0),
    Scene("084_footprint",
          "The tune footprint. The cell is re tracked with the "
          "space charge field frozen from the first pass, each "
          "test particle's tune from an F F T. Turns sets "
          "resolution, one twenty eight gives about 2.8 degrees. "
          "Particles sets the amplitude ladder. Small amplitudes "
          "feel the full force and sit deepest below the bare "
          "tune. The caption reads core tunes near 11.3 degrees "
          "per cell.",
          min_s=8.0),
    Scene("086_scc",
          "A D C H minus beam traps an ion cloud that partially "
          "neutralises its own space charge. This popup computes "
          "the compensation degree from gas physics. Pick the gas "
          "and the pressure on the log slider. Computed mode "
          "solves the Poisson Boltzmann balance. Assumed mode "
          "takes your eta. Compute fills the compensation profile, "
          "plasma potential, and build up time. "
          "Export cards copies suggested space charge comp factors "
          "to the clipboard. Apply to lattice is confirm gated.",
          min_s=9.0),
    Scene("088_h_minus",
          "The species gate flips. H minus's second "
          "electron strips in collisions inside the bunch and in "
          "strong magnetic fields. Run an H minus beam, five "
          "milliamps through the quadrupole line, and the intra "
          "beam stripping card comes alive. The popup implements "
          "Lebedev's model. The measured total, under two parts "
          "per million, rounds to zero watts. Magnetic stripping "
          "evaluates the field particles actually sample, at two "
          "sigma, one sigma, or pole tip in the quadrupoles. Here, "
          "peak field 0.043 tesla, loss rate exactly zero, which "
          "is what you run it to confirm.",
          min_s=9.0),
    Scene("090_ensemble",
          "The ensemble tile says what many imperfect copies of "
          "this machine would do. Behind it is a real "
          "ten seed Monte Carlo on the F O D O. Three percent "
          "gradient errors, 0.15 millimetres of misalignment. "
          "Solid line, the seed mean. Dashed, the one sigma band. "
          "Below, final transmission across seeds. All ten "
          "transmit fully. The errors moved the optics, not the "
          "beam into the wall.",
          min_s=7.0),
    Scene("092_bend_dispersion",
          "The bend line. Ninety degrees in two arcs, and bends "
          "make dispersion. The solid curves are statistical "
          "dispersion, from the tracked beam's sigma matrix. Tick "
          "Transfer matrix model and a background worker "
          "propagates the design dispersion element by element. "
          "The dashed overlay lands on the measured curve, peaking "
          "near two and a half metres. Sigma delta p over p stays "
          "flat. Bends re arrange trajectories, not momenta, and "
          "that flat spread is exactly what the dispersion "
          "multiplies. And the Display "
          "dropdown earns its keep. Raw sigma x includes the "
          "dispersive term. Corrected strips it, and the exit size "
          "drops from 8.6 to 4.1 millimetres.",
          min_s=10.0),
    Scene("094_bend_survey",
          "The same machine as hardware. The dipole field tile "
          "turns each bend's geometry into tesla through the beam "
          "rigidity. Integral B d l is the bending strength a "
          "magnet engineer quotes. And the floor plan is a genuine "
          "survey, aspect locked, both arcs highlighted, entrance "
          "and exit marked. Four point five four metres of path, "
          "ninety degrees of bend.",
          min_s=6.5),
    Scene("096_orbit_bpms",
          "The orbit correction demo. Four monitors, four "
          "steerers. We kicked the first steerer on purpose and "
          "set every monitor's target to zero. The centroid popup "
          "shows the orbit walking off, the goal orbit as hollow "
          "points, and the banner quotes the achieved versus goal "
          "R M S gap. About 0.9 millimetres in the kicked plane. "
          "The B P M table reads the same run through the "
          "lattice's monitor flags, and longitudinal offset "
          "repeats it in phase and energy.",
          min_s=8.0),
    Scene("098_partran",
          "Cross checks. The compare popup overlays any TraceWin "
          "partran dot out file on the current run. Solid HELIX, "
          "dashed file, and the lower panel is the per step "
          "relative difference. Seven axes. Three sizes, three "
          "emittances, energy. This overlay is a second HELIX run "
          "at lower particle count exported in the partran format, "
          "so the residuals are statistics and sampling, not "
          "physics. Clear overlay drops it.",
          min_s=7.5),
    Scene("100_fieldmap_ttf",
          "Field maps, on a synthetic six cell cavity map "
          "generated for the tutorial. Real cavity maps plug in "
          "the same way. Pick the element, the channel, here R F "
          "electric, two dimensional cylindrical, the component, "
          "and the cut plane. Sliders slice the map, and the "
          "crosshair ties the line cuts to the heatmap. The "
          "transit time factor popup computes the phase optimised "
          "T of beta. The dashed line marks the map's optimum "
          "beta, and the dotted line, the beam's actual beta, "
          "sits essentially on top of it.",
          min_s=8.5),
    Scene("103_matrix_viewers",
          "The advanced cards are routers, not popups. The sigma "
          "matrix viewer shows the full six by six second moment "
          "matrix at any recorded step, in our basis or "
          "TraceWin's. Every plot in this tab is a slice of it. "
          "The transfer matrix dialog multiplies element matrices "
          "between any two indices. And the S C convergence card "
          "jumps straight to the Numerics tab.",
          min_s=7.0),
    Scene("106_popup_machinery",
          "Popups are windows, not modes. Open as many as you "
          "like, and every visible one refreshes the moment a run "
          "completes. Watch. We re run the F O D O. Both windows "
          "clear while the tracker works, then repopulate "
          "together. That sync is always on. And the Show lattice "
          "checkbox toggles the orientation strip, off and back "
          "on.",
          min_s=6.5),
    Scene("108_live_preview",
          "One channel is opt in. Every popup carries a live "
          "match preview checkbox. Tick it, start a real Matching "
          "tab optimisation, and the plot re draws with the "
          "matcher's current iterate as it works. Watch the "
          "envelope change while the quadrupole is tuned. The "
          "preview never overwrites committed results. When the "
          "match ends, the popup snaps back.",
          min_s=6.5),
    Scene("110_export",
          "Every popup exports through one dialog, on Control S. "
          "C S V, numpy, JSON, or H D F five for data. P N G, "
          "J P E G, S V G, or P D F for the figure. Here the beam "
          "size data lands as C S V, the figure as P N G. The "
          "toolbar exports the whole run as open P M D, the "
          "portable particle standard, and Import Results reads "
          "it straight back. The header and the sparklines "
          "repopulate from disk, and anything the file does not "
          "carry stays honestly blank.",
          min_s=8.0),
    Scene("120_outro",
          "That is the Results tab at full depth. Nine sections, "
          "forty seven tiles, a window for every question a run "
          "can raise. Sizes, tunes, losses in watts, stripping "
          "physics, dispersion, cross checks, and the sigma "
          "matrix under it all. Run, read the wall, open what "
          "matters, export what you need. Other episodes give the "
          "campaign tabs and the assistant the same depth. See "
          "you there.",
          min_s=7.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def capture_visuals() -> None:
    from PyQt6.QtCore import QEventLoop, QPoint, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import (QApplication, QFileDialog, QMessageBox,
                                 QPushButton, QScrollArea)

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["ep07"])

    # ------------------------------------------------------------------
    # Modal-dialog disarm.  Every QMessageBox static becomes a logging
    # stub (an unexpected dialog would otherwise hang the offscreen
    # build forever); QFileDialog statics return whatever path the
    # current segment scripted into _DIALOG_PATHS.
    # ------------------------------------------------------------------
    _DIALOG_PATHS = {"open": "", "save": ""}

    def _log_box(kind):
        def f(*a, **k):
            print(f"[msgbox-{kind}]", " | ".join(str(x) for x in a[1:3]))
            return QMessageBox.StandardButton.Ok
        return staticmethod(f)
    for kind in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, kind, _log_box(kind))
    QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (_DIALOG_PATHS["open"], ""))
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (_DIALOG_PATHS["save"], ""))

    cards.title_card(s["010_title"], "The Results Tab",
                     "Maximum-detail cut — 47 tiles, every popup")
    cards.outro_card(s["120_outro"], [
        "Next — the Param Study manager",
        "Then — Error Study, Failure Study, Surrogates",
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

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    # ------------------------------------------------------------------
    # Compositing helpers — popups are TOP-LEVEL windows; win.grab()
    # never contains them, so stills/clips that need several windows
    # paint their individual grabs onto a 1920x1080 canvas.
    # ------------------------------------------------------------------
    BG = QColor("#0b1220")

    def montage(images, vertical=False, pad=14):
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(BG)
        p = QPainter(canvas)
        n = len(images)
        if vertical:
            avail_h = 1080 - pad * (n + 1)
            scaled = []
            for img in images:
                sc_img = img.scaledToHeight(
                    min(int(avail_h / n), img.height()),
                    Qt.TransformationMode.SmoothTransformation)
                if sc_img.width() > 1920 - 2 * pad:
                    sc_img = img.scaledToWidth(
                        1920 - 2 * pad,
                        Qt.TransformationMode.SmoothTransformation)
                scaled.append(sc_img)
            total_h = sum(i.height() for i in scaled) + pad * (n - 1)
            y = (1080 - total_h) // 2
            for img in scaled:
                p.drawImage((1920 - img.width()) // 2, y, img)
                y += img.height() + pad
        else:
            avail_w = 1920 - pad * (n + 1)
            scaled = []
            for img in images:
                sc_img = img.scaledToWidth(
                    min(int(avail_w / n), img.width()),
                    Qt.TransformationMode.SmoothTransformation)
                if sc_img.height() > 1080 - 2 * pad:
                    sc_img = img.scaledToHeight(
                        1080 - 2 * pad,
                        Qt.TransformationMode.SmoothTransformation)
                scaled.append(sc_img)
            total_w = sum(i.width() for i in scaled) + pad * (n - 1)
            x = (1920 - total_w) // 2
            for img in scaled:
                p.drawImage(x, (1080 - img.height()) // 2, img)
                x += img.width() + pad
        p.end()
        return canvas

    def open_popup(key, size=(1600, 900)):
        assert win.show_result_plot(key), f"no plot {key}"
        settle(10)
        pop = win.results_tab._popups.get(key)
        pop.resize(*size)
        settle(8)
        return pop

    def popup_shot(scene_key, plot_key, size=(1600, 900)):
        pop = open_popup(plot_key, size)
        pop.grab().save(s[scene_key])
        pop.close()
        settle(2)

    from linac_gen.core.config import BeamConfig
    import numpy as np

    ct = win.convergence_tab
    ct._record_density.setChecked(True)
    ct._snapshot_every_n.setValue(4)

    def load(deck_rel):
        deck = str(deck_rel) if os.path.isabs(str(deck_rel)) \
            else str(ROOT / deck_rel)
        lattice, _meta = _parse_lattice_file(deck)
        win.state.set_lattice(lattice, deck)
        return lattice

    def run_mp(cfg: "BeamConfig") -> None:
        win.state.set_results(None)
        win.state.set_beam_config(cfg)
        win._run_mp()
        t0 = time.time()
        while win.state.results is None and time.time() - t0 < 600:
            settle(10)
            time.sleep(0.05)
        assert win.state.results is not None, "mp run gave no results"
        settle(10)

    def run_env(cfg: "BeamConfig") -> None:
        win.state.set_results(None)
        win.state.set_beam_config(cfg)
        win._run_envelope()
        t0 = time.time()
        while win.state.results is None and time.time() - t0 < 600:
            settle(10)
            time.sleep(0.05)
        assert win.state.results is not None, "envelope run gave no results"
        settle(10)

    def wait_worker(pop, cap_s=180):
        t0 = time.time()
        while getattr(pop, "_worker", None) is not None \
                and pop._worker.isRunning() and time.time() - t0 < cap_s:
            settle(10)
            time.sleep(0.05)
        settle(15)

    rec = Recorder(WORK / "frames", settle)

    # ==================================================================
    # RUN A — the matched FODO from ep01 (2500 macroparticles, healthy)
    # ==================================================================
    fodo = load("examples/fodo_cell.dat")
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.matching import (find_fodo_cells,
                                    find_matched_input_twiss)
    ref = ReferenceParticle(species=PROTON, w_kin=1.0, frequency=352.21)
    tw = find_matched_input_twiss(fodo, ref, *find_fodo_cells(fodo)[0])
    cfg_fodo = BeamConfig(
        species="proton", energy=1.0, frequency=352.21, current=0.0,
        n_particles=2500,
        alpha_x=tw["alpha_x"], beta_x=tw["beta_x"],
        alpha_y=tw["alpha_y"], beta_y=tw["beta_y"])
    run_mp(cfg_fodo)
    go_tab("results")
    settle(8)
    res = win.state.results
    enx = np.asarray(res.emit_nx, dtype=float)
    print(f"[measure] FODO emit_nx {enx[0]:.4f} -> {enx[-1]:.4f} "
          f"({(enx[-1]/enx[0]-1)*100:+.2f}%)")
    for key in ("rms", "ensemble", "ibs"):
        card = next(c for c in win.results_tab._cards if c._key == key)
        print(f"[measure] card {key}: value={card._value_lbl.text()!r} "
              f"trend={card._trend_lbl.text()!r} enabled={card.isEnabled()}")

    # --- 020: the wall — KPI strip, then a REAL scroll through all
    # nine sections, then the click-a-tile composite with the RMS popup.
    scroll = win.results_tab.findChildren(QScrollArea)[0]
    vbar = scroll.verticalScrollBar()
    vbar.setValue(0)
    settle(6)

    def wall_grab():
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(BG)
        p = QPainter(canvas)
        wimg = win.grab().toImage().scaledToWidth(
            1560, Qt.TransformationMode.SmoothTransformation)
        p.drawImage(0, (1080 - wimg.height()) // 2, wimg)
        pop = win.results_tab._popups.get("rms")
        if pop is not None and pop.isVisible():
            pimg = pop.grab().toImage().scaledToWidth(
                980, Qt.TransformationMode.SmoothTransformation)
            px, py = 1920 - pimg.width() - 12, 1080 - pimg.height() - 60
            p.fillRect(px - 5, py - 5, pimg.width() + 10,
                       pimg.height() + 10, QColor("#2b3b55"))
            p.drawImage(px, py, pimg)
        p.end()
        return canvas

    rec.start("020_wall")
    rec.hold(wall_grab, 5.0)            # KPI strip + hint + import button
    vmax = vbar.maximum()
    steps = 30
    for i in range(1, steps + 1):       # genuine scroll through 9 sections
        vbar.setValue(int(vmax * i / steps))
        rec.hold(wall_grab, 0.55)
    rec.hold(wall_grab, 2.0)
    vbar.setValue(0)
    rec.hold(wall_grab, 1.5)
    assert win.show_result_plot("rms")
    settle(8)
    win.results_tab._popups["rms"].resize(1200, 680)
    rec.hold(wall_grab, 4.5)
    rec.finish(scene_by_name("020_wall"))
    win.results_tab._popups["rms"].close()
    settle(4)

    # --- 023: tile anatomy — three genuine card states, enlarged.
    card_imgs = []
    for key in ("rms", "ensemble", "ibs"):
        card = next(c for c in win.results_tab._cards if c._key == key)
        img = card.grab().toImage().scaledToWidth(
            560, Qt.TransformationMode.SmoothTransformation)
        card_imgs.append(img)
    montage(card_imgs).save(s["023_tile_anatomy"])

    # --- 030: RMS popup.
    popup_shot("030_rms", "rms")

    # --- 035: geometric + normalised emittance side by side.
    p1 = open_popup("emit", (940, 760))
    p2 = open_popup("emit_n", (940, 760))
    montage([p1.grab().toImage(), p2.grab().toImage()]).save(s["035_emit"])
    p1.close(); p2.close(); settle(2)

    # --- 040: 6-D / 4-D / eigenemittance popups stacked.
    p1 = open_popup("emit6d", (1250, 380))
    p2 = open_popup("emit4d", (1250, 380))
    p3 = open_popup("eigenemit", (1250, 380))
    montage([p1.grab().toImage(), p2.grab().toImage(),
             p3.grab().toImage()], vertical=True).save(
        s["040_advanced_emittances"])
    p1.close(); p2.close(); p3.close(); settle(2)

    # --- 045: Twiss + longitudinal Twiss + divergence.
    p1 = open_popup("twiss", (900, 900))
    p2 = open_popup("long_twiss", (900, 560))
    p3 = open_popup("divergence", (900, 440))
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    canvas.fill(BG)
    p = QPainter(canvas)
    limg = p1.grab().toImage().scaledToHeight(
        1050, Qt.TransformationMode.SmoothTransformation)
    p.drawImage(12, 15, limg)
    rx = limg.width() + 28
    rw = 1920 - rx - 12
    img2 = p2.grab().toImage().scaledToWidth(
        rw, Qt.TransformationMode.SmoothTransformation)
    img3 = p3.grab().toImage().scaledToWidth(
        rw, Qt.TransformationMode.SmoothTransformation)
    y0 = (1080 - img2.height() - img3.height() - 14) // 2
    p.drawImage(rx, y0, img2)
    p.drawImage(rx, y0 + img2.height() + 14, img3)
    p.end()
    canvas.save(s["045_twiss_family"])
    p1.close(); p2.close(); p3.close(); settle(2)

    # --- 103: the classic matrix dialogs (constructed directly — the
    # card routes end in .exec(), a documented offscreen hang) + the
    # SC-convergence card genuinely routing to the Numerics tab.
    from linac_gen_gui.dialogs.sigma_matrix_dialog import SigmaMatrixDialog
    from linac_gen_gui.dialogs.transfer_matrix_dialog import (
        TransferMatrixDialog)
    holder = {"w": None}

    def dlg_grab():
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(BG)
        p = QPainter(canvas)
        src = holder["w"]
        img = (src.grab().toImage() if src is not None
               else win.grab().toImage())
        img = img.scaledToHeight(
            min(1050, img.height()),
            Qt.TransformationMode.SmoothTransformation)
        if img.width() > 1900:
            img = img.scaledToWidth(
                1900, Qt.TransformationMode.SmoothTransformation)
        p.drawImage((1920 - img.width()) // 2,
                    (1080 - img.height()) // 2, img)
        p.end()
        return canvas

    sdlg = SigmaMatrixDialog(win, win.state.results,
                             beam_config=win.state.beam_config)
    sdlg.resize(1500, 900)
    sdlg.show()
    settle(12)
    holder["w"] = sdlg
    rec.start("103_matrix_viewers")
    rec.hold(dlg_grab, 8.0)
    sdlg.close(); settle(4)
    tdlg = TransferMatrixDialog(win, win.state.lattice,
                                win.state.beam_config)
    tdlg.resize(1500, 980)
    tdlg.show()
    settle(10)
    tdlg._compute_btn.click()
    settle(10)
    holder["w"] = tdlg
    rec.hold(dlg_grab, 8.0)
    tdlg.close(); settle(4)
    holder["w"] = None
    go_tab("results"); settle(6)
    rec.hold(dlg_grab, 1.5)
    win.results_tab.open_plot("convergence")     # routes to Numerics tab
    settle(8)
    rec.hold(dlg_grab, 4.0)
    rec.finish(scene_by_name("103_matrix_viewers"))
    go_tab("results"); settle(6)

    # --- 106: popups refresh together on re-run (always-on sync) +
    # Show-lattice toggle, all recorded.
    pr = open_popup("rms", (940, 700))
    pe = open_popup("emit", (940, 700))

    def duo_grab():
        return montage([pr.grab().toImage(), pe.grab().toImage()])

    rec.start("106_popup_machinery")
    rec.hold(duo_grab, 3.0)
    pr._chk_lat.setChecked(False)
    rec.hold(duo_grab, 1.8)
    pr._chk_lat.setChecked(True)
    rec.hold(duo_grab, 1.8)
    win.state.set_results(None)          # popups clear (run starting)
    win.state.set_beam_config(cfg_fodo)
    win._run_mp()
    rec.wait_until(duo_grab,
                   lambda: win.state.results is not None,
                   cap_s=180, stable_s=0.5)
    rec.hold(duo_grab, 4.0)
    rec.finish(scene_by_name("106_popup_machinery"))
    pr.close(); pe.close(); settle(4)

    # --- 090: a REAL mini Monte-Carlo through the ErrorStudy engine
    # (the same engine the Error Study tab drives) fills the ensemble
    # popup via state.error_study_results.
    from linac_gen.errors.error_model import ErrorStudy
    study = ErrorStudy(win.state.lattice, win.state.beam_config,
                       n_seeds=10)
    study.add_error("QUAD*", "gradient_rel", sigma=0.03)
    study.add_error("QUAD*", "dx", sigma=0.15)
    sres = study.run()
    win.state.error_study_results = sres
    print(f"[measure] ensemble n_seeds={sres.n_seeds}")
    pop = open_popup("ensemble", (1250, 850))
    print("[measure] ensemble summary:", pop._summary.text())
    pop.grab().save(s["090_ensemble"])
    pop.close(); settle(2)
    win.state.error_study_results = None

    # ==================================================================
    # RUN B0 — DTL at 20 mA, 1000 macroparticles: the partran overlay
    # source (engine-faithful export of a REAL second run).
    # ==================================================================
    load("examples/dtl_section.dat")
    cfg_b0 = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                        current=20.0, n_particles=1000)
    run_mp(cfg_b0)
    from linac_gen.io.tracewin_outputs import write_partran_out
    pfile = SCRATCH / "dtl_1000.out"
    res0 = win.state.results
    _oldf = res0.ref_frequency
    try:
        # the GUI worker stores ref_frequency as a scalar; the partran
        # writer expects the recorder's per-step list — same value at
        # every step of this single-frequency machine.
        if not isinstance(_oldf, (list, tuple)):
            res0.ref_frequency = [float(_oldf)] * len(res0.s)
        write_partran_out(res0, win.state.lattice, cfg_b0, pfile)
    finally:
        res0.ref_frequency = _oldf
    print(f"[measure] partran overlay file: {pfile.stat().st_size} bytes")

    # ==================================================================
    # RUN B — DTL at 20 mA, 2500 macroparticles (the overloaded run)
    # ==================================================================
    cfg_dtl = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                         current=20.0, n_particles=2500)
    run_mp(cfg_dtl)
    go_tab("results")
    settle(8)
    res = win.state.results
    tr = np.asarray(res.transmission, dtype=float)
    print(f"[measure] DTL transmission end {tr[-1]:.1f}% "
          f"(loss {100-tr[-1]:.1f}%)")
    print(f"[measure] DTL W {res.ref_w_kin[0]:.2f} -> "
          f"{res.ref_w_kin[-1]:.3f} MeV")
    hx = np.asarray(res.halo_x, dtype=float)
    print(f"[measure] DTL halo_x max {np.nanmax(hx):.2f} end {hx[-1]:.2f}")

    # --- 050: energy popup + the beam-power card (footer tells the
    # overload story) on one canvas.
    p1 = open_popup("energy", (1150, 900))
    pcard = next(c for c in win.results_tab._cards if c._key == "power")
    print(f"[measure] power card footer: {pcard._value_lbl.text()!r} "
          f"trend {pcard._trend_lbl.text()!r}")
    cimg = pcard.grab().toImage().scaledToWidth(
        540, Qt.TransformationMode.SmoothTransformation)
    canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
    canvas.fill(BG)
    p = QPainter(canvas)
    eimg = p1.grab().toImage().scaledToHeight(
        1000, Qt.TransformationMode.SmoothTransformation)
    if eimg.width() > 1300:
        eimg = eimg.scaledToWidth(
            1300, Qt.TransformationMode.SmoothTransformation)
    p.drawImage(24, (1080 - eimg.height()) // 2, eimg)
    p.drawImage(min(eimg.width() + 60, 1920 - cimg.width() - 20),
                (1080 - cimg.height()) // 2, cimg)
    p.end()
    canvas.save(s["050_energy"])
    p1.close(); settle(2)

    # --- 055: the loss family — profile then aperture autopsy, recorded.
    holder2 = {"w": None}

    def one_grab():
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(BG)
        p = QPainter(canvas)
        img = holder2["w"].grab().toImage()
        img = img.scaledToHeight(
            min(1050, img.height()),
            Qt.TransformationMode.SmoothTransformation)
        if img.width() > 1900:
            img = img.scaledToWidth(
                1900, Qt.TransformationMode.SmoothTransformation)
        p.drawImage((1920 - img.width()) // 2,
                    (1080 - img.height()) // 2, img)
        p.end()
        return canvas

    pop = open_popup("loss", (1200, 620))
    holder2["w"] = pop
    rec.start("055_losses")
    rec.hold(one_grab, 6.0)
    pop.close(); settle(4)
    pop = open_popup("aperture_loss", (1500, 860))
    holder2["w"] = pop
    settle(10)
    print("[measure] aperture status:", pop._status.text())
    rec.hold(one_grab, 10.0)
    rec.finish(scene_by_name("055_losses"))
    pop.close(); settle(2)

    # --- 058: loss-power popup (W/m + exit-plane W/cm²).
    pop = open_popup("loss_power", (1250, 880))
    print("[measure] loss_power status:", pop._status.text())
    pop.grab().save(s["058_loss_power"])
    pop.close(); settle(2)

    # --- 060: halo + peak excursion side by side.
    p1 = open_popup("halo", (940, 640))
    p2 = open_popup("peak", (940, 640))
    montage([p1.grab().toImage(), p2.grab().toImage()]).save(
        s["060_halo_peak"])
    p1.close(); p2.close(); settle(2)

    # --- 065: phase space — snapshot scrub + every display control.
    pop = open_popup("phase", (1500, 880))
    assert pop._location.count() > 1, "no snapshots recorded"
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("065_phase_space")
    pop._location.setCurrentIndex(0)                 # exit (final)
    rec.hold(pop_grab, 3.5)
    for i in range(pop._location.count() - 1, 0, -1):
        pop._location.setCurrentIndex(i)             # scrub backwards
        rec.hold(pop_grab, 0.6)
    pop._location.setCurrentIndex(0)
    rec.hold(pop_grab, 2.5)
    for ci in (1, 2, 3):                             # colour-by sweep
        pop._colour.setCurrentIndex(ci)
        rec.hold(pop_grab, 2.2)
    pop._colour.setCurrentIndex(0)
    rec.hold(pop_grab, 1.5)
    pop._basis.setCurrentIndex(1)                    # (z, δ) TraceWin
    rec.hold(pop_grab, 2.5)
    pop._basis.setCurrentIndex(0)
    rec.hold(pop_grab, 1.0)
    if pop._wrap_phi.isEnabled():
        pop._wrap_phi.setChecked(False)              # unfold the train
        rec.hold(pop_grab, 2.5)
        pop._wrap_phi.setChecked(True)
        rec.hold(pop_grab, 1.5)
    pop._params_btn.setChecked(True)                 # parameter table
    rec.hold(pop_grab, 4.5)
    pop._params_btn.setChecked(False)
    rec.hold(pop_grab, 1.5)
    rec.finish(scene_by_name("065_phase_space"))
    pop.close(); settle(2)

    # --- 070: density heatmap — axis sweep + log & overlay toggles.
    pop = open_popup("density_s", (1500, 820))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("070_density")
    rec.hold(pop_grab, 4.0)                          # x (default)
    for ci in (2, 4, 5):                             # x', phi, W
        pop._axis_cb.setCurrentIndex(ci)
        rec.hold(pop_grab, 2.4)
    pop._axis_cb.setCurrentIndex(0)
    rec.hold(pop_grab, 2.0)
    pop._chk_sigma.setChecked(False)
    rec.hold(pop_grab, 1.6)
    pop._chk_sigma.setChecked(True)
    rec.hold(pop_grab, 1.6)
    pop._chk_log.setChecked(False)                   # tails vanish
    rec.hold(pop_grab, 2.4)
    pop._chk_log.setChecked(True)
    rec.hold(pop_grab, 2.0)
    rec.finish(scene_by_name("070_density"))
    pop.close(); settle(2)

    # --- 075: lattice parameters — sweep six genuine popups (DTL has
    # quads, RF gaps AND two solenoids).
    rec.start("075_lattice_params")
    for key, hold_s in (("quad_grad", 4.5), ("quad_gl", 3.5),
                        ("rf_volt", 3.5), ("sync_phase", 3.5),
                        ("bpeak", 3.5), ("int_b2", 3.5)):
        pop = open_popup(key, (1400, 800))
        holder2["w"] = pop
        rec.hold(one_grab, hold_s)
        pop.close(); settle(3)
    rec.finish(scene_by_name("075_lattice_params"))

    # --- 098: TraceWin-compare — load the RUN-B0 partran file (a real
    # HELIX export, never ANL/CEA data), sweep the axis dropdown.
    pop = open_popup("partran", (1400, 860))
    _DIALOG_PATHS["open"] = str(pfile)
    load_btn = [b for b in pop.findChildren(QPushButton)
                if b.text().startswith("Open partran")][0]
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("098_partran")
    rec.hold(pop_grab, 2.5)
    load_btn.click()
    settle(12)
    print("[measure] partran status:", pop._status.text())
    rec.hold(pop_grab, 5.0)
    for ci in (3, 6):                                # eps_x, W
        pop._axis.setCurrentIndex(ci)
        rec.hold(pop_grab, 3.0)
    pop._axis.setCurrentIndex(0)
    rec.hold(pop_grab, 2.0)
    clear_btn = [b for b in pop.findChildren(QPushButton)
                 if b.text().startswith("Clear overlay")][0]
    clear_btn.click()
    rec.hold(pop_grab, 2.5)
    rec.finish(scene_by_name("098_partran"))
    pop.close(); settle(2)

    # --- 110: export round trip — Ctrl+S (CSV + PNG), the actual files,
    # openPMD export, and Import Results repopulating the wall.
    pop = open_popup("rms", (1200, 700))
    csv_path = SCRATCH / "RMS_sigma.csv"
    png_path = SCRATCH / "RMS_sigma.png"
    opmd_path = SCRATCH / "dtl_run.opmd.h5"
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("110_export")
    rec.hold(pop_grab, 3.0)
    _DIALOG_PATHS["save"] = str(csv_path)
    pop._save_plot_data()                            # Ctrl+S slot
    settle(6)
    _DIALOG_PATHS["save"] = str(png_path)
    pop._save_plot_data()
    settle(6)
    files = [(f, (SCRATCH / f).stat().st_size)
             for f in (csv_path.name, png_path.name)]
    print("[measure] saved files:", files)
    term_png = SHOTS / "110_files.png"
    cards.terminal_card(term_png, [
        ("cmd", "ls -la " + SCRATCH.name + "/"),
        ("out", f"{csv_path.name:<24}{csv_path.stat().st_size:>10,} bytes"),
        ("out", f"{png_path.name:<24}{png_path.stat().st_size:>10,} bytes"),
        ("gap", ""),
        ("out", "Ctrl+S — one dialog, data or image formats"),
    ], title="the exported files")
    term_img = QImage(str(term_png))
    rec.hold(lambda: term_img, 5.0)
    pop.close(); settle(4)
    _DIALOG_PATHS["save"] = str(opmd_path)
    win._export_openpmd()
    settle(10)
    print(f"[measure] openPMD file: {opmd_path.stat().st_size} bytes")

    def win_grab():
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(BG)
        p = QPainter(canvas)
        img = win.grab().toImage().scaledToWidth(
            1880, Qt.TransformationMode.SmoothTransformation)
        p.drawImage(20, (1080 - img.height()) // 2, img)
        p.end()
        return canvas

    go_tab("results"); settle(6)
    scroll.verticalScrollBar().setValue(0); settle(4)
    win.state.set_results(None)                      # dashes everywhere
    settle(8)
    rec.hold(win_grab, 2.5)
    _DIALOG_PATHS["open"] = str(opmd_path)
    win.results_tab._import_btn.click()              # loads it back
    settle(20)
    assert win.state.results is not None, "import gave no results"
    rec.hold(win_grab, 5.0)
    rec.finish(scene_by_name("110_export"))

    # ==================================================================
    # RUN C — the FODO with an H- beam at 5 mA: the species gate.
    # (RF-free on purpose: a negative-charge beam in the proton-phased
    # DTL decelerates below zero energy — probed, math domain error.)
    # ==================================================================
    load("examples/fodo_cell.dat")
    from linac_gen.core.particle import H_MINUS
    ref_h = ReferenceParticle(species=H_MINUS, w_kin=3.0,
                              frequency=352.21)
    tw_h = find_matched_input_twiss(win.state.lattice, ref_h,
                                    *find_fodo_cells(win.state.lattice)[0])
    cfg_h = BeamConfig(species="H-", energy=3.0, frequency=352.21,
                       current=5.0, n_particles=1000,
                       alpha_x=tw_h["alpha_x"], beta_x=tw_h["beta_x"],
                       alpha_y=tw_h["alpha_y"], beta_y=tw_h["beta_y"])
    run_mp(cfg_h)
    go_tab("results"); settle(8)
    ibs_card = next(c for c in win.results_tab._cards if c._key == "ibs")
    print(f"[measure] ibs card enabled after H- run: {ibs_card.isEnabled()}")

    def card_grab():
        canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        canvas.fill(BG)
        p = QPainter(canvas)
        img = ibs_card.grab().toImage().scaledToWidth(
            760, Qt.TransformationMode.SmoothTransformation)
        p.drawImage((1920 - img.width()) // 2,
                    (1080 - img.height()) // 2, img)
        p.end()
        return canvas

    rec.start("088_h_minus")
    rec.hold(card_grab, 4.0)                         # the enabled tile
    pop = open_popup("ibs", (1250, 900))
    settle(15)
    print("[measure] ibs summary:", pop._summary.text())
    holder2["w"] = pop
    rec.hold(one_grab, 8.0)
    pop.close(); settle(4)
    pop = open_popup("magstrip", (1250, 900))
    settle(15)
    print("[measure] magstrip summary:", pop._summary.text())
    holder2["w"] = pop
    rec.hold(one_grab, 9.0)
    rec.finish(scene_by_name("088_h_minus"))
    pop.close(); settle(2)

    # ==================================================================
    # RUN D — Hofmann demo linac, envelope at 5 mA (phase probe on):
    # channel tunes, stability chart, tune footprint.
    # ==================================================================
    load("examples/hofmann_stability/hofmann_demo_linac.dat")
    cfg_hof = BeamConfig(
        species="proton", energy=2.5, frequency=162.5, current=5.0,
        n_particles=1000, distribution="gaussian", cutoff=4.0,
        emit_nx=0.146096, alpha_x=-1.366692, beta_x=0.297401,
        emit_ny=0.146096, alpha_y=1.366692, beta_y=0.297401,
        emit_z=0.035, alpha_z=0.0, beta_z=241.441179)
    run_env(cfg_hof)
    go_tab("results"); settle(8)

    pop = open_popup("phase_adv", (1700, 960))
    holder2["w"] = pop
    rec.start("080_channel_tunes")
    rec.wait_until(one_grab,
                   lambda: (getattr(pop, "_worker", None) is None
                            or not pop._worker.isRunning()),
                   cap_s=120, stable_s=0.5)
    settle(15)
    print("[measure] phase_adv info:", pop._info.text())
    rec.hold(one_grab, 9.0)
    pop.close(); settle(4)
    pop = open_popup("tune_depr", (1250, 960))
    holder2["w"] = pop
    rec.wait_until(one_grab,
                   lambda: (getattr(pop, "_worker", None) is None
                            or not pop._worker.isRunning()),
                   cap_s=120, stable_s=0.5)
    settle(15)
    print("[measure] tune_depr info:", pop._info.text())
    rec.hold(one_grab, 9.0)
    rec.finish(scene_by_name("080_channel_tunes"))
    pop.close(); settle(2)

    # --- 082: Hofmann chart computed on camera.
    pop = open_popup("hofmann", (1100, 840))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("082_hofmann")
    rec.hold(pop_grab, 4.0)                          # trajectory only
    pop._btn.click()
    rec.wait_until(pop_grab,
                   lambda: (pop._worker is None
                            or not pop._worker.isRunning()),
                   cap_s=180, stable_s=0.5)
    settle(15)
    print("[measure] hofmann info:", pop._info.text())
    rec.hold(pop_grab, 7.0)
    pop._chk_bands.setChecked(True)                  # legacy bands
    rec.hold(pop_grab, 3.0)
    pop._chk_bands.setChecked(False)
    rec.hold(pop_grab, 1.5)
    rec.finish(scene_by_name("082_hofmann"))
    pop.close(); settle(2)

    # --- 084: tune footprint computed on camera.
    pop = open_popup("footprint", (1050, 800))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("084_footprint")
    rec.hold(pop_grab, 3.0)
    pop._btn.click()
    rec.wait_until(pop_grab,
                   lambda: (pop._worker is None
                            or not pop._worker.isRunning()),
                   cap_s=300, stable_s=0.5)
    settle(15)
    print("[measure] footprint info:", pop._info.text())
    rec.hold(pop_grab, 7.0)
    rec.finish(scene_by_name("084_footprint"))
    pop.close(); settle(2)

    # ==================================================================
    # RUN E — DC H- LEBT (SCC demo): substep recording for a resolved
    # f_c(z), then the compensation analysis computed on camera.
    # ==================================================================
    ct._record_substeps.setChecked(True)
    load("examples/lebt_scc_demo.dat")
    cfg_lebt = BeamConfig(
        species="H-", energy=0.03, frequency=162.5, current=15.0,
        n_particles=1000, distribution="gaussian", cutoff=4.0,
        emit_nx=0.023977, alpha_x=0.0, beta_x=0.6,
        emit_ny=0.023977, alpha_y=0.0, beta_y=0.6,
        emit_z=0.0, continuous=True)
    run_env(cfg_lebt)
    ct._record_substeps.setChecked(False)
    go_tab("results"); settle(8)
    pop = open_popup("scc", (1400, 900))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    ngas = pop._gas.findText("N2")
    rec.start("086_scc")
    rec.hold(pop_grab, 3.0)
    if ngas >= 0:
        pop._gas.setCurrentIndex(ngas)
    rec.hold(pop_grab, 1.5)
    pop._btn.click()
    rec.wait_until(pop_grab,
                   lambda: (pop._worker is None
                            or not pop._worker.isRunning()),
                   cap_s=180, stable_s=0.5)
    settle(15)
    print("[measure] scc info:", pop._info.text())
    rec.hold(pop_grab, 7.0)
    pop._btn_export.click()                          # clipboard only
    settle(6)
    print("[measure] scc after export:", pop._info.text())
    rec.hold(pop_grab, 3.5)
    rec.finish(scene_by_name("086_scc"))
    pop.close(); settle(2)

    # ==================================================================
    # RUN F — the 90-degree bend line: the dispersion family.
    # ==================================================================
    load("examples/bend_line.dat")
    cfg_bend = BeamConfig(species="proton", energy=100.0,
                          frequency=352.21, current=0.0,
                          n_particles=2000)
    run_mp(cfg_bend)
    go_tab("results"); settle(8)
    res = win.state.results
    S = np.asarray(res.sigma_matrix, dtype=float)
    sxx = S[:, 0, 0]
    sww = S[:, 5, 5]
    corr = np.where(sww > 1e-30,
                    sxx - S[:, 0, 5] ** 2 / np.where(sww > 1e-30, sww, 1.0),
                    sxx)
    print(f"[measure] bend sigma_x end raw={np.sqrt(sxx[-1]):.3f} "
          f"corrected={np.sqrt(max(corr[-1], 0)):.3f} mm")

    pop = open_popup("dispersion", (1250, 800))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("092_bend_dispersion")
    rec.hold(pop_grab, 4.5)                          # statistical only
    pop._chk_model.setChecked(True)
    rec.wait_until(pop_grab,
                   lambda: (pop._worker is None
                            or not pop._worker.isRunning()),
                   cap_s=120, stable_s=0.5)
    settle(15)
    print("[measure] dispersion status:", pop._lbl_status.text())
    rec.hold(pop_grab, 6.0)
    pop.close(); settle(4)
    pop = open_popup("dpp", (1250, 620))
    holder2["w"] = pop
    rec.hold(one_grab, 4.0)
    pop.close(); settle(4)
    pop = open_popup("rms", (1250, 800))
    holder2["w"] = pop
    rec.hold(one_grab, 3.5)                          # raw
    pop._mode_combo.setCurrentIndex(1)               # corrected — drops
    rec.hold(one_grab, 4.5)
    pop._mode_combo.setCurrentIndex(0)
    rec.hold(one_grab, 2.0)
    rec.finish(scene_by_name("092_bend_dispersion"))
    pop.close(); settle(2)

    # --- 094: bend hardware tiles + floor plan.
    rec.start("094_bend_survey")
    for key, hold_s in (("bend_b", 4.0), ("bend_bl", 3.5)):
        pop = open_popup(key, (1300, 720))
        holder2["w"] = pop
        rec.hold(one_grab, hold_s)
        pop.close(); settle(3)
    pop = open_popup("floorplan", (1300, 800))
    holder2["w"] = pop
    settle(8)
    print("[measure] floorplan:", pop._info.text())
    rec.hold(one_grab, 6.5)
    rec.finish(scene_by_name("094_bend_survey"))
    pop.close(); settle(2)

    # ==================================================================
    # RUN G — orbit-correction demo: kicked steerer + goal orbit.
    # ==================================================================
    lat = load("examples/correction_demo/correction_demo.dat")
    from linac_gen.elements.steerer import Steerer
    steerers = [e for e in lat.elements if isinstance(e, Steerer)]
    steerers[0].by_l = 5e-4                          # deliberate kick
    for e in lat.elements:                           # flat-orbit goals
        if getattr(e, "is_bpm", False):
            e.x_target_mm = 0.0
            e.y_target_mm = 0.0
    cfg_corr = BeamConfig(species="proton", energy=5.0,
                          frequency=352.21, current=0.0,
                          n_particles=1000)
    run_mp(cfg_corr)
    go_tab("results"); settle(8)
    pop = open_popup("centroid", (1300, 820))
    settle(10)
    print("[measure] centroid banner:", pop._goal_lbl.text())
    holder2["w"] = pop
    rec.start("096_orbit_bpms")
    rec.hold(one_grab, 8.0)
    pop.close(); settle(4)
    pop = open_popup("bpms", (900, 560))
    holder2["w"] = pop
    rows = pop._table.rowCount()
    print(f"[measure] bpm rows: {rows}")
    for r in range(rows):
        print("   ", [pop._table.item(r, c).text() for c in range(5)])
    rec.hold(one_grab, 6.0)
    pop.close(); settle(4)
    pop = open_popup("long_offset", (1100, 620))
    holder2["w"] = pop
    rec.hold(one_grab, 4.0)
    rec.finish(scene_by_name("096_orbit_bpms"))
    pop.close(); settle(2)

    # ==================================================================
    # RUN H — synthetic 6-cell cavity map (generated here, nothing
    # proprietary): field-map viewer + TTF.
    # ==================================================================
    fmdir = SCRATCH / "fmap"
    fmdir.mkdir(exist_ok=True)
    L, R, Nz, Nr = 0.4, 0.03, 120, 24
    Lc = L / 6.0                                     # 6 cells
    zax = np.linspace(0, L, Nz + 1)
    rax = np.linspace(0, R, Nr + 1)
    E0 = 2.5                                         # MV/m
    prof = np.sin(np.pi * zax / Lc)
    Ez = E0 * prof[:, None] * (1 - 0.4 * (rax[None, :] / R) ** 2)
    dEz = E0 * (np.pi / Lc) * np.cos(np.pi * zax / Lc)[:, None]
    Er = -0.5 * rax[None, :] * dEz
    om = 2 * np.pi * 162.5e6
    Bq = om * rax[None, :] / (2 * (2.998e8) ** 2) * (E0 * 1e6) \
        * prof[:, None]

    def _write2d(ext, arr):
        with open(fmdir / ("syncav" + ext), "w") as f:
            f.write(f"{Nz} {L}\n{Nr} {R}\n1.0\n")
            for k in range(Nz + 1):
                f.write(" ".join(f"{arr[k, i]:.6e}"
                                 for i in range(Nr + 1)) + "\n")
    _write2d(".edz", Ez)
    _write2d(".edr", Er)
    _write2d(".bdq", Bq)
    cav_deck = fmdir / "syncav_demo.dat"
    cav_deck.write_text(
        "TITLE Synthetic cavity demo\nFREQ 162.5\nDRIFT 100 30\n"
        "FIELD_MAP 400 400.0 -30.0 30.0 0 1.0 0 0 syncav\n"
        "DRIFT 100 30\nEND\n")
    load(cav_deck)
    cfg_cav = BeamConfig(species="proton", energy=2.5, frequency=162.5,
                         current=0.0, n_particles=800)
    run_mp(cfg_cav)                                  # MP → ref_bg → β_beam
    go_tab("results"); settle(8)
    print(f"[measure] cavity W 2.5 -> {win.state.results.ref_w_kin[-1]:.4f}")
    pop = open_popup("field_map", (1700, 960))
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    rec.start("100_fieldmap_ttf")
    rec.hold(pop_grab, 4.0)
    if "Fz" in pop._component_keys:
        pop._cp_combo.setCurrentIndex(pop._component_keys.index("Fz"))
    rec.hold(pop_grab, 3.0)
    zmax = pop._sliders["z"].maximum()
    for frac in (0.25, 0.5, 0.75):                   # slice sweep
        pop._sliders["z"].setValue(int(zmax * frac))
        rec.hold(pop_grab, 1.6)
    xmax = pop._sliders["x"].maximum()
    pop._sliders["x"].setValue(int(xmax * 0.8))      # off-axis cut
    rec.hold(pop_grab, 2.0)
    pop.close(); settle(4)
    pop = open_popup("ttf", (1400, 800))
    settle(10)
    print("[measure] ttf info:", pop._info.text())
    holder2["w"] = pop
    rec.hold(one_grab, 7.0)
    rec.finish(scene_by_name("100_fieldmap_ttf"))
    pop.close(); settle(2)

    # ==================================================================
    # RUN I — matching_demo: the opt-in live match preview, with a
    # REAL Matching-tab optimisation streaming into the RMS popup.
    # ==================================================================
    load("examples/matching_demo.dat")
    cfg_match = BeamConfig(species="proton", energy=3.0,
                           frequency=352.21, current=0.0)
    run_env(cfg_match)
    win.beam_tab.set_beam_config(cfg_match)   # matching reads the widgets
    go_tab("results"); settle(8)
    pop = open_popup("rms", (1250, 760))
    pop._live_cb.setChecked(True)
    mt = win.matching_tab
    mt._aa_algorithm.setCurrentText("bayesopt")
    mt._aa_iter.setValue(40)     # ~12 s of live preview, not minutes
    settle(6)
    pop_grab = lambda: pop.grab().toImage()          # noqa: E731
    _live_seen = {"n": 0}

    def _match_done():
        if "LIVE" in pop.windowTitle():
            _live_seen["n"] += 1
        return mt._aa_apply.isEnabled()

    rec.start("108_live_preview")
    rec.hold(pop_grab, 2.5)
    mt._aa_run.click()
    rec.wait_until(pop_grab, _match_done, cap_s=45, stable_s=0.3)
    settle(20)
    rec.hold(pop_grab, 3.5)
    rec.finish(scene_by_name("108_live_preview"))
    print(f"[measure] live preview ticks seen: {_live_seen['n']}; "
          f"final title: {pop.windowTitle()!r}")
    pop.close(); settle(2)

    for sc in SCENES:
        if not sc.frames_dir:
            sc.image = s[sc.name]
    # NO win.close(): with dirty state it opens the unsaved-changes
    # confirm — a modal dialog, which offscreen spins/hangs forever.
    # The process exits after rendering; Qt tears down with it.


def main() -> None:
    capture_visuals()
    info = build_video(SCENES, WORK, OUT)
    print(f"rendered {info['mp4']}  ({info['duration']:.1f} s)")
    print(f"captions {info['srt']}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
