"""Episode 4 — The Beam tab in depth (maximum-detail cut).

Build:  PYTHONPATH=.:gui:tutorials python3 tutorials/storyboards/ep04_beam_tab.py
Output: tutorials/rendered/ep04_beam_tab.mp4 (+ .srt)

Every control on the tab is either shown on camera or demonstrated live:
derived-chip recomputation, all six distributions, halo + cutoff, the
dispersion rows, centroid/mismatch demos, a real Apply, a genuine .dst
import (file dialog stubbed — modal dialogs hang offscreen), the DC and
periodic-phase modes, Reset defaults, and two manual-page tours rendered
with QtWebEngine.  All spoken numbers were measured on this machine.
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

WORK = ROOT / "tutorials" / "rendered" / "ep04_work"
SHOTS = WORK / "shots"
OUT = ROOT / "tutorials" / "rendered" / "ep04_beam_tab.mp4"

DST_FIXTURE = ROOT / "examples/batch_mode/02_run_multiparticle/out/chicane_final.dst"


def est(text: str) -> float:
    """Estimated narration seconds for a segment (measured ~14.0 c/s)."""
    return len(text) / 14.0


# ---------------------------------------------------------------------
# Narration segments for motion scenes: holds are derived from est() so
# the on-camera action stays in step with the spoken text.
N032 = [
    "Watch the derived chips, top right. Beta, gamma, beta gamma, "
    "recomputed on every edit, no Apply needed. ",
    "Deuteron, twice the proton mass, and beta falls below six "
    "percent of the speed of light. ",
    "Back to proton. ",
    "Raise the energy. One hundred M e V, beta about nought point "
    "four three. ",
    "Eight hundred, beta nought point eight four, gamma one point "
    "eight five, beta gamma one point five six. ",
    "Beta gamma converts normalised to geometric emittance. Its "
    "growth from nought point nought eight is adiabatic damping, "
    "read off the form. ",
]
N042 = [
    "Watch the preview as the distribution changes. Every frame is "
    "the real generator, one hundred thousand particles rebinned. ",
    "Waterbag. Uniform in a six dimensional ellipsoid, R M S like a "
    "Gaussian, but every particle inside a hard envelope. ",
    "K V. Uniform transverse charge density, the only non trivial "
    "distribution with exactly linear space charge forces. ",
    "Parabolic, a soft edged density matching one of TraceWin's "
    "options. ",
    "And uniform, in HELIX an alias for the waterbag generator, so "
    "with the same seed the picture is identical. ",
]
N044 = [
    "Thermal is the halo distribution. Selecting it wakes the two "
    "halo fields every other shape greys out. ",
    "Fraction, the odds of the wide component, five percent default. "
    "Ratio, how much wider, five. ",
    "Regenerate. The log scale shows a faint cloud around the core. ",
    "Raise the fraction to thirty percent, and the halo brightens "
    "sharply. ",
    "The mixture is renormalised, the combined R M S emittance still "
    "matches the form. ",
]
N046 = [
    "Back on Gaussian, cutoff truncates at that many sigma, one to "
    "ten. ",
    "One point five, and the edge is unmistakable, the tails simply "
    "gone. ",
    "Eight, and the tails run long, what halo and aperture loss "
    "studies need. It truncates the thermal core too. ",
    "Back to four, this form's default. ",
]
N048 = [
    "The manual's distributions chapter opens with a use case table. "
    "Waterbag, benchmarks. Gaussian, production realism. K V, "
    "envelope theory. Parabolic, TraceWin compatibility. Uniform, "
    "stress tests. Thermal, explicit halo. ",
    "Figure four point one draws all six from the same Twiss. Cores "
    "alike at the sigma level, tails completely different. ",
    "And the see also list links the convergence guide, how many "
    "particles you really need. ",
]
N062 = [
    "Centroid offsets on camera. Delta x three millimetres, delta phi "
    "twenty degrees, regenerate. ",
    "The bunch moves, subtly. The panels range around the beam, the "
    "blob stays central, the shift lands in the axis scales, x "
    "centred near three millimetres, phase near twenty. ",
    "That is how injection errors are modelled, and the phase field "
    "accepts plus or minus three thousand six hundred degrees. ",
    "Back to zero, the axes re centre. ",
]
N065 = [
    "Mismatch scales the plane's geometric emittance by one plus the "
    "percentage over one hundred. Delta epsilon x three hundred "
    "percent, four times the emittance, twice the size. ",
    "Read the axes. The horizontal span doubles, y untouched. File "
    "loaded beams ignore these factors. ",
    "One detail. The floor is minus ninety nine point nine nine nine "
    "percent, minus one hundred would zero the emittance. Push lower, "
    "the box pins. ",
    "Back to matched. ",
]
N070 = [
    "The preview is drawn from your numbers. Watch it work. ",
    "Alpha x flips from one point two to minus two point five, and "
    "Regenerate shears the horizontal ellipse while the rest hold "
    "still. ",
    "Four views. x x prime, y y prime, phase against energy "
    "deviation, and real space x y. One hundred forty bins per axis, "
    "every surviving particle binned, no downsampling. The colour "
    "bar, log of one plus N. Seeded generation, identical numbers "
    "redraw the identical picture. ",
]
N085 = [
    "Time to commit. The form asks one hundred twenty thousand "
    "particles, the status label still shows the last preview at one "
    "hundred thousand. Nothing applied yet. ",
    "Click Apply. The preview regenerates, the label reads one "
    "hundred twenty thousand, and the window status bar confirms, "
    "beam config applied. ",
    "Beam settings live in the project file, so Apply marks the "
    "project dirty and the close prompt warns. A project load "
    "applies quietly. The Matching tab flags the project itself. ",
]
N092 = [
    "A real import. The file is the final beam of a batch mode "
    "example, and Continuous is ticked on purpose. Watch it. ",
    "The header fills the form. Five thousand particles, one hundred "
    "sixty two point five megahertz, five milliamps, just under eight "
    "hundred M e V, emittance and Twiss from the actual particles. ",
    "Species is inferred by mass. Proton and H minus tie, so your "
    "choice stands. ",
    "Continuous unticked itself, a dot D S T is bunched by "
    "construction. ",
    "The chip names the file, and the run now tracks exactly these "
    "particles, full count, no subsampling. ",
]
N094 = [
    "Clear undoes it. The chip vanishes, the status line confirms, "
    "file source cleared, using generated distribution. ",
    "The form keeps the imported values, regenerating builds five "
    "thousand particles from that Twiss, not the file. A source "
    "switch, not a new form. ",
    "Reset defaults would clear the file source too. ",
]
N100 = [
    "Now the beam modes. Continuous is for lines before any R F "
    "structure. No bunches. Tick it, and watch both columns. ",
    "The longitudinal Twiss greys out, ignored by the generator, "
    "nothing to zero by hand. The D C delta W field wakes in its "
    "place, the sole energy spread source. ",
    "Untick, everything returns. Tracking a D C line as bunched blows "
    "up the phase spread non physically, and the switch also selects "
    "the continuous space charge kernels. ",
]
N102 = [
    "What does a continuous beam look like. Tick Continuous, two "
    "k e V of spread, regenerate. ",
    "The phase energy panel becomes a flat band, uniform across one "
    "R F period, a Gaussian of one sigma two k e V on top. "
    "Transverse panels untouched, generation here is transverse "
    "phase space plus that band. ",
    "Such a beam tracks in four dimensions until the first R F "
    "element, then switches to six. ",
    "Untick, clear the spread, and the bunched preview returns. ",
]
N104 = [
    "The last checkbox has manners. Periodic phase matters only for a "
    "beam injected D C and bunched later, so it wakes only with "
    "Continuous. ",
    "Tick Continuous, the box enables. ",
    "Tick Periodic phase, the bunch train fold will apply during "
    "tracking. ",
    "Untick Continuous. Periodic phase is disabled but keeps your "
    "tick, an accidental untick and retick destroys nothing. ",
    "The built configuration requires both, so a disabled tick never "
    "reaches a project. ",
]
N106 = [
    "Why the fold exists, in measured numbers. An R F Q makes one "
    "bunch per R F period, the simulation seeds one period. ",
    "Space charge pushes particles across the bucket boundary, and "
    "they land a full bunch spacing away as satellites. A bunch "
    "really four degrees wide reports one hundred eighty three. The "
    "Toutatis fold makes the numbers single bunch values. ",
    "With space charge off nothing else moves, coordinates identical "
    "to a part in ten to the twelve. With it on, transmission sixty "
    "two point zero to sixty point six, emittance nought point one "
    "four two to nought point one nine four. ",
    "Three consequences. Backtracking refuses a folded run. C S R "
    "cannot combine. And epsilon z staircases while particles cross "
    "buckets, so retune objectives built on it. ",
]
N108 = [
    "Housekeeping. The form is scrambled on purpose, fifty M e V, "
    "uniform, off axis, continuous ticked. ",
    "One click of Reset defaults snaps everything back. H minus, "
    "about two point one M e V, one hundred sixty two point five "
    "megahertz, five milliamps, Gaussian at four sigma, one hundred "
    "thousand particles. Toggles clear, a file source would drop "
    "too. ",
    "Reset once wrote a stale bunch train tick into a project, now "
    "it clears the toggles and ends with a quiet apply, ",
    "and one press of Regenerate shows the stock beam. ",
]

SCENES = [
    Scene("010_title",
          "Deep dive number two. The Beam tab, where the injected "
          "particles are defined. Species, energy, current. Six "
          "distribution generators, a live preview. Twiss and "
          "dispersion. Centroids and mismatch. Continuous beams, and "
          "real particle files imported on camera.",
          min_s=6.0),
    Scene("020_layout",
          "Two halves and a footer. Three groups of numbers, Twiss in "
          "the middle, centroids, mismatch, dispersion and derived on "
          "the right. Below, the live preview. At the bottom, the "
          "button row and status label, the tab's feedback channel. "
          "The form maps one to one onto the beam configuration, "
          "saved in the project file, not the lattice file."),
    Scene("030_particle",
          "The first group. Species, proton, deuteron, or H minus. "
          "Kinetic energy, one k e V to ten G e V. Frequency, one to "
          "five thousand megahertz. Peak current, up to one amp, "
          "driving space charge, while duty, valid above zero to one "
          "hundred percent, scales it to a time average. Particle "
          "count, one hundred to two million. One hundred thousand is "
          "the default."),
    Scene("032_derived_live", "".join(N032)),
    Scene("040_distribution",
          "Below them, the distribution, six generators built in. "
          "This form starts from Gaussian at four sigma, right for "
          "most production work. Let the real preview do the "
          "talking."),
    Scene("042_distribution_gallery", "".join(N042)),
    Scene("044_thermal_halo", "".join(N044)),
    Scene("046_cutoff", "".join(N046)),
    Scene("048_manual_distributions", "".join(N048)),
    Scene("050_twiss",
          "The second group, optics, three by three. Alpha sets "
          "convergence, negative diverging. Beta sets size, suffix "
          "millimetres per milliradian, numerically metres per "
          "radian. Transverse emittances, normalised R M S. The "
          "longitudinal plane uses degrees and M e V, eight decimals "
          "so a matched solution round trips without loss. The "
          "Matching tab's Apply writes straight into these fields."),
    Scene("052_centroid",
          "The third group opens with the three read only chips, "
          "beta, gamma, beta gamma. That is the whole derived row. "
          "Then centroid offsets, four dispersion rows, and three "
          "mismatch factors. Each gets a demonstration now."),
    Scene("055_dispersion",
          "Dispersion first. D x, D x prime, D y, D y prime couple "
          "position and angle to energy offset, millimetres and "
          "milliradians per M e V. A matched beam in a bending line "
          "carries exactly this, and the matching dialog fills these "
          "rows for arc and transfer line cells. Straight machines "
          "leave them zero."),
    Scene("062_centroid_demo", "".join(N062)),
    Scene("065_mismatch_demo", "".join(N065)),
    Scene("070_preview", "".join(N070)),
    Scene("080_buttons",
          "The button row. Apply builds and commits the "
          "configuration, nothing takes effect before. Regenerate "
          "preview redraws without applying. Reset defaults restores "
          "every field. Import dot D S T loads a particle file. The "
          "status label reports preview counts, loads, and errors."),
    Scene("085_apply_dirty", "".join(N085)),
    Scene("090_dst",
          "A dot D S T is the TraceWin particle format, and it is the "
          "import format, no selector. HELIX reads and writes it. "
          "Track one line, export, inject into the next. For real."),
    Scene("092_dst_import_live", "".join(N092)),
    Scene("094_dst_clear", "".join(N094)),
    Scene("100_dc", "".join(N100)),
    Scene("102_dc_preview", "".join(N102)),
    Scene("104_periodic_choreo", "".join(N104)),
    Scene("106_periodic_manual", "".join(N106)),
    Scene("108_reset_demo", "".join(N108)),
    Scene("110_outro",
          "That is the Beam tab. Define, preview honestly, apply. "
          "Import a dot D S T to chain machines, go continuous when "
          "nothing is bunched yet. Next, the Numerics tab, and later "
          "the Matching tab fills these Twiss fields for you. See you "
          "there.",
          min_s=6.0),
]


def scene_by_name(name: str) -> Scene:
    return next(sc for sc in SCENES if sc.name == name)


def capture_visuals() -> None:
    # WebEngine import must precede QApplication construction.
    from PyQt6 import QtWebEngineWidgets                     # noqa: F401
    from PyQt6.QtCore import QEventLoop, QPoint, QRect, QTimer, QUrl
    from PyQt6.QtWidgets import QApplication, QGroupBox

    from pipeline import cards
    from pipeline.record import Recorder

    SHOTS.mkdir(parents=True, exist_ok=True)
    s = {sc.name: str(SHOTS / f"{sc.name}.png") for sc in SCENES}

    app = QApplication.instance() or QApplication(["ep04"])
    cards.title_card(s["010_title"], "The Beam Tab",
                     "Particles, distributions, dispersion, beam modes, "
                     ".dst import")
    cards.outro_card(s["110_outro"], [
        "Define · preview · Apply",
        "Import .dst to chain machines",
        "Continuous mode for un-bunched lines",
        "Next — the Numerics tab in depth",
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

    def crop_widget(key: str, widget, pad: int = 8) -> None:
        settle()
        img = win.grab().toImage()
        tl = widget.mapTo(win, QPoint(0, 0))
        x = max(0, tl.x() - pad)
        y = max(0, tl.y() - pad)
        w = min(img.width() - x, widget.width() + 2 * pad)
        h = min(img.height() - y, widget.height() + 2 * pad)
        img.copy(x, y, w, h).save(s[key])

    def go_tab(label: str) -> None:
        for i in range(win._tabs.count()):
            if win._tabs.tabText(i).casefold() == label.casefold():
                win._tabs.setCurrentIndex(i)
                return
        raise LookupError(f"tab {label!r} not found")

    deck = str(ROOT / "examples/dtl_section.dat")
    lattice, _meta = _parse_lattice_file(deck)
    win.state.set_lattice(lattice, deck)
    go_tab("beam")
    settle(10)

    bt = win.beam_tab
    # a physical beam for the preview and the form: the 3 MeV proton
    # that examples/dtl_section.dat expects.
    bt._species.setCurrentText("proton")
    bt._energy.setValue(3.0)
    bt._freq.setValue(352.21)
    bt._current.setValue(20.0)
    bt._regen_preview()
    settle(10)

    groups = {g.title(): g for g in bt.findChildren(QGroupBox)}
    grp_particle = groups["Particle · Energy · RF · Current"]
    grp_twiss = groups["Twiss — X / Y / Z"]
    grp_centroid = groups["Centroid · Mismatch · Derived"]
    grp_preview = groups["Initial phase-space preview"]

    # ---- grab helpers ------------------------------------------------
    def crop_grab(widget, pad: int = 8):
        def g():
            img = win.grab().toImage()
            tl = widget.mapTo(win, QPoint(0, 0))
            x = max(0, tl.x() - pad)
            y = max(0, tl.y() - pad)
            w = min(img.width() - x, widget.width() + 2 * pad)
            h = min(img.height() - y, widget.height() + 2 * pad)
            return img.copy(x, y, w, h)
        return g

    def union_grab(widgets, pad: int = 8):
        def g():
            img = win.grab().toImage()
            x0 = y0 = 10 ** 9
            x1 = y1 = -10 ** 9
            for w in widgets:
                tl = w.mapTo(win, QPoint(0, 0))
                x0 = min(x0, tl.x()); y0 = min(y0, tl.y())
                x1 = max(x1, tl.x() + w.width())
                y1 = max(y1, tl.y() + w.height())
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(img.width(), x1 + pad); y1 = min(img.height(), y1 + pad)
            return img.copy(QRect(x0, y0, x1 - x0, y1 - y0))
        return g

    def full_grab():
        return win.grab().toImage()

    # ---- stills ------------------------------------------------------
    settle()
    win.grab().save(s["020_layout"])
    crop_widget("030_particle", grp_particle)
    crop_widget("040_distribution", grp_particle)
    crop_widget("050_twiss", grp_twiss)
    crop_widget("052_centroid", grp_centroid)

    # 055 — close-up on the four dispersion rows (D_x .. D_y')
    def save_disp_crop() -> None:
        settle()
        img = win.grab().toImage()
        g_tl = grp_centroid.mapTo(win, QPoint(0, 0))
        top = bt._disp_x.mapTo(win, QPoint(0, 0)).y()
        bot = (bt._disp_yp.mapTo(win, QPoint(0, 0)).y()
               + bt._disp_yp.height())
        x = max(0, g_tl.x() - 8)
        w = min(img.width() - x, grp_centroid.width() + 16)
        y = max(0, top - 14)
        h = min(img.height() - y, bot - top + 28)
        img.copy(x, y, w, h).save(s["055_dispersion"])
    save_disp_crop()

    # 080/090 — whole-tab overview and a tight ribbon of the button row
    crop_widget("080_buttons", bt)
    def save_row_ribbon() -> None:
        settle()
        img = win.grab().toImage()
        tl = bt._apply_btn.mapTo(win, QPoint(0, 0))
        y = max(0, tl.y() - 46)
        h = min(img.height() - y, bt._apply_btn.height() + 92)
        img.copy(0, y, img.width(), h).save(s["090_dst"])
    save_row_ribbon()

    rec = Recorder(WORK / "frames", settle)

    # ---- 032: derived chips recompute live ---------------------------
    top_g = union_grab([grp_particle, grp_centroid])
    rec.start("032_derived_live")
    rec.hold(top_g, est(N032[0]))
    bt._species.setCurrentText("deuteron")
    rec.hold(top_g, est(N032[1]))
    bt._species.setCurrentText("proton")
    rec.hold(top_g, est(N032[2]))
    bt._energy.setValue(100.0)
    rec.hold(top_g, est(N032[3]))
    bt._energy.setValue(800.0)
    rec.hold(top_g, 5.0)         # freeze holds through the final segment
    rec.finish(scene_by_name("032_derived_live"))
    bt._energy.setValue(3.0)     # back to the DTL beam (off camera)
    settle(6)

    # ---- 042: four distributions, live -------------------------------
    preview_g = crop_grab(grp_preview)
    rec.start("042_distribution_gallery")
    rec.hold(preview_g, 2.0)                       # gaussian, intro
    bt._dist.setCurrentText("waterbag"); bt._gen_btn.click()
    rec.hold(preview_g, est(N042[0]) + est(N042[1]) - 2.0)
    bt._dist.setCurrentText("kv"); bt._gen_btn.click()
    rec.hold(preview_g, est(N042[2]))
    bt._dist.setCurrentText("parabolic"); bt._gen_btn.click()
    rec.hold(preview_g, est(N042[3]))
    bt._dist.setCurrentText("uniform"); bt._gen_btn.click()
    rec.hold(preview_g, est(N042[4]) - 3.0)
    rec.finish(scene_by_name("042_distribution_gallery"))
    bt._dist.setCurrentText("gaussian"); bt._regen_preview()   # off camera
    settle(6)

    # ---- 044: thermal halo enable + inflate --------------------------
    rec.start("044_thermal_halo")
    rec.hold(full_grab, 1.0)
    bt._dist.setCurrentText("thermal")             # halo fields enable
    rec.hold(full_grab, est(N044[0]) + est(N044[1]) - 1.0)
    bt._gen_btn.click()
    rec.hold(full_grab, est(N044[2]))
    bt._halo_frac.setValue(0.30); bt._gen_btn.click()
    rec.hold(full_grab, est(N044[3]) + 1.0)
    rec.finish(scene_by_name("044_thermal_halo"))
    bt._halo_frac.setValue(0.05)
    bt._dist.setCurrentText("gaussian")
    bt._regen_preview()
    settle(6)

    # ---- 046: cutoff at work -----------------------------------------
    rec.start("046_cutoff")
    rec.hold(preview_g, est(N046[0]))
    bt._cutoff.setValue(1.5); bt._gen_btn.click()
    rec.hold(preview_g, est(N046[1]))
    bt._cutoff.setValue(8.0); bt._gen_btn.click()
    rec.hold(preview_g, est(N046[2]))
    bt._cutoff.setValue(4.0); bt._gen_btn.click()
    rec.hold(preview_g, est(N046[3]))
    rec.finish(scene_by_name("046_cutoff"))

    # ---- 062: centroid offsets ---------------------------------------
    rec.start("062_centroid_demo")
    rec.hold(full_grab, 1.0)
    bt._cx.setValue(3.0); bt._cphi.setValue(20.0)
    rec.hold(full_grab, est(N062[0]) - 1.0)
    bt._gen_btn.click()
    rec.hold(full_grab, est(N062[1]) + est(N062[2]) - 2.0)
    bt._cx.setValue(0.0); bt._cphi.setValue(0.0); bt._gen_btn.click()
    rec.hold(full_grab, est(N062[3]))
    rec.finish(scene_by_name("062_centroid_demo"))

    # ---- 065: mismatch + the -99.999 floor ---------------------------
    rec.start("065_mismatch_demo")
    rec.hold(full_grab, est(N065[0]) - 3.0)
    bt._mx.setValue(300.0)
    rec.hold(full_grab, 3.0)
    bt._gen_btn.click()
    rec.hold(full_grab, est(N065[1]))
    bt._mx.setValue(-150.0)          # pins at the -99.999 floor
    rec.hold(full_grab, est(N065[2]))
    bt._mx.setValue(0.0); bt._gen_btn.click()
    rec.hold(full_grab, est(N065[3]) + 0.5)
    rec.finish(scene_by_name("065_mismatch_demo"))

    # ---- 070: alpha_x tilt regenerated on camera ---------------------
    rec.start("070_preview")
    rec.hold(preview_g, est(N070[0]))
    bt._alpha_x.setValue(-2.5)          # visibly different tilt
    rec.hold(preview_g, 0.8)
    bt._gen_btn.click()
    rec.hold(preview_g, est(N070[1]))
    rec.finish(scene_by_name("070_preview"))
    bt._alpha_x.setValue(1.228)         # back to the DTL beam
    bt._regen_preview()
    settle(6)

    # ---- 085: Apply, status feedback, dirty flag ---------------------
    bt._npart.setValue(120_000)         # form ahead of applied state
    settle(4)
    rec.start("085_apply_dirty")
    rec.hold(full_grab, est(N085[0]))
    bt._apply_btn.click()
    rec.hold(full_grab, est(N085[1]) + 1.5)
    rec.finish(scene_by_name("085_apply_dirty"))
    bt._npart.setValue(100_000)
    bt._apply(quiet=True)
    bt._regen_preview()
    settle(6)

    # ---- 092/094: real .dst import + clear ---------------------------
    # _import_dst opens a modal QFileDialog — the documented forever-hang
    # offscreen — so stub the module's QFileDialog to return the fixture.
    from linac_gen_gui.interphase.tabs import beam_tab as beam_tab_mod

    class _StubFileDialog:
        @staticmethod
        def getOpenFileName(*_a, **_k):
            return (str(DST_FIXTURE), "TraceWin DST (*.dst)")

    _real_filedialog = beam_tab_mod.QFileDialog
    beam_tab_mod.QFileDialog = _StubFileDialog

    rec.start("092_dst_import_live")
    rec.hold(full_grab, 1.5)
    bt._continuous.setChecked(True)     # import will untick it on camera
    rec.hold(full_grab, est(N092[0]) - 1.5)
    bt._import_dst()
    rec.hold(full_grab, est(N092[1]) + 2.0)
    rec.finish(scene_by_name("092_dst_import_live"))
    beam_tab_mod.QFileDialog = _real_filedialog

    rec.start("094_dst_clear")
    rec.hold(full_grab, 1.0)
    bt._clear_file_btn.click()
    rec.hold(full_grab, est(N094[0]) - 1.0)
    bt._gen_btn.click()
    rec.hold(full_grab, est(N094[1]) - 2.0)
    rec.finish(scene_by_name("094_dst_clear"))
    # back to the DTL proton beam for the mode scenes (off camera)
    bt._reset()
    bt._species.setCurrentText("proton")
    bt._energy.setValue(3.0)
    bt._freq.setValue(352.21)
    bt._current.setValue(20.0)
    bt._regen_preview()
    settle(8)

    # ---- 100: DC toggle greys the longitudinal column ----------------
    dc_g = union_grab([grp_particle, grp_twiss])
    rec.start("100_dc")
    rec.hold(dc_g, est(N100[0]))
    bt._continuous.setChecked(True)     # the real toggle slot fires
    rec.hold(dc_g, est(N100[1]))
    bt._continuous.setChecked(False)
    rec.hold(dc_g, 3.0)
    rec.finish(scene_by_name("100_dc"))

    # ---- 102: the DC phi-dW band preview -----------------------------
    rec.start("102_dc_preview")
    rec.hold(full_grab, 1.0)
    bt._continuous.setChecked(True)
    bt._dc_dw.setValue(2.0)
    rec.hold(full_grab, est(N102[0]) - 1.0)
    bt._gen_btn.click()
    rec.hold(full_grab, est(N102[1]) + est(N102[2]) - 2.0)
    bt._continuous.setChecked(False)
    bt._dc_dw.setValue(0.0)
    bt._gen_btn.click()
    rec.hold(full_grab, est(N102[3]))
    rec.finish(scene_by_name("102_dc_preview"))

    # ---- 104: periodic-phase enable choreography ---------------------
    part_g = crop_grab(grp_particle)
    rec.start("104_periodic_choreo")
    rec.hold(part_g, est(N104[0]))
    bt._continuous.setChecked(True)     # periodic phase enables
    rec.hold(part_g, est(N104[1]))
    bt._periodic_phase.setChecked(True)
    rec.hold(part_g, est(N104[2]))
    bt._continuous.setChecked(False)    # disables, keeps the tick
    rec.hold(part_g, est(N104[3]) - 1.0)
    rec.finish(scene_by_name("104_periodic_choreo"))
    bt._periodic_phase.setChecked(False)      # off camera
    settle(4)

    # ---- 108: Reset defaults snap-back -------------------------------
    rec.start("108_reset_demo")
    rec.hold(full_grab, 1.5)
    bt._energy.setValue(50.0)
    rec.hold(full_grab, 1.2)
    bt._dist.setCurrentText("uniform")
    rec.hold(full_grab, 1.2)
    bt._cx.setValue(5.0)
    rec.hold(full_grab, 1.2)
    bt._continuous.setChecked(True)
    rec.hold(full_grab, max(est(N108[0]) - 5.1, 1.0))
    bt._reset_btn.click()
    rec.hold(full_grab, est(N108[1]) + est(N108[2]) - 2.0)
    bt._gen_btn.click()
    rec.hold(full_grab, est(N108[3]))
    rec.finish(scene_by_name("108_reset_demo"))

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

    # 048 — distributions chapter: TL;DR table, Figure 4.1, see-also
    web_load("04_beam/01_distributions.html")
    y_tbl = web_y("tldr-pick-one")
    y_fig = min(web_y("visual-comparison") - 120, web_max_scroll())
    y_see = min(web_y("see-also") - 420, web_max_scroll())
    web_clip("048_manual_distributions", [
        ("jump", y_tbl - 140), ("hold", est(N048[0]) - 2.0),
        ("scroll", y_fig), ("hold", est(N048[1]) - 3.0),
        ("scroll", y_see), ("hold", est(N048[2]) - 1.5),
    ])

    # 106 — beam-tab chapter: DC mode, periodic phase, measured numbers
    web_load("10_gui/03_beam_tab.html")
    y_dc = web_y("dc-vs-bunched-mode")
    y_pp = web_y("periodic-phase-bunch-train")
    y_num = web_strong_y("With space charge off")
    y_con = web_strong_y("Backtracking refuses")
    web_clip("106_periodic_manual", [
        ("jump", y_dc - 150), ("hold", est(N106[0]) - 1.5),
        ("scroll", y_pp - 130), ("hold", est(N106[1]) - 2.5),
        ("scroll", y_num - 260), ("hold", est(N106[2]) - 2.5),
        ("scroll", min(y_con - 220, web_max_scroll())),
        ("hold", est(N106[3]) - 2.0),
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
