"""Ring tab — ring mode (multi-turn tracking of a synchrotron), opt-in.

Shown only while Simulate → Ring Mode is on; a linac session never creates
it.  The ring verdict for the loaded lattice heads four pages:

* **Ring & Optics** — the ring's periodic optics from the ring kernels
  (tunes with their integer part, chromaticity, momentum compaction,
  transition, Twiss and dispersion along the ring; with RF, the synchrotron
  tune and the bucket), the element model they use, and a button that gives
  the Beam tab the beam matched to the ring at the start of the run (the
  Twiss functions x, y and, with the RF on, z, the dispersion, the closed
  orbit as the centroid);
* **Programs** — the ramp (:mod:`.ring_programs`): the reference energy,
  the RF voltage and phase versus time, transition, paraphasing, with a
  preview of the reference particle alone;
* **Space charge** — the 2.5-D model (:mod:`.ring_space_charge`): off,
  particle-in-cell or a frozen Gaussian, the bunch's intensity, the kicks,
  the grid, the pipe, the longitudinal kick, with the analytic tune shift;
* **Injection** — multi-turn injection through a stripping foil
  (:mod:`.ring_injection`): the turns, macroparticles and current, the foil,
  the injected beam, the chopper, the bump and painting magnets' waveforms;
* **Run** — N turns of a bunch through the ring kernels, at the ring's
  fixed reference energy or through the ramp, with or without space charge
  and injection (every CPU core, or a CUDA GPU), starting from the Beam
  tab's beam (its distribution, particles, emittances and Twiss functions,
  at the ring's species and energy) or from none (the injection fills the
  ring); progress, Stop, and a live strip — the
  survival, the normalized emittances x and y and the longitudinal emittance
  turn by turn while the run goes.  A run's results are in the Results tab
  ("Show in the Results tab", or its "Run shown: Ring run"): the beam turn by
  turn in its tiles and, under RING RUN, the ramp, the bunch in eV s, the
  losses, the tunes, phase space at the snapshot turns (with the separatrix
  of the turn's bucket), the periodic optics and the injection.  Every run (a
  stopped one too, and one stopped by quitting HELIX if it ends its turn
  within the five seconds HELIX waits) is written to the calc directory as
  ``<time>_ring.h5``.
"""
from __future__ import annotations

import contextlib
import dataclasses
import sys

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import theme

_RUN_NOTE = ("Turns of the bunch through the ring kernels, at the ring's fixed reference "
             "energy or through the ramp of the Programs page, with the space charge of the "
             "Space charge page and the injection of the Injection page: chromaticity, "
             "nonlinear elements, apertures and the RF bucket included.")

_SOURCES = (("The Beam tab's beam", "beam"),
            ("None: the injection fills the ring", "empty"))

#: the Beam tab's settings "Use the matched beam in the Beam tab" writes
#: (with the RF on at the start; without, the longitudinal ones stay)
_MATCHED_KEYS = ("alpha_x", "beta_x", "alpha_y", "beta_y", "disp_x", "disp_xp", "disp_y",
                 "disp_yp", "alpha_z", "beta_z", "centroid_x", "centroid_xp", "centroid_y",
                 "centroid_yp", "centroid_dphi", "centroid_dw")


def migration_note(old_count) -> str:
    """The Run page's note for a project whose ring run started from a beam
    matched to the ring (``old_count`` particles; None: not saved)."""
    count = (f" of {old_count} particles"
             if isinstance(old_count, int) and not isinstance(old_count, bool) else "")
    return (f"This project's ring run started from a beam{count} matched to the ring. A "
            "ring run now takes the Beam tab's beam, its particles and Twiss functions: "
            "Ring & Optics → Use the matched beam in the Beam tab gives the Beam tab the "
            "beam matched to the ring.")

_OPTICS_NOTE = ("The optics come from the ring kernels themselves: closed orbit and "
                "one-turn map by tracking, so the chromaticity includes every element's "
                "full map. Cartesian multipoles: the bend multipoles expanded in Cartesian "
                "coordinates, exact pole faces by default; "
                "curvilinear multipoles: exact elements with the bend multipoles in the "
                "curved frame, linear pole faces.")

_STOPPED_MSG = "Stopped: the lattice changed."
#: the Run page's line for a run stopped by another lattice (Open, Reload, a
#: project, a matching Apply: a new lattice object — an edit leaves a run going)
_RUN_STOPPED_MSG = "Stopped: the lattice was replaced."

_MODELS = (("Cartesian multipoles", "cartesian"), ("curvilinear multipoles", "curvilinear"))
_EDGES = (("exact", "exact"), ("full", "full"), ("linear", "linear"))
_EDGES_TIP = ("exact: rotation into the pole face, the hard-edge dipole fringe (also at "
              "a face angle of zero), the wedge's quadrupole field in MAD8's form, the "
              "dipole wedge.\nfull: as exact, with the quadrupole's hard-edge fringe and "
              "the wedge's quadrupole field to first order in the face angle.\nlinear: "
              "thin linear kicks with the fringe correction (the curvilinear model's "
              "only pole faces).")
_CONSTANTS = (("CODATA 2018 (default)", "helix"), ("CODATA 2010", "codata2010"),
              ("CODATA 2022", "codata2022"))


def _snapshot_text(saved) -> str:
    """The Snapshot turns field for a project's list (absent: '0, last')."""
    if not isinstance(saved, list):
        return "0, last"
    return ", ".join(str(t) for t in saved)


# (the run's memory estimate lives in linac_gen.ring.memory — shared with the
# assistant's run_ring; these names stay for the tab and its tests, and load
# ring mode only when called)
def _physical_memory() -> int | None:
    from linac_gen.ring.memory import physical_memory
    return physical_memory()


def _run_memory(n_particles: int, n_snapshots: int, n_rows: int, n_probes: int,
                n_record: int) -> int:
    from linac_gen.ring.memory import run_memory
    return run_memory(n_particles, n_snapshots, n_rows, n_probes, n_record)


def _injection_record_memory(injection: dict, n_turns: int) -> int:
    from linac_gen.ring.memory import injection_record_memory
    return injection_record_memory(injection, n_turns)


def _space_charge_memory(sc: dict | None, n_starts: int | None = None) -> int:
    from linac_gen.ring.memory import space_charge_memory
    return space_charge_memory(sc, n_starts)


def _turns_from(config) -> int:
    """The number of turns a saved ring setting holds (10 when absent or not
    a positive whole number)."""
    try:
        n = int((config or {}).get("n_turns", 10))
    except (TypeError, ValueError, AttributeError):
        return 10
    return n if n >= 1 else 10


def _numba_installed() -> bool:
    """Whether numba can be found — without importing it on the GUI thread
    (the worker imports it)."""
    import importlib.util
    try:
        return importlib.util.find_spec("numba") is not None
    except (ImportError, ValueError):
        return False


def _declared_beam(lattice) -> dict | None:
    """The beam the lattice's deck declares, or None
    (:func:`linac_gen.ring.config.declared_beam`: only a declared beam sets
    the ring's species and energy; ``ring_reference_beam`` without one only
    scaled the magnets).  From the settings module, so that the tab does not
    import the ring compiler, and numba with it, on the GUI thread."""
    from linac_gen.ring.config import declared_beam
    return declared_beam(lattice)


def _label(choices, key) -> str:
    return next((label for label, k in choices if k == key), str(key))


def _choice(config, key, choices, default):
    v = (config or {}).get(key, default) if isinstance(config, dict) else default
    if key == "model" and isinstance(v, str):
        # (a project saved with the model names of earlier versions)
        from linac_gen.ring.config import MODEL_ALIASES
        v = MODEL_ALIASES.get(v, v)
    return v if v in [c[1] for c in choices] else default


def _injection_summary(inj) -> str:
    """A run's foil and injection in a sentence ("" without a foil; the real
    particles the ring's)."""
    if not inj:
        return ""
    import numpy as np
    f = inj.get("foil") or {}
    w = inj.get("waste") or {}
    mh = inj.get("mean_hits_circulating")
    hits = (f", {float(mh):.3g} per circulating particle" if mh is not None
            and np.isfinite(float(mh)) else "")
    out = ""
    if inj.get("settings"):
        b = int(inj.get("buckets") or 1)
        out = (f" Injection: {int(np.sum(inj['injected']))} macroparticles, "
               f"{float(np.sum(inj['injected_charge'])):.4g} particles into the ring"
               + (f" ({b} filled buckets)" if b > 1 else "")
               + f"; waste: missed the foil {w.get('missed', 0)}, not stripped "
               f"{w.get('unstripped', 0)}.")
    return out + (f" Foil {f.get('name')}: {int(np.sum(inj['foil_hits']))} crossings{hits}; "
                  f"theta0 {float(f.get('theta0_rad', 0.0)) * 1e6:.4g} urad for one crossing, mean loss "
                  f"{float(f.get('mean_loss_mev', 0.0)) * 1e3:.4g} keV a crossing.")


def _sc_summary(m) -> str:
    """A run's space charge in a sentence ("" without)."""
    if not m:
        return ""
    model = "2.5-D particle-in-cell" if m.get("model") == "pic" else "frozen Gaussian"
    q = m.get("laslett_gaussian")
    where = "at the bunch centre" if m.get("bunched") else "of the coasting beam"
    est = (f"; analytic tune shift {where} dQx {q[0]:.4g}, dQy {q[1]:.4g}" if q else "")
    off = (f"; at most {m['outside_grid_max']} particle(s) off the grid at a kick"
           if m.get("outside_grid_max") else "")
    n_b = m.get("intensity_used")
    n_b = m.get("intensity") if n_b is None else n_b
    src = ("" if m.get("intensity") is not None else
           " (circulating at the end, from the injected current)"
           if m.get("intensity_from") == "injection" else " (from the Beam tab's current)")
    where_n = "the bunch" if m.get("bunched", True) else "the ring"
    bins = ""
    if m.get("bunched"):
        used = {}
        for t, b in m.get("z_bins_used") or ():
            used[int(t)] = int(b)                  # (at a turn, the last bins set)
        if m.get("z_bins") == "auto" and used:
            lo, hi = min(used.values()), max(used.values())
            bins = ("; line-density bins following the bunch, "
                    + (f"{lo}" if lo == hi else f"{lo} to {hi}"))
        elif m.get("z_bins") not in (None, "auto"):
            bins = f"; {m['z_bins']} line-density bins"
    return (f" Space charge: {model}, {float(n_b or 0.0):.4g} particles in {where_n}{src}, "
            f"{m.get('kicks')} kicks per turn{bins}{est}{off}.")


class RingTab(QWidget):
    """Ring mode's page: verdict, optics, run controls and per-turn results."""

    #: assigned by app.py — returns the Beam tab's current BeamConfig
    beam_config_provider = None
    #: assigned by app.py — puts a BeamConfig into the Beam tab
    beam_config_setter = None
    #: "Show in the Results tab" pressed (app.py shows the ring run there)
    show_in_results_requested = pyqtSignal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._worker = None
        self._run_lattice = None        # the lattice object the run was started on
        self._optics_worker = None
        self._matched_worker = None     # Use the matched beam in the Beam tab
        self._load_note = None          # a loaded project's note (Run page)
        self._last_dump = None          # the results file of the current run
        self._dump_note = None          # why it cannot be written (no calc dir)
        self._optics = None             # the last optics result (dict from the worker)
        # (species, energy) of the Beam tab the optics were computed with —
        # None when the lattice declares its own beam (a MAD-X deck's BEAM)
        self._optics_beam = None
        self._build_ui()
        state.lattice_changed.connect(self._on_lattice_changed)
        state.beam_config_changed.connect(self._on_beam_changed)
        state.ring_results_changed.connect(self._show_results)
        self._on_lattice_changed(state.lattice)
        self._show_results(state.ring_results)
        self._record_source()
        self._update_beam_line()

    def showEvent(self, event) -> None:
        # (the Beam tab may have changed meanwhile without an Apply: the Run
        # page says what a run would start with now)
        super().showEvent(event)
        self._update_beam_line()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        title = QLabel("Ring mode")
        title.setStyleSheet(f"font-weight:600; color:{theme.TEXT_0};")
        v.addWidget(title)
        self._verdict = QLabel("")
        self._verdict.setWordWrap(True)
        v.addWidget(self._verdict)
        self._chips = QLabel("")
        self._chips.setStyleSheet(f"color:{theme.ACCENT}; font-family:{theme.FONT_MONO};")
        self._chips.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._chips)
        self._pages = QTabWidget()
        self._pages.addTab(self._build_optics_page(), "Ring && Optics")
        self._pages.addTab(self._build_programs_page(), "Programs")
        self._pages.addTab(self._build_space_charge_page(), "Space charge")
        self._pages.addTab(self._build_injection_page(), "Injection")
        self._pages.addTab(self._build_run_page(), "Run")
        v.addWidget(self._pages, stretch=1)
        self._programs.changed.connect(self._on_programs_changed)
        self._space_charge.changed.connect(self._on_programs_changed)
        self._injection.changed.connect(self._on_programs_changed)
        self._injection.switched.connect(self._on_injection_switched)
        self._on_programs_changed()

    def _build_programs_page(self) -> QWidget:
        from .ring_programs import RingProgramsPage
        self._programs = RingProgramsPage(self.state, options_provider=self._optics_options,
                                          beam_provider=self._beam_config,
                                          turns_setter=self._set_turns,
                                          turns_provider=self._get_turns)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._programs)
        return scroll

    def _build_space_charge_page(self) -> QWidget:
        from .ring_space_charge import RingSpaceChargePage
        self._space_charge = RingSpaceChargePage(self.state,
                                                 optics_provider=self._current_optics,
                                                 beam_provider=self._beam_config,
                                                 injection_provider=self._injection_on)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._space_charge)
        return scroll

    def _build_injection_page(self) -> QWidget:
        from .ring_injection import RingInjectionPage
        self._injection = RingInjectionPage(self.state, optics_provider=self._current_optics)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._injection)
        return scroll

    def _current_optics(self):
        """The optics shown on the Ring & Optics page (its worker's dict) or None."""
        return getattr(self, "_optics", None)

    def _injection_on(self) -> bool:
        """The Injection page's injection is on (False before the page exists)."""
        page = getattr(self, "_injection", None)
        return page is not None and page.is_on()

    def _on_injection_switched(self, on: bool) -> None:
        """Injection turned on starts the ring empty (the Beam tab's beam can
        be chosen after); turned off, the empty start gives way to the Beam
        tab's beam."""
        # (with injection the space charge is the circulating charge it builds)
        self._space_charge.update_estimate()
        keys = [k for _l, k in _SOURCES]
        cur = self._source.currentData()
        if on and cur != "empty":
            self._source.setCurrentIndex(keys.index("empty"))
            self.state.status_message.emit("Injection on: the run starts with an empty ring "
                                           "(Run page: Particles from).")
        elif not on and cur == "empty":
            self._source.setCurrentIndex(keys.index("beam"))

    def _update_beam_line(self, *_args) -> None:
        """The Run page's line under Particles from: the beam the run starts
        with (the Beam tab's), and a loaded project's note."""
        if not hasattr(self, "_beam_line"):
            return
        if self._source.currentData() == "empty":
            text = "No beam at the start: the injection's particles fill the ring."
        else:
            err = None
            try:
                cfg = self._beam_config()
            except Exception as exc:          # noqa: BLE001 — an invalid Beam form
                cfg, err = None, exc
            if err is not None:
                text = f"The Beam tab's settings are not valid: {err}"
            elif cfg is None:
                text = "Set the beam in the Beam tab."
            elif getattr(cfg, "source", "generate") == "file":
                import os
                text = ("The Beam tab's distribution file "
                        f"{os.path.basename(str(cfg.distribution_file or ''))}: its particles "
                        "as read; it must hold the ring's species at the ring's energy.")
            else:
                # (the cutoff shapes the Gaussian and thermal distributions only)
                cut = (f" cut at {float(cfg.cutoff):g} rms"
                       if str(cfg.distribution) in ("gaussian", "thermal") else "")
                text = (f"{int(cfg.n_particles)} particles of the Beam tab: "
                        f"{cfg.distribution}{cut}, normalized "
                        f"rms emittances {float(cfg.emit_nx):g} and {float(cfg.emit_ny):g} "
                        f"π mm mrad, emit_z {float(cfg.emit_z):g} deg·MeV at "
                        f"{float(cfg.frequency):g} MHz, with the Beam tab's Twiss functions "
                        "(Ring & Optics → Use the matched beam in the Beam tab sets those of "
                        "the beam matched to the ring at the start of the run).")
            if self._injects_beam_tab():
                text += " The Injection page injects the Beam tab's beam as well."
        note = getattr(self, "_load_note", None)
        self._beam_line.setText(text + (f"\n{note}" if note else ""))

    def _injects_beam_tab(self) -> bool:
        """Injection on, its injected beam the Beam tab's distribution."""
        try:
            inj = self._injection.injection_config()
        except (AttributeError, ValueError):
            return False
        return bool(inj) and (inj.get("source") or {}).get("kind", "beam") == "beam"

    def _record_source(self) -> None:
        """The Run page's start always goes into the project's ring settings
        (an earlier version left it out when it was its default, the matched
        beam: a project without it is one of those), without marking the
        project changed."""
        cfg = dict(self.state.ring_config or {})
        src = self._source.currentData()
        if cfg.get("source") != src:
            cfg["source"] = src
            self.state.ring_config = cfg

    def set_load_note(self, note: str | None) -> None:
        """A loaded project's note about its ring settings (shown on the Run
        page until the next run or project)."""
        self._load_note = note or None
        self._update_beam_line()

    def commit_pending(self) -> None:
        """Entries still being typed go into the project (before it is saved:
        a Save does not end an edit) — the Run page's snapshot turns, the line
        edits of the Programs, Space charge and Injection pages, and a table
        cell being edited on the Programs or Injection page."""
        if self._snapshots.isModified():
            self._on_run_setting_changed()
        self._programs.commit_pending()
        self._space_charge.commit_pending()
        self._injection.commit_pending()

    def _beam_config(self):
        provider = self.beam_config_provider
        return provider() if callable(provider) else self.state.beam_config

    def _set_turns(self, n: int) -> bool:
        """The Run page's turns (False: not now — a run is going on)."""
        if not self._turns.isEnabled():
            return False
        self._turns.setValue(int(n))
        return True

    def _get_turns(self) -> int:
        return int(self._turns.value())

    def _on_programs_changed(self) -> None:
        if hasattr(self, "_ramp_line"):
            self._ramp_line.setText(self._programs.summary_text() + "\n"
                                    + self._space_charge.summary_text() + "\n"
                                    + self._injection.summary_text())
        # (a matched beam being computed for the pages as they were)
        if hasattr(self, "_to_beam_btn"):
            self._stop_matched("The Programs, Space charge or Injection page changed")
        self._update_beam_line()

    def _build_optics_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        note = QLabel(_OPTICS_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(note)
        box = QGroupBox("Element model")
        form = QFormLayout(box)
        cfg = self.state.ring_config
        self._model = QComboBox()
        for label, key in _MODELS:
            self._model.addItem(label, key)
        self._model.setCurrentIndex([k for _l, k in _MODELS].index(
            _choice(cfg, "model", _MODELS, "cartesian")))
        form.addRow("Model", self._model)
        self._edge = QComboBox()
        self._edge.setToolTip(_EDGES_TIP)
        for label, key in _EDGES:
            self._edge.addItem(label, key)
        self._edge.setCurrentIndex([k for _l, k in _EDGES].index(
            _choice(cfg, "edge_model", _EDGES, "exact")))
        form.addRow("Pole faces", self._edge)
        self._constants = QComboBox()
        for label, key in _CONSTANTS:
            self._constants.addItem(label, key)
        self._constants.setCurrentIndex([k for _l, k in _CONSTANTS].index(
            _choice(cfg, "constants", _CONSTANTS, "helix")))
        form.addRow("Constants", self._constants)
        self._six_d = QCheckBox("with RF: 6-D closed orbit, synchrotron tune, bucket")
        form.addRow("", self._six_d)
        row = QHBoxLayout()
        self._optics_btn = QPushButton("Compute optics")
        self._optics_btn.clicked.connect(self.compute_optics)
        self._to_beam_btn = QPushButton("Use the matched beam in the Beam tab")
        self._to_beam_btn.setToolTip(
            "Give the Beam tab the beam matched to the ring at the start of the run: the "
            "periodic Twiss functions x and y, the dispersion and, with the RF on, the "
            "longitudinal Twiss functions at the start of the ring, from the optics of the "
            "run's first turn (this element model, the Programs page's ramp at turn 0, the "
            "Injection page's magnets at turn 0), and the closed orbit as the centroid; "
            "for a lattice that declares its beam (a MAD-X deck's BEAM), also that species "
            "and the run's starting energy.  The emittances, the distribution and the number "
            "of particles stay the Beam tab's.")
        self._to_beam_btn.clicked.connect(self.use_matched_beam)
        self._to_beam_btn.setEnabled(False)
        row.addWidget(self._optics_btn)
        row.addWidget(self._to_beam_btn)
        row.addStretch(1)
        form.addRow(row)
        self._optics_status = QLabel("")
        self._optics_status.setWordWrap(True)
        form.addRow("", self._optics_status)
        v.addWidget(box)
        # (the pole faces chosen for the cartesian model, restored after a
        # visit to the curvilinear model, which has linear ones only)
        self._cartesian_edge = (self._edge.currentData()
                                if self._model.currentData() != "curvilinear" else "exact")
        self._model.currentIndexChanged.connect(self._on_model_changed)
        self._edge.currentIndexChanged.connect(self._on_edge_changed)
        self._constants.currentIndexChanged.connect(self._on_option_changed)
        self._six_d.toggled.connect(self._on_six_d_toggled)
        self._sync_edge_choices()

        self._optics_text = QLabel("No optics yet.")
        self._optics_text.setWordWrap(True)
        self._optics_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._optics_text.setStyleSheet(f"font-family:{theme.FONT_MONO};")
        v.addWidget(self._optics_text)
        self._twiss_plot = pg.PlotWidget()
        self._twiss_plot.setBackground(theme.BG_1)
        self._twiss_plot.setLabel("left", "beta (m)")
        self._twiss_plot.addLegend()
        self._curve_bx = self._twiss_plot.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[0], width=2),
                                               name="beta x")
        self._curve_by = self._twiss_plot.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[1], width=2),
                                               name="beta y")
        self._twiss_plot.setMinimumHeight(150)
        v.addWidget(self._twiss_plot, stretch=2)
        self._disp_plot = pg.PlotWidget()
        self._disp_plot.setBackground(theme.BG_1)
        self._disp_plot.setLabel("bottom", "s (m)")
        self._disp_plot.setLabel("left", "D x (m)")
        self._disp_plot.setXLink(self._twiss_plot)
        self._curve_dx = self._disp_plot.plot([], [], pen=pg.mkPen(theme.PLOT_PALETTE[2], width=2))
        self._disp_plot.setMinimumHeight(120)
        v.addWidget(self._disp_plot, stretch=1)
        # (a short window scrolls the page rather than squeezing its controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _build_run_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        note = QLabel(_RUN_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(note)
        run_box = QGroupBox("Run")
        form = QFormLayout(run_box)
        cfg = self.state.ring_config or {}
        self._ramp_line = QLabel("")
        self._ramp_line.setWordWrap(True)
        form.addRow("", self._ramp_line)
        self._turns = QSpinBox()
        self._turns.setRange(1, 10_000_000)
        self._turns.setValue(_turns_from(cfg))
        self._turns.valueChanged.connect(self._on_turns_changed)
        form.addRow("Turns", self._turns)
        self._source = QComboBox()
        for label, key in _SOURCES:
            self._source.addItem(label, key)
        self._source.setToolTip("The particles the ring starts with: the Beam tab's beam (its "
                                "distribution, number of particles, emittances and Twiss "
                                "functions, generated at the ring's species and energy), or "
                                "none when the injection fills the ring")
        self._source.setCurrentIndex([k for _l, k in _SOURCES].index(
            _choice(cfg, "source", _SOURCES, "beam")))
        form.addRow("Particles from", self._source)
        # (what the run starts with, from the Beam tab; and a project's note)
        self._beam_line = QLabel("")
        self._beam_line.setWordWrap(True)
        self._beam_line.setStyleSheet(f"color:{theme.TEXT_2};")
        form.addRow("", self._beam_line)
        self._probes = QSpinBox()
        self._probes.setRange(0, 12)
        self._probes.setValue(int(cfg.get("tune_probes", 1)) if
                              str(cfg.get("tune_probes", "")).isdigit() else 1)
        self._probes.setToolTip("0: none; 1: one small-amplitude probe (the tunes); n: an n x n "
                                "grid of amplitudes up to 3 rms sizes (the footprint)")
        form.addRow("Tune probes", self._probes)
        self._snapshots = QLineEdit(_snapshot_text(cfg.get("snapshot_turns")))
        self._snapshots.setToolTip("turns at which the whole distribution is kept: numbers, "
                                   "'last' for the last turn")
        form.addRow("Snapshot turns", self._snapshots)
        self._device = QComboBox()
        self._device.addItem("CPU (every core)", "cpu")
        self._device.addItem("CUDA GPU", "cuda")
        if sys.platform == "darwin":
            # (no CUDA on macOS: the item stays visible, greyed, with the reason)
            item = self._device.model().item(1)
            item.setEnabled(False)
            item.setToolTip("No CUDA GPU on macOS — the CPU runs the same kernels")
        elif _choice(cfg, "device", (("", "cpu"), ("", "cuda")), "cpu") == "cuda":
            self._device.setCurrentIndex(1)
        form.addRow("Device", self._device)
        for sig in (self._source.currentIndexChanged, self._probes.valueChanged,
                    self._snapshots.editingFinished, self._device.currentIndexChanged):
            sig.connect(self._on_run_setting_changed)
        self._source.currentIndexChanged.connect(self._update_beam_line)
        row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self.run)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self.stop)
        self._stop_btn.setEnabled(False)
        row.addWidget(self._run_btn)
        row.addWidget(self._stop_btn)
        row.addStretch(1)
        form.addRow(row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        form.addRow("Progress", self._progress)
        self._turn_label = QLabel("")
        form.addRow("", self._turn_label)
        v.addWidget(run_box)

        res_box = QGroupBox("Results")
        rv = QVBoxLayout(res_box)
        self._summary = QLabel("No ring run yet.")
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv.addWidget(self._summary)
        row = QHBoxLayout()
        self._to_results_btn = QPushButton("Show in the Results tab")
        self._to_results_btn.setToolTip(
            "The ring run in the Results tab: the beam turn by turn in its tiles and, under "
            "RING RUN, the ramp, the bunch in eV s, the losses, the tunes, phase space, the "
            "optics and the injection")
        self._to_results_btn.setEnabled(False)
        self._to_results_btn.clicked.connect(self.show_in_results_requested)
        row.addWidget(self._to_results_btn)
        row.addStretch(1)
        rv.addLayout(row)
        from ..panels.ring_plots import RingLiveStrip
        self._live_strip = RingLiveStrip()
        rv.addWidget(self._live_strip, stretch=1)
        v.addWidget(res_box, stretch=1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    # ------------------------------------------------------------------
    # verdict and results
    # ------------------------------------------------------------------
    def _on_lattice_changed(self, lattice) -> None:
        # another lattice (a new object: Open, Reload, a project, a matching
        # Apply): the run of the old one is over, and nothing it still sends
        # may reach the new lattice's results.  An edit sends the same object
        # again (the command bus: the inspector, undo and redo, the
        # assistant's changes) and leaves a run going — it runs on its own
        # copy.  Either way the optics, the matched beam and the ramp preview
        # describe the lattice as it was: they go.
        if self._worker is not None and lattice is not self._run_lattice:
            running = self._worker.isRunning()
            self._detach()
            if running:
                self._turn_label.setText(_RUN_STOPPED_MSG)
        stopped = self._detach_optics()
        stopped = self._detach_matched() or stopped
        self._show_optics(None)
        self._to_beam_btn.setEnabled(lattice is not None)
        self._programs.invalidate("")
        self._injection.set_lattice_elements()
        self._injection.update_estimate()
        # (the signal can come twice for one new lattice: the second must not
        # erase what the first said)
        if stopped:
            self._optics_status.setText(_STOPPED_MSG)
        elif self._optics_status.text() != _STOPPED_MSG:
            self._optics_status.setText("")
        if lattice is None:
            self._verdict.setText("No lattice loaded.")
            return
        try:
            from linac_gen.ring.detect import detect_ring
            verdict = detect_ring(lattice)
        except Exception as exc:          # noqa: BLE001 — a verdict must never break the tab
            self._verdict.setText(f"Ring check failed: {exc}")
            return
        reason = verdict.reason[:1].upper() + verdict.reason[1:]
        imp = _declared_beam(lattice)
        if imp:
            ref = (f" Reference: {imp['species']} at {imp['energy']:.6g} MeV, the beam the "
                   "deck's magnets were scaled with.")
        elif getattr(lattice, "ring_reference_beam", None):
            # (an elegant deck, a MAD-X BEAM without an energy: the magnets
            # keep the normalized strengths the deck gives)
            ref = (" Reference: the Beam tab's species and energy — the deck declares no "
                   "beam; its magnets keep the normalized strengths it gives.")
        else:
            ref = (" Reference: the Beam tab's species and energy — the lattice declares no "
                   "beam; set them to the energy its magnets are scaled for.")
        self._verdict.setText(
            f"{reason}. Length {verdict.length_m:.6f} m"
            + (f", RF harmonic {verdict.harmonic}" if verdict.harmonic else "") + "." + ref)

    def _show_results(self, results) -> None:
        self._to_results_btn.setEnabled(results is not None)
        if results is None:
            self._summary.setText("No ring run yet.")
            self._live_strip.clear()
            return
        done = results.turns_done
        what = "stopped after" if results.truncated else "done:"
        m = results.meta
        from linac_gen.ring.results import survival_text
        inj = getattr(results, "injection", None) or {}
        n0 = int(results.n_alive[0]) if len(results.n_alive) else 0
        if inj.get("settings"):
            # (with injection: of the particles the ring received — the stripped ones)
            w = inj.get("waste") or {}
            n0 = int(results.ledger.get("injected", 0)) - int(w.get("missed", 0)) \
                - int(w.get("unstripped", 0))
        surv = (survival_text(n0, int(results.n_alive[-1])) if len(results.n_alive) else "n/a")
        t = results.tunes or {}
        tunes = ""
        if t:
            tunes = " NAFF tunes " + ", ".join(
                f"{k.capitalize()} = {t[k]:.6f}" for k in ("qx", "qy", "qs") if k in t) + "."
        lost_all = (f" Every particle was lost by turn {m['all_lost_at_turn']}."
                    if m.get("all_lost_at_turn") is not None else "")
        dump = (f" Saved: {m['dump_path']}." if m.get("dump_path") else
                (f" Not saved: {m['dump_error']}." if m.get("dump_error") else ""))
        every = int(m.get("stats_every", 1) or 1)
        rows = f" Statistics every {every} turns." if every > 1 else ""
        # (the starting beam's mismatch is said below, with its BMAG)
        notes = "".join(f" Note: {n}." for n in m.get("notes", [])
                        if not str(n).startswith("the beam at the start is mismatched"))
        bm = [v for v in (m.get("start_mismatch") or []) if v is not None]
        if bm:
            # (the starting beam against the ring's matched one, per normal mode)
            notes += (" At the start, BMAG (1 when matched) "
                      + ", ".join(f"{p} {v:.4f}" for p, v in zip("xyz", m["start_mismatch"],
                                                                  strict=False)
                                  if v is not None) + ".")
            from linac_gen.ring.engine import MISMATCH_NOTE
            if max(bm) > MISMATCH_NOTE:
                notes += (" Ring & Optics → Use the matched beam in the Beam tab gives the "
                          "Beam tab the matched beam.")
        sc = _sc_summary(m.get("space_charge")) + _injection_summary(inj)
        self._summary.setText(
            f"Run {what} {done} of {results.n_turns_requested} turn(s), "
            f"{m.get('n_particles', '?')} particles, {results.wall_s:.2f} s "
            f"({m.get('device', '?')}, {m.get('threads', '?')} threads). Survival "
            f"{surv}.{tunes}{sc}{lost_all}{rows}{notes}{dump}")
        self._live_strip.show_results(results)

    # ------------------------------------------------------------------
    # optics
    # ------------------------------------------------------------------
    def _sync_edge_choices(self) -> None:
        # the curvilinear model has linear pole faces only; the cartesian
        # model gets back the pole faces chosen for it
        curvilinear = self._model.currentData() == "curvilinear"
        want = "linear" if curvilinear else self._cartesian_edge
        self._edge.blockSignals(True)
        self._edge.setCurrentIndex([k for _l, k in _EDGES].index(want))
        self._edge.blockSignals(False)
        self._edge.setEnabled(not curvilinear)

    def _on_model_changed(self, _i: int) -> None:
        self._sync_edge_choices()
        self._on_option_changed(_i)

    def _on_edge_changed(self, _i: int) -> None:
        if self._model.currentData() != "curvilinear":
            self._cartesian_edge = self._edge.currentData()
        self._on_option_changed(_i)

    def _on_option_changed(self, _i: int) -> None:
        cfg = dict(self.state.ring_config or {})
        cfg.update(self._optics_options())
        self.state.ring_config = cfg
        self.state.mark_project_dirty()
        self._invalidate_optics("The element model changed: compute the optics again.")
        self._stop_matched("The element model changed")
        self._programs.invalidate("")

    def _on_six_d_toggled(self, _on: bool) -> None:
        self._invalidate_optics("The RF setting changed: compute the optics again.")

    def _on_beam_changed(self, cfg) -> None:
        # (a matched beam being computed for the Beam tab as it was: its phase
        # unit — the frequency — or, for a lattice without its own beam, the
        # magnets' scaling may have changed)
        self._stop_matched("The Beam tab changed")
        # (a lattice without its own beam is scaled with the Beam tab's: a
        # new species or energy gives other optics)
        if self._optics_beam is not None and self._beam_key(cfg) != self._optics_beam:
            self._invalidate_optics("The Beam tab's species or energy changed: compute the "
                                    "optics again.")
        if _declared_beam(self.state.lattice) is None:
            # (the ring runs at the Beam tab's beam: a preview of the ramp
            # may no longer hold)
            self._programs.invalidate("")
        self._space_charge.update_estimate()
        self._update_beam_line()

    def _invalidate_optics(self, msg: str) -> None:
        """Optics shown or being computed for other settings are dropped."""
        stopped = self._detach_optics()
        if stopped or self._optics is not None:
            self._show_optics(None)
            self._optics_status.setText(msg)

    @staticmethod
    def _beam_key(cfg):
        if cfg is None:
            return None
        return (str(getattr(cfg, "species", "")), float(getattr(cfg, "energy", 0.0) or 0.0))

    def _optics_options(self) -> dict:
        return {"model": self._model.currentData(), "edge_model": self._edge.currentData(),
                "constants": self._constants.currentData()}

    def load_config(self) -> None:
        """Show the ring settings held in the state (a project was loaded)."""
        self._stop_matched("The project's ring settings were loaded")
        self._show_settings()
        self._invalidate_optics("The project's ring settings were loaded: compute the optics "
                                "again.")

    def reload_settings(self) -> None:
        """The state's ring settings were set from outside the tab (the
        assistant): show them on every page — the optics shown stay unless the
        element model changed."""
        before = self._optics_options()
        self._stop_matched("The ring settings changed")
        self._show_settings()
        if self._optics_options() != before:
            self._invalidate_optics("The element model changed: compute the optics again.")

    def current_settings(self) -> dict:
        """The ring settings the Run button would use: the state's, with the
        Run page's values and each page's settings as they stand (a page whose
        settings are not valid keeps the state's)."""
        from linac_gen.ring.config import parse_snapshot_turns
        cfg = dict(self.state.ring_config or {})
        cfg.pop("n_particles", None)          # (the Beam tab's)
        cfg.update({"n_turns": int(self._turns.value()),
                    "source": self._source.currentData(),
                    "tune_probes": int(self._probes.value()),
                    "device": self._device.currentData(), **self._optics_options()})
        with contextlib.suppress(ValueError):
            cfg["snapshot_turns"] = parse_snapshot_turns(self._snapshots.text())
        for key, page_config in (("ramp", self._programs.ramp_config),
                                 ("space_charge", self._space_charge.space_charge_config),
                                 ("injection", self._injection.injection_config)):
            try:
                value = page_config()
            except ValueError:
                continue
            if value is None:
                cfg.pop(key, None)
            else:
                cfg[key] = value
        # (the steerers' choice is the ramp's: at fixed energy they keep their kicks)
        cfg["corrector_mode"] = (self._programs.corrector_mode() if cfg.get("ramp")
                                 else "follow")
        return cfg

    def _show_settings(self) -> None:
        cfg = self.state.ring_config
        run_widgets = (self._source, self._probes, self._snapshots, self._device)
        for w in (self._turns, self._model, self._edge, self._constants, *run_widgets):
            w.blockSignals(True)
        # (a setting the project does not hold is the Run page's default:
        # nothing is kept from the project open before)
        self._turns.setValue(_turns_from(cfg))
        self._source.setCurrentIndex([k for _l, k in _SOURCES].index(
            _choice(cfg, "source", _SOURCES, "beam")))
        k = cfg.get("tune_probes")
        self._probes.setValue(k if isinstance(k, int) and k >= 0 else 1)
        self._snapshots.setText(_snapshot_text(cfg.get("snapshot_turns")))
        if cfg.get("device") == "cuda" and self._device.model().item(1).isEnabled():
            self._device.setCurrentIndex(1)
        else:
            self._device.setCurrentIndex(0)
        for w in run_widgets:
            w.blockSignals(False)
        self._model.setCurrentIndex([k for _l, k in _MODELS].index(
            _choice(cfg, "model", _MODELS, "cartesian")))
        self._edge.setCurrentIndex([k for _l, k in _EDGES].index(
            _choice(cfg, "edge_model", _EDGES, "exact")))
        self._constants.setCurrentIndex([k for _l, k in _CONSTANTS].index(
            _choice(cfg, "constants", _CONSTANTS, "helix")))
        for w in (self._turns, self._model, self._edge, self._constants):
            w.blockSignals(False)
        self._cartesian_edge = (self._edge.currentData()
                                if self._model.currentData() != "curvilinear" else "exact")
        self._sync_edge_choices()
        self._programs.load(cfg)
        self._space_charge.load(cfg)
        self._injection.load(cfg)
        # (the space-charge estimate depends on the injection, loaded after it)
        self._space_charge.update_estimate()
        self._on_programs_changed()
        self._record_source()
        self._update_beam_line()

    def compute_optics(self) -> None:
        if self._optics_worker is not None and self._optics_worker.isRunning():
            return
        lattice = self.state.lattice
        if lattice is None:
            self._optics_fail("Open a lattice first.")
            return
        try:
            provider = self.beam_config_provider
            beam_config = provider() if callable(provider) else self.state.beam_config
        except Exception as exc:          # noqa: BLE001 — an invalid Beam form
            self._optics_fail(f"Beam settings: {exc}")
            return
        from linac_gen.ring.config import RingConfig
        try:
            config = RingConfig(**self._optics_options())
        except ValueError as exc:
            self._optics_fail(str(exc))
            return
        self._detach_optics()
        self._show_optics(None)
        from ..workers import RingOpticsWorker
        worker = RingOpticsWorker(lattice, config, beam_config, self._six_d.isChecked())
        worker.finished_ok.connect(self._on_optics_done)
        worker.failed.connect(self._on_optics_failed)
        self._optics_worker = worker
        self._optics_beam = (None if _declared_beam(lattice) is not None
                             else self._beam_key(beam_config))
        self._optics_btn.setEnabled(False)
        if _numba_installed():
            self._optics_status.setText("Computing the ring optics (the first time in a "
                                        "session also compiles the ring kernels, a few "
                                        "seconds)…")
        else:
            self._optics_status.setText(
                "Computing the ring optics — numba is not installed, so the ring kernels run "
                "as plain Python: about 20 s for a ring of a thousand elements (the helix "
                "[ring] extra installs numba)…")
        self.state.status_message.emit("Ring optics…")
        worker.start()

    def _on_optics_done(self, result) -> None:
        if self.sender() is not self._optics_worker:
            return
        self._optics_btn.setEnabled(True)
        self._optics_status.setText(f"Done in {result['optics'].wall_s:.2f} s "
                                    f"({result['n_nodes']} kernel nodes"
                                    + ("" if result.get("compiled", True) else ", plain Python")
                                    + ").")
        self._show_optics(result)
        self.state.status_message.emit("Ring optics done.")

    def _on_optics_failed(self, msg: str) -> None:
        if self.sender() is not self._optics_worker:
            return
        self._optics_btn.setEnabled(True)
        self._optics_fail(f"failed — {msg}")

    def _optics_fail(self, msg: str) -> None:
        self._optics_status.setText(msg)
        self.state.status_message.emit(f"Ring optics: {msg}")

    def _show_optics(self, result) -> None:
        self._optics = result
        if hasattr(self, "_space_charge"):
            self._space_charge.update_estimate()
        if hasattr(self, "_injection"):
            self._injection.update_estimate()
        if result is None:
            if self._optics_worker is None:
                self._optics_beam = None
            self._optics_text.setText("No optics yet.")
            self._chips.setText("")
            for c in (self._curve_bx, self._curve_by, self._curve_dx):
                c.setData([], [])
            return
        o = result["optics"]
        st = o.start
        gt = f"{o.gamma_t:.6f}" if o.gamma_t else "none"
        opts = o.options
        lines = [
            f"{_label(_MODELS, opts.get('model'))} model, pole faces "
            f"{opts.get('edge_model')}, constants {result.get('constants', '?')}; "
            f"{result.get('kinetic_mev', float('nan')):.6g} MeV",
            f"Qx {o.qx:.9f}   Qy {o.qy:.9f}",
            f"Q'x {o.dqx:.4f}   Q'y {o.dqy:.4f}   (dQ/d delta)",
            f"momentum compaction {o.alpha_c:.9g}   slip factor {o.eta:.9g}   gamma_t {gt}",
            f"at s = 0:  beta_x {st['betx']:.6f} m   alpha_x {st['alfx']:.6f}   "
            f"beta_y {st['bety']:.6f} m   alpha_y {st['alfy']:.6f}   "
            f"D_x {st['dx']:.6f} m   D'_x {st['dpx']:.6g}",
            f"closed orbit at s = 0 ({o.mode}): x {o.closed_orbit[0] * 1e3:.4g} mm   "
            f"y {o.closed_orbit[2] * 1e3:.4g} mm",
        ]
        b = result.get("bucket")
        if o.qs is not None and b is not None:
            lines.append(f"Qs {o.qs:.6f} (small-amplitude formula {b['qs']:.6f});  bucket: "
                         f"harmonic {b['harmonic']:g}, {b['voltage_mv'] * 1e3:.6g} kV, "
                         f"half-height delta {b['delta_max']:.4g}")
        for w in result.get("warnings", []):
            lines.append(f"note: {w}")
        self._optics_text.setText("\n".join(lines))
        self._chips.setText(f"Q {o.qx:.4f} / {o.qy:.4f}   ξ {o.dqx:.2f} / {o.dqy:.2f}   "
                            f"γt {gt if o.gamma_t is None else f'{o.gamma_t:.4f}'}")
        t = o.table
        self._curve_bx.setData(t["s"], t["betx"])
        self._curve_by.setData(t["s"], t["bety"])
        self._curve_dx.setData(t["s"], t["dx"])

    def use_matched_beam(self) -> None:
        """The Beam tab gets the beam matched to the ring at the start of the
        run, computed in a worker (RingSimulation.matched_beam): the Twiss
        functions x, y and, with the RF on at the start, z, the dispersion per
        MeV and the closed orbit as the centroid — HELIX units, the phase in
        degrees at the Beam tab's frequency.  For a lattice that declares its
        beam (a MAD-X deck's BEAM) also that species and the run's starting
        energy; a lattice without one is scaled with the Beam tab's energy,
        which then stays."""
        if self._matched_worker is not None and self._matched_worker.isRunning():
            return
        lattice = self.state.lattice
        if lattice is None:
            self._optics_fail("Open a lattice first.")
            return
        provider, setter = self.beam_config_provider, self.beam_config_setter
        if not (callable(provider) and callable(setter)):
            return
        try:
            beam_config = provider()
        except Exception as exc:          # noqa: BLE001 — an invalid Beam form
            self._optics_fail(f"Beam settings: {exc}")
            return
        from linac_gen.ring.config import RingConfig
        try:
            ramp = self._programs.ramp_config()
        except ValueError as exc:
            self._optics_fail(f"Programs page: {exc}")
            return
        try:
            injection = self._injection.injection_config()
        except ValueError as exc:
            self._optics_fail(f"Injection page: {exc}")
            return
        if injection and (injection.get("source") or {}).get("kind", "beam") == "beam":
            # (the injected beam is the Beam tab's: the ring's matched beam
            # would replace it — its species, its Twiss at the foil)
            self._optics_fail("The Injection page injects the Beam tab's beam: the ring's "
                              "matched beam would replace it.  For a matched beam at the start "
                              "and an injection, inject a Gaussian at the foil or a particle "
                              "file (Injection page: Injected beam), or turn the injection "
                              "off.")
            return
        try:
            # (the run's settings that shape its first turn's optics)
            config = RingConfig(ramp=ramp, corrector_mode=(self._programs.corrector_mode()
                                                           if ramp else "follow"),
                                injection=injection, source="beam", **self._optics_options())
        except ValueError as exc:
            self._optics_fail(str(exc))
            return
        self._detach_matched()
        # (the Beam tab the result belongs to: its phase unit and, for a lattice
        # without its own beam, the scaling of its magnets)
        self._matched_for = (float(beam_config.frequency), str(beam_config.species),
                             float(beam_config.energy))
        from ..workers import RingMatchedBeamWorker
        worker = RingMatchedBeamWorker(lattice, config, beam_config)
        worker.finished_ok.connect(self._on_matched_done)
        worker.failed.connect(self._on_matched_failed)
        self._matched_worker = worker
        self._to_beam_btn.setEnabled(False)
        self._optics_status.setText("Computing the beam matched to the ring at the start of "
                                    "the run…")
        self.state.status_message.emit("Ring: the matched beam…")
        worker.start()

    def _on_matched_done(self, result) -> None:
        if self.sender() is not self._matched_worker:
            return
        self._release_matched()
        self._to_beam_btn.setEnabled(self.state.lattice is not None)
        provider, setter = self.beam_config_provider, self.beam_config_setter
        try:
            cfg = provider()
        except Exception as exc:          # noqa: BLE001 — an invalid Beam form
            self._optics_fail(f"Beam settings: {exc}")
            return
        # (only a beam the deck declares sets the species and energy; with
        # none the ring runs at the Beam tab's, which stay)
        own = _declared_beam(self.state.lattice) is not None
        now = (float(cfg.frequency), str(cfg.species), float(cfg.energy))
        was = getattr(self, "_matched_for", now)
        if now[0] != was[0] or (not own and now[1:] != was[1:]):
            # (the form changed without an Apply while the beam was computed)
            self._optics_fail("The Beam tab changed during the computation: press Use the "
                              "matched beam in the Beam tab again.")
            return
        vals = {k: float(result[k]) for k in _MATCHED_KEYS if result.get(k) is not None}
        if getattr(cfg, "source", "generate") == "file":
            # (a particle file: only its transverse planes can take Twiss
            # functions — re-shaped, keeping its emittances)
            vals = {k: v for k, v in vals.items()
                    if k in ("alpha_x", "beta_x", "alpha_y", "beta_y")}
            vals["reshape_to_twiss"] = True
        if own:
            # (12 digits: 800 MeV, not E - m's 800.0000000000001 — still the
            # ring's energy to the 1e-9 a run asks of the Beam tab's)
            vals.update(species=result["species"],
                        energy=float(f"{float(result['kinetic_mev']):.12g}"))
        try:
            setter(dataclasses.replace(cfg, **vals))
        except Exception as exc:          # noqa: BLE001 — a value the form refuses
            self._optics_fail(f"The Beam tab did not take the matched beam: {exc}")
            return
        self.state.mark_project_dirty()
        text = self._matched_text(result, cfg, own, normalized=not own and bool(
            getattr(self.state.lattice, "ring_reference_beam", None)))
        if getattr(cfg, "source", "generate") == "file":
            text += (" The Beam tab's beam is read from a particle file: its transverse "
                     "planes are re-shaped to these Twiss functions (keeping its emittances); "
                     "its dispersion, centroid and longitudinal distribution stay the file's.")
        elif bool(getattr(cfg, "continuous", False)):
            text += (" The Beam tab's beam is a continuous one: its longitudinal Twiss "
                     "functions are not used (the phase uniform over a period).")
        self._optics_status.setText(text)
        self.state.status_message.emit("Beam tab: the beam matched to the ring at the start "
                                       "of the run.")

    @staticmethod
    def _matched_text(r, cfg, own: bool, normalized: bool = False) -> str:
        """What "Use the matched beam in the Beam tab" set, in words (``own``:
        the deck declares its beam; ``normalized``: it declares none and its
        magnets keep the normalized strengths it gives)."""
        w0 = float(r["kinetic_mev"])
        where = "turn 0 of the ramp" if r.get("ramp") else "fixed energy"
        rf = (f"; RF {r['rf_frequency_hz'] * 1e-6:.6g} MHz" if r.get("rf_frequency_hz")
              else "")
        parts = [f"Beam tab: the beam matched to the ring at the start of the run "
                 f"({r['species']} {w0:.6g} MeV, {where}{rf}): "
                 f"beta_x {r['beta_x']:.6g} m, alpha_x {r['alpha_x']:.4g}, "
                 f"beta_y {r['beta_y']:.6g} m, alpha_y {r['alpha_y']:.4g}, "
                 f"D_x {r['disp_x']:.6g} mm/MeV"]
        if r.get("beta_z") is not None:
            parts.append(f", beta_z {r['beta_z']:.6g} deg/MeV, alpha_z {r['alpha_z']:.4g} "
                         f"(the phase at the Beam tab's {r['frequency_mhz']:.6g} MHz).")
        else:
            parts.append(". No RF at the start: the beam coasts, and the Beam tab's "
                         "longitudinal settings stay as they are.")
        if any(r.get(k) for k in ("centroid_x", "centroid_xp", "centroid_y", "centroid_yp",
                                  "centroid_dphi", "centroid_dw")):
            parts.append(f" Centroid on the closed orbit: x {r['centroid_x']:.4g} mm, "
                         f"x' {r['centroid_xp']:.4g} mrad, y {r['centroid_y']:.4g} mm, "
                         f"y' {r['centroid_yp']:.4g} mrad.")
        if not own and abs(float(cfg.energy) - w0) > 1e-9 * w0:
            why = ("stays (the deck declares no beam; its magnets keep the normalized "
                   "strengths it gives)" if normalized else
                   "sets the lattice's magnet strengths (it declares no beam) and stays")
            parts.append(f" The run starts at {w0:.6g} MeV; the Beam tab's {float(cfg.energy):g}"
                         f" MeV {why}: the beam is generated at {w0:.6g} MeV.")
        for n in r.get("notes", []):
            if not n.startswith("the ramp starts at"):
                parts.append(f" Note: {n}.")
        return "".join(parts)

    def _on_matched_failed(self, msg: str) -> None:
        if self.sender() is not self._matched_worker:
            return
        self._release_matched()
        self._to_beam_btn.setEnabled(self.state.lattice is not None)
        self._optics_fail(f"The matched beam: {msg}")

    def _release_matched(self) -> None:
        """The matched-beam worker has reported (its result or its failure):
        nothing is being computed any more.  It reports just before its thread
        ends, so the thread may still be running: then it is parked until it
        ends — a QThread dropped while it runs aborts the process."""
        old, self._matched_worker = self._matched_worker, None
        if old is not None and old.isRunning():
            from ..app import _park_worker  # (never drop a live QThread)
            _park_worker(old)

    def _stop_matched(self, reason: str) -> None:
        """A matched beam being computed for settings that just changed is
        dropped, and the page says so.  Once its worker has reported there is
        nothing to drop and nothing is said: the Beam tab has the matched beam,
        or the page says why not."""
        if self._matched_worker is None:
            return
        self._detach_matched()
        self._optics_status.setText(f"{reason}: the matched beam was not set — press Use "
                                    "the matched beam in the Beam tab again.")

    def _detach_matched(self) -> bool:
        """Cut the matched-beam worker loose (its result is never used); True
        when it was still computing."""
        old, self._matched_worker = self._matched_worker, None
        if old is None:
            return False
        for name in ("finished_ok", "failed"):
            with contextlib.suppress(TypeError, RuntimeError):
                getattr(old, name).disconnect()
        running = old.isRunning()
        if running:
            old.request_stop()
            from ..app import _park_worker  # (never drop a live QThread)
            _park_worker(old)
        self._to_beam_btn.setEnabled(self.state.lattice is not None)
        return running

    def _detach_optics(self) -> bool:
        """Cut the optics worker loose (its result is never shown); True when
        it was still computing."""
        old, self._optics_worker = self._optics_worker, None
        if old is None:
            return False
        for name in ("finished_ok", "failed"):
            with contextlib.suppress(TypeError, RuntimeError):
                getattr(old, name).disconnect()
        running = old.isRunning()
        if running:
            old.request_stop()
            from ..app import _park_worker  # (never drop a live QThread)
            _park_worker(old)
        self._optics_btn.setEnabled(True)
        return running

    # ------------------------------------------------------------------
    # run control
    # ------------------------------------------------------------------
    def run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        lattice = self.state.lattice
        if lattice is None:
            self._fail("Open a lattice first.")
            return
        try:
            provider = self.beam_config_provider
            beam_config = provider() if callable(provider) else self.state.beam_config
        except Exception as exc:          # noqa: BLE001 — an invalid Beam form
            self._fail(f"Beam settings: {exc}")
            return
        if beam_config is None:
            self._fail("Set the beam in the Beam tab first.")
            return
        from linac_gen.ring.config import RingConfig, auto_stats_every, parse_snapshot_turns
        n = int(self._turns.value())
        try:
            ramp = self._programs.ramp_config()
        except ValueError as exc:
            self._fail(f"Programs page: {exc}")
            return
        try:
            space_charge = self._space_charge.space_charge_config()
        except ValueError as exc:
            self._fail(f"Space charge page: {exc}")
            return
        try:
            injection = self._injection.injection_config()
        except ValueError as exc:
            self._fail(f"Injection page: {exc}")
            return
        try:
            snaps = parse_snapshot_turns(self._snapshots.text())
            config = RingConfig(n_turns=n, source=self._source.currentData(),
                                tune_probes=int(self._probes.value()), snapshot_turns=snaps,
                                stats_every=auto_stats_every(n),
                                device=self._device.currentData(), ramp=ramp,
                                # (the steerers' choice is the ramp's: at fixed energy
                                # they keep their kicks)
                                corrector_mode=(self._programs.corrector_mode() if ramp
                                                else "follow"),
                                space_charge=space_charge, injection=injection,
                                **self._optics_options())
        except ValueError as exc:
            self._fail(str(exc))
            return
        # (a run that cannot fit in memory is refused before it starts; the
        # Beam tab's distribution has the Beam tab's number of particles)
        from linac_gen.ring.memory import MEMORY_SHARE, run_need
        n_part = (0 if config.source == "empty"
                  else int(getattr(beam_config, "n_particles", 0) or 0))
        if injection:
            n_part += int(injection["turns"]) * int(injection["particles_per_turn"])
        need = run_need(config, int(getattr(beam_config, "n_particles", 0) or 0),
                        len(lattice.elements))
        have = _physical_memory()
        if have and need > MEMORY_SHARE * have:
            self._fail(f"This run would need about {need / 2**30:.1f} GB of the machine's "
                       f"{have / 2**30:.0f} GB: fewer particles or snapshot turns.")
            return
        cfg = dict(self.state.ring_config or {})
        cfg["n_turns"] = config.n_turns
        self.state.ring_config = cfg
        self._retire()
        self.set_load_note(None)
        self._live_strip.clear()
        self._live_gamma = None
        self._last_dump = None
        dump_path, self._dump_note = self._dump_path()
        self._summary.setText(f"Running {n} turn(s) of {n_part} particles… (the "
                              "curves below follow the run; the Results tab and the project "
                              "keep the last run's results until this one ends)")
        from ..workers import RingWorker
        worker = RingWorker(lattice, beam_config, config, dump_path=dump_path,
                            lattice_path=self.state.lattice_path)
        worker.progress.connect(self._on_progress)
        worker.progress_turn.connect(self._on_progress_turn)
        worker.finished_ok.connect(self._on_done)
        worker.aborted.connect(self._on_aborted)
        worker.failed.connect(self._on_failed)
        worker.live.connect(self._on_live)
        worker.dumped.connect(self._on_dumped)
        self._worker = worker
        self._run_lattice = lattice
        self._set_running(True)
        self.state.status_message.emit(f"Ring run: {config.n_turns} turn(s)…")
        worker.start()

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()

    def stop_all(self, reason: str = "Ring mode was switched off") -> None:
        """Ring mode switched off: a run stops at the end of its turn (its turns
        come back as after Stop); the optics, the matched beam and the ramp
        preview being computed are dropped — none of them may reach the Beam
        tab or the project once ring mode is off.  Optics and a preview shown
        stay."""
        self.stop()
        if self._detach_optics():
            self._optics_status.setText(f"{reason}: compute the optics again.")
        self._stop_matched(reason)
        self._programs.stop_preview(f"{reason}: press Preview again.")

    def is_running(self) -> bool:
        """A ring run is in progress."""
        return self._worker is not None and self._worker.isRunning()

    def shutdown_begin(self) -> list:
        workers = []
        for w in (self._worker, self._optics_worker, self._matched_worker):
            if w is not None and w.isRunning():
                w.request_stop()
                workers.append(w)
        return workers + self._programs.shutdown_begin()

    #: results files named in this session (a second run within the same
    #: second gets its own name although the first is not written yet)
    _DUMP_NAMES: set = set()

    @classmethod
    def _dump_path(cls):
        """(<calc dir>/<time>_ring.h5, None), or (None, why) when the calc
        directory cannot be made.  A name taken — on disk or by an earlier
        run of this session — gets _2, _3, ..."""
        import datetime as _dt
        try:
            from ..app import _resolve_calc_dir
            d = _resolve_calc_dir()
            d.mkdir(parents=True, exist_ok=True)
        except Exception as exc:                            # noqa: BLE001
            return None, f"the calc directory cannot be made ({exc})"
        stem = f"{_dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_ring"
        path, k = d / f"{stem}.h5", 1
        while str(path) in cls._DUMP_NAMES or path.exists():
            k += 1
            path = d / f"{stem}_{k}.h5"
        cls._DUMP_NAMES.add(str(path))
        return path, None

    def _on_run_setting_changed(self, *_args) -> None:
        self._snapshots.setModified(False)        # (its text is taken below)
        cfg = dict(self.state.ring_config or {})
        cfg.pop("n_particles", None)          # (the Beam tab's)
        cfg.update({"source": self._source.currentData(),
                    "tune_probes": int(self._probes.value()),
                    "device": self._device.currentData()})
        from linac_gen.ring.config import parse_snapshot_turns
        with contextlib.suppress(ValueError):
            # (as typed: "last" stays the last turn, whatever the number of turns)
            cfg["snapshot_turns"] = parse_snapshot_turns(self._snapshots.text())
        self.state.ring_config = cfg
        self.state.mark_project_dirty()

    def _on_live(self, d) -> None:
        if self.sender() is not self._worker:
            return
        self._live_strip.show_live(d)
        if d.get("ramp") and len(d.get("turns", ())):
            # (a ramp: the reference's gamma at the last statistics row, in the turn label)
            import numpy as np
            bg = float(np.asarray(d.get("beta0gamma0", np.nan), dtype=float).reshape(-1)[-1])
            self._live_gamma = (int(np.asarray(d["turns"])[-1]), float(np.sqrt(1.0 + bg * bg)))
            self._show_turn()

    def _on_dumped(self, path: str) -> None:
        if self.sender() is not self._worker:
            return
        self._last_dump = path
        self.state.status_message.emit(f"Ring results: {path}")

    def _on_turns_changed(self, n: int) -> None:
        # the project's ring settings follow the spin box (saved even
        # without a run)
        cfg = dict(self.state.ring_config or {})
        cfg["n_turns"] = int(n)
        self.state.ring_config = cfg
        self.state.mark_project_dirty()

    def apply_font_size(self, pt: int) -> None:
        """(the app font reaches every widget here — nothing inline to resize)"""

    def _detach(self) -> None:
        # stop the current run without waiting for it: its signals are cut
        # and the thread is parked until it ends (never dropped while alive)
        old, self._worker = self._worker, None
        self._run_lattice = None
        if old is None:
            return
        for name in ("progress", "progress_turn", "finished_ok", "aborted", "failed"):
            with contextlib.suppress(TypeError, RuntimeError):
                getattr(old, name).disconnect()
        if old.isRunning():
            old.request_stop()
            from ..app import _park_worker  # (never drop a live QThread)
            _park_worker(old)
        self._set_running(False)
        self._progress.setValue(0)
        self._turn_label.setText("")
        # (the run's live curves go with it: the strip shows the results the
        # state holds — none, after a new lattice)
        self._show_results(self.state.ring_results)

    def _retire(self) -> None:
        old, self._worker = self._worker, None
        self._run_lattice = None
        if old is None:
            return
        for name in ("progress", "progress_turn", "finished_ok", "aborted", "failed"):
            with contextlib.suppress(TypeError, RuntimeError):
                getattr(old, name).disconnect()
        if old.isRunning():
            old.request_stop()
            if not old.wait(5000):
                from ..app import _park_worker  # (never drop a live QThread)
                _park_worker(old)

    def _set_running(self, on: bool) -> None:
        self._run_btn.setEnabled(not on)
        self._turns.setEnabled(not on)
        self._stop_btn.setEnabled(on)
        # (a turn or gamma of this run never reaches the label after it ends)
        self._turn_now = None
        self._live_gamma = None
        if on:
            self._progress.setValue(0)
            self._turn_label.setText("")

    def _fail(self, msg: str) -> None:
        self._turn_label.setText(msg)
        self.state.status_message.emit(f"Ring run: {msg}")

    # worker slots (bound methods; a stale worker's late signal is ignored)
    def _on_progress(self, pct: int) -> None:
        if self.sender() is not self._worker:
            return
        self._progress.setValue(int(pct))

    def _on_progress_turn(self, turn: int, n: int) -> None:
        if self.sender() is not self._worker:
            return
        self._turn_now = (int(turn), int(n))
        self._show_turn()

    def _show_turn(self) -> None:
        """The turn label during a run: the turn and, in a ramp, the reference's
        gamma at the last statistics row."""
        now = getattr(self, "_turn_now", None)
        if now is None:
            return
        g = getattr(self, "_live_gamma", None)
        self._turn_label.setText(f"turn {now[0]} of {now[1]}" + (
            "" if g is None else f"; the reference's gamma {g[1]:.6f} (turn {g[0]})"))

    def _saved(self, results) -> None:
        m = results.meta
        if self._last_dump:
            m["dump_path"] = self._last_dump
        elif self._dump_note and not m.get("dump_error"):
            m["dump_error"] = self._dump_note
        if m.get("dump_error"):
            self.state.status_message.emit(f"Ring results not saved: {m['dump_error']}")

    def _on_done(self, results) -> None:
        if self.sender() is not self._worker:
            return
        self._set_running(False)
        self._progress.setValue(100)
        self._saved(results)
        self.state.set_ring_results(results)
        self.state.status_message.emit(f"Ring run done: {results.turns_done} turn(s).")

    def _on_aborted(self, results) -> None:
        if self.sender() is not self._worker:
            return
        self._set_running(False)
        if results is not None:
            self._saved(results)
            self.state.set_ring_results(results)
            self._turn_label.setText(f"stopped after turn {results.turns_done}")
        else:
            # (no turn done: the strip and the summary go back to the results
            # the state still holds)
            self._show_results(self.state.ring_results)
            self._turn_label.setText("stopped before the first turn was done")
        self.state.status_message.emit("Ring run stopped.")

    def _on_failed(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        self._set_running(False)
        self._show_results(self.state.ring_results)
        self._fail(f"failed — {msg}")
