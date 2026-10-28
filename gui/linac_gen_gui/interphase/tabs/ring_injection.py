"""The Ring tab's Injection page — multi-turn injection through a stripping
foil (see :mod:`linac_gen.ring.injection`): off or on; the turns that inject,
the macroparticles each brings and the current; the foil (where it is, its
material and thickness, the straggling, mean-loss and stripping models, its
rectangle); the injected beam (the Beam tab's distribution, a particle file
or a Gaussian at the foil; its species, energy, the frequency of its phases,
the linac's bunch rate and its centroid); the chopper; and the bump and
painting magnets with their waveforms.

Every valid change goes into the project's ring settings (``injection``;
removed when off); an entry that is not valid is reported and the project
keeps the last valid settings (a run refuses to start until the page is
valid).  With the optics of the Ring & Optics page, the page shows the real
particles each turn brings and the foil's scattering and loss.  Turning
injection on or off sends :attr:`RingInjectionPage.switched` (the Ring tab
starts the ring empty with injection).
"""
from __future__ import annotations

import math
from decimal import Decimal, DecimalException

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme

_NOTE = ("Multi-turn injection: each turn of the injection brings its macroparticles to the "
         "foil, placed on the linac bunches the chopper keeps; their first crossing strips "
         "them (an H- leaves as a proton of the same velocity; those left H0 or H-, and those "
         "missing the foil, are waste). The circulating beam crosses the foil wherever its "
         "rectangle is: Gaussian scattering of the Highland width and the energy loss of the "
         "foil element (large-angle and nuclear scattering are not modelled). Bump and "
         "painting magnets follow their waveforms turn by turn, so the closed orbit moves "
         "under the injected beam.")

_ON = (("Off", None), ("Injection through the foil", "on"))
_SOURCES = (("The Beam tab's distribution", "beam"), ("A particle file", "file"),
            ("A Gaussian at the foil", "gaussian"))
_SPECIES = (("From the source", None), ("H-", "H-"), ("proton", "proton"))
_DRAWS = (("Shuffled (seeded)", "shuffled"), ("In its own order", "in_order"))
_STRAGGLING = (("The foil element's, else auto", None), ("auto", "auto"),
               ("Landau", "landau"), ("Gaussian", "gaussian"))
_DEDX = (("The foil element's, else MIP", None), ("MIP (tabulated rate)", "mip"),
         ("Bethe", "bethe"))
_STRIP = (("The foil element's, else two-step for H-", None), ("Two-step H- to H0 to p",
                                                               "two_step"),
          ("Off (every ion stripped)", "off"))
_MAG_COLUMNS = ("Name", "Elements", "Model", "Plane", "Kick (mrad)", "Waveform (ms)",
                "Fringe: fint, half gap (mm)", "Faces: e1, e2 (mrad)")
_WAVE_TIP = ("The magnets' strength factor versus time from the start of the run (times in "
             "ms):\n  1   or  constant 1\n  linear t0 t1 start end\n  sqrt t0 t1 start end "
             "(start + (end - start) sqrt(u))\n  sqrt_end t0 t1 start end (end + (start - end) "
             "sqrt(1 - u))\n  table t1:v1 t2:v2 ...\n  file name.csv (t_s,value; a relative "
             "name from the lattice file's folder)")
_MODEL_TIP = ("kick: a kicker's deflection (Kick, in the Plane); dipole: a thick kicker or a "
              "bend with no angle as an exact dipole (its fringe and faces); scale: the "
              "elements' own kicks and multipoles times the waveform")


def _index(choices, key) -> int:
    keys = [k for _l, k in choices]
    return keys.index(key) if key in keys else 0


def _num(text: str, what: str, *, blank=None):
    t = text.strip()
    if not t:
        if blank is ...:
            raise ValueError(f"{what}: give a number")
        return blank
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{what}: {t!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{what}: {t!r} is not a finite number")
    return v


def _scaled(text: str, what: str, power: int, *, blank=None):
    """A number typed in mm (mrad, ms, ...) → SI by a decimal shift of the
    text ('54.7' → 0.0547 exactly as float('0.0547'))."""
    t = text.strip()
    if not t:
        if blank is ...:
            raise ValueError(f"{what}: give a number")
        return blank
    try:
        d = Decimal(t)
        v = float(d.scaleb(power)) if d.is_finite() else math.nan
    except (DecimalException, ArithmeticError, ValueError):
        raise ValueError(f"{what}: {t!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{what}: {t!r} is not a finite number")
    return v


def _text(v, power: int = 0) -> str:
    """A value in SI as text in the page's unit (10^power of it), reading back
    (by :func:`_scaled`) to the same float."""
    if v is None:
        return ""
    s = format(Decimal(repr(float(v))).scaleb(power).normalize(), "f")
    return s


def _fmt(v) -> str:
    if v is None:
        return ""
    r = repr(float(v))
    return r[:-2] if r.endswith(".0") else r


def parse_waveform(text: str, base_dir=None) -> dict:
    """A waveform typed on the page (times in ms; see the tooltip) → the
    settings' dict (times in s); a file's relative name is taken from
    ``base_dir`` (the lattice file's folder)."""
    t = text.strip()
    if not t:
        raise ValueError("give a waveform (1 for a constant strength)")
    parts = t.replace(",", " ").split()
    kind = parts[0].lower()
    if kind == "file":
        name = t.split(None, 1)[1].strip() if len(parts) > 1 else ""
        if not name:
            raise ValueError("file: give the CSV file's name (its columns t_s,value)")
        from pathlib import Path

        from linac_gen.ring.injection.config import read_waveform_csv
        path = Path(name).expanduser()
        if not path.is_absolute() and base_dir:
            path = Path(base_dir) / path
        # (what is wrong with the file, said as it is: InjectionError)
        times, values = read_waveform_csv(path)
        return {"kind": "table", "times_s": list(times), "values": list(values)}
    try:
        if len(parts) == 1 and kind not in ("constant", "table"):
            return {"kind": "constant", "value": float(parts[0])}
        if kind == "constant":
            if len(parts) > 2:
                raise ValueError("constant takes one number")
            return {"kind": "constant", "value": float(parts[1]) if len(parts) > 1 else 1.0}
        if kind in ("linear", "sqrt", "sqrt_end"):
            if len(parts) != 5:
                raise ValueError(f"{kind} needs four numbers: t0 t1 (ms) start end")
            return {"kind": kind, "t_start_s": _scaled(parts[1], "t0", -3),
                    "t_end_s": _scaled(parts[2], "t1", -3), "start": float(parts[3]),
                    "end": float(parts[4])}
        if kind == "table":
            times, values = [], []
            for p in parts[1:]:
                a, b = p.split(":")
                times.append(_scaled(a, "time", -3))
                values.append(float(b))
            return {"kind": "table", "times_s": times, "values": values}
    except (IndexError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(kind):
            raise
        raise ValueError(f"the waveform {t!r} is not one of the forms in the tooltip") from None
    raise ValueError(f"unknown waveform kind {kind!r} (constant, linear, sqrt, sqrt_end, "
                     "table, file)")


def waveform_text(w: dict) -> str:
    """The page's text of a waveform's settings (times in ms)."""
    kind = w.get("kind", "constant")
    if kind == "constant":
        return _fmt(w.get("value", 1.0))
    if kind == "table":
        return "table " + " ".join(f"{_text(t, 3)}:{_fmt(v)}"
                                   for t, v in zip(w["times_s"], w["values"], strict=True))
    return (f"{kind} {_text(w.get('t_start_s', 0.0), 3)} {_text(w.get('t_end_s'), 3)} "
            f"{_fmt(w.get('start', 1.0))} {_fmt(w.get('end', 0.0))}")


def _fringe(text: str, what: str):
    """The fringe column: fint, and the half gap typed in mm (→ m by a
    decimal shift, '11.3' → 0.0113 exactly)."""
    t = text.strip()
    if not t:
        return 0.0, 0.0
    parts = t.replace(";", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"{what}: two numbers separated by a comma")
    return _scaled(parts[0], what, 0), _scaled(parts[1], what, -3)


def _pair(text: str, what: str, power: int):
    t = text.strip()
    if not t:
        return 0.0, 0.0
    parts = t.replace(";", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"{what}: two numbers separated by a comma")
    return _scaled(parts[0], what, power), _scaled(parts[1], what, power)


class RingInjectionPage(QWidget):
    """``optics_provider()`` — the Ring & Optics page's last optics (its
    worker's dict) or None; ``lattice_provider()`` — the window's lattice."""

    changed = pyqtSignal()
    #: injection turned on (True) or off (False) on the page
    switched = pyqtSignal(bool)

    def __init__(self, state, *, optics_provider=None, lattice_provider=None):
        super().__init__()
        self.state = state
        self._optics_provider = optics_provider or (lambda: None)
        self._lattice_provider = lattice_provider or (lambda: state.lattice)
        self._loading = False
        self._was_on = False
        self._map_bins = [40, 40]          # (not on the page: the project's, kept)
        self._build_ui()
        self.load(self.state.ring_config or {})

    # -- the page --------------------------------------------------------------
    def _build_ui(self) -> None:
        from linac_gen.ring.injection.config import PARTICLES_PER_TURN_MAX, TURNS_MAX
        v = QVBoxLayout(self)
        note = QLabel(_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(note)

        box = QGroupBox("Injection")
        f = QFormLayout(box)
        self._on = QComboBox()
        for label, key in _ON:
            self._on.addItem(label, key)
        f.addRow("Injection", self._on)
        self._turns = QSpinBox()
        self._turns.setRange(1, TURNS_MAX)
        self._turns.setToolTip("Turns that inject (from the start turn on)")
        f.addRow("Turns that inject", self._turns)
        self._start = QSpinBox()
        self._start.setRange(0, TURNS_MAX)
        f.addRow("First injection turn", self._start)
        self._per_turn = QSpinBox()
        self._per_turn.setRange(1, PARTICLES_PER_TURN_MAX)
        self._per_turn.setSingleStep(10)
        f.addRow("Macroparticles a turn", self._per_turn)
        self._current = QLineEdit()
        self._current.setToolTip("The average current injected during the pulse, after the "
                                 "chopper [mA]: each turn brings I T / e particles, shared by "
                                 "the filled buckets")
        f.addRow("Current (mA)", self._current)
        self._notch = QSpinBox()
        self._notch.setRange(0, 1_000_000)
        self._notch.setToolTip("Buckets left empty (a notch): the particles of a turn are "
                               "shared by the others")
        f.addRow("Empty buckets (notch)", self._notch)
        v.addWidget(box)

        fb = QGroupBox("Foil")
        ff = QFormLayout(fb)
        self._at = QComboBox()
        self._at.setEditable(True)
        self._at.setToolTip("The element the foil is at (it acts at its exit); blank: the "
                            "lattice's foil element")
        ff.addRow("At element", self._at)
        from linac_gen.elements.foil import supported_materials
        self._material = QComboBox()
        self._material.addItem("The foil element's, else carbon", None)
        for m in supported_materials():
            self._material.addItem(m, m)
        ff.addRow("Material", self._material)
        self._thick = QLineEdit()
        self._thick.setPlaceholderText("blank: the foil element's")
        ff.addRow("Thickness (ug/cm2)", self._thick)
        self._strag = QComboBox()
        for label, key in _STRAGGLING:
            self._strag.addItem(label, key)
        ff.addRow("Energy straggling", self._strag)
        self._dedx = QComboBox()
        for label, key in _DEDX:
            self._dedx.addItem(label, key)
        ff.addRow("Mean energy loss", self._dedx)
        self._strip = QComboBox()
        for label, key in _STRIP:
            self._strip.addItem(label, key)
        ff.addRow("Stripping", self._strip)
        row = QHBoxLayout()
        self._xmin, self._xmax, self._ymin, self._ymax = (QLineEdit() for _ in range(4))
        for w, ph in ((self._xmin, "x min"), (self._xmax, "x max"), (self._ymin, "y min"),
                      (self._ymax, "y max")):
            w.setPlaceholderText(ph)
            w.setToolTip("The foil's edges in absolute ring coordinates [mm]; blank: open on "
                         "that side (or the foil element's extent about its offset)")
            row.addWidget(w)
        ff.addRow("Rectangle (mm)", row)
        v.addWidget(fb)

        sb = QGroupBox("Injected beam")
        sf = QFormLayout(sb)
        self._kind = QComboBox()
        for label, key in _SOURCES:
            self._kind.addItem(label, key)
        sf.addRow("Particles from", self._kind)
        prow = QHBoxLayout()
        self._path = QLineEdit()
        self._path.setToolTip("A TraceWin .dst, HELIX's ASCII distribution (# w_kin_ref "
                              "header) or TraceWin's particle ASCII file")
        self._browse = QPushButton("Browse…")
        self._browse.clicked.connect(self._on_browse)
        prow.addWidget(self._path, stretch=1)
        prow.addWidget(self._browse)
        sf.addRow("File", prow)
        self._species = QComboBox()
        for label, key in _SPECIES:
            self._species.addItem(label, key)
        sf.addRow("Species", self._species)
        self._kinetic = QLineEdit()
        self._kinetic.setPlaceholderText("blank: the source's")
        sf.addRow("Kinetic energy (MeV)", self._kinetic)
        self._freq = QLineEdit()
        self._freq.setPlaceholderText("blank: the source's")
        self._freq.setToolTip("The frequency the source's phases are counted in (and the "
                              "linac's bunch rate, unless given below)")
        sf.addRow("Bunch frequency (MHz)", self._freq)
        self._bunch = QLineEdit()
        self._bunch.setPlaceholderText("blank: the frequency above")
        self._bunch.setToolTip("The linac's bunch rate — the bunches the chopper sees — when it "
                               "differs from the frequency of the source's phases")
        sf.addRow("Linac bunch rate (MHz)", self._bunch)
        grow = QHBoxLayout()
        self._enx, self._eny = QLineEdit(), QLineEdit()
        for w, ph in ((self._enx, "x"), (self._eny, "y")):
            w.setPlaceholderText(ph)
            grow.addWidget(w)
        sf.addRow("Gaussian: emittances (pi mm mrad, normalized rms)", grow)
        trow = QHBoxLayout()
        self._bx, self._ax, self._by, self._ay = (QLineEdit() for _ in range(4))
        for w, ph in ((self._bx, "beta x (m)"), (self._ax, "alpha x"), (self._by, "beta y (m)"),
                      (self._ay, "alpha y")):
            w.setPlaceholderText(ph)
            trow.addWidget(w)
        sf.addRow("Gaussian: Twiss at the foil", trow)
        lrow = QHBoxLayout()
        self._sdp, self._sz, self._cut = QLineEdit(), QLineEdit(), QLineEdit()
        for w, ph in ((self._sdp, "dp/p rms"), (self._sz, "bunch length rms (mm)"),
                      (self._cut, "cut (sigma, at least 1)")):
            w.setPlaceholderText(ph)
            lrow.addWidget(w)
        sf.addRow("Gaussian: momentum spread, length", lrow)
        crow = QHBoxLayout()
        self._cx, self._cpx, self._cy, self._cpy = (QLineEdit() for _ in range(4))
        for w, ph in ((self._cx, "x (mm)"), (self._cpx, "px (mrad)"), (self._cy, "y (mm)"),
                      (self._cpy, "py (mrad)")):
            w.setPlaceholderText(ph)
            w.setToolTip("The injected beam's centroid at the foil, in the ring's coordinates "
                         "(px, py = P/P0)")
            crow.addWidget(w)
        sf.addRow("Centroid at the foil", crow)
        self._draw = QComboBox()
        for label, key in _DRAWS:
            self._draw.addItem(label, key)
        sf.addRow("Particles drawn", self._draw)
        v.addWidget(sb)

        cb = QGroupBox("Chopper")
        cf = QFormLayout(cb)
        self._window = QLineEdit()
        self._window.setToolTip("The linac bunches kept: within this many degrees of the ring's "
                                "RF of the bucket's centre (180: every bunch)")
        cf.addRow("Window (deg of the ring's RF)", self._window)
        self._phase = QLineEdit()
        self._phase.setToolTip("The ring RF phase of the first linac bunch (> 0: late)")
        cf.addRow("Phase of the first bunch (deg)", self._phase)
        v.addWidget(cb)

        mb = QGroupBox("Bump and painting magnets")
        mv = QVBoxLayout(mb)
        self._mags = QTableWidget(0, len(_MAG_COLUMNS))
        self._mags.setHorizontalHeaderLabels(list(_MAG_COLUMNS))
        self._mags.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._mags.horizontalHeader().setStretchLastSection(True)
        self._mags.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._mags.setMinimumHeight(140)
        for c, tip in ((2, _MODEL_TIP), (3, "x or y (kick and dipole models)"),
                       (4, "One deflection per element (mrad, commas), or one for all; blank: "
                           "the elements' own"), (5, _WAVE_TIP)):
            self._mags.horizontalHeaderItem(c).setToolTip(tip)
        mv.addWidget(self._mags)
        brow = QHBoxLayout()
        self._add = QPushButton("Add a group")
        self._add.clicked.connect(self._on_add_group)
        self._remove = QPushButton("Remove")
        self._remove.clicked.connect(self._on_remove_group)
        brow.addWidget(self._add)
        brow.addWidget(self._remove)
        brow.addStretch(1)
        mv.addLayout(brow)
        v.addWidget(mb)

        self._estimate = QLabel("")
        self._estimate.setWordWrap(True)
        self._estimate.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._estimate)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._status)
        v.addStretch(1)
        for w in (self._on, self._material, self._strag, self._dedx, self._strip, self._kind,
                  self._species, self._draw):
            w.currentIndexChanged.connect(self._on_changed)
        self._at.currentTextChanged.connect(self._on_changed)
        for w in (self._turns, self._start, self._per_turn, self._notch):
            w.valueChanged.connect(self._on_changed)
        for w in self._line_edits():
            w.editingFinished.connect(self._on_changed)
        self._mags.itemChanged.connect(self._on_changed)

    def _line_edits(self):
        return (self._current, self._thick, self._xmin, self._xmax, self._ymin, self._ymax,
                self._path, self._kinetic, self._freq, self._bunch, self._enx, self._eny,
                self._bx, self._ax,
                self._by, self._ay, self._sdp, self._sz, self._cut, self._cx, self._cpx,
                self._cy, self._cpy, self._window, self._phase)

    def _sync_visibility(self) -> None:
        on = self._on.currentData() is not None
        kind = self._kind.currentData()
        for w in self.findChildren(QWidget):
            if w is self._on or isinstance(w, QGroupBox) or w is self._estimate \
                    or w is self._status:
                continue
            if isinstance(w, (QLineEdit, QComboBox, QSpinBox, QPushButton, QTableWidget)):
                w.setEnabled(on)
        self._on.setEnabled(True)
        gauss = on and kind == "gaussian"
        for w in (self._enx, self._eny, self._bx, self._ax, self._by, self._ay, self._sdp,
                  self._sz, self._cut):
            w.setEnabled(gauss)
        self._path.setEnabled(on and kind == "file")
        self._browse.setEnabled(on and kind == "file")

    def set_lattice_elements(self) -> None:
        """The foil's element choices: the lattice's markers and foil elements
        whose names are unique (the foil's element must be)."""
        from collections import Counter
        lat = self._lattice_provider()
        cur = self._at.currentText()
        self._at.blockSignals(True)
        self._at.clear()
        self._at.addItem("")
        if lat is not None:
            count = Counter(str(e.name).lower() for e in lat.elements)
            for e in lat.elements:
                n = str(e.name)
                if type(e).__name__ in ("Marker", "Foil") and count[n.lower()] == 1:
                    self._at.addItem(n)
        self._at.setEditText(cur)
        self._at.blockSignals(False)

    # -- the settings ----------------------------------------------------------
    def injection_config(self) -> dict | None:
        """The page's settings (InjectionConfig.to_dict), None when off;
        ValueError when an entry is not valid."""
        if self._on.currentData() is None:
            return None
        from linac_gen.ring.injection.config import InjectionConfig
        kind = self._kind.currentData()
        foil = {"at": self._at.currentText().strip() or None,
                "material": self._material.currentData(),
                "thickness_ug_cm2": _num(self._thick.text(), "foil thickness (ug/cm2)"),
                "straggling": self._strag.currentData(), "dedx_model": self._dedx.currentData(),
                "strip_model": self._strip.currentData(),
                "x_min_m": _scaled(self._xmin.text(), "foil x min (mm)", -3),
                "x_max_m": _scaled(self._xmax.text(), "foil x max (mm)", -3),
                "y_min_m": _scaled(self._ymin.text(), "foil y min (mm)", -3),
                "y_max_m": _scaled(self._ymax.text(), "foil y max (mm)", -3),
                "map_bins": list(self._map_bins)}
        src = {"kind": kind, "species": self._species.currentData(),
               "kinetic_mev": _num(self._kinetic.text(), "kinetic energy (MeV)"),
               "frequency_mhz": _num(self._freq.text(), "bunch frequency (MHz)"),
               "bunch_frequency_mhz": _num(self._bunch.text(), "linac bunch rate (MHz)"),
               "draw": self._draw.currentData(),
               "x_m": _scaled(self._cx.text(), "centroid x (mm)", -3, blank=0.0),
               "px": _scaled(self._cpx.text(), "centroid px (mrad)", -3, blank=0.0),
               "y_m": _scaled(self._cy.text(), "centroid y (mm)", -3, blank=0.0),
               "py": _scaled(self._cpy.text(), "centroid py (mrad)", -3, blank=0.0)}
        if kind == "file":
            src.update(path=self._path.text().strip() or None)
        if kind == "gaussian":
            src.update(emit_nx_um=_num(self._enx.text(), "emittance x", blank=0.0),
                       emit_ny_um=_num(self._eny.text(), "emittance y", blank=0.0),
                       beta_x_m=_num(self._bx.text(), "beta x (m)", blank=1.0),
                       alpha_x=_num(self._ax.text(), "alpha x", blank=0.0),
                       beta_y_m=_num(self._by.text(), "beta y (m)", blank=1.0),
                       alpha_y=_num(self._ay.text(), "alpha y", blank=0.0),
                       sigma_dp=_num(self._sdp.text(), "dp/p rms", blank=0.0),
                       sigma_z_m=_scaled(self._sz.text(), "bunch length (mm)", -3, blank=0.0),
                       cutoff=_num(self._cut.text(), "cut (sigma)"))
        d = {"foil": foil, "source": src, "turns": int(self._turns.value()),
             "start_turn": int(self._start.value()),
             "particles_per_turn": int(self._per_turn.value()),
             "current_ma": _num(self._current.text(), "current (mA)"),
             "notch_buckets": int(self._notch.value()),
             "chopper_window_deg": _num(self._window.text(), "chopper window (deg)", blank=180.0),
             "chopper_phase_deg": _num(self._phase.text(), "chopper phase (deg)", blank=0.0),
             "magnets": self._magnet_groups()}
        return InjectionConfig.from_dict(d).to_dict()

    def _cell(self, r: int, c: int) -> str:
        it = self._mags.item(r, c)
        return it.text().strip() if it is not None else ""

    def _magnet_groups(self) -> list:
        out = []
        for r in range(self._mags.rowCount()):
            name = self._cell(r, 0) or f"group {r + 1}"
            where = f"magnet group {name!r}"
            elements = [s.strip() for s in self._cell(r, 1).split(",") if s.strip()]
            model = (self._cell(r, 2) or "kick").lower()
            plane = self._cell(r, 3).lower() or None
            kick = self._cell(r, 4)
            kicks = None
            if kick:
                kicks = [_scaled(k, f"{where}: kick (mrad)", -3) for k in kick.split(",")]
                if len(kicks) == 1:
                    kicks = kicks[0]
            try:
                wave = parse_waveform(self._cell(r, 5) or "1", base_dir=self._base_dir())
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from None
            g = {"name": name, "elements": elements, "model": model, "plane": plane,
                 "kick_rad": kicks, "waveform": wave}
            fringe, faces = self._cell(r, 6), self._cell(r, 7)
            if model == "dipole":
                g["fint"], g["hgap_m"] = _fringe(fringe, f"{where}: fint, half gap")
                g["e1_rad"], g["e2_rad"] = _pair(faces, f"{where}: e1, e2 (mrad)", -3)
            elif fringe or faces:
                raise ValueError(f"{where}: the fringe and faces are the 'dipole' model's (clear "
                                 "them, or choose dipole)")
            out.append(g)
        return out

    def _base_dir(self):
        """The folder a waveform file's relative name is taken from: the
        lattice file's (None: the working directory)."""
        import os
        path = getattr(self.state, "lattice_path", None)
        return os.path.dirname(os.path.abspath(path)) if path else None

    def summary_text(self) -> str:
        """One line for the Run page."""
        try:
            d = self.injection_config()
        except ValueError as exc:
            return f"Injection: not valid — {exc}"
        if d is None:
            return "Injection: off."
        s = d["source"]
        src = {"beam": "the Beam tab's distribution", "file": "a particle file",
               "gaussian": "a Gaussian"}[s["kind"]]
        mags = f", {len(d['magnets'])} magnet group(s)" if d["magnets"] else ""
        return (f"Injection: {d['turns']} turn(s) from turn {d['start_turn']}, "
                f"{d['particles_per_turn']} macroparticles a turn from {src}, "
                f"{d['current_ma']:g} mA{mags} (Injection page).")

    def load(self, cfg: dict) -> None:
        """Show the settings a project's ring section holds (the defaults for
        what it does not)."""
        from linac_gen.ring.injection.config import InjectionConfig
        self._loading = True
        try:
            self.set_lattice_elements()
            raw = (cfg or {}).get("injection")
            try:
                d = InjectionConfig.from_dict(raw).to_dict() if raw is not None else None
            except ValueError:
                d = None
            f = (d or {}).get("foil") or {}
            s = (d or {}).get("source") or {}
            self._on.setCurrentIndex(_index(_ON, "on" if d else None))
            self._was_on = d is not None
            self._map_bins = list(f.get("map_bins") or [40, 40])
            self._turns.setValue(int((d or {}).get("turns", 100)))
            self._start.setValue(int((d or {}).get("start_turn", 0)))
            self._per_turn.setValue(int((d or {}).get("particles_per_turn", 50)))
            self._current.setText(_fmt((d or {}).get("current_ma", 2.0)))
            self._notch.setValue(int((d or {}).get("notch_buckets", 0)))
            self._at.setEditText(f.get("at") or "")
            self._material.setCurrentIndex(_index(
                [(None, None)] + [(m, m) for m in _materials()], f.get("material")))
            self._thick.setText(_fmt(f.get("thickness_ug_cm2")))
            self._strag.setCurrentIndex(_index(_STRAGGLING, f.get("straggling")))
            self._dedx.setCurrentIndex(_index(_DEDX, f.get("dedx_model")))
            self._strip.setCurrentIndex(_index(_STRIP, f.get("strip_model")))
            for w, k in ((self._xmin, "x_min_m"), (self._xmax, "x_max_m"),
                         (self._ymin, "y_min_m"), (self._ymax, "y_max_m")):
                w.setText(_text(f.get(k), 3))
            self._kind.setCurrentIndex(_index(_SOURCES, s.get("kind", "beam")))
            self._path.setText(s.get("path") or "")
            self._species.setCurrentIndex(_index(_SPECIES, s.get("species")))
            self._kinetic.setText(_fmt(s.get("kinetic_mev")))
            self._freq.setText(_fmt(s.get("frequency_mhz")))
            self._bunch.setText(_fmt(s.get("bunch_frequency_mhz")))
            self._enx.setText(_fmt(s.get("emit_nx_um", 0.0)))
            self._eny.setText(_fmt(s.get("emit_ny_um", 0.0)))
            self._bx.setText(_fmt(s.get("beta_x_m", 1.0)))
            self._ax.setText(_fmt(s.get("alpha_x", 0.0)))
            self._by.setText(_fmt(s.get("beta_y_m", 1.0)))
            self._ay.setText(_fmt(s.get("alpha_y", 0.0)))
            self._sdp.setText(_fmt(s.get("sigma_dp", 0.0)))
            self._sz.setText(_text(s.get("sigma_z_m", 0.0), 3))
            self._cut.setText(_fmt(s.get("cutoff")))
            self._cx.setText(_text(s.get("x_m", 0.0), 3))
            self._cpx.setText(_text(s.get("px", 0.0), 3))
            self._cy.setText(_text(s.get("y_m", 0.0), 3))
            self._cpy.setText(_text(s.get("py", 0.0), 3))
            self._draw.setCurrentIndex(_index(_DRAWS, s.get("draw", "shuffled")))
            self._window.setText(_fmt((d or {}).get("chopper_window_deg", 180.0)))
            self._phase.setText(_fmt((d or {}).get("chopper_phase_deg", 0.0)))
            self._mags.setRowCount(0)
            for g in (d or {}).get("magnets", []):
                self._append_group(g)
            self._status.setText("" if raw is None or d is not None else
                                 "The project's injection settings were not valid: the page "
                                 "shows the defaults.")
        finally:
            self._loading = False
        self._sync_visibility()
        self.update_estimate()

    def _append_group(self, g: dict) -> None:
        r = self._mags.rowCount()
        self._mags.insertRow(r)
        k = g.get("kick_rad")
        kick = "" if k is None else ", ".join(_text(v, 3) for v in k)
        fringe = (f"{_fmt(g.get('fint', 0.0))}, {_text(g.get('hgap_m', 0.0), 3)}"
                  if g.get("model") == "dipole" else "")
        faces = (f"{_text(g.get('e1_rad', 0.0), 3)}, {_text(g.get('e2_rad', 0.0), 3)}"
                 if g.get("model") == "dipole" else "")
        for c, text in enumerate((g.get("name", ""), ", ".join(g.get("elements", [])),
                                  g.get("model", "kick"), g.get("plane") or "", kick,
                                  waveform_text(g.get("waveform") or {}), fringe, faces)):
            self._mags.setItem(r, c, QTableWidgetItem(text))

    def _on_add_group(self) -> None:
        self._loading = True
        try:
            self._append_group({"name": f"group {self._mags.rowCount() + 1}", "elements": [],
                                "model": "kick", "waveform": {"kind": "constant",
                                                              "value": 1.0}})
        finally:
            self._loading = False
        self._on_changed()

    def _on_remove_group(self) -> None:
        rows = sorted({i.row() for i in self._mags.selectedIndexes()}, reverse=True)
        if not rows and self._mags.rowCount():
            rows = [self._mags.rowCount() - 1]
        for r in rows:
            self._mags.removeRow(r)
        self._on_changed()

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "The injected beam's particles", "",
                                              "Particle files (*.dst *.txt *.dat);;All (*)")
        if path:
            self._path.setText(path)
            self._on_changed()

    def _on_changed(self, *_a) -> None:
        self._sync_visibility()
        if self._loading:
            return
        for w in self._line_edits():
            w.setModified(False)
        cfg = dict(self.state.ring_config or {})
        try:
            d = self.injection_config()
        except ValueError as exc:
            self._status.setText(f"Not valid: {exc} (the project keeps the last valid "
                                 "settings).")
        else:
            self._status.setText("")
            if d is None:
                cfg.pop("injection", None)
            else:
                cfg["injection"] = d
        if cfg != (self.state.ring_config or {}):
            self.state.ring_config = cfg
            self.state.mark_project_dirty()
        self.update_estimate()
        self.changed.emit()
        on = self._on.currentData() is not None
        if on != self._was_on:
            self._was_on = on
            self.switched.emit(on)

    def commit_pending(self) -> None:
        """Take an entry still being typed: a cell of the magnets' table whose
        editor is open, and a line edit (only when one was edited)."""
        if self._loading:
            return
        from .ring_programs import commit_cell_editor
        commit_cell_editor(self._mags)            # (its itemChanged takes it)
        if not any(w.isModified() for w in self._line_edits()):
            return
        self._on_changed()

    def is_on(self) -> bool:
        """Injection is on (the page's choice, valid or not)."""
        return self._on.currentData() is not None

    def showEvent(self, event) -> None:          # noqa: N802 (Qt's name)
        super().showEvent(event)
        self.set_lattice_elements()
        self.update_estimate()

    def update_estimate(self, *_a) -> None:
        """The real particles each turn brings and the foil's scattering and
        loss at the ring's energy (with the Ring & Optics page's optics)."""
        try:
            d = self.injection_config()
        except ValueError:
            self._estimate.setText("")
            return
        if d is None:
            self._estimate.setText("")
            return
        res = self._optics_provider()
        if not res or res.get("optics") is None:
            self._estimate.setText("The particles each turn brings: compute the optics on the "
                                   "Ring & Optics page.")
            return
        try:
            self._estimate.setText(_estimate_text(res, d, self._lattice_provider()))
        except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            self._estimate.setText(f"Estimate: {exc}")


def _materials():
    from linac_gen.elements.foil import supported_materials
    return supported_materials()


def _foil_element(lattice, at):
    """The lattice's foil element the injection's foil is (``at`` names it, or
    blank and the lattice has one), or None."""
    if lattice is None:
        return None
    foils = [e for e in lattice.elements if type(e).__name__ == "Foil"]
    if at:
        named = [e for e in foils if str(e.name).lower() == at.lower()]
        return named[0] if len(named) == 1 else None
    return foils[0] if len(foils) == 1 else None


def _estimate_text(res, d, lattice=None) -> str:
    """Real particles a turn and in all, per filled bucket, and the foil's
    Highland width and mean loss at the ring's energy (the foil's settings,
    else its foil element's)."""
    opt = res["optics"]
    t_rev = float(opt.length) / (float(opt.beta0) * 299792458.0)
    rf = bool(res.get("rf_on"))
    # (the RF buckets: the circumference over the RF period of the cavities
    # with a voltage, as a run fills them)
    n_rf = res.get("rf_buckets") or res.get("harmonic")
    if rf and not n_rf:
        raise ValueError("the RF is not at a harmonic of the revolution frequency")
    buckets = (int(n_rf) - int(d["notch_buckets"])) if rf else 1
    if buckets < 1:
        raise ValueError(f"the notch leaves no bucket filled ({n_rf} RF buckets)")
    per_turn = d["current_ma"] * 1e-3 * t_rev / 1.602176634e-19
    total = per_turn * d["turns"]
    where = (f"over {buckets} filled bucket(s): {total / buckets:.4g} particles in each"
             if rf else "in the coasting beam")
    out = (f"Each turn brings {per_turn:.4g} particles ({d['current_ma']:g} mA x "
           f"{t_rev * 1e6:.6g} us); {d['turns']} turns {total:.4g}, {where}; "
           f"{d['particles_per_turn'] * d['turns']} macroparticles.")
    f = d["foil"]
    elem = _foil_element(lattice, f.get("at"))

    def setting(key, attr, default):
        if f.get(key) is not None:
            return f[key]
        return getattr(elem, attr, default) if elem is not None else default

    thick = setting("thickness_ug_cm2", "thickness_ug_cm2", None)
    if thick:
        from linac_gen.ring.injection.foil import FoilSettings, RingFoil
        fs = FoilSettings(name="foil", node=0, material=setting("material", "material", "C"),
                          thickness_ug_cm2=float(thick),
                          straggling=setting("straggling", "straggling", "auto"),
                          dedx_model=setting("dedx_model", "dedx_model", "mip"),
                          strip_model="off", x_min_m=None, x_max_m=None, y_min_m=None,
                          y_max_m=None, map_bins=(1, 1))
        foil = RingFoil(fs, mass_mev=938.27208816)
        w = float(res.get("kinetic_mev", 0.0))
        ph = foil.physics(w)
        out += (f" The foil at {w:.6g} MeV: scattering theta0 {ph['theta0'] * 1e6:.4g} urad "
                f"for one crossing (Highland), mean loss {ph['mean_loss_mev'] * 1e3:.4g} keV a "
                "crossing.")
    else:
        out += " The foil's scattering and loss: give its thickness."
    return out
