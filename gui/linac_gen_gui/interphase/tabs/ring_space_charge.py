"""The Ring tab's Space charge page — the 2.5-D model (see
:mod:`linac_gen.ring.sc`): off, particle-in-cell (the field solved from the
beam at every kick) or a frozen Gaussian; the bunch's intensity, the kicks
per turn, the grid, the pipe, the longitudinal kick and the line density.

Every valid change goes into the project's ring settings (``space_charge``;
removed when off); an entry that is not valid is reported and the project
keeps the last valid settings (a run refuses to start until the page is
valid).  With the optics of the Ring & Optics page, the page shows the
analytic tune shift at the centre of a Gaussian bunch for these settings and
the Beam tab's emittances.  With injection the space charge is that of the
circulating charge the injection builds: a run reports its tune shift at the
end.
"""
from __future__ import annotations

import contextlib
import math
from decimal import Decimal, DecimalException

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import theme

_NOTE = ("The 2.5-D model: at every kick the bunch's transverse distribution gives a "
         "two-dimensional field, felt by each particle in proportion to the line density where "
         "it is; the slope of the line density gives a longitudinal kick. The kicks sit at "
         "element boundaries, each standing for the ring halfway to its neighbours. The tune "
         "probes feel the field without adding to it.")

_MODELS = (("Off", None), ("2.5-D particle-in-cell (the field from the beam, every kick)", "pic"),
           ("Frozen Gaussian (the beam's centroid and rms sizes)", "frozen"))
_DEPOSITS = (("TSC (3 x 3 nodes)", "tsc"), ("CIC (2 x 2 nodes)", "cic"))
_PIPES = (("Open (no wall)", "open"), ("Round", "round"), ("Rectangle", "rectangle"),
          ("Ellipse", "ellipse"))
_LONG = (("From the potential (zero on the wall, or at the reference radius)", "potential"),
         ("A geometry factor g, the same for every particle", "g_factor"), ("Off", "off"))


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


def _fmt(v) -> str:
    """A number as text that reads back to the same float (large ones with an
    exponent: 5e10, 1.2345678901234e10)."""
    if v is None:
        return ""
    v = float(v)
    if abs(v) >= 1e6:
        import numpy as np
        m, e = np.format_float_scientific(v, unique=True, trim="-").split("e")
        return f"{m}e{int(e)}"
    r = repr(v)
    return r[:-2] if r.endswith(".0") else r


def _mm_to_m(text: str, what: str):
    """A length typed in mm → metres, by a decimal shift of the text: '54.7' →
    0.0547 exactly as float('0.0547') (None when blank)."""
    t = text.strip()
    if not t:
        return None
    try:
        d = Decimal(t)
        v = float(d.scaleb(-3)) if d.is_finite() else math.nan
    except (DecimalException, ArithmeticError, ValueError):
        raise ValueError(f"{what}: {t!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{what}: {t!r} is not a finite number")
    return v


def _m_to_mm_text(v) -> str:
    """A length in metres as mm text that reads back (by :func:`_mm_to_m`) to
    the same float."""
    if v is None:
        return ""
    s = format(Decimal(repr(float(v))).scaleb(3).normalize(), "f")
    return s


class RingSpaceChargePage(QWidget):
    """``optics_provider()`` — the Ring & Optics page's last optics (its
    worker's dict) or None; ``beam_provider()`` — the Beam tab's BeamConfig;
    ``injection_provider()`` — whether the Injection page's injection is on."""

    changed = pyqtSignal()

    def __init__(self, state, *, optics_provider=None, beam_provider=None,
                 injection_provider=None):
        super().__init__()
        self.state = state
        self._optics_provider = optics_provider or (lambda: None)
        self._beam_provider = beam_provider or (lambda: state.beam_config)
        self._injection_provider = injection_provider or (lambda: False)
        self._loading = False
        self._saved_r_ref = 1.0          # (kept while a pipe makes it not apply)
        # (a project's value a spin box shows rounded: (shown, exact), the exact
        # one kept until the box is changed)
        self._exact = {}
        self._build_ui()
        self.load(self.state.ring_config or {})

    # -- the page --------------------------------------------------------------
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        note = QLabel(_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(note)

        box = QGroupBox("Model")
        f = QFormLayout(box)
        self._model = QComboBox()
        for label, key in _MODELS:
            self._model.addItem(label, key)
        f.addRow("Space charge", self._model)
        self._intensity = QLineEdit()
        self._intensity.setPlaceholderText("blank: from the Beam tab's current")
        self._intensity.setToolTip("Real particles in the bunch (the whole ring for a coasting "
                                   "beam), e.g. 5e10; blank: the Beam tab's current over its "
                                   "frequency (one bunch), or times the revolution time for a "
                                   "coasting beam.  With injection leave it blank: the charge "
                                   "is the circulating charge the injection builds")
        f.addRow("Particles in the bunch", self._intensity)
        from linac_gen.ring.sc.config import KICKS_MAX, REFRESH_MAX, SPAN_MAX
        self._kicks = QSpinBox()
        self._kicks.setRange(1, KICKS_MAX)
        self._kicks.setToolTip("About this many kicks per turn, at the element boundaries "
                               "nearest equal spacing; at least ten per betatron wavelength")
        f.addRow("Kicks per turn", self._kicks)
        self._refresh = QSpinBox()
        self._refresh.setRange(0, REFRESH_MAX)
        self._refresh.setToolTip("Frozen model: turns between measurements of the beam's "
                                 "centroid, rms sizes and line density (0: only at the start)")
        f.addRow("Refresh every (turns)", self._refresh)
        v.addWidget(box)

        g = QGroupBox("Grid and line density")
        gf = QFormLayout(g)
        row = QHBoxLayout()
        self._nx = QSpinBox()
        self._ny = QSpinBox()
        from linac_gen.ring.sc.config import GRID_MAX
        for w in (self._nx, self._ny):
            w.setRange(8, GRID_MAX)
            w.setSingleStep(8)
        row.addWidget(self._nx)
        row.addWidget(QLabel("x"))
        row.addWidget(self._ny)
        row.addStretch(1)
        gf.addRow("Grid (nx x ny)", row)
        self._deposit = QComboBox()
        for label, key in _DEPOSITS:
            self._deposit.addItem(label, key)
        gf.addRow("Charge shape", self._deposit)
        self._box_sigma = QDoubleSpinBox()
        self._box_sigma.setRange(1.0, SPAN_MAX)
        self._box_sigma.setDecimals(6)
        self._box_sigma.setToolTip("The grid covers the beam out to its extent or this many "
                                   "rms sizes, the smaller; particles beyond feel the grid "
                                   "charge's field (its multipole series far out, the sum over "
                                   "its cells nearer)")
        gf.addRow("Grid covers (rms sizes)", self._box_sigma)
        self._z_bins = QSpinBox()
        self._z_bins.setRange(8, 65536)
        self._z_bins.setValue(64)
        self._z_auto = QCheckBox("auto")
        self._z_auto.setChecked(True)
        self._z_auto.setToolTip("16 bins per rms of the bunch's core, re-sized as the bunch "
                                "changes; the line density smoothed over a width that shrinks "
                                "as the number of particles grows")
        zrow = QHBoxLayout()
        zrow.addWidget(self._z_bins, 1)
        zrow.addWidget(self._z_auto)
        gf.addRow("Line-density bins (RF period)", zrow)
        self._z_smooth = QDoubleSpinBox()
        self._z_smooth.setRange(0.0, SPAN_MAX)
        self._z_smooth.setDecimals(6)
        self._z_smooth.setToolTip("Width of the Gaussian filter on the line density, in bins "
                                  "(0: none)")
        gf.addRow("Line-density filter (bins)", self._z_smooth)
        v.addWidget(g)

        p = QGroupBox("Pipe and longitudinal kick")
        pf = QFormLayout(p)
        self._pipe = QComboBox()
        for label, key in _PIPES:
            self._pipe.addItem(label, key)
        pf.addRow("Pipe (grounded wall)", self._pipe)
        self._pipe_a = QLineEdit()
        self._pipe_a.setToolTip("Radius (round), or half width (rectangle, ellipse) [mm]")
        pf.addRow("Radius / half width (mm)", self._pipe_a)
        self._pipe_b = QLineEdit()
        self._pipe_b.setToolTip("Half height (rectangle, ellipse) [mm]")
        pf.addRow("Half height (mm)", self._pipe_b)
        self._order = QSpinBox()
        self._order.setRange(2, 64)
        self._order.setToolTip("Harmonics fitted to cancel the potential on a rectangle or an "
                               "ellipse")
        pf.addRow("Harmonics fitted", self._order)
        self._r_ref = QLineEdit()
        self._r_ref.setToolTip("With no wall, the potential is zero at this radius [m]: the "
                               "longitudinal kick scales as g = 1 + 2 ln(r_ref / a)")
        pf.addRow("Reference radius (m)", self._r_ref)
        self._long = QComboBox()
        for label, key in _LONG:
            self._long.addItem(label, key)
        pf.addRow("Longitudinal kick", self._long)
        self._g = QLineEdit()
        self._g.setToolTip("g = 1 + 2 ln(b/a) on the axis of a uniform beam of radius a in a "
                           "pipe of radius b")
        pf.addRow("Geometry factor g", self._g)
        v.addWidget(p)

        self._estimate = QLabel("")
        self._estimate.setWordWrap(True)
        self._estimate.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._estimate)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._status)
        v.addStretch(1)
        for w in (self._model, self._deposit, self._pipe, self._long):
            w.currentIndexChanged.connect(self._on_changed)
        self._z_auto.toggled.connect(self._on_changed)
        for w in (self._kicks, self._refresh, self._nx, self._ny, self._z_bins, self._order):
            w.valueChanged.connect(self._on_changed)
        for w in (self._box_sigma, self._z_smooth):
            w.valueChanged.connect(self._on_changed)
        for w in (self._intensity, self._pipe_a, self._pipe_b, self._r_ref, self._g):
            w.editingFinished.connect(self._on_changed)

    def _sync_visibility(self) -> None:
        on = self._model.currentData() is not None
        frozen = self._model.currentData() == "frozen"
        pipe = self._pipe.currentData()
        for w in (self._intensity, self._kicks, self._nx, self._ny, self._deposit,
                  self._box_sigma, self._z_bins, self._z_auto, self._z_smooth, self._pipe,
                  self._long):
            w.setEnabled(on)
        self._z_bins.setEnabled(on and not self._z_auto.isChecked())
        self._refresh.setEnabled(on and frozen)
        self._pipe_a.setEnabled(on and pipe != "open")
        self._pipe_b.setEnabled(on and pipe in ("rectangle", "ellipse"))
        self._order.setEnabled(on and pipe in ("rectangle", "ellipse"))
        self._r_ref.setEnabled(on and pipe == "open")
        self._g.setEnabled(on and self._long.currentData() == "g_factor")

    # -- the settings ----------------------------------------------------------
    def space_charge_config(self) -> dict | None:
        """The page's settings (SpaceChargeConfig.to_dict), None when off;
        ValueError when an entry is not valid."""
        model = self._model.currentData()
        if model is None:
            return None
        from linac_gen.ring.sc.config import SpaceChargeConfig
        pipe = self._pipe.currentData()
        # (only the fields that apply to the choices made are read: a greyed
        # field never blocks the page)
        a = (_mm_to_m(self._pipe_a.text(), "pipe radius / half width (mm)")
             if pipe != "open" else None)
        b = (_mm_to_m(self._pipe_b.text(), "pipe half height (mm)")
             if pipe in ("rectangle", "ellipse") else None)
        if pipe == "open":
            r_ref = _num(self._r_ref.text(), "reference radius (m)", blank=1.0)
            if r_ref > 0.0:
                self._saved_r_ref = r_ref        # (kept for when a pipe is chosen)
        else:
            r_ref = self._saved_r_ref
        g = (_num(self._g.text(), "geometry factor g", blank=None)
             if self._long.currentData() == "g_factor" else None)
        d = {"model": model,
             "intensity": _num(self._intensity.text(), "particles in the bunch", blank=None),
             "kicks_per_turn": int(self._kicks.value()),
             "grid": [int(self._nx.value()), int(self._ny.value())],
             "deposit": self._deposit.currentData(),
             "box_sigma": self._spin_value("box_sigma", self._box_sigma),
             "pipe": pipe, "pipe_a_m": a, "pipe_b_m": b,
             "reference_radius_m": r_ref,
             "boundary_order": int(self._order.value()),
             "longitudinal": self._long.currentData(), "g_factor": g,
             "z_bins": "auto" if self._z_auto.isChecked() else int(self._z_bins.value()),
             "z_smoothing_bins": self._spin_value("z_smoothing_bins", self._z_smooth),
             "refresh_turns": int(self._refresh.value())}
        return SpaceChargeConfig.from_dict(d).to_dict()

    def _spin_value(self, key, box) -> float:
        """The spin box's value, or the project's exact value while the box
        still shows it as loaded (rounded to its decimals)."""
        v = float(box.value())
        shown, exact = self._exact.get(key, (None, None))
        return exact if shown is not None and v == shown else v

    def summary_text(self) -> str:
        """One line for the Run page."""
        try:
            d = self.space_charge_config()
        except ValueError as exc:
            return f"Space charge: not valid — {exc}"
        if d is None:
            return "Space charge: off."
        model = ("2.5-D particle-in-cell" if d["model"] == "pic" else "frozen Gaussian")
        res = self._optics_provider() or {}
        where = "the ring" if res.get("rf_on") is False else "the bunch"
        if self._injection_provider():
            # (a run with injection takes the injected current's charge, and
            # refuses an intensity)
            n = ("the circulating charge the injection builds (its tune shift is reported at "
                 "the end of the run)" if d["intensity"] is None else
                 f"{d['intensity']:.4g} particles in {where}, which a run with injection "
                 "refuses (leave it blank: the charge is the circulating charge the injection "
                 "builds)")
        else:
            n = ("the Beam tab's intensity" if d["intensity"] is None
                 else f"{d['intensity']:.4g} particles in {where}")
        return (f"Space charge: {model}, {n}, about {d['kicks_per_turn']} kicks per turn, grid "
                f"{d['grid'][0]} x {d['grid'][1]}, {d['pipe']} pipe (Space charge page).")

    def load(self, cfg: dict) -> None:
        """Show the settings a project's ring section holds (the defaults for
        what it does not)."""
        from linac_gen.ring.sc.config import SpaceChargeConfig
        self._loading = True
        try:
            raw = (cfg or {}).get("space_charge")
            try:
                d = SpaceChargeConfig.from_dict(raw).to_dict() if raw is not None else None
            except ValueError:
                d = None
            base = d or SpaceChargeConfig().to_dict()
            self._model.setCurrentIndex(_index(_MODELS, d["model"] if d else None))
            self._intensity.setText(_fmt(base["intensity"]))
            self._kicks.setValue(int(base["kicks_per_turn"]))
            self._refresh.setValue(int(base["refresh_turns"]))
            self._nx.setValue(int(base["grid"][0]))
            self._ny.setValue(int(base["grid"][1]))
            self._deposit.setCurrentIndex(_index(_DEPOSITS, base["deposit"]))
            self._box_sigma.setValue(float(base["box_sigma"]))
            self._exact["box_sigma"] = (float(self._box_sigma.value()),
                                        float(base["box_sigma"]))
            auto = base["z_bins"] == "auto"
            self._z_auto.setChecked(auto)
            if not auto:
                self._z_bins.setValue(int(base["z_bins"]))
            self._z_smooth.setValue(float(base["z_smoothing_bins"]))
            self._exact["z_smoothing_bins"] = (float(self._z_smooth.value()),
                                               float(base["z_smoothing_bins"]))
            self._pipe.setCurrentIndex(_index(_PIPES, base["pipe"]))
            self._pipe_a.setText(_m_to_mm_text(base["pipe_a_m"]))
            self._pipe_b.setText(_m_to_mm_text(base["pipe_b_m"]))
            self._order.setValue(int(base["boundary_order"]))
            self._r_ref.setText(_fmt(base["reference_radius_m"]))
            self._saved_r_ref = float(base["reference_radius_m"])
            self._long.setCurrentIndex(_index(_LONG, base["longitudinal"]))
            self._g.setText(_fmt(base["g_factor"]))
            self._status.setText("" if raw is None or d is not None else
                                 "The project's space-charge settings were not valid: the "
                                 "page shows the defaults.")
        finally:
            self._loading = False
        self._sync_visibility()
        self.update_estimate()

    def _on_changed(self, *_a) -> None:
        self._sync_visibility()
        if self._loading:
            return
        for w in (self._intensity, self._pipe_a, self._pipe_b, self._r_ref, self._g):
            w.setModified(False)
        cfg = dict(self.state.ring_config or {})
        try:
            d = self.space_charge_config()
        except ValueError as exc:
            self._status.setText(f"Not valid: {exc} (the project keeps the last valid "
                                 "settings).")
        else:
            self._status.setText("")
            if d is None:
                cfg.pop("space_charge", None)
            else:
                cfg["space_charge"] = d
        if cfg != (self.state.ring_config or {}):
            self.state.ring_config = cfg
            self.state.mark_project_dirty()
        self.update_estimate()
        self.changed.emit()

    # -- the analytic tune shift ---------------------------------------------------
    def showEvent(self, event) -> None:          # noqa: N802 (Qt's name)
        # (the Beam tab may have changed without being applied)
        super().showEvent(event)
        self.update_estimate()

    def commit_pending(self) -> None:
        """Take an entry still being typed in a line edit (only when one was
        edited: settings the page shows rounded or does not apply are not
        rewritten by a save)."""
        edits = (self._intensity, self._pipe_a, self._pipe_b, self._r_ref, self._g)
        if self._loading or not any(w.isModified() for w in edits):
            return
        self._on_changed()

    def update_estimate(self, *_a) -> None:
        """The analytic tune shift for these settings, the Ring & Optics page's
        optics and the Beam tab's emittances (or why it is not shown)."""
        try:
            d = self.space_charge_config()
        except ValueError:
            self._estimate.setText("")
            return
        if d is None:
            self._estimate.setText("")
            return
        res = self._optics_provider()
        if self._injection_provider():
            # (not the Beam tab's current and emittances: the beam the
            # injection builds, known at the end of the run)
            text = ("With injection, the space charge is that of the circulating charge the "
                    "injection builds: a run reports its analytic tune shift at the end.")
            if res and res.get("optics") is not None:
                with contextlib.suppress(AttributeError, KeyError, TypeError, ValueError,
                                         ZeroDivisionError):
                    text += _kicks_text(res, d["kicks_per_turn"])
            self._estimate.setText(text)
            return
        if not res or res.get("optics") is None:
            self._estimate.setText("The analytic tune shift: compute the optics on the Ring & "
                                   "Optics page.")
            return
        try:
            beam = self._beam_provider()
        except Exception as exc:          # noqa: BLE001 — an invalid Beam form
            self._estimate.setText(f"The analytic tune shift: Beam settings — {exc}")
            return
        try:
            self._estimate.setText(_estimate_text(res, beam, d["intensity"])
                                   + _kicks_text(res, d["kicks_per_turn"]))
        except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            self._estimate.setText(f"The analytic tune shift: {exc}")


def _kicks_text(res, kicks: int) -> str:
    """The kicks per betatron wavelength of the kicks the request places (at
    element boundaries: fewer than asked when the elements are fewer)."""
    opt = res["optics"]
    q = max(abs(float(opt.qx)), abs(float(opt.qy)))
    if not q > 0.0:
        return ""
    starts = res.get("element_starts_s")
    placed = kicks
    if starts is not None:
        from linac_gen.ring.sc.placement import placed_count
        placed = placed_count(starts, float(opt.length), kicks)
    per = placed / q
    asked = "" if placed == kicks else f" ({placed} of the {kicks} asked: element boundaries)"
    warn = "" if per >= 10.0 else " — fewer than ten: give more kicks per turn"
    return f" About {per:.3g} kicks per betatron wavelength (Q {q:.4g}){asked}{warn}."


def _estimate_text(res, beam, intensity) -> str:
    """The analytic tune shift for an ``intensity`` (None: the Beam tab's
    current over its frequency — one bunch — or, coasting, times the
    revolution time — the ring, as the run takes it)."""
    import numpy as np

    from linac_gen.ring.sc.laslett import tune_shift
    opt = res["optics"]
    W = np.asarray(opt.W)
    if W.shape[0] != 6 and res.get("rf_on"):
        raise ValueError("the ring has RF: compute its optics with 'with RF: 6-D' (Ring & "
                         "Optics page) for the bunch length")
    bg = float(opt.beta0) * float(opt.gamma0)
    ex = float(beam.emit_nx) * 1e-6 / bg
    ey = float(beam.emit_ny) * 1e-6 / bg
    mass = float(res["p0c_mev"]) / bg
    n_b = intensity
    if n_b is None:
        try:
            n_b = float(beam.current) * 1e-3 / (float(beam.frequency) * 1e6) / 1.602176634e-19
            if W.shape[0] != 6:
                n_b *= float(beam.frequency) * 1e6 * float(opt.length) / (
                    float(opt.beta0) * 299792458.0)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            n_b = 0.0
    if not n_b > 0.0:
        raise ValueError("no intensity — give the particles in the bunch (or a beam current "
                         "in the Beam tab)")
    if W.shape[0] == 6:
        # the matched beam's bunch length: the longitudinal mode's beta with the
        # Beam tab's longitudinal emittance (the Beam tab's beam once Ring &
        # Optics → Use the matched beam in the Beam tab gave it the matched Twiss)
        f_hz = float(beam.frequency) * 1e6
        b0 = float(opt.beta0)
        e0 = mass * float(opt.gamma0)
        ez = float(beam.emit_z) * (b0 * 299792458.0 / (360.0 * f_hz)) / (b0 * b0 * e0)
        sz = math.sqrt(ez * (W[4, 4] ** 2 + W[4, 5] ** 2))
        sd = math.sqrt(ez * (W[5, 4] ** 2 + W[5, 5] ** 2))
        f = 1.0 / (math.sqrt(2.0 * math.pi) * sz)
        where = f"the centre of a Gaussian bunch (rms length {sz:.4g} m)"
    else:
        f = 1.0 / float(opt.length)
        sd = 0.0
        where = "a coasting beam's centre"
    dqx, dqy = tune_shift(opt, emit_x=ex, emit_y=ey, intensity=n_b, line_density=f,
                          sigma_delta=sd, mass_mev=mass)
    big = (" A shift beyond half an integer crosses resonances: check the intensity, the "
           "emittances and the bunch length." if max(abs(dqx), abs(dqy)) > 0.5 else "")
    return (f"Analytic tune shift at {where}: dQx {dqx:.4g}, dQy {dqy:.4g} — for {n_b:.4g} "
            f"particles and the Beam tab's emittances ({float(beam.emit_nx):.4g} / "
            f"{float(beam.emit_ny):.4g} pi mm mrad), from the Ring & Optics page's optics.{big}")
