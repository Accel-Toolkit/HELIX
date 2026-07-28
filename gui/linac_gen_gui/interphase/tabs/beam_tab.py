"""Beam configuration tab.

Three-column dense form matching the BeamConfig dataclass.  Apply pushes
the config into AppState.  A bottom 4-quadrant initial-phase-space
preview updates whenever the form changes (throttled to Apply).
"""
from __future__ import annotations

import math
import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QCheckBox,
    QFileDialog, QMessageBox,
)

from linac_gen_gui.interphase.app_settings import make_settings
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.state import AppState


def _group_qss() -> str:
    return (
        f"QGroupBox {{ color:{theme.TEXT_2}; border:1px solid {theme.BORDER_0};"
        f" border-radius:4px; margin-top:12px; padding-top:6px; }} "
        f"QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 6px;"
        f" color:{theme.TEXT_2}; font-size:10px; letter-spacing:1px;"
        f" text-transform:uppercase; background:{theme.BG_0}; }}"
    )


class BeamTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        # When the user imports a TraceWin .dst file we keep the path and
        # flip ``source`` to "file" so create_beam reads the actual particles
        # back instead of regenerating from Twiss.  Both fields surface on
        # the BeamConfig built by ``_build_cfg``.
        self._source = "generate"
        self._distribution_file: str | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        # --- Three column forms ---------------------------------------
        cols = QHBoxLayout(); cols.setSpacing(10)
        cols.addWidget(self._col_particle(), 1)
        cols.addWidget(self._col_twiss(), 1)
        cols.addWidget(self._col_centroid(), 1)
        v.addLayout(cols)

        # --- Sample beam preview --------------------------------------
        preview = QGroupBox("Initial phase-space preview")
        preview.setStyleSheet(_group_qss())
        p_lay = QHBoxLayout(preview)

        # Four density panels (log-intensity 2-D histograms with colorbars):
        # the three phase-space projections plus the real-space x-y view
        # which surfaces transverse coupling / mismatch at a glance.
        from linac_gen_gui.interphase.plots.plot_style import DensityPanel
        self._panel_xx = DensityPanel(title="x – x'", xlabel="x",   xunits="mm",
                                       ylabel="x'",  yunits="mrad")
        self._panel_yy = DensityPanel(title="y – y'", xlabel="y",   xunits="mm",
                                       ylabel="y'",  yunits="mrad")
        self._panel_zp = DensityPanel(title="φ – dW", xlabel="φ",   xunits="deg",
                                       ylabel="dW",  yunits="MeV")
        self._panel_xy = DensityPanel(title="x – y",  xlabel="x",   xunits="mm",
                                       ylabel="y",   yunits="mm")
        p_lay.addWidget(self._panel_xx, 1)
        p_lay.addWidget(self._panel_yy, 1)
        p_lay.addWidget(self._panel_zp, 1)
        p_lay.addWidget(self._panel_xy, 1)
        v.addWidget(preview, stretch=1)

        # --- Apply / regenerate row -----------------------------------
        row = QHBoxLayout(); row.setSpacing(8)
        self._apply_btn = QPushButton("  Apply")
        self._apply_btn.setIcon(icon("check", 12, "#00161c"))
        self._apply_btn.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; border-radius:3px;"
            f"padding:6px 16px; font-weight:600;"
        )
        self._apply_btn.clicked.connect(self._apply)
        row.addWidget(self._apply_btn)

        self._gen_btn = QPushButton("  Regenerate preview")
        self._gen_btn.setIcon(icon("refresh", 12))
        self._gen_btn.clicked.connect(self._regen_preview)
        row.addWidget(self._gen_btn)

        self._reset_btn = QPushButton("  Reset defaults")
        self._reset_btn.setIcon(icon("rewind", 12))
        self._reset_btn.clicked.connect(self._reset)
        row.addWidget(self._reset_btn)

        self._import_dst_btn = QPushButton("  Import .dst…")
        self._import_dst_btn.setIcon(icon("file", 12))
        self._import_dst_btn.setToolTip(
            "Import a TraceWin .dst particle distribution.  The file's "
            "calculated emittance, Twiss, energy, frequency, current and "
            "particle count populate every field; the simulation reads the "
            "actual particles from the file (source='file')."
        )
        self._import_dst_btn.clicked.connect(self._import_dst)
        row.addWidget(self._import_dst_btn)

        # Active-file chip + clear link.  Hidden until a .dst is loaded.
        self._file_chip = QLabel("")
        self._file_chip.setStyleSheet(
            f"color:{theme.ACCENT}; font-family:{theme.FONT_MONO}; font-size:10px;"
            f"background:{theme.BG_INSET}; padding:3px 6px; border-radius:3px;"
            f"border:1px solid {theme.BORDER_0};"
        )
        self._file_chip.hide()
        row.addWidget(self._file_chip)
        self._clear_file_btn = QPushButton("clear")
        self._clear_file_btn.setStyleSheet(
            f"background:transparent; color:{theme.TEXT_2}; border:0;"
            f"text-decoration:underline; padding:0 4px; font-size:10px;"
        )
        self._clear_file_btn.clicked.connect(self._clear_file_source)
        self._clear_file_btn.hide()
        row.addWidget(self._clear_file_btn)
        row.addStretch(1)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;")
        row.addWidget(self._status)
        v.addLayout(row)

        self._reset()
        # react to energy/species for derived display
        self._energy.valueChanged.connect(lambda *_: self._recompute_derived())
        self._species.currentTextChanged.connect(lambda *_: self._recompute_derived())
        self._recompute_derived()
        self._regen_preview()

    # ------------------------------------------------------------------
    # Form columns
    def _col_particle(self) -> QGroupBox:
        g = QGroupBox("Particle · Energy · RF · Current")
        g.setStyleSheet(_group_qss())
        f = QFormLayout(g)
        self._species = QComboBox(); self._species.addItems(["proton", "deuteron", "H-"])
        self._energy  = self._spin(0.001, 10_000, 0.001, 7, " MeV")
        self._freq    = self._spin(1, 5000, 0.001, 3, " MHz")
        self._current = self._spin(0, 1000, 0.01, 3, " mA")
        self._duty    = self._spin(0.001, 100.0, 0.1, 2, " %")
        self._npart   = QSpinBox(); self._npart.setRange(100, 2_000_000); self._npart.setValue(100_000)
        self._dist    = QComboBox(); self._dist.addItems(
            ["gaussian", "waterbag", "kv", "parabolic", "uniform", "thermal"]
        )
        self._cutoff  = self._spin(1.0, 10.0, 0.5, 2, " σ")
        # Bi-Gaussian (thermal) halo controls — only meaningful when
        # distribution == "thermal"; greyed out otherwise.
        self._halo_frac  = self._spin(0.0, 1.0, 0.01, 3, "")
        self._halo_frac.setValue(0.05)
        self._halo_frac.setToolTip(
            "Halo fraction (bi-Gaussian only): probability that a particle "
            "is drawn from the wide halo component."
        )
        self._halo_ratio = self._spin(1.0, 20.0, 0.1, 2, "×σ")
        self._halo_ratio.setValue(5.0)
        self._halo_ratio.setToolTip(
            "Halo σ ratio (bi-Gaussian only): how much wider the halo is "
            "than the core.  Default 5.0 gives a ~5×RMS halo population."
        )
        self._halo_frac.setEnabled(False)
        self._halo_ratio.setEnabled(False)
        self._dist.currentTextChanged.connect(self._on_distribution_changed)
        # DC / continuous-beam controls (pre-RFQ / LEBT).
        self._continuous = QCheckBox("Continuous beam (pre-RFQ / LEBT)")
        self._continuous.setToolTip(
            "When checked, the beam is generated with uniform phase over one RF "
            "period and propagated in 4-D (no longitudinal dynamics) until the "
            "first RF bunching element.  ε_z / α_z / β_z are ignored; energy "
            "spread comes from the field below."
        )
        self._dc_dw = self._spin(0.0, 1000.0, 0.01, 3, " keV")
        self._dc_dw.setToolTip("1σ energy spread for the continuous beam.")
        self._dc_dw.setEnabled(False)   # unchecked by default → grey out
        self._continuous.toggled.connect(self._on_continuous_toggled)
        for lab, w in [("Species", self._species), ("W_kin", self._energy),
                       ("f_rf", self._freq), ("I (peak)", self._current),
                       ("Duty", self._duty), ("N particles", self._npart),
                       ("Distribution", self._dist), ("Cutoff", self._cutoff),
                       ("Halo fraction", self._halo_frac),
                       ("Halo σ ratio", self._halo_ratio),
                       ("", self._continuous),
                       ("DC ΔW (1σ)", self._dc_dw)]:
            f.addRow(lab, w)
        return g

    def _on_distribution_changed(self, text: str) -> None:
        is_thermal = (text == "thermal")
        self._halo_frac.setEnabled(is_thermal)
        self._halo_ratio.setEnabled(is_thermal)

    def _on_continuous_toggled(self, on: bool) -> None:
        # In continuous mode, longitudinal Twiss is meaningless — grey it out.
        for w in (self._emit_z, self._alpha_z, self._beta_z):
            w.setEnabled(not on)
        self._dc_dw.setEnabled(on)

    def _col_twiss(self) -> QGroupBox:
        g = QGroupBox("Twiss — X / Y / Z")
        g.setStyleSheet(_group_qss())
        f = QFormLayout(g)
        self._emit_nx = self._spin(0, 1000, 0.001, 6, " π·mm·mrad")
        self._alpha_x = self._spin(-100, 100, 0.001, 6, "")
        self._beta_x  = self._spin(0.001, 10_000, 0.001, 5, " mm/mrad")
        self._emit_ny = self._spin(0, 1000, 0.001, 6, " π·mm·mrad")
        self._alpha_y = self._spin(-100, 100, 0.001, 6, "")
        self._beta_y  = self._spin(0.001, 10_000, 0.001, 5, " mm/mrad")
        self._emit_z  = self._spin(0, 1000, 0.0001, 8, " deg·MeV")
        self._alpha_z = self._spin(-100, 100, 0.001, 6, "")
        self._beta_z  = self._spin(0.001, 100_000, 0.01, 5, " deg/MeV")
        for lab, w in [("ε_nx", self._emit_nx), ("α_x", self._alpha_x), ("β_x", self._beta_x),
                       ("ε_ny", self._emit_ny), ("α_y", self._alpha_y), ("β_y", self._beta_y),
                       ("ε_z", self._emit_z),   ("α_z", self._alpha_z), ("β_z", self._beta_z)]:
            f.addRow(lab, w)
        return g

    def _col_centroid(self) -> QGroupBox:
        g = QGroupBox("Centroid · Mismatch · Derived")
        g.setStyleSheet(_group_qss())
        f = QFormLayout(g)
        self._d_beta  = self._ro("—")
        self._d_gamma = self._ro("—")
        self._d_bg    = self._ro("—")
        f.addRow("β",  self._d_beta); f.addRow("γ", self._d_gamma); f.addRow("βγ", self._d_bg)
        self._cx  = self._spin(-1e3, 1e3, 0.001, 6, " mm")
        self._cxp = self._spin(-1e3, 1e3, 0.001, 6, " mrad")
        self._cy  = self._spin(-1e3, 1e3, 0.001, 6, " mm")
        self._cyp = self._spin(-1e3, 1e3, 0.001, 6, " mrad")
        self._cphi = self._spin(-3600, 3600, 0.01, 4, " deg")
        self._cdw  = self._spin(-1e4, 1e4, 0.001, 6, " MeV")
        for lab, w in [("δx", self._cx), ("δx'", self._cxp),
                       ("δy", self._cy), ("δy'", self._cyp),
                       ("δφ", self._cphi), ("δW", self._cdw)]:
            f.addRow(lab, w)
        # Input dispersion (x/y ↔ ΔW correlation, mm/MeV · mrad/MeV) —
        # the matched beam of a bending line carries these; the matching
        # dialog's Apply fills them for arc/BTL cells.
        self._disp_x  = self._spin(-1e5, 1e5, 0.001, 6, " mm/MeV")
        self._disp_xp = self._spin(-1e5, 1e5, 0.001, 6, " mrad/MeV")
        self._disp_y  = self._spin(-1e5, 1e5, 0.001, 6, " mm/MeV")
        self._disp_yp = self._spin(-1e5, 1e5, 0.001, 6, " mrad/MeV")
        for lab, w in [("D_x", self._disp_x), ("D_x'", self._disp_xp),
                       ("D_y", self._disp_y), ("D_y'", self._disp_yp)]:
            f.addRow(lab, w)
        # Floor is -99.999, NOT -100: BeamConfig rejects mismatch <= -100 %
        # (it would zero out the emittance), so a spinbox that can reach
        # exactly -100 lets the form describe an unbuildable beam.
        self._mx = self._spin(-99.999, 1000, 0.1, 3, " %")
        self._my = self._spin(-99.999, 1000, 0.1, 3, " %")
        self._mz = self._spin(-99.999, 1000, 0.1, 3, " %")
        for lab, w in [("Δ ε_x", self._mx), ("Δ ε_y", self._my), ("Δ ε_z", self._mz)]:
            f.addRow(lab, w)
        return g

    # ------------------------------------------------------------------
    def _spin(self, lo, hi, step, decimals, suffix):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setDecimals(decimals); s.setSuffix(suffix)
        return s

    def _ro(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; font-size:11px;"
            f"background:{theme.BG_INSET}; padding:3px 6px; border-radius:3px;"
            f"border:1px solid {theme.BORDER_0};"
        )
        return lbl

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self._species.setCurrentText("H-")
        self._energy.setValue(2.1226695)
        self._freq.setValue(162.5)
        self._current.setValue(5.0)
        self._duty.setValue(100.0)
        self._npart.setValue(100_000)
        self._dist.setCurrentText("gaussian")
        self._cutoff.setValue(4.0)
        self._emit_nx.setValue(0.21); self._alpha_x.setValue(1.228); self._beta_x.setValue(0.316)
        self._emit_ny.setValue(0.21); self._alpha_y.setValue(-0.095394); self._beta_y.setValue(0.113)
        self._emit_z.setValue(0.06231832); self._alpha_z.setValue(0.0); self._beta_z.setValue(819.05492)
        for s in (self._cx, self._cxp, self._cy, self._cyp, self._cphi, self._cdw,
                  self._disp_x, self._disp_xp, self._disp_y, self._disp_yp,
                  self._mx, self._my, self._mz):
            s.setValue(0.0)
        self._source = "generate"
        self._distribution_file = None
        if hasattr(self, "_file_chip"):
            self._file_chip.hide()
            self._clear_file_btn.hide()
        self._apply(quiet=True)

    def _recompute_derived(self) -> None:
        try:
            from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
            sp = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}[self._species.currentText()]
            w = self._energy.value()
            gamma = 1.0 + w / sp.mass
            beta = math.sqrt(max(1 - 1 / (gamma * gamma), 0.0))
            self._d_beta.setText(f"{beta:.6f}")
            self._d_gamma.setText(f"{gamma:.6f}")
            self._d_bg.setText(f"{beta*gamma:.6f}")
        except Exception:
            pass

    def get_beam_config(self):
        """Adapter for classic dialogs that expect a BeamConfigWidget."""
        return self._build_cfg()

    def set_beam_config(self, cfg) -> None:
        """Push a BeamConfig into every widget (used by Open-Project)."""
        # Species / distribution are combo-boxes; the rest are spin-boxes.
        idx = self._species.findText(str(cfg.species))
        if idx >= 0:
            self._species.setCurrentIndex(idx)
        idx = self._dist.findText(str(cfg.distribution))
        if idx >= 0:
            self._dist.setCurrentIndex(idx)
        self._energy.setValue(float(cfg.energy))
        self._freq.setValue(float(cfg.frequency))
        self._current.setValue(float(cfg.current))
        self._duty.setValue(float(cfg.duty_cycle))
        self._npart.setValue(int(cfg.n_particles))
        self._cutoff.setValue(float(cfg.cutoff))
        self._emit_nx.setValue(float(cfg.emit_nx))
        self._alpha_x.setValue(float(cfg.alpha_x))
        self._beta_x.setValue(float(cfg.beta_x))
        self._emit_ny.setValue(float(cfg.emit_ny))
        self._alpha_y.setValue(float(cfg.alpha_y))
        self._beta_y.setValue(float(cfg.beta_y))
        self._emit_z.setValue(float(cfg.emit_z))
        self._alpha_z.setValue(float(cfg.alpha_z))
        self._beta_z.setValue(float(cfg.beta_z))
        self._cx.setValue(float(cfg.centroid_x))
        self._cxp.setValue(float(cfg.centroid_xp))
        self._cy.setValue(float(cfg.centroid_y))
        self._cyp.setValue(float(cfg.centroid_yp))
        self._cphi.setValue(float(cfg.centroid_dphi))
        self._cdw.setValue(float(cfg.centroid_dw))
        # Input dispersion — tolerate older BeamConfig/.lgproj without it.
        self._disp_x.setValue(float(getattr(cfg, "disp_x", 0.0)))
        self._disp_xp.setValue(float(getattr(cfg, "disp_xp", 0.0)))
        self._disp_y.setValue(float(getattr(cfg, "disp_y", 0.0)))
        self._disp_yp.setValue(float(getattr(cfg, "disp_yp", 0.0)))
        self._mx.setValue(float(cfg.mismatch_x))
        self._my.setValue(float(cfg.mismatch_y))
        self._mz.setValue(float(cfg.mismatch_z))
        # DC / continuous-beam fields (tolerate older BeamConfig instances
        # that may not carry the attributes).
        self._continuous.setChecked(bool(getattr(cfg, "continuous", False)))
        self._dc_dw.setValue(float(getattr(cfg, "dc_energy_spread_keV", 0.0)))
        self._on_continuous_toggled(self._continuous.isChecked())
        # Bi-Gaussian (thermal) halo fields — tolerate older configs that
        # don't carry them; they default to (0.05, 5.0) which only matter
        # when distribution == "thermal".
        self._halo_frac.setValue(float(getattr(cfg, "halo_fraction", 0.05)))
        self._halo_ratio.setValue(float(getattr(cfg, "halo_ratio", 5.0)))
        self._on_distribution_changed(self._dist.currentText())
        # Preserve the file-source state (saved-project round-trip / .dst import).
        self._source = str(getattr(cfg, "source", "generate") or "generate")
        path = getattr(cfg, "distribution_file", None)
        self._distribution_file = str(path) if path else None
        self._refresh_file_chip()
        self._apply(quiet=True)
        # A full-config load (project import, matched-beam apply, .dst
        # load) must also refresh the phase-space preview — _apply(quiet=
        # True) deliberately skips it, so regenerate it explicitly here.
        self._regen_preview()

    def _refresh_file_chip(self) -> None:
        if self._source == "file" and self._distribution_file:
            from os.path import basename
            self._file_chip.setText(f"file: {basename(self._distribution_file)}")
            self._file_chip.setToolTip(self._distribution_file)
            self._file_chip.show()
            self._clear_file_btn.show()
        else:
            self._file_chip.hide()
            self._clear_file_btn.hide()

    def _clear_file_source(self) -> None:
        """Drop the .dst file association and revert to generate mode."""
        self._source = "generate"
        self._distribution_file = None
        self._refresh_file_chip()
        self._apply(quiet=True)
        self._status.setText("file source cleared — using generated distribution")
        self.state.status_message.emit("beam source: generate")

    def _import_dst(self) -> None:
        """Open a .dst file and populate every spinbox from the file's
        sample-derived emittance/Twiss + header.  Keep the file path so
        the simulation reads the actual particles back (source='file')."""
        from PyQt6.QtCore import QSettings
        s = make_settings("Helix", "HELIX")
        start_dir = str(s.value("beam_tab/last_dst_dir", ""))
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import TraceWin .dst beam", start_dir,
            "TraceWin DST (*.dst);;All Files (*)"
        )
        if not fp:
            return
        try:
            from linac_gen.io.tracewin_dst import load_dst
            _particles, header = load_dst(fp)
        except Exception as exc:
            QMessageBox.critical(self, "Import .dst failed", str(exc))
            return

        from os.path import dirname, basename
        s.setValue("beam_tab/last_dst_dir", dirname(fp))

        # Species inference.  TraceWin .dst stores only mc² with no charge
        # sign, so proton vs H- is ambiguous on a ~938 MeV file — keep
        # whichever of those is currently selected; for a mass that
        # clearly belongs to a different species (e.g. deuteron at
        # ~1875 MeV) switch over so the run uses the right relativistic
        # parameters.
        mass = float(header.get("mass_MeV", 0.0))
        from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
        sp_mass = {"proton": PROTON.mass, "H-": H_MINUS.mass,
                   "deuteron": DEUTERON.mass}
        cur = self._species.currentText()
        # If the current species' mass is more than ~5 MeV off the file,
        # pick the closest known species by mass.
        if abs(sp_mass.get(cur, 0.0) - mass) > 5.0:
            best = min(sp_mass.items(), key=lambda kv: abs(kv[1] - mass))[0]
            idx = self._species.findText(best)
            if idx >= 0:
                self._species.setCurrentIndex(idx)

        # Header → spinboxes.  Every key is present (load_dst always
        # populates them), but use .get with sensible defaults to be safe.
        self._npart.setValue(int(header.get("n_particles", self._npart.value())))
        self._current.setValue(float(header.get("current_mA", self._current.value())))
        self._freq.setValue(float(header.get("frequency_MHz", self._freq.value())))
        self._energy.setValue(float(header.get("w_kin_ref", self._energy.value())))
        self._emit_nx.setValue(float(header.get("emit_nx", self._emit_nx.value())))
        self._emit_ny.setValue(float(header.get("emit_ny", self._emit_ny.value())))
        self._emit_z.setValue(float(header.get("emit_z", self._emit_z.value())))
        self._alpha_x.setValue(float(header.get("alpha_x", self._alpha_x.value())))
        self._beta_x.setValue(float(header.get("beta_x",  self._beta_x.value())))
        self._alpha_y.setValue(float(header.get("alpha_y", self._alpha_y.value())))
        self._beta_y.setValue(float(header.get("beta_y",  self._beta_y.value())))
        self._alpha_z.setValue(float(header.get("alpha_z", self._alpha_z.value())))
        self._beta_z.setValue(float(header.get("beta_z",  self._beta_z.value())))

        # A .dst stores absolute phi/W per particle, so the loaded beam is
        # always bunched.  Clear the continuous toggle to avoid the factory
        # silently overwriting the longitudinal columns with uniform-phase
        # synthetic values.
        self._continuous.setChecked(False)
        self._on_continuous_toggled(False)

        self._source = "file"
        self._distribution_file = fp
        self._refresh_file_chip()

        # Apply pushes the BeamConfig to AppState; regen rebuilds the
        # phase-space preview from the actual file particles.
        self._apply(quiet=True)
        self._regen_preview()
        self._status.setText(
            f"loaded {basename(fp)} — N={header.get('n_particles')}, "
            f"W={header.get('w_kin_ref'):.4f} MeV, "
            f"εnx={header.get('emit_nx'):.4f}, εny={header.get('emit_ny'):.4f}"
        )
        self.state.status_message.emit(f"beam source: {basename(fp)}")

    def _build_cfg(self):
        from linac_gen.core.config import BeamConfig
        return BeamConfig(
            species=self._species.currentText(),
            energy=self._energy.value(),
            frequency=self._freq.value(),
            current=self._current.value(),
            duty_cycle=self._duty.value(),
            n_particles=self._npart.value(),
            distribution=self._dist.currentText(),
            cutoff=self._cutoff.value(),
            emit_nx=self._emit_nx.value(), alpha_x=self._alpha_x.value(), beta_x=self._beta_x.value(),
            emit_ny=self._emit_ny.value(), alpha_y=self._alpha_y.value(), beta_y=self._beta_y.value(),
            emit_z=self._emit_z.value(), alpha_z=self._alpha_z.value(), beta_z=self._beta_z.value(),
            centroid_x=self._cx.value(), centroid_xp=self._cxp.value(),
            centroid_y=self._cy.value(), centroid_yp=self._cyp.value(),
            centroid_dphi=self._cphi.value(), centroid_dw=self._cdw.value(),
            disp_x=self._disp_x.value(), disp_xp=self._disp_xp.value(),
            disp_y=self._disp_y.value(), disp_yp=self._disp_yp.value(),
            mismatch_x=self._mx.value(), mismatch_y=self._my.value(), mismatch_z=self._mz.value(),
            continuous=bool(self._continuous.isChecked()),
            dc_energy_spread_keV=float(self._dc_dw.value()),
            halo_fraction=float(self._halo_frac.value()),
            halo_ratio=float(self._halo_ratio.value()),
            source=self._source,
            distribution_file=self._distribution_file,
        )

    def _apply(self, *, quiet: bool = False) -> None:
        try:
            cfg = self._build_cfg()
        except Exception as exc:
            self._status.setText(f"invalid: {exc}"); return
        self.state.set_beam_config(cfg)
        if not quiet:
            # User clicked Apply -- beam config lives in the .lgproj
            # JSON (not the .dat), so flag the project dirty so the
            # close/open-project prompt warns before discarding.  Skip
            # for quiet=True calls (initial population, project load,
            # matched-beam apply) which should not mark dirty.
            self.state.mark_project_dirty()
            self._status.setText("applied"); self._regen_preview()
            self.state.status_message.emit("beam config applied")

    def _regen_preview(self) -> None:
        try:
            cfg = self._build_cfg()
            from linac_gen.distributions.factory import create_beam
            beam = create_beam(cfg, seed=42)
            p = beam.alive_particles
            # Density histograms are cheap — use every surviving particle
            # (no downsample) so the log-intensity contrast is genuine.
            self._panel_xx.set_data(p[:, 0], p[:, 1])
            self._panel_yy.set_data(p[:, 2], p[:, 3])
            self._panel_zp.set_data(p[:, 4], p[:, 5])
            self._panel_xy.set_data(p[:, 0], p[:, 2])
            self._status.setText(f"preview: {len(p)} particles")
        except Exception as exc:
            self._status.setText(f"preview failed: {exc}")
