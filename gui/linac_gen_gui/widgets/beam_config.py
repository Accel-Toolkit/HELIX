"""Beam configuration panel widget.

The widget pre-populates with a realistic H- injector configuration
(2.1226695 MeV, 162.5 MHz bunching, 5 mA) so a fresh GUI launch loads
sensible machine-appropriate values rather than synthetic placeholders.
Change any field interactively; :meth:`get_beam_config` returns a
:class:`~linac_gen.core.config.BeamConfig` built from the current spinbox
values.
"""
import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QLabel, QCheckBox,
)
from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS


# Machine-specific default startup values for the H- injector reference
# design.  Gathered in one place so they are easy to retune.
_DEFAULT_SPECIES = "H-"
_DEFAULT_ENERGY_MEV = 2.1226695
_DEFAULT_FREQUENCY_MHZ = 162.5
_DEFAULT_CURRENT_MA = 5.0
_DEFAULT_N_PARTICLES = 100_000
_DEFAULT_DISTRIBUTION = "waterbag"
_DEFAULT_DUTY_CYCLE = 100.0

# Beam-center offsets (δx, δx', δy, δy', δΦ, δw).  All zero at startup.
_DEFAULT_CENTROID = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Per-plane mismatch factors (%).  0 % means the beam's Twiss IS the
# supplied Twiss -- no extra emittance inflation.
_DEFAULT_MISMATCH = (0.0, 0.0, 0.0)

# (emit_n_or_z, alpha, beta) per plane.
# X, Y use normalized RMS emittance (pi.mm.mrad) and beta in mm/(pi.mrad).
# Z uses longitudinal emittance (pi.deg.MeV) and beta in deg/(pi.MeV).
_DEFAULT_TWISS = {
    "X": (0.21,        1.228,        0.316),
    "Y": (0.21,       -0.095394323,  0.113),
    "Z": (0.06231832,  0.0,          819.05492),
}

_SPECIES_MASS = {"proton": PROTON.mass, "deuteron": DEUTERON.mass, "H-": H_MINUS.mass}


class BeamConfigWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Species
        species_group = QGroupBox("Species")
        species_layout = QFormLayout(species_group)
        self._species_combo = QComboBox()
        self._species_combo.addItems(["proton", "deuteron", "H-"])
        self._species_combo.setCurrentText(_DEFAULT_SPECIES)
        self._species_combo.currentTextChanged.connect(self._update_derived)
        species_layout.addRow("Type:", self._species_combo)
        layout.addWidget(species_group)

        # Energy
        energy_group = QGroupBox("Energy")
        energy_layout = QFormLayout(energy_group)
        self._energy_spin = QDoubleSpinBox()
        self._energy_spin.setRange(0.001, 10000)
        self._energy_spin.setDecimals(7)
        self._energy_spin.setValue(_DEFAULT_ENERGY_MEV)
        self._energy_spin.setSuffix(" MeV")
        self._energy_spin.valueChanged.connect(self._update_derived)
        energy_layout.addRow("W_kin:", self._energy_spin)
        self._beta_label = QLabel()
        self._gamma_label = QLabel()
        self._bg_label = QLabel()
        energy_layout.addRow("beta:", self._beta_label)
        energy_layout.addRow("gamma:", self._gamma_label)
        energy_layout.addRow("beta*gamma:", self._bg_label)
        layout.addWidget(energy_group)

        # RF
        rf_group = QGroupBox("RF")
        rf_layout = QFormLayout(rf_group)
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(1, 5000)
        self._freq_spin.setDecimals(3)
        self._freq_spin.setValue(_DEFAULT_FREQUENCY_MHZ)
        self._freq_spin.setSuffix(" MHz")
        rf_layout.addRow("Frequency:", self._freq_spin)
        layout.addWidget(rf_group)

        # Current + duty cycle
        current_group = QGroupBox("Current")
        current_layout = QFormLayout(current_group)
        self._current_spin = QDoubleSpinBox()
        self._current_spin.setRange(0, 1000)
        self._current_spin.setDecimals(3)
        self._current_spin.setValue(_DEFAULT_CURRENT_MA)
        self._current_spin.setSuffix(" mA")
        current_layout.addRow("I (peak):", self._current_spin)

        self._duty_spin = QDoubleSpinBox()
        self._duty_spin.setRange(0.001, 100.0)
        self._duty_spin.setDecimals(3)
        self._duty_spin.setValue(_DEFAULT_DUTY_CYCLE)
        self._duty_spin.setSuffix(" %")
        current_layout.addRow("Duty cycle:", self._duty_spin)

        # Continuous-beam toggle (pre-RFQ / LEBT).  Required for envelope+SC
        # to use the 2-D DC analytic kick — without it EnvelopeSolver
        # dispatches to the 3-D Ferrario kick which early-returns identity
        # for emit_z=0.  See commit c400f5f.
        self._continuous = QCheckBox("Continuous beam (pre-RFQ / LEBT)")
        self._continuous.setToolTip(
            "Enable for pre-RFQ / LEBT continuous beams.\n"
            "Required for envelope+SC to use the 2-D DC analytic kick;\n"
            "without it, ENV+SC silently equals ENV+NOSC for emit_z=0."
        )
        self._dc_dw = QDoubleSpinBox()
        self._dc_dw.setRange(0.0, 100.0)
        self._dc_dw.setDecimals(3)
        self._dc_dw.setSuffix(" keV")
        self._dc_dw.setToolTip("1σ energy spread for the continuous beam.")
        current_layout.addRow("", self._continuous)
        current_layout.addRow("DC energy spread:", self._dc_dw)
        layout.addWidget(current_group)

        # Particles
        part_group = QGroupBox("Particles")
        part_layout = QFormLayout(part_group)
        self._npart_spin = QSpinBox()
        self._npart_spin.setRange(100, 10_000_000)
        self._npart_spin.setValue(_DEFAULT_N_PARTICLES)
        part_layout.addRow("N:", self._npart_spin)
        self._dist_combo = QComboBox()
        self._dist_combo.addItems(["waterbag", "gaussian", "kv", "parabolic", "uniform"])
        self._dist_combo.setCurrentText(_DEFAULT_DISTRIBUTION)
        self._dist_combo.currentTextChanged.connect(self._on_distribution_changed)
        part_layout.addRow("Distribution:", self._dist_combo)

        # Gaussian-only cutoff (σ units).  Active only when distribution = gaussian.
        self._cutoff_spin = QDoubleSpinBox()
        self._cutoff_spin.setRange(1.0, 10.0)
        self._cutoff_spin.setDecimals(2)
        self._cutoff_spin.setValue(4.0)
        self._cutoff_spin.setSuffix(" σ")
        part_layout.addRow("Gaussian cutoff:", self._cutoff_spin)
        layout.addWidget(part_group)
        # Set initial enabled state to match current distribution
        self._on_distribution_changed(self._dist_combo.currentText())

        # Emittances (X, Y, Z)
        for plane, defaults in _DEFAULT_TWISS.items():
            group = QGroupBox(f"{plane} Plane")
            gl = QFormLayout(group)

            emit = QDoubleSpinBox()
            emit.setRange(0, 1000)
            emit.setDecimals(8)
            emit.setValue(defaults[0])
            gl.addRow("emit_n:" if plane != "Z" else "emit_z:", emit)

            alpha = QDoubleSpinBox()
            alpha.setRange(-1000, 1000)
            alpha.setDecimals(6)
            alpha.setValue(defaults[1])
            gl.addRow("alpha:", alpha)

            beta = QDoubleSpinBox()
            beta.setRange(0.001, 10000)
            beta.setDecimals(5)
            beta.setValue(defaults[2])
            gl.addRow("beta:", beta)

            layout.addWidget(group)
            setattr(self, f"_emit_{plane.lower()}", emit)
            setattr(self, f"_alpha_{plane.lower()}", alpha)
            setattr(self, f"_beta_{plane.lower()}", beta)

        # Beam-center offsets.  Each coordinate is added to every generated
        # particle so the whole distribution sits off-axis.
        centroid_group = QGroupBox("Beam Center")
        centroid_layout = QFormLayout(centroid_group)
        centroid_fields = [
            ("dx",    "δx (mm):",     _DEFAULT_CENTROID[0], (-1000.0, 1000.0), 6),
            ("dxp",   "δx' (mrad):",  _DEFAULT_CENTROID[1], (-1000.0, 1000.0), 6),
            ("dy",    "δy (mm):",     _DEFAULT_CENTROID[2], (-1000.0, 1000.0), 6),
            ("dyp",   "δy' (mrad):",  _DEFAULT_CENTROID[3], (-1000.0, 1000.0), 6),
            ("dphi",  "δΦ (deg):",    _DEFAULT_CENTROID[4], (-3600.0, 3600.0), 6),
            ("dw",    "δW (MeV):",    _DEFAULT_CENTROID[5], (-10000.0, 10000.0), 6),
        ]
        for attr, label, default, (lo, hi), decimals in centroid_fields:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(decimals)
            spin.setValue(default)
            centroid_layout.addRow(label, spin)
            setattr(self, f"_centroid_{attr}", spin)
        layout.addWidget(centroid_group)

        # Input dispersion (mm/MeV, mrad/MeV).  The matched beam of a
        # bending line (arc FODO / BTL) carries x–ΔW correlations; the
        # generator applies x += D·ΔW before the centroid offsets.
        disp_group = QGroupBox("Dispersion (per MeV)")
        disp_layout = QFormLayout(disp_group)
        disp_fields = [
            ("disp_x",  "D_x (mm/MeV):"),
            ("disp_xp", "D_x' (mrad/MeV):"),
            ("disp_y",  "D_y (mm/MeV):"),
            ("disp_yp", "D_y' (mrad/MeV):"),
        ]
        for attr, label in disp_fields:
            spin = QDoubleSpinBox()
            spin.setRange(-1e5, 1e5)
            spin.setDecimals(6)
            spin.setValue(0.0)
            disp_layout.addRow(label, spin)
            setattr(self, f"_{attr}", spin)
        layout.addWidget(disp_group)

        # Courant-Snyder mismatch (%). 0 -> beam Twiss is the matched Twiss.
        mismatch_group = QGroupBox("Mismatch (%)")
        mismatch_layout = QFormLayout(mismatch_group)
        for plane, default in zip(("x", "y", "z"), _DEFAULT_MISMATCH):
            spin = QDoubleSpinBox()
            spin.setRange(-99.999, 1000.0)   # -100 % would zero emittance
            spin.setDecimals(3)
            spin.setValue(default)
            spin.setSuffix(" %")
            mismatch_layout.addRow(f"{plane.upper()}:", spin)
            setattr(self, f"_mismatch_{plane}", spin)
        layout.addWidget(mismatch_group)

        layout.addStretch()
        self._update_derived()

    def _on_distribution_changed(self, text: str) -> None:
        """Enable cutoff only when the distribution is gaussian."""
        self._cutoff_spin.setEnabled(text == "gaussian")

    def _update_derived(self):
        mass = _SPECIES_MASS.get(self._species_combo.currentText(), PROTON.mass)
        w_kin = self._energy_spin.value()
        gamma_minus_one = w_kin / mass
        gamma = 1.0 + gamma_minus_one
        # Same numerically-stable formula used in ReferenceParticle.
        bg = math.sqrt(gamma_minus_one * (gamma + 1.0))
        beta = bg / gamma
        self._beta_label.setText(f"{beta:.6f}")
        self._gamma_label.setText(f"{gamma:.6f}")
        self._bg_label.setText(f"{bg:.6f}")

    def get_beam_config(self):
        from linac_gen.core.config import BeamConfig
        return BeamConfig(
            species=self._species_combo.currentText(),
            energy=self._energy_spin.value(),
            frequency=self._freq_spin.value(),
            current=self._current_spin.value(),
            duty_cycle=self._duty_spin.value(),
            n_particles=self._npart_spin.value(),
            distribution=self._dist_combo.currentText(),
            cutoff=self._cutoff_spin.value(),
            emit_nx=self._emit_x.value(),
            alpha_x=self._alpha_x.value(),
            beta_x=self._beta_x.value(),
            emit_ny=self._emit_y.value(),
            alpha_y=self._alpha_y.value(),
            beta_y=self._beta_y.value(),
            emit_z=self._emit_z.value(),
            alpha_z=self._alpha_z.value(),
            beta_z=self._beta_z.value(),
            centroid_x=self._centroid_dx.value(),
            centroid_xp=self._centroid_dxp.value(),
            centroid_y=self._centroid_dy.value(),
            centroid_yp=self._centroid_dyp.value(),
            centroid_dphi=self._centroid_dphi.value(),
            centroid_dw=self._centroid_dw.value(),
            disp_x=self._disp_x.value(),
            disp_xp=self._disp_xp.value(),
            disp_y=self._disp_y.value(),
            disp_yp=self._disp_yp.value(),
            mismatch_x=self._mismatch_x.value(),
            mismatch_y=self._mismatch_y.value(),
            mismatch_z=self._mismatch_z.value(),
            continuous=self._continuous.isChecked(),
            dc_energy_spread_keV=self._dc_dw.value(),
        )
