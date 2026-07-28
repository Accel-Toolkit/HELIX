"""Simulation settings dialog.

Initialises its fields from the main window's live ``_sc_config`` and
field-map class attributes so the values shown reflect what the
simulation will actually use.

The "Numerics" group exposes every advanced toggle:

* **Poisson Green's function** — IGF (default) or legacy point-source.
* **Particle-mesh kernel**     — CIC (default) or TSC.
* **Field-map integrator**     — KD (default) or symplectic DKD.
* **Field-map interpolation**  — linear (default) or tricubic.

Defaults match the LEBT/MEBT validation runs.  Each control is paired
with a tooltip so the user knows what they're flipping.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QDialogButtonBox,
)

from linac_gen.core.config import SpaceChargeConfig


class SimulationSettingsDialog(QDialog):
    def __init__(self, parent=None, current: SpaceChargeConfig | None = None,
                 integrator_kind: str = "kd", interp_kind: str = "linear"):
        super().__init__(parent)
        self.setWindowTitle("Simulation Settings")
        layout = QVBoxLayout(self)

        # Seed from the caller's live config so we never show a value that
        # disagrees with what the simulation actually uses.
        cfg = current or SpaceChargeConfig()

        # ------------------------------------------------------------------
        # Space-charge grid + boundary
        # ------------------------------------------------------------------
        sc_group = QGroupBox("Space Charge")
        sc_layout = QFormLayout(sc_group)

        self._nx = QSpinBox(); self._nx.setRange(8, 256); self._nx.setValue(cfg.nx)
        self._ny = QSpinBox(); self._ny.setRange(8, 256); self._ny.setValue(cfg.ny)
        self._nz = QSpinBox(); self._nz.setRange(8, 256); self._nz.setValue(cfg.nz)
        sc_layout.addRow("nx:", self._nx)
        sc_layout.addRow("ny:", self._ny)
        sc_layout.addRow("nz:", self._nz)

        self._grid_extent = QDoubleSpinBox()
        self._grid_extent.setRange(1.0, 20.0)
        self._grid_extent.setDecimals(2)
        self._grid_extent.setSingleStep(0.5)
        self._grid_extent.setValue(cfg.grid_extent)
        self._grid_extent.setSuffix(" σ")
        sc_layout.addRow("Grid extent:", self._grid_extent)

        self._grid_mode = QComboBox()
        self._grid_mode.addItems(["fixed", "adaptive"])
        self._grid_mode.setCurrentText(cfg.grid_mode)
        sc_layout.addRow("Grid mode:", self._grid_mode)

        self._bc_combo = QComboBox()
        # Only open-BC is implemented (SpaceChargeConfig refuses
        # "periodic" since the 2026-07 review round).
        self._bc_combo.addItems(["open"])
        if cfg.boundary in ("open",):
            self._bc_combo.setCurrentText(cfg.boundary)
        sc_layout.addRow("Boundary:", self._bc_combo)

        layout.addWidget(sc_group)

        # ------------------------------------------------------------------
        # Numerics — advanced toggles for power users
        # ------------------------------------------------------------------
        num_group = QGroupBox("Numerics (advanced)")
        num_layout = QFormLayout(num_group)

        self._green = QComboBox()
        self._green.addItems(["igf", "point"])
        self._green.setCurrentText(getattr(cfg, "green_kind", "igf"))
        self._green.setToolTip(
            "Hockney FFT Green's function.\n"
            "  igf   — Integrated Green Function (Qiang 2006). Recommended.\n"
            "  point — sampled 1/(4πε₀ r). Legacy; kept for back-compat regression."
        )
        num_layout.addRow("Green's function:", self._green)

        self._kernel = QComboBox()
        self._kernel.addItems(["cic", "tsc"])
        self._kernel.setCurrentText(getattr(cfg, "kernel", "cic"))
        self._kernel.setToolTip(
            "Particle-mesh deposition / gather kernel (must match).\n"
            "  cic — 8-corner trilinear, 1st-order. Default; fine for linac "
            "stages where RF cavities reset per-step grid noise.\n"
            "  tsc — 27-cell quadratic. Lower grid noise but ~3× more work. "
            "Use for long no-RF transport (e.g. BTL): CIC's per-step "
            "self-force/aliasing accumulates into longitudinal filamentation "
            "without cavity restoring force."
        )
        num_layout.addRow("Particle-mesh kernel:", self._kernel)

        self._dc_kernel = QComboBox()
        self._dc_kernel.addItems(["uniform", "gaussian", "pic2d"])
        self._dc_kernel.setCurrentText(getattr(cfg, "dc_kernel", "uniform"))
        self._dc_kernel.setToolTip(
            "Continuous (DC) beam space-charge model.\n"
            "  uniform  — analytic linear uniform-elliptical-cylinder kick. "
            "Default; matches the rigid-Σ envelope solver, fastest, closest "
            "to TraceWin's exit σ on PXIE-LEBT.\n"
            "  gaussian — Bassetti-Erskine field of a 2-D Gaussian density. "
            "Per-particle non-linear (rigid σ).\n"
            "  pic2d    — 2-D Hockney FFT PIC over the actual particle "
            "distribution. Captures full per-particle nonlinearity; "
            "TraceWin PICNIC_2D analogue."
        )
        num_layout.addRow("DC SC kernel:", self._dc_kernel)

        self._integrator = QComboBox()
        self._integrator.addItems(["kd", "dkd"])
        self._integrator.setCurrentText(integrator_kind)
        self._integrator.setToolTip(
            "Field-map integrator (FieldMap3D).\n"
            "  kd  — first-order kick-then-drift. Default.\n"
            "  dkd — second-order symplectic Drift-Kick-Drift. Better for "
            "long / periodic / storage-ring trajectories."
        )
        num_layout.addRow("Field-map integrator:", self._integrator)

        self._interp = QComboBox()
        self._interp.addItems(["linear", "cubic"])
        self._interp.setCurrentText(interp_kind)
        self._interp.setToolTip(
            "Field-map interpolation order (FieldMap3D).\n"
            "  linear — trilinear. Default.\n"
            "  cubic  — tricubic; useful for sharp-gradient cavities or quads."
        )
        num_layout.addRow("Field-map interp:", self._interp)

        layout.addWidget(num_group)

        # ------------------------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ----------------------------------------------------------------------
    def get_sc_config(self) -> SpaceChargeConfig:
        return SpaceChargeConfig(
            nx=self._nx.value(), ny=self._ny.value(), nz=self._nz.value(),
            grid_extent=self._grid_extent.value(),
            grid_mode=self._grid_mode.currentText(),
            boundary=self._bc_combo.currentText(),
            kernel=self._kernel.currentText(),
            green_kind=self._green.currentText(),
            dc_kernel=self._dc_kernel.currentText(),
        )

    def get_integrator_kind(self) -> str:
        return self._integrator.currentText()

    def get_interp_kind(self) -> str:
        return self._interp.currentText()
