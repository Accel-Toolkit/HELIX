"""SC convergence check dialog.

Runs a small sweep over ``grid_extent`` and ``n_grid`` with the current
lattice + beam, reports 1%-converged values, and offers an "Apply" button
that pushes the recommended settings into the main window's
``SpaceChargeConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QTextEdit, QMessageBox,
)

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.core.step_config import StepConfig
from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS

SPECIES_MAP = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}


# ---------------------------------------------------------------------------
# Sweep points  (kept intentionally small -- ~40-60 s total on a default laptop)
# ---------------------------------------------------------------------------
GRID_EXT_POINTS = [3.0, 4.0, 5.0, 6.0]        # PIC half-grid in sigmas
N_GRID_POINTS   = [48, 64, 96]                # PIC cells per axis
STEP1_POINTS    = [50, 100, 200, 500]         # integration per metre (drifts / field maps)
STEP2_POINTS    = [25, 50, 100, 200]          # SC kicks per metre (drifts / field maps)
TOL_EMIT        = 0.01                        # "converged" when Δemit_x ≤ 1% of reference


@dataclass
class SweepRow:
    knob: str
    value: float
    sigma_x: float
    sigma_y: float
    emit_x: float
    emit_y: float
    runtime: float


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
class ConvergenceWorker(QThread):
    """Runs the grid_ext + n_grid sweeps in the background."""

    progress = pyqtSignal(int, str)           # % done, status text
    row_done = pyqtSignal(object)             # SweepRow
    finished = pyqtSignal(list, dict)         # rows, recommendation
    error = pyqtSignal(str)

    def __init__(self, lattice, beam_config, sc_config, n_particles=5000):
        super().__init__()
        self._lattice = lattice
        # Remember the lattice's original step_config so we can restore it
        # after each sweep point; sweeps mutate lattice.step_config in place.
        self._original_step_config = getattr(
            lattice, "step_config", StepConfig(),
        )
        self._beam_config = beam_config
        self._sc_config = sc_config
        self._n_particles = n_particles

    # ------------------------------------------------------------------
    def _make_beam(self) -> Beam:
        """Build a beam from the current BeamConfig using the shared factory.

        The factory respects ``cfg.distribution`` (gaussian, waterbag, KV,
        parabolic, uniform) as well as ``cutoff``, centroid and mismatch
        fields.  Earlier revisions of this method ignored the distribution
        choice and silently used a Gaussian — which drifted convergence
        results away from the rest of the app.  The scan uses a reduced
        ``n_particles`` for speed but otherwise matches a real Run MP.
        """
        from dataclasses import replace
        from linac_gen.distributions.factory import create_beam

        cfg = replace(self._beam_config, n_particles=self._n_particles)
        return create_beam(cfg, seed=42)

    def _simulate_one(self, sc_config, step_config=None) -> tuple[float, float, float, float, float]:
        """Run one simulation, return (sx, sy, emit_x, emit_y, runtime).

        ``step_config`` is applied by overwriting ``lattice.step_config``
        for the duration of the run.
        """
        beam = self._make_beam()
        if step_config is not None:
            self._lattice.step_config = step_config
        else:
            self._lattice.step_config = self._original_step_config
        sim = Simulation(self._lattice, beam, space_charge=sc_config)
        t0 = time.time()
        res = sim.run()
        return (res.sigma_x[-1], res.sigma_y[-1],
                res.emit_x[-1], res.emit_y[-1], time.time() - t0)

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            rows: list[SweepRow] = []
            total = (len(GRID_EXT_POINTS) + len(N_GRID_POINTS)
                     + len(STEP1_POINTS) + len(STEP2_POINTS))
            done = 0

            # Sweep 1: grid_extent  (varies SC config; keeps base step_config)
            for v in GRID_EXT_POINTS:
                self.progress.emit(
                    int(100 * done / total),
                    f"Sweeping grid_extent = {v:.1f} sigma ...",
                )
                sc = replace(self._sc_config, grid_extent=v)
                sx, sy, ex, ey, dt = self._simulate_one(sc)
                rows.append(SweepRow("grid_ext", v, sx, sy, ex, ey, dt))
                self.row_done.emit(rows[-1])
                done += 1

            # Sweep 2: n_grid
            for v in N_GRID_POINTS:
                self.progress.emit(
                    int(100 * done / total),
                    f"Sweeping n_grid = {v} cells ...",
                )
                sc = replace(self._sc_config, nx=v, ny=v, nz=v)
                sx, sy, ex, ey, dt = self._simulate_one(sc)
                rows.append(SweepRow("n_grid", v, sx, sy, ex, ey, dt))
                self.row_done.emit(rows[-1])
                done += 1

            # Sweep 3: integration step density (step1) -- affects DRIFT / FIELD_MAP
            base_step = self._original_step_config
            for v in STEP1_POINTS:
                self.progress.emit(
                    int(100 * done / total),
                    f"Sweeping step1 (integration/m) = {v} ...",
                )
                sc = self._sc_config
                step = StepConfig(
                    integration_steps_per_metre=float(v),
                    sc_steps_per_metre=base_step.sc_steps_per_metre,
                )
                sx, sy, ex, ey, dt = self._simulate_one(sc, step_config=step)
                rows.append(SweepRow("step1", float(v), sx, sy, ex, ey, dt))
                self.row_done.emit(rows[-1])
                done += 1

            # Sweep 4: SC kick density (step2) -- affects DRIFT / FIELD_MAP
            for v in STEP2_POINTS:
                self.progress.emit(
                    int(100 * done / total),
                    f"Sweeping step2 (SC kicks/m) = {v} ...",
                )
                sc = self._sc_config
                step = StepConfig(
                    integration_steps_per_metre=base_step.integration_steps_per_metre,
                    sc_steps_per_metre=float(v),
                )
                sx, sy, ex, ey, dt = self._simulate_one(sc, step_config=step)
                rows.append(SweepRow("step2", float(v), sx, sy, ex, ey, dt))
                self.row_done.emit(rows[-1])
                done += 1

            # Restore lattice to its original step config now that sweeps are done.
            self._lattice.step_config = self._original_step_config

            # Recommendations: for each knob, smallest value with <=1% emit_x
            # change relative to the finest sweep point.
            rec: dict[str, float] = {}
            for knob in ("grid_ext", "n_grid", "step1", "step2"):
                kx = [r for r in rows if r.knob == knob]
                if not kx:
                    continue
                ref_emit = kx[-1].emit_x
                for r in kx:
                    if abs(r.emit_x - ref_emit) / max(abs(ref_emit), 1e-30) <= TOL_EMIT:
                        rec[knob] = r.value
                        break
                else:
                    rec[knob] = kx[-1].value

            self.progress.emit(100, "Done.")
            self.finished.emit(rows, rec)
        except Exception as exc:          # noqa: BLE001
            # Make sure we don't leave the lattice with a modified step_config.
            try:
                self._lattice.step_config = self._original_step_config
            except Exception:
                pass
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------
class ConvergenceDialog(QDialog):
    """Modal dialog: run sweeps, display results, offer Apply."""

    def __init__(self, parent, lattice, beam_config, sc_config):
        super().__init__(parent)
        self.setWindowTitle("Space-Charge Convergence Check")
        self.setMinimumSize(720, 560)

        self._lattice = lattice
        self._beam_config = beam_config
        self._sc_config = sc_config
        self._recommended_sc: SpaceChargeConfig | None = None
        self._worker: ConvergenceWorker | None = None

        layout = QVBoxLayout(self)

        header = QLabel(
            "<b>SC Convergence Check</b><br>"
            "Sweeps four numerical parameters for the current lattice and "
            "beam, finds the 1 %-converged values, and offers to apply them:"
            "<ul>"
            "<li><tt>grid_extent</tt> — PIC grid half-size (σ multiples)</li>"
            "<li><tt>n_grid</tt> — PIC cells per axis (<tt>nx = ny = nz</tt>)</li>"
            "<li><tt>step1</tt> — integration sub-steps per metre in drifts / field maps</li>"
            "<li><tt>step2</tt> — space-charge kicks per metre in drifts / field maps</li>"
            "</ul>"
            "Takes ~40–70 seconds with 5 000 particles."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Convergence Study")
        self._run_btn.clicked.connect(self._start_run)
        btn_row.addWidget(self._run_btn)

        self._apply_btn = QPushButton("Apply Recommended")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self._apply_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._status = QLabel("Idle.")
        layout.addWidget(self._status)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFontFamily("Courier")
        layout.addWidget(self._log, stretch=1)

    # ------------------------------------------------------------------
    def _start_run(self) -> None:
        if self._lattice is None or not self._lattice.elements:
            QMessageBox.warning(self, "No lattice",
                                "Load a lattice before running convergence.")
            return

        self._log.clear()
        self._log.append(
            f"Current SC config: nx={self._sc_config.nx}  "
            f"grid_extent={self._sc_config.grid_extent}  "
            f"grid_mode={self._sc_config.grid_mode}\n"
        )
        self._log.append(
            f"Beam: {self._beam_config.species}  "
            f"W={self._beam_config.energy:g} MeV  "
            f"I={self._beam_config.current:g} mA\n"
        )
        self._log.append(
            f"Current step_config: step1 = "
            f"{self._lattice.step_config.integration_steps_per_metre} /m,  step2 = "
            f"{self._lattice.step_config.sc_steps_per_metre} /m\n"
        )
        self._log.append(
            "Running sweeps "
            "(grid_ext ∈ {3,4,5,6} σ, "
            "n_grid ∈ {48,64,96} cells, "
            "step1 ∈ {50,100,200,500}/m, "
            "step2 ∈ {25,50,100,200}/m) ...\n"
        )
        self._run_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._progress.setValue(0)

        self._worker = ConvergenceWorker(
            self._lattice, self._beam_config, self._sc_config, n_particles=5000,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.row_done.connect(self._on_row)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._status.setText(msg)

    def _on_row(self, row: SweepRow) -> None:
        self._log.append(
            f"  {row.knob:10s}={row.value:>6.1f}   "
            f"sigma_x={row.sigma_x:6.3f}  sigma_y={row.sigma_y:6.3f}  "
            f"emit_x={row.emit_x:7.3f}  emit_y={row.emit_y:7.3f}   "
            f"t={row.runtime:5.1f}s"
        )

    def _on_finished(self, rows: list, recommendation: dict) -> None:
        self._status.setText("Done.")
        self._progress.setValue(100)

        grid_ext = recommendation.get("grid_ext", self._sc_config.grid_extent)
        n_grid = int(recommendation.get("n_grid", self._sc_config.nx))
        step1 = float(recommendation.get(
            "step1", self._lattice.step_config.integration_steps_per_metre,
        ))
        step2 = float(recommendation.get(
            "step2", self._lattice.step_config.sc_steps_per_metre,
        ))

        self._recommended_sc = replace(
            self._sc_config,
            grid_extent=grid_ext,
            nx=n_grid, ny=n_grid, nz=n_grid,
        )
        self._recommended_step = StepConfig(
            integration_steps_per_metre=step1,
            sc_steps_per_metre=step2,
        )

        self._log.append("")
        self._log.append("Recommendation (1 % emit_x convergence):")
        self._log.append(f"  grid_extent = {grid_ext:g} sigma")
        self._log.append(f"  n_grid      = {n_grid} cells (nx=ny=nz)")
        self._log.append(f"  step1       = {step1:g} /m  (integration)")
        self._log.append(f"  step2       = {step2:g} /m  (SC kick)")

        current = self._sc_config
        current_step = self._lattice.step_config
        unchanged = (
            grid_ext == current.grid_extent
            and n_grid == current.nx
            and step1 == current_step.integration_steps_per_metre
            and step2 == current_step.sc_steps_per_metre
        )
        if unchanged:
            self._log.append("  -> current settings already converged; no change.")
            self._apply_btn.setEnabled(False)
        else:
            self._log.append("  Click 'Apply Recommended' to update the simulation config.")
            self._apply_btn.setEnabled(True)

        self._run_btn.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._status.setText(f"Failed: {msg}")
        self._log.append(f"\nERROR: {msg}")
        self._run_btn.setEnabled(True)

    def _on_apply(self) -> None:
        if self._recommended_sc is None:
            return
        # Just close with accept; parent reads `self.recommended_sc()` after exec_.
        self.accept()

    # ------------------------------------------------------------------
    def recommended_sc(self) -> SpaceChargeConfig | None:
        return self._recommended_sc

    def recommended_step(self) -> StepConfig | None:
        """Recommended StepConfig (step1 / step2).  May be ``None`` if the
        user closed the dialog without running the study or without applying.
        """
        return getattr(self, "_recommended_step", None)
