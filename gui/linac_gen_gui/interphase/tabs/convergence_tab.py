"""Numerics tab — PIC & integration-step convergence scans.

Four axes, same as the classic ``ConvergenceDialog``:

  1. grid (nx = ny = nz)           — PIC cells per axis
  2. grid_extent                   — PIC half-size in σ multiples
  3. step1 (integration/m)         — drifts / field-map integration density
  4. step2 (SC kicks/m)            — drifts / field-map SC kick density

Plus a fifth, Interphase-only axis (useful for MP noise studies):

  5. n_particles

Convergence metric is ε_x (same as the classic — tolerance 1 %).
"""
from __future__ import annotations

import time
from dataclasses import replace
import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSettings
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QComboBox,
    QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox, QTextEdit,
    QTabWidget, QCheckBox, QToolButton,
)

from linac_gen_gui.interphase.app_settings import make_settings
from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.state import AppState


# --- Axis vocabulary --------------------------------------------------------
AXIS_GRID    = "grid (nx=ny=nz)"
AXIS_EXTENT  = "grid_extent (σ)"
AXIS_STEP1   = "step1 (integration/m)"
AXIS_STEP2   = "step2 (SC kicks/m)"
AXIS_NPART   = "n_particles"
TOL_EMIT     = 0.01      # 1 % — same as classic dialog


# Strong references to scan workers that outlived a relaunch: dropping
# the last reference to a live QThread destroys it mid-run and aborts
# the process.  Pruned when each thread finishes.
_ZOMBIE_WORKERS: list = []


def _park_zombie(worker) -> None:
    _ZOMBIE_WORKERS.append(worker)

    def _prune() -> None:
        try:
            _ZOMBIE_WORKERS.remove(worker)
        except ValueError:
            pass
    worker.finished.connect(_prune)
    if not worker.isRunning():
        _prune()


def _group_qss() -> str:
    return (
        f"QGroupBox {{ color:{theme.TEXT_2}; border:1px solid {theme.BORDER_0};"
        f" border-radius:4px; margin-top:12px; padding-top:6px; }} "
        f"QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 6px;"
        f" color:{theme.TEXT_2}; font-size:10px; letter-spacing:1px;"
        f" text-transform:uppercase; background:{theme.BG_0}; }}"
    )


def _output_tabs_qss() -> str:
    # `min-width:60px` keeps "Plot" / "Table" from being squeezed to
    # "p" / "t" by Qt's default elide-on-overflow when the dock is
    # narrow.  Without it, the longest tab dominates and Qt truncates
    # the shorter labels rather than horizontally scrolling.
    return (
        f"QTabWidget::pane {{ background:{theme.BG_0};"
        f" border:1px solid {theme.BORDER_0}; border-radius:4px; top:-1px; }}"
        f"QTabBar::tab {{ background:{theme.BG_1}; color:{theme.TEXT_2};"
        f" padding:6px 14px; border:1px solid {theme.BORDER_0}; border-bottom:0;"
        f" border-top-left-radius:4px; border-top-right-radius:4px;"
        f" font-size:11px; min-width:60px; }}"
        f"QTabBar::tab:selected {{ background:{theme.BG_0}; color:{theme.TEXT_0};"
        f" border-top:1px solid {theme.ACCENT}; }}"
    )


# ---------------------------------------------------------------------------
class CollapsibleSection(QWidget):
    """A click-to-toggle section: a QToolButton header above a content
    QWidget that hides / shows on click.  Multi-open — sections are
    independent.  The inner content uses a QFormLayout with the macOS
    overrides baked in, so each section looks like the flat panel it
    replaces."""

    toggled_changed = pyqtSignal(str, bool)   # (title, expanded)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title

        self._toggle = QToolButton()
        # QToolButton treats '&' as a mnemonic; escape so '&' renders literally.
        self._toggle.setText(f"  {title.replace('&', '&&')}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.setStyleSheet(
            f"QToolButton {{ border:none; background:transparent;"
            f" padding:4px 8px; font-weight:600; text-align:left;"
            f" color:{theme.TEXT_1}; }}"
            f"QToolButton:hover {{ color:{theme.TEXT_0}; }}"
            f"QToolButton:checked {{ color:{theme.ACCENT}; }}"
        )
        self._toggle.toggled.connect(self._on_toggled)

        self._content = QWidget()
        form = QFormLayout(self._content)
        # Same macOS overrides as the flat panel had — baked in here so
        # every section's field alignment stays consistent.
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setContentsMargins(20, 0, 0, 6)
        self._form = form
        self._content.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.updateGeometry()
        self.toggled_changed.emit(self._title, checked)

    def addRow(self, *args) -> None:
        """Passthrough to the inner form's addRow — accepts both
        ``(label, widget)`` and single-widget (label-less span)."""
        self._form.addRow(*args)

    def setExpanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)

    def isExpanded(self) -> bool:
        return self._toggle.isChecked()

    def title(self) -> str:
        return self._title


# ---------------------------------------------------------------------------
class _ScanWorker(QThread):
    progress  = pyqtSignal(int)             # 0..100
    row_done  = pyqtSignal(int, object)     # (index, result_dict)
    failed    = pyqtSignal(str)
    done      = pyqtSignal(bool)            # stopped_by_user

    def __init__(self, lattice, beam_config, axis: str, values: list,
                 fixed_nx: int, fixed_extent: float,
                 fixed_step1: float, fixed_step2: float,
                 scan_n_particles: int | None = None,
                 parallel_workers: int = 0,
                 lattice_path: str | None = None,
                 use_gpu: str = "auto"):
        super().__init__()
        # Cooperative stop — same threading.Event pattern as the
        # envelope/MP workers (workers.py).  A plain Event (rather than
        # only QThread.requestInterruption) also works when run() is
        # driven synchronously in tests, where the thread never starts.
        import threading
        self._stop_event = threading.Event()
        self.lattice = lattice
        self.cfg = beam_config
        self.axis = axis
        self.values = values
        self.fixed_nx = fixed_nx
        self.fixed_extent = fixed_extent
        self.fixed_step1 = fixed_step1
        self.fixed_step2 = fixed_step2
        # Override n_particles used during the scan (None → use cfg.n_particles)
        self.scan_n_particles = scan_n_particles
        # Process-pool fan-out: 0 → serial, N → up to N parallel points
        self.parallel_workers = int(parallel_workers)
        # Needed for parallel path (lattice must be re-parsed in each worker)
        self.lattice_path = lattice_path
        # PIC FFT backend ("auto" | "cpu" | "gpu" | "cuda" | "mps").  For
        # parallel runs the pool auto-downgrades any GPU-like value to
        # "cpu" per worker since one GPU can't parallelise across processes.
        self.use_gpu = use_gpu

    # -- cooperative stop ------------------------------------------------
    def request_stop(self) -> None:
        """Ask the scan to stop at the next point boundary (thread-safe)."""
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            # Parallel path: use the process pool when the user opted in
            # AND we have a lattice path (worker processes need to re-parse).
            if (self.parallel_workers > 1 and self.lattice_path is not None
                    and self.axis not in (AXIS_STEP1, AXIS_STEP2)):
                # step1/step2 mutate lattice.step_config globally; easier to keep
                # those two in-process and let the pool handle the PIC-heavy
                # grid/extent/n_particles sweeps where the wall-clock win lives.
                self._run_parallel()
                self.done.emit(self._stopping())
                return

            from linac_gen.core.config import SpaceChargeConfig
            from linac_gen.core.step_config import StepConfig
            from linac_gen.core.simulation import Simulation
            from linac_gen.distributions.factory import create_beam

            original_step = getattr(self.lattice, "step_config", StepConfig())
            n = len(self.values)
            # Apply scan-time N override (lets the scan run fast even when
            # BeamConfig.n_particles is large for real MP runs)
            base_n_particles = (
                int(self.scan_n_particles)
                if self.scan_n_particles is not None
                else self.cfg.n_particles
            )
            for i, v in enumerate(self.values):
                if self._stopping():
                    # Leave the live lattice exactly as we found it and
                    # report the partial scan — never terminate() a thread
                    # that is inside numpy/PIC code.
                    self.lattice.step_config = original_step
                    self.done.emit(True)
                    return
                nx = int(self.fixed_nx); extent = float(self.fixed_extent)
                step1 = float(self.fixed_step1); step2 = float(self.fixed_step2)
                n_part = base_n_particles
                if self.axis == AXIS_GRID:
                    nx = int(v)
                elif self.axis == AXIS_EXTENT:
                    extent = float(v)
                elif self.axis == AXIS_STEP1:
                    step1 = float(v)
                elif self.axis == AXIS_STEP2:
                    step2 = float(v)
                elif self.axis == AXIS_NPART:
                    n_part = int(v)

                cfg_i = self.cfg
                if n_part != self.cfg.n_particles:
                    cfg_i = replace(self.cfg, n_particles=int(n_part))

                # Apply step config by mutating the lattice in place
                self.lattice.step_config = StepConfig(
                    integration_steps_per_metre=float(step1),
                    sc_steps_per_metre=float(step2),
                )

                t0 = time.time()
                beam = create_beam(cfg_i, seed=42)
                sc = SpaceChargeConfig(
                    nx=nx, ny=nx, nz=nx, grid_extent=extent,
                    use_gpu=self.use_gpu,
                ) if cfg_i.current > 0 else None
                res = Simulation(self.lattice, beam, space_charge=sc,
                                 should_abort=self._stopping).run()
                elapsed = time.time() - t0
                if self._stopping():
                    # The tracker bailed mid-lattice — this point's numbers
                    # are truncated, not physics; drop them.
                    self.lattice.step_config = original_step
                    self.done.emit(True)
                    return
                self.row_done.emit(i, {
                    "value": v,
                    "sigma_x":   float(res.sigma_x[-1])  if res.sigma_x else 0.0,
                    "sigma_y":   float(res.sigma_y[-1])  if res.sigma_y else 0.0,
                    "sigma_phi": float(res.sigma_phi[-1]) if res.sigma_phi else 0.0,
                    "emit_x":    float(res.emit_x[-1])   if res.emit_x else 0.0,
                    "emit_y":    float(res.emit_y[-1])   if res.emit_y else 0.0,
                    "elapsed":   elapsed,
                })
                self.progress.emit(int((i + 1) * 100 / max(n, 1)))

            # Restore original step_config (whatever axis we were scanning)
            self.lattice.step_config = original_step
            self.done.emit(False)
            return
        except Exception as exc:
            # Best-effort restore
            try:
                self.lattice.step_config = original_step
            except Exception:
                pass
            self.failed.emit(str(exc))

    # ------------------------------------------------------------------
    def _run_parallel(self) -> None:
        """Dispatch every point to a process pool via linac_gen.parallel."""
        from dataclasses import asdict, replace
        from linac_gen.parallel import ScanPoint, run_scan_points

        base_n = (int(self.scan_n_particles)
                  if self.scan_n_particles is not None
                  else self.cfg.n_particles)

        points: list[ScanPoint] = []
        for v in self.values:
            nx = int(self.fixed_nx); extent = float(self.fixed_extent)
            step1 = float(self.fixed_step1); step2 = float(self.fixed_step2)
            n_part = base_n
            if self.axis == AXIS_GRID:    nx = int(v)
            elif self.axis == AXIS_EXTENT: extent = float(v)
            elif self.axis == AXIS_NPART:  n_part = int(v)
            cfg_i = replace(self.cfg, n_particles=int(n_part))
            points.append(ScanPoint(
                lattice_path=self.lattice_path,
                beam_config=asdict(cfg_i),
                nx=nx, grid_extent=extent,
                step1=step1, step2=step2,
                seed=42,
                use_gpu=self.use_gpu,
            ))

        total = len(points)
        done = {"n": 0}

        def on_done(i: int, row: dict) -> None:
            # Proxy emissions through the same signals the serial path uses
            self.row_done.emit(i, {
                "value":    self.values[i],
                "sigma_x":   row["sigma_x"],
                "sigma_y":   row["sigma_y"],
                "sigma_phi": row["sigma_phi"],
                "emit_x":    row["emit_x"],
                "emit_y":    row["emit_y"],
                "elapsed":   row["elapsed"],
            })
            done["n"] += 1
            self.progress.emit(int(100 * done["n"] / max(total, 1)))

        run_scan_points(
            points, on_done=on_done,
            max_workers=max(2, self.parallel_workers),
            should_stop=self._stopping,
        )


# ---------------------------------------------------------------------------
class ConvergenceTab(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._worker: _ScanWorker | None = None
        self._rows: list[dict] = []
        self._converged_idx: int | None = None
        # "Run All" state: queue of remaining axes + per-axis summary
        self._auto_queue: list[str] = []
        self._auto_summary: list[str] = []
        self._auto_mode: bool = False

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # --- Simulation settings panel --------------------------------
        settings = QGroupBox(
            "Simulation settings · used by ‘Run Multi-particle’ and as "
            "the baseline for non-scanned axes"
        )
        settings.setStyleSheet(_group_qss())
        # Group the 21 controls into named collapsible sections instead of
        # one flat 21-row form.  The macOS form-policy overrides that used
        # to live here are now baked into CollapsibleSection's inner form.
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(2)
        self._sections: dict[str, CollapsibleSection] = {}
        # Defaults picked from the MEBT convergence scan: 48³ with 7σ extent
        # matches TraceWin transmission within ~0.7 pp at N=20 k.  Finer grids
        # (96³) are noise-dominated for typical 5 k-particle runs; tighter
        # extent (5 σ) clips the halo and *increases* spurious losses.
        self._fixed_nx    = QSpinBox();       self._fixed_nx.setRange(16, 256);       self._fixed_nx.setValue(48)
        self._fixed_ext   = QDoubleSpinBox(); self._fixed_ext.setRange(1, 20);        self._fixed_ext.setValue(7.0); self._fixed_ext.setSuffix(" σ")
        self._fixed_step1 = QDoubleSpinBox(); self._fixed_step1.setRange(5, 5000);    self._fixed_step1.setValue(100); self._fixed_step1.setSuffix(" /m")
        self._fixed_step2 = QDoubleSpinBox(); self._fixed_step2.setRange(5, 5000);    self._fixed_step2.setValue(50);  self._fixed_step2.setSuffix(" /m")
        # PIC FFT backend.  'auto' picks GPU when cupy is installed and a
        # CUDA device is visible, else CPU.  'gpu' forces GPU and errors if
        # unavailable.  'cpu' disables GPU even if present.
        self._fixed_backend = QComboBox()
        self._fixed_backend.addItems(["auto", "cpu", "gpu", "cuda", "mps"])
        self._fixed_backend.setCurrentText("auto")
        self._fixed_backend.setToolTip(
            "PIC Poisson-solver FFT backend.\n"
            "  auto — cupy/CUDA (float64) if available, else CPU.\n"
            "         Never picks the float32 MPS backend automatically.\n"
            "  cpu  — scipy.fft on the host, even if a GPU is present.\n"
            "  gpu  — any GPU; cupy preferred, MPS fallback (float32,\n"
            "         warns). Errors if none.\n"
            "  cuda — force cupy/CUDA. Errors if cupy unavailable.\n"
            "  mps  — force torch.mps (Apple Silicon Metal GPU, FLOAT32:\n"
            "         ~1e-7 field error, warns). Errors if unavailable."
        )
        # Space-charge compute engine: production numpy/C++ PIC vs the
        # differentiable PyTorch PIC.  Feeds SpaceChargeConfig.sc_backend.
        self._fixed_sc_backend = QComboBox()
        self._fixed_sc_backend.addItems(["numpy", "torch"])
        self._fixed_sc_backend.setCurrentText("numpy")
        self._fixed_sc_backend.setToolTip(
            "Space-charge compute engine (multi-particle runs).\n"
            "  numpy — production numpy/C++ PIC. Default; fastest; handles\n"
            "          bunched and continuous (DC) beams.\n"
            "  torch — differentiable PyTorch PIC. FP64 on CPU, bunched-beam\n"
            "          3-D only, markedly slower. Always sizes its grid\n"
            "          adaptively (the Grid mode control is ignored), then\n"
            "          matches the numpy backend to ~1e-7. Pick it for the\n"
            "          autograd-differentiable kick. A continuous (DC) beam\n"
            "          falls back to numpy automatically."
        )
        # Numerics-advanced toggles.  Defaults match the validated baselines
        # (LEBT-PXIE / MEBT+HWR comparison plots).  Tooltips explain when to
        # flip each one.
        self._fixed_green = QComboBox()
        self._fixed_green.addItems(["igf", "point"])
        self._fixed_green.setCurrentText("igf")
        self._fixed_green.setToolTip(
            "Hockney FFT Green's function.\n"
            "  igf   — Integrated Green Function (Qiang 2006). Recommended.\n"
            "  point — sampled 1/(4πε₀ r). Legacy; back-compat regression only."
        )
        self._fixed_kernel = QComboBox()
        self._fixed_kernel.addItems(["cic", "tsc"])
        self._fixed_kernel.setCurrentText("cic")
        self._fixed_kernel.setToolTip(
            "Particle-mesh deposition / gather kernel.\n"
            "  cic — 8-corner trilinear, 1st-order. Default; fine for linac "
            "stages where RF cavities reset per-step grid noise.\n"
            "  tsc — 27-cell quadratic. Lower grid noise but ~3× more work. "
            "Use for long no-RF transport (e.g. BTL): CIC's per-step "
            "self-force/aliasing accumulates into longitudinal filamentation "
            "without cavity restoring force."
        )
        self._fixed_grid_mode = QComboBox()
        self._fixed_grid_mode.addItems(["fixed", "adaptive"])
        self._fixed_grid_mode.setCurrentText("fixed")
        self._fixed_grid_mode.setToolTip(
            "PIC grid sizing strategy.\n"
            "  fixed    — build grid once at first SC kick from initial σ × "
            "extent. Default; correct when σ stays close to the initial "
            "value (linac stages with cavity refocusing).\n"
            "  adaptive — rebuild grid every SC kick from current σ. Use "
            "for transport lines (e.g. BTL) where σ_z grows several × the "
            "initial value: the fixed grid extent gets outgrown and tail "
            "particles fall off-grid, distorting the field. ~2–3× slower "
            "(rebuilds the FFT solver each kick)."
        )
        self._fixed_dc_kernel = QComboBox()
        self._fixed_dc_kernel.addItems(["uniform", "gaussian", "pic2d"])
        self._fixed_dc_kernel.setCurrentText("uniform")
        self._fixed_dc_kernel.setToolTip(
            "Continuous (DC) beam space-charge model.\n"
            "  uniform  — analytic linear uniform-elliptical-cylinder kick. "
            "Default; matches the rigid-Σ envelope solver, fastest, closest "
            "to TraceWin's exit σ on PXIE-LEBT.\n"
            "  gaussian — Bassetti-Erskine field of a 2-D Gaussian density "
            "(rigid σ, per-particle non-linear).\n"
            "  pic2d    — 2-D Hockney FFT PIC over the actual particle "
            "distribution. Captures full per-particle nonlinearity; "
            "TraceWin PICNIC_2D analogue."
        )
        # Grey out the controls the torch SC engine ignores — it always
        # uses an adaptive grid, runs FP64 on CPU, and has no DC kernels.
        self._fixed_sc_backend.currentTextChanged.connect(
            self._on_sc_engine_changed)
        self._fixed_env_solver = QComboBox()
        self._fixed_env_solver.addItems(["matrix", "sacherer"])
        self._fixed_env_solver.setCurrentText("matrix")
        self._fixed_env_solver.setToolTip(
            "Envelope solver choice (Run Envelope only).\n"
            "  matrix   — element-by-element transfer-matrix Σ propagation "
            "with split-operator SC kicks. Default; tracks accelerating "
            "RF cavities and longitudinal dynamics.\n"
            "  sacherer — coupled KV envelope ODE integrated with RK45. "
            "DC, no acceleration; useful for low-energy transport with "
            "strong SC where the matrix split-operator under-resolves."
        )
        self._fixed_integ = QComboBox()
        self._fixed_integ.addItems(["kd", "dkd"])
        self._fixed_integ.setCurrentText("kd")
        self._fixed_integ.setToolTip(
            "FieldMap3D integrator.\n"
            "  kd  — first-order kick-then-drift. Default.\n"
            "  dkd — symplectic Drift-Kick-Drift; better for long / periodic "
            "trajectories."
        )
        self._fixed_interp = QComboBox()
        self._fixed_interp.addItems(["linear", "cubic"])
        self._fixed_interp.setCurrentText("linear")
        self._fixed_interp.setToolTip(
            "FieldMap3D interpolation order.\n"
            "  linear — trilinear. Default.\n"
            "  cubic  — tricubic; useful for sharp-gradient cavities (takes "
            "effect on next lattice load)."
        )
        # 3-D field-map sampling implementation.  Both paths are BITWISE
        # identical (pinned by tests); the switch exists for verification
        # and debugging, not physics.
        self._fieldmap_sampling = QComboBox()
        self._fieldmap_sampling.addItems(["kernel", "scipy"])
        self._fieldmap_sampling.setToolTip(
            "FieldMap3D sampling implementation (linear interp only).\n"
            "  kernel — fused C++ sampler. Default; measured 1.8× faster\n"
            "           MP+SC and 2.9× faster ENV+SC on the full PIP-II\n"
            "           linac, bit-identical results.\n"
            "  scipy  — legacy per-component RegularGridInterpolator.\n"
            "           Same results, slower; use to cross-check."
        )
        try:
            from linac_gen.elements.field_map_3d import kernel_available
            _kern_ok = kernel_available()
        except Exception:                                  # noqa: BLE001
            _kern_ok = False
        if _kern_ok:
            self._fieldmap_sampling.setCurrentText("kernel")
        else:
            self._fieldmap_sampling.setCurrentText("scipy")
            self._fieldmap_sampling.setEnabled(False)
            self._fieldmap_sampling.setToolTip(
                "Compiled _fieldmap_kernels module not found — using the "
                "scipy sampler.  Build it per linac_gen/csrc/README.md to "
                "enable the faster kernel (results are identical either "
                "way).")
        # Record at every sub-step (finer-than-element sampling).  Essential
        # for seeing transient ε spikes inside long elements (solenoids,
        # long drifts, field maps) — without this the recorder only writes
        # at element exits and long elements look like straight lines.
        from PyQt6.QtWidgets import QCheckBox
        self._record_substeps = QCheckBox("Record per-sub-step")
        self._record_substeps.setChecked(False)
        self._record_substeps.setToolTip(
            "Write a diagnostic record at every integration sub-step, not "
            "just at element exits.  Reveals the fine structure inside long "
            "solenoids and RF cavities (e.g. the paraxial-fringe ε spikes "
            "that show up at each HWR solenoid).  Costs ~50× more rows in "
            "the recorder arrays and a small wall-clock hit; turn on when "
            "you want TraceWin-resolution σ(s) / ε(s) curves."
        )
        # 2-D particle-density grid along s.  Off by default — small
        # memory cost (~1 MB / axis / 1k steps) but pulls in np.histogram
        # at every record site.  Enables the "Density vs s · heatmap"
        # popup on the Results tab.
        self._record_density = QCheckBox("Record particle density (all 6 coords)")
        self._record_density.setChecked(False)
        self._record_density.setToolTip(
            "Histogram every alive particle into a 2-D density grid (axis "
            "vs s) at every diagnostic record point.  Auto-fits the bin "
            "extent to ±1.1·max(|x,y|) on the first record.  Storage is "
            "~1 MB per axis per 1000 s-steps.  Required for the Results "
            "tab's “Density vs s · heatmap” popup."
        )
        # Snapshot every N elements / sub-steps (full 6-D particle dump).
        # 0 = no periodic snapshots.  Snapshots also fire at any
        # ``Marker(snapshot=True)`` regardless of this setting.
        self._snapshot_every_n = QSpinBox()
        self._snapshot_every_n.setRange(0, 1_000_000)
        self._snapshot_every_n.setValue(0)
        self._snapshot_every_n.setToolTip(
            "Take a full particle-array snapshot every N elements (or "
            "every N sub-steps when 'Record per-sub-step' is on).  0 = "
            "off (only Marker(snapshot=True) fires snapshots).  "
            "Storage cost is ~24 bytes × n_particles per snapshot."
        )
        # Snapshot at specific named elements (targeted, cheap — captures
        # only where asked).  Comma-separated element names; the "Add
        # selected" button appends the element currently selected in the
        # Lattice/Results tabs.  The Phase Space plot then offers each
        # captured location in its "location" dropdown.
        from PyQt6.QtWidgets import QHBoxLayout, QWidget as _QW
        self._snapshot_elements = QLineEdit()
        self._snapshot_elements.setPlaceholderText(
            "element names, comma-separated (e.g. QUAD_012, DIAG_3)")
        self._snapshot_elements.setToolTip(
            "Take a full phase-space snapshot at each named element "
            "(exact name match).  View them in the Phase Space plot's "
            "'location' dropdown.  Cheaper than 'every N' — captures only "
            "the elements you list.")
        _add_sel = QPushButton("Add selected")
        _add_sel.setToolTip("Append the element currently selected in the "
                            "Lattice/Results tab.")
        _add_sel.clicked.connect(self._add_selected_snapshot_element)
        self._snapshot_elements_row = _QW()
        _row = QHBoxLayout(self._snapshot_elements_row)
        _row.setContentsMargins(0, 0, 0, 0); _row.setSpacing(4)
        _row.addWidget(self._snapshot_elements, 1)
        _row.addWidget(_add_sel)
        # Density-grid resolution and physical extent (only used when
        # density recording is enabled).
        self._density_bins = QSpinBox()
        self._density_bins.setRange(16, 4096)
        self._density_bins.setValue(200)
        self._density_bins.setToolTip(
            "Number of histogram bins per axis for density-vs-s recording. "
            "Default 200 is fine for most beams; bump to 400+ for sharp "
            "beams with halo (and accept ~4× memory)."
        )
        self._density_extent = QDoubleSpinBox()
        self._density_extent.setRange(0.0, 1000.0)
        self._density_extent.setDecimals(2)
        self._density_extent.setValue(0.0)
        self._density_extent.setSuffix(" mm")
        self._density_extent.setToolTip(
            "Half-width applied to every axis of the density histogram. "
            "0 = auto-fit to ±1.1·max(|coord|) on the first record. "
            "Set explicitly to compare two runs on the same physical "
            "scale."
        )
        # Coherent Synchrotron Radiation in bends.  Multi-particle only —
        # the envelope solver tracks a Σ-matrix and cannot see the bunch
        # profile CSR needs.  Off by default (opt-in physics).
        self._fixed_csr = QCheckBox("CSR in bends (multi-particle only)")
        self._fixed_csr.setChecked(False)
        self._fixed_csr.setToolTip(
            "Apply a 1-D steady-state Coherent Synchrotron Radiation "
            "energy kick (Saldin-Schnizer) inside every Dipole during a "
            "multi-particle run.  CSR drives energy spread and — via the "
            "bend dispersion — transverse emittance growth.  No effect "
            "on envelope runs (CSR needs the actual bunch profile)."
        )
        self._scan_npart  = QSpinBox();       self._scan_npart.setRange(100, 2_000_000)
        self._scan_npart.setValue(5000)       # Match classic by default
        self._scan_npart.setToolTip(
            "N particles used DURING THE SCAN only.  Does not change the "
            "BeamConfig.n_particles used by regular Run Multi-particle.  "
            "Classic convergence dialog hard-codes 5000 for speed — keep "
            "5000 here for apples-to-apples parity."
        )
        # Parallel workers: 0 = serial.  N>1 fans scan points across processes.
        import os as _os
        default_workers = max(2, min(8, (_os.cpu_count() or 4) // 2))
        self._parallel_workers = QSpinBox()
        self._parallel_workers.setRange(0, _os.cpu_count() or 32)
        self._parallel_workers.setValue(default_workers)
        self._parallel_workers.setSuffix(" proc")
        self._parallel_workers.setToolTip(
            "0 = serial (classic behaviour, one scan point at a time).\n"
            "N>1 = run up to N scan points concurrently in worker processes.\n"
            "Results are identical to serial; only wall-clock time changes.\n"
            "(grid/extent/n_particles axes use the pool; step1/step2 stay "
            "serial because they mutate lattice state in-place.)"
        )
        # Seven sections; only "Step density" is open by default (most
        # frequently tweaked, just 2 rows).  Per-user QSettings overrides
        # this on subsequent loads via _restore_section_state below.
        sec_step = CollapsibleSection("Step density")
        # Integration profile preset -- swap step1/step2 between
        # production-accuracy and matching-speed in one click.  Settings:
        #
        #   Production (100/50) -- physics-converged, ~28 s/eval on
        #                          PIP-II HWR.  Use for final validation.
        #   Matching   (30/15)  -- 1/3 step density, ~3x faster.  RK4
        #                          error stays well below the matcher's
        #                          tolerance; drift in matched params
        #                          vs. production typically < 0.5 %.
        #   Custom              -- the user's hand-entered values.
        #
        # Editing step1 or step2 by hand auto-switches the preset to
        # "Custom" so the GUI never silently overwrites a manual setting.
        self._step_preset = QComboBox()
        self._step_preset.addItems([
            "Production (100/50)",
            "Matching (30/15)",
            "Custom",
        ])
        self._step_preset.setCurrentText("Production (100/50)")
        self._step_preset.setToolTip(
            "Production = production-accuracy physics (default).\n"
            "Matching   = ~3x faster per evaluation, RK4 error stays\n"
            "              below the matcher's tolerance.  Use this\n"
            "              for matching iteration, then switch back\n"
            "              to Production for the final validation pass.\n"
            "Custom     = your hand-entered step1 / step2 values."
        )
        self._step_preset.currentTextChanged.connect(self._on_step_preset)
        sec_step.addRow("Integration profile", self._step_preset)
        sec_step.addRow("Base step1 (integration)", self._fixed_step1)
        sec_step.addRow("Base step2 (SC kicks)",    self._fixed_step2)
        # Switching to "Custom" on any manual edit -- otherwise the
        # preset label and the spinbox values would drift apart silently.
        self._fixed_step1.valueChanged.connect(self._on_step_manual_edit)
        self._fixed_step2.valueChanged.connect(self._on_step_manual_edit)
        sec_step.setExpanded(True)
        settings_layout.addWidget(sec_step)
        self._sections["Step density"] = sec_step

        sec_sc = CollapsibleSection("Space charge & PIC")
        sec_sc.addRow("Base PIC grid (nx=ny=nz)", self._fixed_nx)
        sec_sc.addRow("Base grid extent",         self._fixed_ext)
        sec_sc.addRow("PIC backend",              self._fixed_backend)
        sec_sc.addRow("SC engine",                self._fixed_sc_backend)
        sec_sc.addRow("Green's function",         self._fixed_green)
        sec_sc.addRow("Particle-mesh kernel",     self._fixed_kernel)
        sec_sc.addRow("Grid mode",                self._fixed_grid_mode)
        sec_sc.addRow("DC SC kernel",             self._fixed_dc_kernel)
        settings_layout.addWidget(sec_sc)
        self._sections["Space charge & PIC"] = sec_sc

        sec_fm = CollapsibleSection("Field maps (FieldMap3D)")
        sec_fm.addRow("Integrator", self._fixed_integ)
        sec_fm.addRow("Interp",     self._fixed_interp)
        sec_fm.addRow("Sampling",   self._fieldmap_sampling)
        settings_layout.addWidget(sec_fm)
        self._sections["Field maps (FieldMap3D)"] = sec_fm

        sec_env = CollapsibleSection("Envelope solver")
        sec_env.addRow("Solver", self._fixed_env_solver)
        settings_layout.addWidget(sec_env)
        self._sections["Envelope solver"] = sec_env

        sec_coll = CollapsibleSection("Collective effects")
        sec_coll.addRow(self._fixed_csr)
        settings_layout.addWidget(sec_coll)
        self._sections["Collective effects"] = sec_coll

        sec_diag = CollapsibleSection("Diagnostics & recording")
        sec_diag.addRow(self._record_substeps)
        sec_diag.addRow(self._record_density)
        sec_diag.addRow("Snapshot every N",   self._snapshot_every_n)
        sec_diag.addRow("Snapshot at",        self._snapshot_elements_row)
        sec_diag.addRow("Density bins",       self._density_bins)
        sec_diag.addRow("Density extent (±)", self._density_extent)
        settings_layout.addWidget(sec_diag)
        self._sections["Diagnostics & recording"] = sec_diag

        sec_scan = CollapsibleSection("Scan controls")
        sec_scan.addRow("Scan N particles", self._scan_npart)
        sec_scan.addRow("Parallel workers", self._parallel_workers)
        settings_layout.addWidget(sec_scan)
        self._sections["Scan controls"] = sec_scan

        # Restore per-user expanded state (silently keeps defaults on miss)
        # and wire each section's toggle to write its state back.
        self._restore_section_state()
        for sec in self._sections.values():
            sec.toggled_changed.connect(self._persist_section_state)

        v.addWidget(settings)

        # Pick up lattice's current step_config when one loads, so the
        # settings panel starts out consistent with the .dat file.
        state.lattice_changed.connect(self._sync_step_from_lattice)

        # --- Scan controls --------------------------------------------
        controls = QGroupBox("Scan parameters")
        controls.setStyleSheet(_group_qss())
        cf = QFormLayout(controls)
        cf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        cf.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        cf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._axis = QComboBox()
        self._axis.addItems([AXIS_GRID, AXIS_EXTENT, AXIS_STEP1, AXIS_STEP2, AXIS_NPART])
        self._axis.currentTextChanged.connect(self._apply_axis_default)
        cf.addRow("Scan axis", self._axis)

        self._values = QLineEdit("32, 48, 64, 96, 128")
        self._values.setStyleSheet(
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_1};"
            f"border-radius:3px; color:{theme.TEXT_0};"
            f"font-family:{theme.FONT_MONO}; font-size:11px; padding:4px 8px;"
        )
        cf.addRow("Values (comma-sep)", self._values)
        v.addWidget(controls)

        # --- Run row --------------------------------------------------
        row = QHBoxLayout(); row.setSpacing(8)
        self._run_all_btn = QPushButton("  Run All Scans (grid · extent · step1 · step2)")
        self._run_all_btn.setIcon(icon("play", 12, "#00161c"))
        self._run_all_btn.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; border-radius:3px;"
            f"padding:8px 16px; font-weight:600;"
        )
        self._run_all_btn.setToolTip(
            "Scan all four PIC/step parameters in sequence (matches the "
            "classic convergence dialog).  Auto-applies the converged value "
            "for each axis as it finishes."
        )
        self._run_all_btn.clicked.connect(self._start_run_all)
        row.addWidget(self._run_all_btn)

        self._run_btn = QPushButton("  Run Single Axis")
        self._run_btn.setIcon(icon("play", 12))
        self._run_btn.setToolTip(
            "Scan just the selected axis with the Values list above."
        )
        self._run_btn.clicked.connect(self._start_scan)
        row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("  Stop")
        self._stop_btn.setIcon(icon("stop", 12))
        self._stop_btn.clicked.connect(self._stop_scan)
        self._stop_btn.setEnabled(False)   # nothing to stop yet
        self._stop_btn.setToolTip(
            "Stop the running scan at the next element boundary "
            "(keeps the rows already completed)."
        )
        row.addWidget(self._stop_btn)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(240)
        self._progress.setRange(0, 100); self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background:{theme.BG_INSET}; border:1px solid {theme.BORDER_1};"
            f" border-radius:3px; height:10px; }}"
            f"QProgressBar::chunk {{ background:{theme.ACCENT}; }}"
        )
        row.addWidget(self._progress)
        row.addStretch(1)

        self._badge = QLabel("")
        self._badge.setStyleSheet(
            f"color:{theme.ACCENT}; font-family:{theme.FONT_MONO}; font-size:10px;"
            f"padding:4px 8px; border:1px solid {theme.ACCENT_DIM};"
            f"background:rgba(34,211,238,0.08); border-radius:3px;"
        )
        row.addWidget(self._badge)

        self._apply_btn = QPushButton("  Apply converged value")
        self._apply_btn.setIcon(icon("check", 12))
        self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip(
            "Write the converged row's scan value back into the base settings "
            "used by Run Multi-particle."
        )
        self._apply_btn.clicked.connect(self._apply_converged)
        row.addWidget(self._apply_btn)
        v.addLayout(row)

        # --- Tabbed output: Log | Plot | Table -----------------------
        out_tabs = QTabWidget()
        out_tabs.setStyleSheet(_output_tabs_qss())
        # Explicit no-elide so even if the dock is narrow Qt scrolls
        # the tab bar instead of truncating short labels to "p" / "t".
        out_tabs.setElideMode(Qt.TextElideMode.ElideNone)
        out_tabs.setUsesScrollButtons(True)

        # Log pane — formatted like the classic ConvergenceDialog's QTextEdit
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"QTextEdit {{ background:{theme.BG_INSET};"
            f" color:{theme.TEXT_1};"
            f" font-family:{theme.FONT_MONO}; font-size:11px;"
            f" border:1px solid {theme.BORDER_0}; border-radius:3px;"
            f" padding:8px 10px; }}"
        )
        # "Log (classic format)" was the original full label; the
        # "(classic format)" suffix was 14 chars of redundant info
        # that crowded the tab bar -- the tooltip carries it now.
        out_tabs.addTab(self._log, "Log")
        out_tabs.setTabToolTip(
            out_tabs.count() - 1,
            "Per-axis convergence log in the classic ConvergenceDialog "
            "format.",
        )

        from linac_gen_gui.interphase.plots.plot_style import (
            style_plot, add_legend, curve_pen, CURVE_WIDTH_BOLD,
        )
        self._plot = pg.PlotWidget()
        style_plot(self._plot, "ε_x end", "mm·mrad",
                   xlabel="scan value", xunits="")
        add_legend(self._plot)
        self._curve_ex = self._plot.plot(
            pen=curve_pen(theme.ACCENT, width=CURVE_WIDTH_BOLD),
            symbol="o", symbolSize=9, symbolBrush=theme.ACCENT,
            name="ε_x",
        )
        self._curve_sy = self._plot.plot(
            pen=pg.mkPen("#a3e635", width=2.0, style=Qt.PenStyle.DashLine),
            symbol="s", symbolSize=7, symbolBrush="#a3e635",
            name="σ_y",
        )
        out_tabs.addTab(self._plot, "Plot")

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["value", "σ_x [mm]", "σ_y [mm]", "σ_φ [deg]",
             "ε_x [mm·mrad]", "ε_y [mm·mrad]", "t [s]"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        out_tabs.addTab(self._table, "Table")

        v.addWidget(out_tabs, stretch=1)
        # Track per-axis recommendations for the final summary
        self._auto_recs: dict[str, object] = {}

        # ---- project-dirty wiring -------------------------------------
        # Every "fixed" setting (the values that get serialised into the
        # .lgproj JSON) marks the project dirty when the user edits it,
        # so the close/open-project prompt can warn before discarding.
        # Section toggles and scan-axis widgets are session-only and
        # intentionally excluded.
        def _mark_dirty(*_args) -> None:
            try:
                self.state.mark_project_dirty()
            except Exception:    # noqa: BLE001
                pass
        for _w in (self._fixed_nx, self._fixed_ext,
                   self._fixed_step1, self._fixed_step2):
            _w.valueChanged.connect(_mark_dirty)
        for _w in (self._fixed_backend, self._fixed_sc_backend,
                   self._fixed_green, self._fixed_kernel,
                   self._fixed_grid_mode, self._fixed_dc_kernel,
                   self._fixed_env_solver, self._fixed_integ,
                   self._fixed_interp, self._step_preset,
                   self._fieldmap_sampling):
            _w.currentTextChanged.connect(_mark_dirty)
        # Apply the sampling choice IMMEDIATELY on change (and on project
        # load, which fires the same signal).  This is the single source
        # of application, so every run type — MP, envelope, convergence
        # scans, error studies — inherits the choice; spawned workers get
        # it via the env var.  Bound method, not a lambda (house rule).
        self._fieldmap_sampling.currentTextChanged.connect(
            self._apply_fieldmap_sampling)
        # Integrator / interp likewise apply on change (through the env-
        # mirroring helper), so error studies launched without a prior
        # MP/envelope run still use the selected numerics in their
        # spawned workers.
        self._fixed_integ.currentTextChanged.connect(
            self._apply_fieldmap_numerics)
        self._fixed_interp.currentTextChanged.connect(
            self._apply_fieldmap_numerics)
        # Sync the process-global to the combo's initial state (covers a
        # process started with LINAC_GEN_FIELDMAP_KERNEL=0 while the combo
        # constructs to "kernel" — no change signal fires in that case).
        self._apply_fieldmap_sampling()
        self._fixed_csr.toggled.connect(_mark_dirty)
        # snapshot controls are serialized into the project too
        self._snapshot_every_n.valueChanged.connect(_mark_dirty)
        self._snapshot_elements.textChanged.connect(_mark_dirty)

    # ------------------------------------------------------------------
    def _on_step_preset(self, text: str) -> None:
        """Integration-profile preset → step1 / step2 spinbox values.

        Block signals on the spinboxes so this slot doesn't fire
        ``_on_step_manual_edit`` (which would flip the preset back to
        Custom and undo our change).
        """
        if text.startswith("Production"):
            new_step1, new_step2 = 100.0, 50.0
        elif text.startswith("Matching"):
            new_step1, new_step2 = 30.0, 15.0
        else:   # Custom — leave whatever the user has
            return
        for sp, val in ((self._fixed_step1, new_step1),
                        (self._fixed_step2, new_step2)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

    def _on_step_manual_edit(self, *_args) -> None:
        """Any hand-edit of step1 / step2 means "Custom" -- flip the
        preset combobox so the label matches what's actually loaded.

        Idempotent if we're already on Custom.
        """
        if self._step_preset.currentText() != "Custom":
            self._step_preset.blockSignals(True)
            self._step_preset.setCurrentText("Custom")
            self._step_preset.blockSignals(False)

    def _on_sc_engine_changed(self, backend: str) -> None:
        """The torch SC engine ignores grid mode, the FFT device backend
        and the DC kernel — grey them out so the panel shows which
        settings apply to which engine.  All three controls live in the
        'Space charge & PIC' collapsible section, so the visual greying
        is only visible when that section is expanded."""
        is_torch = (backend == "torch")
        if is_torch:
            # Remember the user's grid mode so switching back to numpy restores
            # it, instead of silently leaving numpy stuck on "adaptive" (a
            # different, ~2-3× slower physics path than the "fixed" they chose).
            if self._fixed_grid_mode.currentText() != "adaptive":
                self._saved_grid_mode = self._fixed_grid_mode.currentText()
            self._fixed_grid_mode.setCurrentText("adaptive")
        else:
            saved = getattr(self, "_saved_grid_mode", None)
            if saved is not None:
                self._fixed_grid_mode.setCurrentText(saved)
                # One-shot restore: forget the saved mode so a LATER explicit
                # "adaptive" choice isn't silently overridden by a stale value
                # on the next torch → numpy round-trip.
                self._saved_grid_mode = None
        for w in (self._fixed_grid_mode, self._fixed_backend,
                  self._fixed_dc_kernel):
            w.setEnabled(not is_torch)

    # ------------------------------------------------------------------
    def _apply_fieldmap_sampling(self, text: str | None = None) -> None:
        """Push the Sampling combo state into the process-global switch
        and the env var (spawned workers re-import and read the env).

        Connected to the combo's currentTextChanged, so it fires on user
        edits AND on project load — every run type (MP, envelope,
        convergence scans, error studies) then inherits the choice."""
        try:
            import os
            from linac_gen.elements import field_map_3d
            enabled = ((text or self._fieldmap_sampling.currentText())
                       == "kernel")
            field_map_3d.use_fused_kernel(enabled)
            os.environ["LINAC_GEN_FIELDMAP_KERNEL"] = "1" if enabled else "0"
        except RuntimeError:
            pass                                    # widget destroyed

    def _add_selected_snapshot_element(self) -> None:
        """Append the currently-selected element's name to the snapshot
        field (deduplicated)."""
        try:
            name = getattr(getattr(self.state, "selected", None), "name", None)
            if not name:
                return
            cur = [t.strip() for t in self._snapshot_elements.text().split(",")
                   if t.strip()]
            if name not in cur:
                cur.append(name)
                self._snapshot_elements.setText(", ".join(cur))
        except RuntimeError:
            pass                                    # widget destroyed

    def snapshot_element_names(self) -> set:
        """Parse the 'Snapshot at' field into a set of element names."""
        try:
            return {t.strip() for t in
                    self._snapshot_elements.text().split(",") if t.strip()}
        except RuntimeError:
            return set()

    def _apply_fieldmap_numerics(self, _text: str | None = None) -> None:
        try:
            from linac_gen.elements import field_map_3d
            field_map_3d.set_fieldmap_numerics(
                integrator=self._fixed_integ.currentText(),
                interp=self._fixed_interp.currentText())
        except RuntimeError:
            pass                                    # widget destroyed

    def current_sc_config(self, current: float, *, continuous: bool = False):
        """Build the SpaceChargeConfig these settings describe.

        Single source of truth for every run path — the toolbar MP run
        and the Errors-tab Monte Carlo studies must use the SAME space-
        charge configuration the user set up here.  Returns ``None`` for
        a zero-current beam (SC off).

        ``continuous`` marks a DC beam: the torch PIC is a bunched-beam
        3-D solver with no continuous kernels, so a torch selection falls
        back to numpy rather than silently applying the wrong SC model.
        """
        if current <= 0:
            return None
        from linac_gen.core.config import SpaceChargeConfig
        nx = self._fixed_nx.value()
        grid_mode = self._fixed_grid_mode.currentText()
        sc_backend = self._fixed_sc_backend.currentText()
        if sc_backend == "torch" and continuous:
            sc_backend = "numpy"
        if sc_backend == "torch":
            # The torch PIC sizes its grid adaptively every kick — it has
            # no fixed-grid path — so label the config honestly.
            grid_mode = "adaptive"
        return SpaceChargeConfig(
            nx=nx, ny=nx, nz=nx,
            grid_extent=self._fixed_ext.value(),
            grid_mode=grid_mode,
            use_gpu=self._fixed_backend.currentText(),
            green_kind=self._fixed_green.currentText(),
            kernel=self._fixed_kernel.currentText(),
            dc_kernel=self._fixed_dc_kernel.currentText(),
            sc_backend=sc_backend,
            csr_enabled=bool(self._fixed_csr.isChecked()),
        )

    # ------------------------------------------------------------------
    def _restore_section_state(self) -> None:
        """Restore each section's expanded state from per-user QSettings.
        UI-only state, kept independent of the .lgproj project file.
        Silently falls back to the default expanded set on any read
        failure."""
        try:
            s = make_settings("HELIX", "linac_gen_gui")
            for title, sec in self._sections.items():
                key = f"convergence/section/{title}"
                if s.contains(key):
                    sec.setExpanded(bool(s.value(key, type=bool)))
        except Exception:
            pass

    def _persist_section_state(self, title: str, expanded: bool) -> None:
        """Write a section's expanded state to per-user QSettings."""
        try:
            s = make_settings("HELIX", "linac_gen_gui")
            s.setValue(f"convergence/section/{title}", bool(expanded))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _sync_step_from_lattice(self, lattice) -> None:
        if lattice is None:
            return
        sc = getattr(lattice, "step_config", None)
        if sc is None:
            return
        s1 = float(sc.integration_steps_per_metre)
        s2 = float(sc.sc_steps_per_metre)
        # Block manual-edit signal so the spinbox updates don't kick
        # the preset to "Custom" before we've had a chance to detect it.
        for sp, val in ((self._fixed_step1, s1), (self._fixed_step2, s2)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        # Detect which preset (if any) the project matches.
        self._step_preset.blockSignals(True)
        if (s1, s2) == (100.0, 50.0):
            self._step_preset.setCurrentText("Production (100/50)")
        elif (s1, s2) == (30.0, 15.0):
            self._step_preset.setCurrentText("Matching (30/15)")
        else:
            self._step_preset.setCurrentText("Custom")
        self._step_preset.blockSignals(False)

    def _apply_axis_default(self, axis: str) -> None:
        defaults = {
            AXIS_GRID:   "32, 48, 64, 96, 128",
            AXIS_EXTENT: "3, 4, 5, 6",
            AXIS_STEP1:  "50, 100, 200, 500",
            AXIS_STEP2:  "25, 50, 100, 200",
            AXIS_NPART:  "10000, 30000, 100000, 300000",
        }
        self._values.setText(defaults.get(axis, ""))

    def _parse_values(self) -> list:
        axis = self._axis.currentText()
        raw = [s.strip() for s in self._values.text().split(",") if s.strip()]
        try:
            if axis in (AXIS_EXTENT, AXIS_STEP1, AXIS_STEP2):
                return [float(s) for s in raw]
            return [int(s) for s in raw]
        except Exception as exc:
            raise ValueError(f"could not parse values: {exc}")

    def _start_scan(self) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first."); return
        if self.state.beam_config is None:
            QMessageBox.warning(self, "No beam", "Apply a beam config first."); return
        try:
            values = self._parse_values()
            if not values:
                raise ValueError("no values to scan")
        except ValueError as exc:
            QMessageBox.critical(self, "Scan", str(exc)); return

        # A worker from the previous scan may still be unwinding after a
        # stop request: detach its signals so late emissions can't write
        # into the new table, and give it a moment to exit — replacing
        # the only Python reference to a live QThread aborts the process.
        prev = self._worker
        if prev is not None:
            for sig in (prev.progress, prev.row_done, prev.done, prev.failed):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
            if prev.isRunning():
                prev.request_stop()
                prev.requestInterruption()
                if not prev.wait(2000):
                    # Still unwinding — we are about to rebind
                    # self._worker, so park the straggler instead of
                    # dropping the last reference to a live QThread.
                    _park_zombie(prev)

        self._rows = [None] * len(values)
        self._table.setRowCount(len(values))
        for r, v in enumerate(values):
            self._table.setItem(r, 0, QTableWidgetItem(str(v)))
            for c in range(1, 7):
                self._table.setItem(r, c, QTableWidgetItem("—"))
        self._curve_ex.setData([], []); self._curve_sy.setData([], [])
        self._progress.setValue(0)
        self._badge.setText("running…")
        self._apply_btn.setEnabled(False)

        self._worker = _ScanWorker(
            self.state.lattice, self.state.beam_config,
            self._axis.currentText(), values,
            fixed_nx=self._fixed_nx.value(),
            fixed_extent=self._fixed_ext.value(),
            fixed_step1=self._fixed_step1.value(),
            fixed_step2=self._fixed_step2.value(),
            scan_n_particles=self._scan_npart.value(),
            parallel_workers=self._parallel_workers.value(),
            lattice_path=self.state.lattice_path,
            use_gpu=self._fixed_backend.currentText(),
        )
        self._worker.progress.connect(self._progress.setValue)
        self._worker.row_done.connect(self._on_row)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._run_btn.setEnabled(False)
        self._run_all_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        # If this is a single-axis run (not part of "Run All"), write a fresh header
        if not self._auto_mode:
            self._log.clear()
            self._log_header()
            self._log.append(
                f"Running sweep  {self._axis.currentText()} ∈ {values} ...\n"
            )
        self._worker.start()

    def _start_run_all(self) -> None:
        """Run all four classic axes in sequence using their default value lists."""
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first."); return
        if self.state.beam_config is None:
            QMessageBox.warning(self, "No beam", "Apply a beam config first."); return
        self._auto_mode = True
        self._auto_queue = [AXIS_GRID, AXIS_EXTENT, AXIS_STEP1, AXIS_STEP2]
        self._auto_summary = []
        self._auto_recs = {}
        self._badge.setText("running all scans…")
        self._run_all_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        # Classic-format header in the log
        self._log.clear()
        self._log_header()
        self._log.append(
            "Running sweeps "
            "(grid_ext ∈ {3,4,5,6} σ, "
            "n_grid ∈ {32,48,64,96,128} cells, "
            "step1 ∈ {50,100,200,500}/m, "
            "step2 ∈ {25,50,100,200}/m) ...\n"
        )
        self._kick_next_auto()

    def _kick_next_auto(self) -> None:
        """Pop the next axis off the auto queue and start its scan."""
        if not self._auto_queue:
            self._auto_mode = False
            self._run_all_btn.setEnabled(True)
            self._run_btn.setEnabled(True)
            summary = "  ·  ".join(self._auto_summary) if self._auto_summary else "no data"
            # Write all collected recommendations to baselines at once,
            # now that every sweep has finished.
            self._apply_all_auto_recs()
            self._badge.setText(
                f"all scans done  —  {summary}   · recommended values applied"
            )
            return
        next_axis = self._auto_queue.pop(0)
        self._axis.setCurrentText(next_axis)
        self._apply_axis_default(next_axis)
        self._start_scan()

    def _apply_all_auto_recs(self) -> None:
        """End-of-RunAll commit: write every stashed recommendation at once.

        Mirrors the classic 'Apply Recommended' button behaviour — the
        user sees their scan results reflect a fixed baseline, then the
        whole set of converged values is applied in one go.
        """
        from dataclasses import replace as dc_replace
        recs = self._auto_recs
        if "n_grid" in recs:
            self._fixed_nx.setValue(int(recs["n_grid"]))
        if "grid_ext" in recs:
            self._fixed_ext.setValue(float(recs["grid_ext"]))
        step_changed = False
        if "step1" in recs:
            self._fixed_step1.setValue(float(recs["step1"])); step_changed = True
        if "step2" in recs:
            self._fixed_step2.setValue(float(recs["step2"])); step_changed = True
        if step_changed:
            self._write_step_to_lattice()
        if "n_part" in recs and self.state.beam_config is not None:
            self.state.set_beam_config(
                dc_replace(self.state.beam_config, n_particles=int(recs["n_part"]))
            )
        if recs:
            applied = ", ".join(f"{k}={v}" for k, v in recs.items())
            self.state.status_message.emit(f"Applied recommended values: {applied}")

    def shutdown_begin(self) -> list:
        """App teardown: signal the scan worker (which also tears down
        the process pool via should_stop) and return it for the
        window's bounded wait."""
        w = self._worker
        if w is not None and w.isRunning():
            w.request_stop()
            w.requestInterruption()
            return [w]
        return []

    def _stop_scan(self) -> None:
        """Cooperative stop — the worker exits at the next element/point
        boundary and reports through done(stopped=True).

        Never terminate(): killing a thread inside numpy/PIC code is
        undefined behaviour, and the old immediate button re-enable let a
        second scan start while the first still mutated the live lattice.
        """
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.requestInterruption()
            self._stop_btn.setEnabled(False)
            self._badge.setText("stopping — finishing the current step…")

    def _on_row(self, i: int, d: dict) -> None:
        self._rows[i] = d
        cells = [f"{d['value']}", f"{d['sigma_x']:.3f}", f"{d['sigma_y']:.3f}",
                 f"{d['sigma_phi']:.3f}", f"{d['emit_x']:.4f}",
                 f"{d['emit_y']:.4f}", f"{d['elapsed']:.1f}"]
        for c, txt in enumerate(cells):
            item = QTableWidgetItem(txt)
            if c > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(i, c, item)
        xs = [r["value"] for r in self._rows if r]
        ys_e = [r["emit_x"] for r in self._rows if r]
        ys_s = [r["sigma_y"] for r in self._rows if r]
        self._curve_ex.setData(xs, ys_e)
        self._curve_sy.setData(xs, ys_s)

        # Classic-format per-row log line
        knob = self._classic_knob_name(self._axis.currentText())
        self._log.append(
            f"  {knob:10s}={float(d['value']):>6.1f}   "
            f"sigma_x={d['sigma_x']:6.3f}  sigma_y={d['sigma_y']:6.3f}  "
            f"emit_x={d['emit_x']:7.3f}  emit_y={d['emit_y']:7.3f}   "
            f"t={d['elapsed']:5.1f}s"
        )

    def _on_done(self, stopped: bool = False) -> None:
        self._stop_btn.setEnabled(False)
        if stopped:
            # User cancel: keep the partial rows on screen, but no
            # recommendation from truncated data and no auto-launch of
            # the next Run-All axis.
            self._auto_mode = False
            self._auto_queue = []
            self._run_btn.setEnabled(True)
            self._run_all_btn.setEnabled(True)
            n_done = sum(1 for r in self._rows if r)
            self._badge.setText(
                f"stopped — {n_done}/{len(self._rows)} points completed")
            self._log.append(f"\n[stopped by user after {n_done} points]")
            return
        if not self._auto_mode:
            self._run_btn.setEnabled(True)
        valid = [(i, r) for i, r in enumerate(self._rows) if r]
        self._converged_idx = None
        axis_short = self._axis.currentText().split(" ")[0]

        # Classic-parity recommendation rule:  smallest value whose ε_x is
        # within TOL_EMIT of the finest row's ε_x.  Always succeeds when
        # at least one row ran — the finest matches itself by definition.
        if valid:
            # Order by the swept resolution value so this is correct no matter
            # what order the user typed the (free-text) values in.  The finest
            # row is the LARGEST value (most grid points / highest step
            # density), and "smallest converged" must scan from coarsest up.
            # Previously ``valid[-1]`` assumed ascending entry, so a
            # "128, 64, 32" entry took the COARSEST row as the reference and
            # could apply an under-resolved grid to the baseline.
            ordered = sorted(valid, key=lambda ir: ir[1]["value"])
            ref_emit = ordered[-1][1]["emit_x"]
            chosen = None
            for (i, r) in ordered:
                if abs(r["emit_x"] - ref_emit) / max(abs(ref_emit), 1e-30) <= TOL_EMIT:
                    chosen = (i, r); break
            idx, row = chosen if chosen is not None else ordered[-1]
            self._converged_idx = idx
            self._apply_btn.setEnabled(True)
            self._highlight_row(idx)

            if len(ordered) >= 2:
                last_rel = abs(ordered[-1][1]["emit_x"] - ordered[-2][1]["emit_x"]) \
                           / max(abs(ordered[-1][1]["emit_x"]), 1e-30)
                tail = f" · finest drift {last_rel*100:.2f}%"
            else:
                tail = ""
            self._badge.setText(
                f"converged at {axis_short}={row['value']}{tail}"
            )
            # Record recommendation keyed by classic knob name
            knob = self._classic_knob_name(self._axis.currentText())
            self._auto_recs[knob] = row["value"]

            if self._auto_mode:
                # IMPORTANT: do NOT apply in-between axes during Run-All.
                # Applying here would bump the baseline and contaminate the
                # remaining sweeps (e.g. n_grid→128 would make the extent
                # sweep run at nx=128 instead of the original baseline).
                # The classic dialog keeps the baseline frozen across all
                # four sweeps; we match that here and apply everything
                # only when the whole run finishes.
                self._auto_summary.append(f"{axis_short}={row['value']}")
        else:
            self._badge.setText("done — no rows")
            if self._auto_mode:
                self._auto_summary.append(f"{axis_short}=skipped")

        # If this was the final single-axis (or last auto step), write the
        # classic-format Recommendation footer.
        if (not self._auto_mode) or (self._auto_mode and not self._auto_queue):
            self._log_recommendation_footer()

        if self._auto_mode:
            self._kick_next_auto()

    def _on_fail(self, msg: str) -> None:
        self._stop_btn.setEnabled(False)
        self._run_btn.setEnabled(True)
        self._run_all_btn.setEnabled(True)
        self._auto_mode = False
        self._auto_queue = []
        self._badge.setText(f"failed: {msg}")

    # ------------------------------------------------------------------
    def _highlight_row(self, idx: int) -> None:
        from PyQt6.QtGui import QBrush, QColor
        accent = QColor(34, 211, 238, 40)
        for c in range(self._table.columnCount()):
            item = self._table.item(idx, c)
            if item is not None:
                item.setBackground(QBrush(accent))

    def _apply_converged(self) -> None:
        if self._converged_idx is None:
            return
        row = self._rows[self._converged_idx]
        axis = self._axis.currentText()
        v = row["value"]
        if axis == AXIS_GRID:
            self._fixed_nx.setValue(int(v))
            self.state.status_message.emit(f"Applied grid = {int(v)}")
        elif axis == AXIS_EXTENT:
            self._fixed_ext.setValue(float(v))
            self.state.status_message.emit(f"Applied grid_extent = {float(v):g} σ")
        elif axis == AXIS_STEP1:
            self._fixed_step1.setValue(float(v))
            self._write_step_to_lattice()
            self.state.status_message.emit(f"Applied step1 = {float(v):g} /m")
        elif axis == AXIS_STEP2:
            self._fixed_step2.setValue(float(v))
            self._write_step_to_lattice()
            self.state.status_message.emit(f"Applied step2 = {float(v):g} /m")
        elif axis == AXIS_NPART:
            cfg = self.state.beam_config
            if cfg is not None:
                from dataclasses import replace as dc_replace
                self.state.set_beam_config(dc_replace(cfg, n_particles=int(v)))
                self.state.status_message.emit(f"Applied N particles = {int(v)}")
        self._badge.setText(self._badge.text() + "   — applied ✓")

    # --- Classic-format log helpers -----------------------------------
    _CLASSIC_KNOB_NAMES = {
        AXIS_GRID:   "n_grid",
        AXIS_EXTENT: "grid_ext",
        AXIS_STEP1:  "step1",
        AXIS_STEP2:  "step2",
        AXIS_NPART:  "n_part",
    }

    def _classic_knob_name(self, axis_ui_label: str) -> str:
        return self._CLASSIC_KNOB_NAMES.get(axis_ui_label, axis_ui_label)

    def _log_header(self) -> None:
        """Write the three-line preamble matching the classic dialog."""
        cfg = self.state.beam_config
        lat = self.state.lattice
        self._log.append(
            f"Current SC config: nx={self._fixed_nx.value()}  "
            f"grid_extent={self._fixed_ext.value()}  grid_mode=fixed\n"
        )
        if cfg is not None:
            self._log.append(
                f"Beam: {cfg.species}  W={cfg.energy:g} MeV  "
                f"I={cfg.current:g} mA  dist={cfg.distribution}  "
                f"N_scan={self._scan_npart.value()}  "
                f"(BeamConfig N={cfg.n_particles})\n"
            )
        if lat is not None:
            sc = getattr(lat, "step_config", None)
            if sc is not None:
                self._log.append(
                    f"Current step_config: step1 = "
                    f"{sc.integration_steps_per_metre} /m,  step2 = "
                    f"{sc.sc_steps_per_metre} /m\n"
                )

    def _log_recommendation_footer(self) -> None:
        """Write the 'Recommendation (1 % emit_x convergence):' block that
        the classic dialog prints at the end of the sweeps."""
        self._log.append("")
        self._log.append("Recommendation (1 % emit_x convergence):")
        labels = [
            ("grid_ext", "grid_extent", " sigma"),
            ("n_grid",   "n_grid     ", " cells (nx=ny=nz)"),
            ("step1",    "step1      ", " /m  (integration)"),
            ("step2",    "step2      ", " /m  (SC kick)"),
            ("n_part",   "n_part     ", " particles"),
        ]
        for knob, label, suffix in labels:
            if knob not in self._auto_recs:
                continue
            v = self._auto_recs[knob]
            # Integer formatting for count-ish knobs, float otherwise
            if knob in ("n_grid", "n_part"):
                vs = f"{int(v)}"
            else:
                vs = f"{float(v):g}"
            self._log.append(f"  {label} = {vs}{suffix}")
        self._log.append(
            "  Click 'Apply converged value' (per-axis) to write into the "
            "base settings above."
        )

    def _write_step_to_lattice(self) -> None:
        """Persist step1/step2 into lattice.step_config so every subsequent
        MP run picks them up (the tracker reads from lattice.step_config)."""
        from linac_gen.core.step_config import StepConfig
        if self.state.lattice is None:
            return
        self.state.lattice.step_config = StepConfig(
            integration_steps_per_metre=float(self._fixed_step1.value()),
            sc_steps_per_metre=float(self._fixed_step2.value()),
        )
