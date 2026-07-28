"""Parameter-scan popup.

User picks a lattice element, one of its scalar parameters, a scan range
and number of points, plus a simulation mode (envelope or multi-particle)
and an output quantity to plot.  A QThread worker runs the scan point by
point, patching the parameter, running a fresh simulation, recording the
final-state value of the chosen quantity, then restoring the element's
original value at the end (even on abort / failure).

The dialog is deliberately non-modal so the user can keep interacting
with the main GUI while a scan is running.
"""
from __future__ import annotations

import copy
import math
import traceback
from dataclasses import asdict, fields
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel, QProgressBar,
    QMessageBox, QRadioButton, QButtonGroup, QGroupBox,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.plots.plot_style import style_plot


# ---------------------------------------------------------------------------
# Output quantities — mapping label → ``(attr_on_results, reducer, unit)``.
# Reducer is applied to the array to get one scalar per scan point.
# ---------------------------------------------------------------------------
_OUTPUT_CHOICES: list[tuple[str, str, str, str]] = [
    # label, results-attribute, reducer, unit
    ("σ_x at exit",          "sigma_x",         "last",  "mm"),
    ("σ_y at exit",          "sigma_y",         "last",  "mm"),
    ("σ_φ at exit",          "sigma_phi",       "last",  "deg"),
    ("σ_W at exit (keV)",    "sigma_w",         "last_x1000", "keV"),
    ("ε_nx at exit",         "emit_nx",         "last",  "mm·mrad"),
    ("ε_ny at exit",         "emit_ny",         "last",  "mm·mrad"),
    ("ε_nz at exit",         "emit_nz",         "last",  "mm·mrad"),
    # 4-D coupling-invariant transverse emittance -- this is the actual
    # constraint metric used by MIN_EMIT_4D_GROWTH (the HWR matching
    # standard).  Worth scanning directly rather than as ε_nx, ε_ny
    # which can trade off under x-y coupling without ε_4D moving.
    ("ε_4D at exit",         "emit_4d",         "last",  "mm·mrad"),
    # Eigen-emittances (Balandin normal-mode emittances): invariants of
    # any symplectic transport.  ε_e1, ε_e2 are the two transverse
    # eigenmodes; under x-y or transverse-longitudinal coupling they
    # stay constant while ε_nx / ε_ny / ε_nz oscillate.  Useful for
    # diagnosing whether emittance growth is real or just projection.
    ("ε_e1 at exit",         "emit_e1",         "last",  "mm·mrad"),
    ("ε_e2 at exit",         "emit_e2",         "last",  "mm·mrad"),
    ("Transmission at exit", "transmission",    "last",  "%"),
    ("W_ref at exit",        "ref_w_kin",       "last",  "MeV"),
    # Twiss at exit -- useful for matching a section into a downstream
    # lattice (e.g. HWR exit α, β should hand off cleanly to SSR1).
    ("α_x at exit",          "alpha_x",         "last",  ""),
    ("β_x at exit",          "beta_x",          "last",  "mm/mrad"),
    ("α_y at exit",          "alpha_y",         "last",  ""),
    ("β_y at exit",          "beta_y",          "last",  "mm/mrad"),
    ("σ_x maximum along s",  "sigma_x",         "max",   "mm"),
    ("σ_y maximum along s",  "sigma_y",         "max",   "mm"),
    # 4-D ε growth: useful for the same reason MIN_EMIT_4D_GROWTH exists
    # as a constraint -- the matcher's own objective metric.
    ("ε_4D growth (exit / entry)",  "emit_4d",  "ratio_end_start", ""),
    ("ε_nx growth (exit / entry)",  "emit_nx",  "ratio_end_start", ""),
    ("ε_ny growth (exit / entry)",  "emit_ny",  "ratio_end_start", ""),
    ("ε_nz growth (exit / entry)",  "emit_nz",  "ratio_end_start", ""),
    # Halo metrics -- MP-only because envelope tracks RMS Σ, not the
    # 4σ tail population.  EnvelopeResults leaves these as None and
    # the plot will be empty in env mode (no silent synth -- env
    # genuinely cannot estimate halo).
    ("Halo_x at exit (MP only)",    "halo_x",   "last",  ""),
    ("Halo_y at exit (MP only)",    "halo_y",   "last",  ""),
]


def _reduce(values, reducer: str) -> Optional[float]:
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=float)
    if reducer == "last":
        return float(arr[-1])
    if reducer == "last_x1000":
        return float(arr[-1]) * 1000.0
    if reducer == "max":
        return float(np.max(arr))
    if reducer == "ratio_end_start":
        start = float(arr[0])
        if abs(start) < 1e-30:
            return None
        return float(arr[-1] / start)
    return None


# ---------------------------------------------------------------------------
# Discovering scannable parameters on a given element.
#
# We enumerate every instance attribute that's a plain ``float`` or ``int``,
# minus a hand-curated blacklist of things that shouldn't be user-scanned
# (parser bookkeeping, unit flags, counters).  Works across all element
# types without hard-coding per-class tables.
# ---------------------------------------------------------------------------
_PARAM_BLACKLIST = frozenset({
    "n_steps", "p_flag", "aperture", "ka", "geometry",
    "_step_idx", "_z_map_start", "_z_map_end", "_phi_s_at_entrance",
    "_sync_offset_deg",
})


def _scannable_params(element) -> list[str]:
    names = []
    for attr in sorted(vars(element).keys()):
        if attr.startswith("_") and attr not in ("_sync_offset_deg",):
            continue
        if attr in _PARAM_BLACKLIST:
            continue
        val = getattr(element, attr)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            names.append(attr)
    return names


# ---------------------------------------------------------------------------
# Scan worker — runs N simulations in a QThread, patching the chosen
# parameter in place at each step.  Always restores the original value
# in ``finally``.
# ---------------------------------------------------------------------------
class _ScanWorker(QThread):
    point_done = pyqtSignal(int, float, object)  # (index, value, scalar|None)
    finished_ok = pyqtSignal(object, object)     # (values_array, scalars_array)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, state, element, param_name: str,
                 values: np.ndarray,
                 mode: str,
                 out_attr: str, out_reducer: str,
                 convergence: Optional[dict] = None):
        super().__init__()
        # Snapshot lattice + beam + the scanned element so concurrent edits in
        # the (deliberately non-modal) parent dialog can't race the worker
        # thread — mirrors Error / Failure study, which deep-copy for the same
        # reason.  Every mutation below hits the COPY, never the live objects,
        # so an undo / reorder / delete mid-scan can't corrupt state or crash
        # the sim with "list changed size during iteration".
        live_lat = getattr(state, "lattice", None)
        self._lattice = copy.deepcopy(live_lat) if live_lat is not None else None
        self._beam = copy.deepcopy(getattr(state, "beam_config", None))
        if live_lat is not None and self._lattice is not None:
            try:
                _idx = next(i for i, e in enumerate(live_lat.elements)
                            if e is element)
                self._element = self._lattice.elements[_idx]
            except StopIteration:
                # Identity lookup failed (shouldn't happen — worker is built
                # in the same event-handler turn the element was fetched).
                # Fall back to a NAME match on the copy rather than mixing a
                # live element into a copied lattice (which would scan the
                # live object and produce a flat curve).
                self._element = next(
                    (e for e in self._lattice.elements
                     if getattr(e, "name", None) == getattr(element, "name",
                                                            None)),
                    element)
        else:
            self._element = element
        self.param = param_name
        self.values = np.asarray(values, dtype=float)
        self.mode = mode     # "env" or "mp"
        self.out_attr = out_attr
        self.out_reducer = out_reducer
        # Snapshot of the convergence-tab dropdowns (env_solver, green_kind,
        # kernel, dc_kernel, integrator_kind, interp_kind, nx, grid_extent,
        # step1, step2, backend).  Empty dict ⇒ fall back to library defaults.
        self.convergence = dict(convergence or {})
        self._abort = False

    def request_stop(self):
        self._abort = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        original_value = getattr(self._element, self.param)
        scalars = np.full(self.values.shape, np.nan, dtype=float)
        # Apply integrator/interp class-level toggles once for the whole scan
        # so every point uses the user-selected numerics.
        try:
            from linac_gen.elements import field_map_3d as _fm3
            _fm3.set_fieldmap_numerics(
                integrator=self.convergence.get("integrator_kind") or None,
                interp=self.convergence.get("interp_kind") or None)
        except Exception:
            pass
        # Apply step_config to the live lattice for the MP path.
        try:
            from linac_gen.core.step_config import StepConfig
            s1 = self.convergence.get("step1")
            s2 = self.convergence.get("step2")
            if s1 is not None and s2 is not None and self._lattice is not None:
                self._lattice.step_config = StepConfig(
                    integration_steps_per_metre=float(s1),
                    sc_steps_per_metre=float(s2),
                )
        except Exception:
            pass
        try:
            for i, v in enumerate(self.values):
                if self._abort:
                    break
                setattr(self._element, self.param, float(v))
                scalar = self._run_one()
                if scalar is None or not np.isfinite(scalar):
                    scalars[i] = np.nan
                else:
                    scalars[i] = scalar
                self.point_done.emit(i, float(v), scalars[i])
                self.progress.emit(int(round(100.0 * (i + 1) / max(len(self.values), 1))))
            self.finished_ok.emit(self.values.copy(), scalars.copy())
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
        finally:
            # ALWAYS restore the original value — even on abort / exception —
            # otherwise the user's lattice is silently modified.
            setattr(self._element, self.param, original_value)

    # ------------------------------------------------------------------
    def _run_one(self) -> Optional[float]:
        lat = self._lattice
        cfg = self._beam
        if lat is None or cfg is None:
            raise RuntimeError("Load a lattice and configure the beam first.")
        if self.mode == "env":
            return self._run_envelope(lat, cfg)
        return self._run_mp(lat, cfg)

    def _run_envelope(self, lat, cfg) -> Optional[float]:
        import math as _m
        from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
        from linac_gen.core.reference import ReferenceParticle

        species_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
        sp = species_map.get(cfg.species, PROTON)
        bg = _m.sqrt(max((1 + cfg.energy / sp.mass) ** 2 - 1, 0.0))
        ref = ReferenceParticle(species=sp, w_kin=cfg.energy,
                                frequency=cfg.frequency)
        # Single source of truth for the envelope seed (incl. the
        # first-moment centroid and DC fields) — a hand-rolled copy here
        # silently scanned an on-axis orbit for offset beams.
        from linac_gen.cli.common import _envelope_initial
        initial = _envelope_initial(cfg, ref)
        env_solver = str(self.convergence.get("env_solver", "matrix")).lower()
        if env_solver == "sacherer":
            from linac_gen.tracking.sacherer import SachererSolver
            solver = SachererSolver(lat, ref, initial, current=cfg.current)
        else:
            from linac_gen.tracking.envelope import EnvelopeSolver
            solver = EnvelopeSolver(lat, ref, initial, current=cfg.current)
        res = solver.run()
        # EnvelopeResults carries a PER-ROW ref_frequency list (the ENV
        # writer's FREQ-jump-safe centroid-z depends on it) — never
        # clobber it with a scalar; only backfill when the solver left
        # it empty (SachererSolver builds its own results).
        if not getattr(res, "ref_frequency", None):
            res.ref_frequency = [cfg.frequency] * len(res.s)
        # EnvelopeResults stores GEOMETRIC emit_x/y/z; the parameter-scan
        # output choices in _OUTPUT_CHOICES (and DiagnosticRecorder's
        # convention from MP mode) reference NORMALISED ``emit_nx/ny/nz``
        # which the EnvelopeSolver does not produce.  Without this
        # translation, env-mode scans on emittance metrics get None ->
        # NaN -> empty plot.  Derive ``emit_n*`` here from the geometric
        # arrays scaled by ``ref_beta * ref_gamma`` per step.
        try:
            if hasattr(res, "emit_x") and not hasattr(res, "emit_nx"):
                betas = getattr(res, "ref_beta", None) or []
                gammas = getattr(res, "ref_gamma", None) or []
                if betas and gammas and len(betas) == len(res.emit_x):
                    bg = [b * g for b, g in zip(betas, gammas)]
                    res.emit_nx = [ex * f for ex, f in zip(res.emit_x, bg)]
                    res.emit_ny = [ey * f for ey, f in zip(res.emit_y, bg)]
                    res.emit_nz = [ez * f for ez, f in zip(res.emit_z, bg)]
        except Exception:                                  # noqa: BLE001
            pass
        # Envelope tracks RMS Σ-matrices, not individual particles, so
        # it has no notion of transmission -- ``getattr(res,
        # "transmission", None)`` returns None and the scan plot stays
        # empty when the user picks "Transmission at exit".  Synthesize
        # a constant 100% trace so the curve shows the physically-true
        # result for envelope (lossless by construction) and the plot
        # populates.  Users who need real transmission must switch to
        # multi-particle mode (which records n_alive / n_particles).
        try:
            n = None
            if hasattr(res, "emit_x"):
                n = len(res.emit_x)
            elif hasattr(res, "ref_w_kin"):
                n = len(res.ref_w_kin)
            if n and not hasattr(res, "transmission"):
                res.transmission = [100.0] * n
        except Exception:                                  # noqa: BLE001
            pass
        return _reduce(getattr(res, self.out_attr, None), self.out_reducer)

    def _run_mp(self, lat, cfg) -> Optional[float]:
        from linac_gen.core.config import SpaceChargeConfig
        from linac_gen.core.simulation import Simulation
        from linac_gen.distributions.factory import create_beam

        beam = create_beam(cfg, seed=42)
        nx = int(self.convergence.get("nx", 48))
        ext = float(self.convergence.get("grid_extent", 5.0))
        backend = str(self.convergence.get("backend", "auto"))
        green_kind = str(self.convergence.get("green_kind", "igf"))
        kernel = str(self.convergence.get("kernel", "cic"))
        dc_kernel = str(self.convergence.get("dc_kernel", "uniform"))
        sc = (SpaceChargeConfig(nx=nx, ny=nx, nz=nx, grid_extent=ext,
                                use_gpu=backend,
                                green_kind=green_kind, kernel=kernel,
                                dc_kernel=dc_kernel)
              if cfg.current > 0 else None)
        sim = Simulation(lat, beam, space_charge=sc, record_substeps=False)
        res = sim.run()
        # Same rule as the envelope branch: the recorder's ref_frequency
        # is a per-row list — backfill only if absent, never clobber.
        if not getattr(res, "ref_frequency", None):
            res.ref_frequency = [cfg.frequency] * len(getattr(res, "s", []))
        return _reduce(getattr(res, self.out_attr, None), self.out_reducer)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------
class ParameterScanDialog(QDialog):
    """Non-modal element-parameter scan popup."""

    def __init__(self, parent, state):
        super().__init__(parent)
        self.setWindowTitle("Parameter Scan")
        from linac_gen_gui.interphase.scrollwrap import screen_capped
        self.setMinimumSize(*screen_capped(self, 860, 620))
        self.setModal(False)
        self._state = state
        self._worker: Optional[_ScanWorker] = None

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)

        # ---- Top row: element + parameter + range + mode + output ----
        sel = QGroupBox("Scan definition")
        sel.setStyleSheet(
            f"QGroupBox {{ color:{theme.TEXT_2}; border:1px solid {theme.BORDER_0};"
            f" border-radius:4px; margin-top:12px; padding-top:6px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 6px;"
            f" color:{theme.TEXT_2}; font-size:10px; letter-spacing:1px;"
            f" text-transform:uppercase; background:{theme.BG_0}; }}"
        )
        form = QFormLayout(sel)

        self._element_combo = QComboBox()
        self._element_combo.currentIndexChanged.connect(self._on_element_changed)
        form.addRow("Element:", self._element_combo)

        self._param_combo = QComboBox()
        self._param_combo.currentIndexChanged.connect(self._on_param_changed)
        form.addRow("Parameter:", self._param_combo)

        # Range controls
        rng_row = QHBoxLayout(); rng_row.setSpacing(6)
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setDecimals(6); self._min_spin.setRange(-1e9, 1e9)
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setDecimals(6); self._max_spin.setRange(-1e9, 1e9)
        self._npts_spin = QSpinBox(); self._npts_spin.setRange(2, 200); self._npts_spin.setValue(11)
        rng_row.addWidget(QLabel("from")); rng_row.addWidget(self._min_spin)
        rng_row.addWidget(QLabel("to"));   rng_row.addWidget(self._max_spin)
        rng_row.addSpacing(10)
        rng_row.addWidget(QLabel("points:")); rng_row.addWidget(self._npts_spin)
        rng_row.addStretch(1)
        form.addRow("Range:", self._wrap(rng_row))

        # Mode radio
        mode_row = QHBoxLayout(); mode_row.setSpacing(8)
        self._mode_group = QButtonGroup(self)
        self._mode_env = QRadioButton("Envelope")
        self._mode_mp  = QRadioButton("Multi-particle")
        self._mode_env.setChecked(True)
        self._mode_group.addButton(self._mode_env); self._mode_group.addButton(self._mode_mp)
        mode_row.addWidget(self._mode_env); mode_row.addWidget(self._mode_mp)
        mode_row.addStretch(1)
        form.addRow("Mode:", self._wrap(mode_row))

        # Output quantity
        self._out_combo = QComboBox()
        for label, *_ in _OUTPUT_CHOICES:
            self._out_combo.addItem(label)
        form.addRow("Plot quantity:", self._out_combo)

        v.addWidget(sel)

        # ---- Controls + progress -------------------------------------
        controls = QHBoxLayout(); controls.setSpacing(8)
        self._run_btn = QPushButton("  Run scan")
        self._run_btn.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; border-radius:3px;"
            f"padding:6px 14px; font-weight:600;"
        )
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            f"QPushButton {{ background:{theme.ERR}; color:#ffffff; border:0;"
            f" border-radius:3px; padding:6px 14px; font-weight:600; }} "
            f"QPushButton:disabled {{ background:{theme.BG_2};"
            f" color:{theme.TEXT_2}; border:1px solid {theme.BORDER_1}; }}"
        )
        self._progress = QProgressBar(); self._progress.setRange(0, 100)
        self._progress.setTextVisible(True)
        self._status = QLabel(""); self._status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;"
        )
        controls.addWidget(self._run_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._progress, stretch=1)
        controls.addWidget(self._status)
        v.addLayout(controls)

        self._run_btn.clicked.connect(self._run_scan)
        self._stop_btn.clicked.connect(self._stop_scan)

        # ---- Plot ----------------------------------------------------
        self._plot = pg.PlotWidget()
        style_plot(self._plot, "quantity", "")
        self._curve = self._plot.plot(
            [], [], pen=pg.mkPen(theme.ACCENT, width=2),
            symbol="o", symbolSize=6,
            symbolPen=pg.mkPen(theme.ACCENT), symbolBrush=pg.mkBrush(theme.ACCENT),
        )
        v.addWidget(self._plot, stretch=1)

        # Populate element list now (if a lattice is loaded), and refresh
        # whenever a new one is loaded.
        self._rebuild_elements()
        # BOUND METHOD, never a lambda: this dialog is destroyable
        # (deleteLater on reopen), and PyQt only auto-disconnects a slot
        # when it can identify the receiver QObject — a lambda has none,
        # so its stale connection fires on the DEAD dialog at the next
        # lattice load and raises "wrapped C/C++ object ... deleted".
        # House rule: bound method + RuntimeError guard (see
        # _on_lattice_changed).
        try:
            state.lattice_changed.connect(self._on_lattice_changed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(layout) -> "QWidget":
        from PyQt6.QtWidgets import QWidget
        w = QWidget(); w.setLayout(layout); return w

    # ------------------------------------------------------------------
    def _snapshot_convergence(self) -> dict:
        """Read the parent MainWindow's convergence-tab dropdowns.

        Returns an empty dict if the parent doesn't expose a
        ``convergence_tab`` (e.g. unit-test harness); the worker then
        falls back to library defaults (igf / cic / uniform / matrix /
        kd / linear).
        """
        ct = getattr(self.parent(), "convergence_tab", None)
        if ct is None:
            return {}
        try:
            return {
                "env_solver":      str(ct._fixed_env_solver.currentText()),
                "green_kind":      str(ct._fixed_green.currentText()),
                "kernel":          str(ct._fixed_kernel.currentText()),
                "grid_mode":       str(ct._fixed_grid_mode.currentText()),
                "dc_kernel":       str(ct._fixed_dc_kernel.currentText()),
                "integrator_kind": str(ct._fixed_integ.currentText()),
                "interp_kind":     str(ct._fixed_interp.currentText()),
                "backend":         str(ct._fixed_backend.currentText()),
                "nx":              int(ct._fixed_nx.value()),
                "grid_extent":     float(ct._fixed_ext.value()),
                "step1":           float(ct._fixed_step1.value()),
                "step2":           float(ct._fixed_step2.value()),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def _on_lattice_changed(self, _lat=None) -> None:
        """Slot for the app-wide ``lattice_changed`` signal.

        Guarded because a connection can briefly outlive this dialog's C++
        widgets during teardown; on a dead object, self-disconnect so a
        lingering connection stops firing (belt-and-suspenders on top of
        PyQt's bound-method auto-disconnect)."""
        try:
            self._rebuild_elements()
        except RuntimeError:
            try:
                self._state.lattice_changed.disconnect(
                    self._on_lattice_changed)
            except (TypeError, RuntimeError):
                pass

    def _rebuild_elements(self) -> None:
        self._element_combo.blockSignals(True)
        self._element_combo.clear()
        lat = getattr(self._state, "lattice", None)
        if lat is None:
            self._element_combo.setEnabled(False)
            self._param_combo.setEnabled(False)
            self._status.setText("load a lattice first")
        else:
            self._element_combo.setEnabled(True)
            self._param_combo.setEnabled(True)
            for i, el in enumerate(lat.elements):
                name = getattr(el, "name", f"el{i}")
                self._element_combo.addItem(
                    f"{i:3d}  {type(el).__name__:14s}  {name}", userData=i
                )
            self._status.setText(f"{len(lat.elements)} elements loaded")
        self._element_combo.blockSignals(False)
        self._on_element_changed()

    def _current_element(self):
        lat = getattr(self._state, "lattice", None)
        if lat is None:
            return None
        idx = self._element_combo.currentIndex()
        if idx < 0:
            return None
        return lat.elements[idx]

    def _on_element_changed(self) -> None:
        el = self._current_element()
        self._param_combo.blockSignals(True)
        self._param_combo.clear()
        if el is None:
            self._param_combo.setEnabled(False)
            self._param_combo.blockSignals(False)
            return
        self._param_combo.setEnabled(True)
        for p in _scannable_params(el):
            v = getattr(el, p)
            self._param_combo.addItem(f"{p}   (= {v:g})", userData=p)
        self._param_combo.blockSignals(False)
        self._on_param_changed()

    def _on_param_changed(self) -> None:
        el = self._current_element()
        if el is None or self._param_combo.currentIndex() < 0:
            return
        p = self._param_combo.currentData()
        v = float(getattr(el, p))
        # Sensible default range: ±50 % around current, or ±1 if zero.
        if abs(v) < 1e-9:
            lo, hi = -1.0, 1.0
        else:
            lo, hi = sorted([v * 0.5, v * 1.5]) if v > 0 else sorted([v * 1.5, v * 0.5])
        self._min_spin.setValue(float(lo))
        self._max_spin.setValue(float(hi))

    # ------------------------------------------------------------------
    def _run_scan(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        el = self._current_element()
        if el is None:
            QMessageBox.warning(self, "Parameter Scan", "No element selected."); return
        if self._state.beam_config is None:
            QMessageBox.warning(self, "Parameter Scan",
                                "Apply a beam configuration first (Beam tab).")
            return
        param = self._param_combo.currentData()
        if not param:
            QMessageBox.warning(self, "Parameter Scan", "No parameter selected."); return
        n = self._npts_spin.value()
        lo = self._min_spin.value(); hi = self._max_spin.value()
        if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
            QMessageBox.warning(self, "Parameter Scan",
                                "Invalid range — need lo < hi with non-zero span."); return
        values = np.linspace(min(lo, hi), max(lo, hi), n)

        out_idx = self._out_combo.currentIndex()
        label, attr, reducer, unit = _OUTPUT_CHOICES[out_idx]
        mode = "env" if self._mode_env.isChecked() else "mp"

        # Reset plot + progress
        self._curve.setData([], [])
        self._plot.setLabel("left", f"{label}", units=unit)
        self._plot.setLabel("bottom",
                            f"{type(el).__name__}.{param}",
                            units="")
        self._progress.setValue(0)
        self._status.setText(
            f"running · {n} × {mode} over {param} ∈ [{lo:g}, {hi:g}]"
        )
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        # Snapshot the convergence-tab dropdowns from the parent MainWindow
        # so each scan point uses the same numerics as the main Run button
        # (env_solver, green_kind, kernel, dc_kernel, integrator/interp,
        # grid sizing, step density).  Empty dict ⇒ library defaults.
        convergence = self._snapshot_convergence()

        self._worker = _ScanWorker(
            self._state, el, param, values, mode, attr, reducer,
            convergence=convergence,
        )
        self._worker.progress.connect(self._progress.setValue)
        self._worker.point_done.connect(self._on_point_done)
        self._worker.finished_ok.connect(self._on_scan_done)
        self._worker.failed.connect(self._on_scan_failed)
        # QThread.finished fires AFTER run() returns -- i.e. after the
        # finally: in _ScanWorker.run() has restored original_value.
        # Use it to refresh the element inspector so the user sees the
        # restored value rather than the cached mid-scan value.
        self._worker.finished.connect(self._refresh_after_scan)
        self._worker.start()

        # Incremental plot data — populate as points come in.
        self._xs: list[float] = []
        self._ys: list[float] = []

    def _on_point_done(self, i: int, x: float, y) -> None:
        self._xs.append(float(x))
        if y is None or (isinstance(y, float) and not math.isfinite(y)):
            self._ys.append(math.nan)
        else:
            self._ys.append(float(y))
        xs = np.asarray(self._xs, dtype=float)
        ys = np.asarray(self._ys, dtype=float)
        mask = np.isfinite(ys)
        xs_finite = xs[mask]
        ys_finite = ys[mask]
        self._curve.setData(xs_finite, ys_finite)
        # Force the plot to re-fit its view to the actual data range.
        # Without this, a previous scan's view box can persist (e.g.
        # if the prior run left a wide x-range and the new scan is in
        # a narrow region, the new curve appears as a tiny dot at the
        # edge or off-screen).  We also handle the degenerate "all
        # y values are identical" case (where pyqtgraph's autoRange
        # collapses the y axis to zero height) by padding the y range.
        if xs_finite.size > 0:
            try:
                vb = self._plot.getViewBox()
                vb.enableAutoRange(axis='xy', enable=True)
                # Guard against a flat horizontal line by giving the y
                # axis at least 10% padding if y is identical.
                if ys_finite.size >= 2 and float(ys_finite.max()) == float(ys_finite.min()):
                    yval = float(ys_finite[0])
                    pad = max(abs(yval) * 0.1, 1e-6)
                    self._plot.setYRange(yval - pad, yval + pad, padding=0)
            except Exception:                                  # noqa: BLE001
                pass
        # Also surface the latest value in the status line so the user
        # can see SOMETHING moving even if the plot range is hard to read.
        self._status.setText(
            f"point {len(self._xs)}/{int(self._npts_spin.value())}  "
            f"x={float(x):.6g}  y={'NaN' if math.isnan(self._ys[-1]) else f'{self._ys[-1]:.6g}'}"
        )

    def _on_scan_done(self, xs, ys) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)
        xs = np.asarray(xs); ys = np.asarray(ys)
        n_ok = int(np.sum(np.isfinite(ys)))
        self._status.setText(f"done · {n_ok}/{xs.size} points")

    def _on_scan_failed(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText("failed")
        QMessageBox.critical(self, "Parameter Scan failed", msg)

    def _refresh_after_scan(self) -> None:
        """Re-render the element inspector once the worker has fully
        returned (including its finally:, which restored the original
        parameter value).  Without this the inspector keeps showing
        the last mid-scan value because its spinboxes cache values at
        construction time.  Re-emitting selected_element_changed forces
        the inspector to rebuild from the now-restored attribute."""
        try:
            sel = getattr(self._state, "selected", None)
            if sel is not None:
                self._state.set_selected(sel)
        except Exception:    # noqa: BLE001
            pass

    def _stop_scan(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._status.setText("stopping…")

    # ------------------------------------------------------------------
    def closeEvent(self, ev) -> None:
        # Make sure a scan in progress releases the element's original
        # value before we leave the dialog.
        self._stop_scan()
        if self._worker is not None:
            self._worker.wait(5000)
        super().closeEvent(ev)
