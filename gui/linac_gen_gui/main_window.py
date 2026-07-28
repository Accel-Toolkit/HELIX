"""Main application window."""
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QMenuBar, QToolBar,
    QStatusBar, QFileDialog, QMessageBox, QWidget, QVBoxLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linac_Gen - Particle Tracking")
        self.setMinimumSize(1200, 800)

        self._lattice = None
        self._beam = None
        self._results = None
        self._worker = None

        self._create_menus()
        self._create_toolbar()
        self._create_panels()
        self._create_status_bar()

    def _create_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open Lattice...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_lattice)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Lattice...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_lattice)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")
        run_env_action = QAction("Run &Envelope", self)
        run_env_action.triggered.connect(self._run_envelope)
        sim_menu.addAction(run_env_action)

        run_mp_action = QAction("Run &Multi-particle", self)
        run_mp_action.triggered.connect(self._run_multiparticle)
        sim_menu.addAction(run_mp_action)

        sim_menu.addSeparator()
        match_action = QAction("&Match Beam...", self)
        match_action.triggered.connect(self._open_matching)
        sim_menu.addAction(match_action)

        convergence_action = QAction("Check SC &Convergence...", self)
        convergence_action.triggered.connect(self._open_convergence_check)
        sim_menu.addAction(convergence_action)

        tmatrix_action = QAction("Show &Transfer Matrix...", self)
        tmatrix_action.triggered.connect(self._open_transfer_matrix)
        sim_menu.addAction(tmatrix_action)

        sigma_action = QAction("Show &Sigma Matrix...", self)
        sigma_action.triggered.connect(self._open_sigma_matrix)
        sim_menu.addAction(sigma_action)

        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self._open_sim_settings)
        sim_menu.addAction(settings_action)

    def _create_toolbar(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)
        toolbar.addAction("Open").triggered.connect(self._open_lattice)
        toolbar.addAction("Save").triggered.connect(self._save_lattice)
        toolbar.addSeparator()
        toolbar.addAction("Run Env").triggered.connect(self._run_envelope)
        toolbar.addAction("Run MP").triggered.connect(self._run_multiparticle)
        toolbar.addSeparator()
        toolbar.addAction("Match").triggered.connect(self._open_matching)

    def _create_panels(self):
        # Left: Lattice editor (dock)
        from linac_gen_gui.widgets.lattice_editor import LatticeEditorWidget
        self._lattice_dock = QDockWidget("Lattice", self)
        self._lattice_editor = LatticeEditorWidget()
        self._lattice_dock.setWidget(self._lattice_editor)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._lattice_dock)

        # Right: Beam config (dock)
        from linac_gen_gui.widgets.beam_config import BeamConfigWidget
        self._beam_dock = QDockWidget("Beam Setup", self)
        self._beam_config = BeamConfigWidget()
        self._beam_dock.setWidget(self._beam_config)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._beam_dock)

        # Center: Tab widget for plots
        from linac_gen_gui.widgets.envelope_plot import EnvelopePlotWidget
        from linac_gen_gui.widgets.phase_space_plot import PhaseSpacePlotWidget
        from linac_gen_gui.widgets.loss_map_plot import LossMapPlotWidget
        from linac_gen_gui.widgets.emittance_plot import EmittancePlotWidget

        self._plot_tabs = QTabWidget()
        self._envelope_plot = EnvelopePlotWidget()
        self._phase_space_plot = PhaseSpacePlotWidget()
        self._loss_map_plot = LossMapPlotWidget()
        self._emittance_plot = EmittancePlotWidget()

        self._plot_tabs.addTab(self._envelope_plot, "Envelope")
        self._plot_tabs.addTab(self._phase_space_plot, "Phase Space")
        self._plot_tabs.addTab(self._loss_map_plot, "Loss Map")
        self._plot_tabs.addTab(self._emittance_plot, "Emittance")
        self.setCentralWidget(self._plot_tabs)

        # Bottom dock: Lattice layout strip
        from linac_gen_gui.widgets.lattice_layout import LatticeLayoutWidget
        self._layout_dock = QDockWidget("Lattice Layout", self)
        self._lattice_layout = LatticeLayoutWidget()
        self._layout_dock.setWidget(self._lattice_layout)
        self._layout_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._layout_dock)

        # Space charge config (default, user can change via Settings dialog)
        from linac_gen.core.config import SpaceChargeConfig
        self._sc_config = SpaceChargeConfig()

    def _create_status_bar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    def _open_lattice(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Lattice", "", "TraceWin Files (*.dat);;All Files (*)")
        if filepath:
            try:
                from linac_gen.io.tracewin_parser import parse_tracewin
                self._lattice, metadata = parse_tracewin(filepath)
                self._lattice_editor.set_lattice(self._lattice)
                self._lattice_layout.set_lattice(self._lattice)
                warnings = metadata.get("warnings", [])
                if warnings:
                    self._status.showMessage(f"Loaded with {len(warnings)} warnings")
                else:
                    self._status.showMessage(f"Loaded {len(self._lattice)} elements")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _save_lattice(self):
        if not self._lattice:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Lattice", "", "TraceWin Files (*.dat)")
        if filepath:
            from linac_gen.io.tracewin_writer import write_tracewin
            write_tracewin(self._lattice, filepath)
            self._status.showMessage(f"Saved to {filepath}")

    def _open_matching(self):
        if self._lattice is None:
            QMessageBox.warning(self, "No Lattice", "Please load a lattice first.")
            return
        from linac_gen_gui.dialogs.matching_dialog import MatchingDialog
        dlg = MatchingDialog(self._lattice, self._beam_config, parent=self)
        dlg.exec()

    def _open_sim_settings(self):
        from linac_gen_gui.dialogs.simulation_settings import SimulationSettingsDialog
        from linac_gen.elements.field_map_3d import FieldMap3D
        dlg = SimulationSettingsDialog(
            self,
            current=self._sc_config,
            integrator_kind=getattr(FieldMap3D, "integrator_kind", "kd"),
            interp_kind=getattr(FieldMap3D, "interp_kind", "linear"),
        )
        if dlg.exec():
            self._sc_config = dlg.get_sc_config()
            # FieldMap3D class-level attrs apply to the whole next run.
            new_int = dlg.get_integrator_kind()
            new_interp = dlg.get_interp_kind()
            interp_changed = new_interp != getattr(FieldMap3D, "interp_kind", "linear")
            from linac_gen.elements import field_map_3d as _fm3
            _fm3.set_fieldmap_numerics(integrator=new_int, interp=new_interp)
            cfg = self._sc_config
            msg = (
                f"Settings: grid {cfg.nx}×{cfg.ny}×{cfg.nz} {cfg.grid_extent}σ, "
                f"green={cfg.green_kind}, kernel={cfg.kernel}, "
                f"FM3D={new_int}/{new_interp}"
            )
            # Cubic interp builds a per-element coefficient table at __init__,
            # so the change only takes effect for lattices loaded *after* this.
            if interp_changed:
                msg += "  (reload lattice to apply new interp)"
            self._status.showMessage(msg)

    def _open_sigma_matrix(self):
        """Show the beam sigma matrix at every recorded step of the last run."""
        if self._results is None:
            QMessageBox.warning(self, "No Results",
                                "Run a simulation first (envelope or multi-particle).")
            return
        from linac_gen_gui.dialogs.sigma_matrix_dialog import SigmaMatrixDialog
        beam_config = self._beam_config.get_beam_config()
        dlg = SigmaMatrixDialog(parent=self, results=self._results,
                                beam_config=beam_config)
        dlg.exec()

    def _open_transfer_matrix(self):
        """Show the dialog that computes the linear transfer matrix between two
        user-selected element indices (pure linear transport, no space charge).
        """
        if self._lattice is None or not self._lattice.elements:
            QMessageBox.warning(self, "No Lattice",
                                "Please load a lattice first.")
            return
        from linac_gen_gui.dialogs.transfer_matrix_dialog import TransferMatrixDialog
        beam_config = self._beam_config.get_beam_config()
        dlg = TransferMatrixDialog(parent=self, lattice=self._lattice,
                                   beam_config=beam_config)
        dlg.exec()

    def _open_convergence_check(self):
        """Run a short SC convergence sweep on the current lattice/beam."""
        if self._lattice is None or not self._lattice.elements:
            QMessageBox.warning(self, "No Lattice",
                                "Please load a lattice first.")
            return
        from linac_gen_gui.dialogs.convergence_dialog import ConvergenceDialog
        beam_config = self._beam_config.get_beam_config()
        dlg = ConvergenceDialog(
            parent=self, lattice=self._lattice,
            beam_config=beam_config, sc_config=self._sc_config,
        )
        if dlg.exec():
            rec = dlg.recommended_sc()
            rec_step = dlg.recommended_step()
            if rec is not None:
                self._sc_config = rec
            if rec_step is not None and self._lattice is not None:
                self._lattice.step_config = rec_step
            if rec is not None or rec_step is not None:
                parts = []
                if rec is not None:
                    parts.append(f"nx={rec.nx}, grid_extent={rec.grid_extent}")
                if rec_step is not None:
                    parts.append(
                        f"step1={rec_step.integration_steps_per_metre:g}, "
                        f"step2={rec_step.sc_steps_per_metre:g}"
                    )
                self._status.showMessage("SC + step config updated: " + "; ".join(parts))

    def _build_simulation(self):
        """Build a Simulation from the current lattice and beam config."""
        if self._lattice is None:
            QMessageBox.warning(self, "No Lattice", "Please load a lattice first.")
            return None
        try:
            from linac_gen.distributions.factory import create_beam
            from linac_gen.core.simulation import Simulation
            beam_config = self._beam_config.get_beam_config()
            beam = create_beam(beam_config)
            sim = Simulation(self._lattice, beam, space_charge=self._sc_config)
            # Attach envelope params from beam config for run_envelope()
            # Convert normalized emittance to geometric: emit_geo = emit_n / (beta*gamma)
            bg = beam.ref.bg
            from linac_gen.distributions.factory import geometric_emittances
            _ex, _ey, _ez = geometric_emittances(beam_config, bg)
            sim.beam_envelope_params = {
                "alpha_x": beam_config.alpha_x,
                "beta_x": beam_config.beta_x,
                "emit_x": _ex,   # normalized -> geometric, mismatch applied
                "alpha_y": beam_config.alpha_y,
                "beta_y": beam_config.beta_y,
                "emit_y": _ey,
                "alpha_z": beam_config.alpha_z,
                "beta_z": beam_config.beta_z,
                "emit_z": _ez,
                # DC / continuous-beam metadata — EnvelopeSolver._sc_matrix
                # checks ``_continuous`` to switch from the 3-D Ferrario kick
                # to the 2-D DC analytic.  Missing this flag was the silent
                # failure mode behind the LEBT env+SC mismatch (commit
                # c400f5f); without it Ferrario early-returns identity for
                # emit_z=0 → ENV+SC silently equals ENV+NOSC.
                "continuous": bool(getattr(beam_config, "continuous", False)),
                "dc_energy_spread_keV":
                    float(getattr(beam_config, "dc_energy_spread_keV", 0.0)),
                # Input dispersion (mm/MeV, mrad/MeV) — EnvelopeSolver's
                # _build_sigma_matrix folds these into the Σ[i,5] cross
                # terms; the matched beam of a bending line needs them.
                # All-zero defaults reproduce the previous Σ exactly.
                "disp_x": float(getattr(beam_config, "disp_x", 0.0)),
                "disp_xp": float(getattr(beam_config, "disp_xp", 0.0)),
                "disp_y": float(getattr(beam_config, "disp_y", 0.0)),
                "disp_yp": float(getattr(beam_config, "disp_yp", 0.0)),
            }
            return sim
        except Exception as e:
            QMessageBox.critical(self, "Setup Error", str(e))
            return None

    def _run_envelope(self):
        sim = self._build_simulation()
        if sim is None:
            return
        self._start_worker(sim, mode="envelope")

    def _run_multiparticle(self):
        sim = self._build_simulation()
        if sim is None:
            return
        self._start_worker(sim, mode="multiparticle")

    def _start_worker(self, simulation, mode):
        from linac_gen_gui.workers import SimulationWorker
        if self._worker is not None and self._worker.isRunning():
            self._status.showMessage("Simulation already running")
            return
        self._worker = SimulationWorker(simulation, mode=mode)
        self._worker.finished.connect(self._on_simulation_finished)
        self._worker.error.connect(self._on_simulation_error)
        self._status.showMessage(f"Running {mode} simulation...")
        self._worker.start()

    def _on_simulation_finished(self, results):
        self._results = results

        # Update envelope plot (works for both DiagnosticRecorder and EnvelopeResults)
        if hasattr(results, 'sigma_x') and results.sigma_x:
            self._envelope_plot.update_data(results)
        else:
            self._envelope_plot.clear_data()

        # Update emittance plot
        if hasattr(results, 'emit_x') and results.emit_x:
            self._emittance_plot.update_data(results)
        else:
            self._emittance_plot.clear_data()

        # Update phase space plot (only for multiparticle results with beam)
        self._update_phase_space_reference(results)
        if hasattr(results, 'beam') and results.beam is not None:
            self._phase_space_plot.set_particles(results.beam.alive_particles)
        elif hasattr(self._worker, 'simulation') and hasattr(self._worker.simulation, 'beam'):
            beam = self._worker.simulation.beam
            self._phase_space_plot.set_particles(beam.alive_particles)
        else:
            self._phase_space_plot.clear_data()

        # Update loss map plot (also overlays the transmission curve
        # when we have a multiparticle recorder with .transmission).
        if (hasattr(self._worker, 'simulation') and
                hasattr(self._worker.simulation, 'beam')):
            beam = self._worker.simulation.beam
            loss_table = beam.loss_table
            s_max = self._lattice.total_length if self._lattice else 1.0
            rec = results if hasattr(results, "transmission") else None
            self._loss_map_plot.update_data(
                loss_table, s_max, beam.n_particles, recorder=rec,
            )
        else:
            self._loss_map_plot.clear_data()

        mode = self._worker.mode if self._worker else "simulation"
        # Tag the status with what was actually simulated so the user can
        # verify the distribution / current combo they picked was used.
        cfg = self._beam_config.get_beam_config()
        extra = f"  [{cfg.distribution}, I={cfg.current:g} mA"
        if cfg.distribution == "gaussian":
            extra += f", cutoff={cfg.cutoff:g}σ"
        extra += "]"
        # Auto-dump a run report so diagnostics can be shared verbatim.
        try:
            path = self._dump_run_report(results, mode, cfg)
            self._status.showMessage(
                f"{mode.capitalize()} simulation complete{extra}   →   report: {path}"
            )
        except Exception as exc:
            # Report dump failure shouldn't abort the UI update.
            self._status.showMessage(
                f"{mode.capitalize()} simulation complete{extra}   (report failed: {exc})"
            )

    def _dump_run_report(self, results, mode: str, beam_cfg) -> str:
        """Write a JSON report with every parameter + end-of-lattice summary.

        Returns the absolute path of the written file.  Location is the
        working directory (so it travels with the project) under a
        ``run_reports/`` subfolder with a timestamped name.
        """
        import json, os, datetime
        import numpy as np
        out_dir = os.path.join(os.getcwd(), "run_reports")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(out_dir, f"{mode}-{ts}.json")

        # BeamConfig → plain dict (dataclass)
        from dataclasses import asdict
        beam_dict = asdict(beam_cfg)

        # SC config (dataclass) — we stash the live object in _sc_config
        sc_dict = asdict(self._sc_config) if self._sc_config is not None else None

        # Step config (from the lattice, if present)
        step_dict = None
        if self._lattice is not None:
            sc_cfg = getattr(self._lattice, "step_config", None)
            if sc_cfg is not None:
                step_dict = {
                    "integration_steps_per_metre": sc_cfg.integration_steps_per_metre,
                    "sc_steps_per_metre":         sc_cfg.sc_steps_per_metre,
                }

        # Lattice summary
        lattice_dict = None
        if self._lattice is not None:
            lattice_dict = {
                "n_elements": len(self._lattice.elements),
                "total_length_mm": float(sum(e.length for e in self._lattice.elements)),
                "elements": [
                    {
                        "idx": i,
                        "type": type(e).__name__,
                        "name": getattr(e, "name", ""),
                        "length_mm": float(e.length),
                    }
                    for i, e in enumerate(self._lattice.elements)
                ],
            }

        # End-of-lattice + mid-lattice sample of the recorded envelope
        def _safe_list(attr):
            return [float(v) for v in getattr(results, attr, [])]
        s_arr = _safe_list("s")
        env = {}
        for attr in ("sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                     "emit_x", "emit_y", "emit_z"):
            vals = _safe_list(attr)
            if vals and s_arr:
                env[attr] = {
                    "initial": vals[0],
                    "final":   vals[-1],
                    "max":     float(max(vals)),
                    "min":     float(min(vals)),
                }
        summary = {
            "mode":         mode,
            "timestamp":    ts,
            "python_cwd":   os.getcwd(),
            "beam_config":  beam_dict,
            "space_charge": sc_dict,
            "step_config":  step_dict,
            "lattice":      lattice_dict,
            "envelope":     env,
            "n_recorded_points": len(s_arr),
            "final_s_mm":   s_arr[-1] if s_arr else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return path

    def _on_simulation_error(self, message):
        QMessageBox.critical(self, "Simulation Error", message)
        self._status.showMessage(f"Simulation failed: {message}")

    def _update_phase_space_reference(self, results) -> None:
        """Push (β, γ, m, λ_m) at the final step to the phase-space widget
        so it can render the (z, δ) TraceWin basis."""
        import math
        from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
        from linac_gen.core.constants import C_LIGHT

        species_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
        cfg = self._beam_config.get_beam_config()
        species = species_map.get(cfg.species, PROTON)
        mass = species.mass

        beta_arr = getattr(results, "ref_beta", None) or []
        gamma_arr = getattr(results, "ref_gamma", None) or []
        if beta_arr and gamma_arr:
            beta = float(beta_arr[-1])
            gamma = float(gamma_arr[-1])
        else:
            gamma = 1.0 + cfg.energy / mass
            beta = math.sqrt(max(1.0 - 1.0 / (gamma * gamma), 0.0))

        wl_m = C_LIGHT / (cfg.frequency * 1e6)
        self._phase_space_plot.set_reference(beta, gamma, mass, wl_m)
