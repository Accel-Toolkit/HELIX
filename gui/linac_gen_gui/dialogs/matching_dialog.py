"""Matching dialog: find matched Twiss parameters and apply to beam config."""
import math
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QDialogButtonBox, QMessageBox, QTextEdit,
    QSpinBox, QComboBox,
)
from PyQt6.QtCore import Qt


class MatchingDialog(QDialog):
    """Dialog to compute periodic matched Twiss (with or without SC)."""

    def __init__(self, lattice, beam_config_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Beam Matching")
        self.setMinimumSize(520, 550)
        self._lattice = lattice
        self._beam_config_widget = beam_config_widget
        self._matched_twiss = None

        layout = QVBoxLayout(self)

        # --- Lattice periodicity (governs BOTH calculations below) ---
        mode_group = QGroupBox("Lattice Periodicity")
        mode_form = QFormLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "Whole lattice (periodic ring)",
            "FODO cell → entrance (transfer line)",
        ])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_form.addRow("Mode:", self._mode_combo)
        self._cell_combo = QComboBox()
        self._cell_combo.setEnabled(False)
        mode_form.addRow("FODO cell:", self._cell_combo)
        layout.addWidget(mode_group)
        self._populate_cells()

        # --- Zero-current periodic Twiss ---
        periodic_group = QGroupBox("1. Zero-Current Periodic Twiss")
        periodic_layout = QVBoxLayout(periodic_group)

        self._periodic_btn = QPushButton("Compute Periodic Twiss (no SC)")
        self._periodic_btn.clicked.connect(self._compute_periodic)
        periodic_layout.addWidget(self._periodic_btn)

        self._periodic_result = QFormLayout()
        self._alpha_x_label = QLabel("--")
        self._beta_x_label = QLabel("--")
        self._mu_x_label = QLabel("--")
        self._alpha_y_label = QLabel("--")
        self._beta_y_label = QLabel("--")
        self._mu_y_label = QLabel("--")
        self._periodic_result.addRow("alpha_x:", self._alpha_x_label)
        self._periodic_result.addRow("beta_x (m):", self._beta_x_label)
        self._periodic_result.addRow("mu_x (deg):", self._mu_x_label)
        self._periodic_result.addRow("alpha_y:", self._alpha_y_label)
        self._periodic_result.addRow("beta_y (m):", self._beta_y_label)
        self._periodic_result.addRow("mu_y (deg):", self._mu_y_label)
        # Periodic dispersion (nonzero only for bending lattices — arc
        # FODO / BTL).  Same mm/MeV convention as the SC-matched path.
        self._disp_x_label = QLabel("--")
        self._disp_xp_label = QLabel("--")
        self._disp_y_label = QLabel("--")
        self._disp_yp_label = QLabel("--")
        self._periodic_result.addRow("D_x (mm/MeV):", self._disp_x_label)
        self._periodic_result.addRow("D_x' (mrad/MeV):", self._disp_xp_label)
        self._periodic_result.addRow("D_y (mm/MeV):", self._disp_y_label)
        self._periodic_result.addRow("D_y' (mrad/MeV):", self._disp_yp_label)
        periodic_layout.addLayout(self._periodic_result)

        self._apply_periodic_btn = QPushButton("Apply to Beam Setup")
        self._apply_periodic_btn.setEnabled(False)
        self._apply_periodic_btn.clicked.connect(self._apply_matched)
        periodic_layout.addWidget(self._apply_periodic_btn)
        layout.addWidget(periodic_group)

        # --- SC-matched Twiss ---
        sc_group = QGroupBox("2. Space-Charge Matched Twiss (iterative)")
        sc_layout = QVBoxLayout(sc_group)

        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Max iterations:"))
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(5, 500)
        self._max_iter_spin.setValue(50)
        iter_layout.addWidget(self._max_iter_spin)
        sc_layout.addLayout(iter_layout)

        self._sc_match_btn = QPushButton("Compute SC-Matched Twiss")
        self._sc_match_btn.clicked.connect(self._compute_sc_matched)
        sc_layout.addWidget(self._sc_match_btn)

        self._sc_result = QFormLayout()
        self._sc_alpha_x_label = QLabel("--")
        self._sc_beta_x_label = QLabel("--")
        self._sc_alpha_y_label = QLabel("--")
        self._sc_beta_y_label = QLabel("--")
        self._sc_converged_label = QLabel("--")
        self._sc_result.addRow("alpha_x:", self._sc_alpha_x_label)
        self._sc_result.addRow("beta_x (m):", self._sc_beta_x_label)
        self._sc_result.addRow("alpha_y:", self._sc_alpha_y_label)
        self._sc_result.addRow("beta_y (m):", self._sc_beta_y_label)
        self._sc_result.addRow("Status:", self._sc_converged_label)
        sc_layout.addLayout(self._sc_result)

        self._apply_sc_btn = QPushButton("Apply SC-Matched to Beam Setup")
        self._apply_sc_btn.setEnabled(False)
        self._apply_sc_btn.clicked.connect(self._apply_matched)
        sc_layout.addWidget(self._apply_sc_btn)
        layout.addWidget(sc_group)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        layout.addWidget(self._log)

        # --- Close ---
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _get_ref(self):
        from linac_gen.core.reference import ReferenceParticle
        from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
        config = self._beam_config_widget.get_beam_config()
        species_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
        species = species_map.get(config.species, PROTON)
        return ReferenceParticle(species=species, w_kin=config.energy,
                                 frequency=config.frequency), config

    def _populate_cells(self):
        """Fill the FODO-cell combo from the lattice (QF->QF detection)."""
        self._cell_combo.clear()
        if self._lattice is None or len(self._lattice) == 0:
            return
        try:
            from linac_gen.matching.periodic import find_fodo_cells
            cells = find_fodo_cells(self._lattice)
        except Exception:                                   # noqa: BLE001
            cells = []
        for k, (cs, ce) in enumerate(cells):
            self._cell_combo.addItem(
                f"cell {k}: elements {cs}–{ce}", (cs, ce))

    def _on_mode_changed(self, idx):
        """Enable the FODO-cell selector only in cell mode."""
        self._cell_combo.setEnabled(idx == 1)

    def _compute_periodic(self):
        if self._lattice is None or len(self._lattice) == 0:
            QMessageBox.warning(self, "No Lattice", "Load a lattice first.")
            return
        try:
            ref, config = self._get_ref()
            if self._mode_combo.currentIndex() == 0:
                from linac_gen.matching.periodic import find_periodic_twiss
                twiss = find_periodic_twiss(self._lattice, ref)
                desc = "Periodic (whole lattice)"
            else:
                from linac_gen.matching.periodic import (
                    find_matched_input_twiss)
                cell = self._cell_combo.currentData()
                if cell is None:
                    QMessageBox.warning(
                        self, "No FODO Cell",
                        "No FODO cell detected in this lattice.\n"
                        "Use 'Whole lattice' mode, or load a lattice with a "
                        "focusing-quadrupole periodicity.")
                    return
                twiss = find_matched_input_twiss(
                    self._lattice, ref, cell[0], cell[1])
                desc = (f"Matched input (cell {cell[0]}–{cell[1]} "
                        f"→ entrance)")
            self._matched_twiss = twiss

            self._alpha_x_label.setText(f"{twiss['alpha_x']:.4f}")
            self._beta_x_label.setText(f"{twiss['beta_x']:.4f}")
            self._mu_x_label.setText(f"{twiss['mu_x']:.1f}")
            self._alpha_y_label.setText(f"{twiss['alpha_y']:.4f}")
            self._beta_y_label.setText(f"{twiss['beta_y']:.4f}")
            self._mu_y_label.setText(f"{twiss['mu_y']:.1f}")
            # Periodic dispersion — NaN means near-integer tune where
            # the periodic solution is undefined.
            for key, label in (("disp_x", self._disp_x_label),
                               ("disp_xp", self._disp_xp_label),
                               ("disp_y", self._disp_y_label),
                               ("disp_yp", self._disp_yp_label)):
                v = float(twiss.get(key, 0.0))
                label.setText("undefined (integer tune)"
                              if math.isnan(v) else f"{v:.4f}")
            self._apply_periodic_btn.setEnabled(True)
            # Annotate coupled lattices clearly -- the per-plane Twiss
            # is a *projection* from the full 4×4 matched Σ, not the
            # decoupled Courant-Snyder.  The user needs to know the
            # difference (e.g. solenoid-focused HWR cryomodules).
            coupled_tag = ""
            if twiss.get("coupled"):
                mu_1 = twiss.get("mu_1", twiss["mu_x"])
                mu_2 = twiss.get("mu_2", twiss["mu_y"])
                coupled_tag = (
                    f" [COUPLED -- normal-mode advances μ₁={mu_1:.1f}°, "
                    f"μ₂={mu_2:.1f}°; α/β are Σ projections]"
                )
            disp_tag = ""
            dvals = [float(twiss.get(k, 0.0))
                     for k in ("disp_x", "disp_xp", "disp_y", "disp_yp")]
            if any(v != 0.0 for v in dvals):   # incl. NaN (NaN != 0.0)
                disp_tag = (" [dispersive: D_x="
                            f"{dvals[0]:.3f} mm/MeV, D_y={dvals[2]:.3f}]")
            self._log.append(
                f"{desc}: beta_x={twiss['beta_x']:.3f}, "
                f"beta_y={twiss['beta_y']:.3f}, "
                f"mu_x={twiss['mu_x']:.1f} deg{coupled_tag}{disp_tag}")
        except ValueError as e:
            self._log.append(f"ERROR: {e}")
            QMessageBox.critical(self, "Failed", f"Unstable lattice:\n{e}")
        except Exception as e:
            self._log.append(f"ERROR: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def _compute_sc_matched(self):
        """Iteratively find SC-matched Twiss using envelope solver."""
        if self._lattice is None or len(self._lattice) == 0:
            QMessageBox.warning(self, "No Lattice", "Load a lattice first.")
            return
        if self._mode_combo.currentIndex() == 1:
            self._compute_sc_matched_cell()
            return

        try:
            from linac_gen.matching.periodic import find_periodic_twiss
            from linac_gen.tracking.envelope import EnvelopeSolver
            from linac_gen.core.reference import ReferenceParticle

            ref, config = self._get_ref()
            current = config.current
            if current <= 0:
                self._log.append("Current is 0 -- use zero-current matching instead.")
                self._compute_periodic()
                return

            from linac_gen.distributions.factory import (
                geometric_emittances)
            emit_geo_x, emit_geo_y, _emit_geo_z = geometric_emittances(
                config, ref.bg)

            # Start from zero-current periodic Twiss
            twiss = find_periodic_twiss(self._lattice, ref)
            alpha_x = twiss["alpha_x"]
            beta_x = twiss["beta_x"]
            alpha_y = twiss["alpha_y"]
            beta_y = twiss["beta_y"]
            # Periodic dispersion seed — constant across iterations; for
            # a straight lattice these are exact zeros and the Σ builder
            # short-circuits, so the iteration is bit-identical to before.
            # NaN (integer tune) is sanitised to 0.0: it is only a seed.
            disp_seed = {
                k: (0.0 if math.isnan(float(twiss.get(k, 0.0)))
                    else float(twiss.get(k, 0.0)))
                for k in ("disp_x", "disp_xp", "disp_y", "disp_yp")
            }

            max_iter = self._max_iter_spin.value()
            self._log.append(f"SC matching: I={current} mA, starting from zero-current Twiss...")

            def _twiss_scale():
                # Norm used to convert absolute Twiss errors into relative ones.
                return max(1.0, abs(alpha_x) + abs(alpha_y)
                           + abs(beta_x) + abs(beta_y))

            def _run_envelope(ax, bx, ay, by):
                initial = {
                    "alpha_x": ax, "beta_x": bx,
                    "alpha_y": ay, "beta_y": by,
                    "alpha_z": 0.0, "beta_z": config.beta_z,
                    "emit_x": emit_geo_x, "emit_y": emit_geo_y,
                    "emit_z": _emit_geo_z,
                    **disp_seed,
                }
                ref_iter = ReferenceParticle(
                    species=ref.species, w_kin=config.energy,
                    frequency=config.frequency)
                return EnvelopeSolver(self._lattice, ref_iter, initial,
                                      current=current).run()

            damping = 0.5
            err_history: list[float] = []
            tol = 1e-4  # relative convergence tolerance
            converged = False
            err = float("inf")

            for iteration in range(max_iter):
                env = _run_envelope(alpha_x, beta_x, alpha_y, beta_y)
                ax_out = env.alpha_x[-1]
                bx_out = env.beta_x[-1]
                ay_out = env.alpha_y[-1]
                by_out = env.beta_y[-1]

                err = (abs(ax_out - alpha_x) + abs(bx_out - beta_x)
                       + abs(ay_out - alpha_y) + abs(by_out - beta_y)) / _twiss_scale()
                err_history.append(err)

                if err < tol:
                    converged = True
                    self._log.append(
                        f"  Converged in {iteration + 1} iterations "
                        f"(relative err={err:.2e})")
                    break

                # Oscillation detector: if the error is no better than 2 iters ago,
                # reduce damping to pull the iteration into a stable fixed point.
                if (len(err_history) >= 4
                        and err_history[-1] > 0.9 * err_history[-3]):
                    damping = max(damping * 0.7, 0.1)
                    self._log.append(
                        f"  iter {iteration + 1}: oscillation suspected, "
                        f"damping -> {damping:.2f}")

                alpha_x = (1.0 - damping) * alpha_x + damping * ax_out
                beta_x = (1.0 - damping) * beta_x + damping * bx_out
                alpha_y = (1.0 - damping) * alpha_y + damping * ay_out
                beta_y = (1.0 - damping) * beta_y + damping * by_out

            if not converged:
                self._log.append(
                    f"  Did not converge after {max_iter} iterations "
                    f"(relative err={err:.2e})")
            else:
                # Post-convergence verification: rerun envelope with the final
                # Twiss and check the output still matches the input. If not,
                # the iteration hit a non-fixed-point.
                verify_env = _run_envelope(alpha_x, beta_x, alpha_y, beta_y)
                verify_err = (
                    abs(verify_env.alpha_x[-1] - alpha_x)
                    + abs(verify_env.beta_x[-1] - beta_x)
                    + abs(verify_env.alpha_y[-1] - alpha_y)
                    + abs(verify_env.beta_y[-1] - beta_y)
                ) / _twiss_scale()
                if verify_err > 10.0 * tol:
                    self._log.append(
                        f"  WARNING: verification residual {verify_err:.2e} "
                        f">> tol; result may not be a true periodic fixed point.")

            self._matched_twiss = {
                "alpha_x": alpha_x, "beta_x": beta_x,
                "alpha_y": alpha_y, "beta_y": beta_y,
            }

            self._sc_alpha_x_label.setText(f"{alpha_x:.4f}")
            self._sc_beta_x_label.setText(f"{beta_x:.4f}")
            self._sc_alpha_y_label.setText(f"{alpha_y:.4f}")
            self._sc_beta_y_label.setText(f"{beta_y:.4f}")
            self._sc_converged_label.setText(
                f"Converged ({iteration+1} iter)" if converged else f"Not converged (err={err:.2e})")
            self._apply_sc_btn.setEnabled(True)

            self._log.append(
                f"SC-matched: alpha_x={alpha_x:.4f}, beta_x={beta_x:.4f}, "
                f"alpha_y={alpha_y:.4f}, beta_y={beta_y:.4f}")

        except ValueError as e:
            self._log.append(f"ERROR: {e}")
            QMessageBox.critical(self, "Failed", str(e))
        except Exception as e:
            self._log.append(f"ERROR: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def _compute_sc_matched_cell(self):
        """SC-matched input Twiss for a transfer-line FODO cell.

        FODO-cell periodic state (betatron Twiss + dispersion) under space
        charge, back-propagated to the lattice entrance.
        """
        cell = self._cell_combo.currentData()
        if cell is None:
            QMessageBox.warning(
                self, "No FODO Cell",
                "No FODO cell detected — use 'Whole lattice' mode.")
            return
        try:
            ref, config = self._get_ref()
            current = config.current
            if current <= 0:
                self._log.append(
                    "Current is 0 — use the zero-current matching above.")
                self._compute_periodic()
                return
            from linac_gen.distributions.factory import (
                geometric_emittances)
            emit_geo_x, emit_geo_y, _emit_geo_z = geometric_emittances(
                config, ref.bg)
            base_initial = {
                "alpha_z": 0.0, "beta_z": config.beta_z,
                "emit_x": emit_geo_x, "emit_y": emit_geo_y,
                "emit_z": _emit_geo_z,
            }
            from linac_gen.matching.periodic import find_sc_matched_input_twiss
            self._log.append(
                f"SC cell matching: I={current} mA, "
                f"cell {cell[0]}–{cell[1]} → entrance...")
            tw = find_sc_matched_input_twiss(
                self._lattice, ref, cell[0], cell[1], current, base_initial,
                max_iter=self._max_iter_spin.value())
            self._matched_twiss = {
                "alpha_x": tw["alpha_x"], "beta_x": tw["beta_x"],
                "alpha_y": tw["alpha_y"], "beta_y": tw["beta_y"],
                # SC-matched dispersion — previously computed but DROPPED
                # here, so Apply silently lost it.  Now pushed to the
                # beam widget alongside alpha/beta.
                "disp_x": tw["disp_x"], "disp_xp": tw["disp_xp"],
                "disp_y": tw["disp_y"], "disp_yp": tw["disp_yp"],
            }
            self._sc_alpha_x_label.setText(f"{tw['alpha_x']:.4f}")
            self._sc_beta_x_label.setText(f"{tw['beta_x']:.4f}")
            self._sc_alpha_y_label.setText(f"{tw['alpha_y']:.4f}")
            self._sc_beta_y_label.setText(f"{tw['beta_y']:.4f}")
            self._sc_converged_label.setText(
                "Converged (FODO cell)" if tw["converged"]
                else "Not fully converged")
            self._apply_sc_btn.setEnabled(True)
            self._log.append(
                f"SC-matched input (cell): alpha_x={tw['alpha_x']:.4f}, "
                f"beta_x={tw['beta_x']:.4f}, alpha_y={tw['alpha_y']:.4f}, "
                f"beta_y={tw['beta_y']:.4f}  (D_x={tw['disp_x']:.4f})")
        except ValueError as e:
            self._log.append(f"ERROR: {e}")
            QMessageBox.critical(self, "Failed", str(e))
        except Exception as e:                                  # noqa: BLE001
            self._log.append(f"ERROR: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def _apply_matched(self):
        """Apply the most recent matched Twiss to beam config panel."""
        if self._matched_twiss is None:
            return
        tw = self._matched_twiss
        self._beam_config_widget._alpha_x.setValue(tw["alpha_x"])
        self._beam_config_widget._beta_x.setValue(tw["beta_x"])
        self._beam_config_widget._alpha_y.setValue(tw["alpha_y"])
        self._beam_config_widget._beta_y.setValue(tw["beta_y"])
        # Matched dispersion (bending lattices).  Skip NaN (integer
        # tune — undefined) and tolerate widgets without the disp spins.
        applied_disp = False
        for key in ("disp_x", "disp_xp", "disp_y", "disp_yp"):
            v = float(tw.get(key, 0.0))
            spin = getattr(self._beam_config_widget, f"_{key}", None)
            if spin is None:
                continue
            if math.isnan(v):
                self._log.append(
                    f"WARNING: {key} is undefined (integer tune) — "
                    "leaving the beam-setup value unchanged.")
                continue
            spin.setValue(v)
            applied_disp = applied_disp or (v != 0.0)
        self._log.append(
            "Applied matched Twiss"
            + (" + dispersion" if applied_disp else "")
            + " to Beam Setup panel.")
