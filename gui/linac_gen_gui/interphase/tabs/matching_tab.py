"""Matching tab — MatchingDialog launcher, current Twiss KPIs, and a
Phase-Advance panel powered by ``linac_gen.analysis``.

The Phase-Advance panel auto-detects period candidates in the loaded
lattice (LATTICE-card brackets first, type-sequence repeats second,
"(whole lattice)" always last) and lets the user pick which one to
analyse.  σ₀ comes from the bare transfer matrix; σ comes from the
matched-beam β(s) of the most recent envelope run, when available.
The σ/σ₀ ratio is the standard tune-depression diagnostic.
"""
from __future__ import annotations

import math
import time
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QMessageBox, QComboBox, QFrame, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QSizePolicy,
    QProgressBar,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.icons import icon
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.panels import kpi_card, kpi_set as _kset


class MatchingTab(QWidget):
    # Tab-level relays for the live match preview: the per-run
    # _MatchWorker's signals are re-emitted here so other tabs (the
    # Results tab's popup fan-out) can connect ONCE at app construction
    # instead of chasing every new worker instance.
    preview_results = pyqtSignal(object, int)   # (results, iter_idx)
    match_ended = pyqtSignal()                  # finished OR failed/stopped

    def __init__(self, state: AppState, beam_tab):
        super().__init__()
        self.state = state
        self._beam_tab = beam_tab  # used as BeamConfigWidget adapter

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # --- Description ----------------------------------------------
        hint = QLabel(
            "Twiss matching (including space charge) is implemented in the "
            "classic MatchingDialog.  Click the button below to open it on "
            "the currently-loaded lattice and beam config; matched α/β can "
            "be applied back into the Beam tab from inside the dialog."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_2}; font-size:12px; padding:4px 2px;")
        v.addWidget(hint)

        # --- Current Twiss KPIs ---------------------------------------
        kpi_title = QLabel("CURRENT TWISS  (from Beam tab)")
        kpi_title.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:11px; letter-spacing:1.5px;"
            f"font-weight:700; padding:6px 0 6px 8px;"
            f"border-left:3px solid {theme.ACCENT};"
            f"border-bottom:1px solid {theme.BORDER_0};"
        )
        v.addWidget(kpi_title)

        grid = QGridLayout(); grid.setSpacing(8)
        self._kax = kpi_card("α_x"); self._kbx = kpi_card("β_x", unit="mm/mrad")
        self._kex = kpi_card("ε_nx", unit="π·mm·mrad")
        self._kay = kpi_card("α_y"); self._kby = kpi_card("β_y", unit="mm/mrad")
        self._key = kpi_card("ε_ny", unit="π·mm·mrad")
        grid.addWidget(self._kax, 0, 0); grid.addWidget(self._kbx, 0, 1); grid.addWidget(self._kex, 0, 2)
        grid.addWidget(self._kay, 1, 0); grid.addWidget(self._kby, 1, 1); grid.addWidget(self._key, 1, 2)
        v.addLayout(grid)

        # --- Action row -----------------------------------------------
        row = QHBoxLayout(); row.setSpacing(8)
        self._open_btn = QPushButton("  Open Matching Dialog")
        self._open_btn.setIcon(icon("sliders", 12, "#00161c"))
        self._open_btn.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; border-radius:3px;"
            f"padding:8px 16px; font-weight:600;"
        )
        self._open_btn.clicked.connect(self._open_matching)
        row.addWidget(self._open_btn)
        self._mo_btn = QPushButton("  Multi-objective design…")
        self._mo_btn.setIcon(icon("sliders", 12, "#00161c"))
        self._mo_btn.setStyleSheet(
            f"background:{theme.BG_2}; color:{theme.TEXT_0}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:3px;"
            f"padding:8px 16px;"
        )
        self._mo_btn.setToolTip(
            "Explore the Pareto trade-off between competing objectives "
            "(e.g. emittance growth vs exit energy) over the lattice's "
            "ADJUST knobs.  NSGA-II or Bayesian multi-objective.")
        self._mo_btn.clicked.connect(self._open_multiobjective)
        row.addWidget(self._mo_btn)
        row.addStretch(1)
        self._status = QLabel("")
        self._status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;"
        )
        row.addWidget(self._status)
        v.addLayout(row)

        # --- Phase Advance panel --------------------------------------
        v.addWidget(self._build_phase_advance_panel())

        # --- Auto-Adjust (SET / ADJUST card matcher) ------------------
        v.addWidget(self._build_auto_adjust_panel())

        v.addStretch(1)

        # Per-element transfer-matrix cache is shared across tabs through
        # ``state.matrix_cache``; AppState.set_lattice / set_beam_config
        # replace it on relevant changes, so we don't need our own
        # invalidation handlers.  Background warmer pre-populates it.

        state.beam_config_changed.connect(self._refresh_kpis)
        state.lattice_changed.connect(self._refresh_periods)
        state.results_changed.connect(lambda *_: self._recompute_phase_advance())
        # Lattice mutations (inspector edits, drag-drop reorder, Open) invalidate
        # any cached match result — Apply/Save would push outdated data.
        state.lattice_changed.connect(self._invalidate_match_cache)
        # initial populate
        self._refresh_kpis(state.beam_config)
        if state.lattice is not None:
            self._refresh_periods(state.lattice)

    # ------------------------------------------------------------------
    def _refresh_kpis(self, cfg) -> None:
        if cfg is None:
            return
        _kset(self._kax, f"{cfg.alpha_x:+.3f}")
        _kset(self._kbx, f"{cfg.beta_x:.3f}")
        _kset(self._kay, f"{cfg.alpha_y:+.3f}")
        _kset(self._kby, f"{cfg.beta_y:.3f}")
        _kset(self._kex, f"{cfg.emit_nx:.3f}")
        _kset(self._key, f"{cfg.emit_ny:.3f}")

    # ------------------------------------------------------------------
    # Phase Advance panel
    # ------------------------------------------------------------------
    def _build_phase_advance_panel(self) -> QFrame:
        wrap = QFrame()
        # Raised card: BG_2 sits one notch above the tab background so the
        # panel reads as a distinct surface instead of a sunken well.
        wrap.setStyleSheet(
            f"QFrame {{ background:{theme.BG_2}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:6px; }}"
        )
        wv = QVBoxLayout(wrap)
        wv.setContentsMargins(12, 10, 12, 12); wv.setSpacing(8)

        title = QLabel("PHASE ADVANCE  (per period)")
        title.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:11px; letter-spacing:1.5px;"
            f"font-weight:700; padding:2px 0 6px 8px;"
            f"border-left:3px solid {theme.ACCENT};"
            f"border-bottom:1px solid {theme.BORDER_0};"
        )
        wv.addWidget(title)

        # Period picker + recompute
        prow = QHBoxLayout(); prow.setSpacing(8)
        prow.addWidget(QLabel("Period"))
        self._pa_combo = QComboBox()
        self._pa_combo.setMinimumWidth(280)
        self._pa_combo.currentIndexChanged.connect(
            lambda _i: self._recompute_phase_advance()
        )
        prow.addWidget(self._pa_combo, stretch=1)
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self._recompute_phase_advance)
        prow.addWidget(recompute)
        # Stability LED
        self._pa_led = QLabel("●")
        self._pa_led.setStyleSheet(f"color:{theme.TEXT_3}; font-size:18px;")
        self._pa_led.setToolTip("Stability indicator")
        prow.addWidget(self._pa_led)
        wv.addLayout(prow)

        # KPI grid — three rows:
        #   σ₀            bare channel tunes (transfer matrix / probe)
        #   σ (model)     depressed CHANNEL tunes from the phase-probe
        #                 monodromy — the primary tune-depression source
        #                 (exact w.r.t. the envelope run, matched-state
        #                 independent)
        #   σ (beam)      TraceWin-comparable ∫ds/β of the tracked beam
        #                 — beam-state dependent, needs a matched beam
        grid = QGridLayout(); grid.setSpacing(8)
        self._kpa_sx0 = kpi_card("σ₀_x", unit="deg")
        self._kpa_sy0 = kpi_card("σ₀_y", unit="deg")
        self._kpa_sxm = kpi_card("σ_x (model)", unit="deg")
        self._kpa_sym = kpi_card("σ_y (model)", unit="deg")
        self._kpa_rm  = kpi_card("η (model)")
        self._kpa_sx  = kpi_card("σ_x (beam)",  unit="deg")
        self._kpa_sy  = kpi_card("σ_y (beam)",  unit="deg")
        self._kpa_rx  = kpi_card("σ/σ₀ x (beam)")
        self._kpa_ry  = kpi_card("σ/σ₀ y (beam)")
        self._kpa_sxm.setToolTip(
            "Depressed channel tune from the SC-loaded one-period map\n"
            "(phase-probe monodromy) — matched-channel definition,\n"
            "independent of the tracked beam's match state."
        )
        self._kpa_sym.setToolTip(self._kpa_sxm.toolTip())
        self._kpa_rm.setToolTip(
            "Tune depression η = σ_model/σ₀_model per plane (or paired\n"
            "normal modes I·II when x-y coupled), median over cells."
        )
        self._kpa_sx.setToolTip(
            "Beam phase accumulation ∫ds/β from the tracked beam's\n"
            "moments — TraceWin-comparable, but only equals the channel\n"
            "tune for a matched beam."
        )
        self._kpa_sy.setToolTip(self._kpa_sx.toolTip())
        grid.addWidget(self._kpa_sx0, 0, 0); grid.addWidget(self._kpa_sy0, 0, 1)
        grid.addWidget(self._kpa_rm,  0, 2)
        grid.addWidget(self._kpa_sxm, 1, 0); grid.addWidget(self._kpa_sym, 1, 1)
        grid.addWidget(self._kpa_rx,  1, 2)
        grid.addWidget(self._kpa_sx,  2, 0); grid.addWidget(self._kpa_sy,  2, 1)
        grid.addWidget(self._kpa_ry,  2, 2)
        wv.addLayout(grid)

        self._pa_status = QLabel("")
        self._pa_status.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; font-size:10px;"
        )
        self._pa_status.setWordWrap(True)
        wv.addWidget(self._pa_status)

        # Stash periods locally so _recompute can index them.
        self._pa_periods: list = []
        return wrap

    def _refresh_periods(self, lattice) -> None:
        self._pa_combo.blockSignals(True)
        self._pa_combo.clear()
        self._pa_periods = []
        if lattice is None:
            self._pa_combo.blockSignals(False)
            self._reset_phase_advance_kpis()
            return
        from linac_gen.analysis.period_detect import detect_periods
        try:
            self._pa_periods = detect_periods(lattice)
        except Exception as exc:                       # noqa: BLE001
            self._pa_periods = []
            self._pa_status.setText(f"period detection failed: {exc}")
        for p in self._pa_periods:
            tag = {
                "lattice_card":            "[LATTICE]",
                "lattice_card_recovered":  "[LATTICE+]",
                "type_sequence":           "[auto]",
                "fallback":                "[full]",
            }.get(p.source, "[?]")
            self._pa_combo.addItem(f"{tag}  {p.label}")
        self._pa_combo.blockSignals(False)
        if self._pa_periods:
            self._pa_combo.setCurrentIndex(0)
            self._recompute_phase_advance()
        else:
            self._reset_phase_advance_kpis()

    def _reset_phase_advance_kpis(self) -> None:
        for k in (self._kpa_sx0, self._kpa_sy0, self._kpa_sx,
                  self._kpa_sy, self._kpa_rx, self._kpa_ry,
                  self._kpa_sxm, self._kpa_sym, self._kpa_rm):
            _kset(k, "—")
        self._pa_led.setStyleSheet(f"color:{theme.TEXT_3}; font-size:18px;")
        self._pa_led.setToolTip("Stability indicator")
        self._pa_status.setText("")

    def showEvent(self, ev):                                  # noqa: N802
        super().showEvent(ev)
        if getattr(self, "_pa_stale", False):
            self._pa_stale = False
            self._recompute_phase_advance()

    def _recompute_phase_advance(self) -> None:
        if not self.isVisible():
            # σ₀ is a synchronous one-period transfer-matrix build on the UI
            # thread (seconds on periodic field-map lattices, cold cache).
            # This slot fires on every results/lattice change — including
            # while this tab is hidden.  Defer to the next showEvent instead
            # of freezing the GUI for a panel nobody is looking at.  (Same
            # gate as the results-tab σ₀ popups.)
            self._pa_stale = True
            return
        if not self._pa_periods or self.state.lattice is None:
            self._reset_phase_advance_kpis()
            return
        idx = max(0, self._pa_combo.currentIndex())
        if idx >= len(self._pa_periods):
            return
        period = self._pa_periods[idx]

        # Build a ReferenceParticle from the current beam config.
        cfg = self.state.beam_config
        if cfg is None:
            self._pa_status.setText("Set a beam config first (Beam tab).")
            self._reset_phase_advance_kpis()
            return

        from linac_gen.analysis.phase_advance import beam_phase_advance

        # The "(whole lattice)" fallback isn't a periodic structure — an
        # accelerating linac has no closed one-period matrix (trace |½M|≥1
        # in both planes), and HWR/SSR solenoids couple x↔y on top of that,
        # so σ₀ is undefined.  Skip the σ₀ computation entirely: avoids a
        # multi-minute synchronous freeze building a transfer matrix
        # through 1000+ field maps for an answer that would only show "—"
        # anyway.  ``beam_phase_advance`` with ``sigma0=None`` still gives
        # the total ∫ds/β integral across the span, which is meaningful.
        is_fallback = (period.source == "fallback")

        sigma0 = None
        if not is_fallback:
            try:
                from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
                from linac_gen.core.reference import ReferenceParticle
                sp_map = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}
                sp = sp_map.get(cfg.species, PROTON)
                ref = ReferenceParticle(species=sp, w_kin=cfg.energy,
                                        frequency=cfg.frequency)
                from linac_gen.analysis.phase_advance import (
                    structure_phase_advance,
                )
                sigma0 = structure_phase_advance(
                    self.state.lattice, ref, period,
                    cache=self.state.matrix_cache,
                )
            except Exception as exc:                   # noqa: BLE001
                self._reset_phase_advance_kpis()
                self._pa_status.setText(f"σ₀ failed: {exc}")
                return

        # σ₀ KPIs + stability LED.
        if sigma0 is None:
            _kset(self._kpa_sx0, "—"); _kset(self._kpa_sy0, "—")
            self._pa_led.setStyleSheet(f"color:{theme.TEXT_3}; font-size:18px;")
            self._pa_led.setToolTip(
                "σ₀ not defined — the whole lattice is not a periodic structure"
            )
        elif sigma0.get("coupled"):
            _kset(self._kpa_sx0, "(coupled)")
            _kset(self._kpa_sy0, "(coupled)")
            self._pa_led.setStyleSheet(f"color:{theme.WARN}; font-size:18px;")
            self._pa_led.setToolTip("Coupled lattice — μ undefined per plane")
        else:
            _kset(self._kpa_sx0, _fmt_deg(sigma0.get("mu_x_deg")))
            _kset(self._kpa_sy0, _fmt_deg(sigma0.get("mu_y_deg")))
            stable = sigma0.get("stable_x") and sigma0.get("stable_y")
            if stable:
                self._pa_led.setStyleSheet(f"color:{theme.OK}; font-size:18px;")
                self._pa_led.setToolTip("Stable: |½ tr(M)| < 1 in both planes")
            else:
                self._pa_led.setStyleSheet(f"color:{theme.ERR}; font-size:18px;")
                self._pa_led.setToolTip(
                    f"Unstable: {sigma0.get('reason') or 'see log'}"
                )

        # Status line.
        notes = []
        if is_fallback:
            notes.append(
                "Whole lattice is not periodic — σ₀ undefined.  σ shows "
                "the total β-integrated phase advance across the span.  "
                "For tune depression, pick a LATTICE-bracket section."
            )
        else:
            dw = sigma0.get("dw") or 0.0
            if abs(dw) > 1e-9:
                notes.append(
                    f"period accelerates: W {sigma0['w_in']:.3f} → "
                    f"{sigma0['w_out']:.3f} MeV (μ at entry energy)"
                )

        # Channel tunes (σ_model) — the PRIMARY tune-depression source.
        # Needs probe-bearing envelope results (the GUI's envelope runs
        # record the probe by default since the phase-advance overhaul).
        results = self.state.results
        ch = None
        if (results is not None
                and getattr(results, "element_maps_dep", None)
                and not is_fallback):
            try:
                from linac_gen.analysis.phase_advance import (
                    channel_phase_advance,
                )
                import numpy as _np
                ch = channel_phase_advance(results, period)
            except Exception as exc:                   # noqa: BLE001
                notes.append(f"σ_model failed: {exc}")
                ch = None
        if ch is not None:
            import numpy as _np

            def _med(key):
                a = _np.asarray(ch.get(key, []), dtype=float)
                a = a[_np.isfinite(a)]
                return float(_np.median(a)) if a.size else None

            if ch["coupled_xy"]:
                mI, mII = _med("mu_I_dep_deg"), _med("mu_II_dep_deg")
                eI, eII = _med("eta_I"), _med("eta_II")
                _kset(self._kpa_sxm,
                      f"{mI:.2f} (I)" if mI is not None else "—")
                _kset(self._kpa_sym,
                      f"{mII:.2f} (II)" if mII is not None else "—")
                _kset(self._kpa_rm,
                      (f"{eI:.3f} · {eII:.3f}"
                       if eI is not None and eII is not None else "—"))
                notes.append("modes I/II (x-y coupled channel)")
            else:
                mx, my = _med("mu_x_dep_deg"), _med("mu_y_dep_deg")
                ex, ey = _med("eta_x"), _med("eta_y")
                _kset(self._kpa_sxm, _fmt_deg(mx))
                _kset(self._kpa_sym, _fmt_deg(my))
                _kset(self._kpa_rm,
                      (f"{ex:.3f} · {ey:.3f}"
                       if ex is not None and ey is not None else "—"))
            # Per-cell spread hint (accelerating sections detune cells).
            for key, tag in (("eta_I", "η_I") if ch["coupled_xy"]
                             else ("eta_x", "η_x"),):
                a = _np.asarray(ch.get(key, []), dtype=float)
                a = a[_np.isfinite(a)]
                if a.size >= 2 and a.max() - a.min() > 0.02:
                    notes.append(
                        f"{tag} spreads {a.min():.2f}–{a.max():.2f} over "
                        f"cells (accelerating/quasi-periodic section)")
        else:
            _kset(self._kpa_sxm, "—"); _kset(self._kpa_sym, "—")
            _kset(self._kpa_rm, "—")
            if results is not None and not is_fallback:
                notes.append(
                    "σ_model needs a probe-bearing envelope run — re-run "
                    "the envelope (probe is recorded automatically)")

        # Beam μ — only available with envelope results.
        sigma_beam = None
        if results is not None and getattr(results, "beta_x", None):
            try:
                sigma_beam = beam_phase_advance(results, period, sigma0=sigma0)
            except Exception as exc:                  # noqa: BLE001
                notes.append(f"σ failed: {exc}")
                sigma_beam = None
        if sigma_beam is not None:
            _kset(self._kpa_sx, _fmt_deg(sigma_beam.get("mu_x_deg")))
            _kset(self._kpa_sy, _fmt_deg(sigma_beam.get("mu_y_deg")))
            if sigma0 is not None:
                _kset(self._kpa_rx, _fmt_ratio(sigma_beam.get("sigma_over_sigma0_x")))
                _kset(self._kpa_ry, _fmt_ratio(sigma_beam.get("sigma_over_sigma0_y")))
            else:
                _kset(self._kpa_rx, "—"); _kset(self._kpa_ry, "—")
            if sigma0 is not None and not sigma_beam.get("matched", True):
                bx_pct = sigma_beam["mismatch_x"] * 100
                by_pct = sigma_beam["mismatch_y"] * 100
                bm = sigma_beam.get("mismatch_bmag_x")
                bm_txt = f", BMAG_x={bm:.2f}" if bm else ""
                notes.append(
                    f"BEAM NOT MATCHED — β mismatch x={bx_pct:.1f}%, "
                    f"y={by_pct:.1f}%{bm_txt} (beam σ unreliable; "
                    f"σ_model above is match-independent)"
                )
            if sigma_beam.get("projected_only"):
                notes.append(
                    "⚠ span crosses x-y-coupled records — beam σ is a "
                    "projected value, not a mode tune (trust σ_model)")
            if not sigma_beam.get("resolution_ok", True):
                notes.append(
                    f"⚠ coarse sampling "
                    f"({sigma_beam.get('samples_per_period', 0):.0f} "
                    f"samples/2π < 20) — beam σ under-resolved; enable "
                    f"'Record per-sub-step' or trust σ_model")
            n_skip_x = sigma_beam.get("n_skipped_x", 0)
            n_skip_y = sigma_beam.get("n_skipped_y", 0)
            n_total  = sigma_beam.get("n_total", 0) or 0
            if (n_skip_x or n_skip_y) and n_total:
                notes.append(
                    f"{max(n_skip_x, n_skip_y)}/{n_total} samples "
                    "skipped (beam lost — β=0)"
                )
        else:
            _kset(self._kpa_sx, "—"); _kset(self._kpa_sy, "—")
            _kset(self._kpa_rx, "—"); _kset(self._kpa_ry, "—")
            if results is None:
                notes.append("Run an envelope or MP simulation to populate σ.")
        self._pa_status.setText(" · ".join(notes))

    # ------------------------------------------------------------------
    # Auto-Adjust (SET / ADJUST card-driven matcher) panel
    # ------------------------------------------------------------------
    def _build_auto_adjust_panel(self) -> QFrame:
        wrap = QFrame()
        # Raised card: BG_2 sits one notch above the tab background so the
        # panel reads as a distinct surface instead of a sunken well.
        wrap.setStyleSheet(
            f"QFrame {{ background:{theme.BG_2}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:6px; }}"
        )
        wv = QVBoxLayout(wrap)
        wv.setContentsMargins(12, 10, 12, 12); wv.setSpacing(8)

        title = QLabel("AUTO-ADJUST  (SET / ADJUST cards in lattice)")
        title.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:11px; letter-spacing:1.5px;"
            f"font-weight:700; padding:2px 0 6px 8px;"
            f"border-left:3px solid {theme.ACCENT};"
            f"border-bottom:1px solid {theme.BORDER_0};"
        )
        wv.addWidget(title)

        hint = QLabel(
            "Runs the Levenberg-Marquardt matcher against ADJUST_* "
            "variables and SET_* constraints in the loaded lattice. "
            "Variables and matched values appear below; click "
            "Apply to write them into the elements / beam config, "
            "or Save to export a matched .dat."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_2}; font-size:11px;")
        wv.addWidget(hint)

        # Two rows instead of one: the old single strip held 13 controls
        # and demanded ~1200px of width — at laptop widths Qt compressed
        # it and the trailing buttons clipped.  Row 1 = solver setup,
        # row 2 = algorithm-specific knobs + actions.
        opts = QHBoxLayout(); opts.setSpacing(10)
        opts2 = QHBoxLayout(); opts2.setSpacing(10)
        self._aa_sc = QCheckBox("Space charge")
        opts.addWidget(self._aa_sc)
        opts.addWidget(QLabel("Max iter"))
        self._aa_iter = QSpinBox()
        self._aa_iter.setRange(10, 5000); self._aa_iter.setValue(200)
        opts.addWidget(self._aa_iter)

        # ---- Cost solver: envelope (fast, default) vs multi-particle ----
        # MP mode uses the Convergence-tab SC config and step-config so
        # the matcher sees the SAME physics as the toolbar's Run-MP
        # button -- no hidden default SC behaviour.
        opts.addWidget(QLabel("Cost solver"))
        self._aa_cost_solver = QComboBox()
        self._aa_cost_solver.addItems(["envelope", "mp"])
        self._aa_cost_solver.setCurrentText("envelope")
        self._aa_cost_solver.setToolTip(
            "Forward-pass model for the matcher's cost function:\n"
            "  envelope (default) -- RMS sigma matrix via linear matrix\n"
            "    propagation, linear SC kick.  Fast (5-30 s / eval) and\n"
            "    accurate enough for most matches.  The right pick when\n"
            "    you're iterating on knobs.\n"
            "  mp -- full multi-particle PIC, identical physics to the\n"
            "    toolbar's Run-MP button.  Captures nonlinear SC, halo\n"
            "    formation, and full 4-D coupling.  ~50-100x slower per\n"
            "    eval; use for the final, physics-accurate polish after\n"
            "    an envelope-mode match has got you close.  Uses the\n"
            "    Numerics tab's SC settings (no hidden defaults).\n"
            "  NOTE: centroid constraints (DIAG_POSITION targets,\n"
            "    SET_POSITION) fit under BOTH solvers -- the envelope\n"
            "    tracks a real first moment, and its noiseless fits\n"
            "    converge in minutes where mp takes hours.  Only\n"
            "    MIN_TRANSMISSION still needs 'mp' (the envelope\n"
            "    tracks no particle loss); the strict audit refuses\n"
            "    to silently ignore it.  The 'gradient' algorithm\n"
            "    cannot enforce centroid constraints (torch mirror\n"
            "    has no first moment) -- it refuses with an error."
        )
        opts.addWidget(self._aa_cost_solver)

        opts.addWidget(QLabel("MP particles"))
        self._aa_mp_n = QSpinBox()
        self._aa_mp_n.setRange(50, 200_000)
        self._aa_mp_n.setSingleStep(500)
        self._aa_mp_n.setValue(1000)
        self._aa_mp_n.setToolTip(
            "Number of macroparticles per MP cost evaluation.  Lower = "
            "faster but noisier cost; higher = slower but more "
            "physics-accurate.  Ignored unless Cost solver = mp."
        )
        opts.addWidget(self._aa_mp_n)

        # Grey out MP particle count when cost_solver != "mp"
        def _on_cost_solver_changed(text: str) -> None:
            self._aa_mp_n.setEnabled(text == "mp")
        _on_cost_solver_changed(self._aa_cost_solver.currentText())
        self._aa_cost_solver.currentTextChanged.connect(_on_cost_solver_changed)

        # Escape hatch for the pre-run constraint audit: match() refuses
        # to run when an active constraint would be silently ignored
        # (a stub-evaluator card carrying weight, or MIN_TRANSMISSION
        # under the envelope cost-solver, which tracks no loss).  Mirrors
        # the engine kwarg / CLI --allow-inert-constraints flag.
        self._aa_allow_inert = QCheckBox("Allow inert constraints")
        self._aa_allow_inert.setChecked(False)
        self._aa_allow_inert.setToolTip(
            "By default the matcher REFUSES to run when an active\n"
            "constraint card would be silently ignored — e.g.\n"
            "MIN_TRANSMISSION with the envelope cost-solver (envelope\n"
            "mode tracks no particle loss, so the residual is always 0),\n"
            "or a stub card carrying a nonzero weight.\n\n"
            "Tick this to proceed anyway (the inert constraint simply\n"
            "contributes nothing) — useful for legacy decks.  Fixing the\n"
            "deck or switching Cost solver to mp is the better answer."
        )
        opts.addWidget(self._aa_allow_inert)

        opts.addWidget(QLabel("Algorithm"))
        from linac_gen.matching import MATCH_ALGORITHMS
        self._aa_algorithm = QComboBox()
        self._aa_algorithm.addItems(list(MATCH_ALGORITHMS))
        self._aa_algorithm.setCurrentText("least_squares")
        self._aa_algorithm.setToolTip(
            "Optimiser for the match:\n"
            "  least_squares — local, fast, the default.\n"
            "  differential_evolution / dual_annealing — global search,\n"
            "  slower, and require finite min/max bounds on every ADJUST\n"
            "  variable.  For these two, 'Max iter' counts generations /\n"
            "  annealing iterations rather than function evaluations.\n"
            "  cmaes — Covariance Matrix Adaptation Evolution Strategy.\n"
            "  Robust global, gradient-free, bound-respecting; the right\n"
            "  pick for multimodal landscapes (coupling resonances,\n"
            "  sync-phase sign flips) and the one-sided MIN_EMIT_GROWTH /\n"
            "  SET_KE_OUT_MIN constraints whose flat regions defeat LM.\n"
            "  Default sigma=0.2, package-default popsize; chained with\n"
            "  a least_squares polish for LM-quality precision.\n"
            "  gradient — local search with exact autograd Jacobians from\n"
            "  differentiable tracking; SET_TWISS / SET_SIZE matching of\n"
            "  quad / solenoid / dipole knobs.  Works with or without space\n"
            "  charge — tick 'Space charge' to match through the\n"
            "  differentiable PIC.  Linear lattices only (no RF / field maps).\n"
            "  bayesopt — Gaussian-process Bayesian optimisation (BoTorch).\n"
            "  Sample-efficient global search; the best pick for EXPENSIVE\n"
            "  matches (Cost solver = mp / space charge) where each eval is\n"
            "  seconds-minutes.  Needs finite bounds; chained with an LS\n"
            "  polish.  Tick 'BO physics prior' to warm-start an MP match\n"
            "  from the cheap envelope cost."
        )
        opts.addWidget(self._aa_algorithm)

        # ---- CMA-ES advanced knobs --------------------------------
        # Exposed inline so users don't have to dig for the CLI.  Only
        # active when Algorithm = cmaes; the other optimisers ignore
        # these values entirely.  Defaults match the engine's CLI
        # defaults: sigma=0.2, popsize=0 (cma library auto-default
        # = 4 + floor(3 ln N)), refine=True (LS polish after CMA-ES).
        opts.addStretch(1)
        opts2.addWidget(QLabel("σ₀"))
        self._aa_cmaes_sigma = QDoubleSpinBox()
        self._aa_cmaes_sigma.setRange(0.01, 1.0)
        self._aa_cmaes_sigma.setDecimals(2)
        self._aa_cmaes_sigma.setSingleStep(0.05)
        self._aa_cmaes_sigma.setValue(0.2)
        self._aa_cmaes_sigma.setToolTip(
            "CMA-ES initial step size as a fraction of the bound box "
            "width (default 0.2).  Larger explores more aggressively; "
            "smaller refines a near-matched lattice.  Ignored unless "
            "Algorithm = cmaes."
        )
        opts2.addWidget(self._aa_cmaes_sigma)

        opts2.addWidget(QLabel("popsize"))
        self._aa_cmaes_popsize = QSpinBox()
        self._aa_cmaes_popsize.setRange(0, 200)
        self._aa_cmaes_popsize.setValue(0)
        self._aa_cmaes_popsize.setSpecialValueText("auto")
        self._aa_cmaes_popsize.setToolTip(
            "CMA-ES population size; 0 = library default "
            "(4 + floor(3·ln(N))).  Larger is more robust against "
            "local minima but costs more forward passes per generation."
            "  Ignored unless Algorithm = cmaes."
        )
        opts2.addWidget(self._aa_cmaes_popsize)

        self._aa_no_refine = QCheckBox("no LS polish")
        self._aa_no_refine.setToolTip(
            "When ticked, CMA-ES / bayesopt output is returned without a "
            "scipy.optimize.least_squares refinement.  Use for very "
            "fast smoke tests, or when the one-sided constraint floors "
            "the cost (the LS polish is a no-op).  Ignored unless "
            "Algorithm = cmaes or bayesopt."
        )
        opts2.addWidget(self._aa_no_refine)

        # Bayesian-optimisation: physics-informed warm start.
        self._aa_bo_prior = QCheckBox("BO physics prior")
        self._aa_bo_prior.setToolTip(
            "Bayesian optimisation only: use the cheap envelope cost to "
            "pre-scout the parameter box before the expensive objective.\n"
            "Only has an effect with Algorithm = bayesopt AND "
            "Cost solver = mp (it warm-starts the MP match from the "
            "envelope-good region).  Ignored otherwise."
        )
        opts2.addWidget(self._aa_bo_prior)

        # Grey out algorithm-specific knobs when not applicable.
        def _on_algo_changed(text: str) -> None:
            is_cma = (text == "cmaes")
            is_bo = (text == "bayesopt")
            self._aa_cmaes_sigma.setEnabled(is_cma)
            self._aa_cmaes_popsize.setEnabled(is_cma)
            # LS polish applies to both cmaes and bayesopt.
            self._aa_no_refine.setEnabled(is_cma or is_bo)
            self._aa_bo_prior.setEnabled(is_bo)
        _on_algo_changed(self._aa_algorithm.currentText())
        self._aa_algorithm.currentTextChanged.connect(_on_algo_changed)

        opts2.addStretch(1)
        self._aa_run = QPushButton("  Match")
        self._aa_run.setIcon(icon("play", 12, "#00161c"))
        self._aa_run.setStyleSheet(
            f"background:{theme.ACCENT}; color:#00161c; border:0; "
            f"border-radius:3px; padding:6px 14px; font-weight:600;"
        )
        self._aa_run.clicked.connect(self._on_match_clicked)
        opts2.addWidget(self._aa_run)
        # Dedicated Stop button right next to Match -- much more
        # discoverable than the toolbar Stop, which is wired only to
        # envelope / multi-particle workers (not the matcher).  Calls
        # QThread.requestInterruption on the active match worker; the
        # worker checks this in its progress callback and raises
        # StopIteration to abort cleanly at the next eval boundary.
        self._aa_stop = QPushButton("  Stop")
        self._aa_stop.setIcon(icon("stop", 12, "#ffffff"))
        self._aa_stop.setStyleSheet(
            f"QPushButton {{ background:{theme.ERR}; color:#ffffff; "
            f"border:0; border-radius:3px; padding:6px 14px; "
            f"font-weight:600; }}"
            f"QPushButton:disabled {{ background:{theme.BG_2}; "
            f"color:{theme.TEXT_2}; border:1px solid {theme.BORDER_1}; "
            f"font-weight:400; }}"
        )
        self._aa_stop.setEnabled(False)
        self._aa_stop.setToolTip(
            "Cancel the running match at the next cost-function "
            "evaluation (latency ~ one forward pass)."
        )
        self._aa_stop.clicked.connect(self._on_match_stop)
        opts2.addWidget(self._aa_stop)
        self._aa_apply = QPushButton("Apply")
        self._aa_apply.setEnabled(False)
        self._aa_apply.clicked.connect(self._on_match_apply)
        opts2.addWidget(self._aa_apply)
        self._aa_save = QPushButton("Save matched .dat")
        self._aa_save.setEnabled(False)
        self._aa_save.clicked.connect(self._on_match_save)
        opts2.addWidget(self._aa_save)
        wv.addLayout(opts)
        wv.addLayout(opts2)

        # Result tables
        self._aa_var_table = QTableWidget(0, 4)
        self._aa_var_table.setHorizontalHeaderLabels(
            ["Variable", "Initial", "Matched", "Bounds"]
        )
        self._aa_var_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._aa_var_table.setMinimumHeight(140)
        self._aa_var_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        wv.addWidget(self._aa_var_table)

        self._aa_con_table = QTableWidget(0, 3)
        self._aa_con_table.setHorizontalHeaderLabels(
            ["Constraint", "RMS residual", "Notes"]
        )
        self._aa_con_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._aa_con_table.setMinimumHeight(120)
        self._aa_con_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        wv.addWidget(self._aa_con_table)

        # ----- Live progress row ------------------------------------------
        # Indeterminate busy bar + prominent status text + elapsed-time
        # ticker.  Without this the matching tab gives no feedback during
        # long PIP-II HWR runs (each forward pass can take ~30 s, so the
        # iteration counter only updates every half-minute — easy to
        # mistake for a freeze).  The busy bar runs at 60 Hz from Qt so
        # the user can see at a glance that the worker is alive.
        prog_row = QHBoxLayout()
        prog_row.setContentsMargins(0, 4, 0, 0)
        self._aa_progress = QProgressBar()
        self._aa_progress.setRange(0, 0)            # indeterminate
        self._aa_progress.setTextVisible(False)
        self._aa_progress.setFixedHeight(6)
        self._aa_progress.setStyleSheet(
            f"QProgressBar {{ background:{theme.BG_INSET}; border:0; "
            f"border-radius:3px; }} "
            f"QProgressBar::chunk {{ background:{theme.ACCENT}; "
            f"border-radius:3px; }}"
        )
        self._aa_progress.setVisible(False)
        prog_row.addWidget(self._aa_progress, stretch=1)
        wv.addLayout(prog_row)

        self._aa_status = QLabel("")
        self._aa_status.setStyleSheet(
            f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; "
            f"font-size:12px; font-weight:600; padding:4px 0;"
        )
        self._aa_status.setMinimumHeight(22)
        wv.addWidget(self._aa_status)

        # Holds the most recent MatchResult between Run and Apply/Save.
        self._aa_result = None
        self._aa_worker = None
        # Live convergence-plot dialog (created lazily on first Match
        # click).  Persists across matches so the user can review the
        # last run's trace; replaced on each new Match.
        self._aa_convergence_dlg = None

        # Elapsed-time ticker.  Refreshes the status text every 500 ms
        # so the user sees something moving even when forward passes are
        # slow (FieldMap3D RK4 → ~30 s per eval is normal for PIP-II).
        self._aa_start_time = 0.0
        self._aa_last_iter = 0
        self._aa_last_cost = float("inf")
        self._aa_tick = QTimer(self)
        self._aa_tick.setInterval(500)
        self._aa_tick.timeout.connect(self._refresh_match_status)

        return wrap

    def _build_mp_configs(self):
        """Read the Numerics tab's SC + step settings to build the
        configs the MP cost solver will use.  Returns
        ``(sc_config, step_config)``.  Mirrors the pattern in
        ``app.py:_on_run_envelope`` so MP-mode matching uses the
        identical physics the toolbar's Run-MP button would.

        ``window()`` walks the Qt parent chain to the top-level
        InterphaseWindow which owns ``convergence_tab``.  Falls back
        to None / default StepConfig if the chain breaks (the engine
        constructs a sane default when these are None).
        """
        from linac_gen.core.config import SpaceChargeConfig
        from linac_gen.core.step_config import StepConfig
        try:
            ct = self.window().convergence_tab    # type: ignore[attr-defined]
        except AttributeError:
            return None, None
        try:
            cfg = self.state.beam_config
            nx = ct._fixed_nx.value()
            ext = ct._fixed_ext.value()
            step1 = ct._fixed_step1.value()
            step2 = ct._fixed_step2.value()
            backend = ct._fixed_backend.currentText()
            green_kind = ct._fixed_green.currentText()
            kernel = ct._fixed_kernel.currentText()
            grid_mode = ct._fixed_grid_mode.currentText()
            dc_kernel = ct._fixed_dc_kernel.currentText()
            csr_on = bool(ct._fixed_csr.isChecked())
            sc_backend = ct._fixed_sc_backend.currentText()
            # torch backend is bunched-only; fall back to numpy on DC.
            if sc_backend == "torch" and getattr(cfg, "continuous", False):
                sc_backend = "numpy"
            if sc_backend == "torch":
                grid_mode = "adaptive"
            sc_cfg = (SpaceChargeConfig(nx=nx, ny=nx, nz=nx, grid_extent=ext,
                                        grid_mode=grid_mode,
                                        use_gpu=backend,
                                        green_kind=green_kind, kernel=kernel,
                                        dc_kernel=dc_kernel,
                                        sc_backend=sc_backend,
                                        csr_enabled=csr_on)
                      if cfg.current > 0 else None)
            step_cfg = StepConfig(
                integration_steps_per_metre=float(step1),
                sc_steps_per_metre=float(step2),
            )
            return sc_cfg, step_cfg
        except Exception:                                  # noqa: BLE001
            # Numerics tab not fully initialised (e.g. on test
            # harness); the engine will use defaults.
            return None, None

    def _on_match_clicked(self) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first.")
            return
        try:
            cfg = self._beam_tab.get_beam_config()
            self.state.set_beam_config(cfg)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, "Beam config error", str(exc))
            return

        # The matcher mutates the lattice; deep-copy so we can either
        # apply on success or discard cleanly.
        import copy
        lattice_copy = copy.deepcopy(self.state.lattice)
        beam_copy = copy.deepcopy(cfg)

        self._aa_run.setEnabled(False)
        self._aa_stop.setEnabled(True)
        self._aa_apply.setEnabled(False)
        self._aa_save.setEnabled(False)

        # Reset live-progress state and show the busy bar.  The
        # algorithm + space-charge mode are shown alongside elapsed time
        # so the user can tell from one glance which run they kicked off.
        import time
        self._aa_start_time = time.time()
        self._aa_last_iter = 0
        self._aa_last_cost = float("inf")
        # popsize is reported by the worker once CMA-ES is up; before
        # that, fall back to estimating from the variable count so the
        # "gen" display has something to render on the very first tick.
        self._aa_popsize = 0
        self._aa_progress.setVisible(True)
        self._aa_tick.start()
        algo = self._aa_algorithm.currentText()
        sc = "with SC" if self._aa_sc.isChecked() else "no SC"

        # ---- Sequential-scan setup dialog ------------------------------
        # Coordinate-descent matchers benefit from explicit element
        # selection + scan parameters that the other algorithms don't
        # need; pop a dedicated dialog before constructing the worker.
        # Cancel in the dialog aborts the match entirely (restore button
        # states so the user can pick a different algorithm).
        seqscan_settings: dict = {}
        if algo == "sequential_scan":
            from linac_gen_gui.interphase.dialogs.sequential_scan_setup import (
                SequentialScanSetupDialog,
            )
            dlg = SequentialScanSetupDialog(lattice_copy, parent=self)
            if dlg.exec() != dlg.DialogCode.Accepted:
                # User cancelled the setup dialog (NOT a running match);
                # restore the idle button state.  Apply/Save reflect
                # whether a PRIOR match has a result available (we use
                # self._aa_result, not self._aa_worker -- the latter
                # persists across runs whereas _aa_result is cleared by
                # the lattice-changed signal).
                self._aa_progress.setVisible(False)
                self._aa_tick.stop()
                self._aa_run.setEnabled(True)
                self._aa_stop.setEnabled(False)
                self._aa_apply.setEnabled(self._aa_result is not None)
                self._aa_save.setEnabled(self._aa_result is not None)
                self._aa_status.setText("Sequential scan cancelled.")
                return
            _, seqscan_settings = dlg.get_settings()

        # Build the MP cost-solver config from the Numerics tab so
        # the matcher's MP forward pass sees the SAME SC physics as
        # the toolbar's Run-MP button (no hidden defaults).  The
        # 'gradient' algorithm ignores cost_solver but DOES track a
        # macro-particle bunch through PIC space charge when SC is on —
        # without this config it silently fell back to the engine's
        # 32^3/±4σ defaults while every other path honoured the tab.
        cost_solver = self._aa_cost_solver.currentText()
        mp_sc_config = None
        mp_step_config = None
        if cost_solver == "mp" or (algo == "gradient"
                                   and self._aa_sc.isChecked()):
            mp_sc_config, mp_step_config = self._build_mp_configs()

        self._aa_status.setText(
            f"Matching ({algo}, {sc}, cost={cost_solver}) — "
            f"iter 0  cost=—  elapsed 0.0 s"
        )

        # Disconnect the previous worker's signals before replacing it.
        # Without this, a delayed emission from the OLD worker (e.g. its
        # finished_with arriving after the user clicks Match again) would
        # fire into the slots that now belong to the NEW worker's run and
        # overwrite or corrupt its state.  Qt's `disconnect()` with no
        # args removes all connections; wrap in try/except so a worker
        # that was never connected (rare) doesn't crash the handler.
        prev = getattr(self, "_aa_worker", None)
        if prev is not None:
            for sig in (
                getattr(prev, "progress", None),
                getattr(prev, "popsize_known", None),
                getattr(prev, "finished_with", None),
                getattr(prev, "failed", None),
                getattr(prev, "progress_detail", None),
                getattr(prev, "preview_results", None),
            ):
                if sig is None:
                    continue
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
            # The old worker may still be inside run() (a cost eval can take
            # tens of seconds).  Replacing self._aa_worker would drop our only
            # reference to a live QThread and destroy it mid-run → process
            # abort.  Park it until it finishes, then let go.
            if prev.isRunning():
                # The superseded match's results will be discarded — tell it
                # to stop at its next cost evaluation instead of letting a
                # full CMA-ES run burn cores alongside the new match.
                prev.requestInterruption()
                if not hasattr(self, "_zombie_workers"):
                    self._zombie_workers = []
                self._zombie_workers.append(prev)
                prev.finished.connect(
                    lambda w=prev: (self._zombie_workers.remove(w)
                                    if w in self._zombie_workers else None))

        self._aa_worker = _MatchWorker(
            lattice_copy, beam_copy,
            space_charge=self._aa_sc.isChecked(),
            algorithm=self._aa_algorithm.currentText(),
            max_iter=int(self._aa_iter.value()),
            cmaes_sigma=float(self._aa_cmaes_sigma.value()),
            cmaes_popsize=int(self._aa_cmaes_popsize.value()),
            refine=not self._aa_no_refine.isChecked(),
            bo_prior=self._aa_bo_prior.isChecked(),
            seqscan_settings=seqscan_settings,
            cost_solver=cost_solver,
            mp_n_particles=int(self._aa_mp_n.value()),
            mp_sc_config=mp_sc_config,
            mp_step_config=mp_step_config,
            allow_inert_constraints=self._aa_allow_inert.isChecked(),
        )
        self._aa_worker.progress.connect(self._on_match_progress)
        self._aa_worker.popsize_known.connect(self._on_match_popsize_known)
        self._aa_worker.finished_with.connect(self._on_match_finished)
        self._aa_worker.failed.connect(self._on_match_failed)
        # Live-preview relays (signal→signal for the stream; bound
        # method for the end marker — finished_with/failed have
        # incompatible signatures with the zero-arg match_ended).
        self._aa_worker.preview_results.connect(self.preview_results)
        self._aa_worker.finished_with.connect(self._emit_match_ended)
        self._aa_worker.failed.connect(self._emit_match_ended)

        # Live convergence plot dialog -- subscribes to the same
        # progress signal as the status line.  Stays open after the
        # match finishes so the user can review the trace; closing it
        # mid-run does NOT cancel the match (Stop button does that).
        from linac_gen_gui.interphase.dialogs.match_convergence import (
            MatchConvergenceDialog,
        )
        # Close any prior dialog before opening a new one (e.g. on a
        # second Match click during the same session).
        if getattr(self, "_aa_convergence_dlg", None) is not None:
            try:
                self._aa_convergence_dlg.close()
                # Release the old dialog (and its matplotlib Figure/canvas +
                # redraw QTimer) instead of leaking one per Match click.
                self._aa_convergence_dlg.deleteLater()
            except Exception:    # noqa: BLE001
                pass
        self._aa_convergence_dlg = MatchConvergenceDialog(
            algo=self._aa_algorithm.currentText(),
            popsize=int(self._aa_cmaes_popsize.value()),
            max_iter=int(self._aa_iter.value()),
            parent=self,
        )
        # Multicast worker signals into the dialog.
        self._aa_worker.progress.connect(
            self._aa_convergence_dlg.append_point)
        self._aa_worker.popsize_known.connect(
            self._aa_convergence_dlg.set_popsize)
        # Live emit/energy/element-being-scanned panel (sequential_scan
        # populates the per-element bits; other algorithms emit emit/
        # energy only).
        self._aa_worker.progress_detail.connect(
            self._aa_convergence_dlg.update_detail)
        self._aa_convergence_dlg.show()

        self._aa_worker.start()

    def shutdown_begin(self) -> list:
        """App teardown: request interruption of a running match and
        hand the worker to the window's bounded-wait loop.  A match
        mid-cost-evaluation can take ~30 s to notice — the caller's
        deadline, not this method, decides how long to wait."""
        out: list = []
        w = getattr(self, "_aa_worker", None)
        if w is not None and w.isRunning():
            w.requestInterruption()
            out.append(w)
        # Parked superseded matches (already interruption-requested at park
        # time) must also be handed to the bounded-wait loop, or interpreter
        # teardown destroys a live QThread → qFatal abort on quit.
        for z in list(getattr(self, "_zombie_workers", [])):
            if z is not None and z.isRunning():
                z.requestInterruption()
                out.append(z)
        # The multi-objective dialog owns its own Pareto worker the app's
        # teardown sweep doesn't otherwise see.
        mo = getattr(self, "_mo_dlg", None)
        mw = getattr(mo, "_worker", None) if mo is not None else None
        if mw is not None and mw.isRunning():
            mw.requestInterruption()
            out.append(mw)
        return out

    def _on_match_stop(self) -> None:
        """Cancel the running match.  The worker checks
        ``isInterruptionRequested`` in its progress callback and aborts
        at the next cost-function evaluation."""
        w = self._aa_worker
        if w is None or not w.isRunning():
            return
        w.requestInterruption()
        self._aa_stop.setEnabled(False)
        # Surface a hint -- the actual FAILED status comes from the
        # worker's `failed` emission once the running eval finishes.
        algo = self._aa_algorithm.currentText()
        sc = "with SC" if self._aa_sc.isChecked() else "no SC"
        self._aa_status.setText(
            f"Stopping ({algo}, {sc}) — waiting for current "
            f"evaluation to finish..."
        )

    def _on_match_popsize_known(self, popsize: int) -> None:
        """Worker reports the CMA-ES population size so the status
        label can render `gen N / eval M`."""
        self._aa_popsize = int(popsize)
        self._refresh_match_status()

    def _on_match_progress(self, n_iter: int, cost: float) -> None:
        # Remember the most recent worker-reported iter / cost so the
        # 500 ms ticker can paint elapsed-time updates between them.
        self._aa_last_iter = int(n_iter)
        self._aa_last_cost = float(cost)
        self._refresh_match_status()

    def _refresh_match_status(self) -> None:
        """Refresh the live status text from cached iter/cost + clock."""
        import time
        if self._aa_start_time <= 0:
            return
        algo = self._aa_algorithm.currentText()
        sc = "with SC" if self._aa_sc.isChecked() else "no SC"
        elapsed = time.time() - self._aa_start_time
        if self._aa_last_cost == float("inf"):
            cost_txt = "—"
        else:
            cost_txt = f"{self._aa_last_cost:.4e}"
        # When CMA-ES has reported its popsize, render `gen N · eval M`
        # so the user can map the eval counter onto their `--max-iter`
        # setting (which counts generations, not evals).  Otherwise
        # fall back to the legacy "iter N" text.
        if self._aa_popsize > 0:
            gen = self._aa_last_iter // self._aa_popsize
            counter_txt = (f"gen {gen} · eval {self._aa_last_iter}"
                           f" (popsize {self._aa_popsize})")
        else:
            counter_txt = f"iter {self._aa_last_iter}"
        self._aa_status.setText(
            f"Matching ({algo}, {sc}) — {counter_txt}  "
            f"cost={cost_txt}  elapsed {elapsed:6.1f} s"
        )

    def _on_match_finished(self, lattice_after, beam_after, result) -> None:
        self._aa_tick.stop()
        self._aa_progress.setVisible(False)
        self._aa_start_time = 0.0
        self._aa_run.setEnabled(True)
        self._aa_stop.setEnabled(False)
        self._aa_apply.setEnabled(True)
        self._aa_save.setEnabled(True)
        self._aa_result = (lattice_after, beam_after, result)

        # Populate variable table
        self._aa_var_table.setRowCount(len(result.variables))
        for i, (var, x0, xf) in enumerate(zip(
                result.variables, result.x0, result.x_final)):
            # If the variable is a follower in a link group (different
            # column index), still display the linked column's value.
            self._aa_var_table.setItem(i, 0, QTableWidgetItem(var.label))
            self._aa_var_table.setItem(i, 1, QTableWidgetItem(f"{x0:.6g}"))
            self._aa_var_table.setItem(i, 2, QTableWidgetItem(f"{xf:.6g}"))
            self._aa_var_table.setItem(
                i, 3, QTableWidgetItem(f"[{var.vmin:.4g}, {var.vmax:.4g}]"))

        # Populate constraint table
        self._aa_con_table.setRowCount(len(result.constraints))
        import math
        for i, c in enumerate(result.constraints):
            res = result.per_constraint_residuals.get(c.label)
            if res is None or len(res) == 0:
                rms = 0.0
            else:
                rms = float((res * res).mean()) ** 0.5
            self._aa_con_table.setItem(i, 0, QTableWidgetItem(c.label))
            self._aa_con_table.setItem(
                i, 1, QTableWidgetItem(f"{rms:.4e}"))
            self._aa_con_table.setItem(i, 2, QTableWidgetItem(c.notes or ""))

        # User-initiated cancel now flows through this success path
        # (engine catches StopIteration internally and returns a
        # MatchResult with success=False and "cancelled by user" in
        # the message); distinguish the three outcomes in the status
        # label so the user sees what happened.
        cancelled = "cancelled by user" in (result.message or "")
        if cancelled:
            status_prefix = "Cancelled — best of run"
        elif result.success:
            status_prefix = "OK"
        else:
            status_prefix = "FAILED"
        # Show baseline → final cost when the engine reports a baseline
        # (every algorithm runs an explicit baseline pass at x0 with the
        # chosen cost_solver since dd6614b).  Lets the user see at a
        # glance whether the matcher actually improved over x0.
        baseline = getattr(result, "baseline_cost", None)
        if baseline is not None:
            cost_str = f"cost {baseline:.4e} → {result.cost:.4e}"
        else:
            cost_str = f"cost {result.cost:.4e}"
        self._aa_status.setText(
            f"{status_prefix} — "
            f"{result.n_iter} iters, {cost_str}, "
            f"{result.elapsed_s:.2f}s"
        )

    def _invalidate_match_cache(self, *_args) -> None:
        """Drop any stale ``MatchResult`` and reset Apply/Save state.

        Wired to ``state.lattice_changed`` so editing a SET/ADJUST card
        via the inspector or reordering elements clears the previous
        result.  Without this, the Variable/Constraint tables and the
        Apply button would point at a deep-copy taken before the edit.
        """
        if getattr(self, "_applying_match", False):
            # Apply is installing the matched lattice itself — keep the result
            # valid so Save still works afterwards (see _on_match_apply).
            return
        if self._aa_result is None:
            return
        self._aa_result = None
        self._aa_apply.setEnabled(False)
        self._aa_save.setEnabled(False)
        self._aa_var_table.setRowCount(0)
        self._aa_con_table.setRowCount(0)
        self._aa_status.setText(
            "Lattice changed since last match — click Match again."
        )

    def _emit_match_ended(self, *_args) -> None:
        """Relay any match termination (finished/failed/stopped) as the
        zero-arg ``match_ended`` — consumed by the Results tab to end
        live previews.  Stale-worker guard: superseded workers had their
        signals disconnected wholesale, so only the live worker fires."""
        self.match_ended.emit()

    def _on_match_failed(self, message: str) -> None:
        self._aa_tick.stop()
        self._aa_progress.setVisible(False)
        self._aa_start_time = 0.0
        self._aa_run.setEnabled(True)
        self._aa_stop.setEnabled(False)
        self._aa_apply.setEnabled(False)
        self._aa_save.setEnabled(False)
        # A user-initiated cancel is also a "failure" path in the worker
        # -- distinguish in the surfaced text but don't pop a critical
        # dialog for it (the user knows they clicked Stop).
        cancelled = "cancelled by user" in message
        self._aa_status.setText(
            "Cancelled by user" if cancelled else f"FAILED — {message}"
        )
        if not cancelled:
            QMessageBox.critical(self, "Matching error", message)

    def _on_match_apply(self) -> None:
        if not self._aa_result:
            return
        lattice_after, beam_after, _ = self._aa_result
        # Mutating state.lattice in place isn't safe (other panels hold
        # references); push the matched copy via the state setter.
        try:
            # Keep the on-disk path: the matched lattice is the SAME file,
            # just with new parameter values.  Dropping it turned Save
            # into Save-As and the titlebar into "(no lattice loaded)".
            path = self.state.lattice_path
            # Suppress _invalidate_match_cache for THIS lattice change — the
            # matched lattice is exactly what Apply is installing, so the
            # result must stay valid so the user can still Save it afterwards.
            # (Without this, set_lattice()'s lattice_changed cleared
            # self._aa_result and disabled Save the instant Apply ran.)
            self._applying_match = True
            try:
                self.state.set_lattice(lattice_after, path)
            finally:
                self._applying_match = False
            self.state.set_beam_config(beam_after)
            # Push back into the BeamTab form so the user sees the values.
            if hasattr(self._beam_tab, "set_beam_config"):
                self._beam_tab.set_beam_config(beam_after)
            # set_lattice() reset() the undo stack and cleared bus.dirty,
            # but the matched lattice DOES differ from the on-disk .dat,
            # so re-flag it as dirty so the user gets prompted to save
            # before quitting / loading another lattice.  Same for the
            # project: beam config + matched lattice changed.
            if hasattr(self.state, "bus"):
                self.state.bus.mark_dirty()
            self.state.mark_project_dirty()
            # The lattice now carries FITTED values the source file does
            # not — plain Save must reroute to Save-As (set AFTER
            # set_lattice, which resets the flag).
            self.state.lattice_fitted = True
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, "Apply error", str(exc))
            return
        self._aa_status.setText("Applied — lattice + beam updated. "
                                "Save the .dat / project to persist.")

    def _on_match_save(self) -> None:
        if not self._aa_result:
            return
        lattice_after, _, _ = self._aa_result
        path, _filt = QFileDialog.getSaveFileName(
            self, "Save matched .dat", "", "TraceWin .dat (*.dat)"
        )
        if not path:
            return
        try:
            from linac_gen.io.tracewin_writer import write_tracewin
            write_tracewin(lattice_after, path)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, "Save error", str(exc))
            return
        self._aa_status.setText(f"Wrote {path}")

    def _open_multiobjective(self) -> None:
        """Open the multi-objective (Pareto) design dialog."""
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first.")
            return
        try:
            from linac_gen_gui.interphase.dialogs.multiobjective_dialog import (
                MultiObjectiveDialog,
            )
            # Release the previous instance if it isn't mid-run (it would
            # otherwise pile up parented-forever with its signal wiring);
            # if a Pareto run is still active, keep the old dialog alive —
            # deleting it would drop the only reference to a live QThread.
            old = getattr(self, "_mo_dlg", None)
            if old is not None:
                ow = getattr(old, "_worker", None)
                if ow is None or not ow.isRunning():
                    try:
                        old.close()
                        old.deleteLater()
                    except Exception:    # noqa: BLE001
                        pass
            self._mo_dlg = MultiObjectiveDialog(
                self.state, self._beam_tab, parent=self)
            self._mo_dlg.show()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Multi-objective error", str(exc))

    def _open_matching(self) -> None:  # noqa: D401 — keep original signature
        if self.state.lattice is None:
            QMessageBox.warning(self, "No lattice", "Load a lattice first."); return
        try:
            # Make sure BeamTab's current form values are committed
            cfg_before = self._beam_tab.get_beam_config()
            self.state.set_beam_config(cfg_before)

            from linac_gen_gui.dialogs.matching_dialog import MatchingDialog
            dlg = MatchingDialog(self.state.lattice, self._beam_tab, parent=self)
            dlg.exec()
            # After the dialog closes, the beam config may have changed
            # because the dialog's "Apply to Beam Setup" button mutates
            # alpha_x / beta_x / alpha_y / beta_y on the BeamTab.  Re-
            # collect, compare to the pre-dialog snapshot, and push +
            # mark dirty if anything actually changed.  Without the
            # dirty flag, closing or loading another project would skip
            # the unsaved-changes prompt and the matched Twiss would
            # silently be lost.
            cfg_after = self._beam_tab.get_beam_config()
            self.state.set_beam_config(cfg_after)
            twiss_keys = ("alpha_x", "beta_x", "alpha_y", "beta_y",
                          "alpha_z", "beta_z")
            twiss_changed = any(
                abs(getattr(cfg_after, k, 0.0) - getattr(cfg_before, k, 0.0))
                > 1e-12
                for k in twiss_keys
            )
            if twiss_changed:
                self.state.mark_project_dirty()
                self._status.setText(
                    "closed — beam config updated from dialog "
                    "(project flagged dirty)"
                )
            else:
                self._status.setText(
                    "closed — beam config unchanged"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Matching error", str(exc))


# ---------------------------------------------------------------------------
# Small formatters used by the Phase-Advance KPI cards.  Pulled out of the
# class so they can be unit-tested in isolation without spinning up Qt.
# ---------------------------------------------------------------------------
def _fmt_deg(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ratio(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Match worker — runs ``linac_gen.matching.match`` off the GUI thread.
# ---------------------------------------------------------------------------
class _MatchWorker(QThread):
    """Background matcher.

    Emits ``progress(iter, cost)`` per residual evaluation, and
    ``finished_with(lattice, beam, MatchResult)`` (or ``failed(str)`` on
    exception) on completion.
    """

    progress = pyqtSignal(int, float)
    # Rich per-eval payload (live GUI display): emit_nx/ny/nz at end of
    # line, W_kin out, currently-scanned element + parameter, pass / step
    # counters.  Sequential_scan populates the per-element bits; other
    # algorithms emit emit/energy data only.  Dict carries:
    #   iter, cost, best_cost, emit_nx_out, emit_ny_out, emit_nz_out,
    #   w_kin_out, and (for sequential_scan only): pass, total_passes,
    #   step, total_steps, element_name, attrs, x_values, direction.
    progress_detail = pyqtSignal(dict)
    # Emitted once after CMA-ES starts, so the GUI can render the eval
    # counter as `gen N / eval M`.  Other algorithms never emit this
    # and the GUI falls back to the legacy `iter N` label.
    popsize_known = pyqtSignal(int)
    # Live match preview: the full forward-sim result of the current
    # evaluation (DiagnosticRecorder for cost_solver="mp", else
    # EnvelopeResults) + iteration index, throttled to ~1 emission/s.
    # Consumers must treat the object as READ-ONLY (the engine builds a
    # fresh one per eval and never mutates an emitted instance).
    preview_results = pyqtSignal(object, int)
    finished_with = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, lattice, beam_cfg, *, space_charge: bool,
                 max_iter: int, algorithm: str = "least_squares",
                 cmaes_sigma: float = 0.2,
                 cmaes_popsize: int = 0,
                 refine: bool = True,
                 bo_prior: bool = False,
                 seqscan_settings: dict | None = None,
                 cost_solver: str = "envelope",
                 mp_n_particles: int = 1000,
                 mp_sc_config=None,
                 mp_step_config=None,
                 allow_inert_constraints: bool = False,
                 parent=None):
        super().__init__(parent)
        # Qt's default QThread stack is only ~544 KB on macOS, which is
        # not enough headroom for OpenBLAS's `dgetrf_parallel` workspace
        # when the matcher runs many `np.linalg.inv` calls per residual
        # (FieldMap3D RK4 with deep substeps blows past the guard page
        # and the worker dies with SIGBUS "Thread stack size exceeded").
        # 16 MB matches the main-thread stack and is plenty for any
        # numpy/scipy call chain we can plausibly hit here.
        self.setStackSize(16 * 1024 * 1024)
        self._lattice = lattice
        self._beam_cfg = beam_cfg
        self._space_charge = space_charge
        self._last_preview_t = 0.0    # monotonic clock of last preview emit
        self._algorithm = algorithm
        self._max_iter = max_iter
        self._cmaes_sigma = float(cmaes_sigma)
        self._cmaes_popsize = int(cmaes_popsize)
        self._refine = bool(refine)
        self._bo_prior = bool(bo_prior)
        # sequential_scan settings -- only consulted when algorithm ==
        # "sequential_scan"; otherwise an empty dict that match() ignores.
        self._seqscan_settings = dict(seqscan_settings or {})
        # MP cost-solver settings.  When cost_solver == "envelope" the
        # mp_* values are ignored by the engine.
        self._cost_solver = str(cost_solver)
        self._mp_n_particles = int(mp_n_particles)
        self._mp_sc_config = mp_sc_config
        self._mp_step_config = mp_step_config
        self._allow_inert_constraints = bool(allow_inert_constraints)

    def _maybe_emit_preview(self, lr, it, now: float = None) -> bool:
        """Emit a live-preview iterate at most once per second.

        ``now`` is injectable for tests; production passes None →
        ``time.monotonic()``.  Returns whether an emission happened.
        """
        if now is None:
            now = time.monotonic()
        if now - self._last_preview_t < 1.0:
            return False
        self._last_preview_t = now
        self.preview_results.emit(lr, int(it))
        return True

    def run(self) -> None:                   # noqa: D401 — QThread entry
        # OpenBLAS's parallel LU (`dgetrf_parallel`) allocates a large
        # workspace on the calling thread's stack and SIGBUSes any
        # QThread (default 544 KB stack on macOS).  Env vars set at the
        # top of linac_gen_gui.interphase.__main__ already cap BLAS
        # thread counts at process start; this runtime override via
        # threadpoolctl is a second line of defense for users who
        # bypass the launcher (e.g. import the tab directly).
        try:
            from threadpoolctl import threadpool_limits
            _blas_limit = threadpool_limits(limits=1)
        except ImportError:
            _blas_limit = None
        try:
            from linac_gen.matching import match
            from linac_gen.elements.lattice_commands import (
                Adjust, AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
                AdjustBeamTwiss, AdjustBeamCentroid, AdjustBeamEmit,
                AdjustBeamCurrent, LatticeCommand, SetSize, SetSizeMax,
                SetSizeMin, SetTwiss, SetBeamPhaseAdv,
            )
            # ---- diagnostic dump (stderr) -----------------------------
            # Helps debug "matched value doesn't match what I expect"
            # by printing exactly what the matcher saw.  Goes to the
            # terminal that launched the GUI; never blocks the run.
            import sys
            cfg = self._beam_cfg
            print("[match] === incoming match request ===", file=sys.stderr)
            print(f"[match] beam: species={cfg.species} energy={cfg.energy} "
                  f"freq={cfg.frequency} current={cfg.current} "
                  f"distribution={cfg.distribution}", file=sys.stderr)
            print(f"[match]   X: alpha={cfg.alpha_x} beta={cfg.beta_x} "
                  f"emit_n={cfg.emit_nx}  mismatch={cfg.mismatch_x}",
                  file=sys.stderr)
            print(f"[match]   Y: alpha={cfg.alpha_y} beta={cfg.beta_y} "
                  f"emit_n={cfg.emit_ny}  mismatch={cfg.mismatch_y}",
                  file=sys.stderr)
            print(f"[match]   Z: alpha={cfg.alpha_z} beta={cfg.beta_z} "
                  f"emit_z={cfg.emit_z}  mismatch={cfg.mismatch_z}",
                  file=sys.stderr)
            print(f"[match] flags: SC={self._space_charge} "
                  f"algorithm={self._algorithm} "
                  f"max_iter={self._max_iter}", file=sys.stderr)
            print(f"[match] lattice: {len(self._lattice.elements)} elements",
                  file=sys.stderr)
            for j, elem in enumerate(self._lattice.elements):
                tn = type(elem).__name__
                if isinstance(elem, LatticeCommand):
                    print(f"[match]   [{j:>3d}] {elem.name:<24s} "
                          f"{tn:<22s} args={elem.to_tracewin_args()}",
                          file=sys.stderr)
                elif tn in ("Quadrupole", "Solenoid", "Dipole",
                            "Drift", "Steerer"):
                    grad = (getattr(elem, "gradient", None)
                            or getattr(elem, "field", None)
                            or getattr(elem, "angle", None))
                    print(f"[match]   [{j:>3d}] {elem.name:<24s} "
                          f"{tn:<22s} L={getattr(elem,'length',0):.3g} "
                          f"param={grad}", file=sys.stderr)

            # Compute / emit popsize once, before the worker starts the
            # CMA-ES loop (only meaningful for cmaes; harmless otherwise).
            # If the user explicitly set popsize in the GUI, use that;
            # otherwise compute the cma library's default
            # `4 + floor(3 * ln(N))` from the variable list.  Emitting
            # up-front lets the gen counter render correctly from iter 1.
            popsize_emitted = {"done": False}
            if self._algorithm == "cmaes":
                try:
                    if self._cmaes_popsize > 0:
                        _popsize = int(self._cmaes_popsize)
                        _ncols = -1
                    else:
                        from linac_gen.matching.variables import collect_variables
                        from linac_gen.matching.engine import _link_group_index
                        _vars = collect_variables(self._lattice, self._beam_cfg)
                        _cols, _ncols = _link_group_index(_vars)
                        _ncols = max(int(_ncols), 1)
                        _popsize = 4 + int(math.floor(3.0 * math.log(_ncols)))
                    self.popsize_known.emit(_popsize)
                    popsize_emitted["done"] = True
                    print(f"[match] cmaes popsize = {_popsize}"
                          + (f"  (N_vars = {_ncols})" if _ncols > 0
                             else "  (user override)"),
                          file=sys.stderr)
                except Exception as exc:    # noqa: BLE001
                    print(f"[match] could not pre-compute popsize: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)

            def cb(it, x, cost, info=None):
                # User-initiated cancel: the GUI calls
                # ``QThread.requestInterruption`` on Stop click; the
                # engine has no notion of cancellation, so we surface
                # it as an exception that the outer try/except routes
                # to ``self.failed.emit(...)``.
                if self.isInterruptionRequested():
                    raise StopIteration("cancelled by user")
                self.progress.emit(int(it), float(cost))
                # Rich per-eval payload for the live GUI (emit/energy/
                # current element).  The engine fills `info` for callers
                # that accept a 4-arg signature; it's None for callers
                # that wired a 3-arg cb (test fixtures), in which case
                # we just skip the detail emission.
                if info is not None:
                    # Live preview: ship the full per-eval results object
                    # (throttled — a tiny-lattice eval can be sub-ms and
                    # flooding the GUI queue would starve the event loop).
                    lr = info.get("results")
                    if lr is not None:
                        try:
                            self._maybe_emit_preview(lr, it)
                        except Exception:    # noqa: BLE001
                            pass
                    try:
                        # Keep the detail payload scalar-only: existing
                        # consumers (status line, convergence dialog)
                        # must not receive the big results object.
                        detail = {k: v for k, v in info.items()
                                  if k != "results"}
                        self.progress_detail.emit(detail)
                    except Exception:    # noqa: BLE001
                        pass
                if it <= 5 or it % 10 == 0:
                    print(f"[match]   iter {it:>3d}  x={x.tolist()}  "
                          f"cost={cost:.3e}", file=sys.stderr)

            result = match(
                self._lattice, self._beam_cfg,
                space_charge=self._space_charge,
                algorithm=self._algorithm,
                max_iter=self._max_iter,
                cmaes_sigma=self._cmaes_sigma,
                cmaes_popsize=self._cmaes_popsize,
                refine=self._refine,
                bo_prior=self._bo_prior,
                cost_solver=self._cost_solver,
                allow_inert_constraints=self._allow_inert_constraints,
                mp_n_particles=self._mp_n_particles,
                mp_sc_config=self._mp_sc_config,
                mp_step_config=self._mp_step_config,
                callback=cb,
                **self._seqscan_settings,  # empty unless sequential_scan
            )

            print(f"[match] result: success={result.success} "
                  f"cost={result.cost:.3e} iters={result.n_iter}",
                  file=sys.stderr)
            print(f"[match]   x0      = {result.x0.tolist()}",
                  file=sys.stderr)
            print(f"[match]   x_final = {result.x_final.tolist()}",
                  file=sys.stderr)
            for i, var in enumerate(result.variables):
                print(f"[match]   var[{i}] {var.label:<28s} "
                      f"link={var.link_group} bounds=[{var.vmin}, {var.vmax}]"
                      f"  target_now={getattr(var.target, var.attr)}",
                      file=sys.stderr)
            for c in result.constraints:
                res = result.per_constraint_residuals.get(c.label)
                if res is None:
                    rms = float("nan")
                else:
                    import numpy as _np
                    rms = float(_np.sqrt(_np.mean(_np.asarray(res) ** 2)))
                print(f"[match]   constraint {c.label:<28s} rms={rms:.3e}",
                      file=sys.stderr)
        except Exception as exc:             # noqa: BLE001
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        finally:
            # Release the threadpoolctl BLAS limit (if we set one).
            # threadpoolctl renamed `restore()` -> `restore_original_limits()`;
            # the legacy name silently AttributeErrored under the bare
            # except and leaked the 1-thread cap for the GUI's lifetime.
            if _blas_limit is not None:
                try:
                    _blas_limit.restore_original_limits()
                except Exception:    # noqa: BLE001
                    pass
        self.finished_with.emit(self._lattice, self._beam_cfg, result)
