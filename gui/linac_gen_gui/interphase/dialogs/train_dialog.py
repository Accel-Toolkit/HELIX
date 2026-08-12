"""Multibunch / pulse-study dialogs (Simulate → Multibunch / Pulse Study…).

``TrainConfigDialog`` is the GUI's OPT-IN surface for the rarely-run
multibunch study (plan §3b): the study switch is OFF by default and the
whole parameter form is inert until it is enabled.  OK builds a real
:class:`linac_gen.train.TrainConfig` and reports every validation
refusal IN-DIALOG (red label, verbatim message — never a modal box, so
the offscreen test path can exercise refusals safely).  Physics inputs
are never defaulted: enabling beam loading / HOMs reveals the cavity
sidecar picker, and a missing sidecar is refused listing exactly what is
missing.

``TrainSummaryDialog`` is the v1 results surface: a modeless per-bunch
summary plot (no dedicated tab), fed a TrainResults by the app when a
run (or a mid-train abort's partial result) lands.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme

_MODE_HINTS = {
    "mp": "tracked — one full multiparticle pass per bunch "
          "(10s of bunches)",
    "envelope": "one envelope solve per bunch (no beam loading / HOM "
                "channels)",
    "fast": "per-slot phasor recursion — full ~10⁵-slot pulses in "
            "seconds (centroid energy ledger)",
    "hybrid": "fast pass over the whole pulse + full-MP replay of "
              "selected bunches",
}


class TrainConfigDialog(QDialog):
    """Modal setup for the opt-in multibunch / pulse study.

    After ``exec()`` returns True, ``train_config`` holds the validated
    :class:`~linac_gen.train.TrainConfig` and ``space_charge_enabled``
    says whether the tracked passes should use the Numerics-tab SC
    settings.
    """

    def __init__(self, default_freq_MHz: float = 162.5, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multibunch / Pulse Study")
        self.setMinimumWidth(560)
        self.train_config = None
        self.space_charge_enabled = True

        lay = QVBoxLayout(self)

        intro = QLabel(
            "Simulate a pulse: a train of bunches at the bunch frequency "
            "with a definable chopped fill pattern, coupled bunch-to-"
            "bunch through cavity beam loading, dipole-HOM wakes and/or "
            "direct neighbour space charge.  This is an opt-in study — "
            "with it disabled (and with every physics channel off) HELIX "
            "behaves exactly as a single-bunch run.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{theme.TEXT_2};")
        lay.addWidget(intro)

        # ---- THE opt-in switch (off by default; form inert until on) --
        self._enable = QCheckBox("Enable the multibunch / pulse study")
        self._enable.setChecked(False)
        self._enable.toggled.connect(self._on_enable_toggled)
        lay.addWidget(self._enable)

        self._form_box = QWidget()
        form_lay = QVBoxLayout(self._form_box)
        form_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._form_box)

        # ---- run mode + bunch clock ----------------------------------
        run_group = QGroupBox("Train")
        form = QFormLayout(run_group)
        self._mode = QComboBox()
        self._mode.addItems(["mp", "envelope", "fast", "hybrid"])
        self._mode.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Mode", self._mode)
        self._mode_hint = QLabel(_MODE_HINTS["mp"])
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setStyleSheet(f"color:{theme.TEXT_2};")
        form.addRow("", self._mode_hint)

        self._freq = QDoubleSpinBox()
        self._freq.setRange(1e-3, 1e5)
        self._freq.setDecimals(4)
        self._freq.setSuffix(" MHz")
        self._freq.setValue(float(default_freq_MHz))
        self._freq.setToolTip(
            "Bunch-slot rate (the RF bunch frequency).  Must agree with "
            "the beam's bunch_frequency_MHz when that is set.")
        # The preview's pulse-length line depends on the frequency too.
        self._freq.valueChanged.connect(self._refresh_pattern_preview)
        form.addRow("Bunch frequency", self._freq)

        self._pattern = QLineEdit("1*10")
        self._pattern.setPlaceholderText("e.g. 1*10 0*54 1*26")
        self._pattern.setToolTip(
            "Run-length-encoded fill pattern over the slot axis: "
            "value*count tokens, 1 = bunch present, 0 = chopped slot.")
        self._pattern.textChanged.connect(self._refresh_pattern_preview)
        form.addRow("Pattern (RLE)", self._pattern)
        self._pattern_preview = QLabel("")
        self._pattern_preview.setWordWrap(True)
        form.addRow("", self._pattern_preview)

        # Live pulse-structure strip: the fill pattern drawn against
        # time in the pulse, redrawn as the RLE / frequency change
        # (user request 2026-08-11: "can I see the bunches plot with
        # time, that is the pulse, somewhere?").
        import pyqtgraph as pg

        from linac_gen_gui.interphase.plots.plot_style import style_plot
        self._pattern_plot = pg.PlotWidget()
        style_plot(self._pattern_plot, "", "")
        self._pattern_plot.setFixedHeight(96)
        self._pattern_plot.setLabel("bottom", "time in pulse [µs]")
        self._pattern_plot.hideAxis("left")          # y is just 0/1 fill
        self._pattern_plot.setMouseEnabled(x=True, y=False)
        form.addRow("", self._pattern_plot)

        self._keep_full = QCheckBox(
            "Keep full per-bunch results (uncheck: summary-only, for "
            "big tracked trains)")
        self._keep_full.setChecked(True)
        form.addRow(self._keep_full)
        form_lay.addWidget(run_group)

        # ---- physics channels (all OFF; inputs revealed on demand) ----
        phys_group = QGroupBox("Bunch-coupling physics (all off = "
                               "independent bunches, bit-identical)")
        pf = QVBoxLayout(phys_group)
        self._loading = QCheckBox("Cavity beam loading (fundamental "
                                  "mode)")
        self._loading.toggled.connect(self._refresh_reveals)
        pf.addWidget(self._loading)
        self._hom = QCheckBox("Dipole-HOM long-range wakes / cumulative "
                              "BBU")
        self._hom.toggled.connect(self._refresh_reveals)
        pf.addWidget(self._hom)

        self._sidecar_row = QWidget()
        srow = QHBoxLayout(self._sidecar_row)
        srow.setContentsMargins(20, 0, 0, 0)
        self._sidecar_label = QLabel("Cavity parameters (sidecar)")
        srow.addWidget(self._sidecar_label)
        self._sidecar = QLineEdit()
        self._sidecar.setPlaceholderText(
            "cavity_params.yaml / .json — R/Q, Q_L, detuning, hom_modes")
        srow.addWidget(self._sidecar, stretch=1)
        self._sidecar_browse = QPushButton("Browse…")
        self._sidecar_browse.clicked.connect(self._pick_sidecar)
        srow.addWidget(self._sidecar_browse)
        pf.addWidget(self._sidecar_row)

        self._direct_sc = QCheckBox("Direct bunch-to-bunch space charge "
                                    "(PIC ±1 neighbour images; mode "
                                    "mp/hybrid, numpy PIC)")
        self._direct_sc.toggled.connect(self._refresh_reveals)
        pf.addWidget(self._direct_sc)
        self._dsc_row = QWidget()
        drow = QHBoxLayout(self._dsc_row)
        drow.setContentsMargins(20, 0, 0, 0)
        drow.addWidget(QLabel("Neighbours"))
        self._dsc_neighbors = QComboBox()
        self._dsc_neighbors.addItems(["images", "distinct"])
        self._dsc_neighbors.setToolTip(
            "images: pattern-scaled copies of the live bunch (Toutatis-"
            "validated machinery).  distinct: leading neighbour is the "
            "previously tracked bunch's snapshot.")
        drow.addWidget(self._dsc_neighbors)
        self._dsc_force = QCheckBox("Force engagement (model neighbours "
                                    "regardless of bunch length)")
        drow.addWidget(self._dsc_force)
        drow.addStretch(1)
        pf.addWidget(self._dsc_row)
        form_lay.addWidget(phys_group)

        # ---- numerics / hybrid ---------------------------------------
        opt_group = QGroupBox("Numerics · Hybrid replay")
        of = QFormLayout(opt_group)
        self._sc_check = QCheckBox("In-bunch space charge (Numerics-tab "
                                   "settings)")
        self._sc_check.setChecked(True)
        of.addRow(self._sc_check)
        self._select = QLineEdit()
        self._select.setPlaceholderText(
            "auto  (or absolute slot indices, e.g. 0, 12, 88)")
        self._select.setToolTip(
            "Hybrid mode only: which bunches pass 2 replays full-MP.")
        of.addRow("Replay bunches", self._select)
        form_lay.addWidget(opt_group)

        # ---- in-dialog validation output -----------------------------
        self._error = QLabel("")
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color:{theme.ERR};")
        self._error.setVisible(False)
        lay.addWidget(self._error)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._validate_and_accept)
        self._buttons.rejected.connect(self.reject)
        lay.addWidget(self._buttons)

        self._on_enable_toggled(False)
        self._refresh_pattern_preview()
        self._refresh_reveals()
        self._on_mode_changed(self._mode.currentText())

    # ------------------------------------------------------------------
    def _ok_button(self):
        return self._buttons.button(QDialogButtonBox.StandardButton.Ok)

    def _on_enable_toggled(self, on: bool) -> None:
        self._form_box.setEnabled(bool(on))
        ok = self._ok_button()
        ok.setEnabled(bool(on))
        ok.setToolTip("" if on else
                      "Enable the multibunch study first — it is "
                      "strictly opt-in.")
        if not on:
            self._error.setVisible(False)

    def _on_mode_changed(self, mode: str) -> None:
        self._mode_hint.setText(_MODE_HINTS.get(mode, ""))
        self._select.setEnabled(mode == "hybrid")
        # Envelope mode always applies linear space charge at the beam
        # current — the PIC checkbox would be inert there.
        is_env = (mode == "envelope")
        self._sc_check.setEnabled(not is_env)
        self._sc_check.setToolTip(
            "Envelope mode always uses linear space charge at the beam "
            "current — this switch only affects tracked passes."
            if is_env else
            "Build the in-bunch PIC space charge from the Numerics tab "
            "for every tracked pass (uncheck = explicit off).")

    def _pick_sidecar(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Cavity parameters sidecar", "",
            "Cavity sidecar (*.yaml *.yml *.json);;All Files (*)")
        if fp:
            self._sidecar.setText(fp)

    def _refresh_reveals(self, *_a) -> None:
        need_sidecar = self._loading.isChecked() or self._hom.isChecked()
        self._sidecar_row.setEnabled(need_sidecar)
        self._sidecar_row.setVisible(True)
        self._dsc_row.setEnabled(self._direct_sc.isChecked())

    def _refresh_pattern_preview(self, *_a) -> None:
        from linac_gen.train import PulsePattern
        txt = self._pattern.text().strip()
        try:
            pat = PulsePattern.from_rle(txt)
        except ValueError as exc:
            self._pattern_preview.setStyleSheet(f"color:{theme.ERR};")
            self._pattern_preview.setText(str(exc))
            self._pattern_plot.clear()
            return
        f = float(self._freq.value())
        duty = 100.0 * pat.n_bunches / pat.n_slots
        self._pattern_preview.setStyleSheet(f"color:{theme.TEXT_2};")
        self._pattern_preview.setText(
            f"{pat.n_slots} slots · {pat.n_bunches} bunches "
            f"({duty:.1f}% duty) · pulse "
            f"{pat.pulse_length_us(f):.3g} µs @ {f:g} MHz")
        # Redraw the pulse strip (0/1 fill vs µs).  A polyline over the
        # full slot axis is fine for pyqtgraph even at ~90k slots.
        from linac_gen_gui.interphase.plots.plot_style import curve_pen
        y = pat.filled.astype(np.float64)
        x = np.arange(pat.n_slots, dtype=np.float64) / max(f, 1e-12)
        self._pattern_plot.clear()
        self._pattern_plot.plot(x, y, pen=curve_pen(theme.ACCENT))
        self._pattern_plot.setYRange(-0.1, 1.1, padding=0)

    # ------------------------------------------------------------------
    def _show_error(self, msg: str) -> None:
        self._error.setText(str(msg))
        self._error.setVisible(True)

    def build_train_config(self):
        """Build (and therefore loudly validate) the TrainConfig from
        the current form.  Raises ValueError with the exact refusal."""
        from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics
        from linac_gen.train.cavity_state import CavityStateRegistry

        pat = PulsePattern.from_rle(self._pattern.text().strip())
        loading = self._loading.isChecked()
        hom = self._hom.isChecked()
        mode = self._mode.currentText()
        if mode == "envelope" and (loading or hom):
            raise ValueError(
                "beam loading / HOMs are not wired for mode='envelope' "
                "(the envelope solver has no element hooks) — use mode "
                "'mp', 'fast' or 'hybrid'")
        sidecar = self._sidecar.text().strip() or None
        if not (loading or hom):
            # The picker row is disabled without a channel — leftover
            # text must not ride into the config as an unused input.
            sidecar = None
        if (loading or hom) and sidecar:
            if not os.path.isfile(sidecar):
                raise ValueError(
                    f"cavity_params sidecar not found: {sidecar}")
            # Parse now so format errors are refused here, in-dialog,
            # rather than mid-run in the worker.
            CavityStateRegistry.load_sidecar(
                sidecar, need_fundamental=loading, need_hom=hom)
        select = self._select.text().strip()
        kwargs = {}
        if mode == "hybrid" and select and select.lower() != "auto":
            try:
                kwargs["select_bunches"] = [
                    int(tok) for tok in select.replace(",", " ").split()]
            except ValueError:
                raise ValueError(
                    f"replay bunches {select!r}: expected 'auto' or "
                    "integer slot indices (comma/space separated)")
        return TrainConfig(
            bunch_frequency_MHz=float(self._freq.value()),
            pattern=pat, mode=mode,
            physics=TrainPhysics(
                direct_sc=self._direct_sc.isChecked(),
                beam_loading=loading, hom=hom),
            cavity_params=sidecar,
            keep_full_results=self._keep_full.isChecked(),
            direct_sc_neighbors=self._dsc_neighbors.currentText(),
            direct_sc_force_engage=self._dsc_force.isChecked(),
            **kwargs)

    def _validate_and_accept(self) -> None:
        if not self._enable.isChecked():        # belt & braces (OK is
            return                              # disabled when off)
        try:
            self.train_config = self.build_train_config()
        except (ValueError, TypeError, NotImplementedError) as exc:
            self._show_error(exc)
            return
        self.space_charge_enabled = self._sc_check.isChecked()
        self._error.setVisible(False)
        self.accept()


# ---------------------------------------------------------------------------
class TrainSummaryDialog(QDialog):
    """Modeless per-bunch summary of a multibunch run (v1: one plot).

    ``set_results(train_results)`` accepts a live TrainResults or the
    loader's namespace; quantities offered follow what the run actually
    recorded (fast ΔW ledger and/or the tracked per-bunch summary
    table).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multibunch summary")
        self.setMinimumSize(640, 420)
        self._results = None
        self._series: dict = {}

        lay = QVBoxLayout(self)
        self._header = QLabel("No multibunch results yet.")
        self._header.setWordWrap(True)
        lay.addWidget(self._header)

        # Live-progress row: shown by begin_run() the moment the study
        # starts (the config dialog closes on OK — without this the only
        # run feedback was the main-window status bar), hidden again by
        # set_results()/run_failed().
        self._prog_label = QLabel("")
        self._prog_label.setWordWrap(True)
        self._prog_label.setStyleSheet(f"color:{theme.TEXT_2};")
        lay.addWidget(self._prog_label)
        self._prog = QProgressBar()
        self._prog.setRange(0, 100)
        self._prog.setValue(0)
        lay.addWidget(self._prog)
        self._prog_label.setVisible(False)
        self._prog.setVisible(False)

        row = QHBoxLayout()
        row.addWidget(QLabel("Quantity"))
        self._quantity = QComboBox()
        self._quantity.currentTextChanged.connect(self._replot)
        row.addWidget(self._quantity, stretch=1)
        self._us_axis = QCheckBox("time axis [µs]")
        self._us_axis.setToolTip(
            "Plot against time in the pulse (slot / bunch frequency) "
            "instead of slot index.")
        self._us_axis.toggled.connect(self._replot)
        row.addWidget(self._us_axis)
        lay.addLayout(row)
        self._freq_MHz = 0.0

        import pyqtgraph as pg

        from linac_gen_gui.interphase.plots.plot_style import style_plot
        self._plot = pg.PlotWidget()
        style_plot(self._plot, "", "")
        self._plot.setLabel("bottom", "slot index")
        lay.addWidget(self._plot, stretch=1)

    # ------------------------------------------------------------------
    def begin_run(self, mode: str, n_bunches: int, n_slots: int) -> None:
        """Switch to live-progress mode at study launch.

        Called by the app right after the TrainWorker starts; the plot
        arrives via set_results() when the run (or its partial abort
        result) lands.
        """
        self._results = None
        self._series = {}
        self._quantity.blockSignals(True)
        self._quantity.clear()
        self._quantity.blockSignals(False)
        self._plot.clear()
        self._header.setText(
            f"Multibunch study RUNNING — mode={mode} · {n_bunches} "
            f"bunches over {n_slots} slots.")
        self._prog_label.setText(
            "Design pass first (cavity calibration at nominal voltage, "
            "one full single-bunch pass) — per-bunch progress follows…")
        self._prog.setRange(0, max(int(n_bunches), 1))
        self._prog.setValue(0)
        self._prog_label.setVisible(True)
        self._prog.setVisible(True)

    def on_progress(self, done: int, total: int) -> None:
        """Per-bunch progress from the TrainWorker (queued signal)."""
        if not self._prog.isVisibleTo(self):
            return                     # stale signal after results landed
        if self._prog.maximum() != int(total):
            self._prog.setRange(0, max(int(total), 1))
        self._prog.setValue(int(done))
        self._prog_label.setText(f"bunch {done}/{total} complete")

    def run_failed(self, msg: str) -> None:
        """Reflect a failed study (the app's shared handler owns the
        error box; this just stops the popup claiming it is running)."""
        self._prog.setVisible(False)
        self._prog_label.setVisible(False)
        self._header.setText(f"Multibunch study FAILED — {msg}")

    # ------------------------------------------------------------------
    def set_results(self, results) -> None:
        self._results = results
        self._series = {}
        self._prog.setVisible(False)
        self._prog_label.setVisible(False)
        if results is None:
            self._header.setText("No multibunch results yet.")
            self._quantity.clear()
            self._plot.clear()
            return
        live = hasattr(results, "config")
        cfg = results.config if live else results
        pat = cfg.pattern
        self._freq_MHz = float(getattr(cfg, "bunch_frequency_MHz", 0.0)
                               or 0.0)
        fast = getattr(results, "fast", None)
        truncated = bool(getattr(results, "truncated", False)
                         or (fast is not None
                             and getattr(fast, "truncated", False)))
        n_replay = len(getattr(results, "replay_bunches", {}) or {})
        head = (f"mode={results.mode} · {pat.n_slots} slots · "
                f"{pat.n_bunches} bunches in pattern · "
                f"{len(results.slots)} tracked"
                + (f" · {n_replay} replayed" if n_replay else ""))
        if truncated:
            head += ("   —  ABORTED mid-train: partial result "
                     "(loadable; per-bunch data covers the processed "
                     "prefix)")
        self._header.setText(head)

        # ---- fast ledger ---------------------------------------------
        if fast is not None and len(getattr(fast, "w_exit_MeV", ())):
            slots = np.asarray(fast.slot, float)
            w = np.asarray(fast.w_exit_MeV, float)
            wd = float(fast.w_design_exit_MeV)
            self._series["dW vs design [keV] (fast ledger)"] = (
                slots, (w - wd) * 1e3)
            self._series["W_exit [MeV] (fast ledger)"] = (slots, w)
        # ---- tracked summary table -----------------------------------
        summ = results.summary() if callable(
            getattr(results, "summary", None)) else (results.summary or {})
        s_slots = np.asarray(summ.get("slot", ()), float)
        units = {"ref_w_kin": "MeV", "sigma_x": "mm", "sigma_y": "mm",
                 "sigma_phi": "deg", "sigma_w": "MeV",
                 "emit_x": "mm·mrad", "emit_y": "mm·mrad",
                 "emit_z": "deg·MeV", "transmission": "%",
                 "mean_x": "mm", "mean_y": "mm", "mean_phi": "deg",
                 "mean_w": "MeV"}
        if s_slots.size:
            for key, unit in units.items():
                arr = np.asarray(summ.get(key, ()), float)
                if arr.size and np.isfinite(arr).any():
                    self._series[f"{key} [{unit}] (per bunch)"] = (
                        s_slots, arr)
        # ---- the pulse itself: fill pattern over the slot axis --------
        self._series["fill pattern (1 = bunch present)"] = (
            np.arange(pat.n_slots, dtype=float),
            pat.filled.astype(float))

        self._quantity.blockSignals(True)
        self._quantity.clear()
        self._quantity.addItems(list(self._series))
        self._quantity.blockSignals(False)
        self._replot()

    #: results_tab popup contract name — kept for API symmetry.
    refresh = set_results

    def _replot(self, *_a) -> None:
        self._plot.clear()
        name = self._quantity.currentText()
        series = self._series.get(name)
        if not series:
            return
        x, y = series
        use_us = bool(self._us_axis.isChecked()) and self._freq_MHz > 0
        if use_us:
            x = np.asarray(x, float) / self._freq_MHz
            self._plot.setLabel("bottom", "time in pulse [µs]")
        else:
            self._plot.setLabel("bottom", "slot index")
        from linac_gen_gui.interphase.plots.plot_style import curve_pen
        kwargs = dict(pen=curve_pen(theme.ACCENT))
        # Symbols only for short series — a full-pulse fast run (or the
        # 89k-slot fill pattern) as 10^4-10^5 scatter markers crawls.
        if len(x) <= 5000:
            kwargs.update(symbol="o", symbolSize=4,
                          symbolBrush=theme.ACCENT, symbolPen=None)
        self._plot.plot(x, y, **kwargs)
        label = name.split(" (")[0]
        self._plot.setLabel("left", label)
