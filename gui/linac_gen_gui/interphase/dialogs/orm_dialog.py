"""Orbit-response calibration dialog — Tools → Orbit-Response Calibration (LOCO)…

Load a measured orbit-response matrix (FORMA scan folders, one driven plane
each, or HELIX ORM CSV files), map its devices onto the session lattice
(Mapping tab, editable), compare with the model response (Compare tab),
fit quadrupole scale factors / trim calibrations / BPM gains (Fit tab) and
apply the result as ONE undoable edit or export a recalibrated deck
(Apply / Export tab).

Workers run on deep copies of the lattice and beam config; nothing touches
the live lattice off the GUI thread.  The apply step goes through the
command bus (``MacroCommand`` of ``ParamChangeCommand``), sets
``state.lattice_fitted`` (plain Save reroutes to Save-As) and refuses
atomically when a design gradient no longer matches the calibration.
"""
from __future__ import annotations

import copy
import os
import threading
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.app_settings import make_settings
from linac_gen_gui.interphase.commands import MacroCommand, ParamChangeCommand
from linac_gen_gui.interphase.plots.plot_style import style_plot

_STAGES = ("trims", "trims+quads", "trims+quads+bpms")
_PLANE_NAME = {"x": "horizontal (x)", "y": "vertical (y)"}


def _scalars(obj) -> tuple:
    """Hashable snapshot of the scalar attributes of an element / config (floats by repr: NaN-safe)."""
    out = []
    for k, v in sorted(vars(obj).items()):
        if isinstance(v, float):
            out.append((k, repr(v)))
        elif isinstance(v, (int, str, bool, type(None))):
            out.append((k, v))
    return tuple(out)


def _optics_fingerprint(lattice, beam_cfg) -> tuple:
    """Every scalar of every element plus the beam scalars — any optics edit while a worker runs invalidates its result."""
    els = tuple((type(e).__name__, _scalars(e)) for e in lattice.elements) if lattice is not None else ()
    return els, (_scalars(beam_cfg) if beam_cfg is not None else ())


def _f(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{x:.4g}" if np.isfinite(x) else "—"


# ---------------------------------------------------------------------------
class _OrmWorker(QThread):
    """One ORM task (compare / fit / validate) on lattice + beam SNAPSHOTS."""

    finished_ok = pyqtSignal(object)      # result dict
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, int)       # iteration, max_iter

    def __init__(self, task: str, lattice, beam_cfg, measured: dict, device_map,
                 opts: dict, parent=None):
        super().__init__(parent)
        self._task = task
        self._lattice = copy.deepcopy(lattice)          # deepcopy walks need no WALK_LOCK
        self._beam = copy.deepcopy(beam_cfg)
        self._measured = measured
        self._map = device_map
        self._opts = dict(opts)
        self._stop = threading.Event()
        # numpy / OpenBLAS workspace can blow the macOS default QThread stack
        self.setStackSize(16 * 1024 * 1024)

    def request_stop(self) -> None:
        self._stop.set()

    def _stopping(self) -> bool:
        return self._stop.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            from linac_gen.core.cancelled import OperationCancelled
            from linac_gen.orm import (OrmFitOptions, OrmModel, calibration_from_fit,
                                       compare_orm, fit_orm, resolve_devices,
                                       synthetic_validation)
            o = self._opts
            try:
                sel = resolve_devices(self._lattice, self._measured, self._map)
                if sel.n_bpm == 0 or sel.n_trim == 0:
                    raise ValueError("no measured device matches the lattice — "
                                     "check the Mapping tab")
                model = OrmModel(self._lattice, self._beam, sel)
                R = model.response(should_stop=self._stopping)
                cmp = compare_orm(self._measured, R, sel, model.brho_trim, model.w_trim_MeV,
                                  sys_floor=float(o.get("sys_floor", 0.05)))
                out = {"task": self._task, "selection": sel, "model": model,
                       "response": R, "compare": cmp}
                if self._task == "compare":
                    if o.get("check_tracking"):
                        T = model.tracked_response(should_stop=self._stopping)
                        out["tracking_dev"] = float(model.selfcheck(R, T))
                else:
                    fo = OrmFitOptions(stage=o.get("stage", "trims+quads"),
                                       prior_g=float(o.get("prior_g", 0.10)),
                                       prior_G=float(o.get("prior_G", 0.05)),
                                       sys_floor=float(o.get("sys_floor", 0.05)),
                                       fix_quads=tuple(o.get("fix_quads", ())),
                                       max_iter=int(o.get("max_iter", 12)))

                    def _prog(it, _chi2, _n=fo.max_iter):
                        self.progress.emit(int(it), int(_n))
                    if self._task == "fit":
                        fit = fit_orm(model, self._measured, fo,
                                      should_stop=self._stopping, progress=_prog)
                        out["fit"] = fit
                        out["calibration"] = calibration_from_fit(
                            fit, model, self._measured, lattice_path=o.get("lattice_path"),
                            beam_cfg=self._beam, device_map=self._map)
                    else:
                        out["validation"] = synthetic_validation(
                            model, self._measured, fo, seed=int(o.get("seed", 7)),
                            g_sigma=float(o.get("g_sigma", 0.03)),
                            G_sigma=float(o.get("G_sigma", 0.03)),
                            should_stop=self._stopping, progress=_prog)
                self.finished_ok.emit(out)
            except OperationCancelled:
                self.cancelled.emit()
            except ValueError as exc:                               # HELIX's refusal idiom: the message is the answer
                self.failed.emit(str(exc))
        except Exception as exc:                                    # noqa: BLE001
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
class OrmCalibrationDialog(QDialog):
    """Non-modal orbit-response comparison / LOCO-style calibration popup."""

    def __init__(self, parent, state):
        super().__init__(parent)
        self.setWindowTitle("Orbit-Response Calibration (LOCO)")
        from linac_gen_gui.interphase.scrollwrap import screen_capped
        self.setMinimumSize(*screen_capped(self, 1040, 720))
        self.setModal(False)
        self.state = state
        self._worker: Optional[_OrmWorker] = None
        self._settings = make_settings("HELIX", "OrmCalibration")
        self._paths: list = []            # measured inputs (folders / CSV) in load order
        self._measured: dict = {}         # {"x": MeasuredOrm, "y": MeasuredOrm}
        self._map = None                  # DeviceMap
        self._sel = None                  # OrmSelection of the last resolve
        self._compare = None
        self._model = None
        self._fit = None
        self._cal = None
        self._lattice_at_launch = None
        self._fp_at_launch = None
        self._measured_at_launch = None
        self._map_at_launch = None
        self._lattice_obj = getattr(state, "lattice", None)
        self._compare_text = ""
        self._applying = False
        self._sign_errors: dict = {}
        self._build_ui()
        self._restore_session()
        # Bound methods (never lambdas) with a RuntimeError guard — see
        # tests/gui/test_parameter_scan_dangling.py for the historical bug.
        for sig, slot in ((getattr(state, "lattice_changed", None), self._on_lattice_changed),
                          (getattr(state, "beam_config_changed", None), self._on_beam_changed)):
            if sig is not None:
                sig.connect(slot)

    # ---------------------------------------------------------------- UI
    def _group(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(
            f"QGroupBox {{ color:{theme.TEXT_2}; border:1px solid {theme.BORDER_0};"
            f" border-radius:4px; margin-top:12px; padding-top:6px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left:10px; padding:0 6px;"
            f" color:{theme.TEXT_2}; font-size:10px; letter-spacing:1px;"
            f" text-transform:uppercase; background:{theme.BG_0}; }}")
        return g

    def _accent(self, btn: QPushButton) -> None:
        btn.setStyleSheet(f"background:{theme.ACCENT}; color:#00161c; border:0; "
                          f"border-radius:3px; padding:6px 14px; font-weight:600;")

    def _table(self, headers) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setVisible(False)
        t.setAlternatingRowColors(True)
        return t

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)

        top = QHBoxLayout(); top.setSpacing(8)
        self._load_folder_btn = QPushButton("Load FORMA folder…")
        self._load_folder_btn.setToolTip("One driven plane per folder — load the H and the V scan folders one after the other")
        self._load_folder_btn.clicked.connect(self._load_folder)
        self._load_csv_btn = QPushButton("Load ORM CSV…")
        self._load_csv_btn.clicked.connect(self._load_csv)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear)
        self._status = QLabel("no measurement loaded")
        self._status.setStyleSheet(f"color:{theme.TEXT_2};")
        top.addWidget(self._load_folder_btn); top.addWidget(self._load_csv_btn); top.addWidget(self._clear_btn)
        top.addWidget(self._status, 1)
        v.addLayout(top)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_mapping_tab(), "Mapping")
        self._tabs.addTab(self._build_compare_tab(), "Compare")
        self._tabs.addTab(self._build_fit_tab(), "Fit")
        self._tabs.addTab(self._build_apply_tab(), "Apply / Export")
        v.addWidget(self._tabs, 1)

    def _build_mapping_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(6)
        row = QHBoxLayout()
        self._automap_btn = QPushButton("Auto-map (built-in rules)")
        self._automap_btn.clicked.connect(self._auto_map)
        self._apply_map_btn = QPushButton("Apply table edits")
        self._apply_map_btn.setToolTip("Element column: deck label or HELIX name; sign: +1 / -1; untick include to leave a device out")
        self._apply_map_btn.clicked.connect(self._apply_table_edits)
        self._import_map_btn = QPushButton("Import map JSON…")
        self._import_map_btn.clicked.connect(self._import_map)
        self._export_map_btn = QPushButton("Export map JSON…")
        self._export_map_btn.clicked.connect(self._export_map)
        for b in (self._automap_btn, self._apply_map_btn, self._import_map_btn, self._export_map_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self._map_table = self._table(["device", "kind", "planes", "element (label or name)", "s (m)", "status", "sign", "include"])
        lay.addWidget(self._map_table, 3)
        self._map_notes = QPlainTextEdit(); self._map_notes.setReadOnly(True)
        self._map_notes.setMaximumHeight(120)
        self._map_notes.setStyleSheet(f"font-family:{theme.FONT_MONO}; color:{theme.TEXT_2};")
        lay.addWidget(self._map_notes, 1)
        return w

    def _build_compare_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(6)
        row = QHBoxLayout()
        self._compare_btn = QPushButton("  Compute model and compare")
        self._accent(self._compare_btn)
        self._compare_btn.clicked.connect(self._start_compare)
        self._tracking_chk = QCheckBox("cross-check by tracking (kick-and-read)")
        self._plane_combo = QComboBox()
        self._plane_combo.addItem(_PLANE_NAME["x"], "x"); self._plane_combo.addItem(_PLANE_NAME["y"], "y")
        self._plane_combo.currentIndexChanged.connect(self._refresh_compare)
        self._compare_summary = QLabel("—")
        self._compare_summary.setStyleSheet(f"color:{theme.TEXT_2};")
        self._compare_summary.setWordWrap(True)
        row.addWidget(self._compare_btn); row.addWidget(self._tracking_chk)
        row.addSpacing(12); row.addWidget(QLabel("plane:")); row.addWidget(self._plane_combo)
        row.addWidget(self._compare_summary, 1)
        lay.addLayout(row)

        maps = QHBoxLayout(); maps.setSpacing(6)
        self._heat = {}
        self._heat_img = {}
        cmap = pg.colormap.get("CET-D1A")
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        for key, title in (("meas", "measured (column ÷ max)"), ("model", "model (column ÷ max)"), ("diff", "measured − model")):
            p = pg.PlotWidget(); style_plot(p, "BPM", "", xlabel="trim", xunits="", title=title)
            img = pg.ImageItem(axisOrder="row-major"); img.setLookupTable(lut)
            p.addItem(img); p.invertY(True); p.setMenuEnabled(False)
            self._heat[key] = p; self._heat_img[key] = img
            maps.addWidget(p)
        lay.addLayout(maps, 3)

        bottom = QHBoxLayout(); bottom.setSpacing(6)
        left = QVBoxLayout()
        trow = QHBoxLayout(); trow.addWidget(QLabel("trim:"))
        self._trim_combo = QComboBox(); self._trim_combo.currentIndexChanged.connect(self._refresh_trim_plot)
        trow.addWidget(self._trim_combo, 1)
        left.addLayout(trow)
        self._trim_plot = pg.PlotWidget()
        style_plot(self._trim_plot, "response ÷ column max", "", xlabel="BPM", xunits="", title="measured (points, ±1σ) vs model (line)")
        self._trim_plot.addLegend(offset=(10, 10))
        left.addWidget(self._trim_plot, 1)
        bottom.addLayout(left, 3)
        self._metrics_table = self._table(["trim", "device", "k (mT·m/A)", "kick (mrad/A)", "|r|", "residual", "n BPM", "included"])
        bottom.addWidget(self._metrics_table, 2)
        lay.addLayout(bottom, 3)
        return w

    def _build_fit_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(6)
        opts = self._group("Fit options"); form = QFormLayout(opts)
        self._stage_combo = QComboBox()
        for s in _STAGES:
            self._stage_combo.addItem(s)
        self._stage_combo.setCurrentText("trims+quads")
        form.addRow("Stage:", self._stage_combo)
        prow = QHBoxLayout()
        self._prior_g = QDoubleSpinBox(); self._prior_g.setDecimals(3); self._prior_g.setRange(0.001, 10.0); self._prior_g.setSingleStep(0.01); self._prior_g.setValue(0.10)
        self._prior_G = QDoubleSpinBox(); self._prior_G.setDecimals(3); self._prior_G.setRange(0.001, 10.0); self._prior_G.setSingleStep(0.01); self._prior_G.setValue(0.05)
        self._sys_floor = QDoubleSpinBox(); self._sys_floor.setDecimals(3); self._sys_floor.setRange(0.0, 1.0); self._sys_floor.setSingleStep(0.01); self._sys_floor.setValue(0.05)
        self._max_iter = QSpinBox(); self._max_iter.setRange(1, 200); self._max_iter.setValue(12)
        for lab, wdg in (("prior σ(g):", self._prior_g), ("prior σ(G):", self._prior_G), ("sys. floor:", self._sys_floor), ("max iter:", self._max_iter)):
            prow.addWidget(QLabel(lab)); prow.addWidget(wdg)
        prow.addStretch(1)
        pw = QWidget(); pw.setLayout(prow)
        form.addRow("Weights:", pw)
        self._fix_quads = QLineEdit(); self._fix_quads.setPlaceholderText("quad labels held at scale 1, comma-separated (e.g. Q73)")
        form.addRow("Fix quads:", self._fix_quads)
        lay.addWidget(opts)

        row = QHBoxLayout()
        self._fit_btn = QPushButton("  Fit"); self._accent(self._fit_btn); self._fit_btn.clicked.connect(self._start_fit)
        self._cancel_btn = QPushButton("Cancel"); self._cancel_btn.setEnabled(False); self._cancel_btn.clicked.connect(self._cancel)
        self._validate_btn = QPushButton("Synthetic validation…")
        self._validate_btn.setToolTip("Inject random quad errors into the model, refit, report the recovery")
        self._validate_btn.clicked.connect(self._start_validate)
        self._progress = QProgressBar(); self._progress.setRange(0, 100); self._progress.setValue(0)
        self._fit_status = QLabel("—"); self._fit_status.setStyleSheet(f"color:{theme.TEXT_2};"); self._fit_status.setWordWrap(True)
        row.addWidget(self._fit_btn); row.addWidget(self._cancel_btn); row.addWidget(self._validate_btn)
        row.addWidget(self._progress, 1); row.addWidget(self._fit_status, 2)
        lay.addLayout(row)

        body = QHBoxLayout(); body.setSpacing(6)
        self._quad_table = self._table(["quad", "scale", "± err", "change (%)", "fixed", "reason"])
        body.addWidget(self._quad_table, 2)
        plots = QVBoxLayout()
        self._quad_plot = pg.PlotWidget(); style_plot(self._quad_plot, "gradient change", "%", xlabel="quad", xunits="", title="fitted quadrupole corrections")
        self._resid_plot = pg.PlotWidget(); style_plot(self._resid_plot, "residual / signal", "%", xlabel="trim", xunits="", title="residual per trim: before (grey) → after (colour)")
        self._resid_plot.addLegend(offset=(10, 10))
        plots.addWidget(self._quad_plot, 1); plots.addWidget(self._resid_plot, 1)
        body.addLayout(plots, 3)
        lay.addLayout(body, 1)
        return w

    def _build_apply_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(6)
        row = QHBoxLayout()
        self._bake_chk = QCheckBox("bake into the gradients (default: Quadrupole.gradient_rel, design gradients unchanged)")
        row.addWidget(self._bake_chk); row.addStretch(1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self._apply_btn = QPushButton("  Apply to lattice (undoable)"); self._accent(self._apply_btn); self._apply_btn.clicked.connect(self._apply)
        self._export_deck_btn = QPushButton("Export recalibrated deck…"); self._export_deck_btn.clicked.connect(self._export_deck)
        self._save_cal_btn = QPushButton("Save calibration JSON…"); self._save_cal_btn.clicked.connect(self._save_calibration)
        self._load_cal_btn = QPushButton("Load calibration JSON…"); self._load_cal_btn.clicked.connect(self._load_calibration)
        for b in (self._apply_btn, self._export_deck_btn, self._save_cal_btn, self._load_cal_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self._apply_text = QPlainTextEdit(); self._apply_text.setReadOnly(True)
        self._apply_text.setStyleSheet(f"font-family:{theme.FONT_MONO}; color:{theme.TEXT_2};")
        self._apply_text.setPlainText("Run a fit (Fit tab) or load a calibration JSON.")
        lay.addWidget(self._apply_text, 1)
        return w

    # ---------------------------------------------------------------- session
    def _session(self) -> dict:
        s = getattr(self.state, "orm_session", None)
        if not isinstance(s, dict):
            s = {}
            try:
                self.state.orm_session = s
            except AttributeError:
                pass
        return s

    def _save_session(self) -> None:
        s = self._session()
        s.update({"paths": list(self._paths), "map": self._map, "calibration": self._cal})

    def _restore_session(self) -> None:
        s = self._session()
        try:
            paths = list(s.get("paths") or [])
            if paths:
                self._paths = paths
                self._map = s.get("map")
                self._reload_measured(keep_map=self._map is not None)
            cal = s.get("calibration")
            if cal is not None:
                self._apply_text.setPlainText(self._calibration_text(cal))
                self._cal = cal
        except Exception as exc:                                    # noqa: BLE001 — a broken session must never block the dialog
            self._paths = []; self._measured = {}; self._map = None; self._cal = None
            s.update({"paths": [], "map": None, "calibration": None})
            self._status.setText(f"previous session dropped: {exc}")

    def _last_dir(self) -> str:
        d = self._settings.value("last_dir", "", type=str)
        return d if d and os.path.isdir(d) else ""

    def _remember_dir(self, path: str) -> None:
        p = Path(path)
        self._settings.setValue("last_dir", str(p if p.is_dir() else p.parent))

    # ---------------------------------------------------------------- loading
    def _load_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "FORMA scan folder (one driven plane)", self._last_dir())
        if d:
            self._add_path(d)

    def _load_csv(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "HELIX ORM CSV", self._last_dir(), "ORM CSV (*.csv);;All files (*)")
        if p:
            self._add_path(p)

    def _add_path(self, p: str) -> None:
        self._remember_dir(p)
        if p in self._paths:
            self._status.setText("already loaded: " + os.path.basename(p)); return
        self._paths.append(p)
        self._reload_measured(keep_map=False)

    def _reload_measured(self, *, keep_map: bool) -> bool:
        """(Re)load ``self._paths``; inputs that fail are dropped (reported), the rest stays loaded."""
        from linac_gen.orm import load_measured
        good, bad = [], []
        for path in list(self._paths):
            try:
                load_measured([path])
                good.append(path)
            except Exception as exc:                                # noqa: BLE001 — csv.Error, ValueError, OSError …
                bad.append((path, exc))
        while True:                                                 # then the combination (same plane twice, trims differ)
            try:
                meas = load_measured(good) if good else {}
                break
            except Exception as exc:                                # noqa: BLE001
                bad.append((good.pop(), exc))
        self._paths = good
        self._measured = meas
        self._invalidate_results()
        if not keep_map or self._map is None:
            self._auto_map()
        else:
            self._resolve_and_fill()
        planes = ", ".join(f"{p} ({m.source}, {len(m.trims)} trims)" for p, m in sorted(meas.items()))
        self._status.setText(planes if meas else "no measurement loaded")
        self._save_session()
        if bad:
            QMessageBox.warning(self, "Orbit-response measurement", "Not loaded:\n" + "\n".join(f"{os.path.basename(str(q)) or q}: {e}" for q, e in bad))
        return not bad

    def _clear(self) -> None:
        self._paths = []; self._measured = {}; self._map = None; self._sel = None; self._sign_errors = {}
        self._invalidate_results()
        self._map_table.setRowCount(0); self._map_notes.setPlainText("")
        self._status.setText("no measurement loaded")
        self._save_session()

    def _invalidate_results(self, note: str = "—") -> None:
        """Drop compare / model / fit views — the mapping, measurement, lattice or beam they were built on changed."""
        self._compare = self._model = self._fit = None
        self._compare_text = ""
        for img in self._heat_img.values():
            img.clear()
        self._metrics_table.setRowCount(0); self._quad_table.setRowCount(0)
        self._trim_combo.blockSignals(True); self._trim_combo.clear(); self._trim_combo.blockSignals(False)
        self._trim_plot.clear(); self._quad_plot.clear(); self._resid_plot.clear()
        self._compare_summary.setText(note); self._fit_status.setText(note)
        self._progress.setRange(0, 100); self._progress.setValue(0)

    # ---------------------------------------------------------------- app-state reactions
    def _on_lattice_changed(self, lat) -> None:
        try:
            self._map_table.rowCount()                  # RuntimeError once the C++ widget is gone
        except RuntimeError:
            for sig, slot in ((getattr(self.state, "lattice_changed", None), self._on_lattice_changed),
                              (getattr(self.state, "beam_config_changed", None), self._on_beam_changed)):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass
            return
        if lat is not self._lattice_obj:                # another deck: everything computed is stale
            self._lattice_obj = lat
            self._invalidate_results("lattice replaced — recompute")
            if self._measured and self._map is not None:
                self._resolve_and_fill()
            return
        if self._applying:
            return                                      # our own undoable apply
        if self._measured and self._map is not None:
            self._resolve_and_fill()                    # positions / labels may have moved
        if self._compare is not None or self._fit is not None:
            self._fit_status.setText("lattice edited since the last compute — recompute before applying")

    def _on_beam_changed(self, _cfg) -> None:
        try:
            self._map_table.rowCount()
        except RuntimeError:
            return
        if self._compare is not None or self._fit is not None:
            self._invalidate_results("beam changed — recompute")

    # ---------------------------------------------------------------- mapping
    def _auto_map(self) -> None:
        if self.state.lattice is None or not self._measured:
            self._map_table.setRowCount(0); return
        from linac_gen.orm import default_device_map
        self._map = default_device_map(self.state.lattice, self._measured)
        self._sign_errors = {}
        self._invalidate_results()
        self._resolve_and_fill()
        self._save_session()

    def _resolve_and_fill(self) -> None:
        if self.state.lattice is None or not self._measured or self._map is None:
            return
        from linac_gen.orm import resolve_devices
        try:
            self._sel = resolve_devices(self.state.lattice, self._measured, self._map)
        except ValueError as exc:
            self._map_notes.setPlainText(f"mapping failed: {exc}"); return
        self._fill_map_table()

    def _devices(self) -> list:
        """[(device, kind, planes)] over the loaded measurement, in file order, unique."""
        rows, seen = [], {}
        for p in sorted(self._measured):
            m = self._measured[p]
            for dev in m.trims:
                key = (dev, "trim")
                if key in seen:
                    rows[seen[key]][2].add(p)
                else:
                    seen[key] = len(rows); rows.append((dev, "trim", {p}))
            for rp in m.read_planes():
                for dev in m.bpms[rp]:
                    key = (dev, "BPM")
                    if key in seen:
                        rows[seen[key]][2].add(rp)
                    else:
                        seen[key] = len(rows); rows.append((dev, "BPM", {rp}))
        return rows

    def _fill_map_table(self) -> None:
        sel, dm = self._sel, self._map
        bpm_at = {}
        for rp, devs in sel.bpm_devices.items():
            for i, dev in enumerate(devs):
                if dev is not None:
                    bpm_at[dev] = i
        trim_at = {}
        for kp, devs in sel.trim_devices.items():
            for j, dev in enumerate(devs):
                if dev is not None:
                    trim_at[dev] = j
        rows = self._devices()
        t = self._map_table
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for r, (dev, kind, planes) in enumerate(rows):
            target = (dm.bpms if kind == "BPM" else dm.trims).get(dev, "")
            excluded = dev in (dm.exclude_bpms if kind == "BPM" else dm.exclude_trims) or target in (dm.exclude_bpms if kind == "BPM" else dm.exclude_trims)
            if kind == "BPM":
                idx = bpm_at.get(dev); s_m = sel.s_bpm_m[idx] if idx is not None else None
                label = sel.bpm_labels[idx] if idx is not None else None
            else:
                idx = trim_at.get(dev); s_m = sel.s_trim_m[idx] if idx is not None else None
                label = sel.trim_labels[idx] if idx is not None else None
            if dev in self._sign_errors:
                status = f"invalid sign {self._sign_errors[dev]!r} (use +1 / -1)"
            elif excluded:
                status = "excluded"
            elif dev in sel.dead_devices:
                status = "dead (scan)"
            elif idx is not None:
                status = f"matched → {label}"
            elif target:
                status = "no such element"
            else:
                status = "unmatched (no rule)"
            sign = int(dm.bpm_sign.get(dev, 1)) if kind == "BPM" else 1
            cells = [dev, kind, ",".join(sorted(planes)), target, f"{s_m:.3f}" if s_m is not None else "", status,
                     f"{sign:+d}" if kind == "BPM" else "", ""]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                if c == 7:
                    it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                    it.setCheckState(Qt.CheckState.Unchecked if excluded else Qt.CheckState.Checked)
                elif c in (3, 6) and not (c == 6 and kind != "BPM"):
                    it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
                else:
                    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                t.setItem(r, c, it)
        t.blockSignals(False)
        notes = [sel.summary()]
        for kind in ("bpm", "trim"):
            if sel.unmatched_devices[kind]:
                notes.append(f"unmatched {kind} devices: " + ", ".join(sel.unmatched_devices[kind]))
            if sel.unmatched_elements[kind]:
                notes.append(f"lattice {kind}s without data: " + ", ".join(sel.unmatched_elements[kind]))
        notes += ["note: " + n for n in sel.notes]
        self._map_notes.setPlainText("\n".join(notes))

    def _apply_table_edits(self) -> None:
        if self._map is None:
            return
        from linac_gen.orm import DeviceMap
        dm = DeviceMap(rules="edited")
        t = self._map_table
        self._sign_errors = {}
        for r in range(t.rowCount()):
            dev = t.item(r, 0).text(); kind = t.item(r, 1).text()
            target = (t.item(r, 3).text() or "").strip()
            include = t.item(r, 7).checkState() == Qt.CheckState.Checked
            if kind == "BPM":
                if target:
                    dm.bpms[dev] = target
                raw = (t.item(r, 6).text() or "+1").strip().replace("\u2212", "-")
                if raw in ("+1", "1", "+1.0", "1.0"):
                    pass
                elif raw in ("-1", "-1.0"):
                    dm.bpm_sign[dev] = -1
                else:
                    self._sign_errors[dev] = raw
                    if int(self._map.bpm_sign.get(dev, 1)) < 0:
                        dm.bpm_sign[dev] = -1                     # keep the previous sign
                if not include:
                    dm.exclude_bpms.add(dev)
            else:
                if target:
                    dm.trims[dev] = target
                if not include:
                    dm.exclude_trims.add(dev)
        self._map = dm
        self._invalidate_results()
        self._resolve_and_fill()
        self._save_session()

    def _import_map(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Device map JSON", self._last_dir(), "JSON (*.json)")
        if not p:
            return
        from linac_gen.orm import DeviceMap
        try:
            dm = DeviceMap.from_json(p)
        except (ValueError, OSError, TypeError) as exc:
            QMessageBox.warning(self, "Device map", f"Could not read the map:\n{exc}"); return
        self._map = dm; self._sign_errors = {}
        self._remember_dir(p)
        self._invalidate_results()
        self._resolve_and_fill(); self._save_session()

    def _export_map(self) -> None:
        if self._map is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save device map", os.path.join(self._last_dir(), "orm_device_map.json"), "JSON (*.json)")
        if not p:
            return
        try:
            self._map.to_json(p)
        except OSError as exc:
            QMessageBox.warning(self, "Device map", f"Not written:\n{exc}"); return
        self._remember_dir(p)
        self.state.status_message.emit(f"device map written: {p}")

    # ---------------------------------------------------------------- workers
    def _session_deck(self):
        p = getattr(self.state, "lattice_path", None)
        if not p or not os.path.isfile(str(p)):
            return None
        p = str(p)
        if p.lower().endswith(".lgproj"):
            from linac_gen.io.project import load_project
            return str(load_project(p).lattice_path)
        return p

    def _fit_opts(self) -> dict:
        fix = [s.strip() for s in self._fix_quads.text().split(",") if s.strip()]
        return {"stage": self._stage_combo.currentText(), "prior_g": self._prior_g.value(),
                "prior_G": self._prior_G.value(), "sys_floor": self._sys_floor.value(),
                "max_iter": self._max_iter.value(), "fix_quads": fix,
                "check_tracking": self._tracking_chk.isChecked(),
                "lattice_path": self._session_deck(), "seed": 7, "g_sigma": 0.03, "G_sigma": 0.03}

    def _start(self, task: str) -> None:
        if self.state.lattice is None:
            QMessageBox.warning(self, "Orbit-response calibration", "Load a lattice first."); return
        if getattr(self.state, "beam_config", None) is None:
            QMessageBox.warning(self, "Orbit-response calibration", "Set the beam (Beam tab) first."); return
        if not self._measured:
            QMessageBox.warning(self, "Orbit-response calibration", "Load a measured orbit-response matrix first."); return
        if self._worker is not None and self._worker.isRunning():
            return
        if self._map is None:
            self._auto_map()
        self._lattice_at_launch = self.state.lattice
        self._fp_at_launch = _optics_fingerprint(self.state.lattice, self.state.beam_config)
        self._measured_at_launch = self._measured
        self._map_at_launch = self._map
        w = _OrmWorker(task, self.state.lattice, self.state.beam_config, self._measured, self._map, self._fit_opts(), parent=self)
        w.finished_ok.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(self._on_cancelled)
        w.progress.connect(self._on_progress)
        self._worker = w
        self._set_busy(True, task)
        w.start()

    def _start_compare(self) -> None:
        self._start("compare")

    def _start_fit(self) -> None:
        self._start("fit")

    def _start_validate(self) -> None:
        self._start("validate")

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._fit_status.setText("cancelling…")

    def _set_busy(self, busy: bool, task: str = "") -> None:
        for b in (self._compare_btn, self._fit_btn, self._validate_btn, self._apply_btn, self._automap_btn, self._apply_map_btn,
                  self._load_folder_btn, self._load_csv_btn, self._clear_btn, self._import_map_btn, self._export_map_btn,
                  self._export_deck_btn, self._save_cal_btn, self._load_cal_btn):
            b.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        if busy:
            self._progress.setRange(0, 0)                 # indeterminate until the fit reports iterations
            self._fit_status.setText({"compare": "computing the model response…", "fit": "fitting…", "validate": "synthetic validation…"}.get(task, "working…"))
        else:
            self._progress.setRange(0, 100)

    def _on_progress(self, it: int, n: int) -> None:
        if self.sender() is not self._worker:
            return
        self._progress.setRange(0, 100)
        self._progress.setValue(int(100 * min(it, n) / max(n, 1)))

    def _on_failed(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        self._set_busy(False)
        self._fit_status.setText("failed")
        QMessageBox.critical(self, "Orbit-response calibration failed", msg)

    def _on_cancelled(self) -> None:
        if self.sender() is not self._worker:
            return
        self._set_busy(False)
        self._fit_status.setText("cancelled")

    def _on_done(self, out: dict) -> None:
        w = self.sender()
        if w is not None and w is not self._worker:
            return                              # a cancelled predecessor finishing late
        self._set_busy(False)
        if (self.state.lattice is not self._lattice_at_launch
                or _optics_fingerprint(self.state.lattice, self.state.beam_config) != self._fp_at_launch):
            self.state.status_message.emit("ORM result discarded — the lattice or beam changed while it was computing")
            self._fit_status.setText("discarded: the lattice or beam changed while computing"); return
        if self._measured is not self._measured_at_launch or self._map is not self._map_at_launch:
            self.state.status_message.emit("ORM result discarded — the measurement or mapping changed while it was computing")
            self._fit_status.setText("discarded: the measurement or mapping changed while computing"); return
        self._sel = out["selection"]; self._model = out["model"]; self._compare = out["compare"]
        self._fill_map_table()
        self._refresh_compare()
        task = out["task"]
        if task == "compare":
            txt = "; ".join(self._summary_lines(out["compare"]))
            if "tracking_dev" in out:
                txt += f"; kick-and-read cross-check: max relative deviation {out['tracking_dev']:.2e}"
            self._compare_text = txt
            self._fit_status.setText("model computed"); self._refresh_compare()
            self._tabs.setCurrentIndex(1)
        elif task == "fit":
            self._fit = out["fit"]; self._cal = out["calibration"]
            self._compare_text = "; ".join(self._summary_lines(out["compare"])); self._refresh_compare()
            self._refresh_fit(); self._show_calibration_text(); self._save_session()
            self._progress.setValue(100)
            self._fit_status.setText(" · ".join(self._fit.summary_lines()))
            self._tabs.setCurrentIndex(2)
        elif task == "validate":
            v = out["validation"]
            lines = [f"seed {v['seed']}: injected quad errors {100 * v['g_rms_injected']:.2f} % rms → recovered to "
                     f"{100 * v['g_rms_recovered']:.2f} % rms (max {100 * v['g_max_recovered']:.2f} %); trims to {100 * v['k_rel_rms']:.1f} %"]
            if np.isfinite(v.get("G_rms_recovered", float("nan"))) and v.get("G_rms_recovered", 0) > 0:
                lines.append(f"BPM gains recovered to {100 * v['G_rms_recovered']:.2f} %")
            lines += [f"   {lab}: injected {100 * t:+.2f} %, recovered {100 * f:+.2f} %" for lab, t, f in v["worst_quads"]]
            self._fit_status.setText(lines[0])
            QMessageBox.information(self, "Synthetic validation", "\n".join(lines))

    @staticmethod
    def _summary_lines(cmp) -> list:
        from linac_gen.orm import summary_lines
        return summary_lines(cmp)

    # ---------------------------------------------------------------- compare views
    def _refresh_compare(self) -> None:
        cmp, sel = self._compare, self._sel
        for img in self._heat_img.values():
            img.clear()
        self._metrics_table.setRowCount(0)
        if cmp is None or sel is None:
            self._trim_combo.clear(); self._trim_plot.clear(); return
        plane = self._plane_combo.currentData()
        d = cmp["planes"].get(plane)
        if d is None:
            self._trim_combo.clear(); self._trim_plot.clear()
            self._compare_summary.setText((self._compare_text + " — " if self._compare_text else "") + f"no {_PLANE_NAME[plane]} measurement loaded"); return
        if np.shape(d["aligned"]) != (sel.n_bpm, sel.n_trim):
            self._invalidate_results("mapping changed — recompute"); return
        self._compare_summary.setText(self._compare_text)
        ticks_x = [[(j, lab) for j, lab in enumerate(sel.trim_labels)]]
        ticks_y = [[(i, lab) for i, lab in enumerate(sel.bpm_labels)]]
        for key, A in (("meas", d["measured_norm"]), ("model", d["model_norm"]), ("diff", d["measured_norm"] - d["model_norm"])):
            img = self._heat_img[key]
            img.setImage(np.nan_to_num(np.asarray(A, dtype=float), nan=0.0), levels=(-1.0, 1.0))
            img.setRect(-0.5, -0.5, sel.n_trim, sel.n_bpm)
            p = self._heat[key]
            p.getAxis("bottom").setTicks(ticks_x); p.getAxis("left").setTicks(ticks_y)
            p.setXRange(-0.5, sel.n_trim - 0.5, padding=0); p.setYRange(-0.5, sel.n_bpm - 0.5, padding=0)
        t = self._metrics_table
        t.setRowCount(len(d["per_trim"]))
        for r, row in enumerate(d["per_trim"]):
            cells = [row["trim"], row["device"] or "", _f(row["k_Tm_per_A"] * 1e3) if np.isfinite(row["k_Tm_per_A"]) else "—",
                     _f(abs(row["kick_mrad_per_A"])) if np.isfinite(row["kick_mrad_per_A"]) else "—",
                     f"{abs(row['r']):.3f}" if np.isfinite(row["r"]) else "—",
                     f"{100 * row['nrms_resid']:.0f} %" if np.isfinite(row["nrms_resid"]) else "—",
                     str(row["n_down"]), "yes" if row["included"] else "no"]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable); t.setItem(r, c, it)
        cur = self._trim_combo.currentIndex()
        self._trim_combo.blockSignals(True); self._trim_combo.clear()
        for lab in sel.trim_labels:
            self._trim_combo.addItem(lab)
        self._trim_combo.setCurrentIndex(cur if 0 <= cur < sel.n_trim else 0)
        self._trim_combo.blockSignals(False)
        self._refresh_trim_plot()

    def _refresh_trim_plot(self) -> None:
        self._trim_plot.clear()
        cmp, sel = self._compare, self._sel
        j = self._trim_combo.currentIndex()
        if cmp is None or sel is None or j < 0:
            return
        plane = self._plane_combo.currentData(); d = cmp["planes"].get(plane)
        if d is None or j >= sel.n_trim:
            return
        A, E, R = d["aligned"], d["aligned_err"], d["model"]
        dn = sel.down[:, j] & np.isfinite(A[:, j])
        nm = float(np.nanmax(np.abs(A[dn, j]))) if dn.any() else 1.0
        nm = nm if nm > 0 else 1.0
        x = np.arange(sel.n_bpm)
        ok = np.isfinite(A[:, j])
        err = np.where(np.isfinite(E[:, j]), E[:, j], 0.0) / nm
        col = theme.ERR if plane == "x" else theme.ACCENT
        self._trim_plot.addItem(pg.ErrorBarItem(x=x[ok], y=A[ok, j] / nm, height=2 * err[ok], beam=0.25, pen=pg.mkPen(col, width=1)))
        self._trim_plot.plot(x[ok], A[ok, j] / nm, pen=None, symbol="o", symbolSize=7, symbolBrush=col, symbolPen=None, name="measured ÷ column max")
        if self._fit is not None and plane in self._fit.metrics_after and self._fit.bpms == sel.bpm_labels and self._fit.trims == sel.trim_labels:
            model = self._fit.fitted_block(plane)[:, j]; label = f"fitted model ({self._fit.stage})"
        else:
            k = d["per_trim"][j]["k_Tm_per_A"]
            model = k * R[:, j] if np.isfinite(k) else np.full(sel.n_bpm, np.nan); label = "model × per-trim k"
        m_ok = dn & np.isfinite(model)
        if m_ok.any():
            self._trim_plot.plot(x[m_ok], model[m_ok] / nm, pen=pg.mkPen("#e6e6ee", width=2), symbol="s", symbolSize=5, symbolBrush="#e6e6ee", name=label)
        self._trim_plot.getAxis("bottom").setTicks([[(i, lab) for i, lab in enumerate(sel.bpm_labels)]])
        self._trim_plot.setYRange(-1.3, 1.3, padding=0)

    # ---------------------------------------------------------------- fit views
    def _refresh_fit(self) -> None:
        fit = self._fit
        self._quad_table.setRowCount(0); self._quad_plot.clear(); self._resid_plot.clear()
        if fit is None:
            return
        t = self._quad_table
        t.setRowCount(len(fit.quads))
        for r, lab in enumerate(fit.quads):
            g, e, fx = float(fit.g[r]), float(fit.g_err[r]), bool(fit.g_fixed[r])
            cells = [lab, f"{g:.5f}", f"{e:.5f}" if np.isfinite(e) else "—", f"{100 * (g - 1):+.2f}", "yes" if fx else "", fit.fixed_reason.get(lab, "")]
            for c, txt in enumerate(cells):
                it = QTableWidgetItem(txt); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable); t.setItem(r, c, it)
        x = np.arange(len(fit.quads))
        h = 100 * (fit.g - 1)
        brushes = ["#5a5a66" if fx else theme.ERR for fx in fit.g_fixed]
        self._quad_plot.addItem(pg.BarGraphItem(x=x, height=h, width=0.7, brushes=brushes, pen=None))
        err = np.where(np.isfinite(fit.g_err), fit.g_err, 0.0) * 100
        self._quad_plot.addItem(pg.ErrorBarItem(x=x, y=h, height=2 * err, beam=0.2, pen=pg.mkPen("#e6e6ee", width=1)))
        self._quad_plot.getAxis("bottom").setTicks([[(i, lab) for i, lab in enumerate(fit.quads)]])
        xt = np.arange(len(fit.trims))
        for k, (p, col) in enumerate((("x", theme.ERR), ("y", theme.ACCENT))):
            if p not in fit.metrics_after:
                continue
            off = -0.2 if k == 0 else 0.2
            before = 100 * np.nan_to_num(np.asarray(fit.metrics_before[p]["nrms_per_trim"], dtype=float))
            after = 100 * np.nan_to_num(np.asarray(fit.metrics_after[p]["nrms_per_trim"], dtype=float))
            self._resid_plot.addItem(pg.BarGraphItem(x=xt + off, height=before, width=0.36, brush="#5a5a66", pen=None))
            self._resid_plot.addItem(pg.BarGraphItem(x=xt + off, height=after, width=0.36, brush=col, pen=None, name=f"{_PLANE_NAME[p]} after"))
        self._resid_plot.getAxis("bottom").setTicks([[(i, lab) for i, lab in enumerate(fit.trims)]])
        self._refresh_trim_plot()

    def _show_calibration_text(self) -> None:
        cal = self._cal
        if cal is None:
            self._apply_text.setPlainText("Run a fit (Fit tab) or load a calibration JSON."); return
        self._apply_text.setPlainText(self._calibration_text(cal))

    @staticmethod
    def _calibration_text(cal: dict) -> str:
        from linac_gen.orm import validate_calibration
        validate_calibration(cal)
        lat = cal.get("lattice") if isinstance(cal.get("lattice"), dict) else {}
        lines = [f"calibration: {cal.get('created')} · stage {(cal.get('fit') or {}).get('stage')} · lattice {lat.get('path') or '(in memory)'}"]
        m = cal.get("metrics") or {}
        for p in (m.get("after") or {}):
            b = (m.get("before") or {}).get(p, {}).get("nrms_all"); a = m["after"][p].get("nrms_all")
            if a is not None and b is not None:
                lines.append(f"plane {p}: residual {100 * b:.0f} % → {100 * a:.1f} % of signal")
        lines.append("")
        lines.append(f"{'quad':<12}{'scale':>10}{'± err':>10}{'change':>10}  fixed")
        for q in cal.get("quads", []):
            e = q.get("scale_err")
            lines.append(f"{str(q['label']):<12}{q['scale']:>10.5f}{(f'{e:.5f}' if isinstance(e, (int, float)) else '—'):>10}{100 * (q['scale'] - 1):>+9.2f}%  {'yes' if q.get('fixed') else ''}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- apply / export
    def _apply(self) -> None:
        cal = self._cal
        if cal is None:
            QMessageBox.information(self, "ORM calibration", "Run a fit (Fit tab) or load a calibration JSON first."); return
        lat = self.state.lattice
        if lat is None:
            QMessageBox.warning(self, "ORM calibration", "Load a lattice first."); return
        from linac_gen.orm import apply_calibration
        mode = "bake" if self._bake_chk.isChecked() else "rel"
        try:
            changes = apply_calibration(lat, cal, mode=mode)
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "ORM calibration not applied", str(exc)); return
        if not changes:
            msg = "ORM calibration: nothing to apply — the lattice already carries it (or every scale factor is 1)"
            self.state.status_message.emit(msg); self._fit_status.setText(msg); return
        cmds = [ParamChangeCommand(q, attr, old, new) for q, attr, old, new in changes]
        self._applying = True
        try:
            self.state.bus.do(MacroCommand(cmds, label="ORM calibration"))
        finally:
            self._applying = False
        self.state.lattice_fitted = True          # fitted values in memory → plain Save reroutes to Save-As
        msg = f"ORM calibration applied to {len(cmds)} quad(s) as {'gradient' if mode == 'bake' else 'gradient_rel'} — Undo reverts it"
        self.state.status_message.emit(msg); self._fit_status.setText(msg)

    def _export_deck(self) -> None:
        cal = self._cal
        if cal is None:
            QMessageBox.information(self, "ORM calibration", "Run a fit or load a calibration JSON first."); return
        src = self._session_deck()
        if src is None:
            QMessageBox.warning(self, "Export recalibrated deck", "The session lattice has no deck file on disk — the export rewrites the original .dat text."); return
        default = os.path.join(os.path.dirname(src), Path(src).stem + "_orm_fitted.dat")
        p, _ = QFileDialog.getSaveFileName(self, "Export recalibrated deck", default, "TraceWin deck (*.dat)")
        if not p:
            return
        from linac_gen.orm import export_recalibrated_deck
        try:
            rep = export_recalibrated_deck(src, p, cal)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Export recalibrated deck", f"Not written:\n{exc}"); return
        self._remember_dir(p)
        self.state.status_message.emit(f"recalibrated deck written: {rep['dst']} ({len(rep['changed'])} quads rescaled, verified on reload)")

    def _save_calibration(self) -> None:
        if self._cal is None:
            QMessageBox.information(self, "ORM calibration", "Run a fit first."); return
        p, _ = QFileDialog.getSaveFileName(self, "Save calibration", os.path.join(self._last_dir(), "orm_calibration.json"), "JSON (*.json)")
        if not p:
            return
        from linac_gen.orm import save_calibration
        try:
            save_calibration(p, self._cal)
        except OSError as exc:
            QMessageBox.warning(self, "ORM calibration", f"Not written:\n{exc}"); return
        self._remember_dir(p)
        self.state.status_message.emit(f"calibration written: {p}")

    def _load_calibration(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Load calibration", self._last_dir(), "JSON (*.json)")
        if not p:
            return
        from linac_gen.orm import load_calibration
        try:
            cal = load_calibration(p)
            text = self._calibration_text(cal)                 # render before adopting: a bad record never replaces a good one
        except (ValueError, OSError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "ORM calibration", f"Could not read the calibration:\n{exc}"); return
        self._cal = cal; self._apply_text.setPlainText(text)
        self._remember_dir(p); self._save_session()
        self._tabs.setCurrentIndex(3)

    # ---------------------------------------------------------------- lifecycle
    def closeEvent(self, ev) -> None:
        self._cancel()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        super().closeEvent(ev)
