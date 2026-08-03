"""Analysis views for the Param Study tab.

Four sub-views over a loaded study folder (GUI-launched or CLI-made —
the folder is the contract, not the process that filled it):

* Runs    — sortable table, status-coloured, every KPI + observable
* 1D      — KPI vs one parameter; group-by a second parameter as
            coloured series; seed repeats aggregate to mean ± std
            error bars; lin/log axes
* Map     — two parameters vs a KPI: regular-grid studies render as a
            heatmap, everything else as a colour-mapped scatter
* Overlay — σ(z)/ε(z)/… curves of selected runs read from each run's
            results.h5, legend by run tag

The pure helpers (`aggregate_1d`, `detect_grid`) are module-level for
unit testing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout,
                             QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout,
                             QWidget)

from linac_gen.study.engine import METRIC_KEYS

_PALETTE = ["#4a7ab5", "#c0392b", "#1f6f50", "#8e44ad", "#d68910",
            "#16a085", "#7f8c8d", "#2c3e50", "#a04000", "#5d6d7e"]

#: overlay-able envelope quantities (present per mode where computed)
_OVERLAY_QTYS = ("sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                 "emit_nx", "emit_ny", "emit_z", "alpha_x", "beta_x",
                 "alpha_y", "beta_y", "halo_x", "halo_y",
                 "transmission")


# ----------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------
class StudyModel:
    """All run records of one study folder, loaded from status.json."""

    def __init__(self):
        self.study_dir: Path | None = None
        self.param_names: list[str] = []
        self.obs_names: list[str] = []
        self.records: list[dict] = []      # {index, tag, seed, status,
        #                                    params{}, metrics{}, obs{},
        #                                    results_path, error}

    def load(self, study_dir) -> None:
        from linac_gen.study.engine import StudyManager
        mgr = StudyManager.load(study_dir)
        self.study_dir = Path(study_dir)
        self.param_names = [p.selector for p in mgr.spec.parameters]
        self.obs_names = [o.name for o in mgr.spec.observables]
        self.records = []
        for run in mgr.plan():
            st = mgr._status(run)
            rd = mgr._run_dir(run)
            rec = {"index": run.index, "tag": run.tag, "seed": run.seed,
                   "status": (st or {}).get("status", "pending"),
                   "params": (st or {}).get("params") or dict(run.params),
                   "metrics": (st or {}).get("metrics") or {},
                   "obs": (st or {}).get("observables") or {},
                   "error": (st or {}).get("error"),
                   "results_path": str(rd / "results.h5")}
            self.records.append(rec)

    # convenient views ------------------------------------------------
    def ok_records(self) -> list[dict]:
        return [r for r in self.records if r["status"] == "ok"]

    def value_columns(self) -> list[str]:
        return list(METRIC_KEYS) + self.obs_names

    def column(self, rec: dict, name: str):
        if name in rec["metrics"]:
            return rec["metrics"].get(name)
        return rec["obs"].get(name)


# ----------------------------------------------------------------------
# pure helpers (unit-tested)
# ----------------------------------------------------------------------
def aggregate_1d(records: list[dict], x_name: str, y_name: str,
                 group_name: str | None, column) -> dict:
    """Group records by (group value, x value) over seed repeats.

    Returns {group_value_or_None: (x[], y_mean[], y_std[], n[])},
    x-sorted.  ``column(rec, name)`` extracts a value column.
    """
    buckets: dict = {}
    for rec in records:
        y = column(rec, y_name)
        if y is None:
            continue
        x = rec["params"].get(x_name)
        if x is None:
            continue
        g = rec["params"].get(group_name) if group_name else None
        buckets.setdefault(g, {}).setdefault(float(x), []).append(
            float(y))
    out = {}
    for g, xs in buckets.items():
        xv = np.array(sorted(xs))
        ym = np.array([np.mean(xs[x]) for x in xv])
        ys = np.array([np.std(xs[x]) for x in xv])
        n = np.array([len(xs[x]) for x in xv])
        out[g] = (xv, ym, ys, n)
    return out


def detect_grid(records: list[dict], x_name: str, y_name: str,
                z_name: str, column):
    """(xu, yu, Z) when the runs form a full regular grid, else None.

    Repeats collapse by mean; Z is (len(yu), len(xu)) with NaN holes —
    a grid is declared only when every cell has at least one sample.
    """
    cells: dict = {}
    for rec in records:
        z = column(rec, z_name)
        x = rec["params"].get(x_name)
        y = rec["params"].get(y_name)
        if z is None or x is None or y is None:
            continue
        cells.setdefault((float(x), float(y)), []).append(float(z))
    if not cells:
        return None
    xu = np.array(sorted({k[0] for k in cells}))
    yu = np.array(sorted({k[1] for k in cells}))
    if len(xu) < 2 or len(yu) < 2:
        return None
    if len(cells) != len(xu) * len(yu):
        return None
    Z = np.full((len(yu), len(xu)), np.nan)
    for (x, y), vals in cells.items():
        Z[np.searchsorted(yu, y), np.searchsorted(xu, x)] = \
            float(np.mean(vals))
    return xu, yu, Z


def read_envelope(results_h5: str, quantity: str):
    """(s_mm, values) from a run's results file, or None."""
    import h5py
    try:
        with h5py.File(results_h5, "r") as f:
            env = f.get("envelope")
            if env is None or quantity not in env or "s" not in env:
                return None
            return (np.asarray(env["s"], dtype=float),
                    np.asarray(env[quantity], dtype=float))
    except OSError:
        return None


# ----------------------------------------------------------------------
# panels
# ----------------------------------------------------------------------
class _RunsPanel(QWidget):
    def __init__(self, model: StudyModel, open_results_cb=None):
        super().__init__()
        self.model = model
        self.open_results_cb = open_results_cb
        v = QVBoxLayout(self)
        self.table = QTableWidget(0, 1)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        v.addWidget(QLabel("double-click a run to open it in the "
                           "Results tab"))
        v.addWidget(self.table)

    def refresh(self) -> None:
        m = self.model
        cols = (["run", "tag", "seed", "status"] + m.param_names
                + ["transmission", "sigma_x", "sigma_y", "emit_nx",
                   "emit_ny", "ref_w_kin", "elapsed"]
                + m.obs_names + ["error"])
        self.table.setSortingEnabled(False)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(m.records))
        for r, rec in enumerate(m.records):
            vals = ([rec["index"], rec["tag"], rec["seed"],
                     rec["status"]]
                    + [rec["params"].get(p) for p in m.param_names]
                    + [rec["metrics"].get(k) for k in
                       ("transmission", "sigma_x", "sigma_y", "emit_nx",
                        "emit_ny", "ref_w_kin", "elapsed")]
                    + [rec["obs"].get(o) for o in m.obs_names]
                    + [rec["error"] or ""])
            for c, val in enumerate(vals):
                if isinstance(val, float):
                    it = QTableWidgetItem()
                    it.setData(Qt.ItemDataRole.DisplayRole,
                               round(val, 6))
                else:
                    it = QTableWidgetItem(str(val))
                if rec["status"] == "failed":
                    it.setBackground(QColor(120, 40, 40))
                elif rec["status"] == "pending":
                    it.setForeground(QColor(130, 130, 130))
                it.setData(Qt.ItemDataRole.UserRole, rec["index"])
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)

    def _on_double_click(self, item) -> None:
        if self.open_results_cb is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        rec = next((x for x in self.model.records
                    if x["index"] == idx), None)
        if rec and rec["status"] == "ok":
            self.open_results_cb(rec["results_path"])


class _Plot1DPanel(QWidget):
    def __init__(self, model: StudyModel):
        super().__init__()
        self.model = model
        v = QVBoxLayout(self)
        bar = QGridLayout()
        self.x = QComboBox()
        self.y = QComboBox()
        self.group = QComboBox()
        self.logx = QCheckBox("log x")
        self.logy = QCheckBox("log y")
        for w_, lab, col in ((self.x, "X (parameter)", 0),
                             (self.y, "Y (quantity)", 2),
                             (self.group, "Group by", 4)):
            bar.addWidget(QLabel(lab), 0, col)
            bar.addWidget(w_, 0, col + 1)
        bar.addWidget(self.logx, 0, 6)
        bar.addWidget(self.logy, 0, 7)
        v.addLayout(bar)
        self.plot = pg.PlotWidget(background="k")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(10, 10))
        v.addWidget(self.plot)
        for w_ in (self.x, self.y, self.group):
            w_.currentTextChanged.connect(self.redraw)
        self.logx.toggled.connect(self.redraw)
        self.logy.toggled.connect(self.redraw)

    def refresh_choices(self) -> None:
        m = self.model
        for combo, items in ((self.x, m.param_names),
                             (self.y, m.value_columns()),
                             (self.group, ["(none)"] + m.param_names)):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if keep in items:
                combo.setCurrentText(keep)
            combo.blockSignals(False)
        if self.y.currentText() == "elapsed" or not self.y.currentText():
            if "transmission" in m.value_columns():
                self.y.setCurrentText("transmission")
        self.redraw()

    def redraw(self, *_a) -> None:
        self.plot.clear()
        m = self.model
        xn, yn = self.x.currentText(), self.y.currentText()
        if not xn or not yn:
            return
        gn = self.group.currentText()
        gn = None if gn in ("", "(none)") else gn
        series = aggregate_1d(m.ok_records(), xn, yn, gn, m.column)
        self.plot.setLogMode(self.logx.isChecked(),
                             self.logy.isChecked())
        for i, (g, (xv, ym, ys, n)) in enumerate(
                sorted(series.items(),
                       key=lambda kv: (kv[0] is None, kv[0]))):
            color = _PALETTE[i % len(_PALETTE)]
            name = f"{gn}={g:g}" if g is not None else yn
            self.plot.plot(xv, ym, pen=pg.mkPen(color, width=2),
                           symbol="o", symbolBrush=color,
                           symbolSize=6, name=name)
            if (n > 1).any() and not self.logy.isChecked():
                err = pg.ErrorBarItem(x=xv, y=ym, height=2 * ys,
                                      pen=pg.mkPen(color))
                self.plot.addItem(err)
        self.plot.setLabel("bottom", xn)
        self.plot.setLabel("left", yn)


class _Plot2DPanel(QWidget):
    def __init__(self, model: StudyModel):
        super().__init__()
        self.model = model
        v = QVBoxLayout(self)
        bar = QGridLayout()
        self.x = QComboBox()
        self.y = QComboBox()
        self.z = QComboBox()
        for w_, lab, col in ((self.x, "X", 0), (self.y, "Y", 2),
                             (self.z, "Colour (quantity)", 4)):
            bar.addWidget(QLabel(lab), 0, col)
            bar.addWidget(w_, 0, col + 1)
        v.addLayout(bar)
        self.plot = pg.PlotWidget(background="k")
        v.addWidget(self.plot)
        self._cbar = None
        for w_ in (self.x, self.y, self.z):
            w_.currentTextChanged.connect(self.redraw)

    def refresh_choices(self) -> None:
        m = self.model
        for combo, items in ((self.x, m.param_names),
                             (self.y, m.param_names),
                             (self.z, m.value_columns())):
            keep = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if keep in items:
                combo.setCurrentText(keep)
            combo.blockSignals(False)
        if len(m.param_names) >= 2 \
                and self.y.currentText() == self.x.currentText():
            self.y.setCurrentIndex(1)
        if not self.z.currentText() \
                and "transmission" in m.value_columns():
            self.z.setCurrentText("transmission")
        self.redraw()

    def redraw(self, *_a) -> None:
        self.plot.clear()
        if self._cbar is not None:
            try:
                self.plot.getPlotItem().layout.removeItem(self._cbar)
            except Exception:                           # noqa: BLE001
                pass
            self._cbar = None
        m = self.model
        xn, yn, zn = (self.x.currentText(), self.y.currentText(),
                      self.z.currentText())
        if not xn or not yn or not zn or xn == yn:
            return
        recs = m.ok_records()
        cmap = pg.colormap.get("viridis")
        grid = detect_grid(recs, xn, yn, zn, m.column)
        if grid is not None:
            xu, yu, Z = grid
            img = pg.ImageItem(Z.T)
            img.setColorMap(cmap)
            dx = (xu[-1] - xu[0]) / max(len(xu) - 1, 1)
            dy = (yu[-1] - yu[0]) / max(len(yu) - 1, 1)
            img.setRect(pg.QtCore.QRectF(
                xu[0] - dx / 2, yu[0] - dy / 2,
                xu[-1] - xu[0] + dx, yu[-1] - yu[0] + dy))
            self.plot.addItem(img)
            zmin, zmax = np.nanmin(Z), np.nanmax(Z)
            self._cbar = pg.ColorBarItem(values=(zmin, zmax),
                                         colorMap=cmap, label=zn)
            self._cbar.setImageItem(img,
                                    insert_in=self.plot.getPlotItem())
        else:
            pts = [(rec["params"].get(xn), rec["params"].get(yn),
                    m.column(rec, zn)) for rec in recs]
            pts = [(x, y, z) for x, y, z in pts
                   if None not in (x, y, z)]
            if not pts:
                return
            xs, ys, zs = map(np.array, zip(*pts))
            zmin, zmax = zs.min(), zs.max()
            span = (zmax - zmin) or 1.0
            brushes = [pg.mkBrush(cmap.map((z - zmin) / span,
                                           mode="qcolor"))
                       for z in zs]
            sp = pg.ScatterPlotItem(x=xs, y=ys, brush=brushes, size=10,
                                    pen=pg.mkPen(None))
            self.plot.addItem(sp)
        self.plot.setLabel("bottom", xn)
        self.plot.setLabel("left", yn)
        self.plot.setTitle(zn)


class _OverlayPanel(QWidget):
    def __init__(self, model: StudyModel):
        super().__init__()
        self.model = model
        h = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel("Runs (multi-select)"))
        self.runs = QListWidget()
        self.runs.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.runs.setMaximumWidth(280)
        left.addWidget(self.runs)
        self.qty = QComboBox()
        self.qty.addItems(_OVERLAY_QTYS)
        left.addWidget(QLabel("Quantity"))
        left.addWidget(self.qty)
        draw = QPushButton("Draw")
        draw.clicked.connect(self.redraw)
        left.addWidget(draw)
        h.addLayout(left)
        self.plot = pg.PlotWidget(background="k")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(10, 10))
        h.addWidget(self.plot, stretch=1)
        self.qty.currentTextChanged.connect(self.redraw)

    def refresh_choices(self) -> None:
        self.runs.clear()
        for rec in self.model.ok_records():
            it = QListWidgetItem(f"{rec['index']:04d}  {rec['tag']}")
            it.setData(Qt.ItemDataRole.UserRole, rec["results_path"])
            self.runs.addItem(it)
        for i in range(min(3, self.runs.count())):
            self.runs.item(i).setSelected(True)
        self.redraw()

    def redraw(self, *_a) -> None:
        self.plot.clear()
        qty = self.qty.currentText()
        sel = self.runs.selectedItems()
        for i, it in enumerate(sel[:len(_PALETTE) * 2]):
            data = read_envelope(
                it.data(Qt.ItemDataRole.UserRole), qty)
            if data is None:
                continue
            s_mm, vals = data
            color = _PALETTE[i % len(_PALETTE)]
            self.plot.plot(s_mm * 1e-3, vals,
                           pen=pg.mkPen(color, width=1.5),
                           name=it.text().split("  ", 1)[-1][:40])
        self.plot.setLabel("bottom", "s [m]")
        self.plot.setLabel("left", qty)


class StudyAnalysisPanel(QTabWidget):
    """The right-hand side of the Param Study tab."""

    def __init__(self, open_results_cb=None):
        super().__init__()
        self.model = StudyModel()
        self.runs = _RunsPanel(self.model, open_results_cb)
        self.p1d = _Plot1DPanel(self.model)
        self.p2d = _Plot2DPanel(self.model)
        self.overlay = _OverlayPanel(self.model)
        self.addTab(self.runs, "Runs")
        self.addTab(self.p1d, "1D sweep")
        self.addTab(self.p2d, "Map")
        self.addTab(self.overlay, "Overlay σ(z)")

    def load_study(self, study_dir) -> None:
        self.model.load(study_dir)
        self.refresh_all()

    def refresh_all(self) -> None:
        self.runs.refresh()
        self.p1d.refresh_choices()
        self.p2d.refresh_choices()
        self.overlay.refresh_choices()
