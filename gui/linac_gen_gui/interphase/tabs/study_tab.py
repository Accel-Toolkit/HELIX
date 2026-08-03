"""Param Study tab — configure, launch, and monitor parameter studies.

Thin GUI over :mod:`linac_gen.study`: the tab builds a
:class:`~linac_gen.study.spec.StudySpec` from its tables, creates the
study folder on the GUI thread (so selector/validation errors surface
as dialogs BEFORE anything runs), and hands only the study-directory
path to a QThread worker that drives
:meth:`~linac_gen.study.engine.StudyManager.run`.  Everything on disk —
resume, per-run results, summary.csv — is the engine's contract; the
same study can be continued from the CLI and vice versa.

House rules honoured: 16 MB worker stack, bound-method slots with
stale-sender guards, no lambdas on long-lived signals, deep state reads
on the GUI thread only, ``shutdown_begin() -> list``.
"""
from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from linac_gen.study.spec import ParamSpec, StudySpec
from linac_gen.study.strategies import expand_runs

from ..dialogs.parameter_scan import _scannable_params
from ..panels.element_inspector import _SCHEMA
from ..panels.study_plots import StudyAnalysisPanel

_STRATEGIES = ("oat", "zip", "grid", "random", "lhs")


def _param_label(elem, attr: str) -> str:
    for a, label, unit in _SCHEMA.get(type(elem).__name__, []):
        if a == attr:
            return f"{label} [{unit}]" if unit else label
    return attr


class _StudyWorker(QThread):
    run_done = pyqtSignal(int, object)     # run index, row dict
    progress = pyqtSignal(object)          # StudyProgress
    failed = pyqtSignal(str)
    done = pyqtSignal(bool)                # stopped_by_user

    def __init__(self, study_dir: str, max_workers: int):
        super().__init__()
        self.setStackSize(16 * 1024 * 1024)
        self._study_dir = study_dir
        self._max_workers = int(max_workers)
        import threading
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() \
            or self.isInterruptionRequested()

    def _on_run_done(self, run, row) -> None:
        self.run_done.emit(run.index, row)

    def _on_progress(self, prog) -> None:
        self.progress.emit(prog)

    def run(self) -> None:
        try:
            from linac_gen.study.engine import StudyManager
            mgr = StudyManager.load(self._study_dir)
            mgr.run(max_workers=self._max_workers,
                    serial=self._max_workers <= 1,
                    on_run_done=self._on_run_done,
                    progress_cb=self._on_progress,
                    should_stop=self._stopping)
        except Exception as exc:                        # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(self._stop_event.is_set())


class StudyTab(QWidget):
    """Parameter Study Manager (engine lives in linac_gen.study)."""

    #: assigned by app.py — canonical SC-config builder (same seam as
    #: the Error Study tab)
    sc_config_provider = None
    #: assigned by app.py — open a results.h5 in the Results tab
    open_results_cb = None

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._worker: _StudyWorker | None = None
        self._study_dir: str | None = None
        self._run_rows: dict[int, int] = {}
        self._build_ui()
        state.lattice_changed.connect(self._on_lattice_changed)
        self._on_lattice_changed(state.lattice)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split)

        # ---- left: controls (scrolls internally) ----------------------
        left = QWidget()
        lv = QVBoxLayout(left)

        pick = QGroupBox("Parameters")
        pg = QGridLayout(pick)
        self._elem_combo = QComboBox()
        self._attr_combo = QComboBox()
        self._elem_combo.currentIndexChanged.connect(
            self._refresh_attr_combo)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_param)
        pg.addWidget(QLabel("Element"), 0, 0)
        pg.addWidget(self._elem_combo, 0, 1)
        pg.addWidget(QLabel("Parameter"), 1, 0)
        pg.addWidget(self._attr_combo, 1, 1)
        pg.addWidget(add_btn, 1, 2)

        self._ptable = QTableWidget(0, 7)
        self._ptable.setHorizontalHeaderLabels(
            ["selector", "label", "baseline", "start", "stop", "n",
             "spacing"])
        self._ptable.verticalHeader().setVisible(False)
        self._ptable.itemChanged.connect(self._refresh_run_count)
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._on_remove_param)
        pg.addWidget(self._ptable, 2, 0, 1, 3)
        pg.addWidget(rm_btn, 3, 0, 1, 3)
        lv.addWidget(pick)

        strat = QGroupBox("Strategy")
        sg = QGridLayout(strat)
        self._strategy = QComboBox()
        self._strategy.addItems(_STRATEGIES)
        self._strategy.setCurrentText("grid")
        self._strategy.currentTextChanged.connect(self._refresh_run_count)
        self._repeats = QSpinBox()
        self._repeats.setRange(1, 100)
        self._repeats.valueChanged.connect(self._refresh_run_count)
        self._seed = QSpinBox()
        self._seed.setRange(0, 10 ** 6)
        self._seed.setValue(42)
        self._nsamples = QSpinBox()
        self._nsamples.setRange(1, 100000)
        self._nsamples.setValue(32)
        self._nsamples.valueChanged.connect(self._refresh_run_count)
        self._run_count = QLabel("—")
        sg.addWidget(QLabel("Strategy"), 0, 0)
        sg.addWidget(self._strategy, 0, 1)
        sg.addWidget(QLabel("Seed repeats"), 1, 0)
        sg.addWidget(self._repeats, 1, 1)
        sg.addWidget(QLabel("Base seed"), 2, 0)
        sg.addWidget(self._seed, 2, 1)
        sg.addWidget(QLabel("Samples (random/lhs)"), 3, 0)
        sg.addWidget(self._nsamples, 3, 1)
        sg.addWidget(QLabel("Total runs:"), 4, 0)
        sg.addWidget(self._run_count, 4, 1)
        lv.addWidget(strat)

        ex = QGroupBox("Execution")
        eg = QGridLayout(ex)
        self._name = QLineEdit(time.strftime("study_%Y%m%d_%H%M"))
        self._folder = QLineEdit()
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self._on_browse)
        self._mode = QComboBox()
        self._mode.addItems(["envelope", "mp"])
        self._workers = QSpinBox()
        self._workers.setRange(1, max(1, (os.cpu_count() or 2) - 1))
        self._workers.setValue(min(4, max(1, (os.cpu_count() or 2) - 2)))
        self._start = QPushButton("Start study")
        self._start.clicked.connect(self._on_start)
        self._stop = QPushButton("Stop")
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._on_stop)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1)
        self._status = QLabel("")
        eg.addWidget(QLabel("Name"), 0, 0)
        eg.addWidget(self._name, 0, 1, 1, 2)
        eg.addWidget(QLabel("Folder"), 1, 0)
        eg.addWidget(self._folder, 1, 1)
        eg.addWidget(browse, 1, 2)
        eg.addWidget(QLabel("Mode"), 2, 0)
        eg.addWidget(self._mode, 2, 1)
        eg.addWidget(QLabel("Workers"), 3, 0)
        eg.addWidget(self._workers, 3, 1)
        eg.addWidget(self._start, 4, 0)
        eg.addWidget(self._stop, 4, 1)
        eg.addWidget(self._bar, 5, 0, 1, 3)
        eg.addWidget(self._status, 6, 0, 1, 3)
        open_btn = QPushButton("Open study…")
        open_btn.clicked.connect(self._on_open_study)
        eg.addWidget(open_btn, 4, 2)
        lv.addWidget(ex)
        lv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(430)
        split.addWidget(scroll)

        # ---- right: analysis views (runs / 1D / map / overlay) --------
        self._analysis = StudyAnalysisPanel(
            open_results_cb=self._open_run_results)
        split.addWidget(self._analysis)
        split.setStretchFactor(1, 1)

    def _beam_cfg(self):
        """The live beam config, or defaults when none is set yet."""
        cfg = self.state.beam_config
        if cfg is None:
            from linac_gen.core.config import BeamConfig
            cfg = BeamConfig()
        return cfg

    # ------------------------------------------------------------------
    # picker plumbing
    # ------------------------------------------------------------------
    def _on_lattice_changed(self, lattice) -> None:
        self._elem_combo.blockSignals(True)
        self._elem_combo.clear()
        if lattice is not None:
            self._elem_combo.addItem("Beam (input distribution)", -1)
            for i, el in enumerate(lattice.elements):
                if _scannable_params(el):
                    name = getattr(el, "name", f"@{i + 1}")
                    self._elem_combo.addItem(
                        f"@{i + 1}  {name}  ({type(el).__name__})", i)
        self._elem_combo.blockSignals(False)
        self._refresh_attr_combo()

    def _refresh_attr_combo(self) -> None:
        self._attr_combo.clear()
        lattice = self.state.lattice
        if lattice is None or self._elem_combo.count() == 0:
            return
        idx = self._elem_combo.currentData()
        if idx == -1:                               # beam pseudo-element
            cfg = self._beam_cfg()
            for f in dataclasses.fields(type(cfg)):
                v = getattr(cfg, f.name)
                if isinstance(v, (int, float)) \
                        and not isinstance(v, bool):
                    self._attr_combo.addItem(f.name, f.name)
            return
        el = lattice.elements[idx]
        for attr in _scannable_params(el):
            self._attr_combo.addItem(
                f"{attr} — {_param_label(el, attr)}", attr)

    def _on_add_param(self) -> None:
        lattice = self.state.lattice
        if lattice is None or self._attr_combo.count() == 0:
            return
        attr = self._attr_combo.currentData()
        idx = self._elem_combo.currentData()
        if idx == -1:
            cfg = self._beam_cfg()
            selector = attr
            label = f"beam.{attr}"
            base = float(getattr(cfg, attr))
        else:
            el = lattice.elements[idx]
            selector = f"@{idx + 1}.{attr}"
            label = f"{getattr(el, 'name', '?')}.{attr}"
            base = float(getattr(el, attr))
        span = abs(base) * 0.2 or 1.0
        r = self._ptable.rowCount()
        self._ptable.blockSignals(True)
        self._ptable.insertRow(r)
        for c, text in enumerate([selector, label, f"{base:g}",
                                  f"{base - span:g}", f"{base + span:g}",
                                  "5", "lin"]):
            item = QTableWidgetItem(text)
            if c in (0, 1):
                item.setFlags(item.flags()
                              & ~Qt.ItemFlag.ItemIsEditable)
            self._ptable.setItem(r, c, item)
        self._ptable.blockSignals(False)
        self._refresh_run_count()

    def _on_remove_param(self) -> None:
        rows = sorted({i.row() for i in self._ptable.selectedItems()},
                      reverse=True)
        for r in rows:
            self._ptable.removeRow(r)
        self._refresh_run_count()

    def _param_specs(self) -> list:
        specs = []
        for r in range(self._ptable.rowCount()):
            def cell(c):
                it = self._ptable.item(r, c)
                return it.text() if it else ""
            specs.append(ParamSpec(
                selector=cell(0), display_name=cell(1),
                baseline=float(cell(2)) if cell(2) else None,
                start=float(cell(3)), stop=float(cell(4)),
                n=int(float(cell(5))), spacing=cell(6) or "lin"))
        return specs

    def _draft_spec(self) -> StudySpec:
        cfg = self.state.beam_config
        beam = {}
        if cfg is not None:
            beam = {k: v for k, v in dataclasses.asdict(cfg).items()
                    if v is not None}
        numerics = {}
        if self.sc_config_provider is not None and cfg is not None:
            try:
                sc = self.sc_config_provider(getattr(cfg, "current", 0.0))
                if sc is not None:
                    numerics = {"nx": sc.nx,
                                "grid_extent": sc.grid_extent}
            except Exception:                           # noqa: BLE001
                pass
        return StudySpec(
            name=self._name.text().strip() or "study",
            input=str(self.state.lattice_path or ""),
            mode=self._mode.currentText(),
            strategy=self._strategy.currentText(),
            parameters=self._param_specs(),
            seed=int(self._seed.value()),
            repeats=int(self._repeats.value()),
            n_samples=int(self._nsamples.value()),
            beam=beam, numerics=numerics)

    def _refresh_run_count(self, *_a) -> None:
        try:
            n = len(expand_runs(self._draft_spec()))
            self._run_count.setText(str(n))
        except Exception:                               # noqa: BLE001
            self._run_count.setText("—")

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def _default_folder(self) -> str:
        base = Path.cwd() / "runs" / "studies"
        return str(base)

    def _on_browse(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Study root folder",
            self._folder.text() or self._default_folder())
        if d:
            self._folder.setText(d)

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        lattice = self.state.lattice
        path = self.state.lattice_path
        if lattice is None or not path or not Path(path).exists():
            QMessageBox.warning(
                self, "Param Study",
                "Load a lattice from a file first — studies run "
                "headless against the saved .dat.")
            return
        if getattr(self.state.bus, "dirty", False):
            QMessageBox.warning(
                self, "Param Study",
                "The lattice has unsaved edits.  Save it first — the "
                "study executes the file on disk, and refusing beats "
                "silently studying different physics.")
            return
        if self._ptable.rowCount() == 0:
            QMessageBox.warning(self, "Param Study",
                                "Add at least one parameter.")
            return
        try:
            spec = self._draft_spec()
            root = Path(self._folder.text().strip()
                        or self._default_folder())
            study_dir = root / spec.name
            from linac_gen.study.engine import StudyManager
            if (study_dir / "study.json").exists():
                mgr = StudyManager.load(study_dir)
            else:
                mgr = StudyManager.create(study_dir, spec)
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.critical(self, "Param Study",
                                 f"Study setup failed:\n{exc}")
            return
        self._study_dir = str(study_dir)
        total = len(mgr.plan())
        self._bar.setRange(0, total)
        self._bar.setValue(total - len(mgr.pending()))
        self._status.setText(f"running — {total} run(s), "
                             f"dir: {study_dir}")
        try:
            self._analysis.load_study(str(study_dir))
        except Exception:                               # noqa: BLE001
            pass
        self._worker = _StudyWorker(str(study_dir),
                                    int(self._workers.value()))
        self._worker.run_done.connect(self._on_run_done)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._on_done)
        self._start.setEnabled(False)
        self._stop.setEnabled(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._status.setText("stopping after in-flight runs …")

    # ---- worker slots (stale-sender guarded) --------------------------
    def _on_run_done(self, index: int, row: dict) -> None:
        if self.sender() is not self._worker:
            return
        if self._study_dir is None:
            return
        try:
            self._analysis.load_study(self._study_dir)
        except Exception:                               # noqa: BLE001
            pass

    def _on_progress(self, prog) -> None:
        if self.sender() is not self._worker:
            return
        self._bar.setValue(prog.done)
        eta = (f" — ETA {prog.eta_s / 60.0:.1f} min"
               if prog.eta_s else "")
        self._status.setText(
            f"{prog.done}/{prog.total} done, {prog.failed} failed{eta}")

    def _on_failed(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        self._status.setText(f"study failed: {msg}")
        self._finish_ui()

    def _on_done(self, stopped: bool) -> None:
        if self.sender() is not self._worker:
            return
        self._status.setText(
            ("stopped — resume with Start (completed runs are kept)"
             if stopped else
             f"complete — summary in {self._study_dir}/summary/"))
        if self._study_dir is not None:
            try:
                self._analysis.load_study(self._study_dir)
            except Exception:                           # noqa: BLE001
                pass
        self._finish_ui()

    def _finish_ui(self) -> None:
        self._start.setEnabled(True)
        self._stop.setEnabled(False)

    def _on_open_study(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Open study directory",
            self._folder.text() or self._default_folder())
        if not d:
            return
        try:
            self._analysis.load_study(d)
            self._study_dir = d
            self._status.setText(f"loaded study: {d}")
        except Exception as exc:                        # noqa: BLE001
            QMessageBox.critical(self, "Param Study",
                                 f"Could not load study:\n{exc}")

    def _open_run_results(self, results_path: str) -> None:
        cb = self.open_results_cb
        if cb is not None:
            cb(results_path)

    # ------------------------------------------------------------------
    def showEvent(self, ev) -> None:                    # noqa: N802
        if not self._folder.text():
            self._folder.setText(self._default_folder())
        super().showEvent(ev)

    def shutdown_begin(self) -> list:
        workers = []
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            workers.append(self._worker)
        return workers
