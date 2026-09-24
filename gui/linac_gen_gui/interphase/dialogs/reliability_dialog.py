"""Reliability Study dialog — Tools → Reliability Study…

The window front of :mod:`linac_gen.reliability`: set up a campaign on the
session lattice (preset, legs, scenario classes, circuit map, error
budget, orbit correction, availability blocks, foil scenarios, execution),
export its pending waves as a job folder for another machine and import
the results back, run it in a worker thread that only
ever holds the campaign *path* (the engine reads the deck from disk,
SHA-pinned, and runs its items in the process pool), follow it on the Run
tab, and read the results back from the campaign folder: the fault-leg
ranking and criticality bar, the compensation table with an undoable
*Apply compensator settings*, the imperfection seeds and their robustness
rows, the availability and foil tables, and the HTML report.  The
built-in self-test runs from the same window.

Nothing touches the live lattice off the GUI thread.  A campaign that was
running when the lattice or beam changed is marked stale — its numbers
stay readable but Apply refuses — and Apply goes through the command bus
as ONE ``MacroCommand`` of ``ParamChangeCommand`` (Undo reverts it).
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.app_settings import make_settings
from linac_gen_gui.interphase.commands import MacroCommand, ParamChangeCommand
from linac_gen_gui.interphase.dialogs._fingerprint import optics_fingerprint

LEGS = ("faults", "imperfections", "foil", "availability")
LEG_LABELS = {"faults": "A — fault tolerance and compensation",
              "imperfections": "C — imperfections and faults on error seeds",
              "foil": "D — foil scenarios",
              "availability": "B — availability and beam-trip budget"}
_TITLE = "Reliability Study"


def _f(x, p: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return f"{v:.{p}g}" if v == v else "—"


def _parse_floats(text: str, what: str) -> list:
    """``"0, 435 600"`` → ``[0.0, 435.0, 600.0]`` (comma / space separated;
    finite numbers only)."""
    import math
    out = []
    for tok in text.replace(";", ",").replace(",", " ").split():
        try:
            v = float(tok)
        except ValueError:
            raise ValueError(f"{what}: {tok!r} is not a number") from None
        if not math.isfinite(v):
            raise ValueError(f"{what}: {tok!r} is not a finite number")
        out.append(v)
    return out


def _parse_pairs(text: str, what: str) -> list:
    """``"1,0; 0,1"`` → ``[[1.0, 0.0], [0.0, 1.0]]``."""
    out = []
    for grp in text.split(";"):
        if not grp.strip():
            continue
        vals = _parse_floats(grp, what)
        if len(vals) != 2:
            raise ValueError(f"{what}: {grp.strip()!r} is not an x,y pair")
        out.append(vals)
    return out


def _parse_variants(text: str, what: str) -> dict:
    """``"srf_fdr=62.5, srf_sns=6"`` → ``{"srf_fdr": 62.5, "srf_sns": 6.0}``."""
    out = {}
    for tok in text.replace(";", ",").split(","):
        if not tok.strip():
            continue
        name, sep, val = tok.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"{what}: {tok.strip()!r} is not name=MTBF_hours")
        if name.strip() in out:
            raise ValueError(f"{what}: {name.strip()!r} given twice")
        try:
            v = float(val)
        except ValueError:
            raise ValueError(f"{what}: {val.strip()!r} is not a number") from None
        if not (v > 0) or v == float("inf"):
            raise ValueError(f"{what}: {name.strip()} needs a positive MTBF in hours, got {val.strip()!r}")
        out[name.strip()] = v
    return out


# ---------------------------------------------------------------------------
class _CampaignWorker(QThread):
    """Create-if-needed and run a campaign; only the folder path, the spec
    and the leg list cross the thread boundary."""

    progress = pyqtSignal(object)          # CampaignProgress
    log = pyqtSignal(str)
    done = pyqtSignal(bool)                # stopped by the user
    failed = pyqtSignal(str)

    def __init__(self, campaign_dir: str, spec, legs, max_workers: int, parent=None):
        super().__init__(parent)
        self.setStackSize(16 * 1024 * 1024)
        self._dir = str(campaign_dir)
        self._spec = spec
        self._legs = list(legs) if legs else None
        self._max_workers = int(max_workers)
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def _stopping(self) -> bool:
        return self._stop.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            from linac_gen.reliability.campaign import ReliabilityCampaign
            d = Path(self._dir)
            try:
                if (d / "campaign.json").exists():
                    c = ReliabilityCampaign.load(d)
                    self.log.emit(f"resuming {d}")
                else:
                    c = ReliabilityCampaign.create(d, self._spec)
                    plan = c.plan()
                    self.log.emit(f"created {d}: " + ", ".join(f"{k} {len(v)}" for k, v in plan.items()))
                out = c.run(self._legs, max_workers=self._max_workers,
                            serial=self._max_workers <= 1,
                            progress_cb=self.progress.emit, should_stop=self._stopping)
            except (ValueError, RuntimeError, FileExistsError, FileNotFoundError,
                    NotImplementedError) as exc:                   # the refusal idiom
                self.failed.emit(str(exc))
                return
            self.done.emit(out is None)
        except Exception as exc:                                    # noqa: BLE001
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


class _SelfTestWorker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, out_dir: str, parent=None):
        super().__init__(parent)
        self.setStackSize(16 * 1024 * 1024)
        self._out = out_dir
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            from linac_gen.reliability.selftest import run_selftest
            res = run_selftest(quick=True, out_dir=self._out, progress=self.progress.emit,
                               should_stop=lambda: self._stop.is_set() or self.isInterruptionRequested())
            self.done.emit(res)
        except Exception as exc:                                    # noqa: BLE001
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
class ReliabilityStudyDialog(QDialog):
    """Non-modal Reliability Study window."""

    def __init__(self, parent, state):
        super().__init__(parent)
        self.setWindowTitle(_TITLE)
        from linac_gen_gui.interphase.scrollwrap import screen_capped
        self.setMinimumSize(*screen_capped(self, 1080, 740))
        self.setModal(False)
        self.state = state
        self._worker: Optional[_CampaignWorker] = None
        self._selftest_worker: Optional[_SelfTestWorker] = None
        self._settings = make_settings("HELIX", "ReliabilityStudy")
        self._results_dir: Optional[Path] = None
        self._lattice_at_launch = None
        self._fp_at_launch = None
        self._stale_reason = ""
        self._applying = False
        self._budget_rows: Optional[dict] = None
        self._comp_rows: list = []
        self._build_ui()
        self._restore_session()
        for sig, slot in ((getattr(state, "lattice_changed", None), self._on_lattice_changed),
                          (getattr(state, "beam_config_changed", None), self._on_beam_changed)):
            if sig is not None:
                sig.connect(slot)

    # ---------------------------------------------------------------- UI helpers
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
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        return t

    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12); v.setSpacing(8)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_campaign_tab(), "Campaign")
        self._tabs.addTab(self._build_run_tab(), "Run")
        self._tabs.addTab(self._build_results_tab(), "Results")
        v.addWidget(self._tabs, 1)
        self._status = QLabel("no campaign")
        self._status.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(self._status)

    def _build_campaign_tab(self) -> QWidget:
        from linac_gen.reliability.spec import CLASS_IDS, CLASS_LABELS
        w = QWidget(); lay = QHBoxLayout(w); lay.setSpacing(10)
        left = QVBoxLayout(); right = QVBoxLayout()
        lay.addLayout(left, 1); lay.addLayout(right, 1)

        g = self._group("Campaign"); f = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._preset = QComboBox(); self._preset.addItems(["quick", "full", "custom"])
        self._preset.currentTextChanged.connect(self._on_preset_changed)
        f.addRow("Preset", self._preset)
        legs_box = QHBoxLayout()
        self._leg_chk = {}
        for lg in ("faults", "imperfections", "foil", "availability"):
            cb = QCheckBox(lg); cb.setChecked(True); cb.setToolTip(LEG_LABELS[lg])
            self._leg_chk[lg] = cb; legs_box.addWidget(cb)
        f.addRow("Legs", legs_box)
        self._class_list = QListWidget()
        self._class_list.setMaximumHeight(150)
        for cid in CLASS_IDS:
            it = QListWidgetItem(f"{cid} — {CLASS_LABELS[cid]}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setData(Qt.ItemDataRole.UserRole, cid)
            it.setCheckState(Qt.CheckState.Unchecked)
            self._class_list.addItem(it)
        f.addRow("Scenario classes", self._class_list)
        row = QHBoxLayout()
        self._circuits = QLineEdit(); self._circuits.setPlaceholderText("circuits.json (default: next to the deck, else derived)")
        self._circuits_browse = QPushButton("Browse…"); self._circuits_browse.clicked.connect(self._browse_circuits)
        self._circuits_gen = QPushButton("Generate from deck…"); self._circuits_gen.clicked.connect(self._generate_circuits)
        row.addWidget(self._circuits, 1); row.addWidget(self._circuits_browse); row.addWidget(self._circuits_gen)
        f.addRow("Circuit map", row)
        left.addWidget(g)

        g = self._group("Imperfections (leg C)"); f = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        row = QHBoxLayout()
        self._budget = QLineEdit(); self._budget.setPlaceholderText("error budget JSON ({\"element\": [...], \"beam\": [...]})")
        self._budget_browse = QPushButton("Browse…"); self._budget_browse.clicked.connect(self._browse_budget)
        self._budget_from_tab = QPushButton("Use Error Study tab entries"); self._budget_from_tab.clicked.connect(self._budget_from_error_tab)
        row.addWidget(self._budget, 1); row.addWidget(self._budget_browse); row.addWidget(self._budget_from_tab)
        f.addRow("Error budget", row)
        self._budget_note = QLabel("no budget — the leg is skipped"); self._budget_note.setStyleSheet(f"color:{theme.TEXT_2};")
        f.addRow("", self._budget_note)
        self._n_seeds = QSpinBox(); self._n_seeds.setRange(0, 100000); self._n_seeds.setValue(5)
        f.addRow("Seeds", self._n_seeds)
        fos = QHBoxLayout()
        self._fos_top = QSpinBox(); self._fos_top.setRange(0, 1000); self._fos_top.setValue(3)
        self._fos_seeds = QSpinBox(); self._fos_seeds.setRange(0, 100000); self._fos_seeds.setValue(2)
        fos.addWidget(QLabel("top cases")); fos.addWidget(self._fos_top); fos.addWidget(QLabel("× seeds")); fos.addWidget(self._fos_seeds); fos.addStretch(1)
        f.addRow("Faults on seeds", fos)
        cs = dict(getattr(self.state, "correction_settings", {}) or {})
        self._corr_enable = QCheckBox("Orbit correction on every seed")
        self._corr_enable.setChecked(bool(cs.get("enabled", False)))
        f.addRow("", self._corr_enable)
        self._corr_method = QComboBox(); self._corr_method.addItems(["auto", "one_to_one", "svd"])
        self._corr_method.setCurrentText(str(cs.get("method", "auto")))
        f.addRow("Method", self._corr_method)
        self._corr_n_iter = QSpinBox(); self._corr_n_iter.setRange(1, 50); self._corr_n_iter.setValue(int(cs.get("n_iter", 5)))
        f.addRow("n_iter", self._corr_n_iter)
        self._corr_tol = QDoubleSpinBox(); self._corr_tol.setRange(0.001, 100.0); self._corr_tol.setDecimals(4)
        self._corr_tol.setValue(float(cs.get("tol_mm", 0.05))); self._corr_tol.setSuffix(" mm")
        f.addRow("Tolerance", self._corr_tol)
        self._corr_noise = QDoubleSpinBox(); self._corr_noise.setRange(0.0, 10.0); self._corr_noise.setDecimals(4)
        self._corr_noise.setValue(float(cs.get("bpm_noise", 0.0))); self._corr_noise.setSuffix(" mm")
        f.addRow("BPM noise", self._corr_noise)
        self._corr_backend = QComboBox(); self._corr_backend.addItems(["envelope", "mp"])
        f.addRow("Readings", self._corr_backend)
        self._corr_pairing = QComboBox(); self._corr_pairing.addItems(["cards", "auto"])
        self._corr_pairing.setToolTip("cards: the deck's ADJUST_STEERER cards pair each steerer with a BPM (a deck "
                                      "without cards corrects nothing); auto: every steerer against every BPM "
                                      "(one-to-one when the counts match, else a global SVD)")
        f.addRow("Pairing", self._corr_pairing)
        left.addWidget(g)

        g = self._group("Availability (leg B)"); f = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        row = QHBoxLayout()
        self._blocks = QLineEdit(); self._blocks.setPlaceholderText("blocks.csv (default: the surrogate template, labelled as such)")
        self._blocks_browse = QPushButton("Browse…"); self._blocks_browse.clicked.connect(self._browse_blocks)
        self._blocks_template = QPushButton("Write surrogate template…")
        self._blocks_template.setToolTip("save the surrogate block table to edit with the project's MTBF / MTTR data")
        self._blocks_template.clicked.connect(self._write_blocks_template)
        row.addWidget(self._blocks, 1); row.addWidget(self._blocks_browse); row.addWidget(self._blocks_template)
        f.addRow("Block table", row)
        self._av_hours = QDoubleSpinBox(); self._av_hours.setRange(1.0, 8784.0); self._av_hours.setDecimals(0)
        self._av_hours.setValue(5000.0); self._av_hours.setSuffix(" h / year")
        f.addRow("Operating hours", self._av_hours)
        self._av_trials = QSpinBox(); self._av_trials.setRange(1, 1000000); self._av_trials.setValue(200)
        f.addRow("Trials", self._av_trials)
        self._av_seed = QSpinBox(); self._av_seed.setRange(0, 2**31 - 1); self._av_seed.setValue(0)
        f.addRow("Seed", self._av_seed)
        self._av_variants = QLineEdit("srf_fdr=62.5, srf_sns=6")
        self._av_variants.setToolTip("SRF-trip rate variants run side by side: name=MTBF hours of one trip, comma-separated")
        f.addRow("SRF-trip variants", self._av_variants)
        self._av_bins = QLineEdit("10, 60, 300, 1200, 3600, 14400")
        self._av_bins.setToolTip("trip-duration bin upper edges in seconds (the beam-trip budget)")
        f.addRow("Trip bins (s)", self._av_bins)
        left.addWidget(g)
        left.addStretch(1)

        g = self._group("Compensation (leg A)"); f = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._comp_top = QSpinBox(); self._comp_top.setRange(0, 10000); self._comp_top.setValue(5)
        self._comp_top.setToolTip("compensate the top-N critical cases (0 = none)")
        f.addRow("Top cases", self._comp_top)
        self._comp_iter = QSpinBox(); self._comp_iter.setRange(1, 10000); self._comp_iter.setValue(120)
        self._comp_iter.setToolTip("matcher iterations per case — one envelope run of the deck each")
        f.addRow("Matcher iterations", self._comp_iter)
        self._comp_k = QSpinBox(); self._comp_k.setRange(1, 20); self._comp_k.setValue(1)
        self._comp_k.setToolTip("k-out-of-n: ~k neighbours on each side")
        f.addRow("k neighbours", self._comp_k)
        right.addWidget(g)

        g = self._group("Foil (leg D)"); f = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._foil_thick = QLineEdit("0, 435, 600, 800")
        self._foil_thick.setToolTip("foil thicknesses to run (µg/cm²), comma-separated; 0 = the foil is missing; the deck's own thickness is the nominal scenario")
        f.addRow("Thicknesses (µg/cm²)", self._foil_thick)
        self._foil_offsets = QLineEdit("1,0; 0,1")
        self._foil_offsets.setToolTip("transverse foil offsets dx,dy in mm, pairs separated by semicolons")
        f.addRow("Offsets (mm)", self._foil_offsets)
        self._foil_thinned = QDoubleSpinBox(); self._foil_thinned.setRange(0.0, 0.99); self._foil_thinned.setDecimals(2)
        self._foil_thinned.setSingleStep(0.05); self._foil_thinned.setValue(0.5)
        self._foil_thinned.setToolTip("a thinned foil at this fraction of the nominal thickness (0 = no such scenario)")
        f.addRow("Thinned fraction", self._foil_thinned)
        self._foil_strip = QComboBox(); self._foil_strip.addItems(["two_step", "off"])
        self._foil_strip.setToolTip("stripping model of the tracked (multiparticle) foil runs")
        f.addRow("Strip model", self._foil_strip)
        self._foil_extent = QLineEdit(); self._foil_extent.setPlaceholderText("half-sizes x,y in mm — blank: unbounded")
        f.addRow("Extent (mm)", self._foil_extent)
        self._foil_hits = QDoubleSpinBox(); self._foil_hits.setRange(0.01, 10000.0); self._foil_hits.setDecimals(2)
        self._foil_hits.setValue(1.0); self._foil_hits.setToolTip("foil traversals per particle (scales the deposited power density)")
        f.addRow("Hits per particle", self._foil_hits)
        right.addWidget(g)

        g = self._group("Execution"); f = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        row = QHBoxLayout()
        self._root = QLineEdit(self._settings.value("rootDir", "", type=str))
        self._root.setPlaceholderText("campaign root folder (default: <deck dir>/reliability)")
        self._root_browse = QPushButton("Browse…"); self._root_browse.clicked.connect(self._browse_root)
        row.addWidget(self._root, 1); row.addWidget(self._root_browse)
        f.addRow("Root folder", row)
        self._name = QLineEdit(); self._name.setPlaceholderText("campaign name (default reliability_<preset>)")
        f.addRow("Name", self._name)
        self._workers = QSpinBox(); self._workers.setRange(1, max(1, os.cpu_count() or 1))
        self._workers.setValue(int(self._settings.value("maxWorkers", max(1, min(4, (os.cpu_count() or 2) - 1)), type=int)))
        self._workers.setToolTip("worker processes — the campaign is disk-backed and runs its items in a process pool")
        f.addRow("Workers", self._workers)
        right.addWidget(g)

        btns = QHBoxLayout()
        self._start_btn = QPushButton("Start"); self._accent(self._start_btn); self._start_btn.clicked.connect(self._on_start)
        self._resume_btn = QPushButton("Resume…"); self._resume_btn.setToolTip("open an existing campaign folder and run its pending items")
        self._resume_btn.clicked.connect(self._on_resume)
        self._cancel_btn = QPushButton("Cancel"); self._cancel_btn.setEnabled(False); self._cancel_btn.clicked.connect(self._cancel)
        self._selftest_btn = QPushButton("Self-test"); self._selftest_btn.setToolTip("run the built-in self-test on the shipped demo deck (quick checks)")
        self._selftest_btn.clicked.connect(self._selftest)
        self._export_btn = QPushButton("Export job…")
        self._export_btn.setToolTip("write the pending items of the checked legs as a job folder for another machine "
                                    "(python -m linac_gen reliability run <job>); completed items travel with it")
        self._export_btn.clicked.connect(self._export_job)
        self._import_btn = QPushButton("Import results…")
        self._import_btn.setToolTip("copy a finished job's results back into the campaign, re-summarise and reload the views")
        self._import_btn.clicked.connect(self._import_results)
        for b in (self._start_btn, self._resume_btn, self._cancel_btn, self._selftest_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self._export_btn); btns.addWidget(self._import_btn)
        right.addLayout(btns)
        right.addStretch(1)
        self._preset.setCurrentText(self._settings.value("preset", "quick", type=str))
        self._on_preset_changed(self._preset.currentText())
        return w

    def _build_run_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(6)
        self._bars = {}
        for lg in LEGS:
            row = QHBoxLayout()
            lab = QLabel(LEG_LABELS[lg]); lab.setMinimumWidth(320)
            bar = QProgressBar(); bar.setRange(0, 1); bar.setValue(0); bar.setFormat("%v / %m")
            self._bars[lg] = bar
            row.addWidget(lab); row.addWidget(bar, 1)
            lay.addLayout(row)
        self._overall = QLabel("idle"); self._overall.setStyleSheet(f"color:{theme.TEXT_2};")
        lay.addWidget(self._overall)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        self._log.setStyleSheet(f"font-family:{theme.FONT_MONO}; color:{theme.TEXT_2};")
        lay.addWidget(self._log, 1)
        row = QHBoxLayout()
        self._cancel_btn2 = QPushButton("Cancel"); self._cancel_btn2.setEnabled(False); self._cancel_btn2.clicked.connect(self._cancel)
        self._open_folder_btn = QPushButton("Open campaign folder"); self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._open_folder)
        row.addWidget(self._cancel_btn2); row.addWidget(self._open_folder_btn); row.addStretch(1)
        lay.addLayout(row)
        return w

    def _build_results_tab(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        self._rtabs = QTabWidget()
        # ---- leg A
        a = QWidget(); av = QVBoxLayout(a)
        self._fault_table = self._table(["case", "class", "bracket", "criticality", "critical", "ΔW (MeV)", "Δφ end (°)",
                                         "εn,x growth %", "εn,y growth %", "recovered by"])
        av.addWidget(self._fault_table, 3)
        self._bar = pg.PlotWidget()
        self._bar.setLabel("left", "criticality"); self._bar.setLabel("bottom", "failed element(s) — worst first")
        self._bar.showGrid(x=False, y=True, alpha=0.25)
        font = QFont(); font.setPointSize(7)
        self._bar.getAxis("bottom").setStyle(tickFont=font, autoExpandTextSpace=True)
        av.addWidget(self._bar, 2)
        self._comp_table = self._table(["case", "strategy", "recovered by rule", "energy verdict", "compensators", "knobs",
                                        "W after (MeV)", "εn,x after", "message"])
        av.addWidget(self._comp_table, 2)
        row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply compensator settings"); self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip("apply the selected compensation's settings to the session lattice as ONE undoable edit")
        self._apply_btn.clicked.connect(self._apply_settings)
        self._apply_status = QLabel(""); self._apply_status.setStyleSheet(f"color:{theme.TEXT_2};")
        row.addWidget(self._apply_btn); row.addWidget(self._apply_status, 1)
        av.addLayout(row)
        self._rtabs.addTab(a, "Leg A")
        # ---- leg C
        c = QWidget(); cv = QVBoxLayout(c)
        self._seed_table = self._table(["run", "seed", "kind", "status", "transmission %", "W end (MeV)", "εn,x", "εn,y",
                                        "correction", "orbit rms before (mm)", "after (mm)", "draws"])
        cv.addWidget(self._seed_table, 3)
        self._robust_table = self._table(["case", "variant", "robust fraction", "n seeds"])
        cv.addWidget(self._robust_table, 1)
        self._rtabs.addTab(c, "Leg C")
        # ---- leg B
        b = QWidget(); bv = QVBoxLayout(b)
        self._avail_table = self._table(["variant", "availability mean", "p05", "p95", "closed form", "trips / year"])
        bv.addWidget(self._avail_table, 1)
        self._avail_note = QLabel(""); self._avail_note.setStyleSheet(f"color:{theme.TEXT_2};"); self._avail_note.setWordWrap(True)
        bv.addWidget(self._avail_note)
        self._trip_table = self._table(["trip duration ≤ (s)"])
        self._trip_table.setToolTip("beam-trip budget: events per year in each duration bin, per SRF-trip variant")
        bv.addWidget(self._trip_table, 1)
        self._rtabs.addTab(b, "Leg B")
        # ---- leg D
        d = QWidget(); dv = QVBoxLayout(d)
        self._foil_table = self._table(["scenario", "label", "σx (mm)", "σy (mm)", "W/cm² peak", "stripped (tracked)",
                                        "stripped (model)", "missed"])
        dv.addWidget(self._foil_table, 1)
        self._rtabs.addTab(d, "Leg D")
        # ---- report
        r = QWidget(); rv = QVBoxLayout(r)
        self._report_view = QTextBrowser(); self._report_view.setOpenExternalLinks(True)
        rv.addWidget(self._report_view, 1)
        row = QHBoxLayout()
        self._open_report_btn = QPushButton("Open in browser"); self._open_report_btn.setEnabled(False)
        self._open_report_btn.clicked.connect(self._open_report)
        self._open_folder_btn2 = QPushButton("Open folder"); self._open_folder_btn2.setEnabled(False)
        self._open_folder_btn2.clicked.connect(self._open_folder)
        row.addWidget(self._open_report_btn); row.addWidget(self._open_folder_btn2); row.addStretch(1)
        rv.addLayout(row)
        self._rtabs.addTab(r, "Report")
        lay.addWidget(self._rtabs, 1)
        self._results_note = QLabel("no results loaded"); self._results_note.setStyleSheet(f"color:{theme.TEXT_2};")
        self._results_note.setWordWrap(True)
        lay.addWidget(self._results_note)
        return w

    # ---------------------------------------------------------------- campaign setup
    def _on_preset_changed(self, preset: str) -> None:
        from linac_gen.reliability.spec import PRESETS
        p = PRESETS.get(preset, PRESETS["quick"])
        if preset != "custom":
            wanted = set(p["classes"])
            for i in range(self._class_list.count()):
                it = self._class_list.item(i)
                it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in wanted
                                 else Qt.CheckState.Unchecked)
            self._n_seeds.setValue(int(p["seeds"]["n"]))
            self._fos_top.setValue(int(p["seeds"]["faults_on_seeds"]["top_n"]))
            self._fos_seeds.setValue(int(p["seeds"]["faults_on_seeds"]["n_seeds"]))
            self._comp_top.setValue(int(p["compensation"]["top_n"]))
            self._comp_iter.setValue(int(p["compensation"]["max_iter"]))
            self._comp_k.setValue(int(p["compensation"]["k"]))
            av = p["availability"]
            self._av_trials.setValue(int(av["n_trials"])); self._av_hours.setValue(float(av["hours_per_year"]))
            self._av_seed.setValue(int(av["seed"]))
            self._av_variants.setText(", ".join(f"{k}={v:g}" for k, v in av["variants"].items()))
            self._av_bins.setText(", ".join(f"{v:g}" for v in av["budget_bins_s"]))
            fo = p["foil"]
            self._foil_thick.setText(", ".join(f"{v:g}" for v in fo["thickness_ug_cm2"]))
            self._foil_offsets.setText("; ".join(f"{x:g},{y:g}" for x, y in fo["offsets_mm"]))
            self._foil_thinned.setValue(float(fo["thinned_fraction"] or 0.0))
            self._foil_strip.setCurrentText(str(fo["strip_model"]))
            self._foil_hits.setValue(float(fo["hits_per_particle"]))
        if not self._name.text().strip() or self._name.text().startswith("reliability_"):
            self._name.setText(f"reliability_{preset}")
        self._settings.setValue("preset", preset)

    def _checked_classes(self) -> list:
        return [self._class_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self._class_list.count())
                if self._class_list.item(i).checkState() == Qt.CheckState.Checked]

    def _deck_path(self) -> Optional[Path]:
        p = getattr(self.state, "lattice_path", None)
        return Path(p) if p and Path(p).exists() else None

    def _default_root(self) -> Path:
        d = self._deck_path()
        return (d.parent / "reliability") if d is not None else Path.cwd() / "reliability"

    def _browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Campaign root folder", self._root.text() or str(self._default_root()))
        if d:
            self._root.setText(d); self._settings.setValue("rootDir", d)

    def _browse_circuits(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Circuit map", str(self._default_root().parent), "JSON (*.json)")
        if p:
            self._circuits.setText(p)

    def _generate_circuits(self) -> None:
        lat = getattr(self.state, "lattice", None)
        if lat is None:
            QMessageBox.warning(self, _TITLE, "Load a lattice first."); return
        from linac_gen.reliability.circuits import build_circuit_map
        cm = build_circuit_map(lat)
        default = str((self._deck_path().parent if self._deck_path() else Path.cwd()) / "circuits.json")
        p, _ = QFileDialog.getSaveFileName(self, "Save the derived circuit map", default, "JSON (*.json)")
        if not p:
            return
        cm.save(p)
        self._circuits.setText(p)
        self._status.setText(f"circuit map derived from the deck: {len(cm.spares)} spare(s), foil {cm.foil_element or '—'}, "
                             f"one section — edit the file to add cryomodules, RF stations, magnet circuits, "
                             "sections and the treaty point")

    def _browse_blocks(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Reliability block table", str(self._default_root().parent), "CSV (*.csv)")
        if p:
            self._blocks.setText(p)

    def _write_blocks_template(self) -> None:
        from linac_gen.reliability.availability import write_blocks_template
        default = str((self._deck_path().parent if self._deck_path() else Path.cwd()) / "blocks.csv")
        p, _ = QFileDialog.getSaveFileName(self, "Save the surrogate block table", default, "CSV (*.csv)")
        if not p:
            return
        try:
            write_blocks_template(p)
        except OSError as exc:
            QMessageBox.critical(self, _TITLE, f"Could not write {p}:\n{exc}"); return
        self._blocks.setText(p)
        self._status.setText(f"surrogate block table written: {p} — placeholder MTBF / MTTR values, replace them "
                             "with the project's data before reading anything into the numbers")

    def _browse_budget(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Error budget JSON", str(self._default_root().parent), "JSON (*.json)")
        if p:
            self._budget.setText(p); self._budget_rows = None
            self._budget_note.setText(f"budget from {Path(p).name}")

    def _budget_from_error_tab(self) -> None:
        tab = getattr(self.parent(), "errors_tab", None)
        el = list(getattr(tab, "_element_errors", []) or [])
        bm = list(getattr(tab, "_beam_errors", []) or [])
        if not el and not bm:
            QMessageBox.information(self, _TITLE, "The Error Study tab has no error entries."); return
        self._budget_rows = {"element": [dict(r) for r in el], "beam": [dict(r) for r in bm]}
        self._budget.setText("")
        self._budget_note.setText(f"budget from the Error Study tab: {len(el)} element row(s), {len(bm)} beam row(s)")

    def _error_budget(self) -> dict:
        if self._budget.text().strip():
            doc = json.loads(Path(self._budget.text().strip()).read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                raise ValueError(f"{self._budget.text().strip()}: an error budget is a JSON object with "
                                 "element / beam rows")
            return {"element": list(doc.get("element") or []), "beam": list(doc.get("beam") or [])}
        return dict(self._budget_rows or {"element": [], "beam": []})

    def _build_spec(self, deck: Path):
        from linac_gen.reliability.spec import default_spec
        preset = self._preset.currentText()
        name = self._name.text().strip() or f"reliability_{preset}"
        cfg = getattr(self.state, "beam_config", None)
        beam = asdict(cfg) if cfg is not None else {}
        legs = {lg: cb.isChecked() for lg, cb in self._leg_chk.items()}
        corr = {"enabled": self._corr_enable.isChecked(),
                "method": None if self._corr_method.currentText() == "auto" else self._corr_method.currentText(),
                "n_iter": int(self._corr_n_iter.value()), "tol_mm": float(self._corr_tol.value()),
                "bpm_noise": float(self._corr_noise.value()), "reading_backend": self._corr_backend.currentText(),
                "pairing": self._corr_pairing.currentText()}
        blocks = self._blocks.text().strip()
        if blocks and not Path(blocks).is_file():
            raise ValueError(f"block table not found: {blocks}")
        availability = {"blocks": str(Path(blocks).resolve()) if blocks else None,
                        "hours_per_year": float(self._av_hours.value()), "n_trials": int(self._av_trials.value()),
                        "seed": int(self._av_seed.value()),
                        "variants": _parse_variants(self._av_variants.text(), "SRF-trip variants"),
                        "budget_bins_s": _parse_floats(self._av_bins.text(), "trip bins")}
        if not availability["variants"]:
            raise ValueError("SRF-trip variants: give at least one name=MTBF_hours entry")
        bins = availability["budget_bins_s"]
        if not bins or any(b <= 0 for b in bins) or any(b2 <= b1 for b1, b2 in zip(bins, bins[1:])):
            raise ValueError("trip bins: give positive, strictly increasing upper edges in seconds")
        ext = _parse_floats(self._foil_extent.text(), "foil extent")
        if ext and len(ext) not in (1, 2):
            raise ValueError("foil extent: one half-size or x,y")
        thick = _parse_floats(self._foil_thick.text(), "foil thicknesses")
        if any(t < 0 for t in thick):
            raise ValueError("foil thicknesses: a thickness cannot be negative (0 = the foil is missing)")
        foil = {"thickness_ug_cm2": thick,
                "offsets_mm": _parse_pairs(self._foil_offsets.text(), "foil offsets"),
                "thinned_fraction": float(self._foil_thinned.value()) or None,
                "strip_model": self._foil_strip.currentText(),
                "extent_mm": (ext[0] if len(ext) == 1 else ext) if ext else None,
                "hits_per_particle": float(self._foil_hits.value())}
        spec = default_spec(str(deck), preset=preset, name=name, beam=beam, legs=legs,
                            classes=self._checked_classes() or None,
                            error_budget=self._error_budget(), correction=corr,
                            seeds={"n": int(self._n_seeds.value()),
                                   "faults_on_seeds": {"top_n": int(self._fos_top.value()),
                                                       "n_seeds": int(self._fos_seeds.value())}},
                            compensation={"top_n": int(self._comp_top.value()), "max_iter": int(self._comp_iter.value()),
                                          "k": int(self._comp_k.value())},
                            availability=availability, foil=foil,
                            execution={"max_workers": int(self._workers.value()), "serial": self._workers.value() <= 1})
        circ = self._circuits.text().strip()
        if not circ and (deck.parent / "circuits.json").exists():
            circ = str(deck.parent / "circuits.json")
        if circ:
            spec.circuits = str(Path(circ).resolve())
        spec.validate_shape()
        return spec

    # ---------------------------------------------------------------- start / resume / cancel
    def _refuse_if_not_ready(self) -> bool:
        if self._busy():                                # a campaign or the self-test
            return True
        lat = getattr(self.state, "lattice", None)
        deck = self._deck_path()
        if lat is None or deck is None:
            QMessageBox.warning(self, _TITLE, "Load a lattice from a file first — a campaign executes the saved deck.")
            return True
        if getattr(getattr(self.state, "bus", None), "dirty", False):
            QMessageBox.warning(self, _TITLE, "The lattice has unsaved edits.  Save it first — the campaign executes the "
                                              "file on disk, and refusing beats silently studying different physics.")
            return True
        return False

    def _on_start(self) -> None:
        if self._refuse_if_not_ready():
            return
        deck = self._deck_path()
        try:
            spec = self._build_spec(deck)
        except Exception as exc:                                    # noqa: BLE001
            QMessageBox.critical(self, _TITLE, f"Campaign setup failed:\n{exc}"); return
        root = Path(self._root.text().strip() or self._default_root())
        cdir = root / spec.name
        legs = [lg for lg, cb in self._leg_chk.items() if cb.isChecked()]
        self._launch(cdir, spec, legs)

    def _on_resume(self) -> None:
        if self._busy():
            return
        d = QFileDialog.getExistingDirectory(self, "Existing campaign folder",
                                             self._settings.value("lastCampaignDir", "", type=str) or str(self._default_root()))
        if not d:
            return
        if not (Path(d) / "campaign.json").exists():
            QMessageBox.warning(self, _TITLE, f"{d} holds no campaign.json"); return
        self._launch(Path(d), None, None)

    def _launch(self, cdir: Path, spec, legs) -> None:
        self._results_dir = Path(cdir)
        self._settings.setValue("lastCampaignDir", str(cdir))
        if self._root.text().strip():
            self._settings.setValue("rootDir", self._root.text().strip())
        self._settings.setValue("maxWorkers", int(self._workers.value()))
        self._lattice_at_launch = getattr(self.state, "lattice", None)
        self._fp_at_launch = optics_fingerprint(self._lattice_at_launch, getattr(self.state, "beam_config", None))
        self._stale_reason = ""
        for bar in self._bars.values():
            bar.setRange(0, 1); bar.setValue(0)
        self._log.clear()
        self._log.appendPlainText(f"campaign: {cdir}")
        w = _CampaignWorker(str(cdir), spec, legs, int(self._workers.value()), parent=self)
        w.progress.connect(self._on_progress)
        w.log.connect(self._on_log)
        w.done.connect(self._on_done)
        w.failed.connect(self._on_failed)
        self._worker = w
        self._set_busy(True)
        self._tabs.setCurrentIndex(1)
        self._save_session()
        w.start()

    def _cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._overall.setText("cancelling — the running items finish, queued ones are dropped …")
        if self._selftest_worker is not None and self._selftest_worker.isRunning():
            self._selftest_worker.request_stop()

    def _busy(self) -> bool:
        return any(w is not None and w.isRunning() for w in (self._worker, self._selftest_worker))

    def _set_busy(self, busy: bool) -> None:
        for b in (self._start_btn, self._resume_btn, self._selftest_btn, self._circuits_gen, self._budget_from_tab,
                  self._blocks_template, self._export_btn, self._import_btn):
            b.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy); self._cancel_btn2.setEnabled(busy)
        self._open_folder_btn.setEnabled(self._results_dir is not None)
        self._open_folder_btn2.setEnabled(self._results_dir is not None)
        if busy:
            self._status.setText(f"running — {self._results_dir}")
            self._overall.setText("running …")

    # ---------------------------------------------------------------- worker slots
    def _on_log(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        self._log.appendPlainText(msg)

    def _on_progress(self, p) -> None:
        if self.sender() is not self._worker:
            return
        bar = self._bars.get(getattr(p, "leg", ""))
        if bar is not None:
            bar.setRange(0, max(1, int(p.total))); bar.setValue(int(p.done))
        eta = f", ETA {p.eta_s / 60.0:.1f} min" if getattr(p, "eta_s", None) else ""
        self._overall.setText(f"{p.leg} / {p.phase}: {p.done}/{p.total} ({p.failed} failed){eta}")

    def _on_failed(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        self._set_busy(False)
        self._overall.setText("failed")
        self._status.setText(f"failed — {msg.splitlines()[0][:120]}")
        self._log.appendPlainText("FAILED: " + msg)
        QMessageBox.critical(self, f"{_TITLE} failed", msg)

    def _on_done(self, stopped: bool) -> None:
        if self.sender() is not self._worker:
            return
        self._set_busy(False)
        self._overall.setText("stopped — Resume… continues the pending items" if stopped else "complete")
        if (getattr(self.state, "lattice", None) is not self._lattice_at_launch
                or optics_fingerprint(getattr(self.state, "lattice", None),
                                      getattr(self.state, "beam_config", None)) != self._fp_at_launch):
            self._mark_stale("the lattice or beam changed while the campaign was running")
        self._load_results(self._results_dir)
        if not stopped:
            self._tabs.setCurrentIndex(2)
        self._save_session()

    # ---------------------------------------------------------------- job export / import
    def _campaign_for_job(self):
        """The loaded campaign, or None with the reason shown."""
        if self._busy():
            return None
        cdir = self._results_dir
        if cdir is None or not (Path(cdir) / "campaign.json").exists():
            QMessageBox.information(self, _TITLE, "Start or resume a campaign first — jobs are exported from, and "
                                                  "imported into, its folder.")
            return None
        from linac_gen.reliability.campaign import ReliabilityCampaign
        try:
            return ReliabilityCampaign.load(cdir)
        except (OSError, ValueError, RuntimeError) as exc:            # the refusal idiom
            QMessageBox.critical(self, _TITLE, f"The campaign folder refuses to load:\n{exc}")
            return None

    def _export_job(self) -> None:
        c = self._campaign_for_job()
        if c is None:
            return
        legs = [lg for lg, cb in self._leg_chk.items() if cb.isChecked()] or None
        last = self._settings.value("lastExportDir", "", type=str)
        default = str(Path(last or c.dir.parent) / f"{c.dir.name}_job")
        p, _ = QFileDialog.getSaveFileName(self, "Job folder to write (a new folder)", default, "")
        if not p:
            return
        from PyQt6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)   # copies the completed items: seconds on a linac campaign
        try:
            from linac_gen.reliability.jobs import export_job
            job = export_job(c, p, legs=legs)
            n = len(json.loads((job / "job.json").read_text(encoding="utf-8"))["expected"])
        except (OSError, ValueError, RuntimeError) as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, _TITLE, f"Export refused:\n{exc}"); return
        QApplication.restoreOverrideCursor()
        self._settings.setValue("lastExportDir", str(Path(p).parent))
        pending = len([it for it in c.pending() if not legs or it["leg"] in legs])
        msg = (f"job written: {job} — {n} item(s) of {', '.join(legs or list(c.plan()))}, {pending} pending; "
               f"the exact remote command is in README.txt, then Import results…")
        self._status.setText(msg)
        self._log.appendPlainText(msg)
        QMessageBox.information(self, _TITLE, msg)

    def _import_results(self) -> None:
        c = self._campaign_for_job()
        if c is None:
            return
        last = self._settings.value("lastExportDir", "", type=str)
        d = QFileDialog.getExistingDirectory(self, "Finished job folder", last or str(c.dir.parent))
        if not d:
            return
        if not (Path(d) / "job.json").exists():
            QMessageBox.warning(self, _TITLE, f"{d} holds no job.json — pick a folder written by Export job…"); return
        from PyQt6.QtWidgets import QApplication
        from linac_gen.reliability.jobs import import_results
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            try:
                rec = import_results(d, c)
            except RuntimeError as exc:
                if "missing" not in str(exc):
                    raise
                QApplication.restoreOverrideCursor()
                why = str(exc).replace(" (use --allow-partial to import the rest)", "")
                ans = QMessageBox.question(self, _TITLE, f"{why}\n\nImport the completed items anyway?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if ans != QMessageBox.StandardButton.Yes:
                    return
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                rec = import_results(d, c, allow_partial=True)
            c.summarize()
        except (OSError, ValueError, RuntimeError) as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, _TITLE, f"Import refused:\n{exc}"); return
        QApplication.restoreOverrideCursor()
        self._settings.setValue("lastExportDir", str(Path(d).parent))
        msg = (f"imported {len(rec['imported'])} item(s) from {d}, {len(rec['skipped_ok'])} already complete, "
               f"{len(rec['missing'])} still missing")
        self._log.appendPlainText(msg)
        self._load_results(c.dir)
        self._status.setText(msg)
        self._save_session()

    # ---------------------------------------------------------------- results
    def _load_results(self, cdir) -> None:
        if cdir is None or not (Path(cdir) / "campaign.json").exists():
            self._results_note.setText("no results loaded"); return
        cdir = Path(cdir)
        from linac_gen.reliability.summary import read_csv
        legs = cdir / "legs"

        def rows(rel):
            p = legs / rel
            return read_csv(p) if p.exists() else []
        faults = rows("faults/faults.csv")
        comps = rows("faults/compensation.csv")
        seeds = rows("imperfections/seeds.csv")
        robust = rows("imperfections/robustness.csv")
        avail = rows("availability/availability.csv")
        trips = rows("availability/trip_histogram.csv")
        foil = rows("foil/foil.csv")
        self._fill_fault_table(faults)
        self._render_bar(faults)
        self._fill_comp_table(comps)
        self._fill_seed_tables(seeds, robust)
        self._fill_avail(avail, cdir)
        self._fill_trips(trips)
        self._fill_foil(foil)
        self._load_report(cdir)
        summ = {}
        try:
            summ = json.loads((cdir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        fa = summ.get("faults") or {}
        note = (f"{cdir.name}: {fa.get('n_cases', 0)} fault case(s), {fa.get('n_critical', 0)} critical; "
                f"compensation {summ.get('compensation', {}).get('recovered', 0)} recovered by the rule of "
                f"{summ.get('compensation', {}).get('n', 0)}; {len([s for s in seeds if s.get('kind') == 'seed'])} seed(s); "
                f"{len(foil)} foil scenario(s); availability "
                + (f"FAILED: {str(summ.get('availability_error'))[:80]}" if summ.get("availability_error")
                   else ("run" if avail else "not run")))
        if self._stale_reason:
            note += f"  —  STALE: {self._stale_reason} (Apply refuses)"
        self._results_note.setText(note)
        self._status.setText(f"results: {cdir}")
        self._open_folder_btn.setEnabled(True); self._open_folder_btn2.setEnabled(True)
        self._apply_btn.setEnabled(bool(self._comp_rows) and self._apply_allowed())

    def _fill_fault_table(self, faults) -> None:
        t = self._fault_table
        t.setRowCount(0)
        rows = [r for r in faults if r.get("case_id") != "baseline"]
        rows.sort(key=lambda r: -(float(r["criticality"]) if r.get("criticality") else -1.0))
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r.get("case_id"), r.get("class"), r.get("bracket"), _f(r.get("criticality")), r.get("critical"),
                    _f(r.get("d_energy_mev")), _f(r.get("d_phi_end_deg"), 3), _f(r.get("emit_nx_growth_pct"), 3),
                    _f(r.get("emit_ny_growth_pct"), 3), r.get("recovered_by")]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))

    def _render_bar(self, faults) -> None:
        self._bar.clear()
        rows = [r for r in faults if r.get("criticality") and r.get("bracket") == "rephased" and r.get("case_id") != "baseline"]
        rows.sort(key=lambda r: -float(r["criticality"]))
        top = rows[:15]
        if not top:
            self._bar.getAxis("bottom").setTicks(None); return
        ys = [float(r["criticality"]) for r in top]
        xs = list(range(len(ys)))
        self._bar.addItem(pg.BarGraphItem(x=xs, height=ys, width=0.7, brush=theme.ACCENT))
        self._bar.getAxis("bottom").setTicks([[(i, (top[i].get("elements") or top[i]["case_id"]).replace(";", "+"))
                                               for i in range(len(top))]])
        self._bar.setYRange(0.0, max(ys) * 1.1, padding=0)

    def _fill_comp_table(self, comps) -> None:
        self._comp_rows = list(comps)
        t = self._comp_table
        t.setRowCount(len(comps))
        for i, c in enumerate(comps):
            vals = [c.get("case_id"), c.get("strategy"), c.get("recovered_by_rule"), c.get("recovered"),
                    (c.get("compensators") or "").replace(";", ", "), c.get("n_knobs"), _f(c.get("ref_w_kin_after"), 7),
                    _f(c.get("emit_nx_after")), (c.get("error") or c.get("message") or "")[:60]]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))

    def _fill_seed_tables(self, seeds, robust) -> None:
        t = self._seed_table
        t.setRowCount(len(seeds))
        for i, s in enumerate(seeds):
            vals = [s.get("id"), s.get("seed"), s.get("kind"), s.get("status"), _f(s.get("transmission")),
                    _f(s.get("ref_w_kin"), 7), _f(s.get("emit_nx")), _f(s.get("emit_ny")), s.get("correction_status"),
                    _f(s.get("orbit_rms_before_mm"), 3), _f(s.get("orbit_rms_after_mm"), 3), s.get("n_draws")]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))
        r = self._robust_table
        r.setRowCount(len(robust))
        for i, s in enumerate(robust):
            for j, v in enumerate([s.get("case_id"), s.get("variant"), _f(s.get("robust_fraction"), 3), s.get("n")]):
                r.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))

    def _fill_avail(self, avail, cdir: Path) -> None:
        t = self._avail_table
        t.setRowCount(len(avail))
        for i, a in enumerate(avail):
            vals = [a.get("variant"), _f(a.get("availability_mean"), 5), _f(a.get("availability_p05"), 5),
                    _f(a.get("availability_p95"), 5), _f(a.get("availability_analytic"), 5), _f(a.get("trips_per_year"), 5)]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))
        src = ""
        try:
            src = json.loads((cdir / "legs" / "availability" / "availability.json").read_text(encoding="utf-8")).get("source", "")
        except (OSError, json.JSONDecodeError):
            pass
        self._avail_note.setText(f"block source: {src}" if src else "")

    def _fill_trips(self, trips) -> None:
        t = self._trip_table
        t.setRowCount(0)
        variants = [k for k in (trips[0].keys() if trips else []) if k != "bin_upper_s"]
        t.setColumnCount(1 + len(variants))
        t.setHorizontalHeaderLabels(["trip duration ≤ (s)"] + [f"{v} (events / year)" for v in variants])
        t.setRowCount(len(trips))
        for i, r in enumerate(trips):
            t.setItem(i, 0, QTableWidgetItem(str(r.get("bin_upper_s", ""))))
            for j, v in enumerate(variants):
                t.setItem(i, 1 + j, QTableWidgetItem(_f(r.get(v), 5)))

    def _fill_foil(self, foil) -> None:
        t = self._foil_table
        t.setRowCount(len(foil))
        for i, r in enumerate(foil):
            vals = [r.get("case_id"), r.get("label"), _f(r.get("foil_sigma_x_mm")), _f(r.get("foil_sigma_y_mm")),
                    _f(r.get("foil_w_cm2_peak")), _f(r.get("strip_eff"), 6), _f(r.get("strip_eff_analytic"), 6),
                    _f(r.get("missed_frac"))]
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))

    def _load_report(self, cdir: Path) -> None:
        p = cdir / "report.html"
        if p.exists():
            self._report_view.setSearchPaths([str(cdir)])
            self._report_view.setHtml(p.read_text(encoding="utf-8"))
            self._open_report_btn.setEnabled(True)
        else:
            self._report_view.setPlainText("no report.html yet")
            self._open_report_btn.setEnabled(False)

    def _open_report(self) -> None:
        if self._results_dir is not None and (self._results_dir / "report.html").exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._results_dir / "report.html")))

    def _open_folder(self) -> None:
        if self._results_dir is not None and self._results_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._results_dir)))

    # ---------------------------------------------------------------- staleness / apply
    def _mark_stale(self, reason: str) -> None:
        self._stale_reason = reason
        self._apply_btn.setEnabled(False)
        self._apply_status.setText(f"stale: {reason} — Apply refuses")
        try:
            self.state.status_message.emit(f"Reliability Study results marked stale — {reason}")
        except Exception:                                           # noqa: BLE001
            pass

    def _apply_allowed(self) -> bool:
        if self._stale_reason or self._results_dir is None:
            return False
        lat = getattr(self.state, "lattice", None)
        if lat is None:
            return False
        if self._lattice_at_launch is not None:
            return (lat is self._lattice_at_launch
                    and optics_fingerprint(lat, getattr(self.state, "beam_config", None)) == self._fp_at_launch)
        # loaded from disk: the session deck must be the campaign's, unedited
        deck = self._deck_path()
        try:
            from linac_gen.reliability.campaign import ReliabilityCampaign, _sha256
            c = ReliabilityCampaign.load(self._results_dir)
            return (deck is not None and c.input_path.resolve() == deck.resolve()
                    and not getattr(getattr(self.state, "bus", None), "dirty", False)
                    and _sha256(deck) == c.spec.lattice_sha256)
        except Exception:                                           # noqa: BLE001
            return False

    def _apply_settings(self) -> None:
        if not self._apply_allowed():
            self._mark_stale(self._stale_reason or "the session lattice is not the campaign's deck");
            QMessageBox.warning(self, _TITLE, "The session lattice is not (or no longer) the one the campaign ran on — "
                                              "reload the deck or rerun the campaign."); return
        i = self._comp_table.currentRow()
        if i < 0 or i >= len(self._comp_rows):
            QMessageBox.information(self, _TITLE, "Select a compensation row first."); return
        row = self._comp_rows[i]
        try:
            settings = json.loads(row.get("settings") or "{}")
        except json.JSONDecodeError:
            settings = {}
        if not settings:
            QMessageBox.information(self, _TITLE, "That compensation carries no settings."); return
        lat = self.state.lattice
        by_name = {}
        for e in lat.elements:
            nm = getattr(e, "name", None)
            if nm:
                by_name.setdefault(nm, e)
        cmds = []
        for key, new in settings.items():
            name, _, attr = key.rpartition(".")
            el = by_name.get(name)
            if el is None or not hasattr(el, attr):
                QMessageBox.warning(self, _TITLE, f"{key}: no such element/attribute in the session lattice"); return
            old = getattr(el, attr)
            if old != new:
                cmds.append(ParamChangeCommand(el, attr, old, float(new)))
        if not cmds:
            self._apply_status.setText("nothing to apply — the lattice already carries these settings"); return
        self._applying = True
        try:
            self.state.bus.do(MacroCommand(cmds, label=f"Reliability compensation {row.get('case_id')} ({row.get('strategy')})"))
        finally:
            self._applying = False
        try:
            self.state.lattice_fitted = True
        except Exception:                                           # noqa: BLE001
            pass
        msg = f"applied {len(cmds)} setting(s) of {row.get('case_id')} / {row.get('strategy')} — Undo reverts it"
        self._apply_status.setText(msg)
        try:
            self.state.status_message.emit("Reliability Study: " + msg)
        except Exception:                                           # noqa: BLE001
            pass
        # the applied lattice differs from the campaign's deck: results stay, Apply is done
        self._lattice_at_launch = None
        self._apply_btn.setEnabled(False)

    def _on_lattice_changed(self, lat) -> None:
        try:
            self._fault_table.rowCount()                # RuntimeError once the C++ widget is gone
        except RuntimeError:
            for sig, slot in ((getattr(self.state, "lattice_changed", None), self._on_lattice_changed),
                              (getattr(self.state, "beam_config_changed", None), self._on_beam_changed)):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError, AttributeError):
                    pass
            return
        if self._applying or self._results_dir is None:
            return
        if self._lattice_at_launch is not None and lat is not self._lattice_at_launch:
            self._mark_stale("lattice replaced")
        elif self._lattice_at_launch is not None and optics_fingerprint(
                lat, getattr(self.state, "beam_config", None)) != self._fp_at_launch:
            self._mark_stale("lattice edited since the campaign ran")
        self._apply_btn.setEnabled(bool(self._comp_rows) and self._apply_allowed())

    def _on_beam_changed(self, _cfg) -> None:
        try:
            self._fault_table.rowCount()
        except RuntimeError:
            return
        if self._results_dir is not None and self._lattice_at_launch is not None and optics_fingerprint(
                getattr(self.state, "lattice", None), getattr(self.state, "beam_config", None)) != self._fp_at_launch:
            self._mark_stale("beam changed since the campaign ran")

    # ---------------------------------------------------------------- self-test
    def _selftest(self) -> None:
        if (self._worker is not None and self._worker.isRunning()) or \
                (self._selftest_worker is not None and self._selftest_worker.isRunning()):
            return
        import tempfile
        out = tempfile.mkdtemp(prefix="helix_reliability_selftest_")
        w = _SelfTestWorker(out, parent=self)
        w.done.connect(self._on_selftest_done)
        w.failed.connect(self._on_selftest_failed)
        w.progress.connect(self._on_selftest_progress)
        self._selftest_worker = w
        self._set_busy(True)                            # Start / Resume / Export / Import wait too
        self._log.clear(); self._log.appendPlainText("self-test: quick checks on the shipped demo deck …")
        self._overall.setText("self-test running …")
        self._status.setText("self-test running …")
        self._tabs.setCurrentIndex(1)
        w.start()

    def _on_selftest_progress(self, p) -> None:
        if self.sender() is not self._selftest_worker:
            return
        self._overall.setText(f"self-test: {getattr(p, 'leg', '')} / {getattr(p, 'phase', '')} {getattr(p, 'done', 0)}/{getattr(p, 'total', 0)}")

    def _on_selftest_failed(self, msg: str) -> None:
        if self.sender() is not self._selftest_worker:
            return
        self._set_busy(False)
        self._overall.setText("self-test failed")
        self._log.appendPlainText("SELF-TEST FAILED: " + msg)
        QMessageBox.critical(self, f"{_TITLE} self-test", msg)

    def _on_selftest_done(self, res: dict) -> None:
        if self.sender() is not self._selftest_worker:
            return
        from linac_gen.reliability.selftest import format_result
        self._set_busy(False)
        self._selftest_result = res
        text = format_result(res)
        self._log.appendPlainText(text)
        self._overall.setText(f"self-test {res.get('verdict')}: {res.get('n_checks')} checks, {res.get('n_failed')} failed")
        self._status.setText(f"self-test {res.get('verdict')}")
        QMessageBox.information(self, f"{_TITLE} self-test",
                                f"VERDICT {res.get('verdict')} — {res.get('n_checks')} checks, {res.get('n_failed')} failed "
                                f"({res.get('elapsed_s', 0):.1f} s).\nThe full table is on the Run tab.")

    # ---------------------------------------------------------------- session
    def _session(self) -> dict:
        s = getattr(self.state, "reliability_session", None)
        if not isinstance(s, dict):
            s = {}
            try:
                self.state.reliability_session = s
            except AttributeError:
                pass
        return s

    def _save_session(self) -> None:
        s = self._session()
        s.update({"campaign_dir": str(self._results_dir) if self._results_dir else None,
                  "report_path": str(self._results_dir / "report.html") if self._results_dir else None,
                  "stale": self._stale_reason})

    def _restore_session(self) -> None:
        s = self._session()
        try:
            d = s.get("campaign_dir")
            if d and (Path(d) / "campaign.json").exists():
                self._results_dir = Path(d)
                self._stale_reason = s.get("stale") or ""
                self._load_results(self._results_dir)
            elif d:
                s.update({"campaign_dir": None, "report_path": None})
        except Exception as exc:                                    # noqa: BLE001 — a broken session must never block the dialog
            self._results_dir = None
            s.update({"campaign_dir": None, "report_path": None})
            self._status.setText(f"previous session dropped: {exc}")

    # ---------------------------------------------------------------- lifecycle
    def shutdown_begin(self) -> list:
        workers = []
        for w in (self._worker, self._selftest_worker):
            if w is not None and w.isRunning():
                w.request_stop()
                workers.append(w)
        return workers

    def closeEvent(self, ev) -> None:
        self._cancel()
        for w in (self._worker, self._selftest_worker):
            if w is not None and w.isRunning():
                w.wait(5000)
        self._save_session()
        super().closeEvent(ev)
