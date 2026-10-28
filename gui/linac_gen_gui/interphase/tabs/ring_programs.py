"""The Ring tab's Programs page — the ramp: the reference energy, the RF
voltage and the RF phase versus time, transition, paraphasing.

Each turn, the RF phase gives the reference particle the energy gain the
energy program asks for (see :mod:`linac_gen.ring.programs`); the magnets
follow the reference.  Times on the page are in ms from the start of the
cycle (seconds in the project and in CSV files, whose header names the
columns: ``t_s`` and ``kinetic_mev``, ``momentum_mevc``, ``total_mev``,
``voltage_mv`` or ``phi_deg``).  Every valid change goes into the project's
ring settings (``ramp``, ``corrector_mode``); an invalid one is reported and
the project keeps the last valid programs.  Preview runs the reference
particle alone through the ramp (off the GUI thread): the turns to the end
of the program, transition, the RF frequency, the gain against the voltage.
"""
from __future__ import annotations

import contextlib
import math
import re
from decimal import Decimal, DecimalException

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme

_NOTE = ("A ramp sets the reference energy, the RF voltage and the RF phase turn by turn. "
         "The magnets follow the reference (their normalized strengths stay); each turn's RF "
         "phase gives the reference the gain the energy program asks for, mirrored above "
         "transition. Times in ms from the start of the cycle.")

_ENERGY_KINDS = (("Fixed energy (no programs)", None),
                 ("Flat: fixed energy, with the voltage and phase programs", "flat"),
                 ("Resonant cycle: sine in momentum", "sine"),
                 ("Table", "table"))
_QUANTITIES = (("kinetic energy (MeV)", "kinetic_mev"), ("momentum (MeV/c)", "momentum_mevc"),
               ("total energy (MeV)", "total_mev"))
_VOLTAGE_KINDS = (("The cavities' own voltages", "constant"), ("Table", "table"))
_PHASE_KINDS = (("From the energy gain the turn needs", "auto"),
                ("Table of the synchronous phase", "table"))
_ON_EXCESS = (("cap at 90 deg and count the turns", "cap"),
              ("refuse: the run stops at that turn", "refuse"))
_CORRECTORS = (("follow the reference momentum", "follow"), ("keep their field", "fixed_field"))


def _index(choices, key) -> int:
    keys = [k for _l, k in choices]
    return keys.index(key) if key in keys else 0


def _num(text: str, what: str, *, blank=None):
    t = text.strip()
    if not t:
        if blank is ...:
            raise ValueError(f"{what}: give a number")
        return blank
    try:
        v = float(t)
    except ValueError:
        raise ValueError(f"{what}: {t!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{what}: {t!r} is not a finite number")
    return v


def _ms_to_s(text: str, what: str, *, blank=None):
    """A time typed in ms → seconds, by a decimal shift of the text (no
    rounding beyond the conversion to a float: '0.8' → 0.0008 exactly as
    float('0.8e-3'))."""
    t = text.strip()
    if not t:
        if blank is ...:
            raise ValueError(f"{what}: give a number")
        return blank
    try:
        d = Decimal(t)
        v = float(d.scaleb(-3)) if d.is_finite() else math.nan
    except (DecimalException, ArithmeticError, ValueError):
        raise ValueError(f"{what}: {t!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{what}: {t!r} is not a finite number")
    return v


def _fmt(v) -> str:
    """A number as text that reads back to the same float."""
    if v is None:
        return ""
    r = repr(float(v))
    return r[:-2] if r.endswith(".0") else r


def _fmt_ms(t_s) -> str:
    """Seconds → the ms text, by a decimal shift of the shortest repr."""
    d = Decimal(repr(float(t_s))).scaleb(3)
    txt = format(d, "f")
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt if txt not in ("", "-0") else "0"


def commit_cell_editor(table) -> None:
    """A cell of ``table`` being edited (its editor open) takes what was typed
    into it, and the edit ends — as when the cell loses the focus; the table's
    ``itemChanged`` then takes the value (a Save does not end an edit)."""
    from PyQt6.QtWidgets import QAbstractItemDelegate, QAbstractItemView
    if table.state() != QAbstractItemView.State.EditingState:
        return
    model = table.model()
    for r in range(table.rowCount()):
        for c in range(table.columnCount()):
            editor = table.indexWidget(model.index(r, c))
            if editor is not None:
                table.commitData(editor)
                table.closeEditor(editor, QAbstractItemDelegate.EndEditHint.NoHint)
                return


class TableEditor(QWidget):
    """Two columns — the time in ms and a value — with rows added and removed,
    read from and written to CSV files (times in s there).  ``imported`` carries
    the file's value column (for the energy table: what the values are), sent
    before ``changed``; ``failed`` the reason an import or export did not
    happen."""

    changed = pyqtSignal()
    imported = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, program: str, value_label: str, column: str, column_provider=None):
        super().__init__()
        self.program = program          # energy / voltage / phase (the CSV columns)
        self.column = column            # the value column written to CSV
        # (the energy table's column is its quantity, chosen on the page)
        self.column_provider = column_provider
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["t (ms)", value_label])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(120)
        self.table.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.table)
        row = QHBoxLayout()
        for text, slot in (("Add row", self.add_row), ("Remove row", self.remove_row),
                           ("Import CSV…", self._import_clicked),
                           ("Export CSV…", self._export_clicked)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

    def _on_item_changed(self, _item) -> None:
        self.changed.emit()

    def add_row(self, *_a) -> None:
        n = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(n)
        self.table.setItem(n, 0, QTableWidgetItem(""))
        self.table.setItem(n, 1, QTableWidgetItem(""))
        self.table.blockSignals(False)

    def remove_row(self, *_a) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for r in rows:
            self.table.removeRow(r)
        self.changed.emit()

    def set_points(self, times_s, values) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for t, v in zip(times_s or (), values or (), strict=True):
            n = self.table.rowCount()
            self.table.insertRow(n)
            self.table.setItem(n, 0, QTableWidgetItem(_fmt_ms(t)))
            self.table.setItem(n, 1, QTableWidgetItem(_fmt(v)))
        self.table.blockSignals(False)

    def points(self, what: str) -> tuple[list, list]:
        """(times [s], values); blank rows are skipped, anything else that is
        not a number is an error naming the row."""
        times, values = [], []
        for r in range(self.table.rowCount()):
            cells = [(self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
                     for c in (0, 1)]
            if not any(cells):
                continue
            times.append(_ms_to_s(cells[0], f"{what}, row {r + 1}, time", blank=...))
            values.append(_num(cells[1], f"{what}, row {r + 1}, value", blank=...))
        return times, values

    def _import_clicked(self, *_a) -> None:
        path, _f = QFileDialog.getOpenFileName(self, f"Import the {self.program} table", "",
                                               "CSV (*.csv);;All files (*)")
        if path:
            try:
                self.import_csv(path)
            except (ValueError, OSError, UnicodeError) as exc:
                self.failed.emit(f"Import: {exc}")

    def _export_clicked(self, *_a) -> None:
        path, _f = QFileDialog.getSaveFileName(self, f"Export the {self.program} table",
                                               f"{self.program}.csv",
                                               "CSV (*.csv);;All files (*)")
        if path:
            try:
                self.export_csv(path)
            except (ValueError, OSError) as exc:
                self.failed.emit(f"Export: {exc}")

    def import_csv(self, path: str) -> str:
        """Read a CSV table; returns (and sends) the file's value column."""
        from linac_gen.ring.programs import read_table_csv
        times, values, column = read_table_csv(path, self.program)
        self.set_points(times, values)
        self.imported.emit(column)
        self.changed.emit()
        return column

    def export_csv(self, path: str) -> None:
        """Write the table as a CSV file (its column the page's quantity for
        the energy)."""
        from linac_gen.ring.programs import write_table_csv
        times, values = self.points(f"{self.program} table")
        column = self.column_provider() if callable(self.column_provider) else self.column
        write_table_csv(path, times, values, column)


class RingProgramsPage(QWidget):
    """The Programs page (see the module docstring).  ``options_provider()``
    gives the element-model settings, ``beam_provider()`` the Beam tab's
    BeamConfig, ``turns_setter(n)`` sets the Run page's number of turns,
    ``turns_provider()`` reads it (the preview's turns for a flat program)."""

    changed = pyqtSignal()

    def __init__(self, state, *, options_provider, beam_provider, turns_setter,
                 turns_provider):
        super().__init__()
        self.state = state
        self._options_provider = options_provider
        self._beam_provider = beam_provider
        self._turns_setter = turns_setter
        self._turns_provider = turns_provider
        self._worker = None
        self._preview = None
        self._summary = ""
        self._loading = False
        self._build_ui()
        self.load(self.state.ring_config or {})

    # -- UI ----------------------------------------------------------------------
    def _build_ui(self) -> None:
        v = QVBoxLayout(self)
        note = QLabel(_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{theme.TEXT_2};")
        v.addWidget(note)

        box = QGroupBox("Reference energy")
        form = QFormLayout(box)
        self._e_kind = QComboBox()
        for label, key in _ENERGY_KINDS:
            self._e_kind.addItem(label, key)
        form.addRow("Program", self._e_kind)
        self._sine = QWidget()
        sf = QFormLayout(self._sine)
        sf.setContentsMargins(0, 0, 0, 0)
        self._e_start = QLineEdit()
        self._e_start.setPlaceholderText("the ring's reference energy")
        self._e_top = QLineEdit()
        self._e_freq = QLineEdit()
        self._e_tmin = QLineEdit("0")
        self._e_tstart = QLineEdit("0")
        sf.addRow("Start kinetic energy (MeV)", self._e_start)
        sf.addRow("Top kinetic energy (MeV)", self._e_top)
        sf.addRow("Cycle frequency (Hz)", self._e_freq)
        sf.addRow("Field minimum, t_min (ms)", self._e_tmin)
        sf.addRow("Ramp start, t_start (ms)", self._e_tstart)
        self._e_start.setToolTip("The momentum is held at this energy until t_start; the cycle "
                                 "p_min + (p_max - p_min)(1 - cos 2 pi f (t - t_min))/2 passes "
                                 "through it at t_start")
        form.addRow(self._sine)
        self._e_table_box = QWidget()
        tf = QFormLayout(self._e_table_box)
        tf.setContentsMargins(0, 0, 0, 0)
        self._e_quantity = QComboBox()
        for label, key in _QUANTITIES:
            self._e_quantity.addItem(label, key)
        tf.addRow("Values", self._e_quantity)
        self._e_table = TableEditor("energy", "value", "kinetic_mev",
                                    column_provider=self._energy_column)
        tf.addRow(self._e_table)
        form.addRow(self._e_table_box)
        v.addWidget(box)

        self._volt_box = QGroupBox("RF voltage (all cavities)")
        form = QFormLayout(self._volt_box)
        self._v_kind = QComboBox()
        for label, key in _VOLTAGE_KINDS:
            self._v_kind.addItem(label, key)
        form.addRow("Program", self._v_kind)
        self._v_table = TableEditor("voltage", "V (MV)", "voltage_mv")
        form.addRow(self._v_table)
        v.addWidget(self._volt_box)

        self._phase_box = QGroupBox("RF phase")
        form = QFormLayout(self._phase_box)
        self._p_kind = QComboBox()
        for label, key in _PHASE_KINDS:
            self._p_kind.addItem(label, key)
        self._p_kind.setToolTip("From the gain: phi_s = asin(gain / V), 180 deg - phi_s above "
                                "transition (sine convention: the reference gains V sin phi_s). "
                                "A table is used as it is — the reference follows the energy "
                                "program, the beam the table's phase")
        form.addRow("Program", self._p_kind)
        self._p_excess = QComboBox()
        for label, key in _ON_EXCESS:
            self._p_excess.addItem(label, key)
        form.addRow("Gain beyond the voltage", self._p_excess)
        self._p_table = TableEditor("phase", "phi_s (deg)", "phi_deg")
        form.addRow(self._p_table)
        v.addWidget(self._phase_box)

        self._other_box = QGroupBox("Transition, clock, correctors")
        form = QFormLayout(self._other_box)
        self._gamma_t = QLineEdit()
        self._gamma_t.setPlaceholderText("from the ring's optics")
        form.addRow("Transition gamma", self._gamma_t)
        self._t0 = QLineEdit("0")
        self._t0.setToolTip("The cycle's time at the first turn")
        form.addRow("Clock at turn 0 (ms)", self._t0)
        self._corr = QComboBox()
        for label, key in _CORRECTORS:
            self._corr.addItem(label, key)
        form.addRow("Steerers", self._corr)
        v.addWidget(self._other_box)

        self._para_box = QGroupBox("Paraphasing")
        pv = QVBoxLayout(self._para_box)
        pnote = QLabel("Groups of cavities whose phase is offset by an angle: names, patterns "
                       "(rf*) or numbers (0 = the first cavity after s = 0), separated by "
                       "commas. The phase rule uses the net voltage of all cavities.")
        pnote.setWordWrap(True)
        pnote.setStyleSheet(f"color:{theme.TEXT_2};")
        pv.addWidget(pnote)
        self._para = QTableWidget(0, 2)
        self._para.setHorizontalHeaderLabels(["cavities", "angle (deg)"])
        self._para.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._para.setMinimumHeight(90)
        pv.addWidget(self._para)
        prow = QHBoxLayout()
        add, rem = QPushButton("Add group"), QPushButton("Remove group")
        add.clicked.connect(self._add_group)
        rem.clicked.connect(self._remove_group)
        prow.addWidget(add)
        prow.addWidget(rem)
        prow.addStretch(1)
        pv.addLayout(prow)
        v.addWidget(self._para_box)

        row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setToolTip("The reference particle alone through the ramp")
        self._preview_btn.clicked.connect(self.preview)
        self._to_run_btn = QPushButton("Run to the end of the program")
        self._to_run_btn.setToolTip("Set the Run page's turns to the end of the energy program "
                                    "(after a preview)")
        self._to_run_btn.clicked.connect(self.turns_to_run)
        self._to_run_btn.setEnabled(False)
        row.addWidget(self._preview_btn)
        row.addWidget(self._to_run_btn)
        row.addStretch(1)
        v.addLayout(row)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._status)
        from ..panels.ring_plots import RampPlots
        self._plots = RampPlots()
        self._plots.setMinimumHeight(420)
        v.addWidget(self._plots, stretch=1)

        for w in (self._e_kind, self._e_quantity, self._v_kind, self._p_kind, self._p_excess,
                  self._corr):
            w.currentIndexChanged.connect(self._on_changed)
        for w in (self._e_start, self._e_top, self._e_freq, self._e_tmin, self._e_tstart,
                  self._gamma_t, self._t0):
            w.editingFinished.connect(self._on_changed)
        for t in (self._e_table, self._v_table, self._p_table):
            t.changed.connect(self._on_changed)
            t.failed.connect(self._on_table_failed)
        self._e_table.imported.connect(self._on_energy_imported)
        self._para.itemChanged.connect(self._on_para_item)
        self._sync_visibility()

    def _energy_column(self) -> str:
        return self._e_quantity.currentData()

    def _line_edits(self):
        return (self._e_start, self._e_top, self._e_freq, self._e_tmin, self._e_tstart,
                self._gamma_t, self._t0)

    def commit_pending(self) -> None:
        """Take what is still being typed into the project (before it is
        saved): a table cell whose editor is open, and a line edit — only one
        that was edited, so that a save rewrites nothing else."""
        if self._loading:
            return
        for table in (self._e_table.table, self._v_table.table, self._p_table.table,
                      self._para):
            commit_cell_editor(table)             # (its itemChanged takes it)
        if any(w.isModified() for w in self._line_edits()):
            self._on_changed()

    def _on_energy_imported(self, column: str) -> None:
        # (the file's column says what its values are)
        i = self._e_quantity.findData(column)
        if i >= 0:
            self._e_quantity.blockSignals(True)
            self._e_quantity.setCurrentIndex(i)
            self._e_quantity.blockSignals(False)

    def _on_table_failed(self, msg: str) -> None:
        self._status.setText(msg)

    def _add_group(self, *_a) -> None:
        n = self._para.rowCount()
        self._para.blockSignals(True)
        self._para.insertRow(n)
        self._para.setItem(n, 0, QTableWidgetItem(""))
        self._para.setItem(n, 1, QTableWidgetItem("0"))
        self._para.blockSignals(False)

    def _remove_group(self, *_a) -> None:
        rows = sorted({i.row() for i in self._para.selectedIndexes()}, reverse=True)
        if not rows and self._para.rowCount():
            rows = [self._para.rowCount() - 1]
        for r in rows:
            self._para.removeRow(r)
        self._on_changed()

    def _on_para_item(self, _item) -> None:
        self._on_changed()

    def _sync_visibility(self) -> None:
        kind = self._e_kind.currentData()
        on = kind is not None
        self._sine.setVisible(kind == "sine")
        self._e_table_box.setVisible(kind == "table")
        for w in (self._volt_box, self._phase_box, self._other_box, self._para_box,
                  self._preview_btn):
            w.setEnabled(on)
        self._v_table.setVisible(self._v_kind.currentData() == "table")
        self._p_table.setVisible(self._p_kind.currentData() == "table")
        self._p_excess.setEnabled(self._p_kind.currentData() == "auto")

    # -- the programs ------------------------------------------------------------
    def ramp_config(self) -> dict | None:
        """The page's ramp (``RampConfig.to_dict``), None for fixed energy; a
        ValueError says what is not valid."""
        from linac_gen.ring.programs import RampConfig
        kind = self._e_kind.currentData()
        if kind is None:
            return None
        if kind == "flat":
            energy = {"kind": "flat"}
        elif kind == "sine":
            energy = {"kind": "sine",
                      "start_kinetic_mev": _num(self._e_start.text(), "start kinetic energy"),
                      "end_kinetic_mev": _num(self._e_top.text(), "top kinetic energy",
                                              blank=...),
                      "frequency_hz": _num(self._e_freq.text(), "cycle frequency", blank=...),
                      "t_min_s": _ms_to_s(self._e_tmin.text(), "t_min", blank=0.0),
                      "t_start_s": _ms_to_s(self._e_tstart.text(), "t_start", blank=0.0)}
        else:
            t, val = self._e_table.points("energy table")
            energy = {"kind": "table", "times_s": t, "values": val,
                      "quantity": self._e_quantity.currentData()}
        if self._v_kind.currentData() == "table":
            t, val = self._v_table.points("voltage table")
            voltage = {"kind": "table", "times_s": t, "values_mv": val}
        else:
            voltage = {"kind": "constant"}
        if self._p_kind.currentData() == "table":
            t, val = self._p_table.points("phase table")
            phase = {"kind": "table", "times_s": t, "values_deg": val,
                     "on_excess": self._p_excess.currentData()}
        else:
            phase = {"kind": "auto", "on_excess": self._p_excess.currentData()}
        groups = []
        for r in range(self._para.rowCount()):
            cells = [(self._para.item(r, c).text() if self._para.item(r, c) else "").strip()
                     for c in (0, 1)]
            if not any(cells):
                continue
            # (numbers are places in the ring, anything else a name or pattern)
            toks = [x for x in re.split(r"[,;\s]+", cells[0]) if x]
            cav = [int(x) if re.fullmatch(r"[0-9]+", x) else x for x in toks]
            groups.append({"cavities": cav,
                           "angle_deg": _num(cells[1], f"paraphasing group {r + 1}, angle",
                                             blank=...)})
        gamma_t = _num(self._gamma_t.text(), "transition gamma")
        t0 = _ms_to_s(self._t0.text(), "clock at turn 0", blank=0.0)
        try:
            rc = RampConfig.from_dict({"energy": energy, "voltage": voltage, "phase": phase,
                                       "paraphasing": groups, "gamma_t": gamma_t,
                                       "t_start_s": t0})
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from None
        return rc.to_dict()

    def corrector_mode(self) -> str:
        return self._corr.currentData()

    def summary_text(self) -> str:
        """One line for the Run page: what the run's energy does."""
        try:
            d = self.ramp_config()
        except ValueError as exc:
            return f"Energy: the programs are not valid — {exc}"
        if d is None:
            return "Energy: fixed (no programs)."
        e = d["energy"]
        if e["kind"] == "sine":
            start = ("the ring's energy" if e["start_kinetic_mev"] is None
                     else f"{e['start_kinetic_mev']:.6g} MeV")
            return (f"Energy: the ramp of the Programs page — sine from {start} to "
                    f"{e['end_kinetic_mev']:.6g} MeV at {e['frequency_hz']:.6g} Hz.")
        if e["kind"] == "table":
            return (f"Energy: the ramp of the Programs page — a table of "
                    f"{len(e['times_s'])} points.")
        return "Energy: fixed, with the voltage and phase programs of the Programs page."

    def load(self, cfg: dict) -> None:
        """Show the ramp a ring configuration holds (none: fixed energy)."""
        from linac_gen.ring.programs import RampConfig
        self._loading = True
        try:
            d = cfg.get("ramp") if isinstance(cfg, dict) else None
            rc = None
            if isinstance(d, dict):
                with contextlib.suppress(TypeError, ValueError):
                    rc = RampConfig.from_dict(d)
            self._reset_fields()
            self._corr.setCurrentIndex(_index(_CORRECTORS, (cfg or {}).get("corrector_mode",
                                                                            "follow")))
            if rc is None:
                self._sync_visibility()
                return
            e, vp, ph = rc.energy, rc.voltage, rc.phase
            self._e_kind.setCurrentIndex(_index(_ENERGY_KINDS, e.kind))
            if e.kind == "sine":
                self._e_start.setText(_fmt(e.start_kinetic_mev))
                self._e_top.setText(_fmt(e.end_kinetic_mev))
                self._e_freq.setText(_fmt(e.frequency_hz))
                self._e_tmin.setText(_fmt_ms(e.t_min_s))
                self._e_tstart.setText(_fmt_ms(e.t_start_s))
            elif e.kind == "table":
                self._e_quantity.setCurrentIndex(_index(_QUANTITIES, e.quantity))
                self._e_table.set_points(e.times_s, e.values)
            self._v_kind.setCurrentIndex(_index(_VOLTAGE_KINDS, vp.kind))
            self._v_table.set_points(vp.times_s, vp.values_mv)
            self._p_kind.setCurrentIndex(_index(_PHASE_KINDS, ph.kind))
            self._p_excess.setCurrentIndex(_index(_ON_EXCESS, ph.on_excess))
            self._p_table.set_points(ph.times_s, ph.values_deg)
            self._gamma_t.setText(_fmt(rc.gamma_t))
            self._t0.setText(_fmt_ms(rc.t_start_s))
            self._para.blockSignals(True)
            self._para.setRowCount(0)
            for g in rc.paraphasing:
                n = self._para.rowCount()
                self._para.insertRow(n)
                self._para.setItem(n, 0, QTableWidgetItem(", ".join(str(c) for c in g.cavities)))
                self._para.setItem(n, 1, QTableWidgetItem(_fmt(g.angle_deg)))
            self._para.blockSignals(False)
            self._sync_visibility()
        finally:
            self._loading = False
            self.invalidate("")

    def _reset_fields(self) -> None:
        """Every field at its default (nothing kept from the programs shown
        before)."""
        for combo in (self._e_kind, self._e_quantity, self._v_kind, self._p_kind,
                      self._p_excess, self._corr):
            combo.setCurrentIndex(0)
        for edit, text in ((self._e_start, ""), (self._e_top, ""), (self._e_freq, ""),
                           (self._e_tmin, "0"), (self._e_tstart, "0"), (self._gamma_t, ""),
                           (self._t0, "0")):
            edit.setText(text)
        for table in (self._e_table, self._v_table, self._p_table):
            table.set_points((), ())
        self._para.blockSignals(True)
        self._para.setRowCount(0)
        self._para.blockSignals(False)

    def _on_changed(self, *_a) -> None:
        self._sync_visibility()
        if self._loading:
            return
        for w in self._line_edits():
            w.setModified(False)                  # (their text is taken below)
        self.invalidate("")
        cfg = dict(self.state.ring_config or {})
        cfg["corrector_mode"] = self.corrector_mode()
        try:
            d = self.ramp_config()
        except ValueError as exc:
            self._status.setText(f"Not valid: {exc} (the project keeps the last valid "
                                 "programs).")
        else:
            if d is None:
                cfg.pop("ramp", None)
            else:
                cfg["ramp"] = d
        if cfg != (self.state.ring_config or {}):
            self.state.ring_config = cfg
            self.state.mark_project_dirty()
        self.changed.emit()

    # -- preview -----------------------------------------------------------------
    def invalidate(self, msg: str) -> None:
        """A preview shown or being computed for other settings is dropped."""
        self._detach()
        self._preview = None
        self._to_run_btn.setEnabled(False)
        self._plots.clear("")
        self._status.setText(msg)

    def preview(self, *_a) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        lattice = self.state.lattice
        if lattice is None:
            self._status.setText("Open a lattice first.")
            return
        try:
            d = self.ramp_config()
        except ValueError as exc:
            self._status.setText(f"Not valid: {exc}")
            return
        if d is None:
            self._status.setText("Fixed energy: there is no ramp to preview.")
            return
        try:
            beam = self._beam_provider()
        except Exception as exc:          # noqa: BLE001 — an invalid Beam form
            self._status.setText(f"Beam settings: {exc}")
            return
        from linac_gen.ring.config import RingConfig
        try:
            config = RingConfig(ramp=d, corrector_mode=self.corrector_mode(),
                                **self._options_provider())
        except ValueError as exc:
            self._status.setText(str(exc))
            return
        self.invalidate("The reference through the ramp… (the first time in a session also "
                        "compiles the ring kernels)")
        from ..workers import RingRampWorker
        w = RingRampWorker(lattice, config, beam, int(self._turns_provider()))
        w.finished_ok.connect(self._on_preview_done)
        w.failed.connect(self._on_preview_failed)
        self._worker = w
        self._preview_btn.setEnabled(False)
        w.start()

    def _on_preview_done(self, r) -> None:
        if self.sender() is not self._worker:
            return
        # (the worker stays referenced — its thread may still be ending — until
        # the next preview or a change detaches it)
        self._preview_btn.setEnabled(True)
        self._preview = r
        o = r["orbit"]
        import numpy as np
        m = r["mass_mev"]
        n = int(r["turns"])
        k = o["transition_turn"]
        gt = "none" if r["gamma_t"] is None else f"{r['gamma_t']:.6f}"
        volts = o["voltage_mv"] * r["voltage_eff_factor"]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(volts > 0.0, np.abs(o["gain_mev"]) / volts, np.nan)
        ratio = (f"; largest gain / voltage {np.nanmax(x):.4g} (turn {int(np.nanargmax(x))})"
                 if np.isfinite(x).any() else "")
        capped = int(np.sum(o["capped"]))
        end = (f"{n} turns to the end of the energy program" if r["turns_to_end"] is not None
               else f"{n} turns (the Run page's; a flat program has no end)")
        tr = (f"transition (gamma_t {gt}) at turn {k}" if k is not None
              else f"above transition (gamma_t {gt}) from the start"
              if o.get("above_transition_at_start") else f"transition (gamma_t {gt}) not crossed")
        f = o["frequency_hz"]
        self._summary = (
            f"{end}: {o['energy_mev'][0] - m:.6g} -> {o['energy_end_mev'] - m:.6g} MeV, t "
            f"{o['t_s'][0] * 1e3:.6g} -> {o['t_end_s'] * 1e3:.6g} ms; {tr}; RF "
            f"{f.min() / 1e6:.7g} to {f.max() / 1e6:.7g} MHz{ratio}; "
            + (f"{capped} turn(s) capped at 90 deg." if capped else "none capped."))
        self._status.setText(self._summary)
        self._plots.show_ramp(np.arange(n), o["energy_mev"] - m,
                              o["phi_deg"] + r["phase_offset_deg"], o["voltage_mv"], f, k,
                              "The reference alone.")
        self._to_run_btn.setEnabled(r["turns_to_end"] is not None)

    def _on_preview_failed(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        self._preview_btn.setEnabled(True)
        self._status.setText(f"Preview failed — {msg}")

    def turns_to_run(self, *_a) -> None:
        r = self._preview
        if r is None or r.get("turns_to_end") is None:
            return
        n = int(r["turns_to_end"])
        done = self._turns_setter(n) is not False
        self._status.setText(self._summary + (f" The Run page now runs {n} turns." if done else
                                              " The Run page's turns cannot change while a "
                                              "run goes on."))

    def stop_preview(self, msg: str = "") -> None:
        """A preview being computed is dropped, and ``msg`` says so (a preview
        shown stays)."""
        if self._worker is not None and self._worker.isRunning():
            self._detach()
            self._status.setText(msg)

    def _detach(self) -> None:
        old, self._worker = self._worker, None
        if old is None:
            return
        for name in ("finished_ok", "failed"):
            with contextlib.suppress(TypeError, RuntimeError):
                getattr(old, name).disconnect()
        if old.isRunning():
            old.request_stop()
            from ..app import _park_worker  # (never drop a live QThread)
            _park_worker(old)
        self._preview_btn.setEnabled(self._e_kind.currentData() is not None)

    def shutdown_begin(self) -> list:
        w = self._worker
        if w is not None and w.isRunning():
            w.request_stop()
            return [w]
        return []
