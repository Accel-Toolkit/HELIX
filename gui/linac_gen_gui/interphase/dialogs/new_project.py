"""New Project wizard — name + location + starting point.

Creates ``<location>/<name>/`` with the project's lattice inside and
returns the ingredients via :meth:`NewProjectDialog.result`; the window
then assembles and writes the ``.lgproj`` (see ``app._new_project``).
The dialog never imports the app module: the start directory and the
bundled-example list are injected by the caller, which keeps imports
acyclic and makes the dialog trivially testable.

Starting points:
  * Blank lattice — writes a minimal one-drift TraceWin deck (mm units)
    so the fresh project is immediately loadable and runnable.
  * Import an existing lattice — a ``.dat`` is copied into the project
    folder by default (portable project) or referenced in place when
    unchecked; a MAD-X / MAD8 / Elegant / Bmad / SciBmad / PALS deck is
    parsed and materialised as ``<name>.dat`` inside the project (projects
    stay TraceWin-native), its import warnings handed back in the result.
    A TraceWin ``.dat`` with a ``<deck>.ini`` next to it offers to import
    that options file's beam too (checkbox, off by default): the ``.ini``
    travels with a copied deck and its path is handed back as
    ``tracewin_ini`` for ``app._new_project`` to apply before the
    ``.lgproj`` is written.
  * Bundled example — one of the small self-contained example decks,
    always copied so edits never touch ``examples/``.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QMessageBox, QPushButton,
    QRadioButton, QVBoxLayout,
)

from linac_gen.io.formats import (
    DIALOG_FILTER, IMPORT_SUFFIXES, import_format, parse_lattice_file,
)
from linac_gen.io.tracewin_ini import sibling_ini
from linac_gen_gui.interphase import theme

# Windows refuses these as file/directory names (with or without an
# extension); creating them over SMB or on the native build fails in
# confusing ways, so the wizard rejects them everywhere.
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}
_NAME_OK = re.compile(r"^[A-Za-z0-9._ -]+$")

_STARTER_DECK = """; {name} — created by HELIX New Project
TITLE {name}
FREQ 162.5
DRIFT 100 15
END
"""


def _name_error(name: str) -> str | None:
    """Return a human-readable rejection, or None when ``name`` is safe."""
    if not name:
        return "Enter a project name."
    if not _NAME_OK.match(name):
        return ("Project names may contain only letters, digits, spaces, "
                "dots, hyphens and underscores (ASCII).")
    if name[-1] in ". " or name[0] in ". ":
        return "Project names may not start or end with a dot or space."
    if name.split(".")[0].upper() in _RESERVED:
        return f"“{name}” is a reserved device name on Windows."
    return None


class NewProjectDialog(QDialog):
    """Guided project creation; result() returns the ingredients."""

    def __init__(self, parent=None, *, start_dir: str,
                 examples: list[tuple[str, str]] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(460)
        self._result: dict | None = None
        self._examples = list(examples or [])

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        g = QGridLayout()
        g.setHorizontalSpacing(8)
        g.addWidget(QLabel("Name"), 0, 0)
        self._name = QLineEdit()
        self._name.setPlaceholderText("my_linac")
        g.addWidget(self._name, 0, 1, 1, 2)
        g.addWidget(QLabel("Location"), 1, 0)
        self._location = QLineEdit(start_dir)
        g.addWidget(self._location, 1, 1)
        browse_loc = QPushButton("Browse…")
        browse_loc.clicked.connect(self._browse_location)
        g.addWidget(browse_loc, 1, 2)
        v.addLayout(g)

        hint = QLabel("A folder “<name>” is created inside the location; "
                      "the lattice and the runs/ output directory live "
                      "there, so the whole project can be moved or "
                      "version-controlled as one unit.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_1};")
        v.addWidget(hint)

        box = QGroupBox("Starting point")
        bl = QGridLayout(box)
        bl.setVerticalSpacing(6)

        self._rb_blank = QRadioButton("Blank lattice (one drift, ready to edit)")
        self._rb_blank.setChecked(True)
        bl.addWidget(self._rb_blank, 0, 0, 1, 3)

        self._rb_import = QRadioButton(
            "Import an existing lattice (.dat, MAD-X, MAD8, Elegant, Bmad, SciBmad, PALS)")
        bl.addWidget(self._rb_import, 1, 0, 1, 3)
        self._import_path = QLineEdit()
        self._import_path.setEnabled(False)
        bl.addWidget(self._import_path, 2, 1)
        self._import_browse = QPushButton("Browse…")
        self._import_browse.setEnabled(False)
        self._import_browse.clicked.connect(self._browse_import)
        bl.addWidget(self._import_browse, 2, 2)
        self._copy_in = QCheckBox("Copy the lattice into the project folder")
        self._copy_in.setChecked(True)
        self._copy_in.setEnabled(False)
        self._copy_in.setToolTip(
            "A .dat is copied as-is (or referenced in place when unchecked).\n"
            "Any other format is always converted to <name>.dat inside the "
            "project — the source file is never written.")
        bl.addWidget(self._copy_in, 3, 1, 1, 2)
        self._ini_in = QCheckBox(
            "Also import the beam from the TraceWin .ini next to the deck")
        self._ini_in.setChecked(False)
        self._ini_in.setEnabled(False)
        self._ini_in.setToolTip(
            "Enabled when the imported TraceWin .dat has a <deck>.ini next "
            "to it (TraceWin's project options file).  Its input beam — "
            "particle, energy, frequency, current, emittances, Twiss — then "
            "replaces the current Beam-tab settings before the project is "
            "written; the .ini is copied alongside a copied deck.")
        bl.addWidget(self._ini_in, 4, 1, 1, 2)

        self._rb_example = QRadioButton("Start from a bundled example")
        bl.addWidget(self._rb_example, 5, 0, 1, 3)
        self._example_combo = QComboBox()
        for label, _path in self._examples:
            self._example_combo.addItem(label)
        self._example_combo.setEnabled(False)
        bl.addWidget(self._example_combo, 6, 1, 1, 2)
        if not self._examples:
            self._rb_example.setEnabled(False)
            self._rb_example.setToolTip(
                "No bundled examples found in this installation.")

        for rb in (self._rb_blank, self._rb_import, self._rb_example):
            rb.toggled.connect(self._sync_enabled)
        self._import_path.textChanged.connect(self._sync_enabled)
        v.addWidget(box)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    # -- helpers -------------------------------------------------------
    def _sync_enabled(self) -> None:
        imp = self._rb_import.isChecked()
        self._import_path.setEnabled(imp)
        self._import_browse.setEnabled(imp)
        self._copy_in.setEnabled(imp)
        self._ini_in.setEnabled(imp and self._sibling_ini() is not None)
        self._example_combo.setEnabled(self._rb_example.isChecked())

    def _sibling_ini(self) -> Path | None:
        """``<deck>.ini`` next to the import path when that path is a
        TraceWin ``.dat`` (only TraceWin projects carry one)."""
        src = self._import_path.text().strip()
        if not src or not os.path.isfile(src) or import_format(src) != "tracewin":
            return None
        return sibling_ini(src)

    def _browse_location(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Project location", self._location.text().strip() or os.getcwd())
        if d:
            self._location.setText(d)

    def _browse_import(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Lattice to import", self._location.text().strip(),
            DIALOG_FILTER)
        if fp:
            self._import_path.setText(fp)

    # -- accept --------------------------------------------------------
    def _accept(self) -> None:
        name = self._name.text().strip()
        err = _name_error(name)
        if err:
            QMessageBox.warning(self, "New Project", err)
            return
        location = self._location.text().strip()
        if not location or not os.path.isdir(location):
            QMessageBox.warning(self, "New Project",
                                "Choose an existing location directory.")
            return
        if not os.access(location, os.W_OK):
            QMessageBox.warning(self, "New Project",
                                "The location directory is not writable.")
            return
        if self._rb_import.isChecked():
            src = self._import_path.text().strip()
            if not src or not os.path.isfile(src):
                QMessageBox.warning(self, "New Project",
                                    "Choose the lattice file to import.")
                return
            if import_format(src) is None:
                QMessageBox.warning(
                    self, "New Project",
                    "The imported lattice must be one of: "
                    + " ".join(IMPORT_SUFFIXES) + ".")
                return

        project_dir = Path(location) / name
        if project_dir.exists() and (
                not project_dir.is_dir() or any(project_dir.iterdir())):
            QMessageBox.warning(
                self, "New Project",
                f"“{project_dir}” already exists and is not an empty "
                "directory.")
            return

        import_warnings: list[str] = []
        tracewin_ini: Path | None = None
        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            if self._rb_blank.isChecked():
                lattice_path = project_dir / f"{name}.dat"
                lattice_path.write_text(_STARTER_DECK.format(name=name),
                                        encoding="utf-8")
                mode = "blank"
            elif self._rb_import.isChecked():
                src = Path(self._import_path.text().strip())
                if import_format(src) != "tracewin":
                    # Foreign deck: parse through the shared dispatcher and
                    # materialise it as a TraceWin .dat inside the project
                    # (never in place — the source is not HELIX's to write).
                    lattice_path = project_dir / f"{name}.dat"
                    import_warnings = self._materialise(src, lattice_path)
                    if import_warnings is None:
                        # error already shown; leave no half-made project
                        if project_dir.is_dir() and not any(project_dir.iterdir()):
                            project_dir.rmdir()
                        return
                elif self._copy_in.isChecked():
                    lattice_path = project_dir / src.name
                    shutil.copy2(src, lattice_path)
                else:
                    lattice_path = src
                mode = "import"
                if self._ini_in.isEnabled() and self._ini_in.isChecked():
                    ini_src = self._sibling_ini()
                    if ini_src is not None:
                        if lattice_path.parent == project_dir:
                            tracewin_ini = project_dir / ini_src.name
                            try:
                                shutil.copy2(ini_src, tracewin_ini)
                            except OSError as exc:
                                # The project is still created; only the
                                # beam import is dropped, and said so.
                                tracewin_ini = None
                                QMessageBox.warning(
                                    self, "New Project",
                                    f"{ini_src.name} could not be copied "
                                    f"({exc}); the project is created "
                                    f"without its TraceWin beam — use the "
                                    f"Beam tab's Import TraceWin .ini… "
                                    f"later.")
                        else:
                            tracewin_ini = ini_src
            else:
                _label, src = self._examples[self._example_combo.currentIndex()]
                src = Path(src)
                lattice_path = project_dir / src.name
                shutil.copy2(src, lattice_path)
                mode = "example"
        except OSError as exc:
            QMessageBox.critical(self, "New Project",
                                 f"Could not create the project:\n{exc}")
            return

        self._result = {
            "name": name,
            "project_dir": str(project_dir),
            "lattice_path": str(lattice_path),
            "mode": mode,
            "import_warnings": list(import_warnings),
            "tracewin_ini": str(tracewin_ini) if tracewin_ini else None,
        }
        self.accept()

    def _materialise(self, src: Path, dst: Path) -> list[str] | None:
        """Parse a non-TraceWin *src* and write it as TraceWin *dst*.
        Returns the import warnings, or None after showing the error
        (a half-written *dst* is removed so the folder cleanup can run)."""
        from linac_gen.io.tracewin_writer import write_tracewin
        try:
            session = getattr(self.parent(), "state", None)
            lattice, meta = parse_lattice_file(
                str(src), fallback_beam=getattr(session, "beam_config", None))
            write_tracewin(lattice, str(dst))
        except Exception as exc:
            try:
                if dst.is_file():
                    dst.unlink()
            except OSError:
                pass
            QMessageBox.critical(self, "New Project",
                                 f"Could not import {src.name}:\n{exc}")
            return None
        warns = meta.get("warnings", []) if isinstance(meta, dict) else []
        return [str(w) for w in warns]

    def project_result(self) -> dict | None:
        """Ingredients of the accepted project, or None if cancelled.
        (Named so it cannot shadow ``QDialog.result()``.)"""
        return self._result
