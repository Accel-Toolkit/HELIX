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
  * Import an existing ``.dat`` — copied into the project folder by
    default (portable project), or referenced in place when unchecked.
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

        self._rb_import = QRadioButton("Import an existing lattice (.dat)")
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
        bl.addWidget(self._copy_in, 3, 1, 1, 2)

        self._rb_example = QRadioButton("Start from a bundled example")
        bl.addWidget(self._rb_example, 4, 0, 1, 3)
        self._example_combo = QComboBox()
        for label, _path in self._examples:
            self._example_combo.addItem(label)
        self._example_combo.setEnabled(False)
        bl.addWidget(self._example_combo, 5, 1, 1, 2)
        if not self._examples:
            self._rb_example.setEnabled(False)
            self._rb_example.setToolTip(
                "No bundled examples found in this installation.")

        for rb in (self._rb_blank, self._rb_import, self._rb_example):
            rb.toggled.connect(self._sync_enabled)
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
        self._example_combo.setEnabled(self._rb_example.isChecked())

    def _browse_location(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Project location", self._location.text().strip() or os.getcwd())
        if d:
            self._location.setText(d)

    def _browse_import(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Lattice to import", self._location.text().strip(),
            "TraceWin lattice (*.dat);;All Files (*)")
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
            if not src.lower().endswith(".dat"):
                QMessageBox.warning(self, "New Project",
                                    "The imported lattice must be a .dat file.")
                return

        project_dir = Path(location) / name
        if project_dir.exists() and (
                not project_dir.is_dir() or any(project_dir.iterdir())):
            QMessageBox.warning(
                self, "New Project",
                f"“{project_dir}” already exists and is not an empty "
                "directory.")
            return

        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            if self._rb_blank.isChecked():
                lattice_path = project_dir / f"{name}.dat"
                lattice_path.write_text(_STARTER_DECK.format(name=name),
                                        encoding="utf-8")
                mode = "blank"
            elif self._rb_import.isChecked():
                src = Path(self._import_path.text().strip())
                if self._copy_in.isChecked():
                    lattice_path = project_dir / src.name
                    shutil.copy2(src, lattice_path)
                else:
                    lattice_path = src
                mode = "import"
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
        }
        self.accept()

    def project_result(self) -> dict | None:
        """Ingredients of the accepted project, or None if cancelled.
        (Named so it cannot shadow ``QDialog.result()``.)"""
        return self._result
