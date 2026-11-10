"""Add Element dialog — type combo + schema-driven form.

Returns a freshly-instantiated element via :meth:`AddElementDialog.element`
when accepted.  The dialog reuses :class:`ElementInspector._SCHEMA` for
the per-type field list so the editor and the dialog stay in lock-step.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QFrame, QLabel, QLineEdit, QSpinBox, QVBoxLayout, QMessageBox,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.element_factory import make_default, supported_types
from linac_gen_gui.interphase.panels.element_inspector import _SCHEMA


class AddElementDialog(QDialog):
    """Pick a type, edit its initial parameters, click OK."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add element")
        self.setMinimumWidth(380)
        self._element = None    # populated when user clicks OK

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14); v.setSpacing(10)

        # Header — type picker.
        v.addWidget(QLabel("Type"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(supported_types())
        v.addWidget(self._type_combo)

        # Name override (defaults to the factory's name).
        v.addWidget(QLabel("Name"))
        self._name = QLineEdit()
        v.addWidget(self._name)

        # Form host — rebuilt whenever the type combo changes.
        self._form_host = QFrame()
        self._form_host.setStyleSheet(
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_0};"
            f"border-radius:4px; padding:6px;"
        )
        self._form_lay = QFormLayout(self._form_host)
        self._form_lay.setContentsMargins(6, 6, 6, 6); self._form_lay.setSpacing(4)
        self._field_widgets: dict[str, object] = {}
        v.addWidget(self._form_host)

        # OK / Cancel.
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self._type_combo.currentTextChanged.connect(self._rebuild_form)
        self._rebuild_form(self._type_combo.currentText())

    # ------------------------------------------------------------------
    def _rebuild_form(self, type_name: str) -> None:
        # Clear existing rows.
        while self._form_lay.rowCount():
            self._form_lay.removeRow(0)
        self._field_widgets.clear()

        try:
            template = make_default(type_name)
        except Exception as exc:
            self._template = None
            self._form_lay.addRow(QLabel(f"<error: {exc}>"))
            return
        self._template = template

        if template is None:
            # Field maps need an external file — display guidance.
            self._form_lay.addRow(QLabel(
                f"<i>{type_name} requires an external field-map file. "
                "Use File → Open .field instead.</i>"))
            return

        # Default name preset.
        self._name.setText(getattr(template, "name", type_name))

        schema = _SCHEMA.get(type_name, [])
        derived = getattr(template, "_is_derived", lambda _a: False)
        inputs = getattr(type(template), "_GEOMETRY_ATTRS", ())
        for attr, label, unit in schema:
            value = getattr(template, attr, None)
            editor = self._make_editor(value)
            if derived(attr):
                # follows from the others (a bend's length, from its angle and radius)
                editor.setEnabled(False)
                editor.setToolTip("follows from the other parameters")
            label_text = label + (f"  [{unit}]" if unit else "")
            self._form_lay.addRow(label_text, editor)
            self._field_widgets[attr] = editor
            if attr in inputs and hasattr(editor, "valueChanged"):
                editor.valueChanged.connect(self._refresh_derived)

    def _refresh_derived(self, *_value) -> None:
        """After an edit of a parameter others follow from (a bend's angle or radius): what
        follows is shown, and a field can be edited while it does not follow (a bend with no
        angle takes a length).  Judged on a copy of the template with the form's values."""
        import contextlib
        import copy
        template = self._template
        if template is None or not hasattr(template, "_is_derived"):
            return
        probe = copy.deepcopy(template)
        for attr, ed in self._field_widgets.items():
            if not ed.isEnabled() or probe._is_derived(attr):
                continue
            with contextlib.suppress(Exception):   # an unfinished entry: judged again at OK
                setattr(probe, attr, self._read_editor(ed, getattr(probe, attr)))
        for attr, ed in self._field_widgets.items():
            follows = probe._is_derived(attr)
            if (follows or not ed.isEnabled()) and hasattr(ed, "setValue"):
                ed.blockSignals(True)              # shown, not entered
                try:
                    ed.setValue(getattr(probe, attr))
                finally:
                    ed.blockSignals(False)
            ed.setEnabled(not follows)
            ed.setToolTip("follows from the other parameters" if follows else "")

    def _make_editor(self, value):
        if isinstance(value, bool):
            ed = QLineEdit(str(value)); return ed
        if isinstance(value, int):
            ed = QSpinBox(); ed.setRange(-10**9, 10**9); ed.setValue(value); return ed
        if isinstance(value, float):
            ed = QDoubleSpinBox(); ed.setRange(-1e12, 1e12)
            ed.setDecimals(6); ed.setValue(value); return ed
        if isinstance(value, list):
            # Render lists (e.g. Multipole.knl / ksl) as comma-separated text,
            # matching the element inspector.  Without this branch the list
            # fell through to a QLineEdit whose text was later written back as
            # a raw string, producing an element that crashed tracking.
            is_float_list = (not value) or any(isinstance(v, float) for v in value)
            fmt = (lambda v: f"{float(v):g}") if is_float_list else (lambda v: str(int(v)))
            ed = QLineEdit(", ".join(fmt(v) for v in value))
            return ed
        ed = QLineEdit(str(value) if value is not None else "")
        return ed

    def _read_editor(self, ed, original):
        if isinstance(original, bool):
            return ed.text().strip().lower() in ("true", "1", "yes")
        if isinstance(original, int):
            return int(ed.value())
        if isinstance(original, float):
            return float(ed.value())
        if isinstance(original, list):
            # Parse the comma/space-separated text back into a numeric list
            # (float unless the original was a pure-int list).  A parse error
            # here is caught by _accept's try/except → "Invalid input".
            txt = ed.text().strip()
            if not txt:
                return []
            parts = [p for p in txt.replace(",", " ").split() if p]
            is_float_list = (not original) or any(isinstance(v, float) for v in original)
            return [float(p) for p in parts] if is_float_list \
                else [int(float(p)) for p in parts]
        return ed.text()

    # ------------------------------------------------------------------
    def _accept(self) -> None:
        if self._template is None:
            QMessageBox.warning(self, "Cannot add",
                                "This type cannot be added without an external file.")
            return
        try:
            # Apply name + form values onto the template.
            name = self._name.text().strip() or self._template.name
            self._template.name = name
            derived = getattr(self._template, "_is_derived", lambda _a: False)
            later = []
            for attr, ed in self._field_widgets.items():
                if derived(attr):          # follows from the others: set them first
                    later.append((attr, ed))
                    continue
                original = getattr(self._template, attr)
                setattr(self._template, attr, self._read_editor(ed, original))
            for attr, ed in later:         # e.g. the length a bend with no angle takes
                if not derived(attr) and ed.isEnabled():
                    setattr(self._template, attr,
                            self._read_editor(ed, getattr(self._template, attr)))
        except Exception as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return
        self._element = self._template
        self.accept()

    # ------------------------------------------------------------------
    def element(self):
        """The element instance to insert (or None if user cancelled)."""
        return self._element
