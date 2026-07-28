"""Type-aware element property editor."""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QDoubleSpinBox,
    QScrollArea, QToolButton, QWidget, QSpinBox, QCheckBox, QComboBox,
)

from linac_gen_gui.interphase import theme
from linac_gen_gui.interphase.manual_help import open_for_element
from linac_gen_gui.interphase.state import AppState


# Fields skipped by the LatticeCommand introspector — names that come
# from ``Element`` / ``PassiveElement`` plumbing rather than the command
# itself (a SET_BEAM_ENERGY's "length" or "n_steps" is uninteresting and
# always 0).  The ``flags`` list on ``ADJUST_BEAM_*`` is handled
# specially because it's a list-of-int.
_LATTICE_COMMAND_HIDE = {
    "name", "length", "aperture", "n_steps",
    "KEYWORD", "snapshot", "is_bpm", "_s_entry",
}


# Per-attribute typed dropdowns — used when the underlying value is an int
# but we want the GUI to surface labelled choices instead of a raw spinbox.
# Maps (type_name, attr) → list[(int_value, "Display label")].
_ENUM_CHOICES: dict[tuple[str, str], list[tuple[int, str]]] = {
    ("Aperture", "aperture_type"): [
        (0, "Rectangular"),
        (1, "Circular"),
        (2, "Pepperpot"),
        (3, "Rect + beam-fraction"),
        (4, "Finger (horizontal)"),
        (5, "Finger (vertical)"),
        (6, "Ring"),
    ],
}


# Per-attribute string-valued dropdowns.  Each entry is a list of
# valid string values; the inspector renders them as a QComboBox.
_STR_CHOICES: dict[tuple[str, str], list[str]] = {
    ("VaneRFQ", "field_model"): [
        "2term",
        "8term",
        "8term_full",
        "laplace2d",
        "laplace3d",
    ],
}


def _parse_int_list(text: str) -> list[int]:
    """Permissive ``"1, 0, 2"`` → ``[1, 0, 2]``."""
    out: list[int] = []
    for token in text.replace(",", " ").split():
        try:
            out.append(int(float(token)))
        except ValueError:
            continue
    return out


def _parse_float_list(text: str) -> list[float]:
    """Permissive ``"0.1, 0, 2.5"`` → ``[0.1, 0.0, 2.5]``."""
    out: list[float] = []
    for token in text.replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def _introspect_command_schema(element) -> list[tuple[str, str, str]]:
    """Build a (attr, label, unit) list for a ``LatticeCommand`` from
    its public attributes (the constructor's keyword arguments)."""
    rows: list[tuple[str, str, str]] = []
    for attr, value in vars(element).items():
        if attr.startswith("_") or attr in _LATTICE_COMMAND_HIDE:
            continue
        if attr == "flags" and isinstance(value, list):
            # ``ADJUST_BEAM_*`` flag list — render as a single string row
            # so the user can edit the comma-separated tail without us
            # having to wire list-indexed editors.  Commit parses the
            # text back into ``list[int]`` via the standard string commit
            # path; ``__post_init__``-style coercion isn't run here, so
            # we keep a permissive parser in ``_commit_flags_text``.
            rows.append((attr, "Flags",
                         "list 0=skip, 1=adjust, 2=couple"))
            continue
        # Best-effort unit hint by attribute name.
        unit = ""
        n = attr.lower()
        if n.endswith("_mm") or "size" in n:        unit = "mm"
        elif n.endswith("_deg") or "phi" in n:      unit = "deg"
        elif n.endswith("_mev") or "energy" in n:   unit = "MeV"
        elif "alpha" in n:                          unit = ""
        elif "beta" in n:                           unit = "mm/mrad"
        elif "current" in n:                        unit = "mA"
        rows.append((attr, attr.replace("_", " "), unit))
    return rows


# Misalignment fields shared across most magnets / cavities — appended to
# their schema below.  Keep the order consistent so the inspector layout is
# stable as the user clicks between elements.
_MISALIGN = [
    ("dx",        "Offset Δx",      "mm"),
    ("dy",        "Offset Δy",      "mm"),
    # dz / pitch / yaw are stored and round-tripped but NOT applied by
    # any tracker (only dx, dy, tilt act on the beam) — say so in the
    # label instead of silently accepting dead input.
    ("dz",        "Offset Δz (not used in tracking)",      "mm"),
    ("tilt_deg",  "Tilt (about z)", "deg"),
    ("pitch_deg", "Pitch (about x) (not used in tracking)", "deg"),
    ("yaw_deg",   "Yaw (about y) (not used in tracking)",   "deg"),
]

_SCHEMA: dict[str, list[tuple[str, str, str]]] = {
    "Drift":       [("length", "Length", "mm"),
                    ("aperture", "Aperture", "mm")] + _MISALIGN,
    "Quadrupole":  [("length", "Length", "mm"),
                    ("gradient", "Gradient", "T/m"),
                    ("gradient_rel", "Gradient error (rel.)", ""),
                    ("skew_angle", "Skew angle", "deg"),
                    ("g3", "g₃ (sextupole)", "T/m²"),
                    ("g4", "g₄ (octupole)",  "T/m³"),
                    ("g5", "g₅ (decapole)",  "T/m⁴"),
                    ("g6", "g₆ (dodecapole)","T/m⁵"),
                    ("aperture", "Aperture", "mm")] + _MISALIGN,
    # Sextupole / Octupole are constructed via convenience factories that
    # build an underlying ``Multipole`` instance, so the inspector keys on
    # ``Multipole``.  ``knl`` / ``ksl`` are list-of-int and rendered as
    # comma-separated text by the list-editor branch in _add_field.
    "Multipole":   [("knl",      "knL (normal)", "MAD-X k_n L"),
                    ("ksl",      "ksL (skew)",   "MAD-X k_n L"),
                    ("dx",       "Offset Δx",    "mm"),
                    ("dy",       "Offset Δy",    "mm"),
                    ("tilt_deg", "Tilt",         "deg"),
                    ("aperture", "Aperture",     "mm")],
    "Dipole":      [("length", "Length", "mm"),
                    ("angle", "Bend angle", "rad"),
                    ("field_rel", "Field error (rel.)", ""),
                    ("n_index", "Field index n", ""),
                    ("e1", "Edge angle 1", "rad"),
                    ("e2", "Edge angle 2", "rad"),
                    ("aperture", "Aperture", "mm")] + _MISALIGN,
    "Solenoid":    [("length", "Length", "mm"),
                    ("Bz", "Bz", "T"),
                    ("field_rel", "Field error (rel.)", ""),
                    ("aperture", "Aperture", "mm")] + _MISALIGN,
    "RFGap":       [("amplitude", "Voltage amplitude", "MV"),
                    ("phase", "RF phase", "deg"),
                    ("frequency", "Frequency", "MHz"),
                    ("voltage_rel", "Voltage error (rel.)", ""),
                    ("phase_offset", "Phase offset", "deg"),
                    ("frequency_offset", "Frequency offset", "MHz")] + _MISALIGN,
    # FieldMap / FieldMap3D store three amplitude multipliers: ``scale``
    # (global, seldom touched by the TraceWin parser — defaults to 1.0),
    # ``ke`` (scales RF electric channels), and ``kb`` (scales magnetic
    # channels).  The TraceWin parser reads ke/kb from the FIELD_MAP card
    # and leaves scale at 1.0, so showing only ``scale`` here used to make
    # every field map look like "Amplitude = 1".
    "FieldMap":    [("length", "Length", "mm"),
                    ("ke",     "E-field amplitude (ke)", ""),
                    ("kb",     "B-field amplitude (kb)", ""),
                    ("scale",  "Global scale", ""),
                    ("phase",  "Phase", "deg"),
                    ("frequency", "Frequency", "MHz"),
                    ("voltage_rel", "Amplitude error (rel.)", ""),
                    ("phase_offset", "Phase offset", "deg"),
                    ("frequency_offset", "Frequency offset", "MHz"),
                    ("aperture", "Aperture", "mm"),
                    ("n_steps", "n_steps", "")] + _MISALIGN,
    "FieldMap3D":  [("length", "Length", "mm"),
                    ("ke",     "E-field amplitude (ke)", ""),
                    ("kb",     "B-field amplitude (kb)", ""),
                    ("scale",  "Global scale", ""),
                    ("phase",  "Phase", "deg"),
                    ("frequency", "Frequency", "MHz"),
                    ("voltage_rel", "Amplitude error (rel.)", ""),
                    ("phase_offset", "Phase offset", "deg"),
                    ("frequency_offset", "Frequency offset", "MHz"),
                    ("aperture", "Aperture", "mm"),
                    ("n_steps", "n_steps", "")] + _MISALIGN,
    "Aperture":    [("dx",            "Half-width / radius (dx)", "mm"),
                    ("dy",            "Half-height (dy)",          "mm"),
                    ("aperture_type", "Shape",                     "")],
    "BPM":         [("length", "Length", "mm")],
    "Marker":      [("snapshot", "Trigger snapshot", ""),
                    ("is_bpm",   "Use as BPM",       "")],
    "RfqCell":     [("voltage_V",   "Vane voltage",      "V"),
                    ("r0_mm",       "r₀",                "mm"),
                    ("A10",         "A₁₀",               ""),
                    ("modulation",  "Modulation m",      ""),
                    ("length",      "Length",            "mm"),
                    ("phi_s_deg",   "Synchronous phase", "deg"),
                    ("cell_type",   "Cell type",         ""),
                    ("Tc_mm",       "Tc",                "mm"),
                    ("dP_deg",      "ΔP",                "deg")],
    "Steerer":     [("bx_l", "B·L (x)", "T·m"),
                    ("by_l", "B·L (y)", "T·m")],
    "Foil":        [("material",          "Material",  ""),
                    ("thickness_ug_cm2",  "Thickness", "μg/cm²"),
                    ("aperture",          "Aperture",  "mm")],
    "VaneRFQ":     [("length",      "Length",       "mm"),
                    ("n_steps",      "n_steps",      ""),
                    ("field_model",  "Field model",  ""),
                    ("aperture",     "Aperture",     "mm")],
    "ThinLens":    [("fx", "fx",       "mm"),
                    ("fy", "fy",       "mm"),
                    ("aperture", "Aperture", "mm")],
    "SpaceChargeComp": [("factor", "Compensation factor", "")],
}


class ElementInspector(QFrame):
    def __init__(self, state: AppState):
        super().__init__()
        self.setObjectName("inspector")
        self._state = state
        # Base font size (pt) for the inline row styles; the app pushes
        # updates through apply_font_size so the inspector scales with
        # the global Ctrl+= / Ctrl+- font control.
        self._font_base = theme.FONT_SIZE
        # True while one of OUR editors is pushing a value through the
        # command bus.  bus.do → bus.changed → state.lattice_changed is a
        # synchronous chain, so without this guard every keystroke/spin
        # step would rebuild the editor under the user's cursor.
        self._in_commit = False

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        head = QFrame()
        head.setStyleSheet(f"background:{theme.BG_1}; border-bottom:1px solid {theme.BORDER_0};")
        h_lay = QHBoxLayout(head)
        h_lay.setContentsMargins(10, 8, 10, 8)
        h_lay.setSpacing(8)

        self._swatch = QLabel()
        self._swatch.setFixedSize(22, 22)
        self._swatch.setStyleSheet(f"background:{theme.BG_3}; border-radius:4px;")
        h_lay.addWidget(self._swatch)

        col = QVBoxLayout(); col.setSpacing(0)
        self._title = QLabel("Inspector")
        self._sub = QLabel("select an element")
        self._restyle_header()
        col.addWidget(self._title); col.addWidget(self._sub)
        h_lay.addLayout(col)
        h_lay.addStretch(1)

        # "?" help button — opens the manual chapter for the current
        # selection.  Disabled until an element is selected.
        self._help_btn = QToolButton()
        self._help_btn.setText("?")
        self._help_btn.setToolTip("Open manual for this element  (F1)")
        self._help_btn.setShortcut("F1")
        self._help_btn.setEnabled(False)
        self._help_btn.setStyleSheet(
            f"QToolButton {{ background:{theme.BG_INSET}; "
            f"border:1px solid {theme.BORDER_1}; border-radius:3px;"
            f"color:{theme.TEXT_1}; padding:1px 8px; font-weight:600; }}"
            f"QToolButton:hover {{ background:{theme.BG_3}; "
            f"color:{theme.TEXT_0}; }}"
            f"QToolButton:disabled {{ color:{theme.TEXT_3}; }}"
        )
        self._help_btn.clicked.connect(self._open_manual_for_selected)
        h_lay.addWidget(self._help_btn)
        v.addWidget(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(0, 0, 0, 0); self._body_lay.setSpacing(0)
        self._empty = QLabel("select an element from the outline")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(
            f"color:{theme.TEXT_3}; padding:40px 20px; font-size:{self._fs(-1)}px;"
        )
        self._body_lay.addWidget(self._empty)
        self._body_lay.addStretch(1)
        scroll.setWidget(self._body)
        v.addWidget(scroll, stretch=1)

        state.selected_element_changed.connect(self._on_selected)
        # Refresh displayed values when the lattice changes underneath the
        # selection (undo/redo, orbit correction, edits from other views).
        state.lattice_changed.connect(self._on_lattice_changed)

    # ------------------------------------------------------------------
    def _fs(self, delta: int = 0) -> int:
        """Row font size in px, relative to the app base."""
        return max(8, int(self._font_base) + delta)

    def _restyle_header(self) -> None:
        self._title.setStyleSheet(
            f"color:{theme.TEXT_0}; font-size:{self._fs(0)}px; "
            f"font-weight:500;"
        )
        self._sub.setStyleSheet(
            f"color:{theme.TEXT_2}; font-family:{theme.FONT_MONO}; "
            f"font-size:{self._fs(-2)}px;"
        )

    def apply_font_size(self, base: int) -> None:
        """Rescale the inline-styled rows (called by the app's global
        font control; the app-wide font doesn't reach these because
        they carry explicit inline font sizes)."""
        self._font_base = int(base)
        self._restyle_header()
        # Rebuild the body so every row/editor picks up the new size.
        self._on_selected(self._state.selected)

    # ------------------------------------------------------------------
    def _on_lattice_changed(self, lattice) -> None:
        if self._in_commit:
            # Our own commit — the editor already shows the typed value.
            return
        sel = self._state.selected
        if sel is None:
            return
        elements = getattr(lattice, "elements", None) if lattice is not None else None
        if not elements or not any(e is sel for e in elements):
            # Selection no longer exists (undo of an insert, delete from
            # another view) — clear it instead of rebuilding a ghost
            # editor whose commits would mutate an orphaned element.
            self._state.set_selected(None)
            return
        self._on_selected(sel)

    # ------------------------------------------------------------------
    def _on_selected(self, element) -> None:
        while self._body_lay.count():
            item = self._body_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                # A focused editor may emit editingFinished as it is torn
                # down, committing a stale value mid-rebuild — silence the
                # row and every child editor first.
                w.blockSignals(True)
                for child in w.findChildren(QWidget):
                    child.blockSignals(True)
                w.deleteLater()
        if element is None:
            self._title.setText("Inspector")
            self._sub.setText("select an element")
            self._swatch.setStyleSheet(f"background:{theme.BG_3}; border-radius:4px;")
            self._help_btn.setEnabled(False)
            # Rebuild the placeholder: the previous instance was scheduled
            # for deletion by the clear-loop above, so adding that dangling
            # reference here would raise "wrapped C/C++ object of type
            # QLabel has been deleted" on the next selection change.
            self._empty = QLabel("Select a lattice element to inspect")
            self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._empty.setStyleSheet(
                f"color:{theme.TEXT_3}; padding:40px 20px; font-size:{self._fs(-1)}px;"
            )
            self._body_lay.addWidget(self._empty)
            self._body_lay.addStretch(1)
            return
        type_name = type(element).__name__
        self._title.setText(getattr(element, "name", "?"))
        self._sub.setText(type_name)
        color = theme.EL_COLORS.get(type_name, theme.TEXT_2)
        self._swatch.setStyleSheet(f"background:{color}; border-radius:4px;")
        self._help_btn.setEnabled(True)
        self._help_btn.setToolTip(f"Open manual for {type_name}  (F1)")

        self._add_section("General")
        self._add_field(element, "name", "Name", "")
        self._add_readonly(type_name, "Type", "")
        schema = _SCHEMA.get(type_name, [])

        # ``LatticeCommand`` (SET_*/ADJUST_*) classes don't ship a
        # per-class schema — introspect their public attributes so every
        # command keyword gets an editable inspector for free.  We list
        # the keyword as a read-only header, then expose every
        # non-internal scalar attribute carried by the dataclass-style
        # constructor.  Mutations land via the normal ``_commit`` path,
        # which goes through the undo/redo bus.
        try:
            from linac_gen.elements.lattice_commands import LatticeCommand
            is_cmd = isinstance(element, LatticeCommand)
        except Exception:
            is_cmd = False
        if is_cmd and not schema:
            self._add_readonly(getattr(element, "KEYWORD", "?"),
                               "Keyword", "")
            schema = _introspect_command_schema(element)

        if schema:
            self._add_section("Properties")
            for attr, label, unit in schema:
                if not hasattr(element, attr):
                    continue
                self._add_field(element, attr, label, unit)
        if getattr(element, "is_bpm", False) and \
                getattr(element, "diag_family", None) is not None:
            # DIAG_POSITION diagnostic-matching data.  Read-only: targets
            # come from the deck card (or a runtime BPM-targets file) —
            # None means "plane unconstrained" and must never be coerced
            # into a spin-box 0.0.
            def _t(v):
                return "free" if v is None else f"{v:g}"
            self._add_section("Diagnostic matching")
            self._add_readonly(str(element.diag_family), "Family (N)", "")
            self._add_readonly(_t(element.x_target_mm), "x target", "mm")
            self._add_readonly(_t(element.y_target_mm), "y target", "mm")
            self._add_readonly(f"{element.accuracy_mm:g}",
                               "Accuracy (dm)", "mm")
            ov = getattr(element, "diag_target_override", None)
            if ov is not None:
                self._add_readonly(f"{_t(ov[0])} / {_t(ov[1])}",
                                   "File override x/y", "mm")
        if hasattr(element, "length") and not is_cmd:
            # Layout block irrelevant for zero-length commands.
            self._add_section("Layout")
            self._add_readonly(f"{element.length:g}", "Length", "mm")
        self._body_lay.addStretch(1)

    def _add_section(self, title: str) -> None:
        lbl = QLabel(title.upper())
        # Section header reads as "raised band" — accent stripe on the left,
        # subtle tinted background, brighter text.
        lbl.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:{self._fs(-1)}px; letter-spacing:1.5px; "
            f"font-weight:700; padding:8px 10px 6px 14px; "
            f"border-top:1px solid {theme.BORDER_0}; "
            f"border-left:3px solid {theme.ACCENT}; "
            f"background:{theme.SECTION_BG};"
        )
        self._body_lay.addWidget(lbl)

    def _label_col_width(self) -> int:
        """Label-column width scaled with the font — a hard 96px clips
        labels once the base size grows (rows rebuild on font change,
        so this recomputes with them)."""
        return int(96 * self._font_base / 12)

    def _add_field(self, element, attr: str, label: str, unit: str) -> None:
        row = QFrame()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(12, 4, 10, 4); lay.setSpacing(8)
        lbl = QLabel(label); lbl.setFixedWidth(self._label_col_width())
        lbl.setStyleSheet(
            f"color:{theme.TEXT_1}; font-size:{self._fs(-1)}px; font-weight:500;"
        )
        lay.addWidget(lbl)
        value = getattr(element, attr, "")
        type_name = type(element).__name__
        enum_key = (type_name, attr)
        if isinstance(value, bool):
            editor = QCheckBox()
            editor.setChecked(bool(value))
            editor.toggled.connect(
                lambda checked, e=element, a=attr: self._commit(e, a, bool(checked)))
            editor.setStyleSheet(
                f"QCheckBox {{ color:{theme.TEXT_0}; font-size:{self._fs(-1)}px; }}"
                f"QCheckBox::indicator {{ width:14px; height:14px; "
                f"border:1px solid {theme.BORDER_1}; "
                f"background:{theme.BG_INSET}; border-radius:3px; }}"
                f"QCheckBox::indicator:checked {{ "
                f"background:{theme.ACCENT}; "
                f"border:1px solid {theme.ACCENT}; }}"
            )
            lay.addWidget(editor, stretch=1)
            if unit:
                u = QLabel(unit)
                u.setStyleSheet(
                    f"color:{theme.TEXT_3}; font-family:{theme.FONT_MONO}; font-size:{self._fs(-2)}px;"
                    f"padding:0 6px; border-left:1px solid {theme.BORDER_1};"
                )
                lay.addWidget(u)
            self._body_lay.addWidget(row)
            return
        elif enum_key in _STR_CHOICES and isinstance(value, str):
            editor = QComboBox()
            for choice in _STR_CHOICES[enum_key]:
                editor.addItem(choice)
            for i in range(editor.count()):
                if editor.itemText(i) == value:
                    editor.setCurrentIndex(i); break
            editor.currentTextChanged.connect(
                lambda txt, e=element, a=attr: self._commit(e, a, str(txt))
            )
            editor.setStyleSheet(
                f"QComboBox {{ background:{theme.BG_INSET}; "
                f"border:1px solid {theme.BORDER_1}; border-radius:3px; "
                f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; "
                f"font-size:{self._fs(-1)}px; padding:3px 6px; }}"
            )
            lay.addWidget(editor, stretch=1)
            if unit:
                u = QLabel(unit)
                u.setStyleSheet(
                    f"color:{theme.TEXT_3}; font-family:{theme.FONT_MONO}; font-size:{self._fs(-2)}px;"
                    f"padding:0 6px; border-left:1px solid {theme.BORDER_1};"
                )
                lay.addWidget(u)
            self._body_lay.addWidget(row)
            return
        elif enum_key in _ENUM_CHOICES and isinstance(value, int):
            editor = QComboBox()
            for code, name in _ENUM_CHOICES[enum_key]:
                editor.addItem(f"{code} — {name}", userData=code)
            for i in range(editor.count()):
                if editor.itemData(i) == int(value):
                    editor.setCurrentIndex(i); break
            editor.currentIndexChanged.connect(
                lambda idx, ed=editor, e=element, a=attr:
                    self._commit(e, a, int(ed.itemData(idx)))
            )
            editor.setStyleSheet(
                f"QComboBox {{ background:{theme.BG_INSET}; "
                f"border:1px solid {theme.BORDER_1}; border-radius:3px; "
                f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; "
                f"font-size:{self._fs(-1)}px; padding:3px 6px; }}"
            )
            lay.addWidget(editor, stretch=1)
            if unit:
                u = QLabel(unit)
                u.setStyleSheet(
                    f"color:{theme.TEXT_3}; font-family:{theme.FONT_MONO}; font-size:{self._fs(-2)}px;"
                    f"padding:0 6px; border-left:1px solid {theme.BORDER_1};"
                )
                lay.addWidget(u)
            self._body_lay.addWidget(row)
            return
        elif isinstance(value, list):
            # Lists are rendered as comma-separated text.  Float lists
            # (Multipole.knl, Multipole.ksl) keep full precision; int
            # lists (ADJUST_BEAM_* flags) parse back into int via the
            # int parser.
            # Empty list defaults to FLOAT (matches the add-element dialog):
            # knl/ksl are float lists, and classifying [] as int silently
            # truncated a first entry of "0.5" to 0.
            is_float_list = (not value) or any(isinstance(v, float) for v in value)
            if is_float_list:
                editor = QLineEdit(", ".join(f"{float(v):g}" for v in value))
                editor.editingFinished.connect(
                    lambda e=element, a=attr, ed=editor:
                        self._commit(e, a, _parse_float_list(ed.text())))
            else:
                editor = QLineEdit(", ".join(str(int(v)) for v in value))
                editor.editingFinished.connect(
                    lambda e=element, a=attr, ed=editor:
                        self._commit(e, a, _parse_int_list(ed.text())))
        elif isinstance(value, int):
            editor = QSpinBox(); editor.setRange(-10**9, 10**9); editor.setValue(value)
            editor.valueChanged.connect(lambda v, e=element, a=attr: self._commit(e, a, v))
        elif isinstance(value, float):
            editor = QDoubleSpinBox(); editor.setRange(-1e12, 1e12)
            # Decimals adapt to magnitude so small physical quantities
            # (steerer kicks in T·m, edge angles, relative errors) aren't
            # silently truncated to 1e-6 on a user edit, while normal-scale
            # values keep a clean 6-decimal display.  Zero/unset fields get
            # extra precision since they're often about to hold a small kick.
            if value == 0.0 or not math.isfinite(value):
                _dec = 9
            else:
                _dec = min(12, max(6, 5 - int(math.floor(math.log10(abs(value))))))
            editor.setDecimals(_dec); editor.setValue(value)
            editor.valueChanged.connect(lambda v, e=element, a=attr: self._commit(e, a, float(v)))
        else:
            editor = QLineEdit(str(value) if value is not None else "")
            editor.editingFinished.connect(lambda e=element, a=attr, ed=editor: self._commit(e, a, ed.text()))
        editor.setStyleSheet(
            f"background:{theme.BG_INSET}; border:1px solid {theme.BORDER_1};"
            f"border-radius:3px; color:{theme.TEXT_0};"
            f"font-family:{theme.FONT_MONO}; font-size:{self._fs(-1)}px; padding:3px 6px;"
        )
        lay.addWidget(editor, stretch=1)
        if unit:
            u = QLabel(unit)
            u.setStyleSheet(
                f"color:{theme.TEXT_3}; font-family:{theme.FONT_MONO}; font-size:{self._fs(-2)}px;"
                f"padding:0 6px; border-left:1px solid {theme.BORDER_1};"
            )
            lay.addWidget(u)
        self._body_lay.addWidget(row)

    def _add_readonly(self, value: str, label: str, unit: str) -> None:
        row = QFrame()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(10, 3, 10, 3); lay.setSpacing(8)
        lbl = QLabel(label); lbl.setFixedWidth(self._label_col_width())
        lbl.setStyleSheet(f"color:{theme.TEXT_2}; font-size:{self._fs(-1)}px;")
        lay.addWidget(lbl)
        v = QLabel(value + (f"  {unit}" if unit else ""))
        v.setStyleSheet(
            f"color:{theme.TEXT_0}; font-family:{theme.FONT_MONO}; font-size:{self._fs(-1)}px;"
            f"background:{theme.BG_INSET}; padding:3px 6px; border-radius:3px;"
            f"border:1px solid {theme.BORDER_0};"
        )
        lay.addWidget(v, stretch=1)
        self._body_lay.addWidget(row)

    def _commit(self, element, attr: str, value) -> None:
        try:
            old = getattr(element, attr)
            if old == value:
                return
            from linac_gen_gui.interphase.commands import ParamChangeCommand
            self._in_commit = True
            try:
                self._state.bus.do(ParamChangeCommand(element, attr, old, value))
            finally:
                self._in_commit = False
            # bus.changed → state.lattice_changed (wired in state.py)
            self._state.status_message.emit(f"{element.name}.{attr} = {value}")
        except Exception as exc:
            self._state.status_message.emit(f"failed: {exc}")

    def _open_manual_for_selected(self) -> None:
        sel = self._state.selected
        if sel is None:
            return
        # Prefer the inline Qt popup (renders markdown source directly)
        # over the file:// browser opener (which renders the mkdocs-
        # built HTML and looks broken because Material theme assumes
        # HTTP).  Fall back to the browser only if the markdown source
        # isn't on disk -- e.g. PyInstaller bundles that ship site/
        # but not docs/.
        from linac_gen_gui.interphase.dialogs.manual_popup import open_inline
        ok, msg = open_inline(sel, parent=self)
        if not ok:
            ok, msg = open_for_element(sel)
        try:
            self._state.status_message.emit(msg)
        except Exception:
            pass
