"""Classify lattice elements for failure analysis and enumerate failable ones."""
from __future__ import annotations

from collections import Counter

from linac_gen.failures.failure_mode import FailureKind

ALL_TYPES = ("cavity", "quad", "solenoid", "dipole")

#: The reliability vocabulary, opt-in by naming one of them in ``types``:
#: ``"steerer"`` (a corrector; its only failure is OFF) and ``"spare"`` (an
#: unpowered field-map cavity, ``ke == kb == 0`` — never fails, but it can
#: be powered up as a compensator).  Naming either switches
#: :func:`classify` to its extended mode for the whole selection.
EXTRA_TYPES = ("steerer", "spare")

_EXTENDED_ONLY = {"Steerer": "steerer", "NCells": "cavity"}


def classify(elem, *, extended: bool = False) -> str | None:
    """Map an element to a failure-filter label, or ``None`` if it can't fail.

    FieldMap / FieldMap3D are routed by
    :func:`linac_gen.matching.variables.categorize_fieldmap` — ``ke != 0`` →
    ``"cavity"``, else ``"solenoid"``.

    ``extended`` (the reliability vocabulary, see :data:`EXTRA_TYPES`)
    additionally labels an unpowered field map (``ke == kb == 0``) as
    ``"spare"`` instead of ``"solenoid"``, a ``Steerer`` as ``"steerer"``
    and a multi-gap ``NCells`` cavity as ``"cavity"`` (it fails through
    its error slots but has no matcher knob, so it never compensates).
    The default mode is byte-for-byte the historical classification.
    """
    cls = type(elem).__name__
    if cls in ("FieldMap", "FieldMap3D"):
        if extended:
            ke = float(getattr(elem, "ke", 0.0) or 0.0)
            kb = float(getattr(elem, "kb", 0.0) or 0.0)
            if abs(ke) < 1e-12 and abs(kb) < 1e-12:
                return "spare"
        from linac_gen.matching.variables import categorize_fieldmap
        cat = categorize_fieldmap(elem)        # "cavity" | "solenoid" | "other"
        return cat if cat in ("cavity", "solenoid") else None
    label = {
        "RFGap": "cavity",
        "Quadrupole": "quad",
        "Solenoid": "solenoid",
        "Dipole": "dipole",
    }.get(cls)
    if label is None and extended:
        return _EXTENDED_ONLY.get(cls)
    return label


def valid_kinds(label: str) -> set[FailureKind]:
    """Which failure kinds make sense for a given element-type label."""
    if label == "cavity":
        return {FailureKind.OFF, FailureKind.DETUNE}
    if label in ("quad", "solenoid", "dipole"):
        return {FailureKind.OFF, FailureKind.PARTIAL}
    if label == "steerer":
        return {FailureKind.OFF}                 # no relative slot: absolute off
    return set()                                 # "spare": never fails


def _is_command(elem) -> bool:
    from linac_gen.elements.lattice_commands import LatticeCommand
    return isinstance(elem, LatticeCommand)


def failable_elements(lattice, types=None) -> list[tuple[str, str, str]]:
    """``[(name, type_label, class_name), …]`` for the elements that may fail.

    Returns named, non-command elements whose :func:`classify` label is in
    ``types`` (default: all of :data:`ALL_TYPES`).  Naming any of
    :data:`EXTRA_TYPES` classifies in the extended mode (so an unpowered
    field map is then a ``"spare"``, not a ``"solenoid"``).  **Duplicate-
    named** elements are dropped: the ``NAME.attr`` override selector is
    ambiguous for them (``cli.common.apply_element_override`` raises on
    duplicates), so they can't be safely targeted by name.
    """
    types = set(types) if types is not None else set(ALL_TYPES)
    extended = bool(types & set(EXTRA_TYPES))
    counts = Counter(
        getattr(e, "name", None)
        for e in lattice.elements
        if not _is_command(e) and getattr(e, "name", None)
    )
    out: list[tuple[str, str, str]] = []
    for e in lattice.elements:
        if _is_command(e):
            continue
        name = getattr(e, "name", None)
        if not name or counts[name] > 1:
            continue
        label = classify(e, extended=extended)
        if label is None or label not in types:
            continue
        out.append((name, label, type(e).__name__))
    return out
