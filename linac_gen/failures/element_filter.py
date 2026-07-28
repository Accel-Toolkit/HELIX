"""Classify lattice elements for failure analysis and enumerate failable ones."""
from __future__ import annotations

from collections import Counter

from linac_gen.failures.failure_mode import FailureKind

ALL_TYPES = ("cavity", "quad", "solenoid", "dipole")


def classify(elem) -> str | None:
    """Map an element to a failure-filter label, or ``None`` if it can't fail.

    FieldMap / FieldMap3D are routed by
    :func:`linac_gen.matching.variables.categorize_fieldmap` — ``ke != 0`` →
    ``"cavity"``, else ``"solenoid"``.
    """
    cls = type(elem).__name__
    if cls in ("FieldMap", "FieldMap3D"):
        from linac_gen.matching.variables import categorize_fieldmap
        cat = categorize_fieldmap(elem)        # "cavity" | "solenoid" | "other"
        return cat if cat in ("cavity", "solenoid") else None
    return {
        "RFGap": "cavity",
        "Quadrupole": "quad",
        "Solenoid": "solenoid",
        "Dipole": "dipole",
    }.get(cls)


def valid_kinds(label: str) -> set[FailureKind]:
    """Which failure kinds make sense for a given element-type label."""
    if label == "cavity":
        return {FailureKind.OFF, FailureKind.DETUNE}
    if label in ("quad", "solenoid", "dipole"):
        return {FailureKind.OFF, FailureKind.PARTIAL}
    return set()


def _is_command(elem) -> bool:
    from linac_gen.elements.lattice_commands import LatticeCommand
    return isinstance(elem, LatticeCommand)


def failable_elements(lattice, types=None) -> list[tuple[str, str, str]]:
    """``[(name, type_label, class_name), …]`` for the elements that may fail.

    Returns named, non-command elements whose :func:`classify` label is in
    ``types`` (default: all of :data:`ALL_TYPES`).  **Duplicate-named**
    elements are dropped: the ``NAME.attr`` override selector is ambiguous for
    them (``cli.common.apply_element_override`` raises on duplicates), so they
    can't be safely targeted by name.
    """
    types = set(types) if types is not None else set(ALL_TYPES)
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
        label = classify(e)
        if label is None or label not in types:
            continue
        out.append((name, label, type(e).__name__))
    return out
