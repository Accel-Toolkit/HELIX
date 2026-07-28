"""Pure-function lattice validation.

Returns ``dict[id(elem), list[str]]`` of warning messages, advisory
only — never blocks a Save.  The LatticeTimeline uses these to
render a small triangle badge with the warnings as a tooltip.

Rules:
- Aperture ≤ 0:        "aperture not positive"
- Zero-length non-Marker: "zero length"
- Negative length:     "negative length"
- RFGap / FieldMap*:   no preceding ``frequency`` set on the element
                        AND no FREQ-bearing element earlier in the
                        lattice.  This catches a common error after a
                        FREQ header is dropped.
"""
from __future__ import annotations

from typing import Any


def validate(lattice) -> dict[int, list[str]]:
    if lattice is None:
        return {}
    out: dict[int, list[str]] = {}
    last_freq: float | None = None
    for el in getattr(lattice, "elements", []) or []:
        msgs: list[str] = []
        type_name = type(el).__name__
        # Aperture sanity (skip elements that don't carry one).
        ap = getattr(el, "aperture", None)
        if ap is not None and isinstance(ap, (int, float)) and ap <= 0:
            msgs.append(f"aperture {ap:g} ≤ 0")
        # Length sanity.
        L = getattr(el, "length", None)
        if isinstance(L, (int, float)):
            if L < 0:
                msgs.append(f"length {L:g} < 0")
            elif L == 0 and type_name not in (
                "Marker", "Aperture", "Steerer", "ThinLens", "SpaceChargeComp",
            ):
                msgs.append("zero length")
        # Track running frequency from any element that carries one.
        f = getattr(el, "frequency", None)
        if isinstance(f, (int, float)) and f > 0:
            last_freq = f
        # RF elements with no frequency available anywhere.
        if type_name in ("RFGap", "FieldMap", "FieldMap3D", "RfqCell", "VaneRFQ", "NCells"):
            own = getattr(el, "frequency", None)
            if not (isinstance(own, (int, float)) and own > 0) and last_freq is None:
                msgs.append("no preceding frequency")
        if msgs:
            out[id(el)] = msgs
    return out
