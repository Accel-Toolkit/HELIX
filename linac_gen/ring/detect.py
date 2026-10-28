"""Is this lattice a ring?

An explicit declaration always wins: the TraceWin-compatible deck comments
``;@LG ring`` (or ``ring=true``) and ``;@LG harmonic=84`` (read here — the first
consumer of ``lattice.lg_options``), or ``declared=True`` from the caller (a project with
a ``"ring"`` section, the ``ring`` CLI command, the API).  Otherwise the
geometry is inspected: a lattice whose horizontal bends add up to 360° AND
whose survey closes on itself is reported as ring-like, with the evidence.
Detection never switches HELIX into ring mode on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_TRUE = (True, 1, "1", "true", "yes", "on")


@dataclass
class RingVerdict:
    is_ring: bool
    declared: bool
    reason: str
    total_angle_deg: float = 0.0
    vertical_angle_deg: float = 0.0
    closure_mm: float = float("nan")
    heading_closure_deg: float = float("nan")
    length_m: float = 0.0
    harmonic: int | None = None
    evidence: dict = field(default_factory=dict)


def _lg_flag(value) -> bool:
    return value in _TRUE or (isinstance(value, str) and value.strip().lower() in _TRUE)


def survey_2d(lattice):
    """Horizontal survey of the design orbit: (X, Z) in mm and the heading in
    radians after each element, ignoring misalignments.  Vertical bends do not
    move the horizontal projection (their net angle is reported separately)."""
    from linac_gen.elements.dipole import Dipole

    X = Z = 0.0
    th = 0.0
    vert = 0.0
    for e in lattice.elements:
        L = float(getattr(e, "length", 0.0) or 0.0)
        if isinstance(e, Dipole) and getattr(e, "angle", 0.0):
            a = math.radians(float(e.angle))
            if int(getattr(e, "hv", 0)) == 1:
                vert += a
                X += L * math.sin(th)          # straight in the horizontal projection
                Z += L * math.cos(th)
                continue
            chord = 2.0 * abs(float(e.rho)) * math.sin(abs(a) / 2.0)
            mid = th + a / 2.0
            X += chord * math.sin(mid)
            Z += chord * math.cos(mid)
            th += a
        elif L and getattr(e, "is_curved", False):
            # curved field-map cluster: on from its exit frame, drawn like a
            # Dipole (survey_geometry)
            from linac_gen.elements.curved_field_map import survey_geometry
            _L, P, A = survey_geometry(e)
            X += P[2] * math.sin(th) + P[0] * math.cos(th)
            Z += P[2] * math.cos(th) - P[0] * math.sin(th)
            th += math.atan2(A[0, 2], A[2, 2])
            vert += math.asin(max(-1.0, min(1.0, float(A[1, 2]))))
        elif L:
            X += L * math.sin(th)
            Z += L * math.cos(th)
    return X, Z, th, vert


def detect_ring(lattice, meta: dict | None = None, *, declared: bool = False,
                angle_tol_deg: float = 1e-6, closure_rel: float = 1e-5,
                closure_min_mm: float = 1.0) -> RingVerdict:
    from linac_gen.elements.rf_gap import RFGap

    lg = dict(getattr(lattice, "lg_options", {}) or {})
    if meta and isinstance(meta.get("lg_options"), dict):
        lg.update(meta["lg_options"])
    decl = bool(declared) or _lg_flag(lg.get("ring"))

    harmonic = lg.get("harmonic")
    if harmonic is None:
        for e in lattice.elements:
            h = getattr(e, "harmonic", None) if isinstance(e, RFGap) else None
            if h:
                harmonic = h
                break
    try:
        # (identity for the bare-flag True: ``1 in (True,)`` would drop h = 1)
        harmonic = None if harmonic is None or harmonic is True or harmonic == "" \
            else int(harmonic)
    except (TypeError, ValueError):
        harmonic = None               # (not a number: the evidence keeps what was written)

    X, Z, th, vert = survey_2d(lattice)
    length_mm = sum(float(getattr(e, "length", 0.0) or 0.0) for e in lattice.elements)
    total_deg = math.degrees(th)
    closure = math.hypot(X, Z)
    head = math.degrees(th) - 360.0 * round(math.degrees(th) / 360.0)
    tol_mm = max(closure_min_mm, closure_rel * length_mm)
    turns_ok = abs(abs(total_deg) - 360.0) <= angle_tol_deg
    closed = closure <= tol_mm
    evidence = {"lg_options": {k: lg[k] for k in ("ring", "harmonic") if k in lg},
                "closure_tolerance_mm": tol_mm}

    if decl:
        reason = "declared ring (;@LG ring or a ring project/command)"
        is_ring = True
    elif turns_ok and closed:
        reason = (f"closes on itself: bends add up to {total_deg:.6f} deg and the survey "
                  f"closes within {closure:.3g} mm")
        is_ring = True
    elif turns_ok:
        reason = (f"bends add up to {total_deg:.6f} deg but the survey misses the start by "
                  f"{closure:.3g} mm — not closed (an open spiral or an offset)")
        is_ring = False
    else:
        reason = f"a line: horizontal bends add up to {total_deg:.6f} deg"
        is_ring = False
    return RingVerdict(is_ring=is_ring, declared=decl, reason=reason, total_angle_deg=total_deg,
                       vertical_angle_deg=math.degrees(vert), closure_mm=closure,
                       heading_closure_deg=head, length_m=length_mm / 1000.0,
                       harmonic=harmonic, evidence=evidence)
