"""Physicist-level semantic classification of lattice elements.

``type(e).__name__`` answers "which Python class"; an operator reading
the deck asks "which HARDWARE" — and for FIELD_MAP cards those are not
the same question: the identical class is an RF cavity or a static
solenoid depending on which field CHANNELS the 5-digit ``geom`` index
enables.  This module is the one authoritative place that composes the
already-audited primitives into that answer:

- ``e.field_data.channels`` (keys are :class:`~linac_gen.io.
  tracewin_geom.Channel`) / ``decode_geom(e.geom)`` say what the
  hardware IS (electric vs magnetic, static vs RF, 1-D/2-D/3-D);
- ``e.ke`` / ``e.kb`` say whether it is POWERED.

Both axes are required: ``ke != 0`` alone (the historical
``categorize_fieldmap`` rule) files a cavity parked at zero amplitude
as a solenoid — the MEBT-to-Foil deck carries 12 real HB650 cavities
exactly like that — and ``frequency > 0`` is useless because the parser
stamps the section frequency on static maps too.

Pure functions over public element attributes; no Qt, no I/O, no
solver imports — safe for the assistant layer, the GUI and the CLI.
"""
from __future__ import annotations

from typing import Any

#: name prefixes the parser synthesizes for corrector no-op cards (the
#: deck's XCOR/YCOR markers survive ONLY through their names)
_CORRECTOR_PREFIXES = ("XCOR", "YCOR")
#: diagnostic no-op card prefixes (see the keyword cheatsheet); BPMs are
#: additionally flagged via ``Marker.is_bpm``
_DIAG_PREFIXES = ("BPM", "ACCT", "DCCT", "RPU", "DPI", "RWCM", "ASCN",
                  "FFC", "LASERPROFILE", "FASTGV", "COL", "CHOPPER",
                  "MEBTABSORBER", "DIAG")

_DIM_BY_DIGIT = {1: "1D", 4: "2D", 5: "2D", 6: "2D", 7: "3D", 9: "1D"}


def _channel_facts(e: Any) -> dict | None:
    """RF/static/E/B channel facts for a field-map element, from the
    loaded ``field_data`` channels or (fallback) the raw ``geom`` code.
    None when the element is not a field map."""
    fd = getattr(e, "field_data", None)
    channels = getattr(fd, "channels", None)
    rf = stat_b = stat_e = gz = False
    dims: set[str] = set()
    found = False
    if channels:
        for ch, data in channels.items():
            found = True
            name = getattr(ch, "name", str(ch)).upper()
            g = getattr(data, "geometry", None)
            if g == 9:
                gz = True
            if name.startswith("RF"):
                rf = True
            elif name == "STAT_B":
                stat_b = True
            elif name == "STAT_E":
                stat_e = True
            if g in _DIM_BY_DIGIT:
                dims.add(_DIM_BY_DIGIT[g])
    if not found:
        geom = getattr(e, "geom", None)
        if geom is None:
            return None
        try:
            from linac_gen.io.tracewin_geom import decode_geom
            code = decode_geom(int(geom))
        except Exception:                                   # noqa: BLE001
            return None
        found = True
        rf = bool(code.rf_E or code.rf_B)
        stat_b = bool(code.stat_B) and code.stat_B != 9
        stat_e = bool(code.stat_E)
        gz = code.stat_B == 9
        for d in (code.stat_E, code.stat_B, code.rf_E, code.rf_B):
            if d in _DIM_BY_DIGIT:
                dims.add(_DIM_BY_DIGIT[d])
    if not found:
        return None
    order = {"3D": 3, "2D": 2, "1D": 1}
    dim = max(dims, key=lambda d: order[d]) if dims else "?"
    return {"rf": rf, "stat_b": stat_b, "stat_e": stat_e, "gz": gz,
            "dims": dim}


def classify_element(e: Any) -> dict:
    """``{"kind", "field_type", "dims", "powered"}`` for one element.

    ``kind`` ∈ cavity · solenoid · quad · dipole · corrector · rf_gap ·
    drift · aperture · diagnostic · foil · marker · command · other.
    ``field_type``/``dims``/``powered`` are non-None for field maps only
    (e.g. ``"RF E+B field map"``, ``"3D"``, ``True``).
    """
    cls = type(e).__name__
    # command cards (FREQ / SET_* / ADJUST_* …) declare a KEYWORD
    if getattr(e, "KEYWORD", None) is not None or cls in (
            "Freq", "SetSyncPhase"):
        return {"kind": "command", "field_type": None, "dims": None,
                "powered": None}
    facts = _channel_facts(e)
    if facts is not None:
        ke = float(getattr(e, "ke", 0.0) or 0.0)
        kb = float(getattr(e, "kb", 0.0) or 0.0)
        powered = (abs(ke) + abs(kb)) > 1e-12
        if facts["rf"]:
            kind = "cavity"
            ftype = "RF E+B field map" if facts["stat_b"] is False \
                else "RF field map + static B"
        elif facts["gz"]:
            kind = "quad"
            ftype = "static quad-gradient map G(z)"
        elif facts["stat_b"]:
            kind = "solenoid"
            ftype = "static B field map"
        elif facts["stat_e"]:
            kind = "other"
            ftype = "static E field map"
        else:
            kind = "other"
            ftype = "field map (no channels loaded)"
        return {"kind": kind, "field_type": ftype,
                "dims": facts["dims"], "powered": powered}
    if cls == "Quadrupole":
        return {"kind": "quad", "field_type": None, "dims": None,
                "powered": None}
    if cls == "Solenoid":
        return {"kind": "solenoid", "field_type": None, "dims": None,
                "powered": None}
    if cls in ("Dipole", "Edge"):
        return {"kind": "dipole" if cls == "Dipole" else "dipole_edge",
                "field_type": None, "dims": None, "powered": None}
    if cls in ("RFGap", "NCells", "RfqCell", "VaneRFQ"):
        return {"kind": "cavity", "field_type": None, "dims": None,
                "powered": None}
    if cls == "Steerer":
        return {"kind": "corrector", "field_type": None, "dims": None,
                "powered": None}
    if cls == "Drift":
        return {"kind": "drift", "field_type": None, "dims": None,
                "powered": None}
    if cls == "Aperture":
        return {"kind": "aperture", "field_type": None, "dims": None,
                "powered": None}
    if cls == "Foil":
        return {"kind": "foil", "field_type": None, "dims": None,
                "powered": None}
    if cls == "Marker":
        name = str(getattr(e, "name", "") or "").upper()
        if name.startswith(_CORRECTOR_PREFIXES):
            kind = "corrector"
        elif getattr(e, "is_bpm", False) or name.startswith(
                _DIAG_PREFIXES):
            kind = "diagnostic"
        else:
            kind = "marker"
        return {"kind": kind, "field_type": None, "dims": None,
                "powered": None}
    return {"kind": "other", "field_type": None, "dims": None,
            "powered": None}


def summarize_lattice(lattice: Any) -> dict:
    """Semantic rollup an operator would give: hardware counts (with
    cavities split powered/parked), lengths, RF sections."""
    kinds: dict[str, int] = {}
    length_by_kind: dict[str, float] = {}
    cav_live = cav_parked = 0
    bpm = 0
    rf_sections: list[float] = []
    total_mm = 0.0
    n_cmd = 0
    for e in lattice.elements:
        c = classify_element(e)
        k = c["kind"]
        kinds[k] = kinds.get(k, 0) + 1
        ln = float(getattr(e, "length", 0.0) or 0.0)
        total_mm += ln
        length_by_kind[k] = length_by_kind.get(k, 0.0) + ln
        if k == "command":
            n_cmd += 1
            if type(e).__name__ == "Freq":
                f = float(getattr(e, "frequency_mhz", 0.0)
                          or getattr(e, "frequency", 0.0) or 0.0)
                if f and f not in rf_sections:
                    rf_sections.append(f)
        if k == "cavity" and c["powered"] is not None:
            if c["powered"]:
                cav_live += 1
            else:
                cav_parked += 1
        if getattr(e, "is_bpm", False):
            bpm += 1
    return {
        "n_elements": len(lattice.elements),
        "n_command_cards": n_cmd,
        "length_m": total_mm / 1000.0,
        "kind_counts": dict(sorted(kinds.items(),
                                   key=lambda kv: -kv[1])),
        "cavities": kinds.get("cavity", 0),
        "cavities_powered": cav_live,
        "cavities_parked_zero_amplitude": cav_parked,
        "solenoids": kinds.get("solenoid", 0),
        "quads": kinds.get("quad", 0),
        "dipoles": kinds.get("dipole", 0),
        "correctors": kinds.get("corrector", 0),
        "bpms": bpm,
        "rf_sections_mhz": sorted(rf_sections),
        "length_by_kind_m": {k: round(v / 1000.0, 3)
                             for k, v in sorted(
                                 length_by_kind.items(),
                                 key=lambda kv: -kv[1])},
    }
