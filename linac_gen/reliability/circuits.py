"""The circuit map: which elements fail together and where the report looks.

``circuits.json``::

    {"__kind__": "linac_gen_reliability_circuits", "__version__": 1,
     "groups": {"cryomodule": {"CM1": ["GAP_001", "GAP_002"]},
                "rf_station": {...}, "doublet": {...}, "arc_bus": {...},
                "spares": ["FMAP_161", ...]},
     "sections": [{"name": "LINAC", "end": "GAP_004"}, {"name": "TAIL", "end": "FOIL_001"}],
     "treaty_element": "BPM_003", "foil_element": "FOIL_001"}

Sections are contiguous element ranges ending at the named element (the
first starts at the first element).  :func:`build_circuit_map` derives a
minimal map from a lattice (spares = unpowered field maps, the first
foil, one section) when the user gives none.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CircuitMap", "build_circuit_map"]

_KIND = "linac_gen_reliability_circuits"
GROUP_KINDS = ("cryomodule", "rf_station", "doublet", "arc_bus")


@dataclass
class CircuitMap:
    groups: dict = field(default_factory=dict)
    sections: list = field(default_factory=list)
    treaty_element: str | None = None
    foil_element: str | None = None
    spares: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitMap":
        groups = dict(d.get("groups") or {})
        spares = list(groups.pop("spares", []) or [])
        for k in GROUP_KINDS:
            groups.setdefault(k, {})
        return cls(groups=groups, sections=list(d.get("sections") or []),
                   treaty_element=d.get("treaty_element"),
                   foil_element=d.get("foil_element"), spares=spares)

    def to_dict(self) -> dict:
        g = {k: dict(v) for k, v in self.groups.items()}
        g["spares"] = list(self.spares)
        return {"__kind__": _KIND, "__version__": 1, "groups": g,
                "sections": list(self.sections),
                "treaty_element": self.treaty_element, "foil_element": self.foil_element}

    @classmethod
    def load(cls, path) -> "CircuitMap":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("__kind__") not in (None, _KIND):
            raise ValueError(f"{path}: not a {_KIND} file")
        return cls.from_dict(d)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def members(self, kind: str) -> dict:
        return dict(self.groups.get(kind, {}))

    def section_ranges(self, lattice) -> list[tuple[str, float, float]]:
        """``[(name, s_start_m, s_end_m)]`` from the section end elements
        (the exit face of each end element)."""
        s = 0.0
        exit_s: dict[str, float] = {}
        for el in lattice.elements:
            s += float(getattr(el, "length", 0.0) or 0.0)
            nm = getattr(el, "name", None)
            if nm and nm not in exit_s:
                exit_s[nm] = s
        total = s
        out = []
        start = 0.0
        for sec in self.sections:
            end = exit_s.get(sec.get("end"))
            if end is None:
                raise ValueError(f"circuits: section {sec.get('name')!r} end element "
                                 f"{sec.get('end')!r} not in the lattice")
            out.append((str(sec.get("name")), start * 1e-3, end * 1e-3))
            start = end
        if not out:
            out.append(("LINE", 0.0, total * 1e-3))
        return out

    def check(self, lattice) -> list[str]:
        """Names referenced by the map that are not in the lattice."""
        names = {getattr(e, "name", None) for e in lattice.elements}
        missing = []
        for kind, groups in self.groups.items():
            for gname, members in groups.items():
                for m in members:
                    if m not in names:
                        missing.append(f"{kind}/{gname}/{m}")
        for m in self.spares:
            if m not in names:
                missing.append(f"spares/{m}")
        for sec in self.sections:
            if sec.get("end") not in names:
                missing.append(f"section/{sec.get('name')}/{sec.get('end')}")
        for key in ("treaty_element", "foil_element"):
            v = getattr(self, key)
            if v and v not in names:
                missing.append(f"{key}/{v}")
        return missing


def build_circuit_map(lattice) -> CircuitMap:
    """A minimal map derived from the lattice: spares (unpowered field
    maps), the first foil as ``foil_element``, one section ``LINE``."""
    from linac_gen.failures.element_filter import failable_elements
    spares = [n for (n, lbl, _c) in failable_elements(lattice, types={"spare"})]
    foil = next((getattr(e, "name") for e in lattice.elements
                 if type(e).__name__ == "Foil"), None)
    last = next((getattr(e, "name") for e in reversed(lattice.elements)
                 if getattr(e, "name", None)), None)
    return CircuitMap(groups={k: {} for k in GROUP_KINDS},
                      sections=[{"name": "LINE", "end": last}] if last else [],
                      treaty_element=None, foil_element=foil, spares=spares)
