"""Scenario classes S1–S10 → ``FaultCase`` list on the failure engine's
``FailureScenario`` (per-member modes already supported)."""
from __future__ import annotations

from dataclasses import dataclass

from linac_gen.failures.element_filter import ALL_TYPES, EXTRA_TYPES, failable_elements
from linac_gen.failures.failure_mode import FailureKind, FailureMode
from linac_gen.failures.scenario import FailureScenario, enumerate_scenarios

__all__ = ["FaultCase", "enumerate_cases", "name_to_class_map"]

_MAGNETS = ("quad", "solenoid", "dipole")


@dataclass(frozen=True)
class FaultCase:
    case_id: str
    cls: str
    scenario: FailureScenario
    bracket: str = "rephased"
    label: str = ""

    @property
    def element_names(self) -> tuple:
        return self.scenario.element_names


def name_to_class_map(lattice) -> dict:
    """``{name: class_name}`` over the extended vocabulary (spares, steerers,
    multi-gap cavities included)."""
    return {n: c for (n, _l, c) in failable_elements(
        lattice, types=tuple(ALL_TYPES) + tuple(EXTRA_TYPES))}


def _label_of(lattice) -> dict:
    return {n: lbl for (n, lbl, _c) in failable_elements(
        lattice, types=tuple(ALL_TYPES) + tuple(EXTRA_TYPES))}


def _case(cls: str, tag: str, failures, bracket="rephased") -> FaultCase:
    scen = FailureScenario(failures=tuple(failures),
                           label=" + ".join(f"{n}:{m.label}" for n, m in failures))
    return FaultCase(case_id=f"{cls}_{tag}_{bracket[:2]}", cls=cls, scenario=scen,
                     bracket=bracket, label=scen.label)


def enumerate_cases(lattice, spec, circuits) -> tuple[list[FaultCase], dict]:
    """The fault cases of the enabled classes, in a deterministic order,
    plus the extended name → class map the overrides need."""
    n2c = name_to_class_map(lattice)
    labels = _label_of(lattice)
    opts = spec.class_options or {}
    cases: list[FaultCase] = []
    off = FailureMode(FailureKind.OFF)
    classes = set(spec.classes or [])

    def singles(types, kind=FailureKind.OFF, **kw):
        scen, _m, _names = enumerate_scenarios(lattice, types=types, kind=kind, **kw)
        return scen

    if "S1" in classes or "S10" in classes:
        s1 = singles({"cavity"})
        if "S1" in classes:
            cases += [_case("S1", s.element_names[0], s.failures) for s in s1]
        if "S10" in classes:
            cases += [_case("S10", s.element_names[0], s.failures, "frozen") for s in s1]
    if "S2" in classes:
        cases += [_case("S2", s.element_names[0], s.failures) for s in singles(set(_MAGNETS))]
    if "S3" in classes:
        o = opts.get("S3", {})
        cases += [_case("S3", s.element_names[0], s.failures) for s in singles(
            {"cavity"}, kind=FailureKind.DETUNE, amp_scale=float(o.get("amp_scale", 0.9)),
            phase_deg=float(o.get("phase_deg", 5.0)))]
    if "S4" in classes:
        o = opts.get("S4", {})
        cases += [_case("S4", s.element_names[0], s.failures) for s in singles(
            set(_MAGNETS), kind=FailureKind.PARTIAL, amp_scale=float(o.get("amp_scale", 0.9)))]
    if "S5" in classes:
        cases += [_case("S5", s.element_names[0], s.failures) for s in singles({"steerer"})]
    if "S6" in classes:
        cav = [s.element_names[0] for s in singles({"cavity"})]
        for a, b in zip(cav[:-1], cav[1:]):
            cases.append(_case("S6", f"{a}+{b}", ((a, off), (b, off))))
    groups = {"S7": "cryomodule", "S8": "rf_station"}
    for cls_id, kind in groups.items():
        if cls_id in classes:
            for gname, members in sorted(circuits.members(kind).items()):
                fails = [(m, off) for m in members if m in n2c]
                if fails:
                    cases.append(_case(cls_id, gname, fails))
    if "S9" in classes:
        for kind in ("doublet", "arc_bus"):
            for gname, members in sorted(circuits.members(kind).items()):
                fails = [(m, off) for m in members if m in n2c and labels.get(m) in _MAGNETS]
                if fails:
                    cases.append(_case("S9", gname, fails))
    seen = set()
    out = []
    for c in cases:
        if c.case_id in seen:
            raise ValueError(f"duplicate case id {c.case_id}")
        seen.add(c.case_id)
        out.append(c)
    return out, n2c
