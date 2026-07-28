"""Failure scenarios: a set of (element, FailureMode) + enumeration."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from linac_gen.failures.element_filter import failable_elements, valid_kinds
from linac_gen.failures.failure_mode import FailureKind, FailureMode


@dataclass(frozen=True)
class FailureScenario:
    """One failure scenario: a set of simultaneously-failed elements."""
    failures: tuple[tuple[str, FailureMode], ...]   # ((elem_name, mode), …)
    label: str = ""

    def element_overrides(
        self, name_to_class: dict[str, str]
    ) -> tuple[tuple[str, float], ...]:
        """Flatten every failure's overrides into one ScanPoint tuple."""
        ov: list[tuple[str, float]] = []
        for name, mode in self.failures:
            ov.extend(mode.element_overrides(name, name_to_class[name]))
        return tuple(ov)

    @property
    def element_names(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.failures)


def _label(failures: tuple[tuple[str, FailureMode], ...]) -> str:
    return " + ".join(f"{n}:{m.label}" for n, m in failures)


def enumerate_scenarios(
    lattice, *,
    types=None,
    kind: FailureKind = FailureKind.OFF,
    combination: str = "single",
    amp_scale: float = 1.0,
    phase_deg: float = 0.0,
    custom_sets: list[list[str]] | None = None,
    only_names: list[str] | None = None,
) -> tuple[list[FailureScenario], dict[str, str], list[str]]:
    """Build the failure scenarios for a sweep.

    Returns ``(scenarios, name_to_class, element_names)``.

    * ``"single"`` — one scenario per failable element  → N scenarios.
    * ``"pairs"``  — the N singles (which fill the heatmap diagonal) **plus**
      every unordered pair → N + N(N-1)/2 scenarios.
    * ``"custom"`` — one scenario per list in ``custom_sets``.

    Every failed element uses the same ``kind`` (with ``amp_scale`` /
    ``phase_deg``); elements that don't support ``kind`` (e.g. DETUNE on a
    magnet) are filtered out.
    """
    failable = failable_elements(lattice, types)
    elems = [(n, lbl, c) for (n, lbl, c) in failable if kind in valid_kinds(lbl)]
    if only_names is not None:
        keep = set(only_names)
        elems = [(n, lbl, c) for (n, lbl, c) in elems if n in keep]
    name_to_class = {n: c for (n, _lbl, c) in elems}
    names = [n for (n, _lbl, _c) in elems]
    mode = FailureMode(kind=kind, amp_scale=amp_scale, phase_deg=phase_deg)

    scenarios: list[FailureScenario] = []
    if combination == "single":
        for n in names:
            f = ((n, mode),)
            scenarios.append(FailureScenario(f, label=_label(f)))
    elif combination == "pairs":
        for n in names:                       # singles → heatmap diagonal
            f = ((n, mode),)
            scenarios.append(FailureScenario(f, label=_label(f)))
        for a, b in combinations(names, 2):
            f = ((a, mode), (b, mode))
            scenarios.append(FailureScenario(f, label=_label(f)))
    elif combination == "custom":
        for cset in (custom_sets or []):
            f = tuple((n, mode) for n in cset if n in name_to_class)
            if f:
                scenarios.append(FailureScenario(f, label=_label(f)))
    else:
        raise ValueError(
            f"unknown combination {combination!r} "
            f"(expected 'single', 'pairs', or 'custom')")

    return scenarios, name_to_class, names
