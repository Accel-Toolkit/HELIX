"""Failure modes for element failure analysis.

A :class:`FailureMode` maps an element + mode to ``element_overrides`` — the
``((selector, value), …)`` tuples that
:attr:`linac_gen.parallel.scan_pool.ScanPoint.element_overrides` feeds through
:func:`linac_gen.cli.common.apply_element_override`.

Failures are injected through the elements' **additive error slots**
(``effective = design × (1 + rel)``; see ``rf_gap.py``/``quadrupole.py``/
``solenoid.py``/``dipole.py``/``field_map.py``), so the *override value* is a
RELATIVE delta, not an absolute setting:

* ``OFF``     → rel slot = ``-1.0``                 (zeroes the strength)
* ``PARTIAL`` → rel slot = ``amp_scale - 1.0``      (90 % → ``-0.10``)
* ``DETUNE``  → ``voltage_rel = amp_scale - 1.0`` and/or
                ``phase_offset = phase_deg`` (additive degrees)

All sign logic lives here so the rest of the module never has to reason about
the ``design × (1 + δ)`` convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureKind(str, Enum):
    OFF = "off"          # hard off: element transfers nothing
    DETUNE = "detune"    # cavity: amplitude scale and/or phase offset
    PARTIAL = "partial"  # magnet: field/gradient scaled to a fraction


# element-class name -> (relative-strength slot, phase-offset slot | None)
_REL_SLOT: dict[str, tuple[str, str | None]] = {
    "RFGap":      ("voltage_rel", "phase_offset"),
    "FieldMap":   ("voltage_rel", "phase_offset"),
    "FieldMap3D": ("voltage_rel", "phase_offset"),
    "NCells":     ("voltage_rel", "phase_offset"),
    "Quadrupole": ("gradient_rel", None),
    "Solenoid":   ("field_rel", None),
    "Dipole":     ("field_rel", None),
}

# element classes whose only "off" knob is an absolute kick (no relative slot)
_ABS_OFF: dict[str, tuple[str, ...]] = {
    "Steerer": ("bx_l", "by_l"),
}

# classes for which DETUNE (amplitude + RF phase) is meaningful
_CAVITY = {"RFGap", "FieldMap", "FieldMap3D", "NCells"}


@dataclass(frozen=True)
class FailureMode:
    """One element's failure state.

    ``amp_scale`` is the fraction of nominal strength (PARTIAL/DETUNE);
    ``phase_deg`` is the additive RF-phase offset in degrees (DETUNE only).
    For DETUNE, ``amp_scale == 1.0`` means "no amplitude change" (phase-only).
    """
    kind: FailureKind
    amp_scale: float = 1.0
    phase_deg: float = 0.0

    def element_overrides(
        self, name: str, element_class: str
    ) -> tuple[tuple[str, float], ...]:
        """Return ``((selector, value), …)`` for ``ScanPoint.element_overrides``."""
        if element_class in _ABS_OFF:
            if self.kind is not FailureKind.OFF:
                raise ValueError(
                    f"{element_class} only supports OFF (it has no relative "
                    f"error slot)")
            return tuple((f"{name}.{attr}", 0.0)
                         for attr in _ABS_OFF[element_class])

        if element_class not in _REL_SLOT:
            raise ValueError(
                f"element class {element_class!r} has no failure knob")
        rel_slot, phase_slot = _REL_SLOT[element_class]

        if self.kind is FailureKind.OFF:
            return ((f"{name}.{rel_slot}", -1.0),)

        if self.kind is FailureKind.PARTIAL:
            return ((f"{name}.{rel_slot}", float(self.amp_scale) - 1.0),)

        if self.kind is FailureKind.DETUNE:
            if element_class not in _CAVITY:
                raise ValueError("DETUNE applies only to cavities")
            ov: list[tuple[str, float]] = []
            if self.amp_scale != 1.0:
                ov.append((f"{name}.{rel_slot}", float(self.amp_scale) - 1.0))
            if phase_slot is not None and self.phase_deg:
                ov.append((f"{name}.{phase_slot}", float(self.phase_deg)))
            if not ov:
                raise ValueError(
                    "DETUNE with amp_scale=1.0 and phase_deg=0 is a no-op")
            return tuple(ov)

        raise ValueError(f"unknown failure kind {self.kind!r}")

    @property
    def label(self) -> str:
        if self.kind is FailureKind.OFF:
            return "off"
        if self.kind is FailureKind.PARTIAL:
            return f"partial({self.amp_scale * 100:.0f}%)"
        bits = []
        if self.amp_scale != 1.0:
            bits.append(f"amp={self.amp_scale:.2f}")
        if self.phase_deg:
            bits.append(f"phi={self.phase_deg:+g}")
        return "detune(" + ",".join(bits) + ")"


def can_fail(element_class: str) -> bool:
    """True if ``element_class`` has any failure knob."""
    return element_class in _REL_SLOT or element_class in _ABS_OFF


def is_cavity(element_class: str) -> bool:
    return element_class in _CAVITY
