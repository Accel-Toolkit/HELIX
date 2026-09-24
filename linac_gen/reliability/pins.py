"""Frozen-phase pins: freeze every cavity at its DESIGN operating point.

HELIX re-solves ``SET_SYNC_PHASE`` cavities on every run and applies a
thin ``GAP``'s deck phase as its synchronous phase, so a fault upstream is
always tracked with the downstream RF ideally re-phased.  A reliability
campaign also wants the other bracket — the machine frozen as it was set
before the fault — and that is what a pin does: a field-map cavity keeps
the ψ its design pass calibrated (``sync_phase_pin``, the multibunch
mechanism), and a thin gap keeps the reference RF-clock phase it saw at
its entrance (``RFGap.sync_phase_pin``); the slower reference of a
faulted line then shows up as a phase error at every cavity downstream.

:func:`harvest_pins` runs one envelope design pass on a **deep copy** of
the lattice and returns the pins as ``@index.sync_phase_pin`` overrides,
the form :func:`linac_gen.cli.common.apply_element_override` and the
scan-pool workers transport (as ``linac_gen.train.replay`` does for
bunch trains).  The live lattice is never touched.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

__all__ = ["PinSet", "collect_pins", "harvest_pins"]


@dataclass(frozen=True)
class PinSet:
    """The pins of one design pass.

    ``overrides`` — ``(("@7.sync_phase_pin", 123.4), …)`` in lattice order;
    ``pins`` — ``{name: value}``; ``kinds`` — ``{name: "psi" | "clock"}``
    (a field-map ψ offset in degrees, or a thin gap's entrance clock in
    degrees at the reference frequency); ``skipped`` — pinnable elements
    whose design pass left no value (their calibration never ran)."""
    overrides: tuple = ()
    pins: dict = field(default_factory=dict)
    kinds: dict = field(default_factory=dict)
    skipped: tuple = ()

    def __len__(self) -> int:
        return len(self.overrides)


def _pinnable(el):
    """``"psi"`` for a sync-phase field-map cavity, ``"clock"`` for a thin
    gap, ``None`` for everything else."""
    if not hasattr(el, "sync_phase_pin"):
        return None
    if hasattr(el, "_entry_phi_s") and not hasattr(el, "_sync_offset_deg"):
        return "clock"                                   # RFGap
    if getattr(el, "p_flag", 0) == 1 or getattr(el, "sync_phase", False):
        return "psi"                                     # FieldMap / NCells
    return None


def _clock_frequency_after(el, current: float) -> float:
    """The reference-clock frequency after ``el``: a FREQ card or an RF
    element with a live electric channel switches it (what both engines
    do); everything else leaves it."""
    f = getattr(el, "frequency_mhz", None)
    if f is not None and type(el).__name__ == "Freq":
        return float(f)
    if hasattr(el, "sync_phase_pin"):                       # RFGap / FieldMap / FieldMap3D / NCells
        ke = getattr(el, "ke", None)
        if ke is not None and abs(float(ke or 0.0)) < 1e-12:  # unpowered / magnetic field map
            return current
        eff = float(getattr(el, "effective_frequency", 0.0) or getattr(el, "frequency_mhz", 0.0) or 0.0)
        return eff if eff > 0.0 else current
    return current


def collect_pins(lattice, ref_frequency: float | None = None) -> PinSet:
    """Read the pins a design pass left on ``lattice``'s elements.

    Pure: no tracking, no mutation.  Raises ``NotImplementedError`` when a
    pinnable cavity sits inside a ``SuperposedFieldMap`` — the
    ``@index`` override selectors cannot address a child — or when a thin
    gap's frequency differs from the reference clock's at its entrance
    (``ref_frequency`` = the beam frequency the clock starts at; the walk
    follows FREQ cards and RF elements as both engines do): the clock pin
    is in degrees at the reference frequency, and the two engines switch
    the reference at different points of a per-element jump.
    """
    overrides: list = []
    pins: dict = {}
    kinds: dict = {}
    skipped: list = []
    unindexed: list = []
    f_clock = float(ref_frequency) if ref_frequency else None
    for idx, el in enumerate(lattice.elements):
        kids = getattr(el, "children", None)
        if kids:
            for _z, child in kids:
                if _pinnable(child) is not None:
                    unindexed.append(getattr(child, "name", repr(child)))
            continue
        kind = _pinnable(el)
        if kind == "clock" and f_clock is not None:
            f_gap = float(getattr(el, "effective_frequency", 0.0) or 0.0)
            if f_gap > 0.0 and f_gap != f_clock:
                raise NotImplementedError(
                    f"frozen-phase pin of thin gap {getattr(el, 'name', idx + 1)!r}: its "
                    f"frequency ({f_gap:g} MHz) differs from the reference clock's at its "
                    f"entrance ({f_clock:g} MHz) — a per-element frequency jump is not "
                    "supported by the clock pin; put a FREQ card before the gap")
        if f_clock is not None:
            f_clock = _clock_frequency_after(el, f_clock)
        if kind is None:
            continue
        name = getattr(el, "name", f"@{idx + 1}")
        value = (el._entry_phi_s if kind == "clock"
                 else getattr(el, "_sync_offset_deg", None))
        if value is None:
            skipped.append(name)
            continue
        overrides.append((f"@{idx + 1}.sync_phase_pin", float(value)))
        pins[name] = float(value)
        kinds[name] = kind
    if unindexed:
        raise NotImplementedError(
            "frozen-phase pins cannot address pinned SuperposedFieldMap "
            "children (not top-level elements): " + ", ".join(unindexed))
    return PinSet(overrides=tuple(overrides), pins=pins, kinds=kinds,
                  skipped=tuple(skipped))


def harvest_pins(lattice, beam_cfg, *, mode: str = "envelope",
                 env_solver: str = "matrix", sc_config=None,
                 step_config=None, seed: int = 42) -> PinSet:
    """One design pass on a deep copy of ``lattice``, then
    :func:`collect_pins` on the copy.

    ``mode`` picks the engine the pins are for: ``"envelope"`` (with
    ``env_solver`` as in ``cli.common.run_envelope_sim``) or ``"mp"`` (a
    multiparticle pass with the campaign's ``sc_config`` / ``step_config``
    and 4 macroparticles — the reference clock is particle-independent).
    The two engines accumulate the RF clock with different rounding
    (per sub-step vs per element: ~1e-13 deg without and up to ~1e-11 deg
    with space charge on the demo deck), so a bracket must be pinned by
    the engine that runs it for the nominal frozen and re-phased runs to
    agree bit for bit.  ``beam_cfg`` is the campaign's
    ``BeamConfig``; the live lattice is never touched."""
    work = copy.deepcopy(lattice)
    if mode == "envelope":
        from linac_gen.cli.common import run_envelope_sim
        run_envelope_sim(work, beam_cfg, env_solver=env_solver)
    elif mode == "mp":
        from linac_gen.core.simulation import Simulation
        from linac_gen.core.step_config import StepConfig
        from linac_gen.distributions.factory import create_beam
        cfg = copy.deepcopy(beam_cfg)
        cfg.n_particles = 4
        work.step_config = step_config or StepConfig()
        sc = sc_config if cfg.current > 0 else None
        Simulation(work, create_beam(cfg, seed=seed), space_charge=sc).run()
    else:
        raise ValueError(f"harvest_pins: mode must be 'envelope' or 'mp', got {mode!r}")
    return collect_pins(work, ref_frequency=float(getattr(beam_cfg, "frequency", 0.0) or 0.0))
