"""Optimiser variables collected from ``ADJUST_*`` lattice commands.

If a command references a target that doesn't resolve to anything in
the lattice (e.g. ``ADJUST_STEERER 1`` with no Steerer element present),
the offending command is **skipped with a warning** rather than raising
— matching is often run on lattices that are still being assembled, and
a missing target should not abort the whole run.  Catastrophic config
errors (unknown ``param_idx`` for a known target type) still raise.


Each :class:`Variable` describes one degree of freedom the matcher may
vary: a target object (lattice element, ``BeamConfig``, etc.), an
attribute path on that target, bounds, an initial guess and an optional
*link group* (``ADJUST n`` argument) — variables sharing a link group
are constrained to the same value during optimisation, exposing exactly
one optimiser column.

Mapping of ``ADJUST <target> <param_idx>`` to a concrete element
attribute lives in :data:`_PARAM_INDEX_MAP`; missing entries raise
:class:`MatchingConfigError` with a helpful message.

The walker:

* For each ``Adjust`` command, resolves ``target`` as either an
  integer (diagnostic / element index) or a family-name fragment
  matched against ``elem.name``.  Picks the first matching element
  forward of the command (TraceWin scoping rule).
* For each ``AdjustSteerer*``, links to the ``Steerer`` element with
  matching diagnostic number; raises if none.
* For each ``AdjustBeam*``, walks the per-axis flag tail and emits one
  ``Variable`` per ``flag == 1`` entry, with ``link_group`` set when
  ``flag == 2`` (couple to previous axis).
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any, List, Optional

from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.lattice_commands import (
    Adjust, AdjustBeamCentroid, AdjustBeamCurrent, AdjustBeamEmit,
    AdjustBeamTwiss, AdjustSteerer, AdjustSteererBx, AdjustSteererBy,
    LatticeCommand,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.steerer import Steerer

_log = logging.getLogger(__name__)


class MatchingConfigError(ValueError):
    """Raised when an ADJUST/SET command can't be wired to a real target."""


# ---------------------------------------------------------------------------
# TraceWin parameter-index → element attribute map
# ---------------------------------------------------------------------------
# Manual reference: ``ADJUST N v ...`` where ``v`` selects which scalar on
# the element to vary.  Numbering follows the per-element schema slots.
# Only the parameters we routinely match are listed; missing entries trip
# a clear error.
_PARAM_INDEX_MAP: dict = {
    Drift:      {1: "length"},
    Quadrupole: {1: "length", 2: "gradient", 3: "aperture"},
    Solenoid:   {1: "length", 2: "field",    3: "aperture"},
    RFGap:      {1: "voltage", 2: "phase"},
    Dipole:     {1: "angle",   2: "rho"},
    FieldMap:   {2: "phase",   5: "kb",      6: "ke"},  # selected fields
    FieldMap3D: {2: "phase",   5: "kb",      6: "ke"},  # 3-D variant
    Steerer:    {1: "bx_l",    2: "by_l"},
}


@dataclass
class Variable:
    """One optimiser DoF.

    Attributes
    ----------
    target :
        The Python object whose ``attr`` is varied.  Either a lattice
        element (for ``ADJUST_*``) or a ``BeamConfig`` (for
        ``ADJUST_BEAM_*``).
    attr :
        Attribute name on ``target`` (e.g. ``"gradient"``, ``"alpha_x"``).
    vmin, vmax :
        Bounds; the optimiser's TRF method honours them strictly.
    x0 :
        Initial value (read from ``target`` at collection time).
    link_group :
        Variables with the same non-zero ``link_group`` share a single
        optimiser column.  ``link_group == 0`` is "unlinked".
    label :
        Human-readable name for the result table (e.g.
        ``"QUAD_001.gradient"``).
    source :
        The originating ``LatticeCommand`` (kept for diagnostic reports).
    """

    target: Any
    attr: str
    vmin: float
    vmax: float
    x0: float
    link_group: int = 0
    label: str = ""
    source: Optional[LatticeCommand] = None

    def assign(self, value: float) -> None:
        """Write ``value`` back to ``target.attr`` in place."""
        setattr(self.target, self.attr, float(value))

    def read(self) -> float:
        """Return the current value of ``target.attr``."""
        return float(getattr(self.target, self.attr))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_adjust_target(adjust_cmd: Adjust, lattice, command_idx: int,
                           diag_families=None):
    """Resolve ``Adjust.target`` to a concrete element.

    Integer targets have TWO meanings, disambiguated by the deck itself:

    * **TraceWin family semantics** — when the integer matches a
      ``diag_family`` declared by any ``DIAG_POSITION N …`` marker in the
      lattice, ``ADJUST N v`` is the TraceWin card: N ties the variable to
      diagnostic family N, and the card varies the NEXT non-command
      element after it (TW manual: "the association is made by preceding
      the transport elements with the adjust commands").
    * **Legacy index semantics** — otherwise the integer is a 1-based
      non-command element index across the lattice (the pre-existing
      HELIX convention, pinned by test_variables.py).

    Non-integer targets → family-name match: the *next* element after
    ``command_idx`` whose ``name`` starts with ``target``
    (case-insensitive).  Raise on no match.

    ``diag_families`` — precomputed set of declared diagnostic numbers;
    when None it is derived here (callers in a loop should precompute).
    """
    target = adjust_cmd.target.strip()
    elements = lattice.elements

    # Integer → diagnostic number → 1-based element index.  Parse OUTSIDE
    # the lookup so an out-of-range index raises loudly:
    # MatchingConfigError subclasses ValueError, so the old combined
    # try/except swallowed it and fell through to name matching —
    # silently binding e.g. target "12" to an element named "12MEV_CAV".
    try:
        idx = int(target)
    except ValueError:
        idx = None
    if idx is not None:
        if diag_families is None:
            diag_families = {getattr(e, "diag_family", None)
                             for e in elements
                             if getattr(e, "is_bpm", False)}
            diag_families.discard(None)
        if idx in diag_families:
            # TW family semantics: bind the next non-command element.
            # The integer is usually ALSO a valid legacy 1-based index
            # (any small number is), so a warning here would fire on
            # every genuine TraceWin deck — record the choice at debug
            # level instead for hybrid-deck forensics.
            _log.debug(
                "ADJUST target %d matches a DIAG_POSITION family — using "
                "TraceWin family semantics (binding the element after "
                "the card), not the legacy element-index reading.", idx)
            for elem in elements[command_idx + 1:]:
                if not isinstance(elem, LatticeCommand):
                    return elem
            raise MatchingConfigError(
                f"ADJUST family {idx}: no element follows the card"
            )
        # 1-based; ignore commands when counting.
        seen = 0
        for elem in elements:
            if isinstance(elem, LatticeCommand):
                continue
            seen += 1
            if seen == idx:
                return elem
        raise MatchingConfigError(
            f"ADJUST diagnostic index {idx} not found in lattice"
        )

    # Family-name match — next non-command element forward of the command.
    needle = target.upper()
    for elem in elements[command_idx + 1:]:
        if isinstance(elem, LatticeCommand):
            continue
        if elem.name.upper().startswith(needle):
            return elem
    # Fallback: any element in the lattice.
    for elem in elements:
        if isinstance(elem, LatticeCommand):
            continue
        if elem.name.upper().startswith(needle):
            return elem
    raise MatchingConfigError(
        f"ADJUST target '{target}' did not match any element name "
        f"(searched forward from command and globally)"
    )


def _attr_for(elem, param_idx: int, source) -> str:
    """Look up the attribute name for a given (element-class, param_idx)."""
    from linac_gen.elements.superposed_field_map import SuperposedFieldMap
    if isinstance(elem, SuperposedFieldMap):
        # Cluster children are not lattice elements — a knob cannot be
        # wired to one map inside the cluster (v1).  Raising here routes
        # into collect_variables' established skip-with-warning contract.
        raise MatchingConfigError(
            f"ADJUST cannot address maps inside SUPERPOSE cluster "
            f"'{elem.name}' — set ke/kb/phase on the deck cards or "
            "split the cluster"
        )
    for cls, mapping in _PARAM_INDEX_MAP.items():
        if isinstance(elem, cls) and param_idx in mapping:
            return mapping[param_idx]
    raise MatchingConfigError(
        f"{source.KEYWORD if source else 'ADJUST'} param_idx={param_idx} "
        f"not mapped for element type {type(elem).__name__}"
    )


def _bounds(cmd, default_lo: float = -math.inf, default_hi: float = math.inf):
    """Best-effort bounds extraction; treat zero/zero pair as 'unset'."""
    lo = getattr(cmd, "vmin", None)
    hi = getattr(cmd, "vmax", None)
    if lo is None or hi is None:
        return default_lo, default_hi
    if lo == 0.0 and hi == 0.0:
        return default_lo, default_hi
    if lo > hi:
        warnings.warn(
            f"{getattr(cmd, 'KEYWORD', 'ADJUST')}: vmin ({lo}) > vmax "
            f"({hi}) — swapping.  Check the card; this usually indicates "
            "a slot-order mistake.",
            stacklevel=2,
        )
        lo, hi = hi, lo
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# ADJUST_BEAM_* helpers
# ---------------------------------------------------------------------------
_BEAM_TWISS_AXES = ["alpha_x", "beta_x", "alpha_y", "beta_y",
                    "alpha_z", "beta_z"]
_BEAM_CENTROID_AXES = ["centroid_x", "centroid_xp",
                       "centroid_y", "centroid_yp",
                       "centroid_dphi", "centroid_dw"]
_BEAM_EMIT_AXES = ["emit_nx", "emit_ny", "emit_z"]


def _beam_var_default_bounds(attr: str) -> tuple[float, float]:
    """Reasonable optimiser bounds for a BeamConfig attribute."""
    if attr.startswith("beta_"):
        return (1e-4, 1e3)
    if attr.startswith("emit"):
        return (1e-6, 1e2)
    if attr.startswith("alpha_"):
        return (-50.0, 50.0)
    if attr.startswith("centroid_"):
        return (-100.0, 100.0)
    if attr == "current":
        return (0.0, 1e3)
    return (-1e3, 1e3)


# Namespace base for auto-minted beam-variable link groups — must never
# collide with element ADJUST family numbers taken verbatim from the .dat.
_BEAM_LINK_GROUP_BASE = 1_000_000_000


def _collect_flag_axes(axes: list[str], flags: list[int],
                       beam_cfg, *, link_seed: int,
                       source: LatticeCommand) -> list[Variable]:
    """Turn ``ADJUST_BEAM_*`` flag tail into ``Variable``s.

    Flag values per the manual: ``1`` = adjust, ``2`` = couple to the
    previous-axis variable (link group), ``0`` = skip.
    """
    out: list[Variable] = []
    last_link_group = 0
    next_link = link_seed
    for axis, flag in zip(axes, flags):
        if flag == 0:
            continue
        if flag == 1:
            # Independent DoF — mint a fresh singleton group so a
            # FOLLOWING flag=2 couples to THIS axis, per the documented
            # semantics ("2 = couple to the previous varied axis").  The
            # old code set link=0 and reset the tracker, so the natural
            # "1 2" pattern produced two INDEPENDENT variables (only
            # "2 2" coupled).  A singleton group is equivalent to
            # link_group=0 for the optimizer.
            last_link_group = next_link
            next_link += 1
            link = last_link_group
        elif flag == 2:
            # Couple to previous-axis variable; if none yet, treat as 1.
            if last_link_group == 0:
                last_link_group = next_link
                next_link += 1
            link = last_link_group
        else:
            raise MatchingConfigError(
                f"{source.KEYWORD}: flag value {flag} not in {{0,1,2}}"
            )
        x0 = float(getattr(beam_cfg, axis))
        lo, hi = _beam_var_default_bounds(axis)
        out.append(Variable(
            target=beam_cfg, attr=axis,
            vmin=lo, vmax=hi, x0=x0,
            link_group=link, label=f"beam.{axis}",
            source=source,
        ))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def collect_variables(lattice, beam_cfg) -> List[Variable]:
    """Walk the lattice and return one :class:`Variable` per ADJUST DoF.

    Parameters
    ----------
    lattice :
        :class:`linac_gen.core.lattice.Lattice` containing
        ``LatticeCommand`` instances among its elements.
    beam_cfg :
        :class:`linac_gen.core.config.BeamConfig` to mutate when
        ``ADJUST_BEAM_*`` commands are encountered.

    Raises
    ------
    MatchingConfigError
        On any unresolvable target / parameter / flag.
    """
    variables: List[Variable] = []
    elements = lattice.elements
    # Beam-variable couple groups get a namespace FAR above any plausible
    # .dat ADJUST family number (family numbers from the file are used
    # verbatim as link_group).  Seeding at 1 collided: an ``ADJUST … n=1``
    # merged with the first ADJUST_BEAM_* couple into ONE optimizer
    # column, forcing e.g. a quad gradient numerically equal to a beam
    # Twiss parameter.
    next_link_seed = _BEAM_LINK_GROUP_BASE + 1
    # Declared diagnostic families (DIAG_POSITION N markers) — computed once
    # so integer ADJUST targets can be disambiguated (TW family vs legacy
    # element index) without rescanning the lattice per card.
    diag_families = {getattr(e, "diag_family", None) for e in elements
                     if getattr(e, "is_bpm", False)}
    diag_families.discard(None)
    for i, elem in enumerate(elements):
        if not isinstance(elem, LatticeCommand):
            continue

        if isinstance(elem, Adjust):
            try:
                tgt = _resolve_adjust_target(elem, lattice, i,
                                             diag_families=diag_families)
                # Inside the try so unmapped targets (e.g. a SUPERPOSE
                # cluster container, or an unmapped param_idx) follow
                # the module's documented skip-with-warning contract
                # instead of aborting the whole collect.
                attr = _attr_for(tgt, elem.param_idx, elem)
            except MatchingConfigError as exc:
                warnings.warn(f"matcher: skipping ADJUST: {exc}",
                              stacklevel=2)
                continue
            x0 = float(getattr(tgt, attr))
            lo, hi = _bounds(elem)
            variables.append(Variable(
                target=tgt, attr=attr,
                vmin=lo, vmax=hi, x0=x0,
                link_group=elem.link_group,
                label=f"{tgt.name}.{attr}",
                source=elem,
            ))

        elif isinstance(elem, (AdjustSteererBx, AdjustSteererBy, AdjustSteerer)):
            # Partner-steerer resolution mirrors correction.py: an explicit
            # ``target_name`` wins, else the NEXT Steerer after the card.
            # ``diag_n`` is the DIAGNOSTIC (BPM) number the steerer serves —
            # NOT a steerer counter.  The old code walked to "the diag_n-th
            # Steerer forward of the card", so ``ADJUST_STEERER 7`` varied
            # an unrelated steerer (the two consumers disagreed;
            # correction.py's reading matches TraceWin).
            target = None
            target_name = getattr(elem, "target_name", None)
            if target_name:
                for cand in elements:
                    if isinstance(cand, Steerer) and cand.name == target_name:
                        target = cand
                        break
            if target is None:
                for cand in elements[i + 1:]:
                    if isinstance(cand, Steerer):
                        target = cand
                        break
            if target is None:
                warnings.warn(
                    f"matcher: skipping {elem.KEYWORD}: no Steerer found "
                    f"after the card",
                    stacklevel=2,
                )
                continue

            # ``AdjustSteererBx`` / ``AdjustSteererBy`` subclass
            # ``AdjustSteerer`` so an ``isinstance(elem, AdjustSteerer)``
            # check would over-match.  Use exact ``type`` for the parent
            # to keep BX-only and BY-only commands single-axis.
            do_x = type(elem) is AdjustSteerer or isinstance(elem, AdjustSteererBx)
            do_y = type(elem) is AdjustSteerer or isinstance(elem, AdjustSteererBy)
            for attr_name, do_it in [("bx_l", do_x), ("by_l", do_y)]:
                if not do_it:
                    continue
                vmax = float(getattr(elem, "vmax", 0.0) or 1.0)
                lo = -abs(vmax) if vmax else -1.0
                hi = abs(vmax) if vmax else 1.0
                x0 = float(getattr(target, attr_name))
                variables.append(Variable(
                    target=target, attr=attr_name,
                    vmin=lo, vmax=hi, x0=x0,
                    link_group=0,
                    label=f"{target.name}.{attr_name}",
                    source=elem,
                ))

        elif isinstance(elem, AdjustBeamTwiss):
            link_seed = next_link_seed
            beam_vars = _collect_flag_axes(
                _BEAM_TWISS_AXES, elem.flags, beam_cfg,
                link_seed=link_seed, source=elem,
            )
            next_link_seed += sum(1 for v in beam_vars if v.link_group)
            variables.extend(beam_vars)

        elif isinstance(elem, AdjustBeamCentroid):
            beam_vars = _collect_flag_axes(
                _BEAM_CENTROID_AXES, elem.flags, beam_cfg,
                link_seed=next_link_seed, source=elem,
            )
            next_link_seed += sum(1 for v in beam_vars if v.link_group)
            variables.extend(beam_vars)

        elif isinstance(elem, AdjustBeamEmit):
            beam_vars = _collect_flag_axes(
                _BEAM_EMIT_AXES, elem.flags, beam_cfg,
                link_seed=next_link_seed, source=elem,
            )
            next_link_seed += sum(1 for v in beam_vars if v.link_group)
            variables.extend(beam_vars)

        elif isinstance(elem, AdjustBeamCurrent):
            if not elem.flags or elem.flags[0] != 1:
                continue
            x0 = float(getattr(beam_cfg, "current"))
            lo, hi = _beam_var_default_bounds("current")
            variables.append(Variable(
                target=beam_cfg, attr="current",
                vmin=lo, vmax=hi, x0=x0,
                link_group=0, label="beam.current", source=elem,
            ))

    return variables


# ---------------------------------------------------------------------------
# Sequential-scan helpers (used by the ``sequential_scan`` matcher branch
# and the GUI's Sequential Scan setup dialog).
# ---------------------------------------------------------------------------
def categorize_fieldmap(elem) -> str:
    """Classify a ``FieldMap`` / ``FieldMap3D`` as a solenoid or cavity.

    Returns ``"solenoid"`` when ``ke == 0`` (pure magnetic element, no
    electric acceleration), ``"cavity"`` when ``ke != 0`` (RF cavity).
    Returns ``"other"`` for non-FieldMap inputs.
    """
    if not isinstance(elem, (FieldMap, FieldMap3D)):
        return "other"
    ke = float(getattr(elem, "ke", 0.0) or 0.0)
    return "solenoid" if abs(ke) < 1e-12 else "cavity"


def group_variables_by_element(
    variables: List[Variable],
) -> "OrderedDict[Any, List[Variable]]":
    """Group ``variables`` by their ``target`` element, preserving the
    first-encountered order.

    Variables whose target is not a lattice element (e.g. ``AdjustBeam*``
    on a ``BeamConfig``) are still grouped, but consumers typically
    filter for ``isinstance(target, FieldMap)`` etc. before scanning.

    Returns an ``OrderedDict`` keyed by the target object so callers can
    iterate elements in the order the ADJUST cards appear in the .dat.
    """
    groups: "OrderedDict[Any, List[Variable]]" = OrderedDict()
    for var in variables:
        key = var.target
        if key not in groups:
            groups[key] = []
        groups[key].append(var)
    return groups
