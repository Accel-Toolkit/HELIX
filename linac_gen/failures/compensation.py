"""Fault compensation: re-tune neighbouring elements to recover the beam.

After a failure, pick a compensation zone (k-out-of-n neighbours /
l-neighbouring FODO periods / manual) and re-tune the compensators' amplitude
(+ RF phase, for cavities) via the matcher to restore the design exit energy
and avoid beam loss — the MYRRHA / LightWin local-compensation scheme.

A failed element is compensated within its own family only: RF (cavities, and
unpowered spares when opted in) or transverse focusing (quadrupoles and
solenoids).  A bend is never a compensator — its knob, the angle, moves the
reference path — and a failed bend is not compensable by re-tuning optics; an
element with no family (a steerer) gets no compensators.

Mechanism: deepcopy the lattice, strip its existing ADJUST/SET cards, inject
the failure + temporary `Adjust`/`SetKeOutMin`/`MinTransmission` cards for the
compensators, then call :func:`linac_gen.matching.match` (there is no
lower-level matcher entry that takes hand-built Variable/Constraint lists).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from linac_gen.cli.common import (apply_element_override, make_sc_config,
                                   make_step_config, result_summary,
                                   run_envelope_sim, run_mp_sim)
from linac_gen.elements.lattice_commands import (Adjust, LatticeCommand,
                                                 MinEmitGrowth, MinTransmission,
                                                 SetKeOutMin, SetPhaseOut,
                                                 SetSizeMax)
from linac_gen.failures.element_filter import ALL_TYPES, EXTRA_TYPES, failable_elements
from linac_gen.matching import match

# the compensation family of a failure-vocabulary label: an element is compensated within its
# own family only.  A bend has none: it is never a compensator and no re-tune of the optics
# brings the beam back onto the line it sets.  A label with no family gets no compensators.
_CATEGORY = {"cavity": "rf", "spare": "rf",          # an unpowered cavity compensates cavities
             "quad": "focusing", "solenoid": "focusing"}

# element class -> [(param_idx, attr, role)]  (param_idx per _PARAM_INDEX_MAP).  No Dipole:
# its angle moves the reference path.
_COMP_PARAMS = {
    "RFGap":      [(1, "voltage", "amp"), (2, "phase", "phase")],
    "FieldMap":   [(6, "ke", "amp"), (2, "phase", "phase")],
    "FieldMap3D": [(6, "ke", "amp"), (2, "phase", "phase")],
    "Quadrupole": [(2, "gradient", "amp")],
    "Solenoid":   [(2, "field", "amp")],
}

_CONSTRAINT_KW = {
    "SET_TWISS", "SET_KE_OUT_MIN", "MIN_TRANSMISSION", "MIN_EMIT_GROWTH",
    "MIN_EMIT_4D_GROWTH", "SET_SIZE", "SET_SIZE_MAX", "SET_SIZE_MIN",
    "SET_BEAM_PHASE_ADV", "SET_ADV", "SET_POSITION", "SET_ACHROMAT",
    "SET_SEPARATION", "SET_PHASE_OUT",
    "SET_BEAM_ENERGY", "SET_BEAM_E0_P0",      # TraceWin matching criteria (2026-11)
}


@dataclass
class CompensationConfig:
    strategy: str = "k_out_of_n"     # k_out_of_n | l_neighboring_lattices | manual
    k: int = 2
    l: int = 1
    manual_names: list[str] | None = None
    algorithm: str = "cmaes"
    cost_solver: str = "envelope"    # "envelope" | "mp"
    max_iter: int = 200
    amp_bounds: tuple[float, float] = (0.5, 1.5)     # × current amplitude
    phase_bounds_deg: float = 30.0                   # ± around current phase
    recover_tol_energy_mev: float = 0.05
    transmission_margin_pct: float = 0.5
    space_charge: bool = False
    # --- reliability extensions (every default reproduces the behaviour
    #     above exactly) ---
    extra_types: tuple = ()              # e.g. ("spare",): unpowered field
    #                                      maps become k_out_of_n / manual
    #                                      candidates (the RF family)
    family_of: dict | None = None        # element name -> family label
    amp_bounds_by_family: dict | None = None   # family -> (lo, hi) x the
    #                                      family's NOMINAL amplitude (the
    #                                      signed median of its powered
    #                                      members) instead of x current —
    #                                      the RF headroom rule; a spare
    #                                      (current 0) gets (lo, hi) x nominal
    objectives: dict | None = None       # None = SET_KE_OUT_MIN (+ MIN_
    #                                      TRANSMISSION in mp).  Keys:
    #                                      ke_out_min (bool), ke_weight,
    #                                      transmission (bool), size_max
    #                                      {x_mm, y_mm, weight} over the line,
    #                                      phase_out {phase_deg, weight,
    #                                      tol_deg} = the nominal exit clock,
    #                                      emit_growth (bool, MIN_EMIT_GROWTH
    #                                      x/y/z — a magnet fault leaves the
    #                                      energy untouched, so without it a
    #                                      magnet compensation is a no-op)


@dataclass
class CompensationResult:
    recovered: bool
    compensator_names: list[str]
    settings: dict = field(default_factory=dict)
    residual_cost: float = float("inf")
    metrics_after: dict = field(default_factory=dict)
    match_success: bool = False
    message: str = ""


# link groups of a compensator's knobs that share one optimiser column (a cavity map's ke and
# kb); far from the family numbers a deck's ADJUST cards carry, and from the beam's
_COMP_LINK_BASE = 900_000_000


def _solenoidal(elem) -> bool:
    """A static magnetic field map whose field on the axis is longitudinal: a solenoid.  A 1-D
    Bz or 2-D cylindrical map is one; a 3-D map is when its on-axis Bz exceeds its on-axis
    transverse field and a tenth of its peak; a quadrupole's field vanishes on the axis and a
    dipole's is transverse, and a 1-D gradient G(z) or a 2-D transverse map is neither
    (``field_map_data.on_axis_field_kind``, the rule the Sacherer envelope reads too)."""
    from linac_gen.io.field_map_data import on_axis_field_kind
    from linac_gen.io.tracewin_geom import Channel
    chans = getattr(getattr(elem, "field_data", None), "channels", None) or {}
    if any(ch.is_rf for ch in chans):
        return False
    ch = chans.get(Channel.STAT_B)
    return ch is not None and on_axis_field_kind(ch) == "longitudinal"


def knobs(elem) -> list:
    """The compensator's matcher knobs, ``[(param_idx, attr, role)]``.  A field map's follow its
    field: a cavity (an RF map, an unpowered spare among them) varies ke, with kb in the same
    column when kb equals ke, and its phase; a static solenoid map varies kb; a static
    quadrupole or dipole map none.  Without field data, ke == 0 and kb != 0 reads as a solenoid."""
    cls = type(elem).__name__
    if cls not in ("FieldMap", "FieldMap3D"):
        return list(_COMP_PARAMS.get(cls, []))
    ke = float(getattr(elem, "ke", 0.0) or 0.0)
    kb = float(getattr(elem, "kb", 0.0) or 0.0)
    chans = getattr(getattr(elem, "field_data", None), "channels", None)
    if chans:
        rf = any(ch.is_rf for ch in chans)
        static_solenoid = (not rf) and _solenoidal(elem)
    else:
        rf = not (abs(ke) < 1e-12 and abs(kb) > 1e-12)
        static_solenoid = not rf
    if static_solenoid:
        return [(5, "kb", "amp")]
    if not rf:
        return []
    out = [(6, "ke", "amp")]
    if kb == ke:
        out.append((5, "kb", "amp"))
    return out + [(2, "phase", "phase")]


def _static_magnet_map(elem) -> bool:
    """A powered field map with no RF field: a solenoid's, a quadrupole's (an unpowered spare
    cavity is none)."""
    if type(elem).__name__ not in ("FieldMap", "FieldMap3D"):
        return False
    if float(getattr(elem, "kb", 0.0) or 0.0) == 0.0:
        return False
    chans = getattr(getattr(elem, "field_data", None), "channels", None)
    if chans:
        return not any(ch.is_rf for ch in chans)
    return float(getattr(elem, "ke", 0.0) or 0.0) == 0.0


def _focus_cells(lattice) -> list:
    """The cells of a "neighbouring cells" zone: just after focuser k to focuser k+2, as
    find_fodo_cells takes them, with the focusers the focusing elements (a quadrupole, a
    solenoid, a powered static magnet map; find_fodo_cells also counts an unpowered spare),
    the first cell reaching back to the line's start and the last on to its end."""
    foc = [i for i, e in enumerate(lattice.elements)
           if type(e).__name__ in ("Quadrupole", "Solenoid") or _static_magnet_map(e)]
    cells = [(foc[k] + 1, foc[k + 2]) for k in range(len(foc) - 2) if foc[k + 2] > foc[k] + 1]
    if cells:
        cells[0] = (0, cells[0][1])
        cells[-1] = (cells[-1][0], len(lattice.elements) - 1)
    return cells


def _category(label: str) -> str | None:
    """The compensation family of ``label``, or None: no family, no compensators."""
    return _CATEGORY.get(label)


def _positions(lattice) -> dict[str, int]:
    pos: dict[str, int] = {}
    for i, e in enumerate(lattice.elements):
        nm = getattr(e, "name", None)
        if nm and not isinstance(e, LatticeCommand):
            pos.setdefault(nm, i)
    return pos


def _noncmd_index(lattice) -> dict[str, int]:
    """name -> 1-based index among non-command elements (Adjust integer target)."""
    idx: dict[str, int] = {}
    seen = 0
    for e in lattice.elements:
        if isinstance(e, LatticeCommand):
            continue
        seen += 1
        nm = getattr(e, "name", None)
        if nm:
            idx.setdefault(nm, seen)
    return idx


def select_zone(lattice, failed_names, cfg: CompensationConfig) -> list[str]:
    """Return compensator element NAMES (excludes the failed elements).

    ``cfg.extra_types`` widens the candidate pool with the reliability
    vocabulary (``"spare"`` unpowered cavities); classes without a matcher
    knob in ``_COMP_PARAMS`` (a bend, a multi-gap ``NCells``, a ``Steerer``)
    are never candidates.  ``k_out_of_n`` takes the failed element's family
    only, and none for an element with no family; ``manual`` keeps to the
    failed elements' families too, and says what it drops (a bend, another
    family's element, a name with no knob or not asked for).  Every label is read in the reliability vocabulary, which knows an
    unpowered field map as a spare (RF), not a solenoid, and a multi-gap cavity
    as a cavity; only the labels asked for are candidates."""
    allowed = set(ALL_TYPES) | set(cfg.extra_types or ())
    every = failable_elements(lattice, tuple(ALL_TYPES) + tuple(EXTRA_TYPES))
    failed_label = {n: lbl for (n, lbl, _c) in every}
    elem_of = {getattr(e, "name", None): e for e in lattice.elements}
    label_of = {n: lbl for (n, lbl, _c) in every
                if lbl in allowed and knobs(elem_of[n])}
    pos = _positions(lattice)
    failed = set(failed_names)
    chosen: list[str] = []

    if cfg.strategy == "manual":
        import warnings
        cls_of = {getattr(e, "name", None): type(e).__name__ for e in lattice.elements}
        cats = {_category(failed_label.get(fn, "")) for fn in failed_names} - {None}
        bends, other, unusable = [], [], []
        for n in (cfg.manual_names or []):
            if n in failed or n in chosen:
                continue
            if cls_of.get(n) == "Dipole":
                bends.append(n)
            elif n not in label_of:
                unusable.append(n)
            elif _category(label_of[n]) in cats:
                chosen.append(n)
            else:
                other.append(n)
        for names, why in ((bends, "a bend is never a compensator (its angle moves the "
                                   "reference path)"),
                           (other, "not of the failed element's family"),
                           (unusable, "no compensation knob, or of a kind not asked for "
                                      "(spares need extra_types)")):
            if names:
                warnings.warn(f"manual compensators: {', '.join(names)} dropped — {why}",
                              UserWarning, stacklevel=2)
    elif cfg.strategy == "k_out_of_n":
        seen: set[str] = set()
        for fn in failed_names:
            if fn not in pos:
                continue
            cat = _category(failed_label.get(fn, ""))
            if cat is None:
                continue                    # a bend or an element with no family
            cands = [n for n in label_of
                     if n not in failed and _category(label_of[n]) == cat]
            # nearest first; on a tie, upstream wins, then ascending index
            cands.sort(key=lambda n: (abs(pos[n] - pos[fn]),
                                      0 if pos[n] < pos[fn] else 1, pos[n]))
            for n in cands[:max(1, cfg.k) * 2]:     # ~k each side
                if n not in seen:
                    seen.add(n)
                    chosen.append(n)
    elif cfg.strategy == "l_neighboring_lattices":
        # the failed elements' families only, in cells reaching both ends of the line
        cats = {_category(failed_label.get(fn, "")) for fn in failed_names} - {None}
        cells = _focus_cells(lattice)               # [(start, end)] index pairs
        keep: set[int] = set()
        for ci, (s, e) in enumerate(cells):
            if any(s <= pos.get(fn, -1) <= e for fn in failed_names):
                for cj in range(max(0, ci - cfg.l),
                                min(len(cells) - 1, ci + cfg.l) + 1):
                    keep.add(cj)
        for cj in sorted(keep):
            cs, ce = cells[cj]
            for n, p in pos.items():
                if cs <= p <= ce and n not in failed and n in label_of \
                        and _category(label_of[n]) in cats and n not in chosen:
                    chosen.append(n)
    else:
        raise ValueError(f"unknown strategy {cfg.strategy!r}")
    return chosen


def family_nominals(lattice, cfg: CompensationConfig) -> dict:
    """``{family: signed median amplitude of its POWERED members}`` for the
    families named in ``cfg.family_of`` (``{}`` when no families)."""
    if not cfg.family_of:
        return {}
    import statistics
    vals: dict[str, list[float]] = {}
    for e in lattice.elements:
        nm = getattr(e, "name", None)
        fam = cfg.family_of.get(nm) if nm else None
        if fam is None:
            continue
        amp = next((attr for (_p, attr, role) in knobs(e) if role == "amp"), None)
        if amp is None:
            continue
        v = float(getattr(e, amp))
        if v != 0.0:
            vals.setdefault(fam, []).append(v)
    return {fam: float(statistics.median(v)) for fam, v in vals.items()}


def amp_bounds(cfg: CompensationConfig, name: str, cur: float,
               nominals: dict) -> tuple[float, float]:
    """The matcher's amplitude window for one compensator.

    With a family headroom rule (``cfg.family_of`` + ``amp_bounds_by_family``)
    the window is ``(lo, hi) x`` the family's nominal amplitude, so an
    unpowered spare (``cur == 0``) can be brought up to the family's
    headroom; otherwise the historical ``cfg.amp_bounds x current`` rule
    with its ``cur == 0`` guard."""
    fam = (cfg.family_of or {}).get(name)
    fb = (cfg.amp_bounds_by_family or {}).get(fam) if fam is not None else None
    if fb is not None:
        nominal = nominals.get(fam)
        if nominal is None or nominal == 0.0:
            nominal = cur
        if nominal != 0.0:
            a, b = float(fb[0]) * nominal, float(fb[1]) * nominal
            vmin, vmax = min(a, b), max(a, b)
            if cur != 0.0:          # a powered member keeps its own setting in its window
                vmin, vmax = min(vmin, cur), max(vmax, cur)
            if vmin != vmax:
                return vmin, vmax
    a, b = cfg.amp_bounds[0] * cur, cfg.amp_bounds[1] * cur
    vmin, vmax = min(a, b), max(a, b)
    if vmin == vmax:                                  # cur == 0 guard
        vmin, vmax = cur - 1.0, cur + 1.0
    return vmin, vmax


def objective_cards(cfg: CompensationConfig, baseline_metrics: dict,
                    n_noncmd: int) -> tuple[list, list]:
    """``(leading, trailing)`` constraint cards for a compensation match:
    ``leading`` go before the first element (a ``SET_SIZE_MAX`` window over
    the whole line), ``trailing`` at the end (exit energy, transmission).
    ``cfg.objectives`` None = the historical pair."""
    obj = cfg.objectives or {}
    leading: list = []
    trailing: list = []
    e0 = baseline_metrics.get("ref_w_kin")
    if e0 is not None and obj.get("ke_out_min", True):
        trailing.append(SetKeOutMin(name="__COMP_KE", energy_mev=float(e0),
                                    weight=float(obj.get("ke_weight", 10.0))))
    t0 = baseline_metrics.get("transmission")
    if cfg.cost_solver == "mp" and t0 is not None and obj.get("transmission", True):
        trailing.append(MinTransmission(
            name="__COMP_T", threshold_pct=max(0.0, t0 - cfg.transmission_margin_pct),
            weight=50.0))
    phase = obj.get("phase_out")
    if phase and phase.get("phase_deg") is not None:
        trailing.append(SetPhaseOut(
            name="__COMP_PHI", phase_deg=float(phase["phase_deg"]),
            weight=float(phase.get("weight", 5.0)),
            tol_deg=float(phase.get("tol_deg", 0.0))))
    if obj.get("emit_growth"):
        w = float(obj.get("emit_weight", 1.0))
        for plane in ("X", "Y", "Z"):
            trailing.append(MinEmitGrowth(name=f"__COMP_E{plane}", plane=plane, weight=w))
    size = obj.get("size_max")
    if size:
        leading.append(SetSizeMax(
            name="__COMP_SIZE", k=float(size.get("weight", 1.0)),
            n_elems=int(n_noncmd), x_mm=float(size.get("x_mm", 0.0)),
            y_mm=float(size.get("y_mm", 0.0))))
    return leading, trailing


def _forward_metrics(lattice, beam_cfg, cost_solver: str, seed: int = 42) -> dict:
    from linac_gen.failures.study import recorder_metrics
    if cost_solver == "mp":
        sc = make_sc_config(beam_cfg, {}, {})
        # the lattice's RFQ solver choice is kept (the GUI sets it)
        from dataclasses import replace
        from linac_gen.core.step_config import rfq_fields
        step = replace(make_step_config({}, {}),
                       **rfq_fields(getattr(lattice, "step_config", None)))
        # run_mp_sim returns (recorder, beam) — take the recorder.
        res, _beam = run_mp_sim(lattice, beam_cfg, sc, step, seed=seed)
    else:
        res = run_envelope_sim(lattice, beam_cfg)
    from linac_gen.analysis.aperture_edge import edge_metrics
    return recorder_metrics(res) | edge_metrics(res, lattice)


def compensate(lattice, beam_cfg, scenario, name_to_class, baseline_metrics,
               cfg: CompensationConfig) -> CompensationResult:
    """Try to recover the beam after ``scenario``'s failures by re-tuning a
    compensation zone with the matcher.  ``lattice`` is the NOMINAL lattice
    (deep-copied here); ``beam_cfg`` is a ``BeamConfig``.

    The registered surrogates take no part (``registry.suspended``): the matcher
    moves settings their weights may not take as inputs, which they would answer
    at one step and refuse at the next, and the settings it finds are judged by
    the native elements."""
    from linac_gen.surrogates import registry
    with registry.suspended():
        return _compensate(lattice, beam_cfg, scenario, name_to_class, baseline_metrics, cfg)


def _compensate(lattice, beam_cfg, scenario, name_to_class, baseline_metrics,
                cfg: CompensationConfig) -> CompensationResult:
    work = copy.deepcopy(lattice)

    # 1. strip existing matcher cards so we only adjust OUR compensators.
    def _is_matcher_card(e):
        if not isinstance(e, LatticeCommand):
            return False
        kw = getattr(e, "KEYWORD", "")
        return kw.startswith("ADJUST") or kw in _CONSTRAINT_KW
    work.elements = [e for e in work.elements if not _is_matcher_card(e)]

    # 2. inject the failure.
    for sel, val in scenario.element_overrides(name_to_class):
        apply_element_override(work, sel, val)

    # 3. select compensators.  A failed bend sets a reference path no re-tune of the optics
    #    can bring the beam back onto: not compensable, and the matcher is not run.
    failed_names = list(scenario.element_names)
    bends = [n for n in failed_names if name_to_class.get(n) == "Dipole"]
    if bends:
        return CompensationResult(
            False, [],
            message=(f"not compensable by re-tuning optics: {', '.join(bends)} is a bend, which "
                     "sets the reference path the beam must follow"),
            metrics_after=_forward_metrics(work, beam_cfg, cfg.cost_solver))
    comp_names = select_zone(work, failed_names, cfg)
    if not comp_names:
        return CompensationResult(
            False, [], message="no compensators selected",
            metrics_after=_forward_metrics(work, beam_cfg, cfg.cost_solver))

    ncidx = _noncmd_index(work)
    comp_objs = {getattr(e, "name", None): e for e in work.elements
                 if getattr(e, "name", None) in set(comp_names)}

    # 4. inject Adjust cards (target by 1-based non-command index — robust to
    #    duplicate name prefixes; appending commands doesn't shift the count).
    j = 0
    nominals = family_nominals(work, cfg)
    for k, nm in enumerate(comp_names):
        elem = comp_objs[nm]
        ks = knobs(elem)
        if type(elem).__name__ in ("FieldMap", "FieldMap3D") and any(
                role == "amp" and attr == "ke" for (_p, attr, role) in ks) \
                and float(getattr(elem, "kb", 0.0) or 0.0) not in (0.0, float(elem.ke)):
            import warnings
            warnings.warn(f"compensator '{nm}': its kb differs from its ke, so re-tuning ke alone "
                          "changes the ratio of its RF electric and magnetic fields",
                          UserWarning, stacklevel=2)
        # a cavity map's ke and kb share one column, so its RF fields scale together
        link = _COMP_LINK_BASE + k if sum(role == "amp" for (_p, _a, role) in ks) > 1 else 0
        for (pidx, attr, role) in ks:
            cur = float(getattr(elem, attr))
            if role == "amp":
                vmin, vmax = amp_bounds(cfg, nm, cur, nominals)
            else:                                     # phase
                vmin = cur - cfg.phase_bounds_deg
                vmax = cur + cfg.phase_bounds_deg
            j += 1
            work.elements.append(Adjust(
                name=f"__COMP_{j}", target=str(ncidx[nm]), param_idx=pidx,
                link_group=link if role == "amp" else 0, vmin=vmin, vmax=vmax,
                start_step=abs(vmax - vmin) * 0.1))

    # 5. recovery objectives: exit energy (+ transmission in mp) at the
    #    lattice end; an optional size ceiling over the whole line leads.
    e0 = baseline_metrics.get("ref_w_kin")
    t0 = baseline_metrics.get("transmission")
    leading, trailing = objective_cards(cfg, baseline_metrics, len(ncidx))
    work.elements[0:0] = leading
    work.elements.extend(trailing)

    # 6. match.
    beam_copy = copy.deepcopy(beam_cfg)
    result = match(work, beam_copy, algorithm=cfg.algorithm,
                   cost_solver=cfg.cost_solver, space_charge=cfg.space_charge,
                   max_iter=cfg.max_iter)

    # 7. read matched compensator settings (match mutated `work` in place).
    settings = {}
    for nm in comp_names:
        elem = comp_objs[nm]
        for (_pidx, attr, _role) in knobs(elem):
            settings[f"{nm}.{attr}"] = float(getattr(elem, attr))

    # 8. impact after, and the recovered? verdict.
    metrics_after = _forward_metrics(work, beam_copy, cfg.cost_solver)
    e_after = metrics_after.get("ref_w_kin")
    t_after = metrics_after.get("transmission")
    energy_ok = (e0 is None or
                 (e_after is not None and abs(e_after - e0) <= cfg.recover_tol_energy_mev))
    trans_ok = (cfg.cost_solver != "mp" or t0 is None or t_after is None
                or t_after >= t0 - cfg.transmission_margin_pct)
    recovered = bool(result.success and energy_ok and trans_ok)

    return CompensationResult(
        recovered=recovered, compensator_names=comp_names, settings=settings,
        residual_cost=float(result.cost), metrics_after=metrics_after,
        match_success=bool(result.success), message=result.message)
