"""grad_sensitivities — ranked knob influence via torch autograd.

One reverse pass over the differentiable transfer-matrix path gives
d(exit KPI)/d(knob) for EVERY tunable at once — "which knob most
affects σ_y at the exit?" answered exactly, not by scanning.

Scope honesty (the zero-current envelope of a magnet line):
- tunables are quad gradient / solenoid field / dipole angle — the
  same three the gradient matcher supports;
- the walk takes every element a magnet line carries: the knobs through
  the differentiable torch matrices, elements with no knob (markers,
  apertures, compensation and grid cards, steerers, thin lenses,
  multipoles, matrix elements) as the constant maps the envelope and
  matrix modes use, a foil as the envelope takes it (its scattering
  added to Σ, its mean energy loss taken off the reference), a matrix
  element's reference gain;
- anything else is REFUSED by class name: field maps, RF gaps,
  multi-gap cavities and RFQ cells, which the walk does not model, and
  any class it does not know, a subclass of a magnet among them;
  FREQ and the HELIX energy card (; HELIX_SET_ENERGY) are followed, and
  a card, foil or matrix element that leaves the reference with no
  energy is refused;
- space charge is OFF, centroid effects are not modeled, and the
  result is d/d(design value) with field errors folded downstream.
"""
from __future__ import annotations

from linac_gen.assist.tools import (
    WorkContext, _capture, _ctx_provenance, _need, _ok, _refused, _tool,
)


def _types():
    """(the magnets with a differentiable torch matrix, with the command cards; the classes with
    a constant map; Foil; the magnets alone)."""
    from linac_gen.elements.base import PassiveElement
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.edge import Edge
    from linac_gen.elements.foil import Foil
    from linac_gen.elements.lattice_commands import LatticeCommand
    from linac_gen.elements.matrix_element import MatrixElement
    from linac_gen.elements.multipole import Multipole
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.elements.steerer import Steerer
    from linac_gen.elements.thin_lens import ThinLens
    magnets = (Drift, Quadrupole, Solenoid, Dipole, Edge)
    constant = (PassiveElement, Steerer, ThinLens, Multipole, MatrixElement)
    return magnets + (LatticeCommand,), constant, Foil, magnets


def _known(e, magnets, accepted) -> bool:
    """A magnet by its exact class only: the torch matrix of a subclass would be its base
    class's, whatever map the subclass has.  Anything else by family."""
    if isinstance(e, magnets):
        return type(e) in magnets
    return isinstance(e, accepted)


def _unsupported(elements) -> dict:
    """The classes in ``elements`` the walk cannot represent, with counts: an allowlist, so a
    class it does not know, a subclass of a magnet among them, is refused by name rather than
    met as an error or a wrong map."""
    differentiable, constant, Foil, magnets = _types()
    accepted = differentiable + constant + (Foil,)
    bad: dict[str, int] = {}
    for e in elements:
        if not _known(e, magnets, accepted):
            nm = type(e).__name__
            bad[nm] = bad.get(nm, 0) + 1
    return bad


def _probe_sigma(lattice, ref, sigma_in, last, overrides):
    """Σ after element ``last`` (0-based, inclusive): the zero-current envelope, differentiable in
    the knobs.  The map is composed segment by segment as ``compute_transfer_matrix_torch``
    composes it, and Σ ← M·Σ·Mᵀ is taken at each foil and at the probe, so a line of the types
    that function takes (no foil, no constant map but the identity) gives its answer to the last
    bit."""
    import torch

    from linac_gen.elements.base import PassiveElement
    from linac_gen.elements.lattice_commands import REFERENCE_COMMANDS, Freq
    from linac_gen.elements.multipole import Multipole
    from linac_gen.tracking.matrix_tracking import get_element_matrix
    from linac_gen.tracking.torch_matrices import F64, RefKinematics
    from linac_gen.tracking.torch_tracking import (
        apply_reference_card,
        element_matrix_torch,
        tilt_rotation_matrix_torch,
    )
    differentiable, constant, Foil, magnets = _types()
    accepted = differentiable + constant + (Foil,)
    clock = ref.copy()
    kin = RefKinematics.from_reference(clock)
    S, M = sigma_in, torch.eye(6, dtype=F64)
    for element in lattice.elements[:last + 1]:
        if not _known(element, magnets, accepted):              # never a silent identity
            raise ValueError(f"grad_sensitivities cannot represent {type(element).__name__} "
                             f"'{getattr(element, 'name', '?')}'")
        if isinstance(element, REFERENCE_COMMANDS):
            if isinstance(element, Freq):
                M = torch.as_tensor(element.transfer_matrix(clock), dtype=F64) @ M
            apply_reference_card(element, clock)
            kin = RefKinematics.from_reference(clock)
            continue
        if isinstance(element, Foil):
            # zero-mean scattering: Σ ← Σ + D, and the reference loses the mean loss, as the
            # envelope takes a foil
            D, dE = element.envelope_diffusion(clock)
            S = M @ S @ M.T + torch.as_tensor(D, dtype=F64)
            M = torch.eye(6, dtype=F64)
            if dE > 0.0:
                if not clock.w_kin - dE > 0.0:
                    raise ValueError(f"foil '{element.name}' leaves the reference with no energy "
                                     f"(mean loss {dE:g} MeV, W = {clock.w_kin:g} MeV)")
                clock.w_kin = clock.w_kin - dE
                kin = RefKinematics.from_reference(clock)
            continue
        if isinstance(element, differentiable):
            M_elem = element_matrix_torch(element, kin, overrides=overrides,
                                          on_nonlinear="error")
        else:
            # no knob: the map the envelope and matrix modes use at this reference, a constant
            M_elem = torch.as_tensor(get_element_matrix(element, clock), dtype=F64)
        tilt = (0.0 if isinstance(element, (PassiveElement, Multipole))
                else getattr(element, "tilt_deg", 0.0))
        if abs(float(tilt)) > 1e-12:
            M_elem = (tilt_rotation_matrix_torch(-tilt) @ M_elem
                      @ tilt_rotation_matrix_torch(tilt))
        M = M_elem @ M
        gain = float(getattr(element, "ref_gain", 0.0) or 0.0)    # a matrix element's gain
        if gain:
            if not clock.w_kin + gain > 0.0:
                raise ValueError(f"'{element.name}' leaves the reference with no energy "
                                 f"(W = {clock.w_kin + gain:g} MeV)")
            clock.w_kin = clock.w_kin + gain
            kin = RefKinematics.from_reference(clock)
    return M @ S @ M.T


_KPI_INDEX = {"sigma_x": 0, "sigma_y": 2, "sigma_phi": 4}
_KPI_UNIT = {"sigma_x": "mm", "sigma_y": "mm", "sigma_phi": "deg"}
_KIND_NAMES = {"quad": "Quadrupole", "solenoid": "Solenoid",
               "dipole": "Dipole"}


@_tool(
    "grad_sensitivities",
    "Ranked knob sensitivities via ONE torch autograd pass: exact "
    "d(exit sigma_x/sigma_y/sigma_phi)/d(knob) for every quad gradient, "
    "solenoid field and dipole angle at once — which knob most affects "
    "the KPI, with sign.  Magnet lines (refuses field maps, RF gaps, "
    "multi-gap cavities and RFQ cells); space charge off; tilts, foils "
    "and matrix elements included, no centroid.",
    {"type": "object",
     "properties": {
         "kpi": {"type": "string",
                 "enum": ["sigma_x", "sigma_y", "sigma_phi"],
                 "description": "exit quantity to differentiate"},
         "kinds": {"type": "array",
                   "items": {"type": "string",
                             "enum": ["quad", "solenoid", "dipole"]},
                   "description": "knob families (default: all three)"},
         "top_n": {"type": "integer",
                   "description": "how many ranked knobs to return "
                                  "(default 10)"},
         "at_index": {"type": "integer",
                      "description": "probe after this element (0-based "
                                     "index, the element included) instead "
                                     "of the lattice end"}},
     "required": []},
    "compute")
def _grad_sensitivities(ctx: WorkContext, kpi: str = "sigma_y",
                        kinds=None, top_n: int = 10, at_index=None,
                        _assist_prov=None, **_ignored):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    if kpi not in _KPI_INDEX:
        return _refused(f"unknown kpi {kpi!r} — one of "
                        f"{sorted(_KPI_INDEX)}")
    try:
        import torch
    except Exception:                                       # noqa: BLE001
        return _refused("PyTorch is not available in this environment — "
                        "grad_sensitivities needs the differentiable "
                        "path; use parameter_scan instead")
    from linac_gen.distributions.factory import (
        create_beam, geometric_emittances, input_dispersion,
    )
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.tracking.envelope import _build_sigma_matrix
    from linac_gen.tracking.torch_matrices import F64

    elements = ctx.lattice.elements
    if not elements:
        return _refused("the lattice has no elements")
    # the probe sits after element ``last`` (0-based), which is part of the span
    last = len(elements) - 1 if at_index is None else int(at_index)
    if not (0 <= last < len(elements)):
        return _refused(f"at_index {at_index} outside 0..{len(elements) - 1} (0-based: "
                        "the probe sits after that element)")
    span = elements[:last + 1]

    bad = _unsupported(span)
    if bad:
        listing = ", ".join(f"{n}×{c}" for n, c in sorted(bad.items()))
        return _refused(
            "this lattice span is not faithfully differentiable — the matrix path cannot "
            f"represent: {listing}.  Gradient sensitivities cover magnet lines (drifts, "
            "quadrupoles, solenoids, bends, pole faces, thin lenses, multipoles, correctors, "
            "foils, matrix elements, markers, apertures and command cards); use parameter_scan "
            "for anything with field maps, RF gaps, multi-gap cavities or RFQ cells, which it "
            "does not model.")

    want = {_KIND_NAMES[k] for k in (kinds or _KIND_NAMES)}
    attr_of = {Quadrupole: "gradient", Solenoid: "field", Dipole: "angle"}
    unit_of = {Quadrupole: "T/m", Solenoid: "T", Dipole: "deg"}
    knobs, rows, s_mm = [], [], 0.0
    overrides = {}
    for i, e in enumerate(span):
        s_mm += float(getattr(e, "length", 0.0) or 0.0)
        for cls, attr in attr_of.items():
            if isinstance(e, cls) and cls.__name__ in want:
                t = torch.tensor(float(getattr(e, attr)), dtype=F64,
                                 requires_grad=True)
                overrides[id(e)] = t
                knobs.append(t)
                rows.append({"index": i,
                             "name": getattr(e, "name", type(e).__name__),
                             "kind": cls.__name__.lower(),
                             "param": attr,
                             "value": float(getattr(e, attr)),
                             "s_m": round(s_mm / 1000.0, 4),
                             "knob_unit": unit_of[cls]})
                break
    if not knobs:
        return _refused("no tunable elements (quad/solenoid/dipole) in "
                        "the requested span")

    cfg = ctx.beam_config

    def _forward():
        ref = create_beam(cfg, seed=42).ref
        ex, ey, ez = geometric_emittances(cfg, max(float(ref.bg), 1e-9))
        az, bz = cfg.alpha_z, cfg.beta_z
        if bool(getattr(cfg, "continuous", False)):
            # a continuous beam has no bunch: the envelope seeds no longitudinal block,
            # whatever bunch settings the config still holds
            az, bz, ez = 0.0, 1.0, 0.0
        sigma_in = torch.as_tensor(_build_sigma_matrix(
            cfg.alpha_x, cfg.beta_x, ex,
            cfg.alpha_y, cfg.beta_y, ey,
            az, bz, ez, **input_dispersion(cfg)), dtype=F64)
        return _probe_sigma(ctx.lattice, ref, sigma_in, last, overrides)

    sigma_out, refusal, warns = _capture(_forward)
    if refusal:
        return refusal
    d = sigma_out[_KPI_INDEX[kpi], _KPI_INDEX[kpi]]
    if float(d.detach()) <= 0.0:
        return _refused(f"degenerate beam: Σ[{kpi}] collapsed to zero "
                        "at the probe point — sensitivities undefined")
    val = torch.sqrt(d)
    grads = torch.autograd.grad(val, knobs, allow_unused=True)
    for row, g in zip(rows, grads):
        row["sens"] = 0.0 if g is None else float(g)
        row["sens_unit"] = f"{_KPI_UNIT[kpi]} per {row.pop('knob_unit')}"
    rows.sort(key=lambda r: abs(r["sens"]), reverse=True)
    n = max(1, min(int(top_n), len(rows)))
    return _ok({
        "kpi": kpi,
        "kpi_value": round(float(val.detach()), 6),
        "kpi_unit": _KPI_UNIT[kpi],
        "probe_after_index": last,
        "n_knobs": len(rows),
        "ranked": rows[:n],
        "note": ("one autograd reverse pass; zero-current envelope — space "
                 "charge OFF, centroid effects absent, d/d(design value)"),
    }, _ctx_provenance(ctx), warns)
