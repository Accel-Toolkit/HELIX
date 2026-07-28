"""grad_sensitivities — ranked knob influence via torch autograd.

One reverse pass over the differentiable transfer-matrix path gives
d(exit KPI)/d(knob) for EVERY tunable at once — "which knob most
affects σ_y at the exit?" answered exactly, not by scanning.

Scope honesty (the differentiable path is linear-matrix only):
- tunables are quad gradient / solenoid field / dipole angle — the
  same three the gradient matcher supports;
- decks containing elements the torch path would silently reduce to
  identity (field maps, RF gaps, multi-gap cavities) are REFUSED, as
  are FREQ / SET_BEAM_ENERGY / SET_BEAM_E0_P0 cards (they change
  downstream physics the matrix span cannot see);
- space charge is OFF, centroid effects are not modeled, and the
  result is d/d(design value) with field errors folded downstream.
"""
from __future__ import annotations

from linac_gen.assist.tools import (
    WorkContext, _capture, _ctx_provenance, _need, _ok, _refused, _tool,
)

#: element-class-name fragments the torch matrix path cannot represent
#: (it would silently emit identity — wrong sensitivities, so: refuse)
_NONLINEAR_FRAGMENTS = ("FieldMap", "RFGap", "NCells", "Ncells", "RFQ")


def _scan_unsupported(elements) -> dict:
    from linac_gen.elements.lattice_commands import (
        Freq, SetBeamE0P0, SetBeamEnergy,
    )
    bad: dict[str, int] = {}
    for e in elements:
        nm = type(e).__name__
        if any(f in nm for f in _NONLINEAR_FRAGMENTS) or isinstance(
                e, (Freq, SetBeamE0P0, SetBeamEnergy)):
            bad[nm] = bad.get(nm, 0) + 1
    return bad


_KPI_INDEX = {"sigma_x": 0, "sigma_y": 2, "sigma_phi": 4}
_KPI_UNIT = {"sigma_x": "mm", "sigma_y": "mm", "sigma_phi": "deg"}
_KIND_NAMES = {"quad": "Quadrupole", "solenoid": "Solenoid",
               "dipole": "Dipole"}


@_tool(
    "grad_sensitivities",
    "Ranked knob sensitivities via ONE torch autograd pass: exact "
    "d(exit sigma_x/sigma_y/sigma_phi)/d(knob) for every quad gradient, "
    "solenoid field and dipole angle at once — which knob most affects "
    "the KPI, with sign.  Linear decks only (refuses field-map/RF-gap "
    "lattices); space charge off; matrix path with tilts, no centroid.",
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
                      "description": "probe after this element index "
                                     "instead of the lattice end"}},
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
        create_beam, geometric_emittances,
    )
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.tracking.envelope import _build_sigma_matrix
    from linac_gen.tracking.torch_matrices import F64
    from linac_gen.tracking.torch_tracking import (
        compute_transfer_matrix_torch,
    )

    elements = ctx.lattice.elements
    end = len(elements) if at_index is None else int(at_index)
    if not (0 < end <= len(elements)):
        return _refused(f"at_index {at_index} outside 1..{len(elements)}")
    span = elements[:end]

    bad = _scan_unsupported(span)
    if bad:
        listing = ", ".join(f"{n}×{c}" for n, c in sorted(bad.items()))
        return _refused(
            "this lattice span is not faithfully differentiable — the "
            f"matrix path cannot represent: {listing}.  Gradient "
            "sensitivities are valid only for linear decks (e.g. the "
            "BTL); use parameter_scan for anything with field maps or "
            "RF gaps.")

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
        sigma_in = torch.as_tensor(_build_sigma_matrix(
            cfg.alpha_x, cfg.beta_x, ex,
            cfg.alpha_y, cfg.beta_y, ey,
            cfg.alpha_z, cfg.beta_z, ez), dtype=F64)
        M = compute_transfer_matrix_torch(
            ctx.lattice, ref, start=0, end=end,
            overrides=overrides, on_nonlinear="error")
        sigma_out = M @ sigma_in @ M.T
        return sigma_out

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
        "kpi_value": round(float(val), 6),
        "kpi_unit": _KPI_UNIT[kpi],
        "probe_after_index": end - 1,
        "n_knobs": len(rows),
        "ranked": rows[:n],
        "note": ("one autograd reverse pass; matrix path — space charge "
                 "OFF, centroid effects absent, d/d(design value)"),
    }, _ctx_provenance(ctx), warns)
