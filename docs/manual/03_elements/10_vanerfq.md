# VaneRFQ

A `VaneRFQ` represents an entire RFQ as **one element**: a single
`FieldMapElement` whose per-z geometry (apertures, voltages, `Tc`)
comes from a `.vane` file (Toutatis convention) and whose cell-level
parameters live in an ordered list of `CellSpan` dataclasses — one
per RFQ_CELL line.  It does *not* instantiate an `RfqCell` per row;
the whole RFQ is tracked as one substepped field-map element.

## TL;DR

```python
from linac_gen.elements.vane_rfq import VaneRFQ
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.vane_rfq_helper import replace_rfq_cells_with_vane

# Usual route: parse a .dat with RFQ_CELL lines, then swap the
# contiguous RfqCell chain for one VaneRFQ driven by the .vane file.
lattice, _ = parse_tracewin("examples/lebt_plus_rfq/lebt_plus_rfq_user.dat")
replace_rfq_cells_with_vane(lattice, "Fields/pxie-rfq.vane")
```

`VaneRFQ` is **not** built by the default parser path — the parser
produces a chain of [RfqCell](09_rfqcell.md) elements.  Construct a
`VaneRFQ` manually (with a parsed `VaneGeometry` + `CellSpan` list),
or use `replace_rfq_cells_with_vane` to swap a parsed cell chain
in-place.

## Tutorial

The PXIE RFQ — used in PIP-II's H⁻ injector — has hundreds of cells
over 4.45 m, accelerating from 30 keV to 2.1 MeV.  Rather than
tracking each cell with its cell-constant `r₀`, the `VaneRFQ`
samples `r₀(z)`, voltages, and `Tc` from the `.vane` data per
substep while taking the Crandall coefficients from the cell-level
RFQ_CELL parameters.

### Field models

One class, five `field_model` options — all reading the same `.vane`
geometry but using progressively higher-fidelity field models:

| `field_model` | Field model | Status |
|---|---|---|
| `"2term"` | 2-term Crandall, cell-constant Wangler short form (M1) | **production**; σ_x within ~30 % of the TraceWin reference |
| `"8term"` | matcher-aware per-z `r₀` from `.vane` (M2) | diagnostic |
| `"8term_full"` | full 8-term Crandall analytic expansion (M3.3) | *unstable* (AG resonance) |
| `"laplace2d"` | per-slice 2-D numerical Laplace (M3) | σ_x converges; σ_y blows up |
| `"laplace3d"` | full 3-D Laplace, Shortley-Weller boundary (M3.2) | research-grade; σ diverges either way |

The default is `"2term"` — it is the only stable production path
within the current implementation budget.  The others are kept as
diagnostic/reference paths that document specific failure modes.

### Example

```python
# `lattice` was parsed (and its RfqCell chain replaced) in the TL;DR
# above — locate the VaneRFQ element it produced:
rfq = next(e for e in lattice.elements if isinstance(e, VaneRFQ))
print(f"VaneRFQ spans {len(rfq.cells)} cells, "
      f"total {rfq.length:.0f} mm long.")
```

### Transfer matrix

The `VaneRFQ` follows the same per-substep integration as
[RfqCell → Transfer matrix](09_rfqcell.md#transfer-matrix-and-per-substep-integration),
with `A_quad = (1−A₁₀)/r₀(z)²` re-evaluated each substep from the
`.vane` interpolation.  The element-level matrix is the chain of
per-substep 6×6 Jacobians and includes strong (x ↔ Δφ), (y ↔ Δφ)
couplings (the RFQ's whole job is to bunch the beam by coupling
transverse and longitudinal).

## API reference

```{.python .skip}
linac_gen.elements.vane_rfq.VaneRFQ(
    name: str,
    vane: VaneGeometry,            # parsed .vane contents
    cells: list[CellSpan],         # ordered, contiguous cell list
    n_steps: int | None = None,    # default max(2·slices_in_span, 1000)
    aperture: float = 0.0,         # mm; 0 = no check
    field_model: str = "2term",
    laplace_cache=None,
    laplace_kwargs: dict | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `name` | (required) | identifier |
| `vane` | (required) | `VaneGeometry` from `linac_gen.io.tracewin_vane` — provides `r₀(z)`, voltages, `Tc` |
| `cells` | (required) | ordered `list[CellSpan]`; must be contiguous and start at z = 0 of the element |
| `n_steps` | None | total substeps over the whole element; None → `max(2·slices_in_span, 1000)` (~half a vane slice per substep) |
| `aperture` | 0.0 | loss-tracking aperture (mm); 0 disables |
| `field_model` | `"2term"` | one of `2term`, `8term`, `8term_full`, `laplace2d`, `laplace3d` |
| `laplace_cache` | None | pre-built Laplace cache (laplace2d/3d only) — share across elements |
| `laplace_kwargs` | None | kwargs forwarded to the Laplace cache constructor |

Each `CellSpan` dataclass carries one RFQ_CELL line plus its z
extent: `z_start_mm`, `z_end_mm`, `voltage_V`, `A10`, `modulation`,
`length_mm`, `phi_s_deg`, `cell_type`, `type_prev`, `type_next`,
`r0_dat_mm` (default 5.0), `Tc_mm` (default 0.0), `dP_deg`
(default 0.0).  All lengths in mm, voltages in volts, phases in
degrees.

### Source

* `linac_gen/elements/vane_rfq.py:135` (constructor; `CellSpan` at `:40`)
* `linac_gen/io/vane_rfq_helper.py` (`replace_rfq_cells_with_vane`)
* `linac_gen/io/tracewin_vane.py` (`VaneGeometry` parser)

## See also

* [RfqCell](09_rfqcell.md) — the per-cell element the parser builds.
* LEBT + RFQ worked example.

← [RfqCell](09_rfqcell.md) ·
[Continue to Multipole →](11_multipole.md)
