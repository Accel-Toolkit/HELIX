# RfqCell

A single RFQ (Radio-Frequency Quadrupole) cell — the building block
of an RFQ buncher/accelerator that turns a continuous DC ion beam
into a bunched, accelerated beam suitable for the downstream linac.
Implements the Crandall 2-term potential expansion (M1 in HELIX
nomenclature).

## TL;DR

| | TraceWin / Toutatis | HELIX |
|---|---|---|
| Keyword | `RFQ_CELL` (Toutatis-format `.dat`) | `RfqCell(...)` |
| Voltage | inter-vane voltage V (V) | same `voltage_V` |
| A₁₀ | Crandall 2-term coefficient | same |
| Modulation m | vane modulation factor (m ≥ 1) | same |
| φ_s | synchronous phase (deg) | `phi_s_deg` |
| cell_type | ±2 / ±3 / ±4 | same |

Conventions:

* Voltage is in **volts** (not MV) — typical RFQs run 50-150 kV.
* `cell_type`: ±2 = accelerating, ±3 = front-end / shaper,
  ±4 = transcell.  The sign selects which neighbour cell the
  transverse model couples to.
* Internally split into N substeps (auto-picked, ≤0.1 mm/substep,
  ≥20 substeps).  See [VaneRFQ](10_vanerfq.md) for multi-cell.

## Tutorial

An RFQ cell uses four electrodes (vanes) with a sinusoidally
modulated tip profile to produce a transverse focusing potential
that simultaneously bunches and accelerates.

The Crandall 2-term potential is

$$
U(r, \theta, z) = \frac{V}{2}\left[
A_{01}\left(\frac{r}{r_0}\right)^2 \cos(2\theta)
+ A_{10} I_0(kr) \cos(kz)
\right]
$$

where r₀ is the average radius, k = π/L (cell wavenumber), I₀ is
the modified Bessel function, A₀₁ ≈ 1 (transverse focusing),
A₁₀ encodes modulation strength (longitudinal acceleration).

### Transfer matrix and per-substep integration

Like a FieldMap, an RFQ cell has **no closed-form 6×6 transfer
matrix** — the 2-term potential mixes transverse position with z
non-linearly.  The production integrator in this codebase is the
**2-term Strang Drift–Kick–Drift** splitting: each substep applies
a half-drift, the full AG transverse + longitudinal kick, then the
second half-drift (TraceWin manual convention).  The default
`field_model="2term"` — `A_quad = (1−A₁₀)/R₀²` with
`S = −sign(Type)` — is the path validated against TraceWin envelope
output; three diagnostic-only variants (`crand_x`,
`crand_x_noflip`, `pdf_2term`) are kept for comparison but blow up
in envelope runs and must not be used for production.

A **Boris time-stepper + Hybrid field source** exploration exists
in the separate `rfqtrack` subproject — it is *not* the integrator
behind `RfqCell` in `linac_gen`.

For envelope tracking, the per-substep 6×6 Jacobian is built by
finite differencing the pusher and chained over the cell.

| Block | Effect |
|---|---|
| (x, x', y, y') 4×4 | strong AG focusing from sin(2θ) potential term |
| (Δφ, ΔW) 2×2 | bunching + acceleration from cos(kz) modulation |
| Cross blocks | (x, Δφ) and (y, Δφ) **non-zero** — RFQ deliberately couples transverse and longitudinal |

The transverse-longitudinal coupling is intrinsic to the RFQ — it
is what bunches a continuous beam.  No purely-transverse subspace
is invariant.

### Validation status

The numbers below come from the **rfqtrack subproject's**
Boris time-stepper + Hybrid field source exploration on PXIE NOSC
(continuous beam, no SC) — not from the `linac_gen` `RfqCell`
integrator documented on this page:

| Quantity | Ratio (rfqtrack Boris+Hybrid) / TW |
|---|---|
| σ_x | 0.83 |
| σ_y | 0.88 |
| W (energy) | 0.94 |
| Transmission | 92 % |

In that study the 2-term Strang-splitting integrator showed σ_y
blow-up by 3.1× at the M1 ceiling — the Boris/Hybrid path closed
that gap in rfqtrack.  Within `linac_gen`, the 2-term Strang DKD
remains the production path (σ_x within ~30 % of the TraceWin
reference at the whole-RFQ level, see [VaneRFQ](10_vanerfq.md)).

Higher-fidelity variants (M3-family laplace2d/3d/8-term) hit
structural blockers; M1 is the production path.

## API reference

```{.python .skip}
linac_gen.elements.rfq_cell.RfqCell(
    name: str,
    voltage_V: float,             # inter-vane V (volts)
    r0_mm: float,                 # vane mean radius R₀ (mm)
    A10: float,                   # Crandall 2-term coefficient
    modulation: float,            # m
    length_mm: float,             # cell length L (mm)
    phi_s_deg: float,             # synchronous phase (deg)
    cell_type: int,               # ±2 / ±3 / ±4
    Tc_mm: float = 0.0,
    dP_deg: float = 0.0,
    n_steps: int | None = None,   # auto-picked if None
    type_prev: int | None = None,
    type_next: int | None = None,
    A_quad: float | None = None,
    aperture: float = 0.0,
    field_model: str = "2term",
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `voltage_V` | (required) | V | inter-vane voltage (typical 50-150 kV) |
| `r0_mm` | (required) | mm | vane mean radius R₀ |
| `A10` | (required) | — | Crandall 2-term acceleration coefficient |
| `modulation` | (required) | — | tip modulation m (≥ 1); feeds the 2-term `A_quad` default |
| `length_mm` | (required) | mm | cell length L (no βλ/2 check enforced) |
| `phi_s_deg` | (required) | deg | synchronous phase |
| `cell_type` | (required) | int | ±2 = accelerating, ±3 = front-end / shaper, ±4 = transcell |
| `Tc_mm` | 0.0 | mm | transverse curvature (TraceWin `Tc`) — accepted for parser compatibility, currently unused |
| `dP_deg` | 0.0 | deg | output-phase shift (TraceWin `dP`), applied at cell exit |
| `n_steps` | None | — | auto-picked: max(20, ceil(L/0.1 mm)) when None |
| `type_prev`, `type_next` | None | int | neighbouring cell types (S = −sign(type[n±1])); default to `cell_type` |
| `A_quad` | None | 1/mm² | DC quadrupole coefficient override; None → `(1 − A₁₀)/R₀²` |
| `aperture` | 0.0 | mm | loss-tracking aperture radius; 0 = no check |
| `field_model` | `"2term"` | — | production `2term`; diagnostic-only `crand_x`, `crand_x_noflip`, `pdf_2term` |

### Source

* `linac_gen/elements/rfq_cell.py:135` (constructor)

## See also

* [VaneRFQ](10_vanerfq.md) — `.vane` file wrapper.
* LEBT + RFQ worked example.
* `Tracewin_code/Toutatis_*.pdf` — RFQ-design code reference.

← [FieldMap3D](08_fieldmap3d.md) ·
[Continue to VaneRFQ →](10_vanerfq.md)
