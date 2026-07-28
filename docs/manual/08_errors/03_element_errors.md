# Element-level errors

Random per-seed perturbations to individual elements: alignment
(dx, dy, dz, tilt) and field (gradient_rel, voltage_rel, etc.).
This is what most tolerance studies are about.

## TL;DR

Two categories:

* **Alignment errors** — the element moves or rotates relative to
  the ideal beam axis.  Honoured by the tracker as pre/post coordinate
  shifts and rotations.
* **Field errors** — the element's *field* deviates from the design
  value (e.g. quad gradient is 0.5% high; cavity voltage is 1% low).
  Folded into the element's `effective_X` properties.

Each error is specified as an `ErrorDef`:

```python
from linac_gen.errors.error_model import ErrorDef

err = ErrorDef(
    pattern="QUAD_*",       # which elements (fnmatch wildcard)
    parameter="dx",         # which parameter
    distribution="gaussian", # gaussian or uniform
    sigma=0.2,              # σ for gaussian
    half_width=0.0,         # half-width for uniform
    cutoff=3.0,             # σ truncation for gaussian
)
```

## Alignment parameters

| Parameter | Units | Effect |
|---|---|---|
| `dx`, `dy` | mm | transverse translation (honoured by the tracker) |
| `tilt_deg` | deg | rotation about z (longitudinal) — honoured |
| `dz` | mm | longitudinal translation — **stored but ignored by the tracker** |
| `pitch_deg`, `yaw_deg` | deg | reserved (not yet honoured) |

The tracker applies these as:

1. **Pre-element**: shift particle (x, y) by (-dx, -dy) (move to
   element frame), rotate by -tilt_deg.
2. **Element body**: standard transfer matrix / RK4.
3. **Post-element**: rotate by +tilt_deg, shift by +(dx, dy)
   (move back to lab frame).

Net: a misaligned element kicks the centroid and rotates the σ
matrix in the (x, y) projection.

## Field parameters

| Parameter | Element types | Effect |
|---|---|---|
| `gradient_rel` | Quadrupole | `G → G·(1 + δ)` |
| `field_rel` | Solenoid, Dipole | `B → B·(1 + δ)` (Dipole via `effective_angle`) |
| `voltage_rel` | RFGap, FieldMap, FieldMap3D | `V → V·(1 + δ)` (field maps via `effective_ke/kb`) |
| `phase_offset` | RFGap, FieldMap | `phase → phase + phase_offset` (deg) |
| `frequency_offset` | RFGap, FieldMap | additive Hz / MHz (rarely used) |
| `g3_rel..g6_rel` | Quadrupole | fractional higher-pole content errors |

!!! note "FieldMap amplitude and Dipole field errors (fixed 2026-07)"
    These two used to be silent no-ops (the engine looked for a
    `voltage`/`field` design attribute those elements don't have).
    The draw now lands on the element's own `_rel` perturbation slot,
    which tracking consumes (`effective_ke/kb`, `effective_angle`).
    A `*_rel` error that matches an element with *neither* slot now
    warns once instead of silently vanishing.  Pre-fix tolerance
    studies with field-map amplitude errors should be re-run.

These are *static* per-seed offsets — drawn once at seed start,
held constant through the simulation.  Per-step time-varying
jitter is not modelled (TraceWin's `ERROR_*_DYN` cards are absorbed
as static errors — see [ERROR_* directives](02_error_directives.md)).

## Wildcard patterns

The `pattern` field is fnmatch:

| Pattern | Matches |
|---|---|
| `QUAD_*` | every element whose name starts with `QUAD_` |
| `QF_03` | exactly element `QF_03` |
| `*Cav*` | every element with "Cav" in the name |
| `*` | every element |

Patterns are case-sensitive.  Use multiple ErrorDefs with different
patterns to apply different distributions to different element classes.

## Examples

### All quads aligned to 0.2 mm rms

(The fragments below assume a `study` — here built on a small demo
lattice with quads named `QUAD_01` … `QUAD_06`.)

```python
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.errors.error_model import ErrorStudy

lat = Lattice()
for i in range(1, 7):
    lat.add(Quadrupole(name=f"QUAD_{i:02d}", length=100.0,
                       gradient=10.0 * (-1) ** i))
    lat.add(Drift(name=f"D_{i}", length=150.0))
study = ErrorStudy(lat, BeamConfig(n_particles=500), n_seeds=2)

study.add_error("QUAD_*", "dx", distribution="gaussian", sigma=0.2)
study.add_error("QUAD_*", "dy", distribution="gaussian", sigma=0.2)
```

### One specific quad heavily mispowered

```python
study.add_error("QUAD_05", "gradient_rel",
                distribution="uniform", half_width=0.05)
```

### Cavity LLRF jitter

```python
study.add_error("CAV_*", "voltage_rel",
                distribution="gaussian", sigma=0.005)   # 0.5% rms
study.add_error("CAV_*", "phase_offset",
                distribution="gaussian", sigma=1.0)     # 1° rms
```

### From .dat (auto-imported by parse_tracewin)

```
ERROR_GAUSSIAN_CUT_OFF 3
ERROR_QUAD_NCPL_STAT 6 2 0.2 0.2 0 0 0 0.5 0 0 0 0
ERROR_CAV_NCPL_STAT  2 2 0.3 0.3 0 0 1.0 1.0 0
```

## Cross-references

* [Beam errors](04_beam_errors.md) — input-beam jitter.
* [Running studies](05_running_studies.md).
* [Worked example: tolerance study](../11_examples/08_tolerance_study.md).

← [ERROR_* directives](02_error_directives.md) ·
[Continue to Beam errors →](04_beam_errors.md)
