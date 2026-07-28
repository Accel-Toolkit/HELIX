# Error studies overview

An error study is a Monte-Carlo ensemble: many simulations of the
same lattice with random perturbations sampled from realistic error
distributions.  The output is statistics: average σ, σ-band envelopes,
transmission histogram, worst-case excursion.  Used to validate that
a lattice is *robust* — not just optimal for the design beam.

## TL;DR

```python
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.errors.error_model import ErrorStudy, ErrorDef

# Small demo lattice — swap in your own parse_tracewin(...) lattice.
lattice = Lattice()
for i in range(1, 5):
    lattice.add(Quadrupole(name=f"QUAD_{i:02d}", length=100.0,
                           gradient=10.0 * (-1) ** i))
    lattice.add(Drift(name=f"D_{i}", length=150.0))
beam_config = BeamConfig(n_particles=500)

study = ErrorStudy(lattice, beam_config, n_seeds=2)   # production: 100+
study.add_error(pattern="QUAD_*", parameter="dx",
                distribution="gaussian", sigma=0.2, cutoff=3.0)
study.add_error(pattern="QUAD_*", parameter="dy",
                distribution="gaussian", sigma=0.2, cutoff=3.0)
results = study.run()

print(results.transmission_stats())        # {"mean", "min", "max", "std"}
sigma_x_mean = results.mean("sigma_x")     # 1-D array vs element index
sigma_x_band = results.std("sigma_x")
```

Or, if your `.dat` already has `ERROR_*` directives (e.g.
`ERROR_QUAD_NCPL_STAT`), they're auto-absorbed by `parse_tracewin`
and a bare `study.run()` works:

```python
from linac_gen.io.tracewin_parser import parse_tracewin

lattice, _ = parse_tracewin(
    "examples/error_studies/combined_realistic/combined_realistic.dat")
study = ErrorStudy(lattice, beam_config, n_seeds=2)   # production: 100+
results = study.run()
```

## What's parameterised

Every element supports two error-injection mixin families:

### Element alignment errors

| Parameter | Units | Element types |
|---|---|---|
| `dx`, `dy` | mm | every element |
| `tilt_deg` | deg | every element (rotation about z) |
| `dz` | mm | stored on the element but **ignored by the tracker** |
| `pitch_deg`, `yaw_deg` | deg | reserved (not yet honoured) |

The tracker honours only `dx`, `dy` and `tilt_deg`.  `dz` draws are
applied to the element attribute and carried through the study, but
no longitudinal shift is performed during tracking.

### Element field errors

| Parameter | Units | Element types |
|---|---|---|
| `gradient_rel` | fractional | Quadrupole (also `g3_rel..g6_rel`) |
| `field_rel` | fractional | Solenoid |
| `voltage_rel` | fractional | RFGap |
| `phase_offset` | deg | RFGap, FieldMap |
| `frequency_offset` | MHz | RFGap, FieldMap |

!!! note "FieldMap amplitude and Dipole field errors (fixed 2026-07)"
    `*_rel` draws are applied by scaling the matching design attribute
    (`gradient_rel` → `gradient`, etc.); for elements that carry the
    *relative-perturbation slot itself* — `FieldMap`/`FieldMap3D`
    (`voltage_rel`, consumed via `effective_ke/kb`) and `Dipole`
    (`field_rel`, via `effective_angle`) — the draw now lands on that
    slot directly.  Earlier releases silently **dropped** these two
    (cavity-amplitude errors on field maps and `ERROR_BEND` dg) —
    re-run any pre-fix tolerance study that relied on them.  An error
    that can land nowhere now emits a warning instead of vanishing.

### Beam input errors

The full input beam state can be perturbed per-seed via
`BeamErrorDef`:

| Parameter | Units | Effect |
|---|---|---|
| `centroid_x`, `centroid_y` | mm | input beam centroid |
| `centroid_xp`, `centroid_yp` | mrad | input beam divergence |
| `centroid_dphi`, `centroid_dw` | deg, MeV | input phase + energy |
| `emit_nx_rel`, `emit_ny_rel`, `emit_z_rel` | fractional | ε growth |
| `mismatch_x`, `mismatch_y`, `mismatch_z` | fractional | Twiss mismatch |
| `current_rel` | fractional | beam current jitter |

## Workflow

```
.dat with ERROR_*       (or)    Python ErrorDef list
        │                                │
        └──────────┬─────────────────────┘
                   ▼
              ErrorStudy
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      seed 1     seed 2     seed N
        │          │          │
        ▼          ▼          ▼
    apply random draw
        ▼          ▼          ▼
   Simulation.run()  ...
        ▼          ▼          ▼
        └──────────┬──────────┘
                   ▼
        ErrorStudyResults
        (mean / std / percentile across seeds)
```

## What you get back

`ErrorStudyResults` aggregates the per-seed recorders:

* `mean(q)`, `std(q)`, `percentile(q, p)` — statistics of any recorder
  quantity (`"sigma_x"`, `"emit_nx"`, `"transmission"`, ...) across
  seeds, returned as a 1-D array vs element index.
* `transmission_stats()` — `{"mean", "min", "max", "std"}` of the
  final transmission over all seeds.
* `corrected_kicks(i)` / `correction_history(i)` — per-seed orbit
  correction record (`None` unless `enable_correction()` was used).
* `n_seeds` — seeds actually completed; `n_requested` — seeds asked
  for (they differ when the run was stopped early).
* `_recorders` — the raw per-seed `DiagnosticRecorder` list, for
  forensics on an individual seed.

## Cross-references

* [ERROR_* directives](02_error_directives.md) — TraceWin-format error specs.
* [Element-level errors](03_element_errors.md) — alignment + field details.
* [Beam-level errors](04_beam_errors.md) — input-beam jitter.
* [Running studies](05_running_studies.md) — Python API + GUI tab.
* [Interpreting results](06_interpreting.md) — what to look for.
* [Worked example: tolerance study](../11_examples/08_tolerance_study.md).

← [Multi-objective matching](../07_matching/08_multiobjective.md) ·
[Continue to ERROR_* directives →](02_error_directives.md)
