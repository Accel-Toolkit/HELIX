# Beam-level errors

Random per-seed perturbations to the *input beam state* — centroid,
emittance, mismatch, current.  Models source / chopper jitter,
upstream parameter drift, and (in coupled studies) the
beam-state result of upstream alignment / RF errors.

## TL;DR

```python
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.errors.error_model import ErrorStudy
from linac_gen.errors.beam_error import BeamErrorDef

# Minimal study to hang the beam errors on — use your own lattice.
lat = Lattice(); lat.add(Drift(name="D1", length=500.0))
study = ErrorStudy(lat, BeamConfig(n_particles=500), n_seeds=2)

study.add_beam_error(
    parameter="centroid_x",
    distribution="gaussian",
    sigma=0.1,           # mm
)
study.add_beam_error(
    parameter="current_rel",
    distribution="gaussian",
    sigma=0.01,           # 1% rms current jitter
)
```

Each `BeamErrorDef` produces a per-seed perturbation applied to a
copy of the `BeamConfig` before `create_beam()`.

## Parameters

| Parameter | Units | Effect on `BeamConfig` |
|---|---|---|
| `centroid_x`, `centroid_y` | mm | adds to `BeamConfig.centroid_x/y` |
| `centroid_xp`, `centroid_yp` | mrad | input divergence |
| `centroid_dphi` | deg | input phase offset |
| `centroid_dw` | MeV | input energy offset |
| `emit_nx_rel`, `emit_ny_rel`, `emit_z_rel` | fractional | scales ε_n |
| `mismatch_x`, `mismatch_y`, `mismatch_z` | fractional | additive to existing mismatch |
| `current_rel` | fractional | scales `BeamConfig.current` |

All distributions same as element errors: gaussian (with
`cutoff`) or uniform (with `half_width`).

## Use cases

### Source / chopper stability

The injector source has shot-to-shot variation in current,
energy, centroid:

```python
study.add_beam_error("centroid_x", sigma=0.05, distribution="gaussian")
study.add_beam_error("centroid_y", sigma=0.05, distribution="gaussian")
study.add_beam_error("current_rel", sigma=0.01, distribution="gaussian")
```

### Upstream-section uncertainty

If you're running just the SSR1 section, the *input* beam has
been already perturbed by errors in the upstream MEBT/HWR.  Model
this with input-Twiss jitter:

```python
study.add_beam_error("emit_nx_rel", sigma=0.02, distribution="gaussian")
study.add_beam_error("emit_ny_rel", sigma=0.02, distribution="gaussian")
```

### Mismatch-tolerance study

How much mismatch can the lattice take before transmission falls
below 99 %?  Sweep mismatch:

```python
study.add_beam_error("mismatch_x", half_width=0.10, distribution="uniform")
study.add_beam_error("mismatch_y", half_width=0.10, distribution="uniform")
```

Note that the uniform distribution reads `half_width`, not `sigma` —
a uniform `BeamErrorDef` with `half_width=0.0` is silently skipped,
even if `sigma` was set.

## From `.dat`

The TraceWin `ERROR_BEAM_STAT` directive produces equivalent
`BeamErrorDef` entries automatically:

```
ERROR_BEAM_STAT 2 0.1 0.1 0 0 0 0 0 0 0 0 0 0 1.0
```

Reads as: gaussian, σ_centroid_x = 0.1 mm, σ_centroid_y = 0.1 mm,
and 1 % current jitter — the last slot is in **percent**, which the
parser divides by 100 into a fractional `current_rel` draw (it is
*not* a mA value).

## Cross-references

* [Element errors](03_element_errors.md) — for per-element perturbations.
* [Running studies](05_running_studies.md).
* [Worked example: combined realistic](../11_examples/08_tolerance_study.md).

← [Element errors](03_element_errors.md) ·
[Continue to Running studies →](05_running_studies.md)
