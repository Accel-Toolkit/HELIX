# Worked example: Tolerance study

A "kitchen sink" tolerance study stacking every error type — quad
alignment, gradient jitter, cavity LLRF, dipole alignment, beam
input centroid + current jitter — into one ensemble.  Used as the
template for real PIP-II tolerance budget runs.

## Source

`examples/error_studies/combined_realistic/`:

* `combined_realistic.dat` — full lattice with `ERROR_*` directives
* `combined_realistic.lgproj` — open in GUI
* `README.md` — narrative description

## Errors applied

```
ERROR_GAUSSIAN_CUT_OFF 3                   ; truncate at ±3σ
ERROR_QUAD_NCPL_STAT 6 2 0.2 0.2 0 0 0 0.5 0 0 0 0   ; quads
ERROR_CAV_NCPL_STAT  2 2 0.3 0.3 0 0 1.0 1.0 0       ; cavities
ERROR_BEND_NCPL_STAT 1 2 0.5 0.5 0 0 0 0.5 0          ; dipole
ERROR_BEAM_STAT      2 0.1 0.1 0 0 0 0 0 0 0 0 0 0 1.0 ; input
```

Per-element tolerance budget:

| Element type | Alignment σ | Field error σ | Per-seed budget |
|---|---|---|---|
| Quadrupole | 0.2 mm dx, dy | 0.5 % gradient | 6 quads |
| RF gap | 0.3 mm dx, dy | 1 % voltage, 1° phase | 2 gaps |
| Dipole | 0.5 mm dx, dy | 0.5 % field | 1 bend |
| Input beam | 0.1 mm centroid, 1 % current | — | once per seed |

!!! note "Two of the field errors are currently no-ops"
    The cavity amplitude (`E`) slot only takes effect on `RFGap`
    cavities — on `FieldMap` cavities it is a silent no-op — and the
    dipole `dg` slot is a no-op on `Dipole` (no `field` attribute).
    Alignment and phase errors apply as written.  See
    [Element-level errors](../08_errors/03_element_errors.md).

## Run

### Python

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.errors.error_model import ErrorStudy
from linac_gen.core.config import BeamConfig

lat, _ = parse_tracewin("examples/error_studies/combined_realistic/combined_realistic.dat")

beam_cfg = BeamConfig(species="proton", energy=2.5, frequency=325.0,
                      current=5.0, n_particles=5000,
                      emit_nx=0.25, emit_ny=0.25, emit_z=0.30,
                      alpha_x=0.0, beta_x=0.5,
                      alpha_y=0.0, beta_y=0.5,
                      alpha_z=0.0, beta_z=1.0)

study = ErrorStudy(lat, beam_cfg, n_seeds=100)   # always multi-particle
results = study.run()

stats = results.transmission_stats()
print(f"Mean transmission: {stats['mean']:.2f} %")
print(f"  range over seeds: {stats['min']:.2f} - {stats['max']:.2f}")
```

### GUI

Open `combined_realistic.lgproj` → Error Study tab (errors auto-loaded
from `.dat`) → set n_seeds=100 → Run.

## What you'll see

* **σ-band envelopes**: visibly broader than any single-error case.
* **Transmission histogram**: a left tail extending below 99 % —
  the longest tail comes from the dipole alignment driving an
  orbit excursion through the bend's dispersion.
* **Centroid drift**: deterministic component (from bend
  misalignment) + stochastic (from quads + input).

## Decomposing the budget

To find which error category dominates, comment out 4 of 5
directives at a time and re-run.  The single-error transmission
spread relative to the combined run gives each category's
contribution.

## Cross-references

* [Errors overview](../08_errors/01_overview.md)
* [Interpreting results](../08_errors/06_interpreting.md)
* [GUI → Error Study tab](../10_gui/06_errors_tab.md)

← [Matching walkthrough](07_matching_walkthrough.md) ·
[Continue to Eigenemittance →](09_eigenemittance.md)
