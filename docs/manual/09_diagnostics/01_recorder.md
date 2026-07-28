# Recorder fields

`DiagnosticRecorder` is the per-step diagnostic store.  Every field
on it is a `list[float]` (or list of arrays) with one entry per
recording point.  This page is the field-by-field reference.

## Quick access pattern

```python
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

# Small demo run — any Simulation works the same way.
lattice = Lattice()
for c in range(2):
    lattice.add(Quadrupole(name=f"QF_{c}", length=100.0, gradient=+10.0))
    lattice.add(Drift(name=f"D1_{c}", length=200.0))
    lattice.add(Quadrupole(name=f"QD_{c}", length=100.0, gradient=-10.0))
    lattice.add(Drift(name=f"D2_{c}", length=200.0))
beam = create_beam(BeamConfig(n_particles=1000), seed=42)
sim = Simulation(lattice, beam)

results = sim.run()
# Convert to numpy for ease:
import numpy as np
s_m = np.asarray(results.s) / 1e3            # mm → m
sigma_x = np.asarray(results.sigma_x)        # mm
W = np.asarray(results.ref_w_kin)            # MeV
```

## Reference

### Path & RF state

| Field | Type | Units | Notes |
|---|---|---|---|
| `s` | list[float] | mm | accumulated path length |
| `element_names` | list[str] | — | element name at this index — a `SHIFT_IN_FIELD_MAP` interior diagnostic adds a row carrying the *diagnostic's* name at `s = s_entry + dz`, inside its host [SuperposedFieldMap](../03_elements/19_superposedfieldmap.md) |
| `ref_w_kin` | list[float] | MeV | reference particle kinetic energy |
| `ref_phi_s` | list[float] | deg | synchronous phase |
| `ref_beta` | list[float] | — | β = v/c |
| `ref_gamma` | list[float] | — | γ = (1 + W/mc²) |
| `ref_bg` | list[float] | — | β·γ |
| `ref_frequency` | list[float] | MHz | local RF frequency (changes at FREQ jumps) |
| `mass_mev` | float | MeV | rest mass (single value, not per-step) |

### Beam moments

| Field | Type | Units | Notes |
|---|---|---|---|
| `sigma_x`, `sigma_y` | list[float] | mm | RMS transverse beam size |
| `sigma_phi` | list[float] | deg | RMS phase spread (at local RF freq) |
| `sigma_w` | list[float] | MeV | RMS energy spread |
| `centroid` | list[ndarray (6,)] | mixed | full 6-D centroid |
| `sigma_matrix` | list[ndarray (6,6)] | mixed | full Σ |

### Twiss

| Field | Type | Units | Notes |
|---|---|---|---|
| `alpha_x`, `alpha_y` | list[float] | — | Twiss α |
| `beta_x`, `beta_y` | list[float] | mm/mrad ≡ m | Twiss β |
| `alpha_z` | list[float] | — | longitudinal Twiss α (2026-07) — HELIX-internal (Δφ, ΔW) convention: α_z = −TraceWin's ([appendix E](../appendices/E_migrating_from_tw.md)) |
| `beta_z` | list[float] | deg/MeV | longitudinal Twiss β at the local machine clock (× f_new/f_old across `FREQ`) |

### Emittances

| Field | Type | Units | Notes |
|---|---|---|---|
| `emit_x`, `emit_y` | list[float] | mm·mrad | geometric ε |
| `emit_z` | list[float] | deg·MeV | longitudinal ε (native units) |
| `emit_z_mmmrad` | list[float] | mm·mrad | ε_z converted via local k_φ |
| `emit_nx`, `emit_ny` | list[float] | mm·mrad | normalised (β·γ·ε) |
| `emit_nz` | list[float] | mm·mrad | normalised longitudinal |
| `emit_4d` | list[float] | mm²·mrad² | 4-D coupling-invariant √det(Σ_4D) |
| `emit_n1`, `emit_n2` | list[float] | mm·mrad | 4-D normal-mode eigen-ε |
| `emit_e1`, `emit_e2` | list[float] | mm·mrad | 6-D Balandin eigenemittances (transverse modes) |
| `emit_e3` | list[float] | deg·MeV | 6-D Balandin eigenemittance, longitudinal mode — the mixed-unit Σ makes ε₃ come out in deg·MeV, matching `emit_z` |

For full eigenemittance theory see [Emittances](02_emittances.md).

### Halo

| Field | Type | Units | Notes |
|---|---|---|---|
| `halo_x`, `halo_y` | list[float] | — | kurtosis-based halo parameter h = ⟨x⁴⟩/⟨x²⟩² − 1 (Gaussian ⇒ 2.0, KV ⇒ 1.0); see [Halo analysis](03_halo.md) |

### Beam state

| Field | Type | Units | Notes |
|---|---|---|---|
| `transmission` | list[float] | % | live-particle fraction |
| `x_max`, `y_max` | list[float] | mm | peak transverse excursion (any surviving particle) |
| `continuous_at` | list[bool] | — | True if pre-RFQ (DC); flags σ_φ/σ_W as non-physical |

### Density-vs-s recording (opt-in)

The recorder can also build 2-D density maps (one histogram column
per s-step) for any of the beam coordinates `x, xp, y, yp, phi, w`:

```python
from linac_gen.diagnostics.recorder import DiagnosticRecorder
from linac_gen.tracking.tracker import Tracker

recorder = DiagnosticRecorder()
recorder.configure_density(axes=("x", "y"), n_bins=200)  # before tracking
beam2 = create_beam(BeamConfig(n_particles=1000), seed=7)
Tracker(lattice, beam2, recorder=recorder).run()
img = recorder.density_array("x")   # int32, shape (n_steps, n_bins)
edges = recorder.density_edges["x"] # bin edges for axis labelling
```

* `configure_density(axes, extent=None, n_bins=200)` — opt in;
  missing `extent` entries are auto-fitted on the first record with
  headroom for downstream growth.
* `record_density(beam)` — called automatically by `record()`;
  appends one histogram column per configured axis (zero columns when
  the beam is empty, so the grid stays rectangular vs `s`).
* `density_array(axis)` — returns the `(n_steps, n_bins)` array, or
  `None` if that axis was never recorded.  The raw columns live in
  `recorder.density`, extents in `density_extent`, bin count in
  `density_n_bins`.

### Particle snapshots

* `save_snapshot(beam, s_position)` — stores a full copy of the
  particle array plus the reference-particle state at that `s`.
* `beam_at(s_position)` — returns the stored `(particles, ref)`
  tuple, or raises `KeyError` if no snapshot was saved there.

## Conventions and pitfalls

!!! note "All transverse lengths in mm"
    `s`, `sigma_x`, `x_max`, `centroid[0,2]` — all mm.  Convert to
    m for plot axes if you prefer (`s_m = np.asarray(s) / 1e3`).

!!! note "σ_φ at LOCAL frequency"
    `sigma_phi` is in degrees of the *local* RF frequency at that s.
    In a multi-section linac (162.5 → 325 → 650 MHz), σ_φ doubles or
    quadruples at FREQ jumps for the same physical bunch length.
    This is TraceWin's convention.

!!! warning "continuous_at is True ⇒ longitudinal fields invalid"
    For pre-RFQ (`Beam.continuous=True`), σ_φ, σ_W, ε_z are not
    physical — they're an artefact of imposing a phase coordinate
    on a continuous beam.  Always check `continuous_at` before
    plotting longitudinal fields.

## Source

`linac_gen/diagnostics/recorder.py:1` — full implementation.

## Cross-references

* [Coordinates & units](../02_concepts/01_coordinates.md)
* [Reading results](../06_running/04_results.md)
* [Emittances](02_emittances.md)

← [Errors → Orbit correction](../08_errors/07_correction.md) ·
[Continue to Emittances →](02_emittances.md)
