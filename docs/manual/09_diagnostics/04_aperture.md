# Aperture and losses

Where in the lattice particles are lost, and against what pipe
boundary.  Critical for high-power machines (PIP-II at 1.6 MW
target) where a 0.1% loss in the wrong place causes
unmaintainable activation.

## Aperture profile (pipe boundary vs s)

`aperture_profile()` walks the lattice and returns the beam-pipe
half-widths as piecewise step arrays — made for overlaying on
envelope plots:

```python
from linac_gen.analysis.aperture_profile import aperture_profile
from linac_gen.core.lattice import Lattice
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.drift import Drift

# A toy beamline with a tight collimator in the middle:
lattice = Lattice()
lattice.add(Drift(name="D1", length=300.0, aperture=15.0))
lattice.add(Aperture(name="COLL", dx=2.0, dy=2.0, aperture_type=1))
lattice.add(Drift(name="D2", length=300.0, aperture=15.0))

s_mm, rx_mm, ry_mm = aperture_profile(lattice)

# Overlay the pipe radius on the envelope of a quick run:
from linac_gen.core.config import BeamConfig
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam

beam0 = create_beam(BeamConfig(n_particles=1000), seed=2)
results = Simulation(lattice, beam0).run()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(results.s, results.sigma_x, label="σ_x [mm]")
ax.step(s_mm, rx_mm, where="post", color="k", label="pipe rx [mm]")
ax.set_xlabel("s [mm]"); ax.set_ylabel("mm"); ax.legend()
plt.close(fig)
```

Two aperture sources are honoured: the element's constant
`aperture` (and optional `aperture_y` for rectangular pipes), and —
for `FieldMap` elements parsed with `Ka=1` and a `.ouv` file — the
per-z `pipe_radius_profile`.  Zero-length elements and elements with
`aperture <= 0` are skipped so they don't pull the boundary to zero.
For a circular pipe `rx == ry`.

## How loss is detected and recorded

During multi-particle tracking, any particle exceeding the local
aperture is marked dead via `Beam.record_loss(particle_id, s,
element_name)`.  Each loss is logged once (no duplicates) with the
particle's position and energy at the moment of loss.

The full log is available after the run as `Beam.loss_table` — a
structured numpy array with fields `particle_id`, `s`, `x`, `y`,
`energy`, `element_name`:

```python
from linac_gen.core.config import BeamConfig
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam

# Track a beam that is wider than the 2 mm collimator above, so some
# particles actually get lost:
beam = create_beam(BeamConfig(n_particles=2000, beta_x=8.0, beta_y=8.0),
                   seed=9)
Simulation(lattice, beam).run()

table = beam.loss_table
for row in table[:5]:
    print(f"  particle {row['particle_id']} lost at s={row['s']:.0f} mm "
          f"in {row['element_name']} (x={row['x']:.2f}, y={row['y']:.2f})")
```

The recorder's `transmission` array (per-step live-particle
percentage) gives the cumulative view of the same information —
plot `100 - transmission` for cumulative loss vs s.

## Loss-per-element table

```python
from collections import Counter
losses_by_elem = Counter(beam.loss_table["element_name"])
for elem, count in losses_by_elem.most_common(10):
    print(f"  {elem:20s}  {count} particle(s) lost")
```

## GUI

In the **Results** tab, the envelope plots have a **Show aperture**
checkbox that overlays the `aperture_profile()` boundary on σ_x/σ_y.
The **Loss** tile plots cumulative loss vs s from the transmission
record, and the aperture-losses popup overlays each `loss_table`
hit as a scatter point on the pipe profile.  (Envelope mode tracks
no losses — transmission is implicitly 100 % there.)

## Engineering decisions

| Loss profile shape | Diagnosis |
|---|---|
| Cliff edge at one element | aperture too tight there → loosen, or filter beam upstream |
| Slow ramp throughout | accumulating halo growth → upstream matching issue |
| Localised spike at FREQ jump | bunch length convention issue |
| Cumulative > 1% | reconsider design |

## Watt/m budget

Translating particle counts into power deposition:

$$
P/L\,[\text{W/m}] = \text{(loss fraction per metre)} \cdot
\text{(beam power)}
$$

For PIP-II, the budget is ~1 W/m everywhere except the dump.  See
[H⁻ stripping](05_stripping.md) for the analytic loss-rate
analyzers (used alongside aperture loss).

## Cross-references

* [Halo analysis](03_halo.md)
* [H⁻ stripping](05_stripping.md)
* [Aperture element](../03_elements/12_aperture.md)
* `linac_gen/analysis/aperture_profile.py:43` — `aperture_profile()`.
* `linac_gen/core/beam.py:84` — `record_loss` / `loss_table`.

← [Halo](03_halo.md) ·
[Continue to H⁻ stripping →](05_stripping.md)
