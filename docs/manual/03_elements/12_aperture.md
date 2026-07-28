# Aperture

A thin loss-collection element: any particle outside the aperture is
marked dead and removed from further tracking.  Used to model
collimators, beam-pipe restrictions, beam dumps, and to count
losses for radiation protection / activation studies.

## TL;DR

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `APERTURE dx dy n` | `Aperture(name, dx, dy, aperture_type)` |
| `dx`, `dy` | half-aperture in x, y (mm) | same |
| `n` | 0 = rectangular, 1 = circular, 2-6 = variants | same |

Aperture types (TraceWin `n` flag):

| `aperture_type` | Shape | Behaviour |
|:---:|---|---|
| 0 | rectangular: `\|x\| ≤ dx and \|y\| ≤ dy` | enforced |
| 1 | **circular**: `x² + y² ≤ dx²` — `dy` is ignored | enforced |
| 2 | pepperpot | **warn + pass-through** (not simulated) |
| 3 | rectangular + beam-fraction adjustment | treated as rectangular |
| 4, 5 | horizontal / vertical finger | treated as rectangular |
| 6 | ring (annular) | **warn + pass-through** (not simulated) |

There is no elliptical shape — TraceWin's APERTURE card has none,
and the legacy `ELLIPTICAL` constant was removed (type 1 is a
circle of radius `dx`).

## Tutorial

A round 20 mm radius aperture:

```python
from linac_gen.elements.aperture import Aperture

ap = Aperture(name="A1", dx=20.0, aperture_type=1)  # circular, radius dx
```

Inside a tracking run, the `Aperture.apply(beam)` step:

1. For each live particle, evaluates the shape predicate.
2. Records the failures as lost (`beam.record_loss(...)` sets
   `beam.lost[i] = True`, so they drop out of `beam.alive_mask`).
3. Updates the recorder's `transmission` array at the next diagnostic
   point.

### Transfer matrix

An `Aperture` is a **passive** zero-length element — its transfer
matrix is the 6×6 identity:

$$
M_{\text{aperture}} = I_{6\times 6}
$$

Surviving particles pass through unchanged; lost particles are
masked from later integration steps.  Envelope tracking ignores
the loss mechanism (it propagates the second moment of the
distribution; an ideal Gaussian has no boundary), so for accurate
transmission you need
[multi-particle tracking](../02_concepts/03_tracking_modes.md).

To see **where** particles were lost, use the per-particle loss
bookkeeping on the beam plus the pipe-radius overlay:

```python
from linac_gen.analysis.aperture_profile import aperture_profile
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.drift import Drift

lattice = Lattice()
lattice.add(Drift(name="D1", length=300.0, aperture=15.0))
lattice.add(Aperture(name="COLL", dx=2.0, dy=2.0, aperture_type=1))
lattice.add(Drift(name="D2", length=300.0, aperture=15.0))
beam = create_beam(BeamConfig(n_particles=1000, beta_x=8.0, beta_y=8.0),
                   seed=4)
Simulation(lattice, beam).run()

s_mm, rx_mm, ry_mm = aperture_profile(lattice)   # pipe radius vs s
losses = beam.loss_table       # per-particle (s, element) records (property)
```

See [Aperture profile](../09_diagnostics/04_aperture.md) for details.

## API reference

```{.python .skip}
linac_gen.elements.aperture.Aperture(
    name: str,
    dx: float = 0.0,              # half-aperture x (mm); radius for type 1
    dy: float = 0.0,              # half-aperture y (mm); ≤0 → defaults to dx
    aperture_type: int = 0,       # TraceWin n flag: 0=rect, 1=circular, 2-6=variants
    a: float | None = None,       # legacy alias for dx
    b: float | None = None,       # legacy alias for dy
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `dx` | 0.0 | mm | half-aperture in x (radius for type 1) |
| `dy` | 0.0 | mm | half-aperture in y; when ≤ 0, defaults to `dx`; ignored by type 1 |
| `aperture_type` | 0 | int | shape code (see TL;DR table) |
| `a`, `b` | None | mm | legacy kwargs, mapped onto `dx` / `dy` |

The check runs in `apply(beam)` (Aperture is a zero-length
`PassiveElement`; it has no `track` method).

### Source

`linac_gen/elements/aperture.py:1`

## See also

* [Drift](01_drift.md) — element `aperture` fields are **enforced**
  by the tracker after every element and inside drift sub-step
  bundles; use a dedicated `Aperture` element for shaped
  (rectangular/finger) collimators at a point.
* [Aperture profile analysis](../09_diagnostics/04_aperture.md).

← [Multipole](11_multipole.md) ·
[Continue to Marker →](13_marker.md)
