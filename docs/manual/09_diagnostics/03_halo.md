# Halo analysis

The "halo" is the long-tail population of particles outside the
Gaussian core.  Halo dominates aperture losses in real linacs —
even though halo carries < 1 % of the beam current, it dominates
activation budgets by 1000×.  HELIX provides several halo
diagnostics.

## Kurtosis halo parameter

The simplest halo measure is the dimensionless kurtosis-based halo
parameter h (computed in `linac_gen/diagnostics/moments.py`):

$$
h_x = \frac{\langle x^4 \rangle}{\langle x^2 \rangle^2} - 1
$$

Recorded as `halo_x` and `halo_y` per s-step.  With this definition a
Gaussian profile gives h = 2 (its ⟨x⁴⟩/⟨x²⟩² kurtosis is 3), and
flatter, tail-free distributions land *below* the Gaussian value:

| h | Meaning |
|---|---|
| ≈ 2 | pure Gaussian (matched) |
| < 2 | sub-Gaussian / compact (uniform ≈ 0.8, KV ⇒ 1.0) |
| noticeably > 2 | halo / super-Gaussian tails |
| ≫ 2 (e.g. > 3) | strong halo |

A growing h trace means halo is being produced (mismatch, RF
nonlinearity, SC filamentation).

## From halo to losses

Beyond h, the question is usually "where do the halo particles
actually hit the pipe?".  HELIX answers that with per-particle loss
bookkeeping: every particle that exceeds the local aperture is logged
with its (s, x, y, energy, element) in `Beam.loss_table`, and the
recorder's `transmission` array shows the cumulative survival vs s.
See [Aperture and losses](04_aperture.md) for the API and plots.

## Maximum-excursion (envelope of all surviving particles)

`x_max[i]` and `y_max[i]` record the most-extreme position of any
*surviving* particle at every s.  Plotting `x_max(s)` against the
local aperture shows whether you have margin against worst-case
particle losses:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

# A small run with a halo-carrying thermal beam:
lattice = Lattice()
for c in range(2):
    lattice.add(Quadrupole(name=f"QF_{c}", length=100.0, gradient=+10.0))
    lattice.add(Drift(name=f"D1_{c}", length=200.0))
    lattice.add(Quadrupole(name=f"QD_{c}", length=100.0, gradient=-10.0))
    lattice.add(Drift(name=f"D2_{c}", length=200.0))
beam = create_beam(BeamConfig(n_particles=2000, distribution="thermal"),
                   seed=5)
results = Simulation(lattice, beam).run()

fig, ax = plt.subplots()
ax.plot(results.s, results.x_max, label="max x [mm]")
# overlay aperture
ax.set_xlabel("s [mm]"); ax.set_ylabel("excursion [mm]")
```

## Visualising halo content

For PIP-II-realistic distributions with a halo population, the
[`thermal` distribution](../04_beam/01_distributions.md#thermal-bi-gaussian-halo)
is the right starting point.  After tracking, plot the (x, x')
phase space at end of lattice:

```python
xs = beam.particles[beam.alive_mask, 0]
xps = beam.particles[beam.alive_mask, 1]
plt.scatter(xs, xps, s=0.5, alpha=0.3)
plt.xlabel("x [mm]"); plt.ylabel("x' [mrad]")
plt.close("all")
```

The core appears as a dense ellipse; halo as scattered points
extending well beyond.

## Cross-references

* [Aperture profile](04_aperture.md) — converting halo into loss
  numbers per element.
* [Distributions → thermal](../04_beam/01_distributions.md#thermal-bi-gaussian-halo).
* [Recorder fields](01_recorder.md).

← [Emittances](02_emittances.md) ·
[Continue to Aperture →](04_aperture.md)
