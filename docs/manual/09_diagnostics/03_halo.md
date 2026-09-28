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

## Action scan: the beam outside n·ε_rms ellipses {#action-scan}

The action scan measures how far out the beam reaches in each
phase-space plane, in units of the local rms emittance, at every
recorded position.  For particle i, with the beam's own rms Twiss
parameters at that position,

$$
W_i = \gamma u_i^2 + 2\alpha u_i u'_i + \beta u_i'^2 ,
\qquad n_i = \frac{W_i}{\varepsilon_\mathrm{rms}}
$$

W is the emittance of the ellipse of the beam's rms shape that passes
through the particle (W = 2J, J the action), so **n is the ellipse
size in multiples of the rms emittance**: n = 1 is the rms ellipse (it
reaches 1σ in u and in u′), and a particle at n reaches √n rms sizes
(n = 25 → 5σ).  At every step the recorder counts the particles
outside each ellipse, N_out(n) = #{i : n_i > n}, on a fixed grid of n
(0 to 50 in steps of 0.25, then 51 to 400 in steps of 1).  The planes
are x–x′ and y–y′, dispersion-corrected (the betatron part: the
correlation with the energy deviation removed, the particle-level form
of the Results tab's *Dispersion-corrected* view) and raw, and φ–W.

Reference values (% of the particles present outside the ellipse):

| n | Gaussian 100·e^(−n/2) | uniformly filled ellipse 100·(1 − n/4) |
|---|---|---|
| 1 | 60.7 % | 75 % |
| 2 | 36.8 % | 50 % |
| 4 | 13.5 % | 0 % |
| 5.99 | 5 % | — |
| 9.21 | 1 % | — |
| 13.82 | 0.1 % | — |
| 18.42 | 0.01 % | — |

In the tail (n ≳ 9, beyond 3σ) a beam that keeps more than the
Gaussian fractions has halo; at small n the fractions describe the
shape of the core (the uniform ellipse keeps more than a Gaussian at
n = 1 and 2 and has no halo at all).  The fractional emittances of the
tail diagnostics (`Simulation(tail_fractions=…)`, raw x and y) are
defined as ε_q = ½·quantile_q(W) (ε_99 = 4.6 ε_rms for a Gaussian), so
in the raw planes a fractional emittance ε_q corresponds to
n = 2 ε_q/ε_rms here.

Recording is on by default in the GUI and opt-in from Python and the CLI.  It costs about 20–30 ms per record at 10⁵
particles and about 12 kB of memory per record; its share of the run
time depends on how much tracking happens between records — about 1 %
for a superconducting linac with field maps and space charge (≈ 2 s
of tracking per record at 10⁵ particles), 5–10 % on short test
lattices, and more with per-sub-step recording or without space
charge — a transfer line tracked without space charge (the BTL, 10⁵
particles) measured +40–60 % run time, and the results file grew 2.2×
(254 → 564 kB).  Untick the box for long parameter sweeps where the halo
counts are not needed.

* Python: `Simulation(..., record_action_scan=True)` (forward
  single-bunch multi-particle runs; `run_backtrack()` and the
  multibunch train files do not carry it), or
  `recorder.configure_action_scan()` before tracking;
* GUI: **Record halo action scan** on the Numerics tab is ticked when
  HELIX starts.  A project stores the setting; opening one saved with it
  unticked turns it off, and it then stays off (for later projects that
  do not store it, too) until ticked again.  Run, then open
  **Halo action scan (n·ε_rms)** on the Results tab;
* CLI: `python -m linac_gen run <input> --mode mp --action-scan` — the
  CLI does not read the project's setting, so a project the GUI saved
  with the scan on still needs the flag.

The counts are saved in the results HDF5 file (`action_scan/`, see
[Results files](../06_running/04_results.md)) and read back, live or
from a file, with `action_scan_from_results`:

```python
from linac_gen.diagnostics.action_scan import (
    action_scan_from_results, gaussian_level_n)

beam = create_beam(BeamConfig(n_particles=20_000, distribution="thermal"),
                   seed=5)
results = Simulation(lattice, beam, record_action_scan=True).run()

scan = action_scan_from_results(results)          # -> ActionScan
n_1pct = scan.level_crossing("x", 1.0)            # one value per step
print(f"x: fewer than 1 % beyond n = {n_1pct[-1]:g} at the exit "
      f"(Gaussian: {gaussian_level_n(1.0):.2f})")
print(f"outermost particle: n = {scan.n_max['x'][-1]:.1f}, "
      f"eps_rms,n = {scan.eps_n['x'][-1]:.4f} mm*mrad")
```

`ActionScan` holds the grid `n`, `s` (mm), `n_alive`, and per plane
(`x`, `y`, `z`, `x_raw`, `y_raw`) the counts `(steps, grid)`, `eps`
(geometric: mm·mrad, z in deg·MeV), `eps_n` (normalized, mm·mrad) and
`n_max` (outermost particle).  `percent_outside(plane)` returns the %
of the particles present; `level_crossing(plane, pct)` the smallest n
with fewer than `pct` % outside (NaN where the level is below one
particle).  For one distribution, `ellipse_sizes(particles, plane)`
returns every particle's n and the plane's ε_rms (⟨n⟩ = 2 exactly).

!!! note "Reading the scan"
    * n is relative to the **local** ε_rms: when the core emittance
      grows, the same particle amplitude gives a smaller n.  Read the
      map together with ε_rms(s) (the GUI draws it underneath).
    * DC (unbunched) records have no longitudinal ellipse; the GUI
      masks φ–W there.  A beam bunched from DC without periodic phase
      coordinates spans the whole bunch train in φ.
    * Inside coupling elements (solenoids) x–x′ and y–y′ are
      projections, not normal modes.
    * A long, curved longitudinal distribution (e.g. a debunching
      beam) reaches very large n in φ–W: there n measures the shape,
      not only halo.

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
