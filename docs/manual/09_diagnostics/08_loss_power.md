# Loss power & plane power density

Multi-particle runs record every particle loss (position *s*, transverse
coordinates, **kinetic energy at the loss point**, and the element that
took the hit).  `linac_gen.analysis.loss_power` turns that record into
the quantities machine protection and absorber engineering use.

## Power convention

`BeamConfig.current` is the **peak (in-pulse)** current; `duty_cycle`
converts it to an average:

```
I_avg = current [mA] · 1e-3 · duty_cycle/100          [A]
P     = (I_avg / n_macro) · E [MeV] · 1e6             [W per macroparticle]
```

Energies are taken **at the loss point** — a particle scraped in the
MEBT costs 2 MeV-scale power, not the linac exit energy.  Sanity
anchor: 2 mA × 550 µs × 20 Hz (duty 1.1 %) at 800 MeV is 17.6 kW —
the PIP-II Booster-bound beam against a 25 kW absorber rating.

## CLI

Every `linac_gen run --mode mp` report ends with a power block:

```
[power] I_avg = 5 mA (5 mA x 100% duty)  delivered = 6.95 kW
[power] lost = 3.67 kW over 6907 macroparticles
[power]   APER_001                    3.67 kW  (n=6907)
```

`delivered` is the surviving beam's power at the exit plane; the table
lists the worst loss elements in watts.

## Python API

```python
import numpy as np
from linac_gen.analysis.loss_power import (
    loss_power_table, loss_power_profile, plane_power_density)
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.drift import Drift

# a wide 2 MeV H- beam through a 2 mm collimator
lattice = Lattice()
lattice.add(Drift("D1", length=200.0, aperture=50.0))
lattice.add(Aperture("APER", dx=2.0, dy=2.0, aperture_type=0))
lattice.add(Drift("D2", length=200.0, aperture=50.0))
ref = ReferenceParticle(species=H_MINUS, w_kin=2.0, frequency=162.5)
beam = Beam(ref=ref, n_particles=2000, current=5.0)
rng = np.random.default_rng(11)
beam.particles[:, 0] = rng.normal(0.0, 2.0, 2000)   # x [mm]
beam.particles[:, 2] = rng.normal(0.0, 2.0, 2000)   # y [mm]
results = Simulation(lattice, beam).run()            # or load_results_hdf5(...)

tab = loss_power_table(results.loss_table, current_mA=5.0,
                       duty_pct=100.0, n_macro=results.n_macro)
# structured array: element_name, n_lost, s_min/max_mm, e_mean_mev, watts

s, wpm = loss_power_profile(results.loss_table, current_mA=5.0,
                            duty_pct=100.0, n_macro=results.n_macro,
                            s_end_mm=360_000.0)      # W/m in 1 m bins
```

The 1 m default binning of `loss_power_profile` reads directly against
the ~1 W/m hands-on-maintenance criterion.

**Power density on a dump / window / foil face** (W/cm²) from any
particle distribution at that plane — e.g. the final beam of an MP run
or a loaded `.dst`:

```python
x_mm = beam.particles[beam.alive_mask, 0]            # surviving beam at the exit plane
y_mm = beam.particles[beam.alive_mask, 2]
H, xe, ye = plane_power_density(x_mm, y_mm, energy_mev=2.0,
                                current_mA=5.0, duty_pct=100.0,
                                n_macro=results.n_macro, bins=64)
# H [W/cm²]; integrates back to the total beam power on the plane
```

## HDF5 persistence

MP results carry a `losses/` group (`s`, `x`, `y`, `energy`,
`particle_id`, `element_name`, attr `n_macro`);
`load_results_hdf5` restores it as `results["loss_table"]` /
`results["n_macro"]`, so reloaded runs support the full analysis.

## GUI & assistant

Two Results-tab tiles, both under **LOSSES · TRANSMISSION**:

* **Loss power (W/m · W/cm²)** — the plots.  Top: the lineal loss
  density along the machine in 1 m bins, with the ~1 W/m hands-on
  criterion drawn as a dashed reference line and the lattice-element
  strip underneath.  Bottom: the **W/cm² heat map** of the surviving
  beam on the terminal plane (dump face / window / foil).  The status
  line carries I_avg, total lost power, peak W/m and peak W/cm².
* **Aperture-profile losses** — the per-element table, now with a
  watts column beside the hit counts.

Both read the results-level loss record, so they work on live *and*
reloaded runs; the exit-plane map additionally needs the final
particle distribution and therefore fills on live MP runs.  The
assistant tool `loss_power` returns the same accounting (summary +
per-element table) on request, and `open_plot("loss power")` opens
the tile.

!!! note "Envelope mode"
    Envelope runs track no per-particle losses; the analysis needs a
    multi-particle run.  Zero-current runs have no beam power — the
    CLI block and the popup wattage column stay silent.
