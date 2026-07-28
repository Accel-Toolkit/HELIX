# Interpreting ensemble results

What to look for in `ErrorStudyResults`, and how to translate the
output into engineering decisions ("yes, this lattice is robust to
0.2 mm quad alignment" or "we need a tighter tolerance").

## TL;DR — three plots to always make

1. **σ-band envelopes** — mean ± 1σ of σ_x, σ_y vs s.  Tells you
   how the *typical* beam evolves under errors.
2. **Transmission histogram** — distribution of final transmission
   over seeds.  Tells you the *probability* of beam loss.
3. **Maximum-excursion vs s** — max σ_x, σ_y at every s over all
   seeds.  Tells you the *worst-case* aperture demand.

The GUI's Errors-tab popup produces all three automatically.

## σ-band envelopes

Plot mean σ_x with shaded ±1σ band:

```python
import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt

# The plots below use `results` from a (here: small, 2-seed) study:
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.errors.error_model import ErrorStudy

lat = Lattice()
for i in range(1, 5):
    lat.add(Quadrupole(name=f"QUAD_{i:02d}", length=100.0,
                       gradient=10.0 * (-1) ** i))
    lat.add(Drift(name=f"D_{i}", length=150.0))
study = ErrorStudy(lat, BeamConfig(n_particles=500), n_seeds=2)
study.add_error("QUAD_*", "dx", distribution="gaussian", sigma=0.2)
results = study.run()

s_m = results.mean("s") / 1000.0            # recorder s is in mm
fig, ax = plt.subplots()
ax.plot(s_m, results.mean("sigma_x"), color="C0", label="mean σ_x")
ax.fill_between(s_m,
                results.percentile("sigma_x", 16),
                results.percentile("sigma_x", 84),
                color="C0", alpha=0.3, label="16-84 percentile")
ax.set_xlabel("s [m]"); ax.set_ylabel("σ_x [mm]")
ax.legend()
```

What to look for:

* **Bandwidth** — wide bands ⇒ beam is sensitive to the errors.
  Narrow bands ⇒ robust.
* **Outliers** — if a few seeds have large excursions while most
  are tight, the lattice has *brittle* failure modes.

## Transmission histogram

```python
# Final transmission per seed, from the per-seed recorders.
trans = [r.transmission[-1] for r in results._recorders]
ax.hist(trans, bins=20)
ax.set_xlabel("transmission [%]")
ax.set_ylabel("seeds")

print(results.transmission_stats())   # {"mean", "min", "max", "std"}
```

What to look for:

* **Tail at low transmission** — even rare losses below 99 % can be
  catastrophic for high-power machines (radiation, activation).
  PIP-II target is 99 % across the *worst* seed.
* **Bimodal distribution** — two clusters: "normal" + "catastrophic"
  — usually means a specific error draw triggers a loss
  cascade.  Investigate which seed and what its draw was.

## Worst-case excursion

```python
# The 100th percentile across seeds is the per-s maximum.
ax.plot(s_m, results.percentile("sigma_x", 100), label="max σ_x")
ax.plot(s_m, results.percentile("sigma_y", 100), label="max σ_y")
ax.set_xlabel("s [m]"); ax.set_ylabel("max σ over all seeds [mm]")
```

What to look for:

* **Peaks** at specific elements ⇒ that element is the bottleneck.
  Check: is the design aperture safe even at the peak?
* **Trend over s** — if the max grows linearly, errors are
  accumulating.  Steerer-correction studies are the next step.

## Engineering decisions

| Observation | What to do |
|---|---|
| 99-percentile σ exceeds aperture by < 20% | tighten tolerance: re-run with σ_dx halved |
| Transmission histogram shows tail < 99% | find the worst seed, identify which error draw causes it, address that specific failure mode |
| One element has the peak in max-excursion | local intervention: tighter alignment spec for that one |
| All seeds look similar → narrow bands | lattice is robust; you can move on |

## Worst-seed forensics

To identify which random draw caused the worst seed:

```python
import numpy as np

trans = [r.transmission[-1] for r in results._recorders]
worst_seed = int(np.argmin(trans))
print(f"Worst seed: {worst_seed}")
print(f"  transmission: {trans[worst_seed]:.2f} %")

# The draws are deterministic per seed: regenerate that seed's
# perturbed lattice copy and diff it against the nominal one.
bad_lattice = study._apply_errors(study.base_seed + worst_seed)
for nom, bad in zip(lat.elements, bad_lattice.elements):
    for attr in ("dx", "dy", "tilt_deg", "gradient"):
        v0 = getattr(nom, attr, None)
        v1 = getattr(bad, attr, None)
        if v0 is not None and v1 is not None and v0 != v1:
            print(f"  {nom.name}.{attr}: {v0} -> {v1}")
```

The worst seed's full diagnostics are already in
`results._recorders[worst_seed]` — plot its `sigma_x` /
`transmission` arrays directly to see exactly where the loss
happened.

## Cross-references

* [Running studies](05_running_studies.md)
* [Worked example: tolerance study](../11_examples/08_tolerance_study.md)
* [Aperture profile](../09_diagnostics/04_aperture.md) — where the losses landed.

← [Running studies](05_running_studies.md) ·
[Continue to Orbit correction →](07_correction.md)
