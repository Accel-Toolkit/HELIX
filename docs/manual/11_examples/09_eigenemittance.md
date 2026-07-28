# Worked example: Eigenemittance

A coupled FODO + sextupole + octupole + bi-Gaussian halo lattice
that demonstrates **all six emittances** on the same lattice.
Shows when ε_x, ε_y oscillate while the eigenemittances ε_E1, ε_E2,
ε_E3 stay invariant.

## Source

`examples/impactx_features_demo.py` and the lattice in
`examples/impactx_features/coupled_fodo.dat`:

```bash
python examples/impactx_features_demo.py
```

## Lattice features

* **Coupled FODO** — solenoid + skew quadrupole couples (x, x') with
  (y, y').
* **Thin sextupole and octupole** — exercise the Multipole element.
* **Thermal halo distribution** — bi-Gaussian with halo_fraction=0.05.

## What to plot

Six emittances vs s on the same axes:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam

lat, _ = parse_tracewin("examples/impactx_features/coupled_fodo.dat")
beam = create_beam(BeamConfig(n_particles=2000), seed=1)
results = Simulation(lat, beam).run()

fig, ax = plt.subplots()
ax.plot(results.s, results.emit_x, label="ε_x (projected)", lw=0.8)
ax.plot(results.s, results.emit_y, label="ε_y (projected)", lw=0.8)
ax.plot(results.s, results.emit_4d, label="ε_4D (4-D invariant)", lw=1.2)
ax.plot(results.s, results.emit_n1, label="ε₁ (4-D normal-mode)", lw=1.2)
ax.plot(results.s, results.emit_n2, label="ε₂ (4-D normal-mode)", lw=1.2)
ax.plot(results.s, results.emit_e1, "--", label="ε_E1 (6-D Balandin)")
ax.plot(results.s, results.emit_e2, "--", label="ε_E2 (6-D Balandin)")
ax.plot(results.s, results.emit_e3, "--", label="ε_E3 (6-D Balandin)")
ax.legend(); ax.set_xlabel("s [mm]"); ax.set_ylabel("ε [mm·mrad]")
```

## What you should see

* **ε_x, ε_y** oscillate strongly through the solenoid (coupling
  artefact).
* **ε_4D** stays nearly flat (4-D invariant — captures the x-y
  coupling but not dispersion).
* **ε₁, ε₂** stay flat (the *normal-mode* emittances — these are
  the textbook "real" transverse emittances).
* **ε_E1, ε_E2, ε_E3** stay flat (full 6-D — invariant under all
  linear coupling).

For an ideal lattice with no growth, the eigenemittances are
*constants of motion*.  The projected ε_x, ε_y are not.

## When does this matter?

* **LEBT / solenoid-rich lines** — projected ε_x, ε_y wobble
  badly; eigenemittances are the only meaningful invariant.
* **High-dispersion regions** — ε_4D oscillates; need the full
  6-D ε_E1/2/3.
* **PIP-II SRF cryomodules** with quadrupole-doublet focusing — same
  story.

## Theory

Per V. Balandin, W. Decking, N. Golubeva, "On the calculation of
generalized four-dimensional and six-dimensional eigen-emittances"
(IPAC 2013 / arXiv:1305.1532), the 6-D eigenemittances are computed
from the *trace invariants* of (J·Σ_6D):

$$
I_2 = -\tfrac{1}{2}\,\mathrm{tr}\!\left((J\Sigma)^2\right),\quad
I_4 = +\tfrac{1}{2}\,\mathrm{tr}\!\left((J\Sigma)^4\right),\quad
I_6 = -\tfrac{1}{2}\,\mathrm{tr}\!\left((J\Sigma)^6\right)
$$

These combine into the cubic

$$
x^3 - I_2\,x^2 + \frac{I_2^2 - I_4}{2}\,x
- \left(\frac{I_2^3}{6} - \frac{I_2 I_4}{2} + \frac{I_6}{3}\right) = 0
$$

whose roots are ε_E1², ε_E2², ε_E3².

See [Emittances](../09_diagnostics/02_emittances.md) and the source
file `linac_gen/diagnostics/eigenemittance.py:1`.

## Cross-references

* [Emittances](../09_diagnostics/02_emittances.md)
* [Multipole](../03_elements/11_multipole.md)
* [Distributions → thermal](../04_beam/01_distributions.md#thermal-bi-gaussian-halo)

← [Tolerance study](08_tolerance_study.md) ·
[Continue to Validation → TraceWin parity →](../12_validation/01_tracewin_parity.md)
