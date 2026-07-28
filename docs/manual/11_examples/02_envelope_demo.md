# Worked example: Envelope demo

The same FODO lattice as [Basic FODO](01_basic_fodo.md), but tracked
with the much faster envelope solver.  Demonstrates the speed/fidelity
trade-off: 100× faster, RMS-only output.

## Source

See `examples/envelope_demo.py`:

```bash
python examples/envelope_demo.py
```

Note the shipped script takes the higher-level route — it builds a
`Simulation` and calls `Simulation.run_envelope()`, which wraps the
`EnvelopeSolver` shown below.  Both paths produce the same results;
use the direct `EnvelopeSolver` API when you want explicit control
over the initial Twiss dict.

## The trick

Same `Lattice`, same Twiss in `BeamConfig`.  Only difference:

```python
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON

# The FODO + beam settings from Basic FODO:
lat = Lattice()
for cell in range(2):
    lat.add(Quadrupole(name=f"QF_{cell}", length=100.0, gradient=+10.0))
    lat.add(Drift(name=f"D1_{cell}", length=200.0))
    lat.add(Quadrupole(name=f"QD_{cell}", length=100.0, gradient=-10.0))
    lat.add(Drift(name=f"D2_{cell}", length=200.0))
beam_cfg = BeamConfig(current=5.0, n_particles=1000)

ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
bg = ref.bg

# Convert normalised → geometric ε for the envelope solver
initial_twiss = dict(
    emit_x=beam_cfg.emit_nx / bg, alpha_x=beam_cfg.alpha_x, beta_x=beam_cfg.beta_x,
    emit_y=beam_cfg.emit_ny / bg, alpha_y=beam_cfg.alpha_y, beta_y=beam_cfg.beta_y,
    emit_z=beam_cfg.emit_z, alpha_z=beam_cfg.alpha_z, beta_z=beam_cfg.beta_z,
    continuous=False,
)

solver = EnvelopeSolver(lat, ref, initial_twiss, current=beam_cfg.current)
env_results = solver.run()
```

The output `env_results` has the same field names as a multi-
particle `DiagnosticRecorder` (s, sigma_x, sigma_y, ...) so the
plot code is the same.

## Compare modes

Run both and overlay:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linac_gen.core.simulation import Simulation
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.distributions.factory import create_beam

# The multi-particle counterpart (small PIC grid for this demo):
beam = create_beam(beam_cfg, seed=42)
sim = Simulation(lat, beam,
                 space_charge=SpaceChargeConfig(nx=16, ny=16, nz=16))
mp_results = sim.run()

s_mp = np.asarray(mp_results.s) / 1e3
s_env = np.asarray(env_results.s) / 1e3

fig, ax = plt.subplots()
ax.plot(s_mp, mp_results.sigma_x, label="MP+PIC σ_x")
ax.plot(s_env, env_results.sigma_x, "--", label="Envelope σ_x")
ax.set_xlabel("s [m]"); ax.set_ylabel("σ_x [mm]"); ax.legend()
plt.close(fig)
```

For a matched beam, the two should overlap to within ~1% noise.
Mismatched beams diverge — envelope's linear assumption breaks.

## When envelope is enough

* Matching loops (where you call `EnvelopeSolver.run()` 1000× in a
  Levenberg-Marquardt iteration).
* Parameter scans (current, focus strength, energy).
* Lattice design first-pass.

## When you need MP

* Halo / loss prediction.
* Non-Gaussian distribution effects.
* Aperture-driven optimisation.
* Final design verification.

## Cross-references

* [Tracking modes](../02_concepts/03_tracking_modes.md)
* [Basic FODO](01_basic_fodo.md)

← [Basic FODO](01_basic_fodo.md) ·
[Continue to DC LEBT →](03_lebt_dc.md)
