# Worked example: DC LEBT

A pre-RFQ Low-Energy Beam Transport (LEBT) operating on a continuous
H⁻ beam from the source.  Demonstrates HELIX's DC-mode handling:
`continuous=True`, 2-D analytic SC kick, no longitudinal physics.

## Source

Mirrors `examples/lebt_dc_demo.py`:

```bash
python examples/lebt_dc_demo.py
```

## The setup

A 30 keV H⁻ beam through two solenoids:

```python
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.drift import Drift
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation

beam_cfg = BeamConfig(
    species="H-",
    energy=0.030,                   # 30 keV (LEBT entrance)
    frequency=162.5,                # downstream RFQ frequency
    current=5.0,
    n_particles=10_000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.20, emit_ny=0.20,
    alpha_x=0.0, beta_x=0.10,
    alpha_y=0.0, beta_y=0.10,
    emit_z=0.0,                     # no longitudinal structure
    alpha_z=0.0, beta_z=1.0,
    continuous=True,                # ← critical
)
beam = create_beam(beam_cfg, seed=42)

lat = Lattice()
lat.add(Drift(name="D0", length=200.0))
lat.add(Solenoid(name="SOL1", length=300.0, field=0.20))
lat.add(Drift(name="D1", length=400.0))
lat.add(Solenoid(name="SOL2", length=300.0, field=0.20))
lat.add(Drift(name="D2", length=200.0))

# DC SC: 2-D analytic kick
sc = SpaceChargeConfig(dc_kernel="gaussian")
sim = Simulation(lat, beam, space_charge=sc)
results = sim.run()
```

## What's special

1. **`continuous=True`** in BeamConfig — turns on DC-aware
   physics throughout.
2. **`emit_z=0`** — DC beams have no longitudinal Twiss.
3. **`dc_kernel="gaussian"`** in SpaceChargeConfig — Bassetti-Erskine
   2-D analytic SC kick instead of 3-D PIC.
4. **σ_φ, σ_W, ε_z are not physical** in the recorder — flagged
   by `results.continuous_at[i] == True`.

## ε oscillates through solenoids

Plot `results.emit_x` vs s — you'll see big oscillations through
each solenoid, because the projected ε is not invariant under
solenoid rotation.  Compare to `results.emit_n1` (the eigenemittance)
which stays smooth — that's the *real* invariant.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(results.s, results.emit_x, label="ε_x (projected)")
ax.plot(results.s, results.emit_n1, label="ε₁ (eigen)")
ax.legend(); ax.set_xlabel("s [mm]"); ax.set_ylabel("ε [mm·mrad]")
```

## Cross-references

* [DC mode](../05_space_charge/04_dc_mode.md)
* [Solenoid](../03_elements/03_solenoid.md)
* [Eigenemittances](../09_diagnostics/02_emittances.md)

← [Envelope demo](02_envelope_demo.md) ·
Continue to LEBT + RFQ →
