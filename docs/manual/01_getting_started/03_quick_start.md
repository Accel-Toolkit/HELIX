# Quick start

This page gets you from zero to a working envelope simulation in
under 10 minutes.  Two paths: build a lattice in Python, or open an
existing TraceWin `.dat` file.  Both produce the same `Lattice`
object and route through the same simulation core.

## Path A — Build a lattice in Python

A 2-cell FODO with a 5 mA proton beam and a transverse-only PIC
space-charge solver:

```python
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

# 1.  Reference particle: 3 MeV proton, 352.21 MHz RF
ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)

# 2.  Beam: Gaussian, 10 000 macroparticles, 5 mA
beam_cfg = BeamConfig(
    species="proton",
    energy=3.0, frequency=352.21,
    current=5.0, n_particles=10_000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.25, emit_ny=0.25, emit_z=0.30,
    alpha_x=0.0, beta_x=2.0,
    alpha_y=0.0, beta_y=2.0,
    alpha_z=0.0, beta_z=1.0,
)
beam = create_beam(beam_cfg, seed=42)

# 3.  Lattice: 2 FODO cells (QF — drift — QD — drift) × 2
lattice = Lattice()
for cell in range(2):
    lattice.add(Quadrupole(name=f"QF_{cell}", length=100.0, gradient=+10.0))
    lattice.add(Drift(name=f"D1_{cell}", length=200.0))
    lattice.add(Quadrupole(name=f"QD_{cell}", length=100.0, gradient=-10.0))
    lattice.add(Drift(name=f"D2_{cell}", length=200.0))

# 4.  Run multi-particle tracking with PIC space charge
sc = SpaceChargeConfig(nx=32, ny=32, nz=32, kernel="cic")
sim = Simulation(lattice, beam, space_charge=sc)
results = sim.run()

# 5.  Inspect the diagnostics
print(f"Lattice length:     {results.s[-1] / 1e3:.2f} m")
print(f"Final σ_x:          {results.sigma_x[-1]:.3f} mm")
print(f"Final σ_y:          {results.sigma_y[-1]:.3f} mm")
print(f"Final transmission: {results.transmission[-1]:.2f} %")
```

Expected output:

```
Lattice length:     1.20 m
Final σ_x:          1.245 mm
Final σ_y:          1.250 mm
Final transmission: 100.00 %
```

## Path B — Run an existing TraceWin `.dat`

If you already have a TraceWin lattice file:

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.core.simulation import Simulation

# 1.  Load the lattice.  Header directives are consumed by the parser:
#     FREQ sets each RF element's frequency, PARTRAN_STEP becomes
#     lattice.step_config.  metadata carries "title", "warnings" (a
#     list of non-fatal parse issues) and "lg_options" (;@LG lines).
lattice, metadata = parse_tracewin("examples/pipii/mebt/mebt.dat")
print(f"Loaded {len(lattice.elements)} elements over "
      f"{lattice.total_length / 1e3:.2f} m")

# 2.  Build the beam.  A .dat carries no beam definition, so energy
#     and frequency come from you (here: the PIP-II MEBT values;
#     BeamConfig defaults are 3.0 MeV / 352.21 MHz).
beam_cfg = BeamConfig(
    species="H-",
    energy=2.1,
    frequency=162.5,
    current=5.0,
    n_particles=10_000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.25, emit_ny=0.25, emit_z=0.30,
    alpha_x=-1.99, beta_x=0.20,
    alpha_y=2.21,  beta_y=0.21,
    alpha_z=0.0,   beta_z=1.0,
)
beam = create_beam(beam_cfg, seed=42)

# 3.  Same Simulation API — works on any Lattice
sim = Simulation(lattice, beam)
results = sim.run()
```

HELIX also imports a subset of **MAD-X** lattice files (`.madx` /
`.seq`) — see
[Importing MAD-X lattices](../06_running/02_tracewin_dat.md#importing-mad-x-lattices).

## Path C — Envelope (RMS) tracking

For matching, parameter sweeps, and any case where you don't need
particle-level diagnostics, use the much faster envelope solver:

```python
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON

# Same lattice and beam_cfg as Path A
ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
bg = ref.bg

initial = dict(
    emit_x=beam_cfg.emit_nx / bg, alpha_x=beam_cfg.alpha_x, beta_x=beam_cfg.beta_x,
    emit_y=beam_cfg.emit_ny / bg, alpha_y=beam_cfg.alpha_y, beta_y=beam_cfg.beta_y,
    emit_z=beam_cfg.emit_z,        alpha_z=beam_cfg.alpha_z, beta_z=beam_cfg.beta_z,
)

solver = EnvelopeSolver(lattice, ref, initial, current=5.0)
env_results = solver.run()

print(f"Envelope final σ_x: {env_results.sigma_x[-1]:.3f} mm")
print(f"Envelope took:      {len(env_results.s)} steps")
```

A typical envelope run is **20–100× faster** than the equivalent
multi-particle run — at the cost of losing halo / loss / non-Gaussian
information.  See
[Tracking modes](../02_concepts/03_tracking_modes.md) for guidance.

## Path D — GUI

```bash
./run_gui.sh
```

The GUI gives you the same data flow, but visually:

1. **Lattice tab** → File ▸ Open Lattice → pick a `.dat`
2. **Beam tab** → set species, energy, current, distribution, Twiss
3. **Numerics tab** → choose envelope or MP, set SC parameters
4. Toolbar: **Run**.
5. **Results tab** → click any tile to open a popup plot.

For a tour see [Part X — GUI](../10_gui/01_overview.md).

## What's next?

You now know how to load a lattice, build a beam, and run a
simulation.  Recommended next reading depends on your goal:

| Goal | Read next |
|---|---|
| Understand the coordinate conventions | [Coordinates & units](../02_concepts/01_coordinates.md) |
| Learn what each element does | [Element catalog](../03_elements/00_overview.md) |
| Match Twiss to a periodic lattice | [Matching overview](../07_matching/01_overview.md) |
| Run an alignment-tolerance ensemble | [Errors overview](../08_errors/01_overview.md) |
| Reproduce TraceWin partran output | [Validation](../12_validation/01_tracewin_parity.md) |
| Step-by-step worked example | [Basic FODO](../11_examples/01_basic_fodo.md) |

← Back to [Installation](02_installation.md) ·
[Continue to Concepts →](../02_concepts/01_coordinates.md)
