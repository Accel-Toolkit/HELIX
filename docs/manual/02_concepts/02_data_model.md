# Data model

HELIX has three core data objects: `Lattice`, `ReferenceParticle`, and
`Beam`.  Every simulation is a function `(Lattice, ReferenceParticle,
Beam, SpaceChargeConfig) → Results`.  This page describes what each
object stores and how they fit together.

## Object diagram

```
                       ┌──────────────────────────────────┐
                       │            Lattice               │
                       │  ─────────────────────────────   │
                       │  elements: list[Element]         │
                       │  errors: list[ErrorDef]          │
                       │  beam_errors: list[BeamErrorDef] │
                       │  step_config: StepConfig         │
                       └──────────────────────────────────┘
                                       │
                                       │ feeds
                                       ▼
┌──────────────────────────┐    ┌────────────────────────┐    ┌─────────────────────────┐
│   ReferenceParticle      │    │   Tracker / Solver     │    │   Beam                  │
│  ──────────────────────  │ ──▶│  ────────────────────  │◀── │  ─────────────────────  │
│  species: Particle       │    │  Tracker (multi-part.) │    │  ref: ReferenceParticle │
│  w_kin: float (MeV)      │    │  EnvelopeSolver        │    │  particles: ndarray     │
│  frequency: float (MHz)  │    │  Sacherer ODE          │    │     shape=(N, 6)        │
│  beta, gamma, bg, phi_s  │    │                        │    │  current: float (mA)    │
└──────────────────────────┘    └────────────────────────┘    │  lost: bool ndarray     │
                                          │                    │  continuous: bool       │
                                          │                    └─────────────────────────┘
                                          ▼
                              ┌──────────────────────────┐
                              │    DiagnosticRecorder    │
                              │  ──────────────────────  │
                              │  s, sigma_x, sigma_y,    │
                              │  sigma_phi, transmission,│
                              │  emit_x/y/z/4d/eigen…    │
                              └──────────────────────────┘
```

## Lattice

`linac_gen.core.lattice.Lattice` is an ordered container of elements
plus a global step configuration and error specs.

```python
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

lattice = Lattice()
lattice.add(Drift(name="D0", length=100.0))
lattice.add(Quadrupole(name="QF1", length=50.0, gradient=10.0))
lattice.add(Drift(name="D1", length=200.0))

print(f"{len(lattice.elements)} elements, "
      f"total length = {lattice.total_length:.1f} mm")
```

### Attributes

| Field | Type | Meaning |
|---|---|---|
| `elements` | `list[Element]` | ordered element list |
| `errors` | `list[ErrorDef]` | element-level error specs (alignment, field) |
| `beam_errors` | `list[BeamErrorDef]` | input-beam jitter specs |
| `error_ratios` | `list[float]` | TraceWin `ERROR_SET_RATIO` values |
| `error_cutoff` | `float` | gaussian truncation in σ |
| `step_config` | `StepConfig` | step density (PARTRAN_STEP): `step_config.integration_steps_per_metre` and `step_config.sc_steps_per_metre` |

### Methods

* `lattice.add(element)` — append an element.
* `lattice.total_length` — sum of element lengths in mm (a
  **property**, not a method — no parentheses).
* `lattice.get_element(name)` — exact-name lookup (raises `KeyError`
  if absent).  There is no pattern-matching `find()`.

### Source

`linac_gen/core/lattice.py:1`

## ReferenceParticle

The synchronous design particle.  Holds energy, RF frequency, and
derived kinematic quantities.

```python
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON

ref = ReferenceParticle(
    species=PROTON,
    w_kin=2.5,            # MeV
    frequency=162.5,      # MHz
)
print(f"β = {ref.beta:.5f}")
print(f"γ = {ref.gamma:.5f}")
print(f"βγ = {ref.bg:.5f}")
```

### Attributes

| Field | Type | Units | Meaning |
|---|---|---|---|
| `species` | `Particle` | — | mass + charge dataclass |
| `w_kin` | `float` | MeV | kinetic energy |
| `frequency` | `float` | MHz | local RF frequency |
| `phi_s` | `float` | deg | synchronous phase |
| `beta` | `float` | — | v/c (computed from w_kin) |
| `gamma` | `float` | — | (1 + w_kin/mc²) |
| `bg` | `float` | — | β·γ (computed) |

### Mutability

`w_kin` and `frequency` are mutable — accelerating elements update
them as the reference particle traverses the lattice.  `beta`,
`gamma`, `bg` are properties that recompute on each access.

### Built-in species

`linac_gen.core.particle` ships:

```python
from linac_gen.core.particle import PROTON, H_MINUS, DEUTERON

print(PROTON.mass, "MeV")        # 938.272088
print(H_MINUS.charge, "e")       # -1.0
print(DEUTERON.mass, "MeV")      # 1875.613
```

You can define a custom species by instantiating `Particle(name=...,
mass=..., charge=...)`.

### Source

`linac_gen/core/reference.py:1`, `linac_gen/core/particle.py:1`

## Beam

A 6-D macroparticle ensemble + reference + current.

```python
from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam

beam_cfg = BeamConfig(
    species="proton",
    energy=2.5, frequency=162.5,
    current=5.0, n_particles=10_000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.21, emit_ny=0.21, emit_z=0.06,
    alpha_x=1.23, beta_x=0.32,
    alpha_y=-0.10, beta_y=0.11,
    alpha_z=0.0, beta_z=819.0,
)
beam = create_beam(beam_cfg, seed=42)

print(f"{beam.n_particles} particles, "
      f"{beam.n_alive} alive, "
      f"current {beam.current} mA")
print(f"shape = {beam.particles.shape}")   # (10000, 6)
```

### Attributes

| Field | Type | Meaning |
|---|---|---|
| `ref` | `ReferenceParticle` | this beam's reference state |
| `particles` | `ndarray (N, 6)` | macroparticle states |
| `lost` | `ndarray (N,) bool` | **stored** loss flags — True once a particle is lost |
| `alive_mask` | property `ndarray (N,) bool` | derived `~lost`; there is no `alive` attribute |
| `n_particles` | `int` | total macroparticle count |
| `n_alive` | `int` | live count (`count_nonzero(~lost)`) |
| `current` | `float` | beam current (mA, peak) |
| `continuous` | `bool` | True for DC pre-RFQ beams |

### Distribution generators

| Distribution | Use case | File |
|---|---|---|
| `"gaussian"` | cutoff-truncated Gaussian | `gaussian.py` |
| `"waterbag"` | default; uniform 6-D ellipsoid | `waterbag.py` |
| `"kv"` | Kapchinskij-Vladimirskij (uniform charge density) | `kv.py` |
| `"parabolic"` | parabolic density profile | `parabolic.py` |
| `"uniform"` | uniform cube in normalised phase space | `uniform.py` |
| `"thermal"` | bi-Gaussian core+halo (PIP-II realistic) | `thermal.py` |

File loading is **not** a distribution: set `source="file"` +
`distribution_file` on the `BeamConfig` to load a TraceWin `.dst`
(binary) or ASCII text file instead of generating.

For full distribution reference see
[Distributions](../04_beam/01_distributions.md).

### Source

`linac_gen/core/beam.py:1`, `linac_gen/distributions/`

## BeamConfig

Dataclass for configuring `create_beam()`.  ~30 fields covering
species, energy, current, distribution choice, Twiss, centroid
offsets, mismatch factors, halo parameters, file source.  The full
field-by-field reference lives at
[BeamConfig reference](../04_beam/03_beam_config.md).

## SpaceChargeConfig

Dataclass for configuring the PIC solver: grid size, kernel choice,
Green's-function flavour, GPU / CPU backend, DC kernel for continuous
beams.  Fully documented at
[Space charge → PIC solver](../05_space_charge/02_pic_solver.md).

## How they connect

The `Simulation` facade wires everything:

```python
from linac_gen.core.simulation import Simulation
from linac_gen.core.config import SpaceChargeConfig

sim = Simulation(
    lattice=lattice,           # Lattice
    beam=beam,                 # Beam (carries ReferenceParticle)
    space_charge=SpaceChargeConfig(),
)
results = sim.run()            # multi-particle tracking
# OR
env_results = sim.run_envelope()  # envelope tracking
```

`Simulation.run()` constructs a `Tracker`, which dispatches each
element through its `track(beam)` method, recording diagnostics each
step into a `DiagnosticRecorder`.  The recorder is returned as
`results` — that's the object you plot from.

For the full API see [Python API](../06_running/01_python_api.md).

← [Back to Coordinates](01_coordinates.md) ·
[Continue to Tracking modes →](03_tracking_modes.md)
