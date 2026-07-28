# DC mode (continuous beam)

Pre-RFQ Low-Energy Beam Transports (LEBTs) operate on a continuous,
unbunched beam — there's no longitudinal structure yet.  HELIX has
DC-aware code paths in both the envelope and multi-particle solvers
to handle this regime correctly.

## TL;DR

For any pre-RFQ run:

```python
from linac_gen.core.config import BeamConfig

beam_cfg = BeamConfig(
    species="H-",
    energy=0.030,            # 30 keV LEBT exit
    frequency=162.5,         # downstream RFQ frequency (used for unit conversions)
    current=5.0,
    n_particles=20_000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.20, emit_ny=0.20,
    alpha_x=0.0, beta_x=0.10,
    alpha_y=0.0, beta_y=0.10,
    emit_z=0.0,              # no longitudinal structure
    alpha_z=0.0, beta_z=1.0,
    continuous=True,         # ← critical
    dc_energy_spread_keV=0.0,
)
```

The `continuous=True` flag triggers DC-aware behaviour throughout
the simulator:

* The longitudinal coordinate is **uniformly sampled** over one RF
  period during particle generation (so SC averaging is correct).
* The PIC solver switches to its **2-D analytic / Bassetti-Erskine
  / 2-D PIC** kick (per `dc_kernel`).
* The recorder marks each step with `continuous_at[i] = True` so
  consumers know σ_φ / σ_W / ε_z are not physically meaningful.

## Choosing a DC kernel

`SpaceChargeConfig.dc_kernel` selects the 2-D transverse kick model:

| `dc_kernel` | Field model | Speed | Use when |
|---|---|---|---|
| `"uniform"` | analytic linear uniform-elliptical kick | fastest | matching, fast scans, default |
| `"gaussian"` | Bassetti-Erskine field of a 2-D Gaussian | medium | rigid-σ MP, per-particle non-linearity |
| `"pic2d"` | 2-D Hockney FFT PIC over actual particle distribution | slowest | most accurate; analogue of TraceWin's PICNIC_2D |

```python
from linac_gen.core.config import SpaceChargeConfig

sc = SpaceChargeConfig(
    dc_kernel="gaussian",   # or "uniform" / "pic2d"
)
```

## Common gotchas

!!! warning "Forget `continuous=True` and your LEBT explodes"
    If you forget to set this for a pre-RFQ LEBT, HELIX assumes the
    beam is bunched and imposes a non-physical longitudinal
    structure.  σ_φ blows up, ε_z grows.  Symptoms: looks like SC
    physics is wrong, but it's just a flag.

!!! warning "PXIE LEBT needs Ki=1 + envelope continuous=True"
    Two silent failure modes:
    1. The PXIE LEBT field maps have `Ki=1` (full SC compensation).
       Forgetting this gives the wrong forces.
    2. The envelope solver also needs `continuous=True` to skip
       longitudinal SC.
    Both together: PXIE LEBT mean error 6.93% → 0.84%.

## DC-aware envelope solver

The Sacherer ODE (`linac_gen.tracking.sacherer`) is the DC envelope
counterpart.  It solves the continuous-beam envelope ODE:

$$
\sigma_x'' + \kappa_x(s)\,\sigma_x - \frac{\varepsilon_x^2}{\sigma_x^3}
- \frac{2K}{\sigma_x + \sigma_y} = 0
$$

(and similarly for y) with generalised perveance K = qI / (2π ε₀ m
c³ (βγ)³).  Scope: drifts, hard-edge quadrupoles, 1-D / 3-D solenoid
field maps.  No acceleration.

```python
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.solenoid import Solenoid
from linac_gen.tracking.sacherer import SachererSolver

# LEBT-style solenoid transport line:
lattice = Lattice()
lattice.add(Drift(name="D1", length=200.0))
lattice.add(Solenoid(name="SOL1", length=300.0, field=0.20))
lattice.add(Drift(name="D2", length=400.0))

ref = ReferenceParticle(species=H_MINUS, w_kin=0.033, frequency=162.5)
bg = ref.bg
initial_twiss = dict(emit_x=0.20 / bg, alpha_x=0.0, beta_x=0.5,
                     emit_y=0.20 / bg, alpha_y=0.0, beta_y=0.5)

solver = SachererSolver(lattice, ref, initial_twiss, current=5.0)
res = solver.run()
```

## Cross-references

* [Models](01_models.md) — model decision tree.
* [Tracking modes → DC modes](../02_concepts/03_tracking_modes.md#dc-continuous-beam-modes).
* [Worked example: DC LEBT](../11_examples/03_lebt_dc.md).

← [Kernels](03_kernels.md) ·
[Continue to Convergence guide →](05_convergence.md)
