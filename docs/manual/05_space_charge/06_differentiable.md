# Differentiable PIC (`sc_backend="torch"`)

HELIX ships a second PIC implementation written entirely in PyTorch:
an **end-to-end differentiable** space-charge kick.  It mirrors the
production numpy/C++ PIC (same IGF Hockney FFT Poisson solver, same
CIC/TSC kernels) but every operation is built from `torch` ops, so
gradients flow through the kick via autograd.  This is the engine
behind gradient-based matching *through* non-linear PIC space charge.

## TL;DR

```python
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.simulation import Simulation
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

lattice = Lattice()
lattice.add(Quadrupole(name="QF", length=100.0, gradient=+10.0))
lattice.add(Drift(name="D1", length=200.0))
lattice.add(Quadrupole(name="QD", length=100.0, gradient=-10.0))
lattice.add(Drift(name="D2", length=200.0))
beam = create_beam(BeamConfig(current=5.0, n_particles=2000), seed=42)

sc = SpaceChargeConfig(
    nx=32, ny=32, nz=32,
    grid_extent=4.0,
    kernel="cic",                # or "tsc"
    sc_backend="torch",          # <-- select the differentiable PIC
)
sim = Simulation(lattice, beam, space_charge=sc)
results = sim.run()
```

In the GUI: **Numerics tab → SC engine → `torch`**.

## What it does

The same eight-step PIC cycle as the production engine
([PIC solver](02_pic_solver.md)) — Lorentz boost → grid → deposit →
IGF Hockney FFT → gather → kicks → inverse boost — implemented in
pure `torch.float64` ops on CPU.  No C++ kernel and no GPU dispatch.

## Scope — what it does NOT do

* **Bunched beams only.**  There are no DC/continuous-beam kernels
  in the torch backend.  The GUI detects a continuous beam and
  falls back to `sc_backend="numpy"` automatically.
* **Adaptive grid only.**  The torch kick sizes its grid from the
  current particle distribution at every kick — it ignores
  `SpaceChargeConfig.grid_mode`.  Numerical parity vs the numpy
  backend therefore holds at `grid_mode="adaptive"` (~1e-7); against
  the numpy default `grid_mode="fixed"` the result differs by ~0.5%
  (the genuine fixed-vs-adaptive difference, not a bug).  The GUI
  forces `adaptive` automatically when `torch` is selected.
* **CPU FP64 only.**  The `use_gpu` selector (cpu / cuda / mps) is
  ignored — FP64 is required for parity with the numpy path and
  Apple MPS is FP32-only.

## When to use it

The torch backend is **markedly slower** than the production PIC
(~5× on a small MEBT lattice) for a plain forward run.  Pick it
**only when you need gradients** — i.e. for gradient-based
matching/optimisation through non-linear space charge (see
[Gradient algorithm](../07_matching/06_gradient_algorithm.md)).
For ordinary multi-particle simulations use `sc_backend="numpy"`.

## Numerical parity

| comparison | rtol |
|---|---|
| torch IGF Green tensor vs numpy | 1e-10 |
| torch Poisson solve vs `PoissonSolverFFT` | 1e-9 |
| torch CIC / TSC deposit + gather vs numpy | 1e-12 |
| `TorchPicSolver.kick` vs `PicSolver.kick` (matched grid mode) | 1e-9 |
| End-to-end MP+SC tracking vs numpy `Tracker` | 6e-8 |

Tests under `tests/pic/torch/` and
`tests/tracking/test_torch_step_tracker.py`.

## Implementation

* `linac_gen.pic.torch.poisson.TorchPoissonSolverFFT` — IGF
  Hockney FFT Poisson solver in PyTorch.
* `linac_gen.pic.torch.deposition` / `interpolation` — CIC + TSC
  deposit and gather using `index_put_(accumulate=True)`
  (autograd-safe scatter).
* `linac_gen.pic.torch.sc_kick.torch_pic_sc_kick` — the
  differentiable kick (phase space → coords → adaptive grid →
  deposit → solve → gather → momentum kick).
* `linac_gen.pic.torch.solver.TorchPicSolver` — same `kick(beam, ds)`
  interface as `PicSolver`, selected by `SpaceChargeConfig.sc_backend`.
  **Adaptive-grid only**: it re-fits the grid from the live
  distribution every kick and does not honour `grid_mode="fixed"` —
  such configs get a one-shot `AdaptiveOnlyBackendWarning` because the
  numerical model differs from the numpy fixed-grid solver.
* `linac_gen.tracking.torch_step_tracker.track_beam_torch_stepwise`
  — step-by-step differentiable tracker that interleaves torch
  element maps and the torch SC kick.  Gradient checkpointing is
  available (`checkpoint=True`) for long lattices; the SC residual
  builder enables it automatically.

## Cross-references

* [PIC solver](02_pic_solver.md) — the production numpy / C++ /
  cupy PIC.
* [Models](01_models.md) — picking the right SC model.
* [Gradient algorithm](../07_matching/06_gradient_algorithm.md)
  — gradient matching through the differentiable PIC.
* GUI: [Numerics tab → SC engine](../10_gui/04_convergence_tab.md).

← [Convergence guide](05_convergence.md) ·
[Continue to Running → Python API →](../06_running/01_python_api.md)
