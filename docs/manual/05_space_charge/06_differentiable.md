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
`torch.float64` ops, on the CPU or on an NVIDIA GPU (below).  Without a
gradient the CPU's deposit, gather and Green's function run on HELIX's
threads with the same numbers (see Implementation below).

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
* **Device: the CPU, or an NVIDIA GPU on request.**  `use_gpu="cuda"`
  (or `"gpu"`, or `LINAC_GEN_USE_GPU`) runs the whole kick on an NVIDIA
  GPU in FP64; a run repeats bit for bit there too (torch's CUDA deposit
  sums every cell in sorted order), and its results differ from the CPU's
  in the last digits.  A CUDA device torch cannot run on (a wheel without
  kernels for the card) counts as none.  Every
  other setting — `"auto"` included — and a machine without a CUDA device
  run it on the CPU.  Apple GPUs have no FP64: `"mps"` warns and stays on
  the CPU.  The results file records the device
  (`provenance/backend_torch_device`).  `scripts/verify_cuda.py` checks a
  CUDA machine against the CPU.  *Not yet tested on an NVIDIA GPU*: new in
  version 1.17.0, this option has so far only been tested with its code
  running on the CPU (see
  [Installation](../01_getting_started/02_installation.md)).

## When to use it

Pick it **when you need gradients** — i.e. for gradient-based
matching/optimisation through non-linear space charge (see
[Gradient algorithm](../07_matching/06_gradient_algorithm.md)).  For
ordinary multi-particle simulations use `sc_backend="numpy"`: it has
fixed grids, continuous beams and bunch-train images, which the torch
backend lacks.

Speed is no longer a reason to avoid it.  In a plain forward run (no
gradient) its charge deposit, field gather and Green's function run on
HELIX's CPU threads (see [CPU threads](../06_running/15_threads.md)), and
on the FODO test cell with the same 32³ adaptive grid it took 0.32 s for
20,000 particles and 0.68 s for 100,000, against 0.53 s and 0.78 s for the
numpy PIC (14-core Mac, 2026-10-11).

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
* `linac_gen.pic.torch.cpu_kernels` — without a gradient, the deposit
  and gather run in C++ kernels written in torch's arithmetic
  (`linac_gen._tsc_kernels`), and the gather's sums and the Green's
  function run as torch's own code in fixed pieces on HELIX's threads,
  giving torch's numbers bit for bit at any thread count.  Where torch's
  FFT is pocketfft (macOS, ARM machines) the FFTs run in pieces on HELIX's
  threads too, checked once per grid size against torch's own transform;
  on x86 Linux and Windows torch's FFT is Intel MKL's and stays on one
  thread, since the differentiable path must give the same numbers (see
  [CPU threads](../06_running/15_threads.md)).
  `LINAC_GEN_TORCH_SC_KERNELS=0` runs torch's own code everywhere.
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
