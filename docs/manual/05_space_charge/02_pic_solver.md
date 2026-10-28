# PIC solver

The 3-D Particle-In-Cell solver is HELIX's main space-charge engine
for bunched, multi-particle tracking.  It solves Poisson's equation
in the beam rest frame on a regular Cartesian grid via doubled-grid
Hockney FFT convolution.

## TL;DR

```python
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

# A small FODO + 5 mA beam to feel the space charge:
lattice = Lattice()
for c in range(2):
    lattice.add(Quadrupole(name=f"QF_{c}", length=100.0, gradient=+10.0))
    lattice.add(Drift(name=f"D1_{c}", length=200.0))
    lattice.add(Quadrupole(name=f"QD_{c}", length=100.0, gradient=-10.0))
    lattice.add(Drift(name=f"D2_{c}", length=200.0))
beam = create_beam(BeamConfig(current=5.0, n_particles=2000), seed=42)

sc = SpaceChargeConfig(
    nx=96, ny=96, nz=96,        # the defaults
    grid_extent=5.0,            # ±5σ box
    kernel="cic",               # particle-mesh shape: "cic" or "tsc"
    green_kind="igf",           # Green's function: "igf" or "point"
    use_gpu="auto",             # "cpu" / "gpu" / "cuda" / "mps" / "auto"
)
sim = Simulation(lattice, beam, space_charge=sc)
results = sim.run()
```

A whole combination of these settings can be chosen by name -- e.g.
`SpaceChargeConfig(method="tsc_adaptive")` or `--sc-method tsc_adaptive`
on the command line; see [Space-charge methods](08_methods.md).

Reasonable defaults:

* Grid: the default is `nx=ny=nz=96` (1% σ convergence at 5 mA);
  drop to 64 for quick tests, 128+ for publication quality.
* Extent: `5.0` σ-multiples — captures ~99.99% of a Gaussian.
* Kernel: `"cic"` for legacy compatibility, `"tsc"` for noise-
  sensitive long no-RF transports (BTL-style).
* Green: `"igf"` (always — `"point"` is legacy reproduction only).

Further `SpaceChargeConfig` knobs:

| Field | Default | Values | Meaning |
|---|---|---|---|
| `boundary` | `"open"` | `open` | Poisson boundary condition — only open-BC (Hockney doubled-grid) is implemented; `"periodic"` is refused at validation (2026-07: it used to be accepted and silently ignored) |
| `solver` | `"fft"` | `fft` | Poisson solver flavour (only FFT is implemented) |
| `shape_order` | `1` | `1` | deprecated no-op — deposition order is chosen by `kernel` (`cic`/`tsc`); any value ≠ 1 warns |
| `grid_mode` | `"fixed"` | `fixed` \| `adaptive` | `fixed` = grid frozen after the **first** kick; `adaptive` = grid re-sized to ±`grid_extent`·σ on **every** kick |
| `method` | `None` | a [method](08_methods.md) name, `custom`, `default` | names a whole combination of settings; after construction it names the method the settings match |
| `green_build` | `"exact"` | `exact` \| `symmetric` | how the integrated Green's function is built: every vertex, or one octant mirrored (equal to rounding, about 5× faster; matters on adaptive grids) |
| `grid_card` | `"apply"` | `apply` \| `ignore` \| `auto` | what a `HELIX_SC_GRID` card does; `auto` ignores it on an adaptive grid |
| `reuse_every` | `1` | integer ≥ 1 | adaptive numpy grid only: solve the field every K-th kick and rescale it in between (see [Space-charge methods](08_methods.md#field-solved-every-k-th-kick-reuse_every)) |

## The 8-step PIC cycle

At every integration step, the PIC solver:

1. **Lorentz boost on z** to enter the beam rest frame (γ on z only,
   transverse coordinates unchanged).
2. **Set up grid** — automatically sizes the box to ±`grid_extent`·σ
   in each dimension (or fixed if `grid_mode="fixed"`).
3. **Charge deposition** on the grid — either CIC (8-corner trilinear,
   first-order) or TSC (27-cell quadratic, second-order).
4. **Doubled-grid FFT convolution** with one of two Green's
   functions:
   * **IGF** (Integrated Green Function, default) — closed-form
     8-point cell-difference of the analytic Coulomb antiderivative.
     Symmetric, free of grid-resolution bias near the source
     (Qiang et al. 2006).
   * **point** (legacy) — sampled 1/(4πε₀ r) with self-potential
     regulariser at r = 0.
5. **E-field gather** to particles (matching deposit kernel —
   CIC or TSC).
6. **Transverse momentum kick** — uses the standard Lorentz
   cancellation: net lab-frame transverse force = q·E_⊥_rest / γ;
   the β²γ² factor in the kick formula bundles the γ from force
   cancellation with the βγ from p_z conversion.
7. **Longitudinal energy kick** — q·E_z_rest applied directly to the
   lab-frame ΔW (E_z is invariant under the z-boost).
8. **No explicit boost-back** — the kicks are written straight into
   the lab-frame x′/y′/ΔW columns (the frame transformation is
   folded into the β²γ² factor of step 6), and the reference
   particle is never modified.

Reference for the algorithm: Qiang, Lidia, Ryne, Limborg-Deprey,
*PRSTAB* 9, 044204 (2006); erratum 10, 129901 (2007) — same kernel used by
Cheetah and OPAL.

![PIC cycle](../_build/figures/fig_05_02_pic_cycle.png)

*Figure 5.2 — The 8-step PIC cycle as implemented in HELIX.  Steps 1
and 8 are the Lorentz boosts that bracket the rest-frame Poisson
solve.*

## CPU / GPU backend

The FFT step of every PIC kick is dispatched through
`linac_gen.pic.gpu_backend.FftBackend`.  Three concrete backends are
shipped:

| Backend | Module | Precision | Runs on |
|---|---|---|---|
| `_CpuBackend` | scipy.fft (or numpy.fft fallback) | FP64 | host CPU, on the [CPU threads](../06_running/15_threads.md) setting |
| `_GpuBackend` | cupy / cupyx.scipy.fft | FP64 | NVIDIA CUDA device |
| `_MpsBackend` | torch.fft on MPS | FP32 (Metal hardware) | Apple Silicon GPU |

### Selecting a backend

| `use_gpu=` | meaning |
|---|---|
| `"auto"` | cupy (FP64) if available, else CPU — **never** MPS |
| `"gpu"`  | any GPU available (cupy preferred, MPS fallback), error if neither |
| `"cuda"` | force cupy; error if missing |
| `"mps"`  | force torch.mps; error if missing |
| `"cpu"`  | force scipy.fft on host |

!!! warning "MPS is float32 and requires explicit opt-in"
    Metal hardware has no FP64, so the MPS backend computes the FFTs in
    float32 (~1e-7 relative field error, not float64-reproducible).
    `"auto"` therefore never selects it — on an Apple-Silicon machine
    without CUDA, `"auto"` runs on the (equally fast, full-precision)
    CPU backend.  Requesting MPS explicitly (`"mps"`, or `"gpu"` with no
    CUDA device) works but emits a `UserWarning` stating the precision
    cost.  *(Changed 2026-07: previously `"auto"` silently picked MPS.)*

The env var `LINAC_GEN_USE_GPU` takes precedence over the argument and
accepts the same set of values (plus `0/1` legacy aliases).  The GUI
exposes the same options in the Numerics tab's **PIC backend**
dropdown.

### Numerical parity

* `cuda` vs `cpu` — `~1e-10` relative on σ (FP64 FFTs on both paths).
* `mps`  vs `cpu` — `~6e-7` relative on σ (FP32 FFT round-off floor).
  Far below the PIC macroparticle shot noise (`~1/√N` ≈ `1e-3` for
  `N=10⁴`), so MP+SC residuals vs TraceWin are unchanged.  For runs
  where you need sub-FP32 numerical fidelity (e.g. publication figures
  comparing run-to-run determinism), force `LINAC_GEN_USE_GPU=cpu`.

### Crossover — NVIDIA

For grids ≤ 96³ on WSL2-class PCIe, host↔device transfer cost can
exceed the FFT speedup.  Measured (RTX 2000 Ada, 16-thread scipy.fft):

| grid | CPU | GPU (cuda) | speedup |
|---|---|---|---|
| 48³ | 7.6 ms | 4.9 ms | 1.6× |
| 64³ | 17.5 ms | 11.7 ms | 1.5× |
| 96³ | 41.7 ms | 55.5 ms | 0.8× — CPU wins |
| 128³ | 100.6 ms | 128.7 ms | 0.8× — CPU wins |

If your workload uses ≥96³ grids on WSL2, force `"cpu"` or set
`LINAC_GEN_USE_GPU=0`.

### Crossover — Apple Silicon (MPS)

Apple unified memory makes the host↔device copy effectively free, so
small grids stay competitive.  PyTorch 2.10's inverse-real-FFT path on
MPS is currently slower than scipy.fft (known upstream issue), so the
full Poisson round-trip is roughly neutral on small grids and only the
forward-FFT half clearly wins.  Measured (M3 Max, padded 256³ FFT,
FP32 vs scipy.fft FP64):

| step | scipy.fft (CPU) | torch.fft (MPS) | speedup |
|---|---|---|---|
| `rfftn` 256³ | 7.9 ms | 2.6 ms | **3.0×** |
| `irfftn` 256³ | 11.0 ms | 42.3 ms | 0.26× (upstream issue) |
| **round-trip** | 18.9 ms | 44.9 ms | net negative on this grid |

End-to-end MEBT+HWR MP+SC (N=10000, 64³ grid) wall-clock:

| backend | wall-clock | MPS peak alloc | TraceWin parity |
|---|---|---|---|
| CPU | 97.2 s | 0 MiB | 4.07% mean σ_x |
| MPS | 90.1 s | 108 MiB | 4.07% mean σ_x (same at displayed precision; FP32 ⇒ ~6e-7 relative on σ, *not* bit-identical) |

The MEBT+HWR speedup is modest because deposit/interp and per-element
overhead dominate; lattices with proportionally more PIC kicks
(full PIP-II, long BTL transport) gain more.

## Implementation

`linac_gen.pic.pic_solver.PicSolver` is the entry point.  Internally:

* `linac_gen/pic/charge_deposition.py` — `deposit_cic` /
  `deposit_tsc`, particle → grid.
* `linac_gen/pic/poisson_solver.py` — `PoissonSolverFFT` (Hockney
  FFT solver) with the IGF antiderivative `_antideriv_igf`.
* `linac_gen/pic/field_interpolation.py` — `interpolate_cic` /
  `interpolate_tsc`, grid → particle.
* `linac_gen._pic_kernels` (C++/pybind11) — fast CIC
  deposit/interpolate paths.
* `linac_gen._tsc_kernels` (C++/pybind11) — TSC deposit/interpolate,
  bit-identical to the Python TSC functions.

The C++ kernels are built automatically by pip during install.  If
`_pic_kernels` is not importable, HELIX logs a warning and falls back
to the pure-Python CIC in `charge_deposition.py` /
`field_interpolation.py` (numerically equivalent, slower).  If
`_tsc_kernels` is missing, HELIX logs one line and runs the Python
TSC, with identical results; `LINAC_GEN_TSC_KERNEL=0` forces that path.
Calls outside the compiled TSC kernel's proven envelope (an empty beam,
fewer than 3 nodes per axis, non-finite coordinates or bounds, zero
spacing, inputs that are not float64) always run the Python TSC.

`PoissonSolverFFT` builds the IGF from the antiderivative at the shared
cell vertices (see [Kernels](03_kernels.md#build-cost)).  In adaptive
mode, `update_grid` rebuilds it only when the cell spacing changes.
The differentiable torch kick (`linac_gen.pic.torch.sc_kick`) uses the
same vertex build when no gradient flows through the grid spacing.  For
plain tensors it also reuses the solver of an identical grid from a
small cache (at most 4 grids and 128 MiB); under any gradient, a
`torch.func` transform, tracing or compilation it builds a fresh solver,
and `TorchPicSolver.free_memory()` empties the cache.  With a gradient
through the spacing it builds the original per-corner graph, so
gradients are unchanged.

## Cross-references

* [Models](01_models.md) — broader SC-model decision.
* [Kernels](03_kernels.md) — CIC vs TSC vs IGF vs point.
* [Convergence guide](05_convergence.md) — grid sizing.
* `linac_gen/pic/pic_solver.py:1` — source.

← [Models](01_models.md) ·
[Continue to Kernels →](03_kernels.md)
