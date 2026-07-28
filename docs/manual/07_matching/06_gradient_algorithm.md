# Gradient algorithm

The `gradient` matcher is a local Levenberg–Marquardt solve driven by
an **exact** Jacobian from differentiable tracking, instead of the
finite-difference Jacobian `least_squares` uses.

## TL;DR

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching import match

lattice, _ = parse_tracewin("examples/matching_demo.dat")
beam_cfg = BeamConfig(n_particles=1000)

# No space charge (matrix tracking residual):
r = match(lattice, beam_cfg, algorithm="gradient", max_iter=80)
```

```{.python .skip}
# With non-linear PIC space charge (differentiable PIC residual) —
# long-running; needs torch and a beam current > 0:
r = match(lattice, beam_cfg, algorithm="gradient",
          space_charge=True, max_iter=80)
```

In the GUI: **Matching tab → Algorithm = `gradient`** (and tick
**Space charge** to match through the differentiable PIC).

## How it differs from `least_squares`

Both algorithms feed `scipy.optimize.least_squares` (TRF method), so
they solve the *same* optimisation problem with the same trust-region
step rules.  The only difference is the Jacobian:

| algorithm | Jacobian | exactness | typical iter count |
|---|---|---|---|
| `least_squares` | scipy `2-point` finite differences (N+1 forwards per Jacobian) | approximate | reference |
| `gradient` | `torch.autograd.functional.jacobian` on a differentiable residual | exact | ≈ half |

On a single-knob test the two land on identical solutions; the
exact Jacobian just gets there in fewer iterations (12 → 6 in our
benchmark).

## Scope

`gradient` works on:

* **Variables** — ADJUST cards on Quadrupole `gradient`, Solenoid
  `field`, or Dipole `angle`.
* **Constraints** — `SET_TWISS` / `SET_SIZE`.  `SET_SIZE_MAX/MIN`
  and `SET_BEAM_PHASE_ADV` are not supported and the matcher
  raises — use `least_squares` for those.  Centroid constraints
  (`SET_POSITION`, `DIAG_POSITION` targets) are also refused: the
  torch mirror propagates the envelope only, with no first moment
  (no steerer kicks, no misalignment feed-down) — use
  `least_squares`, `bo`, `cmaes` or `sequential_scan`.
* **Elements** — drift, quadrupole, solenoid, dipole, edge (the
  five linear element types).  RF gaps / field maps are
  unsupported and fail loud — also use `least_squares` there.
* **Control cards** — passive `SET_*` / `ADJUST_*` markers pass
  through (the torch mirror composes commands as identity, which is
  faithful for them).  The runtime-active cards `FREQ`,
  `SET_BEAM_ENERGY` and `SET_BEAM_E0_P0` mutate the reference
  kinematics mid-lattice, which the mirror cannot represent — they
  are **refused** (before the 2026-07 honesty round a FREQ deck under
  space charge ran silently on the wrong lattice).  Longitudinal
  `SET_TWISS` flags (`kaz`/`kbz`) are also refused: no tracking mode
  records longitudinal Twiss, so the residual would be a
  zero-gradient constant.

Outside the supported scope the matcher raises a `ValueError`
naming the offending variable / constraint / element.

## With vs without space charge

* **Without SC** — the residual is the end-of-lattice σ matrix from
  `compute_transfer_matrix_torch` (differentiable matrix tracking).
  Cheap; the matrix tracking is essentially free.
* **With SC** — `build_torch_residual_sc` tracks a macro-particle
  bunch through the differentiable PIC step tracker
  (`track_beam_torch_stepwise`) and forms residuals from
  `cov(tracked)`.  This is the only way to match *through* non-
  linear PIC space charge in HELIX — the classic
  `least_squares` + space-charge path uses the *envelope* SC model,
  which is a fundamentally different (linear-ish, analytic)
  physics approximation.  The solve honours `mp_sc_config` (grid,
  extent, kernel) and `mp_n_particles` when you pass them (2026-07:
  they used to be silently ignored here); the defaults are a
  CPU-friendly 32³ grid, ±4σ extent, 1500 particles.  Continuous
  (DC) beams are refused — the torch model has no DC SC branch.

## Performance

Per-Jacobian cost vs number of knobs `N` (no SC, matrix tracking):

| N | finite diff | autograd | speedup |
|---|---|---|---|
| 1  | 0.4 ms  | 1.0 ms | 0.4× — FD wins (autograd overhead) |
| 4  | 2.7 ms  | 3.3 ms | 0.8× |
| 8  | 10 ms   | 7 ms   | **1.4×** |
| 16 | 37 ms   | 13 ms  | **2.8×** |

Crossover ~5 knobs.  With space charge the crossover sits in the
same range and the autograd Jacobian wins ~2.25× at N=8.

Internally the engine calls
`torch.autograd.functional.jacobian(..., vectorize=True)` and
`build_torch_residual_sc` enables gradient checkpointing in the
step tracker — without those the autograd path memory-thrashes on
lattices with ≳ 4 SC-coupled knobs.

## Implementation

* `linac_gen.matching.engine.match` — the `gradient` branch builds
  a torch residual and calls `scipy.least_squares` with the
  autograd Jacobian.
* `linac_gen.matching.torch_objective.build_torch_residual`
  — no-SC residual (matrix tracking).
* `linac_gen.matching.torch_objective.build_torch_residual_sc`
  — SC residual (differentiable PIC step tracker, checkpointed).
* `linac_gen.matching.torch_objective.check_gradient_supported`
  — the scope check that raises a clear error before the solve
  starts.

## Cross-references

* [Overview](01_overview.md)
* [SET / ADJUST cards](02_set_adjust.md)
* [Python API](03_python_api.md)
* [Differentiable PIC](../05_space_charge/06_differentiable.md)
* GUI: [Matching tab](../10_gui/05_matching_tab.md)

← [Recipes](05_recipes.md) ·
[Continue to Emittance minimisation →](07_emittance_min.md)
