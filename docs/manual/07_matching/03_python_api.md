# Matching Python API

For programmatic matching, HELIX exposes `match()` plus the
`Variable` / `Constraint` / `MatchResult` dataclasses, all importable
from `linac_gen.matching`.

## TL;DR

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching import match

# The lattice carries the ADJUST_* / SET_* cards; match() reads them.
from linac_gen.core.config import BeamConfig
lattice, _ = parse_tracewin("examples/matching_demo.dat")
beam_config = BeamConfig(n_particles=1000)
result = match(lattice, beam_config, algorithm="least_squares")

print(result.report())              # human-readable summary
print(f"matched x : {result.x_final}")
```

`match()` **mutates the lattice and `beam_config` in place** — after a
run their parameters carry the matched values.

## `match()` function

```python
def match(
    lattice,
    beam_cfg: BeamConfig,
    *,
    space_charge: bool = False,
    algorithm: str = "least_squares",
    max_iter: int = 200,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
    # CMA-ES
    cmaes_sigma: float = 0.2,
    cmaes_popsize: int = 0,
    cmaes_parallel: int = 1,
    cmaes_search_solver: str = "auto",
    refine: bool = True,
    # Cost-solver (forward-pass model)
    cost_solver: str = "envelope",
    mp_n_particles: int = 1000,
    mp_seed: int = 42,
    mp_sc_config=None,
    mp_step_config=None,
    # Sequential-scan algorithm
    seqscan_element_names: list[str] | None = None,
    seqscan_passes: int = 2,
    seqscan_steps: int = 7,
    seqscan_step_frac: float = 0.10,
    seqscan_reversal: str = "both_grew",
    seqscan_threshold: str = "input",
    # Progress callback
    callback=None,
) -> MatchResult: ...
```

### Core kwargs

| Argument | Default | Meaning |
|---|---|---|
| `lattice` | — | lattice with `ADJUST_*` / `SET_*` cards; mutated in place |
| `beam_cfg` | — | initial `BeamConfig`; `ADJUST_BEAM_*` mutations land here |
| `space_charge` | `False` | run forward pass at `beam_cfg.current` (slower) |
| `algorithm` | `"least_squares"` | optimiser — see below |
| `max_iter` | `200` | `least_squares`: max function evals; population algorithms: max generations |
| `xtol`, `ftol` | `1e-8` | tolerances for `least_squares` (others use scipy defaults) |
| `callback` | `None` | `callback(iter, x, cost[, info])` per evaluation — 4-arg signature gets a dict with live ε, W_kin, scan context |

### CMA-ES kwargs

| Argument | Default | Meaning |
|---|---|---|
| `cmaes_sigma` | `0.2` | initial step as fraction of bound width |
| `cmaes_popsize` | `0` | population size; `0` = library default `4 + ⌊3·ln N⌋` |
| `cmaes_parallel` | `1` | evaluate population across N processes; `0` = auto-detect `min(popsize, cpu-1)`.  Only worth >1 when per-eval cost is large (FieldMap3D lattices). |
| `cmaes_search_solver` | `"auto"` | objective the parallel workers score the search with: `"auto"` follows `cost_solver` faithfully (parallel + mp genuinely evaluates the mp objective — validated to reproduce serial mp exactly); `"envelope"` = the fast envelope-guided hybrid (envelope-scored search; baseline, polish and final residuals still use the requested cost solver) as an explicit opt-in.  Ignored when `cmaes_parallel == 1`. |
| `refine` | `True` | after CMA-ES **or bayesopt** converges, run a least_squares polish on the best point |

### Bayesian-optimisation kwargs

| Argument | Default | Meaning |
|---|---|---|
| `bo_prior` | `False` | `bayesopt` only: physics-informed warm start — scout the box with the cheap envelope cost and seed the GP's initial design with the envelope-good points.  Only has effect with `cost_solver="mp"` (it warm-starts the expensive MP match from the cheap-physics region). |
| `bo_n_init` | `0` | `bayesopt` only: number of initial space-filling (Sobol) samples before the GP is first fit.  `0` = default `min(2N+2, 12)`. |

`bayesopt` reuses `max_iter` (number of Bayesian iterations) and `refine`
(LS polish from the BO best).  Requires `botorch` + `gpytorch`.

### Cost-solver kwargs (envelope vs MP)

The cost function is computed by a forward pass through the lattice.
You can pick which forward-pass model drives that cost:

| Argument | Default | Meaning |
|---|---|---|
| `cost_solver` | `"envelope"` | `"envelope"` = fast linear matrix Σ-propagation (default).  `"mp"` = full multi-particle tracking through `Simulation` — physics-accurate but **50–100× slower per eval**. |
| `mp_n_particles` | `1000` | MP cost-solver: macroparticle count.  Lower = faster but noisier cost. |
| `mp_seed` | `42` | MP cost-solver: fixed RNG seed so each evaluation at the same `x` produces the same cost (no stochastic chatter). |
| `mp_sc_config` | `None` | `SpaceChargeConfig` for the MP path.  When `None` and `space_charge=True`, uses library defaults; the GUI passes the Numerics tab's config. |
| `mp_step_config` | `None` | `StepConfig` (integration / SC kicks per metre) for MP. |

Pick MP for the final validating match on a converged envelope solution,
or when nonlinear / halo physics drives the cost.  Wall time for
CMA-ES with popsize ≈ 11 and `max_iter=80` runs envelope in minutes,
MP in hours.

### Sequential-scan kwargs

Used only when `algorithm="sequential_scan"`.  See
[Sequential-scan](07_emittance_min.md#sequential-scan) for the
algorithm description.

| Argument | Default | Meaning |
|---|---|---|
| `seqscan_element_names` | `None` | filter: only scan elements with these names.  `None` = scan all elements with ADJUST cards. |
| `seqscan_passes` | `2` | full passes through the element list |
| `seqscan_steps` | `7` | bracket steps per parameter per pass |
| `seqscan_step_frac` | `0.10` | per-step displacement = this × (vmax − vmin) |
| `seqscan_reversal` | `"both_grew"` | `"both_grew"` = flip direction only when BOTH transverse and longitudinal εn exceed the reference (allows plane-exchange).  `"any_grew"` = stricter; flip on either. |
| `seqscan_threshold` | `"input"` | reference for the reversal check.  `"input"` compares trial exit ε to BEAM INPUT ε (tight — natural for coupling-resonance lattices where plane exchange is OK as long as both stay below input).  `"seed_exit"` compares to NOMINAL/unmatched lattice's exit ε (natural for ε minimisation with intrinsic growth — reversal fires only when trial is *worse than baseline*). |
| `seqscan_reject_loss` | `False` | **Hard rejection** of loss-inducing steps.  When `True` + `cost_solver="mp"`: after each step, if trial transmission falls below `seqscan_loss_threshold_pct`, the step is rolled back (best_x is NOT updated to it, even if cost dropped) and direction is flipped.  Closes the ε-gaming gap where killing halo would otherwise look like an improvement.  Default `False` preserves historical behaviour. |
| `seqscan_loss_threshold_pct` | `100.0` | Transmission floor for the rejection check.  `100.0` = any drop below 100% rejects; `99.9` = tolerate sub-permille losses; `99.0` = up to 1%.  Inert in env mode (no transmission tracked). |

Variables and constraints are always collected from the lattice cards;
`match()` does not accept them as arguments.

### Algorithms

`linac_gen.matching.MATCH_ALGORITHMS` lists the choices:

| `algorithm` | Kind | Notes |
|---|---|---|
| `"least_squares"` | local, gradient-based | Levenberg–Marquardt (`method='trf'`); fast; the default; handles open-ended bounds |
| `"differential_evolution"` | global, gradient-free | explores the whole bounded box; **needs finite bounds** on every ADJUST variable |
| `"dual_annealing"` | global, simulated annealing | as above; **needs finite bounds** |
| `"cmaes"` | global-ish, gradient-free | Covariance Matrix Adaptation Evolution Strategy.  Handles correlated variables natively (great for cavity ke+phase pairs).  `refine=True` chains a final LS polish.  Best general-purpose pick for ~5–30 variable matching problems. |
| `"sequential_scan"` | physics-aware coordinate descent | walks elements in lattice order, bracket-scans each ADJUST parameter (FieldMap `kb`/`ke`/`phase` grouped), reverses direction on the configured emittance threshold.  Strong on FieldMap-heavy lattices.  See `seqscan_*` kwargs above. |
| `"bayesopt"` | global, sample-efficient | Gaussian-process Bayesian optimisation (BoTorch `SingleTaskGP` + `qLogExpectedImprovement`).  The accelerator-tuning state of the art (SLAC Xopt/Badger).  **Fewest evaluations** of any global — the right pick for **expensive** matches (`cost_solver="mp"` / space charge) and multimodal / one-sided-constraint landscapes.  `refine=True` chains an LS polish; `bo_prior=True` warm-starts an MP match from the cheap envelope cost.  Needs finite bounds + `botorch`.  NOT a default — for cheap envelope matches `least_squares`/`gradient` win on wall-clock. |
| `"gradient"` | local, exact-Jacobian | Levenberg–Marquardt driven by an exact autograd Jacobian from differentiable matrix tracking; see below |

The global algorithms (`differential_evolution`, `dual_annealing`,
`cmaes`, `sequential_scan`) search a finite box: if any ADJUST variable
has an open-ended bound, `match()` raises a `ValueError` naming the
variable.  Add `min`/`max` to that ADJUST card, or use `least_squares`.
`differential_evolution` and `dual_annealing` run deterministically
(fixed RNG seed).

The `"gradient"` algorithm replaces the finite-difference Jacobian with
an *exact* one obtained by autograd through a differentiable
(PyTorch) reimplementation of the linear matrix tracking.  It supports
**SET_TWISS / SET_SIZE** matching of **quadrupole gradient / solenoid
field / dipole angle** variables; anything outside that subset raises
a clear `ValueError` pointing you back to `least_squares`.  It **does
support space charge**: with `space_charge=True` the residual is built
by `build_torch_residual_sc`, which tracks a macroparticle bunch
through the differentiable PIC step tracker (a non-linear SC model,
so the envelope self-validation step is skipped on that path).  Note
the `match()` docstring in `engine.py` still says "without space
charge" — that line is stale.  Without SC, the match is
self-validated against the numpy matcher residual at the starting
point.

### `MatchResult` dataclass

| Field | Meaning |
|---|---|
| `success` | bool — converged, or final cost < 1e-10.  `False` for `"cancelled by user"` (callback raised `StopIteration`) but `x_final` still holds the best-cost x seen. |
| `message` | optimiser termination message (includes algorithm-specific info, e.g. `"sequential_scan: 2 passes, 7 steps/param, step_frac=0.1, reversal=both_grew, threshold=input"`) |
| `n_iter` | residual-function evaluations used |
| `elapsed_s` | wall time |
| `x0` / `x_final` | initial / matched variable column |
| `baseline_cost` | cost at `x0` from the explicit baseline pass run at the start of every `match()` call.  Use to report "before → after" cost in UI / scripts. |
| `residuals` | final residual vector |
| `cost` | final ½‖residuals‖² |
| `variables` | `Variable` objects, optimiser-column order |
| `constraints` | `Constraint` objects, residual order |
| `per_constraint_residuals` | dict: constraint label → its residual sub-vector |

`MatchResult.report()` returns a formatted multi-line summary.

### Cancellation contract

If the `callback` raises `StopIteration`, the matcher converts it into
an internal `_MatchCancelled` exception, applies `state["best_x"]` to
the lattice (so `lattice` agrees with `result.x_final`), and returns a
`MatchResult` with `success=False` and `message="cancelled by user"`
— rather than letting `StopIteration` escape `match()`.  This means a
GUI Stop button can cleanly populate the variable table from the
returned result, and the lattice is never left in a half-mutated state.
The contract holds for all seven algorithms (LS, DE, DA, CMA-ES
sequential and parallel, gradient, sequential_scan, bayesopt).

## Variables and constraints

Variables and constraints are **collected from the lattice**, not built
by hand.  Two helpers do this (and `match()` calls them for you):

```python
from linac_gen.matching import collect_variables, collect_constraints

variables   = collect_variables(lattice, beam_config)   # ADJUST_* cards
constraints = collect_constraints(lattice)              # SET_* cards
```

Each `Variable` is one optimiser degree of freedom:

| Field | Meaning |
|---|---|
| `target` | the object varied (a lattice element, or the `BeamConfig`) |
| `attr` | attribute name on `target` (`"gradient"`, `"alpha_x"`, …) |
| `vmin`, `vmax` | bounds (`±inf` when the ADJUST card gives none) |
| `x0` | initial value |
| `link_group` | non-zero ⇒ shares one optimiser column with peers |
| `label` | display name, e.g. `"QUAD_001.gradient"` |

Each `Constraint` is one residual contribution: a `label`, an
`evaluator` callable, and a `weight`.  `SET_TWISS`, `SET_SIZE`,
`SET_SIZE_MAX` / `SET_SIZE_MIN` and `SET_BEAM_PHASE_ADV` cards each map
to a constraint.  See [SET / ADJUST cards](02_set_adjust.md).

## Legacy `Matcher` class

`linac_gen.matching` also exports a hand-driven `Matcher` class (from
`matcher.py`) plus its `evaluate_objectives` helper.  It predates the
card-driven `match()` engine: you build the problem manually —
`Matcher(lattice, ref).add_variable(element_name, parameter, min_val,
max_val)`, `add_objective(location, quantity, target)` (quantities
`alpha_x` / `beta_x` / `alpha_y` / `beta_y` / `mu_x` / `mu_y` at
`"END"`), then `solve(method=...)` with `least_squares`,
`nelder_mead`, or `differential_evolution` — and it returns a plain
dict (`success` / `variables` / `residuals`), not a `MatchResult`.
It knows nothing about `ADJUST_*`/`SET_*` cards, cost solvers, or the
cancellation contract.  Prefer `match()` for anything new; `Matcher`
remains for existing scripts that drive it directly.

## Periodic and matched-input Twiss helpers

Separate from `match()`, `linac_gen.matching.periodic` provides
**closed-form** Twiss helpers used by the GUI's *Open Matching Dialog*:

| Function | Use case |
|---|---|
| `find_periodic_twiss(lattice, ref)` | Full-ring / whole-lattice periodic Twiss.  Auto-detects coupling.  Also returns the periodic dispersion `disp_x`/`disp_xp`/`disp_y`/`disp_yp` (mm/MeV, mrad/MeV; exact `0.0` for straight lattices, `NaN` near an integer tune). |
| `find_matched_input_twiss(lattice, ref, cell_start, cell_end)` | Transfer-line: periodic Twiss of a sub-cell, back-propagated to the lattice entrance.  Returns the entrance dispersion too — a bending cell's matched beam needs it (set the beam's `disp_*` fields, or let the matching dialog's Apply do it). |
| `find_sc_matched_input_twiss(lattice, ref, cell_start, cell_end, current, base_initial, *, max_iter=50, tol=1e-4)` | Same as above but with space charge: two-stage iteration (cell-periodic with SC + envelope back-propagation).  Also matches dispersion (8-state).  `max_iter` / `tol` bound the fixed-point iteration (stop when the input-Twiss update falls below `tol`). |
| `find_coupled_matched_twiss(lattice, ref)` | **Coupled** lattices (solenoid HWR, skew quads): eigenvector / Wolski method on the 4×4 transverse map.  Usually accessed indirectly via the auto-routing in `find_periodic_twiss`. |
| `find_fodo_cells(lattice)` | Auto-detect candidate periodic cells.  Recognises Quadrupole, Solenoid, and FieldMap/FieldMap3D classified as solenoid (kb≠0, ke=0). |

### Whole-lattice vs sub-cell mode -- which to use

The two main entry points (`find_periodic_twiss` and
`find_matched_input_twiss`) answer different physical questions.

**`find_periodic_twiss(lattice, ref)` -- "whole-lattice" mode**:
asks *"what input Twiss, tracked once through the whole lattice,
comes back to itself at the end?"*  Computes the one-turn 6×6
transfer matrix and extracts the periodic solution.

* Physically meaningful for **rings**: storage rings, synchrotrons,
  FFAGs -- structures where one circulation = one period.
* Physically meaningful for **a single periodic cell modeled in
  isolation**, asking "what's the solution if this cell repeats
  forever?"
* **Not** physically meaningful for transfer lines or accelerating
  sections (HWR / SSR / MEBT / BTL).  For accelerating sections,
  eigenvalues drift off the unit circle as βγ grows -- the function
  will route to the coupled path and surface an
  ``accelerating-section deviation`` warning; result is the smooth
  approximation, not a strict periodic solution.

**`find_matched_input_twiss(lattice, ref, cell_start, cell_end)` --
"FODO-cell" mode**: asks *"there's a repeating sub-section.  What
Twiss do I inject at s=0 so the beam is matched to that
sub-section's period by the time it arrives?"*  Computes the
periodic Twiss of the picked sub-cell and back-propagates to s=0
through the inverse of the front section.

* The right choice for transfer lines and linacs with a repeating
  sub-section inside them.
* Use `find_fodo_cells(lattice)` to discover candidate cells, or
  pass `cell_start`/`cell_end` manually if you know the period.
* For lattices with bends (BTL arcs, HEBT), use
  `find_sc_matched_input_twiss` instead -- it carries dispersion
  through an 8-state formulation, which `find_matched_input_twiss`
  does not.

| Lattice in `examples/pipii/*` | Recommended |
|---|---|
| HWR cryomodules | cell mode, cell 0 |
| SSR1 / SSR2 / LB650 / HB650 | cell mode, cell 0 |
| MEBT | cell mode |
| BTL (arc) | cell mode + SC (for dispersion) |
| Storage ring (hypothetical) | whole-lattice mode |

### Coupled-lattice support (HWR cryomodules)

For lattices with solenoid focusing, skew quadrupoles, or any other
source of transverse x↔y coupling, the 2×2-block-trace approach used
by the standard Courant–Snyder extraction is invalid -- the trace
is no longer the one-turn phase advance.

`find_periodic_twiss` now **auto-detects** coupling (by catching the
"lattice is coupled" `ValueError` from the underlying `compute_twiss`)
and routes to `find_coupled_matched_twiss`, which:

1. Eigendecomposes the 4×4 transverse one-turn map.
2. Picks the eigenvector from each complex-conjugate pair with
   positive imaginary part.
3. Normalises each via `v† S v = +i` (the symplectic-form
   convention; Wolski 2014 Eq. 3.66).
4. Builds the matched Σ as `Σ = Σ_k 2·Re(v_k v_k†)` (Wolski Eq. 3.92,
   summed over modes per Eq. 3.95).
5. Projects per-plane α / β from Σ for display (matching what x-only
   and y-only diagnostics would measure).

The returned dict has the same shape as the decoupled case, plus:

| Field | Meaning |
|---|---|
| `coupled` | `True` for the eigenvector path, `False` otherwise. |
| `sigma4` | The full 4×4 matched Σ (unit-emittance per normal mode).  Multiply rows/cols 0-1 by ε₁ and 2-3 by ε₂ to scale to a physical beam. |
| `mu_1`, `mu_2` | The two normal-mode phase advances in degrees. |
| `alpha_x`, `beta_x`, `alpha_y`, `beta_y` | Σ projections (NOT decoupled Courant–Snyder; the x- and y-only values an apertured diagnostic would read). |

For an unstable coupled lattice (eigenvalues off the unit circle),
`_build_coupled_matched_sigma` raises `ValueError` with the actual
eigenvalue moduli for diagnosis.

```{.python .skip}
from linac_gen.matching.periodic import find_periodic_twiss

r = find_periodic_twiss(hwr_lattice, ref)   # a coupled (solenoid) lattice
if r.get("coupled"):
    print(f"Coupled lattice -- normal-mode tunes μ₁={r['mu_1']:.1f}°, "
          f"μ₂={r['mu_2']:.1f}°")
    print(f"Projected: β_x={r['beta_x']:.2f}, β_y={r['beta_y']:.2f}")
    # Use the matched Σ directly with the envelope solver:
    Sigma = r["sigma4"]  # 4×4
```

## Source

* `linac_gen/matching/engine.py` — `match()`, `MatchResult`, `MATCH_ALGORITHMS`.
* `linac_gen/matching/variables.py` — `Variable`, `collect_variables`.
* `linac_gen/matching/constraints.py` — `Constraint`, `collect_constraints`.

## Cross-references

* [SET / ADJUST](02_set_adjust.md) — `.dat` card syntax.
* [CLI](04_cli.md) — `python -m linac_gen.matching`.
* [Recipes](05_recipes.md) — worked examples.

← [SET / ADJUST](02_set_adjust.md) ·
[Continue to CLI →](04_cli.md)
