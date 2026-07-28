# Matching CLI

For batch matching without writing Python, HELIX ships a command-line
interface: `python -m linac_gen.matching`.

## TL;DR

```bash
python -m linac_gen.matching examples/matching_demo.dat \
    --algorithm least_squares \
    --max-iter 200 \
    --report
```

This parses the `.dat`, runs the matcher on its embedded `ADJUST_*` /
`SET_*` cards, prints a report, and writes the matched lattice to
`examples/matching_demo.matched.dat`.

## Flags

### Input / output

| Flag | Default | Meaning |
|---|---|---|
| `input` (positional) | required | input lattice `.dat` |
| `--out PATH` | `<input>.matched.dat` | where to write the matched lattice |
| `--no-write` | off | skip writing the matched `.dat` (just report) |
| `--report` | off | print the full variable / constraint report |

### Algorithm

| Flag | Default | Meaning |
|---|---|---|
| `--algorithm NAME` | `least_squares` | one of `least_squares`, `differential_evolution`, `dual_annealing`, `cmaes`, `sequential_scan`, `gradient`, `bayesopt` |
| `--max-iter N` | `200` | `least_squares`: max function evals; population algorithms: max generations |
| `--xtol VAL` | `1e-8` | `least_squares` step tolerance |
| `--ftol VAL` | `1e-8` | `least_squares` residual tolerance |

### CMA-ES

| Flag | Default | Meaning |
|---|---|---|
| `--cmaes-sigma F` | `0.2` | initial step as fraction of bound-box width |
| `--cmaes-popsize N` | `0` | population; `0` = library default `4 + ⌊3·ln(N)⌋` |
| `--cmaes-parallel N` | `1` | evaluate population across N worker processes; `0` = auto-detect.  Only worth >1 for expensive per-eval lattices (FieldMap3D). |
| `--cmaes-search-solver` | `auto` | objective the parallel workers score the search with.  `auto` follows `--cost-solver` faithfully (parallel + mp genuinely evaluates the multiparticle objective); `envelope` is the fast envelope-guided hybrid — envelope-scored search, mp-scored baseline/polish/final — as an **explicit opt-in** (this was the silent behavior before the 2026-07 honesty round).  No effect with `--cmaes-parallel 1`. |
| `--no-refine` | off | skip the `least_squares` polish after CMA-ES / bayesopt |

### Bayesian optimisation (`--algorithm bayesopt`)

| Flag | Default | Meaning |
|---|---|---|
| `--bo-prior` | off | physics-informed warm start: scout the box with the cheap envelope cost before the expensive objective.  Only active with `--cost-solver mp`. |
| `--bo-n-init N` | `0` | initial Sobol samples before the GP is first fit; `0` = default `min(2N+2, 12)`. |

`bayesopt` reuses `--max-iter` (number of Bayesian iterations) and is the
sample-efficient pick for expensive (MP) + multimodal matches.  Example:

```bash
python -m linac_gen.matching examples/ml_bayesopt/bo_demo.dat \
    --algorithm bayesopt --max-iter 30 \
    --energy 2.5 --current 5 --frequency 162.5 --report

# expensive MP match, physics-informed warm start:
python -m linac_gen.matching examples/ml_bayesopt/bo_demo.dat \
    --algorithm bayesopt --cost-solver mp --mp-n-particles 200 \
    --max-iter 8 --bo-prior --space-charge \
    --energy 2.5 --current 5 --frequency 162.5 --report
```

### Cost solver (forward-pass model)

| Flag | Default | Meaning |
|---|---|---|
| `--cost-solver KIND` | `envelope` | `envelope` (fast linear, default) or `mp` (full PIC; **50–100× slower per eval**, physics-accurate) |
| `--mp-n-particles N` | `1000` | MP cost-solver macroparticle count.  Lower = faster but noisier cost. |
| `--mp-seed N` | `42` | MP cost-solver RNG seed for reproducible per-eval cost. |

The MP cost-solver uses library-default `SpaceChargeConfig` /
`StepConfig`; the GUI passes the Numerics tab's config explicitly,
which the CLI doesn't (yet) read from a `.lgproj`.

### Sequential-scan algorithm

| Flag | Default | Meaning |
|---|---|---|
| `--seqscan-passes N` | `2` | full passes through the element list |
| `--seqscan-steps N` | `7` | bracket steps per parameter per pass |
| `--seqscan-step-frac F` | `0.10` | per-step displacement = F × (vmax − vmin) |
| `--seqscan-reversal KIND` | `both_grew` | `both_grew` (flip when both transverse AND longitudinal ε exceed reference) or `any_grew` (stricter; either) |
| `--seqscan-threshold KIND` | `input` | reversal reference: `input` (compare to beam input ε; tight) or `seed_exit` (compare to unmatched-lattice exit ε; natural for ε minimisation with intrinsic growth) |
| `--seqscan-element NAME` | (all ADJUST'd) | scan only elements with this name (repeatable) |
| `--seqscan-reject-loss` | off | hard-reject any step whose trial transmission falls below `--seqscan-loss-threshold-pct`.  best_x is NOT updated to a loss-inducing step even if cost dropped; direction flips.  Closes the ε-gaming gap.  Requires `--cost-solver mp`. |
| `--seqscan-loss-threshold-pct F` | `100.0` | Transmission floor for the rejection check.  `100.0` rejects any loss; `99.9` tolerates sub-permille. |

See [Sequential-scan](07_emittance_min.md#sequential-scan) for the
physics and when to use each combination.

### Forward-pass / space charge

| Flag | Default | Meaning |
|---|---|---|
| `--space-charge` | off | run the forward pass with space charge |
| `--current mA` | `0.0` | beam current used when `--space-charge` is set |

### Beam defaults

A default `BeamConfig` seeds the envelope; override its fields with
`--species`, `--energy`, `--frequency`, and the `--alpha-x` /
`--beta-x` / `--emit-x` family (and the `-y`, `-z` planes).  Run
`python -m linac_gen.matching --help` for the complete list.

The three global algorithms (`differential_evolution`, `dual_annealing`,
`cmaes`) require finite `min`/`max` bounds on every ADJUST card; an
open-ended bound aborts the run with an error naming the offending
variable.

### When to pick which algorithm

| If… | Use |
|---|---|
| LM-friendly objective, all constraints have non-vanishing gradients | `least_squares` |
| Multimodal landscape with coupling resonances, one-sided / clipped residuals (`MIN_EMIT_GROWTH`, `SET_SIZE_MAX`), or sync-phase sign flips | **`cmaes`** + LS polish |
| Highly non-smooth or discontinuous cost | `differential_evolution` or `dual_annealing` |
| Lattice is exclusively linear and you want exact Jacobian gradients | `gradient` |

`cmaes` is the recommended default for emittance-minimisation
workflows because the one-sided `MIN_EMIT_GROWTH` residual has zero
gradient on the "drop" side, which defeats scipy's LM.  The default
`refine=True` chains a `least_squares` polish from the best CMA-ES
point so you still get LM-quality precision on the active constraints.

## How it works

1. Parse the input `.dat` via `parse_tracewin()`.
2. Collect `ADJUST_*` variables and `SET_*` constraints from the lattice.
3. Build a `BeamConfig` from the CLI beam flags.
4. Run `match()` with the chosen `--algorithm`.
5. Apply the matched values back into the lattice elements.
6. Write the matched lattice to `--out` (unless `--no-write`).

The process exit code is `0` on a converged match, `1` otherwise.

## Examples

### CMA-ES with default LS polish

```
$ python -m linac_gen.matching examples/matching_demo.dat \
    --algorithm cmaes --max-iter 80 --report

[match] 8 elements; algorithm=cmaes; cost_solver=envelope; running without SC
[match] baseline cost (at x0) = 1.3240e+00
[match] OK — 880 iters, cost 9.3450e-13, 2.84s

Matching report
═══════════════
  status     : OK
  message    : CMA-ES: tolfun + tolx  (parallel x1) + LS refine
  iterations : 880
  cost       : 9.345004e-13
  elapsed    : 2.84 s

Variables:
  QUAD_001.gradient            -5  →  -5.47142     [-30, 30]

Constraints:
  SET_SIZE          residual rms = 1.3671e-06
```

### Sequential-scan with seed-exit threshold

```bash
python -m linac_gen.matching examples/emittance_min/pipii_hwr_cm_match_4d_relaxed.dat \
    --algorithm sequential_scan \
    --seqscan-passes 2 --seqscan-steps 7 --seqscan-step-frac 0.10 \
    --seqscan-reversal both_grew \
    --seqscan-threshold seed_exit \
    --report
```

### MP-mode cost solver (physics-accurate final polish)

```bash
python -m linac_gen.matching examples/emittance_min/pipii_hwr_cm_match_4d_relaxed.dat \
    --algorithm cmaes --max-iter 40 --cmaes-popsize 11 \
    --cost-solver mp --mp-n-particles 1000 \
    --space-charge --current 10.0
```

Expect 1–2 orders of magnitude longer wall time than the envelope path
because each evaluation runs a full multi-particle simulation.

## Cross-references

* [Python API](03_python_api.md) — the `match()` function.
* [SET / ADJUST](02_set_adjust.md) — `.dat` card syntax.
* [Recipes](05_recipes.md) — worked examples.

← [Python API](03_python_api.md) ·
[Continue to Recipes →](05_recipes.md)
