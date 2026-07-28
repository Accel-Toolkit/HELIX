# Bayesian-optimisation matcher (`algorithm="bayesopt"`)

Demonstrates HELIX's Gaussian-process **Bayesian-optimisation** matcher —
the sample-efficient global optimiser for *expensive* and *multimodal*
matching problems. It is the in-code analogue of the accelerator-tuning
state of the art (SLAC Xopt / Badger) and plugs into the same matcher
seam as the other six algorithms, so it is reachable from the Python API,
the CLI, and the GUI.

## Files

- `bo_demo.dat` — a 0.9 m lattice (2 solenoids, 2 RF gaps) with **6 ADJUST
  knobs** and **4 constraints** (3 one-sided `MIN_EMIT_GROWTH` + 1
  `SET_KE_OUT_MIN`). The seed is deliberately mismatched (~50 % emittance
  growth) so the matcher has real work.
- `run_bayesopt.py` — A/B convergence: `least_squares` vs `cmaes` vs
  `bayesopt` on the identical problem, reporting **evaluations** and
  baseline → final cost.
- `run_bayesopt_prior.py` — physics-informed warm start (`bo_prior`) on an
  expensive multi-particle (MP) match: prior off vs on.

## Run it

```bash
# A/B convergence (envelope cost, ~seconds)
python examples/ml_bayesopt/run_bayesopt.py

# physics-informed warm start on an MP match (~minutes, tiny budget)
python examples/ml_bayesopt/run_bayesopt_prior.py
```

Representative `run_bayesopt.py` output:

```
algorithm              evals      baseline         final     sec
----------------------------------------------------------------
least_squares             99     2.330e-02     2.099e-03     0.1
cmaes                    416     2.330e-02     2.099e-03     0.8
bayesopt                  89     2.330e-02     2.099e-03    11.3
```

All three reach the same low cost; **Bayesian optimisation uses the fewest
evaluations** — the metric that matters when each forward pass is
expensive. Note the wall-time column: on this *cheap* envelope problem the
GP-fitting overhead makes BO slower in seconds even though it uses fewer
evals. BO wins on wall-clock only when each forward pass is costly
(`cost_solver="mp"` / space charge), where evaluations dominate.

## CLI

```bash
python -m linac_gen.matching examples/ml_bayesopt/bo_demo.dat \
    --algorithm bayesopt --max-iter 30 \
    --energy 2.5 --current 5 --frequency 162.5 --report

# expensive MP match with physics-informed warm start:
python -m linac_gen.matching examples/ml_bayesopt/bo_demo.dat \
    --algorithm bayesopt --cost-solver mp --mp-n-particles 200 \
    --max-iter 8 --bo-prior --space-charge \
    --energy 2.5 --current 5 --frequency 162.5 --report
```

## GUI

Matching tab → **Algorithm = bayesopt**. For an expensive match, set
**Cost solver = mp** and tick **BO physics prior** to warm-start from the
cheap envelope cost. The live convergence plot updates per evaluation; the
**Stop** button cancels and keeps the best point found so far.

## Python API

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching import match

lat, _ = parse_tracewin("examples/ml_bayesopt/bo_demo.dat")
cfg = BeamConfig(species="proton", energy=2.5, frequency=162.5, current=5.0,
                 n_particles=1000, distribution="waterbag",
                 emit_nx=0.30, alpha_x=-1.2, beta_x=0.32,
                 emit_ny=0.30, alpha_y=2.0, beta_y=0.05,
                 emit_z=0.40, alpha_z=0.0, beta_z=10.0)
r = match(lat, cfg, algorithm="bayesopt", max_iter=30, refine=True)
print(r.baseline_cost, "→", r.cost, "in", r.n_iter, "evals")
```

## When to use BO

- **Use it** for expensive (MP / space-charge) matches and multimodal /
  one-sided-constraint landscapes where the population globals
  (DE / dual-annealing / CMA-ES) burn hundreds of evaluations.
- **Don't** use it for cheap linear-envelope matches — `least_squares`
  or `gradient` reach the same optimum faster in wall-clock.

Requires `botorch` + `gpytorch` (already in the HELIX environment).
