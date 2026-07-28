# Multi-objective design (Pareto fronts)

Where [`match()`](03_python_api.md) drives the ADJUST knobs to satisfy
`SET_*` residuals (a single scalar cost), **multi-objective design**
explores the *trade-off surface* between two or more *competing*
objectives and returns a **Pareto front** of non-dominated designs.
This is the canonical offline accelerator-design workflow — sweep the
decision box with a multi-objective optimiser using a fast forward
model, then pick a knee point.

Reachable from all three surfaces: the Python API
(`linac_gen.matching.multiobjective.pareto_optimize`), the CLI
(`python -m linac_gen mo`), and the GUI (Matching tab →
**Multi-objective design…**).

## Objectives

Objectives are chosen **by name** from a built-in library keyed to the
diagnostics the forward pass already records. All are framed as
**minimisation** (maximise-style quantities are negated), so a smaller
value is always better. List them with `python -m linac_gen mo
--list-objectives`:

| name | meaning |
|---|---|
| `emit_nx_growth` / `_ny_` / `_nz_` | normalised emittance growth (exit / entry) per plane |
| `emit_4d_growth` | 4-D transverse emittance growth (exit / entry) |
| `transmission_loss` | beam loss `100 − T_final` (%); meaningful only with `cost_solver="mp"` |
| `neg_exit_energy` | `−W_kin,out` [MeV] (minimise → maximise exit energy) |
| `exit_sigma_x` / `_y` | exit RMS beam size [mm] |
| `max_sigma_x` / `_y` | peak RMS beam size along s [mm] |

Pick **≥ 2**. They should genuinely conflict — e.g. `emit_nz_growth`
(longitudinal emittance) vs `neg_exit_energy` (exit energy): pushing the
cavities for energy means more off-crest phase, which grows longitudinal
emittance.

## Algorithms

| `algorithm` | Method | When |
|---|---|---|
| `"nsga2"` (default) | NSGA-II genetic algorithm (pymoo) | robust, population-based; the standard lattice-design tool.  Best when the forward pass is cheap (envelope / surrogate). |
| `"qnehvi"` | Bayesian multi-objective (BoTorch qNEHVI) | sample-efficient — far fewer forward passes — so better when each evaluation is expensive (`cost_solver="mp"` / space charge). |

On the demo lattice qNEHVI reaches a comparable front in ~20 evaluations
vs NSGA-II's ~200.

## Python API

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching.multiobjective import pareto_optimize

lat, _ = parse_tracewin("examples/ml_multiobjective/mo_demo.dat")
cfg = BeamConfig(species="proton", energy=2.5, frequency=162.5, current=5.0,
                 n_particles=1000, distribution="waterbag",
                 emit_nx=0.30, alpha_x=-1.2, beta_x=0.32,
                 emit_ny=0.30, alpha_y=2.0, beta_y=0.05,
                 emit_z=0.40, alpha_z=0.0, beta_z=10.0)
res = pareto_optimize(lat, cfg, ["emit_nz_growth", "neg_exit_energy"],
                      algorithm="nsga2", pop_size=24, n_gen=15)
print(res.n_eval, "evals ->", len(res.pareto_F), "Pareto designs")
# res.pareto_F : (n_front, n_obj) objective vectors
# res.pareto_x : (n_front, n_vars) decision vectors (the ADJUST knobs)
```

`ParetoResult` fields: `objective_names`, `pareto_x`, `pareto_F`,
`all_x`, `all_F`, `variables`, `col_for_var`, `n_cols`, `algorithm`,
`n_eval`, `message`.

The `_x` arrays have one column per **optimiser column** (`n_cols`),
which is the link-group-deduplicated decision width — ADJUST cards
sharing a non-zero `link_group` collapse to a single column, so
`variables` (the full per-DoF list) can be *longer* than `pareto_x` is
wide. When you write the front to CSV or a table, use the helpers so the
labels line up with the columns even on linked lattices:

```python
knee = 0                                # pick a design off the front
labels = res.column_variable_labels()   # one label per _x column
for v, x in zip(res.column_variables(), res.pareto_x[knee]):
    print(f"{v.label:<22s} = {x:.4f}")
```

(`column_variables()` / `column_variable_labels()` are just the full
`variables` / labels when nothing is linked, so they are always safe to
use.)

## CLI

```bash
# list objectives
python -m linac_gen mo --list-objectives

# run, write the front to CSV + a 2-D PNG scatter
python -m linac_gen mo examples/ml_multiobjective/mo_demo.dat \
    --objective emit_nz_growth --objective neg_exit_energy \
    --algorithm nsga2 --pop-size 24 --n-gen 15 \
    --energy 2.5 --current 5 --freq 162.5 \
    --out pareto.csv --plot pareto.png

# expensive MP objectives -> use the sample-efficient Bayesian MO
python -m linac_gen mo examples/ml_multiobjective/mo_demo.dat \
    --objective emit_4d_growth --objective transmission_loss \
    --algorithm qnehvi --cost-solver mp --mp-n-particles 500 \
    --space-charge --pop-size 8 --n-gen 12 \
    --energy 2.5 --current 5 --freq 162.5 --out pareto.csv
```

## GUI

Matching tab → **Multi-objective design…** opens a dialog:

1. Tick **≥ 2 objectives** from the checklist.
2. Choose **Algorithm** (`nsga2` / `qnehvi`), **Cost solver**
   (`envelope` / `mp`), and **pop/init** + **gen/iter** sizes.
3. **Run** (background thread; **Stop** cancels).
4. Inspect the **Pareto-front scatter** (first two objectives, with the
   knee point starred) and the **table** of designs.
5. **Apply knee point** — or select a table row first — writes that
   design's ADJUST values into the lattice and flags the project dirty
   (so the close-without-save prompt warns you).

## Worked example

See [`examples/ml_multiobjective/`](05_recipes.md):
`run_pareto.py` produces the ε_z-growth-vs-exit-energy front, marks the
knee point, and saves `pareto.csv` + `pareto.png`.

## Cross-references

- [Python API](03_python_api.md) — single-objective `match()`.
- [CLI](04_cli.md) — the matcher CLI.
- [Bayesian-optimisation matcher](03_python_api.md) — the single-objective
  GP optimiser (`algorithm="bayesopt"`) that shares the BoTorch backend.

← [Robust emittance minimisation](07_emittance_min.md) ·
[Continue to Errors → Overview →](../08_errors/01_overview.md)
