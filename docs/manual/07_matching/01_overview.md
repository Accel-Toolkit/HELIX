# Matching overview

Matching means tuning lattice parameters until simulated diagnostics
hit a target — typically Twiss values at a specific s, periodic
conditions across a cell, or aperture limits for halo beams.  HELIX
has a full matcher modeled on TraceWin's SET/ADJUST language.

## TL;DR

* **Variables** (the knobs to tune): quad gradients, solenoid fields,
  cavity phase/voltage, drift lengths, dipole angles, steerer kicks,
  input Twiss — declared by `ADJUST_*` cards.
* **Constraints** (the targets to hit): Twiss values, σ values, σ
  bounds, phase advance — declared by `SET_*` cards.
* **Engine**: a selectable scipy optimiser — local Levenberg–Marquardt
  (`least_squares`, the default) or the global `differential_evolution`
  / `dual_annealing` — with an envelope forward pass per step.

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching import match

# The lattice carries its own ADJUST / SET cards:
lattice, _ = parse_tracewin("examples/matching_demo.dat")
beam_config = BeamConfig(n_particles=1000)

result = match(lattice, beam_config)
print(f"Converged in {result.n_iter} iterations: {result.success}")
```

## Tutorial

### When to match

Common matching problems:

* **Periodic FODO matching** — find α, β at the cell entrance such
  that the beam reproduces itself at the cell exit (the matched
  Twiss).
* **Section-to-section transition** — given Twiss at the end of
  MEBT, find quad strengths in HWR that match into the SSR1
  acceptance.
* **Achromat** — set quad strengths in a transfer line so dispersion
  vanishes downstream of a dipole.
* **Tolerance against mismatch** — find the most-robust setting in
  the presence of expected errors.

### Workflow

```
.dat with SET/ADJUST cards
        │
        ▼
parse_tracewin → Lattice (matching variables + constraints registered)
        │
        ▼
match(lattice, beam_config) → MatchResult
        │
        ▼ (matched values written into the lattice in place)
        ▼
Simulation.run() — verify
```

### `.dat` matching example

```
ADJUST QUAD 2 0 -30 30 0.5 0   ; tune the NEXT quad's gradient, unlinked
QUAD 100 5.0 20
DRIFT 200 20
ADJUST QUAD 2 0 -30 30 0.5 0   ; a second, independent variable
QUAD 100 -5.0 20
DRIFT 200 20
SET_SIZE 1 4 0 0 0             ; target σ_x = 4 mm at lattice exit
```

The matcher reads the `SET_*` cards as constraints, the `ADJUST_*`
cards as variables, and runs the optimiser.  Two placement pitfalls
to avoid:

* **Card position matters.**  A name-based `ADJUST` target is
  resolved by searching *forward* from the card first (then globally
  as a fallback) — a card placed *after* its intended quad binds to
  the **next** matching element downstream.  Put each card before its
  target, or use an integer element index as the target to pin it
  unambiguously.
* **Link groups gang variables.**  Cards sharing the same non-zero
  `link_group` (the third field) collapse into **one** optimiser
  column and move in lockstep.  Two cards both using `... 2 1 ...`
  would become a single variable; use `0` (unlinked) or distinct
  group numbers for independent knobs.

### Running the matcher

`match()` collects the variables and constraints from the lattice
cards — there is nothing extra to pass:

```python
from linac_gen.matching import match

lattice, _ = parse_tracewin("examples/matching_demo.dat")  # ADJUST/SET cards included
result = match(lattice, beam_config)       # least_squares (the default)
print(f"matched values : {result.x_final}")
print(f"final cost     : {result.cost:.3e}")
```

To use a global optimiser, pass `algorithm=`:

```python
result = match(lattice, beam_config, algorithm="differential_evolution",
               max_iter=10)   # small budget for this demo lattice
```

## What can be tuned

Variables are declared by `ADJUST_*` cards; each tunes one element
(or beam) parameter:

| Tunes | Declared by | Parameter |
|---|---|---|
| Quad gradient | `ADJUST` on a `QUAD` | `Quadrupole.gradient` |
| Solenoid field | `ADJUST` on a `SOLENOID` | `Solenoid.field` |
| Dipole angle / radius | `ADJUST` on a bend | `Dipole.angle` / `rho` |
| Cavity phase / voltage | `ADJUST` on a `GAP` / `FIELD_MAP` | `RFGap` / `FieldMap` |
| Drift length | `ADJUST` on a `DRIFT` | `Drift.length` |
| Steerer kick | `ADJUST_STEERER` | `Steerer.bx_l` / `by_l` |
| Input Twiss / emittance | `ADJUST_BEAM_TWISS` / `_EMIT` | `BeamConfig` |

## Cross-references

* [SET / ADJUST card reference](02_set_adjust.md) — TraceWin syntax.
* [Python API](03_python_api.md) — `match()` function.
* [CLI](04_cli.md) — `python -m linac_gen.matching`.
* [Recipes](05_recipes.md) — common matching problems.

← [CLI Twiss tools](../06_running/10_cli_twiss.md) ·
[Continue to SET / ADJUST →](02_set_adjust.md)
