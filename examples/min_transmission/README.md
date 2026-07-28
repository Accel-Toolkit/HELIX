# `MIN_TRANSMISSION`: matching without gaming ε via beam loss

Pure emittance minimisation has a notorious failure mode: in
multi-particle mode the matcher can **lower the emittance by letting a
tight aperture scrape off the beam halo** — emittance is computed from
*surviving* particles, so clipping the tails reads as a "better" ε bought
with lost beam. The `MIN_TRANSMISSION threshold weight` card adds a
one-sided penalty that fires only when transmission drops below
`threshold`, forbidding that trade.

It is a **multi-particle-only** constraint (envelope mode tracks no
particle losses), so match with `cost_solver="mp"`.

## Files

- `min_transmission_demo.dat` — the 6-knob lattice with a tight **6 mm**
  aperture downstream of solenoid 2 and a `MIN_TRANSMISSION 99.0 50.0`
  card. A moderate beam transmits ~98% at the seed, so the floor is
  *violated* until the match focuses the beam through.
- `run_min_transmission.py` — matches (cmaes, MP) and reports transmission
  before/after plus the constraint residual.

## Run it

```bash
python examples/min_transmission/run_min_transmission.py
```

Representative output:

```
Seed transmission (unmatched x0)  = 97.75 %

Matched: cost 4.30e-06  (21 evals, ~90s)
Matched transmission              = 99.00 %

Constraint residuals (rms):
  MIN_EMIT_GROWTH:X      0.0000e+00
  MIN_EMIT_GROWTH:Y      2.9316e-03
  MIN_TRANSMISSION:99%   0.0000e+00
  SET_KE_OUT_MIN:3.5MeV  0.0000e+00
```

The seed sits **below** the 99% floor (residual non-zero), so the matcher
focuses the beam through the aperture until transmission reaches 99% and
the `MIN_TRANSMISSION` residual falls to ~0 — *without* trading beam for a
lower ε.

To see the failure mode it prevents, delete the `MIN_TRANSMISSION` line
from the `.dat` and re-run: the matcher is then free to accept a
lower-ε-but-lossier solution.

## The card

```text
MIN_TRANSMISSION 99.0 50.0      ; residual = 50.0 * max(0, 99.0 - T_final) / 100
```

The residual is zero while transmission ≥ threshold and grows linearly
below it. Pick a strict threshold (99.9 %) + large weight (50–100×) for a
CW machine where lost beam means activation; looser (99.0 / 1–10×) for
proof-of-concept matching.

## Surfaces

- **Lattice card** — drop the line into any `.dat` constraints block;
  `collect_constraints()` picks it up automatically (no Python kwarg, CLI
  flag, or GUI control needed).
- **CLI** — `python -m linac_gen.matching … --cost-solver mp --space-charge`.
- **GUI** — it appears as a `MIN_TRANSMISSION` row in the Matching tab's
  **Constraints** table after a Multi-particle match.

For the **hard-rejection** complement (sequential_scan's
`seqscan_reject_loss`), see [`examples/sequential_scan/`](../sequential_scan/README.md)
and [Recipe 7](../../docs/manual/07_matching/05_recipes.md).
