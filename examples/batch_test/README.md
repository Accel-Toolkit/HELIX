# Batch-mode demo (headless CLI, bulk simulations)

Self-contained example of HELIX's three bulk-simulation modes on the
simplest lattice in the repo: a 24-cell FODO (9.6 m, 48 quads at
±25 T/m, no RF), H⁻ at 2.12 MeV, 5 mA, matched Twiss.

## Contents

Tracked inputs (the outputs below are **generated** by running the tests
and are not committed — `*.h5` is gitignored, the CSVs are regenerable):

| file | what | tracked |
|---|---|---|
| `fodo.dat` | the lattice (copy of `examples/halo_fodo.dat`) | ✅ |
| `fodo.lgproj` | project: beam + numerics (relative `lattice_path`) | ✅ |
| `jobs.json` | batch-campaign job file (4 heterogeneous jobs) | ✅ |
| `fodo_results.h5` | output of test 1 | generated |
| `scan.csv` | output of test 2 (10 rows) | generated |
| `batch_summary.csv` | output of test 3 (4 rows) | generated |

## The three tests (run from this directory)

```bash
# 1. single headless MP run -> HDF5
python3 -m linac_gen run fodo.lgproj --mode mp --out .

# 2. parameter scan: quad gradient x beam current, 4 worker processes
python3 -m linac_gen scan fodo.lgproj --mode envelope --parallel 4 \
    --vary "QUAD_001.gradient=23:27:1" --vary "current=2,5" --out scan.csv

# 3. batch campaign: heterogeneous jobs from jobs.json, 4 workers
python3 -m linac_gen batch jobs.json --parallel 4 --out .
```

## Reading the results

* `scan.csv` — one row per (gradient, current) point; σ_y falls
  monotonically as the first quad strengthens (23→27 T/m), as expected.
  `transmission` is empty in envelope mode (no particles to lose).
* `batch_summary.csv` — envelope vs MP nominal agree on σ_x to ~2 %;
  the detuned-quad job shows the expected transverse emittance growth
  (ε_x 3.21 → 3.41 mm·mrad); 100 % transmission everywhere.
* Longitudinal caveat: this FODO has **no RF**, so the bunch debunches —
  MP ε_z grows while the envelope model carries its input ε_z as an
  invariant.  Expected fidelity difference, not an error.

Determinism: same `--seed` ⇒ identical results, independent of
`--parallel` worker count.
