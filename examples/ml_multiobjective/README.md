# Multi-objective lattice design (Pareto fronts)

Where the matcher drives the ADJUST knobs to satisfy `SET_*` residuals (a
single scalar cost), **multi-objective design** explores the *trade-off
surface* between competing objectives and returns a **Pareto front** of
non-dominated designs. Reachable from the Python API, the CLI
(`python -m linac_gen mo`), and the GUI (Matching tab → *Multi-objective
design…*).

This demo uses the classic conflict: **longitudinal emittance growth vs
exit energy**. Pushing the cavities for more energy means running further
off-crest, which grows the longitudinal emittance — so there is no single
best design, only a front of compromises.

## Files

- `mo_demo.dat` — the same 6-knob lattice as `examples/ml_bayesopt/`.
- `run_pareto.py` — runs NSGA-II, prints the front, marks the knee point,
  saves `pareto.csv` + `pareto.png`.

## Run it

```bash
python examples/ml_multiobjective/run_pareto.py
```

Representative output (front of ~10 designs):

```
  emit_nz_growth  exit_energy[MeV]
----------------------------------
          1.0000            4.0830
          ...
          1.0000            4.6154
Knee-point design: emit_nz_growth=1.0000, exit_energy=4.4672 MeV
```

The front spans the achievable exit-energy range while holding
longitudinal emittance flat; the knee point is a balanced pick.

## Objectives library

All objectives are **minimised** (maximise-style ones are negated). List
them with `python -m linac_gen mo --list-objectives`:

| name | meaning |
|---|---|
| `emit_nx_growth` / `_ny_` / `_nz_` | normalised emittance growth (out/in) per plane |
| `emit_4d_growth` | 4-D transverse emittance growth (out/in) |
| `transmission_loss` | beam loss `100 − T_final` (%); MP only |
| `neg_exit_energy` | `−W_kin,out` (minimise → maximise energy) |
| `exit_sigma_x` / `_y` | exit RMS size [mm] |
| `max_sigma_x` / `_y` | peak RMS size along s [mm] |

## CLI

```bash
python -m linac_gen mo examples/ml_multiobjective/mo_demo.dat \
    --objective emit_nz_growth --objective neg_exit_energy \
    --algorithm nsga2 --pop-size 24 --n-gen 15 \
    --energy 2.5 --current 5 --freq 162.5 \
    --out pareto.csv --plot pareto.png
```

## GUI

Matching tab → **Multi-objective design…** → tick ≥2 objectives, choose
the algorithm + cost solver, **Run**. The Pareto front shows as a
scatter + table; **Apply knee point** (or select a row) writes that
design into the lattice (and flags the project dirty).

## Algorithms

- **`nsga2`** (default) — NSGA-II genetic algorithm (pymoo). Robust,
  population-based; the canonical accelerator design tool. Best when the
  forward pass is cheap (envelope / surrogate).
- **`qnehvi`** — Bayesian multi-objective (BoTorch qNEHVI). Sample
  efficient — far fewer forward passes — so better when each evaluation
  is expensive (`--cost-solver mp` / space charge). On this lattice it
  reaches a comparable front in ~20 evals vs NSGA-II's ~200.

Requires `pymoo` (NSGA-II) and `botorch`+`gpytorch` (qNEHVI) — both in the
HELIX environment.
