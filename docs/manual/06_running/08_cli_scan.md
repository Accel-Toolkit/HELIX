# CLI: `scan`

A native parameter scan — sweep one or more variables and collect a CSV
summary, one row per point.  Scans reuse HELIX's process-pool engine,
so they parallelise across cores.

```
python -m linac_gen scan <input> --vary VAR=SPEC [options]
```

## The `--vary` option

`--vary` is **required** and **repeatable**.  Each `--vary VAR=SPEC`
declares one swept variable; repeating it forms the **Cartesian
product** of the value sets.

`SPEC` is one of:

| Form | Meaning | Example |
|---|---|---|
| `start:stop:step` | inclusive range | `current=0:8:2` → 0, 2, 4, 6, 8 |
| `v1,v2,v3` | explicit value list | `nx=32,48,64` |

`VAR` may be:

| Kind of `VAR` | Example | What it sweeps |
|---|---|---|
| a beam field | `current`, `energy`, `emit_nx` | a `BeamConfig` scalar |
| a grid / step parameter | `nx`, `grid_extent`, `step1`, `step2` | a PIC / integration setting |
| an element parameter | `QF.gradient`, `@12.angle` | one element's parameter |

## Options

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `--vary` | `VAR=SPEC` | (required) | a swept variable — repeatable |
| `--out` | `FILE` | `scan.csv` | CSV summary path |
| `--mode` | `envelope` \| `mp` | `envelope` | solver mode for every point |
| `--parallel` | int | `1` | worker processes; `>1` runs the points in a process pool |
| `--seed` | int | `42` | RNG seed |
| `--env-solver` | `matrix` \| `sacherer` | `matrix` | envelope solver kind |

### Fixed overrides

Every `run`-style override is also accepted by `scan` and is held
**fixed** for every point.  A variable that is *swept* by `--vary`
takes precedence over its fixed value at that point.

| Option | Argument | Meaning |
|---|---|---|
| `--energy` | MeV | reference kinetic energy |
| `--current` | mA | beam peak current |
| `--freq` | MHz | RF frequency |
| `--n-particles` | int | macroparticle count (`mp` mode only) |
| `--species` | `proton` \| `H-` \| `deuteron` | particle species |
| `--beam` | `NAME=VALUE` | repeatable; any `BeamConfig` scalar (e.g. `--beam duty_cycle=100`) |
| `--set` | `ELEM.attr=VALUE` | repeatable; one element parameter (selector `NAME.attr` or `@N.attr`) |
| `--nx` | int | PIC grid points per axis (`nx=ny=nz`) |
| `--grid-extent` | float | PIC grid extent in σ |
| `--step1` | float | integration steps / m |
| `--step2` | float | space-charge kicks / m |
| `--backend` | `auto` \| `cpu` \| `gpu` | compute backend (forced to `cpu` when `--parallel > 1`) |
| `--sc` | `NAME=VALUE` | repeatable; any `SpaceChargeConfig` field — e.g. `--sc kernel=tsc` for the PIC deposition kernel (`scan` has no dedicated `--kernel` flag, unlike `run`) |
| `-q`, `--quiet` | flag | suppress per-point progress |

## The CSV summary

One row per scan point.  Columns are the swept variable(s) followed by
the end-of-lattice metrics:

```
sigma_x, sigma_y, sigma_phi, sigma_w, emit_x, emit_y, emit_z,
transmission, ref_w_kin, x_max, y_max, elapsed
```

Cells that a mode does not produce are left blank (e.g. the envelope
solver records no `transmission`).

## Parallelism

`--parallel N` (N > 1) runs the points across `N` worker processes.
Each point is an independent simulation, so the speed-up is near-linear
in cores.  When parallel workers are requested and any point would use
a GPU backend, all points fall back to CPU — one GPU cannot be shared
across processes.  Results are always returned in input order.

## Examples

```sh
# Sweep beam current 0→8 mA, envelope mode
python -m linac_gen scan cell.lgproj --vary current=0:8:2 \
    --out scan_current.csv

# Sweep a quadrupole gradient over an explicit list
python -m linac_gen scan cell.dat --vary QF.gradient=4,5,6,7,8 \
    --energy 800 --species H- --out scan_grad.csv

# 2-D scan (current × gradient = 9 points), 4 workers
python -m linac_gen scan cell.lgproj \
    --vary current=0:4:2 --vary @12.gradient=4:8:2 \
    --parallel 4 --out scan_2d.csv

# Multi-particle convergence scan over the PIC grid size
python -m linac_gen scan cell.lgproj --vary nx=24:64:8 \
    --mode mp --out scan_nx.csv
```

Runnable versions are in
[`examples/batch_mode/06_scan_beam`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/examples/batch_mode/06_scan_beam),
[`07_scan_element`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/examples/batch_mode/07_scan_element) and
[`08_scan_2d_parallel`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/examples/batch_mode/08_scan_2d_parallel).

## Cross-references

* [Batch-mode overview](06_batch_cli.md) — the input / override model.
* [CLI: run](07_cli_run.md) — a single run with the same overrides.
* [CLI: batch](09_cli_batch.md) — heterogeneous multi-run campaigns.

← [CLI: run](07_cli_run.md) ·
[Continue to CLI: batch →](09_cli_batch.md)
