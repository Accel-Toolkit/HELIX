# CLI: `run`

One headless tracking simulation.  Loads a lattice or `.lgproj`
project, applies overrides, runs an envelope / multi-particle / matrix
simulation, and writes the results.

```
python -m linac_gen run <input> [options]
```

`<input>` is a `.dat` / `.madx` lattice or a `.lgproj` project.

## Options

### Mode & output

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `--mode` | `envelope` \| `mp` \| `matrix` | `envelope` | solver mode (see below) |
| `--out` | `DIR` | `.` | output directory |
| `--format` | `hdf5` \| `openpmd` \| `partran` | `hdf5` | results file format |
| `--write-dst` | flag | off | also write the final distribution as a TraceWin `.dst` (multi-particle only) |

### Beam overrides

| Option | Argument | Meaning |
|---|---|---|
| `--energy` | MeV | kinetic energy |
| `--current` | mA | beam current |
| `--freq` | MHz | beam frequency |
| `--n-particles` | int | macroparticle count |
| `--species` | `proton` \| `deuteron` \| `H-` | particle species |
| `--beam` | `NAME=VALUE` | any other `BeamConfig` field — repeatable (e.g. `--beam emit_nx=0.3 --beam beta_x=4`) |

### Element overrides

| Option | Argument | Meaning |
|---|---|---|
| `--set` | `ELEM.attr=VALUE` | set one element parameter — repeatable.  `ELEM` is a name or `@N` index |

### Convergence / PIC overrides

| Option | Argument | Meaning |
|---|---|---|
| `--nx` | int | PIC grid size (`nx=ny=nz`) |
| `--grid-extent` | float | PIC grid extent, in σ |
| `--step1` | float | integration steps per metre |
| `--step2` | float | space-charge kicks per metre |
| `--kernel` | `cic` \| `tsc` | PIC deposit kernel |
| `--backend` | `auto` \| `cpu` \| `gpu` | compute backend |
| `--sc` | `NAME=VALUE` | any other `SpaceChargeConfig` field — repeatable |
| `--env-solver` | `matrix` \| `sacherer` | envelope solver kind (default `matrix`) |

### Misc

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `--seed` | int | `42` | RNG seed for the particle distribution |
| `--fail-under-transmission` | % | — | exit non-zero if final transmission falls below this |
| `-q`, `--quiet` | flag | off | suppress the run report |

## The three modes

* **`envelope`** — rms Σ-matrix tracking.  Fast, no macroparticles.
  Writes the per-step envelope (σ, emittance, Twiss).
* **`mp`** — multi-particle PIC tracking.  Builds a beam of
  `n_particles` macroparticles; space charge is included when the beam
  has current.  Adds transmission, halo, snapshots; `--write-dst` dumps
  the final distribution.
* **`matrix`** — linear transfer-matrix tracking.  Writes the 6×6
  transfer matrix and the per-plane periodic Twiss to a text file.

## Output files

The results file is named `<input-stem>_results.<ext>` in `--out`:

| `--format` | Extension | Notes |
|---|---|---|
| `hdf5` | `.h5` | HELIX-native HDF5 |
| `openpmd` | `.opmd.h5` | openPMD-beamphysics 1.1 — portable interchange |
| `partran` | `.txt` | TraceWin-style tab-separated ASCII |

`--write-dst` additionally writes `<input-stem>_final.dst`; `--mode
matrix` writes `<input-stem>_matrix.txt`.

## Examples

```sh
# Envelope run of a project, results into ./out/
python -m linac_gen run cell.lgproj --mode envelope --out out/

# Multi-particle run; also dump the final .dst; fail if transmission < 90 %
python -m linac_gen run cell.lgproj --mode mp --n-particles 20000 \
    --write-dst --fail-under-transmission 90 --out out/

# Bare lattice — every parameter from the command line
python -m linac_gen run cell.dat --mode mp \
    --energy 800 --species H- --current 5 \
    --beam emit_nx=0.25 --set QF.gradient=8.5 \
    --nx 64 --kernel tsc --out out/

# Linear transfer matrix + Twiss
python -m linac_gen run cell.lgproj --mode matrix --out out/

# openPMD output
python -m linac_gen run cell.lgproj --format openpmd --out out/
```

Runnable versions are in
[`examples/batch_mode/01_run_envelope`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/examples/batch_mode/01_run_envelope) …
[`05_run_output_formats`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/examples/batch_mode/05_run_output_formats).

## Cross-references

* [Batch-mode overview](06_batch_cli.md) — the input / override model.
* [CLI: scan](08_cli_scan.md) — sweep a parameter instead of one run.
* [Reading results](04_results.md) — the output file formats.

← [Batch-mode overview](06_batch_cli.md) ·
[Continue to CLI: scan →](08_cli_scan.md)
