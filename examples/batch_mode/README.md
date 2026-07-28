# Batch-mode CLI — worked examples

Runnable examples for HELIX's headless batch-mode CLI
(`python -m linac_gen {run,scan,batch}`).  Each numbered folder is one
self-contained test case with a commented `run.sh`.

The full reference is in the manual:
[docs/manual/06_running/06_batch_cli.md](../../docs/manual/06_running/06_batch_cli.md).

## Shared input

Both shared files sit in this folder and are used by every case:

* `chicane.dat` — a compact 6-bend chicane lattice (TraceWin format).
* `chicane.lgproj` — a HELIX project wrapping `chicane.dat` with an
  800 MeV H⁻ beam and convergence settings.

## How to run

Run any case from the **repository root**:

```sh
sh examples/batch_mode/01_run_envelope/run.sh
```

Each `run.sh` `cd`s to the repo root itself, so it also works when
invoked directly.  Outputs are written into that case's `out/` folder.
To run them all:

```sh
for s in examples/batch_mode/*/run.sh; do sh "$s"; done
```

## The cases

| Folder | Subcommand | Demonstrates |
|---|---|---|
| `01_run_envelope` | `run` | envelope mode, `.lgproj` input, `--out` |
| `02_run_multiparticle` | `run` | `--mode mp`, `--n-particles`, `--nx`, `--write-dst`, `--fail-under-transmission` |
| `03_run_matrix` | `run` | `--mode matrix` (6×6 transfer matrix + Twiss) |
| `04_run_overrides` | `run` | `--energy/--current/--species`, `--beam`, `--set`, convergence flags |
| `05_run_output_formats` | `run` | `--format hdf5 / openpmd / partran` |
| `06_scan_beam` | `scan` | `--vary` a beam parameter over a range |
| `07_scan_element` | `scan` | `--vary` an element parameter over a value list |
| `08_scan_2d_parallel` | `scan` | multiple `--vary` (Cartesian product), `--parallel` |
| `09_batch` | `batch` | a JSON job file → `batch_summary.csv` |

Together they exercise every option of the batch-mode CLI.
