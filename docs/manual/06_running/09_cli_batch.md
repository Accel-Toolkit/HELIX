# CLI: `batch`

A multi-run campaign from a JSON job file.  Where [`scan`](08_cli_scan.md)
sweeps one lattice over a regular grid, `batch` runs an arbitrary list
of **heterogeneous** jobs — different lattices, beams, element settings
and modes — and collects one row of metrics per job.

```
python -m linac_gen batch <jobfile.json> [options]
```

## Options

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `jobfile` | path | (required) | the JSON job file |
| `--out` | `DIR` | `.` | directory for `batch_summary.csv` |
| `--parallel` | int | `1` | worker processes; `>1` runs jobs in a process pool |
| `-q`, `--quiet` | flag | off | suppress progress output |

## The job file

JSON — either a bare list of jobs, or an object with a `"jobs"` key:

```json
{
  "jobs": [
    {
      "name": "baseline",
      "input": "cell.lgproj",
      "mode": "envelope"
    },
    {
      "name": "low_current",
      "input": "cell.lgproj",
      "mode": "envelope",
      "beam": {"current": 1.0}
    },
    {
      "name": "softer_quad",
      "input": "cell.dat",
      "mode": "mp",
      "beam": {"energy": 800, "species": "H-"},
      "set": {"QF.gradient": 4.0},
      "sc": {"kernel": "tsc"}
    }
  ]
}
```

### Job fields

| Field | Required | Meaning |
|---|---|---|
| `name` | no | label for the summary row (default `job1`, `job2`, …) |
| `input` | **yes** | a `.lgproj` project or a `.dat` / `.madx` lattice |
| `mode` | no | `envelope` (default) or `mp` — any value other than `envelope` runs multi-particle (there is no `matrix` job mode; use [`run --mode matrix`](07_cli_run.md) for that) |
| `beam` | no | object of `BeamConfig` overrides |
| `set` | no | object of element-parameter overrides (`"ELEM.attr": value`) |
| `sc` | no | object of `SpaceChargeConfig` overrides |
| `env_solver` | no | `matrix` (default) / `sacherer` |
| `seed` | no | RNG seed (default `42`) |

A relative `input` path is resolved against the current directory
first, then against the job-file's own directory — so a job file stays
portable next to its lattices.

## Output

`batch` writes **`batch_summary.csv`** into `--out` — one row per job:

```
name, input, mode, sigma_x, sigma_y, sigma_phi, sigma_w,
emit_x, emit_y, emit_z, transmission, ref_w_kin, x_max, y_max, elapsed
```

All jobs run through the same process pool as `scan`; `--parallel N`
runs `N` at a time.  For a job's *full* per-step results, run that case
with [`run`](07_cli_run.md) — `batch` collects end-of-lattice scalars
only.

## Examples

```sh
# Run a campaign, summary into ./runs/
python -m linac_gen batch jobs.json --out runs/

# Four jobs at a time
python -m linac_gen batch jobs.json --parallel 4 --out runs/
```

A complete, runnable job file and script are in
[`examples/batch_mode/09_batch`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/examples/batch_mode/09_batch).

## Cross-references

* [Batch-mode overview](06_batch_cli.md) — the input / override model.
* [CLI: scan](08_cli_scan.md) — a regular sweep of one lattice.
* [CLI: run](07_cli_run.md) — a single run, full per-step output.

← [CLI: scan](08_cli_scan.md) ·
[Continue to CLI: twiss →](10_cli_twiss.md)
