# Batch-mode CLI — overview

HELIX runs headless from the shell via `python -m linac_gen`. No GUI,
no hand-written driver scripts — the CLI drives the very same
`Simulation` / `EnvelopeSolver` / scan-pool engines the GUI uses, so a
batch run is **bit-identical** to the equivalent GUI run.

```
python -m linac_gen <subcommand> …
```

| Subcommand | Purpose | Page |
|---|---|---|
| `run` | one headless simulation (envelope / mp / matrix) | [CLI: run](07_cli_run.md) |
| `scan` | sweep one or more variables → CSV summary | [CLI: scan](08_cli_scan.md) |
| `batch` | a multi-run campaign from a JSON job file | [CLI: batch](09_cli_batch.md) |
| `twiss` | matched Twiss — whole-lattice or FODO-cell input match | [CLI: twiss](10_cli_twiss.md) |
| `backtrack` | backward tracking — reconstruct an upstream distribution from a downstream state | [CLI: backtrack](11_cli_backtrack.md) |
| `mo` | multi-objective design — Pareto front over `ADJUST` knobs | [Multi-objective](../07_matching/08_multiobjective.md) |
| `failures` | element failure impact + recovery → CSV | [Failure studies](../10_gui/06c_failures_tab.md) |
| `match` | the matcher — delegates to `python -m linac_gen.matching` | [Matching CLI](../07_matching/04_cli.md) |

## The input model

Every subcommand takes an **input** that is either:

* a **lattice file** — `.dat` (TraceWin) or `.madx` / `.seq` (MAD-X);
  the beam then starts from `BeamConfig` defaults; or
* a **`.lgproj` project** — beam and convergence settings are read from
  the file (it is the same project the GUI saves).

Command-line options then **override individual scalars** on top of
that — the "configure once, override per run" model.  The resolution
priority for any setting is:

```
   command-line option   >   project (.lgproj) value   >   built-in default
```

This is what makes parameter scans and campaigns possible: keep a
project file as the baseline, and vary only what changes from the shell.

## Three kinds of override

| Mechanism | Targets | Example |
|---|---|---|
| **Beam** | any `BeamConfig` field | `--current 5` · `--beam emit_nx=0.3` |
| **Element** | any element parameter | `--set QF.gradient=8.5` · `--set @12.angle=10` |
| **Convergence / PIC** | step density, grid, kernel, backend | `--nx 64` · `--step1 100` · `--sc green_kind=point` |

An **element selector** is `NAME.attr` (the element whose `.name` is
`NAME`) or `@N.attr` (the N-th element, 1-based, over `lattice.elements`).

## Output

`run` writes a results file per the `--format` flag; `scan` and `batch`
write a CSV summary.  All formats are the same writers the GUI uses —
see [Reading results](04_results.md).

## Exit codes

The process exit code lets shell scripts and CI branch on the outcome:

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the run / scan / batch failed, or a `--fail-under-transmission` threshold was breached |
| `2` | bad arguments, or a missing input file |

## Worked examples

The folder [`examples/batch_mode/`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/examples/batch_mode) holds
one self-contained, runnable case per feature
(`01_run_envelope` … `09_batch`), each with a commented `run.sh`.  Run
any of them from the repository root:

```sh
sh examples/batch_mode/01_run_envelope/run.sh
```

## Cross-references

* [CLI: run](07_cli_run.md) · [CLI: scan](08_cli_scan.md) ·
  [CLI: batch](09_cli_batch.md) · [CLI: twiss](10_cli_twiss.md) — the
  per-subcommand reference.
* [Python API](01_python_api.md) — the same engines, called directly.
* [From the GUI](05_gui.md) — the interactive equivalent.

← [From the GUI](05_gui.md) ·
[Continue to CLI: run →](07_cli_run.md)
