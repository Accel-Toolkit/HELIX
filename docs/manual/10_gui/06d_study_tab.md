# Param Study tab

The Parameter Study Manager: vary one or more lattice, beam or
numerics parameters over a chosen strategy, run one **headless
simulation per point** (each run is an independent subprocess-or-serial
execution of the *saved* lattice file), and analyze the results without
leaving the GUI.  The engine is `linac_gen.study` — the same machinery
behind `python -m linac_gen study` ([CLI: study](../06_running/12_cli_study.md))
and the assistant's `run_study` tool, so studies started in any of the
three are interchangeable: the folder on disk is the source of truth.

> **The study runs the file on disk.**  Unsaved lattice edits are
> refused (not silently ignored): save the lattice first.  This is
> deliberate — a study whose runs disagree with the screen is worse
> than a refusal.

## Defining a study

1. **Parameters** — pick an element and one of its numeric parameters,
   then **Add**.  Each row takes a `start` / `stop` / `n` range (edit
   the cells directly; `spacing` is `lin` or `log`).  Beam and
   numerics knobs use the same selector grammar as the CLI
   (`NAME.attr`, `@N.attr` with 1-based element numbering, a bare
   `BeamConfig` field such as `current`, or `nx` / `grid_extent` /
   `step1` / `step2`).
2. **Strategy** —
   `oat` (one-at-a-time around the baseline), `zip` (vary rows in
   lockstep), `grid` (full cross product — watch the run counter),
   `random` / `lhs` (space-filling samples, count set by *Samples*).
   *Seed repeats* re-runs every point with seeds `seed … seed+R−1`
   for multi-particle error bars.  The **Total runs** counter updates
   live.
3. **Execution** — study name, root folder (each study becomes
   `<root>/<name>/`), `envelope` or `mp` mode, and the worker count
   (one subprocess per run; 1 = serial in-process).

**Start study** creates the folder and begins; the progress bar and
status line track completed/failed runs with an ETA.  **Stop** halts
after the in-flight runs finish — a stopped (or crashed, or
power-cycled) study **resumes** from where it left off the next time
you press Start: completed runs are never re-executed.

## The study folder

```
<root>/<name>/
├─ study.json            # the full spec — re-runnable anywhere
├─ lattice/              # snapshot of the input deck (provenance)
├─ runs/run_00000_…/     # per run: spec.json, results.h5, status.json, log.txt
└─ summary/summary.csv   # parameters + auto KPIs + observables per run
```

`summary.csv` always carries the varied parameters, run status, and
the auto KPIs (final sizes, emittances, energy, transmission); any
extra observables defined in the spec are appended as columns.

## Analyzing results

The right-hand panel has four views, live-updating while the study
runs:

* **Runs** — one row per run with parameters, status and KPIs;
  double-click a row to open its `results.h5` in the Results tab.
* **1D** — an observable against one parameter, grouped by another;
  seed repeats collapse to mean ± std error bars; linear or log axes.
* **Map** — a 2-D heatmap on full grids (scatter fallback for
  random/LHS or partially complete studies).
* **Overlay** — σ(z) envelopes of selected runs on shared axes.

**Open study…** loads any existing study directory — including one
produced by the CLI or the assistant on another machine.  The tab
remembers the last root folder and reopens the last study on startup.

## Headless twins

Everything above is scriptable:

```bash
python -m linac_gen study plan  my_study.json          # preview runs
python -m linac_gen study run   my_study.json --parallel 4
python -m linac_gen study resume <dir> --retry-failed
```

or through the assistant: *"run a grid study of the second quad's
gradient, 8–16 T/m in 5 steps, observable sigma_x at the exit"* — the
`run_study` tool executes the same engine as a background job and
reports the summary location.
