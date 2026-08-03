# CLI: study — parameter studies with per-run folders

`python -m linac_gen study` runs multi-parameter studies headlessly:
every run gets its own folder with a full `results.h5` (provenance
included) and a `status.json`, execution is parallel and **resumable**,
and a `summary.csv` collects one row per run.

```bash
python -m linac_gen study plan  my_study.json          # expanded run table
python -m linac_gen study run   my_study.json --parallel 4
python -m linac_gen study run   runs/studies/my_study  # resume (default)
python -m linac_gen study summarize runs/studies/my_study
```

## The study.json spec

```json
{
  "__kind__": "linac_gen_study",
  "name": "quad_scan",
  "input": "linac.dat",
  "mode": "mp",
  "strategy": "grid",
  "parameters": [
    {"selector": "@2.gradient", "start": 6.0, "stop": 9.0, "n": 7},
    {"selector": "current",     "values": [0.0, 2.0, 5.0]}
  ],
  "observables": [
    {"name": "sx_mebt", "quantity": "sigma_x", "at": {"element": "QUAD_012"}}
  ],
  "beam":     {"energy": 2.1, "n_particles": 10000},
  "numerics": {"nx": 64, "grid_extent": 5.0},
  "repeats":  1
}
```

**Selectors** use the same grammar as `run --set` / `scan --vary`:
`NAME.attr` or `@N.attr` (1-based) for element parameters, a bare
`BeamConfig` field name for beam parameters, and
`nx / grid_extent / step1 / step2` for numerics.

**Strategies** — `oat` (one-at-a-time around per-parameter `baseline`s,
with an all-nominal reference as run 0), `zip` (equal-length lists
varied together), `grid` (Cartesian product), `random` and `lhs`
(Latin-Hypercube; both need `n_samples`, seeded by `sampler_seed`).
`repeats: R` runs every point with seeds `seed … seed+R−1` for MP
statistics.

**Observables** extract scalars from each run's results: any
`envelope/` quantity, at `"end"`, at `{"s_m": …}`, or at
`{"element": NAME}` (resolved against the lattice when the study is
created).

## Folder layout & resume semantics

```
quad_scan/
├── study.json                     # the spec (lattice SHA-256 pinned)
├── lattice/<deck>                 # provenance snapshot
├── runs/run_00000_<tag>/
│   ├── results.h5                 # written atomically (.part → rename)
│   └── status.json                # ok | failed, params, metrics
└── summary/{runs_manifest.json, summary.csv}
```

`run` skips every completed run — a crash (even `kill -9`) loses only
the in-flight runs, and re-running the same command finishes the rest.
`--force` wipes `runs/` and starts over; `--retry-failed` re-queues
failed runs.  Failed runs are **rows, not crashes**: one diverging
parameter combination never sinks the study.

Two integrity guards refuse loudly instead of mixing physics: the
input lattice is SHA-pinned at creation (editing it invalidates the
study), and the spec must still expand to the recorded run plan
(editing `study.json` after a partial run is detected).

## See also

* [CLI: scan](08_cli_scan.md) — flat CSV grid scans without per-run
  artifacts; `study` supersedes it when you need folders, resume,
  non-grid strategies, or observables.
* [CLI: batch](09_cli_batch.md) — heterogeneous job lists.
* `linac_gen/study/` — the engine API (`StudyManager`), usable directly
  from Python.
