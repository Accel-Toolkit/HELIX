# Running an error study

Three paths to launch an error study: Python script, GUI Error Study tab,
or a `.dat` with embedded `ERROR_*` directives.

## Path 1: Python (programmatic)

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.errors.error_model import ErrorStudy

lattice, _ = parse_tracewin("examples/error_studies/quad_alignment_tolerance/quad_alignment.dat")

beam_cfg = BeamConfig(
    species="proton", energy=2.5, frequency=325.0,
    current=5.0, n_particles=5000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.25, emit_ny=0.25, emit_z=0.30,
    alpha_x=0.0, beta_x=0.5,
    alpha_y=0.0, beta_y=0.5,
    alpha_z=0.0, beta_z=1.0,
)

study = ErrorStudy(lattice, beam_cfg, n_seeds=100)
# (errors auto-loaded from the .dat's ERROR_QUAD_NCPL_STAT)
results = study.run()

stats = results.transmission_stats()
print(f"Mean transmission: {stats['mean']:.2f} %")
print(f"  range over seeds: {stats['min']:.2f} - {stats['max']:.2f}")
sx_mean = results.mean("sigma_x")
sx_std = results.std("sigma_x")
print(f"σ_x at end (mean ± 1σ): {sx_mean[-1]:.3f} ± {sx_std[-1]:.3f}")
```

`ErrorStudy.run()`:

1. Draws a per-seed sample of every error (alignment, field,
   beam input).
2. Applies the perturbation to a per-seed lattice/beam copy.
3. Tracks the multi-particle beam through the perturbed lattice
   (**always multi-particle** — there is no envelope mode; pass
   `sc_config` to add PIC space charge).
4. Keeps the seed's `DiagnosticRecorder` (whole seeds only).
5. Returns an `ErrorStudyResults` that aggregates across seeds on
   demand (`mean` / `std` / `percentile` / `transmission_stats`).

### `ErrorStudy` constructor

```{.python .skip}
ErrorStudy(
    lattice: Lattice,
    beam_config: BeamConfig,
    n_seeds: int = 100,
    sc_config: SpaceChargeConfig | None = None,   # None = no space charge
    base_seed: int = 0,                # RNG offset, for reproducibility /
                                       # extending a study with new seeds
)
```

### `run()` runtime contract

```python
results = study.run(n_workers=1, should_stop=None, progress_cb=None)
```

* `should_stop` — optional zero-argument callable, polled at the top
  of each seed loop *and* re-checked after tracking.  A stop request
  mid-seed **discards that seed entirely** — a truncated recorder
  never enters the ensemble (its arrays would be shorter than the
  others', corrupting every mean/std/percentile).
* `progress_cb` — optional callable invoked as
  `progress_cb(i_done, n_requested)` after each completed seed.
* On early stop the returned results carry only whole seeds:
  `results.n_seeds` is the completed count and `results.n_requested`
  the asked-for count, so callers can report "k of n".
* `base_seed` (constructor) offsets every per-seed RNG draw — two
  studies with the same configuration but different `base_seed`
  values produce statistically independent ensembles.
* `n_workers` is accepted for forward compatibility but **currently
  ignored** — seeds always run serially (a value > 1 emits a warning;
  for parallel ensembles run several studies with different
  `base_seed` via `linac_gen.parallel.scan_pool`).

!!! note "Draw conventions (changed 2026-07)"
    Gaussian errors are truncated by **redrawing** until the value
    falls inside ±cutoff·σ (TraceWin convention) — the old clip-based
    truncation piled tail mass exactly on the limits.  RNG streams are
    now purpose-separated per seed (element / beam / BPM-noise), and
    BPM noise varies per seed instead of repeating one sequence across
    the whole ensemble.  Together with the FieldMap-amplitude fix this
    means **error-study results are not bit-reproducible against
    pre-2026-07 runs** — the new statistics are the correct ones.

### Adding errors programmatically

```python
study.add_error(
    pattern="QUAD_*",
    parameter="dx",
    distribution="gaussian",
    sigma=0.2,
    cutoff=3.0,
)
study.add_beam_error(
    parameter="centroid_x",
    distribution="gaussian",
    sigma=0.05,
)
```

These add to the auto-loaded list — you can mix `.dat` ERROR_*
directives and Python `add_error()` calls.

## Path 2: GUI Error Study tab

1. Open the lattice (File → Open Lattice or `Ctrl+O`).
2. Open the **Error Study** tab (between Surrogates and Failure Study).
3. Either:
   * If the `.dat` has `ERROR_*` directives, they're already
     loaded; the form shows them in the "Registered errors" list.
   * Or use the form to add element / beam errors interactively.
4. Set `n_seeds` (default 50; 100-200 for production).
5. Click **Run study**.  Status bar shows progress; results land
   on `state.error_study_results`.
6. Open **Results** tab → click the **Error study ensemble** tile.

For details see [GUI → Error Study tab](../10_gui/06_errors_tab.md).

## Path 3: `.dat` with `ERROR_*` directives

The minimum-effort path: write a TraceWin-compatible `.dat`,
embed `ERROR_*` cards, and HELIX figures out the rest.  Five
example projects under `examples/error_studies/` show the
convention:

```
examples/error_studies/
├── quad_alignment_tolerance/
├── quad_field_errors/
├── cavity_rf_jitter/
├── beam_input_jitter/
└── combined_realistic/
```

Each is a self-contained `.lgproj` you can open in the GUI and
run with one click.

## Performance tips

All rows are multi-particle runs (`ErrorStudy` has no envelope mode);
"no SC" means `sc_config=None`, "MP+SC" means a PIC `sc_config` was
passed.  Times are indicative and scale with `n_particles`.

| Lattice | Seeds | Mode | Wall time |
|---|---|---|---|
| Small (~10 elements, 5000 particles) | 50 | MP, no SC | ~5 s |
| Small | 50 | MP+SC | ~30 s |
| PIP-II MEBT | 100 | MP, no SC | ~1 min |
| PIP-II MEBT | 100 | MP+SC | ~30 min |
| Full PIP-II 256m | 100 | MP, no SC | ~10 min |
| Full PIP-II 256m | 100 | MP+SC | ~12 hours (don't!) |

For full-machine MP+SC studies, parallelise across machines or run
overnight.  Most tolerance work runs without space charge for speed —
final verification only with the PIC solver enabled.

## Cross-references

* [Overview](01_overview.md)
* [ERROR_* directives](02_error_directives.md)
* [Interpreting results](06_interpreting.md)
* [GUI → Error Study tab](../10_gui/06_errors_tab.md)
* [Worked example: tolerance study](../11_examples/08_tolerance_study.md).

← [Beam errors](04_beam_errors.md) ·
[Continue to Interpreting results →](06_interpreting.md)
