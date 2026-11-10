# Training guide

Practical recipes for picking training parameters, scaling with
multiprocessing, managing the cached weights store, and diagnosing
out-of-scope / regression issues.

## Choosing sample count + epochs

How a training fits the network (`train_surrogate`):

* every input and output is normalised by its mean and spread over the
  training split, an output's spread floored at `OUTPUT_STD_FLOOR` (1e-9).
  A map's structurally zero entries (the transverse-to-longitudinal
  couplings of a symmetric cavity) hold only round-off, spread about 1e-15
  to 2e-12 in the example decks' weights; normalised by that spread they weighed
  as much as a real entry and held 99.8 % of a cavity's validation error;
* Adam minimises the mean squared error of the normalised outputs, its
  learning rate falling along a half cosine from `lr` to 1 % of it at the
  last epoch;
* the epoch kept is the one with the lowest validation error, in the same
  units.  A longer run returns the best epoch it passed, not its last; the
  rate's decay is spread over the epochs given, so a run of 200 epochs is
  not the first 200 epochs of a run of 1000.

The weights file records this recipe (`train_recipe`, below).

| Cycle | Samples | Epochs | Hidden | For |
|---|---|---|---|---|
| **Smoke**      | 200   | 40    | `(64,64)`   | checking that the pipeline runs; weights this small have failed the [accuracy check](01_overview.md#the-accuracy-check-of-a-weights-file) |
| **Production** | 5 000 | 1 000 | `(128,128)` | use; the Surrogates tab's **Train all FieldMap\*** defaults |

The production cycle of one HWR cavity took about a minute on one core:
45–53 s to generate the 5000 samples and about 14 s to train.  Its
weights, and those of two MEBT bunchers, met the accuracy check's limits
with a margin of at least 1.9× (block error 2.6e-4 to 1.3e-3 against 3e-3,
determinant error 7e-5 to 5.3e-4 against 1e-3), where the earlier recipe
(spread + 1e-12 with no floor, a constant rate, the epoch kept by
validation MAPE) at the same samples and network missed them.  The 46
weight sets of the development tree's example decks retrained with this
recipe meet them too: the check's 17 points find block errors up to 1.3e-3
and determinant errors up to 5.3e-4 on their decks.  Between its points
the limits can be missed: on 1000 random points of their ranges, most of
the cavity sets exceed a limit at up to 7, every one within 2.1 % of an
edge of the range.

The **val MAPE** a training reports is the mean relative error over the
outputs that vary (spread above the floor): over the zero entries a
relative error measures round-off.  It is a relative error per entry, so
small entries weigh in it; whether weights are used is decided by the
accuracy check, not by it.

## Choosing the sweep ranges

`param_ranges` and `ref_w_kin_range` define the **scope** — the
region of input space the surrogate is trained on.  Outside scope
the surrogate raises `OutOfScopeError` and the envelope hook falls
back to RK4.  So the scope should cover:

* every operating point you intend to query at,
* plus a margin to absorb optimisation excursions (matching can
  push knobs by ~20 %),
* but **not so wide** that the surrogate has to model unnecessary
  regions — every extra dimension of scope dilutes the training
  set's effective density.

### Defaults that work well in practice

| Knob | Default | Rationale |
|---|---|---|
| `ref_w_kin_range` | ±10 % around nominal | matches typical longitudinal Twiss spread |
| `ke ± rel` | 0.20 (= ±20 %) | covers cavity-amplitude scans |
| `phase ± deg` | 20.0 | covers RF-phase scans + matcher excursions |
| `kb ± rel` | 0.20 | for solenoids; matches typical field-tune range |

If your study scans a knob by ±5 %, a ±20 % training scope is
generous and likely fine.  If you scan by ±50 %, train wider —
or split into two surrogates with overlapping scopes.

### Per-element dynamic detection

The GUI dialog auto-detects which knobs are active on each element
(`_detect_sweep_params`):

* **Cavity** (`ke ≠ 0`, `kb ≠ 0`) — sweeps `ke + kb + phase`.
* **Solenoid** (`ke = 0`, `kb ≠ 0`) — sweeps `kb` only.  Phase is
  suppressed because a pure-magnetic field has no electric channel
  for phase to modulate.
* Zero amplitude → that knob is dropped (it's an inactive channel).

You can override via the CLI's `--ke-rel`, `--phase-rel-deg`, or
via `param_ranges` in the Python API.

## Multiprocessing the data generation

The RK4 `fitted_matrix(ref)` calls of the data generation take most of a
training's time: 41 s of the production cycle's 55 s for a MEBT buncher on
one core.  Several workers cut them about 3×.

### Benchmarks (MEBT buncher FMAP_001, 14-core Apple M-series)

The data of the production cycle, 5000 samples over its {ke, phase} sweep
(measured 2026-11-10):

| Workers | Wall-clock | Speedup | Bit-identical? |
|---|---|---|---|
| 1   | 40.8 s | 1.00× | reference |
| 4   | 14.3 s | 2.85× | yes |
| 8   | 12.3 s | 3.32× | yes |
| 12  | 13.6 s | 2.99× | yes |

A sample takes about 8 ms, so starting the worker processes, a few
seconds, is most of a short run: 500 samples took 4.4 s on one worker and
9.8 s on 12.

### Why bit-identical matters

Each worker process runs the **exact same scalar** `fitted_matrix(ref)`
call that the serial path runs — same `deepcopy`, same
`reset_run_state()`, same RK4.  Only the dispatch is parallel.  The
LHS sample design is generated once (deterministic from the seed),
so for a fixed seed the `(X, Y)` training set is **byte-for-byte
identical** to the serial path regardless of worker count.  This
means:

* Trained MLP weights are reproducible — same seed + same code →
  same weights, no matter how many workers.
* The reproducibility CI gate (commit SHA + seed + lattice hash →
  identical weights) stays valid.
* You can mix worker counts across machines without changing the
  scientific results.

### Recommended worker counts

For the production cycle 4 to 12 workers do about equally well (the table
above); for a smoke cycle of a few hundred samples one worker is fastest.

In the GUI the default is `max(1, cpu_count() - 2)`.  In the CLI
the default is 1 (backward compat — pass `--workers N` explicitly).

## Persistence and cache management

### Where weights live

```
linac_gen/surrogates/weights/
└── <SHA256[:16]>/             ← lattice .dat content hash (+ _proton, _deuteron)
    └── <element-name>/
        ├── weights.pt          ← MLP state_dict (torch)
        └── metadata.json       ← scope, seed, commit SHA, val MAPE, etc.
```

### What's in `metadata.json`

The file of the MEBT deck's first buncher, trained with the windows and
defaults of the Surrogates tab's **Train all FieldMap\*** (the
normalisations shortened):

```json
{
  "element_key": "FMAP_001",
  "element_class": "FieldMap3D",
  "architecture": {
    "input_dim": 6, "output_dim": 36,
    "hidden_dims": [128, 128], "activation": "silu",
    "param_names": ["ke", "kb", "phase"]
  },
  "scope": {
    "input_names": ["w_kin", "beta", "gamma", "ke", "kb", "phase"],
    "input_lo": [2.0165, 0.0, 1.0, 0.0544, 0.0544, -110.0],
    "input_hi": [2.2288, 1.0, 1000.0, 0.0816, 0.0816, -70.0]
  },
  "input_norm": {"mean": [...], "std": [...]},
  "output_norm": {"mean": [...], "std": [...]},
  "training_seed": 42, "n_samples": 5000, "epochs": 1000,
  "val_mape": 0.00405,
  "helix_commit_sha": "45d89d5088f5...",
  "lattice_hash": "7429fb95147c412dc782...",
  "created_iso": "2026-11-10T14:36:27.893034Z",
  "species_name": "H-", "species_mass_mev": 939.29408606,
  "train_frequency_mhz": 162.5,
  "train_phi_s_deg": 2553.92,
  "train_settings": {"scale": 1.0, "frequency": 162.5,
                     "sync_phase_pin": null, "integrator_kind": "kd",
                     "interp_kind": "linear"},
  "train_recipe": {"optimizer": "adam",
                   "loss": "mean squared error of the normalised outputs",
                   "lr": 0.001, "lr_schedule": "cosine", "lr_final": 1e-05,
                   "batch_size": 256, "val_frac": 0.2,
                   "output_std_floor": 1e-09, "kept_epoch": 1000,
                   "kept_by": "lowest validation mean squared error",
                   "val_mse": 8.44e-05, "val_mape_outputs": 36,
                   "torch": "2.10.0", "numpy": "1.26.4"},
  "train_structure": {"class": "FieldMap3D", "length": 240.0,
                      "field_file": "QWR-2012-02", "geom": 7700,
                      "aperture": 20.0, "n_steps": 200, "p_flag": 1,
                      "ki": 0.0, "ka": 0}
}
```

`train_phi_s_deg` is the reference phase at the element's entrance, as the
tab's envelope run counts it from the start of the line; the buncher is
after `SET_SYNC_PHASE` (`p_flag` 1), so its map does not depend on it.

`train_settings` holds the element's settings the weights hold fixed (the
example's buncher was trained over `ke`, `kb` and `phase`, so neither
amplitude is among them): a query on an element set otherwise goes native.  Files from before it have none; see
[which element a surrogate answers for](01_overview.md#which-element-a-surrogate-answers-for).

`train_recipe` records how the network was fitted (optimiser, starting and
final learning rate and its schedule, batch size, validation split, output
floor, the kept epoch and its validation error, the number of outputs the
val MAPE is over, the torch and numpy versions), and `train_structure` what
the element was (its class and its structure: length, field file by name,
geometry, aperture, steps, phase reference `p_flag`, and `ki`/`ka` where the
element has them); both are records, which nothing checks, and files from
before them have neither.

`helix_commit_sha` is the commit of the HELIX checkout that ran the
training, or `unknown` for an installed copy.

The metadata is required for reload — never delete it without
deleting `weights.pt` alongside.

### What invalidates a cached surrogate

| Change | Invalidates? | Why |
|---|---|---|
| You edited the `.dat` file | Yes | new lattice hash → new dir |
| Element renamed in the `.dat` | Yes | name is the cache key |
| You bumped a sweep range | Partly | old weights still load but scope shrinks/grows are stale |
| HELIX commit SHA changed | Partly | physics may differ; rerun reproducibility CI to check |
| Re-trained with different `seed` | No | weights differ but both valid |
| The beam's species changed | Yes | a map answers only for its species; each species has its own folder |
| The clock changed | Not for a magnet | a magnet is converted; a cavity with no frequency of its own answers only on the clock it was trained at |
| The phase a cavity whose field follows it is entered at moved | Partly | its phase input is shifted, and may leave the scope |
| A setting the weights hold fixed changed (`kb`, when only `ke` and `phase` were swept) | For that element | each query goes native; `refusals` counts it, and a compare report lists it |
| Trained before the clock was recorded | Checked when loaded | a cavity map whose longitudinal determinant is not 1 holds a frequency jump: marked stale, it goes native |

### Auto-discovery in the GUI

On lattice load, and when the beam changes species, the Surrogates
tab scans the weights dir of that lattice and species and adds a row to the Trained-surrogates table for every
subdirectory whose name matches a field-map element in the lattice.
**Use** ticks default off (deliberate — registering loud).

Cache-aware Train: clicking **Train** for an element with cached
weights opens a 3-way dialog (Open / Retry / Cancel) so re-running
the smoke cycle is a deliberate act, not the default.

### Sharing weights across machines

A weights directory is fully self-contained — copy
`linac_gen/surrogates/weights/<hash>/<elem>/` (`<hash>_proton/` or
`<hash>_deuteron/` for those species) to another machine
that has the same `.dat` file (matching SHA), and the GUI / CLI
will pick it up.  The metadata records the HELIX commit SHA used
to train — verify against the destination machine's HEAD before
trusting science runs.

## Out-of-scope (OOD) handling

A surrogate is admissible at runtime only when each input
dimension is within its scope.  Outside scope:

* `surr.fitted_matrix(ref)` raises `OutOfScopeError`.
* The envelope hook catches it silently and falls back to the
  wrapped RK4.
* The run is **still correct** — just slower for those queries.

How to tell if you're hitting OOD often:

```{.python .skip}
# Wrap fitted_matrix to count.
ood_count = [0]
real_fm = surr.fitted_matrix
def counting_fm(ref, _real=real_fm, _c=ood_count):
    try:
        return _real(ref)
    except OutOfScopeError:
        _c[0] += 1
        raise
surr.fitted_matrix = counting_fm
# ... run envelope ...
print(f"OOD fall-backs: {ood_count[0]}")
```

If many: widen the training scope and retrain.  If a few near the
edges: usually fine.

## Troubleshooting

### "No FieldMap or FieldMap3D elements"

The lattice doesn't contain any surrogatable elements.  Pure
analytic-magnet lattices (drift + quad + bend) don't benefit from
surrogates; the speedup case is field-map elements where RK4 is
expensive.

### `OutOfScopeError` flood

The MLP refused most queries because the runtime input was outside
the trained scope.  Widen `ref_w_kin_range` and `param_ranges`,
retrain, or skip the surrogate for that element.

### Bit-identical compare (rel.diff = 0.00)

The surrogate **didn't answer**.  Either it was never queried — the
compare ran at `current > 0` (or with *Record per-sub-step* on), where the
envelope slice-walks field maps for SC kicks and every partial slice
delegates to the wrapped RK4; the report's `NN full-element queries` line
reads 0, with a note: run at **0 mA** to engage the surrogate on the
full-element matrix — or it refused each traversal: the report's notes list
them by reason (a setting the weights hold fixed changed, outside their
range, …).

### Weights refused by the accuracy check

Causes, most likely first:

1. **Trained with the earlier recipe**, or with the smoke cycle — retrain
   with the production cycle.  More samples, epochs or a larger network
   with the earlier recipe did not bring the refused HWR cavity weights
   within the limits.
2. **Trained on earlier physics** — a cavity after `SET_SYNC_PHASE`
   (`p_flag` 1) is calibrated from its entrance energy and settings for
   every sample and every run, so the calibration is a function of the
   inputs; but weights trained before its ±180° fix (October 2026) hold
   wrong targets where the cavity's offset crosses ±180°.  Retrain them.
3. **Scope too wide** — try halving `ke_rel` / `phase_rel_deg`.

### GUI crashes mid-training

Training runs in a background `QThread` — a crash in the worker
emits the `failed` signal which surfaces in a popup.  Check
`linac_gen/surrogates/weights/<hash>/<elem>/` for partial files
and delete them; the next training attempt will start clean.

### Cancelling a training run

Close the live progress dialog: the trainer polls a stop flag per
sample (data generation, including the multiprocessing branch —
its pool workers are terminated) and per epoch, then exits without
saving anything.  A cancel is reported as *cancelled*, never as a
training failure.

### Weights file refuses to load

`load_surrogate` requires both `weights.pt` AND `metadata.json` in
the directory.  Use `find_cached_surrogate(dir)` which returns
`None` instead of raising for partial / broken directories.

## Cross-references

* [Overview](01_overview.md) — what surrogates are.
* [GUI walkthrough](02_gui.md) / [CLI](03_cli.md) / [Python API](04_python_api.md).
* [Plan + roadmap](https://github.com/Abhishek-Pathak-90/HELIX/blob/master/docs/plans/surrogates.md).

← [Python API](04_python_api.md) ·
[Continue to Appendix A →](../appendices/A_physics_references.md)
