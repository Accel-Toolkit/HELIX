# Training guide

Practical recipes for picking training parameters, scaling with
multiprocessing, managing the cached weights store, and diagnosing
out-of-scope / regression issues.

## Choosing sample count + epochs

Two officially-supported cycle sizes, calibrated against the M2
acceptance gate:

| Cycle | Samples | Epochs | Hidden | Wall (12-core M3 Max) | Val MAPE |
|---|---|---|---|---|---|
| **Smoke**      | 200    | 40  | `(64,64)`        | ~25 s        | 3–6 %    |
| **Production** | 50 000 | 300 | `(128,128,128)`  | ~1.5 h/elem  | < 1 %    |

The smoke cycle is for "does the pipeline work" sanity checks and
quick CI runs — its accuracy is **not** sufficient for science.
Production is the science setting.

If you need to interpolate between the two:

| n_samples | Expected val MAPE (MEBT buncher) |
|---|---|
| 200    | 4–6 %   |
| 1 000  | 1.5–2.5 % |
| 5 000  | 0.5–1 % |
| 20 000 | 0.2–0.5 % |
| 50 000 | < 0.1 % |

Epochs follow a similar pattern — diminishing returns past ~5×
samples/batch_size epochs.  Past 500 epochs you'll start seeing
overfit (val MAPE rises while train loss falls).

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

The RK4 `fitted_matrix(ref)` call dominates wall-clock (>99 %).
Multiprocessing the LHS loop is the only meaningful speedup knob.

### Benchmarks (MEBT buncher FMAP_001, M3 Max, 14 cores)

500-sample data gen + {ke, phase} sweep:

| Workers | Wall-clock | Speedup | Bit-identical? |
|---|---|---|---|
| 1   | 1342 s    | 1.00× | reference |
| 4   |  121 s    | 11.07× | yes |
| 8   |   75 s    | 17.87× | yes |
| 12  |   61 s    | 21.90× | yes |

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

| Machine | Cores | Default | Rationale |
|---|---|---|---|
| Apple Silicon M-series | 14 | 12 | leave 2 cores for OS + GUI |
| Apple Silicon M Pro / Max | 8-16 | `cores - 2` | same |
| Linux server | 32-128 | `cores - 4` | leave headroom for I/O |
| CI runner | 2-4 | 1 | spawn overhead > work for small N |

In the GUI the default is `max(1, cpu_count() - 2)`.  In the CLI
the default is 1 (backward compat — pass `--workers N` explicitly).

## Persistence and cache management

### Where weights live

```
linac_gen/surrogates/weights/
└── <SHA256[:16]>/             ← lattice .dat content hash
    └── <element-name>/
        ├── weights.pt          ← MLP state_dict (torch)
        └── metadata.json       ← scope, seed, commit SHA, val MAPE, etc.
```

### What's in `metadata.json`

```json
{
  "element_key": "FMAP_001",
  "element_class": "FieldMap3D",
  "architecture": {
    "input_dim": 5, "output_dim": 36,
    "hidden_dims": [128, 128, 128], "activation": "silu",
    "param_names": ["ke", "kb", "phase"]
  },
  "scope": {
    "input_names": ["w_kin", "beta", "gamma", "ke", "kb", "phase"],
    "input_lo": [2.0, 0.0, 1.0, 0.0544, 0.0544, -110.0],
    "input_hi": [2.5, 1.0, 1000.0, 0.0816, 0.0816, -70.0]
  },
  "input_norm": {"mean": [...], "std": [...]},
  "output_norm": {"mean": [...], "std": [...]},
  "training_seed": 42, "n_samples": 50000, "epochs": 300,
  "val_mape": 0.0073,
  "helix_commit_sha": "e3bd592a...",
  "lattice_hash": "bb088243c9fb33ce6f9a...",
  "created_iso": "2026-05-24T15:23:11Z"
}
```

The metadata is required for reload — never delete it without
deleting `weights.pt` alongside.

### What invalidates a cached surrogate

| Change | Invalidates? | Why |
|---|---|---|
| You edited the `.dat` file | ✅ | new lattice hash → new dir |
| Element renamed in the `.dat` | ✅ | name is the cache key |
| You bumped a sweep range | ⚠️ | old weights still load but scope shrinks/grows are stale |
| HELIX commit SHA changed | ⚠️ | physics may differ; rerun reproducibility CI to check |
| Re-trained with different `seed` | ❌ | weights differ but both valid |

### Auto-discovery in the GUI

On lattice load the Surrogates tab scans the per-lattice weights
dir and adds a row to the Trained-surrogates table for every
subdirectory whose name matches a field-map element in the lattice.
**Use** ticks default off (deliberate — registering loud).

Cache-aware Train: clicking **Train** for an element with cached
weights opens a 3-way dialog (Open / Retry / Cancel) so re-running
the smoke cycle is a deliberate act, not the default.

### Sharing weights across machines

A weights directory is fully self-contained — copy
`linac_gen/surrogates/weights/<hash>/<elem>/` to another machine
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

The surrogate **isn't being engaged** — most likely you're at
`current = 0` (no SC) and the envelope path skipped
`fitted_matrix_slice`.  Run at ≥ 1 mA to exercise the SC slice
path where the M3 hook engages.

### Val MAPE high (> 5 %) despite production cycle

Causes (in order of frequency):

1. **Scope too wide** — try halving `ke_rel` / `phase_rel_deg`.
2. **MLP too small** — try `hidden_dims=(256, 256, 256)`.
3. **Element has hidden state** — `FieldMap3D._sync_offset_deg` not
   represented in the MLP input.  Currently a known gap; see the
   roadmap for state-aware surrogates.

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
