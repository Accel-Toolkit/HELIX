# Surrogates CLI

Four subcommands wrap the surrogate library for batch and headless
use.  Invoke as:

```bash
python -m linac_gen.surrogates.cli <subcommand> [options]
```

| Subcommand | What it does |
|---|---|
| `train`          | Train a surrogate for one element of a lattice |
| `compare`        | Baseline-vs-surrogate envelope diff + optional PNG |
| `run-envelope`   | Run one envelope pass with surrogates engaged |
| `register-multi` | Register one trained surrogate under multiple element names |

## `train`

Train a `FieldMap3D` surrogate.  Writes `weights.pt` +
`metadata.json` to `--out` (or the per-lattice default
`linac_gen/surrogates/weights/<hash16>/<element>/`).

```bash
python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt/mebt.dat \
    --element FMAP_001 \
    --samples 50000 \
    --epochs 300 \
    --hidden 128,128,128 \
    --workers 12
```

### Arguments

| Flag | Default | Notes |
|---|---|---|
| `--lattice` (required) | — | TraceWin `.dat` file. |
| `--element` (required) | — | element name OR 0-based index. |
| `--samples` | `200` | LHS samples; smoke = 200, production ≥ 50 000. |
| `--epochs`  | `40`  | training epochs; smoke = 40, production 200-500. |
| `--hidden`  | `64,64` | comma-separated MLP hidden-layer sizes. |
| `--w-kin-range` | `2.0,2.5` | `(lo,hi)` MeV scope on incoming kinetic energy. |
| `--ke-rel`  | `0.20` | fractional sweep around the element's current `ke`. |
| `--phase-rel-deg` | `20.0` | absolute degree sweep around current phase. |
| `--ref-w-kin` | `2.12` | template ref kinetic energy (MeV). |
| `--frequency` | `162.5` | template ref frequency (MHz). |
| `--out` | (auto) | output dir; auto = `weights/<hash16>/<elem>/`. |
| `--seed` | `42` | LHS + train RNG seed. |
| `--workers` | `1` | CPU worker count for data generation; ≥2 enables `multiprocessing.spawn`. |

!!! tip "Workers ≥ 2 is the single biggest speedup knob"
    Pass `--workers $(python -c 'import os; print(max(1, os.cpu_count() - 2))')` —
    on a 14-core M3 Max this drops a 500-sample MEBT-buncher cycle from
    22 min to ~60 s (21× speedup) with bit-identical output (the LHS
    seed and per-sample code path are unchanged; only dispatch is
    parallel).

### Worked example

```bash
# Smoke: ~3 min, val MAPE ~5%
python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt/mebt.dat \
    --element FMAP_001 \
    --samples 200 --epochs 40 \
    --workers 12

# Production: ~1.5 h, val MAPE <1%
python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt/mebt.dat \
    --element FMAP_001 \
    --samples 50000 --epochs 300 \
    --hidden 128,128,128 \
    --workers 12
```

## `compare`

Run the envelope twice (baseline vs surrogate-enabled) and emit a
diff report.  Surrogates are loaded from one or more weights
directories.

```bash
python -m linac_gen.surrogates.cli compare \
    --lattice examples/pipii/mebt/mebt.dat \
    --weights linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_001 \
              linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_002 \
    --current 5.0 \
    --out /tmp/mebt_compare.png
```

### Arguments

| Flag | Default | Notes |
|---|---|---|
| `--lattice` (required) | — | TraceWin `.dat`. |
| `--weights` (required) | — | one or more weights directories. |
| `--out` | none | optional PNG of σ_{x,y,φ,W} overlay. |
| `--ref-w-kin` / `--frequency` | 2.12 / 162.5 | ref-particle defaults. |
| `--alpha-{x,y,z}`, `--beta-{x,y,z}`, `--emit-{x_n,y_n,z}` | MEBT entry | initial Twiss. |
| `--current` | 0.0 | beam current in mA; **set > 0 to exercise the SC slice path** (where the M3 hook fully engages). |

### Output

```
loaded surrogate 'FMAP_001' (val MAPE 4.28e-02)
loaded surrogate 'FMAP_002' (val MAPE 5.56e-02)

Running baseline vs surrogate-enabled envelope...

Surrogates engaged: 2 (FMAP_001, FMAP_002)
Wall-clock: baseline 6.64 s  surrogate 4.95 s  speedup 1.34x
End-of-line sigma moments:
     sigma_x  baseline= 1.7951e+00  surrogate= 1.7892e+00  rel.diff=3.3e-03  mm
     sigma_y  baseline= 1.7905e+00  surrogate= 1.7944e+00  rel.diff=2.2e-03  mm
   sigma_phi  baseline= 6.5356e+00  surrogate= 6.4209e+00  rel.diff=1.8e-02  deg
     sigma_w  baseline= 1.1637e-02  surrogate= 1.1681e-02  rel.diff=3.8e-03  MeV
Worst rel.diff: 1.8e-02
Scope OK: True

plot saved to: /tmp/mebt_compare.png
```

## `run-envelope`

Single envelope pass with one or more surrogates engaged — useful
for benchmarking or sanity checks without baselines.

```bash
python -m linac_gen.surrogates.cli run-envelope \
    --lattice examples/pipii/mebt/mebt.dat \
    --use-surrogates linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_001 \
                     linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_002 \
                     linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_003 \
                     linac_gen/surrogates/weights/bb088243c9fb33ce/FMAP_004 \
    --current 5.0
```

Output:

```
registered surrogate 'FMAP_001' (val MAPE 4.28e-02)
... (3 more)
registered 4 surrogates total.

End-of-line:  sigma_x=1.7822 mm  sigma_y=1.8221 mm  ...
```

The same beam-config flags as `compare` apply.

## `register-multi`

Register one trained surrogate under multiple element names — e.g.
share one HWR-cavity surrogate across all 8 cavities in the
cryomodule.  Useful when the underlying `.map` file is the same and
the per-element operating ranges overlap.  At TRACKING time the
lookup is by element name **guarded by the element in hand**
(`get_for_element`, 2026-07; the envelope, MP and backtrack
consumers all use it, last registration wins) — so each element name
needs an explicit registration even when the physics is identical.
The guard verifies the registered surrogate actually wraps the
element being tracked (object identity, or a structural fingerprint —
class, length, field file, drive parameters — so re-loading the same
deck keeps the surrogate engaged): a same-named element from a
**different** lattice (auto-generated names like `FMAP_001` collide
across decks by construction) is *skipped* with a once-per-name
warning and the native element tracks — it can no longer silently
receive the wrong lattice's learned physics.  Registering a
cross-lattice name additionally warns at registration time.  The
`(lattice_hash, element_key)` key exists alongside for on-disk weight
paths and GUI presence checks.

!!! warning "In-process only"
    The registry is an in-process singleton and the CLI process exits
    when the command returns — **nothing is persisted**.  As a
    standalone command `register-multi` therefore only *validates*
    that the surrogate loads and matches the named elements (it
    prints an explicit note saying so).  For actual runs, register
    from the GUI Surrogates tab, or call
    `linac_gen.surrogates.registry.register()` in the same Python
    process as the tracking.  (Until the 2026-07 honesty round this
    command crashed with an `AttributeError` before registering
    anything.)

```bash
python -m linac_gen.surrogates.cli register-multi \
    --source linac_gen/surrogates/weights/<hash16>/CAV1 \
    --lattice examples/pipii/mebt+hwr/mebt+hwr.dat \
    --names CAV1 CAV2 CAV3 CAV4 CAV5 CAV6 CAV7 CAV8
```

### Arguments

| Flag | Default | Notes |
|---|---|---|
| `--source` (required) | — | weights directory of an already-trained surrogate (the one to share).  Must contain `weights.pt`. |
| `--lattice` (required) | — | TraceWin `.dat` — used to compute the lattice hash for the weight-path/GUI bookkeeping key (tracking itself resolves by element name; see above). |
| `--names` (required) | — | one or more element names under which to register this surrogate. |

Names not found in the lattice are skipped with a `SKIP` line.  As a
quick out-of-distribution check, the source surrogate's trained
`w_kin` scope is printed alongside each element's current
`ke` / `kb` / `phase` values, so OOD risk is visible before it bites
at match time.  Exit status is non-zero if nothing was registered.

## Common patterns

### Batch-train all field maps in a lattice

```bash
ELEMS=$(python -c '
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.field_map import FieldMap
lat, _ = parse_tracewin("examples/pipii/mebt+hwr/mebt+hwr.dat")
for e in lat.elements:
    if isinstance(e, (FieldMap, FieldMap3D)):
        print(e.name)
')

for elem in $ELEMS; do
  python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt+hwr/mebt+hwr.dat \
    --element "$elem" \
    --samples 50000 --epochs 300 \
    --workers 12
done
```

### Compare against an existing trained set

```bash
python -m linac_gen.surrogates.cli compare \
  --lattice examples/pipii/mebt+hwr/mebt+hwr.dat \
  --weights linac_gen/surrogates/weights/$(python -c '
from linac_gen.surrogates.registry import hash_lattice_file as h
print(h("examples/pipii/mebt+hwr/mebt+hwr.dat")[:16])
')/*/ \
  --current 5.0 \
  --out /tmp/mebt_hwr_compare.png
```

## Cross-references

* [Overview](01_overview.md) — what surrogates are, when to use.
* [GUI walkthrough](02_gui.md) — equivalent flow in the GUI.
* [Python API](04_python_api.md) — what the CLI wraps.
* [Training guide](05_training_guide.md) — sample sizing, OOD,
  multiprocessing details.

← [GUI walkthrough](02_gui.md) ·
[Continue to Python API →](04_python_api.md)
