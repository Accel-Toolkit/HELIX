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

`--threads N|all`, given before the subcommand (`python -m linac_gen.surrogates.cli --threads 4 train …`), sets the CPU threads; training's worker processes share them ([CPU threads](../06_running/15_threads.md)).

## `train`

Train a `FieldMap3D` surrogate.  Writes `weights.pt` +
`metadata.json` to `--out` (or the per-lattice default
`linac_gen/surrogates/weights/<hash16>/<element>/`, and
`<hash16>_proton/` or `<hash16>_deuteron/` for those species, as the Surrogates tab
writes them).

```bash
python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt/mebt.dat \
    --element FMAP_001 \
    --samples 5000 \
    --epochs 1000 \
    --hidden 128,128 \
    --workers 12
```

### Arguments

| Flag | Default | Notes |
|---|---|---|
| `--lattice` (required) | — | TraceWin `.dat` file. |
| `--element` (required) | — | element name OR 0-based index. |
| `--samples` | `200` | LHS samples; smoke = 200, production 5000. |
| `--epochs`  | `40`  | training epochs; smoke = 40, production 1000 (the learning rate decays over them). |
| `--hidden`  | `64,64` | comma-separated MLP hidden-layer sizes. |
| `--w-kin-range` | `2.0,2.5` | `(lo,hi)` MeV scope on incoming kinetic energy. |
| `--ke-rel`  | `0.20` | fractional sweep around the element's current `ke`. |
| `--phase-rel-deg` | `20.0` | absolute degree sweep around current phase. |
| `--ref-w-kin` | `2.12` | template ref kinetic energy (MeV). |
| `--frequency` | `162.5` | template ref frequency (MHz).  The metadata records the clock of the targets: a cavity's own frequency, or this one for a magnet, which is converted to the clock it is asked at.  A cavity with no frequency of its own answers only on the clock it was trained at. |
| `--species` | `H-` | the species the maps are sampled with: `H-`, `proton` or `deuteron`. |
| `--out` | (auto) | output dir; auto = `weights/<hash16>/<elem>/`, `weights/<hash16>_<species>/<elem>/` for another species than H⁻. |
| `--seed` | `42` | LHS + train RNG seed. |
| `--workers` | `1` | CPU worker count for data generation; ≥2 enables `multiprocessing.spawn`. |

!!! tip "Workers for a production-size run"
    `--workers 4` or more cuts the data generation of 5000 samples about
    3× (41 s to 14 s for a MEBT buncher on a 14-core Mac) with
    bit-identical output (the LHS seed and per-sample code path are
    unchanged; only dispatch is parallel).  For a few hundred samples one
    worker is faster: starting the workers costs more than they save.

The CLI's template is the `--species` beam at reference phase 0.  A cavity whose field
follows the running phase is in scope only near its training phase, so for such a
cavity train through the Surrogates tab or the Python API's `ref_template`.  A map
trained for one species does not answer for another: `compare` and `run-envelope` run
the species the weights record (a file that records none counts as H⁻), and need
`--species` when the weights are for several.  A weight that cannot engage for the beam
says so on stderr.

### Worked example

```bash
# Smoke: checks the pipeline; weights this small have failed the accuracy check
python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt/mebt.dat \
    --element FMAP_001 \
    --samples 200 --epochs 40 \
    --workers 12

# A production-size run.  The CLI trains with lr 3e-3 and batch min(64, samples / 8);
# the cycle the training guide measured is the Surrogates tab's (lr 1e-3, batch 256)
python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt/mebt.dat \
    --element FMAP_001 \
    --samples 5000 --epochs 1000 \
    --hidden 128,128 \
    --workers 12
```

## `compare`

Run the envelope twice (baseline vs surrogate-enabled) and emit a
diff report.  Surrogates are loaded from one or more weights
directories.

```bash
python -m linac_gen.surrogates.cli compare \
    --lattice examples/pipii/mebt/mebt.dat \
    --weights linac_gen/surrogates/weights/7429fb95147c412d/FMAP_001 \
              linac_gen/surrogates/weights/7429fb95147c412d/FMAP_002 \
    --current 0.0 \
    --out /tmp/mebt_compare.png
```

### Arguments

| Flag | Default | Notes |
|---|---|---|
| `--lattice` (required) | — | TraceWin `.dat`. |
| `--weights` (required) | — | one or more weights directories. |
| `--out` | none | optional PNG of σ_{x,y,φ,W} overlay. |
| `--species` | the weights' | the beam's species: `H-`, `proton` or `deuteron`; by default the one the weights record (several need it named). |
| `--ref-w-kin` / `--frequency` | 2.12 / 162.5 | ref-particle defaults. |
| `--alpha-{x,y,z}`, `--beta-{x,y,z}`, `--emit-{x_n,y_n,z}` | MEBT entry | initial Twiss. |
| `--current` | 0.0 | beam current in mA.  **0 (the default) engages surrogates** — the pure-linear path serves the NN full-element matrix; **> 0 exercises the RK4 slice walk** (zero NN queries, zero diff expected). |

### Output

```
loaded surrogate 'FMAP_001' (val MAPE 4.05e-03)
loaded surrogate 'FMAP_002' (val MAPE 3.13e-03)

Running baseline vs surrogate-enabled envelope...

Surrogates engaged: 2 (FMAP_001, FMAP_002)
Wall-clock: baseline 0.133 s  surrogate 0.053 s  speedup 2.49x
End-of-line sigma moments:
     sigma_x  baseline= 1.6476e+00  surrogate= 1.6476e+00  rel.diff=1.20e-05  mm
     sigma_y  baseline= 1.5740e+00  surrogate= 1.5740e+00  rel.diff=8.01e-06  mm
   sigma_phi  baseline= 5.8742e+00  surrogate= 5.8752e+00  rel.diff=1.67e-04  deg
     sigma_w  baseline= 1.0614e-02  surrogate= 1.0612e-02  rel.diff=1.99e-04  MeV
Worst rel.diff: 1.99e-04
NN full-element queries: 2
Scope OK: True
Notes: the surrogate run's time leaves out 0.29 s of one-time accuracy checks

plot saved to: /tmp/mebt_compare.png
```

The wall-clock is one run of each, the baseline first, with the caches
cold: much of its speedup is the order.  Run alternately after a
warm-up, the MEBT's envelope at 0 mA takes 59 ms natively and 48 ms with
all 4 bunchers' networks (1.24×).

`NN full-element queries` counts how many times the surrogate run
actually consulted the network.  At `--current 0` it equals the
number of surrogated field-map traversals; at `--current` > 0 it is
0 and the report carries a note — the SC walk requests partial
slices, which always delegate to RK4.

## `run-envelope`

Single envelope pass with one or more surrogates engaged — useful
for benchmarking or sanity checks without baselines.  It takes the beam arguments of
`compare`, `--species` included.

```bash
python -m linac_gen.surrogates.cli run-envelope \
    --lattice examples/pipii/mebt/mebt.dat \
    --use-surrogates linac_gen/surrogates/weights/7429fb95147c412d/FMAP_001 \
                     linac_gen/surrogates/weights/7429fb95147c412d/FMAP_002 \
                     linac_gen/surrogates/weights/7429fb95147c412d/FMAP_003 \
                     linac_gen/surrogates/weights/7429fb95147c412d/FMAP_004 \
    --current 0.0
```

Output:

```
registered surrogate 'FMAP_001' (val MAPE 4.05e-03)
... (3 more)
registered 4 surrogates total.

End-of-line:  sigma_x=1.6476 mm  sigma_y=1.5741 mm  W=2.1200 MeV  sigma_phi=5.875 deg
NN full-element queries: 4
```

At `--current` > 0 the last line reads
`NN full-element queries: 0` followed by a one-line note — the SC
slice walk stays on RK4, so the surrogates are registered but never
queried.  The same beam-config flags as `compare` apply.

## `register-multi`

Register one trained surrogate under multiple element names — e.g.
share one cavity surrogate across the cavities of a cryomodule.  The
weights answer for a cavity only where its settings are theirs: the
same field file, its inputs inside the trained range, and every
setting not swept in training (`kb`, when only `ke` and `phase` were)
equal to what they hold.  A name whose held settings differ is
registered but printed as `WARN … runs native` with the setting; train
with that setting, or `scale`, as an input to share the weights there.
At TRACKING time the
lookup is by element name **guarded by the element in hand**
(`get_for_element`, 2026-07; the envelope, MP and backtrack
consumers all use it, last registration wins) — so each element name
needs an explicit registration even when the physics is identical.
The guard verifies the registered surrogate actually wraps the
element being tracked (object identity, or its structure — class,
length, field file, geometry, aperture, steps, `p_flag` — so re-loading
the same deck, or an error or failure study's copy, keeps the surrogate
engaged, through a view of the weights bound to that element, which
reads its parameters and shares the NN-call count).  How the element is
set is checked when the weights are asked (see
[which element a surrogate answers for](01_overview.md#which-element-a-surrogate-answers-for)).
Which weights stand for the element is decided once per run; how it is set,
and its structure, is checked at every query (a match runs an element whose
held setting is a variable native for the whole run).  A copy of the element
decides for itself.  A same-named element from a
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

The Surrogates tab's **Train all FieldMap\*** does this: it takes each
element's energy window from an envelope run and sweeps `kb` where the map
has one.  `train` sweeps `ke` and `phase` of a `FieldMap3D` only, over the
`--w-kin-range` given, so a loop over a lattice's cavities needs each
one's energy window:

```bash
ELEMS=$(python -c '
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.elements.field_map_3d import FieldMap3D
lat, _ = parse_tracewin("examples/pipii/mebt/mebt.dat")
for e in lat.elements:
    if isinstance(e, FieldMap3D):
        print(e.name)
')

for elem in $ELEMS; do
  python -m linac_gen.surrogates.cli train \
    --lattice examples/pipii/mebt/mebt.dat \
    --element "$elem" \
    --w-kin-range 2.0,2.5 \
    --samples 5000 --epochs 1000 --hidden 128,128 \
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
