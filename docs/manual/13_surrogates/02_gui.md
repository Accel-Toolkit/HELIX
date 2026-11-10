# Surrogates in the GUI

The **Surrogates tab** is the 5th tab in the InterphaseWindow strip
(Beam · Lattice · Matching · Numerics · **Surrogates** · Error Study ·
Failure Study · Results).  It handles training, persistence,
registration, and comparison through one panel — for envelope runs
and, via the hybrid section at the bottom, multi-particle runs too.

## Layout

```
┌───────────────────────────────────────────────────────┐
│ Train ML surrogates for field-map elements …          │ ← hint
├───────────────────────────────────────────────────────┤
│ ┌─ Train a new surrogate ──────────────────────────┐  │
│ │ Element: [  36: FMAP_001 [FieldMap3D, 240 mm] ▾]│  │
│ │                              [ Train surrogate ]│  │
│ └───────────────────────────────────────────────────┘  │
│ Training 'FMAP_001' (200 samples, 40 epochs)...        │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                       │
├───────────────────────────────────────────────────────┤
│ Trained surrogates                                     │
│ ┌─────────┬──────────┬───────────────┬─────┬─────────┐ │
│ │ Element │ Val MAPE │ Scope         │ Use │ Action  │ │
│ ├─────────┼──────────┼───────────────┼─────┼─────────┤ │
│ │FMAP_001 │ 4.05e-03 │ w∈[2.02,2.23]…│ [ ] │[Compare]│ │
│ │FMAP_002 │ 3.13e-03 │ w∈[2.02,2.23]…│ [ ] │[Compare]│ │
│ │FMAP_003 │ 3.21e-03 │ w∈[2.02,2.23]…│ [ ] │[Compare]│ │
│ │FMAP_004 │ 3.29e-03 │ w∈[2.02,2.23]…│ [ ] │[Compare]│ │
│ └─────────┴──────────┴───────────────┴─────┴─────────┘ │
│ auto-loaded 4 cached surrogate(s) from ...             │
└───────────────────────────────────────────────────────┘
```

(The table's last column, **Accuracy check**, is left out of the sketch.)

## End-to-end workflow

### 1. Load lattice + beam first

Surrogates run inside the envelope solver (and, when hybrid mode is
engaged, the multi-particle tracker), so both inputs must be in
state before you touch the tab.

* **Lattice tab** → open the `.dat` (e.g. `examples/pipii/mebt/mebt.dat`).
* **Beam tab** → species, w_kin, frequency, Twiss/emittance,
  **current** (0 mA engages the surrogates; > 0 exercises the RK4
  slice walk — see
  [which envelope runs engage the NN](01_overview.md#which-envelope-runs-engage-the-nn)) →
  **Apply**.

### 2. Open the Surrogates tab

When a lattice is loaded, and again when the beam changes species, the tab does two
things automatically:

* **Populates the element dropdown** with every surrogatable
  element in the lattice (`FieldMap` 1-D and `FieldMap3D` 3-D).
  The item label shows index, name, concrete type, and length:

      36: FMAP_001  [FieldMap3D, 240 mm]
     430: FMAP_005  [FieldMap, 300 mm]

* **Auto-discovers cached weights** under
  `linac_gen/surrogates/weights/<lattice-hash-16>/`, or
  `<lattice-hash-16>_proton/` and `_deuteron/` for those beams.  Any subdir
  whose name matches a field-map element name is loaded and added
  to the Trained-surrogates table.  Status line confirms:

      auto-loaded N cached surrogate(s) from linac_gen/.../7429fb95147c412d

    A row that must not be used says why in place of its val MAPE, with **Use**
    greyed out: `stale, retrain` for weights that hold a frequency jump, `trained for
    H-` for another species than the beam's.
    Such a row is never in use, and Compare skips it.

    The **Accuracy check** column (right of Action) gives the verdict on weights
    loaded from a file: *not checked yet* until a run uses them inside their range,
    then *within limits* or *refused*, with the largest block and determinant errors
    against the native element (the reason in the tooltip; see
    [the accuracy check](01_overview.md#the-accuracy-check-of-a-weights-file)).
    Refused weights stay tickable: they go native in every run, and Compare at
    0 mA still runs them to show how far off they are.

### 3. Train a surrogate (or load a cached one)

1. Pick an element in the **Element dropdown**.
2. Click **Train surrogate**.
3. **If a cached weights pair exists** for this `(lattice_hash,
   element)` you get a three-way modal:
    - **Open** (default) — load the cached weights, no retraining.
      Row appears instantly in the table.
    - **Retry** — retrain from scratch (the dialog below opens);
      the new weights overwrite the cached ones.
    - **Cancel** — abort.
4. **If no cache**, the training dialog opens immediately.

#### Training dialog parameters

| Field | Default | Notes |
|---|---|---|
| Samples (LHS) | 200 | smoke; the batch trainer's 5000 for production |
| Epochs | 40 | smoke; the batch trainer's 1000 for production |
| Hidden dims | `64,64` | comma-separated MLP layer sizes |
| Workers (CPU) | `cpu_count() - 2` | data-gen parallelism (12 on M3 Max) |
| w_kin lo / hi | 2.0 / 2.5 MeV | scope on incoming kinetic energy |
| ke ± rel | 0.20 | shown for elements with `ke ≠ 0` |
| kb ± rel | 0.20 | shown for elements with `kb ≠ 0` |
| phase ± | 20.0 deg | shown only when ke ≠ 0 (E-channel active) |

The **per-parameter rows are dynamic**: a 1-D solenoid (ke = 0,
kb ≠ 0) sees only the `kb` row; phase is suppressed because a
pure-magnetic field doesn't have an electric channel that phase
modulates.  A row's spin set to 0 disables that sweep.

5. Click **OK** → training runs in a background `QThread` with a
   live progress dialog (data-gen bar, loss curve, the validation MAPE
   of each epoch and of the epoch kept so far, per-entry MAPE heatmap).
   The epoch kept is the one with the lowest validation error, so its
   line can rise; see the [training guide](05_training_guide.md).  The GUI stays responsive throughout.
   **Closing the progress dialog cancels the run**: the trainer stops
   at the next sample / epoch and nothing is written to disk (weights
   are only saved after both stages complete).  The same applies to a
   batch run — closing its dialog stops the current element and skips
   the rest.
6. On completion the new surrogate is written to
   `linac_gen/surrogates/weights/<hash16>/<element>/`, or under
   `<hash16>_proton/` or `<hash16>_deuteron/` for the species trained
   (`weights.pt` + `metadata.json`), and a row appears in the table.

#### Batch training — "Train all FieldMap*"

The **Train all FieldMap\*** button next to Train surrogate trains
the whole lattice in one go.  A selector dialog lists every
FieldMap-class element (all ticked by default, with **Select all** /
**Select none** buttons) above a shared settings block — samples,
epochs, hidden dims, CPU workers (5000 samples, 1000 epochs and
`128,128` by default, the production cycle of the
[training guide](05_training_guide.md)) — that applies to every checked
element.  Elements train sequentially (the inner data-gen is already
multi-process, so parallelising across elements would oversubscribe);
per-element w_kin and ADJUST sweep ranges are still auto-detected.
Closing the batch progress dialog stops the current element and
skips the rest.

### 4. Engage surrogates

Tick the **Use** checkbox on every row you want active.  This calls
`linac_gen.surrogates.registry.register(surr)` — the envelope hook
in `tracking/envelope.py` looks up surrogates by element name on
the full-element matrix request of a 0 mA envelope run and routes
it through the MLP when admissible.  SC-active runs (current > 0)
and per-sub-step recording request partial slices, which always
stay on the wrapped RK4.

You can mix-and-match: use FMAP_001 + FMAP_003, leave FMAP_002 +
FMAP_004 as pure RK4.  Untick to revert.  The **Select all** /
**Deselect all** buttons above the table bulk-tick or untick every
row's Use checkbox (each toggle registers / unregisters that
surrogate exactly as a manual click would).

!!! tip "Use ticks don't survive a GUI restart"
    The cached weights and the Trained-surrogates rows DO persist,
    but the **Use** ticks default to unchecked on every launch —
    re-tick the rows you want active after reopening the GUI.
    Within a session the ticks mirror the runtime registry across
    table rebuilds, so a still-registered surrogate always shows as
    ticked; and retraining an engaged element re-registers the fresh
    weights automatically, so the tick keeps pointing at the new
    model.

### 5. Run a normal envelope sim with surrogates engaged

After ticking **Use** on the rows you want, switch to **Numerics**
or **Results** and run as usual.  In a **0 mA** envelope run without
*Record per-sub-step*, the surrogate replaces RK4 for every
full-element `fitted_matrix(ref)` request where it's both registered
and admissible (input within training scope).  At current > 0, or
with per-sub-step recording on, the run slice-walks RK4 and the
network is never queried — results are identical to the
unregistered run.

### 6. Compare baseline vs surrogate

Click **Compare** on a surrogate's table row to run:

1. clear registry → baseline envelope (pure RK4),
2. re-register that surrogate → surrogate-enabled envelope,
3. diff the σ_{x,y,φ,W} curves end-to-end, report wall-clock.

A summary dialog pops up.  You're prompted to save the σ-curves
PNG.

The compare runs at the **configured beam current**.  Its timing
leaves out the one-time accuracy check of weights it meets first, and the
report says how long that took.  Measured output at 0 mA (MEBT, one
surrogate engaged, production-cycle weights):

    Surrogates engaged: 1 (FMAP_001)
    Wall-clock: baseline 0.130 s  surrogate 0.056 s  speedup 2.32x
    End-of-line sigma moments:
         sigma_x  baseline= 1.6476e+00  surrogate= 1.6476e+00  rel.diff=1.67e-06  mm
         ...
    NN full-element queries: 1

With all 4 bunchers engaged at 0 mA the end-of-line σ rel.diffs are
1.0e-5 … 1.7e-4.  The wall-clock is one run of each, the baseline first:
on a lattice that has not run yet, the baseline also pays for the first
run (127 ms against 50 ms with the 4 bunchers engaged, 2.5×); once the
lattice has run, a compare shows about 1.2× (58–60 ms against 47–49 ms).  At a **non-zero current** the compare returns zero diff
**by design** — the SC walk requests partial slices, which always
delegate to RK4 — and reports `NN full-element queries: 0` with a
note saying the surrogates were registered but never queried.

### 7. (Optional) Engage surrogates in multi-particle runs

The collapsible **Multi-particle surrogates (hybrid mode)** section
beneath the table extends the *same* trained surrogates to the MP
tracker.  Controls:

* **Engage in MP runs** — the master toggle; it drives
  `linac_gen.surrogates.registry.set_mp_enabled()`.  Off by default.
  On alone, each surrogate call delegates to native RK4 —
  bit-identical, no speedup.
* **Experimental: linear-matrix fast path** — second opt-in; with the
  master also on, the per-substep RK4 push is replaced by an analytic
  reference advance plus a cached matrix-slice apply (linear-matrix
  accuracy; halo particles drift).  Run Compare MP after toggling.
* **Residual RK4 substeps (reserved)** — stored on every trained
  surrogate but read by no tracking path yet: the planned hybrid
  linear-anchor + RK4-residual mode is not implemented.
* **Compare MP (baseline vs hybrid)** — runs two full MP
  simulations (pure RK4 vs surrogate-engaged) in the background and
  reports speedup and σ differences.  Run it after every toggle
  change.

## Behaviour on lattice edits

An edit of the lattice in hand — an inspector change, an undo — keeps the
Trained-surrogates table, its **Use** ticks and the runtime registry: the
weights check how their element is set each time they are asked, so a setting
they hold that the edit changed sends that element native until it is set back
(see
[which element a surrogate answers for](01_overview.md#which-element-a-surrogate-answers-for)).
A row whose element the edit removed, replaced or renamed, or whose structure
it changed (length, steps, `p_flag`, …), is dropped and unregistered, and the
status line names it: that element runs native until it is trained again,
unless it was renamed to the name of another row whose element has its
structure, in which case those weights answer for it.  To load the dropped
weights again, undo the edit and choose **Train surrogate** → **Open**, or
re-open the project (which drops unsaved edits); both work only while the deck
file holds the text it had when the weights were trained, since the weights
folder the tab reads is named by a hash of that text.  **File > Save Lattice**
(Ctrl+S) writes the lattice in hand into the deck file in HELIX's own format,
which changes the text after an edit of what its cards hold, or the first time
a deck written by hand or by another program is saved.  After a Matching
Apply (one that varied only beam fields included) or another Apply that writes
fitted values into the lattice (orbit correction, ORM, reliability, a
multi-objective knee point), and for a lattice with no file or one read from
another program's lattice format, it asks for a file name instead (suggesting
`<name>.matched.dat` after an Apply), and the file saved becomes the deck the
tab reads.  **Save Project** leaves the deck file alone.  The dropdown follows
the edit.  A Matching
**Apply** is such an edit: it writes the matched values into the lattice in
hand as one undoable step, so the table, its ticks and the registry stay.

## Behaviour on lattice swap

Loading a different `.dat` (or another beam species, whose weights live in
another folder):

* clears the Trained-surrogates table (its rows held element
  bindings from the *previous* lattice — stale now),
* **clears the runtime registry**,
* re-runs auto-discovery against the new lattice's hash dir,
* fills the dropdown with the new lattice's field maps.

The registry clear is deliberate: the envelope hook looks surrogates
up by *element name* only, so a surrogate registered under project
A's `FMAP_001` would silently be used for project B's `FMAP_001` if
it survived the swap — producing wrong envelope / MP results.  After
a swap, simply re-tick **Use** on the new lattice's rows once its
surrogates are trained or auto-loaded.

## Cached-weights location

Layout per lattice:

    linac_gen/surrogates/weights/
    └── 7429fb95147c412d/          ← SHA256 prefix of the .dat content,
                                     with _proton or _deuteron for those beams
        ├── FMAP_001/
        │   ├── weights.pt
        │   └── metadata.json       ← seed, commit SHA, scope, val MAPE
        ├── FMAP_002/
        ├── FMAP_003/
        └── FMAP_004/

Changing the lattice (even a single `ke` value in the .dat) yields
a different SHA → fresh subdirectory; old weights stay put.  Useful
for reproducibility but can hoard disk over time — delete stale
subdirs manually if needed.

## Cross-references

* [Overview](01_overview.md) — what surrogates are, when to use.
* [CLI](03_cli.md) — train / compare / run-envelope without a GUI.
* [Python API](04_python_api.md) — programmatic equivalents.
* [Training guide](05_training_guide.md) — choosing sample counts,
  parallelism, OOD handling.

← [Overview](01_overview.md) ·
[Continue to CLI →](03_cli.md)
