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
│ │FMAP_001 │ 4.28e-02 │ w∈[2,2.5]  …  │ [ ] │[Compare]│ │
│ │FMAP_002 │ 5.56e-02 │ w∈[2,2.5]  …  │ [ ] │[Compare]│ │
│ │FMAP_003 │ 3.32e-02 │ w∈[2,2.5]  …  │ [ ] │[Compare]│ │
│ │FMAP_004 │ 5.57e-02 │ w∈[2,2.5]  …  │ [ ] │[Compare]│ │
│ └─────────┴──────────┴───────────────┴─────┴─────────┘ │
│ auto-loaded 4 cached surrogate(s) from ...             │
└───────────────────────────────────────────────────────┘
```

## End-to-end workflow

### 1. Load lattice + beam first

Surrogates run inside the envelope solver (and, when hybrid mode is
engaged, the multi-particle tracker), so both inputs must be in
state before you touch the tab.

* **Lattice tab** → open the `.dat` (e.g. `examples/pipii/mebt/mebt.dat`).
* **Beam tab** → species, w_kin, frequency, Twiss/emittance,
  **current** (≥ 1 mA so the SC-engaged path is exercised — see
  [the M3 limitation](01_overview.md#the-m3-honest-limitation)) →
  **Apply**.

### 2. Open the Surrogates tab

On entry the tab does two things automatically:

* **Populates the element dropdown** with every surrogatable
  element in the lattice (`FieldMap` 1-D and `FieldMap3D` 3-D).
  The item label shows index, name, concrete type, and length:

      36: FMAP_001  [FieldMap3D, 240 mm]
     430: FMAP_005  [FieldMap, 300 mm]

* **Auto-discovers cached weights** under
  `linac_gen/surrogates/weights/<lattice-hash-16>/`.  Any subdir
  whose name matches a field-map element name is loaded and added
  to the Trained-surrogates table.  Status line confirms:

      auto-loaded N cached surrogate(s) from linac_gen/.../bb088243c9fb33ce

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
| Samples (LHS) | 200 | smoke; bump to 50 000+ for production |
| Epochs | 40 | smoke; 200-500 for production |
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
   live progress dialog (data-gen bar, loss curves, per-entry MAPE
   heatmap).  The GUI stays responsive throughout.
   **Closing the progress dialog cancels the run**: the trainer stops
   at the next sample / epoch and nothing is written to disk (weights
   are only saved after both stages complete).  The same applies to a
   batch run — closing its dialog stops the current element and skips
   the rest.
6. On completion the new surrogate is written to
   `linac_gen/surrogates/weights/<hash16>/<element>/`
   (`weights.pt` + `metadata.json`) and a row appears in the table.

#### Batch training — "Train all FieldMap*"

The **Train all FieldMap\*** button next to Train surrogate trains
the whole lattice in one go.  A selector dialog lists every
FieldMap-class element (all ticked by default, with **Select all** /
**Select none** buttons) above a shared settings block — samples,
epochs, hidden dims, CPU workers — that applies to every checked
element.  Elements train sequentially (the inner data-gen is already
multi-process, so parallelising across elements would oversubscribe);
per-element w_kin and ADJUST sweep ranges are still auto-detected.
Closing the batch progress dialog stops the current element and
skips the rest.

### 4. Engage surrogates

Tick the **Use** checkbox on every row you want active.  This calls
`linac_gen.surrogates.registry.register(surr)` — the envelope hook
in `tracking/envelope.py` looks up surrogates by element name on
each `fitted_matrix(_slice)` call and routes through the MLP when
admissible.

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
or **Results** and run as usual.  The surrogate replaces RK4 for
every `fitted_matrix(ref)` call where it's both registered and
admissible (input within training scope).

### 6. Compare baseline vs surrogate

Click **Compare** on a surrogate's table row to run:

1. clear registry → baseline envelope (pure RK4),
2. re-register that surrogate → surrogate-enabled envelope,
3. diff the σ_{x,y,φ,W} curves end-to-end, report wall-clock.

A summary dialog pops up.  You're prompted to save the σ-curves
PNG.

Typical output (MEBT, 5 mA, one surrogate engaged):

    Surrogates engaged: 1 (FMAP_001)
    Wall-clock: baseline 6.51 s  surrogate 5.93 s  speedup 1.10x
    End-of-line sigma moments:
         sigma_x  baseline= 1.7951e+00  surrogate= 1.7912e+00  rel.diff=2.2e-03  mm
         ...

For an all-surrogate run (4 bunchers engaged) on the MEBT at 5 mA
the speedup rises to ~1.86× and σ rel.diffs stay under 6 % at the
smoke-cycle accuracy.

### 7. (Optional) Engage surrogates in multi-particle runs

The collapsible **Multi-particle surrogates (hybrid mode)** section
beneath the table extends the *same* trained surrogates to the MP
tracker: each surrogate call applies a fast linear-matrix anchor
plus a cheap RK4 residual for the nonlinear physics.  Controls:

* **Engage in MP runs** — the master toggle; it drives
  `linac_gen.surrogates.registry.set_mp_enabled()`.  Off by
  default, so the envelope-only workflow is unchanged until you opt
  in.
* **Residual RK4 substeps** — substeps per surrogate call
  (default 15, accuracy-first).  Lower is faster but less accurate;
  0 means pure linear matrix.
* **Compare MP (baseline vs hybrid)** — runs two full MP
  simulations (pure RK4 vs hybrid) in the background and reports
  speedup and σ differences.  Run it after every toggle change.

## Behaviour on lattice swap

Loading a different `.dat`:

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
    └── bb088243c9fb33ce/          ← SHA256 prefix of the .dat content
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
