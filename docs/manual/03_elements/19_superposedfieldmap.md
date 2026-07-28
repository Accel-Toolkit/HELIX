# SuperposedFieldMap

**Overlapping field maps** — TraceWin's `SUPERPOSE_MAP` clusters.  A
*cluster* is a run of consecutive `SUPERPOSE_MAP Z0` + `FIELD_MAP` card
pairs ended by the next ordinary element; TraceWin places each map's
origin `Z0` mm from the cluster entrance and integrates **once through
the vector sum** of every map covering each z.  Classic uses: a
quadrupole map inside a solenoid map, steerer maps stacked on a
solenoid (the PIP-II HWR packages), or a buncher cavity partially
overlapping a quad doublet (the PIP-II MEBT).

## TL;DR

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `SUPERPOSE_MAP Z0 [X0 Y0 θ…]` before each `FIELD_MAP` | `SuperposedFieldMap(name, [(z0, map), …])` |
| Cluster span | `max(Z0ᵢ + Lᵢ)` — s advances once | same (one lattice element) |
| Field at z | vector sum of all maps covering z | same — per-slice sum over children |
| `Z0 < 0` | start `−Z0` mm *inside* that map | same |
| X0/Y0/θ operands | **ignored** without `SUPERPOSE_MAP_OUT` | same (ignored, with a parse note) |
| `SUPERPOSE_MAP_OUT` | curved reference trajectory | **refused** (see limitations) |

Conventions:

* **Children are pure samplers.** Each `(z0_mm, FieldMap | FieldMap3D)`
  pair keeps its own file data, scale factors and phase, but its
  integrator is never invoked: the container samples every child's
  E/B at the shared slice midpoint (`z_file = z_local − z0`), sums
  them, and applies **one** energy kick, adiabatic damping, phase
  slip, Lorentz kick and drift per slice.  Single application of the
  shared-slice physics is what makes overlap physically correct — two
  half-strength solenoid maps are *exactly* one full-strength one.
* **One RF clock.** All RF children must share one frequency; a
  cluster mixing RF frequencies is refused at construction (in
  permissive parses the deck falls back to the legacy end-to-end
  layout with a warning).  Static (magnetic) children mix freely with
  RF ones.
* **Position-0 carrier** (TraceWin convention): the child placed at
  `z0 = 0` carries the cluster's aperture, `.ouv` aperture profile and
  `.scc` space-charge-compensation map.  A cluster with no `z0 = 0`
  child gets aperture 0 and a parser warning — add an empty map at
  position 0, as the TraceWin manual recommends.
* **`SET_SYNC_PHASE` is container-aware**: the fixed-point calibration
  walks the *whole* cluster span (β evolves under the summed fields)
  while the voltage integral counts only the target cavity's own
  E-channels, referenced to that cavity's window entry.  For a lone
  `z0 = 0` RF child this degenerates exactly to the standalone
  calibration.
* **Misalignment / `ERROR_CAV` apply cluster-wide** (rigid container
  wrap).  TraceWin applies them per card — a documented divergence;
  the parser warns when alignment errors land on a container.
  `ADJUST` cards targeting a container are warn-skipped.
* A single-child cluster at `z0 = 0` (the common "SUPERPOSE_MAP 0"
  furniture idiom) parses to a **plain `FieldMap`** with provenance,
  so round-trips, surrogates and every existing dispatch see the
  ordinary element.

## Interior diagnostics — `SHIFT_IN_FIELD_MAP`

`SHIFT_IN_FIELD_MAP dz` in front of diagnostic elements places them
`dz` mm *inside* the span of the following field map or cluster
(`dz > 0`; several diagnostics can be stacked).  HELIX consumes the
pair into `interior_markers`: during tracking, an extra recorder row is
emitted when the walk crosses `z = dz`, attributed to the diagnostic's
name at `s = s_entry + dz` — in multi-particle *and* envelope modes,
with or without space charge.  The markers are not lattice elements,
so element positions and `element_exit_idx` are untouched.
`DIAG_POSITION` cards carrying orbit-correction targets are **not**
allowed inside a map (correction drivers resolve BPMs as lattice
elements) — the parser refuses the combination.

## Modelling notes

The container drives the same kick-drift walk as
[FieldMap](07_fieldmap.md), with an accumulated physical z-cursor (the
[NCells](18_ncells.md) idiom) recorded per step so exact backtracking
works (`untrack_rk4` closes to the 1e-12 class).  All four tracking
modes are supported — multi-particle (incl. Strang space-charge
bundles), envelope (incl. the SC bundle loop), matrix, and backtrack.
`n_steps` defaults to the finest child native grid resolved over the
whole span; the magnetic-only 5000 steps/m auto-refine applies iff
*every* child is magnetic-only.  3-D children are integrated with the
container's kick-drift walk (the global DKD selection for standalone
`FieldMap3D` does not apply inside a cluster).

!!! note "Validation anchors"
    The shipped test battery pins: 1-child cluster ≡ plain map to
    1e-12 across all modes; a map *split* into two overlapping files
    with complementary ramps ≡ the unsplit map; two half-strength
    solenoids ≡ one full solenoid; cluster sync-phase ≡ standalone
    sync-phase; envelope-vs-MP parity with space charge; and
    byte-idempotent `.dat` round-trips.  See
    `tests/elements/test_superposed_field_map.py`,
    `tests/io/test_superpose_parsing.py`,
    `tests/tracking/test_superpose_tracking.py`.

!!! success "Validated against real PIP-II TraceWin decks (2026-07)"
    The FDR SC-linac deck (111 `superpose_map` cards: hwrcx + hwrcy
    steerer maps stacked on every HWR/SSR solenoid) and the PIP2IT
    MEBT deck (quad doublets/triplets and a QWR buncher partially
    overlapping a doublet at Z0 = 206.4 mm) import warning-free —
    37 + 9 containers.  Against TraceWin's own exports: every element
    boundary matches over the full 162.9 m FDR line (<0.5 mm); the
    first FDR solenoid+corrector cluster reproduces the TW envelope
    σx/σy/σφ to ≤1.5 %; the MEBT doublet cluster tracks TW to 0.3 %
    and its *cumulative transfer matrix* matches TW's
    `Transfer_matrix1.dat` element-for-element at the few-per-mil
    level.  Residual deviations further downstream trace to the
    standalone RF-cavity model at far-below-design β (present for
    plain maps too, independent of superposition).  The gated anchors
    live in `tests/analysis/test_superpose_realdeck.py` (skipped when
    the never-committed PIP-II decks are absent).

## API reference

```{.python .skip}
linac_gen.elements.superposed_field_map.SuperposedFieldMap(
    name: str,
    children: list,          # [(z0_mm, FieldMap | FieldMap3D), …] in card order
    *,
    aperture: float | None = None,   # default: the z0 == 0 child's
    n_steps: int | None = None,      # default: finest child grid over the span
    interior_markers: list | None = None,  # [(dz_mm, Marker), …] from SHIFT_IN_FIELD_MAP
    dx=0.0, dy=0.0, dz=0.0,          # cluster-wide misalignment
    tilt_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
    voltage_rel=0.0, phase_offset=0.0, frequency_offset=0.0,  # ERROR_CAV slots
)
```

### Source

* `linac_gen/elements/superposed_field_map.py` (container, walk, calibration)
* `linac_gen/io/tracewin_parser.py` (cluster state machine, `SHIFT_IN_FIELD_MAP`)
* `linac_gen/io/tracewin_writer.py` (round-trip emission)

## Limitations (v1)

* `SUPERPOSE_MAP_OUT` (curved reference trajectory through dipole
  maps) is refused at parse — the exit-frame bookkeeping is separate
  work; without it TraceWin itself ignores the X/Y/θ operands, so
  ignoring them here is faithful.
* Mixed RF frequencies in one cluster are refused (single `ref.phi_s`
  clock).
* Misalignment and cavity errors are cluster-wide, not per child.

## See also

* [FieldMap](07_fieldmap.md) — the 1-D child element and slice physics.
* [FieldMap3D](08_fieldmap3d.md) — the 3-D child element.
* [Known limitations](../12_validation/03_known_limitations.md).

← [NCells](18_ncells.md) ·
[Continue to Multipole →](11_multipole.md)
