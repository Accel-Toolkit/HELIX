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
| X0/Y0/θ operands | **ignored** without `SUPERPOSE_MAP_OUT`; place the map with it | same |
| `SUPERPOSE_MAP_OUT` | curved reference trajectory | same — a [curved cluster](#curved-clusters-superpose_map_out) |

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

## Curved clusters — `SUPERPOSE_MAP_OUT` {#curved-clusters-superpose_map_out}

A `SUPERPOSE_MAP_OUT Z X Y θZ θX θY` card in front of a cluster declares
that its maps **bend** the beam — a dipole given as a 3-D magnetic field
map, alone or with other maps around it.  The card gives the exit point
(mm) and the exit direction of the reference (deg) in the cluster's
entrance frame; with it, all six operands of every child's
`SUPERPOSE_MAP Zi Xi Yi θZi θXi θYi` place that map (its origin and its
axes).  HELIX builds a `CurvedFieldMapCluster` and does what TraceWin's
manual describes:

1. **The reference trajectory.**  The reference particle is traced
   through the summed fields of the children (each sampled in its own
   frame, its field turned back into the cluster frame) from the entrance
   origin along +z, until it crosses the exit plane — the plane through
   the card's point, normal to the card's exit direction.  RK4 in the
   path length (0.1 mm steps), landing on the plane to 1e-11 mm.  Any
   bending angle works, also beyond 90° (the map's z then runs backward),
   and the path need not stay in one plane.
2. **The beam around it.**  `s` runs along that path: the element's
   **length is the reference's path length**, and the particles are
   described in planes normal to the reference (x, y along a frame
   transported without rotation about the tangent).  They are pushed in
   `s` with the exact equations of motion in a static magnetic field —
   no small-angle approximation — so x' = p_x/p_s and y' = p_y/p_s stay
   exact whatever the bend.  RK4 with steps of at most ¼ of the finest
   child grid step along z (the 5000 steps/m refinement of straight
   solenoid maps does not apply).
3. **The exit.**  The particles, on the plane normal to the reference at
   its crossing point, are moved into the card's frame, each drifted
   straight onto the card's plane.  A card that does not match the traced
   reference therefore leaves the beam **offset** in position and angle —
   TraceWin keeps that offset too, and so does HELIX (multiparticle
   particles, the envelope centroid).

**The length depends on the entering energy.**  Until the reference is
traced, the length is an estimate from the card (the path through the
point where the entrance and exit axes meet; `length_provisional` is
true).  Every run (multiparticle, envelope, matrix, the backtracking
replay) traces it at the element, for the energy that actually reaches
it; a trace is reused while species, energy, field maps and factors,
placements and misalignment are unchanged — also by copies of the
lattice.  The GUI traces it as soon as a lattice and a beam are known
(opening a project or lattice, applying the Beam tab), when nothing
before the cluster can change the energy, and redraws the lattice views
when a length changes; from Python,
`curved_field_map.prepare_curved_elements(lattice, ref)` does it.  A
reference that cannot reach the exit plane — it leaves the maps away
from the plane, or turns twice — is refused with a message.

**Sign conventions.**  A child's angles are an active rotation of its
axes, about z, then y, then x: θY > 0 turns the map's z axis toward +X,
θX > 0 toward +Y, θZ is right-handed about z.  The exit card's angles
carry the **opposite** sign: θY > 0 turns the exit direction toward −X.
Both are pinned by the TraceWin manual's own example (its dipole +
quadrupole figure, where `SUPERPOSE_MAP_OUT 1573.29 164.27 0 0 0
-9.59023` and `SUPERPOSE_MAP 1100 84.27 0 0 0 9.59023` both turn toward
+X) and by a TraceWin project with a 30° switching dipole, whose traced
path and exit offset HELIX reproduces.  No example pins θX and θZ: the
parser warns when either is used.

**Run modes.**

* *Multiparticle* — with space charge between steps (the beam's
  coordinates are relative to the reference, so the solvers and grids
  apply unchanged), the `.scc` table and the `.ouv` aperture of the
  position-0 child along `s`, and the aperture measured from the
  reference.  The push has a compiled twin (`curved_rk4_steps`), equal to
  the numpy code bit for bit at any thread count;
  `LINAC_GEN_CURVED_KERNEL=0` selects the numpy code.
* *Envelope and matrix* — transfer matrices fitted on the same push
  (whole element and slices), the move into the card frame included; the
  exit offset enters the envelope centroid.  Matrix mode is linear: the
  offset is not part of a matrix.
* *Not available:* backtracking (TraceWin offers none for these maps
  either), the Sacherer envelope solver (use the matrix solver), the
  torch matrix path, ring mode, and MAD-X export (it warns that the line
  continues straight, and `--strict` refuses).  Each says so when asked.

**Children** are static magnetic 3-D Cartesian maps (`FIELD_MAP 70…`);
a cluster with an electric or RF map is refused at construction.  The
maps are sampled trilinearly (the cubic 3-D interpolation setting does not
apply; HELIX says so).  **The move into the card frame** is a straight
drift onto the card plane: the field between the reference's exit plane
and the card plane is not felt — an error of about x·α/ρ for a card that
misses the traced reference by an angle α, negligible for a card set from
the map.  **The phase** keeps each particle's exact path: a particle at an
angle to the reference falls behind it by (1/cos − 1) of the path even
without a field, a term HELIX's straight elements, linear in the angles,
leave out.  **Interior diagnostics** (`SHIFT_IN_FIELD_MAP`) in a curved
cluster are placed along the path; one beyond its end is reported once
the path is traced and not recorded.  The H⁻ magnetic-stripping analysis
takes the largest field normal to the traced reference.
**Misalignment** moves the maps in the entrance frame: the reference is
traced through the moved fields and its mismatch with the card becomes
the beam offset.  The **floor plan** follows the traced reference and
continues from the card's exit frame, drawn like a `BEND` (a bend whose
beam leaves toward −x, dispersion R16 > 0, is drawn toward +x as a
positive-angle bend is); the Lattice and Results tabs show the element
as a bend.

!!! note "Validation"
    `tests/elements/test_curved_field_map.py`: a uniform field over the
    whole arc bends the reference on a circle — for 30°, 90°, 135°, 180°
    and −45° the traced path equals ρθ to 1e-12 and the 6×6 matrix is the
    sector `Dipole`'s (x plane, dispersion, path-length row) to 1e-4,
    with a drift of ρθ in y; a card moved by δ gives the closed-form
    offset of a circle seen from the moved plane; the compiled push
    equals the numpy code bit for bit at 1, 2 and 4 threads; the
    reference particle stays exactly on zero; multiparticle tracking
    equals M·x + offset; slices compose to the whole map; the cards
    round-trip through the deck writer.  On a measured 30° dipole map,
    brute-force Cartesian ray tracing of 3000 particles (positions and
    momenta in the entrance frame, no moving frame) agrees with the
    curved push to 0.1 µm and 0.03 mrad.

## Modelling notes

The container drives the same kick-drift walk as
[FieldMap](07_fieldmap.md), with an accumulated physical z-cursor (the
[NCells](18_ncells.md) idiom) recorded per step so exact backtracking
works (`untrack_rk4` closes to the 1e-12 class).  All four tracking
modes are supported for straight clusters — multi-particle (incl. Strang space-charge
bundles), envelope (incl. the SC bundle loop), matrix, and backtrack.
`n_steps` defaults to the finest child native grid resolved over the
whole span; the magnetic-only 5000 steps/m auto-refine applies iff
*every* child is magnetic-only (1-D or 3-D, no electric channel) — in
multi-particle stepping and in the envelope / matrix fitted maps alike,
so a solenoid with unpowered 3-D correctors superposed on it matches
the solenoid alone.  3-D children are integrated with the
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

```{.python .skip}
linac_gen.elements.curved_field_map.CurvedFieldMapCluster(
    name: str,
    children: list,          # [(z0_mm, FieldMap3D), …] in card order
    placements: list | None = None,  # per child (Z, X, Y, θZ, θX, θY), mm/deg
    exit_frame=(0, 0, 0, 0, 0, 0),   # the SUPERPOSE_MAP_OUT operands
    **kw,                            # as SuperposedFieldMap
)
linac_gen.elements.curved_field_map.prepare_curved_elements(lattice, ref) -> int
```

### Source

* `linac_gen/elements/superposed_field_map.py` (container, walk, calibration)
* `linac_gen/elements/curved_field_map.py` (curved clusters: reference
  trajectory, push, exit, matrices)
* `linac_gen/csrc/fieldmap_kernels.cpp` (`curved_rk4_steps`)
* `linac_gen/io/tracewin_parser.py` (cluster state machine, `SHIFT_IN_FIELD_MAP`)
* `linac_gen/io/tracewin_writer.py` (round-trip emission)

## Limitations (v1)

* Curved clusters (`SUPERPOSE_MAP_OUT`) take static magnetic 3-D maps
  only, run in multiparticle, envelope (matrix solver) and matrix modes,
  and take θX / θZ placements by the manual's figure alone (see
  [Curved clusters](#curved-clusters-superpose_map_out)).
* Mixed RF frequencies in one cluster are refused (single `ref.phi_s`
  clock).
* Misalignment and cavity errors are cluster-wide, not per child.

## See also

* [FieldMap](07_fieldmap.md) — the 1-D child element and slice physics.
* [FieldMap3D](08_fieldmap3d.md) — the 3-D child element.
* [Known limitations](../12_validation/03_known_limitations.md).

← [NCells](18_ncells.md) ·
[Continue to Multipole →](11_multipole.md)
