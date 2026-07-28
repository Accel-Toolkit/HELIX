# Element catalog overview

HELIX has 17 concrete element classes covering every component
you'd find in a proton/H⁻ linac.  Each element is a Python class in
`linac_gen/elements/` with a corresponding TraceWin `.dat` keyword.
This page is the catalog; click an entry to see the chapter for that
element.

## TL;DR — pick by use case

| You want… | Use this element |
|---|---|
| Field-free propagation | [Drift](01_drift.md) |
| Hard-edge focusing magnet | [Quadrupole](02_quadrupole.md) |
| On-axis B-field magnet | [Solenoid](03_solenoid.md) |
| Bending dipole | [Dipole](04_dipole.md) |
| Dipole fringe field | [Edge](05_edge.md) |
| Thin RF kick (TraceWin GAP) | [RFGap](06_rfgap.md) |
| Cavity from a 1-D / 2-D field map | [FieldMap](07_fieldmap.md) |
| Cavity from a 3-D Cartesian field map | [FieldMap3D](08_fieldmap3d.md) |
| One RFQ cell (Crandall 2-term) | [RfqCell](09_rfqcell.md) |
| Whole RFQ from a `.vane` file | [VaneRFQ](10_vanerfq.md) |
| Sextupole / octupole / higher-pole | [Multipole](11_multipole.md) |
| Aperture mask (loss collimator) | [Aperture](12_aperture.md) |
| Named s-position (snapshot trigger) | [Marker](13_marker.md) |
| Steering corrector | [Steerer](14_steerer.md) |
| Stripper / scattering foil | [Foil](15_foil.md) |
| Ideal thin lens (fx / fy focal lengths) | [ThinLens](16_thinlens.md) |
| Space-charge compensation marker (LEBT) | [SpaceChargeComp](17_spacechargecomp.md) |
| Multi-gap cavity (TraceWin `NCELLS`) | [NCells](18_ncells.md) |
| Overlapping field maps (TraceWin `SUPERPOSE_MAP`) | [SuperposedFieldMap](19_superposedfieldmap.md) |

## Element taxonomy

Internally, every element inherits from one of four base classes:

```
                          Element (abstract)
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
TransferMapElement      ThinKickElement         FieldMapElement       PassiveElement
─────────────────────  ─────────────────────  ─────────────────────  ────────────────
linear 6×6 matrix      thin kick / no length   substepped field       no dynamics
                                               integrator

Drift                   RFGap                    FieldMap              Marker
Quadrupole              Steerer                  FieldMap3D            Aperture
Solenoid                Multipole                RfqCell               SpaceChargeComp
Dipole                  Foil                     VaneRFQ               Edge
                        ThinLens                 NCells
                                                 SuperposedFieldMap
```

* **TransferMapElement** — linear elements with a closed-form 6×6
  matrix.  Drift, hard-edge quad, hard-edge solenoid, sector/rect
  dipole.
* **ThinKickElement** — zero-length kick.  RFGap (TraceWin's GAP
  card), Steerer, thin Multipole, Foil, ThinLens.
* **FieldMapElement** — integrated substep-by-substep through a
  tabulated field (first-order midpoint kick-drift for 1-D/2-D
  maps; KD/DKD for 3-D).  Cavities, real solenoids, RFQ cells,
  whole-RFQ `VaneRFQ` (its five field models are `field_model`
  options on the one class, not separate classes).
* **PassiveElement** — zero-length (or matrix-thin), no tracked
  dynamics of their own.  Markers, aperture cuts,
  space-charge-compensation tags, and the dipole Edge (its linear
  kick is applied via `apply()`).

## Mixin capabilities

Two error-injection mixins exist (added 2026-05), but **not every
element carries them**:

* **Misalignment** — `dx, dy, dz, tilt_deg, pitch_deg, yaw_deg`.
  Carried by Quadrupole, Solenoid, Dipole, RFGap, FieldMap, and
  FieldMap3D — plus Drift, which has **Misalignment only** (no
  FieldError).  Note the tracker honours only `dx`, `dy`, and
  `tilt_deg` as pre/post coordinate shifts and rotations;
  `dz`, `pitch_deg`, `yaw_deg` are stored but **not applied**.
* **FieldError** — per-element relative field offsets:
  `gradient_rel` (quad), `field_rel` (sol/bend), `voltage_rel`
  (cav), `phase_offset` (cav, deg), `frequency_offset` (cav).

Edge, Marker, Aperture, Multipole, Steerer, Foil, and ThinLens have
**neither** mixin.  (Multipole exposes plain `dx`/`dy`/`tilt_deg`
attributes of its own, used for feed-down — not the mixin.)

These mixins are wired by Monte-Carlo error studies — see
[Errors → Element-level errors](../08_errors/03_element_errors.md).

## Multi-track chapter convention

Each element chapter uses the same three-section layout:

* **TL;DR** — TraceWin `.dat` keyword ↔ HELIX class one-liner, plus
  the "gotcha" list (units, sign conventions, what HELIX does
  *differently* from TraceWin).
* **Tutorial** — narrative walkthrough.  When does this element
  apply?  What does the physics look like?  Runnable code +
  embedded plot.
* **API reference** — full constructor signature, parameter table,
  source link.

## TraceWin `.dat` keyword map

For a one-page cross-reference of every TraceWin keyword to its
HELIX class, see
[Appendix B — Keyword cheatsheet](../appendices/B_keyword_cheatsheet.md).

## Cross-references

* [Coordinates & units](../02_concepts/01_coordinates.md) — the
  state vector every element transforms.
* [Data model](../02_concepts/02_data_model.md) — how elements
  live inside `Lattice`.
* [Python API](../06_running/01_python_api.md) — building lattices
  programmatically.

[Continue to Drift →](01_drift.md)
