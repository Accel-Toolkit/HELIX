# `ERROR_*` directives

HELIX implements TraceWin's error-directive language so a `.dat`
file authored for TraceWin tolerance studies works in HELIX
unmodified.  This page is the directive-by-directive reference.

## Quick map

| TraceWin directive | What HELIX does |
|---|---|
| `ERROR_GAUSSIAN_CUT_OFF σ_max` | sets the gaussian truncation for the whole file (last card wins) |
| `ERROR_SET_RATIO r1 r2 r3 …` | stores ratios on `lattice.error_ratios` (not consumed) |
| `ERROR_QUAD_NCPL_STAT N r dx dy φx φy φz dG dG3 dG4 dG5 dG6 [Nb]` | quad alignment + field errors on the next N quads |
| `ERROR_CAV_NCPL_STAT N r dx dy φx φy E φ dz [Nb]` | cavity alignment + voltage + phase on the next N cavities |
| `ERROR_BEND_NCPL_STAT N r dx dy φx φy φz dg dz` | dipole alignment + field on the next N bends |
| `ERROR_BEAM_STAT r dx dy dφ dxp dyp de dEx dEy dEz mx my mz dIb` | beam-input jitter (one draw per seed) |

The `ERROR_*` cards come from TraceWin's "Tolerance language"
section; HELIX's parser absorbs them into
`lattice.errors` (element-level) and `lattice.beam_errors`
(beam-level) lists, ready for the `ErrorStudy` engine.

## Distribution code `r`

The `r` slot in the directive selects the distribution:

| `r` | Distribution | HELIX behaviour |
|---|---|---|
| 0 | constant (TraceWin) | approximated as a truncated gaussian (**no constant mode**) — emits a warning at parse |
| 1 | uniform | uniform half-width |
| 2 | gaussian | gaussian σ (with global `ERROR_GAUSSIAN_CUT_OFF`) |
| 4 | binary ±/∓ | gaussian (coarse approximation) |
| 5 | binary 0/+ | gaussian (coarse approximation) |

Note that `r=0` is *not* implemented as a constant offset: HELIX maps
it to an ordinary gaussian draw (σ = the value, cutoff 3 unless a
`ERROR_GAUSSIAN_CUT_OFF` card overrides it).

## `ERROR_QUAD_NCPL_STAT`

```
ERROR_QUAD_NCPL_STAT  N  r  dx  dy  φx  φy  φz  dG  dG3  dG4  dG5  dG6  [Nb]
```

Applies random errors to the **next N quadrupoles** in the lattice
stream:

* `dx`, `dy` — alignment offset (mm)
* `φx`, `φy`, `φz` — rotation (deg) — only `φz` (= `tilt_deg`) is
  honoured by the tracker; pitch/yaw reserved.
* `dG` — gradient relative error (%) — converted to fraction
* `dG3..dG6` — higher-multipole relative errors (%)
* `Nb` (optional, TraceWin v2.16+) — random-subset count
  (parsed-and-ignored)

### Example

```
ERROR_GAUSSIAN_CUT_OFF 3
ERROR_QUAD_NCPL_STAT 6 2 0.2 0.2 0 0 0 0.5 0 0 0 0
```

Reads as: "On the next 6 quads, gaussian σ_dx = σ_dy = 0.2 mm,
σ_dG = 0.5 % (= 0.005 fractional), truncated at 3σ."

## `ERROR_CAV_NCPL_STAT`

```
ERROR_CAV_NCPL_STAT  N  r  dx  dy  φx  φy  E  φ  dz  [Nb]
```

* `E` — voltage relative error (%) → `voltage_rel`
* `φ` — phase offset (deg) → `phase_offset`
* `dz` — longitudinal alignment (mm) → `dz` (stored but ignored by
  the tracker)

!!! note "Amplitude errors on FieldMap cavities (fixed 2026-07)"
    The `E` slot now takes effect on **both** cavity models: on
    `RFGap` via the `voltage` attribute, and on `FieldMap`/`FieldMap3D`
    via the `voltage_rel` slot that `effective_ke/kb` consume.  Earlier
    releases silently dropped it on field maps — pre-fix SRF tolerance
    studies effectively ran with zero amplitude jitter.  Phase offsets
    (`φ`) work on both, as before.

## `ERROR_BEND_NCPL_STAT`

```
ERROR_BEND_NCPL_STAT  N  r  dx  dy  φx  φy  φz  dg  dz
```

* `dg` — bending-field relative error (%) → `field_rel`

!!! note "`dg` (fixed 2026-07)"
    The `field_rel` draw now lands on the Dipole's own `field_rel`
    slot, consumed via `effective_angle` — earlier releases silently
    dropped it.  Alignment errors (`dx`, `dy`, `φz`) apply as before;
    `dz` is still stored-but-inert (a parse warning now says so).

## `ERROR_BEAM_STAT`

```
ERROR_BEAM_STAT  r  dx  dy  dφ  dxp  dyp  de  dEx  dEy  dEz  mx  my  mz  dIb  ...
```

Position-by-position:

| Slot | Symbol | Meaning |
|---|---|---|
| 1 | `r` | distribution code |
| 2-3 | `dx`, `dy` | input centroid x, y (mm) |
| 4 | `dφ` | input centroid Δφ (deg) |
| 5-6 | `dxp`, `dyp` | input centroid x', y' (mrad) |
| 7 | `de` | input centroid ΔW (MeV) |
| 8-10 | `dEx`, `dEy`, `dEz` | ε growth (%) |
| 11-13 | `mx`, `my`, `mz` | mismatch (fractional) |
| 14 | `dIb` | current jitter (%) |
| 15+ | (advanced) | Twiss-α/β min/max ranges (parsed but not yet used) |

## `ERROR_GAUSSIAN_CUT_OFF`

```
ERROR_GAUSSIAN_CUT_OFF  σ_max
```

Sets the truncation in σ-units for **every** gaussian `ErrorDef` in
the file — the cutoff is applied once, at the end of parsing, so the
card's position relative to the `ERROR_*` cards does not matter, and
when several cutoff cards appear the *last* value wins.  3.0 is a
typical TraceWin-compatible value.

## `ERROR_SET_RATIO`

```
ERROR_SET_RATIO  r1  r2  r3  ...
```

TraceWin's multi-scale sweep (run the same study at error amplitudes
scaled by each ratio).  HELIX parses the card and stores the values on
`lattice.error_ratios`, but **`ErrorStudy` never consumes them** — the
ratio sweep is not implemented.  To sweep amplitudes, run several
studies with scaled σ values.

## Variant and deferred directives

| Directive | Status |
|---|---|
| `ERROR_QUAD/CAV/BEND_*_DYN` | **absorbed as static NCPL errors** — the time-varying (dynamic) semantics are ignored |
| `ERROR_QUAD/CAV/BEND_CPL_*` | **absorbed as static NCPL errors** — the coupled-group semantics are ignored |
| `ERROR_BEAM_DYN` | **absorbed as a static beam error** (same handler as `ERROR_BEAM_STAT`) |
| `ERROR_STAT_FILE` | file-driven error lists (deferred; becomes a no-op marker) |
| `ERROR_RFQ_CEL_NCPL_STAT` | per-cell RFQ errors (deferred; becomes a no-op marker) |
| `ADJUST_STEERER` | closed-orbit auto-correction — **shipped**, see [Orbit correction](07_correction.md) |

The parser routes any keyword starting with `ERROR_QUAD`, `ERROR_CAV`
or `ERROR_BEND` — including the `_DYN` and `_CPL` variants — into the
same static, uncoupled handler, so those cards *do* produce errors,
just without their dynamic/coupled meaning.  See
[Known limitations](../12_validation/03_known_limitations.md).

## Cross-references

* [Element-level errors](03_element_errors.md) — what each parameter means physically.
* [Beam-level errors](04_beam_errors.md) — input jitter.
* [Running studies](05_running_studies.md).

← [Overview](01_overview.md) ·
[Continue to Element errors →](03_element_errors.md)
