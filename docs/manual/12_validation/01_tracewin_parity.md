# TraceWin parity

HELIX is benchmarked against TraceWin partran on every PIP-II
section.  This page summarises the parity status: what matches
exactly, what matches within engineering tolerance, what's known
to differ.

## Coverage matrix

| Element type | Parser | Tracker | Matched within | Notes |
|---|:---:|:---:|---|---|
| Drift | ✓ | ✓ | bit-exact | linear matrix |
| Quadrupole (no g3..g6) | ✓ | ✓ | bit-exact | linear matrix |
| Quadrupole (with g3..g6) | ✓ | ✓ | ~1e-9 | thin Multipole kicks |
| Solenoid (hard-edge) | ✓ | ✓ | bit-exact | linear matrix |
| Solenoid (field map) | ✓ | ✓ | < 0.5 % | RK4; default step density |
| Dipole (sector / rect with edges) | ✓ | ✓ | < 0.5 % | hv=1 fix 2026-05-07 |
| RFGap | ✓ | ✓ | < 1 % | matches partran's GAP |
| FieldMap (1-D / 2-D) | ✓ | ✓ | < 0.5 % | every TraceWin geom code |
| FieldMap3D | ✓ | ✓ | < 1 % | KD/DKD integrators |
| RfqCell (M1 2-term) | ✓ | ✓ | ~10 % (σ_y) | Boris+Hybrid; production path |
| RfqCell (M3 family) | ✓ | partial | — | structural blockers; not production |

## Error directives

| Directive | Parser | Effect |
|---|:---:|---|
| `ERROR_QUAD_NCPL_STAT` | ✓ | full coverage including g3..g6 |
| `ERROR_CAV_NCPL_STAT` | ✓ | dx, dy, dz¹, voltage_rel², phase_offset |
| `ERROR_BEND_NCPL_STAT` | ✓ | dx, dy, dz¹, tilt, field_rel² |
| `ERROR_BEAM_STAT` | ✓ | centroid, ε, mismatch, current |
| `ERROR_GAUSSIAN_CUT_OFF` | ✓ | global cutoff (last card wins, whole file) |
| `ERROR_SET_RATIO` | ✓ | parsed and stored; ratio sweep **not consumed** |
| `ERROR_*_DYN` (dynamic) | ✓ | absorbed as **static** NCPL errors (time-varying semantics ignored) |
| `ERROR_*_CPL_*` (coupled) | ✓ | absorbed as **static** NCPL errors (group-draw semantics ignored) |
| `ERROR_STAT_FILE` | ✗ | deferred (file-driven) |
| `ERROR_RFQ_CEL_NCPL_STAT` | ✗ | deferred (per-cell RFQ) |
| `ADJUST_STEERER` | ✓ | **shipped** — closed-orbit auto-correction, see [Orbit correction](../08_errors/07_correction.md) |

¹ `dz` draws are stored on the element but **ignored by the
tracker** (only dx, dy, tilt are honoured).
² Amplitude errors are currently a **silent no-op** on `FieldMap`
cavities (no `voltage` attribute) and on `Dipole` bends (no `field`
attribute); they take effect on `RFGap` and `Solenoid`.  See
[Element-level errors](../08_errors/03_element_errors.md).

## Validation residuals (5 mA SC)

| Section | σ_x rms | σ_y rms | Transmission |
|---|---|---|---|
| MEBT | < 3 % | < 3 % | matches |
| MEBT + HWR | ~ 3 % | ~ 3 % | matches |
| MEBT + HWR + SSR1 + SSR2 | ~ 3 % | ~ 3 % | matches |
| Full PIP-II 256m | 8.6 % rms | 13.3 % rms | within noise |

The full PIP-II numbers are larger than per-section because:

* 5000-particle MP has ~1 % σ statistical noise floor.
* TSC kernel SC kicks differ from envelope's analytic Σ kicks.
* TraceWin partran also uses its own SC algorithm with different
  mesh choices.

End-of-machine values:

| | HELIX | TW | diff |
|---|---|---|---|
| σ_x | 3.69 mm | 3.55 mm | +3.9 % |
| σ_y | 4.71 mm | 4.94 mm | -4.6 % |

## What's known to differ

### 1. σ_φ reporting convention (~4× factor)

HELIX reports σ_φ at the **local cavity frequency**.  TraceWin
reports σ_φ at the **bunch frequency** (162.5 MHz, fixed).
At the end of HB650, the local frequency is 650 MHz, so HELIX's
σ_φ is 4× larger.  The underlying σ_z is the same.

To convert HELIX → TW reporting: `σ_φ_TW ≈ σ_φ_HELIX × 162.5/local_f_MHz`.

### 2. PIC kernel calibration ~3 % offset

A documented absolute-calibration bias.  HELIX is consistently
~3 % higher in σ than TraceWin under the same conditions.  The
ratio is stable across lattices, which strongly suggests it's a
kernel-coefficient issue rather than a code bug.  Investigation
deferred; for design-stage work a ~3 % residual is acceptable.

## Cross-references

* PIP-II validation — the full 256m run.
* [Known limitations](03_known_limitations.md) — deferred features.
* [Convergence guide](04_convergence_guide.md).

← [Eigenemittance](../11_examples/09_eigenemittance.md) ·
Continue to PIP-II validation →
