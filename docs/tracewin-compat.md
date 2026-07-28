# TraceWin compatibility — v2

This page documents the breaking changes in the `linac_gen` parser / writer
after the TraceWin-strict rewrite.  If you have `.dat` files that parsed
cleanly under the legacy parser, read this before re-running them.

## Highlights

- **Global sub-step configuration via `PARTRAN_STEP step1 step2`** replaces
  the old per-element `n_steps` fourth-positional on `DRIFT`, `QUAD`,
  `SOLENOID`, etc.  `DRIFT` and `FIELD_MAP` honour `step1` (integration
  per metre) and `step2` (space-charge per metre); every other element
  is tracked in exactly **2** sub-steps with one space-charge kick at
  the mid-plane.
- **`QUAD` fourth positional is the skew angle Θ (°)**, *not* sub-step
  count.  A legacy line `QUAD 50 5 20 10` is now read as a 10° skew
  quadrupole (produces x–y coupling) — not a 10-step integration.
- **`DRIFT` third positional is the vertical aperture `Ry` (mm)**, not
  sub-step count.  `DRIFT 50 30 10` is a rectangular (30 × 10) aperture,
  not 10 sub-steps.
- **New first-class elements**: `APERTURE`, `EDGE`, `BEND` (with combined-
  function field index `N`), `STEERER` / `THIN_STEERING`.
- **`GAP` voltage is in volts on the card**, consistent with TraceWin.
  The `RFGap` element internally stores voltage in MV; the parser
  divides by 1e6 and the writer multiplies on emit.
- **`FIELD_MAP` now carries the full nine fields** (`geom L θ R kb ke Ki
  Ka filename P`).  The old 5-positional shorthand no longer parses.
- **`NCELLS` (multi-gap cavity) is a first-class element** — expanded at
  track time into one thin RF gap per cell (TraceWin's own model), with all
  three modes, βg>0/=0/<0, phase P=0/1/2 + `SET_SYNC_PHASE`, end-cell
  corrections, and the βs≠0 transit-time tail.  Imports, tracks, round-trips.
- **Unsupported cards are skipped with a warning — and physics
  downgrades are never silent** (honesty round, 2026-07).  Three
  classes: diagnostic-hardware markers stay silent; cards whose
  TraceWin meaning HELIX cannot honour (`SUPERPOSE_MAP` /
  `SHIFT_IN_FIELD_MAP`, `READ_DST` / `BEAM_ROT`, `ERROR_*_DYN`
  treated as static, coupled `ERROR_*_CPL_*` sampled independently,
  negative field-map geometry codes, missing field-map files) each
  emit a warning in `metadata["warnings"]` (shown by the GUI status
  bar, echoed to stderr by the CLI); fit hints (`MATCH_FAM_*`,
  `RFQ_GEOM`, …) are reported once per deck.  Pass `strict=True` to
  `parse_tracewin(...)` to make unknown cards AND every physics
  downgrade fatal — a strict parse never returns a lattice whose
  physics silently differs from the deck (in particular it refuses,
  rather than drops, a `FIELD_MAP` whose field file is missing).

## Cheat sheet — all supported cards

Each row is `CARD  positional_args…` in TraceWin order.  Square brackets
mark optional fields with their default.

```
DRIFT     L(mm)  R(mm)=0  [Ry(mm)=None]  [x_shift(mm)=0]  [y_shift(mm)=0]
QUAD      L(mm)  G(T/m)  R(mm)=0  [Θ(°)=0]  [G3(T/m²)=0]  [G4(T/m³)=0]  [G5(T/m⁴)=0]  [G6(T/m⁵)=0]  [GFR(mm)=0]
SOLENOID  L(mm)  B(T)  R(mm)=0
GAP       E0TL(V)  φs(°)  R(mm)=0  [P=0]
FIELD_MAP geom(int)  L(mm)  [θ(°)=0]  [R(mm)=0]  [kb=1]  [ke=1]  [Ki=0]  [Ka=1]  filename  [P=0]
NCELLS    mode(0=2π,1=π,2=π&2π)  Nc  βg  EoT(V/m)  θs(°)  R(mm)  [P=0]  [kEoTi=0]  [kEoTo=0]  [dzi(mm)=0]  [dzo(mm)=0]  [βs Ts kT's k²T''s Ti … To …]
BEND      α(°)  ρ(mm)  [N=0]  [R(mm)=0]  [HV=0]
EDGE      β(°)  ρ(mm)  [G(mm)=0]  [K1=0.45]  [K2=2.80]  [R(mm)=0]  [HV=0]
APERTURE  dx(mm)  [dy(mm)=0]  [n(type)=0]           # 0 rect, 1 circle, 2 pepperpot, 3 fraction, 4/5 finger, 6 ring
STEERER   bl_x(T·m or V)  bl_y(T·m or V)  [R(mm)=0]  [elec(0/1)=0]
THIN_STEERING   (alias of STEERER)

; Control cards
FREQ           f(MHz)
PARTRAN_STEP   step1(per m)  step2(per m)
TITLE          "<free-form text>"
END

; Recognised but ignored (warn-skip unless strict=True)
LATTICE  LATTICE_END  REPEAT_ELE  SET_ADV
DIAG_PHASE  DIAG_SIZE
MATCH_FAM_GRAD  MATCH_FAM_PHASE  ...
```

`DIAG_POSITION N X Y [dm]` and plain `ADJUST N v` are **active**:
the marker keeps the family number and centroid targets, and integer
`ADJUST` targets that match a declared diagnostic family bind the
next element (TraceWin semantics) and become matcher variables.
Diagnostic matching divergence: TraceWin tunes **sequentially, in
diagnostic-number order**; HELIX solves one **global least-squares
per family** (all targets and all family variables at once).  The
corrected *orbit* agrees; individual fitted parameter values need
not — do not compare knob-by-knob against a TraceWin fit.

Practical guidance for MP-mode diagnostic fits: the centroid reading
carries sampling noise ≈ σ/√N (≈0.06 mm at 5 000 particles for a
4 mm beam) — targets *below* that floor make the optimizer chase the
seed's noise pattern.  Use ≥20 000 particles (or de-weight/free
sub-noise-floor targets), and give the deck's `ADJUST` cards real
`min`/`max` bounds: without them every knob is unbounded, which both
slows convergence and (before HELIX's dead-beam guard) allowed
beam-killing excursions.

**Envelope mode now tracks the beam centroid** (first moment) exactly
like TraceWin's envelope: steerer kicks, `dx`/`dy`/`tilt` misalignment
feed-down, dispersion and freq-jump rescales all propagate, and the TW
ENV export's centroid columns carry real values.  Diagnostic fits under
`cost_solver="envelope"` are therefore both **noiseless and ~100×
faster** (fnalscl: 1.4 min vs 2 h, landing at µm-level residuals) —
use MP mode when you need loss/transmission or nonlinear-SC effects in
the cost, envelope mode for orbit work.  Note the historical envelope
`tilt` conjugation modelled −tilt (invisible to symmetric Σ,
sign-flipping the coupled-plane centroid); it now matches the MP
tracker exactly.

## Migrating a legacy `.dat`

Most legacy files break in just two places — the 4th positional on
`DRIFT` / `QUAD`.  A sample migration:

```diff
- DRIFT 50 30 5           ; old: n_steps=5
+ DRIFT 50 30             ; new: Ry=None (circular); sub-steps via PARTRAN_STEP
```

```diff
- QUAD 50 5 20 10         ; old: n_steps=10
+ QUAD 50 5 20            ; new: Θ=0; 2 sub-steps per QUAD (TraceWin default)
```

Add a `PARTRAN_STEP` near the top of the file to set the global step
density:

```diff
  FREQ 352.21
+ PARTRAN_STEP 100 50
```

`step1=100/m` means a 200 mm drift gets `ceil(0.2 × 100) = 20`
integration sub-steps.  `step2=50/m` applies one space-charge kick
every 20 mm of drift.

For `GAP` cards, if your file used `ttf` as the 4th positional, switch
to `P` (phase convention flag) and absorb TTF into `E0TL`:

```diff
- GAP 1.0 -30 20 0.8                ; old: V0 in MV, ttf=0.8
+ GAP 800000 -30 20 0               ; new: E0TL=0.8 MV × 1.0 * 10^6 = 8e5 V; P=0
```

## Conventions inside the code

- `Quadrupole(skew_angle=Θ)` implements the skew via the rotation sandwich
  `M_skew = R(Θ) · M_normal · R(−Θ)` (coupled 4×4 block).
- `Dipole(field_index=N)` uses `kx² = (1−N)/ρ²`, `ky² = N/ρ²` for the
  body focusing; dispersion appears in the horizontal plane only.
- `Edge(pole_rotation=β, gap=G, k1=K1)` is a thin-lens element; fringe
  correction ψ `= K1·G·(1+sin²β) / (ρ·cos β)` subtracts from β on the
  vertical side.  Compose `EDGE … BEND … EDGE` for rectangular magnets.
- `Aperture(dx, dy, aperture_type=0|1|2|3|4|5|6)` — shape flag matches
  TraceWin.  Pepperpot (2) and ring (6) are parsed but not simulated.
- `parse_tracewin(path, strict=False)` returns `(Lattice, metadata)`
  where `metadata["warnings"]` lists every skipped / fall-back event.

## What is still NOT supported

- `LATTICE n1 n2 … LATTICE_END` blocks — treated as metadata; the
  elements inside are still tracked individually.
- `REPEAT_ELE k n` — not expanded; the repeated block is skipped.
- `DIAG_*`, `SET_ADV`, `MATCH_FAM_*`, `ADJUST*` — skipped.
- Pepperpot / ring apertures (`APERTURE` types 2 and 6) — parsed-as-skip;
  every particle passes through.
- Higher-order multipole components on `QUAD` (G3…G6) — stored on the
  element instance but not yet applied during tracking.
- Electric variants (`QUAD_ELE`, electric `STEERER`) — not yet parsed.

File an issue if any of these block a real linac file you want to run.
