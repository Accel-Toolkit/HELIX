# SET / ADJUST card reference

HELIX's matcher reads `ADJUST_*` cards (variables — knobs the matcher
tunes) and `SET_*` / `MIN_*` cards (constraints — targets the matcher
tries to satisfy) directly from the lattice `.dat`.  This is the
TraceWin matching language; HELIX honours a working subset and adds
two HELIX-specific cards (`MIN_EMIT_4D_GROWTH`, `MIN_TRANSMISSION`)
for SC-driven cryomodule matching.

This page documents every card the parser recognises:

* **Signature** — the keyword + positional fields, in parser order.
* **Parameters** — per-field table with type, default, meaning.
* **Residual** — the cost contribution (for active constraints).
* **Example** — a working `.dat` line.

Source of truth:
[`tracewin_syntax.py`](https://github.com/Abhishek-Pathak-90/HELIX/blob/master/linac_gen/io/tracewin_syntax.py)
(parser schema),
[`lattice_commands.py`](https://github.com/Abhishek-Pathak-90/HELIX/blob/master/linac_gen/elements/lattice_commands.py)
(class definitions),
[`constraints.py`](https://github.com/Abhishek-Pathak-90/HELIX/blob/master/linac_gen/matching/constraints.py)
(evaluators).

---

## Active SET_* constraints

These cards contribute a non-zero residual to the cost.

### `SET_TWISS family αx βx αy βy αz βz kax kbx kay kby kaz kbz`

Equality target on Twiss (α, β) at lattice exit.  No emittance fields
— the matcher cares about the Twiss shape, not the action.

| Pos | Field    | Type  | Units | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1  | `family`   | str   | —     | `""`    | no  | Element family the target applies to. Rarely used in practice; pass `""`. |
| 2  | `alpha_x`  | float | —     | 0.0     | no  | Target Twiss α in the x plane at lattice exit. Pure target value — has no selection role. |
| 3  | `beta_x`   | float | mm/mrad | 0.0   | no  | Target Twiss β in x. Pure target — `β_x = 0` is a *valid target of zero*, not a disable switch. |
| 4  | `alpha_y`  | float | —     | 0.0     | no  | Target Twiss α in y. |
| 5  | `beta_y`   | float | mm/mrad | 0.0   | no  | Target Twiss β in y. |
| 6  | `alpha_z`  | float | —     | 0.0     | no  | Target Twiss α in z — **HELIX-internal (Δφ, ΔW) convention**: negate the TraceWin value when transcribing (see the admonition below and [Migrating from TraceWin](../appendices/E_migrating_from_tw.md)). |
| 7  | `beta_z`   | float | deg/MeV | 0.0   | no  | Target Twiss β in z, in **deg/MeV** (the same unit as `BeamConfig.beta_z` and the `.dst` header), at the **exit machine frequency**. |
| 8  | `kax`      | int   | flag  | 0       | no  | "Use α_x" flag. **`1` puts α_x into the residual; `0` leaves it out.**  The k-flags are the *only* plane/parameter selection mechanism. |
| 9  | `kbx`      | int   | flag  | 0       | no  | "Use β_x" flag — `1` selects β_x. |
| 10 | `kay`      | int   | flag  | 0       | no  | "Use α_y" flag. |
| 11 | `kby`      | int   | flag  | 0       | no  | "Use β_y" flag. |
| 12 | `kaz`      | int   | flag  | 0       | no  | "Use α_z" flag. |
| 13 | `kbz`      | int   | flag  | 0       | no  | "Use β_z" flag. |

Residual: one entry per k-flag equal to 1 — `(measured − target)` for
that Twiss parameter.  **If all six k-flags are zero the card is
skipped entirely** (no constraint is registered), so a `SET_TWISS`
with targets but no flags is silently inert.

!!! success "z-plane flags are functional (2026-07)"
    Both the envelope solver and the multi-particle recorder now record
    longitudinal Twiss (α_z/β_z) from the (Δφ deg, ΔW MeV) beam matrix,
    so `kaz`/`kbz` = 1 are genuine residual axes under
    `least_squares`/`bo`/`cmaes`/`sequential_scan` (the `gradient`
    algorithm's torch mirror does not implement the z residual and
    refuses).  Conventions:

    * **α_z sign** — HELIX's Δφ runs opposite to TraceWin's z (a late
      particle has Δφ > 0), so **negate a TraceWin α_z when
      transcribing the card**, exactly as for the input beam
      ([Migrating from TraceWin](../appendices/E_migrating_from_tw.md)).
      This keeps `SET_TWISS` consistent with `BeamConfig.alpha_z` and
      `ADJUST_BEAM_TWISS`, which use the same internal convention.
    * **β_z unit** — deg/MeV **at the exit machine clock**: across a
      `FREQ` jump β_z scales by f_new/f_old (like `emit_z`), α_z is
      invariant.
    * A degenerate or absent longitudinal record at the **baseline**
      evaluation (legacy/foreign results; DC/continuous beam or
      ε_z ≈ 0 at the exit) makes `match()` **refuse up front**, naming
      the card — the z axes would have no usable gradient and the
      match would silently solve a different problem.  Pass
      `allow_inert_constraints=True` to warn-and-continue (the z
      residual slots then read a fixed penalty — a constant,
      gradient-free cost offset that honestly shows the z targets as
      unmet).  The residual vector length never changes mid-run: any
      degenerate evaluation fills the z slots with the same fixed
      finite penalty instead of dropping them
      (`scipy.least_squares` requires a constant residual shape, and
      the fill is stateless so parallel CMA-ES workers score
      identically to the parent).  **Penalty value**: 10³ per slot,
      in each slot's native units (α_z dimensionless, β_z deg/MeV) —
      deliberately far above typical O(1)–O(10) SET_TWISS residuals
      yet finite for the trust-region step.  A solution whose final
      residuals still carry the penalty is flagged machine-readably:
      `MatchResult.z_penalty_infeasible` is `True` and the matching
      report prints `[INFEASIBLE: SET_TWISS z targets unmet]` next to
      the status — a "successful" override match cannot silently pose
      as having met the z targets.

```
SET_TWISS "" -1.0 2.0 1.0 2.0 0.0 10.0 1 1 1 1 0 1
```

→ Constrain α_x = −1.0, β_x = 2.0, α_y = +1.0, β_y = 2.0, β_z = 10.0
deg/MeV (α_z free — its `kaz` flag is 0); family unused.

### `SET_SIZE k x_mm y_mm phi_or_z k2`

Equality target on σ at lattice exit, per plane.  Residual entries are
included only for planes whose target is positive.

| Pos | Field      | Type  | Units | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `k`        | float | weight | 0.0 | no | Constraint weight. **`k = 0` skips the card entirely** (no constraint registered); any non-zero value multiplies the residual.  HELIX always evaluates at lattice exit regardless of `k`. |
| 2 | `x_mm`     | float | mm   | 0.0 | no | Target σ_x at exit. **`0` = "don't constrain x"** (no residual entry for x). |
| 3 | `y_mm`     | float | mm   | 0.0 | no | Target σ_y at exit. `0` disables y. |
| 4 | `phi_or_z` | float | deg / mm | 0.0 | no | Target σ_φ at exit in degrees when POSITIVE; a **negative value means the target is σ_z = \|value\| in mm** (TraceWin manual rule), converted through the local βλ (2026-07: the z-form used to be silently dropped). `0` disables longitudinal. |
| 5 | `k2`       | int   | flag | 0   | no | Per the TraceWin manual, `k2` selects whether the **transverse** sizes include the beam centroid position (it is *not* the deg/mm selector — that is the sign of field 4).  HELIX evaluates sizes about the centroid (the `k2=0` form) and warns on `k2≠0`; since 2026-07 the `match()` pre-run audit **refuses** an active `k2≠0` card outright unless `allow_inert_constraints=True`. |

Residual: `k × (σ_measured − target)` per plane with a positive target.

```
SET_SIZE 1 3.0 0 0 0
```

→ σ_x → 3.0 mm at exit with weight 1; y and longitudinal
unconstrained; `k2=0` inert.  With `k=0` the whole card would be
skipped.

### `SET_SIZE_MAX k n_elems x_mm y_mm phi_or_z k2`

One-sided upper bound on σ over the last `n_elems` recorded steps.
Used to enforce aperture clearance ("σ_x must stay ≤ 5 mm anywhere in
the last 20 elements").

| Pos | Field      | Type  | Units | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `k`        | float | weight | 0.0 | no | Constraint weight. **`k = 0` skips the card**; non-zero multiplies the residual.  Evaluation always uses the lattice-exit window. |
| 2 | `n_elems`  | int   | —    | 1   | no | Length of the look-back window. Residual takes the **worst** (max) σ across the last `n_elems` recorded steps. |
| 3 | `x_mm`     | float | mm   | 0.0 | no | Upper bound on σ_x within the window. `0` disables x. |
| 4 | `y_mm`     | float | mm   | 0.0 | no | Upper bound on σ_y within the window. `0` disables y. |
| 5 | `phi_or_z` | float | deg  | 0.0 | no | Upper bound on σ_φ within the window. `0` disables longitudinal. |
| 6 | `k2`       | int   | flag | 0   | no | σ_φ vs σ_z selector (HELIX always treats as σ_φ). |

Residual: `k × max(0, σ_worst_in_window − target)` per active plane.

```
SET_SIZE_MAX 1 20 5.0 5.0 0 0
```

→ Inside the last 20 elements, σ_x ≤ 5 mm AND σ_y ≤ 5 mm; longitudinal
not bounded.

### `SET_SIZE_MIN k n_elems x_mm y_mm phi_or_z k2`

Same field layout as `SET_SIZE_MAX` (see table above — including the
`k = 0` skip / weight semantics), but the residual flips sign:
`k × max(0, target − σ_worst)`, where the "worst case" for a floor is
the **smallest** σ in the window (for `SET_SIZE_MAX` it is the
largest).  Prevents over-focusing to a singular waist that downstream
optics can't recover from — including a dip in the *middle* of the
window, which older releases missed (they checked the window maximum
for both bounds).

```
SET_SIZE_MIN 1 5 0.5 0.5 0 0
```

→ Within the last 5 elements, σ_x ≥ 0.5 mm AND σ_y ≥ 0.5 mm.

### `SET_BEAM_PHASE_ADV k n_elems mu_x_deg mu_y_deg mu_z_deg`

Target beam phase advance (∫ds/β) over the **`n_elems` non-command
elements FOLLOWING the card** (TW: the card applies at the entrance of
the element where it is placed).  All three planes are active; μ_z is
measured via the effective (z, z′) β (the recorder-Jacobian conversion
of the native deg/MeV β).

| Pos | Field      | Type  | Units | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `k`        | float | weight | 0.0 | no | Cost multiplier (the constraint's weight). **`k = 0` skips the constraint entirely** (it isn't added to the residual vector). |
| 2 | `n_elems`  | int   | —    | 1   | no | Number of non-command elements after the card that the ∫ds/β span covers. **`0` disables the card** (TW zero-means-off convention). |
| 3 | `mu_x_deg` | float | deg  | 0.0 | no | Target horizontal phase advance. **`0` disables the x plane** (it is NOT driven toward zero). |
| 4 | `mu_y_deg` | float | deg  | 0.0 | no | Target vertical phase advance; `0` disables the plane. |
| 5 | `mu_z_deg` | float | deg  | 0.0 | no | Target longitudinal phase advance; `0` disables the plane. |

Residual: `(μ_x − mu_x_deg, μ_y − mu_y_deg, μ_z − mu_z_deg)` with
disabled planes contributing `0.0`, multiplied by weight `k`.
Degenerate cases are never silently bridged: a span where the beam is
lost (no valid β) takes a fixed `1e3` penalty; a plane whose data was
never recorded (e.g. μ_z on results without a σ-matrix) is skipped
with a warning; a trailing card with no following elements is disabled
with a warning.

```
SET_BEAM_PHASE_ADV 1.0 80 90.0 90.0 0
```

→ Target μ_x = μ_y = 90° over the 80 elements following the card;
μ_z unconstrained; weight 1.

### `SET_KE_OUT_MIN energy_MeV weight`

One-sided floor on the reference particle's end-of-line kinetic
energy.  Prevents the matcher from buying ε reduction by detuning a
cavity off-crest.

| Pos | Field        | Type  | Units | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `energy_mev` | float | MeV   | 0.0     | no       | Lower-bound floor on the reference particle's `W_kin` at lattice exit.  Defaults to 0.0 (no floor) when omitted. |
| 2 | `weight`     | float | —     | 1.0     | no       | Multiplier on the one-sided residual. Recommended **~10× the per-plane ε weight** so the energy floor dominates marginal ε gains.  `0` skips the constraint. |

Residual: `weight × max(0, energy_mev − W_kin_out)`.

```
SET_KE_OUT_MIN 5.0 10.0
```

→ Exit kinetic energy must be ≥ 5 MeV; weight 10.

---

## Active MIN_* constraints

### `MIN_EMIT_GROWTH plane weight`

One-sided per-plane emittance growth penalty.  A drop below the seed's
emittance is free; only growth costs.

| Pos | Field    | Type  | Units      | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `plane`  | str   | —          | —       | **yes**  | Which plane to constrain: exactly `"X"`, `"Y"`, or `"Z"`. Transverse planes are compared in normalised ε (ε_geom · βγ); Z uses the native `emit_z` (deg·MeV), which is already an action variable. |
| 2 | `weight` | float | —          | 1.0     | no       | Cost multiplier on the one-sided residual. `0` skips the constraint. |

Residual: `weight × max(0, εn_out − εn_in)`.  Place **one card per
plane** to constrain all three.

```
MIN_EMIT_GROWTH X 1.0
MIN_EMIT_GROWTH Y 1.0
MIN_EMIT_GROWTH Z 1.0
```

→ Three independent per-plane growth constraints.  Use this when each
plane has a hard independent target; otherwise prefer
`MIN_EMIT_4D_GROWTH` (next), which allows x↔y trade-off as physics.

### `MIN_EMIT_4D_GROWTH weight tol_4d tol_z`

Coupling-aware version: produces a **2-element** residual.  Lets the
matcher trade transverse ε between planes (e.g. through a solenoid
rotation) without paying a cost — penalises only growth in the
*coupled* 4-D area and in the longitudinal plane independently.

| Pos | Field    | Type  | Units | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `weight` | float | —     | 1.0     | no       | Common multiplier on both residual entries. `0` skips the constraint entirely. |
| 2 | `tol_4d` | float | ratio | 1.0     | no       | Multiplicative tolerance on the 4-D transverse growth: residual is `max(0, ε_4D_out − tol_4d × ε_4D_in)`. `1.0` = strict (any growth penalised); `1.10` = allow 10% before residual becomes positive. **Must be ≥ 1.0** (raises ValueError at construction). |
| 3 | `tol_z`  | float | ratio | 1.0     | no       | Same idea for longitudinal: `max(0, εnz_out − tol_z × εnz_in)`. **Must be ≥ 1.0.** |

Definitions: `ε_4D_norm = √det(Σ_4×4_transverse) · (βγ)²`,
`ε_z_norm = emit_z · βγ`.

```
MIN_EMIT_4D_GROWTH 1.0 1.05 1.20
```

→ Allow 5% 4-D transverse growth and 20% longitudinal growth before
either residual becomes positive; weight 1.

### `MIN_TRANSMISSION threshold_pct weight`

End-of-line transmission floor — closes the ε-gaming gap (in MP mode,
the matcher could otherwise improve ε by killing halo particles).

| Pos | Field           | Type  | Units | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `threshold_pct` | float | %     | 99.0    | no       | Transmission floor in percent. **Must be in `[0, 100]`** (raises ValueError at construction otherwise). The matcher penalises trials that finish below this value. |
| 2 | `weight`        | float | —     | 1.0     | no       | Cost multiplier on the one-sided residual. Recommended **10–100×** the per-plane ε weight so loss dominates marginal ε gains. `0` skips the constraint. |

Residual: `weight × max(0, threshold_pct − T_final) / 100`.

The `/100` keeps a 1% loss below threshold producing a residual of
~0.01 — comparable to the other ε-style residuals so weight knobs
live in a familiar range.

```
MIN_TRANSMISSION 99.9 50.0
```

→ Penalise transmission below 99.9% with weight 50.  A 0.5% loss
(T = 99.4) produces a residual of `50 × 0.5 / 100 = 0.25`.

**MP cost-solver only.**  Envelope mode is silently inert (apertures
are no-ops; transmission is always 100%) and prints a one-time stderr
warning so the user knows to switch to `cost_solver="mp"`.

---

## Parsed-but-inert SET_* cards

These cards are recognised by the parser so HELIX can round-trip
existing TraceWin `.dat` files cleanly, but they currently produce
**no residual contribution**.  They appear in the matching report
labelled `... (stub)` with a `notes` field explaining why.

| Card | Signature | Status |
|---|---|---|
| `SET_SYNC_PHASE` | (no args) | Sync-phase assertion card; lattice machinery only. |
| `SET_BEAM_PHASE_ERROR` | `dphi_deg random_flag` | Parsed; not yet wired. |
| `SET_BEAM_E0_P0` | `k dE_MeV dphi_deg ke kp` | Parsed; not yet wired. |
| `SET_BEAM_ENERGY` | `k energy_MeV` | Parsed; not yet wired. |
| `SET_GAUSSIAN_CUT_OFF` | `sigma` | Distribution truncation default; not a matching residual. |
| `SET_POSITION` | `k x_mm xp_mrad y_mm yp_mrad` | Reserved for future per-marker position targeting. |
| `SET_ACHROMAT` | `k f1 f2 plane` | Stub. |
| `SET_SEPARATION` | `k sx sy` | Stub. |
| `SET_ADV` | `kxot kyot` | Stub. |

Adding evaluators for these is a follow-up task; track via PRs against
[`constraints.py`](https://github.com/Abhishek-Pathak-90/HELIX/blob/master/linac_gen/matching/constraints.py).

---

## ADJUST_* — variable cards

HELIX uses a **single generic** `ADJUST` card with a `param_idx`
selector — not the per-element-type variants (`ADJUST_QUAD`,
`ADJUST_SOLENOID`, …) some older TraceWin docs show.

### `ADJUST target param_idx [link_group vmin vmax start_step kn]`

Declares one optimiser variable: `target.<attr>` is varied within
`[vmin, vmax]`.

| Pos | Field        | Type    | Units   | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `target`     | str / int | —     | —       | **yes**  | The element to vary. If **integer N** → 1-based diagnostic number (skips command cards while counting). If **string** → element whose `name` starts with this string, searched first forward from this ADJUST card, then globally as a fallback. Case-insensitive. |
| 2 | `param_idx`  | int       | —     | —       | **yes**  | Which scalar on the element to vary. See the **param_idx → attribute table** below. Unsupported `(element-class, param_idx)` pairs raise `MatchingConfigError` at collection time. |
| 3 | `link_group` | int       | —     | 0       | no       | Group ID for linked variables. **Variables sharing the same non-zero `link_group` collapse into one optimiser column** (knobs move in lockstep — useful for ganged supplies). `0` = unlinked, each variable independent. |
| 4 | `vmin`       | float     | varies | 0.0    | no       | Lower bound on the variable's value. **`vmin = vmax = 0` is the "unset" convention → ±∞ bounds.** Global algorithms (DE, DA, CMA-ES, sequential_scan) require finite bounds and raise `ValueError` otherwise. Units match the underlying attribute (T/m for gradient, T for solenoid field, degrees for cavity phase, …). |
| 5 | `vmax`       | float     | varies | 0.0    | no       | Upper bound on the variable's value (same units as `vmin`). |
| 6 | `start_step` | float     | varies | 0.0    | no       | Initial step hint for population-based optimisers. **`least_squares` ignores this** (it computes its own Jacobian step). Same units as `vmin`/`vmax`. |
| 7 | `kn`         | int       | flag  | 0       | no       | TraceWin sub-flag. HELIX preserves it for round-trip but doesn't change behaviour based on its value. |

**param_idx table** (from `linac_gen/matching/variables.py:68`):

| Element | `1` | `2` | `3` | `5` | `6` |
|---|---|---|---|---|---|
| `Drift`      | length |  |  |  |  |
| `Quadrupole` | length | **gradient** | aperture |  |  |
| `Solenoid`   | length | **field**    | aperture |  |  |
| `Dipole`     | **angle**  | rho |  |  |  |
| `RFGap`      | voltage | phase |  |  |  |
| `FieldMap`   |  | **phase** |  | **kb** | **ke** |
| `FieldMap3D` |  | **phase** |  | **kb** | **ke** |
| `Steerer`    | bx_l   | by_l |  |  |  |

Bold = parameters typically used in matching.  Unsupported
(element-class, param_idx) pairs raise `MatchingConfigError` at
collection time with a clear message.

Examples (HWR cryomodule, from `examples/emittance_min/pipii_hwr_cm_match_4d_relaxed.dat`):

```
ADJUST FMAP 5 1 -1.98 -1.62 0.02 0     ; sol1 kb in [-1.98, -1.62]
ADJUST FMAP 6 2  0.248 0.372 0.005 0   ; cav1 ke in [0.248, 0.372]
ADJUST FMAP 2 3 -60.0 -40.0 1.0 0      ; cav1 phase in [-60°, -40°]
```

The three ADJUSTs above declare three independent variables.  Their
`link_group` values (1, 2, 3) are all *different*, so the optimiser
treats them as separate columns.  Set them all to the same non-zero
integer to make them move in lockstep (useful for ganged supplies).

### `ADJUST_STEERER diag_n vmax first_step`

Declares both x and y steerer strengths as variables (used for
orbit correction).

| Pos | Field        | Type  | Units    | Default | Required | Description |
|---|---|---|---|---|---|---|
| 1 | `diag_n`     | int   | —        | —       | **yes**  | 1-based index into the `STEERER` family. Counts the n-th `Steerer` element following this card (with a global fallback if not found forward). |
| 2 | `vmax`       | float | T·m      | 0.0     | no       | Symmetric bound — actual bounds become `[-vmax, +vmax]`. **`vmax = 0` defaults to `[-1, +1]`** so an unset card still gives the optimiser a usable range. |
| 3 | `first_step` | float | T·m      | 0.0     | no       | Initial step hint for population-based optimisers (ignored by least_squares). |

```
ADJUST_STEERER 3  0.5 0.05
```

→ Steerer #3's `bx_l` and `by_l` both varied in `[-0.5, +0.5]` T·m;
initial step 0.05 T·m.

### `ADJUST_STEERER_BX diag_n vmax first_step`

Same as `ADJUST_STEERER` but **x-axis only** (declares one variable,
`bx_l`).  Use `ADJUST_STEERER_BY` for y-only.

### `ADJUST_BEAM_TWISS diag_n AlpX_flag BetX_flag AlpY_flag BetY_flag AlpZ_flag BetZ_flag`

Declares input Twiss parameters (α, β per plane) as variables — used
for matching a beam *into* a periodic structure.  The matcher mutates
the supplied `BeamConfig` rather than a lattice element.

| Pos | Field        | Type | Default | Required | Description |
|---|---|---|---|---|---|
| 1 | `diag_n`     | int  | 0   | no | 1-based diagnostic number for the target axis (TraceWin convention).  HELIX honours non-zero values when present. |
| 2 | `AlpX_flag`  | int  | 0   | no | Flag for input α_x: **`0` = skip, `1` = adjust, `2` = couple to previous axis**. |
| 3 | `BetX_flag`  | int  | 0   | no | Flag for input β_x (same values as `AlpX_flag`). |
| 4 | `AlpY_flag`  | int  | 0   | no | Flag for input α_y. |
| 5 | `BetY_flag`  | int  | 0   | no | Flag for input β_y. |
| 6 | `AlpZ_flag`  | int  | 0   | no | Flag for input α_z. |
| 7 | `BetZ_flag`  | int  | 0   | no | Flag for input β_z. |

```
ADJUST_BEAM_TWISS 1 1 1 1 1 0 0
```

→ Free input α_x, β_x, α_y, β_y; freeze α_z, β_z.  Six new optimiser
variables (or four, depending on coupling flag interpretation).

### `ADJUST_BEAM_CENTROID diag_n X_flag Xp_flag Y_flag Yp_flag Z_flag Zp_flag`

Frees the beam's input centroid (`x, x', y, y', z, z'`) as
optimisation variables.

| Pos | Field      | Type | Default | Required | Description |
|---|---|---|---|---|---|
| 1 | `diag_n`   | int | 0 | no | Diagnostic number (as in `ADJUST_BEAM_TWISS`). |
| 2 | `X_flag`   | int | 0 | no | Free input `x` centroid offset?  `0`/`1`/`2`. |
| 3 | `Xp_flag`  | int | 0 | no | Free `x'` (transverse angle). |
| 4 | `Y_flag`   | int | 0 | no | Free `y`. |
| 5 | `Yp_flag`  | int | 0 | no | Free `y'`. |
| 6 | `Z_flag`   | int | 0 | no | Free longitudinal centroid offset. |
| 7 | `Zp_flag`  | int | 0 | no | Free `z'`. |

### `ADJUST_BEAM_EMIT diag_n Ex_flag Ey_flag Ez_flag`

Frees the input emittances.

| Pos | Field      | Type | Default | Required | Description |
|---|---|---|---|---|---|
| 1 | `diag_n`   | int | 0 | no | Diagnostic number. |
| 2 | `Ex_flag`  | int | 0 | no | Free input ε_x? |
| 3 | `Ey_flag`  | int | 0 | no | Free input ε_y? |
| 4 | `Ez_flag`  | int | 0 | no | Free input ε_z? |

### `ADJUST_BEAM_CURRENT diag_n I_flag`

Frees the input beam current.

| Pos | Field    | Type | Default | Required | Description |
|---|---|---|---|---|---|
| 1 | `diag_n` | int | 0 | no | Diagnostic number. |
| 2 | `I_flag` | int | 0 | no | Free input current?  `0`/`1`. |

---

## How positioning works

`SET_*` and `ADJUST_*` cards are placed in the lattice in source order
along with element cards.  Most `SET_*` residuals are evaluated at
lattice exit regardless of where the card sits — the in-file position
is cosmetic for those, and the `n_elems` field on `SET_SIZE_MAX`
extends the evaluation window *backward* from the exit.  The exception
is **`SET_BEAM_PHASE_ADV`**, whose span runs *forward* from the card's
own position over its `n_elems` following non-command elements (TW
semantics) — its in-file position matters.

`ADJUST_*` cards declare variables in the order encountered; the
target lookup walks forward from the card.

---

## Complete worked example

Single-quad waist matching (from
[`05_recipes.md`](05_recipes.md) Recipe 1):

```
DRIFT 200 10 0 0 0
ADJUST QUAD_001 2 1 0.5 30 0.5 0   ; QUAD_001.gradient varied in [0.5, 30]
QUAD 100 5.0 10 0 0 0 0 0 0
DRIFT 200 10 0 0 0
SET_SIZE 1 3.0 0 0 0                ; target σ_x = 3.0 mm at exit
END
```

HWR cryomodule emittance + loss minimisation (paraphrased from
`examples/emittance_min/pipii_hwr_cm_match_4d_relaxed.dat`):

```
; ---------- Variables ----------
ADJUST FMAP 5 1 -1.98 -1.62 0.02 0     ; sol1 kb
ADJUST FMAP 6 2  0.248 0.372 0.005 0   ; cav1 ke
ADJUST FMAP 2 3 -60.0 -40.0 1.0 0      ; cav1 phase
; … sol2..4, cav2..4 the same pattern …

; ---------- Constraints ----------
MIN_EMIT_4D_GROWTH 1.0 1.05 1.20       ; ε_4D growth (5% slack), ε_z (20% slack)
SET_KE_OUT_MIN 5.0 10.0                ; exit kinetic energy ≥ 5 MeV
MIN_TRANSMISSION 99.9 50.0             ; transmission ≥ 99.9% (MP only)
END
```

---

## Cross-references

* [Overview](01_overview.md) — what the matcher solves.
* [Python API](03_python_api.md) — kwargs (`cost_solver`, `seqscan_*`, …)
  that control how the constraints are evaluated.
* [CLI](04_cli.md) — `python -m linac_gen.matching` flags.
* [Recipes](05_recipes.md) — worked examples per mode.
* [Emittance minimisation](07_emittance_min.md) — full HWR walkthrough.

← [Overview](01_overview.md) ·
[Continue to Python API →](03_python_api.md)
