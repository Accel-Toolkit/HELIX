# Robust emittance minimisation

A worked recipe for HELIX's two newest matcher pieces:

* **`MIN_EMIT_GROWTH plane weight`** — a one-sided residual that
  penalises end-of-line growth in one plane and ignores drops below the
  seed beam's emittance.  Critical for the longitudinal plane, where
  coupling-resonance crossings can drop ε_z below input — recovery up
  to ε_z,in is free, only true growth is penalised.
* **`MIN_EMIT_4D_GROWTH weight tol_4d tol_z`** — emittance-exchange-aware
  variant.  Combines the *4-D coupled transverse* normalised emittance
  `ε_4D · (βγ)²` (computed from `sqrt(det(Σ_4×4_transverse))`) with the
  normalised longitudinal `ε_z · (βγ)`, each clipped by a multiplicative
  tolerance band.  Use when the lattice physically exchanges phase-space
  area between planes (e.g. solenoid-rotated transport, coupling
  resonances near sync-phase crossings): the strict per-axis
  `MIN_EMIT_GROWTH X|Y|Z` constraint would force the matcher onto a
  structural cost floor it cannot escape.  See [§ Emittance-exchange
  tolerance below](#emittance-exchange-tolerance-min_emit_4d_growth) for
  the recipe.
* **`SET_KE_OUT_MIN energy_MeV weight`** — a one-sided floor on the
  reference particle's exit kinetic energy.  Prevents the matcher from
  silently buying lower emittance by detuning a cavity off-crest.
* **`--algorithm cmaes`** — Covariance Matrix Adaptation Evolution
  Strategy (Hansen).  Robust global, gradient-free, bound-respecting.
  The right pick when the constraint family has flat regions that
  defeat scipy LM.

## When to use this recipe

You have a lattice that grows transverse and/or longitudinal emittance
under nominal settings, and you want to tune **solenoid B-fields**,
**cavity gradients (`ke`)** and **cavity synchronous phases** to drive
the growth back down, subject to:

* Per-element bounds (cryomodule spec sheets) — already supported by
  the existing `ADJUST … vmin vmax` syntax.
* An end-of-line energy floor — supplied by `SET_KE_OUT_MIN`.
* "No penalty for ε_z drops, only ε_z growth" — supplied by
  `MIN_EMIT_GROWTH Z`.

## Lattice syntax

```text
ADJUST <family> <param_idx> <link_group> <vmin> <vmax> <start_step> 0
…
MIN_EMIT_GROWTH X 1.0
MIN_EMIT_GROWTH Y 1.0
MIN_EMIT_GROWTH Z 1.0
SET_KE_OUT_MIN 3.5 10.0      ; W_kin,out ≥ 3.5 MeV, weight 10
END
```

Parameter indices used for the tunables in this workflow
(see [`02_set_adjust.md`](02_set_adjust.md) for the full table):

| Element | param_idx | Attribute | Unit |
|---|---|---|---|
| `Solenoid` | 2 | `field` | T |
| `FieldMap` / `FieldMap3D` | 2 | `phase` | deg |
| `FieldMap` / `FieldMap3D` | 5 | `kb` (magnetic scale) | dimensionless |
| `FieldMap` / `FieldMap3D` | 6 | `ke` (electric scale) | dimensionless |
| `RFGap` (a.k.a. `GAP`) | 1 | `voltage` | MV |
| `RFGap` (a.k.a. `GAP`) | 2 | `phase` | deg |

> The `<family>` in an `ADJUST` card matches the *prefix* of the
> element name, **not** the .dat keyword: e.g. `ADJUST SOL …` for
> `Solenoid` elements (named `SOL_001`, `SOL_002`, …) and `ADJUST
> FMAP …` for `FieldMap`/`FieldMap3D` (named `FMAP_001`, `FMAP_002`,
> …).  `RFGap` parsed from `GAP` cards is named `GAP_001` etc., so
> `ADJUST GAP …` is correct.

## One-sided emittance residuals

The X and Y planes are compared in **normalised emittance** so
adiabatic damping under acceleration does not trivially satisfy the
constraint:

```
r_x = max(0, ε_n,out − ε_n,in)            where ε_n = ε_geom · β γ
r_y = max(0, ε_n,out − ε_n,in)
```

The Z plane uses the **native** `emit_z` (deg·MeV); that unit pair
already counts the conjugate longitudinal action and is invariant
under acceleration:

```
r_z = max(0, ε_z,out − ε_z,in)
```

A drop in any plane contributes zero residual.  Recovery up to the
seed value is also free.

## End-of-line energy floor

```
r_E = w_E · max(0, E_floor − W_kin,out)
```

Pick the weight `w_E` roughly **10× the per-plane emittance weights**;
the energy residual scales in MeV which is typically much larger than
the per-plane Δε in mm·mrad, so a smaller weight would let the matcher
trade away significant energy for minor emittance gains.

## Running the matcher

```bash
python -m linac_gen.matching <lattice.dat> \
    --algorithm cmaes \
    --space-charge --current 5.0 \
    --energy 2.5 --species proton --frequency 162.5 \
    --alpha-x -1.2 --beta-x 0.32 --emit-x 0.30 \
    --alpha-y  2.0 --beta-y 0.05 --emit-y 0.30 \
    --emit-z 0.4 --beta-z 600.0 \
    --max-iter 80 --cmaes-sigma 0.25 \
    --report
```

The default `refine=True` chains a `scipy.optimize.least_squares`
polish from the best CMA-ES point.  Pass `--no-refine` to skip that
polish (useful if the polish moves the solution out of an active-bound
basin).

## Worked example

`examples/emittance_min/min_test.dat` ships as the canonical demo: 8
elements (drift–sol–drift–buncher–drift–sol–drift–accel–drift), 6
tunable knobs, 4 constraints, proton at 2.5 MeV with deliberate
transverse mismatch.

| Quantity | Baseline | After match | Change |
|---|---:|---:|---:|
| ε_nx growth (mm·mrad) | +0.15 | +0.045 | **−70 %** |
| ε_ny growth | +0.15 | +0.045 | **−70 %** |
| ε_z growth | 0.00 | 0.00 | flat |
| W_kin,out (MeV) | 3.93 | ≥3.5 | floor respected |
| Wall time | — | ~1.3 s | 783 evals |

See `examples/emittance_min/README.md` for the exact invocation and
expected output.  A PIP-II HWR cryomodule slice with 12 tunable knobs
ships alongside as `pipii_hwr_cm_match.dat`.

## Algorithm notes

* CMA-ES population defaults to `4 + ⌊3·ln(N)⌋` per Hansen, where N is
  the number of decision variables.  Override with `--cmaes-popsize`.
  Larger populations are more robust against local minima at the cost
  of more forward passes.
* The initial step size `--cmaes-sigma` (default `0.2`) is in units of
  bound-box width.  Larger values explore more aggressively; pick
  `~0.3` if the seed is far from a good point, `~0.1` to refine a
  near-matched lattice.
* Coupling-resonance landscapes are genuinely multimodal — there is
  no guarantee CMA-ES finds the *global* optimum.  When budget allows,
  run 3–5 different seeds (vary the RNG seed inside `cma` or perturb
  the initial values) and keep the best.

## Sequential-scan algorithm (`--algorithm sequential_scan`) { #sequential-scan }

For users who prefer the classical accelerator-physics matching recipe —
coordinate descent with direction reversal + physics-aware ordering —
HELIX exposes a `sequential_scan` algorithm in the matcher.  Unlike the
other algorithms which vary all variables simultaneously, this walks
each tunable element in lattice order and brackets each ADJUST'd
parameter in a one-at-a-time sweep:

```text
for pass in range(N_PASSES):
    for element in lattice_order(selected):
        for group in [(kb,), (ke,), (phase,)]:            # FieldMap
            direction = +1
            for step in range(N_STEPS):
                x_new = x_current + direction × step_frac × (vmax − vmin)
                cost = evaluate(x_new)
                if εnx_norm_out > seed AND εnz_norm_out > seed:
                    direction *= −1               # reverse on dual growth
                track_best(x_new, cost)
                x_current = x_new
```

Key properties:

* **Element selection** — runs over a user-specified subset of elements
  (default: all elements with at least one ADJUST card).
* **Physics ordering** — for FieldMap elements: scan `kb` (solenoid
  focusing) first, then `ke` (cavity amplitude), then `phase` (sync
  phase).  Other element classes use the order of their ADJUST cards.
* **Direction reversal** — defaults to "reverse only when *both*
  transverse and longitudinal normalised emittance exceed the reference"
  (`--seqscan-reversal both_grew`); that is, accept emittance exchange
  between planes as physics (matches what `MIN_EMIT_4D_GROWTH` already
  does in the cost function).  Stricter `--seqscan-reversal any_grew`
  reverses on either plane growing.
* **Reversal reference** — `--seqscan-threshold input` (default)
  compares trial exit ε to the *beam input* ε.  Tight criterion;
  designed for the coupling-resonance case (ε_t and ε_z trade off but
  reversal fires only if both exceed input simultaneously).
  `--seqscan-threshold seed_exit` compares to the *nominal/unmatched
  lattice's exit* ε instead; reversal fires only when the trial is
  genuinely WORSE than baseline.  Natural for ε minimisation in
  lattices with intrinsic growth (most RF cryomodule chains).
* **Best-x tracking + Stop button** — share the same machinery as the
  other algorithms; cancellation always leaves the lattice at the
  best-cost x seen so far.

GUI: pick **Algorithm = sequential_scan** in the Matching tab, click
Match, fill in the Setup dialog (which elements, how many passes /
steps, step size as a fraction of bound width, reversal criterion,
reversal reference), click Start Scan.  Live convergence plot updates
per evaluation; Stop button works as for the other algorithms.

CLI — every `seqscan_*` kwarg is exposed:

```bash
python -m linac_gen.matching examples/emittance_min/pipii_hwr_cm_match_4d_relaxed.dat \
    --algorithm sequential_scan \
    --seqscan-passes 2 \
    --seqscan-steps 7 \
    --seqscan-step-frac 0.10 \
    --seqscan-reversal both_grew \
    --seqscan-threshold seed_exit \
    --seqscan-element FMAP_001 --seqscan-element FMAP_002 \
    --report
```

Repeat `--seqscan-element NAME` to restrict the walk to a subset of
ADJUST'd elements; omit it to scan all of them.

When to pick `sequential_scan` over `cmaes`:

* You want full transparency into which knob did what -- per-element
  scans are easier to interpret than CMA-ES population samples.
* The lattice has weak inter-knob coupling and a small number of
  variables (≤ 10) -- coordinate descent converges in fewer total
  evaluations.
* You're validating a hand-tuned baseline -- sequential_scan with
  small `step_frac` (e.g. 0.02) refines a near-matched lattice
  conservatively.

When to stay with `cmaes`:

* The problem is strongly coupled (12+ variables, HWR-style) -- CMA-ES
  learns the covariance matrix and moves diagonally; coordinate descent
  zigzags.
* You want a single number that captures algorithm confidence -- CMA-ES
  provides one via the population spread; sequential scan does not.

## Emittance-exchange tolerance (`MIN_EMIT_4D_GROWTH`)

For lattices that naturally trade phase-space area between transverse
and longitudinal planes — solenoid-focused cryomodule sections,
coupling-resonance crossings near a sync-phase target — the strict
per-axis `MIN_EMIT_GROWTH X|Y|Z 1.0` triple over-constrains the
matcher.  Even after CMA-ES + LS polish the cost plateaus at a
non-zero structural floor because no setting of the tunable knobs can
drive *all three* axes below their entry emittance simultaneously.

`MIN_EMIT_4D_GROWTH` lets you encode the physics-aware version of
"don't grow total phase-space volume":

```text
MIN_EMIT_4D_GROWTH 1.0 1.05 1.20
;                  |   |    |
;                  |   |    +-- tol_z:  allow 20 % normalised εz growth
;                  |   +------- tol_4d: allow 5 % 4-D transverse growth
;                  +----------- weight: same semantics as MIN_EMIT_GROWTH
```

The two residuals are clipped one-sided:

```text
r_4d = max(0,  ε_4D·(βγ)²|_out  −  tol_4d · ε_4D·(βγ)²|_in)
r_z  = max(0,  ε_z·(βγ) |_out   −  tol_z  · ε_z·(βγ) |_in)
```

Key properties:

* **εx ↔ εy exchange is free.**  ε_4D is `sqrt(det(Σ_4×4_transverse))`
  — invariant under any orthogonal rotation of the transverse phase
  space (solenoid coupling, skew-quad twist).  The matcher can shuffle
  area between x and y without paying any cost.
* **Transverse–longitudinal trade-off is allowed up to the tolerances.**
  A drop in εx at the price of εz growth is fine as long as the εz
  growth stays under `tol_z`.
* **Tolerances are multiplicative bounds** — `1.0` is strict (any
  growth penalised); `1.1` allows 10 % growth before the residual
  becomes positive.  Negative or sub-1.0 values raise at parse time.
* **Drops are free** (one-sided clip), same as `MIN_EMIT_GROWTH`.

Use one `MIN_EMIT_4D_GROWTH` *or* three `MIN_EMIT_GROWTH X|Y|Z` lines,
not both — they over-constrain each other.  Choose the strict form
when each plane has a hard independent target (collimator, foil
stripping section); choose the 4-D form when emittance exchange is
expected.

Working example: `examples/emittance_min/pipii_hwr_cm_match_4d.dat`
(strict) and `pipii_hwr_cm_match_4d_relaxed.dat` (5 % / 20 %
tolerance) — both target the PIP-II HWR cryomodule with identical
ADJUST cards and beam config as the original `pipii_hwr_cm_match.dat`.

## Caveats

* `MIN_EMIT_GROWTH` and `SET_KE_OUT_MIN` are **soft** weighted
  residuals — the matcher minimises sum-of-squares.  If the weight on
  `SET_KE_OUT_MIN` is too small, the matcher can buy emittance at the
  cost of a few MeV of beam energy.  Pick the weight deliberately.
* CMA-ES bounds are penalty-based; transient out-of-spec excursions
  during exploration are normal and the final point is always inside
  the box.
* The Z-plane evaluator does not apply a `β γ` correction — `emit_z`
  in deg·MeV is already an action variable.  Mixing units between
  planes is intentional.

## Pre-equipped PIP-II lattices (`examples/pipii_tunable/`)

Ready-to-match copies of the canonical `examples/pipii/` lattices, with an
`ADJUST` card already on **every active element** and an end-of-line
`MIN_EMIT_4D_GROWTH` + `SET_KE_OUT_MIN` objective — no hand-editing needed.
Each deck also carries a commented `;MIN_TRANSMISSION 99.9 50` line —
uncomment it when matching with the `mp` cost solver so the optimiser
cannot "improve" emittance by scraping halo (the engine refuses the
active card under the envelope cost solver, where it would be inert).

| Lattice | knobs (cav / sol / quad) |
|---|---|
| `btl_tunable.dat`, `btl_with_foil_tunable.dat` | 53 (0 / 0 / 53) |
| `mebt_tunable.dat` | 39 (4 / 0 / 31) |
| `mebt+hwr_tunable.dat` | 63 (12 / 8 / 31) |
| `mebt+hwr+ssr1_tunable.dat` | 103 (28 / 16 / 31) |
| `mebt+hwr+ssr1+ssr2_tunable.dat` | 194 (63 / 37 / 31) |
| `mebt+hwr+ssr1+ssr2+lb650+hb650_tunable.dat` | 350 (123 / 37 / 67) |

Bounds: cavity `ke` ±20 %, sync phase ±10°, solenoid `kb` ±10 %, quad
gradient ±20 %. Steerers are excluded (orbit, not emittance); BTL dipoles
stay fixed (geometry). Regenerate or re-tune bounds with
`examples/pipii_tunable/make_tunable.py` (bounds are CLI flags).  Three
policy flags refine the knob set: `--tune-from-s <metres>` restricts
knobs to elements downstream of a given s; `--fix-phase-maps` gives the
matching cavity families a `ke` knob but no phase knob;
`--cap-family-max` caps each cavity/solenoid's strong-side bound at its
field-map family's max design |value| (hardware ceiling).
`examples/pipii_hwr_ssr1_match/` combines the latter two — a 99-knob
MEBT+HWR+SSR1 match with the QWR buncher phases pinned at −90° and
family-max `ke`/`kb` limits — a cheaper section-sized match than the
full linac.

!!! warning "Equip → perturb → match"
    Two conditions must hold for the matcher to actually *do* something here:

    1. **Run with space charge** (`--space-charge`, or the `mp` cost solver).
       Under linear optics with no space charge, RMS emittance is *exactly
       conserved* — so `MIN_EMIT_4D_GROWTH` is trivially satisfied and the
       match is a no-op.
    2. **Start from a degraded state, not the design.** These are the PIP-II
       *design* lattices — they already grow emittance < 1 % end-to-end, so a
       match started from design correctly changes nothing. Perturb first (an
       off-design input beam, applied errors, or a failed element via the
       [Failure Study tab](../10_gui/06c_failures_tab.md)), then match.

## Cross-references

* [SET / ADJUST card syntax](02_set_adjust.md)
* [Matching CLI](04_cli.md) — full `python -m linac_gen.matching`
  flag list.
* [Recipes](05_recipes.md) — Twiss-matching and gradient-matching
  examples.
* `examples/emittance_min/` — runnable demos.
* `examples/pipii_tunable/` — fully-equipped PIP-II lattices (above).

← [Gradient algorithm](06_gradient_algorithm.md) ·
[Continue to Multi-objective matching →](08_multiobjective.md)
