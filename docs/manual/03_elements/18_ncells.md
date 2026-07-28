# NCells

A **multi-cell RF cavity** — TraceWin's `NCELLS` ("Cavity multi-gap") card.
Models a superconducting elliptical / multi-gap cavity (e.g. the PIP-II
HB650 cavities) exactly as TraceWin does: as **a set of thin RF gaps, one at
the middle of each cell**, expanded and tracked at run time.

## TL;DR

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `NCELLS` | `NCells(...)` |
| Model | one thin gap per cell | same — reuses the [RFGap](06_rfgap.md) kick per cell |
| Cell length | βg·λ (2π mode) / βg·λ/2 (π mode) | same |
| Gap field | `EoT` (V/m); per-cell V = EoT·L_cell | same |
| Phase | `θs` + P flag (0 rel / 1 abs / 2) | resolved at track time from `ref.phi_s` |

Conventions:

* Not a field map — it is a closed-form multi-gap model, so no `.edz`/`.bsz`
  file is needed.  The cavity is realised internally as drift → gap → drift
  per cell, giving space charge *between* cells for free.
* **Phase.** `θs` is the RF phase at the **first gap**.  `P=0` relative (θs is
  the synchronous phase at gap 1), `P=1` absolute (θs is referenced to the
  running RF clock — it grows down the linac to hold a constant effective
  phase), `P=2` relative with the beam phase zeroed at the **first gap**
  (implemented identically to `P=0`; no example deck exercises `P=2`, so
  it is not validated against TraceWin).  A preceding
  `SET_SYNC_PHASE` makes θs the synchronous phase (iterative calibration, as
  for `FIELD_MAP`).
* **Charge sign.** Energy gain uses the **`|q|`** (abs-charge) convention
  TraceWin (and `RfqCell`) use — `dW = |q| · V · T · cos(φ)` — so the deck's θs
  maps directly and H⁻ and protons accelerate alike (θ ≈ crest = 0° for `|q|`).
  For `P=1` the seen phase is `φ = ref.phi_s − θs`.  Both choices are validated
  against a TraceWin run (see below).
* **Geometric β.** `βg > 0`: all cells share a length set by βg (the common
  case).  `βg = 0`: cell length follows the running beam velocity, resolved
  lazily when the beam arrives.  `βg < 0`: seeded at `|βg|` and ramped
  cell-to-cell.

## Modelling notes

Each cell is one gap; the π-mode field polarity alternates cell to cell
(`σ_k = (−1)^{k−1}`), and the ~180°-per-cell running-clock advance cancels that
alternation so every cell accelerates coherently.  The interior is exact
(drift + thin kick with β piecewise-constant between gaps), so `n_steps` sets
only the space-charge-kick cadence, not accuracy.  All four tracking modes are
supported — multi-particle, envelope, matrix, and (reference-only) backtrack —
and MP `track_rk4` agrees with envelope `advance_ref` on the reference energy
to machine precision.

!!! success "Validated against TraceWin"
    The `P=1` absolute-phase convention (seen phase `ref.phi_s − θs`, `|q|`
    energy gain) is validated end-to-end against a TraceWin ENV+SC / MP+SC
    run of the PIP-II HB650 deck (`examples/piplattice/`): the beam
    accelerates 116.1 → **404.803 MeV — matching TraceWin exactly** — and
    every observable (σ_x, σ_y, σ_φ, σ_W, energy) tracks the reference at
    0.98–1.00 correlation in both envelope and multi-particle modes.
    (Reaching this also required the `FREQ` machine-clock semantics and the
    physical H⁻ ion mass — see the example README.)  The φ **sign** matters
    beyond the energy — `sin φ` sets the RF defocus, so `phi_s − θs` (not
    `θs − phi_s`) is the physical one.

!!! tip "Results-tab charts"
    NCELLS cavities appear in the Results tab's lattice-parameter charts:
    **RF voltage (V₀)** shows the total effective cavity voltage
    Σ|EoT·L_cell| per gap (including any ERROR_CAV amplitude factor),
    **Peak E_acc** shows EoT, and **Synchronous phase** shows θs for
    relative / `SET_SYNC_PHASE` cavities.  For **P=1 absolute-phase**
    cavities the raw θs is just the RF-clock setting (a meaningless ramp),
    so the φ_s chart plots the **run-resolved phase the beam saw at
    gap 1** — populated after any tracking run, blank before.

!!! success "Geometric synchronism factor (2026-07) — mode-faithful"
    TraceWin's **matrix/envelope engine** applies a β-dependent cell
    transit-time (synchronism) factor internally even when the card
    carries no βs tail — reverse-engineered, zero free parameters,
    from TraceWin's own per-element transfer-matrix export of
    `fnalscl.dat`: the uniform-field π-mode form **T(β) = sin(u)/u,
    u = (π/2)·βg/β** (classic Panofsky/DTL; the half-sine profile
    closes only half the gap).  Its W-derivative enters the gap matrix
    as asymmetric diagonals (M₆₆ = 1+δ, M₅₅ = 1/(1+δ), determinant
    exactly 1), with the card's `EoT` meaning untouched (synchronous
    energy exact).  With this, **all four longitudinal matrix elements
    match TraceWin's export to ≤0.04% per cavity** (previously up to
    1.8% off), and the envelope-mode σ_W end-of-line offset vs
    TraceWin closes from −3.2% to +0.1%.

    Crucially, this is applied **only on the matrix/envelope path**:
    three MP runs against TraceWin's partran reference showed TW's
    *particle tracker* does NOT carry the factor (stock thin-gap
    tracking deviates 1.4%/4.1% below/above 30 m; adding the factor
    made it worse, up to 33%).  HELIX therefore mirrors TraceWin's own
    two-engine architecture: MP tracking = stock thin-gap kick
    (partran-faithful), matrices = synchronism (matrix-engine-
    faithful).  Guard test:
    `tests/analysis/test_tracewin_crosscheck.py` (gated on the
    never-committed TW export); escape hatch: `cavity.synchronism =
    False`; applied only for `βg > 0`.

    Two scope limits (adversarial review, 2026-07): (1) the factor is
    perturbative (δ ~ 10⁻³/gap on a matched deck) — if the beam is
    grossly off the cavity's geometric β (β → βg/2, where T̂ → 0 and
    the log-derivative diverges), δ is clamped at |δ| > 0.1: the
    cavity falls back to the stock thin-gap matrix and warns once.
    (2) The envelope engine pushes the beam **centroid** through the
    same matrix, so with a nonzero longitudinal centroid offset the
    envelope centroid contracts ~δ per gap relative to MP tracking —
    the same two-engine split as the σ's; only second moments are
    validated against the TW export.

!!! note "βs≠0 transit-time tail — reconstructed"
    The optional `βs Ts kT's k²T''s …` tail applies a Wangler `T(β)` transit-
    time correction about the reference velocity βs.  The manual's `T(β)`
    figure is unavailable, so this path is **reconstructed from the standard
    Wangler expansion** and is not validated against a TraceWin numeric
    reference.  It is inert whenever `βs = 0` (the common case).

## API reference

```{.python .skip}
linac_gen.elements.ncells.NCells(
    name: str, *,
    mode: int,               # 0 = 2π, 1 = π, 2 = π&2π
    n_cells: int,
    beta_g: float,           # geometric β (>0 fixed / =0 running / <0 ramped)
    eot_v_per_m: float,      # effective gap field EoT (V/m)
    theta_s_deg: float,      # RF phase at the first gap (deg)
    aperture_mm: float = 0.0,
    p_flag: int = 0,         # 0 relative / 1 absolute / 2 relative+zeroed
    k_eot_i: float = 0.0,    # input-cell field correction  (1+k)
    k_eot_o: float = 0.0,    # output-cell field correction (1+k)
    dz_i_mm: float = 0.0,    # first-gap displacement
    dz_o_mm: float = 0.0,    # last-gap displacement
    frequency_mhz: float = 0.0,   # from the lattice FREQ card
    sync_phase: bool = False,     # a preceding SET_SYNC_PHASE
    ttf=None,                # optional βs≠0 TTF tail (parse_ttf_tail)
    n_steps: int | None = None,   # default 2·n_cells (SC cadence)
)
```

### Source

* `linac_gen/elements/ncells.py` (element, geometry, phase resolution)
* `linac_gen/io/tracewin_parser.py` (the `NCELLS` branch)
* `linac_gen/io/tracewin_writer.py` (round-trip)

## See also

* [RFGap](06_rfgap.md) — the single-gap kick reused per cell.
* [VaneRFQ](10_vanerfq.md) — the other multi-cell-in-one-element cavity.
* [Migrating from TraceWin](../appendices/E_migrating_from_tw.md) — card mapping.

← [VaneRFQ](10_vanerfq.md) ·
[Continue to SuperposedFieldMap →](19_superposedfieldmap.md)
