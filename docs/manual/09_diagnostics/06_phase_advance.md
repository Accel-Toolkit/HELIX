# Phase advance & tune depression

HELIX reports **three distinct** phase-advance quantities.  Keeping them
separate is the whole point of this page — mixing them is the classic way
to get tune depression wrong.

| symbol | name | what it is | source |
|---|---|---|---|
| **σ₀** | bare channel tune | Floquet phase of the *bare* one-period map | `channel_phase_advance` (or `structure_phase_advance`) |
| **σ_model** | depressed channel tune | Floquet phase of the one-period map **with the frozen linearized space-charge field of a matched envelope** | `channel_phase_advance` / `channel_phase_advance_matched` |
| **Δμ_rms** | beam RMS phase advance | ∫ds/β of the *tracked beam moments* | `beam_phase_advance` |

The **primary tune-depression KPI is** `η_model = σ_model / σ₀_model` —
same cell, same mode, same branch, extracted from the *same* probe walk.
`Δμ_rms` is a beam-state-dependent secondary number (it is what TraceWin's
`kx/ky` columns report) and carries validity flags.

## The phase probe

σ₀_model and σ_model both come from the **envelope phase probe**.  Run the
envelope with `phase_probe=True` and it records, for every element, the
**exact tangent factors it actually applies** to Σ — the identical Strang
slices, midpoint-σ space-charge kicks, frequency-jump rescale `D`, edge
maps — in two parallel products:

* `element_maps_dep[j]` — with the space-charge factors present → the
  depressed one-period map `M_dep`;
* `element_maps_bare[j]` — the *same slices* with the SC factors set to
  identity → the consistent bare map `M_bare`.

Building σ₀ from a *separate* zero-current run would be wrong: a plain
`I=0` run uses whole-element FieldMap maps, not the SC walk's slicing, and
the two disagree at the fraction-of-a-degree level.  σ₀ must be the
`M_bare` of the *same walk*.

```python
from linac_gen.analysis.period_detect import detect_periods
from linac_gen.analysis.phase_advance import channel_phase_advance
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.tracking.envelope import EnvelopeSolver
# ... build `lattice` ...

ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
period = detect_periods(lattice)[0]

initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=1.0,
               alpha_y=0.0, beta_y=1.0, emit_y=1.0,
               alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
results = EnvelopeSolver(lattice, ref, initial, current=5.0,
                         phase_probe=True).run()

ch = channel_phase_advance(results, period)
print(ch["mu_x_bare_deg"], ch["mu_x_dep_deg"], ch["eta_x"])   # per cell
print(ch["eta_machine_x"])                                    # machine η
```

`channel_phase_advance` **requires** probe-bearing results — a plain
tracked run (or an MP/PIC run) raises a clear error, because there is no
recorded tangent map to extract from.  The helper
`run_phase_probe(lattice, ref, initial, current)` runs a probe-enabled
envelope for you (extra keyword arguments pass through to
`EnvelopeSolver`).

### Matched channel — the primary σ_model

On a plain probe run, σ_model still depends on how well *your* input beam
was matched.  For the authoritative depressed tune, use
`channel_phase_advance_matched`, which first solves the full-Σ matched
fixed point of the period (normalized-coordinate iteration with
eigenemittance renormalization — required because det(M₄_dep) < 1 under
acceleration damps a naïve covariance iteration toward a singular state)
and then probes with the matched Σ:

```python
from linac_gen.analysis.phase_advance import channel_phase_advance_matched
ch = channel_phase_advance_matched(lattice, ref, period,
                                   current=5.0, base_initial=initial)
print(ch["eta_machine_x"], ch["matched_state"])
```

When the longitudinal fixed point is not meaningful (strong acceleration
per cell), pass `longitudinal="frozen"`; the result is tagged
`sigma_model_approx="frozen-longitudinal"` and the GUI/CLI say so.

## The two curves, mathematically

The tune-depression popup (and the Matching-tab panel) plot two
different quantities.  Keeping their definitions straight is essential
for reading the plot — in particular for understanding why the beam
markers may legitimately sit **above 1** while the model curve cannot.

### Model curve — eigenphase ratio of the one-cell maps

For each cell the phase probe accumulates two one-cell matrices from
the *same* slice sequence and reference trajectory:

$$
M_{\text{dep}} \;=\; \prod_j M_j^{\text{slice}}\, K_j^{\text{SC}},
\qquad
M_{\text{bare}} \;=\; \prod_j M_j^{\text{slice}}\cdot \mathbb{1},
$$

where \(K_j^{\text{SC}}\) are the linearized space-charge kick matrices
(midpoint-σ Strang factors of the envelope's own field).  The phase
advance is the **eigenphase of the one-cell map**; for an uncoupled
2×2 block, with the det-normalization acceleration requires,

$$
\cos\mu \;=\; \frac{\operatorname{tr} M_2}{2\sqrt{\det M_2}},
\qquad
\det M_2 = \frac{(\beta\gamma)_{\text{in}}}{(\beta\gamma)_{\text{out}}},
$$

(coupled cells: eigenphases of the \(\det^{1/4}\)-normalized 4×4
block, modes I/II paired by symplectic eigenvector overlap).  Then

$$
\sigma_0 = \mu\!\left(M_{\text{bare}}\right),\qquad
\sigma_{\text{model}} = \mu\!\left(M_{\text{dep}}\right),\qquad
\eta_{\text{model}} = \frac{\sigma_{\text{model}}}{\sigma_0}.
$$

This is a **single-particle Floquet quantity** of the linear channel
(lattice + frozen linearized SC field); the beam enters only through
the field linearization.  Because the SC kick is defocusing,
\(\sigma_{\text{model}} < \sigma_0\) always — **η_model ≤ 1
structurally**.  A model value above 1 would be a bug, not physics.

### Beam curve — phase integral of the beam's own ellipse

$$
\Delta\mu_{\text{rms}} \;=\; \int_{s_a}^{s_b}
\frac{ds}{\beta_x^{\text{beam}}(s)},
\qquad
\beta_x^{\text{beam}}(s) = \frac{\Sigma_{11}(s)}{\varepsilon_x(s)},
\quad
\varepsilon_x = \sqrt{\Sigma_{11}\Sigma_{22} - \Sigma_{12}^{2}},
$$

— the same Courant–Snyder phase integral, but evaluated on the
**beam's own statistical β(s)** (the projected rms Twiss of the
propagated moment matrix Σ(s), whether from the envelope solver or a
tracked distribution) rather than on the matched channel β.  The
plotted markers are \(\Delta\mu_{\text{rms}}/\sigma_0\) per cell.
Geometrically, \(\Delta\mu_{\text{rms}}\) is the rotation angle of the
beam's rms ellipse in normalized phase space across the cell.

### The identity connecting them — and why the beam markers can exceed 1

Floquet's theorem: **if** the beam ellipse is matched to the depressed
channel — Σ(s) periodic, so
\(\beta^{\text{beam}}(s) = \beta^{\text{channel}}_{\text{dep}}(s)\) —
the one-period phase integral equals the eigenphase exactly:

$$
\int_{\text{cell}} \frac{ds}{\beta_{\text{matched}}(s)}
\;=\; \mu\!\left(M_{\text{dep}}\right) \;=\; \sigma_{\text{model}},
$$

and the two curves coincide.  For a **mismatched** beam,
\(\beta^{\text{beam}}(s)\) is not periodic: it oscillates about the
matched β at twice the betatron frequency.  Wherever the beam runs
*below* the matched β the integrand \(1/\beta\) is larger, so the
per-cell integral overshoots; a half mismatch-period later it
undershoots.  Beam markers **straddling 1 and alternating cell to
cell are a mismatch signature, not a focusing statement** — the gap
between the beam markers and the model curve is a mismatch meter.

Two further contaminations in coupled (solenoid) sections: (i)
\(\beta_x^{\text{beam}} = \Sigma_{11}/\varepsilon_x^{\text{proj}}\)
uses the *projected* emittance, inflated by the x–y cross terms, so
the integral is not a mode phase at all (the `projected_only` guard);
(ii) the denominator σ₀ is a mode eigenphase, so the ratio is only
apples-to-apples after mode pairing.  This is why the popup draws the
model series (modes I/II) and the beam series as visibly separate
curves instead of merging them into one "η".

**Reading the popup:** quote the model curve (η_model) as the tune
depression; read the beam markers as a match-quality diagnostic.  If
the beam markers matter to you (e.g. for a TraceWin `kx/ky`
comparison, which is the same beam quantity), first rematch the beam
into the section — the markers then collapse onto the model curve, up
to the projected-β caveat.

## Coupled sections (solenoid channels)

Inside a solenoid the transverse planes are coupled: the plane-wise 2×2
trace is meaningless.  `channel_phase_advance` detects coupling
(`coupled_xy=True`) and switches to **normal modes I/II**, extracted from
the det-normalized 4×4 map (`det(M₄)^{1/4}` normalization so adiabatic
damping is not mistaken for instability) and **paired** bare↔depressed and
cell↔cell by symplectic eigenvector overlap.  Machine totals are the sums
of the paired per-cell principal increments — never the aliased eigen-angle
of a whole multi-cell product.  Keys become `mu_{I,II}_{bare,dep}_deg`,
`eta_{I,II}`.

> **Projected β is not a normal-mode tune.** Integrating ∫ds/β on the
> projected x/y β *inside* a coupled solenoid interior overshoots (measured
> +16–20 % through the PIP-II HWR).  `beam_phase_advance` sets
> `projected_only=True` and downgrades its authority when it sees coupling;
> trust the channel modes there.

## Beam RMS phase advance — `beam_phase_advance`

`beam_phase_advance(results, period, sigma0=...)` integrates ∫ds/β from the
tracked beam moments over the period.  It is TraceWin-comparable but
beam-state dependent, and returns validity flags rather than a bare number:

* `mu_x_deg` / `mu_y_deg` / `mu_z_deg`;
* `matched` (bool) with `mismatch_x/y/z` and covariance-based
  `mismatch_bmag_x/y/z` (BMAG) — acceleration/frequency scaling is *not*
  counted as mismatch;
* `projected_only` — the coupled-interior guard above;
* `n_samples` / `samples_per_period` / `resolution_ok` — sampling
  adequacy (≥ 20 samples per 2π);
* `n_skipped_x/y` / `n_total` — invalid-β samples split the integral
  rather than being bridged (bridging fabricates phase across a gap);
* `sigma_over_sigma0_{x,y}` when `sigma0=` is supplied (uses the
  branch-consistent `mu_*_branch_deg` denominator).

`μ_z` uses the **effective** (z, z′) β, converted from the native β_φW via
the recorder Jacobian and the 1/γ² longitudinal drift factor — not the
deg/MeV β, which is off by ~685× at PIP-II injection if used directly.

## Branch conventions

Phase can be quoted three ways; HELIX exposes all three explicitly so η is
never the ratio of two different conventions:

* `mu_*_principal_deg` — folded to [0, 180°] per cell;
* `mu_*_oriented_deg` — signed, mod 360°;
* unwrapped machine totals — the sum of per-cell increments (what η uses).

The CLI `twiss` command prints the principal value, adding the oriented
value in parentheses when they differ; `-q` output is unchanged.

## New space-charge diagnostics

Two diagnostics build directly on the channel tunes:

* **[Hofmann stability analysis](07_hofmann_footprint.md#hofmann-stability-analysis-corrected-anisotropic-solver)** —
  per-cell growth rates from the corrected anisotropic l=2/3/4 KV
  dispersion relations, with the trajectory overlaid on the solved
  growth-rate chart.
* **[Tune footprint](07_hofmann_footprint.md#frozen-sc-tune-footprint)** —
  the incoherent per-particle tune spread under a frozen space-charge
  field.

## Period detection

```python
from linac_gen.analysis.period_detect import detect_periods
for p in detect_periods(lattice):
    print(p.label, p.start, p.end, p.n_repeats)
    print(p.spans())        # explicit (start, end) raw-index pair per repeat
```

`detect_periods()` returns `PeriodicStructure` dataclasses carrying
`start` / `end` (element indices, end-exclusive), `inner_period_length`,
`inner_slice_end` (index just past the first repeat), `n_repeats`, `label`,
`source`, and **`repeat_spans`** — accessed via `p.spans()`, the explicit
contiguous span of each repeat.  Always iterate cells with `p.spans()`
(not a constant raw stride): when skipped commands/markers are unevenly
distributed, a fixed stride lands on the wrong boundaries.

## GUI

The **Results** tab carries the **Phase advance σ₀ · σ** popup (channel
tunes + along-s curves, coupled → eigenmodes), the **tune-depression**
popup (η_model per cell as the primary series, Δμ_rms as markers), plus the
new **Hofmann stability chart** and **Tune footprint** popups.  The
**Matching** tab shows a σ₀ / σ_model + η_model / Δμ_rms panel with the
validity tags.

**Multi-particle results** carry no probe maps (the PIC field has no
recorded tangent map), so the model curves cannot be extracted from
them directly — the beam Δμ_rms markers still work.  The
tune-depression popup then offers a **Compute channel model** button: it
runs a *companion envelope probe* in the background with the current
Beam-tab configuration and plots σ₀/σ_model/η_model alongside the MP
beam markers, labelled with their provenance.  The probe result is
cached and shared, so the Hofmann chart fills from the same run; the
cache invalidates automatically when the lattice or any beam-config
field changes.

## Cross-references

* [Hofmann chart & tune footprint](07_hofmann_footprint.md)
* [Tracking modes](../02_concepts/03_tracking_modes.md)
* [Matching → recipes](../07_matching/05_recipes.md) — periodic Twiss matching.
* `linac_gen/analysis/phase_advance.py:1`
* `linac_gen/analysis/period_detect.py:37` — `PeriodicStructure`.

← [H⁻ stripping](05_stripping.md) ·
[Hofmann chart & tune footprint →](07_hofmann_footprint.md)
