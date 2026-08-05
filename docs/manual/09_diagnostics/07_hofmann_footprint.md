# Hofmann stability & tune footprint

Two space-charge diagnostics that build on the [channel
tunes](06_phase_advance.md).  Both are honest about what is exact and what
is indicative.

## Hofmann stability analysis (corrected anisotropic solver)

HELIX solves the coherent-mode KV dispersion relations of I. Hofmann,
*Phys. Rev. E* **57**, 4713 (1998) for the non-oscillatory `l = 2`,
`3 (even/odd)`, and `4 (even)` branches in their **full anisotropic
forms**, and evaluates them per lattice-period cell directly from the
phase-probe channel model — no external code or chart digitization
involved.

The implementation is a verbatim physics port of the corrected solver from
the PRAB manuscript *"Higher-order anisotropic Hofmann stability charts
with probabilistic margins and a machine-learning surrogate, applied to
PIP-II"* (A. Pathak), which fixes three defects relative to the printed
1998 equations:

1. **Coordinate map** (Eq. (24) + p. 4716): once the plane with the larger
   envelope is Hofmann's "x", *both* `h = a/b` and `α = ν_y/ν_x` follow
   from that same assignment, restoring the emittance identity
   `ε_x/ε_y = h²/α`.
2. **Odd-branch interchange** (p. 4719): swapping `ν_x ↔ ν_y` rescales the
   `ν_x`-normalised quantities too —
   `D_odd(s, α, h, S²) = D_even(s/α², 1/α, 1/h, S²/α²)`.
3. **Eq. (41) sign**: the `S⁴` block enters with `+S⁴`, as required by the
   isotropic-limit reduction to Eq. (42).

### Chart coordinates from a HELIX run

Every input comes from one probe-bearing envelope run
(`EnvelopeSolver(..., phase_probe=True)`), per cell:

* `R = ν_z/ν_x` — ratio of the **depressed** channel tunes
  (`mu_z_dep/mu_x_dep`, identically `(k_z0·η_z)/(k_x0·η_x)`);
* `Y = η_x` — transverse tune depression;
* `ε_z/ε_x` — **geometric** emittance ratio, `emit_z_mmmrad/emit_x`
  sampled at each repeat-span entry (clock-safe across `FREQ` jumps).

```python
from linac_gen.analysis.hofmann_stability import (
    hofmann_stability, anisotropy_margin)
from linac_gen.analysis.hofmann_probabilistic import instability_probability

tab = hofmann_stability(results, period)     # probe-bearing results
print(tab["g_combined"], tab["flagged"])     # per-cell γ/ν_0x + flags
mg = anisotropy_margin(results, period)      # distance-to-onset in ε_z/ε_x
p = instability_probability(tab, N_mc=200)   # Monte-Carlo jitter layer
```

`hofmann_stability` returns, per cell: the per-branch growth rates
`g_l2 / g_l3_even / g_l3_odd / g_l4_even` (all `γ/ν_0x`; the `l=4` odd
branch — anisotropic `S²` block, isotropic-limit `S⁴` block — is off by
default), the combined maximum, the space-charge parameter `S²`, and three
verdicts:

* **`valid`** — `S² ≤ 10`, the manuscript's perturbative-validity gate;
  growth rates outside it are extrapolations (`flagged_extrap`), never
  design evidence.
* **`flagged`** — `γ/ν_0x > 0.01` *inside* the gate.
* **`fold_risk`** — the bare per-cell advance approaches the 180°
  principal-value fold of `mu_folded`; a true advance above 180° would be
  silently reflected, corrupting `R` and `Y`, so treat amber cells with
  suspicion.

**Guards.** The analysis refuses — with a `reason` string, never an
exception — on x–y coupled lattices (solenoid channels: the channel tunes
are normal modes I/II, not the x/z planes of Hofmann's relations) and on
DC/continuous beams (no longitudinal tune).  Cells with `η_x ≥ 1` (no net
depression) are kept but warned about; the solver treats them as trivially
stable.

`anisotropy_margin` holds each cell's `(R, Y)` fixed and raises
`ε_z/ε_x` (1 → 9, inside the `S²` domain) until a higher-order (`l=3`,
`l=4` even) channel first exceeds the threshold — the design question
"how much more anisotropy could this period absorb".  Onsets caused by the
`η̂ = 1` coordinate-flip seam (S² jumping > 2× in one step) are flagged
`is_seam` and excluded from the smooth-margin summary.

`instability_probability` draws `N_mc` perturbations per cell from an
engineering jitter budget (current 3 %, mismatch 10 %, ε-ratio 5 %, tunes
2 % — the manuscript's defaults, overridable via `PerturbationBudget`) and
returns `P(γ/ν_0x > threshold)`.  The full probability *grid* and the
bootstrap contour bands (`ProbabilisticChart` in
`linac_gen.analysis.hofmann_probabilistic`) are headless-only tools —
the bootstrap is hours-class serial; use `n_workers` to farm it.

**Cost** (measured, Apple-silicon laptop): ~31 ms per cell for the three
branch families; a full `chart()` at the GUI default `steps=100` ≈ 4 s and
at `chart()`'s own default `steps=200` ≈ 11 s; probability layer
≈ 0.2 s/cell at `N_mc=200`.

**Trust anchors.** `tests/analysis/test_hofmann_dispersion.py` pins the
solver against Hofmann's printed isotropic forms Eqs. (37)/(42)/(46)
(transcribed independently), the off-isotropic Eq. (24)/Eq. (5)/p. 4719
identities the iso-limit cannot see, and 20 golden chart points that are
bitwise-identical to the source manuscript package.

### Trajectory overlay & legacy bands

`linac_gen.analysis.hofmann` still provides the exact per-cell trajectory
`(k_z/k_x, k_x/k_0x)` (`hofmann_trajectory`; conventions as in Hofmann,
Franchetti, Boine-Frankenheim, Qiang & Ryne, *PRST-AB* **6**, 024202,
2003) and the old `resonance_bands()` overlay, whose `w ∝ (1 − depression)`
band widths are a **qualitative heuristic only** — superseded by the
solved growth-rate chart above, and kept as an optional overlay for
orientation at the rational ratios `k_z/k_x = m/n`.

## Frozen-SC tune footprint

`linac_gen.analysis.footprint.tune_footprint` measures the **incoherent**
per-particle tune spread of one period cell.

A *linear* monodromy gives every particle the same eigenfrequency — no
footprint.  So HELIX tracks a sparse amplitude ladder repeatedly through
the cell with the **real nonlinear element transport**, while the
space-charge field is **frozen** from a reference pass of the **matched
Gaussian field beam**:

* the frozen field is the 2-D Gaussian-equivalent transverse SC field
  (the same `_gauss_field_2d` formula the PIC solver uses), with σ_x(s),
  σ_y(s) taped from a beam drawn from the matched Σ — **not** from the
  sparse probe ladder, whose std would set an arbitrary field strength;
* because the Gaussian field is nonlinear, core particles see the full
  central gradient and tail particles a weaker one → a **real
  amplitude-dependent spread**;
* per-particle tunes come from a windowed-FFT NAFF-lite of the
  Courant–Snyder-normalized coordinate.

```python
from linac_gen.analysis.footprint import tune_footprint
fp = tune_footprint(lattice, ref, period, base_initial=initial,
                    current=5.0, n_turns=256, n_particles=120)
print(fp["mu_x_core_deg"], fp["mu_x_spread_pp_deg"])   # core tune, spread
```

Returned keys include per-particle `qx` / `qy` (fraction of a cell) and
launch amplitudes `ax_sigma` / `ay_sigma`; the small-amplitude
`mu_{x,y}_core_deg`; and the spread `mu_{x,y}_spread_pp_deg` (peak-to-peak)
/ `mu_{x,y}_spread_rms_deg`.

**Physics & honesty.** For a defocusing SC field the core is the *most*
depressed (tune rises with amplitude toward the bare tune), and the
Gaussian core sees ~2× the rms-equivalent gradient — so the ordering is
**core < rms channel tune < bare**, *not* a core-equals-centroid identity.
At zero current the footprint collapses to a point (bare tune, zero width).
This is a **frozen-field** footprint, not a self-consistent PIC footprint
(future work); the longitudinal plane is not modelled, and accelerating
cells use a frozen-energy-per-turn approximation.  Absolute tunes near
strong depression approach the FFT resolution floor (~360°/`n_turns` per
cell) and are approximate.

## GUI

Both live on the **Results** tab.  The **Hofmann stability chart** popup
opens with the exact per-cell trajectory; **Compute chart** then solves
the corrected dispersion relations **off-thread** (progress on the
button, abortable on close) and renders the growth-rate heatmap
`γ/ν_0x(R, Y)` at the trajectory's median per-cell ε_z/ε_x with each cell
classified — green = valid + stable, red = flagged, grey hollow =
`S² > 10` (extrapolation), amber ring = fold risk.  A `P(unstable)`
checkbox adds the per-cell Monte-Carlo probabilities and a
`legacy bands` checkbox restores the old indicative m/n overlay.
Computed charts are cached keyed on the solver fingerprint, so reopening
at the same ε and resolution is instant.  The **Tune footprint (frozen
SC)** popup's **Compute** button runs the footprint **off-thread** — it
re-tracks the cell — then scatters the per-particle tunes coloured by
launch amplitude.

## Assistant

The `hofmann_stability` assistant tool runs the same analysis headlessly
(tier *compute*; runs its own envelope phase probe when the session
results lack probe maps).  Parameters: `period_index`, `margin`,
`probability`, `n_mc`, `threshold`.  It returns the per-cell JSON rows,
the flag summary, and the refusal `reason` when the lattice is coupled or
the beam is DC.

## Cross-references

* [Phase advance & tune depression](06_phase_advance.md)
* `linac_gen/analysis/hofmann_dispersion.py:1`
* `linac_gen/analysis/hofmann_stability.py:1`
* `linac_gen/analysis/hofmann_probabilistic.py:1`
* `linac_gen/analysis/hofmann.py:1`
* `linac_gen/analysis/footprint.py:1`

← [Phase advance](06_phase_advance.md) ·
[Continue to GUI → Overview →](../10_gui/01_overview.md)
