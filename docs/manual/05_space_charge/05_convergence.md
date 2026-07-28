# Convergence guide

Practical guidance on grid sizing, particle count, and step density
for HELIX simulations to actually converge.  Wrong choices produce
plausible-looking but wrong answers — this is where most simulation
hours go.

## TL;DR — production defaults

For a typical PIP-II-stage simulation (post-RFQ, ≤ 5 mA, ≤ 1 km
lattice, 1 % σ accuracy target):

| Parameter | Value | Notes |
|---|---|---|
| Particles `n_particles` | 5000-20000 | 1 % σ floor at √N statistical limit |
| Grid `nx=ny=nz` | 96 (the default; 64 for quick tests) | 96³ converged for 5 mA |
| `grid_extent` | 5.0σ | ~99.99% Gaussian retention |
| `kernel` | `cic` (with RF) or `tsc` (no RF) | see [Kernels](03_kernels.md) |
| `green_kind` | `igf` | always |
| Step density | `PARTRAN_STEP 100 50` | step1 = 100/m integration substeps, step2 = 50/m SC kicks (both apply to DRIFT + FIELD_MAP only) |

**PARTRAN_STEP semantics** — the card carries two steps-per-metre
numbers:

* **step1** drives the integration substep count for **DRIFT and
  FIELD_MAP** elements;
* **step2** sets the **space-charge kick cadence** inside those same
  elements;
* **quads, bends, solenoids** (and other transfer-map elements) are
  always tracked in **exactly 2 substeps** with one mid-plane SC
  kick, regardless of this config.

## Convergence checks

For *any* production run, verify convergence in three dimensions:

### 1. Particle count

Run the same lattice with `n_particles ∈ {2000, 5000, 10000, 20000}`
and plot σ_x at end of lattice.  The 1/√N noise floor visible at
2000 should disappear at 10000.  If σ_x changes significantly
beyond 10000, the simulation hasn't converged — likely indicates
strong non-Gaussian effects.

### 2. Grid size

Run with `nx ∈ {32, 48, 64, 96, 128}` (square cubic grids), plot
σ_x at end of lattice.  Should plateau by 96; if not, double the
extent or check for unusual aspect ratios.

The scans live in the GUI's **Numerics tab** (Scan parameters
group) — see [Numerics tab](../10_gui/04_convergence_tab.md); it
automates the grid, extent, step1, and step2 sweeps.

### 3. Step density

Run with `step1_per_m ∈ {25, 50, 100, 200}`.  Below 50/m,
short-element field maps under-sample.  Above 100/m, gains
diminish but cost rises linearly.

## Memory note: BTL CIC ε_z growth

Per memory note `project_cic_grid_noise_in_no_rf_transport`: in
long no-RF transport sections like the BTL, the CIC kernel + IGF
Green's function suffer from grid-noise accumulation that blows up
ε_z by ×14.  Switching to TSC drops it to ×1.6 (matches TraceWin's
×1.7).  See [Kernels](03_kernels.md).

## Memory note: PXIE LEBT silent failure

Per `feedback_lebt_scc_and_envelope_continuous`: PXIE LEBT needs
`Ki=1` (SC compensation) AND the envelope solver needs
`continuous=True`.  Forgetting either produces 6.93% mean error
that **looks like a physics bug**.  Always verify these flags for
LEBT-class runs.

## Memory note: PIC kernel calibration offset

Per `project_lg_pic_kernel_calibration_offset`: HELIX's PIC kernel
has a documented ~3% σ overshoot vs TraceWin partran on PIP-II.
This is **separate** from grid-noise / convergence issues — it's
a known absolute-calibration bias.  When validating against TraceWin,
expect ~3% residual on σ.  Higher residuals usually indicate a
parameter mismatch or convergence issue.

## Runtime expectations — the transfer-matrix cache and who uses it

HELIX has an **opt-in** per-element transfer-matrix cache keyed on
`(id(elem), elem_fingerprint, ref_fingerprint)` with 12-decimal FP
rounding (`linac_gen/tracking/matrix_tracking.py`).  Opt-in means a
caller must pass a `cache=` dict explicitly; with the default
`cache=None` every matrix is recomputed on every call.

**Who actually passes a cache:** only the GUI's structure-analysis
panels — Phase Advance and Tune Depression — via
`structure_phase_advance(..., cache=...)`.  There, the first analysis
of a freshly-loaded full PIP-II lattice costs ~340 s and repeat
analyses ~3 s (**127× faster**).

**Who does NOT:** envelope and multi-particle *runs* (the solvers call
`get_element_matrix(element, ref)` uncached), and the **matching
engine** — matching mutates element parameters in place on every
optimiser step, which is exactly the situation the cache is documented
as unsafe for.  A simulation run therefore neither benefits from nor
populates the cache; expect the same wall-clock on every run.

!!! warning "Don't try to pre-warm the cache in a background thread"
    A `MatrixCacheWarmer` lives at
    `gui/linac_gen_gui/interphase/services/matrix_cache_warmer.py` but
    is deliberately **not instantiated** by the GUI (see comment at
    `app.py:174-177`).  Past experiments showed that even with a
    `running_changed` guard the warmer's fitted-matrix Jacobian walks competed
    for CPU on project / lattice load and made the GUI feel slow.  If
    you re-enable it without a new design (e.g. a truly idle warmer
    that fires only after N seconds of zero UI events), expect the
    same regression.

## Cross-references

* [Models](01_models.md) — picking the right SC model first.
* [Kernels](03_kernels.md) — CIC vs TSC.
* PIP-II validation — full
  256m benchmark with documented residuals.

← [DC mode](04_dc_mode.md) ·
[Continue to Differentiable PIC →](06_differentiable.md)
