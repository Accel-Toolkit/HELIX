# RfqCell

A single RFQ (Radio-Frequency Quadrupole) cell — the building block
of an RFQ buncher/accelerator that turns a continuous DC ion beam
into a bunched, accelerated beam suitable for the downstream linac.
Implements the Crandall 2-term potential expansion (M1 in HELIX
nomenclature).

## TL;DR

| | TraceWin / Toutatis | HELIX |
|---|---|---|
| Keyword | `RFQ_CELL` (Toutatis-format `.dat`) | `RfqCell(...)` |
| Voltage | inter-vane voltage V (V) | same `voltage_V` |
| A₁₀ | Crandall 2-term coefficient | same |
| Modulation m | vane modulation factor (m ≥ 1) | same |
| φ_s | synchronous phase (deg) | `phi_s_deg` |
| cell_type | ±2 / ±3 / ±4 | same |

Conventions:

* Voltage is in **volts** (not MV) — typical RFQs run 50-150 kV.
* `cell_type`: ±2 = accelerating, ±3 = front-end / shaper,
  ±4 = transcell.  The sign selects which neighbour cell the
  transverse model couples to.
* Internally split into N substeps (auto-picked, ≤0.1 mm/substep,
  ≥20 substeps).  See [VaneRFQ](10_vanerfq.md) for multi-cell.

## Tutorial

An RFQ cell uses four electrodes (vanes) with a sinusoidally
modulated tip profile to produce a transverse focusing potential
that simultaneously bunches and accelerates.

The Crandall 2-term potential is

$$
U(r, \theta, z) = \frac{V}{2}\left[
A_{01}\left(\frac{r}{r_0}\right)^2 \cos(2\theta)
+ A_{10} I_0(kr) \cos(kz)
\right]
$$

where r₀ is the average radius, k = π/L (cell wavenumber), I₀ is
the modified Bessel function, A₀₁ ≈ 1 (transverse focusing),
A₁₀ encodes modulation strength (longitudinal acceleration).

### The θs operand, and when a deck lies about it

`RFQ_CELL`'s sixth operand θs drives the whole longitudinal channel:
integrating the on-axis field over one cell with the phase cursor
running 0 → 180° gives the closed form

$$\Delta W = |q|\,\frac{\pi}{4}\,A_{10}\,V\,\cos\theta_s$$

so θs = −90° means **"do not accelerate here"**, whatever the vane
geometry says. HELIX takes the card at face value, and so does
TraceWin's own *envelope* model — the two agree to 1.955717 MeV on
PXIE. **Toutatis does not.** It builds the field from the vane
geometry; the TraceWin manual says so outright when defining the card's
`dP` operand, which exists to *"reset the output phases of Toutatis who
does not own phase reference"*. A wrong θs is therefore invisible to
Toutatis and fatal to the card model.

The PXIE deck contains exactly that. Cells 195–199 carry θs = −90°
while A₁₀, m, L, dP **and** `pxie-rfq.vane` all continue their smooth
ramp — the geometry there is fully modulated, carrying the same on-axis
E_z as its neighbours to ~1 %/cell. The card model gains 0.4–1.8 keV in
those five cells where Toutatis gains 27–42, which is **92 % of the
137.6 keV (7 %) gap** between them.

Because a cell is synchronous when `L = βλ/2`, the deck's cell
*lengths* already encode the design velocity profile, hence the
intended per-cell ΔW, hence θs:

$$\cos\theta_s(n) = \frac{4\,\Delta W(n)}{\pi\,|q|\,A_{10}(n)\,V(n)},
\qquad \beta(n) = \frac{2L(n)}{\lambda}$$

[`synchronous_phase_from_lengths`](#api-reference) evaluates this with
**no free parameters** — every input is a card operand. It makes θs
live the same way `modulation_consistency` makes `m` live. On the
synthetic [rfq_demo](../11_examples/10_rfq_demo.md) deck, built by an
unrelated generator, it reproduces that deck's own −90 → −28 ramp to
**+3.0 ± 1.0°** and flags nothing.

```python
from linac_gen.io.rfq_phase_repair import (inconsistent_phase_cells,
                                           repair_rfq_phases)
# read-only: which cards disagree with their own cell lengths?
for f in inconsistent_phase_cells(lattice, mass_MeV, charge, freq_MHz):
    print(f.index, f.phi_card_deg, "->", f.phi_derived_deg)

repair_rfq_phases(lattice, mass_MeV, charge, freq_MHz)   # opt-in, in place
```

`repair_rfq_phases` replaces θs only on cards that disagree by more than
25°, anchoring out the derivation's ~2° systematic (the dropped
transit-time factor) using the deck's own nearest consistent cards. It
warns loudly, and `dry_run=True` reports without touching anything.
Nothing is automatic — parsing is unchanged unless you call it, exactly
like [`replace_rfq_cells_with_vane`](10_vanerfq.md).

!!! warning "A repaired deck is a different deck"
    On PXIE this moves the RFQ exit energy from 1.9557 to 2.0986 MeV
    against Toutatis's 2.0933 (**+0.25 %**), and collapses the losses in
    the last 13 cells from 2184 to **37** against Toutatis's **36**.
    Transmission goes 39.7 % → 67.3 %; the remaining gap to Toutatis's
    80.4 % sits in the gentle buncher (cells 60–140) and is a separate,
    open problem. Say that you repaired the deck in anything you publish
    from it, and keep the original alongside.

### Transfer matrix and per-substep integration

Like a FieldMap, an RFQ cell has **no closed-form 6×6 transfer
matrix** — the 2-term potential mixes transverse position with z
non-linearly.  Both models use a per-substep **Strang
Drift–Kick–Drift** splitting (TraceWin manual convention).

**`field_model="tw2term"` (DEFAULT since 2026-07-30)** — the exact
TraceWin annex per-step algorithm, with every formula-transcription
ambiguity resolved numerically against TraceWin's own per-cell
transfer matrices (203 PXIE cells; full derivation and calibration
record in `linac_gen/elements/rfq_coefficients.py`).  Key physics:
sin-phased quadrupole of strength V/R₀² (no `(1−A₁₀)` reduction —
TW's R₀ is *defined* as the radius where the quad term is V/R₀²),
cos-phased RF defocus, exact ±3 front-end / ±4 transcell
coefficient forms (±3 are the sin³/cos³ ramps), per-substep
synchronous-γ advance, TW's sin-phased K₂
momentum rescale, and the **per-particle longitudinal phase slip**
in multiparticle tracking — absent from the legacy path, which is
why a DC beam could never bunch there (legacy PXIE capture: 24 %;
tw2term: **99 %** vs the 98±2 % PIP2IT measurement).  The
synchronous ramp reproduces TW's exit energy to all printed digits.

**`field_model="2term"` (legacy fallback, explicit opt-in)** —
`A_quad = (1−A₁₀)/R₀²` with `S = −sign(Type)`, the empirically-
calibrated 2026-04 path, kept bit-identical; its multiparticle
path has no phase slip and no losses and must not be used for capture
or transmission studies.  Three diagnostic-only variants (`crand_x`,
`crand_x_noflip`, `pdf_2term`) are kept for comparison but blow up
in envelope runs and must not be used for production.

A **Boris time-stepper + Hybrid field source** exploration exists
in the separate `rfqtrack` subproject — it is *not* the integrator
behind `RfqCell` in `linac_gen`.

For envelope tracking, the per-substep 6×6 Jacobian is built by
finite differencing the pusher and chained over the cell.

| Block | Effect |
|---|---|
| (x, x', y, y') 4×4 | strong AG focusing from sin(2θ) potential term |
| (Δφ, ΔW) 2×2 | bunching + acceleration from cos(kz) modulation |
| Cross blocks | (x, Δφ) and (y, Δφ) **non-zero** — RFQ deliberately couples transverse and longitudinal |

The transverse-longitudinal coupling is intrinsic to the RFQ — it
is what bunches a continuous beam.  No purely-transverse subspace
is invariant.

### Validation status

**tw2term vs TraceWin ground truth** (PXIE LEBT+RFQ project,
2026-07-30; pinned in `tests/rfq/test_tw2term_benchmarks.py`):

| Quantity | tw2term | Reference |
|---|---|---|
| Per-cell 4×4 matrix, median rel. error | 1.25 % (200/203 cells < 10 %) | TW `Transfer_matrix1.dat` |
| Cumulative Π det(x-block) (momentum invariant) | 0.1237 | TW 0.1236, physical 0.1238 |
| Synchronous exit energy | 1.955717 MeV (exact) | TW chart 1.955717 MeV |
| MP capture (5 mA DC, no SC) | ~99 % | design 99.8 %, measured 98±2 % |
| Envelope σ vs TW ENV export | **1.1 % (x) / 1.7 % (y)** | vane-based reference — TW's own matrices reproduce it only to ~4 % |

**Smooth TW calibration (vane-field campaign, 2026-07-30)**: on top of
the exact annex algorithm, `step_kicks` applies a small smooth
correction pair (quad ≤1 %, transverse defocus ≤2.5 %, parameterised
by the card A₁₀; `rfq_coefficients.tw_calibration`).  Its physical
origin is the Toutatis vane-field solution: an FD Laplace solve of the
true constant-Tc electrode (pipeline validated to 0.1–0.7 % on the
exact two-term surface) predicts corrections of the same sign and
cell-trend; the magnitudes are calibrated to the 203 ground-truth
matrices (the same mode-faithful method as the fnalscl T(β) factor).
The longitudinal channel (E_z ramp, K₁, K₂) deliberately stays on the
card A₁₀ — it already matched TW exactly.  Two pinned negative
results: per-slice coefficients inverted from the vane-TIP table add
nothing (tips are two-term to noise), and RAW per-cell fitted
corrections make the beam WORSE (cell-to-cell jitter breaks AG
coherence; envelope y 7.4 → 16 %) — smoothness is load-bearing.

Two epistemic cautions: (1) 1.1 % is *below* the ~4 % with which TW's
own matrices reproduce the same chart — part of the calibration
absorbs chart-specific residuals, so treat sub-floor agreement as a
fit property, not extra physics; (2) the calibration is
**single-project** (PXIE) and applies, bounded but unvalidated, to
any other RFQ — set
`linac_gen.elements.rfq_coefficients.TW_CALIBRATION_ENABLED = False`
for the uncalibrated exact-annex model.

**Losses (tw2term only, 2026-07-30)**: the multiparticle path checks
each substep against the actual vane-tip apertures
``x_lim(z)/y_lim(z)`` solved from the two-term equipotential
condition (validated to 0.03–0.14 % against the PXIE
``pxie-rfq.vane`` tip table — no empirical factor needed), plus a
W&nbsp;<&nbsp;0 kill for back-accelerated junk.  ±3 front-end/exit
cells skip the transverse check (their real vanes flare to ~3.5·R₀).
With TraceWin's measured input distribution the full LEBT+RFQ line
reproduces TW's LEBT scraping (77.5 % vs 77.3 % — requires the
solenoids' ``.ouv`` bore profiles, i.e. ``Ka=1`` on their FIELD_MAP
cards as in TW's own decks), the transmission–voltage S-curve has its
shoulder just below nominal voltage as measured at PIP2IT, and the
captured beam exits at ε = 0.133/0.167 π·mm·mrad vs the PIP2IT
measurement 0.17/0.16.  Known gap: RFQ transmission of LEBT survivors
is ~82 %, concentrated in the last tight-bore (3.1 mm) cells — see the
vane-coefficient plan below.

!!! note "The old ~97 % figure was unsourced — corrected 2026-07-31"
    Measured directly from `calculations/toutatis.out` on the same 203
    cards and the same input distribution, **Toutatis transmits
    80.40 %**, not ~97 %.  Its losses sit overwhelmingly in the gentle
    buncher (cells 60–140), with only **36** particles lost in the last
    13 cells.  HELIX loses **2184** there — but that is the θs = −90
    problem above, not a bore/aperture problem: with the deck repaired
    the count drops to **37**.  What remains is a genuine buncher gap
    (HELIX 2074 vs Toutatis 1288), which is where the vane-coefficient
    work should point.

**Reading RFQ output: the bunch train (2026-07-30)**.  An RFQ turns a
DC beam into a *train* of bunches one RF period apart.  HELIX seeds one
RF period of DC beam, and during bunching space charge pushes ~20 % of
the particles across a bucket boundary — physically into the
neighbouring bunch of the same train.  Because Δφ is unwrapped, a raw
φ–ΔW plot then shows a row of vertical stripes 360° apart rather than
one bunch; each stripe is a real, fully accelerated bunch (the leading
one sits slightly high in energy, the trailing one low — the signature
of having slipped a period).

**Viewing it**: the phase-space popup's **fold φ** checkbox (on by
default) folds Δφ into one RF period about the bunch centroid, giving
the single-bunch picture TraceWin/Toutatis draw; unchecking shows the
raw train.  The beam-parameters table and the Ctrl+S export follow the
same checkbox, so the numbers beside the plot always match it, and a
folded table is labelled with how many particles came from adjacent
buckets.

**The fix: periodic phase coordinates (`periodic_phase`)**.  Tick
**Periodic phase (bunch train)** in the Beam tab — or set
`periodic_phase: true` in the project's `beam` block, or
`BeamConfig(periodic_phase=True)` from Python — and the tracker folds
Δφ into one bunch spacing after every element and before every
space-charge kick, the convention Toutatis uses.  A particle that
slips past a bucket boundary then simply re-enters the neighbouring
one, the satellite stripes never form, and `sigma_phi`, `emit_z` and
the z-Twiss become single-bunch numbers **with no statistics treatment
at all** — on the PXIE deck at 5 mA, σ_φ 172° → 5.0° and ε_z 3.05 →
0.073 deg·MeV, while line transmission (62.0 → 60.6 %) and the exit
energy (2.1025 → 2.1035 MeV) barely move.  The energy *spread* improves
genuinely, 24.0 → 15.4 keV, because the folded beam no longer carries
the spurious chirp of a finite three-bucket clump.  Both shipped RFQ
example projects enable it.

The fold is physics-neutral: the RF forces depend on Δφ only through
cos/sin, so shifting a particle by a whole bunch spacing changes
nothing.  With space charge off, a flagged run reproduces the
unflagged one to 4e-14 in energy and 1e-12 transversely, with an
identical loss mask and Δφ differing only by multiples of the spacing.
Four guard rails keep it honest:

* **Opt-in, default off** — no existing run changes.
* **Only a beam that was injected DC and has since been bunched.**  A
  beam born bunched (a `.dst` input, an MEBT deck) is one bunch, not a
  seeded period of a train, and is never touched even with the flag on.
* **The period is the bunch spacing, not the RF period** —
  360·f_local/f_bunch, so it is 360° in the RFQ and **720° after a
  162.5 → 325 MHz jump**.  A non-integer ratio (a sub-harmonic
  buncher) would make the fold change the phase a particle sees, so it
  is skipped with a `PeriodicPhaseWarning`.
* **Backtracking refuses a flagged run.**  The fold discards which
  bunch a particle landed in, so it cannot be undone; `helix backtrack`
  raises rather than return a reconstruction wrong by whole spacings.
* **CSR is rejected outright.**  `csr_enabled` builds its wake from the
  ensemble's absolute longitudinal extent (z from Δφ, then a histogram
  over `z.min()..z.max()`), which is not periodic in the bunch spacing:
  folding changes the force on particles it never moved.  The two
  flags together raise.
* **A frequency that is not a harmonic of the buncher stops the fold.**
  If that happens before any particle has been folded it is a warning
  and the fold is skipped; if folds are already in flight it is an
  error, because every downstream RF element rescales them by
  f_new/f_old and they are no longer whole RF periods.  The test has a
  1e-3 tolerance on the ratio, so ordinary cavity detuning (kHz) does
  not disturb it while a genuine sub-harmonic is caught with three
  orders of magnitude to spare.
* **An imported `MATRIX` element that couples Δφ is rejected.**  An
  Elegant `EMATRIX` applies the full 6×6, so a non-zero `M[i,4]` is
  linear in Δφ rather than 360°-periodic and a fold would shift
  coordinate *i* by `M[i,4]·P`.  Everything HELIX generates itself has
  column 4 = (0,0,0,0,1,0) and is unaffected.
* **A static electrostatic element does not create a train.**  An
  einzel lens or DC extraction column carries an electric field but
  cannot bunch anything; the beam stays DC and is never folded.

The period is taken from the element that actually bunched the beam —
its RF clock is captured at the DC→bunched transition — not from
`BeamConfig.frequency`.  A beam configured at 162.5 MHz and bunched by
a 325 MHz gap therefore folds at 360°, as it must; using the config
frequency would have given 720° and left every satellite in place with
no warning.

*Cost*: ε_z is no longer exactly constant through a drift.  Each
bucket crossing is a step in the reported value, so ε_z(s) carries a
staircase wherever particles are still crossing — measured over 40
identical drifts, a badly debunched beam went from 1.4 % spread to
64 %, while a beam that stays inside its bucket measured exactly
0.000 % either way.  Inherent to the convention, since the particle
really did change bunch.  Re-tune any σ_φ / ε_z matching objective
rather than reusing one across the switch.

*What does change under space charge*: the solver now sees a compact
bunch instead of a clump spread over several buckets, so the charge
density it works from is genuinely different and the transverse
emittance responds.  Measured at 5 mA on the 66 kV deck (3000
particles), ε_nx went 0.142 → 0.194 π·mm·mrad against the PIP2IT
measurement 0.17 — the unfolded run undershoots and the folded one
overshoots by a similar margin.  At 0 mA the two agree exactly, which
is what identifies this as a space-charge effect rather than a
bookkeeping artefact.

*Limitation*: space charge then sees one isolated bunch.  In a real
train the neighbours partially cancel the longitudinal field, so it is
slightly overestimated; periodic SC images are not implemented.

**Without the flag, reported σ_φ / ε_z are train-wide, deliberately.**
For a bunch-train beam the recorded `sigma_phi`, `emit_z` and z-Twiss
are computed on the RAW phase, so an unflagged RFQ run reports
σ_φ ≈ 183° where the individual bunch is ≈ 4°.  Folding the
*statistics* instead was implemented, measured and REJECTED
(2026-07-30, two adversarial reviews — see the evidence in
`diagnostics/moments.py::wrap_phase_column`): the satellite buckets are
not periodic images of the core (they carry a space-charge-generated
≈ −35 keV/bucket chirp and differ at 8–52 σ), so folding is a biased
estimator — ε_z +123 %, σ_W +69 %, α_z with the *wrong sign* — and no
shot-noise-robust rule exists to decide when to apply it at the
statistics level (a compactness gate swung the reported σ_φ by 271 %
from a 0.9 % input change, under-reported genuine debunching 10× on a
mismatched beam, and turned a drift's exactly-constant ε_z into a
staircase).  A visibly wrong number is safer than a subtly wrong one.

That is why the fix lives in the tracking coordinates
(`periodic_phase`, above) rather than in the moments: with the train
gone there is nothing left to decide.  For a run made without the
flag, read bunch length off the folded plot.

**Space charge through the RFQ (Phase 4 audit, 2026-07-30)**: RFQ
cells ride the tracker's Strang SC bundles like any field map, and the
LEBT solenoids' `.scc` neutralisation profiles are honoured.  The
DC→bunched transition fires at the first RFQ cell — physically early
(the beam stays quasi-DC through the shaper) — but a controlled
experiment keeping the DC 2-D kick through the whole line changed
transmission by only 0.1 point at 5 mA, so the early flip is benign
at PIP-II current; neighbour-bunch periodic images (TraceWin PICNIR
practice) are deferred on that evidence.  With SC on, the captured
exit emittances land at 0.170/0.159 π·mm·mrad vs the PIP2IT
measurement 0.17/0.16.  When quoting exit bunch length, use the
*wrapped* Δφ (±n·360° offsets of barely-captured particles inflate
the unwrapped rms from ~5–7° to ~33°).  Note there is currently NO
genuine SC envelope reference in the repo — the historical
`LEBT+RFQ_ENV+SC.txt` is a byte-identical copy of the no-SC export.

**Modulation checks (Phase 3)**: the card's `m` operand is live as a
consistency check — a >15 % mismatch between the card A₁₀ and the
two-term value implied by (R₀, m) warns, as does m > 3.2 (the
documented breakdown of the two-term treatment).  Fields always follow
the card A₁₀, exactly as TraceWin.  A per-slice coefficient inversion
from the `.vane` tip table was tried and REJECTED with evidence: the
PXIE tips are two-term to within the inversion's own noise, so the
residual difference to TW's matrices lives in the Toutatis field
solution, not in tip data.

**Caveat on the reference**: the project decks carry
`RFQ_GEOM 1 pxie-rfq.vane` — TW's matrices and ENV charts were
produced *with the Toutatis vane-geometry tables*, not the pure
card model.  Even TW's **own** matrices reproduce the ENV chart only
to ~4 %, so the ~7 % envelope agreement sits near the reference's
intrinsic floor.  The remaining 10–17 % outliers (a few shaper cells
with near-cancelling F/D halves) are plausibly
vane-solution effects; closing them would require vane-table-driven
coefficients (planned; the `.vane` reader already exists in
`io/tracewin_vane.py`).

The numbers below come from the **rfqtrack subproject's**
Boris time-stepper + Hybrid field source exploration on PXIE NOSC
(continuous beam, no SC) — not from the `linac_gen` `RfqCell`
integrator documented on this page:

| Quantity | Ratio (rfqtrack Boris+Hybrid) / TW |
|---|---|
| σ_x | 0.83 |
| σ_y | 0.88 |
| W (energy) | 0.94 |
| Transmission | 92 % |

In that study the 2-term Strang-splitting integrator showed σ_y
blow-up by 3.1× at the M1 ceiling — the Boris/Hybrid path closed
that gap in rfqtrack.  Within `linac_gen`, the 2-term Strang DKD
remains the production path (σ_x within ~30 % of the TraceWin
reference at the whole-RFQ level, see [VaneRFQ](10_vanerfq.md)).

Higher-fidelity variants (M3-family laplace2d/3d/8-term) hit
structural blockers; M1 is the production path.

## API reference

```{.python .skip}
linac_gen.elements.rfq_cell.RfqCell(
    name: str,
    voltage_V: float,             # inter-vane V (volts)
    r0_mm: float,                 # vane mean radius R₀ (mm)
    A10: float,                   # Crandall 2-term coefficient
    modulation: float,            # m
    length_mm: float,             # cell length L (mm)
    phi_s_deg: float,             # synchronous phase (deg)
    cell_type: int,               # ±2 / ±3 / ±4
    Tc_mm: float = 0.0,
    dP_deg: float = 0.0,
    n_steps: int | None = None,   # auto-picked if None
    type_prev: int | None = None,
    type_next: int | None = None,
    A_quad: float | None = None,
    aperture: float = 0.0,
    field_model: str = "tw2term",
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `voltage_V` | (required) | V | inter-vane voltage (typical 50-150 kV) |
| `r0_mm` | (required) | mm | vane mean radius R₀ |
| `A10` | (required) | — | Crandall 2-term acceleration coefficient |
| `modulation` | (required) | — | tip modulation m (≥ 1); feeds the 2-term `A_quad` default |
| `length_mm` | (required) | mm | cell length L (no βλ/2 check enforced) |
| `phi_s_deg` | (required) | deg | synchronous phase |
| `cell_type` | (required) | int | ±2 = accelerating, ±3 = front-end / shaper, ±4 = transcell |
| `Tc_mm` | 0.0 | mm | transverse curvature (TraceWin `Tc`) — accepted for parser compatibility, currently unused |
| `dP_deg` | 0.0 | deg | output-phase shift (TraceWin `dP`), applied at cell exit |
| `n_steps` | None | — | auto-picked: max(20, ceil(L/0.1 mm)) when None |
| `type_prev`, `type_next` | None | int | neighbouring cell types (S = −sign(type[n±1])); default to `cell_type` |
| `A_quad` | None | 1/mm² | DC quadrupole coefficient override; None → `(1 − A₁₀)/R₀²` |
| `aperture` | 0.0 | mm | scalar radius for the tracker's generic end-of-element check (0 = that check off); under `tw2term`, transverse losses follow the vane-tip profile regardless |
| `field_model` | `"tw2term"` | — | default `tw2term` (exact TW annex, calibrated); legacy fallback `2term`; diagnostic-only `crand_x`, `crand_x_noflip`, `pdf_2term` |

### Source

* `linac_gen/elements/rfq_cell.py:135` (constructor)

## See also

* [VaneRFQ](10_vanerfq.md) — `.vane` file wrapper.
* LEBT + RFQ worked example.
* `Tracewin_code/Toutatis_*.pdf` — RFQ-design code reference.

← [FieldMap3D](08_fieldmap3d.md) ·
[Continue to VaneRFQ →](10_vanerfq.md)
