# Appendix E — Migrating from TraceWin

A porting checklist for users transitioning from TraceWin to HELIX.

## What "just works"

Open a TraceWin `.dat` in HELIX and these are correct out of the
box:

* Element cards: DRIFT, QUAD, SOLENOID, BEND, EDGE, GAP,
  FIELD_MAP, RFQ_CELL, APERTURE, MARKER, THIN_STEERING /
  STEERER.
* Header directives: TITLE, FREQ, PARTRAN_STEP, LATTICE,
  LATTICE_END, FIELD_MAP_PATH.  (`REPEAT_ELE` is **not**
  implemented — expand repeats before importing.)
* Error directives: ERROR_QUAD_NCPL_STAT, ERROR_CAV_NCPL_STAT,
  ERROR_BEND_NCPL_STAT, ERROR_BEAM_STAT,
  ERROR_GAUSSIAN_CUT_OFF.
* Matching directives: SET_TWISS, SET_POSITION, SET_SIZE,
  SET_ACHROMAT, SET_SYNC_PHASE, the generic ADJUST card, and
  ADJUST_STEERER / _BX / _BY.  (There are no per-element
  ADJUST_QUAD / ADJUST_SOLENOID / etc. keywords — element
  parameters go through the generic `ADJUST`.)

Just open the `.dat` in the GUI (Ctrl+O) and click Run.

## Conventions and units

* Length units: **mm** — this is HELIX's internal convention for
  `.dat` files, both reading and writing.
* RF phase units: degrees.
* Energy units: MeV.
* Coordinate convention: x' = pₓ/p_z (kinetic, not canonical).
* σ_φ at *local* RF frequency (changes at FREQ jumps).
* Geometric ε in mm·mrad; longitudinal in deg·MeV.

!!! warning "Longitudinal α_z flips sign vs TraceWin"
    HELIX's longitudinal plane is **(Δφ, ΔW)** and Δφ runs *opposite* to
    TraceWin's z-like coordinate (a **late** particle has Δφ > 0 but sits
    *behind*: z < 0).  The ⟨z·δ⟩ correlation therefore has the opposite sign
    of ⟨Δφ·ΔW⟩, so when transcribing a TraceWin longitudinal Twiss input,
    **negate α_z** (β_z and ε_z carry over unchanged):

    | | TraceWin input | HELIX `alpha_z` |
    |---|---|---|
    | Example (PIP-II HB650) | −0.50 | **+0.50** |

    Getting this sign wrong leaves energy, transmission and the transverse
    envelope untouched but *damps the synchrotron mismatch oscillation* —
    σ_φ/σ_W collapse toward the matched value instead of oscillating.
    Validated against a TraceWin ENV+SC/MP+SC reference of
    `examples/piplattice/` (σ_φ correlation collapses from ~1.0 to 0.71
    with the wrong sign).  Transverse α_x/α_y need **no** flip.

    The same negation rule applies to the **`SET_TWISS` card's α_z
    operand** (functional since 2026-07): the card is compared against
    the recorded HELIX-internal α_z, so transcribe a TraceWin target of
    −0.50 as `+0.50` — consistent with the input beam and
    `ADJUST_BEAM_TWISS`.

!!! note "FREQ switches the machine clock at the card"
    The `FREQ` card is an **active command element**: the reference RF
    clock (`ref.phi_s`) changes at the card's lattice position with a
    time-continuous conversion (`φ ← φ·f₂/f₁`), and the beam's Δφ /
    envelope Σ get the exact frequency-jump rescale there — TraceWin's
    semantics.  Previously the clock only switched at the first
    downstream cavity; with an input beam at a different frequency than
    the lattice (e.g. 804.6 MHz beam into a `FREQ 804.96` line) that left
    the clock ~0.5° behind TraceWin's over the entrance section — a real
    arrival-phase error on absolute-phase (P=1) decks.  (The larger part
    of the historical σ_φ residual on the HB650 benchmark turned out to
    be the H⁻ rest-mass bug, fixed alongside; with both fixes the
    benchmark reaches 0.98–1.00 correlation in every observable.)  σ_φ is
    *reported* in degrees of the machine frequency from the card onward.

!!! warning "No unit-parity claim with real TraceWin"
    `.dat` files written by HELIX round-trip through HELIX's own
    parser but are **not directly portable to the real TraceWin
    program** — check units yourself before exchanging files.  See
    the [.dat file reference](../06_running/02_tracewin_dat.md) for
    the full admonition.

## What's different

### σ_φ reporting at end of multi-section linac

HELIX reports σ_φ at the local cavity frequency at each s.
TraceWin partran reports at the *bunch* frequency (162.5 MHz fixed).
At the end of HB650 (650 MHz), HELIX's number is 4× larger than
TW's for the same physical σ_z.

**To compare**: multiply HELIX's σ_φ by `162.5 / local_f_MHz`.

### PIC kernel calibration

HELIX has a documented ~3 % σ overshoot vs TraceWin partran on
production lattices.  This is a known calibration bias, not a bug.
For design-stage work it's acceptable.

### Quadrupole g3..g6

TraceWin parses but ignores the higher-pole content fields on
QUAD cards.  HELIX honours them as thin Multipole kicks in a
split-operator scheme — the quad's total g3..g6 content is
distributed uniformly across the integration substeps (half-kick,
linear map, half-kick per substep).  Net: a `.dat` authored for
TraceWin that *uses* g3..g6 fields will produce slightly different
results in HELIX (the result HELIX gives is more physically
accurate).

### Dipole hv=1 fixed (was a bug, now fixed)

Until recently HELIX's `BEND` card with `hv=1` (vertical bend) was
stored but ignored — bends were treated as horizontal regardless.
Fixed in commit 2026-05-07.  `.dat` files with vertical bends now
produce correct trajectories.

## What you need to set outside the `.dat`

### PIC / space-charge settings

TraceWin doesn't have explicit PIC-grid controls in the `.dat`, and
in HELIX these are **run settings, not lattice content**: set them in
the GUI's Numerics tab, via CLI flags (`--nx`, `--kernel`,
`--sc NAME=VALUE`), or in `SpaceChargeConfig` from Python.

HELIX does parse comment-prefixed `;@LG key=value` directives
(`;@LG kernel=tsc`, `;@LG nx=64 ny=64 nz=64`, …) and stores them on
`lattice.lg_options` — but **nothing currently consumes them at
runtime**, so treat them as documentation / tooling hooks, not as
configuration.  See
[HELIX extensions](../06_running/03_lg_extensions.md).

### Continuous-beam flag

For pre-RFQ runs, set `continuous=True` in `BeamConfig` (Python) or
tick **Continuous beam (pre-RFQ / LEBT)** on the GUI's Beam tab.  A
`;@LG continuous=true` line in the `.dat` has **no runtime effect**
(see above).  TraceWin doesn't need an explicit flag because partran
detects DC vs bunched from the input distribution; HELIX is explicit.

## MAD-X import

HELIX also opens a subset of **MAD-X** lattice files (`.madx` /
`.seq`) — sequences, common elements, expressions, and the `BEAM`
command; lengths are converted from MAD-X metres to HELIX's internal
millimetres on import.  See
[Importing MAD-X lattices](../06_running/02_tracewin_dat.md#importing-mad-x-lattices).

## What's deferred

Things TraceWin has that HELIX doesn't (yet):

* `ERROR_*_DYN` — dynamic per-step jitter (absorbed as *static*
  errors of the same magnitude, not skipped).
* `ERROR_*_CPL_*` — coupled-error groups (absorbed as static
  uncoupled NCPL errors).
* `ERROR_RFQ_CEL_NCPL_STAT` — per-cell RFQ errors.
* `ERROR_STAT_FILE` — file-driven error lists.
* `REPEAT_ELE` — reported as an unsupported card.

For these, either:

* Use a TraceWin-compatible workaround (static errors instead
  of dynamic).
* Stay in TraceWin for that specific feature.
* Implement it manually in Python (the engine is open).

`ADJUST_STEERER`, formerly on this list, has shipped: it now builds
matcher variables that drive steerer-based orbit correction.

## Workflow comparison

| TraceWin step | HELIX equivalent |
|---|---|
| Open `.dat` in TraceWin GUI | Ctrl+O in HELIX GUI |
| Configure beam in `Beam` window | Beam tab |
| Choose Envelope vs Multi-particle | Numerics tab + tracker mode |
| Set partran SC settings | Numerics tab (PIC grid, kernel) |
| Run Envelope | Run button (`run_envelope()`) |
| Run Partran | Run button (`run()`) |
| View output graphs | Results tab tiles |
| Match Twiss | Matching tab (or SET/ADJUST in `.dat`) |
| Tolerance studies | Error Study tab |

## Cross-references

* [Quick start](../01_getting_started/03_quick_start.md)
* [Keyword cheatsheet](B_keyword_cheatsheet.md)
* [TraceWin parity](../12_validation/01_tracewin_parity.md)

← [Appendix D](D_troubleshooting.md) ·
[Continue to Appendix F →](F_contributing.md)
