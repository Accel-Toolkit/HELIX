# ML surrogates overview

A **surrogate** is an MLP-backed drop-in replacement for an expensive
HELIX element.  At runtime the trained network predicts the 6×6
linearised transfer matrix that the RK4 integrator would have
returned — at ~5 000× the per-call speed.  The use case is
parameter scans, matching loops, and tolerance studies where the
same field map gets pushed thousands of times with small parameter
shifts.

## TL;DR

* **What's surrogatable**: `FieldMap3D` (3-D Cartesian) and
  `FieldMap` (1-D / 2-D cylindrical / 1-D quad-gradient) elements.
  RFQ subclasses (`VaneRFQ`, `RfqCell`) are excluded — they carry
  state the surrogate doesn't replicate yet.
* **What surrogates predict**: the 6×6 transfer matrix from
  `fitted_matrix(ref)` — the envelope-mode contract at
  **current = 0** (the pure-linear path requests exactly that full
  matrix).  Slice-aware surrogates (for the SC-engaged path) are
  future work.
* **How they're trained**: Latin-Hypercube samples over (ref
  kinematics, element params) → ground-truth matrices from the
  existing HELIX RK4 → small MLP fit on flattened 6×6 outputs.
* **How they're used**: register one or more in the global
  registry; the envelope hook in `tracking/envelope.py` picks them
  up automatically on the next run.
* **Three entry points**: a dedicated **Surrogates tab** in the GUI,
  the `python -m linac_gen.surrogates.cli` subcommands, and the
  `linac_gen.surrogates` Python package directly.

```{.python .skip}
from linac_gen.surrogates.training import train_surrogate_for_element
mlp, meta = train_surrogate_for_element(
    element=elem, ref_template=ref,
    n_samples=5_000, ref_w_kin_range=(2.0, 2.5),
    param_ranges={"ke": (0.054, 0.082), "phase": (-110, -70)},
    hidden_dims=(128, 128), epochs=1000,     # the production cycle
    out_dir="weights/FMAP_001",
    n_workers=12,                  # CPU parallel data gen
)
```

## When to use a surrogate

| Use case | Worth surrogating? |
|---|---|
| Single envelope pass, novel lattice | No — training costs far more than it saves |
| Scans, matching, error studies of a deck you keep coming back to | When an element is asked for several thousand times in all; the weights stay on disk for every later run |
| Differentiable matching (gradient path) | Only with M7 — see below |

The break-even point, measured on one core for the MEBT buncher
`FMAP_001`: a production training takes about a minute (41 s for the 5000
samples, 14 s to train), and a call then takes 0.07 ms where the native
map takes 8.3 ms.  The training is repaid after about 7 000 calls of the
element; an envelope run at 0 mA asks each field map once.

## Scope of accuracy

Whether a file's weights are used is decided by
[the accuracy check](#the-accuracy-check-of-a-weights-file): the largest
2×2-block error of their map within 3e-3 and its determinants within 1e-3
of the native map's, at the input of their first use and at 16 points of
their training range.  The [training guide](05_training_guide.md)'s two
cycles, on the four MEBT bunchers:

| | Smoke cycle (200 × 40, `64,64`) | Production cycle (5000 × 1000, `128,128`) |
|---|---|---|
| Largest block error at the check | 0.042–0.093 | 1.7e-4–3.1e-4 |
| Verdict | refused | used |

The smoke cycle is for "does the pipeline work" sanity checks.  A call of
the network takes about 0.07 ms after its one-time check.

## The accuracy check of a weights file

Weights loaded from a file may predate today's physics, or be roughly
trained.  At their first use after a load inside their range, by any run and
any copy of the lattice, their map is compared with the native element's at
the input of that use and at 16 points of their training range (a Latin
hypercube over the energy and the swept inputs, with a fixed seed):

* the largest **2×2-block error** must be within **3e-3**: each block's
  difference over the largest norm of the native blocks in its row and column
  (a diagonal block's own norm in an uncoupled map; in a solenoid turned near
  90°, whose diagonal blocks vanish, its coupling blocks').  It is taken in
  units where a 1 m drift has the same off-diagonal term in every plane: x, x′,
  y, y′ in mm and mrad, and Δφ/√s, ΔW·√s with s a 1 m drift's Δφ–ΔW term, which
  makes that term 1 there too.  In degrees and MeV a cavity's Δφ–ΔW term
  outweighs its longitudinal focusing by 10³ to 10⁵ (on the half-wave-resonator
  cavities), and a norm there does not see the focusing;
* the **determinants** of the transverse 4×4 block and of the longitudinal 2×2
  block must be within **1e-3** of the native map's.  The 4-D transverse and the
  longitudinal emittances follow them; one plane's 2×2 block is no invariant
  where the planes couple, as in a solenoid.  They are compared with the native
  map rather than with p_in/p_out, which the native solenoid map itself misses
  by about 1.5e-3.

Weights within both limits answer.  Others go native in envelope and
multiparticle runs, the gradient matcher stops with an error naming them (use
`least_squares`, or untick them), and a warning says by how much, once per
load.  The Surrogates tab shows each row's verdict.  Its **Compare** still
runs refused weights where they would answer (at 0 mA; in a multiparticle
compare, with the fast path on), to show how far they are from their element,
and its timing leaves out the one-time check.  Weights built in memory through
the Python API are not checked.  The check takes one native evaluation per
point: about 0.4 s for one of the half-wave-resonator maps, 7 s for the 16 of
that deck, once per load.

The weights the development tree keeps for five of its example decks all
pass on their decks: the 16 sets of the half-wave-resonator deck and of each of
its two variants, the 4 MEBT bunchers and the 20 sets of the deck that joins
the two, with block errors up to 1.3e-3 and determinant errors up to 5.8e-4.
With them every exit rms size and emittance of each deck's envelope at 0 mA is
within 0.19 % of the native run (σW the largest).  Before they were retrained
the check refused 40 of these 72 sets: they had been trained with the earlier
recipe of the [training guide](05_training_guide.md), the MEBT bunchers and the
joining deck's solenoids with the smoke cycle, the joining deck's cavities and
bunchers over 2 to 18 MeV, and cavities of two decks on targets computed before
October's ±180° calibration fix.  All 40 were retrained with the production
cycle, at the windows of **Train all FieldMap\***, and so were two more
cavities of the matching decks (6 sets), which passed the check's points but
missed its limits at 2 to 3 % of 1000 random points of their ranges.  The 16
points are drawn with a fixed seed, and the input of the first use is the
17th: a set near a limit can pass in one session and not in another.  Between
the points the limits can still be missed: on 1000 random points of their
ranges, most of the cavity sets exceed a limit somewhere, at up to 7 points,
every one within 2.1 % of an edge of the range (their 99th percentiles at most
2.2e-3 and 8.8e-4), and the bunchers and solenoids at most once.  Keep queries
inside a range rather than at its edge.

## Which element a surrogate answers for

The registry recognises an element by its structure: its class, length,
field file, geometry, aperture, steps, `p_flag`, `ki` and `ka`.  A reloaded
deck, an error study's seed, a failure study's copy or a parameter scan's
point is the same element.  The lookup decides once per run; the weights
check the element each time they are asked:

* its structure must still be what it was when they were built on it (a
  scan or an edit of its length, steps or `p_flag` makes it another
  element);
* their inputs are read from the element as it stands, and must be inside
  their training range;
* the error knobs are folded into an input or refused (below);
* misalignments are applied around the element, as around the native one;
* every other setting must equal what the weights hold, within one part in
  10⁹ (a deck written and read back keeps ten digits): `ke`, `kb`, `phase`,
  `frequency` and `scale` where they are not inputs, a pinned sync-phase
  calibration, an `NCELLS`'s gap voltage, and the field-map integrator and
  interpolation of Simulation Settings.  A setting the map cannot feel is
  left out: `phase` and `frequency` of a map with no RF field, `ke` of one
  with no electric field, `kb` of one with no magnetic field.  Otherwise the
  native element answers.

The held settings are recorded in training, before the samples are taken;
the samples are taken with the element's error knobs at zero, as a query
folds them in.  For a file from before, or weights built in memory, the held
settings are those of the element the weights are first built on.  The GUI
builds them when it loads a lattice; an edit of the lattice in hand, a
Matching Apply among them, keeps them, so an old file holds the settings of
the element as it was loaded (re-opening the project takes them again from
the element as it then is).  Retrain an old
file to have its settings recorded.

Each surrogate counts the traversals it did not answer, by reason
(`refusals`), and a compare report lists them.  A compensation runs native:
its matcher moves settings, and the settings it finds are judged by the
native elements.  A match, or a Pareto exploration, runs an element native
from its first evaluation to its last, and warns, when a variable moves a
setting its weights hold fixed or moves one of their inputs between bounds
that leave their training range; otherwise the weights answer throughout.
A phase input the running phase moves too is checked only as the match
runs: if it left the range at some evaluations, the match warns that its
objective mixed the network and the native element.  A parameter scan of an
input uses the weights at every value inside their range, after the design
check of the next section; a scan of a setting the weights hold fixed runs
that element native at every point.

## A study and its surrogates

A failure study first runs its design lattice natively and with the
registered surrogates, and uses them only if every exit rms size and
emittance (σx, σy, σφ, σW, εx, εy, εz) of the two runs agrees within 0.5 %.
Otherwise, in a multiparticle study (the surrogate path is linear inside a
cavity) and with the Sacherer envelope (which never asks them), it runs
native; its compensations run native in every case.  The Failure Study tab's
status line says which, and why.
A Parameter Scan (Tools menu) runs the same check; its status line counts
the network answers over its points, and its tooltip gives the check and the
traversals left native, by reason.

An error study with the envelope (`ErrorStudy(mode="envelope")`, the Error
Study tab's **forward: envelope**) runs the same check.  With the
surrogates, it counts per seed, in its own thread, the network's answers and
the traversals left to the native element, by reason, and warns when more
than 10 % of the traversals the network would answer in a seed fall outside
the weights' trained range.  Its orbit correction runs native.  Its first
three seeds run again natively, and it reports their largest exit
difference.  It gives each seed's exit values as run (`exit_raw`) and, for a
seed the network answered as it answered the design lattice, as the native
design value plus the network's change from its own design value
(`exit_corrected`; None for the other seeds).  A multiparticle error study
runs native.  A parameter study of the Study tab and a reliability campaign
of the GUI run native for now and say so.

A failure sweep (`python -m linac_gen failures`) and a reliability campaign
(`python -m linac_gen reliability run`) from the command line run their
points through a pool of processes, which start with no surrogate
registered.  With `--surrogates DIR[,DIR...]` (each a set's folder, or a
folder of sets named by element) or `--surrogates auto` (the lattice's own
weights folder, by its hash and the beam's species) the command registers the
weights, runs the design check of the study's envelope runs and, if it
passes, every one-time check still due (at the point of the training range
nearest the element), and gives each point the weights with the command's
metadata — their accuracy verdicts, refusals included, and held settings.
Each worker reads the weights once, checks them against what the command
loaded, and builds them on its own parse of the lattice before the point's
overrides: nothing is decided again in a worker.  Each row then carries the
network's answers (`surrogate_calls`) and the traversals left native, by
reason (`surrogate_refused`), the same serially and in parallel.  Weights not
in a folder, or unreadable or changed on disk since, run native in the
workers, with a warning; a multiparticle run, an orbit correction and a
compensation run native in every case.  A campaign records the choice
(`surrogates.json`, and `summary.json`'s `surrogates`), and a resume with
another one stops: `--force` starts over.

## Which envelope runs engage the NN

The shipped surrogate serves `fitted_matrix(ref)` — the full-element
matrix — and delegates every *partial* `fitted_matrix_slice(ref, ds)`
to the wrapped RK4.  The engagement regimes, each measured on the
MEBT with the production-cycle weights of its 4 bunchers:

* **current = 0, no per-sub-step recording** — the pure-linear path
  requests each surrogated field map's full-element matrix exactly
  once per traversal, and the registry serves it from the NN.
  Measured: 4 cavities → **4 NN queries**, end-of-line rel.diff
  x/y/φ/W = 1.0e-5 / 6.4e-5 / 1.5e-4 / 1.7e-4; the run takes 48 ms
  where the native one takes 59 ms (**1.24×**, medians of 7 alternating
  runs after a warm-up; a compare, which times one run of each with the
  baseline first, prints 2.7×).
* **current > 0 (SC-engaged envelope)** — the SC bundling code in
  `envelope.py` slices the field map so SC kicks can be inserted
  between slices.  Each partial slice falls back to RK4, so the
  surrogate is **bypassed**: measured at 5 mA, **zero** NN queries,
  zero diff and no speedup (0.99×; a compare prints 2.0× there, the
  baseline's first run alone).  The compare report prints
  `NN full-element queries: 0` with a note so a registered-but-idle
  surrogate is visible, not silent.
* **current = 0 with *Record per-sub-step* ticked, or with
  `SHIFT_IN_FIELD_MAP` interior markers** — the sub-stepped walk is
  taken for the interior σ(s) points; same as the SC case: zero NN
  queries, bit-identical to the unregistered run.
* **Multi-particle tracking** — the NN engages only under the
  explicit MP double opt-in (see the M7 section below).

A slice-aware surrogate (`fitted_matrix_slice` predicted directly,
or via matrix log/exp from a full-element prediction) is in the
roadmap (M-future).  A monkey-patch demo of the log/exp approach is
in the repo for benchmarking.

## MP-mode hybrid (M7)

The same trained MLP can be **engaged in multi-particle (MP)
tracking** by ticking *Engage in MP runs* in the Surrogates tab's
"Multi-particle surrogates (hybrid mode)" collapsible section.
That sets `linac_gen.surrogates.registry.set_mp_enabled(True)`, and
the MP tracker hook in `tracking/tracker.py:_track_field_map`
routes per-element `track_rk4` calls through the registered
surrogate.

**Two opt-ins control behaviour**

| Toggle | Default | What it does |
|---|---|---|
| **Engage in MP runs** (master, M7) | OFF | When OFF: MP runs ignore all registered surrogates — bit-identical to baseline.  When ON alone: the surrogate sits in the dispatch chain as a safe delegate (`track_rk4` calls `wrapped.track_rk4`); still bit-identical to baseline, ~1 % dispatch overhead. |
| **Experimental: linear-matrix fast path** (M7-followup) | OFF | When ON (and master also ON): per-substep `wrapped.track_rk4` is replaced by an analytic ref-advance + batched `M_slice @ particles`.  Linear-matrix accuracy applies; halo dynamics may drift. |

The fast path is **double-opt-in** because its accuracy depends on
training quality and lattice content; it's a research-quality
feature, not a drop-in replacement for full RK4.

**Measured behaviour on PIP-II MEBT+HWR** (20 field-maps, 1000
gaussian particles @ 5 mA, M3 Max 14-core), with the weights the
development tree kept before they were retrained (not measured again with
the retrained ones):

| Configuration | σ_x rel.diff | σ_z rel.diff | ε_nx rel.diff | Speedup |
|---|---|---|---|---|
| Master OFF (no surrogates engaged) | — | — | — | baseline (1×) |
| Master ON, fast-path OFF | 0.00 % | 0.00 % | 0.00 % | 0.99× |
| Master ON, fast-path ON, **smoke training** (200 sa × 40 ep) | 4.7 % | **28.4 %** | 1.5 % | 1.22× |
| Master ON, fast-path ON, **moderate training** (2000 sa × 100 ep for cavities, smoke for solenoids) | 4.9 % | **4.7 %** | 1.9 % | 1.66× |

Key observations:

* **σ_z is highly sensitive to cavity training quality** — moderate
  training of the 12 RF cavities dropped σ_z error from 28 % to
  4.7 % (6× improvement).  Smoke training is fundamentally
  insufficient for the fast path on RF-rich lattices.
* **σ_x/σ_y plateau around 5 %** — driven by the MEBT bunchers
  (zero-phase, sparse-matrix RF cavities, val MAPE inflated by
  near-zero denominators).  Training with the production cycle (about
  a minute per element) and narrower `ref_w_kin_range` windows may reduce
  it; this has not been measured.
* **Speedup limited by the 8 1-D solenoids** which lack the
  required wrapped helpers (`_phasor`, `_sample_onaxis`,
  `_scale_factor`) and fall back to baseline RK4.  RF-cavity-dense
  lattices (e.g., HWR/SSR1 cryomodule strings without bunchers)
  should see substantially higher speedups (~3-5×).

**When to use the fast path**

| Use case | Acceptable? |
|---|---|
| Parameter scans for matching (σ moments are the only metric) | Partly — at ~5 % σ_x error, only if your matching residual is well above 5 % |
| Tolerance studies measuring σ jitter under errors | No — the surrogate's ~5 % bias would masquerade as the tolerance signal |
| Halo / aperture / beam-loss studies | No — linear matrix misses nonlinear halo physics |
| TraceWin parity / publication-grade physics | No — use baseline RK4 |
| Speed-priority exploration where you'll re-run RK4 for final results | Yes — use the fast path; always click `Compare MP` first to validate the bias on your own lattice + beam |

Out-of-scope inputs fall back to the wrapped element automatically,
so the run is always **physically defensible** — never silently
returning bad numbers, only either slower (delegate) or
linear-approximation (fast path).

**Implementation notes**

The fast path lives in `SurrogateFieldMap.track_rk4` and uses
several private helpers from the wrapped `FieldMap3D`
(`_phi_sync_rad`, `_phasor`, `_sample_onaxis`, `_scale_factor`,
`_calibrate_sync_phase`, `_z_map_start`, `_step_idx`).  These are
not a stable API; future refactors of `FieldMap3D` could break the
fast path silently.  Defensive `hasattr` guards in
`SurrogateFieldMap` cause the path to fall back to the safe
delegate per element whenever any helper is missing — so a future
break would degrade performance, never correctness.  A cavity whose
effective frequency differs from the running clock also goes to the
delegate for that traversal: only the native element makes the frequency
jump on entry (Δφ and the clock).  After a `FREQ` card there is no jump
left, and the fast path stays on.

A surrogate records what it was trained at, and refuses, or adapts, when asked at
something else.  A refusal hands the element to the native map, which is correct and
slower.

* **Clock.**  A cavity's targets use its own frequency.  A magnet's Δφ row is in
  degrees of the training clock, and it is converted to the clock it is asked at.  A
  cavity with no frequency of its own runs on the clock, and answers only on the one
  it was trained at.
* **Species.**  A map trained for one species does not answer for another.  A file
  that records no species counts as trained for H⁻: the Surrogates tab and the CLI
  trained H⁻ only until files recorded their species.  Loading it says so, and the
  tab's row reads "trained for H-" under another beam.  Metadata built in memory from a
  reference with no species answers for any.
* **Phase.**  A field map whose `p_flag` is not 1 has a field that follows the running
  phase where the RF drives it: an RF channel, electric or magnetic, whose amplitude
  (`scale` times `ke` or `kb`) is not zero.  A static channel, or an RF channel switched
  off, gives the same map at every phase.  Its map then depends only on its own phase
  plus the running one.  Its `phase` input is
  shifted by how far the phase at its entrance is from the training's, and turned by
  360° if that brings it into scope; out of scope it goes native.  With `p_flag` 1 the
  map does not depend on the running phase.  An `NCELLS` surrogate, which only the
  Python API makes, fires each gap at the running phase minus θs with P=1: its
  `theta_s_deg` input is shifted the other way, and with θs not swept it answers only
  within 0.01° of its training phase.  P=0, P=2 and `SET_SYNC_PHASE` do not depend on
  the running phase.

* **Errors.**  The weights hold the design element.  An error study's `phase_offset`,
  `voltage_rel` and `frequency_offset`, and the RF shift of `SET_BEAM_PHASE_ERROR`, each
  give the map of a design element with an input moved, so they are folded into that
  input: the phase (θs for an `NCELLS`), `scale` (or each of `ke` and `kb` that acts),
  the frequency.  When that input was not swept in training the native element tracks
  instead, and the gradient matcher refuses.  So does an RF shift on a `p_flag` 1 map,
  whose calibration it comes after, any `p_flag` 1 map whose calibration train mode or a
  frozen reliability pin keeps (its weights were calibrated afresh at each input), a `voltage_rel` on an `NCELLS`, and a phase offset
  or RF shift on one under `SET_SYNC_PHASE`.

Weights trained before these were recorded hold none of this, and are checked when
loaded.  A cavity map whose longitudinal determinant is not 1 holds a frequency jump in
its targets: it is marked stale, and goes native until retrained (a factor within 5 %
of 1 is not detected).  A magnet map whose Δφ–ΔW term is not a drift's was trained at
another clock, and goes native at this one.  A cavity whose field follows the running
phase is taken to have been trained at phase 0, which the Surrogates tab and the CLI
used.  If its phase input was swept, its first use after loading checks that against the
native element: its map is scanned over the sweep, in 0.25° steps, against one native
evaluation.  When phase 0 fits clearly worse than the best fit (three times its
residual), as for one trained through the Python API at another phase, it goes native,
and the message says near which phase it fits, an estimate, or that the best fit is at
the edge of the sweep.  A roughly trained file can hide a small shift, and one whose
phase was not swept cannot be checked.  Write the phase it was trained at into its
`metadata.json` as `"train_phi_s_deg"` to use it; a phase written there is believed.

The matrix log/expm slice approximation (`_slice_matrix` →
`_get_cached_slice_from_log`) caches `logm(M_full)` once per
element entry and only pays `expm` once per `(ds, w_kin)`
combination across the SC bundle.  Without this cache the
`expm` cost (~5 ms) per substep would dwarf any savings.

!!! warning "Stateful elements not yet supported"
    Elements with active per-instance state (`.scc` profile
    selection, `reset_run_state` dirty bits) currently get a deep-copy
    in the surrogate but the trained MLP does not see the state as an
    input.  The `SET_SYNC_PHASE` calibration of a `FieldMap3D`
    (`_sync_offset_deg`) is not such state: it is made from the
    entrance energy and the cavity's settings, the network's inputs, for
    every sample and every run, and weights asked about a pinned
    calibration (train mode) go native.  For elements whose physics depends on accumulated
    state — multi-cell DTL gaps in particular — this is a known
    gap; the surrogate may silently mispredict.  Refused by
    construction for RFQ classes; treat with care for any
    FieldMap3D with non-default sync handling.

## Errors and matching (M8)

The matrix below reflects what's wired through to the surrogates in
both env and MP modes after M8:

| Workflow / Error type | Envelope mode | MP mode (fast path) |
|---|---|---|
| Beam errors (centroid, current jitter, Twiss mismatch) | Supported — Σ propagation invariant under beam-shape changes | Supported — tracker handles per-particle |
| Field-strength errors on an input (`ke`, `kb`, `phase` swept in training) — in scope | Supported — the surrogate reads the attribute as it stands, its error knobs folded in | Supported — same |
| Field-strength errors on an input — out of scope | Handled — `OutOfScopeError` → fall back to wrapped RK4 | Handled — same |
| Field-strength errors on a setting the weights hold fixed (not swept) | Native — refused at the query, counted in `refusals` | Native — same |
| **Element tilt** (`tilt_deg`) | Supported — M8: `envelope.py` wraps each element's matrix with `R_out @ M @ R_in` mirroring `tracker.py:208-219` | Supported — M7: `tracker.py:196-254` wraps `_track_field_map` and therefore the surrogate's substep loop |
| Element offset (`dx`, `dy`) | Supported — no-op for Σ by design (Σ is invariant under rigid translation around the centroid); matches TraceWin's envelope semantics | Supported — tracker applies per-particle translate |
| `dz`, `pitch_deg`, `yaw_deg` | Not supported — Tier-1 not implemented in baseline tracker either; stored but ignored. Future work. | Not supported — same |
| **Matching (`gradient` algorithm) through a surrogated cavity** | Supported — M8: `element_matrix_torch` has a `SurrogateFieldMap` arm; `check_gradient_supported` accepts it. Surrogated cavities act as passive autograd-differentiable blocks the gradient flows through; one at another frequency than the clock makes its jump, and the walk goes on at its clock, as in numpy.  A surrogate that refuses (a stale file, another species, a cavity asked at another clock than the one it runs on, or one whose field follows the running phase) stops the matcher with an error that points to `least_squares`. | N/A — matcher is env-mode |
| Matching that moves a surrogated map's `ke` / `kb` / `phase` (`least_squares` and the global optimisers) | Supported for an input of the weights while its bounds stay inside their training range; a variable whose bounds leave it, or that moves a setting they hold fixed, runs the element native for the whole match, with a warning. `gradient` matching keeps surrogated cavities passive and tunes quads, solenoids and dipoles only. | N/A |
| Matching via `least_squares` / `differential_evolution` / `dual_annealing` (numpy paths) | Supported — already works with surrogates; the NumPy `fitted_matrix(ref)` path is `torch.no_grad`-wrapped and returns a plain ndarray that the scipy optimisers consume directly. | N/A |
| TraceWin parity / publication-grade physics | Not supported — use baseline RK4; the fast path's linear-matrix approximation is documented as research-grade. | Not supported — same |

Empirical env-vs-MP tilt parity (driver `/tmp/test_env_tilt.py`, MEBT
drift-quad-drift mini-lattice, 10 000 particles): σ_x rel.diff
**2.7e-5** across tilt_deg ∈ {0, 2, 5, 10}° — far below the
particle-statistics floor (~1 / √10 000 ≈ 1 %), confirming the env
matrix wrap is consistent with the MP tracker's per-particle
rotation.

## Cross-references

* [GUI walkthrough](02_gui.md) — the Surrogates tab end-to-end.
* [CLI](03_cli.md) — `python -m linac_gen.surrogates.cli`.
* [Python API](04_python_api.md) — `train_surrogate_for_element`,
  `compare_envelope`, registry.
* [Training guide](05_training_guide.md) — choosing sample counts,
  workers, scope; persistence; troubleshooting.
* [Plan + roadmap](https://github.com/Abhishek-Pathak-90/HELIX/blob/master/docs/plans/surrogates.md)
  — milestones, deferred work, references to prior art (Cheetah,
  HénonNet, RFQNet2, Bmad-X, …).

← [Convergence checklist](../12_validation/04_convergence_guide.md) ·
[Continue to GUI walkthrough →](02_gui.md)
