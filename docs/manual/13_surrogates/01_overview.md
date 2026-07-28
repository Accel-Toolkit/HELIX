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
  `fitted_matrix(ref)` — the envelope-mode contract.  Slice-aware
  surrogates (for the SC-engaged path) are future work.
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
    n_samples=50_000, ref_w_kin_range=(2.0, 2.5),
    param_ranges={"ke": (0.054, 0.082), "phase": (-110, -70)},
    out_dir="weights/FMAP_001",
    n_workers=12,                  # CPU parallel data gen
)
```

## When to use a surrogate

| Use case | Worth surrogating? |
|---|---|
| Single envelope pass, novel lattice | ❌ — training cost > savings |
| Parameter scan (10² runs, same elements) | ✅ — break-even at ~5-10 runs |
| Matching loop (100s of forward passes) | ✅✅ — payoff is large |
| Tolerance study (50-1000 seeds) | ✅✅ — payoff dominates |
| Differentiable matching (gradient path) | ⚠️ — needs M7, see below |

The break-even point is roughly: surrogate training takes ~1 h
production (50 k samples, 1 CPU-worker minute per element) and
saves ~880 ms per envelope `fitted_matrix` call on the MEBT buncher.
A 50-seed tolerance study × 4 bunchers × 80 calls/seed = 16 000
RK4 calls saved → ~4 hours of CPU.  Net win: ~3 hours per study.

## Scope of accuracy

Acceptance gates (per `docs/plans/surrogates.md`, milestone M2):

| Metric | Smoke cycle (200 × 40) | Production cycle (50 000 × 300) |
|---|---|---|
| Matrix Frobenius rel.diff (in-scope) | ~0.3–0.5 % | < 0.1 % |
| Val MAPE | ~3–6 % | < 1 % |
| Symplecticity defect ‖MᵀSM−S‖_F / 6 | not enforced | < 1e-3 |
| Per-call wall-clock | 0.15–0.30 ms | same |

The smoke cycle is for "does the pipeline work" sanity checks.  For
science runs, use the production cycle.

## The M3 honest limitation

The shipped surrogate intercepts `fitted_matrix(ref)` cleanly but
delegates `fitted_matrix_slice(ref, ds)` to the wrapped RK4.  This
matters because:

* **No-SC envelope** — the field-map handling skips
  `fitted_matrix_slice` entirely and uses a single full-element
  matrix per element.  Surrogate engages, full speedup.
* **SC-engaged envelope (current > 0)** — the SC bundling code in
  `envelope.py` slices the field map so SC kicks can be inserted
  between slices.  Each slice call falls back to RK4 in the
  shipped surrogate, so the surrogate is **effectively bypassed**:
  measured on the MEBT at 5 mA, **zero** NN calls are made and there
  is **no speedup**.  Surrogate acceleration currently applies to the
  no-SC envelope path only.

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
| **Engage in MP runs** (master, M7) | OFF | When OFF: MP runs ignore all registered surrogates -- bit-identical to baseline.  When ON alone: the surrogate sits in the dispatch chain as a safe delegate (`track_rk4` calls `wrapped.track_rk4`); still bit-identical to baseline, ~1 % dispatch overhead. |
| **Experimental: linear-matrix fast path** (M7-followup) | OFF | When ON (and master also ON): per-substep `wrapped.track_rk4` is replaced by an analytic ref-advance + batched `M_slice @ particles`.  Linear-matrix accuracy applies; halo dynamics may drift. |

The fast path is **double-opt-in** because its accuracy depends on
training quality and lattice content; it's a research-quality
feature, not a drop-in replacement for full RK4.

**Measured behaviour on PIP-II MEBT+HWR** (20 field-maps, 1000
gaussian particles @ 5 mA, M3 Max 14-core):

| Configuration | σ_x rel.diff | σ_z rel.diff | ε_nx rel.diff | Speedup |
|---|---|---|---|---|
| Master OFF (no surrogates engaged) | -- | -- | -- | baseline (1×) |
| Master ON, fast-path OFF | 0.00 % | 0.00 % | 0.00 % | 0.99× |
| Master ON, fast-path ON, **smoke training** (200 sa × 40 ep) | 4.7 % | **28.4 %** | 1.5 % | 1.22× |
| Master ON, fast-path ON, **moderate training** (2000 sa × 100 ep for cavities, smoke for solenoids) | 4.9 % | **4.7 %** | 1.9 % | 1.66× |

Key observations:

* **σ_z is highly sensitive to cavity training quality** -- moderate
  training of the 12 RF cavities dropped σ_z error from 28 % to
  4.7 % (6× improvement).  Smoke training is fundamentally
  insufficient for the fast path on RF-rich lattices.
* **σ_x/σ_y plateau around 5 %** -- driven by the MEBT bunchers
  (zero-phase, sparse-matrix RF cavities, val MAPE inflated by
  near-zero denominators).  Further reducing this needs
  production-quality training (50 000 sa × 300 ep, ~12 h / element)
  or per-element narrowing of `ref_w_kin_range`.
* **Speedup limited by the 8 1-D solenoids** which lack the
  required wrapped helpers (`_phasor`, `_sample_onaxis`,
  `_scale_factor`) and fall back to baseline RK4.  RF-cavity-dense
  lattices (e.g., HWR/SSR1 cryomodule strings without bunchers)
  should see substantially higher speedups (~3-5×).

**When to use the fast path**

| Use case | Acceptable? |
|---|---|
| Parameter scans for matching (σ moments are the only metric) | ⚠️ At ~5 % σ_x error, only if your matching residual is well above 5 % |
| Tolerance studies measuring σ jitter under errors | ❌ The surrogate's ~5 % bias would masquerade as the tolerance signal |
| Halo / aperture / beam-loss studies | ❌ Linear matrix misses nonlinear halo physics |
| TraceWin parity / publication-grade physics | ❌ Use baseline RK4 |
| Speed-priority exploration where you'll re-run RK4 for final results | ✅ Use the fast path; always click `Compare MP` first to validate the bias on YOUR lattice + beam |

Out-of-scope inputs fall back to the wrapped element automatically,
so the run is always **physically defensible** -- never silently
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
delegate per element whenever any helper is missing -- so a future
break would degrade performance, never correctness.

The matrix log/expm slice approximation (`_slice_matrix` →
`_get_cached_slice_from_log`) caches `logm(M_full)` once per
element entry and only pays `expm` once per `(ds, w_kin)`
combination across the SC bundle.  Without this cache the
`expm` cost (~5 ms) per substep would dwarf any savings.

!!! warning "Stateful elements not yet supported"
    Elements with active per-instance state
    (`FieldMap3D._sync_offset_deg`, `.scc` profile selection,
    `reset_run_state` dirty bits) currently get a deep-copy in the
    surrogate but the trained MLP does not see the state as an
    input.  For elements whose physics depends on accumulated
    state — multi-cell DTL gaps in particular — this is a known
    gap; the surrogate may silently mispredict.  Refused by
    construction for RFQ classes; treat with care for any
    FieldMap3D with non-default sync handling.

## Errors and matching (M8)

The matrix below reflects what's wired through to the surrogates in
both env and MP modes after M8:

| Workflow / Error type | Envelope mode | MP mode (fast path) |
|---|---|---|
| Beam errors (centroid, current jitter, Twiss mismatch) | ✅ Σ propagation invariant under beam-shape changes | ✅ tracker handles per-particle |
| Field-strength errors (`ke`, `kb`, `phase` jitter) -- IN scope | ✅ surrogate reads post-error attr via `getattr(self._wrapped, "ke")` | ✅ same |
| Field-strength errors (`ke`, `kb`, `phase` jitter) -- OUT of scope | ✅ `OutOfScopeError` → fall back to wrapped RK4 | ✅ same |
| **Element tilt** (`tilt_deg`) | ✅ M8: `envelope.py` wraps each element's matrix with `R_out @ M @ R_in` mirroring `tracker.py:208-219` | ✅ M7: `tracker.py:196-254` wraps `_track_field_map` and therefore the surrogate's substep loop |
| Element offset (`dx`, `dy`) | ✅ NO-OP for Σ by design (Σ is invariant under rigid translation around the centroid); matches TraceWin's envelope semantics | ✅ tracker applies per-particle translate |
| `dz`, `pitch_deg`, `yaw_deg` | ⚠️ Tier-1 not implemented in baseline tracker either -- stored but ignored. Future work. | ⚠️ same |
| **Matching (`gradient` algorithm) through a surrogated cavity** | ✅ M8: `element_matrix_torch` has a `SurrogateFieldMap` arm; `check_gradient_supported` accepts it. Surrogated cavities act as passive autograd-differentiable blocks the gradient flows through. | N/A -- matcher is env-mode |
| Matching tuning surrogated `ke` / `phase` directly | ⚠️ Out of scope this iteration. Surrogated cavities are passive in the gradient flow; matcher tunes only quads / solenoids / dipoles. Adding `ADJUST_KE` etc. is a separate plan. | N/A |
| Matching via `least_squares` / `differential_evolution` / `dual_annealing` (numpy paths) | ✅ Already works with surrogates -- the NumPy `fitted_matrix(ref)` path is `torch.no_grad`-wrapped and returns a plain ndarray that the scipy optimisers consume directly. | N/A |
| TraceWin parity / publication-grade physics | ❌ Use baseline RK4 -- the fast path's linear-matrix approximation is documented as research-grade. | ❌ same |

Empirical env-vs-MP tilt parity (driver `/tmp/test_env_tilt.py`, MEBT
drift-quad-drift mini-lattice, 10 000 particles): σ_x rel.diff
**2.7e-5** across tilt_deg ∈ {0, 2, 5, 10}° -- far below the
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
