# Plan: HELIX surrogate elements — envelope-mode first, GUI + CLI workflow

> Revised plan (2026-05-24).  Supersedes the v1 draft.  Phase-1 code
> audit and Phase-2 critical review identified real gaps in the v1 — they
> are addressed up front in **"What the v1 plan got wrong"** below.

## Context

The user wants the ability — in BOTH the GUI and the CLI — to select
any element of a loaded lattice, train an ML surrogate that mimics it
using the **existing HELIX tracker as ground truth**, swap the
surrogate into the lattice, and **compare** surrogate-vs-baseline
results.  Initial integration is the **envelope mode** (the 6×6 matrix
path); multiparticle + gradient matcher come later.

This is the natural follow-on to the v3 differentiable PIC and an
extension of the user's own
[Linac_Gen work, arXiv:2406.16630](https://arxiv.org/abs/2406.16630)
(which already demonstrated ~10× phase-space-matching speedup on
PIP-II with surrogate-style models).

## What the v1 plan got wrong — verified facts

1. **Tracker dispatch is isinstance-based**
   (`linac_gen/tracking/tracker.py:221-232`), not a polymorphic seam.
   A generic `SurrogateElement(Element)` would be silently skipped.
   Surrogates **must** subclass the concrete types:
   `FieldMapElement`, `ThinKickElement`, `TransferMapElement`,
   `PassiveElement`.
2. **Gradient matcher composes per-element 6×6 matrices** via
   `element_matrix_torch` (`linac_gen/tracking/torch_tracking.py:106-127`).
   An `nn.Module` surrogate does NOT auto-integrate; it needs an
   explicit arm there.  ("Autograd just works" was wrong as written.)
3. **Envelope solver uses sub-element slices**
   (`linac_gen/tracking/envelope.py:706-907` calls
   `fitted_matrix_slice(ref, ds, z_from_mm)`).  Full-element-only
   surrogates need a slice fallback.
4. **TraceWin parser is positional**
   (`linac_gen/io/tracewin_syntax.py:87-98`); inline `surrogate=<tag>`
   will be silently ignored or crash.  **Side-car config**, not inline
   kwarg.
5. **Stateful semantics** — `FieldMap3D._sync_offset_deg`,
   `.scc` profiles, `reset_run_state()` — the surrogate must replicate
   them or refuse.  v1 glossed over this.
6. **Verification gates** (<1 % σ MAPE) were too loose.

## User-stated scope (this plan)

- **Element selection** in GUI **and** CLI.
- **Training pipeline** generates samples by passing them through the
  existing element tracker (`fitted_matrix(ref)` for envelope-mode).
- **Envelope-mode integration first.**
- **Swap + compare** workflow built in.

## Gaps flagged in the user's request — defaults applied

| Gap | Default (override at any time) |
|---|---|
| Per-instance vs per-type surrogate | **Per-instance first** (one trained model per specific element).  Per-type generalisation = future work (the SOTA admits it's unsolved for non-linear elements). |
| Training param-sweep range | LHS over element params ±20 % around as-loaded values + LHS over incoming Σ moments, configurable in the train dialog. |
| Comparison metrics | σ_x / σ_y / σ_φ / σ_W overlay curves + end-of-lattice rel.diff table + wall-clock.  Returned as a dataclass + optional PNG. |
| Weight storage policy | Per-project, under `linac_gen/surrogates/weights/<lattice-hash>/<element-key>.pt`, with `metadata.json` (training seed, HELIX commit SHA, lattice hash, element index/name, scope bounds, val MAPE). |
| OOD admissibility | Runtime check of incoming params & Σ against metadata's scope; outside scope → warn + fall back to reference, **never silent**. |
| Stateful elements | First pass **refuses** to surrogate elements with active state; surface a clear error in the train dialog. |
| Sub-element slices | First pass: full-element matrix only; envelope falls back to a numeric (numpy-FD) Jacobian for slices on surrogate output. |
| Mid-run safety | "Use surrogates" toggle locked while a sim / matcher is active. |
| Reproducibility | `metadata.json` is **required**; CI gate checks (seed + commit SHA + lattice hash + RK4-reference checksum). |

## State of the art (synthesised research)

- **MLP + linear residual** (Cheetah's `b_linear + ΔΣ_NN(b_in |
  params)` pattern) is the fastest path to a working surrogate.
  4×256 MLP, >1000× speedup over Ocelot PIC
  ([Kaiser 2024, PRAB 27, 054601](https://arxiv.org/abs/2401.05815)).
- **Symplectic-by-construction** (HénonNet, SympNet) — exact
  symplecticity layer-by-layer; needed for long lattices and matcher
  iteration.  Demonstrated on LANSCE CCL Tank 5.1
  ([IPAC 2023 WEPA078](https://proceedings.jacow.org/ipac2023/pdf/WEPA078.pdf),
  [PRAB 2025 bhpv-bcqk](https://journals.aps.org/prab/abstract/10.1103/bhpv-bcqk)).
- **Polynomial / Lie-map NNs** with a symplecticity-regularisation
  loss (Ivanov & Agapov,
  [PRAB 23, 074601](https://arxiv.org/abs/2007.03555);
  Wagner DLMN, [PRAB 28, 024601](https://arxiv.org/abs/2408.11677)).
- **RFQ surrogate** (Koser et al., RFQNet2,
  [PRAB 2023](https://arxiv.org/abs/2210.11451)): 6×100 MLP,
  217 k samples; 0.97 % transmission MAPE, 1.8 % output energy.
- **Training data**: Latin Hypercube + Sobol standard; adaptive LHS
  cuts data 30-50 %; active learning for high-dimensional sweeps.
- **Differentiability**: `torch.nn.Module → autograd` works in
  Cheetah / Bmad-X / JuTrack out of the box.  Tricks: element params
  as `requires_grad=True` leaves, FP64 weights, smooth activations
  (SiLU / Tanh, not ReLU).
- **Open problems we'll respect, not try to solve:**
  - Cross-element generalisation is unsolved for non-linear cases.
  - Halo / tail particles systematically mispredicted — keep RK4 as
    authority for aperture / loss studies.
  - Sim-to-real transfer needs measured data.
  - Out-of-distribution silent failure — addressed by mandatory
    admissibility check.

## Approach — eight milestones

### M1 — Surrogate infrastructure
- New package `linac_gen/surrogates/`:
  - `base.py` — per-element-type mixins (multiple inheritance is
    safe; `Element` uses plain metaclass):
    - `SurrogateFieldMap(FieldMapElement, torch.nn.Module)` —
      implements `track_rk4(beam, ds)` (NumPy in-place contract for
      the multiparticle tracker) AND `fitted_matrix(ref)`,
      `fitted_matrix_slice(ref, ds, z_from_mm)` for the envelope
      solver.  Forward = small MLP predicting a flattened 6×6
      transfer matrix given (`ref.E`, `ref.β`, `ref.γ`, element
      params).  The numpy contracts wrap a `torch.no_grad()` forward.
    - `SurrogateRfGap(ThinKickElement, torch.nn.Module)` — abstract
      placeholder for M7 (locks the hierarchy now).
  - `training.py` — LHS / Sobol data generation
    (`scipy.stats.qmc`).  Generates input samples, runs them through
    the **existing** element's `fitted_matrix(ref)` for envelope
    ground truth; trains the MLP head; validates on 20 % held-out.
    Persists `(weights.pt, metadata.json)`.
  - `registry.py` — given a lattice + element index/name, look up
    a trained surrogate by `(lattice-hash, element-key)`; lazy-load
    weights.  Provides the admissibility test.
  - `compare.py` — runs the envelope solver twice (baseline +
    surrogate-enabled), returns a diff report.
- `tests/surrogates/test_{base,training,registry,compare,admissibility}.py`.

### M2 — Train a first surrogate (user-selected element)
- For the initial driver test we use the MEBT buncher `FieldMap3D`
  from `examples/pipii/mebt/`.  Once GUI/CLI are in place the user
  picks any element.
- ~50 k LHS samples × parameter sweep × incoming-Σ sweep; pass each
  through `FieldMap3D.fitted_matrix(ref)` to get the 6×6 reference;
  train MLP-residual on 6×6 matrix outputs.
- **Acceptance gates (per-element):**
  - Matrix Frobenius rel.diff <0.5 % in-scope.
  - Symplecticity defect ‖Mᵀ S M − S‖_F / 6 < 1e-3.
  - Val MAPE recorded into metadata.json.

### M3 — Envelope-mode integration
- Hook the surrogate into `envelope.py`'s per-element matrix call:
  when `_assemble_step_matrix` encounters an element with a
  registered + admissible surrogate, route `fitted_matrix(_slice)`
  through the surrogate (with numpy-FD slice fallback if the
  surrogate isn't slice-capable).
- **End-to-end gate — MEBT+HWR envelope comparison**:
  - σ_x / σ_y / σ_φ / σ_W rel.diff <1 % everywhere along s.
  - End-of-line emittance growth <2 %.
  - Phase-advance drift <0.5° per cell.
  - Wall-clock per `fitted_matrix` ≥50× faster than RK4 baseline.

### M4 — Comparison framework
- `surrogates/compare.py` produces a structured diff report (Σ
  curves + table + wall-clock + scope-compliance flag).  Reusable
  from CLI and GUI.

### M5 — CLI workflow
- New `linac-gen` subcommands:
  - `linac-gen surrogate-train --lattice <.dat> --element
    <name|index> [--samples N] [--epochs E] [--out <dir>]`
  - `linac-gen surrogate-compare --lattice <.dat> --element <name>
    [--out <png|html>]`
  - `linac-gen run-envelope --lattice <.dat> --use-surrogates`
- Thin wrappers; all logic in `training.py` + `compare.py`.

### M6 — GUI workflow
- New **Surrogates tab** at
  `gui/linac_gen_gui/interphase/tabs/surrogates_tab.py`:
  - Element dropdown from `state.lattice.elements`.
  - **Train surrogate** button → modal dialog (LHS count, epochs,
    architecture, sweep ±%).  Training runs in a `QThread` worker
    with progress bar; on completion a row is added to the trained-
    surrogates table.
  - Trained-surrogates table: name, training scope, val MAPE, Use
    checkbox, Compare button.
  - Toolbar badge "Surrogates ACTIVE" when any are toggled on.
  - Toggle disabled while a sim / matcher runs.
- New manual page + workflow entry in `docs/manual/10_gui/`.

### M7 — Multiparticle + gradient-matcher integration (deferred)
- Multiparticle: add `SurrogateFieldMap` to `tracker.py:221-232`
  isinstance chain (it's already a `FieldMapElement` subclass, so it
  routes through `_track_field_map`, which calls `track_rk4(beam,
  ds)`, which the surrogate overrides).
- Matcher: extend `element_matrix_torch` (`torch_tracking.py`) with
  an arm for `SurrogateFieldMap` calling
  `surrogate.fitted_matrix_torch(kin)` — autograd-differentiable.

### M8 — Per-type generalisation (future work)
- One model that works across all instances of an element type.
  Open research problem per the SOTA review; do not bake into the
  schedule.

## Critical files

**Create:**
- `linac_gen/surrogates/{__init__,base,training,registry,compare,cli}.py`
- `linac_gen/surrogates/weights/`  (per-project trained-weight store)
- `gui/linac_gen_gui/interphase/tabs/surrogates_tab.py`
- `gui/linac_gen_gui/interphase/dialogs/surrogate_train.py`
- `tests/surrogates/test_*.py`

**Modify (additive, default off):**
- `linac_gen/tracking/envelope.py` — at the `fitted_matrix(_slice)`
  call sites, check the registry; dispatch through the surrogate
  when admissible, else RK4.
- `linac_gen/cli/__init__.py` (or equivalent) — register
  `surrogate-*` subcommands.
- `gui/linac_gen_gui/interphase/app.py` — add the surrogates tab;
  toolbar badge; lock the "Use surrogates" toggle during active runs.
- `linac_gen/elements/field_map_3d.py` — refuse surrogate
  replacement when stateful flags are dirty.

**Reuse, do not modify:**
- `linac_gen/elements/field_map_3d.py:fitted_matrix(ref)` —
  the surrogate's ground-truth source.
- `linac_gen/tracking/envelope.py` inner loop — only the matrix
  call site changes.
- `linac_gen/io/tracewin_parser.py` — surrogate enablement comes
  via side-car config, NOT a new `.dat` token.

## Verification

1. **M2 per-element gates**: matrix Frobenius rel.diff <0.5 %;
   symplecticity defect <1e-3; val MAPE recorded.
2. **M3 envelope gates** (MEBT+HWR):
   - σ moments rel.diff <1 % everywhere along s.
   - End-of-line emittance growth <2 %.
   - Phase-advance drift <0.5° per cell.
   - Wall-clock per `fitted_matrix` ≥50× faster.
3. **OOD safety**: out-of-scope inputs trigger a runtime warning +
   fall back to RK4 (test: deliberately feed an off-scope ref
   energy and assert the warning).
4. **Reproducibility CI gate**: every committed weights file must
   have a `metadata.json` with seed + commit SHA + lattice hash +
   RK4-reference checksum; a CI script re-runs training and
   confirms weights match within 1e-5.
5. **Non-breaking**: with surrogates registered but disabled,
   envelope output is **bit-identical** to today.  All existing
   tests stay green.
6. **GUI tests** (`tests/gui/`) stay green; new surrogates-tab
   tests added.

## Honest risks

- **Per-instance retraining cost** — every lattice / field-map /
  element-setting change invalidates the surrogate.  Mitigated by
  hash-keyed scope in the registry.
- **OOD silent failure** — addressed by mandatory admissibility check.
- **Stateful-element opacity** — refused by M1 design; revisit if
  it bites.
- **Envelope-slice fidelity** — full-element matrix only in v1;
  per-slice error reported in the comparison.  Expect 1-3 % error
  near element boundaries.
- **Reproducibility** — every weights file ships with metadata.json;
  CI verifies.  Without it, surrogates are scientifically
  unverifiable.

## Estimated effort

| Milestone | Effort |
|---|---|
| M1 infrastructure | ~1 week |
| M2 first surrogate | ~1 week |
| M3 envelope integration | 3–5 days |
| M4 comparison framework | 2–3 days |
| M5 CLI | 2–3 days |
| M6 GUI | ~1 week |
| M7 MP / matcher (deferred) | ~1 week |
| M8 per-type (future work) | — |

**Total to a working envelope-mode surrogate the user can train +
swap + compare from GUI/CLI: ~3–4 weeks.**

## References

- Cheetah / SC surrogate — Kaiser et al., PRAB 27, 054601 (2024) —
  [arXiv:2401.05815](https://arxiv.org/abs/2401.05815) ·
  [GitHub](https://github.com/desy-ml/cheetah).
- RFQNet2 — Koser et al., PRAB 2023 —
  [arXiv:2210.11451](https://arxiv.org/abs/2210.11451).
- ImpactX 15-stage LPA surrogate — Sandberg et al., PASC '24 —
  [arXiv:2402.17248](https://arxiv.org/abs/2402.17248) ·
  [docs](https://impactx.readthedocs.io/en/25.11/usage/examples/pytorch_surrogate_model/README.html) ·
  [Zenodo weights](https://zenodo.org/records/10810754).
- Symplectic neural surrogate for beam dynamics — Huang et al.,
  IPAC 2023 ([WEPA078](https://proceedings.jacow.org/ipac2023/pdf/WEPA078.pdf))
  + PRAB 2025 ([bhpv-bcqk](https://journals.aps.org/prab/abstract/10.1103/bhpv-bcqk)).
- Polynomial-NN Taylor maps + symplectic-regularisation loss —
  Ivanov & Agapov, PRAB 23, 074601 —
  [arXiv:2007.03555](https://arxiv.org/abs/2007.03555).
- Matrix Lie maps & NNs —
  [arXiv:1908.06088](https://arxiv.org/abs/1908.06088).
- DLMN (physics-shaped Lie-map NN, SIS18/GSI) — Wagner et al.,
  PRAB 28, 024601 —
  [arXiv:2408.11677](https://arxiv.org/abs/2408.11677).
- Differentiable-tracking benchmark — PRAB 2025 —
  [arXiv:2507.08476](https://arxiv.org/abs/2507.08476).
- JuTrack (Julia + Enzyme AD) —
  [arXiv:2409.20522](https://arxiv.org/abs/2409.20522).
- LCLS-II injector CNN + sim-to-real transfer — Gupta et al. —
  [arXiv:2103.07540](https://arxiv.org/abs/2103.07540).
- SympNet / symplectic Taylor-NN —
  [arXiv:2005.04986](https://arxiv.org/abs/2005.04986) ·
  Daniel et al., PoP 32, 103901 (2025) —
  [link](https://pubs.aip.org/aip/pop/article/32/10/103901/3365622/).
- SPINI (PINN + symplectic integrator) — Liang et al., Sci. Reports
  2025 — [link](https://www.nature.com/articles/s41598-025-28710-2).
- Bmad-X (PyTorch element library) —
  [GitHub](https://github.com/bmad-sim/Bmad-X).
- Fully differentiable accelerator modelling — Gonzalez-Aguilera
  et al., IPAC 2023 —
  [link](https://inspirehep.net/files/2e85e29a660866547e338536c0b69532).
- Linac_Gen (user's own PIP-II surrogate work) —
  [arXiv:2406.16630](https://arxiv.org/abs/2406.16630).
