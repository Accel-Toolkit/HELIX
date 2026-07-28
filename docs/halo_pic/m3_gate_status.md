# M3 gate status — honest verdict and rescope decision (2026-07-16)

## Bottom line

M3's acceptance gates **fail as originally written at the M2 operating point
(N = 20 000, 24³, 50 kicks/m)** — and the decisive experiments show they fail
for structural reasons, not tuning reasons. At this testbed scale the
cost-matched control (plain 48³ coarse) beats every corrector configuration
tried. The corrector concept itself is validated at the field level; the
regime where it can win is production scale (N ≥ 10⁵), not the down-scaled
testbed.

## The decision table (all 8-seed, along-s integrated errors vs the 10-seed
fine reference; tune from Hann/zero-padded FFT of σx(s); N = 20k unless noted)

| configuration            | wall    | dtune (°/cell) | q999 err | ε_rms err | halo err |
|--------------------------|---------|----------------|----------|-----------|----------|
| 24³ coarse               | 5.3 s   | +0.352         | 8.95 %   | 3.49 %    | 3.90 %   |
| 32³ coarse               | 7.0 s   | +0.178         | 7.34 %   | 2.02 %    | 2.48 %   |
| 40³ coarse (3-seed tune) | 8.9 s   | +0.120         | —        | —         | —        |
| **48³ coarse**           | 13.2 s  | **+0.056 ✓**   | 6.27 %   | 1.12 %    | 1.85 %   |
| 24³ + net4, gated (v3)   | 48.2 s  | +0.170         | 7.62 %   | 2.29 %    | 2.72 %   |
| 24³ + net4, gate off     | ~29 s   | **+1.0 (worse)** | 6–15 %  | 0.9–2.4 % | 4–6.6 %  |
| 24³ K=1 ceiling (fine field at coarse nodes, every kick) | 77 s | **+0.085 ✓** | 3.4/12.7 % (2 seeds) | 0.4 % | 1.2–2.5 % |
| sampling floor at N=20k (96³, converged) | — | —      | 4.74 %   | 1.00 %    | 1.28 %   |
| fine reference (N=200k, 96³, 100/m) | 113 s | 0 ± 0.013 | (1.60 % self-scatter) | (0.20 %) | (0.57 %) |

Grid tune error scales as Δ² (0.371 × (24/g)² predicts every row to ~15 %).
It is independent of cadence (50 vs 100/m identical) and of N (20k vs 100k).

## What was established (keeps its value)

1. **The defect is capturable.** σ-adaptive, beam-density-weighted δρ-basis
   projection restores the coarse core-field gradient error from 8.9 % to
   ~1 % at an anchor (oracle test). Weighted capture of the 96³-anchor defect
   is ~0.5–0.8 through the whole breathing cycle after the σ-adaptive fix
   (frozen-box basis collapsed to 0.10 mid-run — the beam breathes ~3.4×).
2. **The field-level correction path passes the tune gate.** K=1
   (fine-restricted field every kick) gives dtune = +0.085 ± 0.001 — under
   the 0.1° gate. The residual is gather/deposit-side error that node-value
   correction cannot remove; it matches the 48³-grid quality, i.e. the
   ceiling of ANY corrector at 24³ has essentially zero gate margin.
3. **The offline summary-feature MLP cannot drive it closed-loop.**
   On unseen seeds the net predicts ~70 % of the per-anchor defect
   (coefficient rel. err 0.30 vs 0.14 on training seeds), and applying it at
   every kick is unstable: early-run predictions (rel. err ~1.4) derail the
   trajectory, features leave the training manifold, dtune blows up to +1.0.
   This is precisely the distribution-shift failure documented by Um et al.
   (solver-in-the-loop) — offline training on uncorrected trajectories is
   insufficient; unrolled-through-the-loop training (planned M4) is REQUIRED,
   not a refinement.
4. **The safety machinery works.** The per-anchor α gate + kill switch
   correctly refused the harmful corrections (gate pass rate 0.39; the
   forced-on run confirms the refusals were right). α=0 fallback remains
   bit-identical to production PicSolver (pinned by test, 10/10 green).
5. **Two gates were mis-specified at N=20k.**
   - Tail ×2: sampling floor is 4.74 % on q999 vs 8.95 % coarse error →
     maximum possible improvement is ×1.89. The gate must be floor-aware
     (fraction of *correctable* error removed) or the testbed must run at
     larger N.
   - Overhead ≤1.15×: a 96³ anchor solve costs ~0.35 s vs a ~9 ms coarse
     kick; at 5–13 s total run time NO anchor cadence satisfies 1.15×. The
     anchor economics only close at production N (≥10⁵), where the coarse
     run costs minutes and anchors are a fixed small cost.

## Why the testbed regime is the wrong battlefield

At N = 20k the remaining tail error at 48³ (6.27 %) is only 1.5 % above the
sampling floor — nearly everything left is macroparticle sampling noise,
which no grid-defect corrector can touch by construction. Grid refinement
(24³ → 48³) costs only 2.5× wall and solves the tune gate outright. The
ML corrector cannot beat that trade at this scale, full stop.

At production scale the trade inverts: with N = 10⁵–10⁶ the deposit/gather
cost dominates and scales with N while the FFT scales with grid volume;
the sampling floor drops as 1/√N (→ ~1–2 % at N=10⁵, ~0.5 % at 10⁶) so the
grid+cadence defect once again dominates the error budget, and a 96³-every-
kick fine solve is genuinely expensive. That is the regime the original plan
arithmetic assumed (N_c ≈ 1e5); the N=20k testbed was a down-scale for
iteration speed that then silently became the operating point.

## Options (decision needed)

**A. Rescope the operating point to production N (recommended).**
Re-run the M2 cost/error surface at N = 1–2×10⁵ (Windows CUDA box for the
fine references), re-derive the operating point, restate the tail gate
floor-aware, keep all M3 infrastructure (it is scale-independent). The
small-N result above becomes the honest "cost-matched control" section of
the paper rather than its refutation.

**B. Proceed to M4 (unrolled training) at 32³/40³.**
Fixes the closed-loop instability by construction (train through the coarse
PIC against distribution-level losses). Higher effort; at 24³ even the
ceiling has no margin, so it must be combined with a grid bump — and at
N=20k the 48³-coarse control still wins on economics. Only sensible
combined with A.

**C. Declare the small-N negative result and stop HALO-PIC.**
The infrastructure (tail diagnostics, testbed, anchored solver, basis) keeps
standalone value; the paper claim would be abandoned.

My recommendation: **A, folding B in afterwards** (M4 was always the next
milestone; the closed-loop evidence now shows it is mandatory). The
same-lattice timing table (user requirement) is then produced at the
production operating point where the comparison is meaningful.

---

# M2b addendum — production-N go/no-go result (2026-07-16, same day)

Full data: `docs/halo_pic/m2b_production_n.json` (sweep script
`scripts/halo_m2b_sweep.py`; 6 seeds/frontier arm, 4/floor, 2/ceiling).

## Pre-registered decision rule

A corrector window exists iff the K=1 ceiling at 32³ (fine field at coarse
nodes every kick — the best ANY corrector can do) beats the plain-coarse
configuration of equal-or-greater wall-clock on tail metrics AND passes the
0.1°/cell tune gate with margin.

## Result: NO-GO on the tail metric — the rule is not met

| N | corrected-32³ realistic wall (K=32 anchors) | ceiling q999 | cost-matched plain q999 | ceiling dtune | cost-matched plain dtune |
|------|------|------|------|------|------|
| 100k | 24.5 s | 2.30 % | 2.35 % | +0.027 ✓ | +0.060 ✓ |
| 200k | 40.4 s | 3.24 % (2 seeds) | 2.40 % | +0.017 ✓ | +0.042 ✓ |

- Tune gate: the ceiling passes with real margin — but so does cost-matched
  plain coarse at both N.
- Tail (q999): a wash at N=100k, a loss (within noise) at N=200k. The
  correctable-above-floor tail error IS grid error, and plain grid
  refinement removes it at the same price as anchored correction.
- Where the ceiling genuinely wins ~2× at equal cost (ε_rms 0.31 vs 0.65 %,
  halo kurtosis 0.57 vs 1.12 % at N=100k) is exactly the rms/core-level
  physics that cheaper models can capture — not the halo-tail claim.

## The structural pinch (why no rescue regime exists in HELIX today)

The corrector pays only where the FIELD SOLVE dominates cost. HELIX's
measured cost profiles pinch the window from both sides:
- Small N / large grid (solve-dominated, e.g. MEBT profile: solve 29 %,
  deposit 7.5 %): tail metrics are floored by macroparticle sampling —
  nothing correctable above the floor (M3 result at N=20k).
- Large N (deposit-dominated, this sweep): grid refinement is relatively
  cheap (32³→64³ costs +52 % at N=200k) and buys the same error reduction.
GPU moves both costs down together and does not reopen the window.

## Disposition

- HALO-PIC's speedup claim is retired on evidence, at both scales, against
  the cost-matched control — the exact control the plan (M6) said kills
  most ML-PIC papers.
- Salvage: (1) the tested infrastructure (tail diagnostics, testbed,
  σ-adaptive weighted δρ basis, anchored trust-region solver with
  bit-identical α=0 fallback) remains in-tree as an experimental research
  tool; (2) the two-scale honest-control study is publishable negative-
  result material and the due-diligence section for any future attempt;
  (3) the profile's original recommendation — C++ field-map interpolation
  kernel (56 % of MEBT wall) — remains the real, boring, guaranteed ~2×
  production speedup.
