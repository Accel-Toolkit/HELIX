# HALO-PIC M2 — coarse-PIC error surface (mismatched-FODO testbed)

**Setup.** 24-cell FODO channel (`examples/halo_fodo.dat`), H⁻ 2.12 MeV, 5 mA,
breathing-mode mismatch m = 1.4, SC-matched input Twiss.  Reference: N = 100 000,
96³ grid, 100 SC kicks/m — **130.5 s wall** on the M2 machine (Apple Silicon,
CPU FP64, threaded scipy FFT).  Sweep: N_c ∈ {2, 5, 10, 20, 50}k ×
grid ∈ {24³, 32³, 48³} × cadence ∈ {25, 50, 100}/m × 3 seeds
(`docs/halo_pic/coarse_sweep.csv`, `scripts/halo_coarse_sweep.py`).
Physics context: at this mismatch the fine run grows ε_rms ×1.55 and
ε_99.9 ×3.01 over 9.6 m — the tail outgrows the core by a factor 2.

## Findings

1. **The sampling floor is hard and high.** At N_c = 2 000 and 5 000, the
   ε_99.9 error sits at **15–17 % regardless of grid or cadence** — pure
   macroparticle-sampling error, invisible to any grid-defect corrector
   (exactly the adversarial-review objection #1).  ⇒ **N_c ≥ 10 000 is
   mandatory**; 20–50 k preferred.

2. **The grid + cadence defect — the corrector's addressable target — is
   real and separable at N_c ≥ 10 k.**  At N_c = 10 k: ε_99.9 error falls
   from 7.7 % (24³, 25/m) to 2.4 % (48³, ≥50/m); rms-ε error from 4.4 %
   to 0.9 %; kurtosis-halo error from 6.3 % to 2.9 %.  The ~5 % tail-error
   delta between the cheap and converged grid/cadence settings at fixed N
   is learnable signal.

3. **Cadence converges at 50 kicks/m.** 50 → 100/m changes nothing
   (< 0.02 % everywhere); 25 → 50/m is worth ~1.5–2 % on both core and
   tail.  Never spend beyond 50/m on this lattice class.

4. **Tail-estimator noise is the measurement-design constraint.**  With
   3 seeds, the endpoint ε_99.9 comparison carries ±2–7 % seed scatter
   (independent sampling at both fidelities; N = 50–100 k puts only
   50–100 particles beyond the 99.9 % contour).  The near-converged point
   (50 k, 48³, 100/m) still shows 3.2 ± 2.1 % — that residual is mostly
   estimator noise, setting a **~2–3 % tolerance floor** for endpoint
   tail comparisons.  ⇒ M3 acceptance must use ≥ 8 seeds and/or
   trajectory-integrated (along-s) tail errors, not 3-seed endpoints.

5. **Cost model (measured, per full 9.6 m run):** fine 130.5 s;
   coarse examples: (10 k, 24³, 25/m) 3.7 s → **35×**;
   (20 k, 24³, 25/m) 4.6 s → **28×**; (50 k, 48³, 50/m) 20.7 s → 6.3×.
   Cheap-vs-converged grid/cadence at fixed N costs 3–6× — that is the
   wall-clock the corrector can reclaim on top of the N and grid savings
   vs the fine reference.

## Gate verdict — PASS, with two amendments

An operating point exists where rms observables are within a few % but the
tail carries a several-times-larger, grid/cadence-dominated error:

> **Operating point for M3: N_c = 20 000, grid 24³, cadence 25/m**
> (4.6 s, 28× vs reference; rms-ε err ≈ 4 %, ε_99.9 err ≈ 6.5 %, halo
> err ≈ 4 %) — corrector target: recover converged-grid quality
> (rms 0.4 %, halo 1.3 %, tail to the ~2–3 % noise floor).

Amendments to the plan:
- (a) Corrector acceptance metrics use ≥ 8 seeds and along-s integrated
  errors (endpoint ε_99.9 at 3 seeds cannot resolve the target).
- (b) N_c below 10 k is excluded from all HALO-PIC configurations
  (sampling-dominated regime — the corrector must refuse via config
  validation rather than silently underperform).

## Reproduction

```
PYTHONPATH=. python3 scripts/halo_coarse_sweep.py \
    --ref-config 100000,96,100 --out docs/halo_pic/coarse_sweep.csv
```
