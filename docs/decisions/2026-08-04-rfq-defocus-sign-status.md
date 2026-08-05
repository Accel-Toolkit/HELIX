# RFQ defocus-channel sign correction — status and promotion criteria

**Date:** 2026-08-04
**Status:** EXPERIMENTAL / OPT-IN (temporary until field-validated)
**Feature:** `apply_rfq_geometry(mode="per_plane_dflip", d_scale=...)`
plus the arm-time defocus audit (committed in `22129c8`); defaults are
bit-identical to v1.3 behavior.

## What the correction is

The vane-geometry profile's axisymmetric RF-defocus (D) channel
accumulates the opposite sign vs the analytic two-term defocus on every
lattice measured so far (PXIE: 100 % of 203 cells, |ratio| 0.81;
MEHIPA example: 93 % of 317 cells, |ratio| 1.61). `per_plane_dflip`
applies the corrected sign, with an explicit `d_scale` magnitude.

## Evidence FOR the sign flip (why it exists)

* PXIE 5 mA identical-beam benchmark vs Toutatis: with the flip, the
  front-end loss burst appears at the correct location and size
  (13.7 vs 13.8 %), loss centroid 1.18 vs 1.12 m (shipped default:
  no burst, centroid 1.95 m), and 71 % of individually-lost particles
  match Toutatis's. sigma_y envelope agreement ~8 % median (was ±23 %).
* The TraceWin annex S-sign formulas (image-decoded) carry negative
  signs where the production convention uses positive.
* The MEHIPA lattice reproduces the same sign structure independently.

## Why it is NOT the default (and must not be yet)

* The arm-time audit compares the profile channel against HELIX's own
  TW-matrix-calibrated card convention — it proves the two paths
  DISAGREE, not which one is physically correct (independent review,
  2026-08-04). Ground truth requires measured Toutatis fields.
* The magnitude is machine-dependent (0.81 vs 1.61, opposite sides of
  unity) and `d_scale = 1/audit-ratio` was shown to be an invalid
  calibration rule; no per-lattice tuning is acceptable as policy.
* The project acceptance criterion (both transverse envelopes matching
  Toutatis, curves-on-curves) is not met: sigma_x carries a real
  ±20-45 % amplitude beat attributed to the in-cell phase structure.

## Promotion criteria (to default, target v1.6) — ALL required

1. Licensed-Toutatis field export (OutputFileField line scans) decoded
   and validated; the true defocus channel measured from fields.
2. In-cell kick model derived/corrected against those fields such that
   the audit ratio is ~1.00 on every imported lattice WITHOUT a knob
   (this retires `d_scale` entirely).
3. sigma_x beat closed to the sigma_y level in the 5 mA benchmark;
   loss locations and transmission revalidated (0 and 5 mA ladders).
4. Pinned-test updates + adversarial review round.

If the field-validated derivation contradicts the flip, the mode is to
be REMOVED, not defaulted.

## Provenance

Benchmark campaign records 2026-08-03/04 (external harness archive
`RFQ_campaign_harness`, e40-e78): loss-location forensics, snapshot-
semantics decode of coupled-code .plt data, two-lattice audit results,
independent read-only code review, and regression checks confirming
defaults remained bit-identical throughout (July-26 worktree vs HEAD:
identical to every digit on the PIP-II MEBT->BTL line; MEBT+HWR
benchmark reproduced at documented tolerances).
