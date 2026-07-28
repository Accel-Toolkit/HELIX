# Quad field-error tolerance

**Study type:** random scaling of quadrupole gradient + the higher-order
multipole content (g3 sextupole, g4 octupole).

## Errors applied

```
ERROR_QUAD_NCPL_STAT 8 2 0  0  0  0  0   1   5   2   0   0
```

Per-seed Gaussian draws on every quadrupole:

* `gradient_rel ~ N(0, 1 %)`         — power-supply / current jitter
* `g3_rel       ~ N(0, 5 %)`         — sextupole-end-field variation
* `g4_rel       ~ N(0, 2 %)`         — octupole-end-field variation

The base lattice has small but non-zero `g3=0.05`, `g4=0.01` design values
on every quad — without those baselines, the `*_rel` errors have nothing
to scale.  Real PIP-II QUAD cards from the `.dat` files carry similar
content.

## What you'll see

* **σ_x ensemble envelope** — broadens monotonically through the
  channel as gradient errors accumulate phase mismatch
* **Halo growth** — the g3/g4 perturbations introduce x²- and x³-shaped
  kicks that drive the bunch tails outward (visible in the
  `Peak excursion` Results-tab tile, not just the σ envelope)
* **Eigenemittance ε₁/ε₂** — should grow noticeably more than ε_x/ε_y
  alone would suggest, because the nonlinear kicks couple the
  transverse phase planes

## Verifying that g3/g4 are actually doing something

Run twice: once with the directive as written (g3/g4 errors active),
once with `dG3 dG4` set to `0`.  The difference in final ε_x and
peak-excursion is the contribution from the wired-up multipole content
(was zero before 2026-05-08 — now real).

## How to run

Same workflow as
[`quad_alignment_tolerance`](../quad_alignment_tolerance/) — just open
this `.lgproj` and click **Run study** in the Errors tab.

## Tweaks worth trying

* **Pure linear**: set `dG3 dG4 dG5 dG6` all to 0 → only `gradient_rel`
  varies.  Compare ensemble σ_x to the full-multipole case.
* **Pure nonlinear**: set `dG` to 0 and only `dG3` to 5 % → see how
  much halo growth comes purely from sextupole jitter.
* **Increase to dG=5 %**: realistic worst case; transmission tail
  appears.
