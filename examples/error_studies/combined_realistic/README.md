# Combined realistic tolerance budget

**Study type:** the kitchen sink — every error category from the
previous four examples stacked into one lattice.  Use as a template
for a real PIP-II-style tolerance run.

## Errors applied

```
ERROR_GAUSSIAN_CUT_OFF 3                    ; truncate every Gaussian at ±3σ
ERROR_QUAD_NCPL_STAT 6 2  0.2 0.2 0 0 0  0.5 0 0 0 0   ; quad align + ΔG
ERROR_CAV_NCPL_STAT  2 2  0.3 0.3 0 0    1.0 1.0 0    ; cav align + V + φ
ERROR_BEND_NCPL_STAT 1 2  0.5 0.5 0 0 0  0.5 0        ; bend align + Δfield
ERROR_BEAM_STAT      2    0.1 0.1 0 0 0 0 0 0 0 0 0 0 1.0  ; input centroid + I
```

| Element type | Alignment σ | Field error σ | Per-seed budget |
|---|---|---|---|
| Quadrupole   | 0.2 mm dx, dy | 0.5 % gradient | 6 quads |
| RF gap       | 0.3 mm dx, dy | 1 % voltage, 1° phase | 2 gaps |
| Dipole       | 0.5 mm dx, dy | 0.5 % field | 1 bend |
| Input beam   | 0.1 mm centroid x/y, 1 mA current | — | once per seed |

All distributions Gaussian truncated at ±3σ.

## What you'll see

* **Transmission histogram** — likely a tail extending below 99 %
  because each error category contributes; the longest tail
  comes from the dipole-misalignment-driven orbit excursion through
  the dispersion of the bend.
* **σ_x / σ_y ensemble envelopes** — visibly broader than any
  single-error case.
* **Centroid drift** — has both a deterministic component (from the
  bend's misalignment) and a stochastic component (from the quads'
  alignment + the input beam's centroid jitter).

## Decomposing the budget

To find which error dominates, comment out four directives at a time
and re-run.  The single-error transmission spread relative to the
combined run gives the % contribution of each category.

## How to run

`File → Open Project…` → `combined_realistic.lgproj` → **Errors** tab →
**Run study** with `n_seeds = 100`.

## Scaling up

To approximate a PIP-II linac stage, drop in a real `.dat` (e.g.
`examples/pipii/mebt+hwr+ssr1+ssr2/mebt+hwr+ssr1+ssr2.dat`) and
prepend the same four `ERROR_*` directives — the parser handles
the directive ordering automatically and the count parameter (`N`)
expands to cover every quad / cavity / bend in the file.
