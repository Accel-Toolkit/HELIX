# Beam-input jitter

**Study type:** the *lattice* is fixed; the *input beam* gets perturbed
seed-to-seed.  Use this when characterising source / chopper / LEBT
stability.

## Errors applied

```
ERROR_BEAM_STAT 2 0.1 0.1 0.5 0 0 0 5 5 0 0 0 0 1.0
```

Per-seed Gaussian draws on the input BeamConfig:

| Slot      | Symbol | σ            | Meaning |
|-----------|--------|--------------|---------|
| 1 (`r`)   | —      | —            | distribution code (2 = Gaussian) |
| 2 (`dx`)  | σ_dx   | 0.1 mm       | input centroid x |
| 3 (`dy`)  | σ_dy   | 0.1 mm       | input centroid y |
| 4 (`dφ`)  | σ_dφ   | 0.5°         | input phase centroid |
| 5–7       | —      | 0            | (dxp, dyp, dE — left at 0) |
| 8 (`dEx`) | %      | 5 %          | x-emittance growth (relative) |
| 9 (`dEy`) | %      | 5 %          | y-emittance growth (relative) |
| 10        | —      | 0            | dEz |
| 11–13     | —      | 0            | mismatch_x, mismatch_y, mismatch_z |
| 14 (`dIb`)| mA     | 1 mA         | beam-current jitter |

The factory regenerates a fresh BeamConfig per seed: a different
centroid, a different ε, a different I.  The lattice itself is
untouched.

## What you'll see

* **Centroid spread** at the lattice exit ≈ what you put in plus the
  optical magnification of the channel (so a 0.1 mm input can become
  0.5 mm at the exit if the optics has a high β at that point).
* **σ_x** envelope: spreads because of ε-growth (5 %) and the slight
  optical mismatch from current jitter altering the SC term.
* **Final transmission**: stays ~100 % unless the input centroid
  excursion exceeds the aperture in worst-case seeds.

## Combining with element errors

`BeamErrorDef` and `ErrorDef` (element errors) coexist — the next
example (`combined_realistic/`) layers both.

## How to run

`File → Open Project…` → `beam_input_jitter.lgproj` → **Errors** tab →
**Run study** with `n_seeds = 100`.
