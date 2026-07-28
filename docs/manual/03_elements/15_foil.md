# Foil

A thin material foil that scatters and slows the beam.  Used to model
stripper foils (H⁻ → p⁺ charge exchange is the canonical PIP-II use)
and any degrader / scattering foil in a beam line.

## TL;DR (TraceWin users)

TraceWin has no foil card, so HELIX declares one with a
comment-prefixed extension that stays invisible to TraceWin:

```
; HELIX_FOIL <name> <material> <thickness_ug_cm2>
; HELIX_FOIL STRIP C 600
```

| Field | Meaning |
|---|---|
| `material` | `C`, `Be`, `Al`, `Cu`, `Mo`, or `W` (chemical symbol) |
| `thickness_ug_cm2` | areal density in μg/cm² (PIP-II strippers: 300–700) |

The foil is a zero-length [ThinKickElement](00_overview.md): it applies
its kick at one s-position and adds no length to the lattice.

## Tutorial (newcomers)

A foil does two things to each macroparticle:

1. **Multiple Coulomb scattering** — many small-angle deflections add
   up to a random transverse kick on `xp`, `yp`.  HELIX uses the
   **Highland (1991)** RMS-angle formula:

   $$
   \theta_0 = \frac{13.6\ \text{MeV}}{\beta c p}\,|z|\,
   \sqrt{x/X_0}\,\bigl[1 + 0.038\ln(x/X_0)\bigr]
   $$

   where `x/X₀` is the foil thickness in radiation lengths.

2. **Energy loss** — the mean ionisation loss (Bethe–Bloch, tabulated
   at minimum-ionising rates per material) is subtracted from `dw`,
   with Bohr/Vavilov **straggling** added as a Gaussian spread.

Both effects scale with charge² and with thickness; the scattering
also grows for low-momentum (low-βcp) beams.  They act only in
multi-particle tracking — there is nothing to apply to an RMS
envelope.

### Example

```python
from linac_gen.elements.foil import Foil

# A 600 μg/cm² carbon stripper foil.
foil = Foil(name="STRIP", material="C", thickness_ug_cm2=600.0,
            seed=1234)        # fixed seed → reproducible kicks
```

Pass a fixed `seed` in tests for reproducibility; leave it `None`
(the default) in production so each run draws a fresh random stream.

## API reference (developers)

```{.python .skip}
linac_gen.elements.foil.Foil(
    name: str,
    material: str = "C",            # C, Be, Al, Cu, Mo, W
    thickness_ug_cm2: float = 600.0,
    aperture: float = 0.0,          # mm; 0 = unrestricted
    seed: int | None = None,        # RNG seed for the per-particle kicks
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | |
| `material` | `"C"` | — | one of C / Be / Al / Cu / Mo / W |
| `thickness_ug_cm2` | 600.0 | μg/cm² | areal density |
| `aperture` | 0.0 | mm | round aperture (0 = none) |
| `seed` | `None` | — | fix for reproducible scattering |

The material database (radiation length X₀ and minimum-ionising
dE/dx) is taken from PDG 2024.

### Source

`linac_gen/elements/foil.py:1`

## See also

* [Element overview](00_overview.md).
* [Stripping diagnostics](../09_diagnostics/05_stripping.md) — H⁻
  stripping loss elsewhere in the line.

← [Steerer](14_steerer.md) ·
[Continue to ThinLens →](16_thinlens.md)
