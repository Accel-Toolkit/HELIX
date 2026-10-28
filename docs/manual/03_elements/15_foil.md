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
| optional tail | `[auto|landau|gaussian] [strip_model=two_step] [dedx_model=bethe] [dx=… dy=…] [extent_mm=x,y] [seed=N]` — written only when non-default (see [Foil scenarios](../16_reliability/05_foil.md)) |

The foil is a zero-length [ThinKickElement](00_overview.md): it applies
its kick at one s-position and adds no length to the lattice.

## Tutorial (newcomers)

A foil does two things to each macroparticle:

1. **Multiple Coulomb scattering** — many small-angle deflections add
   up to a random transverse kick on `xp`, `yp`.  HELIX uses Highland's
   RMS-angle formula in the form of Lynch & Dahl (1991) that the Particle
   Data Group's review gives:

   $$
   \theta_0 = \frac{13.6\ \text{MeV}}{\beta c p}\,|z|\,
   \sqrt{x/X_0}\,\bigl[1 + 0.038\ln\bigl(x z^2/(X_0 \beta^2)\bigr)\bigr]
   $$

   where `x/X₀` is the foil thickness in radiation lengths, `z` the
   particle's charge number and `β` its speed.  (Before 2026-10-24 the
   logarithm was of `x/X₀` alone; with `z²/β²` in it θ0 is 2.3 % larger for
   800 MeV protons and 20 % larger for 24 MeV ones.)

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

Without a `seed` (the default) the foil follows the run: a beam made by
`create_beam` with a seed — the GUI and the CLI always give one, also for
a beam read from a file — draws the foil's kicks from a stream seeded by
that seed and the foil's name, so a run repeats exactly and two foils draw
different numbers.  A `seed` on the foil itself fixes the stream instead,
whatever the beam's seed.  Either way every new beam starts the stream
afresh, so a run that reuses the lattice (as the GUI does) repeats too.  A
beam built by hand without a seed, through a foil without one, draws a
fresh random stream each run.  The bunches of a
[bunch train](../15_multibunch/01_multibunch.md) share their seed, so they
all see the same foil kicks, as they share their macroparticles.

The rms energy loss is small, but it is uncorrelated with the beam: on a
long, strongly chirped bunch — a phase–energy correlation of −0.9998, say —
it widens the thin longitudinal ellipse a lot.  The Landau tail then makes
the post-foil longitudinal emittance converge slowly with the particle
count: in such a case it grew by between 5 % and 190 % over 20 foil seeds
with 5 000 particles, and by 40–67 % with 500 000.  Use 50 000 particles or
more when that number matters.

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
| `seed` | `None` | — | `None`: follow the beam's seed (see above); an integer fixes the stream, restarted for every beam |
| `strip_model` | `"off"` | — | `"two_step"`: H⁻ → H⁰ → p with cross sections calibrated to the PIP-II anchors; unconverted ions on `Beam.unstripped_table` (not a loss) |
| `dedx_model` | `"mip"` | — | `"bethe"`: β-dependent Bethe mean loss instead of the MIP floor |
| `dx`, `dy` | 0.0 | mm | foil offset (the tracker/envelope wrap translates the beam into the foil frame) |
| `extent_mm` | `None` | mm | half-size (one value or `(x, y)`); particles outside miss the foil |

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
