# Leg D — foil scenarios

The foil leg tracks the beam to the stripper foil (`landmarks.foil_element`,
or the first `Foil` of the deck) for its nominal state and a set of
scenarios: the thickness list (`foil.thickness_ug_cm2`, `0` = the foil is
missing), transverse offsets (`foil.offsets_mm`), a thinned foil
(`foil.thinned_fraction`), and, in the full preset, tracked runs with a
snapshot of the particles at the foil.

## What is measured

| Quantity | Model | Source |
|---|---|---|
| `foil_sigma_x_mm`, `foil_sigma_y_mm` | envelope Σ at the foil, or the rms of the snapshot | `results.h5` |
| `foil_w_cm2_peak` | peak power density of the snapshot at the beam's current × duty (× `hits_per_particle`) | `analysis.loss_power.plane_power_density` |
| `strip_eff`, `h0_frac`, `missed_frac` | the tracked H⁻ → H⁰ → p populations and the particles outside the foil extent | the beam's `unstripped_table` |
| `strip_eff_analytic`, `h0_frac_analytic` | the two-step model at the foil energy and scenario thickness | `Foil.stripping_fractions` |

## The stripping model

`strip_model="two_step"` converts a negative-ion beam H⁻ → H⁰ → p with the
cross sections σ(−1→0) and σ(0→+1) **calibrated** so that the stripped
fraction reproduces the two PIP-II design anchors — 99.956 % at 600 µg/cm²
and 99.1 % at 380 µg/cm² for an 800 MeV H⁻ beam on carbon — scaled 1/β² to
other energies and by √(Z/6) to other materials (a convention, not data).
The calibration lands within a few per cent of the measured cross sections
(Gulley et al., PRA 53, 3201 (1996)); the report labels the anchors as
calibration points, not measurements.  One uniform draw per particle, taken
*after* the scattering and loss draws, assigns the charge state, so the
historical kicks of a seeded foil are unchanged; unconverted ions are
recorded on `Beam.unstripped_table` (HDF5 group `unstripped/`) and stay
alive — they are not a loss, so transmission and the loss-power accounting
stay honest.  The model is inert for a positive species.

## Deck line

```
; HELIX_FOIL STRIP C 600 gaussian strip_model=two_step dedx_model=bethe dx=0.5 dy=0 extent_mm=2.5,4 seed=7
```

The optional `key=value` tail (`strip_model`, `dedx_model` — `mip` or the
β-dependent `bethe` mean loss —, `dx`, `dy`, `extent_mm` as one value or
`x,y` half-sizes, `seed`) is written only when non-default, so existing decks
round-trip unchanged; particles outside the extent miss the foil (no kick, no
loss, recorded as `missed`).  See [Foil](../03_elements/15_foil.md).

## Outputs

`legs/foil/foil.csv` and the figures `foil_spot.png` and
`foil_efficiency_vs_thickness.png` (the calibrated curve with the tracked
points on it).
