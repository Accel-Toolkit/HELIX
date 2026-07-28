# Twiss & emittance conventions

Every beam in HELIX is described by 9 numbers per plane: ε (emittance)
+ (α, β) (Twiss).  This page nails down the conventions — units,
sign, normalisation — so you never have to wonder if your input
values mean the same thing HELIX thinks they do.

## TL;DR

| | Transverse (x, y) | Longitudinal (z) |
|---|---|---|
| Geometric emittance ε | mm·mrad | deg·MeV (native) |
| Normalised emittance ε_n = β·γ·ε | mm·mrad | mm·mrad (after deg·MeV → mm·mrad conv.) |
| Twiss α | dimensionless | dimensionless |
| Twiss β | mm/mrad = m | deg/MeV (native) |
| Twiss γ | mrad/mm = 1/m | MeV/deg |

`BeamConfig.emit_nx` and `emit_ny` are **normalised**.  HELIX divides
by β·γ before passing to the distribution generator (geometric ε is
what the matched-Twiss ellipse uses).

## Tutorial

### What are α, β, ε?

The Courant-Snyder parameters describe a phase-space ellipse:

$$
\gamma\, x^2 + 2\alpha\, x\,x' + \beta\, x'^2 = \varepsilon
$$

with γ = (1+α²)/β as a dependent quantity.  Geometrically:

* **β** is the "size" envelope — RMS beam half-width is √(β·ε).
* **α = -dβ/(2ds)** characterises whether the beam is converging
  (α > 0) or diverging (α < 0).
* **ε** is the area/π of the ellipse — invariant under linear
  symplectic transport (in the absence of coupling).

The ellipse rotates and shears as the beam propagates; the
Courant-Snyder invariant (the area) stays the same — that's why ε
is the natural beam-quality metric.

### Geometric vs normalised emittance

For relativistic beams (γ ≫ 1), the geometric ε *shrinks* during
acceleration (adiabatic damping).  To compare beams at different
energies, use the normalised emittance:

$$
\varepsilon_n = \beta\,\gamma\,\varepsilon
$$

ε_n is invariant under acceleration — same beam quality before and
after a cavity.  For a 3 MeV proton (β = 0.0796, γ = 1.003) the
factor βγ ≈ 0.0799; for a 1 GeV proton (β = 0.875, γ = 2.067) it's
≈ 1.81.  So ε_n / ε ≈ 23× difference between the two energies.

### Longitudinal emittance

HELIX records longitudinal emittance natively in **deg·MeV**
(matches TraceWin) and converts to mm·mrad for some plots.

$$
\varepsilon_z\,[\text{mm·mrad}] = \frac{\varepsilon_z\,[\text{deg·MeV}] \cdot 1000}{k_\phi}
$$

where k_φ = 360° / (β·λ_RF) is the phase-to-length conversion.

For β·λ_RF = β · c / f, with λ_RF in mm.

The normalised longitudinal emittance is then ε_nz = β·γ·ε_z(mm·mrad).

### Where it bites

!!! warning "BeamConfig.emit_nx is normalised, not geometric"
    The most common confusion in HELIX usage.  If you write
    `emit_nx=0.21` thinking it's the geometric value, you'll
    over-predict σ_x by a factor of 1/βγ ≈ 11× at LEBT entrance.
    The factory always divides by ref.bg before using it.

!!! warning "BeamConfig.emit_z is in deg·MeV"
    Not normalised, not in mm·mrad.  Native TraceWin units.  HELIX
    converts internally where needed.

### Sample values for PIP-II

| Stage | Energy (MeV) | β·γ | ε_nx (mm·mrad) | β_x (m) | α_x |
|---|---|---|---|---|---|
| LEBT entrance (after source) | 0.030 | 0.0080 | 0.20 | (varies) | (varies) |
| MEBT entrance | 2.10 | 0.0669 | 0.21 | 0.32 | 1.23 |
| MEBT exit | 2.45 | 0.0723 | 0.21 | (matched) | (matched) |
| HWR entrance | 2.45 | 0.0723 | 0.21 | (matched) | (matched) |
| End of LB650 | 200 | 0.686 | 0.25 | (varies) | (varies) |
| End of HB650 | 800 | 1.558 | 0.30 | (varies) | (varies) |

The slight ε_n growth (0.21 → 0.30 mm·mrad over 256m) is
emittance-growth from chromaticity, RF gymnastics, and PIC noise —
within engineering tolerance for the design.

## Cross-references

* [Coordinates & units](../02_concepts/01_coordinates.md) — the (x,
  x', …) state vector.
* [BeamConfig reference](03_beam_config.md) — every Twiss field.
* [Eigenemittances](../09_diagnostics/02_emittances.md) —
  coupling-invariant emittance variants.

← [Distributions](01_distributions.md) ·
[Continue to BeamConfig reference →](03_beam_config.md)
