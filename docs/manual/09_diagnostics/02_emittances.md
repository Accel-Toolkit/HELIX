# Emittances

Emittance is HELIX's primary beam-quality metric.  But "emittance"
covers six different things in HELIX, each useful in a different
regime.  This page disambiguates and explains when to use which.

## Six emittances at a glance

| Symbol | Field | Definition | Use case |
|---|---|---|---|
| ε_x | `emit_x` | √(⟨x²⟩⟨x'²⟩ − ⟨xx'⟩²) | basic transverse RMS |
| ε_n,x | `emit_nx` | β·γ·ε_x | invariant under acceleration |
| ε_z | `emit_z` | longitudinal RMS in deg·MeV | native long. units |
| ε_z(mmmrad) | `emit_z_mmmrad` | ε_z converted to mm·mrad | unified plotting |
| ε_4D | `emit_4d` | √det(Σ_4D) | invariant under x-y coupling |
| ε_1, ε_2 | `emit_n1`, `emit_n2` | symplectic eigenvalues of J·Σ_4D | 4-D normal-mode |
| ε_E1, ε_E2, ε_E3 | `emit_e1/e2/e3` | Balandin trace invariants of Σ_6D | full 6-D |

## When projected ε_x is enough

For an uncoupled lattice (no solenoids, no skew quads, no dispersion-
energy coupling), the projected ε_x and ε_y are constant under
linear symplectic transport.  This is the textbook case and ε_x is
all you need.

## When projected ε_x lies — coupling

The moment a solenoid couples (x, x') with (y, y'), the projected
ε_x and ε_y oscillate visibly through the lattice — even for an
ideal beam with no real growth.  Symptom: σ_x and σ_y wobble; ε_x,
ε_y wobble; you'd swear the beam is heating up but actually nothing
is wrong.

The 4-D invariant captures this:

$$
\varepsilon_{4D} = \sqrt{\det \Sigma_{4D}}
$$

It stays constant through any linear x-y coupling.  And the 4-D
**normal-mode** eigenemittances (ε₁, ε₂) are the genuine
"transverse emittances" — what would σ_x and σ_y be if you
diagonalised away the coupling.

For solenoid-rich LEBT analyses, **always plot ε₁ and ε₂** instead
of ε_x and ε_y.

## When the 4-D isn't enough — 6-D coupling

Dispersion couples (x, x') with (Δφ, ΔW).  In high-dispersion
regions, even ε_4D oscillates while the full 6-D emittance is
invariant.

The Balandin trace invariants (I₂, I₄, I₆) of Σ_6D give three
constants:

$$
I_2 = -\tfrac{1}{2}\,\mathrm{tr}\!\left((J\Sigma)^2\right),\quad
I_4 = +\tfrac{1}{2}\,\mathrm{tr}\!\left((J\Sigma)^4\right),\quad
I_6 = -\tfrac{1}{2}\,\mathrm{tr}\!\left((J\Sigma)^6\right)
$$

These combine into a cubic equation whose roots are the squared
eigenemittances ε_E1², ε_E2², ε_E3²:

$$
x^3 - I_2\,x^2 + \frac{I_2^2 - I_4}{2}\,x
- \left(\frac{I_2^3}{6} - \frac{I_2 I_4}{2} + \frac{I_6}{3}\right) = 0
$$

(See `linac_gen/diagnostics/eigenemittance.py:1` for the full
derivation.)  HELIX records ε_E1 ≥ ε_E2 ≥ ε_E3 as
`emit_e1, emit_e2, emit_e3`.

In a fully uncoupled lattice, ε_E1 = ε_x, ε_E2 = ε_y, ε_E3 = ε_z.
With coupling, the eigenemittances stay invariant while the
projected ε_x/y/z visibly wobble.

## Reference plots

For an example of all six emittances on the same plot — illustrating
exactly when each is meaningful — see
[Worked example: Eigenemittance demo](../11_examples/09_eigenemittance.md).

## Implementation

* `linac_gen.diagnostics.eigenemittance.kinetic_invariants(sigma_6d)` —
  computes (I₂, I₄, I₆).
* `linac_gen.diagnostics.eigenemittance.eigenemittances(sigma_6d)` —
  returns sorted (ε_E1, ε_E2, ε_E3).
* `linac_gen.diagnostics.eigenemittance.eigenemittances_4d(sigma_4d)` —
  4-D version.
* `linac_gen.diagnostics.eigenemittance.eigenemittances_series(...)` —
  per-step over the full lattice.

## Reference

* V. Balandin, W. Decking, N. Golubeva, "On the calculation of
  generalized four-dimensional and six-dimensional eigen-emittances",
  IPAC 2013 / arXiv:1305.1532.
* IMPACT-X documentation — same eigenemittance convention.

## Cross-references

* [Recorder fields](01_recorder.md)
* [Worked example: eigenemittance](../11_examples/09_eigenemittance.md)
* `linac_gen/diagnostics/eigenemittance.py:1` (source).

← [Recorder fields](01_recorder.md) ·
[Continue to Halo →](03_halo.md)
