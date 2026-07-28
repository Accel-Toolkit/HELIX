# H⁻ stripping

For H⁻ ion linacs (PIP-II, ESS, J-PARC), the loosely-bound second
electron can detach via two mechanisms en route, producing
unwanted neutrals or H⁰ that get lost downstream.  HELIX has two
analytic stripping-rate analyzers.

## Magnetic (Lorentz) stripping

Implements Folsom et al. 2021 (*Phys. Rev. Accel. Beams* 24:074201):

$$
\frac{1}{N}\frac{dN}{ds} = \frac{|B_\perp|}{A_1}
\exp\left(-\frac{A_2}{\gamma\,\beta\,c\,|B_\perp|}\right) \quad [\text{1/m}]
$$

with empirical constants A₁ = 3.073 × 10⁻⁶ s·V/m, A₂ = 4.414 × 10⁹ V/m
(Stinson 1969 / Scherk 1979 fits, also used by TraceWin and PyORBIT).

Mode A: post-tracking deterministic analyzer (no per-particle MC):

```python
from linac_gen.analysis.magnetic_stripping import magnetic_stripping_loss
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.solenoid import Solenoid

# H⁻ beam through a small solenoid line (stripping needs B fields):
lattice = Lattice()
lattice.add(Drift(name="D1", length=200.0))
lattice.add(Solenoid(name="SOL1", length=300.0, field=0.30))
lattice.add(Drift(name="D2", length=200.0))
beam_config = BeamConfig(species="H-", n_particles=1000)
beam_cfg = beam_config           # both names used in this chapter
beam = create_beam(beam_config, seed=11)
results = Simulation(lattice, beam).run()

result = magnetic_stripping_loss(results, lattice, beam_config)
# MagStripResult fields (all s-indexed arrays; s in METRES):
#   result.s                      — m
#   result.loss_rate_per_m        — (1/N) dN/ds  [1/m]
#   result.integral_loss          — ∫ rate ds (dimensionless)
#   result.power_loss_per_m_W     — W/m
#   result.integral_power_loss_W  — W
```

Keyword options: `duty_factor` (default `beam_config.duty_cycle/100`),
`current_mA` (default `beam_config.current`), `quad_b_scale`
(`"2sigma"` default / `"1sigma"` / `"pole"`), `fieldmap_sample`
(`"max"` default / `"on_axis"`).  Raises `ValueError` unless
`beam_config.species == "H-"`.

The walk:

1. For each step, look up the element type at `element_names[i]`.
2. Extract the **perpendicular** field |B_⊥| — only the component ⟂ to v
   strips (Folsom Eq. 4). Per element:
   `|B_⊥| = √( B_transverse² + (B_axial·θ_rms)² )`, where `B_transverse` is
   the dipole bend field, the quad `G·r` (at 2σ by default), or a field-map
   `B_r`/`B_x`/`B_y`, and `B_axial` is an on-axis longitudinal field (a
   **solenoid's `B_z`**, a field-map `F_z`) that is ∥ to v on axis and only
   couples through the rms beam divergence `θ_rms` (from the σ-matrix; a
   solenoid does **not** strip a paraxial beam, matching Folsom/TraceWin,
   which model only dipoles and quads).
3. Evaluate Folsom Eq. (5).
4. Integrate: cumulative_loss = 1 − exp(−∫ rate ds).

## Intrabeam stripping (IBS)

The bunch's own electromagnetic field can strip electrons via
beam-beam collisions.  Implements the form-factor model:

```python
from linac_gen.analysis.intrabeam_stripping import ibs_loss

result = ibs_loss(results, beam_config)
# IbsResult fields (s-indexed arrays; s in METRES):
#   result.s               — m
#   result.sigma_strip_cm2 — cross-section used at each step [cm²]
#   result.loss_rate_per_m — (1/N) dN/ds  [1/m]
#   result.integral_loss   — cumulative ∫ rate ds (dimensionless)
```

Keyword options: `duty_factor`, `current_mA`, `non_constant_xs`
(per-step β-dependent σ_H instead of a constant σ_max), and
`theta_z_convention`.  The longitudinal velocity coordinate is
convention-flagged: the default is `"dp_over_p"` (the paper's
convention); pass `theta_z_convention="dv_over_v"` to reproduce
Ostiguy's override (dv/v = dp/p / γ²).  Raises `ValueError` unless
the species is H⁻.

## Visualising stripping losses

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linac_gen.analysis.magnetic_stripping import magnetic_stripping_loss
from linac_gen.analysis.intrabeam_stripping import ibs_loss

mag = magnetic_stripping_loss(results, lattice, beam_cfg)
ibs = ibs_loss(results, beam_cfg)

fig, ax = plt.subplots()
ax.plot(mag.s, mag.integral_loss * 100, label="magnetic (Lorentz)")
ax.plot(ibs.s, ibs.integral_loss * 100, label="intrabeam")
ax.set_xlabel("s [m]"); ax.set_ylabel("cumulative loss [%]")
ax.legend()
```

Both result objects carry `s` in **metres** (converted from the
recorder's mm internally), so no unit juggling is needed for the
x axis.

## Engineering decisions

| Stripping fraction | Implication |
|---|---|
| < 0.01 % | typical PIP-II target — acceptable |
| 0.1 % | needs review; likely localised hot spot |
| > 1 % | redesign required |

For PIP-II's HEBT, magnetic stripping in the dump dipoles and the
beam-bunching cavities are the dominant contributors — both are
analyzed here.

## Cross-references

* [Aperture profile](04_aperture.md) — the other loss mechanism.
* `linac_gen/analysis/magnetic_stripping.py:1`
* `linac_gen/analysis/intrabeam_stripping.py:1`

References:

* Folsom, B. et al. "Stripping mechanisms and remediation for H⁻
  beams", *Phys. Rev. Accel. Beams* 24, 074201 (2021),
  arXiv:2103.16195.
* Stinson, G.M. et al. (1969).
* Ostiguy, J.-F.  Internal note on dv/v vs dp/p convention.

← [Aperture](04_aperture.md) ·
[Continue to Phase advance →](06_phase_advance.md)
