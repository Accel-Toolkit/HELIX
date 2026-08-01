# Space-charge models

HELIX has four distinct space-charge models, each appropriate for a
different physical regime.  Picking the right one is critical:
mismatch between physics and model is the most common source of
"why does my simulation disagree with TraceWin?" complaints.

## TL;DR — pick by regime

| Beam regime | Tracker | SC model | When |
|---|---|---|---|
| Bunched, RMS only | `EnvelopeSolver` | uniform-ellipsoid Σ kick | matching, parameter sweeps |
| Bunched, multi-particle | `Tracker` (with `SpaceChargeConfig`) | 3-D PIC (Hockney FFT) | production tracking |
| Continuous (DC), RMS | `Sacherer ODE` | KV-uniform analytic | pre-RFQ LEBT envelope |
| Continuous (DC), multi-particle | `Tracker` (with `SpaceChargeConfig`, `continuous=True`) | 2-D analytic kick | pre-RFQ LEBT MP |

## Model 1: Uniform triaxial ellipsoid (envelope)

Used by `EnvelopeSolver` for bunched beams.  Treats the beam as a
uniform-density ellipsoid with rms-equivalent semi-axes `(√5·σ_x,
√5·σ_y, √5·γ·σ_z)` and applies an analytic depolarisation-factor
kick using the Carlson R_D elliptic integral (Lapostolle / Wangler
formulation).

* **Pros**: closed-form, exact for matched RMS Gaussians.
* **Cons**: doesn't capture non-Gaussian effects, halo, or
  filamentation.

## Model 2: 3-D PIC (Hockney FFT)

Used by the multi-particle tracker (`Tracker` + `SpaceChargeConfig`).
Solves Poisson's equation on a 3-D Cartesian grid via doubled-grid
FFT convolution with one of two Green's functions.

The full 8-step PIC cycle:

1. Lorentz boost on z to enter the beam rest frame (γ on z only).
2. Grid setup on the boosted coordinates.
3. Charge deposition on the grid (CIC or TSC kernel).
4. Doubled-grid FFT convolution with Green's function.
5. Field gather back to particles (CIC or TSC).
6. Transverse kick applied **directly to the lab-frame x′/y′** —
   the boost back is folded into the β²γ² factor of the kick
   formula (one γ from the E+v×B force cancellation, one βγ from
   the Δp⊥ → x′ conversion).
7. Longitudinal kick applied directly to the lab-frame ΔW
   (E_z is invariant under the z-boost).
8. The **reference particle is never modified** — SC kicks touch
   only the deviation coordinates of the macroparticles.

For details see [PIC solver](02_pic_solver.md) and
[Kernels](03_kernels.md).

* **Pros**: captures non-axisymmetric, non-Gaussian, halo,
  filamentation effects.
* **Cons**: ~10× slower than envelope; statistical noise floor
  ≈ 1/√N on σ.

**Single bunch, no neighbour images.**  The solver sees the
macroparticles it is given and nothing else, so a run downstream of an
RFQ models one bunch of what is physically a train.  In a real train
the neighbouring bunches partially cancel the longitudinal field, so
E_z is slightly **over**estimated; the transverse field is barely
affected at typical bunch aspect ratios.  Periodic images (the
TraceWin PICNIR practice) are not implemented.

This matters most alongside
[`periodic_phase`](../04_beam/03_beam_config.md): with the flag on,
particles that slip a bucket are folded back into the bunch, so the
solver sees a single compact bunch rather than a finite three-bucket
clump.  Measured on the PXIE deck at 5 mA that removes a spurious
≈ −35 keV/bucket chirp (exit energy spread 24.0 → 15.4 keV) and moves
line transmission only slightly (62.0 → 60.6 %).  Without the flag
the clump is what the solver sees, which is a *different* wrong — the
neighbour bunches are present but only three of them, unshielded and
at the wrong separation once they start to drift apart.

## Model 3: Sacherer / KV ODE (continuous beam)

For pre-RFQ continuous beams, `linac_gen.tracking.sacherer` solves
the continuous-beam envelope ODE:

$$
\sigma_x'' + \kappa_x(s)\,\sigma_x - \frac{\varepsilon_x^2}{\sigma_x^3}
- \frac{2K}{\sigma_x + \sigma_y} = 0
$$

(and the analogous equation in y) with generalised perveance K =
qI / (2π ε₀ m c³ (βγ)³).

* **Pros**: textbook DC-beam benchmark, fast.
* **Cons**: no acceleration (assumes constant β, γ), only handles
  drifts + hard-edge quads + 1-D/3-D solenoid maps.

## Model 4: 2-D analytic DC kick (continuous beam, MP)

For continuous-beam multi-particle tracking,
`linac_gen.pic.pic_solver.kick_continuous_2d` applies the closed-
form transverse kick for a uniform-density elliptical-cylinder beam.
Activated automatically when `Beam.continuous == True`.

Three DC-kernel choices via `SpaceChargeConfig.dc_kernel`:

| `dc_kernel` | Model |
|---|---|
| `"uniform"` | analytic linear uniform-elliptical kick (matches rigid-Σ envelope) |
| `"gaussian"` | Bassetti-Erskine field of a 2-D Gaussian (rigid σ; per-particle non-linear) |
| `"pic2d"` | 2-D Hockney FFT PIC over the actual particle distribution (most accurate) |

All three scale the field with the **surviving** current,
`I · n_alive / n_launched` — the same macrocharge convention as the
bunched 3-D PIC (each launched macroparticle carries a fixed share of
the configured current, so the transported current decays with
transmission).  Before 2026-08-01 the DC kernels drove the field from
the configured current outright; on the PXIE LEBT (77 % transmission)
that overdrove σ_x by 16 % at the SOL2 exit against TraceWin partran —
loss-scaled, the agreement is ~1 %.  Lossless beams are unaffected
(the factor is exactly 1).

For details see [DC mode](04_dc_mode.md).

## Coherent synchrotron radiation (CSR)

CSR is a separate collective effect — not one of the four space-charge
models above.  When a short bunch travels around a bend, radiation
emitted toward the tail catches up with the head along the shorter
chord and acts back on the bunch, driving energy spread and — through
the bend dispersion — transverse emittance growth.

HELIX models CSR with a **1-D steady-state wake**
(Saldin–Schneidmiller–Yurkov 1997; Derbenev 1995):

$$
\frac{dW}{ds}(s) = -A \int_{-\infty}^{s} \lambda'(s')\,(s - s')^{-1/3}\,ds'
$$

where λ(s) is the bunch line-charge density and the prefactor scales
as `A ∝ R^{-2/3}` with R the bend radius.  The kick is applied per
sub-step inside every [Dipole](../03_elements/04_dipole.md).

* **Multi-particle only.** CSR needs the actual bunch line-density
  profile, so it acts only in `Tracker` runs — the envelope solver
  ignores it.
* **Steady-state only.** Transient entrance/exit (drift→bend,
  bend→drift) fields are not modelled; the model is valid when the
  bend is long compared with the CSR formation length.
* Enable it with `SpaceChargeConfig(csr_enabled=True)`, or the
  "CSR in bends" checkbox on the
  [Numerics tab](../10_gui/04_convergence_tab.md).

| `SpaceChargeConfig` field | Default | Meaning |
|---|---|---|
| `csr_enabled` | `False` | apply the CSR kick in bends |
| `csr_bins` | `200` | longitudinal line-density bin count |
| `csr_model` | `"1d_steady"` | model selector (only value for now) |

## Decision summary

```
Is the beam bunched?
├── Yes (post-RFQ)
│   ├── RMS only? → EnvelopeSolver (Model 1)
│   └── Multi-particle? → Tracker + SpaceChargeConfig (Model 2)
└── No (pre-RFQ, DC)
    ├── RMS only? → Sacherer ODE (Model 3)
    └── Multi-particle? → Tracker + SpaceChargeConfig with continuous=True (Model 4)
```

## Cross-references

* [Tracking modes](../02_concepts/03_tracking_modes.md) — broader
  overview of solver choice.
* [PIC solver](02_pic_solver.md) — Model 2 details.
* [DC mode](04_dc_mode.md) — Models 3 & 4.
* [Convergence guide](05_convergence.md) — grid sizing, particle
  count, when does it converge?

← [Beam → .dst loading](../04_beam/04_dst_io.md) ·
[Continue to PIC solver →](02_pic_solver.md)
