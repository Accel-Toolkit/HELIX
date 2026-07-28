# Appendix C — Glossary

Domain terms used throughout the manual.

## A

* **Adiabatic damping** — divergences x', y' shrink during
  acceleration as p_in/p_out.  Geometric ε shrinks; normalised ε_n
  is invariant.
* **Aperture** — the local maximum allowed transverse extent.
  Particles outside are lost.
* **AGS** — Alternating Gradient Synchrotron focusing: pair a
  focusing quad with a defocusing quad, net is focusing.

## B

* **Beam pipe** — the vacuum chamber the beam travels in.  Sets
  the aperture.
* **Bend / dipole** — magnet that bends the beam trajectory.
* **Boris pusher** — symplectic time-step integrator for
  charged-particle motion in EM fields.

## C

* **CIC** — Cloud-In-Cell, the 8-corner trilinear particle-mesh
  shape function (1st order).
* **Continuous beam** — DC, unbunched.  No longitudinal structure.
  Pre-RFQ.
* **Crandall 2-term** — RFQ field expansion; HELIX's M1.

## D

* **Drift** — field-free propagation.
* **DC mode** — continuous-beam tracking; HELIX's `continuous=True`
  flag.
* **Dispersion** — D_x = ∂x/∂(Δp/p₀); off-momentum particles take
  different paths.
* **DKD** — Drift-Kick-Drift, second-order symplectic field-map
  integrator (velocity-Verlet).

## E

* **Eigenemittance** — symplectic-eigenvalue invariant emittance;
  4-D (ε₁, ε₂) or 6-D (ε_E1, ε_E2, ε_E3).
* **Emittance** — phase-space ellipse area / π.
  Geometric (mm·mrad) or normalised (β·γ·ε).
* **Envelope solver** — propagates the σ-matrix instead of
  individual particles.

## F

* **Field map** — tabulated electromagnetic field on a regular
  grid, integrated by RK4.
* **FODO** — Focusing-Drift-Defocusing-Drift cell.
* **FREQ jump** — RF frequency boundary in a multi-section linac.

## G

* **Geometric emittance** — ε in mm·mrad, untouched by βγ
  conversion.

## H

* **H⁻** — singly-negative hydrogen ion; PIP-II's particle.  Has
  a loosely-bound second electron prone to stripping.
* **Halo** — beam-tail population beyond the Gaussian core.
  HELIX records the kurtosis halo parameter h = ⟨x⁴⟩/⟨x²⟩² − 1
  (Gaussian ⇒ 2, KV ⇒ 1; values noticeably above 2 indicate halo).
* **HEBT** — High-Energy Beam Transport (downstream of linac).
* **HWR** — Half-Wave Resonator, a low-β SRF cavity type.

## I

* **IBS** — IntraBeam Stripping; H⁻ losing electron via in-bunch
  collisions.
* **IGF** — Integrated Green Function; HELIX's PIC Poisson kernel.
* **Image charge** — charge on the beam-pipe wall that mirrors
  the bunch.  Not modelled in HELIX.

## J

(none)

## K

* **KV** — Kapchinskij-Vladimirskij, the uniform-charge-density
  analytic distribution.  Self-consistent with linear envelope
  dynamics.

## L

* **Larmor** — angular frequency of cyclotron motion in a B-field.
* **LB650** — Low-Beta 650 MHz; one of PIP-II's SRF cavity types.
* **LEBT** — Low-Energy Beam Transport (between source and RFQ).
* **Levenberg-Marquardt** — non-linear least-squares solver used
  by the matcher.

## M

* **MEBT** — Medium-Energy Beam Transport (between RFQ and first
  SRF cavity).
* **Mismatch** — ε scaled by a factor != 1; tests Twiss-tolerance.
* **Multipole** — sextupole, octupole, decapole, dodecapole — thin
  kicks.

## N

* **Normalised emittance** — ε_n = β·γ·ε, invariant under
  acceleration.

## P

* **Partran** — TraceWin's multi-particle tracker; HELIX's
  validation reference.
* **PIC** — Particle-In-Cell; the 3-D space-charge solver method.
* **PIP-II** — Proton Improvement Plan II, a Fermilab linac;
  HELIX's reference benchmark.
* **PMQ** — Permanent Magnet Quadrupole.

## Q

* **Quadrupole** — alternating-gradient focusing magnet.

## R

* **RFQ** — Radio-Frequency Quadrupole; bunches and accelerates
  the DC beam.  PIP-II's first accelerating element.
* **RK4** — fourth-order Runge-Kutta integrator.

## S

* **Sacherer ODE** — DC-beam envelope equation.
* **SC** — Space Charge.
* **Sigma matrix** — 6×6 second-moment matrix of the beam.
* **Skew quadrupole** — quad with magnetic axes rotated about z;
  couples (x, x') with (y, y').
* **SRF** — Superconducting RF (cavity).
* **Solenoid** — axial-B-field magnet, focuses both planes.
* **SSR** — Single Spoke Resonator, mid-β SRF cavity type.
* **σ** — RMS standard deviation; e.g. σ_x.

## T

* **TraceWin** — Saclay's commercial linac code; HELIX's
  validation benchmark.
* **TSC** — Triangular-Shaped Cloud, 27-cell quadratic
  particle-mesh shape (2nd order).
* **Twiss** — α, β, γ Courant-Snyder parameters describing the
  phase-space ellipse.

## V

* **VaneRFQ** — HELIX's whole-RFQ wrapper around RfqCell instances.

## W

* **Waterbag** — uniform-distribution-in-6D-ellipsoid;
  textbook-benchmark distribution.
* **Wangler halo parameter** — kurtosis-style halo measure; HELIX's
  `halo_x`/`halo_y` implement h = ⟨x⁴⟩/⟨x²⟩² − 1 (Gaussian ⇒ 2,
  KV ⇒ 1).

## Z

* **z** — longitudinal coordinate; in HELIX's reduced phase space,
  represented by Δφ.

## Symbols

* **α, β, γ** — Twiss parameters (or β = v/c, γ = Lorentz factor).
* **ε** — emittance.
* **σ** — RMS beam size.
* **Σ** — full 6×6 σ-matrix.
* **κ(s)** — focusing function (envelope ODE coefficient).
* **K** — generalised perveance K = qI / (2πε₀mc³(βγ)³).

← [Appendix B](B_keyword_cheatsheet.md) ·
[Continue to Appendix D →](D_troubleshooting.md)
