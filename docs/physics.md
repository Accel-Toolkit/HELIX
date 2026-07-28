# LG physics model

This document lists the assumptions and conventions baked into each
tracker, integrator, and space-charge solver in `linac_gen`.  Treat it
as the contract between the user and the simulation: if a regime
violates one of these assumptions, the result will look reasonable but
will not be correct.

## Coordinates and units

All particle coordinates are stored in TraceWin's reduced phase space:

| Index | Name        | Units |
|-------|-------------|-------|
| 0     | x           | mm    |
| 1     | x' = px/pz  | mrad  |
| 2     | y           | mm    |
| 3     | y' = py/pz  | mrad  |
| 4     | Δφ          | deg   |
| 5     | ΔW          | MeV   |

Δφ is the phase deviation from the synchronous reference particle at
the lattice's RF reference frequency; ΔW is the kinetic-energy
deviation.  The geometric emittance ε is reported in mm·mrad; the
*normalised* emittance ε_n = β·γ·ε is also recorded for cross-machine
comparisons.

Note that x' = p_x / p_z is *kinetic*, not the canonical p_x / p₀ used
by some simulators.  Under solenoid coupling the projected ε_x and ε_y
oscillate; the 4-D invariant √det(σ_4D) and the symplectic normal-mode
emittances ε_1, ε_2 stay constant and are the right "real" emittances
to track for solenoid-rich beamlines (LEBT in particular).

## Trackers

### Matrix tracker (`linac_gen.tracking.tracker`)
Element-by-element 6×6 transfer-matrix transport for linear elements
(drift, quad, dipole, edge, hard-edge solenoid).  Field maps go through
their dedicated integrator, see below.  Matrix tracking is exact for
its element class and ignores nonlinearities.

### Envelope tracker (`linac_gen.tracking.envelope`)
Propagates the 6×6 σ-matrix.  Space charge is a thin defocusing lens
built from the *uniform triaxial ellipsoid* model (Lapostolle / Wangler)
using the rms-equivalent semi-axes (√5·σ_x, √5·σ_y, √5·γ·σ_z) and
Maxwell depolarisation factors via the Carlson R_D elliptic integral.
This is bunched-beam SC — for a continuous (DC) beam use the Sacherer
ODE solver below.

### Sacherer / KV ODE (`linac_gen.tracking.sacherer`)
Continuous-beam envelope ODE solved with `scipy.integrate.solve_ivp`
(RK45):

    σ_x'' + κ_x(s) σ_x − ε_x²/σ_x³ − 2K/(σ_x + σ_y) = 0
    σ_y'' + κ_y(s) σ_y − ε_y²/σ_y³ − 2K/(σ_x + σ_y) = 0

with generalised perveance K = q·I / (2π ε₀ m c³ (βγ)³) and a focusing
function κ(s) read off element by element.  Honours `SpaceChargeComp`
markers (effective K = K · (1 − comp.factor)) the same way the matrix
envelope does.

Scope: drifts, hard-edge quadrupoles, 1-D / 3-D solenoid field maps
(κ from on-axis B_z).  RF cavities and acceleration are out of scope —
this is a textbook DC-beam benchmark, not a general tracker.

### Field-map integrators (`linac_gen.elements.field_map_3d`)
Two flavours are available, controlled by `FieldMap3D.integrator_kind`:

* `"kd"` (default) — first-order kick-then-drift, evaluated at the
  slice midpoint.  Symplectic (each operator preserves the symplectic
  form) but not second-order; the dominant error is an O(ds²)
  asymmetry in phase advance.
* `"dkd"` — second-order symplectic Drift–Kick–Drift ("velocity Verlet").
  Position-dependent fields are sampled at the half-drifted location;
  removes the leading phase-advance asymmetry.

For LEBT-class trajectories with default step refinement the two are
indistinguishable (<0.2 % on σ).  DKD pays off on long, periodic, or
storage-ring trajectories where phase-space symmetry matters.

The 1-D `FieldMap` (cylindrical and table-driven solenoids) currently
uses the same KD scheme as the legacy default; a DKD variant could be
added with the same recipe if needed.

## Space-charge solvers

### Analytic 2-D DC kick (`linac_gen.pic.pic_solver.kick_continuous_2d`)
Closed-form transverse kick for a uniform-density elliptical beam-pipe
cross-section, suitable only for unbunched (DC) beams.  Faster than PIC
and matches TraceWin's "continuous beam" formula; the longitudinal
force is ignored (zero by construction for a DC beam).

### 3-D PIC (`linac_gen.pic.pic_solver.PicSolver`)
Hockney FFT Poisson solve in the rest frame.  Uses

* Lorentz boost on z to enter the rest frame (boost factor γ on z only).
* CIC (default) or TSC charge deposition on a regular Cartesian grid;
  selectable via `SpaceChargeConfig.kernel = "cic" | "tsc"`.
* Doubled-grid FFT convolution with one of two Green's functions:
  * **IGF** (default) — Integrated Green Function from Qiang et al.
    PRSTAB 9, 044204 (2006).  Closed-form 8-point cell-difference of an
    analytic antiderivative.  Symmetric, free of grid-resolution bias
    near the source.  Same kernel used by Cheetah and OPAL.
  * **point** (legacy) — sampled 1/(4πε₀ r) with self-potential
    regulariser at r = 0.  Kept for back-compat regression.
* CIC (or TSC, matching the deposit kernel) field gather to particles.
* Transverse momentum kick uses the standard Lorentz cancellation:
  net lab-frame transverse force = q·E_⊥_rest / γ; the β²γ² factor in
  the kick formula bundles the γ from the force cancellation with the
  βγ from the p_z conversion.

### Real-space Dirichlet Poisson (`linac_gen.pic.realspace_poisson`)
7-point sparse Laplacian solved with preconditioned CG.  Intended for
projection-style applications where periodic FFT BCs are wrong (e.g.
Helmholtz projection of imperfect EM-solver field maps to enforce
∇·B = 0 only inside the field-map volume).  Slower than the FFT
Hockney path; ship only when Dirichlet boundaries genuinely matter.

## Beam initialisation

Distributions in `linac_gen.distributions`:

* Gaussian (default), uniform, KV, water-bag, parabolic.  Each samples
  the matched Twiss ellipse; the longitudinal phase is uniform across
  one RF period for bunched beams or a single period sample for DC.
* TraceWin .dst binary reader/writer: `linac_gen.io.tracewin_dst`.

## Diagnostics

`linac_gen.diagnostics.recorder.DiagnosticRecorder` exposes per-step:

* σ_x, σ_y, σ_φ, σ_W and Twiss (α, β) per plane.
* Geometric and normalised ε per plane.
* 4-D invariant ε_4D = √det(Σ_4D) — coupling-invariant, smooth through
  solenoids.
* Normal-mode emittances ε_1 ≥ ε_2 (symplectic eigenvalues of J·Σ_4D)
  — the *physical* invariants under x–y coupling.
* Halo parameter h = ⟨x⁴⟩/⟨x²⟩² − 2 per plane.
* Centroid position, transmission, peak excursion (x_max, y_max).
* Reference-particle state: w_kin, β, γ, βγ, φ_s.

## What is NOT modelled

* Wakefields (longitudinal, transverse, resistive-wall).
* Image-charge effects from the beam pipe (open boundary in PIC).
* Synchrotron radiation damping.
* Coupling between transverse and longitudinal beyond the standard
  dispersion / phase-slip first-order mixing.
* Acceleration in the Sacherer ODE solver — that path assumes constant
  β, γ.

For these, use the matrix tracker plus the bunched PIC SC kick (for
electromagnetic SC under acceleration) or extend the relevant module.
