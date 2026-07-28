# Linac_Gen: Particle Tracking Code with PIC Space Charge

**Date:** 2026-04-03
**Status:** Approved

## Overview

Linac_Gen is a full-featured particle tracking code for proton/ion linear accelerators, inspired by TraceWin (CEA Saclay). It provides multi-particle tracking with self-consistent 3D PIC space charge, envelope/matrix tracking, beam matching, Monte Carlo error studies, and an interactive desktop GUI.

**Target use case:** High-intensity proton/ion linac design and simulation (like ESS, LINAC4, SNS, IFMIF).

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python + C/C++ extensions (pybind11) | Python for rapid development, C++ kernels for PIC performance |
| Architecture | Library (`linac_gen`) + Separate GUI (`linac_gen_gui`) | Library usable headless on HPC; GUI is a consumer of the library |
| Space charge | Full 3D PIC with FFT Poisson solver | Required for accurate bunched-beam dynamics |
| GUI framework | PyQt6 + pyqtgraph | Mature desktop toolkit, fast real-time plotting |
| Input format | TraceWin `.dat` subset (core cards; see I/O section for scope) | Leverage existing lattice files, ease adoption |
| Scale | Up to 1M particles, 128^3 grids | Production-quality design studies |
| Coordinates | (x, x', y, y', phi, W) in mm, mrad, deg, MeV | TraceWin/Wangler convention |

## Architecture

### Two-Package Structure

```
Linac_Gen/
├── linac_gen/                    # Core computation library (pip installable)
│   ├── core/                     # Particle, Beam, Lattice data structures
│   │   ├── particle.py           # Particle species (proton, ion, etc.)
│   │   ├── reference.py          # ReferenceParticle (synchronous state)
│   │   ├── beam.py               # Beam class (particle array + statistics)
│   │   ├── lattice.py            # Lattice container (ordered list of elements)
│   │   └── constants.py          # Physics constants
│   ├── elements/                 # Lattice element classes
│   │   ├── base.py               # Abstract base element
│   │   ├── drift.py
│   │   ├── quadrupole.py         # Hard-edge + Enge fringe fields
│   │   ├── solenoid.py
│   │   ├── rf_gap.py             # Thin RF gap with TTF
│   │   ├── field_map.py          # 1D/2D/3D field maps
│   │   ├── dipole.py             # Sector/rectangular + edge focusing
│   │   ├── multipole.py          # Sextupole, octupole, general 2n-pole
│   │   ├── steerer.py            # Thin H/V correctors
│   │   ├── aperture.py           # Circular/rectangular/elliptical
│   │   ├── marker.py             # Named marker, diagnostics point
│   │   ├── space_charge_comp.py  # Partial neutralization
│   │   └── thin_lens.py
│   ├── tracking/                 # Tracking engines
│   │   ├── tracker.py            # Main tracking loop
│   │   ├── matrix_tracking.py    # Linear 6x6 transfer matrix tracking
│   │   ├── envelope.py           # RMS envelope solver (Sacherer)
│   │   └── rk4.py                # Runge-Kutta for field maps
│   ├── pic/                      # PIC space charge module
│   │   ├── pic_solver.py         # Python API for PIC cycle
│   │   ├── charge_deposition.py  # CIC weighting (C extension)
│   │   ├── poisson_solver.py     # FFT Poisson + boundary conditions
│   │   ├── field_interpolation.py
│   │   └── lorentz_boost.py      # Beam frame transforms
│   ├── distributions/            # Beam generators
│   │   ├── waterbag.py
│   │   ├── kv.py
│   │   ├── gaussian.py
│   │   ├── parabolic.py
│   │   ├── uniform.py
│   │   └── from_file.py
│   ├── diagnostics/              # Beam analysis
│   │   ├── twiss.py
│   │   ├── emittance.py
│   │   ├── phase_space.py
│   │   ├── beam_loss.py
│   │   ├── halo.py
│   │   └── moments.py
│   ├── matching/                 # Matching & optimization
│   │   ├── matcher.py
│   │   ├── periodic.py
│   │   └── objectives.py
│   ├── errors/                   # Error study framework
│   │   ├── error_model.py
│   │   ├── monte_carlo.py
│   │   └── correction.py
│   ├── io/                       # Input/Output
│   │   ├── tracewin_parser.py
│   │   ├── tracewin_writer.py
│   │   ├── field_map_reader.py
│   │   └── distribution_io.py
│   └── csrc/                     # C/C++ extensions
│       ├── CMakeLists.txt
│       ├── pic_kernels.cpp       # Charge deposition + field interpolation
│       └── poisson_fft.cpp       # 3D FFT Poisson (FFTW)
│       # v2: boris_pusher.cpp (Boris integrator as alternative to RK4 for B-field maps)
│
├── gui/                          # GUI package (separate distribution)
│   ├── pyproject.toml
│   └── linac_gen_gui/
│       ├── __init__.py
│       ├── app.py                # Entry point
│       ├── main_window.py
│       ├── widgets/
│   │   ├── lattice_editor.py     # Lattice table/tree
│   │   ├── beam_config.py        # Beam parameter panel (always visible)
│   │   ├── envelope_plot.py
│   │   ├── phase_space_plot.py
│   │   ├── loss_map_plot.py
│   │   ├── emittance_plot.py
│   │   ├── lattice_layout.py
│   │   └── error_study_view.py
│   ├── dialogs/
│   │   ├── simulation_settings.py
│   │   ├── matching_dialog.py
│   │   └── error_study_dialog.py
│   └── resources/
│
├── tests/
├── examples/
├── docs/
└── pyproject.toml
```

**Key separation:** `linac_gen` has zero GUI dependencies (only NumPy, SciPy, pybind11 C extensions). `linac_gen_gui` depends on `linac_gen` + PyQt6 + pyqtgraph.

## Core Data Model

### Particle Species

```python
species = Particle(mass=938.272, charge=1, name="proton")    # MeV/c^2, q/e
species = Particle(mass=1875.613, charge=1, name="deuteron")
species = Particle(mass=938.272, charge=-1, name="H-")
species = Particle(mass=A * AMU, charge=Z, name="custom")     # general ion
```

### Reference Particle (Synchronous State)

The `ReferenceParticle` tracks the synchronous particle's absolute state as it progresses through the lattice. This is the single source of truth for the beam's reference energy, phase, and derived relativistic quantities.

```python
beam.ref = ReferenceParticle(
    species=Particle(...),
    w_kin=3.0,          # absolute kinetic energy (MeV) -- updated by RF elements
    phi_s=0.0,          # absolute RF phase (deg) -- updated by drifts and RF
    s=0.0,              # position along lattice (mm)
)
# Derived (read-only, recomputed when w_kin changes):
beam.ref.beta           # relativistic beta = v/c
beam.ref.gamma          # relativistic gamma
beam.ref.bg             # beta*gamma
beam.ref.brho           # magnetic rigidity (T.m)
beam.ref.wavelength     # RF wavelength = c / frequency (mm)
```

**Every element that changes energy or phase updates `beam.ref` first** (RF gaps update `w_kin` and `phi_s`; drifts update `phi_s` and `s`). Transfer matrices and tracking routines read `beta`, `gamma`, `brho` from `beam.ref` — never from the particles.

### Beam Class

```python
beam.ref         # ReferenceParticle (synchronous state)
beam.particles   # shape (N, 6) float64: [x, xp, y, yp, dphi, dW]
                 #   x (mm), xp (mrad): transverse position and slope
                 #   y (mm), yp (mrad): transverse position and slope
                 #   dphi (deg): phase DEVIATION from beam.ref.phi_s
                 #   dW (MeV): kinetic energy DEVIATION from beam.ref.w_kin
beam.species     # Particle object
beam.current     # beam current (mA)
beam.frequency   # RF frequency (MHz)
beam.n_particles # number of macroparticles
beam.lost        # boolean mask, shape (N,) -- True if lost
beam.loss_table  # structured array: (particle_id, s, x, y, energy, element_name)
```

**Critical:** columns 4 and 5 of the particle array are **deviations** (dphi, dW), not absolute values. The absolute phase and energy of particle i are:
- `phi_i = beam.ref.phi_s + beam.particles[i, 4]`
- `W_i = beam.ref.w_kin + beam.particles[i, 5]`

This avoids floating-point precision loss (small deviations around a large reference value) and makes the RF gap model unambiguous: the energy kick is `dW_i = q*V0*T*cos(phi_s + dphi_i) - q*V0*T*cos(phi_s)` for the deviation, while the reference particle gets `beam.ref.w_kin += q*V0*T*cos(phi_s)`.

The `(N, 6)` contiguous float64 array is cache-friendly and maps directly to C extensions. Lost particles are masked, not removed (avoids costly array resizing).

### Beam Configuration

```python
beam_config = BeamConfig(
    species="proton",
    energy=3.0,             # initial absolute kinetic energy (MeV) -> sets beam.ref.w_kin
    frequency=352.21,       # RF frequency (MHz)
    current=60.0,           # beam current (mA)
    n_particles=100_000,
    distribution="waterbag", # "waterbag", "kv", "gaussian", "parabolic", "uniform", "file"
    cutoff=3.0,             # sigma cutoff for gaussian
    emit_nx=0.25,           # normalized RMS emittance x (mm.mrad)
    alpha_x=1.0,
    beta_x=0.12,            # Twiss beta (m)
    emit_ny=0.25,
    alpha_y=-0.5,
    beta_y=0.08,
    emit_z=0.30,            # longitudinal emittance (deg.MeV)
    alpha_z=0.0,
    beta_z=1.5,             # (deg/MeV)
    source="generate",      # "generate" or "file"
    distribution_file=None,
)
```

`BeamConfig.energy` is the **initial** absolute kinetic energy. It initializes `beam.ref.w_kin`. The distribution generator produces particles with deviations `(dphi, dW)` centered at zero, spread according to the longitudinal Twiss/emittance.

Auto-computed display fields: beta, gamma, beta*gamma, geometric emittances, RMS beam sizes, perveance, tune depression estimate.

### Lattice Class

```python
lattice = Lattice()
lattice.elements       # ordered list of Element objects
lattice.total_length   # sum of element lengths (mm)
lattice.frequency      # current RF frequency (MHz)
lattice.get_s_positions()  # cumulative s at each element start/end
```

### Simulation Controller

```python
sim = Simulation(lattice, beam, space_charge=SpaceChargeConfig(...))
sim.run()                    # full multi-particle tracking
sim.run_envelope()           # envelope-only mode
results = sim.get_results()  # DiagnosticResults object
```

## Element Physics Models

### Element Interface (Capability-Based)

Elements implement a subset of interfaces based on their physics:

```python
class Element(ABC):
    name: str
    length: float           # mm (0 for thin elements)
    aperture: float         # mm (0 = no aperture check)
    n_steps: int            # integration sub-steps (0 for thin/passive)

class TransferMapElement(Element):
    """Elements with a linear 6x6 transfer matrix (drift, quad, solenoid, dipole)."""
    def transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        # ds (mm): slice length. If None, uses self.length (full element).
        # The tracker calls this with ds = self.length / (2 * n_steps) for half-steps.
    def track(self, beam: Beam, ds: float = None) -> None:
        # 1. Advances beam.ref (s += ds, phi_s updated for drift phase slip)
        # 2. Applies transfer_matrix(beam.ref, ds) to all particles
        # beam.ref is advanced BEFORE the particle update so that the matrix
        # uses the correct reference state at the slice midpoint.

class ThinKickElement(Element):
    """Zero-length elements that apply instantaneous kicks (RF gap, multipole, steerer)."""
    length = 0
    def apply_kick(self, beam: Beam) -> None:
        # For RF gaps: updates beam.ref.w_kin (synchronous energy gain) and
        # beam.ref.phi_s, then applies deviation kicks to particles.
        # For steerers/multipoles: beam.ref unchanged (no energy/phase change).
    def kick_matrix(self, ref: ReferenceParticle) -> np.ndarray:  # linearized 6x6

class FieldMapElement(Element):
    """Elements tracked via RK4 through imported field data."""
    def track_rk4(self, beam: Beam, ds: float) -> None:
        # 1. Integrates the reference particle through ds via RK4 to get
        #    updated w_kin, phi_s, s. Updates beam.ref at the end of the slice.
        # 2. Integrates all particles through the field, storing results as
        #    deviations from the updated beam.ref.
        # Note: in the tracker loop (half-SC / RK4 / half-SC), the first SC
        # half-kick uses the slice-entrance beam.ref; track_rk4 then advances
        # beam.ref; the second SC half-kick uses the slice-exit beam.ref.
        # This is intentional and consistent with the symmetric splitting.
    # No meaningful transfer_matrix(); envelope mode uses a fitted linear matrix

class PassiveElement(Element):
    """Zero-length elements with no dynamics (aperture, marker, diag, SC comp)."""
    length = 0
    def apply(self, beam: Beam) -> None:  # aperture check, record diagnostics, set SC factor
```

### Tracker Strategies (per element type)

The tracker dispatches to element-specific strategies, not a single universal loop:

```python
# For TransferMapElement (drift, quad, solenoid, dipole) with space charge:
for step in range(element.n_steps):
    apply_half_map(beam, element, ds/2)    # half-step via transfer matrix
    if space_charge_enabled:
        pic_solver.kick(beam, ds)           # full SC kick
    apply_half_map(beam, element, ds/2)    # half-step via transfer matrix
    check_aperture(beam, element)
# -> Symplectic (split-operator on Hamiltonian that decomposes into linear + SC)

# For ThinKickElement (RF gap, multipole, steerer):
element.apply_kick(beam)                   # instantaneous kick, updates beam.ref for RF
# -> No splitting needed; thin kicks are exact maps

# For FieldMapElement (field maps with E and/or B):
for step in range(element.n_steps):
    if space_charge_enabled:
        pic_solver.kick(beam, ds/2)         # half SC kick
    element.track_rk4(beam, ds)             # full RK4 step through interpolated fields
    if space_charge_enabled:
        pic_solver.kick(beam, ds/2)         # half SC kick
    check_aperture(beam, element)
# -> Second-order in the SC-vs-external splitting (symmetric composition).
#    NOT exactly symplectic overall: RK4 itself is not symplectic, so the combined
#    map has O(ds^5) symplectic error per step from RK4. Acceptable for single-pass
#    linac tracking where field maps span a finite region.

# For PassiveElement (aperture, marker, diag, SC comp):
element.apply(beam)                         # no dynamics, just side effects
```

The "symplectic by construction" property applies **only** to the TransferMapElement split-operator loop, not to RK4 field-map tracking.

### Element Catalog

| Element | Key Parameters | Physics Model |
|---------|---------------|---------------|
| Drift | L(mm) | 6x6 matrix with longitudinal velocity-dependent path length: (5,6) = -2piL/(beta^2*gamma^3*mc^2*lambda) |
| Quadrupole | L, G(T/m) | cos/cosh thick-lens 6x6. Strength k=qG/(mc*beta*gamma). Optional Enge fringe field (a1-a6 coefficients) |
| Solenoid | L, B0(T) | 4x4 coupled x-y matrix (Larmor rotation). k_s=qB/(2mc*beta*gamma). Thin fringe kicks at entry/exit |
| RF Gap | V0, phi_s, freq, T/T'/S/S' | Thin-lens: energy kick dW=qV0*T*cos(phi), transverse RF defocusing, adiabatic damping scaling |
| Field Map | filename, type(1D/2D/3D), scale, phase | RK4 integration through imported field data. Tricubic interpolation. 1D off-axis Bessel expansion |
| Dipole | theta, rho, e1, e2 | Sector bend 6x6 with dispersion + thin edge focusing matrices. Combined-function supported via k1 parameter |
| Multipole | order n, b_n, a_n | Thin kick: dx'+idy' = -L/(Brho) * sum (b_n+ia_n)/(n-1)! * (x+iy)^(n-1) |
| Steerer | Bx*L, By*L (T.m) | Thin dipole kick: dx'=qByL/p, dy'=qBxL/p |
| Aperture | type, a, b | Loss check: circular r>R, rectangular |x|>a or |y|>b, elliptical (x/a)^2+(y/b)^2>1 |
| Space Charge Comp | factor (0-1) | Multiplies SC kick by (1-factor) for partial neutralization |
| Marker/Diag | name | Zero-length; triggers diagnostic recording |

### Coordinate Convention

TraceWin/Wangler convention:
- Transverse: (x, x', y, y') in mm, mrad where x' = px/pz
- Longitudinal: (phi, W) in deg, MeV -- phase relative to synchronous particle, kinetic energy deviation
- Synchronous phase: phi_s < 0 for stable acceleration (cos convention: dW = qV0*T*cos(phi_s))
- Lengths in mm, gradients in T/m, fields in T or MV/m, frequency in MHz

### Tracking Loop

See "Tracker Strategies (per element type)" above. The main tracker dispatches based on element interface; there is no single universal loop.

## PIC Space Charge Solver

### PIC Cycle (per kick)

1. **Lorentz boost** to beam rest frame: z_rest = gamma_ref * z_lab (using beam.ref.gamma)
2. **Charge deposition**: CIC (Cloud-In-Cell) weighting onto 3D Cartesian grid. Each particle contributes to 8 nodes (2^3)
3. **Poisson solve**: FFT-based with Hockney's method for open boundaries. Zero-pad to (2Nx, 2Ny, 2Nz), convolve with integrated Green's function, inverse FFT
4. **E-field computation**: central finite differences E = -grad(phi)
5. **Field interpolation**: same CIC shape function (momentum conservation)
6. **Lorentz boost back** to lab frame (see below)
7. **Convert E-field kicks to slope/energy kicks** and apply to particles

### PIC Kick: Field-to-Slope Conversion (Critical)

The Poisson solver produces electric field **E_rest** in the beam rest frame. Converting to kicks on **(x', y', dphi, dW)** requires careful handling:

**Step 6 — Lorentz boost of fields:**
In the beam rest frame, the interaction is purely electrostatic. Boosting back to the lab frame and accounting for the self-magnetic force (beam current creates B_theta which partially cancels the electric repulsion):

```
F_x,lab = q * E_x,rest / gamma_ref       # net transverse force (E - v*B cancellation)
F_y,lab = q * E_y,rest / gamma_ref       # gives the 1/gamma^2 suppression at high energy
F_z,lab = q * E_z,rest / gamma_ref^2     # longitudinal (no magnetic cancellation for z)
```

The factor is `1/gamma` (not `gamma`) for the **net force** because `F_transverse = qE_rest(1 - beta^2)/gamma = qE_rest/gamma` after electric-magnetic cancellation.

**Step 7 — Apply kicks to particle coordinates:**

The state uses slopes `x' = px/pz`, not momenta. The conversion from force to slope kick over step ds:

```
# Transverse: dx' = F_x * ds / (pz * c) where pz ~ p_ref = m*c*beta*gamma for paraxial beam
dx'_i = (q * E_x,rest,i * ds) / (m * c^2 * beta_ref^2 * gamma_ref^2)    # mrad, with unit conversions
dy'_i = (q * E_y,rest,i * ds) / (m * c^2 * beta_ref^2 * gamma_ref^2)

# Longitudinal: energy kick directly
dW_i += (q * E_z,rest,i * ds) / gamma_ref^2                               # MeV, with unit conversions

# Phase: no direct kick (phase changes via the energy-dependent drift, handled in the drift half-step)
```

The reference values `beta_ref`, `gamma_ref` come from `beam.ref`. Per-particle energy deviations are neglected in the SC kick denominator (valid when dW/W << 1, which holds for well-bunched beams). This is the standard approximation used by TraceWin and IMPACT.

### Grid Configuration

```python
sc_config = SpaceChargeConfig(
    nx=64, ny=64, nz=64,
    boundary="open",          # "open" or "conducting"
    solver="fft",
    grid_extent=3.0,          # grid = N * sigma_rms in each dimension
    shape_order=1,            # 1=CIC, 2=TSC (future)
)
```

- Open boundaries: Hockney doubled-grid FFT convolution. Memory: (2N)^3 complex. Cost: O(8N^3 log(8N^3))
- Conducting boundaries: Discrete Sine Transform for Dirichlet phi=0. Circular and rectangular pipe

**Grid extent policy:**

- **Default: fixed grid per run.** Grid bounds are computed once from the initial beam (grid_extent * sigma_rms in each dimension) and held constant for the entire simulation. The Green's function is precomputed once. This ensures reproducibility and avoids numerical noise from remapping.
- **Optional: adaptive resizing** at coarse intervals (e.g., every N_adapt elements, default off). When enabled, the grid is resized only if the beam RMS size changes by more than 30% from the current grid assumption (hysteresis prevents oscillation). A resize triggers Green's function recomputation.
- For conducting boundaries, the grid always matches the pipe geometry (fixed by definition).

```python
sc_config = SpaceChargeConfig(
    ...
    grid_mode="fixed",          # "fixed" (default) or "adaptive"
    adaptive_interval=50,       # recheck every N elements (only if adaptive)
    adaptive_threshold=0.3,     # resize if sigma changes by >30%
)
```

### C/C++ Kernels (pybind11 + FFTW)

| Kernel | Purpose | Expected Speedup |
|--------|---------|-----------------|
| charge_deposition_3d() | Scatter particle charges to grid (1M particles, 8 adds each) | 50-100x vs NumPy |
| poisson_fft_3d() | 3D FFT on doubled grid, Green's function multiply, inverse FFT | 10-20x (FFTW vs scipy) |
| field_interpolation_3d() | Gather fields from grid to particles | 50-100x vs NumPy |

Python handles Lorentz boosts, grid setup, gradient computation (vectorized NumPy).

### Analytical Space Charge (envelope mode)

Uniform ellipsoid model for fast envelope tracking:
```
E_x = 3*I*x / (4*pi*eps0*c*beta*gamma^2*a*(a+b)*L_bunch)
```
Linear forces parameterized by RMS beam sizes. No grid, no particles needed.

## Diagnostics

### What is recorded and when

**Always recorded** (at every element exit, lightweight -- scalars only):

```python
results.s              # s-positions (mm)
results.sigma_x/y/phi  # RMS sizes
results.sigma_w        # RMS energy spread
results.emit_x/y/z     # geometric RMS emittances (mm.mrad, deg.MeV)
results.emit_nx/ny      # normalized RMS emittances
results.alpha_x/y, .beta_x/y, .gamma_x/y  # Twiss parameters
results.centroid        # (N_pos, 6) first moments
results.halo_x/y       # halo parameter per position
results.loss_table      # (particle_id, s, x, y, energy, element)
results.transmission    # cumulative transmission (%) vs s
```

These are computed on-the-fly from the live particle array (moments, not copies). Memory cost: O(N_elements * ~30 floats) -- negligible.

**Reference particle history** (always recorded alongside moments):

```python
results.ref_w_kin      # reference kinetic energy (MeV) at each s-position
results.ref_phi_s      # reference RF phase (deg) at each s-position
results.ref_beta       # relativistic beta at each s-position
results.ref_gamma      # relativistic gamma at each s-position
results.ref_bg         # beta*gamma at each s-position
```

This is essential: since particle arrays store deviations (dphi, dW), the reference state is required to reconstruct absolute coordinates. The reference history is lightweight (5 floats per position) and always stored.

**Full particle snapshots** (expensive, opt-in):

```python
results.beam_at(s)      # returns (particles: ndarray(N,6), ref: ReferenceParticle)
```

Every snapshot is a **(particle_array, reference_state) pair**. The reference state at that s-position is always stored alongside the particle deviations, making each snapshot self-describing.

Snapshots are saved **only at Marker/Diag elements**, not at every element exit. Default: save at first element, last element, and any supported diagnostic element (Marker, DIAG_SIZE, DIAG_EMIT, DIAG_PHASE). The user can request additional snapshot locations via:

```python
sim = Simulation(..., snapshot_locations=[0.0, 500.0, "END"])  # s-positions or "END"
# or
sim = Simulation(..., snapshot_every_n=50)  # every 50th element
```

At 1M particles, each snapshot is ~46 MB (1M * 6 * 8 bytes). With 10 snapshots, that's ~460 MB -- acceptable. With 200 snapshots (every element), it's ~9 GB -- not acceptable. The default prevents this.

**GUI live updates**: the GUI receives **moments only** (sigma, emittance, Twiss) per element via signals, not full particle arrays. Phase space plots at the cursor position are rendered by requesting a single snapshot from the worker thread on demand (one at a time, not stored).

**HDF5 output**: moments stored for all positions; particle snapshots stored only at requested locations.

### Computed quantities

Computed from particles using centralized moments:
- Emittance: eps = sqrt(<x^2><x'^2> - <xx'>^2) over non-lost particles
- Twiss: beta = <x^2>/eps, alpha = -<xx'>/eps
- Halo: H = <x^4>/<x^2>^2 - 1

## Matching Module

```python
matcher = Matcher(lattice, beam)
matcher.add_variable("QUAD_01", "gradient", min=-15, max=15)   # T/m
matcher.add_objective("END", "alpha_x", target=0.0)
result = matcher.solve(method="least_squares")
```

Three modes:
1. **Envelope matching** (fast): transfer matrices + envelope equations, Levenberg-Marquardt
2. **Periodic matching**: find settings producing periodic Twiss for a repeating cell
3. **Multi-particle matching** (slow): full tracking with SC per iteration, Nelder-Mead or differential evolution

Backends: SciPy least_squares, minimize, differential_evolution.

## Error Study Framework

```python
error_study = ErrorStudy(lattice, beam, n_seeds=100)
error_study.add_error("QUAD_*", "dx", distribution="gaussian", sigma=0.1, cutoff=3)
error_study.add_error("CAV_*", "phase", distribution="gaussian", sigma=1.0)
results = error_study.run(n_workers=8)  # multiprocessing
```

Error types: alignment (dx, dy, dz, roll, pitch, yaw), magnetic (gradient_rel, field_rel), RF (phase, amplitude_rel), beam (current, energy, emittance, mismatch).

Aggregation: mean, std, percentiles, min/max over seeds for all diagnostics. Loss map with percentile selection.

### Orbit Correction

```python
error_study.enable_correction(
    bpm_pattern="BPM_*",
    steerer_pattern="STEER_*",
    method="svd",           # or "one_to_one"
    bpm_noise=0.01,         # mm
)
```

Methods: one-to-one (local) and SVD (global response matrix inversion).

## GUI Application

### Framework

PyQt6 + pyqtgraph (fast real-time plotting of 1M+ points). Matplotlib backend for publication-quality export.

### Layout

```
+---------------+----------------------------+---------------+
| Lattice       | Central Plot Area (tabbed) | Beam Setup    |
| Tree/Table    |                            | Panel         |
|               | Tabs: Envelope | Phase     |               |
| - DRIFT 1     |   Space | Loss Map |       | Species       |
| - QUAD 1      |   Emittance | Layout       | Energy        |
| - CAV 1       |                            | RF Frequency  |
| - ...         |   [Interactive Plot]       | Current       |
|               |                            | N Particles   |
|---------------|                            | Distribution  |
| Element       |                            | Twiss X/Y/Z  |
| Properties    |                            | Emittances    |
|               |                            |               |
| L: 100mm      |                            | [Preview]     |
| G: 5.2 T/m    |                            | [Regenerate]  |
| R: 20mm       |----------------------------+---------------|
|               | Lattice Layout Strip       | SC Config     |
|               | [D][Q][D][CAV][Q]...       | nx/ny/nz: 64  |
|               |                            | BC: Open      |
+---------------+----------------------------+---------------+
| Status: Tracking 45% | 850k/1M alive | emit: 0.25 mm.mrad |
+-----------------------------------------------------------+
```

### Panels

- **Left: Lattice Editor** -- tree/table of elements, drag to reorder, right-click insert/delete, color-coded by type
- **Left Bottom: Element Properties** -- editable form for selected element
- **Center: Tabbed Plots** -- Envelope, Phase Space, Loss Map, Emittance, Lattice Layout (all interactive, cursor-synchronized)
- **Right: Beam Setup** -- always-visible panel with species, energy, current, frequency, distribution type, Twiss parameters per plane, emittances. Auto-computes beta/gamma/beam-sizes
- **Right Bottom: SC Config** -- grid dimensions, boundary type, pipe radius
- **Bottom: Lattice Strip** -- thin schematic with synchronized cursor
- **Status Bar** -- progress, particle count, key parameters

### Dialogs

- Simulation Settings, Matching, Error Study, Field Map Viewer, Distribution Viewer

### Threading

- GUI thread: all Qt widgets
- Worker QThread: runs simulation, emits progress/step_complete/finished signals
- Live plot updates as each element completes
- Stop button via shared flag

## I/O

### TraceWin .dat Parser — Compatibility Scope

**"TraceWin .dat compatible" means a defined subset, not full parity.** The parser will handle the most-used element cards; unsupported cards emit a warning and are skipped (not a fatal error), allowing partial lattice import.

**v1 supported cards:**

| Card | Status | Notes |
|------|--------|-------|
| FREQ | Supported | Stateful frequency setting |
| DRIFT | Supported | |
| QUAD | Supported | Hard-edge; Enge fringe via optional extra params |
| SOLENOID | Supported | |
| GAP | Supported | Thin RF gap with TTF file reference |
| FIELD_MAP types 1,2,7 | Supported | 1D electric, 2D electric, 2D E+B |
| THIN_STEERING | Supported | |
| APERTURE types 0,1 | Supported | Circular, rectangular |
| MARKER | Supported | |
| DIAG_SIZE, DIAG_EMIT, DIAG_PHASE | Supported | DIAG_PHASE triggers a full particle snapshot |
| BEND + EDGE | Supported | Sector bend with edge matrices |
| SPACE_CHARGE_COMP | Supported | |
| SET_ADV, ADJUST | Supported | Stored as matching metadata |
| END | Supported | |

**v1 unsupported (skip with warning):**

| Card | Reason |
|------|--------|
| FIELD_MAP types 3,4,5,6,8+ | 3D field maps and exotic types deferred to v2 |
| DTL_CEL, NCELLS | Specialized DTL cell definitions — use FIELD_MAP or GAP instead |
| CHOPPER | Phase-based beam chopping deferred |
| SET_BEAM, INPUT_DIST | Beam redefinition mid-lattice deferred |
| MULTIPOLE (TraceWin syntax) | General multipole with TraceWin-specific parameterization |
| LATTICE (periodicity marker) | Periodic section detection deferred |
| Macro/loop constructs | TraceWin scripting macros not supported |

**Unsupported card handling:** `parser.parse("file.dat", strict=False)` skips unknown cards with a warning list. `strict=True` raises on any unknown card. The warning list is returned so the user knows what was skipped.

### Field Map Reader — v1 Scope

**v1 supported formats:**

| Format | Extension | Description |
|--------|-----------|-------------|
| TraceWin 1D on-axis | .edz | nz, z-range, Ez values. Off-axis via Bessel expansion |
| TraceWin 2D electric (r,z) | .edz (2D header) | nr, nz, dr, dz, then Ez(r,z), Er(r,z) grid |
| TraceWin 2D E+B (r,z) — type 7 | .edz (E+B header) | nr, nz, dr, dz, then Ez(r,z), Er(r,z), Bz(r,z), Br(r,z) grid. Four field components on the same (r,z) mesh. Used for solenoid-overlapped cavities |
| Generic CSV | .csv | Header with column names, then x/y/z/Ex/Ey/Ez[/Bx/By/Bz] columns. Magnetic columns optional |

**v2 additions (deferred):** Superfish T7, CST native export, COMSOL export, 3D Cartesian field maps. These are straightforward extensions of the same FieldMapData structure but require format-specific parsers that are not essential for initial functionality.

### Distribution I/O

Distribution files store **absolute** coordinates (not deviations), matching the TraceWin convention for interoperability. The reference state is stored in a header so round-tripping is unambiguous.

**Export format** (ASCII):
```
# Linac_Gen distribution file
# species: proton
# w_kin_ref: 3.000000 MeV        <- reference kinetic energy
# phi_ref: -30.000000 deg        <- reference RF phase
# frequency: 352.210000 MHz
# current: 60.000000 mA
# n_particles: 100000
# columns: x(mm) xp(mrad) y(mm) yp(mrad) phi_abs(deg) W_abs(MeV)
 0.123  -0.456   0.789  1.234   -45.600   3.0012
-0.567   0.890  -0.345  0.678   -38.200   2.9987
...
```

Columns 5-6 are **absolute** phi and W (not deviations). On import, the loader subtracts the header reference values to produce the internal (dphi, dW) representation. On export, it adds `beam.ref.phi_s` and `beam.ref.w_kin` to the deviations.

**Import from external files** (no header): if the header is missing, the user must provide the reference state via `BeamConfig.energy` and an assumed `phi_ref=0`. A warning is emitted.

TraceWin `.dst` binary format: same absolute-coordinate convention, with reference state in the binary header.

### Results Output

HDF5 with groups:
```
results.h5
├── lattice/              # element list, s-positions
├── beam_config/          # initial beam parameters
├── reference/            # ref.w_kin, ref.phi_s, ref.beta, ref.gamma vs s (full history)
├── envelope/             # sigma_x, sigma_y, emit_x, ... vs s
├── particles/            # snapshots at saved positions
│   ├── s_0000/
│   │   ├── data          # (N, 6) float64 array (deviations)
│   │   └── ref           # {w_kin, phi_s, beta, gamma, s} at this location
│   ├── s_0042/
│   │   ├── data
│   │   └── ref
│   └── ...
├── losses/               # loss table
└── error_study/          # per-seed results (if applicable)
```

Each particle snapshot group contains both the deviation array and the reference state, making every snapshot self-describing.

## Performance Targets

| Scenario | Target Time |
|----------|-------------|
| Envelope tracking, 200-element lattice | < 1 second |
| 100k particles, 64^3 grid, 200 elements | < 2 minutes |
| 1M particles, 128^3 grid, 200 elements | < 30 minutes |
| Error study: 100 seeds x 100k particles | < 3 hours (8 cores) |
