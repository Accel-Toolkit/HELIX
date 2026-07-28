# Quadrupole

A quadrupole focuses the beam in one transverse plane and defocuses
in the other.  The workhorse focusing element of any linac:
alternating QF / QD pairs (a FODO cell) hold the beam together while
it accelerates.

## TL;DR (TraceWin users)

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `QUAD L G aperture [skew g3 g4 g5 g6 gfr]` | `Quadrupole(name, length, gradient, ...)` |
| Length L | mm | mm |
| Gradient G | T/m | T/m |
| Skew | degrees | degrees |
| g3..g6 | T/m^(n+1) | **honoured** as thin Multipole kicks distributed over every tracked slice |
| gfr | (TraceWin: gradient-fall-region) | parsed, currently unused |

Conventions:

* HELIX uses signed gradient: G > 0 focuses x, defocuses y for
  positive-charge particles (proton).  For H⁻, sign flips
  automatically inside the matrix.
* `g3..g6` are TraceWin's higher-multipole content.  Until recently
  these were parsed but ignored; HELIX now applies them as thin
  [Multipole](11_multipole.md) kicks **distributed across every
  tracked slice** (split-operator: half of each slice's share
  before the linear transport, half after), so the integrated kick
  over the full magnet equals the design value.
* `gradient_rel` adds a fractional gradient error
  (`effective_G = G·(1 + gradient_rel)`) — used by error studies.

## Tutorial (newcomers)

A quadrupole's magnetic field grows linearly with distance from the
axis: B_y = G·x, B_x = G·y.  A particle off-axis in x feels a force
pushing it back toward the axis (focusing); a particle off-axis in y
feels a force pushing it *away* (defocusing).  Hence "alternating
gradient" focusing — pair a QF (focuses x) with a QD (focuses y) and
the net effect over a cell is focusing in both planes.

### Hard-edge 6×6 transfer matrix

For a quadrupole of length L and signed strength k² = q·G / (Bρ)
(positive ⇒ focusing in x for positive charge), the full 6×6
transfer matrix in coordinates `(x, x', y, y', Δφ, ΔW)` is:

$$
M_{\text{quad}} =
\begin{pmatrix}
C       & S/k     & 0       & 0       & 0 & 0      \\
-kS     & C       & 0       & 0       & 0 & 0      \\
0       & 0       & C_h     & S_h/k   & 0 & 0      \\
0       & 0       & kS_h    & C_h     & 0 & 0      \\
0       & 0       & 0       & 0       & 1 & M_{56} \\
0       & 0       & 0       & 0       & 0 & 1
\end{pmatrix}
$$

where, for k² > 0:

* **Focusing block** (x, x'):  C = cos(kL), S = sin(kL)
* **Defocusing block** (y, y'):  C_h = cosh(kL), S_h = sinh(kL)

For k² < 0 (gradient sign reversed), the focus and defocus blocks
swap planes — the (x, x') block becomes hyperbolic and (y, y')
becomes trigonometric.  HELIX dispatches via `np.sign(k2)` so the
single matrix formula handles both polarities and both charge
signs (proton vs H⁻) automatically.

The longitudinal **drift-like phase slip** is identical to a free
drift of the same length:

$$
M_{56} = -\,\frac{360 \, L \, [\text{mm}]}{\beta^{3} \, \gamma^{3} \, m \, \lambda_{\text{RF}}}
\quad [\text{deg}/\text{MeV}]
$$

| Block | Effect |
|---|---|
| (x, x') 2×2 | trigonometric / hyperbolic depending on sign(G) |
| (y, y') 2×2 | the *opposite* of the (x, x') sign |
| (Δφ, ΔW) 2×2 | drift-like phase slip; off-energy particles slip in φ |
| Cross blocks | all zero (pure quad has no x-y or transverse-longitudinal coupling) |

A skew quadrupole (`skew_angle ≠ 0`) introduces (x, y), (x, y'),
(x', y), (x', y') cross-block coupling via a sandwich
`R(θ)·M·R(-θ)` rotation.  Tilt-induced misalignment is the same
operation but driven by a per-seed random tilt.

### Example: a single FODO cell

```python
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

beam_cfg = BeamConfig(
    species="proton", energy=3.0, frequency=352.21,
    current=0.0, n_particles=2000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.20, emit_ny=0.20, emit_z=0.30,
    alpha_x=0.0, beta_x=2.0,
    alpha_y=0.0, beta_y=2.0,
    alpha_z=0.0, beta_z=1.0,
)
beam = create_beam(beam_cfg, seed=0)

lat = Lattice()
lat.add(Quadrupole(name="QF", length=100.0, gradient=+10.0))
lat.add(Drift(name="D1", length=200.0))
lat.add(Quadrupole(name="QD", length=100.0, gradient=-10.0))
lat.add(Drift(name="D2", length=200.0))

sim = Simulation(lat, beam)
res = sim.run()
print(f"σ_x along cell: {[f'{v:.2f}' for v in res.sigma_x[::5]]}")
```

You should see σ_x oscillate as the beam alternately focuses and
drifts.

### Higher-pole content

A real quadrupole has imperfections that produce sextupole (g3),
octupole (g4), decapole (g5), and dodecapole (g6) field harmonics.
TraceWin's `.dat` lists these on the QUAD card; HELIX applies them
as thin [Multipole](11_multipole.md) kicks distributed **per
tracked slice** — half of the slice's share before the linear
matrix transport and half after (split-operator, symplectic to
second order).  The element's total g3..g6 content is spread
uniformly across the substeps, so the integrated kick over the full
magnet equals the design value.

### Skew quadrupoles

Setting `skew_angle != 0` rotates the quadrupole's principal axes
about z.  A 45°-skew quad couples x and y.  HELIX implements this by
sandwiching the matrix between two transverse rotations:

$$
M_{\text{skew}} = R(\theta) \cdot M_{\text{normal}} \cdot R(-\theta)
$$

where R is the 6×6 rotation built from the 2-D (x, y) rotation
matrix.

!!! warning "skew_angle vs tilt_deg"
    These are *different* parameters!  `skew_angle` rotates the
    *magnetic* principal axes (a real engineering choice — some
    quads are mounted at 45° on purpose).  `tilt_deg` rotates the
    whole element about z due to *misalignment* — random per-seed
    in error studies.  Both can be non-zero simultaneously.

## API reference (developers)

```{.python .skip}
linac_gen.elements.quadrupole.Quadrupole(
    name: str,
    length: float,                # mm
    gradient: float,              # T/m
    aperture: float = 0.0,        # mm
    skew_angle: float = 0.0,      # deg, rotation about z (engineering)
    g3: float = 0.0,              # T/m²
    g4: float = 0.0,              # T/m³
    g5: float = 0.0,              # T/m⁴
    g6: float = 0.0,              # T/m⁵
    gfr: float = 0.0,             # parsed but unused
    dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    tilt_deg: float = 0.0,
    pitch_deg: float = 0.0, yaw_deg: float = 0.0,
    gradient_rel: float = 0.0,    # fractional gradient error
    n_steps: int = 5,
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `length` | (required) | mm | physical length |
| `gradient` | (required) | T/m | design gradient; positive ⇒ focusing x for proton |
| `aperture` | 0.0 | mm | round-aperture radius |
| `skew_angle` | 0.0 | deg | rotation of principal axes about z |
| `g3..g6` | 0.0 | T/m^(n+1) | higher-pole content |
| `dx..yaw_deg` | 0.0 | mm/deg | misalignment (Misalignment mixin) |
| `gradient_rel` | 0.0 | — | fractional error (FieldError mixin) |
| `n_steps` | 5 | — | tracker substeps |

### Properties

* `effective_gradient` — `gradient * (1 + gradient_rel)`.  This is
  the value the matrix uses; the design `gradient` is preserved.

### Source

`linac_gen/elements/quadrupole.py:1` — ~250 lines including the
multipole-kick machinery.

## See also

* [Multipole](11_multipole.md) — used internally for g3..g6 kicks.
* [Solenoid](03_solenoid.md) — alternative focusing element.
* [Element overview](00_overview.md).
* [Errors → Element-level errors](../08_errors/03_element_errors.md)
  — adding random `gradient_rel` jitter for tolerance studies.

← [Drift](01_drift.md) ·
[Continue to Solenoid →](03_solenoid.md)
