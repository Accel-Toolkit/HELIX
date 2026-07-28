# Drift

A drift is field-free propagation: the beam coasts in a straight line
with no transverse force.  The simplest element in the catalog and
the one that makes up the majority of any real lattice.

## TL;DR (TraceWin users)

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `DRIFT L aperture [aperture_y x_shift y_shift]` | `Drift(name, length, aperture, ...)` |
| Length | mm | mm |
| Aperture | mm (radius for round; `aperture_y` makes it **rectangular**) | same |
| `x_shift`, `y_shift` | beam-frame static offset (mm) | folded into the misalignment `dx`, `dy` |

Conventions:

* HELIX `length` is in **mm** (matches TraceWin).
* The longitudinal `M[4,5]` slip term is included
  (`Δφ[deg] = -360·L·ΔW / (β³·γ³·m·λ_RF)`) — this is what causes
  the bunch to drift in φ when there's an energy spread.
* Drift apertures **are enforced**: the tracker checks apertures
  after every element and inside every drift sub-step bundle.  With
  only `aperture` set the pipe is circular (`x² + y² > aperture²`
  loses the particle); setting `aperture_y > 0` switches to a
  **rectangular** check with half-widths (`aperture`, `aperture_y`).
  Use a dedicated [Aperture](12_aperture.md) element for shaped
  point collimators.

## Tutorial (newcomers)

A drift just propagates the beam.  The full **6×6 transfer matrix**
is

$$
M_{\text{drift}} =
\begin{pmatrix}
1 & L & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & L & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & M_{56} \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

with L in metres and the longitudinal phase-slip element

$$
M_{56} \;=\; -\,\frac{360\,L\,[\text{mm}]}{\beta^{3}\,\gamma^{3}\,m\,\lambda_{\text{RF}}} \quad [\text{deg}/\text{MeV}]
$$

This term causes off-energy particles to slip in phase relative to
the synchronous reference — the longitudinal counterpart of
transverse drift expansion.  HELIX stores lengths in mm and converts
to metres internally before applying the matrix.

A pencil beam with non-zero divergence drifts outward linearly:
position grows like x = x₀ + L·x'₀.  This is why a long beamline
without focusing always blows up — the beam keeps spreading.

| Block | Effect |
|---|---|
| (x, x') 2×2 | identity-with-L: standard transverse drift |
| (y, y') 2×2 | identity-with-L: same in the other plane |
| (Δφ, ΔW) 2×2 | drift-like phase slip via M₅₆ |
| Cross blocks | all zero (drift has no coupling) |

### Example

A 5×100 mm drift chain propagating a low-current 3 MeV proton beam:

```python
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift

ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
beam_cfg = BeamConfig(
    species="proton", energy=3.0, frequency=352.21,
    current=0.0, n_particles=2000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.10, emit_ny=0.10, emit_z=0.20,
    alpha_x=0.0, beta_x=1.0,
    alpha_y=0.0, beta_y=1.0,
    alpha_z=0.0, beta_z=1.0,
)
beam = create_beam(beam_cfg, seed=0)

lat = Lattice()
for i in range(5):
    lat.add(Drift(name=f"D{i}", length=100.0))

sim = Simulation(lat, beam)
res = sim.run()
print(f"σ_x: start = {res.sigma_x[0]:.3f} mm, end = {res.sigma_x[-1]:.3f} mm")
```

The σ_x grows because the matched beta function is short relative to
the drift length — exactly what physics predicts.

## API reference (developers)

```{.python .skip}
linac_gen.elements.drift.Drift(
    name: str,
    length: float,                # mm
    aperture: float = 0.0,        # mm (round radius; informational)
    aperture_y: float | None = None,  # mm; >0 makes the check rectangular
    x_shift: float = 0.0,         # mm; folded into dx
    y_shift: float = 0.0,         # mm; folded into dy
    dx: float = 0.0,              # mm; transverse misalignment
    dy: float = 0.0,
    dz: float = 0.0,              # mm; longitudinal offset
    tilt_deg: float = 0.0,        # rotation about z
    pitch_deg: float = 0.0,       # rotation about x (reserved)
    yaw_deg: float = 0.0,         # rotation about y (reserved)
    n_steps: int = 1,
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | element identifier |
| `length` | (required) | mm | drift length |
| `aperture` | 0.0 | mm | circular pipe radius, **enforced** by the tracker; 0 = no check |
| `aperture_y` | None | mm | when > 0, the check becomes **rectangular** with half-widths (`aperture`, `aperture_y`) |
| `x_shift` | 0.0 | mm | TraceWin DRIFT slot 4; static offset |
| `y_shift` | 0.0 | mm | TraceWin DRIFT slot 5 |
| `dx`, `dy`, `dz` | 0.0 | mm | misalignment offsets (Misalignment mixin); only `dx`/`dy` honoured by the tracker |
| `tilt_deg` | 0.0 | deg | rotation about longitudinal axis |
| `pitch_deg`, `yaw_deg` | 0.0 | deg | stored — not honoured by the tracker |
| `n_steps` | 1 | — | **unused** for transfer-map elements: drift integration density comes from `lattice.step_config` (PARTRAN_STEP); quads/bends/solenoids always track in exactly 2 substeps |

### Methods

* `transfer_matrix(ref, ds=None) → np.ndarray` — 6×6 linear matrix.
  ``ds`` lets you query a partial drift.
* `track(beam, ds=None) → None` — applies the matrix to live
  particles in-place.  Reference particle `s` and `phi_s` advance.

### Source

`linac_gen/elements/drift.py:1` (54 lines).

## See also

* [Quadrupole](02_quadrupole.md) — focusing element.
* [Aperture](12_aperture.md) — for loss collection.
* [Coordinates & units](../02_concepts/01_coordinates.md).

← [Element overview](00_overview.md) ·
[Continue to Quadrupole →](02_quadrupole.md)
