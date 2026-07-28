# Edge

A dipole edge models the fringe-field kick at the entrance or exit of
a real dipole magnet.  Two edges flank a `Dipole` to give a complete
sector- or rectangular-bend model.

## TL;DR (TraceWin users)

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `EDGE pole_rotation rho gap k1 k2 [aperture hv]` | `Edge(name, pole_rotation, rho, gap, ...)` |
| pole_rotation | edge angle in degrees | same |
| rho | bending radius (mm) | same |
| gap | full gap height (mm) | same |
| k1, k2 | fringe coefficients | same |

Conventions:

* `pole_rotation` is the angle between the magnet face and the local
  reference trajectory.  0° = sector face (perpendicular to
  trajectory).  Non-zero ⇒ rectangular or wedge magnet.
* `k1`, `k2` are TraceWin's fringe-field coefficients (defaults
  0.45 and 2.80).  **Only `k1` enters the matrix** — `k2` is parsed
  and stored for round-trip fidelity but not applied.
* The edge applies a thin angular kick: x' ↛ x' + (tan β / ρ)·x in
  the bending plane.

## Tutorial (newcomers)

A real dipole has an extended fringe field beyond its geometric
length.  Two effects matter:

1. **Geometric edge focusing** — the slanted face of a
   rectangular bend acts like a thin lens with focal length
   `f = ρ / tan(β)` where β is the pole rotation angle.  This is
   the dominant fringe contribution for a wedge magnet.
2. **Quadrupole-component fringe** — the field rolloff at the
   edge isn't sharp; it falls off over a length proportional to
   the gap.  HELIX captures this with the first-order `k1` fringe
   correction (the second-order `k2` is stored but not applied).

The edge element implements both as a thin matrix kick.

### Full 6×6 transfer matrix

For an edge with pole rotation β and fringe correction ψ, the
thin-edge transfer matrix in `(x, x', y, y', Δφ, ΔW)` is:

$$
M_{\text{edge}} =
\begin{pmatrix}
1            & 0 & 0            & 0 & 0 & 0 \\
\tan\beta/\rho & 1 & 0            & 0 & 0 & 0 \\
0            & 0 & 1            & 0 & 0 & 0 \\
0            & 0 & -\tan(\beta-\psi)/\rho & 1 & 0 & 0 \\
0            & 0 & 0            & 0 & 1 & 0 \\
0            & 0 & 0            & 0 & 0 & 1
\end{pmatrix}
$$

with the fringe correction (dimensionless)

$$
\psi = \frac{k_1\,g}{\rho} \, \frac{1+\sin^2\beta}{\cos\beta}
$$

where `g` is the full magnetic gap.  ψ is zero when `gap = 0` or
`k1 = 0`.  Note that **`k2` does not appear** — the implemented
matrix uses the first-order ψ only; `k2` is stored for `.dat`
round-tripping but never applied.

| Block | Effect |
|---|---|
| (x, x') 2×2 | thin-lens focusing in bending plane: f_x = ρ / tan(β) |
| (y, y') 2×2 | thin-lens defocusing in non-bending plane: f_y = -ρ / tan(β − ψ) |
| (Δφ, ΔW) 2×2 | identity (zero-length element does no longitudinal work) |
| Cross blocks | all zero |

For a **vertical** bend (`hv = 1`), x and y swap throughout.

A **sector bend** has β = 0 → tan(β) = 0 → no edge focusing →
the Edge element collapses to the identity and can be omitted.

### Example

```python
from linac_gen.elements.edge import Edge
from linac_gen.elements.dipole import Dipole

# A horizontal 30° rectangular bend with 15° symmetric edges
edge_in  = Edge(name="E1_in",  pole_rotation=15.0, rho=1500.0,
                gap=20.0, k1=0.45, hv=0)
bend     = Dipole(name="B1", angle=30.0, rho=1500.0, hv=0)
edge_out = Edge(name="E1_out", pole_rotation=15.0, rho=1500.0,
                gap=20.0, k1=0.45, hv=0)
```

A sector bend (no edge focusing) skips the Edge elements entirely.

## API reference (developers)

```{.python .skip}
linac_gen.elements.edge.Edge(
    name: str,
    pole_rotation: float,         # deg, β
    rho: float,                   # mm
    gap: float = 0.0,             # mm, full pole gap
    k1: float = 0.45,             # fringe coefficient (applied)
    k2: float = 2.80,             # fringe coefficient (stored, NOT applied)
    aperture: float = 0.0,        # stored but not used
    hv: int = 0,                  # 0 = horiz., 1 = vert.
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `pole_rotation` | (required) | deg | edge angle β between face and trajectory |
| `rho` | (required) | mm | bending radius (matches the partner Dipole) |
| `gap` | 0.0 | mm | full magnet gap height; 0 disables the fringe correction |
| `k1` | 0.45 | — | first-order fringe coefficient (TraceWin default) |
| `k2` | 2.80 | — | second-order fringe coefficient — parsed/stored, **not applied** by the matrix |
| `aperture` | 0.0 | mm | stored but not used at this stage |
| `hv` | 0 | int | 0 = horizontal bend, 1 = vertical |

`Edge` is a **PassiveElement** — it has no misalignment or
field-error mixin parameters.

### Source

`linac_gen/elements/edge.py:1`

## See also

* [Dipole](04_dipole.md) — needs paired Edges for realistic fringe.
* [Element overview](00_overview.md).

← [Dipole](04_dipole.md) ·
[Continue to RFGap →](06_rfgap.md)
