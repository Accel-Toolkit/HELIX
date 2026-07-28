# Dipole

A dipole bends the beam.  Used in transfer lines (BTL — beam-transport
lines), dump lines, and any place the trajectory must change direction.

## TL;DR (TraceWin users)

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `BEND θ ρ N_steps β_s [aperture hv]` | `Dipole(name, angle, rho, ...)` |
| θ | bending angle (deg) | same |
| ρ | bending radius (mm) | mm |
| `hv` | 0 = horizontal, 1 = vertical | same |

Conventions:

* HELIX `angle` is **signed** in degrees — positive bends toward +x
  (horizontal hv=0) or +y (vertical hv=1).
* `rho` is **always positive** (bending radius magnitude).
* The combination `(angle, rho, hv)` defines the bend completely;
  the geometric length L = ρ·|angle|·π/180.
* `field_rel` adds a fractional bending-field error.

!!! warning "Vertical dipoles fixed 2026-05-07"
    Until recently, `hv=1` (vertical bend) was stored but ignored —
    all bends were treated as horizontal.  Fixed in commit
    `[dipole hv=1]`.  PIP-II BTL `.dat` files with vertical bends
    now produce the correct trajectory.

## Tutorial (newcomers)

A dipole's magnetic field bends the beam through a circular arc.
For a sector dipole with no field index (n = 0), the implemented
6×6 transfer matrix in HELIX's `(x, x', y, y', Δφ, ΔW)` coordinates
is:

$$
M_{\text{bend}} =
\begin{pmatrix}
\cos\theta            & \rho\sin\theta             & 0 & 0 & 0 & D_x  \\
-\sin\theta/\rho      & \cos\theta                 & 0 & 0 & 0 & D_{x'} \\
0                     & 0                          & 1 & L & 0 & 0                   \\
0                     & 0                          & 0 & 1 & 0 & 0                   \\
0                     & 0                          & 0 & 0 & 1 & M_{56} \\
0                     & 0                          & 0 & 0 & 0 & 1
\end{pmatrix}
$$

with the dispersion terms `D_x = ρ(1−cosθ)/(β²γm)` (mm/MeV),
`D_x' = sinθ/(β²γm)` (mrad/MeV) and the same drift-like phase slip
as a free drift of the arc length:

$$
M_{56} \;=\; -\,\frac{360\,L\,[\text{mm}]}{\beta^{3}\,\gamma^{3}\,m\,\lambda_{\text{RF}}} \quad [\text{deg}/\text{MeV}]
$$

| Block | Effect |
|---|---|
| (x, x') 2×2 | rotation by θ in the bending plane (focuses by 1/ρ on average) |
| (y, y') 2×2 | free drift of length L (no vertical focusing for n=0) |
| (0, 5) and (1, 5) | **dispersion** D_x and D_x' — off-energy particles bend differently |
| (4, 0) and (4, 1) | **zero** — no path-length dependence on x, x' is implemented |
| (4, 5) | drift-like phase slip M₅₆ — no momentum-compaction ρ(θ−sinθ) term is implemented |

The dispersion entries couple to ΔW (MeV) directly, via
Δp/p = ΔW / (β²·γ·m·c²).  Note two deliberate simplifications
relative to the textbook sector-bend matrix: the R51/R52
path-length terms and the momentum-compaction contribution to
M[4,5] are **not implemented** — the longitudinal block is exactly
that of a drift of the same arc length.

For a **vertical** bend (`hv = 1`), the (x ↔ y) planes swap and
the dispersion sign flips on the M[i,5] entries — fixed in the
2026-05-07 dipole bug fix.

For real dipoles, fringe-field effects matter.  Set the `e1` / `e2`
entrance/exit edge angles on the Dipole itself — thin-lens
edge-focusing matrices are then applied at the entrance and exit
automatically, so you do **not** need to pair the bend with
separate [Edge](05_edge.md) elements.  Standalone Edge elements
remain available for `.dat` files that declare explicit EDGE cards.

In multi-particle runs a short bunch traversing a dipole also
radiates coherently.  Enable the 1-D steady-state CSR energy kick
with `SpaceChargeConfig(csr_enabled=True)` — see
[Coherent synchrotron radiation](../05_space_charge/01_models.md).

### Example: 30° horizontal bend

```python
from linac_gen.elements.dipole import Dipole

bend = Dipole(name="B1", angle=30.0, rho=1500.0,
              field_index=0.0, aperture=20.0, hv=0)
print(f"Geometric length: {bend.length:.1f} mm")
```

For a vertical bend, set `hv=1`.

## API reference (developers)

```{.python .skip}
linac_gen.elements.dipole.Dipole(
    name: str,
    angle: float,                 # deg, signed
    rho: float,                   # mm
    e1: float = 0.0,              # deg, entrance edge angle
    e2: float = 0.0,              # deg, exit edge angle
    field_index: float = 0.0,     # combined-function field index N
    aperture: float = 0.0,        # mm
    hv: int = 0,                  # 0 = horizontal, 1 = vertical
    dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    tilt_deg: float = 0.0,
    pitch_deg: float = 0.0, yaw_deg: float = 0.0,
    field_rel: float = 0.0,
    n_steps: int = 5,
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | |
| `angle` | (required) | deg | signed bending angle |
| `rho` | (required) | mm | bending radius |
| `e1`, `e2` | 0.0 | deg | entrance/exit pole-face (edge) angles — thin-lens edge-focusing matrices applied at each face; no separate Edge elements needed |
| `field_index` | 0.0 | — | combined-function field index N (0 = pure dipole) |
| `aperture` | 0.0 | mm | round aperture |
| `hv` | 0 | int | 0 = horiz., 1 = vert. |
| `dx..yaw_deg` | 0.0 | mm/deg | misalignment |
| `field_rel` | 0.0 | — | fractional field error (B → B·(1+field_rel), equivalent to scaling the angle) |
| `n_steps` | 5 | — | tracker substeps |

### Properties

* `length` — derived from `angle` and `rho`: L = ρ·|angle|·π/180.
* `effective_angle` — `angle * (1 + field_rel)`; the per-seed
  magnet-strength error folded into the bend angle (arc length is
  fixed by geometry).

### Source

`linac_gen/elements/dipole.py:1`

## See also

* [Edge](05_edge.md) — standalone fringe-field element (only needed
  when the `.dat` declares explicit EDGE cards; prefer `e1`/`e2` on
  the Dipole itself).
* [Coherent synchrotron radiation](../05_space_charge/01_models.md) —
  the CSR energy kick applied inside dipoles in multi-particle runs.
* [Element overview](00_overview.md).

← [Solenoid](03_solenoid.md) ·
[Continue to Edge →](05_edge.md)
