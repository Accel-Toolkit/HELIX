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
* `rho` is **always positive** (bending radius magnitude).  A signed
  `rho` (the Elegant importer keeps ρ = L/θ) is accepted and means the
  same as its magnitude: the direction is read from the angle alone.
* The combination `(angle, rho, hv)` defines the bend completely;
  the geometric length L = ρ·|angle|·π/180.
* The bend direction changes **only the dispersion column**: the
  transfer matrix of the mirror-image magnet is `M(−θ) = S·M(+θ)·S`
  with `S = diag(−1, −1, 1, 1, 1, 1)` — the 4×4 focusing block is the
  same for either direction and `D_x`, `D_x'` flip sign.
* `field_rel` adds a fractional bending-field error.

!!! note "Negative-angle horizontal bends fixed 2026-09-06"
    Until HELIX 1.10.1 the horizontal body matrix used the signed angle
    in its focusing trigonometry, so a negative-angle `BEND` with ρ > 0
    was tracked as the **inverse** sector map with an un-flipped
    dispersion sign; the hyperbolic combined-function branch (n > 1)
    lacked the ×1000 unit factor on the dispersion terms, and n = 1 had
    no dispersion at all.  All three branches are now pinned against
    MAD-X (cpymad) for both bend directions and every field-index
    regime.  Positive-angle and vertical bends are bit-identical to the
    previous release; in the shipped examples only the PIP-II BAL
    off-axis passages (`examples/pipii/bal`) change.

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
P_x                   & P_{x'}                     & 0 & 0 & 1 & M_{56} \\
0                     & 0                          & 0 & 0 & 0 & 1
\end{pmatrix}
$$

where θ and ρ are the **magnitudes** and the dispersion terms carry the
bend direction, `D_x = sign(θ)·ρ(1−cos|θ|)/(β²γm)` (mm/MeV),
`D_x' = sign(θ)·sin|θ|/(β²γm)` (mrad/MeV).

The **path-length row** says that a particle off the design orbit does
not arrive at the same time.  Its extra path through the magnet is
`ΔL = sin(θ)·x + sign(θ)·ρ(1−cos|θ|)·x' + ρ(|θ|−sin|θ|)·δ`, and a longer
path means a later arrival, so with `Δφ = 360·ΔL/(β·λ_RF)`:

$$
P_x = \frac{360\,\sin\theta}{\beta\,\lambda_{\text{RF}}}\;[\text{deg}/\text{mm}], \qquad
P_{x'} = \frac{0.36\,\text{sign}(\theta)\,\rho(1-\cos|\theta|)}{\beta\,\lambda_{\text{RF}}}\;[\text{deg}/\text{mrad}]
$$

$$
M_{56} \;=\; \frac{360}{\beta^{3}\gamma\,m\,\lambda_{\text{RF}}}
\left[\underbrace{\rho(|\theta|-\sin|\theta|)}_{\text{momentum compaction}} - \frac{L}{\gamma^{2}}\right]
\quad [\text{deg}/\text{MeV}]
$$

As everywhere on this page, ρ, L and λ_RF are in **millimetres** and m in
MeV; the 0.36 in `P_x'` rather than 360 is the mrad in its denominator.
(The `M₅₆` velocity term is the `−360·L/(β³γ³mλ)` this element carried
before, unchanged and still grouped that way in the code, so a bend of
vanishing curvature stays bit-identical to a drift.)

The second term of `M₅₆` is the velocity slip a straight drift of the
same arc length also has; the first is the extra distance an off-energy
particle covers by riding the outside of the arc, and it does **not**
depend on which way the magnet bends.

| Block | Effect |
|---|---|
| (x, x') 2×2 | rotation by θ in the bending plane (focuses by 1/ρ on average) |
| (y, y') 2×2 | free drift of length L (no vertical focusing for n=0) |
| (0, 5) and (1, 5) | **dispersion** D_x and D_x' — off-energy particles bend differently |
| (4, 0) and (4, 1) | **path length** P_x and P_x' — an off-axis particle arrives late |
| (4, 5) | momentum compaction plus the drift-like velocity slip |

The dispersion entries couple to ΔW (MeV) directly, via
Δp/p = ΔW / (β²·γ·m·c²).

!!! note "Path length and momentum compaction added 2026-09-07"
    Until HELIX 1.10.1 the (4,0) and (4,1) entries were zero and M₅₆
    carried only the velocity slip, so a bend had no momentum compaction
    and a bunch could not be compressed.  The row is now pinned against
    MAD-X over 192 configurations and against TraceWin's own exported
    matrices for all 36 PIP-II BTL bends.  It is fixed by symplecticity
    rather than free: `P_x` and `P_x'` are the dispersion column over
    again, scaled by `0.36·β·γ·m/λ_RF` with the planes exchanged, which
    is why the whole map now satisfies `MᵀSM = S` in canonical
    coordinates.

    Reading the results: through a dispersive line the **projected**
    longitudinal emittance grows — on `examples/bend_line.dat` by ×93 in
    envelope mode and ×53 with macroparticles — because the beam acquires
    a genuine position–energy correlation.  That is a projection, not
    emittance growth: the map is symplectic and the six-dimensional
    phase-space volume is conserved, which
    `tests/tracking/test_path_length_row.py` checks directly.

With a **field index** n ≠ 0 the bending plane focuses with
`k_x² = (1−n)/ρ²` and the other plane with `k_y² = n/ρ²`; the
dispersion terms become `D_x = sign(θ)(1−cos k_x L)/(ρ k_x²)`,
`D_x' = sign(θ) sin(k_x L)/(ρ k_x)` for n < 1, their hyperbolic
continuation `(1−cosh k L)/(ρ k²)`, `sinh(k L)/(ρ k)` with
`k² = −k_x²` for n > 1, and the parabolic limit `L²/(2ρ)`, `L/ρ` at
n = 1 — all in the same mm/MeV, mrad/MeV units.

For a **vertical** bend (`hv = 1`), the (x ↔ y) planes swap and
the dispersion sign flips on the M[i,5] entries — fixed in the
2026-05-07 dipole bug fix.

For real dipoles, fringe-field effects matter.  The `e1` / `e2`
entrance/exit edge angles on the Dipole add thin-lens edge matrices at
the two faces of the **full-element** matrix — the one matrix mode and
the envelope solver use.  Multi-particle tracking slices every bend
into sub-steps and therefore never sees `e1`/`e2`; use separate
[Edge](05_edge.md) elements, as every `.dat`, MAD-X, MAD8 and Elegant
import does (their pole faces are emitted as `Edge` elements), when a
lattice must give the same optics in every mode.

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
| `e1`, `e2` | 0.0 | deg | entrance/exit pole-face (edge) angles — thin-lens edge matrices on the full-element matrix only (matrix and envelope modes); multi-particle tracking slices the body and ignores them — use `Edge` elements for mode-independent optics |
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
