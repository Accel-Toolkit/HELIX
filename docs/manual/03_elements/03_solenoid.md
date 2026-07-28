# Solenoid

A solenoid focuses the beam in *both* transverse planes simultaneously
via the on-axis B_z field plus the rotational coupling at entrance and
exit.  The standard focusing element of a low-energy LEBT (where
quadrupoles are too weak) and a common companion to RF cavities in
SRF cryomodules.

## TL;DR (TraceWin users)

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `SOLENOID L B aperture` | `Solenoid(name, length, field, ...)` |
| Length L | mm | mm |
| Field B | T (on-axis B_z) | T |

Conventions:

* **Always coupled**: x and y are linked through Larmor rotation —
  σ_x and σ_y *individually* oscillate even if the beam is round.
  Use [eigenemittances](../09_diagnostics/02_emittances.md) for the
  invariants.
* `field_rel` adds a fractional field error
  (`effective_B = B·(1 + field_rel)`).

## Tutorial (newcomers)

A solenoid produces an axial magnetic field B_z along the beam axis.
The transverse 6-D dynamics combine three effects:

1. **Entrance kick** — radial fringe field gives x' a kick
   proportional to y, and vice versa.
2. **Body** — particles spiral around the axis at the Larmor
   frequency ω_L = q·B / (2m) (in the rest frame).  Effective
   focusing strength k_s = q·B / (2·Bρ).
3. **Exit kick** — opposite of entrance kick.

The net effect is focusing in both planes — the axisymmetric
counterpart to a quadrupole pair.  Unlike a quadrupole, the
four transverse coordinates `(x, x', y, y')` are **fully coupled**:
no purely-x or purely-y subspace is invariant.

### Full 6×6 transfer matrix

Define the Larmor wavenumber

$$
k_s \;=\; \frac{q\,B_z}{2\,B\rho}
\quad \text{(rad/m)}
$$

and let C = cos(k_s L), S = sin(k_s L).  The full 6×6 in
coordinates `(x, x', y, y', Δφ, ΔW)` is:

$$
M_{\text{sol}} =
\begin{pmatrix}
C^2      & SC/k_s   & SC       & S^2/k_s & 0 & 0      \\
-k_s SC  & C^2      & -k_s S^2 & SC      & 0 & 0      \\
-SC      & -S^2/k_s & C^2      & SC/k_s  & 0 & 0      \\
k_s S^2  & -SC      & -k_s SC  & C^2     & 0 & 0      \\
0        & 0        & 0        & 0       & 1 & M_{56} \\
0        & 0        & 0        & 0       & 0 & 1
\end{pmatrix}
$$

with the same drift-like longitudinal phase slip as every other
thick element:

$$
M_{56} = -\,\frac{360 \, L \, [\text{mm}]}{\beta^{3} \, \gamma^{3} \, m \, \lambda_{\text{RF}}}
\quad [\text{deg}/\text{MeV}]
$$

| Block | Effect |
|---|---|
| (x, x', y, y') 4×4 | fully coupled focusing — Larmor rotation + radial focus |
| (x, y), (x, y'), (x', y), (x', y') | non-zero (this is what couples the planes) |
| (Δφ, ΔW) 2×2 | drift-like phase slip identical to any thick element |
| Cross to longitudinal | zero (no acceleration; on-axis B_z does no work) |

The `C²` diagonal terms are why σ_x and σ_y individually appear to
"breathe" through the solenoid even for a round, matched beam: the
projected emittances ε_x and ε_y are no longer invariants of the
motion — only the 4-D ε_⊥ and the symplectic eigenemittances ε_1,
ε_2 are.

### Key consequence: ε_x and ε_y "wobble"

Because the (x, x') and (y, y') subspaces are coupled, the projected
emittances ε_x and ε_y *oscillate* through the solenoid even for an
ideal beam — the *true* invariants are the 4-D ε_⊥ = √det(Σ_4D) and
the symplectic normal-mode emittances ε_1, ε_2.  See
[Emittances](../09_diagnostics/02_emittances.md).

### Example: PXIE-style LEBT solenoid

```python
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.drift import Drift
from linac_gen.core.lattice import Lattice

lat = Lattice()
lat.add(Drift(name="D0", length=200.0))
lat.add(Solenoid(name="SOL1", length=300.0, field=0.20))   # 0.20 T
lat.add(Drift(name="D1", length=400.0))
lat.add(Solenoid(name="SOL2", length=300.0, field=0.20))
lat.add(Drift(name="D2", length=200.0))

print(f"Lattice has {len(lat.elements)} elements, "
      f"{lat.total_length:.0f} mm long.")
```

For a complete worked LEBT example see
[DC LEBT walkthrough](../11_examples/03_lebt_dc.md).

## API reference (developers)

```{.python .skip}
linac_gen.elements.solenoid.Solenoid(
    name: str,
    length: float,                # mm
    field: float,                 # T (on-axis B_z)
    aperture: float = 0.0,        # mm
    dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    tilt_deg: float = 0.0,
    pitch_deg: float = 0.0, yaw_deg: float = 0.0,
    field_rel: float = 0.0,
    n_steps: int = 5,
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `length` | (required) | mm | physical length |
| `field` | (required) | T | on-axis B_z (design value) |
| `aperture` | 0.0 | mm | round-aperture radius |
| `dx..yaw_deg` | 0.0 | mm/deg | misalignment |
| `field_rel` | 0.0 | — | fractional field error (FieldError mixin) |
| `n_steps` | 5 | — | tracker substeps |

### Properties

* `effective_field` — `field * (1 + field_rel)`.

### Source

`linac_gen/elements/solenoid.py:1` (~120 lines).

## See also

* [Quadrupole](02_quadrupole.md) — alternative focusing.
* [FieldMap](07_fieldmap.md) — for solenoids defined by a measured
  field-map (when the hard-edge model is not accurate enough).
* [Emittances](../09_diagnostics/02_emittances.md) — eigenemittance
  diagnostic to handle solenoid-induced x-y coupling.

← [Quadrupole](02_quadrupole.md) ·
[Continue to Dipole →](04_dipole.md)
