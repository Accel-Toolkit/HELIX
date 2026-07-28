# Multipole

A thin 2n-pole magnet kick: sextupole, octupole, decapole, dodecapole.
Used either as standalone elements (chromaticity correction in
storage rings) or as the implementation of higher-pole content on
[Quadrupole](02_quadrupole.md) elements (g3/g4/g5/g6 fields from
TraceWin `.dat`).

## TL;DR

```python
from linac_gen.elements.multipole import Multipole, Sextupole, Octupole

# Generic n-pole kick — knl is the integrated strength in SI units
m = Multipole(name="OCT1", knl=[0, 0, 0, 0.05])  # index 3 = k3L (octupole), 0.05 1/m³
# Convenience factory functions:
sx = Sextupole(name="SX1", k2L=0.10)              # 1/m²
oc = Octupole(name="OCT2", k3L=0.05)              # 1/m³
```

| Parameter | Meaning |
|---|---|
| `knl` | list of `[k0L, k1L, k2L, k3L, ...]` — integrated normal-multipole strengths |
| `ksl` | list of skew variants (sin component) |
| `aperture` | round-aperture radius (mm) |
| `dx, dy, tilt_deg` | transverse offset (mm) / tilt (deg) |

The convention follows MAD-X: index `i` of `knl` holds `k_iL` with
units 1/mⁱ — index 0 → dipole (`k0L`), index 1 → quadrupole
(`k1L`, 1/m), index 2 → sextupole (`k2L`, 1/m²), index 3 →
octupole (`k3L`, 1/m³), …

## Tutorial

A multipole's transverse field component is

$$
B_y + i B_x = \sum_{n=0}^{\infty} (B\rho) \left[k_n + i k_{ns}\right] (x + iy)^n / n!
$$

Multipole elements are **thin** in HELIX — zero physical length, the
kick is applied at a single longitudinal point.  For thick
sextupoles you'd typically split into two thin halves with a drift
in between.

### Thin-kick map

A multipole's kick is **non-linear** in (x, y) for any order n ≥ 2,
so it cannot be written as a fixed 6×6 matrix.  The kicks are:

| Order | Element | Δx' | Δy' |
|:---:|---|---|---|
| 0 | Dipole | -k0l | 0 |
| 1 | Quadrupole | -k1l · x | k1l · y |
| 2 | Sextupole | -k2l · (x² - y²) / 2 | k2l · x · y |
| 3 | Octupole | -k3l · (x³ - 3xy²) / 6 | k3l · (3x²y - y³) / 6 |
| 4 | Decapole | -k4l · (x⁴ - 6x²y² + y⁴) / 24 | k4l · (4x³y - 4xy³) / 24 |

The `(x, y)` and `(x', y')` updates are applied as a single thin
kick; `(Δφ, ΔW)` are unchanged (no longitudinal effect).

### Linearized 6×6 (for completeness)

For matching and envelope tracking, only the linear (k1) component
contributes; higher orders are zero in the linearization around
the closed orbit.  The thin-quadrupole limit is:

$$
M_{\text{multipole},\,k_1} =
\begin{pmatrix}
1     & 0 & 0    & 0 & 0 & 0 \\
-k_1L & 1 & 0    & 0 & 0 & 0 \\
0     & 0 & 1    & 0 & 0 & 0 \\
0     & 0 & k_1L & 1 & 0 & 0 \\
0     & 0 & 0    & 0 & 1 & 0 \\
0     & 0 & 0    & 0 & 0 & 1
\end{pmatrix}
$$

For sextupoles and higher, MP tracking captures the nonlinear
kick exactly while the envelope solver sees only the linearization.

### When to use a Multipole

* **Chromaticity correction** in transfer lines: place at points of
  high dispersion to compensate momentum-dependent focusing errors.
* **Aberration correction** in beam transport.
* **As Quadrupole g3/g4/g5/g6 implementation** — the `Quadrupole`
  element uses thin Multipoles internally for higher-pole content
  (see [Quadrupole](02_quadrupole.md)).
* **IMPACT-X parity** for lattices imported from there — IMPACT-X
  has explicit Sextupole, Octupole etc. classes.

## API reference

```{.python .skip}
linac_gen.elements.multipole.Multipole(
    name: str,
    knl: list[float] | None = None,  # integrated normal strengths (SI)
    ksl: list[float] | None = None,  # integrated skew strengths
    aperture: float = 0.0,           # mm
    dx: float = 0.0, dy: float = 0.0, tilt_deg: float = 0.0,
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `knl` | None (→ []) | 1/mⁱ at index i | `[k0L, k1L, k2L, k3L, ...]` integrated normal strengths |
| `ksl` | None (→ []) | 1/mⁱ at index i | skew counterparts (rotated by π/2n) |
| `aperture` | 0.0 | mm | round-aperture radius; 0 = no check |
| `dx`, `dy` | 0.0 | mm | transverse offsets — plain attributes (no Misalignment mixin); the kick is evaluated at `(x − dx, y − dy)`, producing feed-down |
| `tilt_deg` | 0.0 | deg | rotation about z, applied around the kick |

Convenience **factory functions** (they return a configured
`Multipole`, not subclass instances):

* `Sextupole(name, k2L, skew=False, aperture=0.0, dx=0.0, dy=0.0,
  tilt_deg=0.0)` — puts `k2L` (1/m²) in the normal (or skew) slot
* `Octupole(name, k3L, skew=False, aperture=0.0, dx=0.0, dy=0.0,
  tilt_deg=0.0)` — puts `k3L` (1/m³) in the normal (or skew) slot

### Source

`linac_gen/elements/multipole.py:39` (constructor), `:138-171`
(`Sextupole` / `Octupole` factory functions).

## See also

* [Quadrupole](02_quadrupole.md) — uses Multipole internally for g3..g6.
* [Eigenemittance worked example](../11_examples/09_eigenemittance.md) —
  exercises Sextupole/Octupole.

← [SuperposedFieldMap](19_superposedfieldmap.md) ·
[Continue to Aperture →](12_aperture.md)
