# Steerer

A thin steering corrector — a small dipole magnet that nudges the
beam centroid back to the design orbit.  Used in pairs (one
horizontal, one vertical) at every BPM location for closed-orbit
correction.

## TL;DR

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `THIN_STEERING bx_l by_l aperture [elec]` or `STEERER ...` | `Steerer(name, bx_l, by_l, elec=False)` |
| `bx_l` | integrated B_x · L (T·m) — kicks in y | same (elec=1: ∫E_x·dl in **volts** — kicks in x) |
| `by_l` | integrated B_y · L (T·m) — kicks in x | same (elec=1: ∫E_y·dl in **volts** — kicks in y) |
| `elec` | 0 = magnetic (default), 1 = electric | honoured (2026-07): electric kick via the electric rigidity Eρ = βc·Bρ |

The thin kick — magnetic (crossed planes):
Δx' = +sign(q)·by_l / Bρ, Δy' = +sign(q)·bx_l / Bρ; electric
(same-plane, volts): Δx' = +sign(q)·bx_l / (βc·Bρ),
Δy' = +sign(q)·by_l / (βc·Bρ) — in radians.  (The TraceWin manual
writes the magnetic x'-kick with the opposite sign; a corrector's
polarity is a knob-sign convention absorbed by orbit correction.)

## Tutorial

A steerer is a one-shot kick to redirect the beam centroid.  Closed-
orbit correction loops:

1. Read BPM positions (centroid x, y at each BPM).
2. Solve `M·k = -x_BPM` for steerer strengths `k` (M is the
   response matrix from each steerer to each BPM).
3. Apply the steerers; verify the orbit is corrected.

HELIX ships an orbit-correction driver
(`linac_gen.errors.correction.run_correction_from_lattice`) that
walks the lattice for `ADJUST_STEERER` / `ADJUST_STEERER_BX` /
`ADJUST_STEERER_BY` cards, resolves each `diag_n` to the N-th
`DIAG_POSITION` (BPM) marker, and applies the corresponding
one-to-one or SVD correction.  See
[Errors → Orbit correction](../08_errors/07_correction.md) for
the full workflow.

### 6×6 transfer matrix

A steerer is a **thin element with affine kicks** — the linear
6×6 part is the identity, but the kick adds a constant to x' and y':

$$
M_{\text{steerer}} = I_{6\times 6}, \qquad
\begin{pmatrix}\Delta x' \\ \Delta y'\end{pmatrix}_{\text{kick}}
=
\begin{pmatrix} +\operatorname{sign}(q)\,b_{y,L} / B\rho \\ +\operatorname{sign}(q)\,b_{x,L} / B\rho \end{pmatrix}
$$

where `bx_l = ∫B_x dl` (T·m) gives the y kick, and `by_l = ∫B_y dl`
gives the x kick — both with a **plus** sign.  Positive `by_l`
deflects positive charges in the +x direction; the sign flips
automatically for negative species (H⁻).

| Block | Effect |
|---|---|
| (x, x', y, y') 4×4 linear | identity (no focusing) |
| (Δφ, ΔW) 2×2 linear | identity (no longitudinal effect) |
| Affine offset on x', y' | constant kicks scaled by 1/p |

Because the kicks are independent of (x, y), envelope tracking
sees a steerer as the identity matrix — the kicks shift the
**centroid** but leave σ unchanged.

### Example

```python
from linac_gen.elements.steerer import Steerer

st_x = Steerer(name="ST_H1", bx_l=0.0, by_l=1.0e-3)  # 1 mT·m → kick in x
```

## API reference

```{.python .skip}
linac_gen.elements.steerer.Steerer(
    name: str,
    bx_l: float = 0.0,            # T·m, kicks y
    by_l: float = 0.0,            # T·m, kicks x
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `bx_l` | 0.0 | T·m (elec: V) | integrated B_x; deflects in y (Δy' = +sign(q)·bx_l/Bρ) — elec: ∫E_x·dl, deflects in x |
| `by_l` | 0.0 | T·m (elec: V) | integrated B_y; deflects in x (Δx' = +sign(q)·by_l/Bρ) — elec: ∫E_y·dl, deflects in y |
| `elec` | False | — | electric flavour: same-plane kick / (βc·Bρ) |

That is the **entire** constructor — a `Steerer` has no aperture and
neither error-injection mixin.  The STEERER card's aperture field is
parsed and then dropped; `elec` round-trips through the writer.
Note for orbit correction: the `vmax` T·m clip applies to whatever
the knob's units are — volts for an electric steerer.

### Source

`linac_gen/elements/steerer.py:1`

## See also

* [Errors → Orbit correction](../08_errors/07_correction.md) —
  the TraceWin-card-driven correction workflow that uses these
  steerers.
* [Errors → Beam errors](../08_errors/04_beam_errors.md) — input-
  centroid jitter that steerers correct.

← [Marker](13_marker.md) ·
[Continue to Foil →](15_foil.md)
