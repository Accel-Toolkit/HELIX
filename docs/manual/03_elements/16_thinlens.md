# ThinLens

An ideal thin lens: a zero-length paraxial kick with independent
horizontal and vertical focal lengths.  Used for quick optics
sketches, idealised matching sections, and as a stand-in for any
focusing device you don't want to model in detail.

## TL;DR

```python
from linac_gen.elements.thin_lens import ThinLens

lens = ThinLens(name="L1", fx=2.0, fy=-2.0)   # focal lengths in metres
```

The kick is the textbook paraxial thin-lens map:

```
Δx'[mrad] = -x[mm] / fx[m]
Δy'[mrad] = -y[mm] / fy[m]
```

Positive `f` converges, negative `f` diverges, and the default
`float('inf')` means "no focusing in that plane".

## Tutorial

A `ThinLens` is a [ThinKickElement](00_overview.md): zero length, one
kick at a single s-position.  Common uses:

* **Back-of-envelope optics** — build a doublet or telescope from
  two thin lenses and drifts before committing to real quadrupole
  gradients.
* **Idealised matching** — insert a thin lens where a future magnet
  will sit and let the matcher solve for `fx` / `fy`.
* **Einzel-lens / solenoid stand-in** — set `fx = fy` for an
  axisymmetric electrostatic or solenoidal lens whose detailed field
  you don't have yet.

### Transfer matrix

The linearised 6×6 matrix is identity except for the two focusing
entries:

$$
M_{\text{lens}} =
\begin{pmatrix}
1      & 0 & 0      & 0 & 0 & 0 \\
-1/f_x & 1 & 0      & 0 & 0 & 0 \\
0      & 0 & 1      & 0 & 0 & 0 \\
0      & 0 & -1/f_y & 1 & 0 & 0 \\
0      & 0 & 0      & 0 & 1 & 0 \\
0      & 0 & 0      & 0 & 0 & 1
\end{pmatrix}
$$

An infinite focal length zeroes the corresponding entry.

## API reference

```{.python .skip}
linac_gen.elements.thin_lens.ThinLens(
    name: str,
    fx: float = float('inf'),
    fy: float = float('inf'),
    aperture: float = 0.0,
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `fx` | `float('inf')` | m | horizontal focal length; `inf` = no horizontal focusing |
| `fy` | `float('inf')` | m | vertical focal length; `inf` = no vertical focusing |
| `aperture` | 0.0 | mm | round-aperture radius; 0 = no check |

### Source

`linac_gen/elements/thin_lens.py:31`

← [Foil](15_foil.md) ·
[Continue to SpaceChargeComp →](17_spacechargecomp.md)
