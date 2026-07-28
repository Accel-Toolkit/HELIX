# RFGap

A thin RF gap kick: accelerates the beam, applies adiabatic damping
to the transverse divergences, and provides transverse RF
defocusing (Panofsky-Wenzel).  Used when you want the *effect* of an
accelerating cavity without the cost of a full field-map integration.

## TL;DR (TraceWin users)

| | TraceWin | HELIX |
|---|---|---|
| Keyword | `GAP V φ_s aperture [p_flag]` | `RFGap(name, voltage, phase, frequency, ...)` |
| V | peak voltage (MV) — `e0tl = V·T` | same `voltage` |
| φ_s | synchronous phase (deg) | same `phase` |
| frequency | inferred from current FREQ card | passed explicitly |
| TTF | implicit (folded into V) | explicit `ttf` parameter |

Conventions:

* HELIX `voltage` is the **peak** voltage V₀ — the gap's
  energy-gain-per-particle is q·V₀·T·cos(φ_s) where T is the transit-
  time factor.
* `ttf=1.0` is the default; the TraceWin convention often folds
  T into V₀ giving a smaller "effective" V₀.
* `voltage_rel`, `phase_offset`, `frequency_offset` are the
  FieldError-mixin knobs for cavity LLRF jitter.

## Tutorial (newcomers)

The thin RF gap kick has three components per particle:

1. **Energy kick** — ΔW = q·V₀·T·cos(φ_s + Δφ), where Δφ is the
   particle's phase offset from the synchronous reference.  For
   matched bunches Δφ → 0 and the kick is just q·V₀·T·cos(φ_s).
2. **Adiabatic damping** — divergences x', y' scale by the ratio of
   incoming to outgoing momentum: x'_out = x'_in · p_in / p_out.
   The beam "shrinks" in divergence as it accelerates.
3. **Transverse RF defocusing** — the time-varying longitudinal
   field has a small radial component.  Standard Panofsky-Wenzel:
   the radial kick is Δr' = -π·q·V₀·T·sin(φ_s) / (m·c²·(βγ)³·λ_RF) · r,
   evaluated at the post-acceleration βγ.

The reference particle's W and φ_s update after each gap.

### Linearized 6×6 transfer matrix

A thin RF gap is **non-linear** in (Δφ, ΔW) — the energy gain
folds in cos(φ_s + Δφ).  Linearizing around the synchronous
particle (Δφ → 0), and writing α_p = p_in/p_out for the
adiabatic-damping ratio, the 6×6 transfer matrix is:

$$
M_{\text{RFgap}} =
\begin{pmatrix}
1     & 0       & 0     & 0       & 0    & 0 \\
-k_r  & \alpha_p & 0     & 0       & 0    & 0 \\
0     & 0       & 1     & 0       & 0    & 0 \\
0     & 0       & -k_r  & \alpha_p & 0    & 0 \\
0     & 0       & 0     & 0       & 1    & 0 \\
0     & 0       & 0     & 0       & k_\phi & 1
\end{pmatrix}
$$

with

$$
k_r = \frac{\pi\,q\,V_0\,T\,\sin\phi_s}{m c^2\, (\beta\gamma)^3\, \lambda_{\text{RF}}},
\qquad
k_\phi = -q\,V_0\,T\,\sin\phi_s \cdot \frac{\pi}{180}
\quad [\text{MeV/deg}]
$$

(k_r uses the post-acceleration βγ; the π/180 in k_φ converts the
phase deviation from degrees to radians before the sine
linearization.)

| Block | Effect |
|---|---|
| (x, x') 2×2 | RF transverse defocusing (k_r) + adiabatic damping (α_p < 1) |
| (y, y') 2×2 | identical to (x, x') — axisymmetric thin gap |
| (Δφ, ΔW) 2×2 | longitudinal focusing (k_φ); off-phase particles gain less energy |
| Cross blocks | all zero (no skew, no dispersion in a thin gap) |

For φ_s = 0 (peak crest, no longitudinal focus), k_r = k_φ = 0 and
the gap becomes pure adiabatic damping with no focusing.

For a more accurate cavity model — including off-axis effects,
finite gap length, and asymmetric field profiles — use a
[FieldMap](07_fieldmap.md) or [FieldMap3D](08_fieldmap3d.md), which
integrates the full Lorentz force step-by-step instead of relying
on this linearization.

### Example

A single gap with ϕ_s = -30° (early acceleration phase):

```python
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.drift import Drift
from linac_gen.core.lattice import Lattice

lat = Lattice()
lat.add(Drift(name="D0", length=200.0))
lat.add(RFGap(name="GAP1", voltage=1.0, phase=-30.0, frequency=325.0))  # 1 MV
lat.add(Drift(name="D1", length=200.0))
print(f"Gap energy gain at φ_s: ΔW ≈ V·cos(φ_s) = "
      f"{1.0 * 0.866:.3f} MeV (= 1·cos(30°) ≈ 0.866)")
```

For a more accurate cavity model — including off-axis effects and
asymmetric field profiles — use a [FieldMap](07_fieldmap.md)
or [FieldMap3D](08_fieldmap3d.md).

## API reference (developers)

```{.python .skip}
linac_gen.elements.rf_gap.RFGap(
    name: str,
    voltage: float,               # peak V₀ (MV)
    phase: float,                 # synchronous phase φ_s (deg)
    frequency: float,             # MHz
    ttf: float = 1.0,             # transit-time factor T
    aperture: float = 0.0,        # mm
    p_flag: int = 0,              # TraceWin phase-mode flag
    dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    tilt_deg: float = 0.0,
    pitch_deg: float = 0.0, yaw_deg: float = 0.0,
    voltage_rel: float = 0.0,
    phase_offset: float = 0.0,    # deg, additive
    frequency_offset: float = 0.0,
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `voltage` | (required) | MV | peak voltage; e.g. 1.0 for 1 MV (the parser converts TraceWin's e0tl volts → MV) |
| `phase` | (required) | deg | synchronous phase; -30° is a typical proton-linac value |
| `frequency` | (required) | MHz | local RF frequency |
| `ttf` | 1.0 | — | transit-time factor |
| `voltage_rel` | 0.0 | — | LLRF amplitude error (gaussian σ for tolerance studies) |
| `phase_offset` | 0.0 | deg | LLRF phase error |
| `frequency_offset` | 0.0 | MHz | LLRF frequency error |

### Properties

* `effective_voltage` — `voltage * (1 + voltage_rel)`
* `effective_phase` — `phase + phase_offset`
* `effective_frequency` — `frequency + frequency_offset`

### Source

`linac_gen/elements/rf_gap.py:1`

## See also

* [FieldMap](07_fieldmap.md) — full cavity field model.
* [Errors → Element-level errors](../08_errors/03_element_errors.md) —
  cavity RF-jitter studies.

← [Edge](05_edge.md) ·
[Continue to FieldMap →](07_fieldmap.md)
