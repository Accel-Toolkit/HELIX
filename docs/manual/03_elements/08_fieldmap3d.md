# FieldMap3D

A FieldMap3D integrates particles through a fully 3-D Cartesian field
map: F_x(x,y,z), F_y(x,y,z), F_z(x,y,z) on a regular grid.  Used when
the cavity / magnet has non-axisymmetric structure (asymmetric coupler,
tilted field, real measured 3-D mapping) and the 1-D / 2-D
cylindrical reduction is not adequate.

## TL;DR

Same TraceWin `FIELD_MAP` card as the 1-D / 2-D variant, with
`geom` digit 7 selecting 3-D Cartesian for one or both of E and B
channels:

* `geom = 7700` — 3-D RF E + 3-D RF B (full 3-D cavity).
* `geom = 7000` — 3-D RF B only.
* `geom = 70` — 3-D static B (e.g. measured solenoid 3-D map).

The expected files for `geom = 7700` and prefix `MyCavity`:

```
MyCavity.edx, MyCavity.edy, MyCavity.edz   # 3-D RF E components
MyCavity.bdx, MyCavity.bdy, MyCavity.bdz   # 3-D RF B components
```

## Tutorial

The integrator is selectable via the **class attribute**
`FieldMap3D.integrator_kind`: `"kd"` (default — the legacy
first-order kick-then-drift scheme) or `"dkd"` (second-order
Drift–Kick–Drift, a.k.a. velocity Verlet).  **Only DKD is
symplectic** — it samples position-dependent fields at the
half-drifted location, removing the O(ds²) phase-advance asymmetry
that drives long-trajectory emittance growth in strongly focusing
solenoid maps.  The field at each substep is sampled from the 3-D
table by trilinear interpolation (`interp_kind`, also a class
attribute: `"linear"` or `"cubic"`).  ~3-5× slower per step than
the 2-D cylindrical case, but exact for non-axisymmetric fields.

### When to prefer FieldMap3D

* Cavity has a **fundamental power coupler** that breaks
  axisymmetry (off-axis kick).
* You have a **measured** 3-D field map and don't want to reduce it
  to cylindrical components.
* You need to model **asymmetric solenoid bucking coils** (e.g.
  PXIE LEBT solenoid + bucker).

For axisymmetric cavities the 2-D cylindrical (`geom = 1XX0` or
`5XX0`) is usually preferable — same physics, less storage, faster.

### Performance

A 256-m PIP-II run with mostly 3-D field maps (HWR + SSR1 + SSR2
+ LB650 + HB650) takes about 6 minutes per envelope pass.  Replacing
the LB650/HB650 sections with 1-D models drops it to ~3 minutes
with negligible σ change for the design beam.

### Transfer matrix

Same story as 1-D / 2-D [FieldMap](07_fieldmap.md#transfer-matrix-and-per-step-lorentz-integration):
no closed-form 6×6.  HELIX integrates the Lorentz force per substep
and chains finite-differenced 6×6 Jacobians for the
envelope/matching paths.

The 3-D field source means the **cross blocks** of the transfer
matrix can be non-zero — a tilted solenoid couples (x, y'), an
asymmetric coupler couples (x, Δφ), etc.  This is the main reason
to use `FieldMap3D` over its axisymmetric cousin.

## API reference

```{.python .skip}
linac_gen.elements.field_map_3d.FieldMap3D(
    name: str,
    length: float,                # mm
    field_data: FieldMapData,     # pre-loaded 3-D Cartesian table(s)
    scale: float = 1.0,
    phase: float = 0.0,           # deg
    frequency: float = 0.0,       # MHz
    aperture: float = 0.0,        # mm
    n_steps: int = 100,
    ke: float = 1.0, kb: float = 1.0,
    ki: float = 0.0,
    p_flag: int = 0,
    dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    tilt_deg: float = 0.0,
    pitch_deg: float = 0.0, yaw_deg: float = 0.0,
    voltage_rel: float = 0.0,
    phase_offset: float = 0.0,
    frequency_offset: float = 0.0,
    ka: int = 1,
    field_file: str | None = None,  # provenance (writer round-trip)
    geom: int | None = None,        # provenance (writer round-trip)
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `length` | (required) | mm | physical length |
| `field_data` | (required) | — | `FieldMapData` with 3-D Cartesian channels (geom digit 7) |
| `scale` | 1.0 | — | global field amplitude multiplier |
| `phase` | 0.0 | deg | RF phase φ_s |
| `frequency` | 0.0 | MHz | RF frequency |
| `aperture` | 0.0 | mm | round-aperture radius |
| `n_steps` | 100 | — | substeps |
| `ke`, `kb` | 1.0 | — | E / B amplitude scales |
| `ki` | 0.0 | — | SC compensation intensity (needs `.scc` + active SC solver) |
| `p_flag` | 0 | int | 1 = SET_SYNC_PHASE iterative phase calibration |
| `dx..yaw_deg` | 0.0 | mm/deg | misalignment (Misalignment mixin) |
| `voltage_rel`, `phase_offset`, `frequency_offset` | 0.0 | —/deg/MHz | LLRF error knobs (FieldError mixin) |
| `ka` | 1 | int | TraceWin aperture flag — retained for round-trip, ignored by the 3-D tracker |
| `field_file`, `geom` | None | — | provenance set by the parser |

Two **class attributes** (not constructor parameters) control the
numerics; flip them globally, or set them on an instance:

| Attribute | Default | Values | Notes |
|---|---|---|---|
| `integrator_kind` | `"kd"` | `"kd"` \| `"dkd"` | KD = legacy first-order kick-then-drift; DKD = second-order Drift–Kick–Drift — the **only symplectic** option |
| `interp_kind` | `"linear"` | `"linear"` \| `"cubic"` | grid-interpolation order |

### Sampling implementation (fused C++ kernel)

With `interp_kind = "linear"` the per-substep field sampling runs
through a fused C++ kernel (`linac_gen._fieldmap_kernels`) when the
compiled module is present: one pass computes the cell indices and
weights per particle and applies them to every component of a
channel, instead of one scipy `RegularGridInterpolator` call per
component.  The two paths are **bitwise identical** — pinned by
`tests/elements/test_fieldmap_kernels.py` — so this is a speed
switch, never a physics choice.  Measured on the complete PIP-II
linac (256.5 m, 164 field maps): MP+SC 1.84× and envelope+SC 2.93×
end-to-end.

Selection (default is the kernel whenever it is built):

* **GUI** — Numerics tab → Field map → *Sampling* (`kernel` /
  `scipy`); the choice is saved in the project file.
* **CLI** — `python -m linac_gen run … --fieldmap-sampling scipy`
  (or `fieldmap_sampling` in the project's `convergence` block).
* **API** — `linac_gen.elements.field_map_3d.use_fused_kernel(False)`
  (query with `fused_kernel_enabled()` / `kernel_available()`).
* **Env var** — `LINAC_GEN_FIELDMAP_KERNEL=0` disables it at
  process start.

If the compiled module is absent the scipy path is used silently
(identical results, slower).  Build instructions:
`linac_gen/csrc/README.md`.  Cubic interpolation always uses scipy.

### Source

`linac_gen/elements/field_map_3d.py:88` (constructor), `:366-389`
(integrator selection).

## See also

* [FieldMap](07_fieldmap.md) — 1-D / 2-D variants.
* [Convergence guide](../05_space_charge/05_convergence.md) — choosing
  step count, integrator order.

← [FieldMap](07_fieldmap.md) ·
[Continue to RfqCell →](09_rfqcell.md)
