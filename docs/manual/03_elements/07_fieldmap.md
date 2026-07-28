# FieldMap (1-D / 2-D)

A `FieldMap` element integrates particles through a tabulated field
(static B/E or RF B/E) read from external files.  Used for
realistic cavity and solenoid models that go beyond the hard-edge
analytical matrices.  HELIX implements the *full* TraceWin FIELD_MAP
spec including the 5-digit `geom` encoding, every channel
(static/RF × electric/magnetic), file-naming convention, and the
`FIELD_MAP_PATH` directive.

## TL;DR (TraceWin users)

```
FIELD_MAP  geom  L  θᵢ  R  kb  ke  Ki  Ka  FileName  [p_flag]
```

| Param | Meaning | HELIX kw |
|---|---|---|
| `geom` | 5-digit channel encoding (see below) | `geom` (int) |
| `L` | element length (mm) | `length` |
| `θᵢ` | RF phase offset (deg) | `phase` |
| `R` | aperture radius (mm) | `aperture` |
| `kb` | B-field amplitude scale | `kb` |
| `ke` | E-field amplitude scale | `ke` |
| `Ki` | space-charge compensation scale | `ki` |
| `Ka` | aperture flag (0/1/2) | `ka` |
| `FileName` | field-data file prefix | resolved by the parser; stored as `field_file` |
| `p_flag` | 0 = relative phase, 1 = SET_SYNC_PHASE calibration | `p_flag` |

HELIX honours every parameter exactly as TraceWin specifies them.
Two that deserve a closer look:

* **Ki** only *rescales space-charge kicks*: when `Ki > 0` **and** a
  `FileName.scc` compensation table is present, the tracker applies
  a local SC fraction `1 − Ki·scc(z)` inside the map.  It therefore
  needs an **active space-charge solver** to have any effect — with
  no SC solver (or no `.scc` file) Ki is a no-op.
* **p_flag = 1** triggers the SET_SYNC_PHASE iterative phase
  calibration (a scan-fit for the entrance phase offset that makes
  `phase` behave as the synchronous phase).  It is *not* an
  "absolute phase" switch; the parser forces `p_flag=1` on maps
  under an active `SET_SYNC_PHASE` directive.

## Tutorial

### The `geom` 5-digit encoding

```
geom = aper·10⁴ + rf_B·10³ + rf_E·10² + stat_B·10 + stat_E
```

Each digit selects the geometry of one of five channels:

| Digit | Value | Channel | Common geometry |
|:---:|:---:|---|---|
| 10⁰ | E_static | static electric | 0 (absent), 1 (on-axis), 4 (2-D Er/Ez) |
| 10¹ | B_static | static magnetic | 0, 1 (on-axis Bz), 5 (2-D Br/Bz), 9 (quad gradient) |
| 10² | E_RF | RF electric | 0, 1 (on-axis Ez(z)), 4 (2-D Er, Ez) |
| 10³ | B_RF | RF magnetic | 0, 5 (2-D Br, Bz) |
| 10⁴ | aperture | aperture profile from `.ouv` file | 0 or 1 |

**Examples:**

* `geom = 100` — RF electric on-axis only.  Common for SRF cavities
  modeled by Ez(z).
* `geom = 7700` — full 3-D RF, both E and B from 3-D Cartesian
  files.  See [FieldMap3D](08_fieldmap3d.md).
* `geom = 50` — 2-D static B-type (solenoid Br, Bz).
* `geom = 90` — 1-D static B as quad gradient G(z) — used to model
  PMQs (permanent magnet quadrupoles).

### File naming

For a `FileName` of `MyCavity` and `geom = 100` (RF electric on-axis),
HELIX expects:

```
MyCavity.edz       # 1-D E_z(z) RF electric
```

Suffixes follow the three-letter TraceWin scheme
`.{e|b}{s|d}{x|y|z|r|q}` — field letter (`e` electric / `b`
magnetic), type letter (`s` static / `d` dynamic = RF), then the
component (`x`, `y`, `z`, `r`, or `q` = θ):

| Channel | Type letter | 1-D on-axis suffix |
|---|:---:|---|
| static E | `s` | `.esz` |
| static B | `s` | `.bsz` (also digit 9, quad gradient G(z)) |
| RF E | `d` | `.edz` |
| RF B | `d` | `.bdz` |

For a 2-D cylindrical RF E (geom digit 4), expect `MyCavity.edr`
and `MyCavity.edz` (plus `MyCavity.bdq` for the TM-mode Bθ).  A 2-D
cylindrical B channel (digit 5) uses `.b{s,d}r` / `.b{s,d}z` (plus
`.edq` for the TE-mode Eθ when RF).

The aperture file (when geom 10⁴ digit is 1) is `MyCavity.ouv`.

### `FIELD_MAP_PATH` directive

Field-map files often live in a separate directory.  The TraceWin
`FIELD_MAP_PATH` directive sets the search path:

```
FIELD_MAP_PATH ../Fields
FIELD_MAP 100  300  -30  0  1.0  1.0  0  1  HWR_cavity
```

HELIX honours this directive — paths are resolved relative to the
`.dat` file's directory, then the `FIELD_MAP_PATH` (most recent
wins).

### Tracking through a FieldMap

The 1-D / 2-D `FieldMap` element integrates with a single
**first-order midpoint kick-drift** scheme: at each slice, all
channels are sampled at the slice midpoint, the longitudinal energy
kick and the transverse Lorentz kick are applied, then the particles
drift to the slice end.  There is **no integrator selector** on this
element — the KD / DKD choice exists only on
[FieldMap3D](08_fieldmap3d.md).  The step density is set by
`n_steps` or by the lattice's `PARTRAN_STEP` global (whichever is
finer).

The reference particle's W and φ_s are updated at every substep,
so the FieldMap correctly accelerates the design beam.

### Transfer matrix and per-step Lorentz integration

Unlike Drift, Quadrupole, and Solenoid, a FieldMap has **no closed-
form 6×6 transfer matrix** — its dynamics are nonlinear in
(x, y, z) because the field varies in space and (for RF) time.

The element instead **integrates** the equations of motion at each
substep i ∈ [1, n_steps]:

$$
\begin{aligned}
\frac{d\mathbf{p}}{dt} &= q\,(\mathbf{E}(\mathbf{r}, t) + \mathbf{v} \times \mathbf{B}(\mathbf{r}, t)) \\
\frac{d\mathbf{r}}{dt} &= \frac{\mathbf{p}}{\gamma m}
\end{aligned}
$$

For envelope-mode tracking and matching, HELIX builds a **local
6×6 Jacobian** at each substep by finite-differencing the
integrator (the auto-diff variant is reserved for v2):

$$
M^{(i)} = \frac{\partial \, \mathbf{X}_{i+1}}{\partial \, \mathbf{X}_i}
\quad\text{(6×6, evaluated at the synchronous particle)}
$$

The substep matrices are chained — `M_total = ∏ M^{(i)}` — to give
the element-level transfer matrix exposed via
`FieldMap.transfer_matrix(ref)`.  This is what the
[matching](../07_matching/01_overview.md) and
[envelope](../02_concepts/03_tracking_modes.md) workflows consume.

| Block | Effect |
|---|---|
| (x, x', y, y') 4×4 | depends on field shape: Bz solenoid → coupled focus; on-axis Ez → adiabatic damping + RF defocus |
| (Δφ, ΔW) 2×2 | acceleration + longitudinal RF focusing |
| Cross blocks | non-zero whenever the field has off-axis structure (3-D maps, tilted solenoid, etc.) |

### Example: a PIP-II HWR cavity

The parser builds `FieldMap` objects from FIELD_MAP cards
automatically; to construct one programmatically, use the
`FieldMap.from_file` classmethod, which loads the field data for
you:

```{.python .skip}
from linac_gen.elements.field_map import FieldMap

cav = FieldMap.from_file(
    name="HWR1",
    filepath="Fields/my_cavity.edz",    # 1-D E_z(z) RF electric — point
    length=300.0,                       #   this at YOUR field-map file
    phase=-30.0,                        # synchronous phase φ_s
    frequency=162.5,                    # MHz
    aperture=20.0,                      # mm
    p_flag=0,                           # relative phase
)
```

## API reference

```{.python .skip}
linac_gen.elements.field_map.FieldMap(
    name: str,
    length: float,                # mm
    field_data: FieldMapData,     # pre-loaded field table(s)
    scale: float = 1.0,           # global amplitude multiplier
    kb: float = 1.0,              # B-channel amplitude factor
    ke: float = 1.0,              # E-channel amplitude factor
    ki: float = 0.0,              # SC compensation intensity
    ka: int = 1,                  # aperture flag
    phase: float = 0.0,           # deg
    frequency: float = 0.0,       # MHz
    aperture: float = 0.0,        # mm
    n_steps: int = 100,
    p_flag: int = 0,              # 0 = relative, 1 = SET_SYNC_PHASE calibration
    dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
    tilt_deg: float = 0.0,
    pitch_deg: float = 0.0, yaw_deg: float = 0.0,
    voltage_rel: float = 0.0,
    phase_offset: float = 0.0,
    frequency_offset: float = 0.0,
    field_file: str | None = None,  # provenance (writer round-trip)
    geom: int | None = None,        # provenance (writer round-trip)
)
```

| Parameter | Default | Units | Notes |
|---|---|---|---|
| `name` | (required) | — | identifier |
| `length` | (required) | mm | physical length of the field-map region |
| `field_data` | (required) | — | `FieldMapData`; channel geometry 1, 4, 5, or 9 |
| `scale` | 1.0 | — | global field amplitude multiplier |
| `kb` | 1.0 | — | additional magnetic-channel amplitude factor |
| `ke` | 1.0 | — | additional electric-channel amplitude factor |
| `ki` | 0.0 | — | SC compensation intensity (needs `.scc` + active SC solver) |
| `ka` | 1 | int | aperture flag (1 = use `.ouv` profile if present) |
| `phase` | 0.0 | deg | RF phase offset |
| `frequency` | 0.0 | MHz | RF frequency |
| `aperture` | 0.0 | mm | round-aperture radius; 0 = no check |
| `n_steps` | 100 | — | integration sub-steps |
| `p_flag` | 0 | int | 1 = SET_SYNC_PHASE iterative phase calibration |
| `dx..yaw_deg` | 0.0 | mm/deg | misalignment (Misalignment mixin) |
| `voltage_rel` | 0.0 | — | LLRF amplitude jitter (multiplies ke/kb) |
| `phase_offset` | 0.0 | deg | LLRF phase jitter (adds to `phase`) |
| `frequency_offset` | 0.0 | MHz | LLRF frequency jitter (adds to `frequency`) |
| `field_file` | None | — | resolved field-file prefix (set by the parser) |
| `geom` | None | int | raw geom code (set by the parser) |

The `FieldMap.from_file(name, filepath, length=None, scale=1.0,
kb=1.0, ke=1.0, ki=0.0, ka=1, phase=0.0, frequency=0.0,
aperture=0.0, n_steps=100, fm_type=1, p_flag=0)` classmethod loads
a single field-map file (`fm_type`: 1 = 1-D, 2 = 2-D electric,
7 = 2-D E+B) and infers `length` from the z table when omitted.

### Source

`linac_gen/elements/field_map.py:141` (constructor), `:210`
(`from_file`),
`linac_gen/io/tracewin_geom.py` (geom decoder),
`linac_gen/io/field_map_reader.py` (file loaders).

For the complete spec see
[`docs/fieldmap_guide.md`](https://github.com/Abhishek-Pathak-90/HELIX/blob/master/docs/fieldmap_guide.md)
in the HELIX repo, which has the full TraceWin-compatibility detail.

## See also

* [FieldMap3D](08_fieldmap3d.md) — full 3-D Cartesian variant.
* [RFGap](06_rfgap.md) — thin-kick alternative.
* PIP-II MEBT worked example —
  uses 4 QWR cavity field maps.

← [RFGap](06_rfgap.md) ·
[Continue to FieldMap3D →](08_fieldmap3d.md)
