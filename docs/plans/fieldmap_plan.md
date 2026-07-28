# Field-map implementation plan (1D + 2D-cyl + 3D)

## What TraceWin does

TraceWin's `FIELD_MAP ftype length steps thetas file_prefix amp phase ...` supports several `ftype` codes; relevant ones for us:

| ftype | Geometry | File suffixes | Field quantities |
|---|---|---|---|
| **1** | 1-D (on-axis only) | `.edz` | `E_z(z)` |
| **2** | 2-D cylindrical (E only) | `.edz` (+`.edr` derived) | `E_z(r, z)`, `E_r(r, z)` |
| 3/4 | 2-D Cartesian (y=0 plane) | `.edx`, `.edy` | rarely used for RF |
| **7** | 2-D cylindrical (EM) | `.edz`, `.bsz` | `E_z, E_r, B_z, B_r` |
| **70** | **3-D Cartesian E** | `.edx`, `.edy`, `.edz` | `E_x, E_y, E_z` on (x,y,z) |
| **71** | **3-D Cartesian B** | `.bsx`, `.bsy`, `.bsz` | `B_x, B_y, B_z` on (x,y,z) |
| 72/73 | 3-D cylindrical | `.edz`/`.bsz`-like | r-z-θ grids |
| **74** | 3-D Cartesian RF EM | 6 files | `E_xyz + B_xyz` |

File formats (values in SI: V/m for E, T for B; z in **cm** for 1D/2D, **m** for 3D):

```
# 1D (.edz, ftype 1)
n_z
z_min  z_max                      ← in cm
norm_factor                       ← single line, divides all values
value[0]
value[1]
...
value[n-1]

# 2D cylindrical (.edz, ftype 2 or 7)
n_z n_r                           ← both ints on same line
d_z d_r  (cm)                     ← grid spacing
norm_factor
E_z values  (n_z × n_r, r fastest)
E_r values  (same layout)
# if ftype 7:
B_z values
B_r values

# 3D Cartesian (ftype 70, per-component file .edx / .edy / .edz)
n_x  x_min  x_max                 ← meters for 3D
n_y  y_min  y_max
n_z  z_min  z_max
norm_factor
value(ix=0, iy=0, iz=0)
value(ix=0, iy=0, iz=1)
...
```

Trajectory integration: TraceWin uses an RK-style integrator with the interpolated field at the particle's (x, y, z) and current time `t` (via an RF phasor `cos(2πf·t + φ_0)`). For RF fields all four components scale by the same phasor; static magnets have `frequency=0` and the phasor is 1.

## What we have

| Piece | State |
|---|---|
| 1-D `.edz` reader (ftype 1) | ✅ works |
| 2-D cylindrical `.edz` reader (ftype 2, 7) | ✅ reader works; tracker **only samples r=0** — transverse RF focusing is dropped |
| 3-D reader | ❌ missing |
| RK4 tracker for field-maps | Exists in `tracking/rk4.py`, called per sub-step from `FieldMap.track_rk4` |
| RF phasor | ✅ in `elements/field_map.py` via `phase` + `frequency` |
| Tests | 4 `.edz` fixtures (1D/2D, cavity/sin) + reader unit tests |

## Gaps

1. **Proper 2-D cylindrical tracking** — need to sample `E_z(r,z)` *and* `E_r(r,z)` at the particle's r, not always r=0. Transverse RF focusing (`E_r ≈ −(r/2)·∂E_z/∂z`) comes from this; on the beam axis everything is fine, but off-axis particles are currently un-focused by the RF.
2. **3-D Cartesian reader** — parse three sibling files `.edx/.edy/.edz` (or `.bsx/.bsy/.bsz`) with a common header and per-component data arrays.
3. **3-D Cartesian tracker** — trilinear (and ideally tricubic) interpolation at the particle's (x, y, z); per-step kick using `F = q(E + v × B)`.
4. **Glue to `.dat` parser** — `FIELD_MAP 70/71/74 L steps thetas fname amp phase ...` must dispatch to the 3-D reader and build a `FieldMap3D` element.
5. **GUI element inspector** — type-aware fields for the new element class.
6. **Regression tests** — analytic cavity comparison so we can't drift.

## Proposed design

### Shared data model

```python
# linac_gen/io/field_map_data.py  (extract from reader)
@dataclass
class FieldMapData:
    symmetry: str                           # "1d" | "cylindrical" | "3d"
    z: np.ndarray                           # always
    r: np.ndarray | None = None             # cylindrical
    x: np.ndarray | None = None             # 3d
    y: np.ndarray | None = None             # 3d
    # Electric field
    Ez: np.ndarray | None = None
    Er: np.ndarray | None = None            # cyl
    Ex: np.ndarray | None = None            # 3d
    Ey: np.ndarray | None = None
    # Magnetic field
    Bz: np.ndarray | None = None
    Br: np.ndarray | None = None
    Bx: np.ndarray | None = None
    By: np.ndarray | None = None
    frequency: float = 0.0
    norm_factor: float = 1.0
```

Arrays are stored in the natural grid order: `E_z_3d.shape == (n_x, n_y, n_z)`.

### New element classes

```
linac_gen/elements/field_map.py       (existing — 1D / on-axis)
linac_gen/elements/field_map_cyl.py   (new — 2D cylindrical, uses Ez(r,z) + Er(r,z))
linac_gen/elements/field_map_3d.py    (new — 3D Cartesian, Ex/Ey/Ez (+ Bx/By/Bz))
```

`field_map.py` stays the default (1-D) class so existing tests and lattices don't change. The others are imported only when `ftype` warrants it.

### Interpolation

For each field component:

- **1-D / 2-D cylindrical**: `scipy.interpolate.RegularGridInterpolator` (linear) — built once, cached on the element, vectorised over particle batches.
- **3-D**: same, but the grid is large. Start with `RegularGridInterpolator(method="linear")`. Benchmark; if it's the hot spot, switch to a hand-written trilinear kernel in C++ under `linac_gen/csrc/` (the existing OpenMP scaffolding applies directly — each particle samples 8 corners, race-free).

### Kick assembly

The existing `FieldMap.track_rk4` loop becomes:

```python
for substep in range(n_steps):
    t  = ref.t
    x, y, z = particle.position
    phasor = cos(2π f·t + φ_0)                # 1 if static magnet
    Ex, Ey, Ez = interp_E(x, y, z) * amp * phasor
    Bx, By, Bz = interp_B(x, y, z) * amp * phasor
    # Relativistic momentum kick:  dp/dt = q(E + v × B)
    # (advance_ref handles the on-axis reference update as before.)
```

The 2-D cylindrical form collapses to:

```python
r = sqrt(x² + y²)
Ez, Er = interp_cyl(r, z) * amp * phasor
Ex = Er · x/r ; Ey = Er · y/r              # back to Cartesian
```

### Parser changes

`io/tracewin_parser.py` already dispatches `FIELD_MAP`; we extend the switch to choose the right reader + element class based on `ftype`:

| ftype | reader entry | element class |
|---|---|---|
| 1 | `read_edz_1d(file)` | `FieldMap` (current) |
| 2 or 7 | `read_edz_2d(file, ftype)` | `FieldMap2DCyl` (new) |
| 70 | `read_3d_cart_E(prefix)` | `FieldMap3D(E_only=True)` |
| 71 | `read_3d_cart_B(prefix)` | `FieldMap3D(B_only=True)` |
| 74 | `read_3d_cart_EB(prefix)` | `FieldMap3D(E=True, B=True)` |

`prefix` here is the card's filename without the final `.edz/.edx/.edy/.bsz/...` suffix. Reader probes each sibling and errors if missing.

### File paths

Relative paths in the `.dat` are resolved against the directory containing the `.dat` (same rule TraceWin uses). Absolute paths pass through unchanged.

### Units

| Quantity | In TraceWin file | In `FieldMapData` |
|---|---|---|
| z (1D, 2D) | cm | **mm** |
| x, y, z (3D) | **m** | **mm** |
| E | V/m | **V/m** (unchanged) |
| B | T | **T** (unchanged) |
| frequency | MHz | MHz |

One gotcha: 1D/2D `.edz` files store z in cm but 3D files store z (and x, y) in m. The reader handles this per format — we document it once in `FieldMapData` and check in tests.

## Phased execution

Every phase ends with a green `pytest tests/` and a working `.dat` round-trip.

- **P1 — refactor FieldMapData into its own module** (no behaviour change). Existing reader + element switch to import from new home. Baseline tests still pass.

- **P2 — proper 2-D cylindrical tracking** (uses reader that already exists). New element class `FieldMap2DCyl` samples `E_z(r, z)` + `E_r(r, z)` via RGI. Kick applies `E_z` along z, `E_r · x/r` along x, `E_r · y/r` along y. Regression test: a pillbox cavity `E_z = E_0 · cos(π z/L)` with the paraxial approximation `E_r = −r/2 · ∂E_z/∂z` — analytical transverse kick vs our tracker at 1 % tolerance. 1-D `FieldMap` stays the default for `fm_type=1`.

- **P3 — 3-D Cartesian reader** for `.edx/.edy/.edz` triplets (ftype 70) and `.bsx/.bsy/.bsz` (71). One function per triplet, common header parser. Test-generated 3-D `cos(kz) · exp(−r²)` fixture checked against analytic values at grid corners.

- **P4 — 3-D Cartesian tracker** (new `FieldMap3D` element). Uses `RegularGridInterpolator(method="linear")` for each component; RF phasor applied uniformly. Regression test: 3-D field file generated from the same pillbox used in P2 → 3-D tracker gives the same kicks as 2-D-cyl within 0.1 % (floating-point and interpolation noise).

- **P5 — `.dat` parser dispatch** for ftype ∈ {2, 7, 70, 71, 74} and path resolution relative to the `.dat`. Round-trip: write-then-read a `.dat` that names a 3-D map and have the simulator produce the same output as loading the map directly.

- **P6 — GUI integration**:
  - Element inspector gets type-aware fields (`fm_type`, `file prefix`, `amp`, `phase`, `freq`, `aperture`).
  - Interphase Lattice-tab element summary shows 3D field-maps with a dedicated colour swatch.
  - Optional: a "Field viewer" button that opens a read-only heatmap (`xz` slice for 3-D, `rz` for cyl) using pyqtgraph `ImageView`. Low priority — gated on user interest.

- **P7 — performance pass**:
  - If `RegularGridInterpolator` proves to be a bottleneck for 3-D with many particles, move interpolation to `linac_gen/csrc/field_interp_3d.cpp` with an OpenMP-parallel trilinear kernel. The OMP scaffolding from the parallelization branch applies directly.

- **P8 — docs + examples**:
  - Extend `docs/tracewin-compat.md` with the supported `FIELD_MAP` ftypes and file-format notes.
  - Add `examples/field_map_3d_cavity.dat` referencing a tiny synthetic 3-D map so users have a working reference.

## Scope NOT in this plan (explicit)

- ftype 3/4 (2-D Cartesian y=0 anti/symmetric) — rarely used for RF, easy to add later if needed.
- ftype 72/73 (3-D cylindrical) — covered by the Cartesian variant for all typical cavity shapes; add only if requested.
- User-defined ftype ≥ 100.
- Automatic field-map import from CST / HFSS / COMSOL — out of scope; keep the existing CSV reader as an escape hatch.
- GPU interpolation — later, if ever.

## Test matrix

| Test | What it pins |
|---|---|
| `test_fieldmap_1d_reader` (exists) | 1D `.edz` parse + unit conversion |
| `test_fieldmap_2d_reader` (exists) | 2D `.edz` parse for ftype 2 and 7 |
| `test_fieldmap_2d_cyl_tracker` (P2) | `FieldMap2DCyl` transverse kicks match paraxial formula |
| `test_fieldmap_3d_reader` (P3) | 3D triplet parse + header consistency + norm factor |
| `test_fieldmap_3d_tracker_identity` (P4) | 3D map sampled from an axisymmetric field ≡ 2D cyl tracker |
| `test_fieldmap_dat_dispatch` (P5) | ftype ∈ {1, 2, 7, 70, 71, 74} → correct reader + element |
| `test_fieldmap_3d_parallel_parity` (P4) | 3D tracker with OMP_NUM_THREADS ∈ {1, 4, 8} produces bitwise-identical ε_x end |

Each regression test runs < 3 s on CI.

## Estimated effort

| Phase | LOC | Effort |
|---|---|---|
| P1 refactor | ~80 | 30 min |
| P2 2D-cyl tracker | ~200 | ~2 h |
| P3 3D reader | ~180 | ~1 h |
| P4 3D tracker | ~250 | ~3 h |
| P5 parser dispatch | ~60 | 30 min |
| P6 GUI integration | ~180 | ~1 h |
| P7 C++ 3D interp | ~200 | ~2 h (optional, only if needed) |
| P8 docs + example | ~120 | 30 min |
| **Total** | **~1 100 LOC** | **~9 h** including tests |

The 2-D cylindrical fix (P2) is the single highest-value deliverable: it fixes RF transverse focusing that is currently being silently dropped. If time is tight, ship P1 + P2 + P3 + P5 (without P4 tracker) and keep 3-D as "reader-only" until followup. But the plan above delivers the full 1-D + 2-D + 3-D stack as requested.

## Risk register

| Risk | Mitigation |
|---|---|
| 3-D maps can be huge (100 MB+ per file) → RAM blow-up | Lazy-load + keep one `FieldMapData` per element; share between identical ftypes via content hash cache |
| `RegularGridInterpolator` latency dominates runtime | Planned C++ fallback in P7; gate behind env var `LINAC_GEN_FAST_FIELDMAP=1` |
| TraceWin unit quirks (cm vs m vs mm) mis-converted → silent wrong physics | Pillbox-cavity regression test compares absolute σ_xp kick to analytic formula — catches any off-by-factor-10 |
| RK4 + high-frequency RF phasor under-resolves at coarse `n_steps` | Keep the `n_steps` user knob; document `n_steps ≥ 100 × (length / wavelength)` rule of thumb |
| File-path resolution surprises on Windows WSL | Resolve against `os.path.dirname(dat_path)`; use `pathlib.Path` everywhere |

## Rollout to the two GUIs

- **Classic GUI** (`linac_gen_gui/app.py`): transparent — it already dispatches through the parser. Element inspector just needs the new type mapping.
- **Interphase** (`linac_gen_gui/interphase/`): same — the Lattice tab shows whatever element classes the parser returns. The Matching / Convergence tabs need no change.

## Go/no-go decision points

1. After P2: does the 2-D-cyl tracker improve convergence of transverse emittance growth for a realistic DTL lattice? If yes, P3–P8 are justified.
2. After P4: are 3-D fieldmaps actually in use by any planned lattice? If not, hold P7 C++ acceleration until it's a real bottleneck.
