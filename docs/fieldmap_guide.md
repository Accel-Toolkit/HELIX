# TraceWin FIELD_MAP in Linac_Gen

Linac_Gen implements the full TraceWin FIELD_MAP spec — `geom` 5-digit
encoding, every channel (static/RF × electric/magnetic), file-naming
conventions, `FIELD_MAP_PATH` directive, Ka aperture flag, Ki SC
compensation, and p_flag absolute/relative phase.  This guide summarises
what works today and how to use it.

> **Spec reference:** every semantics claim in this guide is taken from
> the TraceWin user manual at
> `C:/Users/abhishek/TraceWin/doc/tracewin.htm` (sections
> "Field map file syntax" §7860-8080 and "FIELD_MAP" element §17830-18434).

---

## The FIELD_MAP card

```
FIELD_MAP  geom  L  θᵢ  R  kb  ke  Ki  Ka  FileName  [p_flag]
```

| Position | Symbol | Type | Default | Description |
|---|---|---|---|---|
| 1 | `geom` | int | required | 5-digit geometry code (see below) |
| 2 | `L` | float (mm) | required | Element length |
| 3 | `θᵢ` | float (deg) | 0.0 | RF phase offset |
| 4 | `R` | float (mm) | 0.0 | Transverse aperture radius |
| 5 | `kb` | float | 1.0 | Magnetic-field amplitude scale |
| 6 | `ke` | float | 1.0 | Electric-field amplitude scale |
| 7 | `Ki` | float | 0.0 | Space-charge compensation scale |
| 8 | `Ka` | int | 1 | Aperture flag (0/1/2) |
| 9 | `FileName` | str | required | File prefix (no extension; see below) |
| 10 | `p_flag` | int | 0 | Phase reference: 0 = relative, 1 = absolute |

The physical field amplitude at a given channel is
`stored_value × k / Norm`, where `k = ke` for electric channels and
`k = kb` for magnetic channels, and `Norm` is the normalisation constant
stored in the field-map file header.

---

## The `geom` 5-digit encoding

The `geom` integer encodes up to five fields packed in powers of 10:

```
geom = aper·10⁴ + rf_B·10³ + rf_E·10² + stat_B·10 + stat_E
```

Each digit selects the **geometry** of one field channel (0 = channel
absent).  The decoder in `linac_gen/io/tracewin_geom.py` extracts these
five digits:

| Digit position | Channel | Constant | Meaning |
|---|---|---|---|
| ones (10⁰) | static electric | `Channel.STAT_E` | E-field, DC |
| tens (10¹) | static magnetic | `Channel.STAT_B` | B-field, DC |
| hundreds (10²) | RF electric | `Channel.RF_E` | E-field, oscillating |
| thousands (10³) | RF magnetic | `Channel.RF_B` | B-field, oscillating |
| ten-thousands (10⁴) | aperture | `GeomCode.aper` | opens `.ouv` pipe file |

### Digit values and their geometry

| Digit | Geometry | Notes |
|---|---|---|
| 0 | absent | channel disabled |
| 1 | 1-D on-axis `F(z)` | single file `.<fl><tl>z` |
| 4 | 2-D cylindrical E-type | `Er(r,z)` + `Ez(r,z)`; RF adds `.bdq` for Bθ |
| 5 | 2-D cylindrical B-type | `Br(r,z)` + `Bz(r,z)`; RF adds `.edq` for Eθ |
| 6 | 2-D Cartesian | `Fx(x,y)` + `Fy(x,y)` — no element class yet |
| 7 | 3-D Cartesian | `Fx(x,y,z)` + `Fy(x,y,z)` + `Fz(x,y,z)` |
| 8 | 3-D cylindrical | not implemented per TraceWin manual itself |
| 9 | 1-D quad gradient G(z) | `stat_B` slot only; `Bz` file contains T/m gradient |

A **negative** `geom` means "apply 2nd-order off-axis expansion instead
of 1st order" — the sign is captured in `GeomCode.second_order` but the
expansion is not yet used; it is logged and the magnitude is decoded
normally.

---

## Channel → file-extension cheat sheet

The table below is derived directly from `component_files()` in
`linac_gen/io/tracewin_geom.py`.  `fl` is the field letter (`e` or `b`)
and `tl` is the type letter (`s` static, `d` dynamic/RF).

| Channel | `fl` | `tl` | digit 1 | digit 4/5 | digit 7 |
|---|---|---|---|---|---|
| `STAT_E` | `e` | `s` | `.esz` | `.esr` `.esz` | `.esx` `.esy` `.esz` |
| `STAT_B` | `b` | `s` | `.bsz` | `.bsr` `.bsz` | `.bsx` `.bsy` `.bsz` |
| `RF_E` | `e` | `d` | `.edz` | `.edr` `.edz` (+ `.bdq`) | `.edx` `.edy` `.edz` |
| `RF_B` | `b` | `d` | `.bdz` | `.bdr` `.bdz` (+ `.edq`) | `.bdx` `.bdy` `.bdz` |

The `.bdq` (Bθ) file is added for RF_E digit-4 (TM-mode solenoid field).
The `.edq` (Eθ) file is added for RF_B digit-5 (TE-mode).

---

## Unit conventions

- **Spatial coordinates in field files:** metres (reader converts to mm internally)
- **Electric field values:** MV/m
- **Magnetic field values:** T
- **Frequency (from FREQ directive):** MHz
- **Physical amplitude at tracking time:**
  `stored_value × k / Norm` where `k = ke` (electric) or `k = kb` (magnetic)
- **Phasor (RF channels only, per manual §18122):**
  - RF electric: `E(t) = E₀ · cos(ωt + φ)`
  - RF magnetic: `B(t) = B₀ · sin(ωt + φ)` — 90° offset versus electric
  - Static channels: phasor = 1 always

---

## Common `geom` values — real-world examples

| `geom` | What it represents | Files opened |
|---|---|---|
| `1` | 1-D static electric | `.esz` |
| `10` | 1-D static magnetic | `.bsz` |
| `100` | 1-D RF electric cavity | `.edz` |
| `400` | 2-D cyl RF electric (TM) | `.edr` `.edz` `.bdq` |
| `10` or `90` | 1-D quad gradient G(z) | `.bsz` (digit 9 enforced on `STAT_B` only) |
| `70` | 3-D static magnetic (dipole / quad / solenoid) | `.bsx` `.bsy` `.bsz` |
| `700` | 3-D RF electric cavity | `.edx` `.edy` `.edz` |
| `7700` | 3-D RF electric + magnetic (full EM cavity) | `.edx` `.edy` `.edz` `.bdx` `.bdy` `.bdz` |
| `10070` | 3-D static magnetic + aperture override | `.bsx` `.bsy` `.bsz` (and reads `.ouv`) |

---

## Physics reference — Lorentz force in engineering units

From the element docstring (matches TraceWin manual §18070):

```
E in MV/m, B in T, positions in mm, angles in mrad, Δs in m

  Δx' [rad] = q · Δs · E_x / (γβ² · mc²)
             + q · Δs · (y'·Bz − By) · 299.792458 / (γβ · mc²)

  Δy' [rad] = q · Δs · E_y / (γβ² · mc²)
             + q · Δs · (Bx − x'·Bz) · 299.792458 / (γβ · mc²)
```

where `q` is in units of the elementary charge, `mc²` is the rest energy
in MeV, and the constant `299.792458 = c × 10⁻⁶` converts `B[T]` to
`MV/m`-equivalent units.

---

## `FIELD_MAP_PATH` global directive

```
FIELD_MAP_PATH  /abs/or/relative/path
```

Sets a global prefix directory for subsequent `FIELD_MAP` filenames.
Resolution order for a `FIELD_MAP FileName`:

1. If `FileName` is an absolute path — use it as-is.
2. Else if a `FIELD_MAP_PATH` has been set — resolve relative to that.
3. Else — resolve relative to the directory containing the `.dat` file.

A relative path given to `FIELD_MAP_PATH` is resolved against the
`.dat` file's directory at parse time.  The directive is persistent for
all subsequent `FIELD_MAP` cards until the end of the file.

---

## `FileName` syntax notes

- Extensions are optional in the card: both `scl_cav` and `scl_cav.edz`
  resolve to the same set of files.  The reader strips any known suffix
  (`.edz`, `.bsz`, `.edx`, etc.) before appending the channel-specific
  extensions.
- Paths containing spaces must be **quoted** in the `.dat` file:
  ```
  FIELD_MAP  70  120.0  -10.0  20.0  1.0 1.0 0.0 1  "my maps/cav 01"
  ```
  The parser uses `shlex.split` so POSIX quoting rules apply.
- Relative paths are resolved as described under `FIELD_MAP_PATH` above.

---

## p_flag — absolute vs relative RF phase

The optional last parameter on the `FIELD_MAP` card (also on `GAP`):

| `p_flag` | Meaning |
|---|---|
| `0` (default) | **Relative phase** — `φ_sync = θᵢ + φ_ref`, where `φ_ref` is the running reference-particle phase accumulated through the lattice |
| `1` | **Absolute phase** — `φ_sync = θᵢ` exactly, ignoring the reference-particle phase |
| `2`, `3` | TraceWin extension variants — parsed and stored; same behaviour as `1` in Linac_Gen currently |

Implementation in `FieldMap._phi_sync_rad()`:

```python
if self.p_flag == 1:
    return self.phase * (np.pi / 180.0)          # absolute
return (self.phase + ref.phi_s) * (np.pi / 180.0)  # relative
```

---

## Ka — aperture flag

Stored as `fd.ka` on the returned `FieldMapData`.

| `Ka` | Meaning |
|---|---|
| `0` | No aperture override — use the standard circular aperture (`R`) |
| `1` | Load `<prefix>.ouv` pipe-radius profile (z in m, r in m) and store as `fd.pipe_radius_profile = (z_mm, r_mm)` |
| `2` | TraceWin 2nd variant — parsed and stored; no extra file |

When `Ka=1` the reader raises `FileNotFoundError` if the `.ouv` file is
absent.  Per-step application of the aperture profile during tracking is
a follow-up item (see **Deferred** below); the file is loaded and flagged
so it is available when the tracker is extended.

---

## Ki — space-charge compensation

`Ki` (field 7 on the card) is the scale factor for the space-charge
compensation current profile.  When non-zero:

1. The reader looks for `<prefix>.scc` — a two-column text file
   (z in m, current in A).
2. The file is loaded, the z-column converted to mm, and stored as
   `fd.scc_profile` (ndarray, shape (N, 2)) with `fd.scc_scale = Ki`.
3. The reader raises `FileNotFoundError` if `Ki != 0` and the file is
   absent.

Application of the compensation during tracking is a follow-up item
(see **Deferred** below).

---

## File header formats

### 1-D file (digit 1)

```
Nz   Zmax[m]         ← Nz = number of intervals; Nz+1 data points follow
Norm
Fz[0]               ← F at z=0
Fz[1]               ← F at z=1·Zmax/Nz
...
Fz[Nz]              ← F at z=Zmax
```

Grid: `z = linspace(0, Zmax, Nz+1)` converted to mm.

### 2-D cylindrical file (digit 4 or 5)

```
Nz   Zmax[m]
Nr   Rmax[m]
Norm
Fz values           ← (Nz+1)×(Nr+1) values, r is fastest axis:
                       for k=0..Nz: for i=0..Nr: F(z_k, r_i)
Fr values           ← same layout
[Fq values if RF mode]
```

Return-shape: `FieldChannel.Fz.shape == (Nr+1, Nz+1)` after transposing
to (r, z) order for interpolation.

### 3-D Cartesian file (digit 7)

```
Nz   Zmax[m]
Nx   Xmin[m]   Xmax[m]
Ny   Ymin[m]   Ymax[m]
Norm
                    ← (Nz+1)×(Ny+1)×(Nx+1) values; x is FASTEST axis:
                       for k=0..Nz: for j=0..Ny: for i=0..Nx: F(z_k, y_j, x_i)
```

The reader reshapes as `(Nz+1, Ny+1, Nx+1)` then transposes to
`(Nx+1, Ny+1, Nz+1)` so `FieldChannel.Fz.shape == (n_x, n_y, n_z)`,
matching the `RegularGridInterpolator` axes order `(x, y, z)`.

---

## End-to-end example

Minimal `.dat` with `FIELD_MAP_PATH` and a 3-D static-magnetic solenoid:

```
TITLE  MEBT solenoid section
FREQ   325.0
FIELD_MAP_PATH  /data/linac/fieldmaps

DRIFT   50.0  20.0
FIELD_MAP  70  200.0  0.0  20.0  1.0  1.0  0.0  0  sol_01  0
DRIFT   50.0  20.0
END
```

Required files (all in `/data/linac/fieldmaps/`):

```
sol_01.bsx
sol_01.bsy
sol_01.bsz
```

Each file follows the 3-D Cartesian header format.  The `kb=1.0` means
the stored values are used verbatim (no extra scale beyond `1/Norm`).

---

## Test suite — behaviour pinned

| Test file | Scope |
|---|---|
| `tests/io/test_tracewin_geom.py` | `decode_geom` / `component_files` / `enabled_channels` |
| `tests/io/test_field_map_reader.py` | 1-D and 2-D legacy reader round-trips |
| `tests/io/test_field_map_tracewin_spec.py` | TraceWin-canonical 1-D and 2-D header format |
| `tests/io/test_field_map_3d_reader.py` | 3-D reader: shape, units, norm_factor, missing-file |
| `tests/io/test_tracewin_fieldmap_reader.py` | `read_tracewin_fieldmap` multi-channel round-trip |
| `tests/io/test_tracewin_field_map_dispatch.py` | Parser routes every geom to the right element class |
| `tests/io/test_field_map_path_directive.py` | `FIELD_MAP_PATH` directory override |
| `tests/io/test_fieldmap_scc_ka.py` | Ki `.scc` loading; Ka=1 `.ouv` loading |
| `tests/elements/test_field_map.py` | 1-D on-axis and paraxial tracking |
| `tests/elements/test_field_map_2d_cyl.py` | 2-D cyl tracker picks up Er at r > 0 |
| `tests/elements/test_field_map_3d.py` | 3-D tracker agrees with 2-D cyl on axis-symmetric field |
| `tests/elements/test_field_map_3d_multichannel.py` | Multi-channel 3-D element tracking |
| `tests/elements/test_field_map_p_flag.py` | Absolute vs relative phase (p_flag=0/1) |
| `tests/elements/test_field_map_solenoid.py` | PXIE solenoid regression (geom=70) |

---

## Currently deferred (documented as follow-up)

- **`SUPERPOSE_MAP` / `SUPERPOSE_MAP_OUT`** — positioning of overlapping
  field maps is not yet implemented.
- **`EXCITATION_CURVE`** — `kb` as a function of power-supply current;
  the manual §18230 describes a lookup-table mechanism not yet wired up.
- **Negative `geom`** — the 2nd-order off-axis expansion; `second_order`
  is stored on `GeomCode` but the expansion factor is not applied at
  tracking time.
- **Ka=1 per-z aperture enforcement** — the `.ouv` file is loaded and
  stored on `fd.pipe_radius_profile`; the tracker does not yet query it
  step-by-step.
- **Ki SC compensation during tracking** — the `.scc` profile is loaded;
  the space-charge correction force is not yet computed or applied.
- **Digit 8 (3-D cylindrical)** — flagged "not implemented yet" in the
  TraceWin manual itself; `component_files()` raises `NotImplementedError`.
- **Digit 6 (2-D Cartesian)** — `component_files()` returns the correct
  extension list but no element class handles this geometry yet.
- **TraceWin-writer round-trip** — `FIELD_MAP` elements can be read and
  tracked but are not yet serialised back to `.dat` format by
  `tracewin_writer.py`.
