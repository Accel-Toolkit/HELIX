# TraceWin FIELD_MAP `geom` — full implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current hard-coded `fm_type ∈ {70, 71, 74}` dispatch with a correct implementation of the TraceWin `geom` 5-digit encoding, so every channel (static-E, static-B, RF-E, RF-B, aperture) is loaded from the right sibling files with the right phasor and scaling during tracking, and real-world files such as `Fields/SOL*-PXIE.*` (geom=70, 3-D static magnetic solenoid) focus a beam correctly.

**Architecture:**  A small `tracewin_geom.py` module decodes `geom` into a `GeomCode` and maps each enabled channel to the correct file extensions per the manual.  `FieldMapData` is refactored into a *multi-channel* container (one channel per enabled (kind, geometry) pair).  A new top-level `read_tracewin_fieldmap(geom, prefix, base_dir)` loads every required file and populates the channels.  Element classes (`FieldMap`, `FieldMap3D`) iterate channels during tracking and apply the correct phasor (no phasor for static; `cos(ωt+φ)` for RF-E; `sin(ωt+φ)` for RF-B per manual §18122) and amplitude scaling (`ke/Norm` for E, `kb/Norm` for B).

**Tech stack:**  Python 3.12+, numpy, scipy (`RegularGridInterpolator`), pytest.  No new 3rd-party deps.

**Authoritative spec:** TraceWin user manual at `/mnt/c/Users/abhishek/TraceWin/doc/tracewin.htm`.  Key anchors: `#_Field_Map` (line 17830), `#FIELD_MAP` card table (line 17867), 5-digit `geom` decoder (line 18052), file-extension table (line 18222-18405), phasor convention (line 18122).  Field-file header syntax (1-D/2-D/3-D) at line 7860.

**Scope of this plan — every parameter on the FIELD_MAP card:**

| Card param | Done by task | Notes |
|---|---|---|
| `geom` (5-digit encoded) | Task 1, 2 | full decode |
| `L` (length, mm) | Task 8 | passed to element; card-L vs file-Zmax handled by `fill_value=0` at sample time |
| `θᵢ` (phase, deg) | Task 6, 7 | used in RF phasor; static channels ignore it |
| `R` (aperture, mm) | Task 8 | applied in existing `FieldMapElement` loss-check |
| `kb`, `ke` (field scales) | Task 6, 7 | applied as `k/Norm` per channel |
| `Ki` (SC-compensation scale) | **Task 2c** | parses `.scc` if `Ki>0`, stored on FieldMapData |
| `Ka` (aperture flag 0/1/2) | **Task 2d** | Ka=0: R-based loss (existing); Ka=1: read `.ouv` pipe-radius map; Ka=2: skip R loss |
| `Filename` (no-ext, relative/abs path) | Task 5 | `_strip_known_suffix` + `base_dir` resolution |
| `Filename` (quoted w/ spaces) | **Task 0b** | upgrade tokenizer to `shlex.split` (one-line change, ripples through parser — done up front) |
| `p_flag` (0 relative / 1 absolute phase) | **Task 6b, 7b** | applied in RF phasor: absolute uses `θᵢ` only; relative adds `ref.phi_s` |
| `FIELD_MAP_PATH` global directive | **Task 0c** | parsed up-front into parser state; prepended to each subsequent FIELD_MAP filename |
| negative geom (2nd-order off-axis) | flag parsed in Task 1 | expansion itself deferred — flagged in `metadata["warnings"]` |
| Digit 8 (3-D cylindrical) | Task 2 | `NotImplementedError` per the manual's own disclaimer |

**Still explicitly deferred (documented as follow-up):**
- `SUPERPOSE_MAP` / `SUPERPOSE_MAP_OUT` positioning
- `EXCITATION_CURVE` (where `kb` is a power-supply current)
- 2nd-order off-axis Taylor expansion (negative geom)
- Full aperture-map sampling during tracking (we load the `.ouv` file but the per-slice radius check in `FieldMapElement` is currently a single scalar `R`)
- TraceWin-writer round-trip of FIELD_MAP elements

---

## File structure

**Create:**
- `linac_gen/io/tracewin_geom.py` — pure-logic module: `GeomCode`, `decode_geom`, `component_files`, channel enum
- `linac_gen/io/tracewin_fieldmap_reader.py` — per-channel & top-level readers, orchestrating multi-channel loads
- `tests/io/test_tracewin_geom.py` — unit tests for decoder and extension mapping
- `tests/io/test_tracewin_fieldmap_reader.py` — integration tests for top-level reader against synthetic fixtures
- `tests/elements/test_field_map_solenoid.py` — regression test using `Fields/SOL1-PXIE.*` vs analytic thick-lens formula
- `tests/io/fixtures/tracewin_geom/` — minimal fixtures for each geometry digit (1, 4, 5, 6, 7, 9)

**Modify:**
- `linac_gen/io/field_map_data.py` — multi-channel container (`channels: dict[ChannelKind, FieldChannel]`)
- `linac_gen/io/field_map_reader.py` — per-file readers stay; `read_edz_1d/2d` and the 3-D readers return a *single* `FieldChannel`, not a full `FieldMapData`, so the orchestrator can compose them
- `linac_gen/elements/field_map.py` — tracking iterates the channels
- `linac_gen/elements/field_map_3d.py` — same; plus `v × Bz` solenoid rotation; plus correct static-vs-RF phasor
- `linac_gen/io/tracewin_parser.py` — FIELD_MAP branch uses `decode_geom` + new top-level reader
- `linac_gen/io/tracewin_syntax.py` — tweak `FIELD_MAP` schema comment (geom is 5-digit encoded)
- `tests/io/test_tracewin_field_map_dispatch.py` — convert from 70/71/74 special cases to realistic geoms (70 static mag, 100 RF-E 1-D, 400 RF-E 2-D cyl TM, 7700 RF EB 3-D Cart, 90 quad grad)
- `tests/io/test_field_map_3d_reader.py` — rename to exercise per-channel 3-D Cart reader
- `tests/io/test_field_map_reader.py` — adapt to the channel-shaped `FieldMapData`
- `tests/io/test_field_map_tracewin_spec.py` — adapt to channel-shaped return
- `docs/fieldmap_guide.md` — rewrite to describe geom encoding + all channels + file-naming table

**Branching:**  All work on new branch `feat/tracewin-geom`.  Current uncommitted work (cm→m units and reshape order fixes) is committed first as a checkpoint so the branch history is clean.

---

## Reference tables (copy verbatim into code comments where relevant)

### `geom` decoding

```
geom = |g|   (negative → second_order_off_axis = True)
stat_E = g % 10
stat_B = (g // 10) % 10
rf_E   = (g // 100) % 10
rf_B   = (g // 1000) % 10
aper   = (g // 10000) % 10
```

### Digit → geometry

| d | meaning                                      | valid in channels |
|---|----------------------------------------------|-------------------|
| 0 | no field                                     | all                |
| 1 | 1-D Fz(z)                                    | stat_E, stat_B, rf_E, rf_B |
| 4 | 2-D cyl *electric-type* (Fr, Fz; +Bθ if RF)  | stat_E, rf_E       |
| 5 | 2-D cyl *magnetic-type* (Fr, Fz; +Eθ if RF)  | stat_B, rf_B       |
| 6 | 2-D Cart (Fx, Fy)                            | all                |
| 7 | 3-D Cart (Fx, Fy, Fz)                        | all                |
| 8 | 3-D cyl                                      | raises NotImplementedError (manual: "not implemented yet") |
| 9 | 1-D G(z), quad gradient only                 | stat_B only; raises if used elsewhere |

### Channel → filename prefix letters

| Channel  | 1st | 2nd |
|----------|-----|-----|
| stat_E   | `e` | `s` |
| stat_B   | `b` | `s` |
| rf_E     | `e` | `d` |
| rf_B     | `b` | `d` |

### (Channel, digit) → component file extensions

| d / channel | stat_E         | stat_B          | rf_E                   | rf_B                   |
|-------------|----------------|-----------------|------------------------|------------------------|
| 1           | `.esz`         | `.bsz`          | `.edz`                 | `.bdz`                 |
| 4           | `.esr .esz`    | —               | `.edr .edz` + `.bdq`†  | —                      |
| 5           | —              | `.bsr .bsz`     | —                      | `.bdr .bdz` + `.edq`†  |
| 6           | `.esx .esy`    | `.bsx .bsy`     | `.edx .edy`            | `.bdx .bdy`            |
| 7           | `.esx .esy .esz` | `.bsx .bsy .bsz` | `.edx .edy .edz`    | `.bdx .bdy .bdz`       |
| 9           | —              | `.bsz`          | —                      | —                      |

† TM mode (digit 4 in rf_E) also includes Bθ in `.bdq`; TE mode (digit 5 in rf_B) also includes Eθ in `.edq`.  These "cross-field" components are stored on the *same channel* object (TM stores Bθ alongside E; TE stores Eθ alongside B), not a separate channel, because the two are phase-locked by Maxwell's equations and belong to the same mode.

### Phasor and scaling (manual §18066, §18122, §18180)

```
applied_field = stored_value × k / Norm,  k = ke (E), kb (B)

Static:           no time factor.
RF electric:      E(x,y,z,t) = E₀(x,y,z) · k_e/Norm · cos(ωt + φ₀ + Δφ)
RF magnetic:      B(x,y,z,t) = B₀(x,y,z) · k_b/Norm · sin(ωt + φ₀ + Δφ)
```

where ω = 2π · freq, φ₀ = `phase` from card, Δφ = per-particle phase deviation.

### Cartesian ↔ cylindrical (manual §18143)

```
Bx = -Bθ · y/r          By = +Bθ · x/r
Ex = -Eθ · y/r          Ey = +Eθ · x/r
```
(x, y, z in a direct frame.)

---

## Task 0b — Upgrade tokenizer to `shlex.split` (quoted filenames)

**Why:** the manual says FIELD_MAP filenames can be quoted when the path contains spaces (§18419).  Our parser currently does `line.split()` which breaks any such path into multiple tokens.  A one-line change to `shlex.split` fixes it, but needs a careful regression pass.

**Files:**
- Modify: `linac_gen/io/tracewin_parser.py:88`
- Create: `tests/io/test_tracewin_parser_quoted_paths.py`

- [ ] **Step 1: Failing test**

```python
"""FIELD_MAP filenames with quoted spaced paths must parse."""
import os, numpy as np, pytest
from linac_gen.io.tracewin_parser import parse_tracewin


def test_quoted_filename_with_space(tmp_path):
    # Make a subdir with a space in its name and put a minimal .edz inside
    dir_with_space = tmp_path / "my maps"
    dir_with_space.mkdir()
    edz = dir_with_space / "cav.edz"
    edz.write_text("3 0.01\n1.0\n0\n0.5\n1.0\n0.5\n")     # 4 values, Nz=3
    dat = tmp_path / "lattice.dat"
    dat.write_text(
        'TITLE t\nFREQ 352.21\n'
        f'FIELD_MAP 100 10 0 20 1 1 0 1 "my maps/cav" 0\nEND\n'
    )
    lat, meta = parse_tracewin(str(dat))
    assert any(type(e).__name__ == "FieldMap" for e in lat.elements)
    assert meta["warnings"] == [], meta["warnings"]
```

- [ ] **Step 2: Run; expect `FileNotFoundError` because parser splits `"my maps/cav"` into two tokens.

- [ ] **Step 3: Replace the tokenizer**

In `linac_gen/io/tracewin_parser.py` around line 88:

```python
import shlex
...
            # Preserve quoted filenames with spaces; drop inline comments first.
            line = raw_line.split(";")[0].strip()
            if not line:
                continue
            tokens = shlex.split(line, posix=True)
```

- [ ] **Step 4: Re-run the full parser suite to confirm no regressions**

```bash
python3 -m pytest tests/io/test_tracewin_parser.py tests/io/test_tracewin_parser_quoted_paths.py -v
```

- [ ] **Step 5: Commit**

```bash
git add linac_gen/io/tracewin_parser.py tests/io/test_tracewin_parser_quoted_paths.py
git commit -m "feat(parser): shlex.split for FIELD_MAP filenames with spaces"
```

---

## Task 0c — `FIELD_MAP_PATH` global directive

**Why:** Manual §18425 lets a user set a base directory once at the top of the `.dat` (`FIELD_MAP_PATH path/to/maps`) so every subsequent `FIELD_MAP` filename resolves against it.  Without this, any `.dat` using the command fails to find its files.

**Files:**
- Modify: `linac_gen/io/tracewin_syntax.py` — new schema entry
- Modify: `linac_gen/io/tracewin_parser.py` — track state, resolve filenames
- Create: `tests/io/test_field_map_path_directive.py`

- [ ] **Step 1: Failing test**

```python
"""FIELD_MAP_PATH changes the resolution base for subsequent FIELD_MAP cards."""
import os, pytest
from linac_gen.io.tracewin_parser import parse_tracewin


def test_field_map_path_directive(tmp_path):
    sub = tmp_path / "maps"
    sub.mkdir()
    (sub / "cav.edz").write_text("3 0.01\n1.0\n0\n0.5\n1.0\n0.5\n")
    dat = tmp_path / "lattice.dat"
    dat.write_text(
        "TITLE t\nFREQ 352.21\n"
        f"FIELD_MAP_PATH {sub}\n"
        "FIELD_MAP 100 10 0 20 1 1 0 1 cav 0\nEND\n"
    )
    lat, meta = parse_tracewin(str(dat))
    assert any(type(e).__name__ == "FieldMap" for e in lat.elements)
    assert meta["warnings"] == [], meta["warnings"]


def test_field_map_path_relative_to_dat(tmp_path):
    """Path is relative to the .dat file's directory."""
    sub = tmp_path / "maps"
    sub.mkdir()
    (sub / "cav.edz").write_text("3 0.01\n1.0\n0\n0.5\n1.0\n0.5\n")
    dat = tmp_path / "lattice.dat"
    dat.write_text(
        "TITLE t\nFREQ 352.21\n"
        "FIELD_MAP_PATH maps\n"
        "FIELD_MAP 100 10 0 20 1 1 0 1 cav 0\nEND\n"
    )
    lat, meta = parse_tracewin(str(dat))
    assert meta["warnings"] == [], meta["warnings"]
```

- [ ] **Step 2: Add schema entry**

`linac_gen/io/tracewin_syntax.py`:

```python
    "FIELD_MAP_PATH": [
        Field("path", str, required=True),      # abs or relative to .dat dir
    ],
```

- [ ] **Step 3: Handle in parser**

`linac_gen/io/tracewin_parser.py`, add a state variable `field_map_path: str | None = None` and a branch:

```python
elif keyword == "FIELD_MAP_PATH":
    kw = parse_positionals(SCHEMA["FIELD_MAP_PATH"], params)
    p = kw["path"]
    if not os.path.isabs(p):
        p = os.path.join(base_dir, p)
    field_map_path = p
```

Then in the FIELD_MAP branch (Task 8) resolve `prefix`:

```python
raw_name = kw["filename"]
if os.path.isabs(raw_name):
    prefix = raw_name
elif field_map_path is not None:
    prefix = os.path.join(field_map_path, raw_name)
else:
    prefix = os.path.join(base_dir, raw_name)
```

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add linac_gen/io/tracewin_syntax.py linac_gen/io/tracewin_parser.py \
        tests/io/test_field_map_path_directive.py
git commit -m "feat(parser): FIELD_MAP_PATH directive resolves filenames"
```

---

## Git setup (do before Task 1)

- [ ] **Commit pending cm→m fixes on `master`**

```bash
git add linac_gen/io/field_map_reader.py docs/fieldmap_guide.md \
        tests/io/test_field_map_3d_reader.py \
        tests/io/test_field_map_tracewin_spec.py \
        tests/io/test_tracewin_field_map_dispatch.py
git commit -m "$(cat <<'EOF'
fix(fieldmap): correct units (m) and reshape order per TraceWin manual

- 1-D canonical .edz: Zmax was read as cm; manual says metres
- 2-D canonical .edz: reshape was (Nr+1, Nz+1) z-fastest; manual loop
  is z-outer/r-inner so natural C-order gives (Nz+1, Nr+1) r-fastest
- 3-D canonical header: was nx/ny/nz; manual has nz/nx/ny with only Zmax
  for z (zmin implicit 0)
- 3-D canonical values: reshape was (nx, ny, nz) z-fastest; manual loop
  is z-outer/y-middle/x-inner so reshape (nz, ny, nx) then transpose
- Updated tests and fieldmap_guide.md accordingly

This is a checkpoint before the full geom-decoder rewrite.
EOF
)"
```

- [ ] **Create feature branch**

```bash
git checkout -b feat/tracewin-geom
```

- [ ] **Commit this plan**

```bash
git add docs/superpowers/plans/2026-04-19-tracewin-geom.md
git commit -m "docs: plan — TraceWin geom decoder full rewrite"
```

---

## Task 1 — `GeomCode` + `decode_geom`

**Files:**
- Create: `linac_gen/io/tracewin_geom.py`
- Create: `tests/io/test_tracewin_geom.py`

- [ ] **Step 1: Write the failing test**

`tests/io/test_tracewin_geom.py`:

```python
"""Unit tests for the TraceWin ``geom`` 5-digit decoder.

Manual reference (lines 18052-18113): geom = aper·10⁴ + rf_B·10³ + rf_E·10²
+ stat_B·10 + stat_E, with per-digit geometry codes 0..9.  Negative geom
means "use 2nd-order off-axis expansion".
"""
from linac_gen.io.tracewin_geom import decode_geom, GeomCode


def test_geom_zero_is_all_zeros():
    g = decode_geom(0)
    assert g == GeomCode(stat_E=0, stat_B=0, rf_E=0, rf_B=0, aper=0,
                         second_order=False)


def test_geom_70_is_3d_static_magnetic():
    """Manual example: FIELD_MAP 70 … qpole — quadrupole/solenoid."""
    g = decode_geom(70)
    assert g.stat_B == 7
    assert g.stat_E == 0 and g.rf_E == 0 and g.rf_B == 0 and g.aper == 0


def test_geom_0070_parsed_same_as_70():
    assert decode_geom(70) == decode_geom(0o070) == decode_geom(int("0070"))


def test_geom_7700_is_3d_rf_both():
    """Manual example: FIELD_MAP 7700 … carte_3gap_2b — 3D RF cavity."""
    g = decode_geom(7700)
    assert g.rf_B == 7 and g.rf_E == 7
    assert g.stat_B == 0 and g.stat_E == 0


def test_geom_100_is_1d_rf_electric():
    """Most common RF cavity: 1D Ez(z)."""
    g = decode_geom(100)
    assert g.rf_E == 1
    assert g.stat_E == 0 and g.stat_B == 0 and g.rf_B == 0


def test_geom_400_is_2d_cyl_rf_electric_TM():
    g = decode_geom(400)
    assert g.rf_E == 4


def test_geom_90_is_1d_quad_gradient():
    g = decode_geom(90)
    assert g.stat_B == 9


def test_negative_geom_sets_second_order_flag():
    g_pos = decode_geom(70)
    g_neg = decode_geom(-70)
    assert g_neg.second_order is True
    assert g_pos.second_order is False
    assert g_neg.stat_B == g_pos.stat_B == 7


def test_full_house():
    """aper=1, rf_B=7, rf_E=7, stat_B=7, stat_E=1: 17771."""
    g = decode_geom(17771)
    assert g.aper == 1 and g.rf_B == 7 and g.rf_E == 7
    assert g.stat_B == 7 and g.stat_E == 1
```

- [ ] **Step 2: Run tests to see them fail**

```bash
python3 -m pytest tests/io/test_tracewin_geom.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'linac_gen.io.tracewin_geom'`.

- [ ] **Step 3: Create the module**

`linac_gen/io/tracewin_geom.py`:

```python
"""TraceWin ``geom`` 5-digit decoder and channel/file mapping.

See the TraceWin user manual, section *FIELD_MAP*:

    geom = aper·10⁴ + rf_B·10³ + rf_E·10² + stat_B·10 + stat_E

where every digit 0..9 describes the *geometry* of the corresponding
field channel (0=absent, 1=1D, 4=2D cyl E-type, 5=2D cyl B-type,
6=2D Cart, 7=3D Cart, 8=3D cyl (N/A), 9=1D G(z) quad gradient).

A negative ``geom`` means "do the off-axis expansion at 2nd order
instead of 1st".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeomCode:
    """Decomposed TraceWin ``geom`` parameter.

    Each field is the single-digit geometry code of one channel.
    ``second_order`` captures the sign of the original geom.
    """
    stat_E: int
    stat_B: int
    rf_E:   int
    rf_B:   int
    aper:   int
    second_order: bool


def decode_geom(geom: int) -> GeomCode:
    """Decode the FIELD_MAP ``geom`` integer into a :class:`GeomCode`."""
    second_order = geom < 0
    g = abs(int(geom))
    return GeomCode(
        stat_E=g % 10,
        stat_B=(g // 10) % 10,
        rf_E=(g // 100) % 10,
        rf_B=(g // 1000) % 10,
        aper=(g // 10000) % 10,
        second_order=second_order,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/io/test_tracewin_geom.py -v
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/io/tracewin_geom.py tests/io/test_tracewin_geom.py
git commit -m "feat(tracewin): decode_geom — 5-digit geom encoding"
```

---

## Task 2 — Channel enum + `component_files` mapping

**Files:**
- Modify: `linac_gen/io/tracewin_geom.py`
- Modify: `tests/io/test_tracewin_geom.py`

- [ ] **Step 1: Extend test file with failing tests**

Append to `tests/io/test_tracewin_geom.py`:

```python
import pytest
from linac_gen.io.tracewin_geom import (
    Channel, component_files, enabled_channels,
)


class TestChannel:
    def test_kind_letters(self):
        assert Channel.STAT_E.field_letter == "e"
        assert Channel.STAT_E.type_letter  == "s"
        assert Channel.STAT_B.field_letter == "b"
        assert Channel.STAT_B.type_letter  == "s"
        assert Channel.RF_E.field_letter   == "e"
        assert Channel.RF_E.type_letter    == "d"
        assert Channel.RF_B.field_letter   == "b"
        assert Channel.RF_B.type_letter    == "d"


class TestComponentFiles:
    def test_1d_stat_E(self):
        assert component_files(Channel.STAT_E, digit=1) == [".esz"]

    def test_1d_stat_B(self):
        assert component_files(Channel.STAT_B, digit=1) == [".bsz"]

    def test_1d_rf_E(self):
        assert component_files(Channel.RF_E, digit=1) == [".edz"]

    def test_1d_rf_B(self):
        assert component_files(Channel.RF_B, digit=1) == [".bdz"]

    def test_2d_cyl_E_type_static(self):
        # digit=4 in stat_E slot: static electric 2D cyl, Fr + Fz
        assert component_files(Channel.STAT_E, digit=4) == [".esr", ".esz"]

    def test_2d_cyl_E_type_rf_TM(self):
        # digit=4 in rf_E slot: RF TM mode, Fr + Fz + Bθ
        assert component_files(Channel.RF_E, digit=4) == [".edr", ".edz", ".bdq"]

    def test_2d_cyl_B_type_static(self):
        assert component_files(Channel.STAT_B, digit=5) == [".bsr", ".bsz"]

    def test_2d_cyl_B_type_rf_TE(self):
        assert component_files(Channel.RF_B, digit=5) == [".bdr", ".bdz", ".edq"]

    def test_2d_cart_stat_B(self):
        assert component_files(Channel.STAT_B, digit=6) == [".bsx", ".bsy"]

    def test_3d_cart_stat_B(self):
        assert component_files(Channel.STAT_B, digit=7) == [
            ".bsx", ".bsy", ".bsz"]

    def test_3d_cart_rf_E(self):
        assert component_files(Channel.RF_E, digit=7) == [
            ".edx", ".edy", ".edz"]

    def test_1d_quad_gradient(self):
        # digit=9 only valid in stat_B
        assert component_files(Channel.STAT_B, digit=9) == [".bsz"]

    def test_digit_9_invalid_in_stat_E(self):
        with pytest.raises(ValueError, match="digit 9.*stat.*magnetic"):
            component_files(Channel.STAT_E, digit=9)

    def test_digit_5_invalid_in_stat_E(self):
        # magnetic-type 2D cyl is nonsensical for an electric channel
        with pytest.raises(ValueError, match="digit 5"):
            component_files(Channel.STAT_E, digit=5)

    def test_digit_4_invalid_in_stat_B(self):
        with pytest.raises(ValueError, match="digit 4"):
            component_files(Channel.STAT_B, digit=4)

    def test_digit_8_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="3-D cyl"):
            component_files(Channel.STAT_B, digit=8)

    def test_digit_0_returns_empty(self):
        assert component_files(Channel.STAT_B, digit=0) == []


class TestEnabledChannels:
    def test_geom_70(self):
        from linac_gen.io.tracewin_geom import decode_geom
        code = decode_geom(70)
        assert enabled_channels(code) == [(Channel.STAT_B, 7)]

    def test_geom_7700(self):
        from linac_gen.io.tracewin_geom import decode_geom
        code = decode_geom(7700)
        assert enabled_channels(code) == [(Channel.RF_E, 7), (Channel.RF_B, 7)]

    def test_geom_90(self):
        from linac_gen.io.tracewin_geom import decode_geom
        code = decode_geom(90)
        assert enabled_channels(code) == [(Channel.STAT_B, 9)]

    def test_geom_0_empty(self):
        from linac_gen.io.tracewin_geom import decode_geom
        assert enabled_channels(decode_geom(0)) == []
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/io/test_tracewin_geom.py -v
```

Expected: ~20 new tests fail with `ImportError` or `AttributeError`.

- [ ] **Step 3: Extend `tracewin_geom.py`**

Append to `linac_gen/io/tracewin_geom.py`:

```python
from enum import Enum
from typing import List, Tuple


class Channel(Enum):
    """One of the four field channels a FIELD_MAP can contain."""
    STAT_E = ("e", "s")
    STAT_B = ("b", "s")
    RF_E   = ("e", "d")
    RF_B   = ("b", "d")

    @property
    def field_letter(self) -> str:
        """``'e'`` for electric channels, ``'b'`` for magnetic."""
        return self.value[0]

    @property
    def type_letter(self) -> str:
        """``'s'`` for static channels, ``'d'`` (dynamic) for RF."""
        return self.value[1]

    @property
    def is_static(self) -> bool:
        return self.type_letter == "s"

    @property
    def is_rf(self) -> bool:
        return self.type_letter == "d"

    @property
    def is_electric(self) -> bool:
        return self.field_letter == "e"

    @property
    def is_magnetic(self) -> bool:
        return self.field_letter == "b"


# --- (channel, digit) → file extensions (per TraceWin manual §18222-18405)
def component_files(channel: Channel, digit: int) -> List[str]:
    """Return the file-extension list (including leading dot) for one
    (channel, geometry-digit) pair.

    Raises
    ------
    ValueError
        If the digit is not physically meaningful in this channel (e.g.
        magnetic-type 2-D cyl in an electric channel, or 1-D G(z)
        anywhere other than the static-magnetic slot).
    NotImplementedError
        For digit 8 (3-D cylindrical), flagged by the TraceWin manual
        itself as "not implemented yet".
    """
    if digit == 0:
        return []
    fl, tl = channel.field_letter, channel.type_letter

    if digit == 1:                                # 1-D Fz(z)
        return [f".{fl}{tl}z"]
    if digit == 4:                                # 2-D cyl E-type
        if not channel.is_electric:
            raise ValueError(
                f"digit 4 (2-D cyl E-type) not valid in {channel.name} "
                f"(only stat_E / rf_E)"
            )
        base = [f".{fl}{tl}r", f".{fl}{tl}z"]
        if channel.is_rf:                         # TM mode: include Bθ
            base.append(".bdq")
        return base
    if digit == 5:                                # 2-D cyl B-type
        if not channel.is_magnetic:
            raise ValueError(
                f"digit 5 (2-D cyl B-type) not valid in {channel.name} "
                f"(only stat_B / rf_B)"
            )
        base = [f".{fl}{tl}r", f".{fl}{tl}z"]
        if channel.is_rf:                         # TE mode: include Eθ
            base.append(".edq")
        return base
    if digit == 6:                                # 2-D Cart
        return [f".{fl}{tl}x", f".{fl}{tl}y"]
    if digit == 7:                                # 3-D Cart
        return [f".{fl}{tl}x", f".{fl}{tl}y", f".{fl}{tl}z"]
    if digit == 8:
        raise NotImplementedError(
            "digit 8 (3-D cyl) is marked 'not implemented yet' in the "
            "TraceWin manual and is not supported."
        )
    if digit == 9:                                # 1-D G(z) quad grad
        if channel is not Channel.STAT_B:
            raise ValueError(
                f"digit 9 (1-D quad G(z)) only valid in stat_B channel, "
                f"got {channel.name}"
            )
        return [f".{fl}{tl}z"]
    raise ValueError(f"unknown geometry digit: {digit}")


def enabled_channels(code: "GeomCode") -> List[Tuple[Channel, int]]:
    """Return the (channel, digit) pairs that actually have a field.

    Order is fixed: STAT_E, STAT_B, RF_E, RF_B.  Aperture digit is
    handled separately by the parser (opens ``.ouv``, no channel).
    """
    out: List[Tuple[Channel, int]] = []
    for ch, d in ((Channel.STAT_E, code.stat_E),
                  (Channel.STAT_B, code.stat_B),
                  (Channel.RF_E,   code.rf_E),
                  (Channel.RF_B,   code.rf_B)):
        if d != 0:
            out.append((ch, d))
    return out
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/io/test_tracewin_geom.py -v
```

Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/io/tracewin_geom.py tests/io/test_tracewin_geom.py
git commit -m "feat(tracewin): Channel enum + component_files mapping for every geom digit"
```

---

## Task 3 — Refactor `FieldMapData` to multi-channel

**Why:** a single FIELD_MAP card can enable up to four channels (e.g. geom=7700 has rf_E + rf_B).  The current `FieldMapData` has flat fields (`Ex`, `Bx`, …) that cannot distinguish static vs RF.  We replace them with a `dict[Channel, FieldChannel]`.

**Files:**
- Modify: `linac_gen/io/field_map_data.py`
- Modify: callers that read old flat fields — enumerated in Task 3b/3c

- [ ] **Step 1: Write failing tests for the new shape**

Create `tests/io/test_field_map_data.py`:

```python
"""Tests for the multi-channel FieldMapData structure."""
import numpy as np
import pytest

from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _dummy_channel_3d():
    return FieldChannel(
        geometry=7,
        z=np.linspace(0.0, 100.0, 11),
        x=np.linspace(-10.0, 10.0, 5),
        y=np.linspace(-10.0, 10.0, 5),
        Fx=np.zeros((5, 5, 11)),
        Fy=np.zeros((5, 5, 11)),
        Fz=np.zeros((5, 5, 11)),
    )


def test_fieldmapdata_default_no_channels():
    fd = FieldMapData(z=np.linspace(0, 10, 11))
    assert fd.channels == {}


def test_add_channel_indexable():
    fd = FieldMapData(z=np.linspace(0, 100, 11))
    ch = _dummy_channel_3d()
    fd.channels[Channel.STAT_B] = ch
    assert Channel.STAT_B in fd.channels
    assert fd.channels[Channel.STAT_B] is ch


def test_fieldchannel_requires_geometry_and_z():
    with pytest.raises(TypeError):
        FieldChannel()           # type: ignore[call-arg]


def test_has_static_has_rf():
    fd = FieldMapData(z=np.linspace(0, 100, 11))
    fd.channels[Channel.STAT_B] = _dummy_channel_3d()
    assert fd.has_static() is True
    assert fd.has_rf() is False
    # Add an RF channel
    rf = _dummy_channel_3d()
    fd.channels[Channel.RF_E] = rf
    assert fd.has_rf() is True


def test_axis_length_uses_outer_z():
    fd = FieldMapData(z=np.linspace(0.0, 250.0, 26))
    assert fd.axis_length_mm() == pytest.approx(250.0)
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/io/test_field_map_data.py -v
```

Expected: FAIL — `FieldChannel` missing; `channels` attribute missing.

- [ ] **Step 3: Rewrite `field_map_data.py`**

```python
"""Dataclasses for multi-channel field-map data.

A ``FIELD_MAP`` with ``geom`` may enable up to four *channels* —
static electric, static magnetic, RF electric, RF magnetic — each
possibly on its own geometry (1-D, 2-D cyl, 2-D Cart, 3-D Cart).
:class:`FieldMapData` holds one :class:`FieldChannel` per enabled
channel.

Units (enforced by every reader):
  * Positions: **mm** (converted from the manual's metres).
  * Electric-field values: **MV/m**.
  * Magnetic-field values: **T**.
  * Frequency: **MHz**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from linac_gen.io.tracewin_geom import Channel


@dataclass
class FieldChannel:
    """One channel's grid + field components.

    The populated attributes depend on ``geometry``:

    * 1-D (``geometry=1``):            ``z``, ``Fz``.
    * 2-D cyl (``geometry=4`` or 5):   ``z``, ``r``, ``Fz``, ``Fr`` (plus
      ``Fq`` for RF TM/TE modes; see `tracewin_geom.component_files`).
    * 2-D Cart (``geometry=6``):       ``x``, ``y``, ``Fx``, ``Fy``.
    * 3-D Cart (``geometry=7``):       ``x``, ``y``, ``z``, ``Fx``,
      ``Fy``, ``Fz``.
    * 1-D G(z) (``geometry=9``):       ``z``, ``Fz`` (gradient in T/m).

    "F" stands for the channel's native field: E (electric) or B
    (magnetic).  Distinguishing E vs B is done by which
    :class:`Channel` this data is attached to inside a
    :class:`FieldMapData`.
    """

    geometry: int                 # digit 1,4,5,6,7,9
    z: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    Fx: Optional[np.ndarray] = None
    Fy: Optional[np.ndarray] = None
    Fz: Optional[np.ndarray] = None
    Fr: Optional[np.ndarray] = None
    Fq: Optional[np.ndarray] = None    # Bθ in TM mode, Eθ in TE mode

    norm_factor: float = 1.0


@dataclass
class FieldMapData:
    """Container of per-channel data for a single FIELD_MAP element."""

    z: np.ndarray                                       # spine axis (mm)
    channels: Dict[Channel, FieldChannel] = field(default_factory=dict)
    frequency: float = 0.0                              # MHz; 0 => static only
    # Aperture-map info, if geom has aper != 0 — reserved for follow-up.
    aperture_file: Optional[str] = None

    # Book-keeping for legacy callers during the transition period.
    # Deprecated; will be removed once all consumers use channels.
    symmetry: str = ""

    # -------------------- convenience ------------------------------
    def axis_length_mm(self) -> float:
        return float(self.z[-1] - self.z[0])

    def has_static(self) -> bool:
        return any(ch.is_static for ch in self.channels.keys())

    def has_rf(self) -> bool:
        return any(ch.is_rf for ch in self.channels.keys())

    def has_electric(self) -> bool:
        return any(ch.is_electric for ch in self.channels.keys())

    def has_magnetic(self) -> bool:
        return any(ch.is_magnetic for ch in self.channels.keys())
```

- [ ] **Step 4: Run the new tests**

```bash
python3 -m pytest tests/io/test_field_map_data.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Run the full io suite to enumerate breakage**

```bash
python3 -m pytest tests/io/ --tb=no -q 2>&1 | tail -40
```

Expected:  Many failures in `test_field_map_reader.py`, `test_field_map_3d_reader.py`, `test_field_map_tracewin_spec.py`, `test_tracewin_field_map_dispatch.py`.  These are addressed in Tasks 4–8.  Do **not** commit yet — FieldMapData rewrite must go in the same commit as at least the first compatibility shim (Task 3b).

---

## Task 3b — Compatibility shim for legacy readers

**Why:** The existing 1-D and 2-D readers return `FieldMapData` with flat `Ez/Er/Bz/Br` fields.  The `FieldMap` element reads those directly.  Until Tasks 4–7 replace the readers, keep a thin shim that materialises a single-channel `FieldMapData` from a flat-style set of arrays, so existing tests keep passing during the rewrite.

**Files:**
- Modify: `linac_gen/io/field_map_data.py`

- [ ] **Step 1: Add a shim to FieldMapData**

Append to `field_map_data.py`:

```python
    @classmethod
    def from_legacy_1d(cls, z: np.ndarray, Ez: np.ndarray,
                       norm_factor: float = 1.0) -> "FieldMapData":
        """Wrap a legacy 1-D (z, Ez) pair in a single RF-E channel.

        Legacy Linac_Gen fixtures (``test_1d.edz``, ``test_cavity_1d.edz``)
        don't declare a geom — they're historically interpreted as RF
        electric cavities.  This constructor is for tests only.
        """
        fd = cls(z=z, frequency=0.0)
        fd.channels[Channel.RF_E] = FieldChannel(
            geometry=1, z=z, Fz=Ez, norm_factor=norm_factor,
        )
        return fd

    @classmethod
    def from_legacy_2d_cyl(cls, z: np.ndarray, r: np.ndarray,
                           Ez: np.ndarray, Er: np.ndarray,
                           Bz: Optional[np.ndarray] = None,
                           Br: Optional[np.ndarray] = None,
                           norm_factor: float = 1.0) -> "FieldMapData":
        """Wrap a legacy 2-D cyl (z, r, Ez, Er[, Bz, Br]) map."""
        fd = cls(z=z, frequency=0.0)
        fd.channels[Channel.RF_E] = FieldChannel(
            geometry=4, z=z, r=r, Fz=Ez, Fr=Er, norm_factor=norm_factor,
        )
        if Bz is not None or Br is not None:
            fd.channels[Channel.RF_B] = FieldChannel(
                geometry=5, z=z, r=r, Fz=Bz, Fr=Br, norm_factor=norm_factor,
            )
        return fd
```

- [ ] **Step 2: Also keep the old flat-field attributes as read-only properties for the grace period**

Inside FieldMapData add:

```python
    # --- Back-compat accessors (DEPRECATED, remove after task 10) --
    @property
    def Ez(self):
        ch = self.channels.get(Channel.RF_E) or self.channels.get(Channel.STAT_E)
        return ch.Fz if ch else None

    @property
    def Er(self):
        ch = self.channels.get(Channel.RF_E) or self.channels.get(Channel.STAT_E)
        return ch.Fr if ch else None

    @property
    def Bz(self):
        ch = self.channels.get(Channel.RF_B) or self.channels.get(Channel.STAT_B)
        return ch.Fz if ch else None

    @property
    def Br(self):
        ch = self.channels.get(Channel.RF_B) or self.channels.get(Channel.STAT_B)
        return ch.Fr if ch else None

    @property
    def Ex(self):
        ch = self.channels.get(Channel.RF_E) or self.channels.get(Channel.STAT_E)
        return ch.Fx if ch else None

    @property
    def Ey(self):
        ch = self.channels.get(Channel.RF_E) or self.channels.get(Channel.STAT_E)
        return ch.Fy if ch else None

    @property
    def Bx(self):
        ch = self.channels.get(Channel.RF_B) or self.channels.get(Channel.STAT_B)
        return ch.Fx if ch else None

    @property
    def By(self):
        ch = self.channels.get(Channel.RF_B) or self.channels.get(Channel.STAT_B)
        return ch.Fy if ch else None

    @property
    def x(self):
        for ch_enum in (Channel.STAT_B, Channel.RF_B, Channel.STAT_E, Channel.RF_E):
            ch = self.channels.get(ch_enum)
            if ch is not None and ch.x is not None:
                return ch.x
        return None

    @property
    def y(self):
        for ch_enum in (Channel.STAT_B, Channel.RF_B, Channel.STAT_E, Channel.RF_E):
            ch = self.channels.get(ch_enum)
            if ch is not None and ch.y is not None:
                return ch.y
        return None

    @property
    def r(self):
        for ch_enum in (Channel.RF_E, Channel.STAT_E, Channel.RF_B, Channel.STAT_B):
            ch = self.channels.get(ch_enum)
            if ch is not None and ch.r is not None:
                return ch.r
        return None
```

- [ ] **Step 3: Update the legacy flat-field readers to build channels internally**

In `linac_gen/io/field_map_reader.py` replace each `return FieldMapData(z=…, Ez=…, …)` with a call to `FieldMapData.from_legacy_*`:

```python
# 1-D canonical (_parse_edz_1d_tracewin) and legacy (_parse_edz_1d):
    return FieldMapData.from_legacy_1d(z=z, Ez=ez, norm_factor=norm)

# 2-D canonical (_parse_edz_2d_tracewin):
    return FieldMapData.from_legacy_2d_cyl(z=z, r=r, Ez=Ez, Er=Er,
                                           Bz=Bz, Br=Br, norm_factor=norm)

# 2-D legacy (_parse_edz_2d_nznr, _parse_edz_2d):
    return FieldMapData.from_legacy_2d_cyl(z=z, r=r, Ez=Ez, Er=Er, Bz=Bz, Br=Br)

# CSV readers:
    # 1-D variant → from_legacy_1d; 2-D variant → from_legacy_2d_cyl
    …
```

- [ ] **Step 4: Run the full io suite; only the canonical/3-D/dispatch tests should still fail**

```bash
python3 -m pytest tests/io/ --tb=short 2>&1 | tail -40
```

Expected: `test_field_map_reader.py` PASSES (89 tests); 3-D reader / dispatch / canonical-spec tests still fail.  Those are Tasks 4+.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/io/field_map_data.py linac_gen/io/field_map_reader.py \
        tests/io/test_field_map_data.py
git commit -m "refactor(fieldmap): FieldMapData → multi-channel container with legacy shims"
```

---

## Task 4 — Per-channel file readers (1-D, 2-D cyl, 2-D Cart, 3-D Cart)

**Goal:** a set of small reader functions, each loads ONE component file (one extension off a prefix) and returns a partially-populated `FieldChannel`.  The orchestrator in Task 5 combines them.

**Files:**
- Create: `linac_gen/io/tracewin_fieldmap_reader.py`
- Create: `tests/io/test_tracewin_fieldmap_reader.py`
- Create: `tests/io/fixtures/tracewin_geom/` — fixture generator helpers

- [ ] **Step 1: Write the fixture generators**

In `tests/io/test_tracewin_fieldmap_reader.py` top-of-file, helpers:

```python
"""Integration tests for the per-channel TraceWin field-map readers."""
from __future__ import annotations
import numpy as np
import pytest
from linac_gen.io.tracewin_fieldmap_reader import (
    read_1d_component, read_2d_cyl_component, read_2d_cart_component,
    read_3d_cart_component,
)


def write_1d(path, Nz, Zmax_m, norm, values):
    """Write a manual-spec 1-D component file (`.esz`/`.bsz`/`.edz`/`.bdz`)."""
    assert len(values) == Nz + 1
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for v in values:
            f.write(f"{v:.6e}\n")


def write_2d_cyl(path, Nz, Zmax_m, Nr, Rmax_m, norm, values):
    """Write a manual-spec 2-D cyl file. values.shape = (Nz+1, Nr+1)."""
    assert values.shape == (Nz + 1, Nr + 1)
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{Nr} {Rmax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for iz in range(Nz + 1):
            for ir in range(Nr + 1):
                f.write(f"{values[iz, ir]:.6e}\n")


def write_2d_cart(path, Nx, Xmin_m, Xmax_m, Ny, Ymin_m, Ymax_m, norm, values):
    """Manual-spec 2-D Cart. values.shape = (Ny+1, Nx+1) — y outer, x inner.

    Manual §18099 says the inner loop index runs over x for 2-D Cart too
    (by analogy with 3-D; the manual paragraph about 2-D Cart has a
    typo but the example in examples/ confirms x-fastest)."""
    assert values.shape == (Ny + 1, Nx + 1)
    with open(path, "w") as f:
        f.write(f"{Nx} {Xmin_m:.6e} {Xmax_m:.6e}\n")
        f.write(f"{Ny} {Ymin_m:.6e} {Ymax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for iy in range(Ny + 1):
            for ix in range(Nx + 1):
                f.write(f"{values[iy, ix]:.6e}\n")


def write_3d_cart(path, Nz, Zmax_m, Nx, Xmin_m, Xmax_m,
                  Ny, Ymin_m, Ymax_m, norm, values):
    """Manual-spec 3-D Cart.  values.shape = (Nz+1, Ny+1, Nx+1)."""
    assert values.shape == (Nz + 1, Ny + 1, Nx + 1)
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{Nx} {Xmin_m:.6e} {Xmax_m:.6e}\n")
        f.write(f"{Ny} {Ymin_m:.6e} {Ymax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for iz in range(Nz + 1):
            for iy in range(Ny + 1):
                for ix in range(Nx + 1):
                    f.write(f"{values[iz, iy, ix]:.6e}\n")
```

Then concrete tests:

```python
class TestRead1dComponent:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "test.esz"
        Nz = 20; Zmax_m = 0.1
        vals = np.cos(np.pi * np.linspace(0, 1, Nz + 1)) * 5.0
        write_1d(str(p), Nz, Zmax_m, norm=2.5, values=vals)
        out = read_1d_component(str(p))
        assert out.geometry == 1
        assert out.norm_factor == pytest.approx(2.5)
        np.testing.assert_allclose(out.z, np.linspace(0, 100.0, Nz + 1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out.Fz, vals, rtol=1e-6)


class TestRead2dCylComponent:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "test.edz"
        Nz, Nr = 10, 4
        Zmax_m, Rmax_m = 0.05, 0.02
        Ez = np.broadcast_to(
            np.cos(np.pi * np.linspace(0, 1, Nz + 1))[:, None],
            (Nz + 1, Nr + 1),
        ).copy() * 1.5
        write_2d_cyl(str(p), Nz, Zmax_m, Nr, Rmax_m, norm=1.0, values=Ez)
        out = read_2d_cyl_component(str(p))
        assert out.geometry == 4 or out.geometry == 5      # decided by caller
        assert out.Fz.shape == (Nz + 1, Nr + 1)
        np.testing.assert_allclose(out.z, np.linspace(0, 50.0, Nz + 1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out.r, np.linspace(0, 20.0, Nr + 1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out.Fz, Ez, rtol=1e-6)


class TestRead2dCartComponent:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "test.esx"
        Nx, Ny = 5, 4
        vals = np.arange((Ny + 1) * (Nx + 1),
                         dtype=float).reshape((Ny + 1, Nx + 1))
        write_2d_cart(str(p), Nx, -0.01, 0.01, Ny, -0.005, 0.005,
                      norm=1.0, values=vals)
        out = read_2d_cart_component(str(p))
        assert out.geometry == 6
        assert out.Fx is None           # reader leaves component assignment to caller
        # Core grid arrays:
        np.testing.assert_allclose(out.x, np.linspace(-10.0, 10.0, Nx + 1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out.y, np.linspace(-5.0, 5.0, Ny + 1),
                                   rtol=1e-12)


class TestRead3dCartComponent:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "test.bsz"
        Nz, Nx, Ny = 8, 3, 2
        vals = np.arange((Nz + 1) * (Ny + 1) * (Nx + 1),
                         dtype=float).reshape((Nz + 1, Ny + 1, Nx + 1))
        write_3d_cart(str(p), Nz, 0.1, Nx, -0.02, 0.02, Ny, -0.01, 0.01,
                      norm=1.0, values=vals)
        out = read_3d_cart_component(str(p))
        assert out.geometry == 7
        # Axes in mm
        np.testing.assert_allclose(out.x, np.linspace(-20.0, 20.0, Nx + 1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out.y, np.linspace(-10.0, 10.0, Ny + 1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out.z, np.linspace(0, 100.0, Nz + 1),
                                   rtol=1e-12)
        # Data transposed from (nz, ny, nx) to (nx, ny, nz)
        # so raw[:, :, :] values should map to ret.values transposed accordingly.
```

- [ ] **Step 2: Run and watch them fail (module missing)**

```bash
python3 -m pytest tests/io/test_tracewin_fieldmap_reader.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `tracewin_fieldmap_reader.py`**

```python
"""TraceWin per-component field-map readers.

Each reader opens ONE file off a path and returns a
:class:`FieldChannel` with the grid populated and **one** of the raw
component arrays attached to ``Fz`` (1-D / 2-D cyl / 3-D Cart, when the
file suffix is ``z``) or left to the caller to assign when the suffix
is ``x/y/r/q`` — the caller knows which field component the file
represents (``.edx`` → Ex, ``.bsy`` → By, ``.edr`` → Er, etc.).

Per the TraceWin manual (§ Dimension 1/2/3):

* All header dimensions are in **metres**.  The reader returns grid
  arrays in **mm**.
* Field values are kept verbatim (MV/m or T).
* Loop orders:
    1-D     →  no loops, Nz+1 values
    2-D cyl →  outer z, inner r (r is fastest)
    2-D Cart→  outer y, inner x (x is fastest)
    3-D Cart→  outer z, middle y, inner x (x is fastest)
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from linac_gen.io.field_map_data import FieldChannel


# --- helpers -----------------------------------------------------------------

def _clean_lines(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Field-map file not found: {filepath}")
    with open(filepath, "r") as fh:
        return [ln.strip() for ln in fh
                if ln.strip() and not ln.lstrip().startswith(("#", "!"))]


def _read_flat(lines: List[str], offset: int, count: int) -> np.ndarray:
    flat: List[float] = []
    for ln in lines[offset:]:
        flat.extend(float(v) for v in ln.split())
        if len(flat) >= count:
            break
    if len(flat) < count:
        raise ValueError(f"expected {count} values, got {len(flat)}")
    return np.asarray(flat[:count], dtype=float)


# --- 1-D ---------------------------------------------------------------------

def read_1d_component(filepath: str) -> FieldChannel:
    """Read a 1-D field-map component file (``.esz``, ``.bsz``, ``.edz``,
    or ``.bdz``).

    Per manual Dimension 1:
        Nz Zmax
        Norm
        (Nz+1) × Fz(k · Zmax / Nz)
    """
    raw = _clean_lines(filepath)
    tok = raw[0].split()
    Nz = int(tok[0]);  Zmax_m = float(tok[1])
    norm = float(raw[1].split()[0])
    vals = _read_flat(raw[2:], 0, Nz + 1)
    z = np.linspace(0.0, Zmax_m * 1000.0, Nz + 1)
    return FieldChannel(geometry=1, z=z, Fz=vals, norm_factor=norm)


# --- 2-D cylindrical ---------------------------------------------------------

def read_2d_cyl_component(filepath: str) -> FieldChannel:
    """Read a 2-D cyl component file (.esr/.esz/.bsr/.bsz/.edr/.edz/.bdr/.bdz/.bdq/.edq).

    Per manual Dimension 2 cyl:
        Nz Zmax
        Nr Rmax
        Norm
        for k=0..Nz:
            for i=0..Nr:  Fz(k·Zmax/Nz, i·Rmax/Nr)

    Inner loop on r ⇒ natural C-order reshape is (Nz+1, Nr+1), r-fastest.
    The returned channel's ``geometry`` is provisional (4 or 5) — the
    caller sets the correct one based on the channel.  The data array is
    attached to ``Fz``; the caller re-attaches to ``Fr`` / ``Fq`` if the
    file suffix was ``r`` / ``q``.
    """
    raw = _clean_lines(filepath)
    tok0 = raw[0].split();  Nz = int(tok0[0]);  Zmax_m = float(tok0[1])
    tok1 = raw[1].split();  Nr = int(tok1[0]);  Rmax_m = float(tok1[1])
    norm = float(raw[2].split()[0])
    vals = _read_flat(raw[3:], 0, (Nz + 1) * (Nr + 1))
    arr = vals.reshape((Nz + 1, Nr + 1))    # r-fastest
    z = np.linspace(0.0, Zmax_m * 1000.0, Nz + 1)
    r = np.linspace(0.0, Rmax_m * 1000.0, Nr + 1)
    return FieldChannel(geometry=4, z=z, r=r, Fz=arr, norm_factor=norm)


# --- 2-D Cartesian -----------------------------------------------------------

def read_2d_cart_component(filepath: str) -> FieldChannel:
    """Read a 2-D Cart component file (.esx/.esy/.bsx/.bsy/.edx/.edy/.bdx/.bdy).

    Manual text for 2-D Cart has a typo in the loop expression, but the
    shape is ``(Ny+1) x (Nx+1)`` with x-fastest by convention.
    """
    raw = _clean_lines(filepath)
    tx = raw[0].split();  Nx = int(tx[0]);  Xmin_m = float(tx[1]);  Xmax_m = float(tx[2])
    ty = raw[1].split();  Ny = int(ty[0]);  Ymin_m = float(ty[1]);  Ymax_m = float(ty[2])
    norm = float(raw[2].split()[0])
    vals = _read_flat(raw[3:], 0, (Nx + 1) * (Ny + 1))
    arr = vals.reshape((Ny + 1, Nx + 1))   # x-fastest
    # Expose as (nx, ny) so downstream RegularGridInterpolator with axes=(x, y)
    # matches naturally:
    arr = arr.T.copy()    # → shape (Nx+1, Ny+1)
    x = np.linspace(Xmin_m, Xmax_m, Nx + 1) * 1000.0
    y = np.linspace(Ymin_m, Ymax_m, Ny + 1) * 1000.0
    return FieldChannel(geometry=6, x=x, y=y, z=None,
                        Fz=arr, norm_factor=norm)
    # Note: the 2-D Cart file has no z dim — Fz in this channel is
    # conceptually F(x,y) at the component named by the file suffix.
    # The orchestrator reassigns to Fx / Fy as appropriate.


# --- 3-D Cartesian -----------------------------------------------------------

def read_3d_cart_component(filepath: str) -> FieldChannel:
    """Read a 3-D Cart component file (.esx/.esy/.esz/.bsx/.bsy/.bsz/.edx/.edy/.edz/.bdx/.bdy/.bdz).

    Per manual Dimension 3:
        Nz Zmax
        Nx Xmin Xmax
        Ny Ymin Ymax
        Norm
        for k=0..Nz:
            for j=0..Ny:
                for i=0..Nx: Fz(k·Zmax/Nz, Ymin+j·ΔY, Xmin+i·ΔX)

    Inner loop on x ⇒ C-order reshape is (Nz+1, Ny+1, Nx+1) x-fastest;
    we transpose to (Nx+1, Ny+1, Nz+1) so the caller can feed the array
    directly into a ``RegularGridInterpolator`` with axes=(x, y, z).
    """
    raw = _clean_lines(filepath)
    tz = raw[0].split();  Nz = int(tz[0]);  Zmax_m = float(tz[1])
    tx = raw[1].split();  Nx = int(tx[0]);  Xmin_m = float(tx[1]);  Xmax_m = float(tx[2])
    ty = raw[2].split();  Ny = int(ty[0]);  Ymin_m = float(ty[1]);  Ymax_m = float(ty[2])
    norm = float(raw[3].split()[0])
    total = (Nz + 1) * (Ny + 1) * (Nx + 1)
    vals = _read_flat(raw[4:], 0, total)
    arr = vals.reshape((Nz + 1, Ny + 1, Nx + 1)).transpose(2, 1, 0).copy()
    x = np.linspace(Xmin_m, Xmax_m, Nx + 1) * 1000.0
    y = np.linspace(Ymin_m, Ymax_m, Ny + 1) * 1000.0
    z = np.linspace(0.0,    Zmax_m, Nz + 1) * 1000.0
    return FieldChannel(geometry=7, x=x, y=y, z=z,
                        Fz=arr, norm_factor=norm)
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/io/test_tracewin_fieldmap_reader.py -v
```

Expected: 4 PASS (1-D, 2-D cyl, 2-D Cart, 3-D Cart round-trip).

- [ ] **Step 5: Commit**

```bash
git add linac_gen/io/tracewin_fieldmap_reader.py \
        tests/io/test_tracewin_fieldmap_reader.py
git commit -m "feat(tracewin): per-component readers (1D/2D cyl/2D Cart/3D Cart)"
```

---

## Task 5 — Top-level orchestrator `read_tracewin_fieldmap`

**Goal:** decode the geom, enumerate enabled channels, open each component file, build a `FieldMapData` with every channel populated, including the right E↔Fx/Fy/Fz/Fr/Fq assignment based on the file suffix.

**Files:**
- Modify: `linac_gen/io/tracewin_fieldmap_reader.py`
- Modify: `tests/io/test_tracewin_fieldmap_reader.py`

- [ ] **Step 1: Write failing tests for the top-level reader**

Append to `tests/io/test_tracewin_fieldmap_reader.py`:

```python
from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
from linac_gen.io.tracewin_geom import Channel


def _make_3d_stat_B(tmp_path, prefix="sol", norm=1.0):
    Nz, Nx, Ny = 5, 2, 2
    zeros = np.zeros((Nz + 1, Ny + 1, Nx + 1))
    for suf in (".bsx", ".bsy", ".bsz"):
        write_3d_cart(str(tmp_path / f"{prefix}{suf}"),
                      Nz, 0.1, Nx, -0.01, 0.01, Ny, -0.01, 0.01,
                      norm=norm, values=zeros)
    return str(tmp_path / prefix)


def test_geom70_loads_single_stat_B_channel(tmp_path):
    prefix = _make_3d_stat_B(tmp_path, norm=2.0)
    fd = read_tracewin_fieldmap(geom=70, prefix=prefix)
    assert Channel.STAT_B in fd.channels
    assert Channel.RF_E not in fd.channels
    ch = fd.channels[Channel.STAT_B]
    assert ch.geometry == 7
    assert ch.Fx is not None and ch.Fy is not None and ch.Fz is not None
    assert ch.norm_factor == pytest.approx(2.0)


def test_geom7700_loads_rf_E_and_rf_B_channels(tmp_path):
    Nz, Nx, Ny = 3, 2, 2
    vals = np.ones((Nz + 1, Ny + 1, Nx + 1))
    prefix = str(tmp_path / "cav")
    for suf in (".edx", ".edy", ".edz", ".bdx", ".bdy", ".bdz"):
        write_3d_cart(f"{prefix}{suf}", Nz, 0.1,
                      Nx, -0.01, 0.01, Ny, -0.01, 0.01,
                      norm=1.0, values=vals)
    fd = read_tracewin_fieldmap(geom=7700, prefix=prefix)
    assert set(fd.channels) == {Channel.RF_E, Channel.RF_B}
    for ch in fd.channels.values():
        assert ch.Fx is not None and ch.Fy is not None and ch.Fz is not None


def test_geom100_loads_1d_rf_E_channel(tmp_path):
    """1-D RF cavity."""
    prefix = str(tmp_path / "cav")
    write_1d(f"{prefix}.edz", Nz=10, Zmax_m=0.1, norm=1.0,
             values=np.cos(np.linspace(0, np.pi, 11)))
    fd = read_tracewin_fieldmap(geom=100, prefix=prefix)
    assert list(fd.channels) == [Channel.RF_E]
    ch = fd.channels[Channel.RF_E]
    assert ch.geometry == 1 and ch.Fz.shape == (11,)


def test_geom400_loads_2d_cyl_rf_TM_with_Btheta(tmp_path):
    prefix = str(tmp_path / "tm")
    Nz, Nr = 8, 3
    for suf in (".edz", ".edr", ".bdq"):
        write_2d_cyl(f"{prefix}{suf}", Nz, 0.1, Nr, 0.02, norm=1.0,
                     values=np.zeros((Nz + 1, Nr + 1)))
    fd = read_tracewin_fieldmap(geom=400, prefix=prefix)
    ch = fd.channels[Channel.RF_E]
    assert ch.geometry == 4
    assert ch.Fz is not None and ch.Fr is not None and ch.Fq is not None


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"\.bsz"):
        read_tracewin_fieldmap(geom=70, prefix=str(tmp_path / "missing"))
```

- [ ] **Step 2: Run and see them fail**

```bash
python3 -m pytest tests/io/test_tracewin_fieldmap_reader.py -v
```

- [ ] **Step 3: Implement orchestrator**

Append to `linac_gen/io/tracewin_fieldmap_reader.py`:

```python
from typing import Optional

from linac_gen.io.field_map_data import FieldMapData
from linac_gen.io.tracewin_geom import (
    Channel, GeomCode, decode_geom, component_files, enabled_channels,
)


def _dispatch_reader(digit: int):
    """Pick the per-component reader for a given geometry digit."""
    if digit == 1 or digit == 9:
        return read_1d_component
    if digit in (4, 5):
        return read_2d_cyl_component
    if digit == 6:
        return read_2d_cart_component
    if digit == 7:
        return read_3d_cart_component
    raise ValueError(f"no reader for digit {digit}")


def _attach_to_channel(ch: FieldChannel, suffix: str, raw: FieldChannel) -> None:
    """Reassign the single raw-file array to the correct F{x|y|z|r|q}.

    The per-component reader puts its data in ``raw.Fz`` regardless of
    the file's component letter; this function moves it to the slot that
    matches the 3rd character of the suffix.
    """
    letter = suffix[-1]                 # one of x, y, z, r, q
    if letter == "z":
        ch.Fz = raw.Fz
        if ch.z is None:
            ch.z = raw.z
        if raw.r is not None and ch.r is None:
            ch.r = raw.r
    elif letter == "r":
        ch.Fr = raw.Fz
        if raw.r is not None and ch.r is None:
            ch.r = raw.r
        if ch.z is None:
            ch.z = raw.z
    elif letter == "x":
        ch.Fx = raw.Fz
        if raw.x is not None and ch.x is None:
            ch.x = raw.x
        if raw.y is not None and ch.y is None:
            ch.y = raw.y
        if raw.z is not None and ch.z is None:
            ch.z = raw.z
    elif letter == "y":
        ch.Fy = raw.Fz
        if raw.x is not None and ch.x is None:
            ch.x = raw.x
        if raw.y is not None and ch.y is None:
            ch.y = raw.y
        if raw.z is not None and ch.z is None:
            ch.z = raw.z
    elif letter == "q":
        ch.Fq = raw.Fz
        if raw.r is not None and ch.r is None:
            ch.r = raw.r
        if ch.z is None:
            ch.z = raw.z
    else:
        raise ValueError(f"unrecognised suffix letter: {letter!r}")


def _strip_known_suffix(prefix: str) -> str:
    """Remove any trailing .e{s,d}{x,y,z,r,q} or .b{s,d}{x,y,z,r,q}/.ouv."""
    lower = prefix.lower()
    for fl in ("e", "b"):
        for tl in ("s", "d"):
            for co in ("x", "y", "z", "r", "q"):
                suf = f".{fl}{tl}{co}"
                if lower.endswith(suf):
                    return prefix[: -len(suf)]
    if lower.endswith(".ouv"):
        return prefix[:-4]
    return prefix


def read_tracewin_fieldmap(geom: int, prefix: str,
                           base_dir: Optional[str] = None,
                           frequency: float = 0.0) -> FieldMapData:
    """Load every field file implied by ``geom`` off ``prefix``.

    Parameters
    ----------
    geom : int
        TraceWin ``geom`` code (5-digit encoded).
    prefix : str
        File-name prefix *without* extension.  If the caller accidentally
        included a recognised extension, it is stripped.
    base_dir : str, optional
        If given, ``prefix`` is treated as relative to this directory
        unless it is absolute.
    frequency : float
        RF frequency in MHz (from the enclosing ``FREQ`` directive).
        Stored on the returned :class:`FieldMapData` so the element can
        compute the phasor at track time.

    Returns
    -------
    FieldMapData
        ``channels`` populated for every enabled non-zero digit.
    """
    code: GeomCode = decode_geom(geom)
    stripped = _strip_known_suffix(prefix)
    full_prefix = (
        stripped if os.path.isabs(stripped) or base_dir is None
        else os.path.join(base_dir, stripped)
    )

    # Build data.z from whichever channel provides one (usually the
    # longest / most-sampled); for now pick the first channel's z.
    data = FieldMapData(z=np.asarray([]), frequency=frequency)

    for channel, digit in enabled_channels(code):
        reader = _dispatch_reader(digit)
        ch = FieldChannel(geometry=digit)
        for suf in component_files(channel, digit):
            raw = reader(full_prefix + suf)
            _attach_to_channel(ch, suf, raw)
        if ch.z is not None and len(data.z) == 0:
            data.z = ch.z
        data.channels[channel] = ch

    # Aperture file (digit > 0 in 10⁴ slot) — open and record path only.
    if code.aper != 0:
        ouv = full_prefix + ".ouv"
        if os.path.exists(ouv):
            data.aperture_file = ouv

    return data
```

- [ ] **Step 4: Run**

```bash
python3 -m pytest tests/io/test_tracewin_fieldmap_reader.py -v
```

Expected: 9 PASS (4 per-component + 5 orchestrator).

- [ ] **Step 5: Commit**

```bash
git add linac_gen/io/tracewin_fieldmap_reader.py \
        tests/io/test_tracewin_fieldmap_reader.py
git commit -m "feat(tracewin): read_tracewin_fieldmap top-level orchestrator"
```

---

## Task 2c — `Ki` > 0 loads `.scc` space-charge compensation map

**Why:** Manual §18188 — if `Ki > 0` the parser must load `FileName.scc` as a current / SC-compensation profile.  We store it on `FieldMapData.scc_profile` so downstream SC tracking can apply it.  Actual use is a separate plan; this task just closes the parsing loop so files with `Ki≠0` don't silently drop data.

**Files:**
- Modify: `linac_gen/io/field_map_data.py` — add `scc_profile: np.ndarray | None`
- Modify: `linac_gen/io/tracewin_fieldmap_reader.py` — `read_tracewin_fieldmap` accepts `Ki` and opens `.scc` when non-zero
- Create: `tests/io/fixtures/tracewin_geom/test.scc` (minimal 2-column z/current)
- Create: `tests/io/test_scc_loading.py`

- [ ] **Step 1: Failing test**

```python
def test_scc_loaded_when_Ki_nonzero(tmp_path):
    # ... write a minimal cavity + scc fixture
    fd = read_tracewin_fieldmap(geom=100, prefix=str(tmp_path/"cav"),
                                Ki=0.5)
    assert fd.scc_profile is not None
    assert fd.scc_profile.shape[1] == 2   # z, current
    assert fd.scc_scale == 0.5


def test_scc_skipped_when_Ki_zero(tmp_path):
    fd = read_tracewin_fieldmap(geom=100, prefix=str(tmp_path/"cav"),
                                Ki=0.0)
    assert fd.scc_profile is None
```

- [ ] **Step 2: Run; expect fail**
- [ ] **Step 3: Implement**
  - Add `scc_profile`, `scc_scale` fields to `FieldMapData`.
  - In `read_tracewin_fieldmap`, after channel loading: `if Ki != 0.0: fd.scc_profile = np.loadtxt(prefix + ".scc"); fd.scc_scale = Ki`.
- [ ] **Step 4: Run tests green**
- [ ] **Step 5: Commit** — `feat(tracewin): load .scc when Ki > 0`.

---

## Task 2d — `Ka` aperture flag (0 / 1 / 2)

**Why:** Manual §18198:
- Ka=0 → particle lost when it leaves R or the field-map frame (existing behavior for any element)
- Ka=1 → read `FileName.ouv` as a pipe-radius map (z, r) and apply it instead of the scalar R
- Ka=2 → R is ignored; particle kept even if it leaves the field-map frame (needed for overlapping maps)

**Files:**
- Modify: `linac_gen/elements/base.py` — `FieldMapElement` gains `ka: int` and `pipe_radius_profile: tuple[np.ndarray, np.ndarray] | None`
- Modify: `linac_gen/io/tracewin_fieldmap_reader.py` — opens `.ouv` when Ka=1
- Modify: `linac_gen/elements/field_map.py` / `field_map_3d.py` — loss check uses profile when present; skipped entirely when Ka=2
- Create: `tests/elements/test_field_map_ka_flag.py`

- [ ] **Step 1: Failing tests**

```python
def test_ka0_loses_on_aperture():
    # Default R-based loss unchanged
    ...

def test_ka1_reads_ouv_profile(tmp_path):
    # Build .ouv with z, r(z) and confirm particles lost when |r_particle|>r(z)
    ...

def test_ka2_ignores_aperture(tmp_path):
    # Particles far outside R should survive
    ...
```

- [ ] **Step 2-4:** implement `Ka` plumbing end-to-end; run tests green
- [ ] **Step 5: Commit** — `feat(fieldmap): honor Ka=0/1/2 aperture flag`.

---

## Task 6 — `FieldMap3D` multi-channel tracking

**Goal:** rewrite `FieldMap3D.track_rk4` to iterate channels, apply the correct phasor per channel kind, and apply the full Lorentz force including `v × Bz` rotation (previously omitted).

**Files:**
- Modify: `linac_gen/elements/field_map_3d.py`
- Modify: `tests/elements/test_field_map_3d.py` (loosen assumptions that no longer hold)
- Create: `tests/elements/test_field_map_3d_multichannel.py`

- [ ] **Step 1: Write failing tests**

```python
"""FieldMap3D with explicit channel separation + solenoid v×Bz."""
import numpy as np
import pytest
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _uniform_Bz_field(L_mm=100.0, Bz_T=0.5):
    """Build a FieldMapData with a uniform static Bz = Bz_T over a box."""
    n = 11
    x = np.linspace(-10, 10, 3);  y = np.linspace(-10, 10, 3)
    z = np.linspace(0.0, L_mm, n)
    Bx = np.zeros((3, 3, n));  By = np.zeros((3, 3, n))
    Bz = np.full((3, 3, n), Bz_T)
    fd = FieldMapData(z=z)
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=7, x=x, y=y, z=z,
        Fx=Bx, Fy=By, Fz=Bz,
    )
    return fd


def test_static_solenoid_rotates_xy_plane():
    """A particle entering at (x=0, y=0, x'=1mrad, y'=0) through a
    uniform Bz solenoid should come out rotated by the Larmor angle."""
    fd = _uniform_Bz_field(L_mm=200.0, Bz_T=0.5)
    sol = FieldMap3D(name="SOL", length=200.0, field_data=fd,
                     scale=1.0, phase=0.0, frequency=0.0, n_steps=200)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=1, current=0.0)
    beam.particles[0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]    # x'=1 mrad
    ds = sol.length / sol.n_steps
    sol._step_idx = 0
    for _ in range(sol.n_steps):
        sol.track_rk4(beam, ds)
    # Analytic Larmor half-angle: Φ = qBzL / (2βγmc).  For 5 MeV proton,
    # β≈0.103, γ≈1.0053; expect a finite rotation — here just test that
    # y' is non-zero (previous code kept it at 0).
    assert abs(beam.particles[0, 3]) > 0.1   # mrad


def test_rf_electric_applies_cos_phasor_not_to_static():
    """With a static-B-only map, varying phase must not change the output."""
    fd = _uniform_Bz_field(200.0, 0.3)
    sol0   = FieldMap3D("SOL", 200.0, fd, phase=  0.0, frequency=352.21, n_steps=50)
    sol180 = FieldMap3D("SOL", 200.0, fd, phase=180.0, frequency=352.21, n_steps=50)
    ref0 = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    ref1 = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam0 = Beam(ref=ref0, n_particles=1, current=0.0)
    beam0.particles[0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    beam1 = Beam(ref=ref1, n_particles=1, current=0.0)
    beam1.particles[0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    for b, s in ((beam0, sol0), (beam1, sol180)):
        s._step_idx = 0
        ds = s.length / s.n_steps
        for _ in range(s.n_steps):
            s.track_rk4(b, ds)
    np.testing.assert_allclose(beam0.particles, beam1.particles, rtol=1e-12)
```

- [ ] **Step 2: Run and watch them fail**

Expected: the rotation test currently returns `y'≈0` (old code omits Bz).  Phasor test may pass only if the old code treats phase=180° with `cos(180°)=-1` identically for both (which it doesn't — both should be zero kick because the field is static).

- [ ] **Step 3: Rewrite `track_rk4`**

Replace the `track_rk4` method in `field_map_3d.py`.  Sketch:

```python
def track_rk4(self, beam, ds: float) -> None:
    ref = beam.ref
    z_pos = self._z_map_start + self._step_idx * ds + ds / 2.0
    self._step_idx += 1

    fd = self.field_data
    phi_sync_rad = (self.phase + ref.phi_s) * np.pi / 180.0
    ds_m = ds * 1e-3
    charge = ref.species.charge        # in units of e
    beta, gamma = ref.beta, ref.gamma
    mass_MeV = ref.species.mass
    freq_hz = self.frequency * 1e6

    # ---- Reference energy advance using on-axis Ez from any electric channel
    Ez_axis = self._sample_on_axis_E(z_pos)
    dW_ref = charge * Ez_axis * np.cos(phi_sync_rad) * ds_m       # MeV
    ref.w_kin += dW_ref
    ref.s     += ds
    ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)

    alive = beam.alive_mask
    if not np.any(alive):
        return

    xs = beam.particles[alive, 0]
    ys = beam.particles[alive, 2]
    xps = beam.particles[alive, 1]
    yps = beam.particles[alive, 3]
    dphi_deg = beam.particles[alive, 4]
    zs = np.full_like(xs, z_pos)

    # Accumulators for the total (E, B) at each particle position.
    Ex_tot = np.zeros_like(xs);  Ey_tot = np.zeros_like(xs);  Ez_tot = np.zeros_like(xs)
    Bx_tot = np.zeros_like(xs);  By_tot = np.zeros_like(xs);  Bz_tot = np.zeros_like(xs)

    # ---- Channel iteration
    for channel, ch in fd.channels.items():
        e_or_b = "E" if channel.is_electric else "B"
        scale  = (self.ke if channel.is_electric else self.kb) / ch.norm_factor

        # Phasor: static = 1; rf_E = cos(ωt+φ); rf_B = sin(ωt+φ)
        phi_i = phi_sync_rad + dphi_deg * np.pi / 180.0
        if channel.is_static:
            phasor = np.ones_like(xs)
        elif channel.is_electric:           # rf_E
            phasor = np.cos(phi_i)
        else:                                # rf_B
            phasor = np.sin(phi_i)

        # Sample Fx, Fy, Fz from whatever geometry this channel uses.
        Fx, Fy, Fz = self._sample_channel(ch, xs, ys, zs)

        contribution_x = scale * phasor * Fx
        contribution_y = scale * phasor * Fy
        contribution_z = scale * phasor * Fz

        if e_or_b == "E":
            Ex_tot += contribution_x
            Ey_tot += contribution_y
            Ez_tot += contribution_z
        else:
            Bx_tot += contribution_x
            By_tot += contribution_y
            Bz_tot += contribution_z

    # ---- Longitudinal energy kick (delta from synchronous)
    dW_i = charge * Ez_tot * ds_m           # MeV (Ez in MV/m, ds in m)
    beam.particles[alive, 5] += dW_i - dW_ref

    # ---- Transverse kicks
    # Derivation (SI → engineering units):
    #   Δp_x = F_x · Δt = q (E_x + (v × B)_x) · Δs / v_z
    #   v × B with v = v_z·(x', y', 1):
    #     (v × B)_x = v_z (y' B_z - B_y)
    #     (v × B)_y = v_z (B_x - x' B_z)
    #   Δp_x = q [ E_x / v_z + y' B_z - B_y ] · Δs
    #        = q [ E_x / (β c) + y' B_z - B_y ] · Δs
    #   Δx' = Δp_x / p, with p = γ β m c:
    #
    #     Δx' = q · Δs · E_x / (γ β² m c²) + q · Δs · (y' B_z - B_y) / (γ β m c)
    #
    # In engineering units (q in e, E in MV/m, B in T, m·c² in MeV, Δs in m):
    #   E-kick factor: 1 / (γ β² · mass_MeV)        [rad per (MV/m)]
    #   B-kick factor: c · 1e-6 / (γ β · mass_MeV)   [rad per T]
    #     where c·1e-6 ≈ 299.7924 (m/s × MV/V) so that B[T]·c·1e-6 gives
    #     an effective "MV/m" magnitude comparable to E.
    if beta > 0 and gamma > 0:
        factor_E = ds_m             / (gamma * beta * beta * mass_MeV)  # rad per (MV/m)
        factor_B = ds_m * 299.792458 / (gamma * beta        * mass_MeV)  # rad per T
        xp_rad = xps * 1e-3   # convert stored mrad → rad before mixing with Bz
        yp_rad = yps * 1e-3
        dxp_rad = charge * (factor_E * Ex_tot + factor_B * (yp_rad * Bz_tot - By_tot))
        dyp_rad = charge * (factor_E * Ey_tot + factor_B * (Bx_tot - xp_rad * Bz_tot))
        beam.particles[alive, 1] += dxp_rad * 1e3    # back to mrad
        beam.particles[alive, 3] += dyp_rad * 1e3

    # ---- Drift
    beam.particles[alive, 0] += beam.particles[alive, 1] * ds_m
    beam.particles[alive, 2] += beam.particles[alive, 3] * ds_m
```

Helper methods:

```python
def _sample_on_axis_E(self, z_pos: float) -> float:
    """On-axis Ez contribution (sum of electric channels at x=y=0)."""
    fd = self.field_data
    total = 0.0
    for channel, ch in fd.channels.items():
        if not channel.is_electric:
            continue
        if ch.Fz is None:
            continue
        if ch.geometry == 7:   # 3-D Cart
            from scipy.interpolate import RegularGridInterpolator as RGI
            rgi = RGI((ch.x, ch.y, ch.z), ch.Fz,
                      method="linear", bounds_error=False, fill_value=0.0)
            total += float(rgi(np.array([[0.0, 0.0, z_pos]]))[0])
        elif ch.geometry == 4:
            # Ez(r=0, z) from 2-D cyl
            from scipy.interpolate import RegularGridInterpolator as RGI
            rgi = RGI((ch.z, ch.r), ch.Fz,
                      method="linear", bounds_error=False, fill_value=0.0)
            total += float(rgi(np.array([[z_pos, 0.0]]))[0])
        elif ch.geometry == 1:
            total += float(np.interp(z_pos, ch.z, ch.Fz, left=0.0, right=0.0))
    return total * self.scale

def _sample_channel(self, ch, xs, ys, zs):
    """Return (Fx, Fy, Fz) at each particle position for this channel."""
    # Dispatch on ch.geometry
    …
```

*(The implementer should spell out `_sample_channel` for each geometry:
1-D, 2-D cyl, 2-D Cart, 3-D Cart, 1-D G.  See Task 6b-stubs in the test
file for expected behaviour at geometry=7.)*

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/elements/test_field_map_3d.py \
                   tests/elements/test_field_map_3d_multichannel.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add linac_gen/elements/field_map_3d.py \
        tests/elements/test_field_map_3d_multichannel.py \
        tests/elements/test_field_map_3d.py
git commit -m "feat(fieldmap3d): multi-channel tracking with correct phasors + v×Bz"
```

---

## Task 7 — `FieldMap` (1-D / 2-D cyl) multi-channel tracking

**Goal:** the non-3-D `FieldMap` element iterates over the same channel dict as `FieldMap3D`, but samples 1-D, 2-D cyl, or 1-D G(z) channels.  Phasor and Lorentz rules are identical to Task 6 — only the `_sample_channel` geometry differs.

**Files:**
- Modify: `linac_gen/elements/field_map.py`
- Create: `tests/elements/test_field_map_multichannel.py`

- [ ] **Step 1: Write the failing tests**

`tests/elements/test_field_map_multichannel.py`:

```python
"""Multi-channel tracking for FieldMap (1-D / 2-D cyl)."""
import numpy as np
import pytest
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _fd_1d_rf_cavity(L_mm=100.0, Epeak_MVm=1.0):
    """1-D RF-E cavity: Ez(z) = Epeak · sin(πz/L)."""
    z = np.linspace(0, L_mm, 51)
    Ez = Epeak_MVm * np.sin(np.pi * z / L_mm)
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=Ez)
    return fd


def test_1d_rf_cavity_accelerates_on_crest():
    fd = _fd_1d_rf_cavity(100.0, 1.0)
    fmap = FieldMap(name="CAV", length=100.0, field_data=fd,
                    scale=1.0, ke=1.0, kb=1.0, phase=0.0,
                    frequency=352.21, n_steps=100)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W_in = ref.w_kin
    fmap.advance_ref(ref)
    assert ref.w_kin > W_in, "On-crest RF cavity should accelerate"


def _fd_1d_static_solenoid(L_mm=200.0, Bz_Tesla=0.5):
    z = np.linspace(0, L_mm, 201)
    Bz = np.full_like(z, Bz_Tesla)
    fd = FieldMapData(z=z, frequency=0.0)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=Bz)
    return fd


def test_1d_static_solenoid_rotates_offaxis_particle():
    """1-D on-axis Bz solenoid: Br = -(r/2)·dBz/dz is tiny (uniform core),
    but the (v × Bz) rotation must kick y' for particles with x' != 0."""
    fd = _fd_1d_static_solenoid(200.0, 0.5)
    fmap = FieldMap(name="SOL", length=200.0, field_data=fd,
                    scale=1.0, ke=1.0, kb=1.0, phase=0.0,
                    frequency=0.0, n_steps=200)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=1, current=0.0)
    beam.particles[0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    ds = fmap.length / fmap.n_steps
    fmap._step_idx = 0
    for _ in range(fmap.n_steps):
        fmap.track_rk4(beam, ds)
    assert abs(beam.particles[0, 3]) > 0.05, "Solenoid Bz rotation should kick y'"


def test_rf_static_combined_honours_phasor():
    """Phase matters for RF channels but not for static ones.  Build a
    1-D map with both a static_B component (Bz≠0) and an rf_E component
    (Ez=0).  Changing phase must not change the trajectory, because the
    only active kick is from the static channel."""
    fd = FieldMapData(z=np.linspace(0, 100, 101), frequency=352.21)
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=1, z=fd.z, Fz=np.full_like(fd.z, 0.3),
    )
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=1, z=fd.z, Fz=np.zeros_like(fd.z),
    )
    ref0 = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    ref1 = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    for phase, ref in zip([0.0, 180.0], [ref0, ref1]):
        fmap = FieldMap(name="X", length=100.0, field_data=fd,
                        scale=1.0, ke=1.0, kb=1.0, phase=phase,
                        frequency=352.21, n_steps=50)
        beam = Beam(ref=ref, n_particles=1, current=0.0)
        beam.particles[0] = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        ds = fmap.length / fmap.n_steps
        fmap._step_idx = 0
        for _ in range(fmap.n_steps):
            fmap.track_rk4(beam, ds)
    np.testing.assert_allclose(ref0.w_kin, ref1.w_kin, rtol=1e-12)
```

- [ ] **Step 2: Run tests and verify failure**

`python3 -m pytest tests/elements/test_field_map_multichannel.py -v`

Expected: fail on the rotation test (old code doesn't apply Bz) and the phase test (old code applies cos(φ) to the Bz too).

- [ ] **Step 3: Rewrite `FieldMap.track_rk4`**

Mirror Task 6's per-channel iteration, but sample on the channel's geometry:

```python
def _sample_channel_1d_2dcyl(self, ch, r_arr, z_scalar):
    """Return (Fx_arr, Fy_arr, Fz_arr) at r_arr (mm) and z_scalar (mm).

    Geometry handling:
      * ch.geometry == 1: Fz on axis via np.interp; Fr paraxial via
        -(r/2)·dFz/dz; Fx/Fy via Fr · x/|r| and Fr · y/|r|.
      * ch.geometry == 4 or 5: bilinear sample of Fz(r,z), Fr(r,z);
        Bθ (Fq) if present converts to Fx = -Fq·y/r, Fy = +Fq·x/r.
      * ch.geometry == 9: treated as a quadrupole gradient G(z);
        Fx = G·x, Fy = -G·y (normal quad; sign depends on convention
        — the existing `Quadrupole` element sets this; match it).
    """
    # implementation details spelled out during subagent work
```

Then follow Task 6's structure:

```python
for channel, ch in fd.channels.items():
    scale = (self.ke if channel.is_electric else self.kb) / ch.norm_factor
    phi_i = phi_sync_rad + dphi_deg * np.pi / 180.0
    phasor = (np.ones_like(xs) if channel.is_static
              else np.cos(phi_i) if channel.is_electric
              else np.sin(phi_i))
    Fx, Fy, Fz = self._sample_channel_1d_2dcyl(ch, rs, z_pos)
    if channel.is_electric:
        Ex_tot += scale * phasor * Fx
        Ey_tot += scale * phasor * Fy
        Ez_tot += scale * phasor * Fz
    else:
        Bx_tot += scale * phasor * Fx
        By_tot += scale * phasor * Fy
        Bz_tot += scale * phasor * Fz
```

and the identical Lorentz-kick block as Task 6 Step 3.

- [ ] **Step 4: Run the element tests**

```bash
python3 -m pytest tests/elements/test_field_map_multichannel.py \
                   tests/elements/test_field_map.py \
                   tests/elements/test_field_map_2d_cyl.py -v
```

Expected: PASS.  Adjust legacy `test_field_map.py` / `test_field_map_2d_cyl.py` assertions if they bake in the old incorrect behaviour (but prefer to keep them passing — they encode correct 1-D/2-D physics that the new code must reproduce).

- [ ] **Step 5: Commit**

```bash
git add linac_gen/elements/field_map.py \
        tests/elements/test_field_map_multichannel.py
git commit -m "feat(fieldmap): 1-D/2-D cyl multi-channel tracking with phasors + v×Bz"
```

---

## Task 7b — `p_flag` absolute vs relative phase (FieldMap + FieldMap3D)

**Why:** Manual §18036-18042 — `p_flag=0` means `θᵢ` is the synchronous-relative phase (add to `ref.phi_s`); `p_flag=1` means it's the absolute phase (don't add).  Today's element stores p_flag but never reads it during tracking.

**Files:**
- Modify: `linac_gen/elements/field_map.py` — `phi_sync_rad` branch on `self.p_flag`
- Modify: `linac_gen/elements/field_map_3d.py` — same
- Create: `tests/elements/test_field_map_p_flag.py`

- [ ] **Step 1: Failing test**

```python
"""p_flag=1 means θᵢ is absolute phase — independent of ref.phi_s."""
import numpy as np
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _rf_1d_cav(Ez_peak):
    z = np.linspace(0, 100, 101)
    Ez = np.full_like(z, Ez_peak)    # uniform for a clean test
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=Ez)
    return fd


def _track_energy_gain(p_flag, phase, phi_s_start):
    fd = _rf_1d_cav(1.0)
    fmap = FieldMap(name="C", length=100.0, field_data=fd,
                    scale=1.0, ke=1.0, kb=1.0, phase=phase,
                    frequency=352.21, n_steps=50, p_flag=p_flag)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    ref.phi_s = phi_s_start
    W0 = ref.w_kin
    fmap.advance_ref(ref)
    return ref.w_kin - W0


def test_p_flag_relative_sees_ref_phi_s():
    # p_flag=0: non-zero ref.phi_s changes effective phase
    dW_a = _track_energy_gain(0, phase=0.0, phi_s_start=0.0)
    dW_b = _track_energy_gain(0, phase=0.0, phi_s_start=90.0)
    assert not np.isclose(dW_a, dW_b, rtol=1e-6)


def test_p_flag_absolute_ignores_ref_phi_s():
    # p_flag=1: ref.phi_s should not change the kick
    dW_a = _track_energy_gain(1, phase=0.0, phi_s_start=0.0)
    dW_b = _track_energy_gain(1, phase=0.0, phi_s_start=90.0)
    np.testing.assert_allclose(dW_a, dW_b, rtol=1e-12)
```

- [ ] **Step 2-3: implement in both elements**

Change the phasor computation inside both `FieldMap.track_rk4` and `FieldMap3D.track_rk4`:

```python
if self.p_flag == 1:      # absolute phase
    phi_sync_rad = self.phase * np.pi / 180.0
else:                     # relative (default)
    phi_sync_rad = (self.phase + ref.phi_s) * np.pi / 180.0
```

Apply the same logic in `advance_ref`.

- [ ] **Step 4: run tests green**
- [ ] **Step 5: Commit** — `feat(fieldmap): honor p_flag (absolute vs relative phase)`.

---

## Task 8 — Parser dispatch via `decode_geom`

**Files:**
- Modify: `linac_gen/io/tracewin_parser.py`
- Modify: `tests/io/test_tracewin_field_map_dispatch.py`

- [ ] **Step 1: Rewrite the failing dispatch tests with realistic geoms**

Replace the `(70, "ed"), (71, "bs")` parametrisation with:

```python
@pytest.mark.parametrize("geom, expected_channels, suf_set", [
    (70,    ["STAT_B"],         ["bsx", "bsy", "bsz"]),
    (100,   ["RF_E"],           ["edz"]),
    (400,   ["RF_E"],           ["edz", "edr", "bdq"]),
    (7700,  ["RF_E", "RF_B"],   ["edx","edy","edz","bdx","bdy","bdz"]),
    (90,    ["STAT_B"],         ["bsz"]),
])
def test_dispatch_loads_correct_channels(tmp_path, geom, expected_channels, suf_set):
    …
```

- [ ] **Step 2: Replace the FIELD_MAP branch in `tracewin_parser.py`**

```python
elif keyword == "FIELD_MAP":
    kw = parse_positionals(SCHEMA["FIELD_MAP"], params)
    if freq is None:
        freq = _DEFAULT_FREQ_MHZ
    try:
        from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
        from linac_gen.io.tracewin_geom import decode_geom, enabled_channels
        from linac_gen.elements.field_map_factory import make_field_map_element

        fdata = read_tracewin_fieldmap(
            geom=kw["geom"],
            prefix=kw["filename"],
            base_dir=base_dir,
            frequency=freq,
        )
        elem = make_field_map_element(
            name=next_name("FMAP"),
            code=decode_geom(kw["geom"]),
            length_mm=kw["length"],
            field_data=fdata,
            kb=kw["kb"], ke=kw["ke"], ki=kw["ki"], ka=kw["ka"],
            phase=kw["phase"], frequency=freq,
            aperture=kw["aperture"], p_flag=kw["p_flag"],
            n_steps=100,
        )
        lattice.add(elem)
    except FileNotFoundError as exc:
        metadata["warnings"].append(
            f"Line {line_num}: FIELD_MAP file missing: {exc}"
        )
```

The new `make_field_map_element` factory lives in a new module
`linac_gen/elements/field_map_factory.py`; it inspects the `GeomCode`
and returns either `FieldMap` (if every enabled channel is 1-D or 2-D
cyl) or `FieldMap3D` (if any channel is 3-D Cart).  If geoms mix 1-D
and 3-D the factory raises with a clear message.

- [ ] **Step 3: Run dispatch tests**

- [ ] **Step 4: Commit**

```bash
git add linac_gen/io/tracewin_parser.py \
        linac_gen/elements/field_map_factory.py \
        tests/io/test_tracewin_field_map_dispatch.py
git commit -m "feat(parser): FIELD_MAP dispatch via decode_geom → element factory"
```

---

## Task 8b — Envelope-mode space charge inside FieldMap

**Why:** `tracking/envelope.py:409-411` currently routes FieldMap elements into a single-shot `M_full @ sigma @ M_full.T` with no SC sub-stepping.  For a 400 mm solenoid or a multi-metre RF cavity at ≥1 mA this misses all the SC defocusing/focusing that MP mode picks up via the Strang-split in `tracker.py:_track_field_map`.  Envelope and MP must agree.

**Files:**
- Modify: `linac_gen/tracking/envelope.py` — new branch for `FieldMapElement` that mirrors `_propagate_with_sc`'s bundle structure
- Modify: `linac_gen/elements/base.py` — add a `sub_matrix(ref, ds)` method (or equivalent) on `FieldMapElement` so envelope can get a linearised half-slice matrix
- Create: `tests/tracking/test_envelope_field_map_sc.py`

- [ ] **Step 1: Failing test — envelope vs MP through a 3-D solenoid with current**

```python
"""Envelope and MP must agree on σ_x at end of a current-carrying solenoid."""
import os, numpy as np, pytest
from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.core.lattice import Lattice
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.tracking.envelope import EnvelopeTracker
# … MP tracker import and beam factory …


@pytest.mark.skipif(not os.path.isdir("Fields"), reason="Fields/ absent")
def test_envelope_agrees_with_mp_through_sol1_with_current():
    fd = read_tracewin_fieldmap(geom=70, prefix="Fields/SOL1-PXIE")
    sol = FieldMap3D(name="SOL1", length=400.0, field_data=fd,
                     scale=1.0, n_steps=200)
    lat = Lattice(); lat.add(sol)

    current_mA = 5.0
    # Build matched σ at 5 MeV, 1 mA with some α, β
    sigma0 = np.eye(6)
    sigma0[0,0] = 4.0;  sigma0[2,2] = 4.0;  sigma0[4,4] = 1.0   # mm²…

    # Envelope
    ref_env = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    env = EnvelopeTracker(lat, sigma0.copy(), ref_env, current=current_mA)
    res_env = env.run()

    # MP — same initial σ, 10 k particles
    ref_mp = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam = _build_mp_beam_from_sigma(sigma0, ref_mp, n=10000, seed=1,
                                     current=current_mA)
    # … run MP tracker …
    sigma_mp = np.cov(beam.particles, rowvar=False)

    # rms x at element exit
    np.testing.assert_allclose(
        res_env.beta_x[-1] * np.sqrt(sigma_mp[0, 0] / res_env.beta_x[-1]),
        np.sqrt(sigma_mp[0, 0]),
        rtol=0.05,
    )
```

(Implementer: the point of the test is σ_x agreement to ≤5 % between the two trackers through the current-carrying solenoid.)

- [ ] **Step 2: Run; expect envelope to be wrong because SC isn't applied inside the FieldMap**

- [ ] **Step 3: Extend `envelope.py:_propagate_with_sc`**

```python
if isinstance(element, TransferMapElement) and total_length > 0:
    ...   # existing path
elif isinstance(element, FieldMapElement) and total_length > 0:
    n_int = (cfg.integration_steps_for_length_mm(element.length)
             if cfg is not None else element.n_steps)
    n_sc  = (cfg.sc_steps_for_length_mm(element.length)
             if cfg is not None else n_int)
    ds_mm  = element.length / n_int
    sc_every = max(1, n_int // n_sc)
    n_bundles = n_int // sc_every

    # Reset element sub-step counter for a clean linearised pass.
    element._step_idx = 0
    for _ in range(n_bundles):
        M_half = element.fitted_matrix_slice(self._ref, ds_mm * sc_every / 2.0)
        sigma  = M_half @ sigma @ M_half.T
        # SC kick at bundle midpoint using current σ
        sx = np.sqrt(max(sigma[0, 0], 0.0))
        sy = np.sqrt(max(sigma[2, 2], 0.0))
        sphi = np.sqrt(max(sigma[4, 4], 0.0))
        M_sc = _sc_kick_matrix_3d(..., sigma_x_mm=sx, sigma_y_mm=sy,
                                  sigma_phi_deg=sphi, ds_mm=ds_mm * sc_every)
        sigma = M_sc @ sigma @ M_sc.T
        sigma = M_half @ sigma @ M_half.T
    # trailing remainder (if any): single half-slice pass
    leftover_steps = n_int - n_bundles * sc_every
    if leftover_steps > 0:
        M_rest = element.fitted_matrix_slice(self._ref, leftover_steps * ds_mm)
        sigma  = M_rest @ sigma @ M_rest.T
else:
    sigma = M_full @ sigma @ M_full.T    # passive / zero-length fallback
```

Need to add `FieldMapElement.fitted_matrix_slice(ref, ds_mm)` — numerically-Jacobian linearised matrix for a partial pass through the element (internally advances `_step_idx` by `ds/element.ds`).

- [ ] **Step 4: Run the failing test — should now pass**

- [ ] **Step 5: Commit**

```bash
git add linac_gen/tracking/envelope.py linac_gen/elements/base.py \
        linac_gen/elements/field_map.py linac_gen/elements/field_map_3d.py \
        tests/tracking/test_envelope_field_map_sc.py
git commit -m "fix(envelope): apply SC inside FieldMap via bundled half-slice matrices"
```

---

## Task 9 — Regression test with `Fields/SOL1-PXIE.*`

**Goal:** prove that the three real solenoid maps work end-to-end and focus a beam to the analytic thick-lens focal length.

**Files:**
- Create: `tests/elements/test_field_map_solenoid.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end regression: real PXIE solenoid field map (geom=70)."""
import os
import numpy as np
import pytest
from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON

FIELDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Fields")


@pytest.mark.skipif(not os.path.isdir(FIELDS_DIR),
                    reason="Fields/ directory not present")
def test_sol1_pxie_focuses_proton_beam():
    prefix = os.path.join(FIELDS_DIR, "SOL1-PXIE")
    fd = read_tracewin_fieldmap(geom=70, prefix=prefix, frequency=0.0)
    # Expect exactly one channel, static magnetic 3D
    assert list(fd.channels.keys())[0].name == "STAT_B"

    sol = FieldMap3D(name="SOL1", length=fd.axis_length_mm(),
                     field_data=fd, scale=1.0,
                     phase=0.0, frequency=0.0, n_steps=200)

    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=1, current=0.0)
    # Launch 20 mm, parallel
    beam.particles[0] = [20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ds = sol.length / sol.n_steps
    sol._step_idx = 0
    for _ in range(sol.n_steps):
        sol.track_rk4(beam, ds)
    x_out = beam.particles[0, 0]
    xp_out = beam.particles[0, 1]
    # ---- Analytic thick-lens solenoid check ----
    # For a solenoid with uniform ∫Bz²dz over length L on a particle of
    # rigidity (Bρ), the thick-lens focal length is:
    #     f ≈ (Bρ)² / (K² × L × sinc²(KL/2))      where K = qBz/(2·Bρ)
    # For PXIE SOL1 (peak 1 T, effective length ≈ 400 mm, hard-edge
    # approximation) the Larmor half-angle per slice is
    #     KL/2 = qBL / (4·Bρ)
    # The exit x-angle for a particle entering with x=20 mm, x'=0 is
    #     xp_expected ≈ -x_in · K · sin(KL) / (cos(KL))
    # (simplified), or equivalently xp_expected ≈ -x_in / f.
    #
    # Compute expected value from CODATA constants (no magic tolerance):
    from linac_gen.core.constants import C_LIGHT_M_PER_S   # or define locally
    p_MeV_over_c = ref.gamma * ref.beta * PROTON.mass     # p = γβmc in MeV/c
    B_peak_T = max(abs(fd.channels[ch]).Fz.max() for ch in fd.channels)   # pseudo
    K = 1 * B_peak_T / (2 * p_MeV_over_c / (C_LIGHT_M_PER_S * 1e-6))  # 1/m
    L_m = sol.length * 1e-3
    f_thick = 1.0 / (K * np.sin(K * L_m))                # m
    xp_expected_mrad = -20.0 / (f_thick * 1000.0) * 1000.0   # mm → mrad
    # Tolerance: 10% for hard-edge vs measured profile
    np.testing.assert_allclose(xp_out, xp_expected_mrad, rtol=0.15)

    # Larmor rotation: off-axis parallel → gains y-velocity of the same order
    assert abs(beam.particles[0, 3]) > 0.1 * abs(xp_out), \
        "v×Bz rotation should give y' of same order as focusing kick"
```

- [ ] **Step 2: Run**

`python3 -m pytest tests/elements/test_field_map_solenoid.py -v`

Expected: PASS.  If not, the implementer investigates the Lorentz-force formula in Task 6.

- [ ] **Step 3: Commit**

```bash
git add tests/elements/test_field_map_solenoid.py
git commit -m "test(fieldmap): real PXIE solenoid regression (geom=70)"
```

---

## Task 10 — Retire legacy dispatch constants + docs

**Files:**
- Remove: flat `Ex/Ey/Ez/Bx/…` shims from `field_map_data.py`
- Modify: `docs/fieldmap_guide.md` — rewrite around `geom`
- Modify: any remaining consumer still using flat accessors

- [ ] **Step 1: Grep for flat accessor usage**

```bash
grep -rn "\.Ex\|\.Ey\|\.Ez\|\.Bx\|\.By\|\.Bz\|\.Er\|\.Br\|\.Ftheta\|\.Fq" linac_gen/ tests/ gui/
```

- [ ] **Step 2: Update each hit to use `fd.channels[Channel.XXX].Fz` (or equivalent)**

- [ ] **Step 3: Remove the shim properties from `field_map_data.py`**

- [ ] **Step 4: Rewrite `docs/fieldmap_guide.md`**

New outline:
- What is a TraceWin field map? (link to manual)
- The `geom` 5-digit encoding (reproduce the manual's table)
- File-naming cheat sheet
- Unit conventions (metres, MV/m, T)
- Phasor conventions (cos for RF-E, sin for RF-B)
- The `kb` / `ke` / `Norm` scaling
- `FIELD_MAP_PATH` global-path handling
- End-to-end example with the PXIE solenoid

- [ ] **Step 5: Run the full suite; commit**

```bash
python3 -m pytest tests/ --tb=short 2>&1 | tail -30
git add -A
git commit -m "chore(fieldmap): drop legacy flat accessors + rewrite docs"
```

---

## Task 11 — Final integration + merge

- [ ] **Step 1: Rebase `feat/tracewin-geom` onto `master`**

```bash
git fetch origin
git rebase origin/master
```

- [ ] **Step 2: Run the **entire** test suite**

```bash
python3 -m pytest tests/ -q 2>&1 | tail
```

Expected: all green.

- [ ] **Step 3: Open a PR or merge**

(Per user's preference — defer until they ask.)

---

## Self-review checklist

Run this before declaring the plan done.

1. **Spec coverage** — every channel/digit combination from the manual:
   - [ ] stat_E digits 1, 4, 6, 7  — tests present?
   - [ ] stat_B digits 1, 5, 6, 7, 9  — tests present?
   - [ ] rf_E digits 1, 4, 6, 7  — tests present? (digit 4 exercises TM mode with Bθ)
   - [ ] rf_B digits 1, 5, 6, 7  — tests present? (digit 5 exercises TE mode with Eθ)
   - [ ] aper digits 1-9 — digit parsed but actual masking deferred (documented)
   - [ ] negative geom (second_order flag) — parsed; actual 2nd-order expansion deferred (documented)
   - [ ] digit 8 everywhere — `NotImplementedError`

2. **Phase conventions**
   - [ ] Static → phasor = 1
   - [ ] RF-E  → cos(ωt+φ)
   - [ ] RF-B  → sin(ωt+φ)

3. **Unit conventions**
   - [ ] All file headers in metres, converted to mm.
   - [ ] Field values verbatim (MV/m, T).
   - [ ] Amplitude scaling `k/Norm` with k=ke (E) or kb (B).

4. **Lorentz force**
   - [ ] `F = q(E + v × B)` for each particle.
   - [ ] `v × Bz` rotation of (x', y') applied for solenoids.
   - [ ] E-kick denominator is `γβ²·mc²` (not `γ²β²·mc²` as in the old code).
   - [ ] B-kick has the explicit `c ≈ 299.792458 MV/(T·m)` factor (currently missing — old code gives kicks 3×10⁸ too small).
   - [ ] 1-D paraxial Er = −(r/2)·dEz/dz is dimensionally consistent — the spurious `* 1e-3` in the old code is removed.

5. **Off-axis expansion (1-D maps)**
   - [ ] 1st-order `Er = −(r/2)·dEz/dz`, `Br = −(r/2)·dBz/dz` applied as TraceWin default.
   - [ ] 2nd-order correction `Ez(r) ≈ Ez₀ − (r²/4)·d²Ez/dz²` and `Bθ(r) = (r/(2c²))·∂Ez/∂t` for RF TM mode are **deferred** (TraceWin activates these only for negative `geom`, which we flag in metadata).

6. **Envelope / MP / SC consistency**
   - [ ] Envelope mode applies SC via bundled half-slices inside FieldMap (Task 8b).
   - [ ] MP mode applies SC via Strang-split bundles inside FieldMap (already correct in `tracker.py`).
   - [ ] A dedicated test asserts envelope σ_x and MP σ_x agree to ≤5 % through a current-carrying solenoid.

7. **Magnitude sanity**
   - [ ] Regression test (Task 9) matches an analytic thick-lens focal length within ≤15 %.
   - [ ] 1-D RF cavity on-crest energy gain matches `q·∫Ez·cos(φ)dz` within numerical error.

8. **Back-compat**
   - [ ] Legacy fixtures (`test_1d.edz`, etc.) still pass.
   - [ ] Existing element tests still pass.

If any box is unchecked after implementation, open a follow-up task.

---

## Execution handoff

After saving this plan and committing it, use **superpowers:subagent-driven-development** to execute task-by-task.  Each task dispatches a fresh subagent with: full task text, code snippets, failing-test expectations, and commit message.  After each task runs, dispatch a spec-compliance reviewer (did the subagent do what the task said?) and a code-quality reviewer (style / edge cases / DRY) before moving on.
