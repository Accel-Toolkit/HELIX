# `.dat` file reference

HELIX reads TraceWin `.dat` files natively.  This page is the
keyword reference: every directive HELIX recognises, with parameter
slots and behaviour notes.

!!! warning "HELIX `.dat` files use millimetres — not directly portable to real TraceWin"
    All lengths in a HELIX `.dat` are interpreted and written in
    **mm**, matching the internal HELIX unit convention
    (`linac_gen/io/tracewin_parser.py`, `linac_gen/io/tracewin_writer.py`).
    A file written by HELIX round-trips cleanly through HELIX's own
    parser, but **no unit-for-unit parity with the real TraceWin
    program is claimed** — do not feed HELIX-written `.dat` files to
    TraceWin (or vice versa) without checking the units yourself.

## File anatomy

```
; Comments start with semicolon
TITLE Example PIP-II MEBT
FREQ 162.5
PARTRAN_STEP 100 50

; Element cards (each on its own line)
DRIFT 100 20
QUAD 100 8.5 20
DRIFT 100 20

; Subsection markers
LATTICE
DRIFT 200 20
LATTICE_END

; Error directives (consumed by error_study)
ERROR_GAUSSIAN_CUT_OFF 3
ERROR_QUAD_NCPL_STAT 6 2 0.2 0.2 0 0 0 0.5 0 0 0 0

; HELIX-specific extensions (comment-prefixed)
;@LG dc_kernel=gaussian
;@LG nx=64 ny=64 nz=64

END
```

## Header / control directives

| Keyword | Slots | Meaning |
|---|---|---|
| `TITLE` | string | lattice title (informational) |
| `FREQ` | f (MHz) | sets reference frequency; updates with each card |
| `PARTRAN_STEP` | step1 step2 (per metre) | integration step density → `lattice.step_config` |
| `LATTICE` | (none) | start of lattice subsection |
| `LATTICE_END` | (none) | end of lattice subsection |
| `END` | (none) | end of file |
| `FIELD_MAP_PATH` | path | directory for `FIELD_MAP` files |

`REPEAT_ELE` is **not implemented**: the parser has no branch for it,
so it lands on the unsupported-card path (a warning in
`metadata["warnings"]`, or `ValueError` with `strict=True`).  Expand
repeats manually before importing.

## Element cards

| Keyword | Parameter slots | HELIX class |
|---|---|---|
| `DRIFT` | L aperture [aperture_y x_shift y_shift] | [Drift](../03_elements/01_drift.md) |
| `QUAD` | L G aperture [skew g3 g4 g5 g6 gfr] | [Quadrupole](../03_elements/02_quadrupole.md) |
| `SOLENOID` | L B aperture | [Solenoid](../03_elements/03_solenoid.md) |
| `BEND` | angle ρ [field_index aperture hv] | [Dipole](../03_elements/04_dipole.md) |
| `EDGE` | pole_rotation ρ gap k1 k2 [aperture hv] | [Edge](../03_elements/05_edge.md) |
| `GAP` | e0tl φ_s aperture [p_flag] | [RFGap](../03_elements/06_rfgap.md) |
| `FIELD_MAP` | geom L θ_i R kb ke Ki Ka FileName [p_flag] | [FieldMap](../03_elements/07_fieldmap.md) |
| `RFQ_CELL` | (Toutatis format) | [RfqCell](../03_elements/09_rfqcell.md) |
| `THIN_STEERING` / `STEERER` | bx_l by_l aperture [elec] | [Steerer](../03_elements/14_steerer.md) |
| `APERTURE` | dx dy aperture_type | [Aperture](../03_elements/12_aperture.md) |
| `MARKER` | name | [Marker](../03_elements/13_marker.md) |

## Error directives

See [Errors → ERROR_* directives](../08_errors/02_error_directives.md)
for full coverage.  Quick map:

| Keyword | Effect |
|---|---|
| `ERROR_GAUSSIAN_CUT_OFF σ_max` | sets default truncation |
| `ERROR_SET_RATIO r1 r2 …` | multi-scale sweep ratios |
| `ERROR_QUAD_NCPL_STAT N r dx dy φx φy φz dG dG3 dG4 dG5 dG6 [Nb]` | quad alignment + field |
| `ERROR_CAV_NCPL_STAT N r dx dy φx φy E φ dz [Nb]` | cavity alignment + voltage + phase |
| `ERROR_BEND_NCPL_STAT N r dx dy φx φy φz dg dz` | dipole alignment + field |
| `ERROR_BEAM_STAT r dx dy dφ dxp dyp de dEx dEy dEz mx my mz dIb` | beam-input jitter |

`r` = distribution code: 0 → gaussian (no constant special-case; the
global cutoff applies as usual); 1 → uniform; 2 → gaussian; 4 / 5
(binary) → treated as gaussian (coarse approximation).

## Matching directives

See [Matching → SET / ADJUST](../07_matching/02_set_adjust.md):

| Keyword (with slots) | Effect |
|---|---|
| `SET_SYNC_PHASE` | (no args) interpret the next `FIELD_MAP`'s θᵢ as the synchronous phase |
| `SET_TWISS family αx βx αy βy αz βz kax kbx kay kby kaz kbz` | Twiss target (no emittance slots; the k-flags select which values are constrained) |
| `SET_POSITION k x xp y yp` | centroid-position constraint |
| `SET_SIZE k x y φ k2` | beam-size constraint |
| `SET_ACHROMAT k f1 f2 plane` | dispersion = 0 |
| `ADJUST target param_idx [link_group vmin vmax start_step kn]` | generic matching variable — `target` is a family/section, `param_idx` the parameter index |
| `ADJUST_STEERER N vmax first_step` | steerer variable (both planes) for orbit correction |
| `ADJUST_STEERER_BX` / `ADJUST_STEERER_BY` | single-knob variants: `_BX` = `bx_l` knob → **vertical** plane; `_BY` = `by_l` → **horizontal** |
| `ADJUST_BEAM_TWISS` / `_CENTROID` / `_EMIT` / `_CURRENT` | beam-input matching variables (`diag_n` + flag tail) |

There are no per-element `ADJUST_QUAD` / `ADJUST_SOLENOID` /
`ADJUST_DIPOLE` / `ADJUST_DRIFT` / `ADJUST_GAP` keywords — element
parameters are adjusted through the generic `ADJUST` card.

## HELIX-specific extensions

Comment-prefixed (won't break TraceWin):

```
;@LG dc_kernel=gaussian
;@LG kernel=tsc green_kind=igf
;@LG use_gpu=auto
;@LG nx=64 ny=64 nz=64
```

A `; HELIX_FOIL` directive declares a scattering / stripper
[Foil](../03_elements/15_foil.md) element — also comment-prefixed, so
TraceWin ignores it:

```
; HELIX_FOIL <name> <material> <thickness_ug_cm2>
; HELIX_FOIL STRIP C 600
```

A `; HELIX_SC_GRID` directive changes the 3-D bunched PIC solver's
grid extent (in beam sigmas) **from its position onward** — also
comment-prefixed.  Use it when one deck spans sections with different
grid needs, e.g. a linac at the project default followed by a long
no-RF transfer line whose dispersive orbits and halo need a much wider
grid (the standalone PIP-II BTL project uses 20 σ):

```
; HELIX_SC_GRID <extent_sigma>
; HELIX_SC_GRID 20
```

It applies in both fixed and adaptive grid modes (fixed mode re-derives
its frozen grid once, from the beam at the card's position).  The card
affects only the bunched 3-D PIC solver: DC/continuous kick models and
envelope runs ignore it (the tracker warns once if it can have no
effect).  Several cards may appear; each takes effect from its own
position.

See [HELIX extensions](03_lg_extensions.md).

## Importing MAD-X lattices

Besides TraceWin `.dat`, HELIX reads a subset of **MAD-X** lattice
files (`.madx` / `.seq`).  In the GUI, `Open Lattice…` accepts them
directly; programmatically:

```python
from pathlib import Path
from linac_gen.io.madx_parser import parse_madx

# A minimal MAD-X file to demonstrate (or point at your own):
Path("ring.madx").write_text("""\
lq := 0.1;
qf: quadrupole, l=lq, k1=2.5;
qd: quadrupole, l=lq, k1=-2.5;
cell: sequence, l=1.0;
  qf, at=0.25;
  qd, at=0.75;
endsequence;
beam, particle=proton, energy=0.941;
use, sequence=cell;
""")
lattice, meta = parse_madx("ring.madx")   # → (Lattice, metadata)
Path("ring.madx").unlink()
```

`meta` carries `"title"`, `"warnings"` (a list — unsupported
constructs are skipped, not fatal) and `"reference"` (the
`ReferenceParticle` built from the `BEAM` command).

**Supported:** `!` / `//` / `/* */` comments; `name = expr` and
`name := expr` assignments with `+ - * / ^`, parentheses, the
constants `pi` / `twopi` / `e` / `clight`, and `sqrt` / `abs` /
`sin` / `cos`; element definitions; `SEQUENCE … ENDSEQUENCE` with
`elem, at=…` members (gap-filling drifts are inserted automatically);
the `BEAM` command; `USE, SEQUENCE=…`.

**Element mapping:**

| MAD-X | HELIX |
|---|---|
| `DRIFT` | `Drift` |
| `QUADRUPOLE` | `Quadrupole` (`k1` → gradient via Bρ) |
| `SBEND` | `Edge` + `Dipole` + `Edge` |
| `RBEND` | `Edge` + `Dipole` + `Edge` (edges pick up ±angle/2) |
| `SEXTUPOLE` / `MULTIPOLE` | `Multipole` |
| `RFCAVITY` | `RFGap` |
| `SOLENOID` | `Solenoid` (`ks` → field via Bρ) |
| `MARKER` | `Marker` |
| `MONITOR` / `HMONITOR` / `VMONITOR` | `Marker` (BPM-flagged) |

**Not supported** (skipped with a warning): `MACRO`, `LINE = (…)`
expansion, `IF` / `WHILE`, `MATCH` / `TRACK` blocks, and deferred
expressions that reference later-defined variables.

> MAD-X lengths are in **metres**; HELIX converts them to its
> internal millimetres on import.

## Importing MAD8 flat files (`.lat`)

HELIX also reads **MAD8 flat / SAVELINE lattices** (`.lat` / `.flat`),
the dialect used by the PIP-II BTL exports — `Open Lattice…` in the
GUI, or:

```python
from linac_gen.io.mad8_parser import parse_mad8
lattice, meta = parse_mad8("BTL2025v0703.lat")   # → (Lattice, metadata)
```

**Supported:** `!` comments and `&` continuations; deferred parameters
`name := expr` (and `name = expr`) with references to other parameters
and `NAME[ATTR]` element-attribute references; element definitions
`name: TYPE, attr=expr, …`; `name: LINE = (A, B, -C, 2*D)` with
recursive expansion, reversal, and integer repetition.  The element
mapping matches the MAD-X table above, plus: `KICKER` / `HKICKER` /
`VKICKER` → `Marker` + full-length body `Drift` (geometry preserved;
non-zero kicks warn), plain `MONITOR` → non-BPM `Marker`, a `TILT` on
a quadrupole → `skew_angle`, and `TILT=±π/2` on a bend → a vertical
bend (`hv=1`).

**Rigidity and charge sign.**  MAD strengths are normalized
(K1 = (q/p)·∂B/∂x); HELIX stores the lab-frame gradient, so the
conversion is **G = sign(q)·K1·|Bρ|** — for H⁻ every gradient sign
flips relative to a proton import.  Bρ is resolved in order from the
`brho=` argument, a `BRHO := …` parameter in the file, or a `BEAM`
statement; if none is available the import **fails loudly** (a wrong
Bρ would silently mis-scale every magnet).  The species defaults to
H⁻ (`species="proton"` to override).  A warning always states the
assumed reference — set the Beam tab to match before running.

**Automatic periodicity.**  The `.lat` LINE hierarchy declares the
machine's cell structure, which a flat TraceWin file loses.  The
importer identifies repeated cells (LINE-valued entries of the root
line's sections with identical transport signatures and significant
element counts) and declares them as `LATTICE n 0` / `LATTICE_END`
brackets automatically, so per-cell phase advance σ₀, tune depression
η, and the Hofmann chart work immediately.  Disable with
`parse_mad8(..., auto_periods=False)`.

> HELIX never writes MAD files: with a `.lat` (or `.madx`) lattice
> loaded, **Save** routes to *Save As…* (TraceWin `.dat`) instead of
> overwriting the source.  Declared `LATTICE` brackets survive the
> `.dat` export and round-trip.

## Cross-references

* [Keyword cheatsheet](../appendices/B_keyword_cheatsheet.md) —
  one-page printable.
* [Element catalog](../03_elements/00_overview.md).
* [Matching SET/ADJUST](../07_matching/02_set_adjust.md).
* [Errors directives](../08_errors/02_error_directives.md).

← [Python API](01_python_api.md) ·
[Continue to HELIX extensions →](03_lg_extensions.md)
