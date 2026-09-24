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

**Labels.** A card may carry a TraceWin label (`Q01: QUAD …`, `D01T : THIN_STEERING …`,
`D01BPM:DIAG_POSITION …`).  The parser keeps the raw label on every element the
line creates as `element.label` (duplicates allowed — fnalscl labels both halves
of its first trim `D01T`) while `element.name` stays the generated unique
identifier (`QUAD_001`, `STEER_001`; BPM markers are named after their label,
de-duplicated).  The writer emits the label again on single-line cards, so a
saved deck keeps the device names the control system uses; unlabeled decks are
written without labels.

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
| `QUADRUPOLE` | `Quadrupole` (`k1` → gradient via Bρ; `tilt` → `skew_angle`, a bare `tilt` = π/4) |
| `SBEND` | `Edge` + `Dipole` + `Edge` (ρ > 0 with the direction in the angle's sign; `k1` → field index n = −k1·ρ²; `tilt=+π/2` / `−π/2` → vertical bend towards +y / −y; pole faces `e1`/`e2` → β = sign(angle)·e, because MAD-X's edge focusing is `h·tan e` with the signed curvature while HELIX's is `tan β/ρ`; `hgap`/`fint`/`fintx` → edge gap and K1, with MAD-X's default `fint = 0`) |
| `RBEND` | `Edge` + `Dipole` + `Edge` (edges pick up ±angle/2; same extras as `SBEND`). `l` is the straight chord under MAD-X's default `OPTION, RBARC=true` and the arc is `l·(θ/2)/sin(θ/2)`; `OPTION, RBARC=false` (LHC-style files) makes `l` the arc length |
| `KICKER` / `HKICKER` / `VKICKER` | `Steerer` (kick angles → ∫B·dl through the charge-signed Bρ, so an H⁻ deck kicks the same way MAD-X does) |
| `RCOLLIMATOR` / `ECOLLIMATOR` | `Aperture` (rectangular / circular; `xsize`, `ysize` are half-apertures) |
| `SEXTUPOLE` / `MULTIPOLE` | `Multipole` (`MULTIPOLE tilt` → `tilt_deg`, same rotation sense as MAD-X's `tilt` and `Quadrupole.skew_angle`; a `SEXTUPOLE tilt` is not imported) |
| `MATRIX` | `MatrixElement` (`rm11…rm66`, `kick1…kick6` converted from MAD-X's canonical basis at the local reference energy; a `kick6` becomes a ΔW offset on every particle — HELIX's reference stays at the `BEAM` energy) |
| `RFCAVITY` | `RFGap` (`volt` in MV; `lag` follows the MAD-X definition — reference gain `V·sin(2π·lag)`, **not** multiplied by the charge, whereas HELIX gains `q·V·T·cos φs` — so the synchronous phase is `360·lag − 90°`, plus 180° when `volt` and the charge have opposite signs; TTF = 1) |
| `SOLENOID` | `Solenoid` (`ks` → field via Bρ) |
| `MARKER` | `Marker` |
| `MONITOR` / `HMONITOR` / `VMONITOR` | `Marker` (BPM-flagged) |

**Not supported** (skipped with a warning): `MACRO`, `LINE = (…)`
expansion, `IF` / `WHILE`, `MATCH` / `TRACK` blocks, and deferred
expressions that reference later-defined variables.

> MAD-X lengths are in **metres**; HELIX converts them to its
> internal millimetres on import.

> The `lag` convention is pinned against MAD-X itself (a tracked
> reference particle through one `RFCAVITY`, `tests/io/test_madx_conventions.py`).
> Decks imported before 2026-09-03 carried every cavity 90° off
> (zero reference gain at `lag=0`); re-import them.

## Exporting to MAD-X

The path also runs the other way: HELIX writes the loaded lattice as a
MAD-X `SEQUENCE` file, built as the **exact inverse of the importer
above** (same unit and rigidity conversions, so export → import gives the
lattice back to machine precision).  Three entry points:

* **GUI** — *File → Export Lattice as MAD-X…*.  The Beam tab's species
  and energy set the rigidity every `k1` / `ks` is normalised by, so the
  beam must be configured first; warnings are listed in the completion
  box and counted in the status bar.
* **CLI** — `python -m linac_gen export deck.dat out.madx --energy 2.1
  --species H-` (`--energy` is required for a bare lattice; a `.lgproj`
  supplies its Beam settings).  `--strict` refuses the export if any
  element has no MAD-X equivalent; `--linearize` writes them as `MATRIX`
  elements instead (below); `--sequence` / `--title` name the sequence
  and the `TITLE` line.  See [CLI](06_batch_cli.md).
* **Python**:

```python
from pathlib import Path
from linac_gen.cli.common import build_ref
from linac_gen.core.config import BeamConfig
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.madx_parser import parse_madx
from linac_gen.io.madx_writer import write_madx
from linac_gen.io.tracewin_parser import parse_tracewin

lattice, _ = parse_tracewin("examples/fodo_cell.dat")
ref = build_ref(BeamConfig(species="H-", energy=3.0, frequency=352.21))
warnings = write_madx(lattice, "fodo_cell.madx", ref)   # → list[str]
assert warnings == []
back, meta = parse_madx("fodo_cell.madx")
quads = [e for e in lattice.elements if isinstance(e, Quadrupole)]
back_quads = [e for e in back.elements if isinstance(e, Quadrupole)]
assert [q.gradient for q in back_quads] == [q.gradient for q in quads]
Path("fodo_cell.madx").unlink()
```

The file carries a header comment (HELIX version, reference particle,
Bρ), a `TITLE`, the `BEAM` line, the element definitions, one
`SEQUENCE … ENDSEQUENCE` with `refer=entry` and explicit `at=` entry
positions, and a closing `USE`.  Lengths are in metres, angles in
radians, `k1` in 1/m², `ks` in 1/m, `volt` in MV, `freq` in MHz.

**Element mapping** (Bρ is the reference rigidity, signed by the charge
— an H⁻ deck exports with the `k1` signs MAD-X expects for a negative
particle; H⁻ and deuterons become `BEAM, particle=ion, mass=…,
charge=…` since MAD-X has no named particle for them):

| HELIX | MAD-X |
|---|---|
| `Drift` | `DRIFT` |
| `Quadrupole` | `QUADRUPOLE` (`k1 = G/Bρ`; `skew_angle` → `tilt`; `g3…g6` / `gfr` are not representable and warn) |
| `Edge` + `Dipole` + `Edge` | one `SBEND` (`l` = arc, signed `angle`, `e1`/`e2 = sign(angle·ρ)·β` from the pole rotations, `k1 = −n/ρ²`, `tilt=pi/2` for a vertical bend, `hgap`/`fint`/`fintx` from the edge gaps and K1s; a non-default edge K2 warns).  `SET_*`/`ADJUST_*` cards between the edges and the body stay comments, a magnet split into consecutive `BEND` cards keeps its one edge pair, and edges whose ρ or plane disagree with the body are folded with a warning.  A lone `Dipole` becomes an `SBEND` with `e1=e2=0`; `RBEND` is never written |
| `Solenoid` | `SOLENOID` (`ks = B/Bρ`) |
| `RFGap` | `RFCAVITY, l=0` (`volt = sign(q)·V·TTF` — MAD-X does not multiply the RF kick by the charge and has no transit-time factor, so the sign carries the species and a TTF ≠ 1 is folded into `volt` with a warning; `lag = (φs + 90°)/360°`, the inverse of the import rule; `freq` in MHz) |
| `Steerer` (magnetic) | `KICKER` (`hkick`/`vkick` from ∫B·dl and the signed Bρ; electric steerers are not representable) |
| `Multipole` | `MULTIPOLE, knl={…}, ksl={…}, tilt` (same sense, see the import row) |
| `Aperture` | `RCOLLIMATOR` / `ECOLLIMATOR` with both `xsize`/`ysize` (MAD-X `TRACK` losses) and `apertype`/`aperture` (its `APERTURE` module); the rectangular family (types 0, 3, 4, 5) is rectangular, type 1 circular; pepperpot and ring are not representable |
| `Marker` / BPM marker | `MARKER` / `MONITOR` |
| `FREQ`, `SET_*`, `ADJUST_*`, `DIAG_*`, `LATTICE`, `SPACE_CHARGE_COMP`, … | a `! HELIX: …` comment at the element's position — nothing vanishes silently |

**Not representable in MAD-X** — field maps (`FIELD_MAP`,
`SUPERPOSE_MAP`), `NCELLS`, RFQ cells, foils, explicit-matrix
elements, thin lenses and electric steerers.  By default each becomes a
`MARKER` plus a body `DRIFT` of the element's length (the geometry
survives) and a warning naming it; `write_madx(…,
on_unsupported="error")` / `export --strict` refuses the export
instead.  Misalignments (`dx`, `dy`, tilt) are not exported and warn —
MAD-X carries them through `EALIGN`, which this writer does not emit.

**Linearised export** (`write_madx(…, linearize=True)` /
`export --linearize`): field maps, `NCELLS`, RFQ cells, thin lenses and
explicit-matrix elements are written as MAD-X `MATRIX` elements built
from their first-order map at the local reference energy (the same map
the matrix mode uses), converted to MAD-X's canonical basis
(`x, px, y, py, t, pt` — `t = −cΔt`, `pt = ΔE/p₀c`, both pinned
against MAD-X `TRACK`).  MAD-X cannot move its reference energy along a
sequence, so an accelerating element's map is written about the entrance
momentum with the exit angles rescaled by `p₀/p_out` (the canonical,
symplectic form) and its reference gain emitted as `kick6` (a `pt`
shift on every particle); the warning says so.  MAD-X `TWISS`
symplectifies `MATRIX` elements, so a map that is not symplectic (a
non-Maxwellian synthetic field, a coarse fit) is flagged — `TRACK`
applies it verbatim.  Foils and electric steerers still fall under the
unsupported rule.

`pt` is a deviation about the *entrance* momentum, so the linearised
form is valid only while the accumulated gain stays small compared with
`p₀c`: measured with cpymad on the PIP-II MEBT–SSR2 line (2.1 → 10 MeV),
MAD-X `TWISS` fails once the accumulated `pt` reaches a few 10⁻², and
`TRACK`'s `pt` overflows before the end.  The exporter reports the total
linearised gain as a fraction of `p₀c` and escalates the wording at 10⁻²
and 0.3; for a whole linac export one section at a time from its own
entrance energy.  `ERROR_*` cards (an error study, not lattice elements)
are not exported in either format — MAD-X models tolerances through
`EALIGN`/`EFCOMP` — and the writer says so.

**Limits inherited from MAD-X.** A sequence has one `BEAM` energy: every
`k1`/`ks` is normalised by the *entrance* rigidity (the exact inverse of
the importer), `TWISS` does not follow the energy gain of an
accelerating line (`TRACK` sees it as `pt`), and `RFCAVITY` carries no
transverse RF (de)focusing — an accelerating deck therefore exports with
a one-line warning and MAD-X reproduces its optics only up to the first
cavity.  MAD-X sequences cannot step backwards, so a negative-length
drift refuses the export, as does a zero-length lattice.  Element names
are made MAD-X-legal (illegal characters, leading digits, reserved
command/class names such as `start` or `wire`, names longer than 40
characters, and a clash with the sequence name — the last two crash
MAD-X outright rather than erroring).

> Verified against MAD-X 5.09.03 through `cpymad`: every public example
> deck loads in MAD-X, MAD-X's own cumulative R-matrix for the exported
> line equals HELIX's transfer matrix (transverse 4×4, both proton and
> H⁻), and the bundled `examples/madx/*.madx` files survive a MAD-X →
> HELIX → MAD-X round trip with the same Twiss (`tests/io/
> test_madx_oracle.py`; the tests skip when `cpymad` is not installed).

## Importing MAD8 lattices (`.lat`, `.flat`, `.mad8`)

HELIX reads **MAD8 lattices** (`.lat` / `.flat` / `.mad8`): SAVELINE
flat files such as the PIP-II BTL and BAL exports, hand-written decks,
and MAD8 files written by other codes.  Use `Open Lattice…` in the GUI,
any CLI command, or:

```{.python data-needs="BTL2025v0703.lat"}
from linac_gen.io.mad8_parser import parse_mad8
lattice, meta = parse_mad8("BTL2025v0703.lat")   # → (Lattice, metadata)
```

**Language.**

- `!` comments and `&` continuations. As in MAD8, text after the `&` on the same line is ignored.
- `;` separates statements on one line.
- `COMMENT` … `ENDCOMMENT` blocks.
- `CALL, FILENAME=…` reads another file (the path is taken relative to the calling file first).
- `RETURN` and `STOP`.
- Parameters: `name := expr`, `name = expr`, `name: CONSTANT = expr` and `SET, name, expr`. They are evaluated when first used, so their order in the file does not matter. If a name is defined twice, the last definition wins and a warning lists it.
- Expressions: `+ − * / ^`, Fortran `D` exponents and `NAME[ATTR]` references to element attributes (also `BEAM[…]`).
  - Functions: the MAD8 set `SQRT LOG EXP SIN COS TAN ASIN ABS MAX MIN`, plus `ACOS ATAN SINH COSH TANH LOG10`.
  - Constants, with MAD8's values: `PI TWOPI DEGRAD RADDEG E EMASS PMASS CLIGHT`.
  - `RANF()`, `GAUSS()` and `TGAUSS()` take their mean value, with a warning. HELIX imports the nominal machine.
- Elements: `name: CLASS, attr=expr, …`.
  - MAD8 keyword abbreviations work (`QUAD`, `SEXT`, `RFCAV`, …).
  - Class inheritance works (`QF2: QF, K1=…`), and so do attribute changes (`QF, K1=…`).
  - Word attributes such as `TYPE=…` are kept out of the arithmetic.
- Beam lines: `name: LINE = (…)` with `-NAME`, `N*NAME`, nested groups `N*(A, -(B, C))`, and lines with arguments (`CELL(X, Y): LINE = (X, D, Y)` used as `CELL(QF, QD)`).
- `name: SEQUENCE[, REFER=…, L=…]` … `ENDSEQUENCE[, AT=…]` with `AT` / `FROM` placement (`FROM` counts from the earlier member's centre, as MAD-X does). The drifts between members are generated.
- `USE, name` selects the beam line. Without it, the importer takes the largest line that no other line references, and warns. `parse_mad8(..., line="NAME")` overrides both.
- Action commands (`TWISS`, `MATCH` … `ENDMATCH`, `SELECT`, `EALIGN`, …) are skipped and listed in one warning. Misalignment and field-error commands (`EALIGN`, `EFIELD`, …) are **not** applied. Model errors with the Error Study instead.

**Elements.** The element mapping follows the MAD-X table above, plus these MAD8 cases:

| MAD8 | HELIX |
|---|---|
| `KICKER` / `HKICKER` / `VKICKER` with zero kick | `Marker` + a `Drift` of the full length (the BTL/BAL corrector layout) |
| the same with a kick | `Steerer` between half-length drifts. `TILT` rotates the kick. |
| plain `MONITOR` (also `IMONITOR`, `WIRE`, `PROFILE`, …) | non-BPM `Marker` + body `Drift` |
| `H/VMONITOR` | BPM marker |
| `TILT` on a quadrupole | `skew_angle` (a bare `TILT` = 45°) |
| `TILT=±π/2` on a bend | a vertical bend (`hv=1`) |
| `OCTUPOLE` | thin `Multipole` (k3·L) between half-length drifts |
| `MULTIPOLE, KnL=…, Tn=…` | `knl[n] = KnL·cos((n+1)Tn)`, `ksl[n] = −KnL·sin((n+1)Tn)`. A bare `Tn` = π/(2(n+1)). |
| zero-angle `SBEND`/`RBEND` with `K1` | `Quadrupole` (exact: with no curvature the bend map is the quadrupole map) |
| `RFCAVITY` with `HARMON` | frequency = HARMON·βc/C, where C is the length of the line used |
| `RFCAVITY` with neither `HARMON` nor `FREQ` | a `Drift` of its length (the cavity has no frequency) |
| `LCAVITY` (`DELTAE` MeV, `PHI0` in turns, `FREQ` MHz) | `RFGap` with gain DELTAE·cos 2πPHI0. Magnets after it are converted with the local rigidity. |
| `LUMP, LINE=…` | the referenced line, expanded in place |
| anything else (`ELSEPARATOR`, `WIGGLER`, `BEAMBEAM`, `SROT`, …) | a `Drift` of its length, or a `Marker` when L = 0. One warning per class says the field is not modelled. |

A multipole's dipole term `K0L` is applied as an orbit kick about a
straight reference, as the HELIX MAD-X importer also does. MAD-X instead
bends the reference orbit by `K0L`. A file with a non-zero `K0L` gets a
warning that says so.

**Rigidity and charge sign.**  MAD strengths are normalized
(K1 = (q/p)·∂B/∂x). HELIX stores the lab-frame gradient, so the
conversion is **G = sign(q)·K1·|Bρ|**. For H⁻ every gradient sign
flips relative to a proton import. Bρ comes from, in this order:

1. a `BEAM` statement that names the particle or its energy: `PARTICLE` and/or `MASS` + `CHARGE`, with `ENERGY` (total, GeV), `PC` or `GAMMA`. MAD8's defaults apply: positron, 1 GeV. A `BEAM` that carries only emittances or `NPART` is not a rigidity source. `BEAM[PC]`, `BEAM[GAMMA]` and the other derived quantities can be referenced in expressions.
2. the `brho=` argument.
3. a `BRHO := …` parameter in the file. Its species is the `species=` argument, else the fallback beam's species, else H⁻. For H⁻ it is refused when it implies an impossible energy (above 20 GeV). Some BAL exports write `P0/C*1.0E11` with P0 in MeV/c and c in cm/s, which is 1000 times too large.
4. the **fallback beam**: the project's beam, or the Beam tab when a bare lattice is opened in the GUI. It is used only when the file declares no usable rigidity, and a warning (plus a note in the GUI status bar) says so.

With none of these the import **fails loudly**: a wrong Bρ would
silently mis-scale every magnet. The CLI only has a fallback when it is
given a project (`.lgproj`), and a scan converts the magnets with the
project's beam as saved, never with the per-point `--beam` overrides.
At GUI start-up a restored lattice uses the last session's beam. A `BEAM` particle that
HELIX does not model (electron, positron, antiproton, …) keeps its own
rigidity for the magnets, and the tracking reference becomes a proton
at that rigidity. The magnetic optics then match MAD8, and a warning
states the substitution. After an `LCAVITY` the magnets follow the real
particle's momentum; a tracked proton through the same cavity would not,
and a warning says so. A warning always states the reference used,
so set the Beam tab to match before running.

**Checked against MAD-X.** On every corpus lattice MAD-X can read, the
imported transverse transfer matrix matches MAD-X's reading of the same
file to ≤ 6·10⁻⁸, and to 10⁻¹⁰ or better on most. For this comparison
RF voltages and kicks are zeroed in both codes: MAD-X's twiss neither
accelerates the reference nor linearises about a kicked orbit the way
HELIX's matrix does. The files checked are the Fermilab Main Injector
(fractional tunes 0.4252865423 / 0.4152846703 in both codes), the CSNS RCS, two
SNS ring decks, the PIP-II BAL, BTL2022, an electron-lens FODO series
and a 10 GeV electron ring written with keyword abbreviations and
nested groups. `tests/io/test_mad8_parser.py` carries the MAD-X
comparisons; they skip when `cpymad` is not installed.

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

## Importing Elegant lattices (`.lte`)

HELIX also reads **Elegant lattice-element files** (`.lte`) —
`Open Lattice…` in the GUI (filter *Elegant (\*.lte)*), any CLI
`<input>`, or:

```{.python data-needs="ring.lte"}
from linac_gen.io.elegant_parser import parse_elegant
lattice, meta = parse_elegant("ring.lte", species="H-",
                              w_kin=1000.0, frequency=352.21)
```

**Supported:** `!`/`#` comments, `&` and trailing-comma continuation;
`line=(...)` beamlines with leading-`-` reversal and `N*elem`
repetition; element templates (a type token that names a defined
element inherits its attributes) and `name[prop]=value` overrides.
Lengths in metres, angles in radians, `k1`/`k2` in MAD-normalised
units, cavity `volt` in V, `phase` in degrees (90° = crest), `freq` in
Hz — the unit and rigidity conversions are shared with the MAD-X
importer above.  An order-1 `ematrix` imports verbatim as a
`MatrixElement` (explicit 6×6 map; the longitudinal block is kept in
Elegant's `(t, p)` basis — no basis change is applied, and the importer
warns in `metadata["warnings"]` whenever that block is non-trivial:
a non-identity longitudinal 2×2, transverse↔longitudinal coupling, or
a longitudinal offset.  A purely transverse map imports exactly,
without a warning).

**Degrades explicitly, never silently:** CSR/LSC drifts → `Drift`
(+warning); `charge`/`wake` → `Marker` (+warning); unknown types →
`Drift` (+warning).  `.lte` files carry no beam energy (it lives in
the run file), so pass `species` / `w_kin` / `frequency` — the
strength conversion is self-consistent for any value.

> Import-only: with an `.lte` lattice loaded, **Save** routes to
> *Save As…* (TraceWin `.dat`) and never overwrites the Elegant
> source.

## Importing Bmad, SciBmad and PALS lattices (via lattix)

Bmad lattices (`.bmad`), SciBmad lattices — Julia source for the
`Beamlines.jl` package (`.jl` / `.scibmad`) — and PALS documents
(`.pals.yaml` / `.pals.yml` / `.pals.json`, the Particle Accelerator
Lattice Standard) open like any other lattice: `Open Lattice…` in the
GUI (filters *Bmad*, *SciBmad*, *PALS*), any CLI `<input>`, the
*Import an existing lattice* page of the New Project wizard, the
assistant's `load_lattice` tool, or:

```{.python data-needs="examples/scibmad/fodo.jl examples/bmad/fodo.bmad" data-requires="lattix"}
from linac_gen.io.lattix_bridge import parse_with_lattix
lattice, meta = parse_with_lattix("examples/scibmad/fodo.jl")   # or .bmad / .pals.yaml
lattice, meta = parse_with_lattix("examples/bmad/fodo.bmad", species="proton")
for w in meta["warnings"]:                                       # the fidelity ledger
    print(w)
```

These formats are read by the **lattix** translator, an optional
dependency that is not yet published: install its checkout into the same
environment, or point `HELIX_LATTIX_ROOT` at it (a `Translator` checkout
next to the HELIX repository is found without either).  Without it the
import fails with a message that says so;
nothing else in HELIX depends on it.  lattix parses the file into its own
code-neutral representation and its HELIX adapter builds the `Lattice`
from that — the deck is **parsed, never run**: a SciBmad file is read by
a pure-Python parser of the restricted Julia subset lattice files use
(`using Beamlines`, `@elements begin … end`, element constructors with
keyword arguments, `Beamline`/`Branch`, simple arithmetic and `DefExpr`);
Julia is not needed and statements outside that subset are reported by
line, not executed.

**Units and conventions.**  Lengths in metres, angles in radians.
Normalised strengths (`k1`, `Kn1`, `Ksol`, …) are converted to HELIX
gradients with the **signed rigidity of the deck's own reference
particle** (Bmad `parameter[particle]`/`p0c`, Beamlines `species_ref` +
`E_ref`/`pc_ref`, PALS `ReferenceP`); `meta["warnings"]` ends with the
same *set the Beam tab to match* note the MAD8 importer prints.  Bmad
`lcavity`/`rfcavity` phases and Beamlines `phi0` (radians, crest = 0)
become the equivalent raw RF phase of a HELIX `GAP`; a thick cavity
becomes drift–gap–drift; a Bmad `patch` becomes a marker; the
reference-energy jumps of a SciBmad `Branch` (`dE_ref`) follow the
cavities.  Every approximation and every dropped element is listed by
name in `meta["warnings"]` (and `meta["fidelity"]` keeps both of lattix's
reports in full) — the same ledger the CLI echoes as `parse warning:` and
the GUI counts in the status bar.

**Species.**  HELIX tracks protons, H⁻ and deuterons.  A deck whose
reference particle is something else — Bmad's default is the
**positron** — is refused with a message naming the deck's reference;
re-open it with `species="proton"` (`h-`, `deuteron`) and the strengths
are converted for that particle at the deck's momentum.  H⁻ flips the
sign of every normalised strength, as the deck's own code would.

The three bundled examples `examples/bmad/fodo.bmad`,
`examples/scibmad/fodo.jl` and `examples/pals/fodo.pals.yaml` are the
FODO cell of `examples/madx/fodo.madx` written out by lattix; imported,
each gives the same HELIX lattice as the MAD-X importer.  A SciBmad file
carries its reference particle on a leading `Marker(species_ref = …,
E_ref = …)`; that marker is imported as a zero-length `Marker` (so the
`.jl` version of the cell has one element more than the `.bmad` one, and
element indices are shifted by one).

> Import-only: with a Bmad / SciBmad / PALS lattice loaded, **Save**
> routes to *Save As…* (TraceWin `.dat`) and never overwrites the
> source; the New Project wizard materialises such a deck as
> `<name>.dat` inside the project.

## Importing TraceWin project settings (`.ini`)

A TraceWin project is a `.dat` deck **and** a `<project>.ini` — the
binary options file that holds what was typed into TraceWin's *Main*
and *Beam* panels: particle, energy, frequency, current, macro-particle
count, the normalised emittances and the Twiss parameters of both input
beams, the PICNIC meshes.  HELIX decodes it, so a deck can start from
the beam it was designed for instead of the stock `BeamConfig`
default (3 MeV proton) — the mistake behind more than one "why is the
800 MeV line lost at 2 MeV" run.

### When to import it, and how

Use the `.ini` whenever you have a TraceWin **project** — a deck *and*
its options file, typically a folder handed over from a TraceWin user
or exported from TraceWin — and you want HELIX to start from the beam
that deck was designed for.  Do not look for one when you only have a
bare `.dat` (a deck written by hand, by HELIX, or converted from
MAD-X/MAD8/Elegant carries no `.ini`), and do not expect it to be read
when you open a `.lgproj`: a saved project already holds the beam you
chose, and it always wins.

| Situation | Route | What happens |
|---|---|---|
| You open a TraceWin `.dat` in the GUI and `<deck>.ini` sits next to it | **File → Open Lattice…** | HELIX asks *Import its beam?* — **Yes** fills the Beam tab and makes it the session beam; **No** keeps the current beam ([Lattice tab](../10_gui/02_lattice_tab.md)) |
| The deck is already open, or the `.ini` lives elsewhere / under another name | **Beam tab → Import TraceWin .ini…** | pick the file; the form is replaced and the project marked dirty ([Beam tab](../10_gui/03_beam_tab.md#importing-a-tracewin-ini-project-beam)) |
| You are creating a HELIX project from a TraceWin deck | **File → New Project… → Import an existing lattice** | tick *Also import the beam from the TraceWin .ini next to the deck*; the `.ini` is copied with the deck and the `.lgproj` is written with its beam ([workflows](../10_gui/08_workflows.md#i-have-a-tracewin-project-dat-ini-and-want-its-beam-in-helix)) |
| You want a HELIX project file from the shell | `python -m linac_gen twini deck.ini --lgproj` | writes `deck.lgproj` next to the `.ini`, pointing at `deck.dat` |
| You want one headless run / scan / study without a project | `python -m linac_gen run deck.dat --tracewin-ini …` (also scan, batch, twiss, backtrack, failures, mo, export, orm) | the beam is read for that command only; `--energy` etc. still override |
| You only want to see what the file says | `python -m linac_gen twini deck.ini --report` or the assistant's `inspect_tracewin_ini` | nothing is loaded or written |

**Step by step (command line):**

1. Look before you load — the report lists what will be applied, what
   is recorded only, and what the file does not describe:

    ```text
    python -m linac_gen twini deck.ini --report
    ```

    Check the particle row (an ion or a `My_particle` slot has no HELIX
    species — pass `--species`), the energy and frequency, and the
    *Warnings* block (a `FREQ` mismatch with the deck, a DC beam, a
    renamed file).
2. Either turn the pair into a HELIX project once —

    ```text
    python -m linac_gen twini deck.ini --lgproj          # deck.lgproj next to the .ini
    ```

    — and work from `deck.lgproj` from then on (GUI **File → Open
    Project…**, or any CLI command with the `.lgproj` as input); or use
    the `.ini` for a single command:

    ```text
    python -m linac_gen run deck.dat --tracewin-ini --mode envelope --out runs
    ```

3. Set what the `.ini` does not carry: the distribution type and
   cut-off, a DC beam's energy spread, centroid offsets.  On the CLI
   `--beam distribution=gaussian --beam cutoff=3`; in a project, edit
   the Beam tab and **Save Project**.
4. Run.  The first `FREQ` card of the deck is compared with the beam
   frequency and any difference is warned about; HELIX rescales the
   longitudinal plane at the card.

**After an import, check** the Beam tab (or the printed summary): the
species, energy and current are the project's; `alpha_z` already
carries HELIX's sign (do not flip it again); a DC project shows
**Continuous beam** ticked; a value outside the form's range (say
5 000 000 particles) was clamped and reported; the console lists every
warning.  Then **Save Project** — until you do, the imported beam lives
only in the session.

### Commands

```text
python -m linac_gen twini deck.ini                 # the converted beam + warnings
python -m linac_gen twini deck.ini --report        # every decoded slot: applied / recorded / unknown / not decoded
python -m linac_gen twini deck.ini --json          # machine-readable
python -m linac_gen twini deck.ini --lgproj        # write deck.lgproj next to the .ini
python -m linac_gen twini deck.ini --lgproj my.lgproj --lattice other.dat --beam 2 --species H- --force
```

`twini` prints the converted beam, and with `--lgproj` writes a project
file — the same format the GUI saves — that points at the sibling
`deck.dat` (or `--lattice`).  The deck is parsed for the check that its
first `FREQ` card matches the beam frequency.  The same conversion is
one call in Python:

```{.python data-needs="tests/io/fixtures/tracewin_ini/ads.ini"}
from linac_gen.io.tracewin_ini import load_tracewin_ini, to_beam_config, report
ini = load_tracewin_ini("tests/io/fixtures/tracewin_ini/ads.ini")
beam, warnings = to_beam_config(ini)      # beam 1 → BeamConfig in HELIX units
print(beam.species, beam.energy, beam.emit_nx, beam.alpha_z, beam.beta_z)
print(report(ini))                        # applied / recorded / unknown / not decoded
```

**Precedence.** A `.lgproj` always wins: its beam is what you saved, and
the `.ini` is read only when you ask for it (`twini`, or the
`--tracewin-ini` option on the CLI runners).  Nothing reads a `.ini`
silently.  Scalar overrides (`--energy`, `--freq`, `--species`) still
apply on top of the imported beam, with a warning: `emit_z` and
`beta_z` were converted at the `.ini` energy, frequency and species and
are not re-derived for the overridden values.

**Applied to the beam** (every slot *verified* against at least two
independent projects, the LightWin `ads.ini` ↔ σ-matrix identity and a
TraceWin run — see [Appendix G](../appendices/G_tracewin_ini_format.md)
for the offsets and the evidence): species from the particle-table row
(proton, deuteron, H-), energy, frequency, current, `n_particles`,
`emit_nx`/`emit_ny` (normalised, π mm mrad), `alpha_x`/`beta_x`,
`alpha_y`/`beta_y`, and the longitudinal plane converted into HELIX's
(Δφ, ΔW) units:

| HELIX | from the `.ini` |
|---|---|
| `emit_z` (π deg MeV) | `eps_z1` · 360 · f · mc² / c |
| `beta_z` (deg/MeV) | `betz1` · 360 · f / ((βγ)³ · c · mc²) |
| `alpha_z` | **−`alpz1`** — the usual TraceWin sign flip, applied once, here |

For the ADS sample: 3e-7 π m rad → 0.0338 π deg MeV, 2.5932 m/rad →
37.11 deg/MeV, α_z +0.17661 → −0.17661.

**Identified, not applied:** thread count and the PICNIC r/z and x/y
meshes are *probable* slots and, more to the point, not HELIX's
space-charge grid — they are printed in the report and recorded under
`convergence.tracewin_ini` of a written project, never applied.

**Degrades explicitly, never silently:** a DC beam (`eps_z1` = 0, LEBT
projects) → `continuous = True` with a warning that the energy spread
is not decoded; a particle row with no HELIX species (an ion, a
`My_particle` slot) → your `--species` or `proton`, with a warning that
names the row and says the `.ini` rest mass was not used; a bunched beam
with `betz1` = 0, an unpopulated beam 2, a zero transverse emittance →
refused; a file of a size other than the two known layouts (31 624 and
44 824 bytes) or a wrong magic → refused with the known sizes named (a
layout tag that disagrees with the size — TraceWin's own `*.old.ini` —
is a warning, not a refusal).  The distribution type,
duty cycle, energy spread, centroid offsets and `.dst` path are *not
decoded*; the report lists them so you set them yourself.

## Cross-references

* [Keyword cheatsheet](../appendices/B_keyword_cheatsheet.md) —
  one-page printable.
* [Appendix G — the TraceWin `.ini` format](../appendices/G_tracewin_ini_format.md).
* [Element catalog](../03_elements/00_overview.md).
* [Matching SET/ADJUST](../07_matching/02_set_adjust.md).
* [Errors directives](../08_errors/02_error_directives.md).

← [Python API](01_python_api.md) ·
[Continue to HELIX extensions →](03_lg_extensions.md)
