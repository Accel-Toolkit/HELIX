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
| `QUADRUPOLE` | `Quadrupole` (`k1` → gradient via Bρ; `tilt` → `skew_angle`, a bare `tilt` = π/4) |
| `SBEND` | `Edge` + `Dipole` + `Edge` (ρ > 0 with the direction in the angle's sign; `k1` → field index n = −k1·ρ²; `tilt=+π/2` / `−π/2` → vertical bend towards +y / −y; pole faces `e1`/`e2` → β = sign(angle)·e, because MAD-X's edge focusing is `h·tan e` with the signed curvature while HELIX's is `tan β/ρ`; `hgap`/`fint`/`fintx` → edge gap and K1, with MAD-X's default `fint = 0`) |
| `RBEND` | `Edge` + `Dipole` + `Edge` (edges pick up ±angle/2; same extras as `SBEND`). `l` is the straight chord under MAD-X's default `OPTION, RBARC=true` and the arc is `l·(θ/2)/sin(θ/2)`; `OPTION, RBARC=false` (LHC-style files) makes `l` the arc length |
| `KICKER` / `HKICKER` / `VKICKER` | `Steerer` (kick angles → ∫B·dl through the charge-signed Bρ, so an H⁻ deck kicks the same way MAD-X does) |
| `RCOLLIMATOR` / `ECOLLIMATOR` | `Aperture` (rectangular / circular; `xsize`, `ysize` are half-apertures) |
| `SEXTUPOLE` / `MULTIPOLE` | `Multipole` (`tilt` → `tilt_deg`, negated: `Multipole.tilt_deg` rotates in the opposite sense to MAD-X's `tilt` and to `Quadrupole.skew_angle`) |
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
| `Multipole` | `MULTIPOLE, knl={…}, ksl={…}, tilt` (negated, see the import row) |
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

## Importing MAD8 flat files (`.lat`)

HELIX also reads **MAD8 flat / SAVELINE lattices** (`.lat` / `.flat`),
the dialect used by the PIP-II BTL exports — `Open Lattice…` in the
GUI, or:

```{.python data-needs="BTL2025v0703.lat"}
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

## Cross-references

* [Keyword cheatsheet](../appendices/B_keyword_cheatsheet.md) —
  one-page printable.
* [Element catalog](../03_elements/00_overview.md).
* [Matching SET/ADJUST](../07_matching/02_set_adjust.md).
* [Errors directives](../08_errors/02_error_directives.md).

← [Python API](01_python_api.md) ·
[Continue to HELIX extensions →](03_lg_extensions.md)
