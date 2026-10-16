# CLI: `twiss`

Matched Twiss parameters for a lattice — straight from the shell, no GUI.
Two jobs in one subcommand:

* the **whole-lattice periodic Twiss** of a true periodic structure, and
* the **input matched Twiss** of a one-pass transfer line — the periodic
  Twiss of a single FODO cell, back-propagated to the lattice entrance;
  for a deck with an **RFQ**, the RFQ's matched input at its entrance
  ([RFQs](#rfqs)).

```
python -m linac_gen twiss <input> [options]
```

`<input>` is a `.dat` / `.madx` lattice or a `.lgproj` project.

## Options

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `--threads` | N \| `all` | `$HELIX_THREADS`, else `$OMP_NUM_THREADS`, else all | CPU threads for the space-charge FFT, field maps and charge deposition; results are identical for any number ([CPU threads](15_threads.md)) |
| `--mode` | `whole` \| `cell` | `cell` | which matched Twiss to compute (see below) |
| `--cell` | int | `0` | which detected cell to use, 0-based (`cell` mode): RFQ periods first, then FODO cells |
| `--list-cells` | flag | off | list the detected cells (RFQ periods and FODO cells) and exit |
| `--current` | mA | — | match with space charge at this beam current (`cell` mode): an RFQ for a DC input beam, a FODO cell for the bunched beam the GUI's cell matching uses.  The emittances come from the project, `--tracewin-ini`, or `--emit-nx`/`--emit-ny` |
| `--emit-nx`, `--emit-ny` | mm mrad | — | normalised rms emittances for `--current`; override the input's beam (required with `--current` on a bare lattice without `--tracewin-ini`) |
| `--cell-start` | int | — | manually specify the inclusive start element index of the periodic cell (overrides the `--cell` auto-pick when auto-detection doesn't find the period you want) |
| `--cell-end` | int | — | manually specify the inclusive end element index; required if `--cell-start` is set |
| `--tracewin-ini [INI]` | flag or path | — | bare lattice only: energy, species, frequency and Twiss from beam 1 of the deck's TraceWin `.ini` (`<deck>.ini`, or `INI`); `--energy` etc. still override (warned: `emit_z`/`beta_z` are not re-derived); refused with a `.lgproj` |
| `--energy` | MeV | — | kinetic energy — **required for a bare `.dat`** (or `--tracewin-ini`) |
| `--freq` | MHz | — | beam frequency |
| `--species` | `proton` \| `deuteron` \| `H-` | — | particle species |
| `-q`, `--quiet` | flag | off | print only the four numbers `αx βx αy βy` |
| `--disp` | flag | off | with `-q`: append the four periodic-dispersion numbers `Dx Dx' Dy Dy'` (mm/MeV, mrad/MeV).  Without `-q` it is a no-op — dispersion rows (`D_x … eta_x`) already print whenever the lattice bends |

A `.lgproj` project already carries the beam energy / frequency /
species. For a bare lattice you must supply at least `--energy` (and
usually `--species`), since a TraceWin `.dat` stores no beam energy.

## The two modes

### `--mode cell` (default) — transfer-line input match

A beam transfer line is **not** periodic end-to-end, but its FODO cells
*are* transversely periodic. The correct *input* Twiss is therefore the
periodic Twiss of one FODO cell — **back-propagated to the lattice
entrance** so it can serve as the starting beam at `s = 0`.

`twiss` finds the FODO cells automatically (it needs at least three
focusing elements — see [cell detection](#cell-detection)),
computes the periodic Twiss of cell `--cell`, and transports it
backwards to the entrance. This is the right number to seed a
transfer-line simulation with.

### `--mode whole` — periodic ring

The periodic Twiss of the **whole lattice** treated as one period —
`cos μ = ½ Tr M` on the end-to-end transfer matrix. Correct for a true
periodic structure (a ring, or a single cell evaluated on its own); it
is **not** the correct input for a multi-cell transfer line — use
`cell` for that.

## Coupled lattices (HWR cryomodules, solenoid focusing)

`twiss` **auto-detects transverse x↔y coupling** (solenoid focusing,
skew quads, HWR-style cryomodules) in *either* mode. When it does, the
standard 2×2-block Courant–Snyder extraction is invalid, so it routes to
the eigenvector / Wolski method on the full 4×4 transverse map and the
report gains a `COUPLED` line with the two normal-mode tunes:

```
$ python -m linac_gen twiss examples/emittance_min/pipii_hwr_cm_match.dat \
    --mode cell --energy 2.1 --freq 162.5 --species proton
[twiss] matched input Twiss — FODO cell 0 ... back-propagated to s=0
  alpha_x = +0.412300    beta_x = 0.884512 m    mu_x = 71.2204 deg
  alpha_y = +0.398880    beta_y = 0.901133 m    mu_y = 71.5582 deg
  COUPLED (eigenvector method) — normal-mode tunes mu_1 = 71.21 deg, mu_2 = 71.56 deg
  per-plane alpha/beta above are projections of the matched 4×4 Σ
```

The `mu_x` / `mu_y` values are the **principal** phase advance folded to
[0, 180°].  When a cell exceeds 180°, the folded value alone is ambiguous,
so `twiss` appends the oriented (mod-360°) branch:

```text
  alpha_x = +1.918003    beta_x = 13.814052 m    mu_x = 89.7421 deg (oriented 270.2579)
```

so neither convention is lost.  `-q` (quiet) output is unchanged.

Two things to know when you see `COUPLED`:

* The per-plane `alpha`/`beta` are **projections** of the matched 4×4 Σ
  (what an x-only or y-only diagnostic would read), **not** decoupled
  Courant–Snyder values. `mu_1`/`mu_2` are the true normal-mode tunes.
* An accelerating section (βγ growing along the lattice) is not strictly
  periodic; `twiss` emits an `accelerating-section deviation …` note and
  returns the closest-to-periodic ("smooth") Σ. A truly unstable lattice
  (eigenvalues far off the unit circle) exits with code `1`.

For the **space-charge / dispersion-matched** input Twiss of a cell, pass
`--current` (it calls `find_sc_matched_input_twiss`, 8-state,
dispersion-aware — the GUI Matching Dialog's **Compute SC-Matched Twiss**)
or use the [Python API](../07_matching/03_python_api.md#periodic-and-matched-input-twiss-helpers).

## RFQs

A deck with RFQ cells is matched to the RFQ's first regular focusing
period — its first two regular cells of opposite sign, after the radial
matching (±3) and transition (±4) cells — carried back to the **RFQ
entrance**.  It is the default cell (`[0]` in `--list-cells`):

```
$ python -m linac_gen twiss examples/rfq_demo/rfq_demo.lgproj
[twiss] matched input Twiss at the RFQ entrance (RFQ_001, element 1) — RFQ period RFQ_002 + RFQ_003 (elements 2..3)
  alpha_x = -0.179174    beta_x = 0.008089 m    mu_x = 37.1813 deg
  alpha_y = +2.299155    beta_y = 0.040944 m    mu_y = 37.1948 deg
  caution: the match depends on the RF phase: re-computed 30-60° from the synchronous phase, it changes by a mismatch factor of up to 2.65 (the radial matching section is 0.5 focusing periods long), so no single input ellipse suits the whole DC beam — check this input with a multiparticle run
```

With space charge (`--current`) the input beam is DC, as the beam
entering an RFQ is; the phase advances printed are the zero-current
ones:

```
$ python -m linac_gen twiss examples/rfq_demo/rfq_demo.lgproj --current 15
[twiss] space-charge (15 mA) matched input Twiss at the RFQ entrance (RFQ_001, element 1) — RFQ period RFQ_002 + RFQ_003 (elements 2..3), DC input beam
  alpha_x = -0.215758    beta_x = 0.008443 m    mu_x = 37.1813 deg
  alpha_y = +2.460487    beta_y = 0.043381 m    mu_y = 37.1948 deg
  the phase advances are the zero-current values
```

* **Behind a LEBT** the elements upstream are not back-propagated
  through: the result is the target the LEBT must deliver at the RFQ
  entrance, and a `note:` line says how far into the lattice the RFQ
  starts.
* **The caution line** appears when the match depends on the RF phase.
  It is the synchronous particle's match, while a DC beam enters at every
  phase; re-computed 30° and 60° either side, it should change by a
  mismatch factor below 0.1.  The demo's radial matching section is a
  single cell (half a focusing period).  Check a flagged input with a
  multiparticle run ([RFQ_CELL → Matched input Twiss](../03_elements/09_rfqcell.md#matched-input-twiss)).
  With `-q` it goes to stderr as `warning:`.
* `--cell-start`/`--cell-end` inside an RFQ keep the manual meaning
  (back-propagated to `s = 0`); with `--current` they start from a DC
  beam too.

## Cell detection

`--list-cells` reports every cell `twiss` found, with the element-index
range of each.  A deck with an RFQ lists its RFQ period first
(`1 cell candidate(s): [0] RFQ period RFQ_002 + RFQ_003, elements 2..3
(matched at the RFQ entrance, RFQ_001, element 1)`); without one the
list is the FODO cells:

```
$ python -m linac_gen twiss btl.lgproj --list-cells
12 FODO-cell candidate(s):
  [  0]  elements 11..131
  [  1]  elements 131..251
  ...
```

A "focusing element" for detection purposes is a `Quadrupole`, a
`Solenoid`, **or** a `FieldMap` / `FieldMap3D` that classifies as a
solenoid (kb≠0, ke=0 — the HWR-cryomodule case).  A cell spans one
full focusing period, focuser *k* → focuser *k+2*, so each candidate
contains one focusing and one defocusing element; mixing types (e.g.
`[QUAD, drift, SOL, drift, QUAD]`) is allowed.

Pick one with `--cell N`. Detection needs an RFQ or at least three
focusing elements; on a lattice with neither (`no cell detected`), use
`--mode whole` or specify the period manually with `--cell-start` /
`--cell-end`.

## Output

By default, a labelled report:

```
[twiss] matched input Twiss — FODO cell 0 (elements 11..131) back-propagated to s=0
  alpha_x = +1.918003    beta_x = 13.814052 m    mu_x = 89.7421 deg
  alpha_y = -0.916102    beta_y = 5.387298 m    mu_y = 90.4115 deg
```

With `--quiet`, just the four numbers, whitespace-separated
(`alpha_x beta_x alpha_y beta_y`), ready to pipe into a script:

```
$ python -m linac_gen twiss btl.lgproj -q
1.91800300e+00 1.38140520e+01 -9.16102000e-01 5.38729800e+00
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success (also when `--list-cells` just prints) |
| `1` | the Twiss computation failed — e.g. an unstable / non-periodic cell |
| `2` | bad arguments — input not found, no cell in `cell` mode, `--cell` out of range, a bad beam spec, or `--current` that is negative, used with `--mode whole`, or lacks the emittances |

## Examples

```sh
# Input matched Twiss of a transfer line (the project carries the beam)
python -m linac_gen twiss examples/pipii/btl/btl.lgproj

# Same, but from a bare lattice — supply the beam on the command line
python -m linac_gen twiss btl.dat --energy 800 --species H-

# Inspect the detected cells, then match on the third one
python -m linac_gen twiss btl.lgproj --list-cells
python -m linac_gen twiss btl.lgproj --cell 2

# Whole-lattice periodic Twiss of a single periodic cell
python -m linac_gen twiss cell.dat --mode whole --energy 100

# Quiet — capture the four numbers in a shell script
read ax bx ay by < <(python -m linac_gen twiss btl.lgproj -q)

# RFQ: matched input at the RFQ entrance, zero current and 15 mA
python -m linac_gen twiss examples/rfq_demo/rfq_demo.lgproj
python -m linac_gen twiss examples/rfq_demo/rfq_demo.lgproj --current 15

# RFQ in a bare deck: give the beam, and the emittances for --current
python -m linac_gen twiss rfq.dat --energy 0.035 --species H- --freq 162.5 \
    --current 5 --emit-nx 0.11 --emit-ny 0.11
```

## Cross-references

* [Matching → Overview](../07_matching/01_overview.md) — the theory:
  periodic Twiss, Courant–Snyder extraction, transfer-line input matching.
* [Matching → Python API](../07_matching/03_python_api.md) —
  `find_periodic_twiss`, `find_fodo_cells`, `find_matched_input_twiss`
  and the RFQ helpers `find_rfq_periods` / `find_rfq_matched_input_twiss`
  called directly.
* [Batch-mode overview](06_batch_cli.md) — the other `python -m linac_gen`
  subcommands.

← [CLI: batch](09_cli_batch.md) ·
[Continue to Matching → Overview →](../07_matching/01_overview.md)
