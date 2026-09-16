# Appendix G — The TraceWin `.ini` options file

TraceWin stores a project's beam and run settings — everything you type
into its *Main* and *Beam* panels — in `<project>.ini`, a binary dump of
its options structure.  The format is undocumented.  This appendix
records what HELIX's reader (`linac_gen/io/tracewin_ini.py`) knows about
it, how each field was established, and how to extend the map.  The user
side of the feature (opening a deck with its `.ini`, the `twini`
converter, the GUI prompt) is in
[Importing TraceWin project settings](../06_running/02_tracewin_dat.md#importing-tracewin-project-settings-ini).

## Two layouts, one prefix

| size (bytes) | generation | notes |
|---|---|---|
| 31 624 | TraceWin ≈ 2017 | LEBT/RFQ-era Fermilab projects |
| 44 824 | TraceWin 2019 and later | current projects, the LightWin sample |

The file starts with the ASCII magic `TraceWin_options_file`; the
`int32` at `0x68` holds the options-struct size of the TraceWin that
last touched the file (31 624 or 44 824).  That tag is informative, not
authoritative: when a newer TraceWin upgrades a project it keeps the
old file as `<project>.old.ini` at the *old* size but writes the *new*
tag into it (seven such files on the reference machine).  The two
generations are prefix-identical — a project saved in both differs in
a single byte inside the common part — and the newer layout appends
13 200 bytes.  The reader therefore keys the layout on the file size
alone, reports a differing tag as a warning, and refuses any other
size with a message naming the known sizes, never guessed.

## Field map

Offsets are bytes from the start of the file; all numbers are
little-endian.  **verified** = the decoded value reproduced the
project's known input in at least two independent projects;
**probable** = the slot is where the C struct implies (a beam-2 twin of a
verified slot, or a constant plausible value) but no project varied it.
Probable slots are reported and recorded, never applied to a beam.

| offset | type | name | unit | status |
|---|---|---|---|---|
| `0x0859` | string | last project directory | | probable |
| `0x0c41` | string | last project directory (2nd slot) | | probable |
| `0x17f9` | string | project name (the `.dat` stem) | | verified |
| `0x2ed0` | int32 | `nbr_part1` | | verified |
| `0x2ed4` | int32 | `nbr_part2` | | probable |
| `0x2ed8` | int32 | `particle_index1` (row of the particle table) | | verified |
| `0x2edc` | int32 | `particle_index2` | | probable |
| `0x2ee8` | int32 | `nbr_thread` | | probable |
| `0x2efc` | int32 | `picnic_r_mesh` | | probable |
| `0x2f00` | int32 | `picnic_z_mesh` | | probable |
| `0x2f04` | int32 | `picnic_xy_mesh` (x) | | probable |
| `0x2f08` | int32 | `picnic_xy_mesh` (y) | | probable |
| `0x2f24` | f64 | `freq1` | MHz | verified |
| `0x2f34` | f64 | `current1` | A | verified |
| `0x2f44` | f64 | `energy1` | eV | verified |
| `0x2f54` | f64 | `etnx1` (rms, normalised) | π m rad | verified |
| `0x2f64` | f64 | `etny1` (rms, normalised) | π m rad | verified |
| `0x2f74` | f64 | `eps_z1` (rms, normalised, = βγ·ε_zδ) | π m rad | verified |
| `0x30dc` | f64 | `alpx1` | | verified |
| `0x30e4` | f64 | `betx1` | m | verified |
| `0x30fc` | f64 | `alpy1` | | verified |
| `0x3104` | f64 | `bety1` | m | verified |
| `0x311c` | f64 | `alpz1` in the (z, z′ = dz/ds) plane | | verified |
| `0x3124` | f64 | `betz1` = γ²·β_zδ | m/rad | verified |
| `0x31fc` | 12 × 44 B | particle table | | verified |

Beam-2 twins of the scalar slots sit 8 bytes after their beam-1 slot
(`freq2` at `0x2f2c`, …) and the Twiss twins 16 bytes after
(`alpx2` at `0x30ec`, …); all of them are *probable* because none of
the sample projects populates a second beam.

### Particle table

Twelve 44-byte records at `0x31fc`: `f64` rest mass in eV, `int32`
charge state, `char[32]` name.  Rows 0–6 are TraceWin's built-ins
(Positron, Electron, Proton, H-, Deuton, H2+, H3+); rows 7–11 are the
user-defined slots and may hold ions, `My_particle` placeholders or
leftovers from another project.  `particle_index1` selects the row.
HELIX maps a row to one of its three species by charge sign and rest
mass (within 0.5 MeV): proton, deuteron, H-.  Any other row is
unrepresentable; the reader warns, names the row and falls back to the
species you chose (or `proton`), and the report says the `.ini` rest
mass was not used.

### Not decoded

These TraceWin options exist in the file but their offsets are not
located; they are listed in every report as *not decoded* and never
guessed: the distribution type, duty cycle, energy spread
(`spreadw1`/`dw1`), the centroid offsets `x1 … zp1` (one 2017-layout
LEBT file shows five small doubles at `0x3024`–`0x3044` that may be
them), `dst_file1`/`use_dst_file`, `part_step`, `random_seed`, `vfac`
and the loss/emittance limits.  `mass1`/`charge1` come from the
particle table row instead.

## How the values were established

* **Value matching.** 54 project files from two TraceWin generations
  (Fermilab PIP-II projects 2017–2026, the LightWin ADS sample) were
  scanned for the doubles and integers that reproduce each project's
  known inputs (frequency, current, energy, particle count, emittances,
  Twiss).  A slot is *verified* only when it matched in at least two
  independent projects.
* **σ-matrix identity.** LightWin ships the ADS beam both as `ads.ini`
  and as a 6×6 σ-matrix in `lightwin.toml`.  The transverse Twiss and
  normalised emittances agree exactly; the longitudinal block fixes the
  meaning of `eps_z1` (= βγ·ε_zδ, i.e. normalised in the (z, δ) plane)
  and of `betz1` (= γ²·β_zδ, the (z, z′) plane in m/rad), with
  `alpz1` equal to α_zδ.
* **A TraceWin run.** For a PIP-II SCL project the converted β_z
  (61.219 deg/MeV from `betz1` = 8 m/rad at 116.1 MeV, 804.96 MHz)
  reproduces the σ_φ = 7.455° that `tracewin.out` reports at the
  entrance to 1e-4.

## Conversion to HELIX units

With `f` the beam frequency in Hz, `mc²` the rest energy in MeV, `c` the
speed of light and βγ at the input energy:

| HELIX field | from the `.ini` |
|---|---|
| `energy` (MeV) | `energy1` × 1e-6 |
| `current` (mA) | `current1` × 1e3 |
| `emit_nx`, `emit_ny` (π mm mrad, normalised) | `etnx1`, `etny1` × 1e6 |
| `alpha_x`, `beta_x` (m), `alpha_y`, `beta_y` (m) | verbatim |
| `emit_z` (π deg MeV) | `eps_z1` · 360 · f · mc² / c — energy-independent |
| `beta_z` (deg/MeV) | `betz1` · 360 · f / ((βγ)³ · c · mc²) |
| `alpha_z` | **− `alpz1`** |
| `n_particles` | `nbr_part1` |

The α_z negation is the same TraceWin convention as everywhere else in
HELIX (see [Appendix E](E_migrating_from_tw.md)): HELIX's longitudinal
plane is (Δφ, ΔW) and a late particle has Δφ > 0 but z < 0.  It is
applied once, in the reader; nothing downstream flips it again.

Derivation: Δφ = −360 f z/(βc) and ΔW = mc² β² γ δ give
ε_ΔφΔW = ε_zδ · 360 f mc² βγ/c = `eps_z1` · 360 f mc²/c and
β_Δφ = σ_φ²/ε_ΔφΔW = β_zδ · 360 f/(β³ γ c mc²) with β_zδ = `betz1`/γ².
For the ADS sample (20 MeV proton, 100 MHz): `eps_z1` = 3e-7 →
`emit_z` = 0.03380 π deg MeV; `betz1` = 2.5932 → `beta_z` = 37.11 deg/MeV;
σ_φ = 1.120°, σ_W = 30.6 keV, both equal to the σ-matrix values.

A DC beam (`eps_z1` = 0, the LEBT projects) becomes `continuous = True`;
its energy spread is not decoded, so `dc_energy_spread_keV` keeps
whatever value the current beam has, and the warning says so.  A bunched
beam with `betz1` = 0 is refused rather than guessed.

The PICNIC mesh sizes and thread count are *identified but not applied*:
TraceWin's r/z and x/y meshes are not HELIX's space-charge grid
(`grid_nx` over ±5 σ), so they are recorded in the project file under
`convergence.tracewin_ini` and shown in the report for the record.

## Extending the map

`scripts/tracewin_ini_probe.py` is the read-only mapper the field table
was built with.  To locate a slot that is still *not decoded*:

1. In TraceWin, open a project, **change exactly one setting** (say the
   energy spread), and save the project under a new name in the *same*
   folder (moving the project rewrites the last-directory strings and
   adds noise to the diff).
2. Diff the two files:

    ```sh
    python3 scripts/tracewin_ini_probe.py diff before.ini after.ini:spreadw1
    ```

    Every run of differing bytes is printed with its `int32` and `f64`
    readings in both files.  A run inside a mapped slot is named
    (`known: energy1 (verified)`); an unmapped run comes with a
    ready-to-paste `FieldSpec("spreadw1", 0x…, "<d", …, "probable", …)`
    line.  More than `--max-runs` runs (default 4) exits 3 — you changed
    more than one thing.
3. Cross-check with the value you typed, in the unit TraceWin stores
   (energy in eV, current in A, emittances in π m rad):

    ```sh
    python3 scripts/tracewin_ini_probe.py find 20e6 tests/io/fixtures/tracewin_ini/ads.ini
    ```

    prints every offset whose reading equals the value (×1, ×1e3, ×1e6,
    ×1e-3, ×1e-6); with several files it keeps only offsets that match
    in every one.  `dump FILE.ini --raw` shows the reader's view of a
    file with every non-zero slot of the beam window.
4. Add the `FieldSpec` to `FIELDS` in `linac_gen/io/tracewin_ini.py`
   with status **probable**, and to the table above.  Promote it to
   **verified** only once the decoded value has reproduced the typed
   input in two independent projects; only verified slots may be
   applied to a beam in `to_beam_config`.  Pin it in
   `tests/io/test_tracewin_ini.py` the way the existing slots are: a
   byte-patched variant of the MIT fixture, never a round trip.

Note the packing: the options struct is 4-byte packed, so its doubles
sit at offsets ≡ 4 (mod 8) (`freq1` at `0x2f24`) — an 8-byte-aligned
scan misses the whole beam block.  The probe scans on the 4-byte grid.
