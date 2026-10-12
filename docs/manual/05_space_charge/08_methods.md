# Space-charge methods

The 3-D PIC solver has about ten settings: the charge-assignment
kernel, whether the grid is fixed or follows the beam, its size and
extent, the Green's function and how it is built, how often the field is
solved, the engine, and so on.  A **space-charge method** is a named
combination of them, so a run can be described in one word and the
combinations that belong together stay together.

## TL;DR

```bash
helix run project.lgproj --sc-method tsc_adaptive     # one run
helix run project.lgproj --sc-method list             # the methods
helix scan project.lgproj --sc-method cic_adaptive_reuse4 ...
```

```python
from linac_gen.core.config import SpaceChargeConfig
sc = SpaceChargeConfig(method="tsc_adaptive")          # its defaults
sc = SpaceChargeConfig(method="tsc_adaptive", nx=64)   # a bigger grid
```

* In a project file: `"sc_method": "tsc_adaptive"` in the
  `convergence` block.
* In the GUI: *Numerics → Space charge & PIC → Space-charge method*.
* For the assistant: the `sc_method` argument of `run_mp`, `run_train`
  and `parameter_scan`.

## The methods

Each method **pins** the settings that define it; changing one of them
makes it a different method.  Its **params** (grid size, extent, Green's
function and its build, what a `HELIX_SC_GRID` card does) have the
method's own defaults and can be changed without leaving the method.

| Method | What it is | Pinned | Default params |
|---|---|---|---|
| `cic_fixed` | CIC, fixed grid | CIC, fixed grid, numpy engine, field solved every kick | 48³, ±7σ, IGF, exact build, cards applied |
| `tsc_fixed` | TSC, fixed grid | TSC, fixed grid, numpy, every kick | 48³, ±7σ, IGF, exact build, cards applied |
| `cic_adaptive` | CIC, adaptive grid | CIC, adaptive grid, numpy, every kick | 48³, ±6σ, IGF, symmetric build, cards ignored |
| `tsc_adaptive` | TSC, adaptive grid | TSC, adaptive grid, numpy, every kick | as `cic_adaptive` |
| `cic_adaptive_reuse2`, `cic_adaptive_reuse4` | CIC, adaptive grid, field solved every 2nd / 4th kick | CIC, adaptive, numpy, every 2nd / 4th kick | as `cic_adaptive` |
| `tsc_adaptive_reuse2`, `tsc_adaptive_reuse4` | TSC, adaptive grid, field solved every 2nd / 4th kick | TSC, adaptive, numpy, every 2nd / 4th kick | as `cic_adaptive` |
| `differentiable_pic` | the PIC written in PyTorch ([Differentiable PIC](06_differentiable.md)) | torch engine, adaptive grid | 48³, ±6σ, IGF, CIC |
| `halo_pic` | HALO-PIC (experimental; needs a trained corrector) | HALO engine | the classic grid |
| `custom` | every setting chosen individually | — | the classic settings |

* **Fixed grid**: the grid is sized from the beam at the first
  space-charge kick and kept for the rest of the run (its Green's
  function is built once).
* **Adaptive grid**: the grid is resized to ±extent·σ at every kick, so
  the cells keep the same size relative to the beam (its Green's
  function is rebuilt whenever the cell spacing changes).
* **CIC / TSC**: cloud-in-cell (each particle shared among 8 grid nodes)
  or triangular-shaped cloud (27 nodes; smoother, less grid noise) —
  see [Kernels](03_kernels.md).
* **Classic settings**: what `SpaceChargeConfig()` gave before named
  methods: 96³, ±5σ, fixed grid, CIC, IGF, numpy engine.

`default` names HELIX's default method.  None is set, so `default`
means the classic settings.

## How a run's settings are decided

1. The method named on the command line (`--sc-method NAME` or
   `--sc method=NAME`), else the project's `sc_method`.
2. With neither, a project that stores any space-charge key runs exactly
   as it did before named methods existed (the classic settings, its
   keys, the command-line flags); a project with none gets the classic
   settings.
3. A project's `sc_method` decides the pinned settings; a stored key
   that contradicts one is overruled, with a note.  Its other keys set
   the params.
4. A command line naming the project's own method, or `custom`, keeps
   the project's settings; any other method starts from its own
   defaults and keeps the project's settings that belong to no method
   (DC kernel, CSR, FFT device, bunch-train images).  A note says so.
5. Then `--nx`, `--grid-extent`, `--kernel`, `--backend`, and last the
   generic `--sc NAME=VALUE` overrides.

A setting that contradicts the named method makes the run the method
the settings match: `--sc-method tsc_adaptive --sc grid_mode=fixed` runs
`tsc_fixed`.  The label always describes the settings that ran, also in
the results file.

## Settings the methods add

### `HELIX_SC_GRID` cards (`grid_card`)

`apply` honours a lattice's `HELIX_SC_GRID` card (it sets the grid
extent from that element on), `ignore` ignores it with one warning, and
`auto` ignores it on an adaptive grid and applies it on a fixed one.  On
an adaptive grid the card's wider extent would only make the cells
coarser (at ±20σ a 48-cell axis has 0.85σ cells), so the adaptive
methods use `auto`.  The cards a run reached and what was done with them
are recorded in the results file (`sc_grid_cards`).

### Green's-function build (`green_build`)

`exact` builds the integrated Green's function on every vertex of the
doubled grid; `symmetric` builds one octant and mirrors it.  The
Green's function is even in each coordinate, so the two agree to
rounding (about 5·10⁻¹³ of the peak at 48³); the mirrored one is exactly
even.  It matters for adaptive grids, which rebuild it often; the
adaptive methods use `symmetric`.

### Field solved every K-th kick (`reuse_every`)

With `reuse_every = K > 1` (adaptive numpy grid only), the field is
solved on every K-th kick.  On the kicks between, the last solved field,
stored in units of the beam's rms sizes, is rescaled to the beam's
current centre, rms sizes and charge — one gather instead of a deposit
and a Poisson solve.  The rescaling is exact for a beam whose shape does
not change between solves.  The field is solved again early after a
`HELIX_SC_GRID` card, on a bunch-train kick, after more than 100 mm of
travel, after more than 1 % particle loss since the last solve, and for
a beam with no extent.  Backtracking undoes only `K = 1` exactly; with
`K > 1` the backward walk warns and uses the plain adaptive grid.

## Which method to use

* **The default** (a fixed grid with CIC; see below) is among the
  fastest.  Its grid is sized at the first space-charge kick and kept, so where the
  beam's size or shape changes a lot along the line the cells end up
  too coarse for the beam (or the beam outgrows the grid) and the error
  grows.
* **An adaptive grid** (`cic_adaptive`, `tsc_adaptive`) keeps the cells
  in proportion to the beam at every kick.  It is more accurate where
  the beam changes along the line, at about twice the run time: it
  rebuilds the Green's function whenever the cell spacing changes.
* **TSC** smooths grid noise; prefer it on long transport without RF
  (see [Kernels](03_kernels.md)).  On lines with frequent RF, CIC is
  about as accurate and cheaper.
* **Solving the field every 2nd or 4th kick** (`*_reuse2`, `*_reuse4`)
  brings an adaptive grid's cost down to about a fixed grid's, at a
  small extra error from rescaling the field between solves.
* On a new line, a grid and particle-count scan
  ([Convergence guide](05_convergence.md)) is still how to know a result
  is converged.

Without a method chosen nothing changes: a project keeps its own
settings, and `SpaceChargeConfig()` keeps the classic ones (the GUI's
new-project settings are 48³, ±7σ, fixed grid, CIC).

## Projects

A project without `sc_method` keeps running exactly as it always did.
`scripts/sc_method_migrate.py` shows and changes what a project stores;
nothing is written without `--write`:

```bash
python scripts/sc_method_migrate.py PROJECT.lgproj            # report
python scripts/sc_method_migrate.py --pin PROJECT.lgproj      # store every setting
python scripts/sc_method_migrate.py --switch NAME PROJECT.lgproj
python scripts/sc_method_migrate.py --restore PROJECT.lgproj
```

* `--pin` stores every setting explicitly under the project's own method
  (its `sc_method`, else the method its keys match, else `custom` when
  it stores no space-charge setting), after checking that nothing
  changes, also for a command line naming any method.  A pinned project
  ignores later changes of the default method.
* `--switch NAME` writes method `NAME` with its defaults, keeps the
  settings that belong to no method, and keeps the previous keys under
  `sc_previous`; `--restore` brings them back.  A GUI save keeps
  `sc_previous`.

The GUI saves `sc_method` and every space-charge setting explicitly.

## Results file

The HDF5 provenance group records the method (`sc_method`), the registry
version, every space-charge setting, the `HELIX_SC_GRID` cards and what
was done with them, the solver's counters (`sc_solver_stats`: Green's
function builds, solved and reused kicks) and the threads in use (which
do not change the results — [CPU threads](../06_running/15_threads.md)).

## Cross-references

* [PIC solver](02_pic_solver.md) — the PIC cycle.
* [Kernels & Green's function](03_kernels.md).
* [Convergence guide](05_convergence.md).
* `linac_gen/pic/methods.py` — the registry and the resolution rules.
* `linac_gen/pic/field_reuse.py` — field reuse.

← [LEBT compensation (SCC)](07_lebt_scc.md)
