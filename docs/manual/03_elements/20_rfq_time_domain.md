# Time-domain RFQ solver

## TL;DR

For multiparticle runs, HELIX tracks an RFQ with **time** as the
independent variable instead of cell by cell along z, by default
whenever it can:

* through the quasi-static 3-D field of its real vanes;
* with space charge computed at a common time and periodic over the
  bunch spacing βλ;
* with particles lost when they touch the electrodes.

This is how Toutatis and LINACSrfqSIM model an RFQ.  On the PXIE RFQ it
reproduces Toutatis's transmission, exit energy and loss locations at
0 and 5 mA, from the same input particles, with no tuning.

```{.python .skip}
Simulation(lat, beam, space_charge=sc).run()    # rfq_solver="auto"
```

`StepConfig.rfq_solver` chooses the model:

| value | the RFQ is tracked by |
|---|---|
| `"auto"` (default) | this solver whenever it can run the lattice and beam (see Requirements); otherwise the per-cell model ([RfqCell](09_rfqcell.md)), with a log notice that says why.  A run that sets up the per-cell model explicitly — `Simulation(rfq_geometry=...)` other than `"auto"` (`"off"` is the card model), or a vane profile armed with `apply_rfq_geometry` — keeps the per-cell model |
| `"time_domain"` | this solver; a case it cannot run stops the run with the reason |
| `"cells"` | the per-cell model |

Only the RFQ cells change solver: the solver finds the chain of
`RFQ_CELL` elements itself, and everything before and after it (an
LEBT, an MEBT) is tracked as before.  The per-cell model remains the
only model for envelope runs and backtracking.  The results file
records what tracked the RFQ (`rfq_solver_used` in `provenance/`) and,
when `"auto"` kept the per-cell model, why (`rfq_solver_note`).

Where to choose it:

| Place | Setting |
|---|---|
| GUI | Numerics tab → **RFQ solver**: *Auto (time domain when possible)*, *Per cell (two-term model)* or *Time domain (3-D vane field)*, and **RFQ time steps** per RF period |
| project file (`.lgproj`) | `convergence` keys `rfq_solver`, `rfq_td_steps_per_period`, `rfq_td_sc_every` (absent: the defaults) |
| command line | `run` / `scan --rfq-solver auto\|cells\|time_domain [--rfq-td-steps N]` ([run](../06_running/07_cli_run.md)) |
| Python | `dataclasses.replace(lat.step_config, rfq_solver="cells")` |
| study spec | `"numerics": {"rfq_solver": "cells"}`; `rfq_td_steps` can be a swept parameter |
| assistant | `run_mp` with `rfq_solver` |

The GUI choice applies to every multiparticle run started from it:
the toolbar run, convergence and parameter scans, matching, and the
error, failure and parameter studies.

## Physics

**Field.**
- The vanes are taken from the deck's `RFQ_GEOM` `.vane` file, read with
  its own pole numbering (pole 1 = +y, 2 = +x, 3 = −y, 4 = −x).
- A deck without `RFQ_GEOM` gets its vanes from the `RFQ_CELL` cards,
  with the per-cell model's own geometry:
  - accelerating (±2) and transition (±4) cells: the two-term
    equipotential tip condition;
  - front-end (+3) and exit (−3) cells: the vanes flare as those cells'
    quadrupole ramp, R = r0/√C1 with C1 = sin³(πz/2L) or cos³(πz/2L),
    up to 3.5·r0;
  - tip radius: the card's `Tc`, else 0.75·r0;
  - vanes between the `RFQ_GAP_RMS_FFS` gaps, else where the flares
    reach 3.5·r0;
  - end-plate holes 1.076·r0.

  On PXIE the card-built tips match the vane file to a median 0.03 %
  in the regular cells (95 % of points within 0.25 %), 3 % in the
  entrance flare and under 1 % in the exit flare, except one cell: the
  deck gives cell 201 as unmodulated, while the real vanes demodulate
  over its 53 mm.
- Each vane is a tip arc of transverse radius ρ(z) at tip distance R(z)
  from the axis, continued by flat flanks.
- The grounded end plates sit at the chain's two ends, at the gaps given
  by `RFQ_GAP_RMS_FFS`: 2.5 mm thick, with a hole of 6.0 mm × r0 /
  5.576 mm (PXIE's proportion; r0 from the first and last cards).
- Laplace's equation is solved on a quadrant grid (the pole pairs are
  mirror images):
  - second-order finite differences;
  - Shortley–Weller cut cells at the curved surfaces in all three
    directions;
  - overlapping z windows, each padded by 20 mm;
  - algebraic multigrid (`pyamg`).
- Grid: 0.1 mm transversely; longitudinally the shortest cell length / 32.
- The near-axis potential is stored and interpolated tricubically, with
  E = −∇φ taken from the interpolant.
- The RF time dependence multiplies the static field: E(r, t) = E(r)·sin(ωt).
- Accuracy: on an electrode set whose exact solution is known (two-term
  equipotentials), the solver's error is 5×10⁻⁵ of V/2 at 0.1 mm, and
  second order in the step.

**Push.**
- Kick–drift–kick leapfrog (velocity Verlet) in (r, γβ), with the
  signed species charge.
- 48 steps per RF period by default (`rfq_td_steps_per_period`).
- On PXIE, 48 and 96 steps give the same transmission, energy and
  emittances.  Toutatis's own PXIE runs used 12 steps per period, which
  shifts the 5 mA transmission by about 2 points.

**Space charge.**
- Particles are deposited (cloud in cell) at a common time on a grid one
  bunch spacing βλ long, periodic in z, open in x and y.
- The solver works mode by mode: an FFT along z, then for each mode
  k the 2-D Green's function K₀(|k|r)/2π (−ln r/2π for k = 0),
  convolved on a doubled transverse grid.
- This is the field of an infinite train of identical bunches.  For the
  DC beam at the entrance that is exact; through the buncher it supplies
  the neighbouring bunches.
- The grid adapts to the beam:
  - the transverse grid spans ±4.5 rms;
  - the cells per period follow the bunch core length.
- The kick is applied every `rfq_td_sc_every` = 4 steps.
- The charge per macro-particle is the usual I/f/N_launched.
- The space-charge compensation factor applies inside the RFQ as it does
  elsewhere.
- Vane image charges are not included yet.  On PXIE at 5 mA they change
  the core field by under 1.5 %, and the halo field near the vanes by
  about ±14 %.

**Injection and exit.**
- The beam enters as given at the chain entrance. Each particle's phase
  becomes an arrival time, and the particle is drifted back to a common
  start time upstream of the entrance plate.
- At every cell exit, each particle's crossing is recorded, interpolated
  inside the time step.
- A DC beam is one period of a steady train, with no phase origin of
  its own.  The bunch core is centred at the RFQ exit, on the reference
  particle's phase.
  - The RFQ's buckets need not line up with the injection window:
    particles injected near its edges are captured one period earlier
    or later.  Each captured particle is counted in the bucket it ends
    in — in a steady train it is the neighbouring window's particle of
    the main bunch — so the accelerated beam leaves as one bunch, with
    or without `periodic_phase` (PXIE at 0 mA: σφ 6.4°).
  - A slow particle that is not captured keeps its lag.  With
    `periodic_phase` (the PXIE projects set it) such particles are
    folded onto the bunch after every cell, as after any element; σφ of
    all particles at the PXIE exit is 6.9° with the fold and about 2000°
    without it.
  - Every particle's shift is a whole number of periods, the same at
    every cell, so its phase is continuous through the RFQ.  Between the
    input and the first cell the whole beam moves by the constant that
    centres the exit bunch — for a DC beam, a relabelling of time.
  - A DC beam that has passed a static electric element (einzel lens,
    extraction column) is accepted.  As in a per-cell run, HELIX then
    never marks it a bunch train, so `periodic_phase` does not fold it.
- The reference particle is the one the RFQ cells produce in a per-cell
  run, so the reference itself, and the phasing of the RF elements
  downstream, are the same in both solvers.  The beam's centroid can sit
  off it in energy: on PXIE the bunch leaves 23 keV above the reference
  (Toutatis reports about +28 keV), mostly from cell 201 (see *Why the
  exit energies differ*).  The run records this offset
  (`exit_w_offset_kev`) and warns when it exceeds 0.5 % of the energy;
  downstream RF elements see the beam off their design energy by that
  much, as in TraceWin with a Toutatis RFQ.
- Results keep one row per cell, named after the cell.  With
  `record_substeps` each cell gets 4 rows, the last at its exit; they
  sit at positions where a per-cell run records a row too, with the
  same reference particle.  (The per-cell model records every 0.1 mm,
  tens of thousands of planes over an RFQ, too many to keep for every
  particle.)
- Downstream, the space-charge solver's bunch-train setting (neighbour
  bunches on while the bunch is long, see
  [neighbour images](../05_space_charge/01_models.md#model-2-3-d-pic-hockney-fft)) is left as a per-cell
  run leaves it: it is evaluated from the bunch length at every cell
  exit.

**Losses.**
- A particle is lost when it is inside metal (a vane tip, a vane body or
  an end plate), turns back, or is still inside the RFQ after three times
  the entrance transit time.
- It is also lost when it leaves the stored field region, the near-axis
  box beyond the vane tips (7.9 mm from the axis for PXIE), between the
  vanes.  No field is stored there.  Without this rule such a particle
  flew on in zero field: one did, to 1 m off axis at the exit, and its
  offset dominated the rms rows along the RFQ.
- Each loss is recorded at the particle's own z, with the name of the
  cell it died in.

**Statistics of the accelerated beam.**  Some particles fall behind the
bunch: they lag the reference particle by more than half of its energy gain.
On PXIE at 5 mA, about 10 in 15 500 leave the RFQ near 28 keV, 2 MeV below
the bunch, and a few more are lost on the vanes later.  They stay in the
beam until then: the transmission, the losses and the exported
distribution count them.  But with `rfq_stats_accelerated` (on by
default), the recorded statistics leave them out.  These are the rms sizes,
emittances, energy spread, Twiss parameters and density.  Each particle is
left out from the cell after the last one where it lagged the reference
energy by less than a tenth of the reference gain, that is, from where it
fell behind the bunch, so every plot and tile describes the accelerated
beam.  Otherwise those few particles set the exit energy spread: 57 keV
against 8.1 keV.  Turn the setting off, with Numerics →
*RFQ statistics: accelerated particles only* or `--rfq-stats all`, to count
every surviving particle.  The run's log line gives the count.

## Results on the PXIE RFQ

Toutatis's own 7786 input particles at the RFQ entrance (H⁻, 30 keV →
2.1 MeV, 162.5 MHz, 60 kV, 203 cells):

| | Toutatis | time-domain, vane file | time-domain, vanes from the cards | per-cell (`antisym`) |
|---|---|---|---|---|
| transmission, 0 mA | 98.37 % | 98.50 % | 98.19 % | 98.82 % |
| transmission, 5 mA (3.893 mA alive) | 80.40 % | 80.91 % | 79.28 % | 83.48 % |
| exit ⟨W⟩, 5 mA | 2.1211 MeV | 2.1221 MeV | 2.0998 MeV | 2.0988 MeV |

The losses along z follow Toutatis: the counts of particles alive at
the cell exits outside the buncher's loss burst agree within 0.6 % of
the input (0.2 % at 0 mA), and the burst (about 1260 particles at 5 mA between 0.65
and 1.08 m) sits at the same place, the time-domain losses trailing by
about 5 cm.  At 5 mA, σx and σy at the cell exits lie within a median
of 8 % of Toutatis's table (the median over the cells beyond 0.25 m of
|σ_HELIX/σ_Toutatis − 1|).  That table holds fixed-time snapshots,
while HELIX records fixed-z crossings.

Exit emittances (normalised rms, of the accelerated particles,
W > 2.0 MeV — Toutatis's exit file also holds about ten unaccelerated
particles, which raise its 5 mA x-emittance from 0.387 to 0.465):

| mm·mrad, x / y | Toutatis | time-domain |
|---|---|---|
| 0 mA | 0.300 / 0.298 | 0.314 / 0.304 |
| 5 mA | 0.387 / 0.352 | 0.396 / 0.407 |

At 5 mA the vertical emittance is 16 % above Toutatis's, the one clear
difference left.  It is not the time step: at Toutatis's 12 steps per
period it rises to 0.427.  Vane image charges, which Toutatis includes
and this solver does not yet, are the leading candidate.

**Why the exit energies differ.**  Up to cell 199 the vane file and
the vanes built from the cards give the same bunch energy, within
1 keV.  The deck gives cell 201 as unmodulated (A10 = 0), while the
real vanes demodulate gradually over its 53 mm and still accelerate
the bunch there: the vane file ends 21 keV higher.  The per-cell
model, which follows the cards, ends 23 keV low against Toutatis for
the same reason.

**Regression mode.**  `cards_geometry.two_term_field_map` gives the
time-domain solver the per-cell model's own analytic two-term field.
From the same particles at 0 mA it gains the same energy as the
per-cell model (2.0997 and 2.0981 MeV for the accelerated particles),
so the cards' synchronous phases agree with a single RF clock.  The
median deviations of σx and σy at the cell exits from the run on the
real vanes are:

| | σx | σy |
|---|---|---|
| time domain, vanes built from the cards | 1.8 % | 2.0 % |
| time domain, the cards' two-term field | 14 % | 15 % |
| per-cell model | 38 % | 11 % |

The field of the real vanes, which the two-term form approximates,
accounts for about 15 % in beam size; the per-cell model's transverse
treatment adds most of the rest in x.

## Using it

**Requirements:**
- the vanes: an `RFQ_GEOM` card naming the `.vane` file, or no
  `RFQ_GEOM` card (the cards give the electrodes, see Physics);
- a DC (continuous) beam at the RFQ entrance;
- one contiguous chain of `RFQ_CELL` elements, with only markers between
  cells;
- one vane voltage;
- no misalignment, or the same misalignment on every cell;
- at least 3 cells, and mirror-symmetric pole pairs (tip distances and
  radii equal to 1 µm, equal voltages, one voltage along z);
- `pyamg` (installed with HELIX), to build a field map (a cached map
  loads without it).

These are checked when the run starts, before anything is tracked.
With `"auto"` a violation means the RFQ is tracked cell by cell, with a
log notice; with `"time_domain"` it raises an `RfqTdError`.  A deck that
names a vane file that cannot be read is never run from the cards'
electrodes instead: `"time_domain"` fails, `"auto"` uses the per-cell
model.

**First run.**  The field map is built once: about three minutes on
8 cores for PXIE (about 200 MB).  It is cached under the user cache
directory (`~/Library/Caches/HELIX/rfq_td` on macOS, or
`HELIX_CACHE_DIR`), keyed by the vane geometry, the cell edges and the
solver settings — not the voltage: a voltage scan reuses one map.  The
cache keeps the most recently used maps within 4 GB
(`HELIX_RFQTD_CACHE_GB`).  The GUI status bar shows the build window by
window; **Stop** ends the run as stopped, and nothing is cached.
- Two processes that need the same map build it once: the second waits,
  and Stop ends the wait.  A lock left by a process that died is taken
  over; an unreadable map is rebuilt.
- Inside a `multiprocessing` pool worker (e.g. parallel CMA-ES matching)
  the map is built in that one process.
- A window whose Laplace solve does not converge stops the build with
  an error; nothing is cached.

**Run time** (PXIE, 8 cores): 7786 particles in about 4 s at 0 mA and
50 s at 5 mA with space charge; 100 000 particles at 5 mA in
1.5 minutes.  Identical runs give bitwise identical results, for any
number of threads, and a field map built afresh on the same machine
(a cleared cache) is bit for bit the one it replaces: the multigrid
solver's set-up, which draws a random start vector, runs with numpy's
global generator in a fixed state (`linac_gen.core.amg.seeded`), as
does the per-cell model's vane-geometry profile solve.

**Settings** (`StepConfig`):

| field | default | meaning |
|---|---|---|
| `rfq_solver` | `"auto"` | `"auto"`: this solver when it can run, else the per-cell model; `"time_domain"`: always this solver; `"cells"`: the per-cell model |
| `rfq_td_steps_per_period` | 48 | leapfrog steps per RF period |
| `rfq_td_sc_every` | 4 | steps between space-charge kicks |
| `rfq_stats_accelerated` | `True` | statistics leave out the particles that leave the RFQ unaccelerated (see Physics) |

**Not supported yet** (with `"auto"`, these run on the per-cell model):
- image charges;
- displaced or deformed vanes (a full-domain field solve);
- vane voltage varying along z;
- a bunched input beam;
- bunch-train (multibunch) runs;
- envelope mode and backtracking, which keep the per-cell model
  (backtracking through the RFQ warns that it does).

**Plots.**  The Results tab's [RFQ panels](../10_gui/07_results_tab.md#rfq-panels)
draw these vanes, their field map once it is cached (on-axis field,
peak surface field against the Kilpatrick limit) and the run cell by
cell.

## Source

* `linac_gen/rfq_td/geometry.py`: electrodes and the loss test.
* `linac_gen/rfq_td/cards_geometry.py`: electrodes from the
  `RFQ_CELL` cards.
* `linac_gen/rfq_td/laplace.py`, `build.py`, `fieldmap.py`: field solve,
  cache and interpolation.
* `linac_gen/rfq_td/space_charge.py`: periodic-train Poisson solver.
* `linac_gen/rfq_td/tracker.py`: time loop.
* `linac_gen/rfq_td/chain.py`: how the tracker runs and replays the RFQ.
* `linac_gen/csrc/rfq_td_kernels.cpp`: the compiled field gather and
  leapfrog step.
