# Ring runs

A ring run tracks a bunch turn after turn through the ring kernels — the
element maps of the [element models](03_models.md), the same ones the
[optics](02_optics.md) come from — at the ring's fixed reference energy, or
through a ramp ([The ramp](05_programs.md)).
Coordinates are the ring's: x, y [m]; px, py, the transverse momenta over
the reference momentum; ζ = s − β0·c·t [m]; δ = (P − P0)/P0.

## The particles

A run starts from the **Beam tab's beam**, as a linac run does: its
distribution, number of particles, emittances, Twiss functions, dispersion
and centroid, converted exactly to ring coordinates.  There is one beam, with
one set of Twiss functions and one number of particles, and the Run page
says what a run will start with.  The beam is generated at the ring's species
and energy (in a ramp, the energy of its first turn).  A Beam tab holding
another species or energy keeps its shape, emittances and Twiss functions,
and the run says so in its notes.

**Ring & Optics → Use the matched beam in the Beam tab** gives the Beam tab
the beam matched to the ring at the start of the run.  It is computed from
the optics of the run's first turn: this element model, the Programs page's
ramp at turn 0 with its RF, and the Injection page's magnets at turn 0.  It
sets:

* the periodic Twiss functions α and β in x and y at the start of the ring;
* the periodic dispersion per MeV of kinetic energy, D/(β0·p0c), in mm/MeV
  and mrad/MeV;
* with the RF on at the start, the longitudinal Twiss functions of the 6-D
  normal mode, with the phase in degrees of the Beam tab's frequency.
  Without RF the beam coasts, and the longitudinal settings stay as they are;
* the closed orbit at the start as the centroid (zero on a ring without
  errors or bump);
* for a lattice that declares its beam (a MAD-X or MAD8 `BEAM` that gives
  the energy), that species and the run's starting energy.  For a lattice
  that declares none (a TraceWin or elegant deck, a `BEAM` with only the
  particle) the Beam tab's species and energy stay: the ring runs at them,
  a deck of normalized strengths with the strengths it gives.

The emittances, the distribution and the number of particles stay the Beam
tab's.  A beam read from a particle file takes the transverse Twiss functions
only, re-shaped onto them (its emittances kept); its dispersion, centroid and
longitudinal distribution stay the file's.  A continuous beam (the Beam tab's
DC mode) takes no longitudinal Twiss functions.  While the Injection page
injects the Beam tab's beam, the button refuses: the ring's matched beam
would replace the injected one.  Inject a Gaussian at the foil or a particle
file instead, for a matched beam at the start and an injection.  A beam
made this way is as well matched as one generated from the ring's normal form.  On
the simple Booster at 800 MeV its BMAG (below) is 1.0000 in x, y and z, and
over 60 turns its rms bunch length stays within 0.6 % (the noise of 2000
particles).  With a β_z 1.5 times too large (BMAG 1.084) the bunch length
swings between 11.9 and 17.7 cm, against 11.9 and 17.8 cm predicted from
the BMAG.  The Beam tab's Gaussian (and thermal) generator sets the second
moments exactly and leaves the sample mean free, so the beam's centre starts
within about σ/√N of the closed orbit (N the number of particles); its
other distributions (waterbag, uniform, parabolic, KV) set the expected
second moments, so theirs carry about 1/√N of noise too.  The command line's
matched source, below, sets both exactly.

Every run that starts with a beam (not an empty ring, nor a run continued
from a checkpoint) compares it against the ring's matched beam, mode by
mode: BMAG = tr(S) / (2·√det S), where S holds the beam's second moments in
the matched normalized coordinates.  For an uncoupled plane this is
(β·γ_m − 2·α·α_m + γ·β_m)/2, with m the matched Twiss functions; it is 1
when the beam is matched.  The Run page's summary gives it as "At the start,
BMAG (1 when matched) x 1.0000, y 1.0000, z 1.0000.", the command line as
"start: BMAG against the ring's matched beam (1 when matched) x 1.0000, …",
and the results file as `start_mismatch` in `/meta`.  The run adds a note
when a mode is above 1.01, where the beam's β swings about 14 % either side
of the matched one.

**None** (`source = "empty"`): the ring starts empty and the injection fills
it turn by turn ([Injection](07_injection.md)).  Turning injection on selects
it; the Beam tab's beam can be selected afterwards.

A project saved by an earlier version whose run started from a beam matched
to the ring, with its own number of particles, now starts from its Beam
tab's beam: its distribution and number of particles.  The Run page says
so, with the old number of particles; Use the matched beam in the Beam tab
gives the Beam tab's beam the matched Twiss functions.

From the command line and the API there is a third start, **matched**
(`source = "matched"`).  It is the command line's default for a bare lattice
file, which has no beam settings of its own: a Gaussian of `n_particles`
matched to the ring's periodic optics with the beam's normalized rms
emittances.  Each normal mode is cut at the beam's cutoff (in amplitude),
and the beam is then given exactly the normal-mode emittances asked for: its
centroid exactly on the closed orbit and its second moments set exactly (a
cut at 3 rms alone would leave 0.949 of the emittance).  The projected emittances a run prints include the energy
spread's share through the dispersion: on the simple Booster, 1.518 π mm
mrad in x for 1.5 asked.  With the RF on, the beam is matched in six
dimensions, to the 6-D normal form of the one-turn map with the RF bucket.
Without RF the transverse planes are matched (with the dispersion) and the
beam coasts: ζ uniform over the ring, δ Gaussian with the beam's rms energy
spread, √(ε_z·(1 + α_z²)/β_z).

The Beam tab's longitudinal emittance `emit_z` is in deg·MeV, the degrees
those of the Beam tab's frequency: it becomes ε(ζ, δ) = emit_z ·
β0c/(360 f) / (β0² E0) [m], E0 the total energy.  A Beam tab left at a
linac's 162.5 MHz therefore asks for a longitudinal emittance 3.6 times
smaller than the same number at the Booster's 44.7 MHz.

## A turn

One pass through every node of the ring.  A particle outside an aperture is
lost there — its turn, element, position and cause are kept, with its
coordinates at the aperture; a particle whose map is not defined (a
transverse momentum beyond the total momentum) is lost as *invalid*.  An
aperture moves with its element: a misaligned quadrupole's or bend's is
checked in the element's frame (its offsets and roll), a drift's at the
drift's offset, as linac tracking checks them; a multipole's, which applies
its offsets inside its own kick, in the lab frame, as in linac tracking.  After
the turn ζ is brought back into the RF period C/h (the circumference without
RF): the RF maps are periodic in it, so this changes nothing in the motion
and keeps the coordinate bounded.

Without a ramp the reference energy is fixed: cavities whose phases would
give the reference particle energy are refused, and so is an RF phase on the
wrong side of transition (below transition the bucket is at a sine lag of 0°,
HELIX RF phase −90°; above it at 180°).  A ramp sets each cavity's voltage,
phase and frequency every turn, and changes the reference after each cavity
([The ramp](05_programs.md)).  With space charge the turn is tracked from
kick to kick, the beam's field applied at each ([Space charge](06_space_charge.md)).
With a stripping foil the turn stops there: the circulating particles that
cross it are scattered and slowed down, and an injection turn's particles
are added ([Injection](07_injection.md)).

## What is kept

| | |
|---|---|
| Statistics | every `stats_every` turns: the number of circulating particles, their centroid and their 6 × 6 second moments.  The GUI and the command line take every turn up to 100 000 turns and every k turns beyond (at most 100 000 rows, about 600 bytes each); the Python default is every turn |
| Emittances | from the second moments: the rms emittance of each plane, √det of its 2 × 2 block — x, y in m·rad (normalized: × β0γ0), ζ–δ in m (in eV·s: × p0c/c), with the reference of each statistics row (it changes in a ramp) |
| Snapshots | the whole distribution (with particle ids) at the chosen turns; `0` is the start, `last` the last turn |
| Losses | per particle lost: its id, the turns it completed (0 for a loss in the first turn: it is missing from the statistics from turn 1 on), the node, element, position s and cause, and its coordinates |
| Tune probes | particles tracked with the beam to the end of the run, their coordinates kept every turn for the first 4096 turns (below) |

The statistics are sums in a fixed order (numpy's pairwise sums, not a BLAS
library, whose sums may change with the memory alignment or the number of
threads): the same bits on every machine and in every run.

## Tunes: probes and NAFF

A tune probe starts at the closed orbit with a small amplitude (1 % of the
beam's rms size in each normal mode; 20 % with space charge, where it feels
the beam's field — see [Space charge](06_space_charge.md#the-kicks)).  Its turn-by-turn coordinates,
normalized with the ring's normal form, give its tunes by NAFF — the
frequency of the largest peak of the Hann-windowed Fourier transform,
refined by maximizing it (Laskar's method), a constant (a closed-orbit
offset) fitted with it.  For a pure tone the frequency comes out to rounding
from three periods in the record on; a tune needs at least three periods of
its motion in the run (a synchrotron tune of 0.024 needs 125 turns) and is
not reported below that.  On the simple Fermilab Booster a probe's NAFF
tunes over 1000 turns are the eigentunes of the one-turn map to 2 × 10⁻⁹.
The probes' tunes come from their first 4096 turns (a longer run tracks
them on, for their survival, without keeping their coordinates).

With the RF on, the transverse tunes so measured are those of the 6-D motion:
they differ from the 4-D tunes of the optics (RF off) by the coupling of the
synchrotron motion through the dispersion at the cavities (8 × 10⁻⁶ in Qx on
the simple Fermilab Booster) — the results give both.

**A footprint**: with *n* probes per plane, an n × n grid of amplitudes from
0.1 to `probe_max_sigma` (default 3) rms sizes; each gives its tunes, and its
tune diffusion — the change between the tunes of the first and the second
half of the record (√(ΔQx² + ΔQy²)).  With space charge the Results tab's
**Ring tunes** marks the bare working point (+) and the analytic shift at
the bunch centre (×, when it is within half an integer) and gives the
probe's tunes less the bare ones, beside the analytic shift.

## Checkpoints

Every `checkpoint_every` turns the run writes everything it needs to continue
— the particles (with their energy variables, bit for bit), the statistics
so far, the snapshots, the probes — to a checkpoint file (written to a
temporary file and renamed, so it is never half written).  A run restarted
from it ends exactly — every bit — as the run that was not interrupted.  The
continuation may ask for more turns and other snapshots; it must have the
checkpoint's tune probes, their largest amplitude, its statistics interval
and its seed (an injection draws every turn's bunch and foil scattering
from the seed; another setting would mix two runs in one result: refused —
the command line takes them from the checkpoint unless they are given).  A
checkpoint of another ring (another compiled lattice or element model) is
refused too.

## Stop, memory and quitting

**Stop** takes effect at the end of the turns being tracked: the run
launches turns in groups of about a quarter of a second, but a group is at
least one turn, and one turn of a large bunch can take seconds (3.6 s for
58 000 particles of the simple Booster on 14 cores).  The turns done are
kept and written.

**Memory**: a run needs up to about 600 bytes per particle (the particles
with their energy variables and bookkeeping, and the arrays made while they
are generated: 567 bytes measured for a matched beam, 414 for the Beam tab's
distribution), 56 bytes per particle for each snapshot, the statistics rows
and the probes' record; with space charge, its Green's functions (kept up to
256 MiB), the deposit's grids, a solve's working arrays and the frozen
model's fields (kept up to 512 MiB).  The GUI refuses a run that would need
more than 70 % of the machine's memory, before it starts.

Quitting HELIX during a run stops it and waits five seconds at most: a run
that cannot end its turn and write its results by then is not saved.

## Devices and speed

**CPU**: the particles are spread over every core (the CPU threads setting);
the results are the same bit for bit for any number of threads.  **CUDA GPU**:
the same kernels, compiled for the GPU; the particles stay on the device
between the statistics turns, which are computed on the host.  On macOS there
is no CUDA.  The GPU compiler fuses multiply-adds and the GPU's sine and
cosine are not the CPU's, so the results differ from the CPU's at rounding
level, growing slowly with the turns — 7 × 10⁻¹³ of each coordinate's
largest value after one turn of the simple Booster and 5 × 10⁻¹⁰ after
1000, as estimated by fusing the multiply-adds on the CPU (the GPU
measurements are part of the CUDA sessions).  The launches on the GPU are split over particles and turns to
take about a quarter of a second each: under Windows a GPU that also drives
a display stops a kernel that runs longer than its timeout (2 s by default).

Without numba (the `[ring]` extra: `pip install -e ".[ring]"`, see
[Installation](../01_getting_started/02_installation.md)) ring mode runs the
same code as plain Python — the same results, about a hundred times slower.  [GPU runs](08_gpu.md)
says what a GPU run needs, what runs where and how to check a GPU.

The simple Fermilab Booster, no space charge, on a 14-core Mac (10
performance and 4 efficiency cores), time per turn of the tracking:

| | 58 000 particles | 5 000 particles |
|---|---|---|
| curvilinear model, 6th order with 6 steps in every magnet | 2.46 s | 0.207 s |
| Cartesian model, expanded bend maps | 2.10 s | 0.176 s |
| Cartesian model, exact bends (the default: fourth order, about 100 kicks per bend) | 3.56 s | 0.300 s |

The time is proportional to the number of particles.  A whole run adds the
statistics and the tune probe: 5 000 particles of the default model take
0.320 s per turn with statistics every turn — 1000 turns in 5.3 minutes.
Through a ramp a launch tracks one turn (the programs are written between
turns); that costs no measurable time: timed side by side with a run at fixed
energy, 0.341 against 0.346 s per turn for the same bunch (about 1 % either
way between runs).
The [ring benchmarks](../12_validation/06_ring_benchmarks.md) page compares
these times with a reference code's on the same machine.

## From the command line

```
python -m linac_gen ring run booster.madx --turns 1000 --particles 58000 \
    --beam emit_nx=1.5 --beam emit_ny=1.5 --beam emit_z=5 --beam beta_z=100 \
    --beam frequency=44.7044096 --snapshot-turns 0,last --out runs
python -m linac_gen ring run booster.madx --turns 5000 --checkpoint-every 500 --out runs
python -m linac_gen ring run booster.madx --turns 5000 --out runs \
    --restart runs/booster_ring_checkpoint.npz
```

The run writes `<stem>_ring.h5` to `--out` and prints the survival, the
beam's BMAG at the start, the emittances at the start and the end, and the
probe's tunes against the optics.  A project's run starts as its Run page
saved it: from its beam (with its injection, if it has one, as well), or
from an empty ring that its injection fills.  `--particles` changes the
number of particles (it is refused for a beam read from a distribution file,
whose particles are all used), and `--source matched` starts from a Gaussian
matched to the ring instead.  An injection given with `--injection` fills an
empty ring.  A bare lattice file, which has no beam settings, starts from the
matched Gaussian of `--particles` with the `--beam` emittances.  `--probes n` makes a footprint; `--stats-every k` sets the statistics
interval; `--device cuda` runs on the GPU; `--threads N` the CPU threads;
`--json` prints the summary as JSON.  The element-model options are those
of `ring optics`.  `--ramp FILE` runs through a ramp, `--to-end` to the end of
its energy program ([The ramp](05_programs.md)); `--space-charge FILE` or
`--intensity N` with space charge ([Space charge](06_space_charge.md));
`--injection FILE` fills the ring through a stripping foil
([Injection](07_injection.md)).

## In Python

```python
from linac_gen.cli.common import load_lattice
from linac_gen.core.config import BeamConfig
from linac_gen.ring import RingConfig, RingSimulation

# the simple Fermilab Booster (the deck HELIX's tests use)
lattice = load_lattice("tests/ring/fixtures/simple_booster.madx")
beam = BeamConfig(species="proton", energy=800.0, frequency=44.7044096,
                  emit_nx=1.5, emit_ny=1.5, emit_z=5.0, beta_z=100.0)
res = RingSimulation(lattice, beam, RingConfig(n_turns=20, n_particles=1000,
                                               snapshot_turns=[0, "last"])).run()
print(res.survival[-1], res.emit_normalized[-1], res.optics["eigen_tunes_6d"])
```

`RingSimulation(..., coords=X)` starts from chosen particles (ring
coordinates, shape (6, n)); `beam=` from a HELIX Beam at the ring's
reference; `run(checkpoint_path=..., restart_from=...)` writes and continues
checkpoints, `run(live_callback=f)` calls `f(stats, n_turns, reference)`
after each statistics row (each row's reference in `stats["ref_bg"]` and
`stats["ref_p0c"]`); `RingConfig(ramp=...)` runs through a ramp,
`RingConfig(space_charge=...)` with space charge, `RingConfig(injection=...)`
with injection;
`linac_gen.ring.results.save_ring_hdf5` / `load_ring_hdf5` write and read the
results file.

## The results file

HDF5, `run_type = "ring"`, schema 3 (schema 1 and 2 files, written before
the ramp and before injection, are still read); every dataset, and how to
read a file back, are on [Results files](11_results.md):

| Group | Contents |
|---|---|
| `turns/` | `turn`, `n_alive`, `mean` (6), `cov` (6 × 6), and the reference `beta0gamma0`, `p0c_mev` per statistics row |
| `snapshots/turn_<t>/` | `coords` (6, m), `pid`; in a ramp, the reference and RF of the turn after it (attributes) |
| `ramp/` | a ramp's record per turn (`t_s`, `kinetic_mev`, `gain_mev`, `voltage_mv`, `phi_deg`, `frequency_hz`, `above_transition`, `capped`) and its summary (attributes: transition turn, γt, turns capped, the reference at the end, the program) |
| `injection/` | with a foil: the injection and the foil turn by turn, the closed orbit at the foil, the waveforms, the hit and deposit maps, the hits per particle, and the foil's settings and physics ([Injection](07_injection.md#what-a-run-keeps)) |
| `losses/` | `pid`, `turn` (turns completed before the loss), `node`, `element`, `s`, `cause`, `coords` |
| `probes/` | `tbt` (probes × min(turns, 4096) + 1 × 6), `amplitudes_sigma`, `tunes`, `diffusion`, `state` |
| `tunes`, `optics`, `ledger` | the probe's tunes, the ring's periodic optics (with the 6-D eigentunes and the RF bucket), the particle bookkeeping |
| `optics_table/` | the run's periodic optics at every element's exit (at the start of a ramp): Twiss functions, phase advance, dispersion, closed orbit |
| `meta` | the reference, the ring's length and elements, the device, the settings, the statistics interval, notes; with space charge its settings, kicks and analytic tune shift (`space_charge`) |
| `beam_config/` | the beam settings the particles came from (written by the GUI and the command line) |
| `provenance/` | the code version and commit, the lattice file's path and hash, the seed, the threads |
