# Command line

Ring mode's three commands, every option in one place:

    python -m linac_gen ring optics INPUT [options]   # the periodic optics
    python -m linac_gen ring ramp   INPUT [options]   # the reference through a ramp
    python -m linac_gen ring run    INPUT [options]   # a bunch, turn after turn

`INPUT` is a `.lgproj` project or a lattice file (`.madx`, `.dat`, `.seq`,
`.lte`, …).  A project's own ring settings (its model, programs, space charge
and injection) and beam are used unless an option replaces them; a lattice
file alone runs the default settings and the default beam settings —
normalized rms emittances 0.25 π mm mrad and emit_z 0.3 deg·MeV at
352.21 MHz, a short bunch for a ring: give the beam's with `--beam` (`ring
run` says so when it runs on them).  The pages of this chapter explain what
each setting does; this one lists them.

## Options of every command

| Option | Meaning |
|---|---|
| `--threads N|all` | CPU threads for the ring kernels (default: `$HELIX_THREADS`, else `$OMP_NUM_THREADS`, else every CPU); the results are the same bit for bit for any number |
| `--energy MEV` | the ring's kinetic energy: needed when the lattice does not state its beam (a MAD-X or MAD8 `BEAM` that gives the energy does, as does a project; a TraceWin or elegant deck, or a MAD-X `BEAM` with only the particle, does not — their magnets keep the normalized strengths the deck gives), else it replaces the lattice's |
| `--species NAME` | the beam the lattice's magnets were set for, when the lattice does not say (`proton`, `H-`); the ring tracks protons |
| `--model {cartesian,curvilinear}` | the element maps ([Element models](03_models.md)): Cartesian bend multipoles, as MAD-X decks define them (default), or exact elements with the bend multipoles in the curved frame |
| `--edge-model {exact,full,linear}` | the pole faces: exact (frame rotation and hard-edge fringe; the Cartesian model's default), full (also the quadrupole fringe and the wedge's quadrupole field), linear (thin linear kicks; the curvilinear model's) |
| `--constants {helix,codata2010,codata2022}` | the physical constants of the reference (default `helix`) |
| `--harmonic H` | lock every cavity to harmonic H of the revolution frequency |
| `--bend-steps N` | integration steps per bend (default 0: automatic) |
| `--json` | print the summary as JSON (non-finite numbers as `null`) |

## ring optics

The tunes, chromaticities, momentum compaction, transition and the periodic
Twiss of the compiled ring ([Ring optics](02_optics.md)).

| Option | Meaning |
|---|---|
| `--six-d` | also the 6-D closed orbit, the synchrotron tune and the RF bucket (the RF on) |
| `--twiss-out FILE` | write the Twiss table per element (`.csv` or `.json`) |

## ring ramp

The reference particle alone through a ramp: the turns to the end of the
energy program, transition, the RF frequency and the phase and gain against
the voltage, turn by turn ([The ramp](05_programs.md)).

| Option | Meaning |
|---|---|
| `--ramp FILE` | the ramp: a JSON (or YAML) file of the energy, voltage and phase programs, tables inline or in CSV files |
| `--fixed-energy` | no ramp, although the project holds one |
| `--gamma-t G` | transition gamma (default: the ramp's own, else the ring's optics) |
| `--turns N` | turns to compute (default: to the end of the energy program) |
| `--csv FILE` | write the reference turn by turn (time, energy, gain, voltage, phase, frequency) |

## ring run

A bunch turn after turn ([Ring runs](04_run.md)); the results go to
`<stem>_ring.h5` in the output directory ([Results files](11_results.md)).

**Turns and the ramp**

| Option | Meaning |
|---|---|
| `--turns N` | the number of turns (default 100) |
| `--ramp FILE` | the ramp (as for `ring ramp`) |
| `--fixed-energy` | no ramp, although the project holds one |
| `--gamma-t G` | transition gamma of the ramp |
| `--to-end` | run the turns to the end of the ramp's energy program (instead of `--turns`) |
| `--corrector-mode {follow,fixed_field}` | in a ramp, steerers follow the reference momentum (default) or keep their field |

**Space charge and injection**

| Option | Meaning |
|---|---|
| `--space-charge FILE` | the space-charge settings: a JSON (or YAML) file ([Space charge](06_space_charge.md)) |
| `--intensity N` | real particles in the bunch with space charge (alone: space charge with the default settings) |
| `--no-space-charge` | no space charge, although the project holds its settings |
| `--injection FILE` | injection through a stripping foil: a JSON (or YAML) file of its settings ([Injection](07_injection.md)) |
| `--no-injection` | no injection, although the project holds its settings |

**The beam**

| Option | Meaning |
|---|---|
| `--particles N` | macroparticles at the start (default: a project's beam's; 10 000 for the matched source); refused for a beam read from a distribution file, whose particles are all used, and for an empty start |
| `--source {beam,matched,empty}` | beam: the project's beam (its distribution, particles, emittances, Twiss functions, with `--beam`'s changes), generated at the ring's species and energy; matched: a Gaussian matched to the ring's optics with the beam's emittances (the default for a bare lattice file, and for a project without beam settings); empty: no beam at the start.  The default for a project is the start its Run page saved (its beam, with its injection too; or an empty ring with its injection); with `--injection FILE`, empty |
| `--beam NAME=VALUE` | a beam setting, repeatable: `emit_nx`, `emit_ny` (π mm mrad, normalized rms), `emit_z`, `beta_z`, `frequency`, `cutoff` |

**What is kept, and where**

| Option | Meaning |
|---|---|
| `--snapshot-turns T1,T2,...` | turns at which the whole distribution is kept (`0` the start, `last` the last turn) |
| `--stats-every K` | turns between statistics rows (default: every turn, or every k turns to keep at most 100 000 rows) |
| `--probes N` | tune probes: 0 none, 1 the tunes (default; a project's run: the probes its Run page saved), n an n × n footprint |
| `--probe-max-sigma S` | the footprint's largest amplitude in rms beam sizes (default 3) |
| `--checkpoint-every K` | turns between checkpoints (`<stem>_ring_checkpoint.npz`; 0 none) |
| `--restart CHECKPOINT` | continue the run of this checkpoint, bit for bit (its probes, statistics interval and seed unless given; other ones are refused) |
| `--seed S` | the random seed of the beam, the probes and the injection's draws (default 1; with `--restart`, the checkpoint's) |
| `--device {cpu,cuda}` | the device ([GPU runs](08_gpu.md)) |
| `--out DIR` | the output directory (default `.`) |

With `--json` the summary holds the turns done and asked for, the particles
at the start (with injection: those the ring received), the survival and the
particles lost (with the elements where most were lost), the betatron and
longitudinal emittances at the start and at the end, the tunes (NAFF, the
optics' and the 6-D eigentunes), the wall time, the device and threads, the
run's notes, and — when they were on — the ramp's, the space charge's and the
injection's summaries.

## Examples

The simple Fermilab Booster of HELIX's tests, with a beam of 1.5 π mm mrad
and a bunch of about 0.2 m (the deck's `BEAM` gives the 800 MeV protons):

    # its optics, with the 6-D ones
    python -m linac_gen ring optics tests/ring/fixtures/simple_booster.madx --six-d

    # 1000 turns of 5000 particles at 800 MeV, the RF on, a snapshot at the end
    python -m linac_gen ring run tests/ring/fixtures/simple_booster.madx \
        --turns 1000 --particles 5000 --snapshot-turns last --out runs \
        --beam emit_nx=1.5 --beam emit_ny=1.5 --beam emit_z=5 --beam beta_z=100 \
        --beam frequency=44.7044096

    # the same with space charge (1.5e10 protons, the default settings)
    python -m linac_gen ring run tests/ring/fixtures/simple_booster.madx \
        --turns 1000 --particles 5000 --intensity 1.5e10 --out runs \
        --beam emit_nx=1.5 --beam emit_ny=1.5 --beam emit_z=5 --beam beta_z=100 \
        --beam frequency=44.7044096
