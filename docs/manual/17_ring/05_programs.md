# The ramp

A ramp accelerates the beam: turn by turn, it sets the reference energy, the
RF voltage and the RF phase.  The magnets follow the reference particle —
their strengths normalized to its momentum stay the same, as in a machine
whose magnet currents track the momentum — so a ramp is defined by the
reference energy alone.  Without a ramp the ring runs at its fixed reference
energy (see [Ring runs](04_run.md)).

## The rule, turn by turn

At the start of every turn, with *t* the reference particle's own time since
the start of the cycle (the sum of each node's length over β*c* over the
turns before) and *E* its total energy:

1. **The gain the turn needs**: ΔE = E_prog(t + C/(βc)) − E, the program's
   energy one revolution later minus the reference's energy now.  The
   reference catches up with the program every turn, so rounding in its
   time does not accumulate into an energy error.
2. **The voltage** V(t) from the voltage program, shared by the cavities in
   proportion to their own voltages (evenly when the lattice gives none).
3. **The phase**: φs = asin(ΔE/V) in the sine convention (a cavity gives the
   reference V sin φs; HELIX's linac RF phase is φs − 90°).  Above transition
   — γ at the start of the turn above γt — the phase is π − φs.  A gain larger
   than the voltage can give (any gain, with no voltage) is capped at ±90° and
   the turn counted, or, if asked, refused: the run then stops at that turn
   with the turns done and says which turn and why (a refusal at the first
   turn starts no run).
4. **The frequency** f = hβc/C: every cavity stays locked to the reference
   particle, and ζ wrapped into the RF period C/h after a turn stays exact.

Each cavity then gives every particle V_k sin(φ_k − 2πfζ/(β0c)) and the
reference V_k sin φ_k (with a phase table, its share of the program's gain:
the reference follows the energy program).  A cavity whose voltage is
negative — an H⁻ deck's, whose charge signs every voltage, or a deck's own
sign — gives V_k sin(φ_k + π): the rule counts it with that offset, so the
net voltage still gives the gain.  Every cavity must be at the ring's one
harmonic: a ramp refuses cavities at other harmonics or at a fixed
frequency.  After each cavity, a reference change re-expresses the particles
relative to the new reference exactly: with p0 → p0′ and β0 → β0′,
p_τ′ = (p0c·p_τ − ΔE)/(p0′c), p_x and p_y scale by p0/p0′ (the transverse
momenta shrink as the momentum grows: adiabatic damping) and ζ by β0′/β0.
The reference's time over the turn adds each stretch between cavities at its
own velocity.  None of this is applied without a ramp: a ring run at fixed
energy is unchanged, bit for bit.

The magnets' normalized strengths need no update, and the RF focusing of
each turn is the bucket of its own voltage and phase.  Steerers follow the
reference momentum too, or, if asked, keep their field (their kick then
falls as 1/p).

## Programs

**Reference energy.**

* *Flat*: the energy stays; the voltage and phase programs still apply (for
  example a voltage table for capture at fixed energy).
* *Resonant cycle*: the momentum of a resonant magnet cycle,
  p(t) = p_c − p_a cos 2πf(t − t_min), lowest at t_min and at the top energy
  half a cycle later (p_c + p_a); its centre p_c and amplitude p_a are set so
  that it passes through the start momentum at t_start, and it is held at
  the start momentum before t_start.  Given: the start and top kinetic
  energies (the start defaults to the ring's reference energy, the lattice's
  or the one asked for), the cycle frequency, t_min (the field's minimum)
  and t_start.  The program ends at the top, t_min + 1/(2f).
* *Table*: times and kinetic energies, momenta or total energies, linear
  between the points, the end values held; it ends at its last time.

**RF voltage**: the cavities' own voltages, or a table of the total voltage
against time (linear between the points, the end values held).

**RF phase**: from the gain (above), or a table of the synchronous phase of
the net voltage against time, used as it is — the table carries its own
transition jump.  With
a table the reference particle (and the magnets) follow the energy program
and the beam follows the table's phase: a beam whose phase does not give the
program's gain moves off the reference momentum, as in a machine running
without its radial loop.

**Transition**: γt from the ring's optics (the 4-D momentum compaction at the
start of the ramp, 1/√α_c), or a given value.  The jump comes on the first
turn whose reference starts above it; the results name that turn, or say the
ramp started above transition.

**Paraphasing**: groups of cavities whose phases are offset by an angle θ_i
(given by name, name pattern or the cavity's place in the ring, 0 being the
first after s = 0 — needed when the cavities share a name).  The phase rule
then uses the net voltage: with A = Σ V_i cos θ_i and B = Σ V_i sin θ_i,
R = √(A² + B²), φ = asin(ΔE/R) − atan2(B, A), and π − asin(ΔE/R) − atan2(B, A)
above transition; a gain beyond R is capped or refused as above; groups
whose voltages cancel, and a cavity in two groups, are refused.

**The clock**: the cycle's time at turn 0 (default 0).  Starting a run later
in the cycle starts it at the program's energy there; the ring is compiled at
that energy.

### Program files

The command line reads a ramp from a JSON file (YAML with PyYAML installed)
in this layout; any table may instead be a CSV file next to it, named by
`"file"`:

```json
{
 "energy": {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 6000.0,
            "frequency_hz": 15.0, "t_min_s": 0.0005, "t_start_s": 0.0008},
 "voltage": {"kind": "table", "file": "volts.csv"},
 "phase": {"kind": "auto", "on_excess": "cap"},
 "paraphasing": [{"cavities": [0, 2, 4], "angle_deg": 20.0}],
 "gamma_t": 5.449167323,
 "t_start_s": 0.004
}
```

A CSV table's header names its columns: `t_s` (seconds), then one of
`kinetic_mev`, `momentum_mevc`, `total_mev` (an energy table: the column
sets what the values are), `voltage_mv` or `phi_deg`; lines starting with
`#` are comments.  Numbers may be written in any form Python reads
(`6e3` too).

```
t_s,voltage_mv
0.0,0.4
0.003,1.2
0.03,1.2
```

## The Programs page

**Ring → Programs** holds the same programs: the energy program (fixed, flat,
resonant cycle or table), the voltage and phase programs, transition gamma
(blank: from the ring's optics), the clock at turn 0, the steerers, and the
paraphasing groups.  Times on the page are in ms.  Each table can be read
from and written to a CSV file in the format above (for the energy table,
the file's column sets the quantity, and the quantity chosen names the
column written).  The steerers' choice applies to a ramp only: at fixed
energy they keep their kicks.  Every valid change goes
into the project; an entry that is not valid is reported and the project
keeps the last valid programs (a run refuses to start until the page is
valid).

**Preview** runs the reference particle alone through the ramp, off the GUI
thread (up to a million turns): the turns to the end of the energy program,
the energy from start to end, the transition turn, the RF frequency range,
the largest gain against the voltage and the turns capped, with the kinetic
energy, the synchronous phase, the voltage and the frequency plotted turn by
turn.  A program whose end is already past at turn 0 has no turns to run,
and says so.  **Run to the end
of the program** sets the Run page's number of turns to the program's.

While a run goes through a ramp, the Run page's turn label gives the
reference's γ at the last statistics row.  After it, the Results tab's
**Ring ramp** shows the same curves for the run, with the transition turn
marked.  **Ring phase space** draws the separatrix of each snapshot turn's
own bucket — the accelerating bucket, shorter than the stationary one and
asymmetric — and the emittances are normalized with each statistics row's
own reference.

## The accelerating bucket

With the synchrotron motion's Hamiltonian in (φ, δ), φ = φs − (2πh/C)ζ,

H = πhη δ² + (V/(β0²E)) (cos φ + φ sin φs),

the separatrix runs through the unstable fixed point π − φs, and

δ² = (V/(πhηβ0²E)) [cos φ_u + φ_u sin φs − cos φ − φ sin φs].

Its other end is where δ² first turns negative going from φs away from the
unstable point.  At φs = 0 it is the stationary bucket.  The small-amplitude
synchrotron tune is Qs = √(h|η|V|cos φs|/(2πβ0²E)).  The matched beam of a
ramp is matched to the RF focusing of its first turn, V_k cos φ_k about a
stationary closed orbit; a ramp starting with no RF focusing (no voltage, or
the phase at 90°) starts with a coasting beam.

## What a ramp run keeps

Besides what every run keeps ([Ring runs](04_run.md)): each statistics row's
reference (β0γ0 and p0c), and per turn, at its start, the reference's time and
kinetic energy, the gain, the voltage, the phase of the phase rule (the net
voltage's phase adds the offset of the paraphasing groups and of negative
voltages), the frequency, whether the turn is above transition and whether
it was capped; the transition turn (None when not crossed; whether the ramp
started above transition), the turns capped, the reference at the end, the
turns to the end of the program (None beyond two million) and the program as
it ran.  Each snapshot keeps
the reference and RF of the turn after it, what its bucket needs.  A
checkpoint keeps the reference's time and energy as exact hexadecimal
numbers: a restarted ramp ends bit for bit as the uninterrupted one, and a
checkpoint is not continued with another ramp.

## From the command line

```
python -m linac_gen ring ramp booster.madx --ramp ramp.json --csv turns.csv
python -m linac_gen ring run booster.madx --ramp ramp.json --to-end --particles 5000 \
    --checkpoint-every 1000 --out runs
```

`ring ramp` runs the reference alone and prints the turns to the end of the
energy program, the reference's energy and time at the end, transition, the
RF frequency range, the largest gain per turn and against the voltage, and
the turns capped; `--csv` writes the turn table (time, kinetic energy, γ,
gain, voltage, the synchronous phase of the net voltage, frequency, above
transition, capped), `--turns N` sets
the number of turns (needed for a flat program), `--json` prints the summary
as JSON.  `ring run --ramp FILE` runs a bunch through the ramp, `--to-end`
for the turns to the end of the program; a refused turn stops it with the
turns done written and exit status 1.  Both take `--gamma-t` (replacing the
ramp's) and the element-model options of `ring optics`; `ring run` also takes
`--corrector-mode follow|fixed_field`.  A project's own ramp (saved from the
Programs page) is used unless `--ramp` gives another or `--fixed-energy` none.

## In Python

```python
from linac_gen.cli.common import load_lattice
from linac_gen.core.config import BeamConfig
from linac_gen.ring import RingConfig, RingSimulation

lattice = load_lattice("tests/ring/fixtures/simple_booster.madx")
beam = BeamConfig(species="proton", energy=800.0, frequency=44.7044096,
                  emit_nx=1.5, emit_ny=1.5, emit_z=5.0, beta_z=100.0)
# 800 to 806 MeV in 0.1 ms at 1.2 MV
ramp = {"energy": {"kind": "table", "times_s": [0.0, 1e-4], "values": [800.0, 806.0]},
        "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [1.2, 1.2]},
        "gamma_t": 5.449167323}
r = RingSimulation(lattice, beam, RingConfig(ramp=ramp)).ramp()   # on the compiled ring
n = r.turns_to_end()                                # 54 turns
orbit = r.orbit(n)                                  # the reference alone
res = RingSimulation(lattice, beam, RingConfig(n_turns=n, n_particles=500,
                                               ramp=ramp)).run()
print(res.ramp["kinetic_end_mev"], res.emit_normalized[-1])
```

A ramp file reads with `linac_gen.ring.programs.ramp_from_file(path)` (a
`RampConfig`; `.to_dict()` for `RingConfig(ramp=...)`).  `res.ramp` holds the
turn-by-turn record and the summary above; `res.row_beta0gamma0` and
`res.row_p0c_mev` each row's reference.
