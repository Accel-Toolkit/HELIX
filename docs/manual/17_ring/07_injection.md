# Injection

Ring mode fills a ring turn by turn through a stripping foil: each turn of
the injection brings its macroparticles to the foil, where their first
crossing strips them — an H⁻ leaves as a proton — and the circulating beam
crosses the foil again wherever it overlaps it.  Bump and painting magnets
follow their waveforms turn by turn, so the closed orbit moves under the
injected beam and paints it into the ring's phase space.  Injection is
opt-in: a run without it is unchanged, bit for bit.

## The foil

**Where.**  The foil acts at the exit of an element of the ring: a foil
element of the lattice (a `; HELIX_FOIL` line of a TraceWin deck, or one
added in the lattice editor), or any element named in the settings — a
marker where the foil sits, for a MAD-X deck.  A ring has one foil: a
lattice with a foil element elsewhere than the foil the settings name is
refused (it would not act).  A foil element acts on the circulating beam
even without injection settings (a stored beam over a foil left in).

**Its rectangle** is in absolute ring coordinates at the foil: x and y from
`x_min_m` to `x_max_m` and from `y_min_m` to `y_max_m`.  A side left out is
open — or, for a foil element, at its extent about its offset.  A particle
crosses the foil when its position is inside the rectangle, edges included.
Over a closed rectangle the run maps the crossings and the energy they leave
on `map_bins` bins (x, y; default 40 × 40).

**Scattering.**  Multiple Coulomb scattering is a Gaussian kick in each
plane with the Highland width (V. L. Highland, Nucl. Instrum. Methods 129,
497 (1975)) in the form of G. R. Lynch and O. I. Dahl (Nucl. Instrum.
Methods B 58, 6 (1991)) that the Particle Data Group's review gives:

θ0 = (13.6 MeV / (β c p)) |z| √(x/X0) [1 + 0.038 ln(x z² / (X0 β²))],

added to the slopes x′ and y′ (the position is kept: the foil is thin).  It
is the formula of HELIX's foil element ([Foil](../03_elements/15_foil.md)),
evaluated for a proton at the ring's reference energy.  A particle that
crosses the foil again gets, at its k-th crossing, the growth of θ0² from
k − 1 to k thicknesses, √(θ0²(k x) − θ0²((k − 1) x)), so that its crossings
add up to Highland at their total thickness: θ0² is not additive over
layers (its logarithm grows with the thickness), and kicks of the
one-crossing width would leave a particle that crossed 600 µg/cm² 100 times
1.3 times too narrow in angle.  (A foil thinner than 4 × 10⁻¹² radiation
lengths, where the formula's logarithm turns the width negative, scatters
at no crossing: a foil that only counts its hits.)  Painting against a code
that scatters by single collisions, the foil's share of the horizontal
action comes out about 12 % larger in HELIX at the median
([validation](../12_validation/06_ring_benchmarks.md)).
Two limits matter:

* Highland's formula fits the core of the angle distribution to about 11 %
  for 10⁻³ < x/X0 < 100.  A stripping foil is thinner — 600 µg/cm² of carbon
  is x/X0 = 1.4 × 10⁻⁵ — and there θ0 is an extrapolation of the core's
  width (100 crossings of it, x/X0 = 1.4 × 10⁻³, are inside the range).
* Single large-angle Coulomb scattering and nuclear scattering, the tails
  that give the losses a foil causes, are not modelled.

**Energy loss.**  Each crossing loses the foil element's mean ionization
loss — the tabulated minimum-ionizing rate (`dedx_model = "mip"`, a floor
below about 2 GeV) or Bethe's formula (`"bethe"`) — with Landau straggling
for a thin foil and Gaussian (Bohr) straggling for a thick one (`"auto"`
chooses by Vavilov's κ).  The loss comes off the energy with the slopes
kept.

**Stripping.**  With the two-step model (`strip_model = "two_step"`) an
ion's first crossing leaves it a proton, H⁰ or H⁻ with the fractions of the
cross sections σ(−1→0) and σ(0→+1) of HELIX's foil element: calibrated for
carbon to two design values at 800 MeV (99.956 % stripped at 600 µg/cm²,
99.1 % at 380 µg/cm²; close to the yields measured by Gulley et al., Phys.
Rev. A 53, 3201 (1996)) and scaled as 1/β² with the velocity; for other
materials as √(Z/6), a convention.  The H⁰ and H⁻ are waste.  With `"off"`
every ion is stripped.  Left out, the strip model is the foil element's — a
`; HELIX_FOIL` line's own default is `"off"` — or, at any other element,
`"two_step"` for an H⁻ beam; a run that strips every H⁻ ion says so in its
notes.

**Energy in the foil.**  Turn by turn the run records the ionization energy
the crossings leave in the foil (each crossing's sampled loss) and, apart,
the kinetic energy (γ − 1) m_e c² of the electrons stripped from the
injected ions — 0.435 MeV each at 800 MeV (γ the ion's); two for an H⁻ that
leaves as a proton, one for an H⁰.  Where those electrons go — back into the
foil, or to a catcher — depends on the magnetic field around the foil, which
ring mode does not model.

Each pass of the foil draws its random numbers from its own stream — the
run's seed, the purpose (the circulating beam's crossings, the injected
ions', the linac bunches, the source's order) and the turn — in the order of
the particles' slots.  A run restarted from a checkpoint and a CUDA run draw
the same numbers.

## The injected beam

**Sources.**

* `"beam"` — the Beam tab's distribution (HELIX's generators, or the
  particle file the Beam tab reads), generated with as many particles as the
  injection needs.  Its species, energy and frequency are the injected
  beam's.
* `"file"` — a TraceWin `.dst`, HELIX's ASCII distribution (its
  `# w_kin_ref`, `# species:` and `# frequency:` header lines, as HELIX
  writes them), or TraceWin's particle ASCII file (the header line
  `Number_of_particle Mass(MeV) Beam_energy(MeV) Beam_Frequency(MHz) ...`,
  its particles with `Loss` 0).  The species comes from the settings, else
  from the file — a `.dst`'s or a particle file's rest mass and charge, a
  HELIX file's `# species:` line; a file that does not say is refused.
* `"gaussian"` — a Gaussian at the foil (an H⁻ beam unless `species` says
  otherwise): the normalized rms emittances (`emit_nx_um`, `emit_ny_um`,
  π mm mrad), the Twiss parameters, the rms momentum spread `sigma_dp`, a
  linac bunch's rms length `sigma_z_m`, an optional cut in σ (at least 1: a
  pair of normal numbers outside the cut is drawn again).

The settings' `species` and `frequency_mhz` replace the source's own.
`kinetic_mev` moves the beam to that energy — its reference energy, the
particles' offsets from it kept (for a Gaussian, the beam's energy).
`frequency_mhz` is the frequency the source's phases are counted in;
`bunch_frequency_mhz` is the linac's bunch rate — the bunches the chopper
sees — when it differs (a beam whose phases are given at 650 MHz from a
linac that bunches at 162.5 MHz, say; left out, the same).

**Stripping conserves velocity.**  An H⁻ leaves the foil as a proton with
the same velocity and direction: its momentum is m_p/m_H⁻ of the ion's,
1.0880 × 10⁻³ below it.  An 800 MeV H⁻ becomes a 799.1296 MeV proton —
δ = −7.067 × 10⁻⁴ in a ring whose reference is an 800 MeV proton.  The
injected particles keep their arrival time.

**The centroid.**  `x_m`, `px`, `y_m`, `py` place the injected beam's
reference trajectory in the ring's coordinates at the foil (px, py = P/P0);
each particle's offsets from its own reference are added to it.

**The draw.**  The source is a pool of particles.  The k-th injection turn
takes particles k n to k n + n − 1 of it (n per turn) — in a seeded shuffle
of the pool (`"shuffled"`) or in its own order (`"in_order"`) — and starts
over when the pool runs out (the run's notes say how many times).

## Turns, current and buckets

The injection runs from turn `start_turn` for `turns` turns, `particles_per_turn`
macroparticles each (at most 10⁸ macroparticles in all).  `current_ma` is
the average current injected during the pulse, after the chopper: a turn of
length T brings I T / e particles.  They share the ring's filled buckets —
with the RF on, the RF buckets C/L (L the RF period: C/h for the greatest
common divisor h of the harmonics of the cavities with a voltage) less
`notch_buckets`; without RF, the whole ring — and each macroparticle of the
tracked bucket stands for I T / (e B n) of them (B the filled buckets).

The real particles the injection's record counts — injected, crossing the
foil, the energy they leave there — are the ring's: every filled bucket's,
B times the tracked one's.  The ledger, like the macroparticles' weights,
counts the tracked bucket's.

With space charge the bunch's intensity is the macroparticles' sum: leave
the space-charge intensity out.  The grid and the frozen model's Gaussian
follow the circulating charge: the macroparticles of a beam at the start and
of the injected beam carry different weights and count by them.  A frozen
model is measured again every turn that injects and on the turn after the
last — the beam is growing — and then as its settings say.  The run reports
the intensity of the circulating beam at the end, and the analytic tune
shift of the beam then.  The run's statistics — sizes, emittances, and the
emittances and bunch length that tune shift takes — count macroparticles:
with a beam at the start and an injection they are per macroparticle, not
per charge.

A run whose injection starts after its last turn, or ends after it, says so
in its notes.

## The chopper

Linac bunch j (counted from the injection's start) reaches the foil at
t_j = j / f_L, f_L the bunch rate.  The injection turn that starts at t_k
receives the bunches with t_j in [t_k, t_k + T).  A bunch arriving dt after
the reference particle passes the foil is behind it by β0 c dt: it lands at
ζ = −β0 c dt − L φ0/360, wrapped into the RF period L (the circumference
without RF) — every bucket is the same, so the wrap puts it in the tracked
one.  `chopper_phase_deg` is the ring RF phase φ0 of the first bunch
(> 0: late).  With the RF on, the chopper keeps the bunches within
`chopper_window_deg` of the bucket's centre (180: every bunch).  Each
macroparticle is placed on one of the turn's kept bunches (drawn at random
from its own stream), at its own offset in the bunch.  A turn whose bunches
the chopper keeps none of injects nothing; the run's notes count such turns.
A source with no bunch frequency is an unbunched linac beam, spread over the
period — with the RF on and a chopper window below 180° it is refused before
the run starts.

## Bump and painting magnets

A magnet group names elements of the ring — every element of that name,
wherever it is placed — and how their strength follows the group's
waveform w(t):

* `"kick"` — a kicker's deflection, `kick_rad` × w(t) in the group's
  `plane` (without `kick_rad`, the element's own kick × w(t)); with a plane,
  only that plane's deflection follows the waveform — the other keeps the
  element's own setting;
* `"dipole"` — a thick kicker (as the importers write it: half drift, thin
  kick, half drift), a bend with no angle, or a drift (a magnet the deck
  holds with no field — a bend with neither angle nor field is read as one),
  compiled as an exact dipole in a straight frame: the hard-edge dipole
  fringe at each face (exact, with the fringe integral `fint`, the half gap
  `hgap_m`, and the face angles `e1_rad`, `e2_rad` with their wedges), and
  the exact trajectory through the uniform field between (a circle arc in
  the straight frame).  Its field is k0 = −kick/L × w(t) (L its length; in
  the y plane the same, in a frame turned by 90°).  A magnet with no field
  of its own needs the group's plane and `kick_rad`.  A bump of four such
  magnets has its exact height at the foil — the arcs', not the paraxial
  one;
* `"scale"` — the elements' own kicks and multipole strengths × w(t) (the
  error multipoles of a bump magnet, say).

An element belongs to one group.  The waveforms are functions of the time t
from the start of the run (in a ramp, the ramp's clock less its start):

| kind | the factor |
|---|---|
| `"constant"` | `value` |
| `"table"` | linear between (`times_s`, `values`), the end values held |
| `"linear"` | `start` until `t_start_s`, linear to `end` at `t_end_s`, held after |
| `"sqrt"` | start + (end − start) √u, u = (t − t_start)/(t_end − t_start) in [0, 1] |
| `"sqrt_end"` | end + (start − end) √(1 − u) |

A closed orbit moved as √t away from the injected beam gives each turn's
particles an action proportional to t: a distribution uniform in action, as
painting intends.  The strengths are normalized: a waveform gives the
deflection at the reference momentum of the moment (in a ramp, the magnets
follow the energy).  Each turn the magnets are set at its start (one turn
per launch); at turn 0 the optics and the matched beam see them.

The run records the 4-D closed orbit at the foil (δ = 0, RF off) at the
start of every turn, and after the last — a Newton search from the last
orbit, made again only when the magnets changed.  It is not recorded in a
ramp (the reference changes within the turn).

## How a turn goes

With a foil the turn stops there: the beam is tracked to the foil, the
circulating particles inside its rectangle cross it, the turn's injected
particles are added at the foil (stripped, or waste), and the beam is
tracked to the end of the turn.  With space charge the foil is one more
event between the kicks, before a kick at the same node (the particles it
injects feel that kick).  The tune probes do not cross the foil.  On a CUDA
GPU the particles inside the foil's rectangle are found on the GPU, and only
they come to the host for the foil and go back, with the slots the
injection fills; the slots used so far are the ones tracked, on the CPU and
on the GPU alike.  A ring that starts empty (the
Run page's **None: the injection fills the ring**, `source = "empty"`) is
not stopped as "every particle lost" before its injection is over.

## The Injection page

**Ring → Injection** holds the settings: injection off or on; the turns,
the first injection turn, the macroparticles a turn, the current and the
empty buckets; the foil (its element, chosen from the lattice's markers and
foil elements or typed; material, thickness, straggling, mean loss,
stripping, rectangle in mm); the injected beam (the Beam tab's, a file, or
a Gaussian with its emittances, Twiss, momentum spread and bunch length; the
species, energy, the frequency of its phases and the linac's bunch rate; the
centroid in mm and mrad; the draw); the chopper; and the magnet groups in a
table — name, elements, model, plane, kicks (mrad), waveform (times in ms:
`1`, `linear t0 t1 start end`, `sqrt t0 t1 start end`, `sqrt_end t0 t1
start end`, `table t1:v1 t2:v2 ...`, `file name.csv` — a relative name from
the project's folder), and for a dipole its fringe (fint, half gap in mm)
and faces (e1, e2 in mrad).  With the optics computed it shows the real
particles each turn brings and — with the foil's thickness, typed or its
foil element's — the foil's θ0 and mean loss.  Every valid change goes into
the project; an entry that is not valid is reported and the project keeps
the last valid settings.

While the injected beam is the Beam tab's distribution, the Beam tab holds
the injected beam: Ring & Optics → Use the matched beam in the Beam tab then
refuses (it would replace the injected beam with the ring's matched one).
For a matched beam at the start of the ring as well as an injection, inject
a Gaussian at the foil or a particle file.

Turning injection on starts the ring empty (the Run page's **Particles
from** becomes **None: the injection fills the ring**; another source can be
chosen after), and turning it off selects the Beam tab's beam again.  The Ring &
Optics page shows the ring without the injection's magnets; a run's optics
are those of its turn 0 — the bump magnets as exact dipoles, at their
strengths then.

The Results tab's **Ring injection** shows the macroparticles injected, the
waste and the circulating beam's foil crossings turn by turn (each at the
turn counted from 1, where the circulating count rises), the particles
circulating, the closed orbit at the foil with each turn's magnets, and the
foil's hit map.  With injection the
survival — on the Run page's live strip and in the Results tab's
transmission — is of the particles the ring received (those it started with
and the stripped ones so far) or, for a ring that started empty, the count
of particles circulating.

## What a run keeps

`RingResults.injection` (the results file's `/injection` group):

* per turn: `injected`, `stripped`, `h0`, `hminus`, `missed`
  (macroparticles), `injected_charge` (the real particles injected into the
  ring), `foil_hits` (the circulating beam's crossings, macroparticles),
  `foil_hits_injected`, `deposit_mev` (the ionization energy left in the
  foil, MeV) and `electron_mev` (the stripped electrons' kinetic energy,
  MeV);
* `closed_orbit_foil` (turns + 1, 4: x, px, y, py; row k at the start of
  turn k, the last row after the last turn; NaN in a ramp) and `waveforms`
  (turns + 1, groups: the waveforms then) with `waveform_names`;
* `hit_map`, `deposit_map_mev` (the real particles crossing each bin and
  the ionization energy they leave, over a closed rectangle, `map_extent_m`),
  `hits_histogram` (the macroparticles crossing the foil k times),
  `mean_hits_circulating`;
* `foil` (its settings with θ0, the mean loss, the straggling used and the
  stripping fractions), `settings`, `source`, `waste` (missed and not
  stripped, macroparticles and the tracked bucket's real particles),
  `buckets`, `notes`.

The real particles and energies are the ring's (all its filled buckets).
The ledger holds exactly: injected = circulating + lost + waste.  A
checkpoint keeps the injection's record; it is not continued with other
injection settings.

## From the command line

```
python -m linac_gen ring run booster.madx --injection inj.json --turns 400 --out runs
```

`--injection FILE` reads the settings from a JSON file (YAML with PyYAML
installed); `--no-injection` runs without the settings a project holds.
With injection the run starts with an empty ring unless `--source` says
otherwise.  A magnet group's table may come from a CSV file (`"waveform":
{"file": "bump.csv"}`, header `t_s,value`); it and a file source's path are
taken relative to the settings file.

```json
{"foil": {"at": "foil", "thickness_ug_cm2": 600,
          "x_min_m": -0.01, "x_max_m": 0.01, "y_min_m": 0.0, "y_max_m": 0.02},
 "source": {"kind": "gaussian", "species": "H-", "kinetic_mev": 800,
            "frequency_mhz": 162.5, "emit_nx_um": 0.3, "emit_ny_um": 0.3,
            "beta_x_m": 10, "beta_y_m": 10, "sigma_dp": 1e-4, "y_m": 0.01},
 "turns": 300, "particles_per_turn": 20, "current_ma": 2.0,
 "chopper_window_deg": 120,
 "magnets": [{"name": "bump", "elements": ["kb1", "kb2", "kb3", "kb4"],
              "model": "dipole", "fint": 0.5, "hgap_m": 0.02,
              "waveform": {"kind": "sqrt", "t_start_s": 0, "t_end_s": 5.6e-4,
                           "start": 1.0, "end": 0.3}}]}
```

The run's summary prints the foil's θ0 and mean loss, the macroparticles
injected and left as waste, the real particles injected into the ring, the
foil crossings, and the ionization energy left in the foil with the
stripped electrons' energy apart.  With `--json`, a number that is not
finite (the emittances at the start of a ring that starts empty) is `null`.

## In Python

```python
from linac_gen.cli.common import load_lattice
from linac_gen.core.config import BeamConfig
from linac_gen.elements.marker import Marker
from linac_gen.ring import RingConfig, RingSimulation

lattice = load_lattice("tests/ring/fixtures/simple_booster.madx")
lattice.elements.insert(1, Marker(name="foil"))
beam = BeamConfig(species="proton", energy=800.0, frequency=44.7044096)
inj = {"foil": {"at": "foil", "thickness_ug_cm2": 600.0,
                "x_min_m": -0.01, "x_max_m": 0.01, "y_min_m": -0.01, "y_max_m": 0.01},
       "source": {"kind": "gaussian", "species": "H-", "kinetic_mev": 800.0,
                  "frequency_mhz": 162.5, "emit_nx_um": 0.3, "emit_ny_um": 0.3,
                  "beta_x_m": 30.0, "beta_y_m": 5.0},
       "turns": 3, "particles_per_turn": 50, "current_ma": 2.0}
res = RingSimulation(lattice, beam, RingConfig(n_turns=4, source="empty", injection=inj,
                                               tune_probes=0)).run()
print(res.n_alive, res.ledger["waste"], res.injection["foil"]["theta0_rad"])
```

`linac_gen.ring.injection.config.InjectionConfig` validates the settings
(its `to_dict()` is what `RingConfig(injection=...)` takes);
`linac_gen.ring.injection.config.injection_from_file` reads a settings file.
