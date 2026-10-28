# Space charge

Ring mode's space charge is the 2.5-D model: at each kick the bunch's
transverse distribution, taken over the whole bunch, gives a two-dimensional
field, and each particle feels it in proportion to the line density where it
is.  The slope of the line density gives a longitudinal field.  This is the
field of a bunch long compared with its width — a synchrotron's bunch — with
one transverse shape along the bunch.  It is opt-in: a run without space
charge is unchanged, bit for bit.

## The kick

With N_b the real particles in the bunch, f(ζ) its normalized line density
(∫ f dζ = 1 over the RF period), r_c = e²/(4πε0 mc²) the classical radius of
the particle and ds the length a kick stands for:

* the transverse kick, dp_x = K_t f(ζ) F_x, dp_y = K_t f(ζ) F_y, with
  K_t = 2 r_c N_b ds / (β0² γ0³);
* the longitudinal kick, dp_τ = −K_l f′(ζ) Φ, with K_l = 2 r_c N_b ds / (β0 γ0³).

Φ = −∫ ρ ln|r − r′| dA′ is the potential of the normalized transverse
distribution ρ (∫ ρ dA = 1) in units of 1/(2πε0), and F = −∇Φ its field.
The transverse kick is the electric field of the bunch less the magnetic
force of its current (a factor 1/γ0² altogether); the longitudinal one is
the field E_z = −λ′φ/γ0² of a long bunch, λ = eN_b f.  β0 and γ0 are the
reference's at the kick (in a ramp, after the cavities before it).

A **coasting beam** (no RF) has a uniform line density, N_b over the
circumference (N_b is then the whole ring's), and no longitudinal kick.

## The field

The circulating particles' weights go to a grid of nx × ny nodes (default
64 × 64, at most 1024 × 1024) with the TSC shape (3 × 3 nodes; or CIC, 2 × 2).  The grid is
centred on the beam and covers it out to its extent or `box_sigma` rms sizes
(default 4), whichever is smaller — the centroid and rms sizes of the beam's
charge, each macroparticle counted by its weight (a beam at the start and an
injected one carry different weights); its cells come from a fixed ladder of
sizes (ratio 2^(1/8)) and its origin sits on whole cells, so the Green's
functions computed for one kick serve the next ones.  The charge of each
node is spread over its cell, and the potential and field at every node are
exact sums over the cells: Green's functions integrated over a cell (the
integrated Green's function of Qiang, Lidia, Ryne and Limborg-Deprey, Phys.
Rev. ST Accel. Beams 9, 044204 (2006), here in two dimensions), convolved
with the charge by FFT on a grid twice the size (Hockney and Eastwood,
*Computer Simulation Using Particles*, 1988).  A particle gathers the field
with the shape it was deposited with, so its own charge gives it no kick and
the kicks conserve the bunch's transverse momentum.  Particles beyond the
grid feel the field of the grid's charge: its multipole series (to order 16)
about the beam's centre where that series converges fast (beyond 1.5 times
the farthest charge — a flat beam's grid reaches farther in its wide plane
than a particle just past it in the narrow one), the direct sum over the
charged cells nearer (that sum costs each such particle about the number of
charged cells: microseconds on 64 × 64, about a millisecond on 1024 × 1024 —
a beam with long tails beyond `box_sigma` on a fine grid kicks noticeably
slower; a larger `box_sigma` keeps more of it on the grid).  They carry no
charge into the field, which is
normalized to the whole bunch's charge (with the grid at 4 rms sizes they are
a few in ten thousand of a Gaussian bunch; a particle-in-cell run reports the
most at any kick).

The field of a Gaussian beam computed this way, on the exact charge of each
cell, is the Bassetti–Erskine field to 3.0–3.4 × 10⁻³ of its largest value
inside 3 rms sizes on 64 × 64 nodes over 8 rms sizes and 7.0–8.0 × 10⁻⁴ on
128 × 128 (the larger figures for a beam three times wider than high; the
error falls by 4.2–4.5 each time the cell is halved, second order).  A run's
grid is a little coarser for the same nodes: its usable region is two cells
narrower and its cells are rounded up to the ladder (up to 9 %).

The deposit is made of integers — weight × shape × 2^k, rounded, 2^k
chosen so that the bunch sums to about 2^60 — whose sums do not depend on
their order: a run gives the same bits for any number of CPU threads, and
wherever numpy happens to place its arrays (the products of complex numbers
are written on their real and imaginary parts: numpy's complex multiply
rounds differently for differently aligned arrays).  The GPU adds the
integers atomically; a particle's grid coordinate is a division and its
weights are written without a multiply-add the GPU could fuse, so for the
same particles it gets the CPU's integers — checked in numba's CUDA
simulator; the CUDA session checks it on a GPU.

## The pipe

* **Open** (default): no wall.  The transverse field is the free-space one;
  the potential, which the longitudinal kick needs, is zero at the reference
  radius r_ref (default 1 m) from the beam.
* **Round** (radius b): exact images — a charge q at w inside a grounded
  circle has its image −q at b²/w̄; their field is a power series in the
  position, summed from the grid charge's moments to the order that holds
  to 10⁻¹³ anywhere inside the pipe (at most 128: a beam reaching 90 % of the
  radius keeps 10⁻⁶ at the wall).  Charge at or beyond the wall — the share a
  particle close to it spreads onto a node past it, or particles no aperture
  removed — is left out of the images; a run counts the kicks where that
  happened.  Each particle inside the wall adds the images' field and
  potential at its own position — the series, not values interpolated from
  the grid's nodes, which near the wall would mix nodes on both sides of it;
  a particle sums the terms its own radius needs for the same accuracy (far
  fewer near the beam's centre).
  A particle at or beyond the wall (about to be lost at an aperture) feels
  the beam's own field, as with no wall — there the images' series does not
  converge — and only the series' constant term, so that the potential has
  no step at the wall.
* **Rectangle, ellipse** (half width a, half height b): the free-space
  potential at points on the wall, cancelled there in the least-squares
  sense by a harmonic of `boundary_order` terms (default 24) — the method of
  F. W. Jones (EPAC 2000, p. 1381); the wall potential comes from the grid's
  node charges as point charges (within 10⁻⁴ of the beam's field of the
  cell-integrated charges), and the least-squares operator is computed
  without BLAS, so it does not change with BLAS's threads.  As for the round
  pipe, the harmonic is added at each particle inside the wall.  A shape the
  harmonics cannot fit is refused: before a run, unit line charges at the
  centre and out to half the aperture — (0.5 a, 0), (0, 0.5 b),
  (0.35 a, 0.35 b), (0.3 a, −0.2 b) — must leave less than 0.01 of potential
  on the wall, checked at sixteen times as many wall points as the fit used
  — evenly spaced, with a rectangle's corners (where the harmonics fit it
  worst) and an ellipse's axis ends among them (a unit line charge's
  potential changes by 1 per e-fold of distance).  The fit's error is
  harmonic inside the wall, so this bounds the error of the potential
  anywhere inside for a beam within half the aperture (a beam nearer the
  wall sees more).  For example, with 24 harmonics a rectangle of 60 × 25 mm
  leaves 5.5 × 10⁻³ and one of 40 × 10 mm is refused (0.026; with 32
  harmonics too, 0.011; with 40 it leaves 5.6 × 10⁻³, with 48,
  3.5 × 10⁻³); rectangles eight or more times wider than
  high are refused at every number up to 64 (the residual does not always
  fall as the number grows).  The run's summary reports what the fit leaves.
  Charge at or beyond the wall is cancelled inside with the rest by a fitted
  pipe; a round pipe leaves it out of its images (and counts it).

The pipe is the same all around the ring and centred on the reference orbit:
a beam off centre feels its images.  It only grounds the field — particles
are lost at the lattice's apertures, not at this wall.  The longitudinal kick scales with the
geometry: on the axis of a uniform round beam of radius a in a pipe of radius
b, Φ = ½ + ln(b/a), the geometry factor g = 2Φ = 1 + 2 ln(b/a) — with no wall,
the reference radius stands for b.

## The line density

The circulating particles' weights go to `z_bins` bins over the RF period
C/h (CIC, periodic); the line density f(ζ), which scales the transverse
kick, and its slope f′(ζ), which the longitudinal kick follows, are filtered
by Gaussians and the slope taken spectrally.

**"auto", the default.**  The bins follow the bunch: 16 bins per rms of the
bunch's core, as a power of two from 16 to 65 536.  The core's rms is 1.4826
times the median absolute deviation of ζ — a halo does not hide a short core
— measured on the circle of the RF period (a bunch lying across the period's
ends is measured whole) from the line deposited at the first kick of each
turn, the deposit's own widening taken off.  The bins it asks for are used
from the next turn on, and they change only once it asks for more than 1.68
times the bins in use, or less than 1/1.68 of them, so that a bunch near the
point between two powers of two does not switch them to and fro.  The noise
of N particles is held down by smoothing, not by coarser bins: the line
density — and with it its slope — is filtered over (4/3)^(1/5) σ N^(−1/5)
(an rms width; σ the core's rms), the normal-reference bandwidth of kernel
density estimation, which balances an estimate's noise against its smoothing
(B. W. Silverman, *Density Estimation for Statistics and Data Analysis*,
1986), and never over less than `z_smoothing_bins` bins.  The width shrinks
as particles are added, and with it both the noise and the smoothing: for
5000 particles it is 0.19 σ (a Gaussian keeps 98 % of its peak, and 95 % of
its slope's linear part), for 50 000 it is 0.12 σ (99 % and 98 %).  A run starts with the bins and width of
its starting beam (a ring that starts empty: 256 bins until a turn finds
particles); with fewer than 100 particles the bins and width stay as they
are.  The frozen model changes them only on the turns it measures
the beam again.  They come from the deposit's integers: a run gives the same
bins — and the same bits — on any number of threads, on a GPU as on the CPU,
and from a checkpoint.  The method assumes one bunch per RF period.

A number from 8 to 65 536 fixes the bins; the filter is then
`z_smoothing_bins` bins wide (default 1).

**Why the noise matters.**  A macroparticle line density is noisy, and the
kicks it gives heat the beam.  On the simple Booster's bunch inside its
bucket (the [validation case](../12_validation/06_ring_benchmarks.md#losses-space-charge-and-painting-over-many-turns),
the longitudinal kick on, 300 turns), the longitudinal emittance grows by
2.58 % with 50 000 particles and 256 fixed bins, by 2.40 % with "auto"; with
5000, by 2.62 % with "auto", 3.65 % with 256 fixed bins and 11.0 % with 1024.  Through transition with the transverse
kick alone (the bunch rotating to 3.4 cm), the horizontal betatron emittance
grows by 36.2 % on average with "auto" at 5000 particles (four samples of
the bunch, 34.4–37.9 %) and by 36.1 % at 20 000; with 2048 fixed bins by
41.2 % at 5000 (39.2–42.9 %), with 4096 by 48.5 % (44.7–49.9 %) — finer
bins, more noise — although the line density the particles see differs by
less than 0.3 % between the two.

**What the bins and the filter keep.**  A Gaussian bunch of rms length σ
looks like one of √(σ² + (filter)² + Δz²/3), its slope correspondingly
lower.  Near transition a bunch can be short: one 3.4 cm long needs bins of
a few millimetres — with 256 bins over the simple Booster's 5.6 m RF period
its peak line density comes out 20 % low, and the space charge it feels with
it.  A run notes it when, for the shortest bunch core over the turns with
statistics, the bins and the filter keep less than 97 % of a Gaussian's
peak; the note gives the core's length, the bins and what kept the line
density coarse — too few bins (fixed), the smoothing few particles need
("auto": more particles resolve it), the bins following a bunch that
shortened faster than they did (they follow it a turn late, in factors of
two), or their largest number — and the record keeps them
(`line_density_resolution`).  The
fraction is a Gaussian's — a flat-topped bunch loses less, a sharp peak more.

The **longitudinal kick** is computed from the potential (default), from a geometry factor g
given for every particle (`g_factor`), or left out.

## The kicks

About `kicks_per_turn` kicks (default 100, at most 1 000 000) at element
boundaries — the boundary nearest each of equally spaced positions from the
start of the ring, so that a ring of identical cells gets the same kicks in
every cell when the number asked is a multiple of the cells; none falls
inside an element (a zero-length element at the very end of the ring starts
where the ring starts: it gets no kick of its own; an element of negative
length is refused).  Each stands for the ring from halfway to the kick before to halfway
to the next.  ζ is brought back into the RF period once a turn, at its end,
as without space charge (in a ramp the period in ζ changes within the turn);
the line density is periodic, so a particle a little beyond the period
between two kicks is binned where it belongs.  As a rule of thumb, give ten kicks per betatron wavelength or
more (the Space charge page shows how many the settings give); a run
reports the largest spacing.  With space
charge the turn is tracked from kick to kick, one turn per launch.

The **tune probes** feel the field without adding to it.  With space charge
the single tune probe starts at 0.2 rms beam sizes (instead of 0.01): the
beam's centroid jitters by about 1/√n rms sizes and drives a probe at the bare
tune, which at 0.01 rms sizes would swamp its own motion.  A Gaussian beam's
tune shift there is 1.2 % below its centre's.

The **frozen model** replaces the field solve by the field of a Gaussian
with the centroid and rms sizes of the beam's charge in x and y at each kick (an x–y tilt
of the beam is not modelled; solved on the same grid, with the same pipe),
and the line density by the beam's, measured every `refresh_turns` turns (0:
only at the start).  Between measurements a turn costs only the kicks; the
fields of the first kicks are kept, up to 512 MiB, the others solved again
each time (the same bits; a turn visits the kicks in order, so dropping the
oldest to make room would keep none it needs next).

## Intensity and noise

`intensity` is the real particles in the bunch, positive; left blank, it is
the Beam tab's current over its frequency for a bunched beam (one bunch), and
the current times the revolution time for a coasting beam (the whole ring).  Each
macroparticle carries its share; a lost one takes its charge with it.

A field made from n macroparticles carries their sampling noise.  Turn after
turn the noise heats the beam: an emittance growth that falls as the number
of particles grows and depends on the grid.  Before reading a growth as
physics, run with more particles and another grid: what does not change is
the beam's.  A few particles per cell in the beam's core is too few for long
runs (5 000 particles on a 128 × 128 grid over 8 rms sizes give about three
at the centre); 64 × 64 with 50 000 particles gives about 150.  The
[validation page](../12_validation/06_ring_benchmarks.md#space-charge) shows
such a check.

## The analytic tune shift

The incoherent tune shift at the centre of a Gaussian beam (L. J. Laslett,
BNL-7534, 1963),

dQ_x = −(r_c N_b f / (2π β0² γ0³)) ∮ β_x / (σ_x (σ_x + σ_y)) ds,

σ_x = √(ε_x β_x + (D_x σ_δ)²) and f the line density at the centre
(1/(√(2π) σ_ζ) for a Gaussian bunch, 1/C for a coasting beam); a KV beam of
the same rms sizes has half of it, the same for every particle.  The Space
charge page shows it for its settings, the Beam tab's emittances and the
optics of the Ring & Optics page — for a ring with RF, from its 6-D optics
(the bunch length; with 4-D optics the page asks for them), for a ring
without RF, a coasting beam's line density 1/C; a run keeps it for its
starting beam.

## The Space charge page

**Ring → Space charge** holds the settings: the model (off, 2.5-D
particle-in-cell, frozen Gaussian), the particles in the bunch, the kicks per
turn, the frozen model's refresh, the grid, the charge shape, how far the
grid reaches, the line-density bins and filter, the pipe and its size (mm),
the harmonics fitted to a rectangle or an ellipse, the reference radius, the
longitudinal kick and its geometry factor g.  With the optics computed it
shows the analytic tune shift and the kicks per betatron wavelength — of the
kicks the request places (at element boundaries: fewer than asked when the
ring's elements are fewer).  Fields that do not apply to the choices made are
greyed out and not read.  Every valid change goes into the project; an entry
that is not valid is reported and the project keeps the last valid settings
(a run refuses to start until the page is valid).  The Run page shows the
settings in a line, and a run's summary says what it measured: the kicks, the
analytic tune shift, the particles that were off the grid.

## What a run keeps

Besides what every run keeps ([Ring runs](04_run.md)), the results' meta
section `space_charge` holds the settings, the intensity used
(`intensity_used`: the settings', or from the Beam tab's current — over its
frequency for a bunched beam, times the revolution time for a coasting one),
the kicks' positions and lengths, the largest spacing, whether the beam was
bunched and had a longitudinal kick, the most particles off the grid at a
kick, the kicks with charge beyond a round pipe's wall, the potential a
fitted pipe leaves on its wall (`pipe_fit_residual`) and the analytic tune
shift; for a bunched beam, the line-density bins used and from which turn
(`z_bins_used`, the first and last hundred changes of a long run;
`z_bins_changes`, how many in all; `z_bins_auto`) and
`line_density_resolution` — the shortest core, its turn, the bins and the
line filter's width then, the fraction of a Gaussian's peak they keep, and
with "auto" what kept them coarse (`limited_by`).
A checkpoint keeps the frozen model's measurements, "auto"'s bins and width
and the run's counters (the most particles off the grid at a kick, the kicks
with charge beyond a round pipe's wall — a continued run reports the
uninterrupted run's); a checkpoint is not continued with other space-charge
settings.

## From the command line

```
python -m linac_gen ring run booster.madx --turns 1000 --particles 50000 \
    --space-charge sc.json --out runs
python -m linac_gen ring run booster.madx --turns 1000 --intensity 5e10 --out runs
```

`--space-charge FILE` reads the settings from a JSON file (YAML with PyYAML
installed) — the keys below, any left out at their defaults; `--intensity N`
sets the particles in the bunch (alone: the default settings with that
intensity); `--no-space-charge` runs without the settings a project holds.

```json
{"model": "pic", "intensity": 5e10, "kicks_per_turn": 96, "grid": [64, 64],
 "deposit": "tsc", "box_sigma": 4, "pipe": "round", "pipe_a_m": 0.04,
 "longitudinal": "potential", "z_bins": 128, "z_smoothing_bins": 1}
```

The other keys: `reference_radius_m` (open pipe), `pipe_b_m` and
`boundary_order` (rectangle, ellipse), `g_factor` (with
`"longitudinal": "g_factor"`), `refresh_turns` (frozen model).  The run's
summary prints the space charge it had and the analytic tune shift.

## In Python

```python
from linac_gen.cli.common import load_lattice
from linac_gen.core.config import BeamConfig
from linac_gen.ring import RingConfig, RingSimulation

lattice = load_lattice("tests/ring/fixtures/simple_booster.madx")
beam = BeamConfig(species="proton", energy=800.0, frequency=44.7044096,
                  emit_nx=2.5, emit_ny=2.5, emit_z=5.0, beta_z=100.0)
sc = {"intensity": 1e10, "kicks_per_turn": 24, "grid": [32, 32]}
res = RingSimulation(lattice, beam, RingConfig(n_turns=3, n_particles=500,
                                               space_charge=sc)).run()
print(res.meta["space_charge"]["laslett_gaussian"], res.emit_normalized[-1])
```

`linac_gen.ring.sc.config.SpaceChargeConfig` validates the settings (its
`to_dict()` is what `RingConfig(space_charge=...)` takes);
`linac_gen.ring.sc.laslett.tune_shift` gives the analytic tune shift for an
optics and a beam.
