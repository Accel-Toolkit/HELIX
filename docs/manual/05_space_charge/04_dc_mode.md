# DC mode (continuous beam)

Pre-RFQ Low-Energy Beam Transports (LEBTs) operate on a continuous,
unbunched beam — there's no longitudinal structure yet.  HELIX has
DC-aware code paths in both the envelope and multi-particle solvers
to handle this regime correctly.

## TL;DR

For any pre-RFQ run:

```python
from linac_gen.core.config import BeamConfig

beam_cfg = BeamConfig(
    species="H-",
    energy=0.030,            # 30 keV LEBT exit
    frequency=162.5,         # downstream RFQ frequency (used for unit conversions)
    current=5.0,
    n_particles=20_000,
    distribution="gaussian", cutoff=4.0,
    emit_nx=0.20, emit_ny=0.20,
    alpha_x=0.0, beta_x=0.10,
    alpha_y=0.0, beta_y=0.10,
    emit_z=0.0,              # no longitudinal structure
    alpha_z=0.0, beta_z=1.0,
    continuous=True,         # ← critical
    dc_energy_spread_keV=0.0,
)
```

The `continuous=True` flag triggers DC-aware behaviour throughout
the simulator:

* The longitudinal coordinate is **uniformly sampled** over one RF
  period during particle generation (so SC averaging is correct).
* The PIC solver switches to its **2-D analytic / Bassetti-Erskine
  / 2-D PIC** kick (per `dc_kernel`).
* The recorder marks each step with `continuous_at[i] = True` so
  consumers know σ_φ / σ_W / ε_z are not physically meaningful.

## Choosing a DC kernel

`SpaceChargeConfig.dc_kernel` selects the 2-D transverse kick model:

| `dc_kernel` | Field model | Speed | Use when |
|---|---|---|---|
| `"uniform"` | analytic linear uniform-elliptical kick | fastest | matching, fast scans, default |
| `"gaussian"` | Bassetti-Erskine field of a 2-D Gaussian | medium (see below) | rigid-σ MP, per-particle non-linearity |
| `"pic2d"` | 2-D Hockney FFT PIC over actual particle distribution | slowest | most accurate; analogue of TraceWin's PICNIC_2D |

```python
from linac_gen.core.config import SpaceChargeConfig

sc = SpaceChargeConfig(
    dc_kernel="gaussian",   # or "uniform" / "pic2d"
)
```

The `"gaussian"` field is that of an upright Gaussian with the beam's
projected σ_x and σ_y: an x–y correlation of the beam (a beam rotated by a
solenoid, say) is not taken into account, where `"uniform"` turns to the
beam's principal axes once the correlation stands above its sampling noise.
Within that model it is within 1e-10 of the exact field wherever the field is
above 1e-2 of its peak, and within 1e-12 of the peak everywhere, for any
σ_x/σ_y, a round beam included.  It costs two Faddeeva-function evaluations
per particle and kick, more than `"uniform"`; for a beam a few per cent from
round it can cost more than `"pic2d"`.

## Common gotchas

!!! warning "Forget `continuous=True` and your LEBT explodes"
    If you forget to set this for a pre-RFQ LEBT, HELIX assumes the
    beam is bunched and imposes a non-physical longitudinal
    structure.  σ_φ blows up, ε_z grows.  Symptoms: looks like SC
    physics is wrong, but it's just a flag.

!!! warning "PXIE LEBT needs Ki=1 + envelope continuous=True"
    Two silent failure modes:
    1. The PXIE LEBT field maps have `Ki=1` (full SC compensation).
       Forgetting this gives the wrong forces.
    2. The envelope solver also needs `continuous=True` to skip
       longitudinal SC.
    Both together: PXIE LEBT mean error 6.93% → 0.84%.

## DC-aware envelope solver

The Sacherer ODE (`linac_gen.tracking.sacherer`) is the DC envelope
counterpart.  It solves the continuous-beam envelope ODE:

$$
\sigma_x'' + \kappa_x(s)\,\sigma_x - \frac{\varepsilon_x^2}{\sigma_x^3}
- \frac{2K}{\sigma_x + \sigma_y} = 0
$$

(and similarly for y) with generalised perveance K = qI / (2π ε₀ m
c³ (βγ)³).  No acceleration.  What it reads, element by element:

* drifts, hard-edge quadrupoles and `SOLENOID` cards;
* **solenoid field maps** — a static magnetic map whose field on the
  axis is longitudinal (a 1-D B_z or 2-D cylindrical map, or a 3-D map
  whose on-axis B_z dominates: the rule the failure compensation uses
  to tell a solenoid).  κ(s) = (B_z(s) / (2Bρ))² in both planes, with
  B_z on the axis (a 3-D map's interpolated at x = 0, y = 0) at the
  amplitude every other mode uses: scale × kb × (1 + `voltage_rel`) /
  the file's `Norm`.  So kb = 0 and a failure's OFF (`voltage_rel` = −1)
  are no field, and an error study's `voltage_rel` draws move it;
* **steering correctors** — a powered 3-D map whose field on the axis is
  a dipole's (transverse, with no quadrupole gradient of a quadrupole's
  size there) is left out, as a `STEERER` card is: it steers the beam,
  which the solver does not follow (a B_z it holds on the axis focuses
  the beam as a solenoid map's does).  The correctors of one map, or a
  `SUPERPOSE_MAP` cluster's corrector children together (their fields
  summed, each over its window), are left out while they bend the beam
  by at most 20 mrad (θ = the largest |∫ amplitude × (B_x, B_y) dz|
  along the element, over Bρ: the largest angle the beam reaches in it,
  both planes together) and focus it, as the beam sees it, by at most
  1/f = 0.0088 /m: the largest |∫ amplitude × (G_n + i G_s) e^(−2iθ_L) dz|
  along the element, over Bρ, of their normal and skew quadrupole
  gradient on the axis turned by the Larmor angle θ_L = sign(q) ∫ B_z dz
  / (2Bρ) the beam turns by in the element's on-axis B_z (every child's,
  at its amplitude and the cluster's).  Each of the two opposite gradient
  lobes at a corrector's ends counts, and a solenoid that turns the beam
  between them keeps them from cancelling.  Beyond either bound they are
  refused.  The focusing bound lies where leaving a corrector's gradient
  out costs 0.5 % of an rms size: the LEBT example's first two corrector
  pairs reach it at 2.4 to 2.5 mrad, standing alone, superposed on their
  solenoid or both of a pair together (at their 16 mrad ADJUST limit a
  corrector focuses the beam by 0.057 to 0.059 /m); the third pair and
  the SCL corrector maps focus it by at most 0.0033 /m at 20 mrad and
  are refused beyond 20 mrad;
* **`SUPERPOSE_MAP` clusters** — the sum of the children's on-axis B_z,
  each over its own window, times the cluster's own 1 + `voltage_rel`;
  its steering-corrector children are bounded together and left out;
* RF maps and gaps as drifts (the DC approximation); a solenoid channel
  on an RF map still focuses.

A powered map it has no model of — a 1-D G(z) gradient, a 3-D map whose
quadrupole gradient on the axis makes more than a quarter of its peak
field at the grid's half-width (a quadrupole whose magnetic centre lies
on its grid, at any strength), a 3-D map whose field on the axis is a
solenoid's that its grid point nearest the axis does not show, or no
solenoid's (a dipole's, or one that vanishes there) where that point
shows a solenoid's, a 3-D map whose field vanishes on the axis with no
gradient there (a multipole), correctors that bend the beam by more
than 20 mrad or focus it, as the beam sees it, by more than
1/f = 0.0088 /m (a dipole with a gradient part, a quadrupole whose
magnetic centre lies off its grid, a corrector powered past the bound),
a static electric map, or a curved `SUPERPOSE_MAP_OUT` cluster — is
refused before the run with the element's name (each of the correctors
it sums): use `env_solver='matrix'` for such a line.  A 2-D Cartesian
map is refused too, and no HELIX mode tracks one.  In an error study, a
seed whose draws push a corrector past either bound (the LEBT example's
first two pairs past 2.4 to 2.5 mrad) is refused as any run is, and the
study stops at that seed; the matrix envelope runs every seed.

Four limits of the model to keep in mind on a real LEBT:

* the σ it gives are those of the rotating (Larmor) frame.  For a round
  beam they are the laboratory ones; behind a solenoid's rotation an
  elliptical beam's planes are mixed (after a quarter turn, swapped);
* space-charge compensation comes from the `SPACE_CHARGE_COMP` cards
  alone, not from a field map's `Ki`/`.scc` profile, which the matrix
  envelope and the multiparticle run apply inside the map;
* it sees a map's on-axis B_z and the paraxial field that goes with it,
  nothing else a 3-D map holds (a quadrupole part, a radial field that
  differs from the one its own B_z implies; a solenoid map's dipole or
  quadrupole part on the axis is neither seen nor bounded), and it has no
  centroid.  A corrector it leaves out takes its gradient with it, up to
  the focusing bound: on a 30 keV H⁻ beam without space charge, the LEBT
  example's first two corrector pairs at that bound (2.4 to 2.5 mrad, one
  corrector or both of a pair) move σ_x and σ_y in a 20000-particle
  multiparticle run by up to 0.53 % superposed on their solenoid (0.50 %
  from their gradient alone; the matrix envelope: 0.46 %) and 0.47 %
  standing just before it (0.24 % from their gradient alone; the matrix
  envelope: 0.34 %), where the run's rms sizes along the beam's principal
  axes, which a round beam hardly defines, scatter with the seed by about
  0.2 %.  With the line's own 6.8 mA beam (the DC LEBT deck with
  `SUPERPOSE_MAP` solenoids) they move σ_x and σ_y by up to 0.76 %, their
  gradient alone by at most 0.29 % (0.31 % in the matrix envelope), the rest
  from their steering, the steered beam losing up to 0.7 % of its
  particles at the apertures downstream.
  And a beam steered off a solenoid's axis meets a field the model does
  not see, which no envelope follows: standing just before its solenoid,
  the first corrector at its 16 mrad limit (refused) moves a
  multiparticle run's rms sizes by 4.0 % from its steering alone and the
  matrix envelope's not at all; the third pair at 20 mrad, standing just
  before its solenoid, by up to 1.9 % in a multiparticle run, nearly all
  from its steering, and superposed on it by at most 0.3 %, where the
  matrix envelope moves by up to 6.8 %.  With the deck's own 6.8 mA beam
  the third pair at its 16 mrad limit, superposed on its solenoid, steers
  the beam into the apertures before the RFQ: a multiparticle run loses
  15 to 22 % of its particles and the rms sizes of the rest move by up to
  19 %, the matrix envelope's by at most 0.7 %;
* of the hard-edge cards, a quadrupole is read at its nominal gradient,
  not its `gradient_rel`, and focuses x when that gradient is positive
  whatever the sign of the charge (the other modes then focus y for an
  H⁻ beam); a `SOLENOID` card reads its `field_rel`, but an OFF
  (`field_rel` = −1) as the nominal field.

```python
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.solenoid import Solenoid
from linac_gen.tracking.sacherer import SachererSolver

# LEBT-style solenoid transport line:
lattice = Lattice()
lattice.add(Drift(name="D1", length=200.0))
lattice.add(Solenoid(name="SOL1", length=300.0, field=0.20))
lattice.add(Drift(name="D2", length=400.0))

ref = ReferenceParticle(species=H_MINUS, w_kin=0.033, frequency=162.5)
bg = ref.bg
initial_twiss = dict(emit_x=0.20 / bg, alpha_x=0.0, beta_x=0.5,
                     emit_y=0.20 / bg, alpha_y=0.0, beta_y=0.5)

solver = SachererSolver(lattice, ref, initial_twiss, current=5.0)
res = solver.run()
```

## The envelope through an RFQ: bunched vs TraceWin-compatible DC

`EnvelopeSolver` with `initial["continuous"]=True` flips a DC beam to
**bunched** at the first RF bunching element: it seeds a uniform bunch
(σφ = 180°/√3 ≈ 103.9°), evolves the longitudinal envelope and switches
to bunched space charge.  That is the more physical picture — but it is
**not what TraceWin's envelope mode computes**.  TraceWin-env keeps the
beam DC through the RFQ: its genuine ENV+SC export carries *no*
longitudinal envelope at all (σφ ≡ σ_dp/p ≡ σ_W ≡ 0 in every RFQ row).

With space charge on, the two models therefore diverge inside an RFQ —
this is a model-class difference, not an error in either code.
Measured on the PXIE 2-solenoid LEBT+RFQ line at 5 mA against the
TraceWin ENV+SC export:

| HELIX envelope mode | RFQ σ deviation | full line |
|---|---|---|
| bunched (default) | 27–64 % | 27–32 % |
| `rfq_dc_envelope=True` | 5–11 % | **4.3 % (x) / 3.7 % (y)** |

LEBT-only agreement is 0.2 % either way, and the energy ramp is
identical in both modes (exit 1955.72 vs TW 1955.77 keV).

The bunched envelope through an RFQ also does not converge as the RFQ
is sliced more finely.  On `examples/lebt_plus_rfq/rfq_only_matched.lgproj`
(3.9 mA) the exit σW is 3.9, 7.8, 16.1, 0.5, 6.0 and 7.3 keV for a
space-charge step setting (step2) of 50, 100, 200, 400, 800 and 1600 per
metre, and σx moves between 0.28 and 0.58 mm.  Space charge is not the
main cause: at 1e-9 mA the exit σW still goes 1.9, 3.1, 4.4, 5.2, 5.7 and
5.8 keV over the same settings.  That beam enters with no energy spread
(`dc_energy_spread_keV = 0`), so the seeded longitudinal ellipse has zero
area all the way through, and its σφ / σW then follow the slicing of
the transport rather than any beam quality.  Read RFQ beam quality from
multiparticle tracking.

```python
from linac_gen.tracking.envelope import EnvelopeSolver

res = EnvelopeSolver(lattice, ref, initial_twiss, current=5.0,
                     rfq_dc_envelope=True).run()   # TW-comparable
```

Use `rfq_dc_envelope=True` **only** for validating against a TraceWin
envelope export; leave the default for physics work.  Neither envelope
model is the truth channel for an RFQ — that is multiparticle
tracking.  Pinned by `tests/rfq/test_env_rfq_dc.py`, including a guard
asserting the *default* mode does **not** match a TW export (if it ever
starts to, the default's physics changed).

## Cross-references

* [Models](01_models.md) — model decision tree.
* [Tracking modes → DC modes](../02_concepts/03_tracking_modes.md#dc-continuous-beam-modes).
* [Worked example: DC LEBT](../11_examples/03_lebt_dc.md).

← [Kernels](03_kernels.md) ·
[Continue to Convergence guide →](05_convergence.md)
