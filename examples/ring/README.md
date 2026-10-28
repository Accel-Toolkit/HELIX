# Ring mode examples

Two projects on the simple Booster test ring
(`tests/ring/fixtures/simple_booster.madx`: 24 identical cells, 474.2 m, RF
harmonic 84; the deck's beam is 800 MeV protons).  Open one with
**File → Open Project…**: ring mode comes on and the Ring tab opens with every
page set.  Press **Run** on the Run page: the strip under the summary follows
the survival and the emittances turn by turn.  When the run ends, **Show in
the Results tab** opens it in the Results tab: the beam turn by turn in its
tiles (sizes, emittances, energy spread, Twiss, the Σ matrix) and, under
RING RUN, the bunch in eV s, the tune diagram, phase space at the snapshot
turns, the ring's periodic optics, and for the transition project the ramp.

| Project | What it runs |
|---|---|
| `simple_booster_storage_sc.lgproj` | 300 turns at 800 MeV of a bunch matched in the RF bucket, with space charge (transverse and longitudinal kicks) |
| `simple_booster_transition_sc.lgproj` | 600 turns of a ramp through transition (crossed at turn 300), with the transverse space-charge kick |

A run takes a few minutes: about 0.5 s a turn on 14 cores, about 1 s on 6.
The projects point to the deck by a relative path: keep them in this folder
of the repository.

## Settings common to both

* **Beam tab:** the beam a run starts from, 5000 protons in a Gaussian cut
  at 4 rms; normalized rms emittances of 2.5 π mm mrad in x and y; emit_z
  8.5379 deg·MeV at 44.70441 MHz (the ring's RF frequency at 800 MeV), which
  is an rms longitudinal emittance of 5.305e-4 eV s (0.01 eV s for 6π times
  the rms).  Its Twiss functions are those of the beam matched to the ring at
  the start of the run, set with **Ring & Optics → Use the matched beam in
  the Beam tab**: the periodic Twiss functions and dispersion at the start of
  the ring, and the longitudinal Twiss functions of the RF bucket at the
  run's first turn.  The button also sets the energy, the run's starting
  energy.  If you change the ramp's start or the RF voltage, press it again.
* **Ring & Optics:** Cartesian bend multipoles, exact pole faces, CODATA 2018
  constants.
* **Run page:** the Beam tab's beam; one small-amplitude tune probe; the CPU
  (every core).  The summary gives the beam's BMAG at the start against the
  ring's matched beam: 1.0000 in x, y and z for both projects.
* **Space charge page:** the 2.5-D particle-in-cell model with 1.5e10
  protons in the bunch, 100 kicks per turn, a 64 × 64 grid (TSC charge
  shape) and no pipe (the potential is zero at 1 m).  The line-density bins
  are **auto** (the default): the number of bins follows the bunch's length
  (16 bins per rms of its core), and the line density is smoothed over a
  width set by the number of particles.  The storage project has the
  longitudinal kick from the potential; the transition project has it off.

The numbers below come from runs with seed 1 (the GUI's).  A GUI run and a
command-line run of the same project give identical results.  The horizontal
betatron emittance is the part of the x–px emittance not correlated with the
energy deviation δ (the dispersion removed).

## Storage at 800 MeV — `simple_booster_storage_sc.lgproj`

|  | auto bins (as saved) | 1024 fixed bins |
|---|---|---|
| survival | 100 % | 100 % |
| line-density bins (the bunch's rms length is 30 cm) | 256 (2.2 cm each) | 1024 (5.5 mm each) |
| rms longitudinal emittance, turn 0 → 300 | 5.305e-4 → 5.403e-4 eV s (+1.9 %) | 5.305e-4 → 5.893e-4 eV s (+11.1 %) |
| normalized rms betatron emittances, turn 0 → 300 | x +0.3 %, y −0.05 % | x −0.4 %, y +0.4 % |
| tune probe (NAFF): Qx, Qy, Qs | 0.598, 0.701, 0.0134 | 0.649, 0.709, 0.0166 |

Without space charge the fractional tunes are 0.71 and 0.81 and Qs is 0.0237
(**Ring & Optics**, with the 6-D box ticked).  The analytic tune shift at the
centre of a Gaussian bunch is −0.159 in x and −0.157 in y.

**What to try:** on the Space charge page, untick **auto** next to
**Line-density bins (RF period)**, set 1024, and Run again.  The 1024 bins
resolve the 30 cm bunch hardly better (the line density's peak is 2.4 %
lower with auto's smoothing), but each bin holds a quarter of the particles.
The noise in the slope of the line density then kicks the particles in energy
on every turn, and the longitudinal emittance grows about six times faster.
The tune probe starts at 0.2 rms.  With 1024 bins it is kicked out to 0.81 m
in ζ (2.2 rms of the bunch) by turn 147, where it sees less space charge, so
its tunes are no longer those of a small amplitude.  With auto it stays
within 17.5 cm of the centre.

The x emittance of the Run page's strip and of the Results tab's emittance
tiles is the projected one: it includes the dispersion term (D = 3.19 m at
the start of the ring) and follows the energy spread.  The y emittance and
the longitudinal emittance carry no such term.  The normalized-emittance
window's **Dispersion-corrected** mode and the KPI "ε_x growth" give the
betatron emittance.

## Through transition — `simple_booster_transition_sc.lgproj`

**Programs page:** the reference energy follows a sine from 800 MeV to 6 GeV
at 15 Hz (its minimum at 0.5 ms, flat before 0.8 ms); the RF voltage rises
from 0.4 MV to 1.2 MV over the first 3 ms (a table); transition gamma
5.449167323.  The clock at turn 0 is 20.106046 ms, so the run starts at
4059.1 MeV, 300 turns before transition, and ends at 4288.6 MeV.
**Preview** shows the turn table, with transition at turn 300.  The Beam
tab holds the beam matched to the ring at that start: 4059.1 MeV, a
dispersion of 0.661 mm/MeV (D = 3.19 m per unit of δ, divided by β0·p0c),
β_z 0.605 deg/MeV and α_z −0.0035 at 44.70441 MHz.  So close to transition
the slip factor is small: Qs is 0.0022, and the matched bunch is short.

|  | auto bins (as saved) | 4096 fixed bins | no space charge |
|---|---|---|---|
| survival | 100 % | 100 % | 100 % |
| line-density bins | 2048; 4096 from turn 553; 2048 from turn 599 | 4096 | — |
| bunch length σ_ζ: start, shortest, turn 600 | 4.16, 2.69 (turn 229), 3.40 cm | 4.16, 2.69 (turn 228), 3.47 cm | 4.16, 3.11 (turn 260), 3.62 cm |
| horizontal betatron emittance, turn 0 → 300 | 2.50 → 3.92 (+57 %) | 2.50 → 4.68 (+87 %) | 2.50 → 2.50 |
| vertical emittance, turn 0 → 300 | +0.4 % | +0.2 % | −0.04 % |
| rms longitudinal emittance: to turn 300, to turn 600 | −9 %, +52 % | −7 %, +71 % | +8 %, +41 % |

(Emittances in π mm mrad, normalized rms.)  Space charge drives the
horizontal growth: without it the betatron emittances stay within 0.1 %.  The
growth is in the horizontal plane only, the plane with dispersion, while the
longitudinal emittance falls (−9 % to transition, against +8 % without space
charge).  It is converged in the number of particles: with 20 000 particles
and auto bins it is +60 % to transition, and the longitudinal emittance −9 %
and then +62 % (5000 particles: +52 %, the growth after transition the least
certain number here).  So 5000 particles with auto bins already give the
horizontal answer, while 5000 particles with 4096 fixed bins overstate it
(+87 %) through the noise of the finely binned line density.  The manual's page "Ring-mode
benchmarks" (section "Losses, space charge and painting over many turns")
shows reference code B finding a horizontal growth of the same kind (more
than 40 % before transition) for another bunch on this ring.

The bins follow the bunch: the Results text gives "line-density bins
following the bunch, 2048 to 4096", and the command line lists each change.

**What to try:**

* untick **auto** and set 4096 bins: the horizontal growth rises to +87 %;
* set **Longitudinal kick** to **From the potential**.  The bunch was matched
  to the RF without space charge, and below transition the space-charge force
  from the slope of the line density opposes the RF.  So the bunch lengthens,
  from 4.2 cm to 10.8 cm by turn 280, and its longitudinal emittance grows by
  55 % to transition.  After transition the line density breaks up into
  short-wavelength modulations (above transition the longitudinal
  space-charge force is destabilizing), and by turn 600 the longitudinal
  emittance is 90 times its start.  The horizontal betatron emittance then
  stays within about 1 % (+1.1 % at turn 600).  A bunch with ten times the
  longitudinal emittance (emit_z 85.379 deg·MeV, same Twiss functions) behaves
  the same way: +245 % to transition, 24 times by turn 600.  At 1.5e10 protons, space charge is strong this close
  to transition.

## From the command line

    python -m linac_gen ring run examples/ring/simple_booster_storage_sc.lgproj --turns 300
    python -m linac_gen ring run examples/ring/simple_booster_transition_sc.lgproj --turns 600

The command line takes the project's beam (its 5000 particles), element
model, ramp and space charge.  The number of turns and the snapshot turns
come from its own options (`--turns`, `--snapshot-turns`); `--particles`
changes the number of particles.
