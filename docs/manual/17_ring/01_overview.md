# Ring mode

Ring mode tracks a beam turn after turn around a synchrotron.  It is opt-in:
a linac session never sees it, and switching it on changes nothing in the
linac tabs or their results.  It stores a beam at the ring's fixed energy or
accelerates it through a ramp — the reference energy, the RF voltage and the
RF phase programmed against time, transition crossed with its phase jump —
with the RF bucket, chromaticity, nonlinear magnets, apertures and space
charge (the 2.5-D model), and fills it turn by turn through a stripping foil
with bump and painting magnets (injection).

## Switching it on

**Simulate → Ring Mode** adds the **Ring** tab after the linac tabs and
shows it; unticking the entry hides the tab again (a run in progress stops).
A project remembers ring mode in a `"ring"` section, written only once ring
mode has been used — a linac project is saved exactly as before.

When the lattice you open closes on itself — its horizontal bends add up to
360° and its survey returns to the start, or the deck declares `;@LG ring` —
the **Lattice** tab shows a note offering ring mode: **Use ring mode** switches
it on, **Keep as line** hides the note for this lattice (the choice is saved
with the project), **Why?** shows the evidence.  Ring mode is never switched
on by itself, and a lattice that does not bend through 360° never loads any
of it.

Outside the GUI, `python -m linac_gen ring` runs the same computations
([Command line](10_cli.md)), and the AI assistant has ring tools of its own
— the verdict, the optics, the ramp and its preview, ring runs
([AI Assistant](../14_assistant/01_assistant.md)).

## The Ring tab

The tab opens with the **verdict** for the loaded lattice — whether it closes
on itself, its length, the RF harmonic number when the deck gives one, and the
reference beam — and a line of the latest optics (tunes, chromaticities,
transition gamma).  Below it are five pages.

**Ring & Optics** computes the ring's periodic optics from the ring kernels:
tunes with their integer part, chromaticities, momentum compaction and
transition, the Twiss functions and dispersion along the ring, and with RF
the synchrotron tune and the bucket.  **Use the matched beam in the Beam tab**
gives the Beam tab the beam matched to the ring at the start of the run: the
Twiss functions in x, y and, with the RF on, z, the dispersion and the closed
orbit, from the optics of the run's first turn.  The
element model (Cartesian or curvilinear multipoles), the pole-face model and the
physical constants are chosen here and saved with the project.  See
[Ring optics](02_optics.md) and [Element models](03_models.md).

**Programs** sets the ramp: the reference energy (fixed, flat, a resonant
cycle or a table), the RF voltage and phase programs, transition gamma, the
clock at turn 0, paraphasing groups; tables are read from and written to CSV
files.  **Preview** runs the reference particle alone through the ramp — the
turns to the end of the program, transition, the RF frequency, the gain
against the voltage — and **Run to the end of the program** gives the Run
page that number of turns.  See [The ramp](05_programs.md).

**Space charge** sets the 2.5-D model: off, particle-in-cell or a frozen
Gaussian, the particles in the bunch, the kicks per turn, the grid, the pipe
(open, round, rectangle, ellipse) and the longitudinal kick, with the
analytic tune shift for the Beam tab's emittances once the optics are
computed.  See [Space charge](06_space_charge.md).

**Injection** sets the multi-turn injection through a stripping foil: the
turns that inject, the macroparticles each brings and the current; the foil
(where it is, its material and thickness, its rectangle); the injected beam
(the Beam tab's, a particle file or a Gaussian at the foil; an H⁻ is
stripped to a proton of the same velocity); the chopper; and the bump and
painting magnets with their waveforms.  See [Injection](07_injection.md).

**Run** tracks a bunch for a number of turns through the same ring kernels, at
the ring's fixed energy or through the ramp of the Programs page, with the
space charge of the Space charge page and the injection of the Injection
page:

* **Particles.**  The Beam tab's beam, as for a linac run (its
  distribution, number of particles, emittances and Twiss functions,
  generated at the ring's species and energy), or none (the injection fills
  the ring).  A run that starts with a beam gives its BMAG against the
  ring's matched beam.  A MAD-X or MAD8 deck whose `BEAM` gives the energy
  states the beam its magnets are scaled for: the ring runs at that
  reference.  A lattice that states no beam — a TraceWin or elegant deck, a
  MAD-X or MAD8 `BEAM` with only the particle — runs at the Beam tab's
  species and energy (set them to the ring's); a deck of normalized
  strengths keeps them at that energy.  The Beam tab's longitudinal
  emittance is in deg·MeV at the Beam tab's frequency.
* **Tune probes** (one small-amplitude probe gives the tunes; n gives an
  n × n footprint), **snapshot turns** (the whole distribution at those
  turns; `last` is the last turn), the **device** (every CPU core, or a
  CUDA GPU).
* **Run** and **Stop**.  Stop takes effect at the end of the turn being
  tracked — a turn of a large bunch can take seconds — and keeps the turns
  done.  The survival and emittances update as it runs.  A run that would
  need more than 70 % of the machine's memory is refused before it starts.
* **Results.**  While it runs, a strip of three plots: the survival, the
  normalized emittances x and y, and the longitudinal emittance turn by turn
  (statistics every turn, or every k turns for more than 100 000 turns).
  When it ends, its results are in the **Results** tab (**Show in the
  Results tab**, or the Results tab's **Run shown: Ring run**): the beam turn
  by turn in its tiles — sizes, emittances, Twiss, energy, survival,
  centroid, dispersion, the Σ matrix — and, under **RING RUN**, a ramp's
  reference energy, synchronous phase, voltage and frequency with the
  transition turn marked, the bunch in eV·s, phase space at the snapshot
  turns with the separatrix of the turn's RF bucket, losses along the ring
  and over the turns, the NAFF tunes and the footprint on a tune diagram,
  the run's periodic optics, and with a foil the injection turn by turn, the
  closed orbit at the foil and the foil's hit map
  ([Ring runs in the Results tab](../10_gui/07_results_tab.md#ring-runs)).
  Every run, a stopped one too, is written to the calc directory as
  `<time>_ring.h5` — one stopped by quitting HELIX too, if it ends its turn
  and writes its results within the five seconds HELIX waits.

See [Ring runs](04_run.md).  Every computation works on a copy of the lattice:
the lattice in the window is never modified.

A ring that is unstable at the reference energy — the magnets scaled for
another energy, an RF phase on the wrong side of transition — is refused with
a message, rather than tracked until the numbers overflow.  So are cavities
that would give the reference particle energy in a run without a ramp: the
reference energy changes only through a ramp.
