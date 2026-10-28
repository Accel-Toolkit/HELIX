# A first ring run in the GUI

The simple Fermilab Booster of HELIX's tests
(`tests/ring/fixtures/simple_booster.madx`), from opening it to the results.
The [overview](01_overview.md#the-ring-tab) describes every page of the Ring
tab; this page walks through them once.

1. **Open the ring.**  **File → Open Lattice…** and choose
   `simple_booster.madx`.  Its bends add up to 360°, so the **Lattice** tab
   shows a note: click **Use ring mode** (or tick **Simulate → Ring Mode**).
   The **Ring** tab appears after the linac tabs, with the verdict: the ring
   closes on itself, 474.2 m, RF harmonic 84, its beam an 800 MeV proton.
2. **Its optics.**  On **Ring & Optics**, tick **with RF: 6-D closed
   orbit, synchrotron tune, bucket** and **Compute optics**: the tunes 6.71
   and 6.81, the chromaticities −8.65 and −8.09, transition gamma 5.449, the
   synchrotron tune 0.0237.
3. **The beam.**  A run starts from the Beam tab's beam, whose defaults are
   a linac's.  Set the emittances and the number of particles for the ring,
   for example normalized rms emittances of 1.5 π mm mrad, emit_z 5 deg·MeV
   at the RF frequency, 44.7044 MHz, and 5000 particles (a turn in about a
   third of a second on a 14-core Mac).  Then **Use the matched beam in the
   Beam tab** (Ring & Optics) gives it the beam matched to the ring at the
   start of the run: the Twiss functions in x and y, the dispersion, the
   longitudinal Twiss functions of the RF bucket, the deck's species and
   energy.  (A deck that states no beam — an elegant deck, a MAD-X or MAD8
   `BEAM` without an energy — keeps the normalized strengths of its magnets, and
   the ring runs at the Beam tab's species and energy: the verdict says so,
   and the matched beam leaves them as they are.)
4. **A run at fixed energy.**  On **Run**, the line under **Particles
   from** says what the run starts with; set the turns, then **Run**.  The
   summary gives the beam's BMAG at the start (1 when matched).  The survival and emittances update as it goes, in the strip
   below the summary; **Stop** ends it at the end of the turn being tracked
   and keeps the turns done.  **Show in the Results tab** then opens the
   run in the Results tab: its tiles show the beam turn by turn (sizes,
   emittances, Twiss, energy, survival, centroid, dispersion, the Σ
   matrix), and **RING RUN** at the top adds phase space at the snapshot
   turns with the RF bucket's separatrix, the losses along the ring, the
   tune diagram (NAFF), the bunch in eV·s and the ring's periodic optics.
5. **Through a ramp.**  On **Programs**, choose a resonant cycle for the
   energy and a voltage table ([The ramp](05_programs.md) gives an example),
   then **Preview**: the reference particle alone through the ramp — the
   turns to the end of the program, the transition turn, the RF frequency.
   **Run to the end of the program** gives the Run page that number of
   turns.  The Results tab's **Ring ramp** then shows the reference energy,
   synchronous phase, voltage and frequency turn by turn, transition marked.
6. **With space charge.**  On **Space charge**, choose the particle-in-cell
   model and the real particles in the bunch (for example 1.5 × 10¹⁰); the
   page shows the analytic tune shift for the Beam tab's emittances (with
   the 6-D optics of step 2: about −0.35 here).  The line-density bins are
   "auto" by default: they follow the bunch's length, and the line density
   is smoothed as the number of particles needs
   ([The line density](06_space_charge.md#the-line-density)).
7. **Injection.**  On **Injection**, switch it on: the Run page's source
   becomes "None: the injection fills the ring".  Choose the foil — **At
   element** lists the lattice's markers and foil elements; the simple
   Booster has none, so type the name of the element at whose exit the foil
   sits (a name the lattice holds once) — its material, thickness and
   rectangle, the injected beam, the turns and the current, and the bump and
   painting magnets with their waveforms; the page estimates the particles a
   turn brings ([Injection](07_injection.md)).  With space charge, leave the
   Space charge page's intensity blank: the charge is the circulating charge
   the injection builds, and the run gives its tune shift at the end.
8. **Keep it.**  **File → Save Project** writes the ring settings into the
   project's `"ring"` section, with an entry still being typed or a table
   cell still being edited as it stands.  Every run — a stopped one too — is
   written to the calc directory as `<time>_ring.h5`
   ([Results files](11_results.md)).

Every computation works on a copy of the lattice: the lattice in the window
is never modified.  An edit of the lattice during a run (the element
inspector, undo, the assistant's changes) leaves the run going on its copy,
the lattice as it was at **Run**, and its results arrive as usual; the
optics, a matched beam being computed and the ramp's preview go, since they
describe the lattice before the edit.  Another lattice — opened, reloaded,
from a project or from a matching Apply — stops the run, and the Run page
says "Stopped: the lattice was replaced."  Switching ring mode off stops a
run at the end of its turn and drops what the Ring & Optics and Programs
pages are computing.
