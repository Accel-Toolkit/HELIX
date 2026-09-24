# Reliability Study window

**Tools → Reliability Study…** opens a non-modal window that runs the four legs
of a reliability study on the session lattice as one resumable campaign —
the same engine as `python -m linac_gen reliability`
([CLI page](../06_running/14_cli_reliability.md)) and the assistant tool
`run_reliability`; the chapter [Reliability studies](../16_reliability/01_overview.md)
explains what each leg computes.  The window needs a lattice loaded **from a
file** and saved: the campaign executes the deck on disk, pins its digest and
refuses unsaved edits.

## Campaign tab

| Group | What you set |
|---|---|
| **Campaign** | the preset (`quick` / `full` / `custom` — choosing one fills the scenario classes, the seed counts and the compensation budget), the legs to run, the S1–S10 scenario classes, and the circuit map: **Browse…** a `circuits.json`, or **Generate from deck…** to derive one (spares, the first foil, one section) and save it for editing — cryomodules, RF stations, magnet circuits, the sections and the treaty point are yours to add |
| **Imperfections (leg C)** | the error budget as a JSON file or **Use Error Study tab entries** (the Error Study tab's element and beam error rows), the number of seeds, the faults-on-seeds sizing, and the orbit-correction settings (defaults from the session's correction settings) — **Pairing** `cards` corrects through the deck's `ADJUST_STEERER` cards (a deck without cards corrects nothing), `auto` pairs every steerer with every BPM (one-to-one when the counts match, else a global SVD) |
| **Availability (leg B)** | the block table (**Browse…** a `blocks.csv`, or **Write surrogate template…** to save the placeholder table — labelled as such — and edit it with the project's MTBF / MTTR data), the operating hours per year, the number of Monte Carlo trials and seed, the SRF-trip variants as `name=MTBF hours` entries, and the trip-duration bins of the beam-trip budget |
| **Compensation (leg A)** | how many critical cases to compensate, the matcher iterations per case (one envelope run of the deck each) and the number of neighbours |
| **Foil (leg D)** | the thickness list (µg/cm², `0` = the foil is missing; the deck's own thickness is the nominal scenario), the transverse offsets as `dx,dy` pairs, the thinned fraction (`0` = no such scenario), the stripping model and foil extent of the tracked runs, and the hits per particle |
| **Execution** | the root folder (default `<deck dir>/reliability/`), the campaign name, and the number of worker processes |

Choosing a preset fills the leg B and D entries too; a malformed entry (a
thickness that is not a number, a variant without its MTBF) is refused before
anything is created.

**Start** creates the campaign folder `<root>/<name>/` (or resumes it when it
exists) and runs it; **Resume…** picks any existing campaign folder — one run
on the CUDA machine and copied back included; **Cancel** stops after the
running items; **Self-test** runs the built-in
[self-test](../16_reliability/07_selftest.md) on the shipped demo deck and
shows its table on the Run tab.

**Export job…** writes the pending items of the checked legs of the loaded
campaign as a job folder for another machine (the completed items travel with
it, so the remote resumes rather than re-runs; `README.txt` holds the exact
command), and **Import results…** brings a finished job's results back: both
hashes and every item's status, results file and provenance are checked before
anything is copied, an item the campaign already completed is never
overwritten, a job with missing items asks before importing the rest, and the
views reload from the re-summarised folder — the window equivalent of the
[CLI's export / import](../06_running/14_cli_reliability.md#job-export-and-import).
Both buttons wait while a campaign or the self-test runs.

## Run tab

One progress bar per leg, the current wave and ETA, the log, and **Open
campaign folder**.  The worker holds only the campaign path: the engine reads
the deck from disk and runs its items in a process pool, so the session
lattice is never touched off the GUI thread.

## Results tab

* **Leg A** — every fault case worst first (criticality score, the rule's
  verdict, exit-energy deficit, exit-clock slip, emittance growths, recovered
  by) with the criticality bar, and the compensation table.  **Apply
  compensator settings** writes the selected compensation's settings onto the
  session lattice as one undoable edit (Undo reverts it); it refuses when the
  results are *stale* — the lattice was replaced or edited, or the beam
  changed, since the campaign ran — or when the session deck is not the one
  the campaign ran on.
* **Leg C** — the error seeds (transmission, energy, emittances, correction
  status, orbit rms before and after the correction) and the robustness of
  the top faults on those seeds.
* **Leg B** — availability per SRF-trip variant with the block source (the
  surrogate template is labelled as such) and the beam-trip budget: events
  per year in each trip-duration bin, one column per variant.
* **Leg D** — the foil scenarios (spot, power density, stripping fractions).
* **Report** — the self-contained HTML report, **Open in browser**, **Open
  folder**.

All views are read back from the campaign folder, so `summarize` on the CLI
and the window always agree.  The window remembers its last campaign for the
session (closing and reopening it restores the results).
