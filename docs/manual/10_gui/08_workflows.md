# GUI workflows

Task-oriented quick reference: "I want to do X, where do I click?"

## "I want to load a TraceWin .dat and run it"

1. **Lattice tab → Open…** (or File → Open Lattice…, Ctrl+O) →
   pick `.dat`.
2. **Beam tab** → set energy, current, distribution → **Apply**.
3. **Numerics tab** → leave defaults for first run.
4. **Toolbar** → click **Run Envelope** (fast σ-only pass) or
   **Run Multi-particle**.
5. **Results tab** → click any tile.

## "I have a TraceWin project (.dat + .ini) and want its beam in HELIX"

TraceWin keeps the input beam you typed into its *Main*/*Beam* panels
in `<project>.ini`, next to the deck.  Three routes; all three end with
the beam in the Beam tab as the session beam, the project marked dirty,
and every conversion warning in the console.

**Route A — open the deck and say yes** (a TraceWin project folder,
first look):

1. **File → Open Lattice…** (Ctrl+O) → pick `<project>.dat`.
2. HELIX finds `<project>.ini` next to it and asks *Import its beam
   into the Beam tab?  This replaces the current beam settings.*  Click
   **Yes**.  (**No** keeps whatever beam you had — nothing is read
   silently, and opening a `.lgproj` never asks.)
3. **Beam tab** → check species, energy, frequency, current, particle
   count, ε and Twiss.  `alpha_z` already has HELIX's sign.  A LEBT
   project (no longitudinal emittance) arrives with **Continuous beam**
   ticked; set the DC energy spread yourself.
4. Set what the file does not carry — distribution type, cut-off,
   centroids — then **Apply** if you changed anything.
5. **File → Save Project** so the `.lgproj` carries the beam; from now
   on open the project, not the deck.

**Route B — the deck is already open, or the `.ini` lives elsewhere:**

1. **Beam tab → Import TraceWin .ini…** → pick the file (any name,
   any folder).
2. Read the status line next to the button (species, energy,
   frequency, current, particle count, and a warning count) and the
   console.
3. Continue with steps 4–5 of route A.

**Route C — a new HELIX project from a TraceWin deck:**

1. **File → New Project…** (Ctrl+N) → name and location → **Import an
   existing lattice** → browse to `<project>.dat`.
2. The checkbox **Also import the beam from the TraceWin .ini next to
   the deck** lights up when a sibling `.ini` exists; tick it.  With
   *Copy the lattice into the project folder* the `.ini` is copied
   alongside.
3. **OK** — the project opens with the imported beam and its `.lgproj`
   is already written.

**When something does not fit:** a user-defined particle in the file
(an ion, a `My_particle` slot) has no HELIX species — the import keeps
the species selected in the Beam tab and says so in the console; pick
the right one first.  A value outside a field's range is clamped and
reported.  The PICNIC mesh sizes are recorded in the project file but
not applied — HELIX's space-charge grid is set on the Numerics tab.
For a second TraceWin input beam, or a deck whose `.ini` you want to
use for one headless run only, use the CLI (`twini --beam 2`,
`--tracewin-ini`): see [Importing TraceWin project settings](../06_running/02_tracewin_dat.md#importing-tracewin-project-settings-ini).

## "I want to match Twiss to a target"

The matcher's variables and constraints come **exclusively** from
`ADJUST_*` / `SET_*` cards in the `.dat` — there is no GUI entry for
them.  The realistic flow:

1. Load lattice (see above).
2. Add a [Marker](../03_elements/13_marker.md) at the target s
   (Lattice tab → **+ Add…** → Marker).
3. **Save** the lattice (Ctrl+S), then add the matching cards to the
   `.dat` in a text editor: `ADJUST` cards on the quads you want to
   tune and a `SET_TWISS` (or other `SET_*`) card at the target —
   see [SET / ADJUST cards](../07_matching/02_set_adjust.md) for the
   syntax.
4. **Reload** the lattice (Lattice tab → Reload).
5. **Matching tab → Match**; watch the live convergence dialog.
6. **Apply** to commit the matched values, then **Save** the matched
   `.dat` (or use **Save matched .dat** to export a copy).
7. Verify with **Run Envelope** and the Results tab.

## "I want to run an alignment-tolerance study"

1. Load lattice.
2. **Error Study tab → Element errors** → add `dx, dy` errors at
   `QUAD_*` with σ = 0.2 mm.
3. Set **n_seeds = 50** (or more for production).
4. Click **Run study**.
5. **Results tab → Error-study ensemble** tile.

## "I want to see what's happening at element X"

1. **Lattice tab** → click element X in the timeline or listing.
2. Inspector on the right shows all parameters.
3. (Optional) Add a Marker before/after for a snapshot — or set
   **Numerics tab → Snapshot every N** for periodic dumps.
4. After running: **Results tab → PHASE SPACE · DIAGNOSTICS →
   Phase space (4-panel)** tile shows the full phase-space at the
   snapshot.

## "I want to start a new project"

1. **File → New Project…** (Ctrl+N, or the toolbar **New** button).
2. Pick a name and a location — a folder `<name>/` is created there.
3. Choose the starting point: a **blank lattice** (one editable
   drift), **import** an existing `.dat` (copied into the folder by
   default so the project stays portable; when a TraceWin `<deck>.ini`
   sits next to it, a checkbox also imports that project's beam — see
   [Beam tab](03_beam_tab.md#importing-a-tracewin-ini-project-beam)),
   or a bundled **example** (FODO cell, solenoid channel, DTL section).
4. The project opens immediately; simulation outputs land in
   `<name>/runs/`, so the whole folder can be moved, archived or
   version-controlled as one unit.  **File → Save Project** updates
   the same `.lgproj` from then on.

## "I want to compare two simulations"

1. Run simulation 1 → File → Save Project As… `sim1.lgproj`.
2. Modify whatever you want changed → Run simulation 2 →
   File → Save Project As… `sim2.lgproj`.
3. Open both projects in two HELIX windows side by side.
4. Tile-by-tile visual comparison.

(A built-in diff/overlay is on the roadmap.)

## "I want to use the GPU"

1. **Numerics tab** → PIC backend → **gpu** (or **auto**; the combo
   also offers **cuda** and **mps** to pin a specific device type).
2. Run as usual.

If the GPU isn't available you'll see a console warning; HELIX
falls back to CPU.

## "I want to run a multi-particle simulation through the differentiable PIC"

1. **Numerics tab** → **SC engine** → `torch`.
2. (Grid mode, PIC backend, DC kernel will grey out — expected.)
3. **Toolbar** → click **Run Multi-particle**.

The torch backend is ~5× slower than the default numpy PIC but
matches it numerically.  Pick it only when you need an autograd-
differentiable kick (e.g. chaining the simulation into a gradient-
based optimiser); for ordinary forward runs leave SC engine on
`numpy`.  See
[Differentiable PIC](../05_space_charge/06_differentiable.md).

## "I want to match through non-linear PIC space charge"

1. Load a lattice with `ADJUST_*` + `SET_TWISS` / `SET_SIZE` cards.
   The lattice must be all-linear (drift / quad / solenoid /
   dipole / edge — no RF or field maps).
2. **Matching tab** → **Algorithm** → `gradient`.
3. Tick **Space charge**.
4. Click **Match**.
5. **Apply** or **Save matched .dat** when done.

This is the only HELIX matcher that solves *through* non-linear
PIC space charge.  See
[Gradient algorithm](../07_matching/06_gradient_algorithm.md).

## "I want to reconstruct the input beam from an exit distribution"

1. Either run a forward multi-particle simulation first (its final
   beam becomes selectable as the source), or have an exit-plane
   `.dst` at hand (measured or exported).
2. Toolbar **Simulate → Backtrack Distribution…**
3. Pick the source, the element range (defaults: full lattice), the
   field-map inverse (`rk4`, the exact default — round trips close at
   ~1e-13), and optionally tick space charge (uses the Numerics-tab SC
   settings) or set an output `.dst` for the reconstructed entrance
   distribution.
4. Click **OK** — progress appears on the toolbar; **Stop** aborts at
   the next element boundary.
5. The Results tab then shows the backward walk reversed to increasing
   s: **index 0 is the reconstructed entrance**.  Losses cannot be
   undone — the reconstruction covers survivors only.

CLI equivalent: [`python -m linac_gen backtrack`](../06_running/11_cli_backtrack.md).

## Cross-references

* [Quick start](../01_getting_started/03_quick_start.md)
* [Tab-by-tab tour](01_overview.md)

← [Results tab](07_results_tab.md) ·
[Continue to Worked examples → Basic FODO →](../11_examples/01_basic_fodo.md)
