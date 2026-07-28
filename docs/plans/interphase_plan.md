# Interphase GUI Implementation Plan

> Parallel PyQt6 GUI modeled on the "Interphase" Claude-Design mockup.
> Existing `linac_gen_gui` stays untouched; this lives beside it as
> `linac_gen_gui/interphase/` and launches via `python -m linac_gen_gui.interphase`.

## Goals

1. Dark-theme, TraceWin-class IDE for particle tracking, **fully wired to the
   existing `linac_gen` backend** (envelope, PIC, matching, diagnostics,
   TraceWin I/O, transfer/sigma matrix dialogs).
2. Every interactive button triggers a **real calculation**, not a stub.
3. Layout, density, typography, palette match the web mockup closely.
4. Coexists with the classic GUI — user can launch either one.

## Non-goals

- Not trying to replicate the mockup's fake real-time PIC animation.
- No remote collaboration / cloud / EPICS integration (mockup has
  placeholder pills for those; we omit).
- Not a rewrite of backend code. GUI glue only.

## Stack choices

| Need | Choice | Rationale |
|---|---|---|
| Framework | PyQt6 | Matches existing GUI; mature, accessible |
| Styling | QSS (Qt stylesheets) | Lets us mirror the CSS tokens 1:1 |
| Plots | pyqtgraph | Already used; fast; dark-theme native |
| 3D | pyqtgraph.opengl | Optional 3D lattice view; ship with graceful fallback |
| Layout | QGridLayout + QDockWidget | 5-row chrome + rail/sidebar/stage/inspector split |
| Icons | Inline SVG via `QIcon.fromSvg` helper | Mockup uses lucide-style strokes; we reproduce as SVG paths |
| Fonts | Inter + JetBrains Mono | Bundle or fall back to system equivalents |

## Architecture

```
gui/linac_gen_gui/interphase/
    __init__.py
    __main__.py                   # python -m linac_gen_gui.interphase
    app.py                        # QApplication + main window factory
    theme.py                      # dark-theme QSS, design tokens, icon helper
    icons.py                      # SVG icon set (mirrors design/icons.jsx)
    state.py                      # AppState dataclass, Qt signals for changes
    workers.py                    # QThread wrappers for envelope / PIC / matching
    chrome/
        titlebar.py
        menubar.py
        subbar.py                 # breadcrumb + s-cursor scrubber
        statusbar.py
    nav/
        rail.py                   # 17-button left rail
        sidebar.py                # outline / search / bookmarks
    inspector/
        inspector.py              # right panel, context-sensitive
        element_fields.py         # type-specific property editors
    dock/
        bottom_dock.py            # tabs: Console | Problems | Output | Diagnostics
        tweak_panel.py            # accent / density / palette
    plots/
        autosize.py               # pyqtgraph wrapper that hugs parent
        envelope_plot.py
        phase_space_plot.py       # 4-quadrant x-xp / y-yp / x-y / phi-dW
        lattice_track.py          # horizontal lattice timeline with element bars
        loss_plot.py
        transmission_plot.py
    views/
        base.py                   # QWidget subclass with stage-toolbar helper
        lattice.py                # KPIs + timeline + live envelope + inspector
        beam.py                   # wraps existing BeamConfigWidget
        envelope.py               # Envelope solver output (σ_x,y,φ,W + Twiss)
        tracking.py               # PIC multiparticle run
        phase_space.py
        optics.py                 # Twiss matching (wraps MatchingDialog)
        errors.py                 # Error-model MC wrapper
        correction.py             # Closed-orbit / steering-magnet fit
        diagnostics.py            # BPM / transmission / loss map
        view3d.py                 # 3D lattice (graceful fallback if no GL)
        rf.py                     # RF / longitudinal phase analysis
        fieldmaps.py              # Field-map viewer
        tunemap.py                # Working-point tune scan
        console.py                # Python REPL (IPython-lite)
        jobs.py                   # Recent runs / queue
        results.py                # Run A/B compare
        reports.py                # PDF/HTML export
    launcher.py                   # first-run wizard / lattice picker (optional)
```

## Phase breakdown

Each phase ends with a checkpoint: launch the app, click around, verify
no regressions; commit.

### Phase 0 — Scaffolding & theme (checkpoint: empty app shell runs)

- `theme.py`: CSS-variable → QSS translation. Produce a `dark_qss()`
  helper that returns the stylesheet string with the tokens substituted.
- `icons.py`: port ~60 lucide-style SVG icons from `icons.jsx`.
  Each icon returns a `QIcon` at requested px size.
- `app.py`: QApplication with theme applied, empty QMainWindow, 5-row
  layout grid (titlebar 28, menubar 38, subbar 34, main 1fr, statusbar 22).
- `__main__.py`: entry point.

Deliverable: `python -m linac_gen_gui.interphase` opens a dark window
with five empty horizontal strips.

### Phase 1 — Chrome (checkpoint: chrome renders, state flows)

- `state.py`: `AppState` with `workspace`, `selected_element`, `s_cursor`,
  `running`, `tweaks`; emits `pyqtSignal` on change.
- `titlebar.py`: traffic lights, app name "INTERPHASE", project path
  (reads current lattice name), right-side EPICS/GPU pills omitted.
- `menubar.py`: File / Edit / Lattice / Beam / Simulate / Analyze /
  Optimize / View / Tools / Plugins / Window / Help menus + Run/Pause/
  Step/Stop button group + pills.
- `subbar.py`: workspace breadcrumb, selection label, s-cursor
  `QSlider` over total lattice length, zoom/layer/filter buttons.
- `statusbar.py`: running state, step count, `s / total`, current
  kinetic energy from ref, ε_n, loss %, FPS.

Menu entries fire actions on `AppState`. State changes cascade to subbar
labels and statusbar readouts.

### Phase 2 — Navigation shell (checkpoint: rail + sidebar + stage switch)

- `rail.py`: 17-button vertical rail (4-wide) + tooltip on hover. Click
  changes `AppState.workspace`.
- `sidebar.py`: outline tree built from loaded lattice, grouped by
  element-name prefix (SRC/LEBT/MEBT/DTL/SCL/HEBT/TGT), search box,
  bookmarks section.
- `stage-tabs` and `stage-toolbar`: QTabBar-style strip over the view
  for "open files" + per-view toolbar.
- `views/base.py`: `WorkspaceView(QWidget)` with helper to add
  stage-toolbar groups and KPI cards.

Every rail click loads the matching view (empty stub initially). Clicking
sidebar elements sets `AppState.selected_element`.

### Phase 3 — Inspector (checkpoint: type-aware property editor)

- `inspector.py`: right-side QScrollArea + section headers.
- `element_fields.py`: per-element-type field sets
  - Drift: length, aperture
  - Quadrupole: length, gradient (with sign flip button), aperture,
    dx/dy alignment, tilt
  - Bend: angle, radius, n-index, edge angles
  - RFGap / FieldMap: amplitude, phase, frequency, file path
  - Solenoid, Sextupole, Octupole, Marker, Aperture, BPM, Collimator
- Two-way binding: editing a field mutates the actual `Element`, emits
  `lattice_changed`, triggers view updates if `auto-rerun` is enabled.

### Phase 4 — Lattice workspace (checkpoint: default view feels real)

- KPI row: element count, total length, ref energy, initial Twiss, σ_x
  final from latest envelope run.
- `lattice_track.py`: horizontal QGraphicsScene strip showing all
  elements colored by type, element-name ruler, clickable selection,
  s-cursor overlay. Same color palette as mockup
  (drift=gray, quad=cyan, bend=orange, cav=lime, etc.).
- Live envelope preview (small σ_x, σ_y curve beneath the timeline).
- AI Suggestions panel: placeholder that becomes "Check convergence"
  actions or deletes.
- Inspector auto-populates with selected element.

### Phase 5 — Core science workspaces (checkpoint: each triggers a real run)

- `beam.py`: wraps existing `BeamConfigWidget` (reuse, don't rebuild).
- `envelope.py`: "Run Envelope" button → `EnvelopeSolver` on QThread,
  returns results, plots σ_x, σ_y, σ_φ, and Twiss α/β per plane.
- `tracking.py`: "Run Multi-particle" button → `Simulation.run()` with
  PIC solver; live progress bar; result plots matching existing.
- `phase_space.py`: our existing `PhaseSpacePlotWidget` upgraded to 4-in-1
  grid (x-xp, y-yp, x-y, φ-dW) with the TraceWin basis toggle already built.
- `optics.py`: embeds existing `MatchingDialog` contents (not as dialog).
- `diagnostics.py`: transmission plot, loss map (existing widgets), BPM
  readout table.
- `console.py`: QPlainTextEdit-backed REPL that evaluates Python in a
  sandbox with `linac_gen`, `lattice`, `beam`, `results` injected.
- `jobs.py`: QTableView over an in-memory "run history" list, persisted
  to `run_reports/*.json` (already dumped by classic GUI).

### Phase 6 — Secondary workspaces (checkpoint: each renders something useful)

- `errors.py`: run one error seed (shift/tilt misalignments)
  → reports per-plane emittance growth; Monte-Carlo scan over N seeds.
- `correction.py`: zero-steerer closed-orbit response matrix; SVD fit
  to null out BPM offsets.
- `rf.py`: longitudinal phase space & RF phase-scan helper
  (calls `SCPhaseScan` or similar).
- `fieldmaps.py`: viewer for TraceWin FieldMap files (we already parse
  them); heatmap of E/B vs (r, z).
- `tunemap.py`: scan Q_x, Q_y over a grid; colour by stability metric.
- `view3d.py`: pyqtgraph.opengl lattice if available, else a 2D schematic
  with Z-into-page isometric cheat. Fallback must be graceful.
- `results.py`: pick two prior runs from `run_reports/`, overlay σ_x, σ_y,
  σ_φ with colour-coded diff shading.
- `reports.py`: Export a single-page HTML report (config + plots + KPIs)
  using matplotlib + the parser we already have.

### Phase 7 — Bottom dock + tweak panel (checkpoint: polish)

- `bottom_dock.py`: tabbed panel (Console | Problems | Output | Diagnostics)
  docked at the bottom; height persisted.
- `tweak_panel.py`: floating popover with accent colour swatches, density
  slider, plot palette, dock on/off, inspector-left/off/right. Applies by
  re-composing the QSS on the fly.

### Phase 8 — Integration tests & polish

- Unit tests for `theme.py` (QSS compiles), `icons.py` (every name
  resolves to QIcon), `state.py` (signals emit correctly).
- Manual smoke test checklist in `docs/smoke_test_interphase.md` covering
  all 17 workspaces against the FODO example lattice.
- Update `run_gui.sh` / `run_gui.bat` to offer both GUIs via `--classic`
  / `--interphase` flag.

## Wiring to existing backend

Every workspace pulls from the same runtime:

```
linac_gen.io.tracewin_parser.parse_tracewin()       → load lattice
linac_gen.distributions.factory.create_beam(cfg)    → sample particles
linac_gen.tracking.envelope.EnvelopeSolver          → envelope mode
linac_gen.core.simulation.Simulation + PicSolver     → multiparticle
linac_gen.matching.MatchingDialog helpers            → optics
linac_gen.diagnostics.recorder.DiagnosticRecorder    → snapshots
linac_gen.tracking.matrix_tracking.compute_transfer  → transfer-matrix dialog (keep)
linac_gen.tracking.longitudinal_coords               → (z, δ) TraceWin basis toggle
```

Nothing in `linac_gen/` gets edited — this is purely a new front end.

## Risk & mitigation

| Risk | Mitigation |
|---|---|
| QSS can't reach CSS fidelity | Accept ~90% match; prioritise correctness over pixel-perfection |
| pyqtgraph 3D unavailable on user's WSL | `view3d.py` has a 2D fallback, no crash |
| IPython REPL complexity | Ship a simple `exec()`-based mini-REPL; refactor to IPython later if needed |
| Scope creep (17 workspaces!) | Phase gates: deliver phases 0-5 first (launchable + science works); phases 6-8 are polish |
| Classic GUI regression | Interphase code lives in its own package; zero edits to `linac_gen_gui/main_window.py` |

## Estimated LOC

| Area | Lines |
|---|---|
| `theme.py` + `icons.py` | ~700 |
| Chrome (5 files) | ~500 |
| Nav (rail + sidebar) | ~350 |
| Inspector + element fields | ~600 |
| Plot widgets (6) | ~600 |
| Views (17) | ~2 000 |
| Dock + tweak | ~300 |
| State + workers + app | ~400 |
| Tests | ~400 |
| **Total** | **≈ 5 850 LOC** |

## Execution order for this session

I'll proceed phase-by-phase, committing after each checkpoint. You
can stop me at any phase — the GUI will be functional (though
increasingly feature-rich) at every checkpoint after Phase 1.

---

**Phases 0 → 5 = minimum viable**: dark-theme app, chrome, all 17
workspace buttons, inspector, Lattice view, all five core-science
workspaces working (beam/envelope/tracking/phase-space/optics).

**Phases 6 → 8 = full scope**: twelve more workspace implementations,
bottom dock polish, tweak panel, tests.

Ready to build Phase 0 on approval.
