# Interphase v2 — TraceWin-style tabbed UI plan

## What changes

**Goal:** simpler, systematic 5-tab GUI with the same dark theme. Replaces the 17-workspace rail sprawl with TraceWin-style top-level tabs matching real user workflows.

**Tabs (left to right):**

1. **Beam** — particle species, energy, RF, current, Twiss X/Y/Z, centroid, mismatch, derived β/γ, sample-beam preview.
2. **Lattice** — open/save .dat, element outline tree, layout strip, element inspector, element-type summary.
3. **Results** — every plot of every quantity, as nested sub-tabs. Central place to inspect envelope + MP runs.
4. **Matching** — Twiss matching (envelope + SC), before/after comparison, apply to beam.
5. **Convergence** — PIC grid / extent / N-particle scan with plots and table.

The Run buttons (Envelope, Multiparticle) + progress bar sit in a **global toolbar** so they work from any tab. Same for File menu and s-cursor.

## What gets removed

- 17-button left rail
- Left outline sidebar (merged into Lattice tab)
- Right inspector dock (merged into Lattice tab)
- Sub-bar with breadcrumbs (folded into toolbar)
- Bottom dock (console etc.)
- Tweak panel (theme becomes fixed)
- Separate Console / Jobs / Errors / Correction / RF / Fieldmap / Tunemap / 3D / Results A-B / Reports workspaces

Console + Jobs move to **Tools menu** as popup/modal. 3D, Tune, Fieldmap, Errors, Correction stay in the backlog for later — none are blocking.

## Final layout

```
┌────────────────────────────────────────────────────────────┐
│ Title bar  INTERPHASE · sns_baseline_v12.lat · 12:04:33  │ 28
├────────────────────────────────────────────────────────────┤
│ [File▾] [Lattice▾] [Simulate▾] [Tools▾] [Help▾]         │
│  ║ [▶ Run Envelope] [⚡ Run Multi-particle] [■] ║────────║ │ 44
│     progress bar          s = 1600 / 1600 mm  ──●──       │
├────────────────────────────────────────────────────────────┤
│ ┌ Tabs ────────────────────────────────────────────────┐ │
│ │  Beam │ Lattice │ Results │ Matching │ Convergence │ │ 34
│ ├────────────────────────────────────────────────────────┤ │
│ │                                                       │ │
│ │                 ACTIVE TAB CONTENT                   │ │ 1fr
│ │                                                       │ │
│ └────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────┤
│  READY · 21 elements · s = 1600 · σ_x = 6.4 · loss 0.0 % │ 22
└────────────────────────────────────────────────────────────┘
```

## Tab-by-tab contents

### 1 · Beam
Three-column dense form (same fields as current Beam view, restyled):
- **Column A** — species, W_kin, frequency, current, duty cycle, N particles, distribution, cutoff
- **Column B** — Twiss X, Y, Z (ε_n, α, β each)
- **Column C** — centroid (δx, δx', δy, δy', δφ, δW), mismatch (x, y, z), derived β/γ/βγ
- **Bottom strip** — [Apply] · [Generate Sample Beam] · inline 4-quadrant initial phase-space preview

Same backend wiring as current `BeamView`.

### 2 · Lattice
Horizontal layout:
- **Sub-toolbar** — [Open…] [Save] [Save As…] [Reload] · total length · element count · "dirty" indicator
- **Left column (240 px)** — outline tree (grouped by prefix, live search, element-type swatches)
- **Center (flex)** — lattice timeline strip (clickable, s-cursor) + element-type distribution bar chart
- **Right column (300 px)** — element inspector (type-aware: Drift, Quad, Dipole, Solenoid, RFGap, FieldMap, Sextupole, Octupole, Aperture, BPM, Marker…)
- **Bottom** — selected element details (s-start, s-end, delta length)

Uses existing `LatticeTimeline`, `Inspector`, and `Sidebar` components; just re-arranges them inside a single tab.

### 3 · Results
KPI row on top (always visible):
- σ_x end · σ_y end · σ_z end · ε_x growth · loss · run time

Nested sub-tabs below KPI row:
1. **RMS σ** — σ_x, σ_y, σ_z stacked (linked x-axis)
2. **Emittance** — ε_x, ε_y, ε_z stacked
3. **Twiss** — α_x, β_x, α_y, β_y stacked
4. **Energy · Transmission** — W_kin, γ, transmission
5. **Loss** — cumulative loss curve + per-element loss histogram
6. **Centroid** — ⟨x⟩, ⟨y⟩, ⟨φ⟩
7. **BPMs** — table
8. **Phase Space** — 4-quadrant scatter with (Δφ,ΔW)↔(z,δ) basis toggle
9. **Sigma Matrix** — step picker + 6×6 table (reuses existing dialog)
10. **Transfer Matrix** — element-range picker + 6×6 table (reuses existing dialog)
11. **Jobs** — previous `run_reports/*.json` in a table

### 4 · Matching
Clear workflow instead of a cryptic dialog:
- **Target block** — radio buttons: "Periodic cell (from lattice)" · "End of element range" · "Manual α/β"
- **Range picker** — start/end element selectors (when relevant)
- **Current Twiss** KPIs
- **Matched Twiss** KPIs
- **Matched − Current** diff KPIs
- **Run**: [Compute] button → populates Matched column
- **Plot** — before/after σ_x, σ_y over the lattice
- **Apply**: [Copy matched → Beam] button

Uses `linac_gen.matching.periodic.find_periodic_twiss` for the periodic mode; for ranged matching, runs envelope twice (current and matched) and overlays.

### 5 · Convergence
Parameter scan, inline — no dialog:
- **Scan axis** dropdown — grid (nx=ny=nz), grid_extent, n_particles
- **Scan values** — comma-sep list or min/max/step
- **Run scan** button with progress bar
- **Plot** — final σ_y (or chosen metric) vs scan parameter, with convergence line
- **Table** — every row: parameter, σ_x end, σ_y end, σ_φ end, ε_x end, elapsed
- **Recommendation badge** — "Converged at nx ≥ 96 (Δσ_y < 0.5%)"

Reuses the logic in the classic `ConvergenceDialog` but laid out in a tab.

## File structure

```
gui/linac_gen_gui/interphase/
  app.py                     ← rewrite (≈ 200 LOC, was ≈ 300)
  theme.py                   ← keep
  icons.py                   ← keep
  state.py                   ← remove `workspace`, add `current_tab`
  workers.py                 ← keep + add ScanWorker
  chrome/
    titlebar.py              ← keep
    toolbar.py               ← NEW (replaces menubar.py + subbar.py)
    statusbar.py             ← keep
  tabs/                      ← NEW top-level
    beam_tab.py              ← from views/beam.py
    lattice_tab.py           ← from views/lattice.py + inspector + sidebar
    results_tab.py           ← nested QTabWidget, composes old views
    matching_tab.py          ← from views/optics.py, expanded
    convergence_tab.py       ← NEW (ports ConvergenceDialog logic)
  panels/                    ← NEW reusable pieces
    outline_tree.py          ← from nav/sidebar.py
    element_inspector.py     ← from inspector/inspector.py
    kpi_row.py               ← from views/base.py
    sample_phase_preview.py  ← NEW, tiny
  plots/                     ← keep
```

**Deleted:** `chrome/menubar.py`, `chrome/subbar.py`, `nav/rail.py`, `nav/sidebar.py` (superseded by panels/outline_tree.py), `inspector/*` (superseded by panels/element_inspector.py), `views/lattice.py`, `views/beam.py`, `views/envelope.py`, `views/tracking.py`, `views/diagnostics.py`, `views/phase_space.py`, `views/optics.py`, `views/console.py`, `views/jobs.py`, `views/base.py`.

**Moved to Tools menu popup dialogs:**
- Python Console (QDialog wrapping ConsoleView)
- Run history (QDialog wrapping JobsView)

The classic GUI under `linac_gen_gui/main_window.py` stays **untouched**. `python -m linac_gen_gui` still runs it.

## Phased execution (implementation order)

Each phase ends with a working checkpoint.

- **P1 — Shell rewrite** (≈ 200 LOC): titlebar, toolbar, tab widget, statusbar; five empty tabs. Verify dark theme still loads, auto-load FODO still works, clock ticks, status updates.

- **P2 — Beam tab** (≈ 300 LOC): port form, wire Apply, add derived β/γ readout. Verify "Apply" fires beam_config_changed signal.

- **P3 — Lattice tab** (≈ 400 LOC): outline tree + timeline + inspector composed in one tab. Verify click-to-select cascades through all three panels.

- **P4 — Results tab** (≈ 500 LOC + reuses ≈ 400 from existing plots): KPI row + 11 nested sub-tabs. Verify each plot renders from Envelope and MP results.

- **P5 — Matching tab** (≈ 300 LOC): target picker + before/after plot + Apply button. Verify periodic match copies back into Beam.

- **P6 — Convergence tab** (≈ 350 LOC): scan controls + `ScanWorker` QThread + results plot + table. Verify scan over nx ∈ {32, 48, 64, 96, 128} at 5 mA gives the expected convergence curve.

- **P7 — Tools popup dialogs** (≈ 200 LOC): Console and Jobs as QDialog. File menu Open/Save/Recent; Export Report; About.

- **P8 — Polish** (≈ 150 LOC): Settings dialog for PIC defaults, smoke test, README.

**Minimum viable:** P1–P4 (Beam + Lattice + Results) — already covers 90 % of actual usage. P5–P6 are analysis shortcuts; P7–P8 are polish.

**Total estimate:** ≈ 2 200 LOC net (deleting ≈ 2 500 from the old interphase, adding ≈ 1 700 net new — the rewrites are simpler than what they replace).

## Design invariants (no regressions)

- Same dark theme / cyan accent / monospace data / Inter text.
- Every plot currently accessible stays accessible (just reorganised).
- Classic GUI untouched.
- Backend untouched — no `linac_gen/` edits.
- All existing backend tests still pass (848).

## Open decisions (confirm before coding)

1. **Tab order** — your suggestion was Beam → Lattice → Results → Matching → Convergence. TraceWin users typically open Lattice first. I lean toward Lattice → Beam → Results → Matching → Convergence, but your order works too. **Default to your order unless you want to flip.**
2. **Console** — keep as Tools menu popup (recommended) or drop entirely? I'd keep it for power users; it's a one-screen dialog.
3. **Simulation settings** (PIC grid / extent / step config) — separate Settings dialog, or inline in the Convergence tab? I'd do separate dialog triggered from toolbar.
4. **Matching scope** — v1 ships "periodic" matching only (fast, useful). Element-range matching (match last 4 elements before dump) is v2; mention I'll stub the radio-button but only wire periodic initially?

## What to do next

Approve the plan (or flag changes) and I'll start P1. I'll keep the changes atomic per phase so we can stop at any checkpoint.
