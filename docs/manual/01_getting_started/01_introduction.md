# Introduction

HELIX is an open-source toolkit for end-to-end simulation of charged-particle
linear accelerators.  It is written in Python with optional C++ /
CUDA acceleration, ships a TraceWin-compatible lattice language, and
provides both a programmable API and a complete graphical workbench.

## Where HELIX fits

| Need | HELIX provides |
|---|---|
| Read TraceWin `.dat` files | Full parser including `ERROR_*` directives, `FIELD_MAP`, `RFQ_CELL`, matching cards |
| Envelope (RMS) tracking | 6×6 σ-matrix propagation with uniform-ellipsoid space charge |
| Multi-particle tracking | Up to ~10⁶ macroparticles with 3-D PIC space charge (CPU or GPU) |
| Continuous (DC) beams | Sacherer ODE for pre-RFQ LEBT, 2-D analytic SC kick, DC ↔ bunched transition |
| Matching & optimisation | Levenberg-Marquardt SET/ADJUST language matching, Python and CLI |
| Tolerance / error studies | Monte-Carlo ensembles over alignment, field, RF, beam-input errors |
| Diagnostics | σ, Twiss, halo, eigenemittance, transmission, aperture loss, H⁻ stripping |
| GUI workbench | Tabbed PyQt6 application with embedded plots |

## Comparison to other codes

HELIX shares a lot of physics convention with **TraceWin** (Saclay), so
TraceWin users transition quickly.  It also borrows architectural
patterns and a few diagnostic features from **IMPACT-X** (Berkeley Lab).

| | HELIX | TraceWin | IMPACT-X | PyORBIT |
|---|---|---|---|---|
| Language | Python (+C++/CUDA) | Fortran (closed) | C++ | C++ + Python |
| Open source | Yes | No | Yes | Yes |
| Native `.dat` parser | Yes (TraceWin format) | Yes | Limited | No |
| Envelope tracker | Yes | Yes | No | Yes |
| 3-D PIC SC | Yes (CPU + GPU) | Yes (PICNIR) | Yes | Yes |
| GUI | PyQt6 (HELIX) | Native Windows | None | None |
| Eigenemittances | Yes (Balandin 6-D) | No | Yes | No |
| TraceWin error directives | Yes | Yes | Manual | Manual |
| H⁻ stripping (analytic) | Yes (Folsom 2021) | Yes | No | Yes |

For a quantitative comparison of HELIX vs TraceWin partran on
production lattices, see
[TraceWin parity](../12_validation/01_tracewin_parity.md) and
PIP-II validation.

## Audience tracks

This manual serves three audiences in parallel.  Pick the entry point
that best matches your background:

### TraceWin user

You already know the `.dat` syntax, partran output, and SET/ADJUST
matching.  Your fast path:

1. **[Installation](02_installation.md)** — most users want
   `pip install -e .` or the bundled Windows `.exe`.
2. **[Quick start](03_quick_start.md)** — open an existing `.dat`,
   run, plot.  10 minutes.
3. **[Migrating from TraceWin](../appendices/E_migrating_from_tw.md)**
   — porting checklist, parity gotchas.
4. **[Keyword cheatsheet](../appendices/B_keyword_cheatsheet.md)** —
   every `.dat` keyword ↔ HELIX class on one page.

### Linac newcomer

No prior linac experience required, but Python literacy is assumed.
Recommended path:

1. **[Installation](02_installation.md)**
2. **[Coordinates & units](../02_concepts/01_coordinates.md)** — the
   6-D phase space we work in.
3. **[Data model](../02_concepts/02_data_model.md)** — `Lattice`,
   `Beam`, `ReferenceParticle`.
4. **[Tracking modes](../02_concepts/03_tracking_modes.md)** —
   when to use envelope vs multi-particle.
5. **[Basic FODO worked example](../11_examples/01_basic_fodo.md)** —
   end-to-end first lattice.
6. **[Glossary](../appendices/C_glossary.md)** — every domain term
   defined.

### HELIX developer

Extending the code, adding elements, debugging physics:

1. **[Data model](../02_concepts/02_data_model.md)** — internals.
2. **[Python API](../06_running/01_python_api.md)** — `Simulation`,
   `Tracker`, `EnvelopeSolver`.
3. **[Contributing](../appendices/F_contributing.md)** — coding
   conventions, test discipline, PR flow.
4. **[Physics references](../appendices/A_physics_references.md)** —
   every paper cited by the codebase.

## What HELIX does not do

A non-exhaustive list of out-of-scope physics — for these, use a
specialised code or extend HELIX:

* **Wakefields** (longitudinal, transverse, resistive-wall).
* **Image-charge / pipe-current effects** beyond simple aperture cuts.
* **Synchrotron radiation damping** (irrelevant for proton/H⁻ linacs;
  relevant for high-energy electron storage rings).
* **Time-varying / dynamic errors** — TraceWin's `ERROR_*_DYN`
  directives are deferred.  Static per-seed errors cover ~95 % of
  real workflows.

(Steerer-based orbit correction, formerly on this list, has since
shipped: `ADJUST_STEERER` / `ADJUST_STEERER_BX` / `ADJUST_STEERER_BY`
now build matcher variables that drive the targeted steerers.)

For the full deferred-feature list see
[Known limitations](../12_validation/03_known_limitations.md).

## Getting help

* **GitHub issues** — bug reports, feature requests.
* **GUI** — every input field has a tooltip; hover for hints.
* **Slash command** — when running HELIX, `--help` on any CLI shows
  flags; `Tools → About` in the GUI shows version & build details.

Continue to [Installation](02_installation.md) →
