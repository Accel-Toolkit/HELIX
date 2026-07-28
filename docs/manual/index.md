# HELIX User Manual

**HELIX** is an open-source Python toolkit for end-to-end simulation
of charged-particle linear accelerators.  It combines a TraceWin-compatible
lattice language, a fast envelope solver, a multi-particle tracker with a
3-D particle-in-cell space-charge solver, and a complete GUI workbench
into one tool.

This manual is the comprehensive reference: every element, every
configuration knob, every command, every diagnostic, with worked
examples and validated benchmarks.

---

## Choose your track

The manual is structured to serve three audiences in parallel:

=== "TraceWin user"

    You already know your way around a `.dat` file and partran output.
    The fastest entry points:

    * **[Quick start](01_getting_started/03_quick_start.md)** — load a
      `.dat`, run, plot.  10 minutes.
    * **[Migrating from TraceWin](appendices/E_migrating_from_tw.md)**
      — porting checklist, gotchas.
    * **[Keyword cheatsheet](appendices/B_keyword_cheatsheet.md)** —
      every TraceWin keyword ↔ HELIX class on one page.
    * **[TraceWin parity](12_validation/01_tracewin_parity.md)** —
      what matches, what's known to differ.

=== "Linac newcomer"

    No prior linac experience required, just Python literacy.
    Recommended path:

    * **[Introduction](01_getting_started/01_introduction.md)** — what
      is a linac, what does HELIX do.
    * **[Coordinates & units](02_concepts/01_coordinates.md)** — the
      6-D phase space we work in.
    * **[Basic FODO](11_examples/01_basic_fodo.md)** — your first
      lattice, end-to-end.
    * **[Glossary](appendices/C_glossary.md)** — every domain term
      defined.

=== "HELIX developer"

    Extending the code, adding elements, debugging physics.

    * **[Data model](02_concepts/02_data_model.md)** — Lattice, Beam,
      ReferenceParticle internals.
    * **[Python API](06_running/01_python_api.md)** — Simulation,
      Tracker, EnvelopeSolver entry points.
    * **[Contributing](appendices/F_contributing.md)** — coding
      conventions, test discipline, PR flow.
    * **[Physics references](appendices/A_physics_references.md)** —
      every paper cited by the codebase.

---

## What's in the manual

| Part | Content |
|---|---|
| [I. Getting Started](01_getting_started/01_introduction.md) | Install, quick start, first run |
| [II. Concepts](02_concepts/01_coordinates.md) | Coordinates, data model, tracking modes |
| [III. Elements](03_elements/00_overview.md) | 17 element types, each with TL;DR + tutorial + API ref |
| [IV. Beam](04_beam/01_distributions.md) | Distributions, Twiss, BeamConfig, file I/O |
| [V. Space charge](05_space_charge/01_models.md) | Models, PIC, kernels, DC mode, convergence |
| [VI. Running](06_running/01_python_api.md) | Python API, `.dat` reference, results, GUI, batch-mode CLI |
| [VII. Matching](07_matching/01_overview.md) | SET / ADJUST cards, matching engine, recipes |
| [VIII. Errors](08_errors/01_overview.md) | Tolerance studies, alignment, RF jitter |
| [IX. Diagnostics](09_diagnostics/01_recorder.md) | Recorder, emittances, halo, stripping |
| [X. GUI](10_gui/01_overview.md) | Tab-by-tab tour, workflows |
| [XI. Worked examples](11_examples/01_basic_fodo.md) | 9 end-to-end walkthroughs |
| [XII. Validation](12_validation/01_tracewin_parity.md) | TraceWin parity, PIP-II benchmark |
| [XIII. ML Surrogates](13_surrogates/01_overview.md) | Train MLP drop-ins for field maps; GUI / CLI / API |
| [Appendices](appendices/A_physics_references.md) | References, keyword cheatsheet, glossary, troubleshooting, TraceWin migration, contributing |

---

## Conventions used in this manual

!!! info "TL;DR cards"
    Every element and major-feature chapter starts with a one-paragraph
    summary card — for readers who already know the territory and just
    want the HELIX-specific facts.

!!! tip "Tutorials"
    The middle section of each chapter is a narrative walkthrough.  All
    code blocks are runnable; copy them into a Python REPL or a `.py`
    file and they will execute against the current HELIX install.

!!! abstract "API reference"
    The end of each chapter has the Python signature and parameter
    table — for developers reading code or extending HELIX.

!!! warning "Caveats"
    Boxed callouts mark known issues, deferred features, or
    convention differences vs. TraceWin / IMPACT-X / other codes.

---

## Building this manual locally

```bash
# from repo root
pip install -e ".[docs]"
mkdocs serve --config-file docs/mkdocs.yml
# → http://127.0.0.1:8000/
```

To regenerate every figure from scratch:

```bash
python docs/manual/_build/regen_plots.py
```

To verify every code snippet still runs:

```bash
python docs/manual/_build/verify_snippets.py
```

---

*HELIX is developed at Fermi National Accelerator Laboratory and
collaborating institutions.  See [Contributing](appendices/F_contributing.md)
for how to participate.*
