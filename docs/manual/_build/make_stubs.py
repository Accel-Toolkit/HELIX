"""Generate Markdown stubs for every chapter that doesn't have real content yet.

Each stub uses the manual's three-track structure (TL;DR / Tutorial /
API reference) with placeholder text.  Re-running this script does NOT
overwrite chapters that have grown past the stub stage — it skips files
larger than 1500 bytes (a stub is ~700-900 bytes).

Usage:
    python docs/manual/_build/make_stubs.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUB_LIMIT = 1500   # bytes — files bigger than this are considered "real"

# (relative_path, title, audience_track_hint)
CHAPTERS = [
    # 03_elements ---------------------------------------------------------
    ("03_elements/00_overview.md",   "Element catalog overview",          "all"),
    ("03_elements/01_drift.md",      "Drift",                             "elem"),
    ("03_elements/02_quadrupole.md", "Quadrupole",                        "elem"),
    ("03_elements/03_solenoid.md",   "Solenoid",                          "elem"),
    ("03_elements/04_dipole.md",     "Dipole",                            "elem"),
    ("03_elements/05_edge.md",       "Edge",                              "elem"),
    ("03_elements/06_rfgap.md",      "RFGap",                             "elem"),
    ("03_elements/07_fieldmap.md",   "FieldMap (1-D / 2-D)",              "elem"),
    ("03_elements/08_fieldmap3d.md", "FieldMap3D",                        "elem"),
    ("03_elements/09_rfqcell.md",    "RfqCell",                           "elem"),
    ("03_elements/10_vanerfq.md",    "VaneRFQ",                           "elem"),
    ("03_elements/11_multipole.md",  "Multipole",                         "elem"),
    ("03_elements/12_aperture.md",   "Aperture",                          "elem"),
    ("03_elements/13_marker.md",     "Marker",                            "elem"),
    ("03_elements/14_steerer.md",    "Steerer",                           "elem"),
    # 04_beam -------------------------------------------------------------
    ("04_beam/01_distributions.md",  "Distributions",                     "all"),
    ("04_beam/02_twiss.md",          "Twiss & emittance conventions",     "all"),
    ("04_beam/03_beam_config.md",    "BeamConfig field reference",        "ref"),
    ("04_beam/04_dst_io.md",         "TraceWin .dst loading",             "all"),
    # 05_space_charge -----------------------------------------------------
    ("05_space_charge/01_models.md",         "Space-charge models",       "all"),
    ("05_space_charge/02_pic_solver.md",     "PIC solver",                "all"),
    ("05_space_charge/03_kernels.md",        "Kernels & Green's function","all"),
    ("05_space_charge/04_dc_mode.md",        "DC mode (continuous beam)", "all"),
    ("05_space_charge/05_convergence.md",    "Convergence guide",         "all"),
    # 06_running ----------------------------------------------------------
    ("06_running/01_python_api.md",   "Python API",                       "ref"),
    ("06_running/02_tracewin_dat.md", ".dat file reference",              "ref"),
    ("06_running/03_lg_extensions.md","HELIX-specific extensions",        "ref"),
    ("06_running/04_results.md",      "Reading results",                  "all"),
    # 07_matching ---------------------------------------------------------
    ("07_matching/01_overview.md",   "Matching overview",                 "all"),
    ("07_matching/02_set_adjust.md", "SET / ADJUST card reference",       "ref"),
    ("07_matching/03_python_api.md", "Matching Python API",               "ref"),
    ("07_matching/04_cli.md",        "Matching CLI",                      "all"),
    ("07_matching/05_recipes.md",    "Matching recipes",                  "all"),
    # 08_errors -----------------------------------------------------------
    ("08_errors/01_overview.md",         "Error studies overview",        "all"),
    ("08_errors/02_error_directives.md", "ERROR_* directives",            "ref"),
    ("08_errors/03_element_errors.md",   "Element-level errors",          "all"),
    ("08_errors/04_beam_errors.md",      "Beam-level errors",             "all"),
    ("08_errors/05_running_studies.md",  "Running an error study",        "all"),
    ("08_errors/06_interpreting.md",     "Interpreting ensemble results", "all"),
    # 09_diagnostics ------------------------------------------------------
    ("09_diagnostics/01_recorder.md",       "Recorder fields",            "ref"),
    ("09_diagnostics/02_emittances.md",     "Emittances",                 "all"),
    ("09_diagnostics/03_halo.md",           "Halo analysis",              "all"),
    ("09_diagnostics/04_aperture.md",       "Aperture profile",           "all"),
    ("09_diagnostics/05_stripping.md",      "H⁻ stripping",               "all"),
    ("09_diagnostics/06_phase_advance.md",  "Phase advance",              "all"),
    # 10_gui --------------------------------------------------------------
    ("10_gui/01_overview.md",         "GUI overview",                     "all"),
    ("10_gui/02_lattice_tab.md",      "Lattice tab",                      "all"),
    ("10_gui/03_beam_tab.md",         "Beam tab",                         "all"),
    ("10_gui/04_convergence_tab.md",  "Numerics tab",                  "all"),
    ("10_gui/05_matching_tab.md",     "Matching tab",                     "all"),
    ("10_gui/06_errors_tab.md",       "Error Study tab",                       "all"),
    ("10_gui/07_results_tab.md",      "Results tab",                      "all"),
    ("10_gui/08_workflows.md",        "GUI workflows",                    "all"),
    # 11_examples ---------------------------------------------------------
    ("11_examples/01_basic_fodo.md",            "Worked example: Basic FODO",          "all"),
    ("11_examples/02_envelope_demo.md",         "Worked example: Envelope demo",       "all"),
    ("11_examples/03_lebt_dc.md",               "Worked example: DC LEBT",             "all"),
    ("11_examples/04_lebt_rfq.md",              "Worked example: LEBT + RFQ",          "all"),
    ("11_examples/05_pipii_mebt.md",            "Worked example: PIP-II MEBT",         "all"),
    ("11_examples/06_pipii_full.md",            "Worked example: Full PIP-II linac",   "all"),
    ("11_examples/07_matching_walkthrough.md",  "Worked example: Matching walkthrough","all"),
    ("11_examples/08_tolerance_study.md",       "Worked example: Tolerance study",     "all"),
    ("11_examples/09_eigenemittance.md",        "Worked example: Eigenemittance",      "all"),
    # 12_validation -------------------------------------------------------
    ("12_validation/01_tracewin_parity.md",   "TraceWin parity",                       "all"),
    ("12_validation/02_pipii_validation.md",  "PIP-II validation",                     "all"),
    ("12_validation/03_known_limitations.md", "Known limitations & deferred features", "all"),
    ("12_validation/04_convergence_guide.md", "Convergence checklist",                 "all"),
    # appendices ----------------------------------------------------------
    ("appendices/A_physics_references.md",   "Appendix A — Physics references",        "ref"),
    ("appendices/B_keyword_cheatsheet.md",   "Appendix B — TraceWin keyword cheatsheet","ref"),
    ("appendices/C_glossary.md",             "Appendix C — Glossary",                  "all"),
    ("appendices/D_troubleshooting.md",      "Appendix D — Troubleshooting",           "all"),
    ("appendices/E_migrating_from_tw.md",    "Appendix E — Migrating from TraceWin",   "all"),
    ("appendices/F_contributing.md",         "Appendix F — Contributing",              "ref"),
]


STUB_TEMPLATE_ELEM = """# {title}

!!! abstract "Status"
    This chapter is scheduled for Phase 2 of the manual.  Until it
    fills in, see the source: [`linac_gen/elements/`](https://github.com/Abhishek-Pathak-90/HELIX/tree/master/linac_gen/elements).

## TL;DR (TraceWin users)

*To be written.*  HELIX `{title}` ↔ TraceWin equivalent — see the
[Keyword cheatsheet](../appendices/B_keyword_cheatsheet.md).

## Tutorial (newcomers)

*To be written.*  Will explain what this element does, when to use
it, with a runnable code example and a plot.

## API reference (developers)

*To be written.*  Constructor signature + parameter table + source
link.

## See also

* [Element catalog overview](00_overview.md)
* [Tracking modes](../02_concepts/03_tracking_modes.md)
"""


STUB_TEMPLATE_REF = """# {title}

!!! abstract "Status"
    This chapter is scheduled for a later phase of the manual.

## Reference

*Content to be written.*  Will be a comprehensive reference table
covering every field / parameter / keyword in this category.

## See also

* [Quick start](../01_getting_started/03_quick_start.md)
* [Python API](../06_running/01_python_api.md)
"""


STUB_TEMPLATE_ALL = """# {title}

!!! abstract "Status"
    This chapter is scheduled for a later phase of the manual.

## TL;DR

*To be written.*

## Tutorial

*To be written.*

## API reference

*To be written.*

## See also

* [Quick start](../01_getting_started/03_quick_start.md)
* [Tracking modes](../02_concepts/03_tracking_modes.md)
"""


TEMPLATES = {
    "elem": STUB_TEMPLATE_ELEM,
    "ref": STUB_TEMPLATE_REF,
    "all": STUB_TEMPLATE_ALL,
}


def main() -> int:
    n_written, n_skipped = 0, 0
    for rel_path, title, hint in CHAPTERS:
        p = ROOT / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size > STUB_LIMIT:
            n_skipped += 1
            continue
        body = TEMPLATES[hint].format(title=title)
        p.write_text(body, encoding="utf-8")
        n_written += 1
    print(f"Stub generation: {n_written} written, {n_skipped} skipped (already filled in).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
