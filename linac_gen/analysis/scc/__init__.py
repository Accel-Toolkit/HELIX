"""LEBT space-charge-compensation (SCC) analysis.

Ported from the author's standalone SCC simulator
(~/Desktop/Projects/scc, GPL-3.0-or-later, same author) — the ANALYSIS
layer only: gas library + cross sections (``constants``), compensation/
loss physics (``physics``), and the self-consistent steady-state
neutralisation balance (``radial`` + ``balance`` — a 1-D radial
nonlinear Poisson–Boltzmann solve, no particles).  The source tool's
envelope/macroparticle/PIC trackers are NOT ported: HELIX's own DC
tracking supersedes them, and the computed ``f_c`` feeds HELIX's
existing :class:`~linac_gen.elements.space_charge_comp.SpaceChargeComp`
cards.

Calibration disclosure (see the ``constants`` docstring): cross-section
SHAPES and peak positions are sourced; the twelve absolute magnitudes
are an inherited calibration to Valerio-Lizarraga CERN-THESIS-2015-121
(~10x published bare sigma_ion) and must not be quoted as literature
values.  Keep the physics files byte-diffable against the source repo.
"""
from linac_gen.analysis.scc.balance import SCCNotConverged, SCCResult, SelfConsistentSCC
from linac_gen.analysis.scc.constants import (
    BEAM_SPECIES,
    GAS_LIBRARY,
    Species,
    sigma_capture,
    sigma_ionization,
    sigma_stripping,
)
from linac_gen.analysis.scc.physics import (
    GasMix,
    beam_loss_rate_per_length,
    beam_potential_axis,
    compensation_fraction,
    compensation_time,
    generalized_perveance,
    rms_to_edge_radius,
)
from linac_gen.analysis.scc.radial import RadialGrid

__all__ = [
    "BEAM_SPECIES", "GAS_LIBRARY", "Species", "GasMix", "RadialGrid",
    "SelfConsistentSCC", "SCCResult", "SCCNotConverged",
    "sigma_ionization", "sigma_stripping", "sigma_capture",
    "compensation_time", "compensation_fraction",
    "beam_loss_rate_per_length", "beam_potential_axis",
    "generalized_perveance", "rms_to_edge_radius",
]
