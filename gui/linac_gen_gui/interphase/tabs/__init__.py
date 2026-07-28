"""Top-level tabs (Beam · Lattice · Matching · Numerics · Surrogates · Error Study · Failure Study · Results)."""
from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
from linac_gen_gui.interphase.tabs.lattice_tab import LatticeTab
from linac_gen_gui.interphase.tabs.matching_tab import MatchingTab
from linac_gen_gui.interphase.tabs.convergence_tab import ConvergenceTab
from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
from linac_gen_gui.interphase.tabs.error_study_tab import ErrorStudyTab
from linac_gen_gui.interphase.tabs.failure_study_tab import FailureStudyTab
from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

__all__ = [
    "BeamTab", "LatticeTab", "MatchingTab",
    "ConvergenceTab", "SurrogatesTab", "ErrorStudyTab", "FailureStudyTab",
    "ResultsTab",
]
