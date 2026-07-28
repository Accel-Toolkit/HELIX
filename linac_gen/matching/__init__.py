"""Matching module: optimise lattice parameters to meet beam optics objectives.

Two layers of API:

* The legacy hand-driven :class:`Matcher` (manual ``add_variable`` /
  ``add_objective`` calls) — kept as-is for back-compat.
* The new lattice-driven :func:`match` engine (collects ``Variable``s
  and ``Constraint``s from ``ADJUST_*`` / ``SET_*`` cards in the lattice,
  runs a selectable scipy optimiser — least_squares, differential
  evolution, or dual annealing).  See :mod:`linac_gen.matching.engine`.
"""
from linac_gen.matching.constraints import Constraint, collect_constraints
from linac_gen.matching.engine import MATCH_ALGORITHMS, MatchResult, match
from linac_gen.matching.matcher import Matcher
from linac_gen.matching.objectives import evaluate_objectives
from linac_gen.matching.periodic import (
    find_fodo_cells, find_matched_input_twiss,
    find_periodic_twiss, find_sc_matched_input_twiss,
)
from linac_gen.matching.variables import (
    MatchingConfigError, Variable, collect_variables,
)

__all__ = [
    # Legacy
    "Matcher", "find_periodic_twiss", "evaluate_objectives",
    # Transfer-line input matching
    "find_matched_input_twiss", "find_fodo_cells",
    "find_sc_matched_input_twiss",
    # New
    "Variable", "Constraint", "MatchResult",
    "collect_variables", "collect_constraints", "match",
    "MATCH_ALGORITHMS", "MatchingConfigError",
]
