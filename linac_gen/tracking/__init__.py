# linac_gen/tracking/__init__.py
"""Tracking package — forward (Tracker/EnvelopeSolver) and backward
(backtrack_distribution) transport.

Submodules are imported directly throughout the codebase
(``from linac_gen.tracking.tracker import Tracker``); the backtracking
entry points are additionally re-exported here for API convenience.
"""
from linac_gen.tracking.backtrack import (   # noqa: F401
    BacktrackWarning, BoundaryState, backtrack_distribution,
    build_replay_table,
)
