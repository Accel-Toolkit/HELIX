"""Algebraic-multigrid solves (pyamg) that repeat bit for bit.

pyamg's smoothed-aggregation set-up estimates a spectral radius starting
from ``np.random.rand`` -- numpy's global generator, with no way to pass
one.  Two set-ups of the same matrix therefore build slightly different
solvers, the solutions they converge to differ within the solver
tolerance (2e-8 relative in an RFQ field map), and every set-up advances
the caller's global random sequence.  :func:`seeded` runs a set-up and
solve with that generator in a fixed state and puts the caller's state
back afterwards.  (The global generator is shared by all threads; HELIX
itself never draws from it.)
"""
from __future__ import annotations

import contextlib

import numpy as np

#: state of numpy's global generator inside :func:`seeded`
SEED = 0


@contextlib.contextmanager
def seeded():
    """numpy's global generator seeded with :data:`SEED` inside the block;
    the caller's state is restored on exit."""
    state = np.random.get_state()
    np.random.seed(SEED)
    try:
        yield
    finally:
        np.random.set_state(state)
