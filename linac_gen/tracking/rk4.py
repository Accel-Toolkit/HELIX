# linac_gen/tracking/rk4.py
"""RK4-based utilities for field map tracking.

This module provides helper functions used by FieldMap.track_rk4 and
FieldMap.fitted_matrix for numerical integration through external fields.
"""
import numpy as np


def rk4_step(state, ds, deriv_func):
    """Single 4th-order Runge-Kutta step.

    Parameters
    ----------
    state : array, shape (6,) or (N, 6)
        Current particle state(s).
    ds : float
        Step size (same units as deriv_func output).
    deriv_func : callable
        ``deriv_func(state) -> derivatives`` with the same shape as *state*.

    Returns
    -------
    new_state : same shape as *state*
        State after advancing by *ds*.
    """
    k1 = deriv_func(state)
    k2 = deriv_func(state + 0.5 * ds * k1)
    k3 = deriv_func(state + 0.5 * ds * k2)
    k4 = deriv_func(state + ds * k3)
    return state + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def numerical_jacobian(track_func, ref_state, eps=None, rel_scale=None):
    """Compute 6x6 Jacobian of a tracking function by central differences.

    Parameters
    ----------
    track_func : callable
        ``track_func(state_6d) -> state_6d``  Tracks a single particle
        (represented as a 6-vector of deviations) through the full element
        and returns the output 6-vector.
    ref_state : array, shape (6,)
        The reference (central) state, usually all zeros.
    eps : array, shape (6,), optional
        Absolute step sizes for each coordinate.  Defaults are chosen to
        be small relative to typical linac beam sizes (mm / mrad / deg / MeV):
        ``[0.01, 0.01, 0.01, 0.01, 0.01, 0.001]``.
    rel_scale : float, optional
        If given, each coordinate's step is widened to
        ``max(eps[j], rel_scale * |ref_state[j]|)``.  Useful for
        off-origin reference states or highly relativistic beams where
        the default absolute step is numerically noisy.

    Returns
    -------
    M : array, shape (6, 6)
        Linearised transfer matrix (Jacobian).
    """
    if eps is None:
        eps = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.001])
    eps = np.asarray(eps, dtype=np.float64).copy()
    if rel_scale is not None:
        for j in range(6):
            eps[j] = max(eps[j], rel_scale * abs(float(ref_state[j])))

    M = np.zeros((6, 6))
    for j in range(6):
        state_p = ref_state.copy()
        state_m = ref_state.copy()
        state_p[j] += eps[j]
        state_m[j] -= eps[j]
        out_p = track_func(state_p)
        out_m = track_func(state_m)
        M[:, j] = (out_p - out_m) / (2.0 * eps[j])
    return M


def numerical_jacobian_batched(track_batch, ref_state, eps=None, rel_scale=None):
    """:func:`numerical_jacobian` with every probe tracked in ONE call.

    Same probes, same order, same arithmetic: ``track_batch(P)`` receives the
    ``(12, 6)`` array of probe states — ``+eps_j`` then ``-eps_j`` for
    ``j = 0..5``, built exactly as the scalar helper builds its operands —
    and returns the ``(12, 6)`` tracked states; column ``j`` is
    ``(Y[2j] - Y[2j+1]) / (2 eps_j)``, one correctly rounded subtraction and
    one division per element, so every column is the same IEEE operation on
    the same operands as before.  The field-map trackers are vectorised over
    particles and advance the reference from on-axis quantities only, so the
    twelve single-particle tracks become one twelve-particle track with
    bit-identical output (tests/elements/test_fieldmap_jacobian_batched.py)
    and roughly an order of magnitude less Python overhead per slice.
    """
    if eps is None:
        eps = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.001])
    eps = np.asarray(eps, dtype=np.float64).copy()
    if rel_scale is not None:
        for j in range(6):
            eps[j] = max(eps[j], rel_scale * abs(float(ref_state[j])))

    P = np.empty((12, 6), dtype=np.float64)
    for j in range(6):
        state_p = ref_state.copy()
        state_m = ref_state.copy()
        state_p[j] += eps[j]
        state_m[j] -= eps[j]
        P[2 * j] = state_p
        P[2 * j + 1] = state_m
    Y = np.asarray(track_batch(P), dtype=np.float64)
    M = np.zeros((6, 6))
    for j in range(6):
        M[:, j] = (Y[2 * j] - Y[2 * j + 1]) / (2.0 * eps[j])
    return M
