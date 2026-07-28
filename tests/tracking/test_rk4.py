# tests/tracking/test_rk4.py
"""Tests for the rk4_step integrator and numerical_jacobian utility."""
import numpy as np
import pytest

from linac_gen.tracking.rk4 import rk4_step, numerical_jacobian


# ------------------------------------------------------------------ #
#  rk4_step: basic correctness
# ------------------------------------------------------------------ #

class TestRk4StepScalar:
    """Test rk4_step with a scalar (1-D) ODE: dy/dt = -y, y(0) = 1."""

    @staticmethod
    def _exp_decay(state):
        return -state

    def test_single_step_close_to_exact(self):
        """One RK4 step of dy/ds = -y should approximate exp(-ds)."""
        y0 = np.array([1.0])
        ds = 0.1
        y1 = rk4_step(y0, ds, self._exp_decay)
        expected = np.exp(-ds)
        # RK4 local error is O(ds^5); for ds=0.1 this is ~1e-7
        np.testing.assert_allclose(y1[0], expected, rtol=1e-6)

    def test_multiple_steps_accumulate(self):
        """100 steps of dy/ds = -y over ds=0.01 each -> y ~ exp(-1)."""
        y = np.array([1.0])
        ds = 0.01
        for _ in range(100):
            y = rk4_step(y, ds, self._exp_decay)
        np.testing.assert_allclose(y[0], np.exp(-1.0), rtol=1e-8)

    def test_returns_same_shape(self):
        """Output shape must match input shape."""
        y0 = np.array([2.5])
        y1 = rk4_step(y0, 0.05, self._exp_decay)
        assert y1.shape == y0.shape

    def test_zero_ds_returns_unchanged(self):
        """Step size zero should leave state unchanged."""
        y0 = np.array([3.14])
        y1 = rk4_step(y0, 0.0, self._exp_decay)
        np.testing.assert_array_equal(y1, y0)


class TestRk4StepHarmonic:
    """Test rk4_step on a 2-D harmonic oscillator: state = [x, v].

    Equations:  dx/dt = v,   dv/dt = -x
    Exact:      x(t) = cos(t),  v(t) = -sin(t)
    """

    @staticmethod
    def _harmonic(state):
        x, v = state
        return np.array([v, -x])

    def test_quarter_period(self):
        """Integrate over t = pi/2 (quarter period): x->0, v->-1."""
        state = np.array([1.0, 0.0])
        # Use many small steps for accuracy
        ds = np.pi / 2 / 1000
        for _ in range(1000):
            state = rk4_step(state, ds, self._harmonic)
        np.testing.assert_allclose(state[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(state[1], -1.0, atol=1e-7)

    def test_full_period_returns_to_start(self):
        """After one full period (2*pi), state should return to initial."""
        state0 = np.array([1.0, 0.0])
        state = state0.copy()
        ds = 2 * np.pi / 1000
        for _ in range(1000):
            state = rk4_step(state, ds, self._harmonic)
        np.testing.assert_allclose(state, state0, atol=1e-6)


class TestRk4StepBatch:
    """Test rk4_step with a 2-D batch state (N, 6)."""

    @staticmethod
    def _neg_identity(state):
        """dy/ds = -y element-wise."""
        return -state

    def test_batch_shape_preserved(self):
        """(N, 6) input should return (N, 6) output."""
        state = np.ones((5, 6))
        out = rk4_step(state, 0.1, self._neg_identity)
        assert out.shape == (5, 6)

    def test_batch_matches_scalar(self):
        """Each row of a batch should match the scalar result."""
        row = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        state_batch = np.tile(row, (4, 1))
        ds = 0.05

        out_batch = rk4_step(state_batch, ds, self._neg_identity)
        out_scalar = rk4_step(row, ds, self._neg_identity)

        for i in range(4):
            np.testing.assert_array_equal(out_batch[i], out_scalar)

    def test_batch_independent_rows(self):
        """Different initial conditions should evolve independently."""
        rng = np.random.default_rng(0)
        state = rng.standard_normal((10, 6))
        ds = 0.01
        out = rk4_step(state, ds, self._neg_identity)
        expected = state * np.exp(-ds)
        np.testing.assert_allclose(out, expected, rtol=1e-8)

    def test_does_not_modify_input(self):
        """rk4_step must not modify the input state array."""
        state = np.array([1.0, 0.0])
        original = state.copy()
        _ = rk4_step(state, 0.1, lambda s: -s)
        np.testing.assert_array_equal(state, original)


# ------------------------------------------------------------------ #
#  rk4_step: order-of-accuracy check
# ------------------------------------------------------------------ #

class TestRk4StepAccuracy:
    """Verify RK4 is 4th order in step size."""

    @staticmethod
    def _deriv(state):
        """dy/ds = y  ->  y(s) = exp(s)."""
        return state

    def test_fourth_order_convergence(self):
        """Global error should scale as O(ds^4)."""
        y_exact = np.exp(1.0)  # y(1) with y(0) = 1

        errors = []
        for n in [10, 20, 40]:
            y = np.array([1.0])
            ds = 1.0 / n
            for _ in range(n):
                y = rk4_step(y, ds, self._deriv)
            errors.append(abs(y[0] - y_exact))

        # Each doubling of steps should reduce error by ~16x (4th order)
        ratio_1 = errors[0] / errors[1]
        ratio_2 = errors[1] / errors[2]
        assert ratio_1 > 10.0, f"Expected ~16x error reduction, got {ratio_1:.2f}"
        assert ratio_2 > 10.0, f"Expected ~16x error reduction, got {ratio_2:.2f}"


# ------------------------------------------------------------------ #
#  numerical_jacobian: basic tests
# ------------------------------------------------------------------ #

class TestNumericalJacobian:
    """Tests for the numerical_jacobian utility."""

    def test_identity_function(self):
        """f(x) = x should give 6x6 identity Jacobian."""
        def _identity(state):
            return state.copy()

        ref = np.zeros(6)
        M = numerical_jacobian(_identity, ref)
        np.testing.assert_allclose(M, np.eye(6), atol=1e-10)

    def test_linear_map(self):
        """f(x) = A @ x for known A should recover A."""
        A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        def _linear(state):
            return A @ state

        ref = np.zeros(6)
        M = numerical_jacobian(_linear, ref)
        np.testing.assert_allclose(M, A, atol=1e-8)

    def test_output_shape(self):
        """Output should always be (6, 6)."""
        M = numerical_jacobian(lambda s: s, np.zeros(6))
        assert M.shape == (6, 6)

    def test_drift_like_map(self):
        """A simple drift: x += xp * L_m, y += yp * L_m."""
        L_m = 0.1  # 100 mm drift in meters

        def _drift(state):
            out = state.copy()
            out[0] += state[1] * L_m  # x += xp * L
            out[2] += state[3] * L_m  # y += yp * L
            return out

        ref = np.zeros(6)
        M = numerical_jacobian(_drift, ref)
        # M[0,1] should be L_m = 0.1
        np.testing.assert_allclose(M[0, 1], L_m, atol=1e-8)
        np.testing.assert_allclose(M[2, 3], L_m, atol=1e-8)
        # Diagonal should be 1
        for i in range(6):
            np.testing.assert_allclose(M[i, i], 1.0, atol=1e-8)
