"""Tests for PoissonSolverFFT.update_grid — the in-place grid-extent
re-target used by adaptive grid mode.  Should be bit-equivalent to building
a fresh PoissonSolverFFT with the new extents (modulo the small
numpy↔cupy transcendental drift described in test_poisson_solver_gpu_parity).
"""
import numpy as np
import pytest

from linac_gen.pic.poisson_solver import PoissonSolverFFT


@pytest.mark.parametrize("green_kind", ["igf", "point"])
def test_update_grid_matches_fresh_instance_cpu(green_kind):
    """Solver A initialized with extent X, then update_grid(Y) → must match
    a fresh solver B built with extent Y (CPU backend, exact equality)."""
    n = np.array([16, 16, 16])

    A_min = np.array([-1.0, -1.0, -1.0])
    A_max = np.array([+1.0, +1.0, +1.0])
    B_min = np.array([-2.5, -1.7, -3.1])
    B_max = np.array([+2.5, +1.7, +3.1])

    A = PoissonSolverFFT(A_min, A_max, n, use_gpu="cpu", green_kind=green_kind)
    B = PoissonSolverFFT(B_min, B_max, n, use_gpu="cpu", green_kind=green_kind)
    A.update_grid(B_min, B_max)

    np.testing.assert_array_equal(A.grid_min, B.grid_min)
    np.testing.assert_array_equal(A.grid_max, B.grid_max)
    np.testing.assert_array_equal(A.dx, B.dx)
    np.testing.assert_allclose(A._G_fft, B._G_fft, rtol=0, atol=0,
                               err_msg="CPU G_fft must match bit-for-bit")


def test_update_grid_solve_matches_fresh_instance_cpu():
    """End-to-end: a re-targeted solver should give identical fields to a
    fresh solver on the same grid + ρ."""
    rng = np.random.default_rng(7)
    rho = rng.standard_normal((16, 16, 16))
    n = np.array([16, 16, 16])
    A_min = np.array([-1.0, -1.0, -1.0])
    A_max = np.array([+1.0, +1.0, +1.0])
    B_min = np.array([-2.0, -1.5, -3.0])
    B_max = np.array([+2.0, +1.5, +3.0])

    A = PoissonSolverFFT(A_min, A_max, n, use_gpu="cpu")
    B = PoissonSolverFFT(B_min, B_max, n, use_gpu="cpu")
    A.update_grid(B_min, B_max)

    Ea = A.solve(rho)
    Eb = B.solve(rho)
    for a, b, name in zip(Ea, Eb, ("Ex", "Ey", "Ez")):
        np.testing.assert_allclose(a, b, rtol=0, atol=0,
                                   err_msg=f"{name}: re-targeted solver must match fresh")


def test_update_grid_repeated_calls_stable_cpu():
    """Calling update_grid many times with varying extents shouldn't drift
    or accumulate numerical error: each rebuild is from scratch via the
    closed-form Green's function."""
    n = np.array([16, 16, 16])
    target_min = np.array([-1.7, -2.3, -1.1])
    target_max = np.array([+1.7, +2.3, +1.1])
    rng = np.random.default_rng(0)

    solver = PoissonSolverFFT(np.array([-1., -1., -1.]),
                              np.array([+1., +1., +1.]),
                              n, use_gpu="cpu")
    for _ in range(20):
        rand_min = rng.uniform(-3, -0.5, size=3)
        rand_max = rng.uniform(0.5, 3, size=3)
        solver.update_grid(rand_min, rand_max)

    solver.update_grid(target_min, target_max)
    fresh = PoissonSolverFFT(target_min, target_max, n, use_gpu="cpu")

    np.testing.assert_allclose(solver._G_fft, fresh._G_fft, rtol=0, atol=0,
                               err_msg="Repeated update_grid must be path-independent")
