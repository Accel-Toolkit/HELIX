"""PoissonSolverFFT.solve without its redundant work gives the old solve's
bits (linac_gen.pic.poisson_solver._lean_solve): the padded charge buffer is
not zeroed again, the inverse FFT works in place on the spectrum, and the
field differences run in the compiled kernel -- at any thread count, over
repeated solves of one solver (the buffer reused), with a NaN charge in
between.  LINAC_GEN_LEAN_SOLVE=0 restores the old path."""
from __future__ import annotations

import numpy as np
import pytest

import linac_gen.pic.poisson_solver as ps
from linac_gen.pic import fft_split
from linac_gen.pic.poisson_solver import PoissonSolverFFT
from tests.helpers import kept_thread_setting

GRIDS = [(8, 8, 8), (17, 12, 9), (32, 32, 32), (48, 40, 56)]


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _solver(grid, lean, monkeypatch):
    monkeypatch.setenv("LINAC_GEN_LEAN_SOLVE", "1" if lean else "0")
    return PoissonSolverFFT(np.array([-1.0, -2.0, -1.5]),
                            np.array([1.0, 2.0, 1.2]), np.array(grid),
                            use_gpu="cpu")


def _solve(solver, rho, lean, monkeypatch):
    monkeypatch.setenv("LINAC_GEN_LEAN_SOLVE", "1" if lean else "0")
    with np.errstate(all="ignore"):
        return solver.solve(rho)


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: "x".join(map(str, g)))
def test_lean_solve_equals_the_old_solve(grid, monkeypatch):
    """The grids from 32^3 on are split by fft_split (the in-place inverse
    applies there); the smaller ones are single scipy calls."""
    rng = np.random.default_rng(7)
    rhos = [rng.normal(size=grid) for _ in range(3)]
    rhos[1][1, 2, 3] = np.nan
    with kept_thread_setting() as threads:
        for t in (1, 4, 14):
            threads.set_threads(t)
            lean = _solver(grid, True, monkeypatch)
            old = _solver(grid, False, monkeypatch)
            for k, rho in enumerate(rhos):
                a = _solve(lean, rho, True, monkeypatch)
                b = _solve(old, rho, False, monkeypatch)
                for fa, fb in zip(a, b):
                    assert np.array_equal(_bits(fa), _bits(fb)), (t, k)
                    assert fa.flags.c_contiguous and fa.dtype == np.float64
                # outside its charge block the reused buffer keeps +0.0
                nx, ny, nz = grid
                pad = lean._rho_padded.copy()
                pad[:nx, :ny, :nz] = 0.0
                assert not _bits(pad).any()


def test_in_place_inverse_is_used_only_when_given_up(monkeypatch):
    """fft_split.irfftn transforms the caller's spectrum in place only with
    overwrite_x and a writeable C-contiguous array; the result is the same."""
    from scipy import fft as sfft
    rng = np.random.default_rng(3)
    shape = (64, 64, 64)
    X = sfft.rfftn(rng.normal(size=shape), workers=1)
    want = sfft.irfftn(X, s=shape, workers=1)
    kept = X.copy()
    assert np.array_equal(_bits(fft_split.irfftn(X, shape, 4)), _bits(want))
    assert np.array_equal(_bits(X), _bits(kept))            # X untouched
    ro = X.copy()
    ro.flags.writeable = False
    assert np.array_equal(_bits(fft_split.irfftn(ro, shape, 4,
                                                 overwrite_x=True)),
                          _bits(want))
    assert np.array_equal(_bits(ro), _bits(kept))           # read-only: copied
    strided = np.zeros((64, 64, 66), dtype=complex)[:, :, :33]
    strided[...] = X
    assert np.array_equal(_bits(fft_split.irfftn(strided, shape, 4,
                                                 overwrite_x=True)),
                          _bits(want))
    assert np.array_equal(_bits(strided), _bits(kept))      # strided: copied
    given = X.copy()
    assert np.array_equal(_bits(fft_split.irfftn(given, shape, 4,
                                                 overwrite_x=True)),
                          _bits(want))
    assert not np.array_equal(_bits(given), _bits(kept))    # used in place


def _numpy_fields(phi, d, monkeypatch):
    monkeypatch.setenv("LINAC_GEN_LEAN_SOLVE", "0")
    out = ps._field_from_potential(phi, *d, np)
    monkeypatch.setenv("LINAC_GEN_LEAN_SOLVE", "1")
    return out


try:  # noqa: SIM105  -- on macOS the kernel needs torch's OpenMP runtime
    import torch as _torch_runtime  # type: ignore  # noqa: F401
except Exception:                                     # pragma: no cover
    pass
needs_kernel = pytest.mark.skipif(ps._field_kernel() is None,
                                  reason="_tsc_kernels without field_from_potential")


@needs_kernel
@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 3, 4), (5, 7, 2), (3, 3, 3),
                                   (17, 9, 33), (48, 48, 48)],
                         ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("layout", ["c", "corner", "fortran", "reversed"])
def test_field_kernel_equals_numpy(shape, layout, monkeypatch):
    """Strided views (solve passes the padded potential's corner), signed
    zeros, NaN and inf, two-node axes (boundary planes only), any threads."""
    rng = np.random.default_rng(13)
    big = rng.normal(size=tuple(2 * s for s in shape))
    phi = {"c": rng.normal(size=shape),
           "corner": big[:shape[0], :shape[1], :shape[2]],
           "fortran": np.asfortranarray(rng.normal(size=shape)),
           "reversed": big[::-2, ::2, ::-2][:shape[0], :shape[1], :shape[2]]
           }[layout]
    if layout == "c":
        phi.flat[::7] = 0.0
        phi.flat[1::11] = -0.0
        phi.flat[3] = np.nan
        phi.flat[5] = np.inf
    d = (np.float64(0.37), np.float64(2.5e-3), np.float64(7e-4))
    with np.errstate(all="ignore"):
        want = _numpy_fields(phi, d, monkeypatch)
        with kept_thread_setting() as threads:
            for t in (1, 3, 14):
                threads.set_threads(t)
                got = ps._field_from_potential(phi, *d, np)
                for g, w in zip(got, want):
                    assert np.array_equal(_bits(g), _bits(w)), t
                    assert g.flags.c_contiguous


@needs_kernel
@pytest.mark.parametrize("shape", [(1, 4, 4), (4, 1, 4), (4, 4, 1)])
def test_field_kernel_leaves_short_axes_to_numpy(shape, monkeypatch):
    """An axis of one node: numpy's IndexError, with or without the kernel."""
    phi = np.ones(shape)
    assert ps._field_kernel()(phi, 0.1, 0.1, 0.1) is None
    for lean in ("1", "0"):
        monkeypatch.setenv("LINAC_GEN_LEAN_SOLVE", lean)
        with pytest.raises(IndexError):
            ps._field_from_potential(phi, 0.1, 0.1, 0.1, np)


def test_old_path_when_switched_off(monkeypatch):
    """LINAC_GEN_LEAN_SOLVE=0: the buffer is zeroed before every solve, the
    inverse gets a copy, and the kernel is not called."""
    calls = []
    monkeypatch.setattr(ps, "_field_kernel",
                        lambda: (lambda *a: calls.append(1)))
    solver = _solver((8, 8, 8), False, monkeypatch)
    rho = np.ones((8, 8, 8))
    _solve(solver, rho, False, monkeypatch)
    solver._rho_padded[-1, -1, -1] = 5.0          # zeroed again by the next solve
    _solve(solver, rho, False, monkeypatch)
    assert solver._rho_padded[-1, -1, -1] == 0.0
    assert not calls


@needs_kernel
def test_field_kernel_computes_in_the_callers_fp_environment(monkeypatch):
    """With flush-to-zero on in the calling thread, the kernel's threads
    flush too: numpy's bits (potential differences that are subnormal)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    # subnormal potentials whose differences divide to normal fields: no
    # floating-point exception, so the kernel's own values come back
    phi = (rng.uniform(1.0, 4.0, size=(40, 40, 40))
           * rng.choice([-1.0, 1.0], size=(40, 40, 40)) * 1e-309)
    d = (np.float64(1e-10), np.float64(2e-10), np.float64(3e-10))
    plain = _numpy_fields(phi, d, monkeypatch)
    if not torch.set_flush_denormal(True):
        pytest.skip("no flush-to-zero control on this CPU")
    try:
        want = _numpy_fields(phi, d, monkeypatch)
        with kept_thread_setting() as threads:
            threads.set_threads(4)
            assert ps._field_kernel()(phi, *d, team_points=0) is not None
            got = ps._field_from_potential(phi, *d, np)
    finally:
        torch.set_flush_denormal(False)
    assert not all(np.array_equal(_bits(a), _bits(b)) for a, b in zip(plain, want))
    for g, w in zip(got, want):
        assert np.array_equal(_bits(g), _bits(w))


@needs_kernel
def test_floating_point_exceptions_are_numpys(monkeypatch):
    """A field that raises a floating-point exception (here a zero spacing:
    division by zero) is left to numpy, whose warnings and errors under
    numpy.errstate the caller gets as before, with the same values."""
    phi = np.random.default_rng(3).normal(size=(12, 10, 8))
    d = (np.float64(0.0), np.float64(0.2), np.float64(0.3))
    assert ps._field_kernel()(phi, *d) is None
    with pytest.raises(FloatingPointError):
        with np.errstate(all="raise"):
            ps._field_from_potential(phi, *d, np)
    with pytest.warns(RuntimeWarning):
        with np.errstate(all="warn"):
            ps._field_from_potential(phi, *d, np)
    with np.errstate(all="ignore"):
        got = ps._field_from_potential(phi, *d, np)
        want = _numpy_fields(phi, d, monkeypatch)
    for g, w in zip(got, want):
        assert np.array_equal(_bits(g), _bits(w))


def test_the_lookup_never_imports_torch():
    """Until something imports torch (on macOS the kernel needs its OpenMP
    runtime) the numpy code runs; the lookup does not import torch."""
    import subprocess
    import sys
    code = ("import sys; import numpy as np; "
            "import linac_gen.pic.poisson_solver as ps; "
            "s = ps.PoissonSolverFFT(np.full(3, -1.0), np.full(3, 1.0), "
            "np.array([8, 8, 8]), use_gpu='cpu'); s.solve(np.ones((8, 8, 8))); "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out[-1] == "False"
