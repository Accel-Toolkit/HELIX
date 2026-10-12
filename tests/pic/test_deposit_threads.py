"""The CIC charge deposit sums every cell in particle order.  So:

* any thread count gives the same bits (raw bits, signed zeros included);
* it equals the kernel it replaced on one thread -- what every GUI run
  computed -- through golden values that kernel wrote (2026-10-11), where
  the build matches the machine that wrote them;
* it agrees with the pure-Python deposit to rounding.
"""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pytest

# torch first: on macOS the kernel binds its OpenMP symbols to the libomp
# torch loads (tests/pic/test_openmp_single_runtime.py)
pytest.importorskip("linac_gen.pic.pic_solver")
P = pytest.importorskip("linac_gen._pic_kernels")
if not hasattr(P, "set_num_threads"):
    pytest.skip("_pic_kernels built before thread control (2026-10-11)",
                allow_module_level=True)

from linac_gen.pic.charge_deposition import deposit_cic as py_deposit  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "cic_deposit_golden.npz"
THREADS = (1, 2, 3, 4, 7, 14)


@pytest.fixture(autouse=True)
def _restore_kernel_threads():
    from linac_gen.parallel.threads import sync_kernels
    yield
    sync_kernels()


def _cases():
    rng = np.random.default_rng(7)
    g = lambda lo, hi, ng: (np.full(3, float(lo)), np.full(3, float(hi)),  # noqa: E731
                            np.array(ng, dtype=np.int32))
    yield "empty", np.zeros((0, 3)), np.zeros(0), g(-1, 1, (8, 8, 8))
    yield "one", np.array([[0.1, -0.2, 0.3]]), np.array([2.0]), g(-1, 1, (4, 5, 6))
    yield "min_grid", rng.normal(0, 0.4, (600, 3)), np.full(600, -1.0), g(-1, 1, (2, 2, 2))
    x = rng.normal(0, 1.0, (60000, 3))
    yield "gauss_neg", x, rng.uniform(-1.5, -0.5, 60000), g(-4, 4, (33, 47, 29))
    q = rng.uniform(0.5, 1.5, 60000) * np.where(rng.random(60000) < 0.5, -1, 1)
    yield "gauss_mixed", x, q, g(-4, 4, (48, 48, 48))
    x = rng.normal(0, 2.0, (5000, 3))
    x[:20] = 0.0; x[20:40, 0] = 1.0; x[40:60] = [1.0, -1.0, 1.0]
    x[60, 1] = np.nan; x[61, 2] = np.inf; x[62, 0] = -np.inf; x[63] = 1e300
    yield "edges", x, np.full(5000, 3.0), g(-1, 1, (16, 16, 16))
    x = rng.normal(0, 1.0, (8000, 3)); x[:, 0] = 0.123
    yield "one_plane", x, np.full(8000, -2.0), g(-3, 3, (24, 24, 24))
    yield "many", rng.normal(0, 1.0, (300000, 3)), np.ones(300000), g(-5, 5, (64, 32, 16))
    # an FMA that underflows leaves -0.0 in a cell; the old kernel's last
    # step (+ 0.0) made it +0.0, and so must every path here
    yield "underflow", np.array([[0.5, 0.5, 1e-320]]), np.array([-1e-12]), \
        g(0, 1, (2, 2, 2))


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.mark.parametrize("case", list(_cases()), ids=lambda c: c[0])
def test_same_bits_for_any_thread_count(case):
    _, x, q, (lo, hi, ng) = case
    P.set_num_threads(1)
    ref = P.deposit_cic(x, q, lo, hi, ng)
    for t in THREADS[1:]:
        P.set_num_threads(t)
        # a build without OpenMP runs one thread whatever it is told
        assert P.get_num_threads() == (t if P._openmp_enabled else 1)
        assert np.array_equal(_bits(P.deposit_cic(x, q, lo, hi, ng)), _bits(ref)), t


@pytest.mark.parametrize("key", ["a", "b"])
def test_equals_the_replaced_kernel_on_one_thread(key):
    g = np.load(GOLDEN)
    written_on = (str(g["system"]), str(g["machine"]))
    if (platform.system(), platform.machine()) != written_on:
        pytest.skip(f"golden values written on {written_on} (clang contracts "
                    "acc += a*b into an FMA there; another compiler or "
                    "machine rounds differently)")
    args = [g[f"{key}_{k}"] for k in ("pos", "q", "lo", "hi", "ng")]
    for t in (1, 3, 14):
        P.set_num_threads(t)
        assert np.array_equal(_bits(P.deposit_cic(*args)), _bits(g[f"{key}_rho"]))


@pytest.mark.parametrize("case", [c for c in _cases()
                                  if c[0] in ("gauss_neg", "one_plane", "many")],
                         ids=lambda c: c[0])
def test_agrees_with_the_python_deposit(case):
    _, x, q, (lo, hi, ng) = case
    P.set_num_threads(4)
    np.testing.assert_allclose(P.deposit_cic(x, q, lo, hi, ng),
                               py_deposit(x, q, lo, hi, ng),
                               rtol=1e-12, atol=1e-12 * np.abs(q).max())


def test_an_underflowing_fma_leaves_no_negative_zero():
    x = np.array([[0.5, 0.5, 1e-320]]); q = np.array([-1e-12])
    lo, hi = np.zeros(3), np.ones(3)
    for t in (1, 3):
        P.set_num_threads(t)
        rho = P.deposit_cic(x, q, lo, hi, np.array([2, 2, 2], dtype=np.int32))
        assert not np.signbit(rho[rho == 0.0]).any()


def test_malformed_input_is_refused():
    x = np.zeros((4, 3)); q = np.ones(4)
    lo, hi = np.full(3, -1.0), np.full(3, 1.0)
    ng = np.array([4, 4, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="shape"):
        P.deposit_cic(np.zeros((4, 2)), q, lo, hi, ng)
    with pytest.raises(ValueError, match="one value per particle"):
        P.deposit_cic(x, np.ones(3), lo, hi, ng)
    with pytest.raises(ValueError, match="at least 2 nodes"):
        P.deposit_cic(x, q, lo, hi, np.array([4, 1, 4], dtype=np.int32))
    with pytest.raises(ValueError, match="3 entries"):
        P.deposit_cic(x, q, lo[:2], hi, ng)
