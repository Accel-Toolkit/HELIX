"""The C++ TSC twin (``linac_gen._tsc_kernels``) must equal the Python TSC
kernels BIT FOR BIT (``np.array_equal``): same expressions, the deposit summed
in ``np.add.at`` order, no FMA contraction.  That contract is what lets HELIX
use it with every existing result unchanged.  Every CI build job (ubuntu,
macOS, Windows, where the workflow runs them) runs this file; with
``LINAC_GEN_REQUIRE_CPP=1`` a missing module is a failure, not a skip.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from linac_gen.pic import pic_solver as ps            # imports torch first (one libomp)
from linac_gen.pic.charge_deposition import deposit_tsc as py_dep
from linac_gen.pic.field_interpolation import interpolate_tsc as py_gat

try:
    from linac_gen import _tsc_kernels as K
    HAS = True
except ImportError:                                    # pragma: no cover - depends on build
    K = None
    HAS = False

needs = pytest.mark.skipif(not HAS, reason="C++ TSC kernel not built")


def test_module_built_when_required():
    if os.environ.get("LINAC_GEN_REQUIRE_CPP") == "1":
        assert HAS, "LINAC_GEN_REQUIRE_CPP=1 but linac_gen._tsc_kernels is not importable"


def _beam(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)) * [1.5, 1.2, 2.8], rng


def _cases():
    pos, rng = _beam(50_000)
    q = np.full(len(pos), 3.1e-16)
    off = pos.copy()
    off[:200] *= 8.0
    off[200:210] = [[1e3, -1e3, 0.0]] * 10
    box = (np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0]))
    return {
        "gaussian_48": (pos, q, (48, 48, 48), None),
        "beyond_every_face": (off, q, (48, 48, 48), None),
        "anisotropic_40_48_64": (pos, q, (40, 48, 64), None),
        "smallest_3": (pos[:2000], q[:2000], (3, 3, 3), None),
        "mixed_4_3_5": (pos[:2000], q[:2000], (4, 3, 5), None),
        "weighted_charges": (pos, q * rng.uniform(0.0, 1.0, len(pos)), (48, 48, 48), None),
        "single_particle": (pos[:1], q[:1], (8, 8, 8), box),
        "on_nodes": (np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]]), q[:2], (5, 5, 5), box),
    }


def _grid(pos, n, box):
    if box is not None:
        return box[0], box[1]
    return pos.mean(0) - 5.0 * pos.std(0), pos.mean(0) + 5.0 * pos.std(0)


@needs
@pytest.mark.parametrize("name", list(_cases()))
def test_bit_identical_to_python(name):
    pos, q, n, box = _cases()[name]
    n = np.asarray(n, dtype=np.int64)
    gmin, gmax = _grid(pos, n, box)
    assert np.array_equal(K.deposit_tsc(pos, q, gmin, gmax, n), py_dep(pos, q, gmin, gmax, n))
    rng = np.random.default_rng(11)
    E = [rng.standard_normal(tuple(n)) for _ in range(3)]
    assert np.array_equal(K.interpolate_tsc(*E, pos, gmin, gmax, n), py_gat(*E, pos, gmin, gmax, n))


@needs
def test_repeated_calls_identical():
    pos, q, n, _ = _cases()["gaussian_48"]
    n = np.asarray(n, dtype=np.int64)
    gmin, gmax = _grid(pos, n, None)
    first = K.deposit_tsc(pos, q, gmin, gmax, n)
    for _ in range(3):
        assert np.array_equal(K.deposit_tsc(pos, q, gmin, gmax, n), first)


_THREAD_SCRIPT = r"""
import sys, numpy as np
from linac_gen.pic import pic_solver            # torch first, then the kernel
from linac_gen import _tsc_kernels as K
rng = np.random.default_rng(5)
pos = rng.standard_normal((40000, 3)) * [1.5, 1.2, 2.8]
q = np.full(len(pos), 1e-15); n = np.array([32, 32, 32], dtype=np.int64)
gmin, gmax = pos.mean(0) - 5 * pos.std(0), pos.mean(0) + 5 * pos.std(0)
E = [rng.standard_normal((32, 32, 32)) for _ in range(3)]
np.savez(sys.argv[1], rho=K.deposit_tsc(pos, q, gmin, gmax, n),
         e=K.interpolate_tsc(*E, pos, gmin, gmax, n))
"""


@needs
def test_independent_of_thread_count(tmp_path):
    if not K._openmp_enabled:
        pytest.skip("single-threaded build: the thread count cannot change anything")
    outs = []
    for t in ("1", "3"):
        f = tmp_path / f"t{t}.npz"
        env = dict(os.environ, OMP_NUM_THREADS=t)
        r = subprocess.run([sys.executable, "-c", _THREAD_SCRIPT, str(f)], env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(np.load(f))
    assert np.array_equal(outs[0]["rho"], outs[1]["rho"])
    assert np.array_equal(outs[0]["e"], outs[1]["e"])


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_dispatch_keeps_python_semantics_outside_the_twin():
    """n < 3 (numpy's index wrap) and non-finite positions stay on Python."""
    pos, _rng = _beam(500)
    q = np.full(len(pos), 1e-15)
    n2 = np.array([2, 4, 4], dtype=np.int64)
    gmin, gmax = _grid(pos, n2, None)
    assert np.array_equal(ps.deposit_tsc(pos, q, gmin, gmax, n2), py_dep(pos, q, gmin, gmax, n2))
    bad = pos.copy()
    bad[3, 1] = np.nan
    n = np.array([8, 8, 8], dtype=np.int64)
    a = ps.deposit_tsc(bad, q, gmin, gmax, n)
    b = py_dep(bad, q, gmin, gmax, n)
    assert np.array_equal(a, b, equal_nan=True)
    assert ps.deposit_tsc(pos[:0], q[:0], gmin, gmax, n).shape == (8, 8, 8)


def test_kill_switch_forces_python():
    code = ("from linac_gen.pic import pic_solver as ps\n"
            "assert ps._USE_CPP_TSC is False\n"
            "assert ps.deposit_tsc is ps._py_deposit_tsc\n"
            "assert ps.interpolate_tsc is ps._py_interpolate_tsc\n"
            "print('python-tsc')")
    env = dict(os.environ, LINAC_GEN_TSC_KERNEL="0")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0 and "python-tsc" in r.stdout, r.stderr[-2000:]


@needs
def test_picsolver_kick_identical_to_python_kernels(monkeypatch):
    """End to end through PicSolver.kick: the C++ twin and the Python
    kernels give the same particles bit for bit."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle

    assert ps._USE_CPP_TSC
    parts = np.random.default_rng(3).standard_normal((20_000, 6)) * [1.5, 1.5, 1.2, 1.5, 8.0, 0.01]

    def kick_once():
        b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5),
                 n_particles=len(parts), current=5.0)
        b.bunch_frequency = 162.5
        b.particles[:] = parts
        ps.PicSolver(SpaceChargeConfig(nx=24, ny=24, nz=24, kernel="tsc", use_gpu="cpu")).kick(b, 2.0)
        return b.particles.copy()

    served = []                                          # the twin must really run

    def counted(fn, tag):
        def wrapper(*a):
            r = fn(*a)
            served.append((tag, r is not None))
            return r
        return wrapper

    monkeypatch.setattr(ps, "_cpp_deposit_tsc", counted(ps._cpp_deposit_tsc, "deposit"))
    monkeypatch.setattr(ps, "_cpp_interpolate_tsc",
                        counted(ps._cpp_interpolate_tsc, "gather"))
    with_cpp = kick_once()
    assert served == [("deposit", True), ("gather", True)]
    monkeypatch.setattr(ps, "deposit_tsc", ps._py_deposit_tsc)
    monkeypatch.setattr(ps, "interpolate_tsc", ps._py_interpolate_tsc)
    with_py = kick_once()
    assert not np.array_equal(with_cpp, parts)          # the kick did something
    assert np.array_equal(with_cpp, with_py)



def _dispatch_vs_python(pos, q, gmin, gmax, n):
    a = ps.deposit_tsc(pos, q, gmin, gmax, n)
    b = py_dep(pos, q, gmin, gmax, n)
    assert np.array_equal(a, b, equal_nan=True)
    rng = np.random.default_rng(1)
    E = [rng.standard_normal(tuple(int(v) for v in n)) for _ in range(3)]
    assert np.array_equal(ps.interpolate_tsc(*E, pos, gmin, gmax, n),
                          py_gat(*E, pos, gmin, gmax, n), equal_nan=True)
    return E


def _edge_case(case):
    pos, _rng = _beam(400)
    q = np.full(len(pos), 1e-15)
    n = np.array([8, 8, 8], dtype=np.int64)
    gmin, gmax = _grid(pos, n, None)
    pos, gmin, gmax = pos.copy(), gmin.copy(), gmax.copy()
    if case == "float32":
        pos, gmin, gmax = (a.astype(np.float32) for a in (pos, gmin, gmax))
    elif case == "zero_spacing":
        gmax[1] = gmin[1]
    elif case == "nan_bound":
        gmin[2] = np.nan
    elif case == "inf_bound":
        gmax[0] = np.inf
    elif case == "nan_position":
        pos[3, 1] = np.nan
    elif case == "inf_position":
        pos[9, 2] = -np.inf
    elif case == "far_particle":
        pos[5, 0] = 1e300
    return pos, q, gmin, gmax, n


EDGE = ["float32", "zero_spacing", "nan_bound", "inf_bound", "nan_position",
        "inf_position", "far_particle"]
DECLINED = ["zero_spacing", "nan_bound", "nan_position", "inf_position", "far_particle"]


def test_dispatch_admits_picsolver_inputs():
    pos, _rng = _beam(1000)
    n = np.array([48, 48, 48], dtype=np.int64)
    gmin, gmax = _grid(pos, n, None)
    assert ps._cpp_tsc_ok(pos, gmin, gmax, n)
    assert ps._cpp_tsc_ok(pos, gmin, gmax, n.astype(np.int32))
    assert not ps._cpp_tsc_ok(pos.astype(np.float32), gmin, gmax, n)
    assert not ps._cpp_tsc_ok(pos, gmin, gmax, np.array([48, 2, 48]))
    assert not ps._cpp_tsc_ok(pos, gmin, gmax, np.array([48.5, 48.0, 48.0]))
    assert not ps._cpp_tsc_ok(pos, gmin, gmax, [48, 48, 48])


def test_non_integral_grid_size_keeps_python_result():
    """pybind11 would truncate 8.5 to 8 (spacing L/7, not Python's L/7.5)."""
    rng = np.random.default_rng(6)
    pos = rng.standard_normal((500, 3))
    q = np.full(len(pos), 1e-15)
    gmin, gmax = np.full(3, -4.0), np.full(3, 4.0)
    n = np.array([8.5, 8.0, 8.0])
    assert np.array_equal(ps.deposit_tsc(pos, q, gmin, gmax, n), py_dep(pos, q, gmin, gmax, n))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("case", EDGE)
def test_dispatch_equals_python_on_edge_inputs(case):
    """Whatever the twin does with an edge input -- decline it, or compute
    it with the same arithmetic -- the dispatcher returns Python's result."""
    _dispatch_vs_python(*_edge_case(case))


@needs
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("case", DECLINED)
def test_twin_declines_outside_the_envelope(case):
    pos, q, gmin, gmax, n = _edge_case(case)
    assert K.deposit_tsc(pos, q, gmin, gmax, n) is None
    E = [np.zeros(tuple(n)) for _ in range(3)]
    assert K.interpolate_tsc(*E, pos, gmin, gmax, n) is None


@needs
def test_twin_computes_infinite_spacing_like_python():
    """An infinite upper bound maps every particle to s = 0: inside the
    envelope, and computed with Python's arithmetic."""
    pos, q, gmin, gmax, n = _edge_case("inf_bound")
    assert np.array_equal(K.deposit_tsc(pos, q, gmin, gmax, n), py_dep(pos, q, gmin, gmax, n))


def test_gather_with_oversized_field_arrays_stays_on_python():
    pos, _rng = _beam(300)
    n = np.array([8, 8, 8], dtype=np.int64)
    gmin, gmax = _grid(pos, n, None)
    E = [np.random.default_rng(2).standard_normal((9, 8, 8)) for _ in range(3)]
    assert np.array_equal(ps.interpolate_tsc(*E, pos, gmin, gmax, n),
                          py_gat(*E, pos, gmin, gmax, n))


@needs
def test_direct_call_validates_shapes():
    pos, _rng = _beam(100)
    q = np.full(len(pos), 1e-15)
    n = np.array([8, 8, 8], dtype=np.int64)
    gmin, gmax = _grid(pos, n, None)
    E = [np.zeros((8, 8, 8))] * 3
    with pytest.raises(ValueError):
        K.deposit_tsc(pos[:, :2], q, gmin, gmax, n)
    with pytest.raises(ValueError):
        K.deposit_tsc(pos, q[:-1], gmin, gmax, n)
    with pytest.raises(ValueError):
        K.deposit_tsc(pos, q, gmin[:2], gmax, n)
    with pytest.raises(ValueError):
        K.deposit_tsc(pos, q, gmin, gmax, np.array([8, 2, 8], dtype=np.int64))
    with pytest.raises(ValueError):
        K.interpolate_tsc(np.zeros((8, 8, 7)), E[1], E[2], pos, gmin, gmax, n)
    with pytest.raises(ValueError):
        K.interpolate_tsc(*E, pos[:, :2], gmin, gmax, n)
