"""The ring kernels give the same answer on every backend: plain Python, the
compiled serial and multi-core CPU drivers (bit for bit), and the CUDA kernel
(in numba's CUDA simulator on a machine without a GPU, run in a separate
process because the simulator must be switched on before numba loads)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring.compile import blocks as B
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.kernels import drivers
from linac_gen.ring.kernels import nodes as N
from linac_gen.ring.optics import _energy
from linac_gen.ring.reference import RingReference

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"


@pytest.fixture(scope="module")
def booster():
    lat, meta = parse_madx(str(FIXTURE))
    w = meta["reference"].w_kin
    ref = RingReference.from_kinetic("proton", w)
    return {m: compile_ring(lat, ref, lattice_energy=w, model=m) for m in ("cartesian", "curvilinear")}


def _particles(n, seed=7):
    rng = np.random.default_rng(seed)
    X = np.zeros((6, n))
    X[0] = rng.normal(0, 3e-3, n)
    X[1] = rng.normal(0, 1e-4, n)
    X[2] = rng.normal(0, 2e-3, n)
    X[3] = rng.normal(0, 3e-4, n)
    X[4] = rng.normal(0, 0.5, n)
    X[5] = rng.normal(0, 1e-3, n)
    return X


def _run(fn, rl, X, n_turns):
    X = np.array(X, copy=True)
    n = X.shape[1]
    ptau, rpp, rvv = (np.ascontiguousarray(a) for a in _energy(X[5], rl.reference.beta0))
    state = np.ones(n, dtype=np.int8)
    lt = np.full(n, -1, dtype=np.int32)
    ln = np.full(n, -1, dtype=np.int32)
    lc = np.zeros(n, dtype=np.int8)
    fn(X, ptau, rpp, rvv, state, lt, ln, lc, *rl.node_table(), 0, rl.n_nodes, n_turns, 0, 0.0)
    return X, state, lt, ln, lc


@pytest.mark.parametrize("model", ["cartesian", "curvilinear"])
def test_serial_parallel_and_python_kernels_agree_bit_for_bit(booster, model):
    pytest.importorskip("numba")
    from linac_gen.ring.kernels.cpu import PARALLEL_LOCK, backend
    be = backend(allow_python=False)
    rl = booster[model]
    X = _particles(256)
    a = _run(be.track_turns, rl, X, 3)
    with PARALLEL_LOCK:
        b = _run(be.track_turns_parallel, rl, X, 3)
    for u, v in zip(a, b, strict=True):
        assert np.array_equal(u, v)
    assert np.all(a[1] == 1)
    # the plain-Python source on a few particles, one turn
    c = _run(drivers.track_turns, rl, X[:, :4], 1)
    d = _run(be.track_turns, rl, X[:, :4], 1)
    for u, v in zip(c, d, strict=True):
        assert np.array_equal(u, v)


def test_losses_are_recorded_at_the_aperture(booster):
    pytest.importorskip("numba")
    import dataclasses

    from linac_gen.ring.kernels.cpu import backend
    rl = booster["cartesian"]
    # an aperture node of 2 mm radius after node 10
    j = 10
    kinds = np.insert(rl.kinds, j + 1, N.N_APERTURE)
    blk = B.aperture("circle", 2e-3)
    params = np.concatenate([rl.params, blk])
    offsets = np.insert(rl.offsets, j + 1, rl.params.size)
    rla = dataclasses.replace(rl, kinds=kinds, offsets=offsets, params=params,
                              ref_beta0=np.insert(rl.ref_beta0, j + 1, rl.ref_beta0[0]),
                              ref_p0c=np.insert(rl.ref_p0c, j + 1, rl.ref_p0c[0]),
                              ref_bg2=np.insert(rl.ref_bg2, j + 1, rl.ref_bg2[0]))
    X = _particles(64)
    Xo, state, lt, ln, lc = _run(backend(allow_python=False).track_turns, rla, X, 2)
    lost = state == 0
    assert lost.any() and not lost.all()
    assert set(ln[lost]) == {j + 1} and set(lc[lost]) == {N.LOSS_APERTURE}
    assert set(lt[lost]) <= {0, 1} and 0 in set(lt[lost])
    # the lost particles stopped at the aperture: outside it, and at its entrance
    r = np.hypot(Xo[0, lost], Xo[2, lost])
    assert np.all(r > 2e-3)
    assert np.all(ln[~lost] == -1) and np.all(lt[~lost] == -1)


#: two valid particles, one whose first map is not defined for it (|px| > 1 + delta)
#: and one that leaves the maps' domain later in the turn
_INVALID = np.array([[1e-3, -2e-3, 0.0, 2.0], [1e-5, 0.0, 1.5, 0.3], [0.0, 1e-3, 0.0, 0.0],
                     [0.0, -2e-5, 0.0, 0.0], [0.1, -0.2, 0.0, 0.0], [1e-4, -1e-3, 0.0, 0.0]])


@pytest.mark.parametrize("model", ["cartesian", "curvilinear"])
def test_a_particle_outside_the_maps_domain_is_lost_as_invalid(booster, model):
    pytest.importorskip("numba")
    from linac_gen.ring.kernels.cpu import PARALLEL_LOCK, backend
    rl = booster[model]
    be = backend(allow_python=False)
    X, state, lt, ln, lc = _run(be.track_turns, rl, _INVALID, 1)
    assert state.tolist() == [1, 1, 0, 0]
    assert lc[2] == lc[3] == N.LOSS_INVALID and lt[2] == lt[3] == 0
    assert ln[2] == 0 and ln[3] > 0
    # (kept at the entrance of the node where its map failed: finite)
    assert np.all(np.isfinite(X)) and X[1, 2] == 1.5
    with PARALLEL_LOCK:
        par = _run(be.track_turns_parallel, rl, _INVALID, 1)
    for u, v in zip((X, state, lt, ln, lc), par, strict=True):
        assert np.array_equal(u, v)


def test_without_numba_the_python_kernels_lose_the_same_particles(booster, tmp_path):
    # ring mode's plain-Python fallback (numba not installed): the math
    # module raises where the compiled kernels give NaN, so the fallback
    # uses functions that give NaN too — the same particles are lost at the
    # same nodes, and the others tracked bit for bit as compiled
    pytest.importorskip("numba")
    from linac_gen.ring.kernels.cpu import backend
    out = tmp_path / "py.npz"
    code = f"""
import sys
sys.modules["numba"] = None                  # numba not installed
import numpy as np
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring import compile as rc
from linac_gen.ring.kernels import _math, cpu
from linac_gen.ring.optics import _energy
from linac_gen.ring.reference import RingReference
assert not _math._COMPILED and not cpu.numba_available()
be = cpu.backend()
assert not be.compiled
lat, meta = parse_madx({str(FIXTURE)!r})
w = meta["reference"].w_kin
ref = RingReference.from_kinetic("proton", w)
res = {{}}
for model in ("cartesian", "curvilinear"):
    rl = rc.compile_ring(lat, ref, lattice_energy=w, model=model)
    X = np.array({_INVALID.tolist()!r})
    ptau, rpp, rvv = (np.ascontiguousarray(a) for a in _energy(X[5], ref.beta0))
    st = np.ones(4, np.int8); lt = np.full(4, -1, np.int32)
    ln = np.full(4, -1, np.int32); lc = np.zeros(4, np.int8)
    be.track_turns(X, ptau, rpp, rvv, st, lt, ln, lc, *rl.node_table(), 0, rl.n_nodes, 1, 0, 0.0)
    for k, v in zip(("X", "state", "lt", "ln", "lc"), (X, st, lt, ln, lc)):
        res[model + "_" + k] = v
np.savez({str(out)!r}, **res)
print("PYTHON_FALLBACK_OK", flush=True)
"""
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"})
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=900)
    assert "PYTHON_FALLBACK_OK" in r.stdout, (r.returncode, r.stderr[-3000:])
    got = np.load(out)
    for model in ("cartesian", "curvilinear"):
        want = _run(backend(allow_python=False).track_turns, booster[model], _INVALID, 1)
        for k, v in zip(("X", "state", "lt", "ln", "lc"), want, strict=True):
            assert np.array_equal(got[f"{model}_{k}"], v), (model, k)


def test_cuda_kernel_in_the_simulator_matches_the_cpu():
    pytest.importorskip("numba")
    code = r"""
import numpy as np
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.kernels import cuda as kc
from linac_gen.ring.kernels.cpu import backend
from linac_gen.ring.optics import _energy
from linac_gen.ring.reference import RingReference
lat, meta = parse_madx(%r)
w = meta["reference"].w_kin
ref = RingReference.from_kinetic("proton", w)
worst = 0.0
for model in ("cartesian", "curvilinear"):
    rl = compile_ring(lat, ref, lattice_energy=w, model=model)
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1e-3, (6, 4)); X[4] *= 300; X[5] *= 1.0
    outs = []
    for fn in (backend(allow_python=False).track_turns, kc.track_turns):
        Y = X.copy()
        ptau, rpp, rvv = (np.ascontiguousarray(a) for a in _energy(Y[5], ref.beta0))
        st = np.ones(4, dtype=np.int8); lt = np.full(4, -1, dtype=np.int32)
        ln = np.full(4, -1, dtype=np.int32); lc = np.zeros(4, dtype=np.int8)
        fn(Y, ptau, rpp, rvv, st, lt, ln, lc, *rl.node_table(), 0, rl.n_nodes, 1, 0, 0.0)
        outs.append(Y)
    d = np.abs(outs[0] - outs[1]) / np.maximum(np.abs(outs[0]), 1e-300)
    worst = max(worst, float(np.nanmax(np.where(outs[0] == 0, 0.0, d))))
print("CUDASIM_REL_DIFF", worst, flush=True)
""".replace("%r", repr(str(FIXTURE)))
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(ROOT)})
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=900)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("CUDASIM_REL_DIFF")]
    assert lines, (out.returncode, out.stderr[-3000:])
    assert float(lines[-1].split()[1]) <= 1e-15


@pytest.mark.parametrize("rate, n, k", [(1e-6, 100_000, 7), (1e-4, 100_000, 2), (2e-9, 5000, 900)])
def test_cuda_launches_stay_short(monkeypatch, rate, n, k):
    # the Launcher's sizes, against a fake GPU taking ``rate`` seconds per
    # particle-turn: after the first warp no launch is longer than
    # MAX_LAUNCH_S (Windows stops a display GPU's kernel after 2 s), and
    # every particle makes every turn, in order
    from linac_gen.ring.kernels import cuda as kc
    clock = [0.0]
    turns = np.zeros(n, dtype=np.int64)
    log = []

    class _Kernel:
        def __getitem__(self, _cfg):
            def launch(*a):
                kk, turn, i0, i1 = a[16], a[17], a[19], a[20]
                assert np.all(turns[i0:i1] == turn)
                turns[i0:i1] += kk
                clock[0] += rate * (i1 - i0) * kk
                log.append((i1 - i0) * kk * rate)
            return launch

    class _Cuda:
        @staticmethod
        def synchronize():
            pass

    monkeypatch.setattr(kc, "kernel", lambda: _Kernel())
    monkeypatch.setattr(kc, "cuda", _Cuda)
    monkeypatch.setattr(kc.time, "perf_counter", lambda: clock[0])
    ln = kc.Launcher([None] * 8, [None] * 6, n, 0, 10, 0.0)
    ln.track(0, k)
    assert np.all(turns == k)
    assert max(log[1:] or [0.0]) <= kc.MAX_LAUNCH_S * (1 + 1e-9)
    assert log[0] == pytest.approx(kc.FIRST_BATCH * rate)
    # (and not needlessly short: the launches after the first sweep fill it)
    assert len(log) <= 3 + int(np.ceil(rate * n * k / kc.MAX_LAUNCH_S)) * 2


def test_cuda_batching_changes_no_bit():
    pytest.importorskip("numba")
    code = r"""
import numpy as np
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.kernels import cuda as kc
from linac_gen.ring.optics import _energy
from linac_gen.ring.reference import RingReference
ref = RingReference.from_kinetic("proton", 2000.0)
brho = ref.p0c / 299.792458
lat = Lattice()
for e in (Drift(name="d1", length=250.0), Quadrupole(name="q1", length=1000.0, gradient=brho),
          Drift(name="d2", length=500.0), Quadrupole(name="q2", length=1000.0, gradient=-brho),
          Drift(name="d3", length=250.0)):
    lat.add(e)
rl = compile_ring(lat, ref, lattice_energy=2000.0)
rng = np.random.default_rng(2)
n = 300
X = rng.normal(0, 1e-3, (6, n))
out = []
for budget in (kc.MAX_LAUNCH_S, 0.0):
    kc.kernel()
    host = [X.copy(), *(np.ascontiguousarray(a) for a in _energy(X[5].copy(), ref.beta0)),
            np.ones(n, np.int8), np.full(n, -1, np.int32), np.full(n, -1, np.int32),
            np.zeros(n, np.int8)]
    dev = [kc.cuda.to_device(np.ascontiguousarray(a)) for a in host]
    table = [kc.cuda.to_device(np.ascontiguousarray(a)) for a in rl.node_table()]
    ln = kc.Launcher(dev, table, n, 0, rl.n_nodes, 0.0, max_launch_s=budget)
    ln.track(0, 3)
    out.append((dev[0].copy_to_host(), ln.launches))
same = np.array_equal(out[0][0], out[1][0])
print("CUDASIM_BATCH", int(same), out[0][1], out[1][1], flush=True)
"""
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(ROOT)})
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDASIM_BATCH")]
    assert line, (r.returncode, r.stderr[-3000:])
    same, few, many = (int(v) for v in line[-1].split()[1:])
    # (a zero budget: a block of particles and a turn per launch)
    assert same == 1 and many > few and many >= 3 * (300 // 128)
