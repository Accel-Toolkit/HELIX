#!/usr/bin/env python3
"""Ring kernels: achieved floating-point rate against the device's FP64 peak.

    python benchmarks/ring/gpu_roofline.py [--particles 58000] [--turns 5]

The work of one particle-turn is counted once, by running the kernel source
on one particle with numbers that count their own operations (an add,
subtract, multiply or divide is one operation — comparisons are not
counted; a square root, sine, cosine, ... a fixed number, OP_COST); the ring
is the simple Fermilab Booster (tests/ring/fixtures).  Then N
particles are timed on the CPU (every core) and, when one is there, on the
CUDA GPU (in short launches, after one turn that sizes them); the peak FP64
rate of the GPU comes from its streaming multiprocessors, clock and FP64
units per multiprocessor (the table below, by compute capability).

The count is of the arithmetic the source asks for, with each function at
its OP_COST: about half of the count is sines, cosines and square roots,
and a division counts one operation where a GPU spends several — so the
rate is a model of the work, not a measurement of the instructions run;
compare runs with it, not with the peak to the percent.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DECK = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
KDIR = ROOT / "linac_gen" / "ring" / "kernels"

#: floating-point operations charged per elementary function (a common
#: rough count of their FP64 instruction sequences)
OP_COST = {"sqrt": 4, "sin": 20, "cos": 20, "tan": 25, "asin": 25, "atan": 25, "atan2": 30,
           "sinh": 25, "cosh": 25, "log1p": 25, "fabs": 1, "floor": 1}
#: FP64 units per streaming multiprocessor, by compute capability (data-centre
#: parts: P100, V100, A100 32, H100 64; consumer parts 1/32 or 1/64 of their
#: FP32 units: Pascal 4, Turing / Ampere / Ada (RTX 2000 Ada: 22 SMs) /
#: Blackwell 2)
FP64_PER_SM_CLOCK = {(6, 0): 32, (7, 0): 32, (8, 0): 32, (9, 0): 64,
                     (6, 1): 4, (7, 5): 2, (8, 6): 2, (8, 9): 2, (12, 0): 2}

COUNT = [0]


class N(float):
    """A float that counts the operations done with it."""
    def _r(self, v):
        COUNT[0] += 1
        return N(v)

    def __add__(self, o): return self._r(float(self) + float(o))
    def __radd__(self, o): return self._r(float(o) + float(self))
    def __sub__(self, o): return self._r(float(self) - float(o))
    def __rsub__(self, o): return self._r(float(o) - float(self))
    def __mul__(self, o): return self._r(float(self) * float(o))
    def __rmul__(self, o): return self._r(float(o) * float(self))
    def __truediv__(self, o): return self._r(float(self) / float(o))
    def __rtruediv__(self, o): return self._r(float(o) / float(self))
    def __pow__(self, o): return self._r(float(self) ** float(o))
    def __neg__(self): return N(-float(self))
    def __abs__(self): return N(abs(float(self)))


def _fn(name):
    f = getattr(math, name)

    def g(*a):
        COUNT[0] += OP_COST.get(name, 1)
        r = f(*[float(x) for x in a])
        return N(r) if isinstance(r, float) else r
    return g


def _load(name, extra):
    src = (KDIR / f"{name}.py").read_text(encoding="utf-8")
    lines = [ln for ln in src.splitlines()
             if not ln.strip().startswith(("from math import", "from linac_gen.ring.kernels"))]
    mod = types.ModuleType(f"count_{name}")
    ns = mod.__dict__
    for nm in ("asin", "atan", "atan2", "cos", "cosh", "fabs", "log1p", "sin", "sinh", "sqrt",
               "tan", "floor"):
        ns[nm] = _fn(nm)
    ns["isfinite"] = math.isfinite
    ns["nan"] = math.nan
    ns["jitable"] = lambda f=None, **_k: f
    ns.update(extra)
    exec(compile("\n".join(lines), f"<count {name}>", "exec"), ns)
    return mod


def flops_per_particle_turn(rl) -> int:
    cart = _load("cartesian", {})
    curv = _load("curvilinear", {})
    nd = _load("nodes", {"_cart": cart, "_curv": curv})
    ends = list(rl.offsets[1:]) + [rl.params.size]
    blocks = [(int(rl.kinds[j]), [N(float(v)) for v in rl.params[rl.offsets[j]:ends[j]]])
              for j in range(rl.n_nodes)]
    b0, p0c, bg2 = N(float(rl.reference.beta0)), N(float(rl.reference.p0c)), \
        N(float((rl.reference.beta0 * rl.reference.gamma0) ** 2))
    x, px, y, py, zeta, delta = (N(v) for v in (1e-3, 1e-5, -5e-4, 2e-5, 0.1, 2e-4))
    ptau, rpp, rvv = cart.energy_from_delta(delta, b0)
    COUNT[0] = 0
    for kind, p in blocks:
        (x, px, y, py, zeta, delta, ptau, rpp, rvv, _l) = nd.track_node(
            kind, p, 0, b0, p0c, bg2, x, px, y, py, zeta, delta, ptau, rpp, rvv)
    return COUNT[0]


def _device_arrays(X, rl):
    """The particle arrays of the kernels for X (6, n)."""
    from linac_gen.ring.optics import _energy
    pt, rp, rv = (np.ascontiguousarray(v) for v in _energy(X[5], rl.reference.beta0))
    n = X.shape[1]
    return [np.array(X), pt, rp, rv, np.ones(n, np.int8), np.full(n, -1, np.int32),
            np.full(n, -1, np.int32), np.zeros(n, np.int8)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, default=58000)
    ap.add_argument("--turns", type=int, default=5)
    a = ap.parse_args(argv)
    from linac_gen.cli.common import load_lattice
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.engine import wrap_length
    from linac_gen.ring.kernels import cuda as kc
    from linac_gen.ring.kernels.cpu import PARALLEL_LOCK, backend
    lat = load_lattice(str(DECK))
    rl = compile_from_config(copy.deepcopy(lat), RingConfig())
    fl = flops_per_particle_turn(rl)
    out = {"flop_per_particle_turn": fl, "particles": a.particles, "turns": a.turns}
    L = wrap_length(rl) or 0.0
    rng = np.random.default_rng(1)
    X0 = np.ascontiguousarray(rng.normal(0, 1, (6, a.particles)) * np.array(
        [[2e-3], [6e-5], [1e-3], [2e-4], [0.2], [3e-4]]))

    def run(fn, X):
        fn(*_device_arrays(X, rl), *rl.node_table(), 0, rl.n_nodes, a.turns, 0, L)

    be = backend(allow_python=False)
    with PARALLEL_LOCK:
        run(be.track_turns_parallel, X0[:, :64])
        t = time.perf_counter()
        run(be.track_turns_parallel, X0)
        dt = time.perf_counter() - t
    out["cpu_gflops"] = fl * a.particles * a.turns / dt * 1e-9
    out["cpu_s_per_turn"] = dt / a.turns
    if kc.available():
        import numba
        from numba import cuda
        kc.kernel()
        dev = [cuda.to_device(np.ascontiguousarray(v)) for v in _device_arrays(X0, rl)]
        table = [cuda.to_device(np.ascontiguousarray(v)) for v in rl.node_table()]
        ln = kc.Launcher(dev, table, a.particles, 0, rl.n_nodes, L)
        ln.track(0, 1)                      # (compiles, and sizes the launches)
        t = time.perf_counter()
        ln.track(1, a.turns)
        dt = time.perf_counter() - t
        out["cuda_gflops"] = fl * a.particles * a.turns / dt * 1e-9
        out["cuda_s_per_turn"] = dt / a.turns
        out["cuda_launches"] = ln.launches
        out["cuda_longest_launch_s"] = round(ln.longest_s, 4)
        if getattr(numba.config, "ENABLE_CUDASIM", 0):
            out["device"] = "numba CUDA simulator"
        else:
            d = cuda.get_current_device()
            cc = tuple(d.compute_capability)
            name = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
            out.update({"device": name, "compute_capability": list(cc)})
            per = FP64_PER_SM_CLOCK.get(cc)
            if per:
                sms = d.MULTIPROCESSOR_COUNT
                clock_ghz = d.CLOCK_RATE * 1e-6
                peak = sms * clock_ghz * per * 2.0          # (a fused multiply-add is 2)
                out.update({"fp64_peak_gflops": peak,
                            "fraction_of_fp64_peak": out["cuda_gflops"] / peak})
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
