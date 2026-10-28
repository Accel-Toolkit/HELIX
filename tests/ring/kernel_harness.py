"""Run a few ring nodes on a set of particles (tests only)."""
from __future__ import annotations

import numpy as np

from linac_gen.ring.kernels import cartesian as cart
from linac_gen.ring.kernels import drivers


def table(nodes):
    """(kinds, offsets, params) of [(kind, block), ...]."""
    kinds, offsets, params = [], [], []
    for kind, blk in nodes:
        kinds.append(int(kind))
        offsets.append(len(params))
        params.extend(float(v) for v in blk)
    return (np.array(kinds, dtype=np.int64), np.array(offsets, dtype=np.int64),
            np.array(params, dtype=np.float64))


def run(nodes, X, *, beta0, p0c, mass, backend="python"):
    """Track the particles X (6, n) once through ``nodes`` (no apertures) and
    return the coordinates after them.  ``backend``: "python" (the plain
    source), "numba" (the compiled serial driver)."""
    kinds, offsets, params = table(nodes)
    n = kinds.size
    X = np.array(X, dtype=np.float64, copy=True)
    m = X.shape[1]
    ptau = np.empty(m)
    rpp = np.empty(m)
    rvv = np.empty(m)
    for i in range(m):
        ptau[i], rpp[i], rvv[i] = cart.energy_from_delta(X[5, i], beta0)
    rb = np.full(n, beta0)
    rp = np.full(n, p0c)
    rg = np.full(n, (p0c / mass) ** 2)
    out = np.empty((0, 6, m))
    if backend == "python":
        drivers.track_probe(X, ptau, rpp, rvv, kinds, offsets, params, rb, rp, rg, 0, n, out, False)
    else:
        from linac_gen.ring.kernels.cpu import backend as be
        be(allow_python=False).track_probe(X, ptau, rpp, rvv, kinds, offsets, params, rb, rp, rg,
                                           0, n, out, False)
    return X
