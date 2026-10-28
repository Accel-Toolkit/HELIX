"""The ring kernels on a CUDA GPU (numba.cuda), from the same source as the
CPU drivers: the kernel below calls :func:`.drivers.advance_particle`, one
thread per particle.

On a machine without a GPU, numba's CUDA simulator (``NUMBA_ENABLE_CUDASIM=1``
set before numba is imported) runs these kernels in Python — the tests use it
to check the CUDA path on the Mac.  Float64 throughout; the GPU compiler fuses
multiply-adds and the GPU's sine, cosine, ... are not the CPU's, so results
agree with the CPU's to rounding — about 1e-15 relative in one element, more
after many: not bit for bit.

Every launch is kept short (:data:`MAX_LAUNCH_S`): a GPU that also drives a
display is stopped by Windows when one kernel runs past its timeout (two
seconds by default, the registry's TdrDelay).  :class:`Launcher` runs k
turns as launches over batches of particles and of turns, waits for each and
sizes the next from the time it took; the particles do not interact, so the
batching changes no bit.
"""
from __future__ import annotations

import time

import numpy as np

cuda = None              # numba.cuda, bound on first use (a module global: the simulator swaps it)
_KERNEL = None
THREADS_PER_BLOCK = 128
#: the longest one launch of the turn kernel should take [s]
MAX_LAUNCH_S = 0.25
#: particles in the first launch of a Launcher, before its rate is known
#: (one warp: on a GPU with few FP64 units a warp already fills them)
FIRST_BATCH = 32


def _turns_kernel(coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause,
                  kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1, n_turns, turn0,
                  zeta_wrap, i0, i1):
    i = i0 + cuda.grid(1)
    if i < i1:
        _advance(i, coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause,
                 kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1, n_turns, turn0,
                 zeta_wrap)


_advance = None


def available() -> bool:
    """True when numba can reach a CUDA device (or its simulator is on)."""
    try:
        from numba import cuda as _c
        return bool(_c.is_available())
    except Exception:            # numba missing, no driver, ...
        return False


def kernel():
    """The compiled CUDA turn kernel (built once)."""
    global cuda, _KERNEL, _advance
    if _KERNEL is None:
        from numba import cuda as _cuda

        from linac_gen.ring.kernels.drivers import advance_particle
        # (a launch of fewer than 128 blocks — a small beam, the foil's few
        # particles — is no fault: numba's occupancy warning switched off; it
        # reads the setting at every launch, where a warnings filter would be
        # undone by an enclosing catch_warnings)
        try:
            from numba.core import config as _config
            _config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
        except ImportError:                      # pragma: no cover
            pass
        cuda = _cuda
        _advance = advance_particle
        _KERNEL = cuda.jit(_turns_kernel)
    return _KERNEL


class Launcher:
    """Turns of the particles held on the device (``dev``: coords, ptau, rpp,
    rvv, state, lost_turn, lost_node, lost_cause as device arrays) through the
    nodes j0..j1 of ``table`` (the ring's node table on the device), in
    launches of at most about :data:`MAX_LAUNCH_S` each.

    ``track(turn, k)`` returns when the k turns are done.  The first launch
    is one warp for one turn; each later one is sized from the time per
    particle-turn the last one took: all the particles for as many turns as
    fit, or — when one turn of all of them takes longer — one turn of as many
    particles as fit (whole blocks).  Every particle makes the same turns in
    each sweep.
    """

    def __init__(self, dev, table, n: int, j0: int, j1: int, zeta_wrap: float,
                 max_launch_s: float = MAX_LAUNCH_S):
        self.k = kernel()
        self.dev = list(dev)
        self.table = list(table)
        self.n = int(n)
        self.j0, self.j1 = int(j0), int(j1)
        self.zwrap = float(zeta_wrap)
        self.max_launch_s = float(max_launch_s)
        self.rate = None                     # seconds per particle-turn, last launch
        self.launches = 0
        self.longest_s = 0.0

    def _launch(self, i0: int, i1: int, kk: int, turn: int) -> None:
        blocks = (i1 - i0 + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        t = time.perf_counter()
        self.k[blocks, THREADS_PER_BLOCK](*self.dev, *self.table, self.j0, self.j1, int(kk),
                                          int(turn), self.zwrap, int(i0), int(i1))
        cuda.synchronize()
        dt = time.perf_counter() - t
        self.rate = max(dt, 1e-9) / ((i1 - i0) * kk)
        self.launches += 1
        self.longest_s = max(self.longest_s, dt)

    def track(self, turn: int, k: int) -> None:
        n, done = self.n, 0
        if n == 0:
            return
        while done < k:
            if self.rate is None:
                kk = 1
            else:
                kk = int(min(k - done, max(1, self.max_launch_s / (self.rate * n))))
            i0 = 0
            while i0 < n:
                if self.rate is None:
                    b = FIRST_BATCH
                else:
                    fit = int(self.max_launch_s / (self.rate * kk))
                    b = max(THREADS_PER_BLOCK, fit // THREADS_PER_BLOCK * THREADS_PER_BLOCK)
                i1 = min(n, i0 + b)
                self._launch(i0, i1, kk, turn + done)
                i0 = i1
            done += kk


# -- the foil's step on the device ----------------------------------------------
# A run with a foil finds on the GPU the circulating particles inside the
# foil's rectangle (the comparisons of injection.foil.RingFoil.inside, an open side
# none); the host gets their slots (one byte a slot), then those particles
# alone, sends them back through the foil and scatters them into place.
_FOIL = None


def _foil_mask_kernel(coords, state, n, hx0, x0, hx1, x1, hy0, y0, hy1, y1, mask):
    i = cuda.grid(1)
    if i < n:
        ok = state[i] == 1
        if ok:
            x = coords[0, i]
            y = coords[2, i]
            if hx0 and not (x >= x0):
                ok = False
            if hx1 and not (x <= x1):
                ok = False
            if hy0 and not (y >= y0):
                ok = False
            if hy1 and not (y <= y1):
                ok = False
        mask[i] = 1 if ok else 0


def _gather_kernel(coords, idx, k, out):
    j = cuda.grid(1)
    if j < k:
        i = idx[j]
        for r in range(6):
            out[r, j] = coords[r, i]


def _scatter_kernel(coords, ptau, rpp, rvv, idx, k, X, pt, rp, rv):
    j = cuda.grid(1)
    if j < k:
        i = idx[j]
        for r in range(6):
            coords[r, i] = X[r, j]
        ptau[i] = pt[j]
        rpp[i] = rp[j]
        rvv[i] = rv[j]


def _put_kernel(coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause, a0, k,
                X, pt, rp, rv, st, lt, ln, lc):
    j = cuda.grid(1)
    if j < k:
        i = a0 + j
        for r in range(6):
            coords[r, i] = X[r, j]
        ptau[i] = pt[j]
        rpp[i] = rp[j]
        rvv[i] = rv[j]
        state[i] = st[j]
        lost_turn[i] = lt[j]
        lost_node[i] = ln[j]
        lost_cause[i] = lc[j]


def foil_kernels():
    """(mask, gather, scatter, put): the foil step's kernels (built once)."""
    global _FOIL
    kernel()
    if _FOIL is None:
        _FOIL = (cuda.jit(_foil_mask_kernel), cuda.jit(_gather_kernel),
                 cuda.jit(_scatter_kernel), cuda.jit(_put_kernel))
    return _FOIL


def _blocks(n: int) -> int:
    return max(1, (int(n) + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK)


def _check_slots(dev, idx) -> None:
    """(a slot out of range would read or write past the device arrays: a
    GPU raises nothing)"""
    if int(idx.min()) < 0 or int(idx.max()) >= dev[0].shape[1]:
        raise ValueError(f"foil step: slots {int(idx.min())}..{int(idx.max())} of "
                         f"{dev[0].shape[1]}")


def foil_slots(dev, n: int, bounds, mask) -> np.ndarray:
    """The slots (ascending) among the first ``n`` of the device particles
    ``dev`` that circulate inside the rectangle ``bounds`` = (x_min, x_max,
    y_min, y_max), None an open side; ``mask`` a device uint8 array of at
    least n."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if n > min(mask.size, dev[0].shape[1]):
        raise ValueError(f"foil_slots: {n} slots asked of {dev[0].shape[1]} (mask {mask.size})")
    k_mask = foil_kernels()[0]
    args = []
    for b in bounds:
        args += [b is not None, 0.0 if b is None else float(b)]
    k_mask[_blocks(n), THREADS_PER_BLOCK](dev[0], dev[4], int(n), *args, mask)
    cuda.synchronize()
    return np.flatnonzero(mask[:n].copy_to_host()).astype(np.int64)


def gather(dev, idx: np.ndarray) -> np.ndarray:
    """coords[:, idx] of the device particles, to the host (6, k)."""
    k = int(idx.size)
    if not k:
        return np.zeros((6, 0))
    _check_slots(dev, idx)
    out = cuda.device_array((6, k), dtype=np.float64)
    foil_kernels()[1][_blocks(k), THREADS_PER_BLOCK](dev[0], cuda.to_device(idx), k, out)
    cuda.synchronize()
    return out.copy_to_host()


def scatter(dev, idx: np.ndarray, X, pt, rp, rv) -> None:
    """coords[:, idx], ptau, rpp, rvv of the device particles from the host's."""
    k = int(idx.size)
    if not k:
        return
    _check_slots(dev, idx)
    foil_kernels()[2][_blocks(k), THREADS_PER_BLOCK](
        dev[0], dev[1], dev[2], dev[3], cuda.to_device(idx), k,
        *(cuda.to_device(np.ascontiguousarray(a, dtype=np.float64)) for a in (X, pt, rp, rv)))
    cuda.synchronize()


def put_slots(dev, host, a0: int, a1: int) -> None:
    """Slots a0..a1-1 of the host particles ``host`` (the arrays of ``dev``,
    in its order) to the device — the ones an injection filled."""
    k = int(a1) - int(a0)
    if k <= 0:
        return
    if int(a0) < 0 or int(a1) > dev[0].shape[1]:
        raise ValueError(f"put_slots: slots {a0}..{a1} of {dev[0].shape[1]}")
    parts = [cuda.to_device(np.ascontiguousarray(host[0][:, a0:a1]))]
    parts += [cuda.to_device(np.ascontiguousarray(a[a0:a1])) for a in host[1:]]
    foil_kernels()[3][_blocks(k), THREADS_PER_BLOCK](*dev, int(a0), k, *parts)
    cuda.synchronize()


def track_turns(coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause,
                kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1, n_turns, turn0,
                zeta_wrap=0.0):
    """Host arrays in, host arrays updated in place (copies to and from the
    device: a convenience for tests and short runs; a ring run keeps its
    particles on the device)."""
    kernel()
    n = coords.shape[1]
    host = (coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause)
    dev = [cuda.to_device(np.ascontiguousarray(a)) for a in host]
    table = [cuda.to_device(np.ascontiguousarray(a))
             for a in (kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2)]
    Launcher(dev, table, n, j0, j1, zeta_wrap).track(int(turn0), int(n_turns))
    for h, d in zip(host, dev, strict=True):
        h[...] = d.copy_to_host()
