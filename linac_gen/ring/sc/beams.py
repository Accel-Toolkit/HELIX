"""The beam as the space-charge kicks see it: its moments, its deposit and its
kick, on the CPU (:class:`CpuBeam`, numba's multi-core kernels) or on a CUDA
GPU (:class:`CudaBeam`, the particles left on the device; the grids — a few
tens of kB — go to the host for the field solve and back).

The CPU deposit sums into one integer grid per chunk of particles, the number
of chunks set by the number of particles alone, then adds the chunks: the
same integers for any number of threads.
"""
from __future__ import annotations

import numpy as np

from linac_gen.ring.sc import kernels as K

#: the most particle chunks of a CPU deposit (each its own integer grid)
MAX_CHUNKS = 32
#: particles per chunk, at least
CHUNK_MIN = 2048
#: the chunks' grids take at most about this many bytes (fewer chunks on a large grid)
DEPOSIT_BYTES = 64 * 2 ** 20


class CpuBeam:
    def __init__(self, coords, ptau, rpp, rvv, state, weight):
        from linac_gen.ring.kernels.cpu import PARALLEL_LOCK
        self.arrays = (coords, ptau, rpp, rvv, state)
        self.weight = weight
        self.n = int(coords.shape[1])
        self.lock = PARALLEL_LOCK
        self.fns = K.sc_backend()
        self._buf = {}

    def _buffers(self, chunks, nx, ny, nz):
        # (the grids by their shape, the line by its own: the line-density
        # bins of z_bins "auto" change without new grids; only the last shape
        # of each is kept)
        g = self._buf.get("grid")
        if g is None or g.shape != (chunks, nx, ny):
            g = self._buf["grid"] = np.zeros((chunks, nx, ny), np.int64)
        line = self._buf.get("line")
        if line is None or line.shape != (chunks, nz):
            line = self._buf["line"] = np.zeros((chunks, nz), np.int64)
        out = self._buf.get("outside")
        if out is None or out.shape != (chunks,):
            out = self._buf["outside"] = np.zeros(chunks, np.int64)
        for a in (g, line, out):
            a.fill(0)
        return g, line, out

    def moments(self, xp: float, yp: float) -> np.ndarray:
        nb = max(1, -(-self.n // K.MOMENT_BLOCK))
        out = np.empty((nb, K.N_MOMENTS))
        coords, _pt, _rp, _rv, state = self.arrays
        with self.lock:
            self.fns[0](coords, self.weight, state, self.n, K.MOMENT_BLOCK, float(xp),
                        float(yp), out)
        return out

    def deposit(self, kp, ki, scale: float, do_grid: bool):
        nx, ny = (int(ki[K.KI_NX]), int(ki[K.KI_NY])) if do_grid else (1, 1)
        # (set by the number of particles and the grid alone, never by the
        # threads: the integer sums are the same whatever the chunks anyway)
        chunks = int(max(1, min(MAX_CHUNKS, self.n // CHUNK_MIN,
                                DEPOSIT_BYTES // (8 * nx * ny))))
        grid, line, outside = self._buffers(chunks, nx, ny, int(ki[K.KI_NZ]))
        coords, _pt, _rp, _rv, state = self.arrays
        with self.lock:
            self.fns[1](coords, self.weight, state, self.n, kp, ki, float(scale), grid, line,
                        outside, bool(do_grid))
        return grid.sum(axis=0), line.sum(axis=0), int(outside.sum())

    def kick(self, fx, fy, phi, fl, dfl, far, cells, bnd, kp, ki) -> None:
        coords, ptau, rpp, rvv, state = self.arrays
        with self.lock:
            self.fns[2](coords, ptau, rpp, rvv, state, self.n, fx, fy, phi, fl, dfl, far, cells,
                        bnd, kp, ki)


class CudaBeam:
    """``dev`` = device arrays (coords, ptau, rpp, rvv, state); ``weight`` the
    host weights (copied once)."""
    THREADS = 128

    def __init__(self, dev, weight):
        from numba import cuda
        self.cuda = cuda
        self.dev = list(dev)
        self.n = int(self.dev[0].shape[1])
        self.weight = cuda.to_device(np.ascontiguousarray(weight, dtype=float))
        self.k_moments, self.k_deposit, self.k_kick = K.cuda_kernels()
        self._dbuf = {}

    def _device(self, name, host):
        """A device buffer of ``host``'s shape and type, refilled with it."""
        host = np.ascontiguousarray(host)
        b = self._dbuf.get(name)
        if b is None or b.shape != host.shape or b.dtype != host.dtype:
            b = self._dbuf[name] = self.cuda.to_device(host)
        else:
            b.copy_to_device(host)
        return b

    def _blocks(self, n):
        return max(1, (n + self.THREADS - 1) // self.THREADS)

    def moments(self, xp: float, yp: float) -> np.ndarray:
        nb = max(1, -(-self.n // K.MOMENT_BLOCK))
        out = self._device("moments", np.zeros((nb, K.N_MOMENTS)))
        coords, _pt, _rp, _rv, state = self.dev
        self.k_moments[self._blocks(nb), self.THREADS](coords, self.weight, state, self.n,
                                                       K.MOMENT_BLOCK, float(xp), float(yp),
                                                       out)
        self.cuda.synchronize()
        return out.copy_to_host()

    def deposit(self, kp, ki, scale: float, do_grid: bool):
        nx, ny = (int(ki[K.KI_NX]), int(ki[K.KI_NY])) if do_grid else (1, 1)
        nz = int(ki[K.KI_NZ])
        grid = self._device("grid", np.zeros((nx, ny), np.int64))
        line = self._device("line", np.zeros(nz, np.int64))
        outside = self._device("outside", np.zeros(1, np.int64))
        coords, _pt, _rp, _rv, state = self.dev
        self.k_deposit[self._blocks(self.n), self.THREADS](
            coords, self.weight, state, self.n, self._device("kp", kp), self._device("ki", ki),
            float(scale), grid, line, outside, bool(do_grid))
        self.cuda.synchronize()
        return grid.copy_to_host(), line.copy_to_host(), int(outside.copy_to_host()[0])

    def kick(self, fx, fy, phi, fl, dfl, far, cells, bnd, kp, ki) -> None:
        coords, ptau, rpp, rvv, state = self.dev
        d = self._device
        self.k_kick[self._blocks(self.n), self.THREADS](
            coords, ptau, rpp, rvv, state, self.n, d("fx", fx), d("fy", fy), d("phi", phi),
            d("fl", fl), d("dfl", dfl), d("far", far), d("cells", cells), d("bnd", bnd),
            d("kp", kp), d("ki", ki))
        self.cuda.synchronize()
