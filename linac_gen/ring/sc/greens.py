"""The 2-D field solve of the space-charge kicks.

The grid holds node charges q_ij (fractions of the bunch, from the deposit),
each taken as spread uniformly over its node's cell.  The dimensionless
potential Phi = -∫ rho ln|r - r'| dA' and the field F = -grad Phi at every
node are then exact sums over the cells: Green's functions integrated over a
cell (the integrated Green's function of J. Qiang, S. Lidia, R. D. Ryne and
C. Limborg-Deprey, Phys. Rev. ST Accel. Beams 9, 044204 (2006), here in two
dimensions with closed-form primitives), convolved with the charge by FFT on
a grid twice the size in each direction (zero padding: the open-boundary
convolution of R. W. Hockney and J. W. Eastwood, Computer Simulation Using
Particles, 1988).  The cell integrals hold for any cell aspect ratio.

The primitives (u, v the offsets from a cell's corners):

* ∫∫ ln(u² + v²) du dv = u v ln(u² + v²) - 3 u v + u² atan(v/u) + v² atan(u/v);
* ∫∫ u / (u² + v²) du dv = v ln(u² + v²) / 2 + u atan(v/u) (up to a term in v
  alone, which the four corners cancel) — and the same with u, v exchanged
  for F_y.

They are evaluated in units of sqrt(dx dy), which keeps their size moderate
whatever the cells; F is then exactly antisymmetric, so a particle's own
charge gives it no kick when it is gathered with the shape it was deposited
with.  The FFTs are scipy's on one thread (the same bits in every run).
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict

import numpy as np
from scipy import fft as _fft

#: the transforms of the Green's functions kept, at most this many bytes (the
#: least recently used go first; a 64 x 64 grid's take 0.5 MB, 128 x 128 2 MB)
CACHE_BYTES = 256 * 2 ** 20
_CACHE: OrderedDict = OrderedDict()
_CACHE_USED = [0]
_LOCK = threading.Lock()


def _prim_log_loop(U, V, out):
    for i in range(U.shape[0]):
        u = U[i]
        v = V[i]
        r2 = u * u + v * v
        o = -3.0 * u * v
        if r2 > 0.0:
            o += u * v * math.log(r2)
        if u != 0.0:
            o += u * u * math.atan(v / u)
        if v != 0.0:
            o += v * v * math.atan(u / v)
        out[i] = o


def _prim_field_loop(U, V, out):
    for i in range(U.shape[0]):
        u = U[i]
        v = V[i]
        r2 = u * u + v * v
        o = 0.0
        if r2 > 0.0:
            o = 0.5 * v * math.log(r2)
        if u != 0.0:
            o += u * math.atan(v / u)
        out[i] = o


_LOOPS = None


def _loops():
    """The primitives' loops, compiled by numba when present: scalar
    logarithms and arc tangents, the same bits in every process (numpy's
    vector loops may round otherwise, and some builds pick them by where
    the arrays sit in memory)."""
    global _LOOPS
    if _LOOPS is None:
        try:
            import numba
            _LOOPS = (numba.njit(cache=False)(_prim_log_loop),
                      numba.njit(cache=False)(_prim_field_loop))
        except ImportError:
            _LOOPS = (_prim_log_loop, _prim_field_loop)
    return _LOOPS


def _prim_log(U, V):
    """∫∫ ln(U² + V²) dU dV (a primitive), elementwise."""
    U = np.ascontiguousarray(U, dtype=float)
    V = np.ascontiguousarray(V, dtype=float)
    out = np.empty(U.size)
    _loops()[0](U.ravel(), V.ravel(), out)
    return out.reshape(U.shape)


def _prim_field(U, V):
    """∫∫ U/(U² + V²) dU dV up to a function of V alone: V ln(U² + V²)/2 +
    U atan(V/U), elementwise."""
    U = np.ascontiguousarray(U, dtype=float)
    V = np.ascontiguousarray(V, dtype=float)
    out = np.empty(U.size)
    _loops()[1](U.ravel(), V.ravel(), out)
    return out.reshape(U.shape)


def _cells(A):
    """Four-corner differences: the integral over each cell."""
    return A[1:, 1:] - A[:-1, 1:] - A[1:, :-1] + A[:-1, :-1]


def cell_greens(nx: int, ny: int, dx: float, dy: float):
    """(G_phi, G_x, G_y), each (2nx, 2ny) in the FFT's circular order (offset
    (0, 0) at [0, 0], offset -m at index 2nx - m; index nx unused): the
    potential and field at a node from a unit charge spread over the cell of a
    node (m, n) cells away.  The offsets m, n >= 0 are integrated; the others
    follow by symmetry (G_x odd in m, G_y odd in n, G_phi even), exactly."""
    s = float(np.sqrt(dx * dy))
    cx = (np.arange(0, nx + 1) - 0.5) * (dx / s)
    cy = (np.arange(0, ny + 1) - 0.5) * (dy / s)
    U, V = np.meshgrid(cx, cy, indexing="ij")
    q_phi = -0.5 * _cells(_prim_log(U, V)) - math.log(s)
    q_x = _cells(_prim_field(U, V)) / s
    q_y = _cells(_prim_field(V, U)) / s
    q_x[0, :] = 0.0                       # (odd in m: zero on its axis)
    q_y[:, 0] = 0.0

    def full(q, sx, sy):
        out = np.zeros((2 * nx, 2 * ny))
        out[:nx, :ny] = q
        out[nx + 1:, :ny] = sx * q[:0:-1, :]            # offsets -(nx-1)..-1
        out[:nx, ny + 1:] = sy * q[:, :0:-1]
        out[nx + 1:, ny + 1:] = (sx * sy) * q[:0:-1, :0:-1]
        return out
    return full(q_phi, 1.0, 1.0), full(q_x, -1.0, 1.0), full(q_y, 1.0, -1.0)


def potential_at(px, py, X, Y, q, dx, dy):
    """Phi at arbitrary points (px, py) from cell charges q at node positions
    X, Y (cells dx x dy): a direct sum of the cell integrals (for the few
    points of a pipe's boundary)."""
    s = float(np.sqrt(dx * dy))
    sel = q != 0.0
    xs, ys, qs = X[sel], Y[sel], q[sel]
    out = np.empty(len(px))
    hx, hy = 0.5 * dx / s, 0.5 * dy / s
    for k in range(len(px)):
        u = (px[k] - xs) / s
        v = (py[k] - ys) / s
        tot = (_prim_log(u + hx, v + hy) - _prim_log(u - hx, v + hy)
               - _prim_log(u + hx, v - hy) + _prim_log(u - hx, v - hy))
        out[k] = float(np.sum(qs * (-0.5 * tot))) - float(np.sum(qs)) * math.log(s)
    return out


def _transforms(nx: int, ny: int, dx: float, dy: float):
    """The transforms of (G_phi, G_x, G_y), each as its (real, imaginary)
    parts in contiguous arrays."""
    key = (int(nx), int(ny), float(dx), float(dy))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    g = cell_greens(nx, ny, dx, dy)
    out = []
    for a in g:
        t = _fft.rfft2(a, workers=1)
        out.append((np.ascontiguousarray(t.real), np.ascontiguousarray(t.imag)))
    out = tuple(out)
    size = sum(r.nbytes + i.nbytes for r, i in out)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = out
            _CACHE_USED[0] += size
        while _CACHE_USED[0] > CACHE_BYTES and len(_CACHE) > 1:
            _k, old = _CACHE.popitem(last=False)
            _CACHE_USED[0] -= sum(r.nbytes + i.nbytes for r, i in old)
    return out


def _product(qr, qi, g):
    """The spectrum Q x G from real and imaginary parts, in real arithmetic:
    numpy's complex multiply takes other roundings depending on where the
    arrays sit in memory (the same effect as the linac PIC's 0b8f531)."""
    gr, gi = g
    out = np.empty(qr.shape, dtype=complex)
    out.real = qr * gr - qi * gi
    out.imag = qr * gi + qi * gr
    return out


def solve(q: np.ndarray, dx: float, dy: float, *, potential: bool = True):
    """(Phi or None, F_x, F_y) at the nodes of the grid holding the cell
    charges ``q`` (nx, ny), open boundary."""
    nx, ny = q.shape
    gp, gx, gy = _transforms(nx, ny, dx, dy)
    shape = (2 * nx, 2 * ny)
    Q = _fft.rfft2(np.ascontiguousarray(q, dtype=float), s=shape, workers=1)
    qr, qi = np.ascontiguousarray(Q.real), np.ascontiguousarray(Q.imag)
    fx = _fft.irfft2(_product(qr, qi, gx), s=shape, workers=1)[:nx, :ny]
    fy = _fft.irfft2(_product(qr, qi, gy), s=shape, workers=1)[:nx, :ny]
    phi = (np.ascontiguousarray(_fft.irfft2(_product(qr, qi, gp), s=shape, workers=1)[:nx, :ny])
           if potential else None)
    return phi, np.ascontiguousarray(fx), np.ascontiguousarray(fy)


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _CACHE_USED[0] = 0
