"""Space-charge particle kernels: moments, deposit, kick — one source for the
CPU (numba, serial or multi-core; plain Python without numba) and CUDA.

**Deposit.**  Each circulating particle's weight goes to the grid nodes
around it with the TSC (triangular-shaped cloud: 3 x 3 nodes) or CIC (2 x 2)
shape, and to the line-density bins (CIC, periodic in zeta), as integers:
weight x shape x 2^k rounded, 2^k chosen so that the whole bunch sums to
about 2^60.  Integer sums do not depend on their order, so the grid comes out
the same, bit for bit, for any number of threads (the CPU sums per chunk,
then the chunks; the GPU adds atomically).  A particle's grid coordinate is
a division ((x - x0)/dx, which no compiler fuses with the add that follows),
the shape weights are written without a multiply followed by an add, and the
scale is a power of two, so a GPU that fuses multiply-adds computes the same
integers for the same particles.

**Moments** (the grid's box, the frozen model's Gaussian): per block of
:data:`MOMENT_BLOCK` particles, in order, then the blocks in order — the same
sums for any number of threads.  The centroid and the rms sizes are the
charge's: each particle's offsets weighted by its weight relative to the
block's largest, r = w / w_b — exactly 1 when the weights are equal, so that
the sums are then the plain sums of the offsets, bit for bit.

**Kick.**  A particle on the grid gathers F_x, F_y (and Phi) with the shape
it was deposited with, so its own charge gives it no kick; off the grid it
takes the field of the grid's charge — its multipole series about the beam's
centre where the series converges fast (beyond 1.5 times the farthest
charge), else the direct sum over the charged cells (each cell integrated
within 2.5 cells of the particle, a point charge beyond).  A particle inside
the pipe's wall adds the pipe's harmonic at its position (beyond the wall
only its constant term).  Then (r_c the classical radius, N_b the
bunch's real particles, ds the kick's length, f the normalized line density)

    dpx = KT f(zeta) F_x,  dpy = KT f(zeta) F_y,   KT = 2 r_c N_b ds / (beta0^2 gamma0^3)
    dptau = -KL f'(zeta) Phi,                       KL = 2 r_c N_b ds / (beta0 gamma0^3)

— the lab-frame field of a long bunch with the magnetic force (1/gamma0^2
of the electric) and the longitudinal field -lambda' phi / gamma0^2.
"""
from __future__ import annotations

from math import atan, ceil, exp, floor, log, sqrt

from linac_gen.ring.kernels._jit import jitable
from linac_gen.ring.kernels.cartesian import energy_from_ptau

#: a round pipe's series: its terms kept down to this size relative to the
#: first, anywhere inside the pipe (each particle sums the terms it needs)
SERIES_TOL = 1e-13
#: the most terms of a round pipe's series
ROUND_MAX_ORDER = 128
_LOG_TOL = log(SERIES_TOL)

try:
    from numba import prange
except ImportError:          # plain Python
    prange = range

# kick parameters (float64 array)
KP_X0 = 0          # x of grid node 0 [m]
KP_Y0 = 1
KP_DX = 2          # the grid's cell size dx [m] (coordinates are divided by it)
KP_DY = 3
KP_XC = 4          # centre of the grid charge's multipole series
KP_YC = 5
KP_XP = 6          # centre of the pipe (of its harmonic series)
KP_YP = 7
KP_IR = 8          # 1/R, the scale of the pipe's series
KP_Z0 = 9          # centre of line-density bin 0 [m]
KP_DZ = 10         # the line density's bin size [m]
KP_KT = 11         # transverse factor
KP_KL = 12         # longitudinal factor
KP_PHI_G = 13      # the g-factor model's Phi (g/2)
KP_F0 = 14         # a coasting beam's line density [1/m]
KP_BETA0 = 15
KP_RCONV = 16      # off the grid, the series is used beyond this distance from its centre
KP_DXC = 17        # the grid's cells (for the direct sum)
KP_DYC = 18
KP_PA = 19         # the pipe: radius or half width ...
KP_PB = 20         # ... and half height (the pipe's correction applies inside it only)
KP_N = 21

# integer parameters (int64 array)
KI_NX = 0
KI_NY = 1
KI_NZ = 2
KI_ORDER = 3       # 1 CIC, 2 TSC
KI_NFAR = 4        # order of the multipole series off the grid
KI_NBND = 5        # order of the pipe's series
KI_LONG = 6        # 0 off, 1 potential, 2 g-factor
KI_LINE = 7        # 0 uniform (coasting beam), 1 the measured line density
KI_NCELL = 8       # charged cells for the direct sum (cells[0..2, :n])
KI_PIPE = 9        # 0 open, 1 round, 2 rectangle, 3 ellipse
KI_N = 10

#: particles summed in order per block for the moments
MOMENT_BLOCK = 2048
#: moments columns: count, sum w, sum r dx, sum r dy, sum r dx^2, sum r dy^2, min x, max x,
#: min y, max y, sum r, w_b (r = w / w_b, w_b the block's largest weight)
N_MOMENTS = 12


@jitable
def shape_1d(u, order):
    """(first node, w0, w1, w2) of the TSC (order 2) or CIC (order 1) shape at
    grid coordinate u (w2 = 0 for CIC).  TSC's middle weight is 1 - w0 - w2,
    not 3/4 - f^2: no multiply-add, the same bits on a GPU that fuses them."""
    if order == 2:
        c = floor(u + 0.5)
        f = u - c
        a = 0.5 - f
        b = 0.5 + f
        w0 = 0.5 * (a * a)
        w2 = 0.5 * (b * b)
        w1 = 1.0 - w0 - w2
        return int(c) - 1, w0, w1, w2
    c = floor(u)
    f = u - c
    return int(c), 1.0 - f, f, 0.0


@jitable
def on_grid(u, v, nx, ny):
    """Whether grid coordinates (u, v) are far enough inside for the shape."""
    return u >= 1.0 and u < nx - 2.0 and v >= 1.0 and v < ny - 2.0


@jitable
def fixed(t, scale):
    """t x scale as an integer (scale a power of two: t x scale is exact)."""
    return int(floor(t * scale + 0.5))


@jitable
def line_cell(zeta, z0, dz, nz):
    """(bin k0, bin k1, w0, w1): zeta between the centres of bins k0 and k1
    (periodic over nz bins of size dz whose bin 0 is centred at z0)."""
    u = (zeta - z0) / dz
    c = floor(u)
    f = u - c
    k0 = int(c) % nz
    if k0 < 0:
        k0 += nz
    k1 = k0 + 1
    if k1 >= nz:
        k1 -= nz
    return k0, k1, 1.0 - f, f


@jitable
def gather(A, i0, j0, a0, a1, a2, b0, b1, b2, order):
    if order == 2:
        return (a0 * (b0 * A[i0, j0] + b1 * A[i0, j0 + 1] + b2 * A[i0, j0 + 2])
                + a1 * (b0 * A[i0 + 1, j0] + b1 * A[i0 + 1, j0 + 1] + b2 * A[i0 + 1, j0 + 2])
                + a2 * (b0 * A[i0 + 2, j0] + b1 * A[i0 + 2, j0 + 1] + b2 * A[i0 + 2, j0 + 2]))
    return (a0 * (b0 * A[i0, j0] + b1 * A[i0, j0 + 1])
            + a1 * (b0 * A[i0 + 1, j0] + b1 * A[i0 + 1, j0 + 1]))


@jitable
def far_field(dx, dy, far, nfar, need_phi):
    """(F_x, F_y, Phi) at offset (dx, dy) from the centre of the grid's charge
    from its moments M_n = sum q (w_q)^n (far[0] real, far[1] imaginary parts):
    F_x - i F_y = sum M_n / w^(n+1), Phi = -M_0 ln|w| + Re sum_(n>=1) M_n/(n w^n)."""
    d = dx * dx + dy * dy
    tr = dx / d
    ti = -dy / d
    sr = far[0, nfar]
    si = far[1, nfar]
    for n in range(nfar - 1, -1, -1):
        r = sr * tr - si * ti + far[0, n]
        si = sr * ti + si * tr + far[1, n]
        sr = r
    fxr = sr * tr - si * ti
    fxi = sr * ti + si * tr
    phi = 0.0
    if need_phi:
        phi = -far[0, 0] * 0.5 * log(d)
        if nfar >= 1:
            pr = far[0, nfar] / nfar
            pi = far[1, nfar] / nfar
            for n in range(nfar - 1, 0, -1):
                r = pr * tr - pi * ti + far[0, n] / n
                pi = pr * ti + pi * tr + far[1, n] / n
                pr = r
            phi += pr * tr - pi * ti
    return fxr, -fxi, phi


@jitable
def _prim_field_s(u, v):
    """∫∫ u/(u² + v²) du dv up to a function of v: v ln(u² + v²)/2 + u atan(v/u)."""
    r2 = u * u + v * v
    out = 0.0
    if r2 > 0.0:
        out = 0.5 * v * log(r2)
    if u != 0.0:
        out += u * atan(v / u)
    return out


@jitable
def _prim_log_s(u, v):
    """∫∫ ln(u² + v²) du dv."""
    r2 = u * u + v * v
    out = -3.0 * u * v
    if r2 > 0.0:
        out += u * v * log(r2)
    if u != 0.0:
        out += u * u * atan(v / u)
    if v != 0.0:
        out += v * v * atan(u / v)
    return out


@jitable
def direct_field(x, y, cells, ncell, dxc, dyc, need_phi):
    """(F_x, F_y, Phi) at (x, y) from the charged cells (cells[0], cells[1]:
    the nodes; cells[2]: their charges), each spread over its cell: the cell
    integral within 2.5 times the longer cell side of the point (in x and in
    y), a point charge beyond."""
    s = sqrt(dxc * dyc)
    hx = 0.5 * dxc / s
    hy = 0.5 * dyc / s
    ls = log(s)
    # (the cell integral wherever the point is within 2.5 of the longer cell
    # side: a flat cell is a poor point charge across its whole width)
    near = 2.5 * (dxc if dxc > dyc else dyc)
    fx = 0.0
    fy = 0.0
    phi = 0.0
    for c in range(ncell):
        u = x - cells[0, c]
        v = y - cells[1, c]
        q = cells[2, c]
        if abs(u) < near and abs(v) < near:
            U = u / s
            V = v / s
            gx = (_prim_field_s(U + hx, V + hy) - _prim_field_s(U - hx, V + hy)
                  - _prim_field_s(U + hx, V - hy) + _prim_field_s(U - hx, V - hy))
            gy = (_prim_field_s(V + hy, U + hx) - _prim_field_s(V - hy, U + hx)
                  - _prim_field_s(V + hy, U - hx) + _prim_field_s(V - hy, U - hx))
            fx += q * gx / s
            fy += q * gy / s
            if need_phi:
                gp = (_prim_log_s(U + hx, V + hy) - _prim_log_s(U - hx, V + hy)
                      - _prim_log_s(U + hx, V - hy) + _prim_log_s(U - hx, V - hy))
                phi += q * (-0.5 * gp - ls)
        else:
            r2 = u * u + v * v
            fx += q * u / r2
            fy += q * v / r2
            if need_phi:
                phi -= q * 0.5 * log(r2)
    return fx, fy, phi


@jitable
def boundary_field(dx, dy, bnd, nb, inv_r, need_phi):
    """(F_x, F_y, Phi) of the pipe's harmonic h = Re sum c_n s^n, s = (dx + i dy)/R
    (bnd[0] real, bnd[1] imaginary parts of c_n): F_x - i F_y = -h'."""
    sr = dx * inv_r
    si = dy * inv_r
    dr = 0.0
    di = 0.0
    if nb >= 1:
        dr = nb * bnd[0, nb]
        di = nb * bnd[1, nb]
        for n in range(nb - 1, 0, -1):
            r = dr * sr - di * si + n * bnd[0, n]
            di = dr * si + di * sr + n * bnd[1, n]
            dr = r
    fx = -dr * inv_r
    fy = di * inv_r
    phi = 0.0
    if need_phi:
        pr = bnd[0, nb]
        pi = bnd[1, nb]
        for n in range(nb - 1, -1, -1):
            r = pr * sr - pi * si + bnd[0, n]
            pi = pr * si + pi * sr + bnd[1, n]
            pr = r
        phi = pr
    return fx, fy, phi


@jitable
def round_terms(dx, dy, nb, inv_r):
    """The terms of a round pipe's series a particle at offset (dx, dy) needs:
    they fall as (r rho / b)^n, and rho <= exp(log(tol) / nb) for the nb terms
    that hold the series to the tolerance anywhere inside (an order at its
    cap: all of them)."""
    if nb <= 1 or nb >= ROUND_MAX_ORDER:
        return nb
    t = sqrt(dx * dx + dy * dy) * inv_r * exp(_LOG_TOL / nb)
    if t <= 1e-300:
        return 1
    return max(1, min(nb, int(ceil(_LOG_TOL / log(t)))))


@jitable
def inside_pipe(dx, dy, kind, a, b):
    """Whether offset (dx, dy) from the pipe's centre is inside its wall (an
    open pipe: everywhere)."""
    if kind == 1:
        return dx * dx + dy * dy < a * a
    if kind == 2:
        return abs(dx) < a and abs(dy) < b
    if kind == 3:
        return (dx / a) * (dx / a) + (dy / b) * (dy / b) < 1.0
    return True


@jitable
def kick_particle(x, y, zeta, px, py, ptau, Fx, Fy, Phi, fl, dfl, far, cells, bnd, kp, ki):
    """(px, py, ptau) after the kick (see the module docstring)."""
    nx = ki[KI_NX]
    ny = ki[KI_NY]
    order = ki[KI_ORDER]
    need_phi = ki[KI_LONG] == 1
    u = (x - kp[KP_X0]) / kp[KP_DX]
    v = (y - kp[KP_Y0]) / kp[KP_DY]
    if on_grid(u, v, nx, ny):
        i0, a0, a1, a2 = shape_1d(u, order)
        j0, b0, b1, b2 = shape_1d(v, order)
        fx = gather(Fx, i0, j0, a0, a1, a2, b0, b1, b2, order)
        fy = gather(Fy, i0, j0, a0, a1, a2, b0, b1, b2, order)
        phi = gather(Phi, i0, j0, a0, a1, a2, b0, b1, b2, order) if need_phi else 0.0
    else:
        wx = x - kp[KP_XC]
        wy = y - kp[KP_YC]
        if wx * wx + wy * wy >= kp[KP_RCONV] * kp[KP_RCONV]:
            fx, fy, phi = far_field(wx, wy, far, ki[KI_NFAR], need_phi)
        else:
            fx, fy, phi = direct_field(x, y, cells, ki[KI_NCELL], kp[KP_DXC], kp[KP_DYC],
                                       need_phi)
    # (the pipe's harmonic at the particle, inside the wall: beyond it the
    # series need not converge — the lattice's apertures lose the particle —
    # and only its constant term is added, so that the potential has no step
    # there)
    ox = x - kp[KP_XP]
    oy = y - kp[KP_YP]
    if inside_pipe(ox, oy, ki[KI_PIPE], kp[KP_PA], kp[KP_PB]):
        nb = ki[KI_NBND]
        if ki[KI_PIPE] == 1:
            nb = round_terms(ox, oy, nb, kp[KP_IR])
        bx, by, bphi = boundary_field(ox, oy, bnd, nb, kp[KP_IR], need_phi)
        fx += bx
        fy += by
        phi += bphi
    elif need_phi:
        phi += bnd[0, 0]
    if ki[KI_LINE] == 1:
        k0, k1, w0, w1 = line_cell(zeta, kp[KP_Z0], kp[KP_DZ], ki[KI_NZ])
        fz = w0 * fl[k0] + w1 * fl[k1]
        dfz = w0 * dfl[k0] + w1 * dfl[k1]
    else:
        fz = kp[KP_F0]
        dfz = 0.0
    px = px + kp[KP_KT] * fz * fx
    py = py + kp[KP_KT] * fz * fy
    if ki[KI_LONG] == 1:
        ptau = ptau - kp[KP_KL] * dfz * phi
    elif ki[KI_LONG] == 2:
        ptau = ptau - kp[KP_KL] * dfz * kp[KP_PHI_G]
    return px, py, ptau


@jitable
def kick_one(i, coords, ptau, rpp, rvv, Fx, Fy, Phi, fl, dfl, far, cells, bnd, kp, ki):
    pt0 = ptau[i]
    px, py, pt = kick_particle(coords[0, i], coords[2, i], coords[4, i], coords[1, i],
                               coords[3, i], pt0, Fx, Fy, Phi, fl, dfl, far, cells, bnd, kp,
                               ki)
    coords[1, i] = px
    coords[3, i] = py
    if pt != pt0:
        ptau[i] = pt
        d, rp, rv = energy_from_ptau(pt, kp[KP_BETA0])
        coords[5, i] = d
        rpp[i] = rp
        rvv[i] = rv


@jitable
def moments_block(b, coords, weight, state, n, block, xp, yp, out):
    """Row b of ``out``: the moments of particles b*block .. (b+1)*block-1,
    summed in order (offsets from the pivot (xp, yp)), each particle's
    offsets weighted by r = w / w_b, w_b the largest weight of the block's
    circulating particles (r = 1 for every one where none is positive)."""
    lo = b * block
    hi = lo + block
    if hi > n:
        hi = n
    wb = 0.0
    for i in range(lo, hi):
        if state[i] == 1 and weight[i] > wb:
            wb = weight[i]
    cnt = 0.0
    sw = 0.0
    sr = 0.0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    syy = 0.0
    mnx = 1e300
    mxx = -1e300
    mny = 1e300
    mxy = -1e300
    for i in range(lo, hi):
        if state[i] != 1:
            continue
        x = coords[0, i]
        y = coords[2, i]
        dx = x - xp
        dy = y - yp
        r = weight[i] / wb if wb > 0.0 else 1.0
        cnt += 1.0
        sw += weight[i]
        sr += r
        sx += r * dx
        sy += r * dy
        sxx += r * dx * dx
        syy += r * dy * dy
        if x < mnx:
            mnx = x
        if x > mxx:
            mxx = x
        if y < mny:
            mny = y
        if y > mxy:
            mxy = y
    out[b, 0] = cnt
    out[b, 1] = sw
    out[b, 2] = sx
    out[b, 3] = sy
    out[b, 4] = sxx
    out[b, 5] = syy
    out[b, 6] = mnx
    out[b, 7] = mxx
    out[b, 8] = mny
    out[b, 9] = mxy
    out[b, 10] = sr
    out[b, 11] = wb


@jitable
def grid_weights(x, y, kp, ki):
    """(on the grid?, i0, j0, a0, a1, a2, b0, b1, b2) of a particle at (x, y)."""
    u = (x - kp[KP_X0]) / kp[KP_DX]
    v = (y - kp[KP_Y0]) / kp[KP_DY]
    if not on_grid(u, v, ki[KI_NX], ki[KI_NY]):
        return False, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    i0, a0, a1, a2 = shape_1d(u, ki[KI_ORDER])
    j0, b0, b1, b2 = shape_1d(v, ki[KI_ORDER])
    return True, i0, j0, a0, a1, a2, b0, b1, b2


# ---------------------------------------------------------------------------
# CPU drivers (compiled by sc_backend; plain Python without numba)
# ---------------------------------------------------------------------------
def moments(coords, weight, state, n, block, xp, yp, out):
    for b in prange(out.shape[0]):
        moments_block(b, coords, weight, state, n, block, xp, yp, out)


def deposit(coords, weight, state, n, kp, ki, scale, grid, line, outside, do_grid):
    """Into grid[c] / line[c] / outside[c] per chunk c of the particles (the
    chunks summed after: integers, any order)."""
    chunks = grid.shape[0]
    order = ki[KI_ORDER]
    for c in prange(chunks):
        lo = (c * n) // chunks
        hi = ((c + 1) * n) // chunks
        for i in range(lo, hi):
            if state[i] != 1:
                continue
            w = weight[i]
            if do_grid:
                ok, i0, j0, a0, a1, a2, b0, b1, b2 = grid_weights(coords[0, i], coords[2, i],
                                                                  kp, ki)
                if ok:
                    for di in range(order + 1):
                        a = a0 if di == 0 else (a1 if di == 1 else a2)
                        wa = w * a
                        for dj in range(order + 1):
                            b = b0 if dj == 0 else (b1 if dj == 1 else b2)
                            grid[c, i0 + di, j0 + dj] += fixed(wa * b, scale)
                else:
                    outside[c] += 1
            if ki[KI_LINE] == 1:
                k0, k1, w0, w1 = line_cell(coords[4, i], kp[KP_Z0], kp[KP_DZ], ki[KI_NZ])
                line[c, k0] += fixed(w * w0, scale)
                line[c, k1] += fixed(w * w1, scale)


def kick(coords, ptau, rpp, rvv, state, n, Fx, Fy, Phi, fl, dfl, far, cells, bnd, kp, ki):
    for i in prange(n):
        if state[i] == 1:
            kick_one(i, coords, ptau, rpp, rvv, Fx, Fy, Phi, fl, dfl, far, cells, bnd, kp, ki)


_BACKEND = None


def sc_backend():
    """(moments, deposit, kick_parallel, kick_serial), compiled once (plain
    Python without numba)."""
    global _BACKEND
    if _BACKEND is None:
        from linac_gen.ring.kernels.cpu import _BUILD_LOCK, backend, numba_available
        backend()                           # (sets numba's threading layer first)
        with _BUILD_LOCK:
            if _BACKEND is None:
                if not numba_available():
                    _BACKEND = (moments, deposit, kick, kick)
                else:
                    import numba
                    par = dict(cache=False, parallel=True, error_model="numpy")
                    _BACKEND = (numba.njit(**par)(moments), numba.njit(**par)(deposit),
                                numba.njit(**par)(kick),
                                numba.njit(cache=False, error_model="numpy")(kick))
    return _BACKEND


# ---------------------------------------------------------------------------
# CUDA kernels (built on first use; ``cuda`` is a module global so that
# numba's CUDA simulator can swap it while it runs them)
# ---------------------------------------------------------------------------
cuda = None
_CUDA = None


def _moments_kernel(coords, weight, state, n, block, xp, yp, out):
    b = cuda.grid(1)
    if b < out.shape[0]:
        moments_block(b, coords, weight, state, n, block, xp, yp, out)


def _deposit_kernel(coords, weight, state, n, kp, ki, scale, grid, line, outside, do_grid):
    i = cuda.grid(1)
    if i >= n or state[i] != 1:
        return
    w = weight[i]
    order = ki[KI_ORDER]
    if do_grid:
        ok, i0, j0, a0, a1, a2, b0, b1, b2 = grid_weights(coords[0, i], coords[2, i], kp, ki)
        if ok:
            for di in range(order + 1):
                a = a0 if di == 0 else (a1 if di == 1 else a2)
                wa = w * a
                for dj in range(order + 1):
                    b = b0 if dj == 0 else (b1 if dj == 1 else b2)
                    cuda.atomic.add(grid, (i0 + di, j0 + dj), fixed(wa * b, scale))
        else:
            cuda.atomic.add(outside, 0, 1)
    if ki[KI_LINE] == 1:
        k0, k1, w0, w1 = line_cell(coords[4, i], kp[KP_Z0], kp[KP_DZ], ki[KI_NZ])
        cuda.atomic.add(line, k0, fixed(w * w0, scale))
        cuda.atomic.add(line, k1, fixed(w * w1, scale))


def _kick_kernel(coords, ptau, rpp, rvv, state, n, Fx, Fy, Phi, fl, dfl, far, cells, bnd, kp,
                 ki):
    i = cuda.grid(1)
    if i < n and state[i] == 1:
        kick_one(i, coords, ptau, rpp, rvv, Fx, Fy, Phi, fl, dfl, far, cells, bnd, kp, ki)


def cuda_kernels():
    """(moments, deposit, kick) CUDA kernels."""
    global cuda, _CUDA
    if _CUDA is None:
        from numba import cuda as _cuda
        cuda = _cuda
        _CUDA = (cuda.jit(_moments_kernel), cuda.jit(_deposit_kernel), cuda.jit(_kick_kernel))
    return _CUDA
