"""Space charge at a common time for the time-domain RFQ solver.

Poisson's equation for the macro-particles, periodic in z with the local
bunch period P = β·λ (the beam is one period of an infinite train of
identical bunches — the DC beam at the RFQ entrance is exactly that) and
open in x, y.  An FFT along z separates the longitudinal modes
k = 2πm/P; each obeys (∇⊥² − k²)φ_k = −ρ_k/ε₀, solved by Hockney's
doubled-grid convolution with the cell-integrated Green's function

    g_0(r) = −ln(r)/(2π),        g_k(r) = K₀(|k| r)/(2π)   (k ≠ 0).

Charge is deposited with HELIX's cloud-in-cell kernel (the deterministic
C++ one when built) on a grid one slice longer than the period, whose
last slice is folded onto the first; the field is gathered the same
way.  The transverse grid spans ±``extent`` rms sizes around the
centroid (with hysteresis, so the RF flutter of the beam size does not
re-size it every step); the number of slices per period follows the
bunch core length (``cells_per_sigma_z`` per rms), so a long DC beam
and the short bunch at the RFQ exit are both resolved.  Particles
outside the transverse grid get the monopole field of the period's
line charge.

Units: mm, V, C.  The charge is deposited unsigned: the field returned
is that of a positive beam, and q·E is then +|q|·E for either species.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import k0

EPS0_MM = 8.8541878128e-12 * 1e-3          # F/mm


def _cell_integrated_green(nx, ny, hx, hy, k, nq=4, nsub=24, near=2):
    """(2nx, 2ny) doubled-grid array of the cell-integrated g_k (mm^2).

    Cells within ``near`` of the origin: Gauss-Legendre ``nq``^2 (and a
    fine midpoint subdivision for the singular 3x3 block); farther cells,
    where g_k is smooth on the cell scale, the midpoint value times the
    cell area (relative error ~ (h/r)^2/24 <= 0.3 %)."""
    ix = np.arange(2 * nx)
    ix = np.where(ix < nx, ix, ix - 2 * nx).astype(float)
    iy = np.arange(2 * ny)
    iy = np.where(iy < ny, iy, iy - 2 * ny).astype(float)

    def g(r):
        if k == 0.0:
            return -np.log(r) / (2.0 * np.pi)
        return k0(k * r) / (2.0 * np.pi)

    R = np.sqrt((ix[:, None] * hx) ** 2 + (iy[None, :] * hy) ** 2)
    R[0, 0] = 1.0                                     # replaced below
    G = g(R) * hx * hy
    gq, wq = np.polynomial.legendre.leggauss(nq)
    W = 0.25 * hx * hy * wq[:, None] * wq[None, :]
    for di in range(-near, near + 1):
        for dj in range(-near, near + 1):
            if max(abs(di), abs(dj)) <= 1:
                s = (np.arange(nsub) + 0.5) / nsub - 0.5
                xx = (di + s[:, None]) * hx
                yy = (dj + s[None, :]) * hy
                val = g(np.sqrt(xx * xx + yy * yy)).sum() * hx * hy / nsub ** 2
            else:
                xx = (di + 0.5 * gq[:, None]) * hx
                yy = (dj + 0.5 * gq[None, :]) * hy
                val = float((g(np.sqrt(xx * xx + yy * yy)) * W).sum())
            G[di % (2 * nx), dj % (2 * ny)] = val
    G[nx, :] = 0.0
    G[:, ny] = 0.0
    return G


class PeriodicTrainSpaceCharge:
    """Self-field of a periodic bunch train at a common time."""

    def __init__(self, macro_charge_C: float, wavelength_mm: float, *,
                 nx: int = 64, nz_min: int = 32, nz_max: int = 256,
                 extent: float = 4.5, cells_per_sigma_z: float = 6.0,
                 grid_tol: float = 0.02, magnetic: bool = True,
                 cache_size: int = 64, cache_bytes: int = 256 * 1024 ** 2):
        self.qm = float(macro_charge_C)
        self.lam = float(wavelength_mm)
        self.nx = int(nx)
        self.nz_min = int(nz_min)
        self.nz_max = int(nz_max)
        self.extent = float(extent)
        self.cps = float(cells_per_sigma_z)
        self.tol = float(grid_tol)
        self.magnetic = bool(magnetic)
        self._cache: dict = {}
        self._cache_size = int(cache_size)
        # a byte budget too: the spectra of a fine late bunch are ~17 MB
        # each, and the period grows along the RFQ, so old ones are rarely
        # reused (64 of them were 1.1 GB on PXIE)
        self._cache_budget = int(cache_bytes)
        self._cache_nbytes = 0
        self._half = None
        self.n_green = 0
        self.n_solve = 0
        self.last = {}

    # -- Green's functions, snapped to geometric ladders so they can be
    #    reused from one solve to the next ---------------------------------
    def _greens(self, half_wanted, P, nz):
        a = math.log1p(self.tol)
        ih = int(math.ceil(math.log(2.0 * half_wanted / (self.nx - 1)) / a))
        ip = int(round(math.log(P) / a))
        h = math.exp(ih * a)
        Pq = math.exp(ip * a)
        key = (ih, ip, nz)
        hit = self._cache.pop(key, None)         # re-inserted: most recent
        if hit is None:
            from linac_gen.pic.poisson_solver import _guarded_copy
            ks = 2.0 * np.pi * np.arange(nz // 2 + 1) / Pq
            G = np.stack([np.fft.rfft2(_cell_integrated_green(
                self.nx, self.nx, h, h, kk)) for kk in ks])
            # kept where no other array can start or end flush against it,
            # so the spectral product always takes numpy's vectorised loop
            # (repeatable runs; see PIC's _precompute_greens_fft)
            hit = (_guarded_copy(G), ks)
            self.n_green += 1
            nbytes = hit[0].nbytes
            while self._cache and (
                    len(self._cache) >= self._cache_size
                    or self._cache_nbytes + nbytes > self._cache_budget):
                old = self._cache.pop(next(iter(self._cache)))
                self._cache_nbytes -= old[0].nbytes
            self._cache_nbytes += nbytes
        self._cache[key] = hit
        return h, Pq, hit[0], hit[1]

    def field(self, x, y, z, beta_z):
        """E (V/mm) of the periodic beam at the particles (positive charge),
        plus nothing else: callers apply it."""
        from linac_gen.pic.pic_solver import deposit_cic, interpolate_cic
        n = x.size
        P0 = float(np.mean(beta_z)) * self.lam
        ang = 2.0 * np.pi * z / P0
        C, S = float(np.mean(np.cos(ang))), float(np.mean(np.sin(ang)))
        zc = (math.atan2(S, C) * P0 / (2.0 * np.pi)
              if math.hypot(C, S) > 0.05 else float(np.median(z)))
        xc, yc = float(np.mean(x)), float(np.mean(y))
        zf = (z - zc + 0.5 * P0) % P0 - 0.5 * P0
        sigz = 0.5 * (np.percentile(zf, 84) - np.percentile(zf, 16))
        want = self.cps * P0 / max(sigz, 1e-9)
        nz = int(2 ** math.ceil(math.log2(max(want, 1.0))))
        nz = int(min(max(nz, self.nz_min), self.nz_max))
        need = self.extent * max(float(np.std(x)), float(np.std(y)), 1e-3)
        if (self._half is None or need > self._half
                or need < 0.75 * self._half):
            self._half = need
        h, Pq, G, ks = self._greens(self._half, P0, nz)
        # the grid and the fold use the exact period; the Green's functions
        # come from the nearest ladder period (k off by <= 1 %, while a
        # mismatched fold would leave a density step in a DC beam)
        P = P0
        ks = ks * (Pq / P0)
        half = 0.5 * h * (self.nx - 1)
        nx = ny = self.nx
        x0, y0 = xc - half, yc - half
        zlo = zc - 0.5 * P
        # positions folded into one period; z in [zlo, zlo + P)
        zp = zlo + (z - zlo) % P
        inside = ((x >= x0) & (x <= x0 + 2 * half)
                  & (y >= y0) & (y <= y0 + 2 * half))
        pos = np.column_stack([x[inside], y[inside], zp[inside]])
        gmin = np.array([x0, y0, zlo])
        gmax = np.array([x0 + 2 * half, y0 + 2 * half, zlo + P])
        ngrid = np.array([nx, ny, nz + 1], np.int64)
        rho = deposit_cic(pos, np.full(pos.shape[0], self.qm), gmin, gmax,
                          ngrid)
        rho = np.asarray(rho)
        rho[:, :, 0] += rho[:, :, -1]                  # periodic fold
        rho = rho[:, :, :-1]
        rk = np.fft.rfft(rho, axis=2) / nz                  # (nx, ny, M)
        M = rk.shape[2]
        # all longitudinal modes in one batched 2-D convolution; each
        # transform runs on one thread, so the result does not depend on
        # the thread count
        from scipy import fft as sfft
        from linac_gen.parallel.threads import get_threads
        wk = max(1, int(get_threads()))
        pad = np.zeros((2, M, 2 * nx, 2 * ny))
        pad[0, :, :nx, :ny] = np.moveaxis(rk.real, 2, 0)
        pad[1, :, :nx, :ny] = np.moveaxis(rk.imag, 2, 0)
        spec = sfft.rfft2(pad, axes=(2, 3), workers=wk)
        # in place: numpy rounds a complex product differently (no fused
        # multiply-add) when a fresh output lands flush against an input
        np.multiply(spec, G[None, :, :, :], out=spec)
        conv = sfft.irfft2(spec, s=(2 * nx, 2 * ny), axes=(2, 3),
                           workers=wk)[:, :, :nx, :ny]
        phik = np.moveaxis(conv[0] + 1j * conv[1], 0, 2) / EPS0_MM
        phi = np.fft.irfft(phik * nz, n=nz, axis=2)
        Ez = np.fft.irfft(-1j * ks[None, None, :] * phik * nz, n=nz, axis=2)
        Ex = -np.gradient(phi, h, axis=0)
        Ey = -np.gradient(phi, h, axis=1)
        # one extra slice (= the first) so the gather sees the period's end
        Ex = np.concatenate([Ex, Ex[:, :, :1]], axis=2)
        Ey = np.concatenate([Ey, Ey[:, :, :1]], axis=2)
        Ez = np.concatenate([Ez, Ez[:, :, :1]], axis=2)
        ex = np.zeros(n)
        ey = np.zeros(n)
        ez = np.zeros(n)
        if pos.shape[0]:
            E = np.asarray(interpolate_cic(
                np.ascontiguousarray(Ex), np.ascontiguousarray(Ey),
                np.ascontiguousarray(Ez), pos, gmin, gmax, ngrid))
            ex[inside], ey[inside], ez[inside] = E[:, 0], E[:, 1], E[:, 2]
        out = ~inside
        if out.any():
            lam_line = self.qm * int(inside.sum()) / P      # C/mm
            dx = x[out] - xc
            dy = y[out] - yc
            r2 = dx * dx + dy * dy
            f = lam_line / (2.0 * np.pi * EPS0_MM * r2)
            ex[out] = f * dx
            ey[out] = f * dy
        self.n_solve += 1
        self.last = dict(period_mm=P, half_mm=half, nz=nz,
                         n_outside=int(out.sum()))
        return ex, ey, ez

    def kick(self, X, Y, Z, ux, uy, uz, active, dt_ns, k_e):
        """Apply Δ(γβ) = k_e·E·dt (|q| E for either species) to the active
        particles; ``k_e`` converts V/mm to 1/ns of γβ."""
        idx = np.flatnonzero(active)
        if idx.size < 2 or self.qm == 0.0:
            return
        g = np.sqrt(1.0 + ux[idx] ** 2 + uy[idx] ** 2 + uz[idx] ** 2)
        bz = uz[idx] / g
        ex, ey, ez = self.field(X[idx], Y[idx], Z[idx], bz)
        ft = k_e * dt_ns
        fm = (1.0 - bz * bz) if self.magnetic else 1.0   # v×B of the beam
        ux[idx] += ft * ex * fm
        uy[idx] += ft * ey * fm
        uz[idx] += ft * ez
