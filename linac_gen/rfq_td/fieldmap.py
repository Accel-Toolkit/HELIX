"""Near-axis potential of the RFQ electrodes, stored per z window.

A map is a list of blocks, each uniform in z, all on the transverse grid
x, y = 0, h, 2h … (the quadrant; the vane pairs are mirror images).  The
potential is the static one at the RF crest (vane voltages as in the
geometry); the tracker multiplies the field by the RF time dependence.

Evaluation is tricubic (Catmull–Rom, C¹) on φ with E = −∇φ taken
analytically from the interpolant, so the field is curl-free by
construction and reproduces any quadratic potential exactly (the
quadrupole term).  Mirror symmetry supplies the ghost nodes at x, y < 0.

Maps are cached on disk under :func:`cache_dir`, keyed by a SHA-256 of
the electrode geometry, the cell edges, the solver settings and
:data:`MAP_VERSION` — never by file name or modification time.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: bump when the solver or the stored layout changes meaning
MAP_VERSION = 1

_KERNEL = None


def _kernel():
    """The compiled field gather (``linac_gen._rfq_td_kernels``), or False.
    ``LINAC_GEN_RFQTD_KERNEL=0`` forces the numpy reference."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = False
        if os.environ.get("LINAC_GEN_RFQTD_KERNEL", "1") != "0":
            try:
                # torch's OpenMP runtime first: the kernel binds to it
                import linac_gen.pic.pic_solver  # noqa: F401
                from linac_gen import _rfq_td_kernels as k
                from linac_gen.parallel.threads import sync_kernels
                sync_kernels()
                _KERNEL = k
            except ImportError:
                _KERNEL = False
    return _KERNEL


def cache_dir() -> Path:
    """Where time-domain RFQ field maps are cached: ``$HELIX_CACHE_DIR``
    (tests point it at a temporary directory), else the platform's user
    cache directory."""
    env = os.environ.get("HELIX_CACHE_DIR")
    if env:
        base = Path(env)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "HELIX"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA",
                                   Path.home() / "AppData" / "Local")) \
            / "HELIX" / "Cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME",
                                   Path.home() / ".cache")) / "helix"
    return base / "rfq_td"


@dataclass
class RfqFieldMap:
    """Blocks of stored potential; see the module docstring."""

    h: float                  # transverse step (mm), all blocks
    core0: np.ndarray         # (B,) z where block b's core starts (mm)
    core1: np.ndarray         # (B,) z where it ends
    z0: np.ndarray            # (B,) z of block b's first stored slice
    hz: np.ndarray            # (B,) longitudinal step
    nz: np.ndarray            # (B,) stored slices
    nx: np.ndarray            # (B,) stored transverse nodes (x = y)
    off: np.ndarray           # (B,) offset of block b in ``phi``
    phi: np.ndarray           # flat float32, block b laid out [ix, iy, iz]
    key: str = ""

    @classmethod
    def from_blocks(cls, h, cores, blocks, key=""):
        """``cores``: [(core0, core1)], ``blocks``: [(z_first, hz, phi3)]."""
        order = np.argsort([c[0] for c in cores])
        core0, core1, z0, hz, nz, nx, off, data = [], [], [], [], [], [], [], []
        o = 0
        for i in order:
            zf, dz, p = blocks[i]
            p = np.ascontiguousarray(p, dtype=np.float32)
            if p.shape[0] != p.shape[1]:
                raise ValueError("blocks must be square in x, y")
            core0.append(float(cores[i][0]))
            core1.append(float(cores[i][1]))
            z0.append(float(zf))
            hz.append(float(dz))
            nz.append(p.shape[2])
            nx.append(p.shape[0])
            off.append(o)
            data.append(p.ravel())
            o += p.size
        core0 = np.array(core0)
        core1 = np.array(core1)
        if np.any(np.abs(core0[1:] - core1[:-1]) > 1e-6):
            raise ValueError("field-map blocks are not contiguous")
        return cls(h=float(h), core0=core0, core1=core1,
                   z0=np.array(z0), hz=np.array(hz),
                   nz=np.array(nz, np.int64), nx=np.array(nx, np.int64),
                   off=np.array(off, np.int64),
                   phi=np.concatenate(data), key=key)

    @property
    def z_span(self):
        return float(self.core0[0]), float(self.core1[-1])

    def scaled(self, factor: float) -> "RfqFieldMap":
        """The same map with every potential multiplied by ``factor`` (a
        vane voltage different from the geometry's)."""
        return RfqFieldMap(h=self.h, core0=self.core0, core1=self.core1,
                           z0=self.z0, hz=self.hz, nz=self.nz, nx=self.nx,
                           off=self.off,
                           phi=(self.phi * np.float32(factor)),
                           key=f"{self.key}*{factor!r}")

    def nbytes(self) -> int:
        return int(self.phi.nbytes)

    # ---- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
        os.close(fd)
        try:
            np.savez(tmp, h=self.h, core0=self.core0, core1=self.core1,
                     z0=self.z0, hz=self.hz, nz=self.nz, nx=self.nx,
                     off=self.off, phi=self.phi,
                     meta=json.dumps({"key": self.key,
                                      "version": MAP_VERSION}))
            try:
                os.chmod(tmp, 0o644)        # mkstemp makes it private
            except OSError:
                pass
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path: Path) -> "RfqFieldMap":
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("version") != MAP_VERSION:
                raise ValueError(f"field map {path} has version "
                                 f"{meta.get('version')}, need {MAP_VERSION}")
            return cls(h=float(f["h"]), core0=f["core0"], core1=f["core1"],
                       z0=f["z0"], hz=f["hz"], nz=f["nz"].astype(np.int64),
                       nx=f["nx"].astype(np.int64),
                       off=f["off"].astype(np.int64), phi=f["phi"],
                       key=meta.get("key", ""))

    # ---- evaluation -------------------------------------------------------
    def field(self, x, y, z, active=None, *, use_kernel=True):
        """Static field (V/mm) at points (mm): ``(Ex, Ey, Ez, n_outside)``.
        Points outside the stored region (beyond the near-axis box or the
        z span) get zero field and are counted so callers can warn --
        except upstream of the first block, which lies before the
        entrance plate and is field-free (particles start there)."""
        x = np.ascontiguousarray(x, float)
        y = np.ascontiguousarray(y, float)
        z = np.ascontiguousarray(z, float)
        n = x.size
        Ex = np.zeros(n)
        Ey = np.zeros(n)
        Ez = np.zeros(n)
        sel = np.ones(n, bool) if active is None else np.asarray(active, bool)
        k = _kernel() if use_kernel else False
        if k:
            n_out = k.field_gather(x, y, z, sel.view(np.uint8) if sel.dtype == bool
                                   else sel.astype(np.uint8), self.h,
                                   self.core0, float(self.core1[-1]), self.z0,
                                   self.hz, self.nz, self.nx, self.off,
                                   self.phi, Ex, Ey, Ez)
            return Ex, Ey, Ez, int(n_out)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            return Ex, Ey, Ez, 0
        zz = z[idx]
        b = np.searchsorted(self.core0, zz, side="right") - 1
        upstream = b < 0
        inz = (~upstream) & (zz <= self.core1[-1])
        b = np.clip(b, 0, self.core0.size - 1)
        ax = np.abs(x[idx]) / self.h
        ay = np.abs(y[idx]) / self.h
        nxb = self.nx[b]
        ok = inz & (ax < nxb - 2) & (ay < nxb - 2)
        n_out = int(idx.size - ok.sum() - upstream.sum())
        if not ok.any():
            return Ex, Ey, Ez, n_out
        idx, b, ax, ay, zz, nxb = idx[ok], b[ok], ax[ok], ay[ok], zz[ok], nxb[ok]
        ix = ax.astype(np.int64)
        iy = ay.astype(np.int64)
        tx = ax - ix
        ty = ay - iy
        fz = (zz - self.z0[b]) / self.hz[b]
        iz = np.clip(fz.astype(np.int64), 1, self.nz[b] - 3)
        tz = fz - iz
        wx, dwx = _catmull_rom(tx)
        wy, dwy = _catmull_rom(ty)
        wz, dwz = _catmull_rom(tz)
        gx = np.zeros(idx.size)
        gy = np.zeros(idx.size)
        gz = np.zeros(idx.size)
        nzb = self.nz[b]
        base = self.off[b]
        for a in range(4):
            ii = np.abs(ix - 1 + a)              # mirror: index -1 -> 1
            for c in range(4):
                jj = np.abs(iy - 1 + c)
                row = base + (ii * nxb + jj) * nzb + iz - 1
                wxy = wx[a] * wy[c]
                dxy = dwx[a] * wy[c]
                ydx = wx[a] * dwy[c]
                for e in range(4):
                    v = self.phi[row + e].astype(np.float64)
                    gx += dxy * wz[e] * v
                    gy += ydx * wz[e] * v
                    gz += wxy * dwz[e] * v
        Ex[idx] = -gx / self.h * np.where(x[idx] >= 0, 1.0, -1.0)
        Ey[idx] = -gy / self.h * np.where(y[idx] >= 0, 1.0, -1.0)
        Ez[idx] = -gz / self.hz[b]
        return Ex, Ey, Ez, n_out

    def push(self, X, Y, Z, ux, uy, uz, active, Ex, Ey, Ez, c1, c2, cdt):
        """One leapfrog step of the active particles, in place, by the
        compiled kernel (``rfq_td_kernels.cpp: push`` -- the numpy step of
        :class:`~linac_gen.rfq_td.tracker.TimeDomainTracker` and
        :meth:`field` in one pass).  ``Ex, Ey, Ez`` hold the field at the
        current positions and receive the field at the new ones.  Returns
        the active particles outside the stored region, or None when the
        kernel is unavailable (the caller then takes the numpy step)."""
        k = _kernel()
        if not k or not hasattr(k, "push"):
            return None
        arrs = (X, Y, Z, ux, uy, uz, Ex, Ey, Ez)
        if not all(isinstance(a, np.ndarray) and a.dtype == np.float64
                   and a.flags.c_contiguous and a.flags.writeable
                   for a in arrs) or active.dtype != bool \
                or not active.flags.c_contiguous:
            return None                 # the kernel would work on copies
        return int(k.push(X, Y, Z, ux, uy, uz, active.view(np.uint8),
                          Ex, Ey, Ez, float(c1), float(c2), float(cdt),
                          self.h, self.core0, float(self.core1[-1]),
                          self.z0, self.hz, self.nz, self.nx, self.off,
                          self.phi))


def _catmull_rom(t):
    """Weights and their derivatives for the 4-point Catmull–Rom spline."""
    t2 = t * t
    t3 = t2 * t
    w = (0.5 * (-t3 + 2 * t2 - t), 0.5 * (3 * t3 - 5 * t2 + 2),
         0.5 * (-3 * t3 + 4 * t2 + t), 0.5 * (t3 - t2))
    d = (0.5 * (-3 * t2 + 4 * t - 1), 0.5 * (9 * t2 - 10 * t),
         0.5 * (-9 * t2 + 8 * t + 1), 0.5 * (3 * t2 - 2 * t))
    return w, d


def map_key(electrodes, edges_mm, settings: dict) -> str:
    """Content hash identifying a field map."""
    h = hashlib.sha256()
    h.update(f"rfqtd-map-v{MAP_VERSION}".encode())
    h.update(electrodes.content_hash().encode())
    h.update(np.ascontiguousarray(edges_mm, dtype=np.float64).tobytes())
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()
