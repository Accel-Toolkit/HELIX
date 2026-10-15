"""Electrode geometry of a four-vane RFQ (time-domain solver).

Coordinates are the RFQ chain's own: z = 0 at the entrance of the first
``RfqCell`` (the inner face of the entrance plate), lengths in mm,
voltages in V at the RF crest.

Each vane tip is a circular arc of transverse radius ρ(z) whose closest
point lies at the tabulated tip distance R(z) from the axis; beyond the
arc's tangent point the vane body continues with flat flanks opening at
the break-out angle (0° = parallel flanks), the PARMULT/Toutatis
"constant transverse curvature" vane.  Vane metal exists only inside
the tabulated z span.  The entrance and exit end plates are grounded
discs with a circular beam hole.

The quadrant field solver needs mirror symmetry (the −x pole equal to
the +x pole, −y equal to +y); :meth:`RfqElectrodes.check_symmetric`
refuses anything else.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


#: end plates: PXIE's 6.0 mm hole for its r0 = 5.576 mm, scaled with r0
_PXIE_HOLE_MM, _PXIE_R0_MM = 6.0, 5.576
#: opposite poles must agree to this (mm): far below the 0.1 mm grid, and
#: above the rounding of a vane table printed to a micrometre
SYMMETRY_TOL_MM = 1e-3


def end_plate_hole(r0_mm: float) -> float:
    """End-plate hole radius (mm) for an RFQ of mean aperture ``r0_mm``:
    PXIE's proportion (6.0 mm at 5.576 mm; exactly 6.0 there)."""
    return _PXIE_HOLE_MM * (float(r0_mm) / _PXIE_R0_MM)


@dataclass(frozen=True)
class EndPlate:
    """Grounded end plate: metal on one side of ``z_face`` over
    ``thickness`` mm, outside a beam hole of radius ``hole_radius``."""

    z_face: float            # mm, the face looking at the vanes
    side: int                # -1: metal at z < z_face (entrance); +1: exit
    thickness: float = 2.5   # mm
    hole_radius: float = 6.0  # mm
    voltage: float = 0.0     # V

    def sdf(self, x, y, z):
        zc = self.z_face + self.side * 0.5 * self.thickness
        r = np.sqrt(x * x + y * y)
        return np.maximum(np.abs(z - zc) - 0.5 * self.thickness,
                          self.hole_radius - r)


@dataclass
class RfqElectrodes:
    """Tabulated four-vane geometry plus end plates."""

    z: np.ndarray            # (N,) mm, strictly increasing
    R_x: np.ndarray          # (N,) mm  tip distance of the +x pole
    R_y: np.ndarray          # (N,) mm  tip distance of the +y pole
    rho_x: np.ndarray        # (N,) mm  transverse tip radius, +x pole
    rho_y: np.ndarray        # (N,) mm
    V_x: float               # V, +x (and -x) pole at the RF crest
    V_y: float               # V, +y (and -y) pole
    R_mx: np.ndarray = None  # -x pole (defaults to R_x)
    R_my: np.ndarray = None
    plates: list = field(default_factory=list)
    breakout_deg: float = 0.0
    source: str = ""

    def __post_init__(self):
        self.z = np.ascontiguousarray(self.z, dtype=float)
        for name in ("R_x", "R_y", "rho_x", "rho_y"):
            setattr(self, name, np.ascontiguousarray(getattr(self, name),
                                                     dtype=float))
        if self.R_mx is None:
            self.R_mx = self.R_x.copy()
        if self.R_my is None:
            self.R_my = self.R_y.copy()
        if self.z.ndim != 1 or self.z.size < 2 \
                or np.any(np.diff(self.z) <= 0):
            raise ValueError("electrode table z must be 1-D and strictly "
                             "increasing")
        n = self.z.size
        for name in ("R_x", "R_y", "rho_x", "rho_y", "R_mx", "R_my"):
            a = getattr(self, name)
            if a.shape != (n,):
                raise ValueError(f"{name} has shape {a.shape}, expected ({n},)")
            if not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError(f"{name} must be finite and positive")

    # ------------------------------------------------------------------
    @classmethod
    def from_vane(cls, vane, *, z_offset_mm: float = 0.0,
                  plates: Sequence[EndPlate] = (), breakout_deg: float = 0.0,
                  source: str = "") -> "RfqElectrodes":
        """From a parsed :class:`~linac_gen.io.tracewin_vane.VaneGeometry`
        (file pole numbering: 1 = +y, 2 = +x, 3 = −y, 4 = −x — the plane
        accessors of that class).  ``z_offset_mm`` shifts the table into
        chain coordinates (0 for a file whose z starts at the RFQ
        entrance, the TraceWin convention)."""
        z = np.asarray(vane.z, float) * 1e3 + z_offset_mm
        volts = {k: np.asarray(getattr(vane, f"voltage_{k}"), float)
                 for k in ("x", "y", "mx", "my")}
        for k, v in volts.items():
            if np.ptp(v) > 1e-9 * max(abs(v).max(), 1.0):
                raise ValueError("vane voltages vary along z; the "
                                 "time-domain solver needs one voltage per "
                                 "pole pair")
        vx, vy = volts["x"], volts["y"]
        for a, b, name in ((vx, volts["mx"], "x"), (vy, volts["my"], "y")):
            if abs(a[0] - b[0]) > 1e-6 * max(abs(a[0]), 1.0):
                raise ValueError(
                    f"the ±{name} poles have different voltages ({a[0]:g}, "
                    f"{b[0]:g} V): the time-domain solver needs mirror-"
                    "symmetric pole pairs")
        for a, b, name in ((vane.Tc_x, vane.Tc_mx, "x"),
                           (vane.Tc_y, vane.Tc_my, "y")):
            d = np.abs(np.asarray(a, float) - np.asarray(b, float)) * 1e3
            if d.max() > SYMMETRY_TOL_MM:
                raise ValueError(
                    f"the ±{name} poles have different tip radii (max "
                    f"{d.max():.3g} mm): the time-domain solver needs "
                    "mirror-symmetric pole pairs")
        return cls(z=z,
                   R_x=np.asarray(vane.aperture_x, float) * 1e3,
                   R_y=np.asarray(vane.aperture_y, float) * 1e3,
                   R_mx=np.asarray(vane.aperture_mx, float) * 1e3,
                   R_my=np.asarray(vane.aperture_my, float) * 1e3,
                   rho_x=np.asarray(vane.Tc_x, float) * 1e3,
                   rho_y=np.asarray(vane.Tc_y, float) * 1e3,
                   V_x=float(vx[0]), V_y=float(vy[0]),
                   plates=list(plates), breakout_deg=breakout_deg,
                   source=source)

    # ------------------------------------------------------------------
    @property
    def z_span(self):
        return float(self.z[0]), float(self.z[-1])

    def check_symmetric(self, tol_mm: float = SYMMETRY_TOL_MM) -> None:
        """The quadrant solver assumes mirror-symmetric pole pairs (tip
        distances equal to ``tol_mm``; tip radii and voltages are checked
        by :meth:`from_vane`)."""
        for a, b, name in ((self.R_x, self.R_mx, "x"), (self.R_y, self.R_my, "y")):
            if np.abs(a - b).max() > tol_mm:
                raise ValueError(
                    f"the ±{name} poles differ (max {np.abs(a - b).max():.3g} "
                    "mm): displaced or deformed vanes need a full-domain "
                    "field solve, which the time-domain solver does not "
                    "have yet")

    def content_hash(self) -> str:
        """SHA-256 of everything the field depends on."""
        h = hashlib.sha256()
        for a in (self.z, self.R_x, self.R_y, self.rho_x, self.rho_y,
                  self.R_mx, self.R_my):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(repr((round(self.V_x, 9), round(self.V_y, 9),
                       round(self.breakout_deg, 9),
                       [(p.z_face, p.side, p.thickness, p.hole_radius,
                         p.voltage) for p in self.plates])).encode())
        return h.hexdigest()

    # ------------------------------------------------------------------
    def _pole_sdf(self, u, w, z, R, rho):
        """Signed distance to one pole in its own (u along the pole, w
        across) frame; negative inside the metal."""
        Rz = np.interp(z, self.z, R)
        rz = np.interp(z, self.z, rho)
        uc = Rz + rz
        a = np.radians(self.breakout_deg)
        ut = uc - rz * np.sin(a)
        circ = np.sqrt((u - uc) ** 2 + w ** 2) - rz
        body = (np.abs(w) - rz * np.cos(a) - (u - ut) * np.tan(a)) * np.cos(a)
        d = np.where(u >= ut, body, circ)
        zcut = np.maximum(self.z[0] - z, z - self.z[-1])
        return np.maximum(d, zcut)

    def sdf_quadrant(self, x, y, z):
        """Level sets in the quadrant x, y >= 0: (d_min, which, V) with
        which = 0 (+x pole), 1 (+y pole), 2+k (plate k)."""
        ds = [self._pole_sdf(x, y, z, self.R_x, self.rho_x),
              self._pole_sdf(y, x, z, self.R_y, self.rho_y)]
        vs = [self.V_x, self.V_y]
        for p in self.plates:
            ds.append(p.sdf(x, y, z))
            vs.append(p.voltage)
        D = np.stack(np.broadcast_arrays(*ds))
        which = np.argmin(D, axis=0)
        dmin = np.take_along_axis(D, which[None], 0)[0]
        return dmin, which, np.asarray(vs)[which]

    def inside(self, x, y, z) -> np.ndarray:
        """True where a point (any quadrant) is inside metal.

        Only points that can be in metal get the full test: no vane comes
        closer to the axis than its tip, so a point nearer the axis than
        every tip of its z-bin, and outside the end plates' slabs or within
        their holes, is in the bore (the same answer, at a fraction of the
        cost -- most particles are far from the vanes)."""
        x = np.abs(np.asarray(x, dtype=float))
        y = np.abs(np.asarray(y, dtype=float))
        z = np.asarray(z, dtype=float)
        x, y, z = np.broadcast_arrays(x, y, z)
        r2 = x * x + y * y
        z0, inv_dz, rmin2 = self._bore()
        # uniform bins: a point within rounding of a bin edge may land in
        # the neighbouring bin, whose bound also covers that edge
        b = np.clip(((z - z0) * inv_dz).astype(np.int64), 0, rmin2.size - 1)
        cand = (z >= self.z[0]) & (z <= self.z[-1]) & (r2 >= rmin2[b])
        for p in self.plates:
            zc = p.z_face + p.side * 0.5 * p.thickness
            hole = max(p.hole_radius - 1e-9, 0.0)
            cand |= ((np.abs(z - zc) <= 0.5 * p.thickness + 1e-9)
                     & (r2 >= hole * hole))
        out = np.zeros(z.shape, dtype=bool)
        idx = np.flatnonzero(cand)
        if idx.size:
            d, _, _ = self.sdf_quadrant(x.ravel()[idx], y.ravel()[idx],
                                        z.ravel()[idx])
            out.ravel()[idx] = d < 0.0
        return out

    def _bore(self):
        """(z of the first bin edge, 1 / bin width, squared radius per
        bin) such that no vane point lies closer to the axis than that
        radius within the bin: the minimum tip distance over the table
        points in the bin and the first ones beyond both its edges (the
        linear interpolant never dips below its end points), less a
        rounding margin.  Built once."""
        cached = getattr(self, "_bore_cache", None)
        if cached is not None:
            return cached
        zt = self.z
        rmin = np.minimum.reduce([self.R_x, self.R_y, self.R_mx, self.R_my])
        nb = max(1, int(np.ceil((zt[-1] - zt[0]) / 1.0)))       # ~1 mm bins
        dz = (zt[-1] - zt[0]) / nb
        edges = zt[0] + dz * np.arange(nb + 1)
        lo = np.clip(np.searchsorted(zt, edges[:-1], side="right") - 2, 0,
                     zt.size - 1)
        hi = np.clip(np.searchsorted(zt, edges[1:], side="left") + 1, 0,
                     zt.size - 1)
        r = np.array([rmin[a:b + 1].min() for a, b in zip(lo, hi)])
        r = r * (1.0 - 1e-9)
        self._bore_cache = (float(zt[0]), 1.0 / dz, r * r)
        return self._bore_cache

    def min_aperture(self, z0: float, z1: float) -> float:
        """Smallest tip distance over [z0, z1] (mm); inf outside the
        tabulated span."""
        m = (self.z >= z0) & (self.z <= z1)
        if not m.any():
            lo = np.interp([z0, z1], self.z, np.minimum(self.R_x, self.R_y))
            return float(lo.min()) if z1 >= self.z[0] and z0 <= self.z[-1] \
                else float("inf")
        return float(min(self.R_x[m].min(), self.R_y[m].min()))
