"""Curved field-map clusters — TraceWin ``SUPERPOSE_MAP_OUT``.

A ``SUPERPOSE_MAP_OUT Z X Y θZ θX θY`` card before a set of superposed
field maps declares that the maps DEFLECT the beam (a dipole given as a
3-D field map, alone or with other maps around it).  TraceWin's manual
("Field map with curved reference trajectory", "Positioning of field
maps"):

* the card gives the exit point (mm) and the exit reference direction
  (deg; rotations about Z, then Y, then X) in the cluster's entrance
  frame; with it, every child's ``SUPERPOSE_MAP Zi Xi Yi θZi θXi θYi``
  placement counts (without it only Zi does — :class:`SuperposedFieldMap`);
* "TraceWin calculates the deviated reference trajectory until it crosses
  the exit plane.  The beam dynamics is then calculated around this
  reference trajectory.  At the exit of calculation, TraceWin displaces
  and rotates the beam according to the positions and angles given with
  SUPERPOSE_MAP_OUT" — a mis-set exit frame shows as a position/angle
  shift of the beam;
* inside the maps the beam is described in planes orthogonal to the
  reference trajectory, and s runs along that trajectory.

:class:`CurvedFieldMapCluster` does exactly that:

1. :func:`integrate_reference` traces the reference particle (RK4 in its
   path length) through the summed child fields, from the entrance origin
   along +z until it crosses the card's exit plane, and records a
   rotation-minimising frame (t, e1, e2) with the curvatures κ1, κ2
   (t' = κ1 e1 + κ2 e2) along it — any bending angle, also beyond 90°.
2. Particles are pushed in that path length s, in the local planes, with
   the exact equations of motion in a static magnetic field (no small-angle
   approximation; HELIX coordinates x' = u1/ut, y' = u2/ut stay exact).
3. At the exit the particles, which sit on the plane orthogonal to the
   reference at its crossing point, are moved rigidly into the card's frame
   (each drifted straight onto the card plane), so the mismatch between the
   traced reference and the card stays as a beam offset, as in TraceWin.

Sign conventions (pinned by a TraceWin project whose 30° switching-dipole
map leaves toward −x with θY = +30° on the exit card — the traced path and
exit offset reproduce TraceWin's — and by the manual's own example, figure
"field_map_dipole", where ``SUPERPOSE_MAP_OUT 1573.29 164.27 0 0 0
-9.59023`` and ``SUPERPOSE_MAP 1100 84.27 0 0 0 9.59023`` both turn toward
+X): a child's angles are an active rotation of its frame with θY > 0
turning its z axis toward +X; the exit card's angles carry the opposite
sign.  θX (> 0: z toward +Y) and θZ (right-handed about z) are read from
the manual's figure only — the parser warns when they are used.

The push has a compiled twin, ``curved_rk4_steps`` in
``csrc/fieldmap_kernels.cpp``, BITWISE equal to :func:`curved_rk4_numpy`
(same operations in the same order, only sqrt and division, no FP
contraction); ``LINAC_GEN_CURVED_KERNEL=0`` selects the numpy twin.
Static magnetic 3-D maps only (an electric or RF map in a curved cluster
is refused), sampled trilinearly; no backtracking (TraceWin offers none
either).  The move into the card frame is a straight drift: the field
between the reference's exit plane and the card plane is not felt (an
error of about x·α/ρ for a card mismatched by α).  The phase keeps each
particle's exact path: a particle at an angle to the reference falls
behind it by (1/cos − 1) of the path even without a field, which HELIX's
straight elements, linear in the angles, leave out.
"""
from __future__ import annotations

import hashlib
import math
import os as _os
import warnings

import numpy as np

from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.superposed_field_map import SuperposedFieldMap

__all__ = ["CurvedFieldMapCluster", "ReferenceTrajectory", "child_rotation",
           "exit_rotation", "integrate_reference", "curved_rk4_numpy",
           "curved_kernel_available", "has_curved_elements",
           "prepare_curved_elements"]

# (MeV/c) per (T·m): 1/(Bρ) [1/(T·m)] = 299.792458 / p[MeV/c]
_C_CONV = 299.792458

# Stage-frame record: r0 (3), t (3), e1 (3), e2 (3), κ1, κ2
_NFRAME = 14

try:  # noqa: SIM105
    import torch  # type: ignore  # noqa: F401  (one OpenMP image, see field_map_3d)
except Exception:                                     # pragma: no cover
    pass
else:
    # a run that loads torch here keeps its CPU work on one thread
    from linac_gen.parallel.threads import sync_torch as _sync_torch
    _sync_torch()
try:
    from linac_gen._fieldmap_kernels import curved_rk4_steps as _curved_kernel
except ImportError:                                   # pragma: no cover
    _curved_kernel = None
try:
    from linac_gen._fieldmap_kernels import interp3_multi as _interp3_multi
except ImportError:                                   # pragma: no cover
    _interp3_multi = None
_USE_KERNEL = (_curved_kernel is not None
               and _os.environ.get("LINAC_GEN_CURVED_KERNEL", "1") != "0")


def curved_kernel_available() -> bool:
    """True when the compiled push is importable."""
    return _curved_kernel is not None


def use_curved_kernel(enabled: bool) -> None:
    """Switch the compiled push on/off (results are identical either way)."""
    global _USE_KERNEL
    _USE_KERNEL = bool(enabled) and _curved_kernel is not None


# --------------------------------------------------------------------------- #
#  Frames
# --------------------------------------------------------------------------- #
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def child_rotation(theta_z: float, theta_x: float, theta_y: float) -> np.ndarray:
    """Axes of a map placed with ``SUPERPOSE_MAP … θZ θX θY`` (deg), as
    columns in the cluster frame (v_cluster = R @ v_map).

    Rotations applied about Z, then Y, then X (TraceWin).  θY > 0 turns the
    map's z axis toward +X (the manual's dipole + quadrupole example);
    θX > 0 toward +Y; θZ is right-handed about z."""
    rz = _rot_z(math.radians(float(theta_z)))
    ry = _rot_y(math.radians(float(theta_y)))
    rx = _rot_x(-math.radians(float(theta_x)))
    return rx @ ry @ rz


def exit_rotation(theta_z: float, theta_x: float, theta_y: float) -> np.ndarray:
    """Axes of the ``SUPERPOSE_MAP_OUT`` exit frame (columns e1, e2, t in
    the entrance frame): the card's angles carry the opposite sign to a
    child's — θY > 0 turns the exit frame toward −X (a 30° switching
    dipole: +30° with the exit at x < 0; the manual's example: −9.59° at
    x = +164.27 mm)."""
    return child_rotation(-float(theta_z), -float(theta_x), -float(theta_y))


# --------------------------------------------------------------------------- #
#  The summed field of the children (static magnetic, cluster frame)
# --------------------------------------------------------------------------- #
class _ChildPack:
    """One static-magnetic channel of one child map, with its placement."""

    __slots__ = ("gx", "gy", "gz", "fields", "slots", "amp", "R", "o", "zstart",
                 "child", "ch_enum")

    def __init__(self, gx, gy, gz, fields, slots, amp, R, o, zstart, child, ch_enum):
        self.gx, self.gy, self.gz = gx, gy, gz
        self.fields = fields              # (M, nx, ny, nz) C-contiguous
        self.slots = slots                # rows of Fx, Fy, Fz in fields, or -1
        self.amp = float(amp)             # scale · kb_eff / norm
        self.R = np.ascontiguousarray(R, dtype=float)
        self.o = np.ascontiguousarray(o, dtype=float)
        self.zstart = float(zstart)
        self.child = child
        self.ch_enum = ch_enum

    def as_tuple(self):
        return (self.gx, self.gy, self.gz, self.fields,
                np.asarray(self.slots, dtype=np.int64), self.amp,
                self.R, self.o, self.zstart)


def _stack_channel(ch):
    names = [c for c in ("Fx", "Fy", "Fz") if getattr(ch, c, None) is not None]
    stacked = np.ascontiguousarray(np.stack(
        [np.asarray(getattr(ch, c), dtype=float) for c in names]))
    slots = [names.index(c) if c in names else -1 for c in ("Fx", "Fy", "Fz")]
    ax = [np.ascontiguousarray(a, dtype=float) for a in (ch.x, ch.y, ch.z)]
    return ax, stacked, slots


def _map_fingerprint(child) -> str:
    """Content fingerprint of a child's magnetic map (axes, components,
    normalisation), kept on the child while the same arrays stay attached.
    The trace cache keys on it rather than on object identity, so a copy of
    the lattice (deepcopy, a worker process) reuses the trace and a changed
    map is traced again."""
    arrays = []
    for ch_enum, ch in child.field_data.channels.items():
        arrays.append((ch_enum.name, float(ch.norm_factor),
                       [getattr(ch, a, None)
                        for a in ("x", "y", "z", "Fx", "Fy", "Fz")]))
    ids = tuple((nm, nf, tuple(id(a) for a in arrs)) for nm, nf, arrs in arrays)
    cached = getattr(child, "_curved_fingerprint", None)
    if cached is not None and cached[0] == ids:
        return cached[1]
    h = hashlib.blake2b(digest_size=16)
    for nm, nf, arrs in arrays:
        h.update(nm.encode())
        h.update(repr(nf).encode())
        for a in arrs:
            if a is None:
                h.update(b"-")
                continue
            a = np.ascontiguousarray(a, dtype=np.float64)
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
    fp = h.hexdigest()
    child._curved_fingerprint = (ids, fp)
    return fp


def _sample(pack: _ChildPack, xs, ys, zs):
    """Trilinear (Fx, Fy, Fz) of one pack at map-frame points — the compiled
    sampler when present (bitwise equal to scipy RGI linear, fill 0)."""
    n = xs.shape[0]
    if _interp3_multi is not None:
        out = _interp3_multi(np.ascontiguousarray(xs), np.ascontiguousarray(ys),
                             np.ascontiguousarray(zs), pack.gx, pack.gy, pack.gz,
                             pack.fields)
    else:                                             # pragma: no cover
        from scipy.interpolate import RegularGridInterpolator as RGI
        pts = np.column_stack([xs, ys, zs])
        out = np.stack([RGI((pack.gx, pack.gy, pack.gz), f, method="linear",
                            bounds_error=False, fill_value=0.0)(pts)
                        for f in pack.fields])
    comp = []
    for k in range(3):
        j = pack.slots[k]
        comp.append(out[j] if j >= 0 else np.zeros(n))
    return comp


def cluster_field(packs, rx, ry, rz):
    """Summed (Bx, By, Bz) [T] at cluster-frame points (mm) — the numpy
    twin of the compiled ``cluster_field`` (same operations, same order)."""
    n = rx.shape[0]
    bx = np.zeros(n)
    by = np.zeros(n)
    bz = np.zeros(n)
    for p in packs:
        R, o = p.R, p.o
        dx = rx - o[0]
        dy = ry - o[1]
        dz = rz - o[2]
        xm = (R[0, 0] * dx + R[1, 0] * dy) + R[2, 0] * dz
        ym = (R[0, 1] * dx + R[1, 1] * dy) + R[2, 1] * dz
        zm = ((R[0, 2] * dx + R[1, 2] * dy) + R[2, 2] * dz) + p.zstart
        fx, fy, fz = _sample(p, xm, ym, zm)
        fx = fx * p.amp
        fy = fy * p.amp
        fz = fz * p.amp
        bx = bx + ((R[0, 0] * fx + R[0, 1] * fy) + R[0, 2] * fz)
        by = by + ((R[1, 0] * fx + R[1, 1] * fy) + R[1, 2] * fz)
        bz = bz + ((R[2, 0] * fx + R[2, 1] * fy) + R[2, 2] * fz)
    return bx, by, bz


# --------------------------------------------------------------------------- #
#  The reference trajectory
# --------------------------------------------------------------------------- #
class ReferenceTrajectory:
    """The reference particle's path through a curved cluster (entrance
    frame, mm): nodes ``s`` (0 … ``length``) with the position ``r``, the
    tangent ``t``, a rotation-minimising frame ``e1``, ``e2`` and the
    curvatures ``k1``, ``k2`` (t' = k1 e1 + k2 e2, 1/mm)."""

    def __init__(self, s, r, t, e1, e2, k1, k2, *, card_point, card_axes):
        self.s = np.asarray(s, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.e1 = np.asarray(e1, dtype=float)
        self.e2 = np.asarray(e2, dtype=float)
        self.k1 = np.asarray(k1, dtype=float)
        self.k2 = np.asarray(k2, dtype=float)
        self.length = float(self.s[-1])
        self.card_point = np.asarray(card_point, dtype=float)
        self.card_axes = np.asarray(card_axes, dtype=float)   # columns e1, e2, t
        e1c, e2c, tc = self.card_axes[:, 0], self.card_axes[:, 1], self.card_axes[:, 2]
        d = self.r[-1] - self.card_point
        te = self.t[-1]
        #: the traced reference's exit in the card frame: x, x', y, y' (mm, mrad)
        self.residual = (float(d @ e1c), 1e3 * float(te @ e1c) / float(te @ tc),
                         float(d @ e2c), 1e3 * float(te @ e2c) / float(te @ tc))

    def frames_at(self, s_query) -> np.ndarray:
        """Stage frames (n, 14) at path lengths ``s_query``: cubic Hermite
        on positions (derivative t) and on the frame vectors (derivatives
        t' = k1 e1 + k2 e2, e1' = −k1 t, e2' = −k2 t), re-orthonormalised;
        curvatures interpolated linearly.  Exact at the nodes."""
        sq = np.atleast_1d(np.asarray(s_query, dtype=float))
        s = self.s
        i = np.clip(np.searchsorted(s, sq, side="right") - 1, 0, s.size - 2)
        h = s[i + 1] - s[i]
        u = np.where(h > 0, (sq - s[i]) / np.where(h > 0, h, 1.0), 0.0)
        u = np.clip(u, 0.0, 1.0)
        h00 = (2 * u ** 3 - 3 * u ** 2 + 1)[:, None]
        h10 = (u ** 3 - 2 * u ** 2 + u)[:, None]
        h01 = (-2 * u ** 3 + 3 * u ** 2)[:, None]
        h11 = (u ** 3 - u ** 2)[:, None]
        hh = h[:, None]
        k1a, k1b = self.k1[i][:, None], self.k1[i + 1][:, None]
        k2a, k2b = self.k2[i][:, None], self.k2[i + 1][:, None]
        ta, tb = self.t[i], self.t[i + 1]
        e1a, e1b = self.e1[i], self.e1[i + 1]
        e2a, e2b = self.e2[i], self.e2[i + 1]
        r = (h00 * self.r[i] + h10 * hh * ta + h01 * self.r[i + 1] + h11 * hh * tb)
        dta = k1a * e1a + k2a * e2a
        dtb = k1b * e1b + k2b * e2b
        t = h00 * ta + h10 * hh * dta + h01 * tb + h11 * hh * dtb
        e1 = h00 * e1a + h10 * hh * (-k1a * ta) + h01 * e1b + h11 * hh * (-k1b * tb)
        t /= np.linalg.norm(t, axis=1)[:, None]
        e1 = e1 - (np.sum(e1 * t, axis=1))[:, None] * t
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(t, e1)
        k1 = (1.0 - u) * self.k1[i] + u * self.k1[i + 1]
        k2 = (1.0 - u) * self.k2[i] + u * self.k2[i + 1]
        out = np.empty((sq.size, _NFRAME))
        out[:, 0:3] = r
        out[:, 3:6] = t
        out[:, 6:9] = e1
        out[:, 9:12] = e2
        out[:, 12] = k1
        out[:, 13] = k2
        # exact node values where a query lands on a node
        on = np.isclose(sq, s[i], rtol=0.0, atol=1e-12)
        if np.any(on):
            j = i[on]
            out[on, 0:3] = self.r[j]
            out[on, 3:6] = self.t[j]
            out[on, 6:9] = self.e1[j]
            out[on, 9:12] = self.e2[j]
            out[on, 12] = self.k1[j]
            out[on, 13] = self.k2[j]
        return out


def _map_coords(pack, r):
    """A cluster-frame point in a pack's map frame (the sampler's own
    arithmetic: x_map = Rᵀ(r − o), z offset by the map start)."""
    R, o = pack.R, pack.o
    d = r - o
    return np.array([(R[0, 0] * d[0] + R[1, 0] * d[1]) + R[2, 0] * d[2],
                     (R[0, 1] * d[0] + R[1, 1] * d[1]) + R[2, 1] * d[2],
                     ((R[0, 2] * d[0] + R[1, 2] * d[1]) + R[2, 2] * d[2])
                     + pack.zstart])


def _inside_any(packs, r) -> bool:
    """True when ``r`` lies in some child's map box (outside every box the
    sampled field is exactly zero)."""
    for p in packs:
        m = _map_coords(p, r)
        if (p.gx[0] <= m[0] <= p.gx[-1] and p.gy[0] <= m[1] <= p.gy[-1]
                and p.gz[0] <= m[2] <= p.gz[-1]):
            return True
    return False


def _next_box_entry(packs, r, t) -> float:
    """Path length along the straight line r + λ t to the first child map
    box it enters (math.inf when it meets none) — slab intersection in each
    map's frame."""
    best = math.inf
    for p in packs:
        m0 = _map_coords(p, r)
        d = p.R.T @ t
        lo_all, hi_all = -math.inf, math.inf
        hit = True
        for a, ax in enumerate((p.gx, p.gy, p.gz)):
            lo, hi = float(ax[0]), float(ax[-1])
            if abs(d[a]) < 1e-15:
                if not (lo <= m0[a] <= hi):
                    hit = False
                    break
                continue
            l1 = (lo - m0[a]) / d[a]
            l2 = (hi - m0[a]) / d[a]
            lo_all = max(lo_all, min(l1, l2))
            hi_all = min(hi_all, max(l1, l2))
        if hit and lo_all <= hi_all and hi_all >= 0.0:
            best = min(best, max(lo_all, 0.0))
    return best


def integrate_reference(packs, q_over_p: float, card_point, card_axes, *,
                        h: float = 0.1, s_max: float | None = None) -> ReferenceTrajectory:
    """Trace the reference through the summed child fields.

    ``q_over_p``: charge/(Bρ) in 1/(T·mm).  RK4 in the path length with
    step ``h`` (mm) from the origin along +z; the frame (t, e1, e2) is
    transported without rotation about t (e1' = −(t'·e1) t).  Stops on the
    exit plane (point ``card_point``, normal = the card's z axis), landing
    on it to 1e-11 mm.  Outside every map the field is exactly zero, so
    there the path is a straight line: it lands on the plane, or jumps to
    the next map it meets, at once — and a reference that leaves the maps
    away from the plane is refused at once.  Also refused: more than two
    full turns, or ``s_max`` of path (default: 20 × the distance to the
    exit point)."""
    P = np.asarray(card_point, dtype=float)
    A = np.asarray(card_axes, dtype=float)
    nrm = A[:, 2]
    if s_max is None:
        s_max = 20.0 * max(float(np.linalg.norm(P)), 1.0) + 100.0

    def field(r):
        bx, by, bz = cluster_field(packs, np.array([r[0]]), np.array([r[1]]),
                                   np.array([r[2]]))
        return np.array([bx[0], by[0], bz[0]])

    def deriv(r, t, e1):
        tp = q_over_p * np.cross(t, field(r))
        return t, tp, -(tp @ e1) * t

    def step(r, t, e1, hs):
        a1 = deriv(r, t, e1)
        a2 = deriv(r + 0.5 * hs * a1[0], t + 0.5 * hs * a1[1], e1 + 0.5 * hs * a1[2])
        a3 = deriv(r + 0.5 * hs * a2[0], t + 0.5 * hs * a2[1], e1 + 0.5 * hs * a2[2])
        a4 = deriv(r + hs * a3[0], t + hs * a3[1], e1 + hs * a3[2])
        r2 = r + hs / 6.0 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        t2 = t + hs / 6.0 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        e12 = e1 + hs / 6.0 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        t2 = t2 / np.linalg.norm(t2)
        e12 = e12 - (e12 @ t2) * t2
        e12 = e12 / np.linalg.norm(e12)
        return r2, t2, e12

    def node(r, t, e1):
        tp = q_over_p * np.cross(t, field(r))
        e2 = np.cross(t, e1)
        return float(tp @ e1), float(tp @ e2), e2

    r = np.zeros(3)
    t = np.array([0.0, 0.0, 1.0])
    e1 = np.array([1.0, 0.0, 0.0])
    S, Rr, T, E1, E2, K1, K2 = [], [], [], [], [], [], []

    def push(s, r, t, e1):
        k1, k2, e2 = node(r, t, e1)
        for lst, v in zip((S, Rr, T, E1, E2, K1, K2), (s, r, t, e1, e2, k1, k2),
                          strict=True):
            lst.append(v)

    # The exit is the first crossing of the plane from its upstream side
    # (g < 0 → g ≥ 0) while moving along its normal (t·n > 0), after at
    # least one step: a 180° bend starts ON its exit plane (g = 0), and
    # beyond 90° the map z is no longer monotonic.
    s = 0.0
    push(s, r, t, e1)
    turned = 0.0
    n_done = 0
    while True:
        if n_done % 16 == 0 and not _inside_any(packs, r):
            g = float((r - P) @ nrm)
            tn = float(t @ nrm)
            lam_plane = -g / tn if (tn > 0.0 and g < 0.0) else math.inf
            lam_box = _next_box_entry(packs, r, t)
            if lam_plane < lam_box:
                # straight to the plane through field-free space (exact)
                push(s + lam_plane, r + lam_plane * t, t.copy(), e1.copy())
                return ReferenceTrajectory(S, Rr, T, E1, E2, K1, K2,
                                           card_point=P, card_axes=A)
            if not math.isfinite(lam_box):
                raise ValueError(
                    "SUPERPOSE_MAP_OUT: the reference particle leaves the "
                    "field maps without reaching the exit plane — check the "
                    "exit frame and the field scaling")
            if lam_box > h:
                # across a field-free gap to the next map
                r = r + lam_box * t
                s += lam_box
                push(s, r, t, e1)
        n_done += 1
        r2, t2, e12 = step(r, t, e1, h)
        if ((r - P) @ nrm < 0.0 and (r2 - P) @ nrm >= 0.0
                and float(t2 @ nrm) > 0.0):
            break
        r, t, e1 = r2, t2, e12
        s += h
        push(s, r, t, e1)
        turned += h * math.hypot(K1[-1], K2[-1])
        if turned > 4.0 * math.pi:
            raise ValueError(
                "SUPERPOSE_MAP_OUT: the reference particle turns more than two "
                "full turns in the maps without reaching the exit plane — "
                "check the exit frame and the field scaling")
        if s > s_max:
            raise ValueError(
                "SUPERPOSE_MAP_OUT: the reference particle does not reach the "
                f"exit plane within {s_max:.0f} mm of path — check the exit "
                "frame and the field scaling")
    # land on the plane: secant on the partial step from the last node
    g0 = float((r - P) @ nrm)
    lo, hi = 0.0, h
    g_lo, g_hi = g0, float((r2 - P) @ nrm)
    hs = h * (-g_lo) / (g_hi - g_lo)
    for _ in range(60):
        rr, tt, ee = step(r, t, e1, hs)
        g = float((rr - P) @ nrm)
        if abs(g) < 1e-11:
            break
        if g < 0.0:
            lo, g_lo = hs, g
        else:
            hi, g_hi = hs, g
        hs_new = lo + (hi - lo) * (-g_lo) / (g_hi - g_lo)
        hs = hs_new if lo < hs_new < hi else 0.5 * (lo + hi)
    push(s + hs, rr, tt, ee)
    return ReferenceTrajectory(S, Rr, T, E1, E2, K1, K2,
                               card_point=P, card_axes=A)


# --------------------------------------------------------------------------- #
#  The particle push (numpy twin of the compiled curved_rk4_steps)
# --------------------------------------------------------------------------- #
def _deriv(f, x, y, u1, u2, packs, kq, inv_beta, inv_beta_ref, c_phi):
    r0x, r0y, r0z = f[0], f[1], f[2]
    tx, ty, tz = f[3], f[4], f[5]
    e1x, e1y, e1z = f[6], f[7], f[8]
    e2x, e2y, e2z = f[9], f[10], f[11]
    k1, k2 = f[12], f[13]
    h = (1.0 - k1 * x) - k2 * y
    ut = np.sqrt((1.0 - u1 * u1) - u2 * u2)
    rx = (r0x + x * e1x) + y * e2x
    ry = (r0y + x * e1y) + y * e2y
    rz = (r0z + x * e1z) + y * e2z
    bx, by, bz = cluster_field(packs, rx, ry, rz)
    ux = (u1 * e1x + u2 * e2x) + ut * tx
    uy = (u1 * e1y + u2 * e2y) + ut * ty
    uz = (u1 * e1z + u2 * e2z) + ut * tz
    cx = uy * bz - uz * by
    cy = uz * bx - ux * bz
    cz = ux * by - uy * bx
    fe1 = (cx * e1x + cy * e1y) + cz * e1z
    fe2 = (cx * e2x + cy * e2y) + cz * e2z
    g = h / ut
    dx = g * u1
    dy = g * u2
    du1 = g * (kq * fe1) - k1 * ut
    du2 = g * (kq * fe2) - k2 * ut
    dphi = c_phi * (g * inv_beta - inv_beta_ref)
    return dx, dy, du1, du2, dphi


def curved_rk4_numpy(P, frames, ds: float, n_steps: int, packs, kq, inv_beta,
                     inv_beta_ref: float, c_phi: float) -> None:
    """``n_steps`` RK4 steps of ``ds`` (mm) along the reference for the
    particles ``P`` ((n, 6) HELIX coordinates, updated in place).

    ``frames``: (2·n_steps + 1, 14) stage frames at s0, s0 + ds/2, s0 + ds, …
    ``kq``: per-particle charge/(Bρ) [1/(T·mm)]; ``inv_beta``: 1/β per
    particle; ``c_phi``: 360/λ [deg/mm].  ΔW is unchanged (static field)."""
    xp = P[:, 1] * 1e-3
    yp = P[:, 3] * 1e-3
    inv = 1.0 / np.sqrt((1.0 + xp * xp) + yp * yp)
    x = P[:, 0].copy()
    y = P[:, 2].copy()
    u1 = xp * inv
    u2 = yp * inv
    phi = P[:, 4].copy()
    half = 0.5 * ds
    sixth = ds / 6.0
    for k in range(n_steps):
        fa, fm, fb = frames[2 * k], frames[2 * k + 1], frames[2 * k + 2]
        a = _deriv(fa, x, y, u1, u2, packs, kq, inv_beta, inv_beta_ref, c_phi)
        b = _deriv(fm, x + half * a[0], y + half * a[1], u1 + half * a[2],
                   u2 + half * a[3], packs, kq, inv_beta, inv_beta_ref, c_phi)
        c = _deriv(fm, x + half * b[0], y + half * b[1], u1 + half * b[2],
                   u2 + half * b[3], packs, kq, inv_beta, inv_beta_ref, c_phi)
        d = _deriv(fb, x + ds * c[0], y + ds * c[1], u1 + ds * c[2],
                   u2 + ds * c[3], packs, kq, inv_beta, inv_beta_ref, c_phi)
        x = x + sixth * (((a[0] + 2.0 * b[0]) + 2.0 * c[0]) + d[0])
        y = y + sixth * (((a[1] + 2.0 * b[1]) + 2.0 * c[1]) + d[1])
        u1 = u1 + sixth * (((a[2] + 2.0 * b[2]) + 2.0 * c[2]) + d[2])
        u2 = u2 + sixth * (((a[3] + 2.0 * b[3]) + 2.0 * c[3]) + d[3])
        phi = phi + sixth * (((a[4] + 2.0 * b[4]) + 2.0 * c[4]) + d[4])
    ut = np.sqrt((1.0 - u1 * u1) - u2 * u2)
    P[:, 0] = x
    P[:, 1] = (u1 / ut) * 1e3
    P[:, 2] = y
    P[:, 3] = (u2 / ut) * 1e3
    P[:, 4] = phi


def _push(P, frames, ds, n_steps, packs, kq, inv_beta, inv_beta_ref, c_phi):
    """The compiled push when switched on, else the numpy twin."""
    if _USE_KERNEL and _curved_kernel is not None and P.shape[0]:
        buf = np.ascontiguousarray(P, dtype=np.float64)
        _curved_kernel(buf, np.ascontiguousarray(frames, dtype=np.float64),
                       float(ds), int(n_steps), [p.as_tuple() for p in packs],
                       np.ascontiguousarray(kq, dtype=np.float64),
                       np.ascontiguousarray(inv_beta, dtype=np.float64),
                       float(inv_beta_ref), float(c_phi))
        P[:, :] = buf
        return
    curved_rk4_numpy(P, frames, ds, n_steps, packs, kq, inv_beta,
                     inv_beta_ref, c_phi)


def survey_geometry(element):
    """A curved cluster's exit as the floor-plan surveys draw it (results
    tab, ring detection): ``(path, P, A)`` — the path length (mm), the exit
    point P (lateral, vertical, forward; mm) and the exit axes A (columns:
    lateral, vertical, forward), both in the entrance frame.

    The surveys draw a Dipole of positive angle toward +x while its beam
    bends toward −x (dispersion R16 > 0, the HELIX and TraceWin
    convention); a curved cluster's card frame is physical, so it is
    turned by 180° about the beam axis (x, y → −x, −y, a proper rotation)
    to draw both alike — a 30° bend leaving at x < 0 (R16 > 0) is drawn
    like a +30° Dipole."""
    L, P, A = element.floor_transform()
    F = np.diag([-1.0, -1.0, 1.0])
    return L, F @ P, F @ A @ F


def survey_points(element, ds_mm: float):
    """Points along a curved cluster's reference (drawing convention of
    :func:`survey_geometry`, entrance frame, mm) about every ``ds_mm`` of
    path, without the end points: ``(s, points)``.  Empty before the
    reference is traced (the surveys then draw the chord)."""
    traj = getattr(element, "reference_trajectory", None)
    if traj is None or traj.length <= 0.0:
        return np.zeros(0), np.zeros((0, 3))
    n = max(4, int(traj.length / max(float(ds_mm), 1e-6)))
    s = np.linspace(0.0, traj.length, n + 1)[1:-1]
    r = traj.frames_at(s)[:, 0:3]
    return s, r * np.array([-1.0, -1.0, 1.0])


def has_curved_elements(lattice) -> bool:
    """True when ``lattice`` holds a curved field-map cluster."""
    return any(getattr(e, "is_curved", False)
               for e in getattr(lattice, "elements", None) or [])


_ENERGY_NEUTRAL = frozenset({
    "Drift", "Quadrupole", "Solenoid", "Dipole", "Edge", "Marker", "Aperture",
    "SpaceChargeComp", "Steerer", "Multipole", "ThinLens", "ScGridDirective"})
_ENERGY_COMMANDS = frozenset({"SetBeamEnergy", "SetBeamPhaseError"})


def _energy_neutral(e) -> bool:
    """True for an element that cannot change the reference energy (the
    trace of a curved cluster depends on the species and the energy only)."""
    if getattr(e, "is_curved", False):
        return True
    name = type(e).__name__
    if name in _ENERGY_NEUTRAL:
        return True
    if name == "MatrixElement":
        off = getattr(e, "offset", None)
        return off is None or float(off[5]) == 0.0
    from linac_gen.elements.base import FieldMapElement, PassiveElement
    from linac_gen.elements.lattice_commands import LatticeCommand
    if isinstance(e, FieldMapElement):
        from linac_gen.tracking.tracker import _is_magnetic_only_fieldmap
        return _is_magnetic_only_fieldmap(e)
    if isinstance(e, LatticeCommand):
        return name not in _ENERGY_COMMANDS
    return isinstance(e, PassiveElement)


def prepare_curved_elements(lattice, ref, *, replay: bool = True) -> int:
    """Trace the reference through the curved clusters of ``lattice`` for
    the beam entering it (``ref``, not modified), so their lengths — the
    reference's path — and every s position after them are known before a
    run.  Runs trace at the element anyway, for the energy that reaches it;
    an element reuses its trace while nothing changed.

    ``replay=True`` walks the reference with matrix mode's replay (it
    resets and advances the field maps before each cluster).
    ``replay=False`` touches no other element: it traces only clusters
    that nothing before them can change the energy of (no RF, foil or
    energy command), at the beam's energy — safe beside a run of the same
    lattice (the GUI's mode); the others wait for a run.

    Returns the number of clusters traced.  A trace that fails raises
    ValueError after the others were tried; a lattice without a curved
    cluster is not touched."""
    els = list(getattr(lattice, "elements", None) or [])
    idx = [i for i, e in enumerate(els) if getattr(e, "is_curved", False)]
    if not idx:
        return 0
    errors, n = [], 0
    if replay:
        from linac_gen.tracking.matrix_tracking import _replay_ref
        r = ref.copy()
        done = 0
        for i in idx:
            _replay_ref(els[done:i], r, lattice)
            try:
                els[i].prepare_reference(r.copy())
                n += 1
            except ValueError as exc:
                errors.append(str(exc))
            done = i
    else:
        neutral = True
        for i, e in enumerate(els):
            if getattr(e, "is_curved", False) and neutral:
                try:
                    e.prepare_reference(ref.copy())
                    n += 1
                except ValueError as exc:
                    errors.append(str(exc))
            elif not _energy_neutral(e):
                neutral = False
            if not neutral and i > idx[-1]:
                break
    if errors:
        raise ValueError("; ".join(errors))
    return n


def _exit_to_card(P, traj: ReferenceTrajectory, inv_beta, c_phi: float):
    """Move particles from the plane orthogonal to the reference at its
    exit into the card's frame, in place: each particle drifts straight onto
    the card plane (the two planes cross at the reference's exit point); its
    phase gains the extra path at its own speed.  Returns the mask of the
    particles that cannot reach the card plane (moving away from it): their
    coordinates become NaN."""
    if P.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    re, te = traj.r[-1], traj.t[-1]
    e1e, e2e = traj.e1[-1], traj.e2[-1]
    Pc = traj.card_point
    e1c, e2c, tc = traj.card_axes[:, 0], traj.card_axes[:, 1], traj.card_axes[:, 2]
    xp = P[:, 1] * 1e-3
    yp = P[:, 3] * 1e-3
    inv = 1.0 / np.sqrt(1.0 + xp * xp + yp * yp)
    u = (xp * inv)[:, None] * e1e + (yp * inv)[:, None] * e2e + inv[:, None] * te
    r = re + P[:, 0][:, None] * e1e + P[:, 2][:, None] * e2e
    utc = u @ tc
    bad = ~(utc > 0.0)
    utc = np.where(bad, 1.0, utc)
    lam = -((r - Pc) @ tc) / utc
    r2 = r + lam[:, None] * u
    d = r2 - Pc
    P[:, 0] = d @ e1c
    P[:, 2] = d @ e2c
    P[:, 1] = 1e3 * (u @ e1c) / utc
    P[:, 3] = 1e3 * (u @ e2c) / utc
    P[:, 4] = P[:, 4] + c_phi * lam * inv_beta
    if np.any(bad):
        P[bad, :5] = np.nan
    return bad


# --------------------------------------------------------------------------- #
#  The element
# --------------------------------------------------------------------------- #
class CurvedFieldMapCluster(SuperposedFieldMap):
    """A TraceWin field-map cluster with ``SUPERPOSE_MAP_OUT``: a curved
    reference trajectory through superposed static magnetic 3-D maps.

    Parameters
    ----------
    name : str
    children : list[tuple[float, FieldMap3D]]
        ``(z0, map)`` pairs in card order (as :class:`SuperposedFieldMap`).
    placements : list[tuple]
        Per child, the full ``SUPERPOSE_MAP`` operands ``(Z, X, Y, θZ, θX,
        θY)`` (mm, deg); default: ``(z0, 0, 0, 0, 0, 0)``.
    exit_frame : tuple
        The ``SUPERPOSE_MAP_OUT`` operands ``(Z, X, Y, θZ, θX, θY)``.

    ``length`` is the reference's path length once :meth:`prepare_reference`
    has traced it for the entering energy (s runs along the arc); before
    that it is an estimate from the exit frame (``length_provisional``).
    """

    #: the step policy: at most this fraction of the finest child grid step
    _GRID_FRACTION = 0.25
    #: the element applies its own misalignment (it moves its maps); the
    #: tracker / envelope / matrix wrappers leave it alone
    self_aligned = True
    is_curved = True

    def __init__(self, name: str, children: list, placements=None,
                 exit_frame=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), **kw):
        super().__init__(name, children, **kw)
        for _z0, child in self.children:
            if not isinstance(child, FieldMap3D):
                raise ValueError(
                    f"curved field-map cluster '{name}': child "
                    f"'{getattr(child, 'name', child)}' is not a 3-D Cartesian "
                    "map (FIELD_MAP 70); only static magnetic 3-D maps can "
                    "curve the reference trajectory in this version")
            for ch_enum in child.field_data.channels:
                if ch_enum.is_electric or not ch_enum.is_static:
                    raise ValueError(
                        f"curved field-map cluster '{name}': child "
                        f"'{child.name}' has an electric or RF channel "
                        f"({ch_enum.name}); only static magnetic maps can "
                        "curve the reference trajectory in this version")
        if placements is None:
            placements = [(z0, 0.0, 0.0, 0.0, 0.0, 0.0) for z0, _c in self.children]
        if len(placements) != len(self.children):
            raise ValueError("one placement per child is needed")
        self.placements = [tuple(float(v) for v in p) for p in placements]
        self.exit_frame = tuple(float(v) for v in exit_frame)
        if len(self.exit_frame) != 6:
            raise ValueError("SUPERPOSE_MAP_OUT needs six operands "
                             "(Z X Y θZ θX θY)")
        self.map_span = float(self.length)      # the straight span of the maps
        self.length = self._provisional_length()
        self.length_provisional = True
        finest = min(min(np.diff(np.asarray(child.field_data.channels[c].z,
                                             dtype=float)).min()
                         for c in child.field_data.channels)
                     for _z, child in self.children)
        self._ds_max = max(self._GRID_FRACTION * float(finest), 1e-3)
        self.n_steps = max(1, int(math.ceil(self.length / self._ds_max)))
        # the largest map diagonal: the trace's path bound scales with it
        self._map_diag = max(
            math.sqrt(sum(float(np.ptp(np.asarray(getattr(ch, a), dtype=float))) ** 2
                          for a in ("x", "y", "z")))
            for _z, child in self.children
            for ch in child.field_data.channels.values())
        self._traj = None
        self._traj_key = None
        self._packs = None
        self._packs_key = None
        self._s_cursor = 0.0
        self._exited = False
        self._warned = set()
        self._traj_fail = None          # (key, message) of a failed trace
        self.exit_label = None          # the SUPERPOSE_MAP_OUT card's label

    # ------------------------------------------------------------------ #
    #  Geometry
    # ------------------------------------------------------------------ #
    def card_frame(self):
        """(exit point, axes) of the card in the entrance frame (mm)."""
        Z, X, Y, tz, tx, ty = self.exit_frame
        return np.array([X, Y, Z]), exit_rotation(tz, tx, ty)

    def _provisional_length(self) -> float:
        """Before the reference is traced: the path through the vertex where
        the entrance axis meets the exit axis (or the chord without one)."""
        P, A = self.card_frame()
        tc = A[:, 2]
        chord = float(np.linalg.norm(P))
        # entrance line r = a ẑ, exit line r = P + b tc: closest approach
        ez = np.array([0.0, 0.0, 1.0])
        c = float(ez @ tc)
        if abs(1.0 - abs(c)) < 1e-12:
            return chord
        w = -P                                   # entrance origin − exit point
        d_ = float(ez @ w)
        e_ = float(tc @ w)
        a = (c * e_ - d_) / (1.0 - c * c)        # along the entrance axis
        b = (e_ - c * d_) / (1.0 - c * c)        # along the exit axis
        v1 = a * ez
        v2 = P + b * tc
        vertex = 0.5 * (v1 + v2)
        length = float(np.linalg.norm(vertex)) + float(np.linalg.norm(P - vertex))
        return length if np.isfinite(length) and length > 0 else chord

    def floor_transform(self):
        """(path length, exit point, exit axes) in the entrance frame: the
        survey continues from the card's exit frame."""
        P, A = self.card_frame()
        return float(self.length), P, A

    # ------------------------------------------------------------------ #
    #  Field and reference
    # ------------------------------------------------------------------ #
    def _misalignment_rt(self):
        """Rotation/translation of the whole cluster's maps in the entrance
        frame from the misalignment (dx, dy, tilt about z)."""
        tilt = math.radians(float(getattr(self, "tilt_deg", 0.0) or 0.0))
        R = _rot_z(tilt)
        o = np.array([float(getattr(self, "dx", 0.0) or 0.0),
                      float(getattr(self, "dy", 0.0) or 0.0), 0.0])
        return R, o

    def _field_key(self):
        return (tuple((k, _map_fingerprint(c), float(c.scale), float(c.kb),
                       float(c.voltage_rel), float(c._z_map_start))
                      for k, (_z, c) in enumerate(self.children)),
                float(getattr(self, "voltage_rel", 0.0) or 0.0),
                float(getattr(self, "dx", 0.0) or 0.0),
                float(getattr(self, "dy", 0.0) or 0.0),
                float(getattr(self, "tilt_deg", 0.0) or 0.0),
                tuple(self.placements))

    def packs(self):
        """The children's static magnetic channels with their placements."""
        key = self._field_key()
        if self._packs is not None and self._packs_key == key:
            return self._packs
        Rm, om = self._misalignment_rt()
        amp_cluster = 1.0 + float(getattr(self, "voltage_rel", 0.0) or 0.0)
        out = []
        for (_z0, child), pl in zip(self.children, self.placements, strict=True):
            Z, X, Y, tz, tx, ty = pl
            Rc = child_rotation(tz, tx, ty)
            oc = np.array([X, Y, Z])
            R = Rm @ Rc
            o = Rm @ oc + om
            for ch_enum, ch in child.field_data.channels.items():
                ax, stacked, slots = _stack_channel(ch)
                amp = child._scale_factor(ch_enum, ch) * amp_cluster
                out.append(_ChildPack(ax[0], ax[1], ax[2], stacked, slots, amp,
                                      R, o, float(child._z_map_start), child, ch_enum))
        self._packs = out
        self._packs_key = key
        return out

    def prepare_reference(self, ref) -> ReferenceTrajectory:
        """Trace (or reuse) the reference for the entering particle ``ref``
        and set ``length`` to its path length."""
        key = (float(ref.species.mass), float(ref.species.charge),
               float(ref.w_kin), self._field_key(), self.exit_frame)
        if self._traj is not None and self._traj_key == key:
            return self._traj
        p = float(ref.bg) * float(ref.species.mass)            # MeV/c
        q_over_p = float(ref.species.charge) * _C_CONV * 1e-3 / p
        P, A = self.card_frame()
        # path bound: generous for any bend inside the maps (a full turn
        # stays within a few map diagonals), finite for a reference that
        # never reaches the exit plane
        s_max = 20.0 * float(np.linalg.norm(P)) + 5.0 * self._map_diag + 100.0
        if self._traj_fail is not None and self._traj_fail[0] == key:
            raise ValueError(self._traj_fail[1])
        try:
            traj = integrate_reference(self.packs(), q_over_p, P, A,
                                       s_max=s_max)
        except ValueError as exc:
            msg = f"'{getattr(self, 'label', None) or self.name}': {exc}"
            self._traj_fail = (key, msg)
            raise ValueError(msg) from None
        self._traj = traj
        self._traj_key = key
        self.length = traj.length
        self.length_provisional = False
        self.n_steps = max(1, int(math.ceil(self.length / self._ds_max)))
        self._check_after_trace()
        return traj

    def _warn_once(self, key, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(msg, stacklevel=3)

    def _check_after_trace(self) -> None:
        """What only the traced path can tell: interior diagnostics
        (SHIFT_IN_FIELD_MAP) beyond its end are never reached; and the
        maps are sampled trilinearly whatever the 3-D interpolation
        setting says."""
        for dz, mk in getattr(self, "interior_markers", None) or []:
            if dz > self.length + 1e-9:
                self._warn_once(
                    ("marker", getattr(mk, "name", ""), round(self.length, 6)),
                    f"'{self.name}': SHIFT_IN_FIELD_MAP {dz:g} mm lies beyond "
                    f"the path of the curved cluster ({self.length:.3f} mm) — "
                    f"diagnostic '{getattr(mk, 'name', mk)}' is not recorded")
        if any(str(getattr(c, "interp_kind", "linear")) != "linear"
               for _z, c in self.children):
            self._warn_once(
                ("interp",),
                f"'{self.name}': curved field-map clusters sample their maps "
                "trilinearly; the cubic interpolation setting does not apply")

    @property
    def reference_trajectory(self):
        return self._traj

    def __getstate__(self):
        # copies and pickles leave out the stacked copies of the maps (as
        # large as the maps themselves); the trace, small, travels along
        state = dict(self.__dict__)
        state["_packs"] = None
        state["_packs_key"] = None
        return state

    # ------------------------------------------------------------------ #
    #  Walk state
    # ------------------------------------------------------------------ #
    def reset_run_state(self) -> None:
        super().reset_run_state()
        self._s_cursor = 0.0
        self._slice_reached_end = False
        self._exited = False

    def _save_walk_state(self):
        return (self._step_idx, self._s_cursor)

    def _restore_walk_state(self, state) -> None:
        self._step_idx, self._s_cursor = state

    # ------------------------------------------------------------------ #
    #  Tracking
    # ------------------------------------------------------------------ #
    def _particle_constants(self, ref, dW):
        m = float(ref.species.mass)
        W = float(ref.w_kin) + dW
        p = np.sqrt(W * (W + 2.0 * m))
        kq = float(ref.species.charge) * _C_CONV * 1e-3 / p
        inv_beta = (W + m) / p
        return kq, inv_beta

    def _stage_frames(self, s_values, kq_ref: float) -> np.ndarray:
        """Frames at the RK4 stage points: position and axes from the traced
        reference, curvatures recomputed from the field there with the push's
        own arithmetic (κ = kq_ref (t × B)·e), so that the particle with zero
        deviation — the reference — stays exactly at zero in every mode."""
        f = self._traj.frames_at(s_values)
        bx, by, bz = cluster_field(self.packs(), f[:, 0], f[:, 1], f[:, 2])
        tx, ty, tz = f[:, 3], f[:, 4], f[:, 5]
        # the push's u for u1 = u2 = 0 is exactly t: (0·e1 + 0·e2) + 1·t
        cx = ty * bz - tz * by
        cy = tz * bx - tx * bz
        cz = tx * by - ty * bx
        f[:, 12] = kq_ref * ((cx * f[:, 6] + cy * f[:, 7]) + cz * f[:, 8])
        f[:, 13] = kq_ref * ((cx * f[:, 9] + cy * f[:, 10]) + cz * f[:, 11])
        return f

    def _steps(self, P, ref, s0: float, ds: float, n: int) -> None:
        kq, inv_beta = self._particle_constants(ref, P[:, 5])
        kq_ref, ib_ref = self._particle_constants(ref, np.zeros(1))
        frames = self._stage_frames(s0 + 0.5 * ds * np.arange(2 * n + 1), float(kq_ref[0]))
        c_phi = 360.0 / ref.wavelength if ref.wavelength > 0 else 0.0
        _push(P, frames, ds, n, self.packs(), kq, inv_beta, float(ib_ref[0]), c_phi)

    def _at_end(self, s: float) -> bool:
        return s >= self.length - 1e-9 * max(1.0, self.length)

    def track_rk4(self, beam, ds: float) -> None:
        """One step of ``ds`` (mm of reference path) for every alive
        particle; the step that reaches the end also moves the particles into
        the card's exit frame."""
        ref = beam.ref
        if self._exited:
            raise RuntimeError(
                f"'{self.name}': a step past the end of the curved cluster "
                "(the beam is already in the exit card's frame)")
        if self._s_cursor == 0.0 or self._traj is None:
            self.prepare_reference(ref)
        s_end = self._s_cursor + ds
        exit_now = self._at_end(s_end)
        alive = beam.alive_mask
        if np.any(alive):
            idx = np.flatnonzero(alive)
            P = beam.particles[idx].copy()
            before = P.copy()
            self._steps(P, ref, self._s_cursor, ds, 1)
            if exit_now:
                _kq, inv_beta = self._particle_constants(ref, P[:, 5])
                c_phi = 360.0 / ref.wavelength if ref.wavelength > 0 else 0.0
                _exit_to_card(P, self._traj, inv_beta, c_phi)
            bad = ~np.all(np.isfinite(P[:, :5]), axis=1)
            if np.any(bad):
                P[bad] = before[bad]
            beam.particles[idx] = P
            for i in idx[bad]:
                beam.record_loss(int(i), ref.s + ds, self.name)
        ref.s += ds
        if ref.wavelength > 0:
            ref.phi_s += 360.0 * ds / (ref.beta * ref.wavelength)
        self._s_cursor = s_end
        self._step_idx += 1
        self._exited = exit_now

    # ------------------------------------------------------------------ #
    #  Reference walks
    # ------------------------------------------------------------------ #
    def advance_ref(self, ref) -> None:
        self.prepare_reference(ref)
        L = self.length
        ref.s += L
        if ref.wavelength > 0:
            ref.phi_s += 360.0 * L / (ref.beta * ref.wavelength)

    def advance_ref_over(self, ref, z_from_mm: float, z_to_mm: float) -> None:
        if z_from_mm <= 0.0:
            self.prepare_reference(ref)
        L = z_to_mm - z_from_mm
        if L <= 0.0:
            return
        ref.s += L
        if ref.wavelength > 0:
            ref.phi_s += 360.0 * L / (ref.beta * ref.wavelength)

    # ------------------------------------------------------------------ #
    #  Matrices
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fd_eps(ref) -> np.ndarray:
        """Finite-difference steps of the fitted matrices: the other field
        maps' (0.01 mm, mrad, deg), with the energy step scaled to the
        energy — 1e-5·W, at most their 1 keV, which is 3 % of a 30 keV
        beam and biased the dispersion terms by 7e-4 there."""
        return np.array([0.01, 0.01, 0.01, 0.01, 0.01,
                         min(1e-3, 1e-5 * float(ref.w_kin))])

    def _track_probes(self, ref, P, s0: float, s1: float) -> np.ndarray:
        L = s1 - s0
        n = max(1, int(math.ceil(L / self._ds_max - 1e-9)))
        Q = np.array(P, dtype=float, copy=True)
        self._steps(Q, ref, s0, L / n, n)
        return Q

    def exit_map(self, ref):
        """(M, o): the linear part and the offset of the move into the card
        frame, about the reference (envelope after the last slice)."""
        from linac_gen.tracking.rk4 import numerical_jacobian_batched
        self.prepare_reference(ref)

        def batch(P):
            Q = np.array(P, dtype=float, copy=True)
            _kq, inv_beta = self._particle_constants(ref, Q[:, 5])
            c_phi = 360.0 / ref.wavelength if ref.wavelength > 0 else 0.0
            _exit_to_card(Q, self._traj, inv_beta, c_phi)
            return Q
        M = numerical_jacobian_batched(batch, np.zeros(6), eps=self._fd_eps(ref))
        o = batch(np.zeros((1, 6)))[0]
        return M, o

    def orbit_offset(self, ref, ds=None):
        """The exit offset — the traced reference seen from the card frame —
        for the whole element (``ds`` None or the full length) and for the
        slice that reached the end (the envelope's last ``fitted_matrix_slice``);
        None for any other slice."""
        whole = ds is None or abs(float(ds) - float(self.length)) <= 1e-9 * max(1.0, self.length)
        if not whole and not getattr(self, "_slice_reached_end", False):
            return None
        self._slice_reached_end = False
        _M, o = self.exit_map(ref)
        return o if np.any(o != 0.0) else None

    def fitted_matrix(self, ref) -> np.ndarray:
        """6×6 over the whole element, the move into the card frame included."""
        from linac_gen.tracking.rk4 import numerical_jacobian_batched
        self.prepare_reference(ref)
        saved = self._save_walk_state()

        def batch(P):
            Q = self._track_probes(ref, P, 0.0, self.length)
            _kq, inv_beta = self._particle_constants(ref, Q[:, 5])
            c_phi = 360.0 / ref.wavelength if ref.wavelength > 0 else 0.0
            _exit_to_card(Q, self._traj, inv_beta, c_phi)
            return Q
        try:
            M = numerical_jacobian_batched(batch, np.zeros(6),
                                           eps=self._fd_eps(ref))
        finally:
            self._restore_walk_state(saved)
        return M

    def fitted_matrix_slice(self, ref, ds_mm: float, _z_from_mm=None) -> np.ndarray:
        """6×6 over one slice of the path; the slice that reaches the end
        includes the move into the card frame (its offset then comes from
        :meth:`orbit_offset`).  Advances the walk cursor like the other field
        maps."""
        from linac_gen.tracking.rk4 import numerical_jacobian_batched
        z_from = self._s_cursor if _z_from_mm is None else float(_z_from_mm)
        if z_from <= 0.0 or self._traj is None:
            self.prepare_reference(ref)
        if self._at_end(z_from):
            raise RuntimeError(
                f"'{self.name}': a matrix slice past the end of the curved "
                "cluster (its exit into the card frame is already included)")
        z_to = z_from + ds_mm
        reaches_end = self._at_end(z_to)

        def batch(P):
            Q = self._track_probes(ref, P, z_from, z_to)
            if reaches_end:
                _kq, inv_beta = self._particle_constants(ref, Q[:, 5])
                c_phi = 360.0 / ref.wavelength if ref.wavelength > 0 else 0.0
                _exit_to_card(Q, self._traj, inv_beta, c_phi)
            return Q
        M = numerical_jacobian_batched(batch, np.zeros(6), eps=self._fd_eps(ref))
        self._step_idx += 1
        self._s_cursor = z_to
        self._slice_reached_end = reaches_end
        return M

    # ------------------------------------------------------------------ #
    #  Not supported
    # ------------------------------------------------------------------ #
    def _straight_only(self, *a, **k):
        raise RuntimeError(
            f"'{self.name}': a straight-axis SUPERPOSE_MAP routine was called on "
            "a curved cluster (SUPERPOSE_MAP_OUT) — this is a bug")

    _active = _straight_only
    _onaxis_dW = _straight_only
    _ref_walk = _straight_only
    _calibrate_child_sync_phase = _straight_only
    _probe_cluster_voltage_integral = _straight_only

    def untrack_rk4(self, *a, **k):
        raise NotImplementedError(
            f"'{self.name}': backtracking through a curved field-map cluster "
            "(SUPERPOSE_MAP_OUT) is not supported — TraceWin does not offer it "
            "either")

    def __repr__(self) -> str:
        kids = ", ".join(f"{child.name}@{z0:g}mm" for z0, child in self.children)
        return (f"CurvedFieldMapCluster({self.name!r}, path={self.length:g} mm"
                f"{' (provisional)' if self.length_provisional else ''}, "
                f"exit={self.exit_frame}, [{kids}])")
