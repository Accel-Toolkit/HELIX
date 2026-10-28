"""The beam pipe of the space-charge kicks: the harmonic correction that makes
the potential zero on a grounded wall.

Every boundary is a harmonic polynomial h(w) = Re sum_n c_n (w/R)^n in
w = (x - x_p) + i (y - y_p) about the pipe's centre, added to the free-space
potential (and its gradient to the field):

* **open** — no wall: c_0 = Q ln r_ref, the potential zero at the reference
  radius r_ref (what the longitudinal kick needs; the transverse field is the
  free-space one);
* **round** (radius b) — exact images: a line charge q at w_j inside the
  circle has its image -q at b^2/conj(w_j), whose potential inside is
  q ln b - q Re sum_n (w conj(w_j) / b^2)^n / n, so c_0 = Q ln b and
  c_n = -conj(M_n) / (n b^n) with R = b and the moments M_n = sum q w_j^n;
  the series is taken to the order where (r_max/b)^n < 1e-13 (r_max the
  farthest charge: the largest term anywhere inside the pipe), at most
  :data:`ROUND_MAX_ORDER`; charge at or beyond the wall — the share a
  particle close to it spreads onto a node past it, or particles no aperture
  removed — is left out of the images (``charge_beyond`` says whether the
  last solve had some; the space charge counts the kicks where some was);
* **rectangle / ellipse** — the method of F. W. Jones, "A method for
  incorporating image forces in multiparticle tracking with space charge",
  EPAC 2000, p. 1381: the free-space potential at points on the wall, and the
  harmonic of ``order`` terms that cancels it there in the least-squares sense
  (its pseudo-inverse computed once per pipe by the one-sided Jacobi SVD in
  plain numpy — no BLAS, so the same bits for any number of BLAS threads —
  dropping the singular values below 1e-10 of the largest; a pipe the
  harmonics cannot fit is refused).

All the arithmetic on complex numbers is done on their real and imaginary
parts: numpy's complex multiply rounds differently depending on where the
arrays sit in memory.

The pipe only grounds the field: particles are lost at the lattice's
apertures, not at this wall; one beyond the wall (where the harmonic need not
converge) feels the free-space field only.  A boundary holds no state between
kicks (a round pipe's ``charge_beyond`` is its last solve's).
"""
from __future__ import annotations

import math

import numpy as np

from linac_gen.ring.sc.kernels import ROUND_MAX_ORDER, SERIES_TOL

#: wall points per fitted harmonic coefficient (real and imaginary parts)
FIT_POINTS_PER_TERM = 2
#: the fit's check: wall points per point of the fit (evenly spaced, a
#: rectangle's corners and an ellipse's axis ends among them)
CHECK_POINTS_PER_FIT_POINT = 16


class OpenBoundary:
    shape = "open"

    def __init__(self, reference_radius: float):
        self.r_ref = float(reference_radius)
        self.centre = (0.0, 0.0)
        self.scale = 1.0

    def coefficients(self, q, X, Y, *, potential_at=None) -> np.ndarray:
        return np.array([float(np.sum(q)) * math.log(self.r_ref)], dtype=complex)

    def inside(self, x, y) -> np.ndarray:
        return np.ones(np.shape(x), dtype=bool)


class RoundPipe:
    shape = "round"

    def __init__(self, radius: float):
        self.b = float(radius)
        self.centre = (0.0, 0.0)
        self.scale = self.b
        self.charge_beyond = False      # whether the last solve had charge at or beyond the wall

    def coefficients(self, q, X, Y, *, potential_at=None) -> np.ndarray:
        w = (X - self.centre[0]) + 1j * (Y - self.centre[1])
        inside = np.abs(w) < self.b * (1.0 - 1e-9)
        self.charge_beyond = bool(np.any((q != 0.0) & ~inside))
        sel = (q != 0.0) & inside
        qs = q[sel]
        w = w[sel]
        Q = float(np.sum(qs))
        if qs.size == 0:
            return np.array([0.0], dtype=complex)
        wr = np.ascontiguousarray(w.real)
        wi = np.ascontiguousarray(w.imag)
        rmax = float(np.sqrt(np.max(wr * wr + wi * wi)))
        rho = rmax / self.b
        # (the series at a point w from the charges: terms (|w| r_max / b^2)^n,
        # at most rho^n anywhere inside the pipe)
        n = 1 if rho <= 0.0 else int(min(ROUND_MAX_ORDER, max(1, math.ceil(
            math.log(SERIES_TOL) / math.log(rho)))))
        c = np.zeros(n + 1, dtype=complex)
        c[0] = Q * math.log(self.b)
        sr, si = wr / self.b, wi / self.b
        pr = np.ascontiguousarray(qs, dtype=float)
        pi = np.zeros_like(pr)
        for k in range(1, n + 1):
            pr, pi = pr * sr - pi * si, pr * si + pi * sr
            # -conj(sum p)/k
            c[k] = complex(-float(np.sum(pr)) / k, float(np.sum(pi)) / k)
        return c

    def inside(self, x, y) -> np.ndarray:
        u = np.asarray(x) - self.centre[0]
        v = np.asarray(y) - self.centre[1]
        return u * u + v * v < self.b * self.b


class FittedPipe:
    """A rectangle or an ellipse (half width a, half height b): a harmonic of
    ``order`` terms fitted to cancel the free-space potential on the wall.

    The fit's quality is measured once, for unit line charges at the centre
    and out to half the aperture: the largest potential left on the wall,
    ``fit_residual`` — on :data:`CHECK_POINTS_PER_FIT_POINT` times as many
    wall points as the fit's, evenly spaced, among them a rectangle's corners
    (where the harmonics fit it worst) and an ellipse's axis ends; in the
    units of Phi, where the potential of a unit charge changes by 1 per
    e-fold of distance.  The fit's error is harmonic inside the wall, so this
    bounds the error of the potential anywhere inside for a beam within half
    the aperture.  A pipe the harmonics cannot fit (beyond
    :data:`FIT_RESIDUAL_MAX`: a flat one) is refused."""

    def __init__(self, shape: str, a: float, b: float, order: int):
        if shape not in ("rectangle", "ellipse"):
            raise ValueError(f"space charge: pipe {shape!r} is not a rectangle or an ellipse")
        self.shape = shape
        self.a, self.b = float(a), float(b)
        self.order = int(order)
        self.centre = (0.0, 0.0)
        self.scale = max(self.a, self.b)
        m = FIT_POINTS_PER_TERM * (2 * self.order + 1)
        self.px, self.py = _wall_points(shape, self.a, self.b, m)
        self._A = _harmonic_columns(self.px / self.scale, self.py / self.scale, self.order)
        self.pinv = _pinv_jacobi(self._A)
        self.fit_residual = self._check_fit()
        if self.fit_residual > FIT_RESIDUAL_MAX:
            raise ValueError(
                f"space charge: a {shape} pipe of half widths {self.a:g} x {self.b:g} m is not "
                f"fitted by {self.order} harmonics (a unit line charge within half its "
                f"aperture leaves a potential of {self.fit_residual:.2g} on the wall, more than "
                f"{FIT_RESIDUAL_MAX:g}): try more harmonics (up to 64 — the residual does not "
                "always fall with their number), a round pipe, or no pipe")

    def _check_fit(self) -> float:
        qx, qy = _check_points(self.shape, self.a, self.b,
                               CHECK_POINTS_PER_FIT_POINT * len(self.px))
        worst = 0.0
        for x0, y0 in ((0.0, 0.0), (0.3 * self.a, -0.2 * self.b), (0.5 * self.a, 0.0),
                       (0.0, 0.5 * self.b), (0.35 * self.a, 0.35 * self.b)):
            def free(px, py, x0=x0, y0=y0):
                return -0.5 * np.log((px - x0) ** 2 + (py - y0) ** 2)
            c = self.coefficients(None, None, None, potential_at=free)
            h, _fx, _fy = harmonic_on_grid(c, self.scale, self.centre, qx, qy, potential=True)
            worst = max(worst, float(np.max(np.abs(free(qx, qy) + h))))
        return worst

    def coefficients(self, q, X, Y, *, potential_at=None) -> np.ndarray:
        if potential_at is None:
            raise ValueError("space charge: the fitted pipe needs the wall potential")
        phi = np.ascontiguousarray(potential_at(self.px, self.py), dtype=float)
        sol = -np.sum(self.pinv * phi[None, :], axis=1)
        c = np.zeros(self.order + 1, dtype=complex)
        c[0] = sol[0]
        c[1:] = sol[1::2] + 1j * sol[2::2]
        return c

    def inside(self, x, y) -> np.ndarray:
        u = (np.asarray(x) - self.centre[0]) / self.a
        v = (np.asarray(y) - self.centre[1]) / self.b
        if self.shape == "rectangle":
            return (np.abs(u) < 1.0) & (np.abs(v) < 1.0)
        return u * u + v * v < 1.0


def _harmonic_columns(sr, si, order: int) -> np.ndarray:
    """The fit's matrix: 1, Re s^n, -Im s^n (n = 1..order) at the wall points
    s = sr + i si (real arithmetic: numpy's complex multiply may round
    otherwise depending on where its arrays sit in memory)."""
    sr = np.ascontiguousarray(sr, dtype=float)
    si = np.ascontiguousarray(si, dtype=float)
    cols = [np.ones(sr.size)]
    pr, pi = np.ones(sr.size), np.zeros(sr.size)
    for _n in range(order):
        pr, pi = pr * sr - pi * si, pr * si + pi * sr
        cols += [pr, -pi]
    return np.ascontiguousarray(np.stack(cols, axis=1))


#: a fitted pipe's singular values below this fraction of the largest are dropped
PINV_RCOND = 1e-10
#: the largest potential a fitted pipe may leave on its wall for a unit line
#: charge within half its aperture (Phi's units: 1 per e-fold of distance)
FIT_RESIDUAL_MAX = 0.01


def _pinv_jacobi(A: np.ndarray, rcond: float = PINV_RCOND, sweeps: int = 80) -> np.ndarray:
    """The pseudo-inverse of A (m x n, m >= n) by the one-sided Jacobi SVD
    (Hestenes): column pairs rotated until orthogonal, in a fixed order, with
    elementwise operations and numpy's sums only — no BLAS or LAPACK, whose
    results may change with their threads — the singular values below
    ``rcond`` of the largest dropped (an ill-conditioned fit stays bounded)."""
    m, n = A.shape
    U = np.ascontiguousarray(np.array(A, dtype=float).T)      # (n, m): A's columns
    V = np.eye(n)                                             # (n, n): V's columns as rows
    for _sweep in range(sweeps):
        off = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                ui, uj = U[i], U[j]
                alpha = float(np.sum(ui * ui))
                beta = float(np.sum(uj * uj))
                gamma = float(np.sum(ui * uj))
                if gamma == 0.0 or alpha == 0.0 or beta == 0.0:
                    continue
                rel = abs(gamma) / math.sqrt(alpha * beta)
                off = max(off, rel)
                if rel < 1e-15:
                    continue
                zeta = (beta - alpha) / (2.0 * gamma)
                tt = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + tt * tt)
                s = c * tt
                U[i], U[j] = c * ui - s * uj, s * ui + c * uj
                vi, vj = V[i], V[j]
                V[i], V[j] = c * vi - s * vj, s * vi + c * vj
        if off < 1e-15:
            break
    sig = np.sqrt(np.sum(U * U, axis=1))
    smax = float(np.max(sig)) if sig.size else 0.0
    P = np.zeros((n, m))
    for k in range(n):
        if sig[k] > rcond * smax:
            P += np.multiply.outer(V[k], U[k]) / (sig[k] * sig[k])
    return np.ascontiguousarray(P)


def _wall_points(shape, a, b, m):
    """The fit's m points on the wall: equal steps of the angle for an
    ellipse, equal steps along the perimeter for a rectangle, offset by half
    a step (from the angle 0; from a corner)."""
    if shape == "ellipse":
        t = 2.0 * np.pi * (np.arange(m) + 0.5) / m
        return a * np.cos(t), b * np.sin(t)
    per = 4.0 * (a + b)
    d = per * (np.arange(m) + 0.5) / m
    x = np.empty(m)
    y = np.empty(m)
    for k, s in enumerate(d):
        if s < 2 * a:                       # bottom, left to right
            x[k], y[k] = -a + s, -b
        elif s < 2 * a + 2 * b:             # right, upwards
            x[k], y[k] = a, -b + (s - 2 * a)
        elif s < 4 * a + 2 * b:             # top, right to left
            x[k], y[k] = a - (s - 2 * a - 2 * b), b
        else:                               # left, downwards
            x[k], y[k] = -a, b - (s - 4 * a - 2 * b)
    return x, y


def _check_points(shape, a, b, m):
    """About m points on the wall for the fit's check, evenly spaced: an
    ellipse's at equal steps of the angle from its axis (its four axis ends
    among them), a rectangle's on each side from corner to corner."""
    if shape == "ellipse":
        n = 4 * (-(-int(m) // 4))
        t = 2.0 * np.pi * np.arange(n) / n
        return a * np.cos(t), b * np.sin(t)
    na = max(1, math.ceil(m * a / (2.0 * (a + b))))      # (the steps along a side of 2a)
    nb = max(1, math.ceil(m * b / (2.0 * (a + b))))
    xs = np.linspace(-a, a, na + 1)
    ys = np.linspace(-b, b, nb + 1)
    return (np.concatenate([xs, np.full(nb + 1, a), xs, np.full(nb + 1, -a)]),
            np.concatenate([np.full(na + 1, -b), ys, np.full(na + 1, b), ys]))


def make_boundary(cfg):
    """The boundary of a :class:`~linac_gen.ring.sc.config.SpaceChargeConfig`."""
    if cfg.pipe == "open":
        return OpenBoundary(cfg.reference_radius_m)
    if cfg.pipe == "round":
        return RoundPipe(cfg.pipe_a_m)
    return FittedPipe(cfg.pipe, cfg.pipe_a_m, cfg.pipe_b_m, cfg.boundary_order)


def harmonic_on_grid(c, scale, centre, X, Y, *, potential: bool):
    """(h or None, F_x, F_y) of the harmonic Re sum c_n (w/R)^n at the nodes
    (Horner's rule on the real and imaginary parts)."""
    sr = (np.asarray(X, dtype=float) - centre[0]) / scale
    si = (np.asarray(Y, dtype=float) - centre[1]) / scale
    n = len(c) - 1
    pr = np.full(sr.shape, c[n].real)
    pi = np.full(sr.shape, c[n].imag)
    if n >= 1:
        dr = np.full(sr.shape, n * c[n].real)
        di = np.full(sr.shape, n * c[n].imag)
    else:
        dr = np.zeros(sr.shape)
        di = np.zeros(sr.shape)
    for k in range(n - 1, -1, -1):
        if potential:
            pr, pi = pr * sr - pi * si + c[k].real, pr * si + pi * sr + c[k].imag
        if k >= 1:
            dr, di = dr * sr - di * si + k * c[k].real, dr * si + di * sr + k * c[k].imag
    fx = -dr / scale
    fy = di / scale
    return (pr if potential else None), fx, fy
