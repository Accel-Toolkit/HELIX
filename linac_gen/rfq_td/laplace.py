"""Laplace's equation for the RFQ electrodes (time-domain solver).

Second-order finite differences on a tensor grid of the quadrant
x, y >= 0 (the vane pairs are mirror images), with Shortley–Weller
cut-cell distances to the electrode surfaces in all three directions,
mirror (Neumann) conditions on the symmetry planes and on the far faces
of the box, solved with algebraic multigrid (pyamg).  The RFQ is solved
in overlapping z windows; each window is padded on both ends so the
artificial end condition decays before the stored core (20 mm pads
leave ~1e-4 of the vane voltage at r < 3 mm on PXIE).

Validated against an exact solution: electrodes shaped as equipotentials
of the two-term potential, outer faces held at it — the solver
reproduces the potential to 2e-4 (h = 0.2 mm) / 5e-5 (h = 0.1 mm) of
V/2, second order (tests/rfq_td/test_laplace.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from linac_gen.core.amg import seeded


def _pyamg():
    try:
        import pyamg
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "building a time-domain RFQ field map needs pyamg "
            "(`pip install pyamg`, or the HELIX 'rfq' extra); a map "
            "that is already cached loads without it") from exc
    return pyamg


def stretched_axis(h_fine: float, x_fine: float, x_max: float,
                   growth: float = 1.12, h_max: float = 0.6) -> np.ndarray:
    """0, h, 2h … x_fine, then geometrically growing steps up to x_max."""
    n = int(round(x_fine / h_fine))
    xs = list(h_fine * np.arange(n + 1))
    h = h_fine
    while xs[-1] < x_max - 1e-9:
        h = min(h * growth, h_max)
        xs.append(min(xs[-1] + h, x_max))
    return np.asarray(xs)


def solve_window(sdf, xs, ys, zs, *, tol: float = 1e-10,
                 theta_min: float = 1e-3, dirichlet=None,
                 dirichlet_faces=("x1", "y1"), source=None,
                 info: dict | None = None) -> np.ndarray:
    """Shortley–Weller Laplace (or Poisson) solve on the grid xs × ys × zs.

    ``sdf(X, Y, Z) -> (d, which, V)``: signed distance to the electrodes
    (negative inside metal) and the potential of the nearest electrode.
    x = 0 and y = 0 are mirror planes; the far faces are mirror planes
    too unless ``dirichlet`` (a callable g(X, Y, Z)) is given, which
    then holds the faces named in ``dirichlet_faces`` ("x1", "y1", "z0",
    "z1").  ``source`` (same shape as the grid, V/mm²) turns it into
    Poisson's equation ∇²φ = source.  Returns φ in volts on the grid,
    metal nodes carrying their electrode's potential.
    """
    pyamg = _pyamg()
    t0 = time.time()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d, _which, Vnode = sdf(X, Y, Z)
    inside = d < 0.0
    fixed = inside.copy()
    Vfix = np.where(inside, Vnode, 0.0)
    if dirichlet is not None:
        face = np.zeros(d.shape, bool)
        if "x1" in dirichlet_faces:
            face[-1, :, :] = True
        if "y1" in dirichlet_faces:
            face[:, -1, :] = True
        if "z0" in dirichlet_faces:
            face[:, :, 0] = True
        if "z1" in dirichlet_faces:
            face[:, :, -1] = True
        face &= ~inside
        Vfix = np.where(face, dirichlet(X, Y, Z), Vfix)
        fixed |= face
    free = ~fixed
    nfree = int(free.sum())
    idx = np.full(d.shape, -1, np.int64)
    idx[free] = np.arange(nfree)

    rows, cols, vals = [], [], []
    rhs = np.zeros(nfree)
    diag = np.zeros(nfree)
    for ax, c in enumerate((xs, ys, zs)):
        n = c.size
        dl = np.diff(c)
        ip = np.arange(n) + 1
        ip[-1] = n - 2                       # mirror at the far face
        im = np.arange(n) - 1
        im[0] = 1                            # mirror at the near face
        hp = np.empty(n)
        hp[:-1] = dl
        hp[-1] = dl[-1]
        hm = np.empty(n)
        hm[1:] = dl
        hm[0] = dl[0]
        shape = [1, 1, 1]
        shape[ax] = n
        nb = {}
        for tag, sel, hh in (("p", ip, hp), ("m", im, hm)):
            H = np.broadcast_to(hh.reshape(shape), d.shape)
            d_nb = np.take(d, sel, axis=ax)
            in_nb = np.take(inside, sel, axis=ax)
            fx_nb = np.take(fixed, sel, axis=ax)
            V_nb = np.take(Vfix, sel, axis=ax)
            i_nb = np.take(idx, sel, axis=ax)
            cut = free & in_nb                # an electrode surface between
            th = np.ones(d.shape)
            th[cut] = np.clip(d[cut] / (d[cut] - d_nb[cut]), theta_min, 1.0)
            nb[tag] = (H * th, free & fx_nb, V_nb, i_nb)
        hpe, fxp, Vp, ipn = nb["p"]
        hme, fxm, Vm, imn = nb["m"]
        cp = 2.0 / ((hpe + hme) * hpe)
        cm = 2.0 / ((hpe + hme) * hme)
        np.add.at(diag, idx[free], -(cp[free] + cm[free]))
        for cc, fx, Vb, inb in ((cp, fxp, Vp, ipn), (cm, fxm, Vm, imn)):
            mk = free & ~fx
            rows.append(idx[mk])
            cols.append(inb[mk])
            vals.append(cc[mk])
            mb = free & fx
            np.add.at(rhs, idx[mb], -cc[mb] * Vb[mb])
    rows.append(np.arange(nfree))
    cols.append(np.arange(nfree))
    vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nfree, nfree))
    if source is not None:
        rhs = rhs + np.asarray(source, float)[free]
    s = 1.0 / np.abs(A.diagonal())          # unit diagonal for the AMG
    A = sp.diags(s) @ A
    b = rhs * s
    t1 = time.time()
    res = []
    with seeded():          # the same solver, and solution, every time
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="nonsymmetric",
                                               max_coarse=2000)
        sol = ml.solve(b, x0=np.zeros(nfree), tol=tol, accel="bicgstab",
                       residuals=res, maxiter=400)
    phi = Vfix.astype(np.float64)
    phi[free] = sol
    if info is not None:
        rr = float(np.linalg.norm(b - A @ sol)
                   / max(np.linalg.norm(b), 1e-300))
        info.update(unknowns=nfree, assemble_s=t1 - t0,
                    solve_s=time.time() - t1, iterations=len(res),
                    rel_residual=rr)
    return phi


@dataclass(frozen=True)
class Window:
    """One solve: stored core [z0, z1] (mm), longitudinal step, outer
    box half-size and stored near-axis half-size (mm)."""

    z0: float
    z1: float
    hz: float
    x_max: float
    x_store: float


def plan_windows(edges_mm: np.ndarray, electrodes, *, hz_per_cell: int = 32,
                 min_core_mm: float = 40.0, end_hz: float = 0.2,
                 upstream_mm: float = 20.0, downstream_mm: float = 20.0,
                 x_max: float = 16.0, x_store_min: float = 8.0,
                 x_store_max: float = 12.0) -> list:
    """Window plan along the chain: an entrance window covering the
    upstream drift, the entrance gap and the first cell; bulk windows of
    whole cells (core >= ``min_core_mm``, h_z = shortest cell /
    ``hz_per_cell``); an exit window over the last two cells, the exit
    gap and ``downstream_mm`` beyond the exit plate."""
    e = np.asarray(edges_mm, float)
    if e.size < 4:
        raise ValueError("the time-domain RFQ solver needs at least 3 cells")
    L = np.diff(e)

    def box(z0, z1):
        m = (electrodes.z >= z0 - 1.0) & (electrodes.z <= z1 + 1.0)
        if m.any():
            reach = max((electrodes.R_x[m] + 2 * electrodes.rho_x[m]).max(),
                        (electrodes.R_y[m] + 2 * electrodes.rho_y[m]).max())
            tip = max(electrodes.R_x[m].max(), electrodes.R_y[m].max())
        else:
            reach, tip = x_max, x_store_min
        xm = max(x_max, float(np.ceil(reach + 2.0)))
        xs = float(np.clip(np.ceil(tip + 0.5), x_store_min, x_store_max))
        return xm, xs

    wins = []
    z0, z1 = e[0] - upstream_mm, e[1]
    wins.append(Window(z0, z1, min(end_hz, L[0] / hz_per_cell), *box(z0, z1)))
    i, last = 1, e.size - 3
    while i < last:
        j = i + 1
        while j < last and e[j] - e[i] < min_core_mm:
            j += 1
        z0, z1 = e[i], e[j]
        hz = float(np.clip(L[i:j].min() / hz_per_cell, 0.05, 2.0))
        wins.append(Window(z0, z1, hz, *box(z0, z1)))
        i = j
    z0, z1 = e[last], e[-1] + downstream_mm
    wins.append(Window(z0, z1, min(end_hz, L[last:].min() / hz_per_cell),
                       *box(z0, z1)))
    return wins


def solve_block(electrodes, win: Window, *, h: float = 0.1,
                pad_mm: float = 20.0, info: dict | None = None):
    """Solve one window and return its stored block
    ``(z_first, hz, phi[nxs, nxs, nzs] float32)``: the near-axis core plus
    two stencil slices on each side."""
    xs = stretched_axis(h, win.x_store, win.x_max)
    ncore = max(int(round((win.z1 - win.z0) / win.hz)), 1)
    hz = (win.z1 - win.z0) / ncore
    npd = int(np.ceil(pad_mm / hz))
    zs = win.z0 + hz * np.arange(-npd, ncore + npd + 1)
    phi = solve_window(electrodes.sdf_quadrant, xs, xs, zs, info=info)
    ns = int(round(win.x_store / h)) + 1
    k0, k1 = npd - 2, npd + ncore + 3
    return float(zs[k0]), float(hz), np.ascontiguousarray(
        phi[:ns, :ns, k0:k1], dtype=np.float32)
