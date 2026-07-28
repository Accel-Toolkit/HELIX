"""M3.2 — 3-D Laplace solver with embedded vane boundaries.

Engaged via ``field_model="laplace3d"``.  Solves the full 3-D Laplace
equation on a Cartesian ``(Nz × Nx × Ny)`` grid with the four hyperbolic
vane surfaces embedded as Dirichlet boundaries.

Boundary modes
--------------
``boundary="binary"`` — pin Φ = V_vane on every grid node that falls
*inside* the vane material.  Simple but produces sub-grid stair-stepping
in the embedded surface representation, which on coarse grids dominates
the field gradients at the axis.  Catastrophic on the PXIE benchmark
(σ_x ~ 1e4 mm).  Kept as a debugging fallback only.

``boundary="shortley_weller"`` (default) — Toutatis-equivalent embedded
boundary scheme (Duperrier 2000 thesis, Eq. 4.17).  For each free-space
node whose 6 axial neighbours might cross a vane surface, we compute
the *fractional* distance ``h_i ≤ h`` to the surface along each axis
and use those distances directly in the FD stencil:

  ∂²Φ/∂x²|₀ ≈ 2·[Φ_e/(h_e(h_e+h_w)) + Φ_w/(h_w(h_e+h_w))
                 − Φ_0/(h_e·h_w)]

(and similarly for y, z; in 3-D the Laplacian sums the three forms.)
Vane voltages enter the RHS via the boundary term ``Φ_e = V_vane`` when
the east neighbour is a conductor.  This eliminates the stair-step
error on the typical RFQ grid resolution; convergence to TraceWin/
Toutatis becomes a function of grid resolution alone.

Solver
------
Default solver is **pyamg** smoothed-aggregation V-cycle when the
package is available (multigrid is what Toutatis itself uses for its
Poisson solve, Toutatis_3.pdf §4.1).  Falls back to
``scipy.sparse.linalg.spsolve`` if pyamg is missing.

Cost
----
* PXIE LEBT+RFQ: ``nx=ny=33, z_subsample=8`` → 2.4·10⁶ unknowns.
  pyamg V-cycle setup ~5 s, solve ~1 s.  spsolve ~30 s, much higher
  memory (LU fill-in).
* Memory: ``Φ_static`` cache is ``Nz × Nx × Ny × 4 B`` (float32) → ~10 MB
  at default resolution.

API mirrors :class:`Laplace2DCache` so ``vane_rfq.py`` can swap solvers
behind the same dispatch.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

try:
    import pyamg
    _HAVE_PYAMG = True
except ImportError:
    _HAVE_PYAMG = False

from linac_gen.elements.vane_rfq_laplace2d import (
    _Caches,
    _vane_masks,
)
from linac_gen.io.tracewin_vane import VaneGeometry


# --------------------------------------------------------------------
# Shortley-Weller distance-to-vane helpers
# --------------------------------------------------------------------
def _dist_to_vane_x(x0_m: float, y0_m: float, a_v_m: float,
                    direction: int) -> float:
    """Distance along ±x from (x0, y0) to the hyperbolic vane
    ``x² − y² = a_v²`` in direction ``direction`` (+1 or −1).

    The vane surface in the +x branch is at ``x = √(a_v² + y²)``;
    in the −x branch at ``x = −√(a_v² + y²)``.  Returns ``+∞`` if
    the ray from (x0,y0) along ``direction·x̂`` does not cross the
    surface within the half-plane.
    """
    if a_v_m <= 0.0:
        return float('inf')
    x_surf = np.sqrt(a_v_m * a_v_m + y0_m * y0_m)
    if direction > 0:
        # Going in +x direction; surface in +x half-plane at +x_surf.
        d = x_surf - x0_m
        return d if d > 0.0 else float('inf')
    else:
        # Going in −x direction; surface in −x half-plane at −x_surf.
        d = x0_m - (-x_surf)
        return d if d > 0.0 else float('inf')


def _dist_to_vane_y(x0_m: float, y0_m: float, a_v_m: float,
                    direction: int) -> float:
    """Distance along ±y to the hyperbolic vane ``y² − x² = a_v²``
    (vanes 2 and 4)."""
    if a_v_m <= 0.0:
        return float('inf')
    y_surf = np.sqrt(a_v_m * a_v_m + x0_m * x0_m)
    if direction > 0:
        d = y_surf - y0_m
        return d if d > 0.0 else float('inf')
    else:
        d = y0_m - (-y_surf)
        return d if d > 0.0 else float('inf')


def _shortley_weller_distances_slice(
    x_m: np.ndarray, y_m: np.ndarray, dx_m: float, dy_m: float,
    a1: float, a2: float, a3: float, a4: float,
    V1: float, V2: float, V3: float, V4: float,
    inside_mask: np.ndarray,
):
    """For one z-slice, compute per-cell Shortley-Weller transverse
    distances and BC voltages.

    Returns
    -------
    h_xp, h_xm, h_yp, h_ym : (nx, ny) float arrays
        Fractional distances (m) to the *nearest* obstacle in each
        direction.  Equals ``dx_m`` (or ``dy_m``) when the neighbour
        cell is also free; less than that when a vane surface is
        between the central cell and its neighbour.
    bc_xp, bc_xm, bc_yp, bc_ym : (nx, ny) float arrays
        Voltage applied at the obstacle (volts).  Zero on cells where
        the corresponding ``h_*`` equals the full grid step (no BC).
    is_bc_xp, is_bc_xm, is_bc_yp, is_bc_ym : (nx, ny) bool arrays
        True iff the corresponding direction has a vane within ``h``
        of the central cell (so a Dirichlet BC term enters the RHS).
    inside_mask : (nx, ny) bool — inside-vane mask at this slice
        (re-supplied by the caller).
    """
    nx = x_m.size
    ny = y_m.size
    h_xp = np.full((nx, ny), dx_m)
    h_xm = np.full((nx, ny), dx_m)
    h_yp = np.full((nx, ny), dy_m)
    h_ym = np.full((nx, ny), dy_m)
    bc_xp = np.zeros((nx, ny))
    bc_xm = np.zeros((nx, ny))
    bc_yp = np.zeros((nx, ny))
    bc_ym = np.zeros((nx, ny))
    is_bc_xp = np.zeros((nx, ny), dtype=bool)
    is_bc_xm = np.zeros((nx, ny), dtype=bool)
    is_bc_yp = np.zeros((nx, ny), dtype=bool)
    is_bc_ym = np.zeros((nx, ny), dtype=bool)

    # For each free cell, scan in each of the four cardinal directions:
    # if the neighbour is inside-vane, find the actual fractional
    # distance to the vane surface along that ray; the BC voltage is
    # the voltage of the vane the ray hits.
    for i in range(nx):
        for j in range(ny):
            if inside_mask[i, j]:
                continue
            x0 = x_m[i]; y0 = y_m[j]
            # +x neighbour: cell (i+1, j)
            if i + 1 < nx and inside_mask[i + 1, j]:
                # Find which vane bites in +x — try +x vane (a1) and
                # also -y/+y vanes whose surface might bend into +x
                # (only relevant near the axis-diagonal corners).
                d_v1 = _dist_to_vane_x(x0, y0, a1, +1)
                d_v3 = _dist_to_vane_x(x0, y0, a3, +1)  # tiny if x0<0
                # Pick the smallest positive d.
                d_min = min(d_v1, d_v3, dx_m)
                if d_min < dx_m:
                    h_xp[i, j] = max(d_min, 0.05 * dx_m)
                    bc_xp[i, j] = V1 if d_v1 <= d_v3 else V3
                    is_bc_xp[i, j] = True
            # −x neighbour: cell (i-1, j)
            if i - 1 >= 0 and inside_mask[i - 1, j]:
                d_v1 = _dist_to_vane_x(x0, y0, a1, -1)
                d_v3 = _dist_to_vane_x(x0, y0, a3, -1)
                d_min = min(d_v1, d_v3, dx_m)
                if d_min < dx_m:
                    h_xm[i, j] = max(d_min, 0.05 * dx_m)
                    bc_xm[i, j] = V3 if d_v3 <= d_v1 else V1
                    is_bc_xm[i, j] = True
            # +y neighbour: cell (i, j+1)
            if j + 1 < ny and inside_mask[i, j + 1]:
                d_v2 = _dist_to_vane_y(x0, y0, a2, +1)
                d_v4 = _dist_to_vane_y(x0, y0, a4, +1)
                d_min = min(d_v2, d_v4, dy_m)
                if d_min < dy_m:
                    h_yp[i, j] = max(d_min, 0.05 * dy_m)
                    bc_yp[i, j] = V2 if d_v2 <= d_v4 else V4
                    is_bc_yp[i, j] = True
            # −y neighbour
            if j - 1 >= 0 and inside_mask[i, j - 1]:
                d_v2 = _dist_to_vane_y(x0, y0, a2, -1)
                d_v4 = _dist_to_vane_y(x0, y0, a4, -1)
                d_min = min(d_v2, d_v4, dy_m)
                if d_min < dy_m:
                    h_ym[i, j] = max(d_min, 0.05 * dy_m)
                    bc_ym[i, j] = V4 if d_v4 <= d_v2 else V2
                    is_bc_ym[i, j] = True

    return (h_xp, h_xm, h_yp, h_ym,
            bc_xp, bc_xm, bc_yp, bc_ym,
            is_bc_xp, is_bc_xm, is_bc_yp, is_bc_ym)


class Laplace3DCache:
    """3-D Laplace cache with vane Dirichlet BCs.

    Parameters
    ----------
    vane : VaneGeometry
    nx, ny : int
        Transverse grid resolution.
    box_factor : float
        Half-width of the simulation box, in units of the maximum
        aperture across all slices (default 1.5).
    z_subsample : int
        Sub-sample the .vane z grid (default 8 — keeps the unknown
        count tractable).
    boundary : {"shortley_weller", "binary"}
        Embedded-boundary scheme (default Shortley-Weller — Toutatis
        equivalent).
    solver : {"auto", "pyamg", "spsolve"}
        Linear-system solver; ``"auto"`` picks pyamg when available.
    verbose : bool

    Notes
    -----
    For the PXIE benchmark the practical setup is
    ``nx=ny=33, z_subsample=8`` → ~2.4·10⁶ unknowns.  pyamg V-cycle
    setup ~5 s, solve ~1 s on the default grid.
    """

    def __init__(self, vane: VaneGeometry, *,
                 nx: int = 33, ny: int = 33,
                 box_factor: float = 1.5,
                 z_subsample: int = 8,
                 boundary: str = "shortley_weller",
                 solver: str = "auto",
                 verbose: bool = False):
        if nx < 5 or ny < 5:
            raise ValueError("Laplace3DCache needs nx, ny ≥ 5")
        if box_factor <= 1.0:
            raise ValueError("box_factor must be > 1.0")
        if boundary not in {"shortley_weller", "binary"}:
            raise ValueError(
                f"boundary must be 'shortley_weller' or 'binary'; "
                f"got {boundary!r}")
        if solver not in {"auto", "pyamg", "spsolve"}:
            raise ValueError(
                f"solver must be 'auto', 'pyamg', or 'spsolve'; "
                f"got {solver!r}")
        self.boundary = boundary
        if solver == "auto":
            solver = "pyamg" if _HAVE_PYAMG else "spsolve"
        if solver == "pyamg" and not _HAVE_PYAMG:
            raise RuntimeError(
                "pyamg requested but not installed; "
                "`pip install pyamg` or use solver='spsolve'.")
        self.solver = solver

        # ---- z grid ---------------------------------------------------
        if z_subsample < 1:
            z_subsample = 1
        if z_subsample == 1:
            idx = np.arange(vane.n_slices)
        else:
            idx = np.arange(0, vane.n_slices, z_subsample)
            if idx[-1] != vane.n_slices - 1:
                idx = np.append(idx, vane.n_slices - 1)
        self._z_idx = idx
        self.nz = idx.size
        self.z_mm = vane.z[idx] * 1000.0
        self.z_m = vane.z[idx].copy()

        self._a1 = np.asarray(vane.aperture_v1[idx], dtype=float)
        self._a2 = np.asarray(vane.aperture_v2[idx], dtype=float)
        self._a3 = np.asarray(vane.aperture_v3[idx], dtype=float)
        self._a4 = np.asarray(vane.aperture_v4[idx], dtype=float)
        self._V1 = np.asarray(vane.voltage_v1[idx], dtype=float)
        self._V2 = np.asarray(vane.voltage_v2[idx], dtype=float)
        self._V3 = np.asarray(vane.voltage_v3[idx], dtype=float)
        self._V4 = np.asarray(vane.voltage_v4[idx], dtype=float)

        # ---- transverse grid -----------------------------------------
        a_max_m = float(np.max([self._a1.max(), self._a2.max(),
                                self._a3.max(), self._a4.max()]))
        x_lim_m = float(box_factor) * a_max_m
        self.nx = int(nx)
        self.ny = int(ny)
        self._x_m = np.linspace(-x_lim_m, x_lim_m, self.nx)
        self._y_m = np.linspace(-x_lim_m, x_lim_m, self.ny)
        self._dx_m = float(self._x_m[1] - self._x_m[0])
        self._dy_m = float(self._y_m[1] - self._y_m[0])
        self._dz_m = float(self.z_m[1] - self.z_m[0]) if self.nz > 1 else 1.0
        self._x_mm = self._x_m * 1000.0
        self._y_mm = self._y_m * 1000.0
        self._dx_mm = self._dx_m * 1000.0
        self._dy_mm = self._dy_m * 1000.0

        n_total = self.nz * self.nx * self.ny
        if verbose:
            mb = n_total * 4 / (1024 * 1024)
            print(f"Laplace3DCache: solving {n_total:,d} unknowns "
                  f"on ({self.nz}, {self.nx}, {self.ny}) grid "
                  f"(box ±{x_lim_m*1e3:.2f} mm, dz={self._dz_m*1e3:.3f} mm) "
                  f"— Φ cache ~{mb:.1f} MB float32, "
                  f"boundary={self.boundary}, solver={self.solver}")

        # ---- assemble inside-vane masks per slice --------------------
        inside_mask = np.zeros((self.nz, self.nx, self.ny), dtype=bool)
        bc_phi_inside = np.zeros((self.nz, self.nx, self.ny), dtype=float)
        for k in range(self.nz):
            m1, m2, m3, m4 = _vane_masks(
                self._x_m, self._y_m,
                float(self._a1[k]), float(self._a2[k]),
                float(self._a3[k]), float(self._a4[k]),
            )
            inside_mask[k] = m1 | m2 | m3 | m4
            bc_phi_inside[k][m1] = self._V1[k]
            bc_phi_inside[k][m2] = self._V2[k]
            bc_phi_inside[k][m3] = self._V3[k]
            bc_phi_inside[k][m4] = self._V4[k]

        # ---- assemble system -----------------------------------------
        if self.boundary == "binary":
            L, rhs = self._build_binary(inside_mask, bc_phi_inside,
                                        verbose=verbose)
        else:
            L, rhs = self._build_shortley_weller(
                inside_mask, bc_phi_inside, verbose=verbose)

        # ---- solve ---------------------------------------------------
        if verbose:
            print(f"  solving with {self.solver}...")
        if self.solver == "pyamg":
            ml = pyamg.smoothed_aggregation_solver(L)
            phi_flat, _ = self._pyamg_solve(ml, rhs, verbose=verbose)
        else:
            phi_flat = spsolve(L, rhs)
        self.phi_static = phi_flat.reshape(
            self.nz, self.nx, self.ny).astype(np.float32)
        if verbose:
            print(f"  done.")

        # Pin Φ on inside-vane nodes to the exact vane voltage so the
        # downstream interpolator returns physically meaningful values
        # for particles that drift into the vane region (very rare).
        self.phi_static[inside_mask] = bc_phi_inside[inside_mask].astype(
            np.float32)

        # ---- 3-D RGI for runtime particle queries --------------------
        self._rgi = RegularGridInterpolator(
            (self.z_mm, self._x_mm, self._y_mm),
            self.phi_static,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )

        # ---- on-axis derivative caches (envelope tracking) -----------
        ix_axis = int(np.argmin(np.abs(self._x_m)))
        iy_axis = int(np.argmin(np.abs(self._y_m)))
        if (ix_axis in (0, self.nx - 1)
                or iy_axis in (0, self.ny - 1)):
            raise ValueError("Laplace3DCache axis sample at grid edge")
        phi3 = phi_flat.reshape(self.nz, self.nx, self.ny).astype(np.float64)
        phi_axis = phi3[:, ix_axis, iy_axis].copy()
        phi_xp   = phi3[:, ix_axis + 1, iy_axis].copy()
        phi_xm   = phi3[:, ix_axis - 1, iy_axis].copy()
        phi_yp   = phi3[:, ix_axis, iy_axis + 1].copy()
        phi_ym   = phi3[:, ix_axis, iy_axis - 1].copy()
        K_xx = (phi_xp - 2.0 * phi_axis + phi_xm) / (self._dx_mm ** 2)
        K_yy = (phi_yp - 2.0 * phi_axis + phi_ym) / (self._dy_mm ** 2)
        dphi_dz = np.gradient(phi_axis, self.z_mm)
        Ez = -dphi_dz
        self._caches = _Caches(z_mm=self.z_mm.copy(),
                               K_xx=K_xx, K_yy=K_yy, Ez=Ez)
        self._phi_axis_f64 = phi_axis

    # ------------------------------------------------------------------
    # Binary mask assembler (legacy, kept as fallback)
    # ------------------------------------------------------------------
    def _build_binary(self, inside_mask, bc_phi_inside, *, verbose=False):
        nz, nx, ny = self.nz, self.nx, self.ny
        n_total = nz * nx * ny
        dx2 = 1.0 / (self._dx_m * self._dx_m)
        dy2 = 1.0 / (self._dy_m * self._dy_m)
        dz2 = 1.0 / (self._dz_m * self._dz_m)
        # 3-D 7-point Laplacian via direct row assembly so we can fix
        # boundary rows in the same pass.
        bc_mask = inside_mask.copy()
        bc_mask[:, 0, :] = True
        bc_mask[:, -1, :] = True
        bc_mask[:, :, 0] = True
        bc_mask[:, :, -1] = True
        bc_mask[0, :, :] = True
        bc_mask[-1, :, :] = True

        def _idx(k, i, j):
            return k * (nx * ny) + i * ny + j

        if verbose:
            print(f"  assembling binary-mask 3-D Laplacian...")
        # COO-style triplets for fast assembly.
        rows = []
        cols = []
        data = []
        rhs = np.zeros(n_total)
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    p = _idx(k, i, j)
                    if bc_mask[k, i, j]:
                        rows.append(p); cols.append(p); data.append(1.0)
                        rhs[p] = float(bc_phi_inside[k, i, j])
                        continue
                    diag = -2.0 * (dx2 + dy2 + dz2)
                    rows.append(p); cols.append(p); data.append(diag)
                    # x neighbours
                    rows.append(p); cols.append(_idx(k, i - 1, j)); data.append(dx2)
                    rows.append(p); cols.append(_idx(k, i + 1, j)); data.append(dx2)
                    # y neighbours
                    rows.append(p); cols.append(_idx(k, i, j - 1)); data.append(dy2)
                    rows.append(p); cols.append(_idx(k, i, j + 1)); data.append(dy2)
                    # z neighbours
                    rows.append(p); cols.append(_idx(k - 1, i, j)); data.append(dz2)
                    rows.append(p); cols.append(_idx(k + 1, i, j)); data.append(dz2)
        from scipy.sparse import coo_matrix
        L = coo_matrix(
            (data, (rows, cols)), shape=(n_total, n_total)
        ).tocsr()
        return L, rhs

    # ------------------------------------------------------------------
    # Shortley-Weller assembler (Toutatis-equivalent)
    # ------------------------------------------------------------------
    def _build_shortley_weller(self, inside_mask, bc_phi_inside, *,
                               verbose=False):
        """Assemble the 3-D Poisson operator with cut-cell stencils.

        Mathematical statement (Toutatis Eq. 4.17, 3-D extension):

          (1/h_xp + 1/h_xm)/((h_xp+h_xm)/2) · Φ_0
          − Φ_e/(h_xp·(h_xp+h_xm)/2)
          − Φ_w/(h_xm·(h_xp+h_xm)/2)
          + (similar for y, z)
          = ρ_0  (= 0 for a static Laplace)

        with all "missing" axial steps (no obstacle) equal to the
        uniform grid step h.  Conductor-side terms move to the RHS
        with their known voltage.

        Outer grid faces (k=0, k=nz-1, i=0, i=nx-1, j=0, j=ny-1) get
        a Neumann condition (Toutatis 4.3.1.5): ∂Φ/∂n = 0, implemented
        as a one-sided ghost ``Φ_outside = Φ_inside``.
        """
        nz, nx, ny = self.nz, self.nx, self.ny
        n_total = nz * nx * ny
        dx_m = self._dx_m; dy_m = self._dy_m; dz_m = self._dz_m

        def _idx(k, i, j):
            return k * (nx * ny) + i * ny + j

        # Pre-compute per-slice transverse SW distances.
        if verbose:
            print(f"  pre-computing per-slice Shortley-Weller distances...")
        slice_data = [None] * nz
        for k in range(nz):
            slice_data[k] = _shortley_weller_distances_slice(
                self._x_m, self._y_m, dx_m, dy_m,
                float(self._a1[k]), float(self._a2[k]),
                float(self._a3[k]), float(self._a4[k]),
                float(self._V1[k]), float(self._V2[k]),
                float(self._V3[k]), float(self._V4[k]),
                inside_mask[k],
            )

        if verbose:
            print(f"  assembling Shortley-Weller 3-D Laplacian...")
        rows = []; cols = []; data = []
        rhs = np.zeros(n_total)

        for k in range(nz):
            (h_xp, h_xm, h_yp, h_ym,
             bc_xp, bc_xm, bc_yp, bc_ym,
             is_bc_xp, is_bc_xm, is_bc_yp, is_bc_ym) = slice_data[k]
            inside = inside_mask[k]

            for i in range(nx):
                for j in range(ny):
                    p = _idx(k, i, j)
                    # Inside conductor → pin Φ.
                    if inside[i, j]:
                        rows.append(p); cols.append(p); data.append(1.0)
                        rhs[p] = float(bc_phi_inside[k, i, j])
                        continue

                    # ----- x direction --------------------------------
                    if i == 0:
                        # Neumann: Φ_w := Φ_0; effectively the −x term
                        # contributes 0 to the second derivative.
                        hxp = h_xp[i, j]; hxm = hxp
                        coeff_w = 0.0
                        rhs_w = 0.0
                        coeff_e = 2.0 / (hxp * (hxp + hxm))
                        diag_x = -coeff_e
                        col_e_idx = _idx(k, i + 1, j)
                        col_w_idx = None
                    elif i == nx - 1:
                        hxm = h_xm[i, j]; hxp = hxm
                        coeff_e = 0.0
                        rhs_e = 0.0
                        coeff_w = 2.0 / (hxm * (hxp + hxm))
                        diag_x = -coeff_w
                        col_e_idx = None
                        col_w_idx = _idx(k, i - 1, j)
                    else:
                        hxp = h_xp[i, j]; hxm = h_xm[i, j]
                        denom = 0.5 * (hxp + hxm)
                        coeff_e = 1.0 / (hxp * denom)
                        coeff_w = 1.0 / (hxm * denom)
                        diag_x = -(coeff_e + coeff_w)
                        col_e_idx = _idx(k, i + 1, j)
                        col_w_idx = _idx(k, i - 1, j)

                    # ----- y direction --------------------------------
                    if j == 0:
                        hyp = h_yp[i, j]; hym = hyp
                        coeff_n = 2.0 / (hyp * (hyp + hym))
                        coeff_s = 0.0
                        diag_y = -coeff_n
                        col_n_idx = _idx(k, i, j + 1)
                        col_s_idx = None
                    elif j == ny - 1:
                        hym = h_ym[i, j]; hyp = hym
                        coeff_n = 0.0
                        coeff_s = 2.0 / (hym * (hyp + hym))
                        diag_y = -coeff_s
                        col_n_idx = None
                        col_s_idx = _idx(k, i, j - 1)
                    else:
                        hyp = h_yp[i, j]; hym = h_ym[i, j]
                        denom = 0.5 * (hyp + hym)
                        coeff_n = 1.0 / (hyp * denom)
                        coeff_s = 1.0 / (hym * denom)
                        diag_y = -(coeff_n + coeff_s)
                        col_n_idx = _idx(k, i, j + 1)
                        col_s_idx = _idx(k, i, j - 1)

                    # ----- z direction (no SW: vane modulation along z
                    # is captured by the per-slice apertures, not by
                    # cut-cells — z neighbours are always at distance h)
                    if k == 0:
                        hzp = dz_m; hzm = dz_m
                        coeff_zp = 2.0 / (hzp * (hzp + hzm))
                        coeff_zm = 0.0
                        diag_z = -coeff_zp
                        col_zp = _idx(k + 1, i, j)
                        col_zm = None
                    elif k == nz - 1:
                        hzp = dz_m; hzm = dz_m
                        coeff_zp = 0.0
                        coeff_zm = 2.0 / (hzm * (hzp + hzm))
                        diag_z = -coeff_zm
                        col_zp = None
                        col_zm = _idx(k - 1, i, j)
                    else:
                        hzp = dz_m; hzm = dz_m
                        coeff_zp = 1.0 / (hzp * 0.5 * (hzp + hzm))
                        coeff_zm = 1.0 / (hzm * 0.5 * (hzp + hzm))
                        diag_z = -(coeff_zp + coeff_zm)
                        col_zp = _idx(k + 1, i, j)
                        col_zm = _idx(k - 1, i, j)

                    # ----- assemble row -------------------------------
                    rows.append(p); cols.append(p)
                    data.append(diag_x + diag_y + diag_z)

                    # x neighbours: SW boundary moves to RHS.
                    if col_e_idx is not None:
                        if is_bc_xp[i, j]:
                            rhs[p] -= coeff_e * float(bc_xp[i, j])
                        else:
                            rows.append(p); cols.append(col_e_idx)
                            data.append(coeff_e)
                    if col_w_idx is not None:
                        if is_bc_xm[i, j]:
                            rhs[p] -= coeff_w * float(bc_xm[i, j])
                        else:
                            rows.append(p); cols.append(col_w_idx)
                            data.append(coeff_w)
                    # y neighbours
                    if col_n_idx is not None:
                        if is_bc_yp[i, j]:
                            rhs[p] -= coeff_n * float(bc_yp[i, j])
                        else:
                            rows.append(p); cols.append(col_n_idx)
                            data.append(coeff_n)
                    if col_s_idx is not None:
                        if is_bc_ym[i, j]:
                            rhs[p] -= coeff_s * float(bc_ym[i, j])
                        else:
                            rows.append(p); cols.append(col_s_idx)
                            data.append(coeff_s)
                    # z neighbours (interior only — pure 7-point)
                    if col_zp is not None:
                        rows.append(p); cols.append(col_zp)
                        data.append(coeff_zp)
                    if col_zm is not None:
                        rows.append(p); cols.append(col_zm)
                        data.append(coeff_zm)

        from scipy.sparse import coo_matrix
        L = coo_matrix(
            (data, (rows, cols)), shape=(n_total, n_total)
        ).tocsr()
        return L, rhs

    # ------------------------------------------------------------------
    def _pyamg_solve(self, ml, rhs, *, verbose=False):
        """V-cycle solve until the residual drops below ``tol``.

        Shortley-Weller produces a non-symmetric operator (different
        h_xp / h_xm at neighbours give asymmetric off-diagonal pairs),
        so GMRES is used instead of CG.
        """
        residuals = []
        x = ml.solve(rhs, tol=1e-8, residuals=residuals,
                     accel='gmres', maxiter=200)
        if verbose:
            print(f"  pyamg V-cycle: {len(residuals)} iters, "
                  f"residual {residuals[-1]:.2e}")
        return x, residuals

    # ------------------------------------------------------------------
    # Public scalar lookups (envelope dispatch)
    # ------------------------------------------------------------------
    def K_xx_axis(self, z_mm) -> np.ndarray:
        return np.interp(z_mm, self._caches.z_mm, self._caches.K_xx)

    def K_yy_axis(self, z_mm) -> np.ndarray:
        return np.interp(z_mm, self._caches.z_mm, self._caches.K_yy)

    def Ez_axis(self, z_mm) -> np.ndarray:
        return np.interp(z_mm, self._caches.z_mm, self._caches.Ez)

    def Phi_static(self, x_mm, y_mm, z_mm):
        scalar = (np.ndim(x_mm) == 0 and np.ndim(y_mm) == 0
                  and np.ndim(z_mm) == 0)
        x = np.atleast_1d(x_mm).astype(float)
        y = np.atleast_1d(y_mm).astype(float)
        z = np.atleast_1d(z_mm).astype(float)
        zb, xb, yb = np.broadcast_arrays(z, x, y)
        pts = np.stack([zb.ravel(), xb.ravel(), yb.ravel()], axis=-1)
        result = self._rgi(pts).reshape(zb.shape)
        return float(result.flat[0]) if scalar else result

    @property
    def memory_MB(self) -> float:
        return self.phi_static.nbytes / (1024.0 * 1024.0)
