"""M3 — Numerical 2-D Laplace per z-slice for the four-vane RFQ.

Engaged via ``field_model="laplace2d"``.  The RFQ potential is built from
scratch by solving ``∇²Φ = 0`` on a 2-D Cartesian grid at every .vane
z-slice with Dirichlet BCs imposed on the four hyperbolic vane surfaces
``x²−y² = ±a_v(z)²``  /  ``y²−x² = ±a_v(z)²`` (RFQUIK / Toutatis form,
no ``Tc`` correction).

The static solution ``Φ_static(x,y,z)`` is cached as a ``(Nz, Nx, Ny)``
``float32`` array; runtime field lookup uses central finite difference
of the cached Φ in all three axes.  The instantaneous field at design
phase ``φ_s`` is obtained by multiplying the static result by the
appropriate ``sin(φ_s)`` / ``cos(φ_s)`` time modulation that the rest
of :mod:`linac_gen.elements.vane_rfq` already applies to the analytic
2-term Crandall fields — so the M3 dispatch slots in cleanly between
the M1 (2-term) and M2 (matcher-aware) paths.

Caveats — empirical observations on the PXIE LEBT+RFQ benchmark
---------------------------------------------------------------
With ``nx=64, z_subsample=2`` and the corrected envelope initial
conditions (σ_x_in = 4.87 mm), M3 brings ``σ_x_exit`` from M1's
1.84 mm down to 0.68 mm versus TraceWin's 0.51 mm — a 2.5× tightening.
The y-plane is unstable in this configuration: the per-slice 2-D
Laplace gives ``∂²Φ/∂x²+∂²Φ/∂y² = 0`` exactly, removing the common-mode
RF-defocusing term that M1 captures via its ``(π/L)²·(A·V/2)·C₂``
analytic expression.  A naive splice (``M1 rf_defoc`` + ``M3 K_xx,
K_yy``) is destabilising in PXIE; the correct fix is full 3-D Laplace
or a smooth (non-discrete) vane mask, both of which are out of scope
for the present milestone.  ``laplace2d`` is therefore best treated
as a transverse-x demonstration of the right *direction*; it is not
yet production-grade for σ_y.

Solver
------
Scipy's :func:`scipy.sparse.linalg.spsolve` is used as the default solver
because it has no extra dependency.  Per-slice solve takes ~30–80 ms on
a 64×64 grid; full PXIE init (17 567 slices) is therefore ~8–20 minutes
at native resolution.  For tractable iteration the cache supports
``z_subsample`` to reduce the z-grid by an integer factor.

For the ~30 % of cells that lie inside vane material the corresponding
rows of the Laplacian are replaced with identity rows pinned to the
vane voltage; the remaining rows form an interior 2-D Poisson system
that ``spsolve`` factorises and solves directly.  Future work could
swap in :mod:`pyamg` for a 100× speedup, but is not required to validate
M3 against TraceWin.

Outputs
-------
:class:`Laplace2DCache` exposes:

* ``Phi_static(x_mm, y_mm, z_mm)``      — Φ_static in volts.
* ``E_static(x_mm, y_mm, z_mm)``        — (E_x, E_y, E_z) gradients
                                          of Φ_static, V/mm.
* ``K_xx_axis(z_mm)``, ``K_yy_axis(z_mm)`` — second derivatives at axis,
                                          V/mm² — the linearised
                                          quadrupole strengths used by
                                          the envelope-tracker dispatch.
* ``Ez_axis(z_mm)``                     — −∂Φ_static/∂z at axis, V/mm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, eye, kron, lil_matrix, diags
from scipy.sparse.linalg import spsolve

from linac_gen.io.tracewin_vane import VaneGeometry


# Hyperbolic-vane "inside" predicate.  A grid point ``(x, y)`` (m) is
# inside the +x vane (vane 1) iff ``x > 0`` and ``x² − y² ≥ a₁²``.  The
# other three vanes are mirror reflections — this matches the standard
# four-vane RFQ where vane pairs (1,3) and (2,4) sit at orthogonal angles
# with hyperbolic asymptotes at ±45°.


def _vane_masks(x_m: np.ndarray, y_m: np.ndarray,
                a1: float, a2: float, a3: float, a4: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for the four hyperbolic vane interiors.

    Inputs ``x_m``, ``y_m`` are 1-D coordinate arrays (m), the four
    apertures are scalars (m).  Returns four 2-D ``(Nx, Ny)`` masks for
    vanes 1..4 respectively.  The hyperbola asymptotes meet at the
    origin so a grid point can be inside at most one vane region.
    """
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    diff = X * X - Y * Y    # > 0 in the +x and -x quadrants
    sumf = -diff            # > 0 in the +y and -y quadrants
    m1 = (X > 0) & (diff >= a1 * a1)
    m2 = (Y > 0) & (sumf >= a2 * a2)
    m3 = (X < 0) & (diff >= a3 * a3)
    m4 = (Y < 0) & (sumf >= a4 * a4)
    return m1, m2, m3, m4


def _vane_chi_smooth(x_m: np.ndarray, y_m: np.ndarray,
                     a1: float, a2: float, a3: float, a4: float,
                     eps_m: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smooth (level-set) inside-vane indicators χ_v ∈ [0, 1].

    The signed-distance level set for vane 1 (+x branch) is
    ``ψ_v1 = x − √(y² + a₁²)`` — *positive* when the point is to the
    right of vane 1's hyperbolic inner surface (deep in vane material),
    *negative* outside.  The half-plane is handled implicitly: for
    x < 0, ``ψ_v1`` is always ≤ −√(y²+a₁²) < 0 so χ_v1 → 0 cleanly,
    no hard ``np.where`` step needed.

    χ_v = ½(1 + tanh(ψ_v / eps_m)) gives a C∞ indicator with a tanh
    transition over a few ``eps_m`` near the surface.
    """
    X, Y = np.meshgrid(x_m, y_m, indexing='ij')
    # Hyperbolic surface distances (one branch each):
    #   vane 1: x = +√(y²+a₁²) → ψ_v1 = x − √(...)
    #   vane 2: y = +√(x²+a₂²) → ψ_v2 = y − √(...)
    #   vane 3: x = −√(y²+a₃²) → ψ_v3 = −x − √(...) = −(x + √(...))
    #   vane 4: y = −√(x²+a₄²) → ψ_v4 = −y − √(...)
    psi1 =  X - np.sqrt(Y * Y + a1 * a1)
    psi2 =  Y - np.sqrt(X * X + a2 * a2)
    psi3 = -X - np.sqrt(Y * Y + a3 * a3)
    psi4 = -Y - np.sqrt(X * X + a4 * a4)
    chi1 = 0.5 * (1.0 + np.tanh(psi1 / eps_m))
    chi2 = 0.5 * (1.0 + np.tanh(psi2 / eps_m))
    chi3 = 0.5 * (1.0 + np.tanh(psi3 / eps_m))
    chi4 = 0.5 * (1.0 + np.tanh(psi4 / eps_m))
    return chi1, chi2, chi3, chi4


def _build_laplacian_2d(nx: int, ny: int,
                        dx: float, dy: float) -> csr_matrix:
    """5-point Laplacian on an ``(nx, ny)`` Cartesian grid.

    Row-major flattening ``index(i, j) = i*ny + j`` with ``i`` along x.
    The boundary rows have one fewer neighbour by construction (no
    wrap-around), so the operator is well-defined as ``∇²Φ`` interior to
    the grid; the outer ring is later forced to ``Φ = 0`` (vacuum BC)
    by replacing those rows with identity entries in :func:`_solve_slice`.
    """
    Lx = diags([1.0, -2.0, 1.0], [-1, 0, 1],
               shape=(nx, nx), format='csr') / (dx * dx)
    Ly = diags([1.0, -2.0, 1.0], [-1, 0, 1],
               shape=(ny, ny), format='csr') / (dy * dy)
    return (kron(Lx, eye(ny, format='csr'))
            + kron(eye(nx, format='csr'), Ly)).tocsr()


def _solve_slice(L_base: csr_matrix,
                 x_m: np.ndarray, y_m: np.ndarray,
                 a1: float, a2: float, a3: float, a4: float,
                 V1: float, V2: float, V3: float, V4: float
                 ) -> np.ndarray:
    """Solve ∇²Φ = 0 with vane Dirichlet BCs at one z-slice.

    Returns ``Φ`` as a ``(Nx, Ny)`` ``float64`` array (V).  The vane
    interiors are pinned to the corresponding ``V_v`` and the outermost
    grid ring is pinned to 0 V (vacuum).  Both choices represent
    Dirichlet BCs imposed by replacing the relevant rows of the
    Laplacian with identity rows.
    """
    nx, ny = x_m.size, y_m.size
    n_total = nx * ny

    m1, m2, m3, m4 = _vane_masks(x_m, y_m, a1, a2, a3, a4)
    bc_mask = m1 | m2 | m3 | m4
    bc_phi = np.zeros((nx, ny))
    bc_phi[m1] = V1
    bc_phi[m2] = V2
    bc_phi[m3] = V3
    bc_phi[m4] = V4

    # Outer ring → Φ = 0 (vacuum BC).
    bc_mask[0, :] = True
    bc_mask[-1, :] = True
    bc_mask[:, 0] = True
    bc_mask[:, -1] = True

    bc_flat = bc_mask.ravel()
    rhs = np.zeros(n_total)
    rhs[bc_flat] = bc_phi.ravel()[bc_flat]

    # Replace boundary rows with identity rows (Φ_i = rhs_i).
    L = L_base.tolil(copy=True)
    for i in np.flatnonzero(bc_flat):
        L.rows[i] = [i]
        L.data[i] = [1.0]
    L = L.tocsr()

    phi_flat = spsolve(L, rhs)
    return phi_flat.reshape(nx, ny)


def _solve_slice_smooth(L_base: csr_matrix,
                        x_m: np.ndarray, y_m: np.ndarray,
                        a1: float, a2: float, a3: float, a4: float,
                        V1: float, V2: float, V3: float, V4: float,
                        eps_m: float, alpha: float
                        ) -> np.ndarray:
    """Smooth-mask (penalty BC) variant of :func:`_solve_slice`.

    Solves ``L · Φ + α · χ_total · Φ = α · Σ χ_v · V_v`` where
    ``χ_v`` are the four smoothed level-set indicators (``[0, 1]``,
    transition width ``eps_m``).  As ``α → ∞`` and ``eps_m → 0`` this
    recovers the binary Dirichlet BC; for finite values the BC varies
    *smoothly* with the aperture, eliminating the stair-step Φ_axis
    that the binary mask produces sub-grid.

    Outer ring still has the binary Φ = 0 BC (vacuum) so the system
    stays well-posed.
    """
    nx, ny = x_m.size, y_m.size
    n_total = nx * ny

    chi1, chi2, chi3, chi4 = _vane_chi_smooth(x_m, y_m,
                                              a1, a2, a3, a4, eps_m)
    chi_total = chi1 + chi2 + chi3 + chi4
    chi_voltage = (chi1 * V1 + chi2 * V2 + chi3 * V3 + chi4 * V4)

    # Outer ring still gets a hard Φ = 0 vacuum BC; the rest of the
    # cells use the penalty form.
    edge_mask = np.zeros((nx, ny), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    interior = ~edge_mask
    chi_total_flat = (chi_total * interior).ravel()
    chi_v_flat     = (chi_voltage * interior).ravel()

    # System: (L + α·diag(χ)) Φ = α · χ_voltage  on interior;
    # rows for edge cells become identity rows pinned to 0.
    L = L_base.tolil(copy=True)
    diag_pen = alpha * chi_total_flat
    L_csr = L.tocsr()
    L_pen = L_csr + csr_matrix((diag_pen,
                                (np.arange(n_total),
                                 np.arange(n_total))),
                               shape=(n_total, n_total))
    rhs = alpha * chi_v_flat

    # Edge rows → identity, RHS=0.
    L_pen = L_pen.tolil()
    for i in np.flatnonzero(edge_mask.ravel()):
        L_pen.rows[i] = [i]
        L_pen.data[i] = [1.0]
        rhs[i] = 0.0
    L_pen = L_pen.tocsr()

    phi_flat = spsolve(L_pen, rhs)
    return phi_flat.reshape(nx, ny)


@dataclass
class _Caches:
    """Per-slice scalar derivative caches at the on-axis sample point.

    The envelope tracker only needs these axis-line scalars; particle
    tracking uses the full 3-D interpolator instead.
    """
    z_mm: np.ndarray         # (Nz,) z grid in mm
    K_xx: np.ndarray         # (Nz,) ∂²Φ_static/∂x² at axis  V/mm²
    K_yy: np.ndarray         # (Nz,) ∂²Φ_static/∂y² at axis  V/mm²
    Ez:   np.ndarray         # (Nz,) −∂Φ_static/∂z  at axis  V/mm


class Laplace2DCache:
    """Per-z 2-D Laplace cache with vane-shape Dirichlet BCs.

    Parameters
    ----------
    vane : VaneGeometry
        Parsed ``.vane`` data.  Gives apertures and voltages per slice.
    nx, ny : int
        Grid resolution in x, y (default 64 each).
    box_factor : float
        Half-width of the simulation box, in units of the maximum
        aperture across all slices (default 1.5).  ``1.5`` keeps the
        box just outside the vane tips so the Dirichlet BC at the box
        edge (vacuum) sits inside vane material for most z.
    z_subsample : int
        Sub-sample the .vane z grid by this factor (default 1, no
        subsampling).  Useful for fast tests and for keeping init time
        manageable on large lattices.
    verbose : bool
        Print one line every 5 % through the slice loop (default False).

    Attributes
    ----------
    nz : int
    z_mm : np.ndarray
        Cached z grid in mm.
    phi_static : np.ndarray  (nz, nx, ny)
        Cached Φ_static in V.

    Notes
    -----
    Memory budget: ``nz × nx × ny × 4`` bytes (``float32``).  For PXIE
    at native resolution: 17 567 × 64 × 64 × 4 ≈ 287 MB.
    """

    def __init__(self, vane: VaneGeometry, *,
                 nx: int = 64, ny: int = 64,
                 box_factor: float = 1.5,
                 z_subsample: int = 1,
                 mask: str = "binary",
                 smooth_eps_grid: float = 1.0,
                 smooth_alpha_factor: float = 1.0e3,
                 verbose: bool = False):
        if nx < 5 or ny < 5:
            raise ValueError("Laplace2DCache needs nx, ny ≥ 5")
        if box_factor <= 1.0:
            raise ValueError("box_factor must be > 1.0 (else box is "
                             "inside the largest vane aperture)")
        if mask not in {"binary", "smooth"}:
            raise ValueError("mask must be 'binary' or 'smooth'")

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
        self.z_mm = vane.z[idx] * 1000.0          # mm
        self.z_m = vane.z[idx].copy()             # m  (internal)

        # Per-slice apertures & voltages (m, V).
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
        # mm equivalents for the interpolator and FD lookups.
        self._x_mm = self._x_m * 1000.0
        self._y_mm = self._y_m * 1000.0
        self._dx_mm = self._dx_m * 1000.0
        self._dy_mm = self._dy_m * 1000.0

        # ---- build static Laplace cache ------------------------------
        L_base = _build_laplacian_2d(self.nx, self.ny,
                                     self._dx_m, self._dy_m)
        self.phi_static = np.empty((self.nz, self.nx, self.ny),
                                   dtype=np.float32)
        # Axis-line samples accumulated in float64 *before* the float32
        # truncation that happens when filling ``phi_static``.  The
        # float32 quantisation (~10⁻⁷ relative) is fine for the 3-D RGI
        # but blows up the small slice-to-slice Φ differences that FD
        # uses to estimate ``Ez_axis`` — so the axis line, which is what
        # the envelope tracker queries, is preserved at full precision
        # in a small (Nz,) auxiliary array.
        ix_axis, iy_axis = self._axis_indices()
        if (ix_axis in (0, self.nx - 1)
                or iy_axis in (0, self.ny - 1)):
            raise ValueError("Laplace2DCache axis sample at grid edge — "
                             "increase box_factor or nx so x=0 lies "
                             "well inside the box")
        phi_axis_f64 = np.empty(self.nz, dtype=np.float64)
        phi_xp_f64   = np.empty(self.nz, dtype=np.float64)
        phi_xm_f64   = np.empty(self.nz, dtype=np.float64)
        phi_yp_f64   = np.empty(self.nz, dtype=np.float64)
        phi_ym_f64   = np.empty(self.nz, dtype=np.float64)
        if verbose:
            print(f"Laplace2DCache: solving {self.nz} slices on "
                  f"{self.nx}×{self.ny} grid (box ±{x_lim_m*1e3:.2f} mm)")
        report_every = max(1, self.nz // 20)
        # Smooth-mask transition width is set as a multiple of the grid
        # spacing — narrower than one cell would not resolve the tanh
        # transition, much wider would smear out the actual vane edge.
        eps_m = smooth_eps_grid * self._dx_m
        # The penalty α must be large compared to the Laplacian's
        # diagonal magnitude (4/dx² in m⁻²) for the BC to be enforced
        # accurately, but not so large that the system becomes
        # ill-conditioned.  ``smooth_alpha_factor`` ≈ 10³ gives BC
        # accuracy of ~0.1 % on a 33-pt grid without numerical issues.
        L_diag_scale = 4.0 / (self._dx_m * self._dx_m)
        alpha = smooth_alpha_factor * L_diag_scale
        self._mask = mask
        self._smooth_eps_m = eps_m
        self._smooth_alpha = alpha
        for k in range(self.nz):
            if mask == "smooth":
                phi_k = _solve_slice_smooth(
                    L_base,
                    self._x_m, self._y_m,
                    float(self._a1[k]), float(self._a2[k]),
                    float(self._a3[k]), float(self._a4[k]),
                    float(self._V1[k]), float(self._V2[k]),
                    float(self._V3[k]), float(self._V4[k]),
                    eps_m=eps_m, alpha=alpha,
                )
            else:
                phi_k = _solve_slice(
                    L_base,
                    self._x_m, self._y_m,
                    float(self._a1[k]), float(self._a2[k]),
                    float(self._a3[k]), float(self._a4[k]),
                    float(self._V1[k]), float(self._V2[k]),
                    float(self._V3[k]), float(self._V4[k]),
                )
            phi_axis_f64[k] = phi_k[ix_axis, iy_axis]
            phi_xp_f64[k]   = phi_k[ix_axis + 1, iy_axis]
            phi_xm_f64[k]   = phi_k[ix_axis - 1, iy_axis]
            phi_yp_f64[k]   = phi_k[ix_axis, iy_axis + 1]
            phi_ym_f64[k]   = phi_k[ix_axis, iy_axis - 1]
            self.phi_static[k] = phi_k.astype(np.float32)
            if verbose and (k + 1) % report_every == 0:
                print(f"  slice {k+1}/{self.nz}")

        # ---- 3-D RGI for runtime particle queries --------------------
        self._rgi = RegularGridInterpolator(
            (self.z_mm, self._x_mm, self._y_mm),
            self.phi_static,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )

        # ---- on-axis derivative caches (envelope tracking) -----------
        # Per-slice 2-D Laplace gives ``∂²Φ/∂x² + ∂²Φ/∂y² = 0`` exactly,
        # so the in-plane K_xx and K_yy are purely anti-symmetric.  The
        # missing common-mode term (longitudinal–transverse coupling,
        # i.e. RF defocusing) is restored from ``K_zz = ∂²Φ/∂z²`` —
        # but only when the z-grid is fine enough to give a clean
        # ``K_zz``; otherwise the correction adds more noise than
        # signal and the bare 2-D values are kept.
        K_xx_2d = (phi_xp_f64 - 2.0 * phi_axis_f64 + phi_xm_f64) / (self._dx_mm ** 2)
        K_yy_2d = (phi_yp_f64 - 2.0 * phi_axis_f64 + phi_ym_f64) / (self._dy_mm ** 2)
        # The discrete vane mask flips one grid cell at a time as the
        # aperture changes sub-grid between consecutive z slices, so
        # phi_axis_f64 stair-steps even when the underlying physics
        # demands a smooth profile.  Naive FD reports delta-spikes —
        # smooth Φ_axis on a scale matched to the .vane native step
        # before differentiating.
        from scipy.ndimage import gaussian_filter1d
        if self.nz >= 5:
            sigma_slices = max(1.0, float(self.nz) / 100.0)
            phi_axis_smooth = gaussian_filter1d(
                phi_axis_f64, sigma=sigma_slices, mode='nearest')
        else:
            phi_axis_smooth = phi_axis_f64
        dphi_dz = np.gradient(phi_axis_smooth, self.z_mm)        # V/mm
        Ez = -dphi_dz                                            # V/mm
        # Common-mode correction is currently disabled: deriving K_zz
        # from a second numerical gradient over the cached z-grid
        # amplifies the residual mask-flip noise far above the true
        # physical magnitude.  Keep K_xx_2d, K_yy_2d as the operating
        # values — they capture the dominant quadrupole anisotropy and
        # match M1 in the matcher limit.
        K_xx = K_xx_2d
        K_yy = K_yy_2d
        self._caches = _Caches(z_mm=self.z_mm.copy(),
                               K_xx=K_xx, K_yy=K_yy, Ez=Ez)
        self._phi_axis_f64 = phi_axis_f64        # raw, for diagnostics
        self._phi_axis_smooth = phi_axis_smooth  # what Ez_axis came from

    # ------------------------------------------------------------------
    # Axis-line second-derivative & ∂z caches
    # ------------------------------------------------------------------
    def _axis_indices(self) -> tuple[int, int]:
        """Indices of (x≈0, y≈0) on the grid."""
        ix0 = int(np.argmin(np.abs(self._x_m)))
        iy0 = int(np.argmin(np.abs(self._y_m)))
        return ix0, iy0

    # ------------------------------------------------------------------
    # Public scalar lookups (envelope dispatch)
    # ------------------------------------------------------------------
    def K_xx_axis(self, z_mm: float | np.ndarray) -> np.ndarray:
        """``∂²Φ_static/∂x²`` at axis, V/mm² (linear interp in z)."""
        return np.interp(z_mm, self._caches.z_mm, self._caches.K_xx)

    def K_yy_axis(self, z_mm: float | np.ndarray) -> np.ndarray:
        return np.interp(z_mm, self._caches.z_mm, self._caches.K_yy)

    def Ez_axis(self, z_mm: float | np.ndarray) -> np.ndarray:
        """``−∂Φ_static/∂z`` at axis, V/mm (linear interp in z)."""
        return np.interp(z_mm, self._caches.z_mm, self._caches.Ez)

    # ------------------------------------------------------------------
    # Particle-tracking lookups (3-D)
    # ------------------------------------------------------------------
    def Phi_static(self, x_mm, y_mm, z_mm):
        """Static potential Φ at (x, y, z), V.  All inputs in mm.

        Returns a Python ``float`` for scalar inputs; an ``np.ndarray``
        of the broadcast shape for array inputs.
        """
        scalar_inputs = (np.ndim(x_mm) == 0 and np.ndim(y_mm) == 0
                         and np.ndim(z_mm) == 0)
        x_arr = np.atleast_1d(x_mm).astype(float)
        y_arr = np.atleast_1d(y_mm).astype(float)
        z_arr = np.atleast_1d(z_mm).astype(float)
        z_b, x_b, y_b = np.broadcast_arrays(z_arr, x_arr, y_arr)
        pts = np.stack([z_b.ravel(), x_b.ravel(), y_b.ravel()], axis=-1)
        result = self._rgi(pts).reshape(z_b.shape)
        return float(result.flat[0]) if scalar_inputs else result

    def E_static(self, x_mm, y_mm, z_mm,
                 *, eps_mm: float = 0.5
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gradients ``E = −∇Φ_static`` at (x, y, z), V/mm.

        Central finite difference with step ``eps_mm`` (default 0.5 mm).
        ``eps_mm`` should be larger than one grid step in the
        corresponding axis but smaller than the cavity scale.
        """
        e = float(eps_mm)
        Ex = -(self.Phi_static(x_mm + e, y_mm, z_mm)
              - self.Phi_static(x_mm - e, y_mm, z_mm)) / (2.0 * e)
        Ey = -(self.Phi_static(x_mm, y_mm + e, z_mm)
              - self.Phi_static(x_mm, y_mm - e, z_mm)) / (2.0 * e)
        Ez = -(self.Phi_static(x_mm, y_mm, z_mm + e)
              - self.Phi_static(x_mm, y_mm, z_mm - e)) / (2.0 * e)
        return Ex, Ey, Ez

    # ------------------------------------------------------------------
    @property
    def memory_MB(self) -> float:
        """Cache size in MB (Φ_static only — derivative caches are
        small)."""
        return self.phi_static.nbytes / (1024.0 * 1024.0)
